// Link to the dream engine. Implements the frozen protocol from docs/DESIGN.md.
//   client -> server: binary [u32 LE headerLen][JSON header][JPEG]
//   server -> client: binary result (same framing) | text info/dropped/stats
// Events (via .on(name, fn)): 'status' (string), 'info' (obj), 'result'
// ({header, bytes} | {header, texture, flipY}), 'dropped' (id), 'stats' (obj).
import * as THREE from 'three';

const enc = new TextEncoder();
const dec = new TextDecoder();

class Emitter {
  constructor() { this._h = {}; }
  on(ev, fn) { (this._h[ev] ||= []).push(fn); return this; }
  emit(ev, a) { for (const fn of this._h[ev] || []) { try { fn(a); } catch (e) { console.error(e); } } }
}

export function frameBytes(header, payload) {
  const h = enc.encode(JSON.stringify(header));
  const out = new Uint8Array(4 + h.length + payload.byteLength);
  new DataView(out.buffer).setUint32(0, h.length, true);
  out.set(h, 4);
  out.set(payload instanceof Uint8Array ? payload : new Uint8Array(payload), 4 + h.length);
  return out;
}

export function parseFrame(buf) {
  const dv = new DataView(buf);
  const hl = dv.getUint32(0, true);
  const header = JSON.parse(dec.decode(new Uint8Array(buf, 4, hl)));
  return { header, bytes: new Uint8Array(buf, 4 + hl) };
}

export class Link extends Emitter {
  constructor(url) {
    super();
    this.url = url;
    this.kind = 'server';
    this.ws = null;
    this.status = 'connecting';
    this.info = null;
    this.stats = null;
    this.inFlight = new Map(); // id -> sentAt(ms)
    this.retry = 0;
    this.closed = false;
    this.bytesOut = 0;
    this._connect();
  }
  get ready() { return !!this.ws && this.ws.readyState === 1; }
  get inFlightCount() { return this.inFlight.size; }
  _setStatus(s) { if (s !== this.status) { this.status = s; this.emit('status', s); } }
  _connect() {
    if (this.closed) return;
    this._setStatus(this.retry ? 'reconnecting' : 'connecting');
    let ws;
    try { ws = new WebSocket(this.url); } catch (e) { this._scheduleReconnect(); return; }
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    ws.onopen = () => { this.retry = 0; this._setStatus('connected'); };
    ws.onmessage = (ev) => this._onMessage(ev.data);
    ws.onclose = () => {
      this._flushInFlight();
      this.ws = null;
      this._setStatus('disconnected');
      this._scheduleReconnect();
    };
    ws.onerror = () => { /* onclose follows */ };
  }
  _scheduleReconnect() {
    if (this.closed) return;
    this.retry++;
    const delay = Math.min(5000, 500 * 2 ** Math.min(this.retry, 4));
    setTimeout(() => this._connect(), delay);
  }
  _flushInFlight() {
    for (const id of this.inFlight.keys()) this.emit('dropped', id);
    this.inFlight.clear();
  }
  _onMessage(data) {
    if (typeof data === 'string') {
      let m; try { m = JSON.parse(data); } catch { return; }
      if (m.type === 'info') { this.info = m; this.emit('info', m); }
      else if (m.type === 'dropped') { if (this.inFlight.delete(m.id)) this.emit('dropped', m.id); }
      else if (m.type === 'stats') { this.stats = m; this.emit('stats', m); }
      else if (m.type === 'error') { console.warn('[dream] server error', m); if (m.id != null && this.inFlight.delete(m.id)) this.emit('dropped', m.id); }
      return;
    }
    const { header, bytes } = parseFrame(data);
    if (header.type !== 'result') return;
    const sentAt = this.inFlight.get(header.id);
    this.inFlight.delete(header.id);
    header.latency = sentAt != null ? performance.now() - sentAt : null;
    this.emit('result', { header, bytes });
  }
  // payload: Uint8Array JPEG
  send(header, payload) {
    if (!this.ready) return false;
    const buf = frameBytes(header, payload);
    this.ws.send(buf);
    this.bytesOut += buf.byteLength;
    this.inFlight.set(header.id, performance.now());
    return true;
  }
  // expire requests that never came back (server restarted etc.)
  pump(now) {
    for (const [id, t] of this.inFlight) {
      if (now - t > 6000) { this.inFlight.delete(id); this.emit('dropped', id); }
    }
  }
  close() { this.closed = true; try { this.ws && this.ws.close(); } catch {} }
}

// ---------------------------------------------------------------------------
// MockLink: in-browser fake diffusion. Decodes the JPEG we would have sent,
// runs a painterly GPU filter (kuwahara + palette drift + seeded domain warp),
// and returns it as a texture after a realistic latency, with the same
// latest-wins single-worker scheduling as the real server.
const MOCK_FRAG = /* glsl */ `
uniform sampler2D uIn;
uniform vec2 uTexel;
uniform float uStrength;
uniform float uSeed;
uniform vec3 uTint;
uniform float uTime;
varying vec2 vUv;
float h12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float vn(vec2 p){ vec2 i = floor(p), f = fract(p); vec2 u = f*f*(3.0-2.0*f);
  return mix(mix(h12(i), h12(i+vec2(1,0)), u.x), mix(h12(i+vec2(0,1)), h12(i+vec2(1,1)), u.x), u.y); }
float fbm(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++){ s += a * vn(p); p = p * 2.1 + 3.7; a *= 0.5; } return s; }
vec3 samp(vec2 uv){ return texture(uIn, vec2(uv.x, 1.0 - uv.y)).rgb; }
vec3 hueRot(vec3 c, float a){ const vec3 k = vec3(0.57735); float ca = cos(a); return c * ca + cross(k, c) * sin(a) + k * dot(k, c) * (1.0 - ca); }
void main(){
  // seeded, screen-anchored warp (like fixed diffusion noise)
  vec2 q = vUv * 5.0 + uSeed * 1.7;
  vec2 w = vec2(fbm(q), fbm(q + 9.2)) - 0.5;
  vec2 uv = vUv + w * 0.018 * uStrength;
  // 4-sector kuwahara, radius 3
  vec3 m[4]; vec3 s[4];
  for (int k = 0; k < 4; k++) { m[k] = vec3(0); s[k] = vec3(0); }
  for (int j = -3; j <= 3; j++) for (int i = -3; i <= 3; i++) {
    vec3 c = samp(uv + vec2(i, j) * uTexel * 1.3);
    if (i <= 0 && j <= 0) { m[0] += c; s[0] += c * c; }
    if (i >= 0 && j <= 0) { m[1] += c; s[1] += c * c; }
    if (i <= 0 && j >= 0) { m[2] += c; s[2] += c * c; }
    if (i >= 0 && j >= 0) { m[3] += c; s[3] += c * c; }
  }
  vec3 best = vec3(0); float bv = 1e9;
  for (int k = 0; k < 4; k++) {
    vec3 mu = m[k] / 16.0; vec3 v = abs(s[k] / 16.0 - mu * mu);
    float vs = v.r + v.g + v.b;
    if (vs < bv) { bv = vs; best = mu; }
  }
  vec3 c = best;
  // dream grade: lift, saturate, tint shadows, drift hue by region
  float l = dot(c, vec3(0.299, 0.587, 0.114));
  c = mix(vec3(l), c, 1.0 + 0.5 * uStrength);
  c = hueRot(c, (fbm(vUv * 2.0 + uSeed) - 0.5) * 1.2 * uStrength);
  c = mix(c, c * uTint * 1.4, (1.0 - l) * 0.35 * uStrength);
  // keep mean luminance (a real model roughly preserves tone at moderate strength)
  float l2 = dot(c, vec3(0.299, 0.587, 0.114));
  c *= (l + 0.01) / (l2 + 0.01);
  // gilded brush filigree following luminance contours
  float f = fbm(uv * 18.0 + w * 6.0);
  c += uTint * smoothstep(0.62, 0.8, f) * 0.12 * uStrength * (0.2 + l);
  // soft posterize
  vec3 pc = floor(c * 7.0 + 0.5) / 7.0;
  c = mix(c, pc, 0.3 * uStrength);
  gl_FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
`;

export class MockLink extends Emitter {
  constructor(renderer, { latency = 150, width = 512, height = 512 } = {}) {
    super();
    this.kind = 'mock';
    this.renderer = renderer;
    this.latency = latency;
    this.status = 'connected';
    this.info = { type: 'info', engine: 'mock (in-browser)', model: 'kuwahara dream filter', width, height, fps_estimate: 1000 / latency, device: 'webgl' };
    this.stats = null;
    this.inFlight = new Map();
    this.queue = null;      // latest-wins pending job
    this.busy = null;       // running job
    this.done = [];
    this.pool = Array.from({ length: 4 }, () => new THREE.WebGLRenderTarget(width, height, { depthBuffer: false }));
    this.poolIdx = 0;
    this.inTex = new THREE.Texture();
    this.inTex.flipY = false;
    this.inTex.colorSpace = THREE.NoColorSpace;
    this.mat = new THREE.ShaderMaterial({
      uniforms: {
        uIn: { value: this.inTex }, uTexel: { value: new THREE.Vector2(1 / width, 1 / height) },
        uStrength: { value: 0.6 }, uSeed: { value: 0 }, uTint: { value: new THREE.Vector3(1, 0.8, 0.6) }, uTime: { value: 0 },
      },
      vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
      fragmentShader: MOCK_FRAG, depthTest: false, depthWrite: false,
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 2, 0, 0, 2], 2));
    this.quad = new THREE.Mesh(geo, this.mat);
    this.quad.frustumCulled = false;
    this.cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.frames = 0; this.lastStats = performance.now();
    this.tint = new THREE.Vector3(1, 0.8, 0.6);
    setTimeout(() => { this.emit('status', 'connected'); this.emit('info', this.info); }, 0);
  }
  get ready() { return true; }
  get inFlightCount() { return this.inFlight.size; }
  setTint(rgb) { this.tint.set(rgb[0], rgb[1], rgb[2]); }
  send(header, payload) {
    const now = performance.now();
    this.inFlight.set(header.id, now);
    const blob = new Blob([payload], { type: 'image/jpeg' });
    const job = { header, now, bitmap: null };
    createImageBitmap(blob).then((bm) => { job.bitmap = bm; }).catch(() => { job.failed = true; });
    if (this.queue) { this.inFlight.delete(this.queue.header.id); this.emit('dropped', this.queue.header.id); this.queue.bitmap?.close?.(); }
    this.queue = job;
    return true;
  }
  pump(now) {
    // worker model: one job at a time, each takes ~latency ms
    if (this.busy && now >= this.busy.doneAt && this.busy.bitmap) {
      const job = this.busy; this.busy = null;
      this._run(job);
    }
    if (!this.busy && this.queue && (this.queue.bitmap || this.queue.failed)) {
      const job = this.queue; this.queue = null;
      if (job.failed) { this.inFlight.delete(job.header.id); this.emit('dropped', job.header.id); }
      else { job.doneAt = now + this.latency * (0.85 + 0.3 * Math.random()); this.busy = job; }
    }
    if (now - this.lastStats > 1000) {
      const fps = this.frames * 1000 / (now - this.lastStats);
      this.frames = 0; this.lastStats = now;
      this.stats = { type: 'stats', fps, ms_infer: this.latency, queue: this.queue ? 1 : 0 };
      this.emit('stats', this.stats);
    }
  }
  _run(job) {
    const r = this.renderer;
    const h = job.header;
    this.inTex.image = job.bitmap;
    this.inTex.needsUpdate = true;
    const rt = this.pool[this.poolIdx++ % this.pool.length];
    this.mat.uniforms.uStrength.value = h.strength ?? 0.6;
    this.mat.uniforms.uSeed.value = ((h.seed ?? 0) % 97) * 0.37;
    this.mat.uniforms.uTint.value.copy(this.tint);
    const prev = r.getRenderTarget();
    r.setRenderTarget(rt);
    r.render(this.quad, this.cam);
    r.setRenderTarget(prev);
    job.bitmap.close?.();
    this.inTex.image = null;
    const sentAt = this.inFlight.get(h.id);
    this.inFlight.delete(h.id);
    this.frames++;
    const latency = sentAt != null ? performance.now() - sentAt : null;
    this.emit('result', {
      header: { type: 'result', id: h.id, width: rt.width, height: rt.height, ms_infer: this.latency, ms_total: latency, latency },
      texture: rt.texture, flipY: false,
    });
  }
  close() {}
}
