// Dream controller: schedules captures, encodes them off the critical path,
// talks to the link, and turns results into (spread-out) paint jobs.
import * as THREE from 'three';
import { LiveViews, LIVE_SLOTS } from './live.js';

const SLOT_COUNT = 5;   // capture slots in flight; live views hold theirs on top of these

// Foveal warp pass: resamples a capture's perspective render (warp times the image's
// density) into the image sent to the model, magnified in the middle (live.js liveWarp is
// the forward map; this is its inverse). Where the warp squeezes, four taps spread over
// the footprint keep the periphery from aliasing.
const WARP_FRAG = /* glsl */ `
uniform sampler2D uSrc;
uniform vec2 uSrcTexel;
uniform float uM;
varying vec2 vUv;
void main(){
  vec2 q = vUv * 2.0 - 1.0;
  vec2 p = q / (uM - (uM - 1.0) * abs(q));
  vec2 d = 1.0 + (uM - 1.0) * abs(p);
  vec2 o = 0.35 * (d * d - 1.0) * uSrcTexel;   // source pixels per image pixel = d^2 (1 in the middle)
  vec2 uv = p * 0.5 + 0.5;
  gl_FragColor = 0.25 * (texture(uSrc, uv + vec2(-o.x, -o.y)) + texture(uSrc, uv + vec2(o.x, -o.y))
    + texture(uSrc, uv + vec2(-o.x, o.y)) + texture(uSrc, uv + vec2(o.x, o.y)));
}
`;

// Depth for engines that paint with it (/api/info `depth`): the capture's inverse depth averaged
// over each latent pixel's footprint (8x8 image pixels, through the same foveal warp as the
// image), from the capture's own depth buffer: inverse depth is linear in the buffer's value,
// so averaging raw values averages inverse depth (4 x 4 nearest taps spread over the footprint). Packed as 16 bits of log inverse depth (0.05..1000 m)
// in two 8-bit channels, so any GPU can render and read it back.
const DEPTHLAT_FRAG = /* glsl */ `
uniform sampler2D uDepth;       // the capture's depth buffer (perspective, warp x the image's density)
uniform vec2 uDepthTexel;
uniform vec2 uNF;               // capture near / far
uniform float uM;
varying vec2 vUv;
void main(){
  vec2 q = vUv * 2.0 - 1.0;
  vec2 p = q / (uM - (uM - 1.0) * abs(q));
  vec2 d = 1.0 + (uM - 1.0) * abs(p);
  vec2 st = 2.0 * d * d * uDepthTexel;   // the footprint is 8 d^2 depth texels a side: 4 x 4 taps
  vec2 c = p * 0.5 + 0.5;
  float s = 0.0;
  for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) s += texture(uDepth, c + (vec2(float(i), float(j)) - 1.5) * st).r;
  float iz = (uNF.y - (uNF.y - uNF.x) * s / 16.0) / (uNF.x * uNF.y);
  float v = clamp((log(max(iz, 1e-3)) + 6.9078) / 9.9035, 0.0, 0.99998);   // log(1/1000) .. log(1/0.05)
  float hi = floor(v * 255.0) / 255.0;
  gl_FragColor = vec4(hi, (v - hi) * 255.0, 0.0, 1.0);
}
`;

export class Dream {
  constructor({ renderer, scene, level, materials, shared, painter, link, settings, onEvent }) {
    this.renderer = renderer;
    this.scene = scene;
    this.level = level;
    this.materials = materials;
    this.painter = painter;
    this.settings = settings;
    this.onEvent = onEvent || (() => {});
    this.width = 512; this.height = 512;
    this.slots = [];
    this.nextId = 1;
    this.lastCapture = -1e9;
    this.jobs = [];
    this.pendingResults = [];
    this.stats = { keyframes: 0, dreamFps: 0, latency: 0, encodeMs: 0, captures: 0, results: 0, dropped: 0, paintMs: 0, lastResultAt: 0, activity: 0 };
    this._resTimes = [];
    this.encoder = new JpegEncoder();
    this.live = new LiveViews(shared, renderer, { count: settings.live ?? 5, tau: settings.liveTau ?? 0.35, fadeIn: settings.liveFade ?? 0.25 });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 2, 0, 0, 2], 2));
    this.warpMat = new THREE.ShaderMaterial({
      uniforms: { uSrc: { value: null }, uSrcTexel: { value: new THREE.Vector2() }, uM: { value: 1 } },
      vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
      fragmentShader: WARP_FRAG, depthTest: false, depthWrite: false,
    });
    this.warpQuad = new THREE.Mesh(geo, this.warpMat);
    this.warpQuad.frustumCulled = false;
    this.warpCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.warpRT = null;
    this.depthMat = new THREE.ShaderMaterial({
      uniforms: { uDepth: { value: null }, uDepthTexel: { value: new THREE.Vector2() }, uNF: { value: new THREE.Vector2(0.1, 800) }, uM: { value: 1 } },
      vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
      fragmentShader: DEPTHLAT_FRAG, depthTest: false, depthWrite: false,
    });
    this.depthRT = null;  // the latent-size depth sent with each frame
    this.nearRT = null;   // the capture's inverse depth at quarter size (the capture's depth cue)
    this.nearOK = renderer.extensions.has('EXT_color_buffer_float') || renderer.extensions.has('EXT_color_buffer_half_float');
    this._allocSlots(512, 512);
    this.setLink(link);
  }

  // w x h is the image the model gets. With a foveal warp (settings.warp = m > 1) each slot
  // renders its perspective view at m times that density and a shared pass warps it down.
  _allocSlots(w, h) {
    for (const s of this.slots) { s.rt.dispose(); s.rt.depthTexture?.dispose(); }
    this.warpRT?.dispose();
    this.width = w; this.height = h;
    this.warp = Math.min(2, Math.max(1, this.settings.warp || 1));   // (above 2 the warp pass's 4 taps leave gaps)
    const rw = Math.round(w * this.warp), rh = Math.round(h * this.warp);
    this.warpRT = this.warp > 1 ? new THREE.WebGLRenderTarget(w, h, { depthBuffer: false, type: THREE.UnsignedByteType }) : null;
    this.nearRT?.dispose();
    this.depthRT?.dispose();
    this.depthRT = new THREE.WebGLRenderTarget(Math.max(1, Math.ceil(w / 8)), Math.max(1, Math.ceil(h / 8)), { depthBuffer: false, type: THREE.UnsignedByteType });
    this.depthPx = new Uint8Array(this.depthRT.width * this.depthRT.height * 4);
    this.nearRT = this.nearOK ? new THREE.WebGLRenderTarget(Math.max(1, Math.round(rw / 4)), Math.max(1, Math.round(rh / 4)), {
      depthBuffer: true, type: THREE.HalfFloatType, generateMipmaps: true,
      minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter,
    }) : null;
    this.slots = [];
    for (let i = 0; i < SLOT_COUNT + LIVE_SLOTS; i++) {   // live views (and ones fading out) hold slots
      const depthTexture = new THREE.DepthTexture(rw, rh);
      depthTexture.type = THREE.UnsignedIntType;
      depthTexture.minFilter = THREE.NearestFilter; depthTexture.magFilter = THREE.NearestFilter;
      const rt = new THREE.WebGLRenderTarget(rw, rh, { depthBuffer: true, depthTexture, type: THREE.UnsignedByteType });
      this.slots.push({
        index: i, rt, id: -1, state: 'free', camera: new THREE.PerspectiveCamera(80, w / h, 0.1, 800),
        viewProj: new THREE.Matrix4(), pos: new THREE.Vector3(), near: 0.1, far: 800, fov: 80, sentAt: 0,
        pixels: new Uint8Array(w * h * 4), zone: 0, imgW: w, imgH: h, warp: this.warp,
      });
    }
  }

  setLink(link) {
    this.link = link;
    if (!link) return;
    link.on('info', (info) => {
      this.info = info;
      this._fit = null;   // re-fit the capture on the next update
      this.onEvent('info', info);
    });
    link.on('status', (s) => this.onEvent('status', s));
    link.on('stats', (s) => this.onEvent('stats', s));
    link.on('dropped', (id) => {
      const s = this.slots.find((x) => x.id === id && x.state === 'sent');
      if (s) { s.state = 'free'; s.id = -1; }
      this.stats.dropped++;
    });
    link.on('result', (res) => this._onResult(res));
  }

  _onResult(res) {
    const id = res.header.id;
    const slot = this.slots.find((x) => x.id === id && x.state === 'sent');
    if (!slot) return;
    slot.state = 'decoding';
    const now = performance.now();
    const lat = res.header.latency ?? (now - slot.sentAt);
    this.stats.latency = this.stats.latency ? this.stats.latency * 0.8 + lat * 0.2 : lat;
    this._resTimes.push(now);
    while (this._resTimes.length && now - this._resTimes[0] > 2000) this._resTimes.shift();
    this.stats.results++;
    if (res.texture) {
      this.pendingResults.push({ slot, texture: res.texture, flipY: res.flipY, owned: false });
      return;
    }
    this.lastResultBytes = res.bytes;
    const blob = new Blob([res.bytes], { type: 'image/jpeg' });
    createImageBitmap(blob).then((bm) => {
      const tex = new THREE.Texture(bm);
      tex.flipY = false;
      tex.colorSpace = THREE.NoColorSpace;
      tex.generateMipmaps = false;
      tex.minFilter = THREE.LinearFilter;
      tex.needsUpdate = true;
      this.pendingResults.push({ slot, texture: tex, flipY: true, owned: true, bitmap: bm });
    }).catch((e) => { console.warn('[dream] decode failed', e); slot.state = 'free'; slot.id = -1; });
  }

  // 0 standing still .. 1 walking or turning briskly (eased over ~0.5 s)
  get motion() { return this._motion?.calm ?? 0; }

  forget() {
    this.painter.clear();
    this.live.clear();
    this.memory?.forget();
    this.onEvent('forget');
  }

  // A result is held by its paint job and (centre views only) by the live layer; the slot
  // (whose depth the live layer samples) and the texture go back when both let go.
  _release(r) {
    if (--r.refs > 0) return;
    if (r.owned) { r.texture.dispose(); r.bitmap?.close?.(); }
    r.slot.state = 'free'; r.slot.id = -1;
  }

  // Capture shape and field of view for this screen. Engines that take any size
  // (info.flexible) get the shape, within their pixel budget, whose field of view covers
  // the screen plus a margin at the highest resolution: 512x320 on a 16:9 screen, 320x512
  // on a phone held upright (a 384x384 engine's budget). Fixed-size engines keep their
  // shape and only the field of view adapts, never wider than the old 96 degrees.
  _fitCapture(info, camera) {
    const ew = info.width | 0 || 512, eh = info.height | 0 || 512;
    const rad = THREE.MathUtils.degToRad, margin = rad(4);
    const tv = Math.tan(rad(camera.fov / 2) + margin);
    const th = Math.tan(Math.atan(Math.tan(rad(camera.fov / 2)) * camera.aspect) + margin);
    const fit = (w, h) => { const t = Math.max(tv, th * h / w); return { w, h, t, f: h / 2 / t }; };
    let best = fit(ew, eh);
    if (info.flexible) {
      const P = ew * eh;
      for (let w = 256; w <= 768; w += 64) {
        for (let h = 256; h <= 768; h += 64) {
          if (w * h > 1.12 * P || w * h < 0.8 * P) continue;
          const c = fit(w, h);
          if (c.f > best.f * 1.02) best = c;
        }
      }
    }
    let fov = THREE.MathUtils.radToDeg(2 * Math.atan(best.t));
    if (best.w === ew && best.h === eh) fov = Math.min(fov, 96);
    if (this.settings.captureFov > 0) fov = this.settings.captureFov;
    return { w: best.w, h: best.h, fov, aspect: camera.aspect, dispFov: camera.fov };
  }

  // Called once per frame from the main loop, before the main render.
  update(now, dt, camera, zoneIndex, zone) {
    const S = this.settings;
    const link = this.link;
    if (this.info && (!this._fit || this._fit.aspect !== camera.aspect || this._fit.dispFov !== camera.fov)) {
      this._fit = this._fitCapture(this.info, camera);
      this.live.clear();   // the live layer's depth bias assumes one capture geometry
      if (this._fit.w !== this.width || this._fit.h !== this.height) this._wantSize = [this._fit.w, this._fit.h];
    }
    // a new foveal warp (settings.warp changed at runtime) re-allocates the slots like a new capture size
    const warp = Math.min(2, Math.max(1, S.warp || 1));
    if (warp !== this.warp && !this._wantSize) this._wantSize = [this.width, this.height];
    // anything that reframes a capture without moving the camera starts a new framing
    const frame = `${this._fit?.w}x${this._fit?.h}@${this._fit?.fov}|${warp}|${S.glance}|${S.fovea}`;
    if (frame !== this._framing) { this._framing = frame; this.stats.keyframes++; }
    this._trackMotion(camera, dt);
    link?.pump?.(now);
    // turn arrived results into paint jobs
    while (this.pendingResults.length) {
      const r = this.pendingResults.shift();
      const spread = Math.max(1, S.paintSpread | 0);
      const rate = 1 - Math.pow(1 - S.paintRate, 1 / spread);
      r.refs = 1;
      this.jobs.push({ result: r, slot: r.slot, texture: r.texture, flipY: r.flipY, remaining: spread, rate });
      r.slot.state = 'painting';
      if (r.slot.side === 0 && this.live.enabled) {
        r.refs++;
        this.live.add(r.slot, r.texture, r.flipY, now, () => this._release(r));
      }
    }
    // run paint jobs (each result is applied over a few frames: no visible ticks)
    const t0 = performance.now();
    let painted = 0;
    for (const job of this.jobs) {
      if (painted >= 3) break;
      this.painter.paint(job);
      this.memory?.markPainted(this.painter.meshes);
      job.remaining--; painted++;
    }
    this.memory?.tick(now);
    if (painted) this.stats.paintMs = this.stats.paintMs * 0.9 + (performance.now() - t0) * 0.1;
    this.jobs = this.jobs.filter((job) => {
      if (job.remaining > 0) return true;
      this._release(job.result);
      return false;
    });
    this.live.update(now);
    this.stats.activity = this.stats.activity * 0.97 + (painted ? 0.03 : 0);
    // relax: the dream slowly forgets unseen places (half-life S.relaxHalfLife s)
    if (!this._lastRelax) this._lastRelax = now;
    if (now - this._lastRelax > 2000 && S.relaxHalfLife > 0) {
      const dtS = (now - this._lastRelax) / 1000;
      this.painter.relax(Math.pow(0.5, dtS / S.relaxHalfLife));
      this._lastRelax = now;
    }
    // pending resize from server info
    if (this._wantSize) {   // new capture size: drain, then re-allocate (no captures meanwhile)
      this.live.clear();
      if (this.slots.every((s) => s.state === 'free')) { this._allocSlots(...this._wantSize); this._wantSize = null; }
    }
    // stats
    const rt = this._resTimes;
    this.stats.dreamFps = rt.length > 1 ? (rt.length - 1) * 1000 / (rt[rt.length - 1] - rt[0] + 1e-3) : 0;
    if (rt.length && now - rt[rt.length - 1] > 3000) this.stats.dreamFps = 0;
    // capture?
    if (!link || !link.ready || S.paused || this._wantSize) return;
    const busy = this.slots.filter((s) => s.state === 'encoding').length + link.inFlightCount;
    if (busy >= S.maxInFlight) return;
    if (now - this.lastCapture < 1000 / S.captureRate) return;
    const slot = this.slots.find((s) => s.state === 'free');
    if (!slot) return;
    this.lastCapture = now;
    this._capture(slot, camera, zoneIndex, zone);
  }

  // Track camera motion so captures can lead it: by the time a result comes
  // back (~2 inference times) the player is looking where we aimed.
  _trackMotion(camera, dt) {
    const e = this._euler || (this._euler = new THREE.Euler(0, 0, 0, 'YXZ'));
    e.setFromQuaternion(camera.quaternion, 'YXZ');
    const m = this._motion || (this._motion = { yaw: e.y, pitch: e.x, pos: camera.position.clone(), yawVel: 0, pitchVel: 0, vel: new THREE.Vector3(), calm: 0 });
    if (dt > 0) {
      let dy = e.y - m.yaw; dy = Math.atan2(Math.sin(dy), Math.cos(dy));
      const k = 1 - Math.exp(-dt * 6);
      m.yawVel += (dy / dt - m.yawVel) * k;
      m.pitchVel += ((e.x - m.pitch) / dt - m.pitchVel) * k;
      const v = camera.position.clone().sub(m.pos).divideScalar(dt);
      if (v.lengthSq() < 400) m.vel.lerp(v, k);
      // 0 standing still .. 1 walking (3.4 m/s) or turning briskly, eased over ~0.5 s
      const moving = Math.min(1, m.vel.length() / 3.4 + Math.abs(m.yawVel) / 1.2);
      m.calm += (moving - m.calm) * (1 - Math.exp(-dt / 0.5));
    }
    m.yaw = e.y; m.pitch = e.x; m.pos.copy(camera.position);
  }

  _capture(slot, camera, zoneIndex, zone) {
    const r = this.renderer;
    const S = this.settings;
    const cam = slot.camera;
    const m = this._motion;
    // aim where the camera will be while this result is on screen: render, readback and
    // encode (encodeMs), the round trip (latency), then about half the fade-in
    const lead = Math.min(0.6, (this.stats.encodeMs + (this.stats.latency || 300)) / 1000 + 0.12) * S.lead;
    const e = new THREE.Euler(0, 0, 0, 'YXZ').setFromQuaternion(camera.quaternion, 'YXZ');
    let yaw = e.y, pitch = e.x;
    cam.position.copy(camera.position);
    if (m) {
      yaw += THREE.MathUtils.clamp(m.yawVel * lead, -0.45, 0.45);
      pitch += THREE.MathUtils.clamp(m.pitchVel * lead, -0.25, 0.25);
      cam.position.addScaledVector(m.vel, Math.min(lead, 0.4));
      // gentle saccades while idle widen coverage without fighting the view
      const idle = Math.max(0, 1 - Math.abs(m.yawVel) * 2 - m.vel.length() * 0.3);
      const t = performance.now() / 1000;
      yaw += S.saccade * idle * Math.sin(t * 1.7 + Math.sin(t * 0.37) * 2.0);
      pitch += S.saccade * 0.4 * idle * Math.sin(t * 1.13 + 1.3);
    }
    pitch = THREE.MathUtils.clamp(pitch, -1.5, 1.5);
    slot.side = 0;
    // View keyframing: hold the capture pose until the (predicted) view has
    // moved enough. Identical framings give the model identical noise/framing,
    // so repeated results agree and the EMA converges crisp instead of mushy.
    const kf = this._kf;
    if (S.kfDist > 0 && kf) {
      const dYaw = Math.atan2(Math.sin(yaw - kf.yaw), Math.cos(yaw - kf.yaw));
      const moved = cam.position.distanceTo(kf.pos);
      if (moved < S.kfDist && Math.abs(dYaw) < S.kfAngle && Math.abs(pitch - kf.pitch) < S.kfAngle * 0.8) {
        cam.position.copy(kf.pos); yaw = kf.yaw; pitch = kf.pitch;
        // Once the centre view has converged, glance sideways: fixed side
        // framings (so they converge too) paint the periphery of the screen
        // and beyond while you stand and look.
        kf.n++;
        const seq = [0, 0, 0, 1, 0, -1, 0, 1, 0, -1];
        const side = kf.n < seq.length ? seq[kf.n] : ((kf.n & 1) ? ((kf.n >> 1) & 1 ? -1 : 1) : 0);
        yaw += side * S.glance;
        slot.side = side;
      } else { kf.pos.copy(cam.position); kf.yaw = yaw; kf.pitch = pitch; kf.n = 0; this.stats.keyframes++; }
    } else if (!kf) {
      this._kf = { pos: cam.position.clone(), yaw, pitch, n: 0 };
      this.stats.keyframes++;
    }
    cam.rotation.set(pitch, yaw, 0, 'YXZ');
    // Foveated captures: a capture pixel spread over the whole screen covers ~4 screen
    // pixels across at 1080p. Centre captures alternate (at rest) or go two narrow to one
    // wide (walking); a narrow one divides tan(fov/2) by S.fovea, putting that many times
    // more pixels on the middle of the view. The live layer shows whichever view is finer
    // (live.js), so the wide ones keep the edges painted. Turning, or with no wide view
    // left in the live layer, captures stay wide: new scenery enters at the edges.
    const fov = this._fit ? this._fit.fov : (S.captureFov || 96);
    slot.narrow = false;
    if (slot.side === 0 && S.fovea > 1) {
      const walking = m && m.vel.length() > 1.7;
      const turning = m && Math.abs(m.yawVel) + Math.abs(m.pitchVel) > 0.8;
      const hasWide = this.live.views.some((v) => !v.slot.narrow);
      slot.narrow = !turning && hasWide && (this._narrowRun || 0) < (walking ? 2 : 1);
      this._narrowRun = slot.narrow ? (this._narrowRun || 0) + 1 : 0;
    }
    const rad = THREE.MathUtils.degToRad;
    cam.fov = slot.narrow ? THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(rad(fov) / 2) / S.fovea)) : fov;
    cam.aspect = this.width / this.height;
    cam.near = 0.1; cam.far = 800;
    cam.updateProjectionMatrix();
    cam.updateMatrixWorld(true);
    slot.viewProj.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    slot.pos.copy(cam.position);
    slot.near = cam.near; slot.far = cam.far; slot.fov = cam.fov;
    slot.zone = zoneIndex;
    slot.id = this.nextId++;
    // which held framing this capture belongs to (keyframing off: each capture is its own)
    slot.kfId = S.kfDist > 0 ? this.stats.keyframes : -slot.id;
    slot.state = 'encoding';
    const t0 = performance.now();
    // the capture's depth cue (a look's; off by default): its inverse depth at quarter size
    // first, which the capture shader blurs into each pixel's neighbourhood (materials.js)
    const prevOverride = this.scene.overrideMaterial;
    const CU = this.materials.capture.uniforms;
    const sepOn = this.nearRT && (S.capSep > 0 || S.edgeKeep > 0);
    const sendDepth = !!this.info?.depth;
    CU.uCapDepthSep.value = sepOn ? S.capSep : 0;
    CU.uCapEdgeKeep.value = sepOn ? S.edgeKeep : 0;
    if (sepOn) {
      this.scene.overrideMaterial = this.materials.capNear;
      r.setRenderTarget(this.nearRT);
      r.setClearColor(0x000000, 1);   // cleared = infinitely far
      r.clear(true, true, false);
      r.render(this.scene, cam);
      CU.uCapNear.value = this.nearRT.texture;   // (three builds its mipmaps after the render)
      CU.uCapRes.value.set(slot.rt.width, slot.rt.height);
    }
    // render the capture look (layer 0 only: decor is never painted)
    this.scene.overrideMaterial = this.materials.capture;
    r.setRenderTarget(slot.rt);
    r.setClearColor(0x000000, 1);
    r.clear(true, true, false);
    r.render(this.scene, cam);
    this.scene.overrideMaterial = prevOverride;
    // the image the model gets: the render itself, or its foveal warp (shared target: the
    // readback below is queued on the GPU before anything else can draw into it)
    let src = slot.rt;
    if (this.warpRT) {
      const U = this.warpMat.uniforms;
      U.uSrc.value = slot.rt.texture;
      U.uSrcTexel.value.set(1 / slot.rt.width, 1 / slot.rt.height);
      U.uM.value = this.warp;
      r.setRenderTarget(this.warpRT);
      r.render(this.warpQuad, this.warpCam);
      src = this.warpRT;
    }
    // the depth the engine paints with, read back beside the image
    let depthRead = null;
    if (sendDepth) {
      const DU = this.depthMat.uniforms, D = this.depthRT;
      DU.uDepth.value = slot.rt.depthTexture;
      DU.uDepthTexel.value.set(1 / slot.rt.width, 1 / slot.rt.height);
      DU.uNF.value.set(slot.near, slot.far);
      DU.uM.value = this.warp;
      this.warpQuad.material = this.depthMat;
      r.setRenderTarget(D);
      r.render(this.warpQuad, this.warpCam);
      this.warpQuad.material = this.warpMat;
      const px = this.depthPx;
      depthRead = (this._syncRead ? Promise.resolve(syncRead(r, D, px))
        : r.readRenderTargetPixelsAsync(D, 0, 0, D.width, D.height, px))
        .then(() => packDepth(px, D.width, D.height)).catch(() => null);
    }
    r.setRenderTarget(null);
    const id = slot.id;
    const w = this.width, h = this.height;
    // what this capture asks for is fixed now, when it was rendered, not when its encode
    // finishes: behind closed eyes (main.js) the room is re-dreamt harder with a variant in
    // front of the prompt, switched at once (cut); only a depth engine keeps the shape at 0.9
    const eyesShut = !!S.eyesClosed;
    const ask = { strength: eyesShut && this.info?.depth ? S.eyesStrength : S.strength, variant: S.promptSuffix || '', cut: eyesShut };
    let read;
    if (this._syncRead) {
      syncRead(r, src, slot.pixels);
      read = Promise.resolve(slot.pixels);
    } else {
      read = r.readRenderTargetPixelsAsync(src, 0, 0, w, h, slot.pixels).catch((e) => {
        console.warn('[dream] async readback failed, falling back to sync readPixels', e);
        this._syncRead = true;
        if (slot.id !== id) return slot.pixels;   // (dropped meanwhile: nothing is sent)
        // the shared warp target may hold a later capture by now: warp this slot's render again
        if (src === this.warpRT) {
          const U = this.warpMat.uniforms, t = r.getRenderTarget();
          U.uSrc.value = slot.rt.texture;
          U.uSrcTexel.value.set(1 / slot.rt.width, 1 / slot.rt.height);
          U.uM.value = slot.warp;
          r.setRenderTarget(this.warpRT); r.render(this.warpQuad, this.warpCam); r.setRenderTarget(t);
        }
        syncRead(r, src, slot.pixels);
        return slot.pixels;
      });
    }
    read.then(async (px) => {
      if (slot.id !== id) return;
      const [jpeg, depth] = await Promise.all([this.encoder.encode(px, w, h, S.jpegQuality), depthRead]);
      const enc = performance.now() - t0;
      this.stats.encodeMs = this.stats.encodeMs ? this.stats.encodeMs * 0.8 + enc * 0.2 : enc;
      if (slot.id !== id) return;
      let prompt = (S.promptOverride && S.promptOverride.trim()) || zone?.prompt || 'a dreamlike painting';
      // a room re-dreamt behind closed eyes: the variant goes first, where the text encoder
      // weighs words most
      if (ask.variant) prompt = ask.variant + ', ' + prompt;
      // Motion calm: moving, the model dreams a little less, so each new viewpoint keeps
      // more of the reprojected dream it is shown instead of re-inventing its detail;
      // standing still it dreams at full strength. Steps of 1/3 keep DeepCache reusing.
      const calm = Math.round((this._motion?.calm ?? 0) * 3) / 3;
      const header = {
        type: 'frame', id, width: w, height: h, prompt,
        negative: zone?.negative ?? null, strength: +(ask.strength * (1 - (S.motionCalm ?? 0) * calm)).toFixed(3),
        // each kind of capture (wide, narrow, left and right glance) has its own seed: its own
        // noise, and on engines with cross-frame attention its own anchor, so each keeps
        // following its own last picture instead of another kind's
        seed: (S.seed + zoneIndex * 7919 + (slot.narrow ? 1 : slot.side > 0 ? 2 : slot.side < 0 ? 3 : 0)) >>> 0, format: 'jpeg',
      };
      if (depth) header.depth = depth;
      if (ask.cut) header.cut = true;
      // the framing id: `fid` in every look, so a pool paints a held view on one engine; `kf` too
      // in the fresh look, so the server tells an engine whether a frame repeats the last framing
      // it painted on this stream and a new framing gets a plain pass
      if (S.kfDist > 0) {
        header.fid = slot.kfId;
        if (S.plainWalk) header.kf = slot.kfId;
      }
      if (this.link.send(header, jpeg)) {
        slot.state = 'sent'; slot.sentAt = performance.now();
        this.stats.captures++;
        this.lastJpeg = jpeg;
      } else { slot.state = 'free'; slot.id = -1; }
    }).catch((e) => { console.warn('[dream] capture failed', e); slot.state = 'free'; slot.id = -1; });
  }
}

// A synchronous read. three.js leaves its async readback's pixel-pack buffer bound until the
// read completes, and WebGL2 then turns a readPixels into an array into a silent no-op.
function syncRead(r, rt, px) {
  const gl = r.getContext();
  gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
  r.readRenderTargetPixels(rt, 0, 0, rt.width, rt.height, px);
}

// Latent-size depth (RGBA8 from DEPTHLAT_FRAG, bottom-up) -> the frame header's `depth`: relative
// inverse depth scaled so the 2nd..98th percentiles span -1..1 (near = 1, the scaling the depth
// model was trained on: per image, not absolute), one byte each, rows top first, base64.
function packDepth(px, w, h) {
  const n = w * h, iz = new Float32Array(n);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = ((h - 1 - y) * w + x) * 4;
      const v = (px[i] + px[i + 1] / 255) / 255;
      iz[y * w + x] = Math.exp(v * 9.9035 - 6.9078);
    }
  }
  const s = iz.slice().sort();
  let lo = s[Math.floor(0.02 * (n - 1))], hi = s[Math.ceil(0.98 * (n - 1))];
  // a nearly flat view (a wall faced squarely) must not stretch its rounding steps to the full
  // range: keep a span of at least 15% of the median inverse depth, centred between the percentiles
  const minSpan = 0.15 * s[n >> 1];
  if (hi - lo < minSpan) { const c = (hi + lo) / 2; lo = c - minSpan / 2; hi = c + minSpan / 2; }
  const k = 255 / Math.max(hi - lo, 1e-9);
  let str = '';
  for (let i = 0; i < n; i++) str += String.fromCharCode(Math.max(0, Math.min(255, Math.round((iz[i] - lo) * k))));
  return { w, h, data: btoa(str) };
}

// RGBA (bottom-up, from readPixels) -> JPEG bytes. Uses OffscreenCanvas when
// available; the encode itself is async in the browser.
export class JpegEncoder {
  constructor() {
    this.canvas = null; this.ctx = null; this.img = null;
  }
  _ensure(w, h) {
    if (this.canvas && this.canvas.width === w && this.canvas.height === h) return;
    if (typeof OffscreenCanvas !== 'undefined' && !this.noOffscreen) this.canvas = new OffscreenCanvas(w, h);
    else { this.canvas = document.createElement('canvas'); this.canvas.width = w; this.canvas.height = h; }
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: false });
    this.img = this.ctx.createImageData(w, h);
  }
  async encode(px, w, h, quality = 0.88) {
    this._ensure(w, h);
    const dst = this.img.data, row = w * 4;
    for (let y = 0; y < h; y++) dst.set(px.subarray((h - 1 - y) * row, (h - y) * row), y * row);
    this.ctx.putImageData(this.img, 0, 0);
    let blob = null;
    if (this.canvas.convertToBlob) {
      try { blob = await this.canvas.convertToBlob({ type: 'image/jpeg', quality }); } catch (e) {
        // some engines lack JPEG in OffscreenCanvas: fall back to a DOM canvas for good
        console.warn('[dream] OffscreenCanvas JPEG failed, using <canvas>', e);
        this.noOffscreen = true; this.canvas = null;
        return this.encode(px, w, h, quality);
      }
    } else blob = await new Promise((res) => this.canvas.toBlob(res, 'image/jpeg', quality));
    if (!blob) throw new Error('JPEG encode produced no blob');
    return new Uint8Array(await blob.arrayBuffer());
  }
}
