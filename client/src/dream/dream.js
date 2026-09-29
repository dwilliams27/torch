// Dream controller: schedules captures, encodes them off the critical path,
// talks to the link, and turns results into (spread-out) paint jobs.
import * as THREE from 'three';

const SLOT_COUNT = 5;

export class Dream {
  constructor({ renderer, scene, level, materials, painter, link, settings, onEvent }) {
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
    this._allocSlots(512, 512);
    this.setLink(link);
  }

  _allocSlots(w, h) {
    for (const s of this.slots) { s.rt.dispose(); s.rt.depthTexture?.dispose(); }
    this.width = w; this.height = h;
    this.slots = [];
    for (let i = 0; i < SLOT_COUNT; i++) {
      const depthTexture = new THREE.DepthTexture(w, h);
      depthTexture.type = THREE.UnsignedIntType;
      depthTexture.minFilter = THREE.NearestFilter; depthTexture.magFilter = THREE.NearestFilter;
      const rt = new THREE.WebGLRenderTarget(w, h, { depthBuffer: true, depthTexture, type: THREE.UnsignedByteType });
      this.slots.push({
        index: i, rt, id: -1, state: 'free', camera: new THREE.PerspectiveCamera(80, w / h, 0.1, 800),
        viewProj: new THREE.Matrix4(), pos: new THREE.Vector3(), near: 0.1, far: 800, fov: 80, sentAt: 0,
        pixels: new Uint8Array(w * h * 4), zone: 0,
      });
    }
  }

  setLink(link) {
    this.link = link;
    if (!link) return;
    link.on('info', (info) => {
      const w = info.width | 0 || 512, h = info.height | 0 || 512;
      if (w !== this.width || h !== this.height) {
        // only re-allocate when nothing is in flight to avoid tearing slots
        if (this.slots.every((s) => s.state === 'free')) this._allocSlots(w, h);
        else this._wantSize = [w, h];
      }
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

  forget() {
    this.painter.clear();
    this.onEvent('forget');
  }

  // Called once per frame from the main loop, before the main render.
  update(now, dt, camera, zoneIndex, zone) {
    const S = this.settings;
    const link = this.link;
    this._trackMotion(camera, dt);
    link?.pump?.(now);
    // turn arrived results into paint jobs
    while (this.pendingResults.length) {
      const r = this.pendingResults.shift();
      const spread = Math.max(1, S.paintSpread | 0);
      const rate = 1 - Math.pow(1 - S.paintRate, 1 / spread);
      this.jobs.push({ ...r, remaining: spread, rate });
      r.slot.state = 'painting';
    }
    // run paint jobs (each result is applied over a few frames: no visible ticks)
    const t0 = performance.now();
    let painted = 0;
    for (const job of this.jobs) {
      if (painted >= 3) break;
      this.painter.paint(job);
      job.remaining--; painted++;
    }
    if (painted) this.stats.paintMs = this.stats.paintMs * 0.9 + (performance.now() - t0) * 0.1;
    this.jobs = this.jobs.filter((job) => {
      if (job.remaining > 0) return true;
      if (job.owned) { job.texture.dispose(); job.bitmap?.close?.(); }
      job.slot.state = 'free'; job.slot.id = -1;
      return false;
    });
    this.stats.activity = this.stats.activity * 0.97 + (painted ? 0.03 : 0);
    // relax: the dream slowly forgets unseen places (half-life S.relaxHalfLife s)
    if (!this._lastRelax) this._lastRelax = now;
    if (now - this._lastRelax > 2000 && S.relaxHalfLife > 0) {
      const dtS = (now - this._lastRelax) / 1000;
      this.painter.relax(Math.pow(0.5, dtS / S.relaxHalfLife));
      this._lastRelax = now;
    }
    // pending resize from server info
    if (this._wantSize && this.slots.every((s) => s.state === 'free')) { this._allocSlots(...this._wantSize); this._wantSize = null; }
    // stats
    const rt = this._resTimes;
    this.stats.dreamFps = rt.length > 1 ? (rt.length - 1) * 1000 / (rt[rt.length - 1] - rt[0] + 1e-3) : 0;
    if (rt.length && now - rt[rt.length - 1] > 3000) this.stats.dreamFps = 0;
    // capture?
    if (!link || !link.ready || S.paused) return;
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
    const m = this._motion || (this._motion = { yaw: e.y, pitch: e.x, pos: camera.position.clone(), yawVel: 0, pitchVel: 0, vel: new THREE.Vector3() });
    if (dt > 0) {
      let dy = e.y - m.yaw; dy = Math.atan2(Math.sin(dy), Math.cos(dy));
      const k = 1 - Math.exp(-dt * 6);
      m.yawVel += (dy / dt - m.yawVel) * k;
      m.pitchVel += ((e.x - m.pitch) / dt - m.pitchVel) * k;
      const v = camera.position.clone().sub(m.pos).divideScalar(dt);
      if (v.lengthSq() < 400) m.vel.lerp(v, k);
    }
    m.yaw = e.y; m.pitch = e.x; m.pos.copy(camera.position);
  }

  _capture(slot, camera, zoneIndex, zone) {
    const r = this.renderer;
    const S = this.settings;
    const cam = slot.camera;
    const m = this._motion;
    const lead = Math.min(0.6, (this.stats.latency || 300) / 1000) * S.lead;
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
    }
    cam.rotation.set(pitch, yaw, 0, 'YXZ');
    cam.fov = S.captureFov;
    cam.aspect = this.width / this.height;
    cam.near = 0.1; cam.far = 800;
    cam.updateProjectionMatrix();
    cam.updateMatrixWorld(true);
    slot.viewProj.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    slot.pos.copy(cam.position);
    slot.near = cam.near; slot.far = cam.far; slot.fov = cam.fov;
    slot.zone = zoneIndex;
    slot.id = this.nextId++;
    slot.state = 'encoding';
    const t0 = performance.now();
    // render the capture look (layer 0 only: decor is never painted)
    const prevOverride = this.scene.overrideMaterial;
    this.scene.overrideMaterial = this.materials.capture;
    r.setRenderTarget(slot.rt);
    r.setClearColor(0x000000, 1);
    r.clear(true, true, false);
    r.render(this.scene, cam);
    r.setRenderTarget(null);
    this.scene.overrideMaterial = prevOverride;
    const id = slot.id;
    const w = this.width, h = this.height;
    const read = this._syncRead
      ? Promise.resolve().then(() => { r.readRenderTargetPixels(slot.rt, 0, 0, w, h, slot.pixels); return slot.pixels; })
      : r.readRenderTargetPixelsAsync(slot.rt, 0, 0, w, h, slot.pixels).catch((e) => {
        console.warn('[dream] async readback failed, falling back to sync readPixels', e);
        this._syncRead = true;
        r.readRenderTargetPixels(slot.rt, 0, 0, w, h, slot.pixels);
        return slot.pixels;
      });
    read.then(async (px) => {
      if (slot.id !== id) return;
      const jpeg = await this.encoder.encode(px, w, h, S.jpegQuality);
      this.stats.encodeMs = this.stats.encodeMs * 0.8 + (performance.now() - t0) * 0.2;
      if (slot.id !== id) return;
      const prompt = (S.promptOverride && S.promptOverride.trim()) || zone?.prompt || 'a dreamlike painting';
      const header = {
        type: 'frame', id, width: w, height: h, prompt,
        negative: zone?.negative ?? null, strength: S.strength,
        seed: (S.seed + zoneIndex * 7919) >>> 0, format: 'jpeg',
      };
      if (this.link.send(header, jpeg)) {
        slot.state = 'sent'; slot.sentAt = performance.now();
        this.stats.captures++;
        this.lastJpeg = jpeg;
      } else { slot.state = 'free'; slot.id = -1; }
    }).catch((e) => { console.warn('[dream] capture failed', e); slot.state = 'free'; slot.id = -1; });
  }
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
