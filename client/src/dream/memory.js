// Dream memory: the atlas survives a reload. It is split into tiles; the paint pass marks
// the tiles it touched, and every few seconds a couple of dirty tiles are converted to
// 8-bit (straight colour + confidence), read back asynchronously, deflated and written to
// IndexedDB. On load the stored tiles are drawn back into the atlas before the first frame.
// Records are keyed by a fingerprint of the atlas layout, so a changed level never
// restores someone else's paint. Everything fails soft: no IndexedDB, no memory.
import * as THREE from 'three';

const DB = 'hypnagogia', STORE = 'atlas', VERSION = 1;
const QUAD_VERT = /* glsl */ `
varying vec2 vUv;
void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, VERSION);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
function idb(db, mode, fn) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, mode);
    const out = fn(tx.objectStore(STORE));
    tx.oncomplete = () => resolve(out?.result ?? out);
    tx.onerror = tx.onabort = () => reject(tx.error);
  });
}
async function squeeze(bytes) {
  if (typeof CompressionStream === 'undefined') return { enc: 'raw', data: bytes.buffer.slice(0) };
  const s = new Blob([bytes]).stream().pipeThrough(new CompressionStream('deflate'));
  return { enc: 'deflate', data: await new Response(s).arrayBuffer() };
}
async function unsqueeze(rec) {
  if (rec.enc === 'raw') return new Uint8Array(rec.data);
  const s = new Blob([rec.data]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(s).arrayBuffer());
}

// FNV-1a over the atlas layout: seed, size and a stride through the atlasUv attribute.
function layoutKey(level, size) {
  const g = level.geometries ? level.geometries[0] : level.geometry;
  const uv = g.attributes.atlasUv.array;
  let h = 0x811c9dc5;
  const step = Math.max(1, Math.floor(uv.length / 4096));
  for (let i = 0; i < uv.length; i += step) { h ^= Math.round(uv[i] * size * 8); h = Math.imul(h, 0x01000193) >>> 0; }
  return `s${level.seed}-a${size}-n${uv.length}-${h.toString(16)}`;
}

export class DreamMemory {
  constructor(renderer, painter, level, { tile = 512, interval = 3000, perTick = 2 } = {}) {
    this.renderer = renderer;
    this.painter = painter;
    this.size = painter.size;
    this.tile = Math.min(tile, this.size);
    this.n = this.size / this.tile;               // tiles per side
    this.interval = interval;
    this.perTick = perTick;
    this.key = layoutKey(level, this.size);
    this.dirty = new Set();
    this.busy = false;
    this.last = 0;
    this.db = null;
    this.gen = 0;          // bumped by forget(): a save that started before it must not land
    this.flushLeft = 0;    // tiles still to save right away (flush() snapshots the dirty count)
    this.failures = 0;     // consecutive failed saves (quota...): back off, up to 64x the interval
    this.stats = { restored: 0, saved: 0, bytes: 0 };
    this._tilesPerMesh(painter.meshes);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 2, 0, 0, 2], 2));
    this.quad = new THREE.Mesh(geo);
    this.quad.frustumCulled = false;
    this.cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    // atlas tile -> 8-bit straight colour + confidence
    this.packMat = new THREE.ShaderMaterial({
      uniforms: { uAtlas: { value: painter.texture }, uOrigin: { value: new THREE.Vector2() }, uScale: { value: 1 / this.n } },
      vertexShader: QUAD_VERT,
      fragmentShader: /* glsl */ `
        uniform sampler2D uAtlas; uniform vec2 uOrigin; uniform float uScale; varying vec2 vUv;
        void main(){ vec4 a = textureLod(uAtlas, uOrigin + vUv * uScale, 0.0);
          gl_FragColor = vec4(a.a > 1e-4 ? clamp(a.rgb / a.a, 0.0, 1.0) : vec3(0.0), clamp(a.a, 0.0, 1.0)); }`,
      depthTest: false, depthWrite: false,
    });
    this.packRT = new THREE.WebGLRenderTarget(this.tile, this.tile, { type: THREE.UnsignedByteType, depthBuffer: false });
    this.buf = new Uint8Array(this.tile * this.tile * 4);
    // stored tile -> premultiplied atlas texels, drawn over exactly that tile
    this.unpackMat = new THREE.ShaderMaterial({
      uniforms: { uTile: { value: null } },
      vertexShader: QUAD_VERT,
      fragmentShader: /* glsl */ `
        uniform sampler2D uTile; varying vec2 vUv;
        void main(){ vec4 c = texture(uTile, vUv); gl_FragColor = vec4(c.rgb * c.a, c.a); }`,
      depthTest: false, depthWrite: false, blending: THREE.NoBlending,
    });
  }

  // Which tiles each paint chunk can touch (its triangles' atlas bounding boxes).
  _tilesPerMesh(meshes) {
    for (const m of meshes) {
      const g = m.geometry, idx = g.index.array, uv = g.attributes.atlasUv.array;
      const set = new Set();
      const end = g.drawRange.start + g.drawRange.count;
      for (let i = g.drawRange.start; i < end; i += 3) {
        let u0 = 1, v0 = 1, u1 = 0, v1 = 0;
        for (let k = 0; k < 3; k++) {
          const j = idx[i + k] * 2;
          u0 = Math.min(u0, uv[j]); u1 = Math.max(u1, uv[j]); v0 = Math.min(v0, uv[j + 1]); v1 = Math.max(v1, uv[j + 1]);
        }
        const tx0 = Math.max(0, Math.floor(u0 * this.n)), tx1 = Math.min(this.n - 1, Math.floor(u1 * this.n));
        const ty0 = Math.max(0, Math.floor(v0 * this.n)), ty1 = Math.min(this.n - 1, Math.floor(v1 * this.n));
        for (let ty = ty0; ty <= ty1; ty++) for (let tx = tx0; tx <= tx1; tx++) set.add(ty * this.n + tx);
      }
      m.userData.tiles = [...set];
    }
  }

  // After a paint pass: every visible chunk's tiles are now dirty.
  markPainted(meshes) {
    for (const m of meshes) if (m.visible) for (const t of m.userData.tiles) this.dirty.add(t);
  }

  async open() {
    if (typeof indexedDB === 'undefined') return false;
    try { this.db = await openDb(); } catch (e) { console.warn('[memory] IndexedDB unavailable', e); return false; }
    // dreams of other layouts (seeds, older builds) are kept a month, then dropped
    const old = Date.now() - 30 * 24 * 3600 * 1000;
    idb(this.db, 'readwrite', (s) => {
      const req = s.openCursor();
      req.onsuccess = () => { const c = req.result; if (!c) return; if (!(c.value?.t > old)) c.delete(); c.continue(); };
    }).catch(() => {});
    return true;
  }

  // Draw every stored tile of this layout back into the atlas.
  async restore() {
    if (!this.db) return 0;
    const prefix = this.key + '/';
    let recs;
    try {
      recs = await idb(this.db, 'readonly', (s) => {
        const out = [];
        const req = s.openCursor(IDBKeyRange.bound(prefix, prefix + '\uffff'));
        req.onsuccess = () => { const c = req.result; if (c) { out.push([c.key, c.value]); c.continue(); } };
        return out;
      });
    } catch (e) { console.warn('[memory] restore failed', e); return 0; }
    const r = this.renderer, prev = r.getRenderTarget(), atlas = this.painter.atlas;
    const tex = new THREE.DataTexture(null, this.tile, this.tile, THREE.RGBAFormat, THREE.UnsignedByteType);
    tex.minFilter = tex.magFilter = THREE.NearestFilter;
    tex.generateMipmaps = false;
    this.unpackMat.uniforms.uTile.value = tex;
    this.quad.material = this.unpackMat;
    for (const [key, rec] of recs) {
      const t = +key.slice(prefix.length);
      if (!(t >= 0 && t < this.n * this.n) || rec.tile !== this.tile) continue;
      let bytes;
      try { bytes = await unsqueeze(rec); } catch { continue; }
      if (bytes.length !== this.tile * this.tile * 4) continue;
      tex.image = { data: bytes, width: this.tile, height: this.tile };
      tex.needsUpdate = true;
      const tx = t % this.n, ty = Math.floor(t / this.n);
      atlas.viewport.set(tx * this.tile, ty * this.tile, this.tile, this.tile);   // GL rows: v = 0 at the bottom
      r.setRenderTarget(atlas);
      r.render(this.quad, this.cam);
      this.stats.restored++;
    }
    atlas.viewport.set(0, 0, this.size, this.size);
    r.setRenderTarget(prev);
    tex.dispose();
    return this.stats.restored;
  }

  // Call once per frame; saves a couple of dirty tiles every `interval` ms (backing off
  // after failures), or right away what flush() asked for.
  tick(now) {
    if (!this.db || this.busy || !this.dirty.size) return;
    const flushing = this.flushLeft > 0;
    if (!flushing && now - this.last < this.interval * (1 << Math.min(this.failures, 6))) return;
    this.last = now;
    this.busy = true;
    const tiles = [...this.dirty].slice(0, flushing ? this.flushLeft : this.perTick);
    this.flushLeft = Math.max(0, this.flushLeft - tiles.length);
    for (const t of tiles) this.dirty.delete(t);
    this._save(tiles).then(() => { this.failures = 0; }, (e) => {
      if (++this.failures === 3) console.warn('[memory] saves keep failing; backing off', e);
      this.flushLeft = 0;
      for (const t of tiles) this.dirty.add(t);
    }).finally(() => { this.busy = false; if (this.flushLeft) this.tick(performance.now()); });
  }

  async _save(tiles) {
    const r = this.renderer, gen = this.gen;
    for (const t of tiles) {
      const tx = t % this.n, ty = Math.floor(t / this.n);
      this.packMat.uniforms.uOrigin.value.set(tx / this.n, ty / this.n);
      this.quad.material = this.packMat;
      const prev = r.getRenderTarget();
      r.setRenderTarget(this.packRT);
      r.render(this.quad, this.cam);
      r.setRenderTarget(prev);
      if (this.syncRead) r.readRenderTargetPixels(this.packRT, 0, 0, this.tile, this.tile, this.buf);
      else {
        try { await r.readRenderTargetPixelsAsync(this.packRT, 0, 0, this.tile, this.tile, this.buf); } catch (e) {
          console.warn('[memory] async readback failed, using readPixels', e);
          this.syncRead = true;
          r.readRenderTargetPixels(this.packRT, 0, 0, this.tile, this.tile, this.buf);
        }
      }
      const { enc, data } = await squeeze(this.buf);
      if (gen !== this.gen) return;   // forgotten while this was in flight
      await idb(this.db, 'readwrite', (s) => s.put({ enc, data, tile: this.tile, t: Date.now() }, `${this.key}/${t}`));
      this.stats.saved++;
      this.stats.bytes += data.byteLength;
    }
  }

  // The page is being hidden (tab switch, app switch on a phone): save everything dirty,
  // after any save in flight. Best effort: a page that is really closing may not finish.
  flush() {
    if (!this.db) return;
    this.flushLeft = this.dirty.size;
    this.tick(performance.now());
  }

  // The player asked the dream to forget: drop this layout's records too.
  async forget() {
    this.dirty.clear();
    this.flushLeft = 0;
    this.gen++;
    if (!this.db) return;
    const prefix = this.key + '/';
    try { await idb(this.db, 'readwrite', (s) => s.delete(IDBKeyRange.bound(prefix, prefix + '\uffff'))); } catch (e) { console.warn('[memory] forget failed', e); }
  }
}
