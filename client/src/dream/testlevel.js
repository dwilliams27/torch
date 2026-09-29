// A small but dramatic stand-in level that satisfies the Level contract
// (docs/DESIGN.md). Used for ?level=test and as a fallback when the world
// module is missing/broken. Two zones: an open-roofed nave and a tall library.
import * as THREE from 'three';

class Builder {
  constructor(atlasSize, tpm, pad) {
    this.size = atlasSize; this.tpm = tpm; this.pad = pad;
    this.quads = [];
  }
  // o, u, v: arrays [x,y,z]; normal = normalize(u x v)
  quad(o, u, v, surface, zone, density = 1) {
    this.quads.push({ o, u, v, surface, zone, density });
  }
  // vertical wall from (x0,z0) to (x1,z1); normal = (dir) x up
  wall(x0, z0, x1, z1, y0, y1, surface, zone, density) {
    this.quad([x0, y0, z0], [x1 - x0, 0, z1 - z0], [0, y1 - y0, 0], surface, zone, density);
  }
  // wall with a rectangular opening at [a,b] (distance along wall) up to height h (relative to y0)
  wallDoor(x0, z0, x1, z1, y0, y1, a, b, h, surface, zone) {
    const L = Math.hypot(x1 - x0, z1 - z0), dx = (x1 - x0) / L, dz = (z1 - z0) / L;
    const P = (s) => [x0 + dx * s, z0 + dz * s];
    const [ax, az] = P(a), [bx, bz] = P(b);
    if (a > 0) this.wall(x0, z0, ax, az, y0, y1, surface, zone);
    if (b < L) this.wall(bx, bz, x1, z1, y0, y1, surface, zone);
    if (y0 + h < y1) this.wall(ax, az, bx, bz, y0 + h, y1, surface, zone);
  }
  floor(x0, z0, x1, z1, y, surface, zone, up = true) {
    if (up) this.quad([x0, y, z0], [0, 0, z1 - z0], [x1 - x0, 0, 0], surface, zone);
    else this.quad([x0, y, z0], [x1 - x0, 0, 0], [0, 0, z1 - z0], surface, zone);
  }
  // solid box, outward faces (no bottom unless asked)
  box(x0, y0, z0, x1, y1, z1, surface, zone, bottom = false) {
    this.floor(x0, z0, x1, z1, y1, surface, zone, true);
    if (bottom) this.floor(x0, z0, x1, z1, y0, surface, zone, false);
    this.wall(x1, z0, x0, z0, y0, y1, surface, zone); // -z
    this.wall(x0, z1, x1, z1, y0, y1, surface, zone); // +z
    this.wall(x0, z0, x0, z1, y0, y1, surface, zone); // -x
    this.wall(x1, z1, x1, z0, y0, y1, surface, zone); // +x
  }
  // room interior (inward faces). opts: {ceiling:bool, floorSurf, wallSurf, ceilSurf}
  room(x0, y0, z0, x1, y1, z1, s, zone, opts = {}) {
    this.floor(x0, z0, x1, z1, y0, opts.floorSurf ?? s, zone, true);
    if (opts.ceiling !== false) this.floor(x0, z0, x1, z1, y1, opts.ceilSurf ?? s, zone, false);
    const w = opts.wallSurf ?? s;
    const doors = opts.doors || {};
    const side = (key, ax, az, bx, bz) => {
      const d = doors[key];
      if (d) this.wallDoor(ax, az, bx, bz, y0, y1, d[0], d[1], d[2], w, zone);
      else this.wall(ax, az, bx, bz, y0, y1, w, zone);
    };
    side('s', x0, z0, x1, z0);
    side('n', x1, z1, x0, z1);
    side('w', x0, z1, x0, z0);
    side('e', x1, z0, x1, z1);
  }
  build() {
    const { size, pad } = this;
    let tpm = this.tpm;
    // shelf packing with shrink-on-overflow
    let placed = null;
    for (let attempt = 0; attempt < 12 && !placed; attempt++) {
      placed = this._pack(tpm);
      if (!placed) tpm *= 0.88;
    }
    if (!placed) throw new Error('testlevel: atlas overflow');
    this.tpmFinal = tpm;
    const n = this.quads.length;
    const pos = new Float32Array(n * 12), nor = new Float32Array(n * 12), uv = new Float32Array(n * 8);
    const auv = new Float32Array(n * 8), surf = new Float32Array(n * 4), zon = new Float32Array(n * 4);
    const idx = new Uint32Array(n * 6);
    const U = new THREE.Vector3(), V = new THREE.Vector3(), N = new THREE.Vector3(), O = new THREE.Vector3(), P = new THREE.Vector3();
    this.quads.forEach((q, i) => {
      O.fromArray(q.o); U.fromArray(q.u); V.fromArray(q.v);
      N.crossVectors(U, V).normalize();
      const uh = U.clone().normalize(), vh = V.clone().normalize();
      const c = placed[i];
      const corners = [[0, 0], [1, 0], [1, 1], [0, 1]];
      corners.forEach(([s, t], k) => {
        P.copy(O).addScaledVector(U, s).addScaledVector(V, t);
        const vi = i * 4 + k;
        P.toArray(pos, vi * 3); N.toArray(nor, vi * 3);
        uv[vi * 2] = P.dot(uh); uv[vi * 2 + 1] = P.dot(vh);
        auv[vi * 2] = (c.x + s * c.w) / size; auv[vi * 2 + 1] = (c.y + t * c.h) / size;
        surf[vi] = q.surface; zon[vi] = q.zone;
      });
      idx.set([i * 4, i * 4 + 1, i * 4 + 2, i * 4, i * 4 + 2, i * 4 + 3], i * 6);
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    g.setAttribute('atlasUv', new THREE.BufferAttribute(auv, 2));
    g.setAttribute('surface', new THREE.BufferAttribute(surf, 1));
    g.setAttribute('zone', new THREE.BufferAttribute(zon, 1));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.computeBoundingSphere(); g.computeBoundingBox();
    return g;
  }
  _pack(tpm) {
    const { size, pad } = this;
    const items = this.quads.map((q, i) => {
      const lu = Math.hypot(...q.u), lv = Math.hypot(...q.v);
      const d = tpm * q.density;
      return { i, w: Math.max(2, Math.ceil(lu * d)), h: Math.max(2, Math.ceil(lv * d)) };
    });
    const order = items.slice().sort((a, b) => b.h - a.h);
    const out = new Array(items.length);
    let x = pad, y = pad, rowH = 0;
    for (const it of order) {
      if (it.w + 2 * pad > size || it.h + 2 * pad > size) return null;
      if (x + it.w + pad > size) { x = pad; y += rowH + 2 * pad; rowH = 0; }
      if (y + it.h + pad > size) return null;
      out[it.i] = { x, y, w: it.w, h: it.h };
      x += it.w + 2 * pad;
      rowH = Math.max(rowH, it.h);
    }
    return out;
  }
}

export function generateTestLevel(seed = 1, opts = {}) {
  const atlasSize = opts.atlasSize || 4096;
  const B = new Builder(atlasSize, 24, 4);
  const S = { stone: 0, tile: 1, plaster: 2, wood: 3, crystal: 4, water: 5, sky: 6, glow: 7, brick: 8, metal: 9 };
  const surfaceTypes = [
    { name: 'nave stone', color: [0.62, 0.66, 0.74], pattern: 'stone', emissive: 0 },
    { name: 'floor tile', color: [0.55, 0.6, 0.66], pattern: 'tile', emissive: 0 },
    { name: 'plaster', color: [0.86, 0.78, 0.66], pattern: 'plaster', emissive: 0 },
    { name: 'shelf wood', color: [0.55, 0.34, 0.18], pattern: 'wood', emissive: 0 },
    { name: 'crystal', color: [0.75, 0.45, 0.95], pattern: 'crystal', emissive: 0.35 },
    { name: 'water', color: [0.18, 0.4, 0.5], pattern: 'water', emissive: 0.1 },
    { name: 'sky', color: [0.3, 0.35, 0.6], pattern: 'sky', emissive: 0 },
    { name: 'glow strip', color: [1.0, 0.7, 0.35], pattern: 'glow', emissive: 1 },
    { name: 'brick', color: [0.62, 0.36, 0.28], pattern: 'brick', emissive: 0 },
    { name: 'bronze', color: [0.6, 0.46, 0.3], pattern: 'metal', emissive: 0 },
  ];
  const zones = [
    { id: 'nave', name: 'The Drowned Nave', subtitle: 'where the tide keeps the hymns',
      prompt: 'a vast drowned gothic cathedral nave, moonlight through water, bioluminescent algae on stone pillars, volumetric light shafts, oil painting, luminous, dreamlike, highly detailed',
      negative: 'text, watermark, blurry, lowres',
      fog: [0.05, 0.1, 0.16], fogDensity: 0.018, light: [0.45, 0.8, 1.0], ambient: [0.1, 0.14, 0.2], sky: [0.2, 0.3, 0.55], mood: 'drowned choir' },
    { id: 'library', name: 'The Amber Library', subtitle: 'every book is the same afternoon',
      prompt: 'an endless vertical library, amber lamplight, towering bookshelves, brass railings, warm dust in the air, baroque painting, rich color, dreamlike',
      negative: 'text, watermark, blurry, lowres',
      fog: [0.16, 0.09, 0.04], fogDensity: 0.025, light: [1.0, 0.7, 0.35], ambient: [0.2, 0.13, 0.07], sky: [0.5, 0.3, 0.2], mood: 'warm dust' },
  ];

  // --- the nave (zone 0): open to the sky
  B.room(-12, 0, 0, 12, 16, 48, S.stone, 0, { ceiling: false, floorSurf: S.tile, doors: { n: [10.5, 13.5, 4.5] } });
  // water channel down the middle
  B.floor(-3, 4, 3, 38, 0.02, S.water, 0);
  // pillars
  for (let z = 6; z <= 36; z += 6) {
    for (const x of [-7, 7]) {
      B.box(x - 0.7, 0, z - 0.7, x + 0.7, 12, z + 0.7, S.stone, 0);
      B.box(x - 1.0, 12, z - 1.0, x + 1.0, 12.6, z + 1.0, S.brick, 0);
    }
  }
  // clerestory bridges between pillar tops
  B.box(-8, 12.6, 17.2, 8, 13.1, 18.8, S.metal, 0, true);
  B.box(-8, 12.6, 29.2, 8, 13.1, 30.8, S.metal, 0, true);
  // dais with steps
  for (let i = 0; i < 5; i++) B.box(-6 + i * 0.4, 0, 38 + i * 0.4, 6 - i * 0.4, 0.3 * (i + 1), 48, S.tile, 0);
  // floating monolith
  B.box(-1.2, 5, 43, 1.2, 11, 45, S.crystal, 0, true);
  // glow strips on the walls
  for (let z = 3; z < 46; z += 6) {
    B.quad([-11.98, 2.0, z + 1.6], [0, 0, -1.6], [0, 0.18, 0], S.glow, 0);
    B.quad([11.98, 2.0, z], [0, 0, 1.6], [0, 0.18, 0], S.glow, 0);
  }
  // --- corridor
  B.room(-1.5, 1.5, 48, 1.5, 4.5, 62, S.brick, 1, { floorSurf: S.wood, doors: { s: [0, 3, 3], n: [0, 3, 3] } });
  // --- library (zone 1)
  B.room(-10, 1.5, 62, 10, 26, 82, S.plaster, 1, { floorSurf: S.wood, doors: { s: [8.5, 11.5, 3] } });
  // shelves along walls, three storeys
  for (let lvl = 0; lvl < 3; lvl++) {
    const y0 = 1.5 + lvl * 7;
    for (let z = 64; z < 80; z += 4) {
      B.box(-10, y0, z, -9.2, y0 + 5.5, z + 3.2, S.wood, 1);
      B.box(9.2, y0, z, 10, y0 + 5.5, z + 3.2, S.wood, 1);
    }
    if (lvl > 0) {
      // mezzanine walkways
      B.box(-10, y0 - 0.3, 62, -7, y0, 82, S.wood, 1, true);
      B.box(7, y0 - 0.3, 62, 10, y0, 82, S.wood, 1, true);
    }
  }
  // bridge across the library
  B.box(-7, 8.2, 71, 7, 8.5, 73, S.metal, 1, true);
  // central crystal lamp
  B.box(-0.8, 12, 71.2, 0.8, 16, 72.8, S.crystal, 1, true);
  // --- sky box
  const R = 300, H = 160;
  B.quad([-R, -20, -R + 24], [2 * R, 0, 0], [0, H, 0], S.sky, 0, 0.02);
  B.quad([R, -20, R + 24], [-2 * R, 0, 0], [0, H, 0], S.sky, 0, 0.02);
  B.quad([-R, -20, R + 24], [0, 0, -2 * R], [0, H, 0], S.sky, 0, 0.02);
  B.quad([R, -20, -R + 24], [0, 0, 2 * R], [0, H, 0], S.sky, 0, 0.02);
  B.quad([-R, H - 20, -R + 24], [2 * R, 0, 0], [0, 0, 2 * R], S.sky, 0, 0.02);

  const geometry = B.build();
  const lights = [];
  for (let z = 6; z <= 36; z += 6) {
    lights.push({ position: [-5.8, 3.2, z], color: [1.0, 0.6, 0.3], intensity: 6, radius: 10, flicker: true });
    lights.push({ position: [5.8, 3.2, z], color: [1.0, 0.6, 0.3], intensity: 6, radius: 10, flicker: true });
  }
  lights.push({ position: [0, 8, 44], color: [0.6, 0.4, 1.0], intensity: 14, radius: 16, flicker: false });
  lights.push({ position: [0, 1, 20], color: [0.2, 0.7, 1.0], intensity: 6, radius: 14, flicker: false });
  lights.push({ position: [0, 3.5, 55], color: [1.0, 0.75, 0.4], intensity: 4, radius: 7, flicker: true });
  lights.push({ position: [0, 14, 72], color: [1.0, 0.6, 0.9], intensity: 16, radius: 18, flicker: false });
  lights.push({ position: [-6, 4, 66], color: [1.0, 0.65, 0.3], intensity: 6, radius: 9, flicker: true });
  lights.push({ position: [6, 4, 78], color: [1.0, 0.65, 0.3], intensity: 6, radius: 9, flicker: true });
  lights.push({ position: [0, 20, 72], color: [1.0, 0.8, 0.5], intensity: 10, radius: 14, flicker: false });

  return {
    seed,
    atlas: { size: atlasSize, texelsPerMeter: B.tpmFinal },
    geometry,
    surfaceTypes,
    zones,
    zoneAt: (p) => (p.z > 50 ? 1 : 0),
    lights,
    spawn: { position: [0, 1.6, 3], yaw: Math.PI }, // yaw 0 looks toward -z (three.js default)
    collision: null,
    decor: null,
    isTestLevel: true,
  };
}
