// generateLevel(seed, {atlasSize}) -> Level (see docs/DESIGN.md "Client contracts").
// Geometry/atlas/collision are produced by levelcore.js (pure JS, node-testable); this wraps
// them into THREE objects and adds non-painted animated decor (drifting motes per zone).
import * as THREE from 'three';
import { buildLevelData, zoneIndexAt } from './levelcore.js';

export { buildLevelData };

export function generateLevel(seed = 1, opts = {}) {
  if (typeof location !== 'undefined' && opts.atlasSize == null) {
    const q = new URLSearchParams(location.search);
    if (q.get('atlas')) opts = { ...opts, atlasSize: parseInt(q.get('atlas'), 10) || 4096 };
  }
  const d = buildLevelData(seed, opts);
  const F = d.flat;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(F.position, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(F.normal, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(F.uv, 2));
  g.setAttribute('atlasUv', new THREE.BufferAttribute(F.atlasUv, 2));
  g.setAttribute('surface', new THREE.BufferAttribute(F.surface, 1));
  g.setAttribute('zone', new THREE.BufferAttribute(F.zone, 1));
  g.setIndex(new THREE.BufferAttribute(F.index, 1));
  g.computeBoundingBox();
  g.computeBoundingSphere();

  const zoneVols = d.zoneVols;
  const zoneAt = (p) => zoneIndexAt(zoneVols, p.x, p.y, p.z);
  const decor = makeMotes(d, seed);

  return {
    seed: d.seed,
    atlas: { size: d.atlas.size, texelsPerMeter: d.atlas.texelsPerMeter, fill: d.atlas.fill, charts: d.atlas.charts, pad: d.atlas.pad },
    geometry: g,
    surfaceTypes: d.surfaceTypes,
    zones: d.zones,
    zoneAt,
    lights: d.lights,
    spawn: d.spawn,
    collision: d.collision,
    decor,
    tour: d.tour,
    stats: { triangles: F.triCount, vertices: F.vertexCount, lights: d.lights.length, buildMs: d.buildMs },
  };
}

// Soft drifting light motes (dust, spores, sparks) in each zone. Not painted; additive.
function makeMotes(d, seed) {
  let s = (seed * 2654435761) >>> 0;
  const rnd = () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const perZone = [260, 520, 700, 520, 420, 520, 380, 520];
  const pos = [], col = [], ph = [];
  for (let zi = 0; zi < d.zones.length; zi++) {
    const z = d.zones[zi];
    const vols = d.zoneVols.filter((v) => v.zone === zi && v.prio >= 0);
    if (!vols.length) continue;
    const v = vols.reduce((a, b) => ((a.max[0] - a.min[0]) * (a.max[2] - a.min[2]) > (b.max[0] - b.min[0]) * (b.max[2] - b.min[2]) ? a : b));
    // clamp huge volumes (void / desert) to a sensible region
    const mn = v.min.slice(), mx = v.max.slice();
    if (zi === 0) { mn[0] = -14; mx[0] = 14; mn[1] = -6; mx[1] = 12; mn[2] = 2; mx[2] = 52; }
    if (zi === 7) { mn[0] = -60; mx[0] = 60; mn[1] = 0.5; mx[1] = 30; mn[2] = -420; mx[2] = -180; }
    for (let i = 0; i < perZone[zi]; i++) {
      pos.push(mn[0] + (mx[0] - mn[0]) * rnd(), mn[1] + (mx[1] - mn[1]) * rnd(), mn[2] + (mx[2] - mn[2]) * rnd());
      const c = z.light, w = 0.55 + 0.45 * rnd();
      col.push(c[0] * w, c[1] * w, c[2] * w);
      ph.push(rnd() * 100);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  g.setAttribute('phase', new THREE.Float32BufferAttribute(ph, 1));
  const mat = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uScale: { value: 400 } },
    vertexShader: /* glsl */`
      attribute vec3 color; attribute float phase;
      uniform float uTime; uniform float uScale;
      varying vec3 vColor; varying float vA;
      void main() {
        vec3 p = position;
        float t = uTime * 0.25 + phase;
        p += vec3(sin(t * 0.9 + phase) * 0.9, sin(t * 0.6 + phase * 1.7) * 0.7 + sin(t * 0.21) * 0.4, cos(t * 0.8 + phase * 0.5) * 0.9);
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_Position = projectionMatrix * mv;
        float dist = -mv.z;
        gl_PointSize = clamp(uScale * 0.05 / max(dist, 0.1), 1.0, 9.0);
        vA = (0.35 + 0.65 * (0.5 + 0.5 * sin(t * 2.3 + phase * 3.1))) * smoothstep(60.0, 8.0, dist) * smoothstep(0.3, 1.5, dist);
        vColor = color;
      }`,
    fragmentShader: /* glsl */`
      varying vec3 vColor; varying float vA;
      void main() {
        vec2 c = gl_PointCoord - 0.5;
        float a = smoothstep(0.5, 0.0, length(c));
        gl_FragColor = vec4(vColor * a * vA, 1.0);
      }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  });
  const pts = new THREE.Points(g, mat);
  pts.frustumCulled = false;
  pts.name = 'motes';
  const root = new THREE.Object3D();
  root.name = 'decor';
  root.add(pts);
  // call decor.userData.update(timeSeconds, drawingBufferHeightPx) each frame
  root.userData.update = (time, heightPx) => {
    mat.uniforms.uTime.value = time;
    mat.uniforms.uScale.value = heightPx;
  };
  return root;
}
