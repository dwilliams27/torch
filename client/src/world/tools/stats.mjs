// Level verification: node client/src/world/tools/stats.mjs [seed] [--atlas 4096] [--raster] [--reach]
// Checks: triangle count, atlas fill, chart rect overlaps, padding, chart winding (3D + atlas),
// texel-level overlap (--raster), reachability of every zone from spawn with the real physics (--reach).
import { buildLevelData, zoneIndexAt } from '../levelcore.js';
import { stepCharacter, CHAR } from '../collision.js';

const args = process.argv.slice(2);
const seed = parseInt(args.find((a) => /^\d+$/.test(a)) || '1', 10);
const atlasSize = args.includes('--atlas') ? parseInt(args[args.indexOf('--atlas') + 1], 10) : 4096;
const doRaster = args.includes('--raster') || args.includes('--all');
const doReach = args.includes('--reach') || args.includes('--all');

const t0 = Date.now();
const L = buildLevelData(seed, { atlasSize });
const t1 = Date.now();
const F = L.flat;
const out = {};
out.seed = seed;
out.buildMs = t1 - t0;
out.triangles = F.triCount;
out.vertices = F.vertexCount;
out.charts = L.atlas.charts;
out.atlas = { size: L.atlas.size, texelsPerMeter: +L.atlas.texelsPerMeter.toFixed(2), fill: +(L.atlas.fill * 100).toFixed(1) + '%', shelfUsed: +(L.atlas.used * 100).toFixed(1) + '%' };
out.lights = L.lights.length;
out.colliders = L.collision.prims.length;

// per-zone triangles & chart area
const zoneTris = new Array(L.zones.length).fill(0);
for (let i = 0; i < F.index.length; i += 3) zoneTris[F.zone[F.index[i]]]++;
out.zoneTris = Object.fromEntries(L.zones.map((z, i) => [z.id, zoneTris[i]]));

// rect overlaps + padding + uv containment
const rects = L.pack.rects;
let overlaps = 0;
const sorted = rects.map((r, i) => ({ ...r, i })).sort((a, b) => a.x - b.x);
for (let a = 0; a < sorted.length; a++) {
  const A = sorted[a];
  for (let b = a + 1; b < sorted.length && sorted[b].x < A.x + A.W; b++) {
    const B = sorted[b];
    if (B.x < A.x + A.W && A.x < B.x + B.W && B.y < A.y + A.H && A.y < B.y + B.H) overlaps++;
  }
}
out.rectOverlaps = overlaps;
let outside = 0, flipped3d = 0, flippedAtlas = 0, degenerate = 0;
const S = L.atlas.size, pad = L.atlas.pad;
for (let c = 0; c < L.charts.length; c++) {
  const r = rects[c];
  if (r.x + r.W > S || r.y + r.H > S) outside++;
}
let uvOut = 0;
const P = F.position, N = F.normal, A = F.atlasUv;
for (let i = 0; i < F.index.length; i += 3) {
  const a = F.index[i], b = F.index[i + 1], c = F.index[i + 2];
  const ex = [P[b * 3] - P[a * 3], P[b * 3 + 1] - P[a * 3 + 1], P[b * 3 + 2] - P[a * 3 + 2]];
  const fx = [P[c * 3] - P[a * 3], P[c * 3 + 1] - P[a * 3 + 1], P[c * 3 + 2] - P[a * 3 + 2]];
  const g = [ex[1] * fx[2] - ex[2] * fx[1], ex[2] * fx[0] - ex[0] * fx[2], ex[0] * fx[1] - ex[1] * fx[0]];
  const nn = [N[a * 3] + N[b * 3] + N[c * 3], N[a * 3 + 1] + N[b * 3 + 1] + N[c * 3 + 1], N[a * 3 + 2] + N[b * 3 + 2] + N[c * 3 + 2]];
  const gl = Math.hypot(g[0], g[1], g[2]);
  if (gl < 1e-9) { degenerate++; continue; }
  if (g[0] * nn[0] + g[1] * nn[1] + g[2] * nn[2] < 0) { flipped3d++; if (process.env.DEBUG_FLIP) { const ch = L.charts[F.chartOf[a]]; console.error('flip', L.surfaceTypes[ch.surface].name, L.zones[ch.zone].id, P[a*3].toFixed(1), P[a*3+1].toFixed(1), P[a*3+2].toFixed(1)); } }
  const ua = [A[a * 2], A[a * 2 + 1]], ub = [A[b * 2], A[b * 2 + 1]], uc = [A[c * 2], A[c * 2 + 1]];
  const cr = (ub[0] - ua[0]) * (uc[1] - ua[1]) - (ub[1] - ua[1]) * (uc[0] - ua[0]);
  if (cr < 0) flippedAtlas++;
  // containment in own rect (inside padding)
  const r = rects[F.chartOf[a]];
  for (const u of [ua, ub, uc]) {
    const X = u[0] * S, Y = u[1] * S;
    if (X < r.x + pad - 1e-3 || X > r.x + r.W - pad + 1e-3 || Y < r.y + pad - 1e-3 || Y > r.y + r.H - pad + 1e-3) uvOut++;
  }
}
out.rectsOutsideAtlas = outside;
out.uvOutsideOwnChart = uvOut;
out.flippedWinding3D = flipped3d;
out.flippedWindingAtlas = flippedAtlas;
out.degenerateTris = degenerate;

// texel raster check: any texel center covered by triangles of two different charts, or twice by the same chart
if (doRaster) {
  const owner = new Int32Array(S * S).fill(-1);
  let cross = 0, self = 0, covered = 0;
  const cnt = new Uint8Array(S * S);
  for (let i = 0; i < F.index.length; i += 3) {
    const ia = F.index[i], ib = F.index[i + 1], ic = F.index[i + 2];
    const ch = F.chartOf[ia];
    const ax = A[ia * 2] * S, ay = A[ia * 2 + 1] * S, bx = A[ib * 2] * S, by = A[ib * 2 + 1] * S, cx = A[ic * 2] * S, cy = A[ic * 2 + 1] * S;
    const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (Math.abs(area) < 1e-9) continue;
    const x0 = Math.max(0, Math.floor(Math.min(ax, bx, cx))), x1 = Math.min(S - 1, Math.ceil(Math.max(ax, bx, cx)));
    const y0 = Math.max(0, Math.floor(Math.min(ay, by, cy))), y1 = Math.min(S - 1, Math.ceil(Math.max(ay, by, cy)));
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const px = x + 0.5, py = y + 0.5;
      const w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area;
      const w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area;
      const w2 = 1 - w0 - w1;
      if (w0 <= 1e-7 || w1 <= 1e-7 || w2 <= 1e-7) continue;
      const k = y * S + x;
      if (owner[k] === -1) { owner[k] = ch; covered++; cnt[k] = 1; }
      else if (owner[k] !== ch) cross++;
      else { if (cnt[k] < 255) cnt[k]++; if (cnt[k] === 2) self++; }
    }
  }
  out.raster = { coveredTexels: covered, coverage: +(covered / (S * S) * 100).toFixed(1) + '%', crossChartOverlapTexels: cross, selfOverlapTexels: self };
}

// reachability: BFS over standing positions using the real character physics
if (doReach) {
  const W = L.collision;
  const q = [], seen = new Map();
  const key = (x, y, z) => `${Math.round(x * 2)},${Math.round(y * 4)},${Math.round(z * 2)}`;
  const settle = (st) => {
    for (let i = 0; i < 400 && !st.grounded; i++) { stepCharacter(W, st, 1 / 60); if (st.y < W.killY) return false; }
    return st.grounded;
  };
  const s0 = { x: L.spawn.position[0], y: L.spawn.position[1] - CHAR.eye + 0.3, z: L.spawn.position[2], vx: 0, vy: 0, vz: 0, grounded: false };
  settle(s0);
  q.push([s0.x, s0.y, s0.z]); seen.set(key(s0.x, s0.y, s0.z), 1);
  const zoneReached = new Array(L.zones.length).fill(0);
  const dirs = [];
  for (let k = 0; k < 8; k++) dirs.push([Math.cos(k * Math.PI / 4), Math.sin(k * Math.PI / 4)]);
  let head = 0, falls = 0;
  const maxNodes = 400000;
  while (head < q.length && q.length < maxNodes) {
    const [x, y, z] = q[head++];
    zoneReached[zoneIndexAt(L.zoneVols, x, y + 1, z)]++;
    for (const [dx, dz] of dirs) {
      const st = { x, y, z, vx: dx * 3.5, vy: 0, vz: dz * 3.5, grounded: true };
      for (let i = 0; i < 9; i++) stepCharacter(W, st, 1 / 60); // ~0.5 m
      st.vx = 0; st.vz = 0;
      if (!settle(st)) { falls++; continue; }
      const nx = Math.round(st.x * 2) / 2, nz = Math.round(st.z * 2) / 2;
      // snap to grid cell center if standable there
      const g = W.groundHeight(nx, nz, st.y + CHAR.step);
      let px = st.x, py = st.y, pz = st.z;
      if (g > -Infinity && Math.abs(g - st.y) < 0.5 && !W.blocked(nx, g, nz)) { px = nx; py = g; pz = nz; }
      const k = key(px, py, pz);
      if (seen.has(k)) continue;
      seen.set(k, 1);
      q.push([px, py, pz]);
    }
  }
  out.reach = { nodes: q.length, falls, zones: Object.fromEntries(L.zones.map((zz, i) => [zz.id, zoneReached[i]])) };
  out.reach.allZonesReached = zoneReached.every((v) => v > 0);
}

console.log(JSON.stringify(out, null, 1));
