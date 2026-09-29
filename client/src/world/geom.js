// Geometry builder: architectural primitives -> charts (unique atlas patches) + colliders.
// Pure JS (no three.js) so node tools can build + verify levels.
//
// A *chart* is a set of triangles with a 2D parametrisation in meters (chart coords C).
// Every chart gets its own padded rectangle in the dream atlas (see atlas.js), so the
// painter can store per-surface paint without any two surfaces sharing texels.
// Invariants (checked by tools/stats.mjs):
//   * triangles are CCW seen from the side the normal points to (three.js FrontSide)
//   * triangles are CCW in chart space too, hence CCW in atlasUv space
//   * chart coords are injective within a chart (no fold-overs)

import { CollisionWorld } from './collision.js';

// ------------------------------------------------------------------ vec3 helpers
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const len = (a) => Math.hypot(a[0], a[1], a[2]);
export const norm = (a) => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
export const lerp3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
const UP = [0, 1, 0];

// rotation matrices (row-major 3x3, world = M * local)
export function matYaw(a) { const c = Math.cos(a), s = Math.sin(a); return [c, 0, s, 0, 1, 0, -s, 0, c]; }
function matX(a) { const c = Math.cos(a), s = Math.sin(a); return [1, 0, 0, 0, c, -s, 0, s, c]; }
function matZ(a) { const c = Math.cos(a), s = Math.sin(a); return [c, -s, 0, s, c, 0, 0, 0, 1]; }
export function matMul(A, B) {
  const R = new Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++)
    R[r * 3 + c] = A[r * 3] * B[c] + A[r * 3 + 1] * B[3 + c] + A[r * 3 + 2] * B[6 + c];
  return R;
}
// yaw (Y), then pitch (X), then roll (Z) in local space
export function matEuler(pitch, yaw, roll) { return matMul(matMul(matYaw(yaw), matX(pitch)), matZ(roll)); }
export const apply = (M, v) => [M[0] * v[0] + M[1] * v[1] + M[2] * v[2], M[3] * v[0] + M[4] * v[1] + M[5] * v[2], M[6] * v[0] + M[7] * v[1] + M[8] * v[2]];

function newellNormal(pts) {
  let nx = 0, ny = 0, nz = 0;
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    nx += (a[1] - b[1]) * (a[2] + b[2]);
    ny += (a[2] - b[2]) * (a[0] + b[0]);
    nz += (a[0] - b[0]) * (a[1] + b[1]);
  }
  return norm([nx, ny, nz]);
}

// clip convex polygon by plane dot(p, n) = d ; returns [below(<=d), above(>=d)]
function splitPoly(pts, n, d) {
  const below = [], above = [];
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    const da = dot(a, n) - d, db = dot(b, n) - d;
    if (da <= 0) below.push(a);
    if (da >= 0) above.push(a);
    if ((da < 0 && db > 0) || (da > 0 && db < 0)) {
      const p = lerp3(a, b, da / (da - db));
      below.push(p); above.push(p);
    }
  }
  return [below.length >= 3 ? below : null, above.length >= 3 ? above : null];
}

function polyArea(pts) {
  let a = [0, 0, 0];
  for (let i = 1; i + 1 < pts.length; i++) a = add(a, cross(sub(pts[i], pts[0]), sub(pts[i + 1], pts[0])));
  return len(a) / 2;
}

// pattern uv (world meters): floors (x,z); walls (horizontal tangent, y)
function worldUV(p, n) {
  if (Math.abs(n[1]) > 0.7) return [p[0], p[2]];
  const t = norm(cross(UP, n));
  return [dot(p, t), p[1]];
}

export function archCurve(w, pointed = 0, segs = 12) {
  // points from right spring (w/2,0) over the crown to left spring (-w/2,0), relative to spring center
  const out = [];
  if (pointed <= 0.001) {
    const R = w / 2;
    for (let i = 0; i <= segs; i++) { const t = Math.PI * i / segs; out.push([R * Math.cos(t), R * Math.sin(t)]); }
    return out;
  }
  const Rp = w / 2 + pointed * w / 2;
  const cxr = w / 2 - Rp;                        // center of right arc
  const alpha = Math.acos(Math.min(1, (Rp - w / 2) / Rp));
  const half = Math.max(2, Math.ceil(segs / 2));
  for (let i = 0; i <= half; i++) { const t = alpha * i / half; out.push([cxr + Rp * Math.cos(t), Rp * Math.sin(t)]); }
  for (let i = half - 1; i >= 0; i--) { const t = alpha * i / half; out.push([-(cxr + Rp * Math.cos(t)), Rp * Math.sin(t)]); }
  return out;
}
export function archRise(w, pointed = 0) {
  if (pointed <= 0.001) return w / 2;
  const Rp = w / 2 + pointed * w / 2;
  return Math.sqrt(Rp * Rp - (Rp - w / 2) * (Rp - w / 2));
}

// --------------------------------------------------------------------- Builder
export class Builder {
  constructor(surfaceTypes) {
    this.surfaceTypes = surfaceTypes;
    this.surfIdx = new Map(surfaceTypes.map((s, i) => [s.name, i]));
    this.charts = [];
    this.col = new CollisionWorld(4);
    this.zone = 0;
    this.densMul = 1;
    this.farY = Infinity;       // faces above farY get farDens
    this.lowY = -Infinity;      // faces below lowY get farDens too (e.g. walls of an abyss)
    this.farDens = 0.5;
    this.maxEdge = 26;
    this.lights = [];
    this.zoneVols = [];
    this.tour = [];
    this.triCount = 0;
  }

  S(s) {
    if (typeof s === 'number') return s;
    const i = this.surfIdx.get(s);
    if (i === undefined) throw new Error('unknown surface ' + s);
    return i;
  }

  light(p, color, intensity = 1, radius = 12, flicker = false) {
    this.lights.push({ position: [p[0], p[1], p[2]], color, intensity, radius, flicker, zone: this.zone });
  }
  zoneBox(x0, y0, z0, x1, y1, z1, zone = this.zone, prio = 0) {
    this.zoneVols.push({ zone, prio, min: [Math.min(x0, x1), Math.min(y0, y1), Math.min(z0, z1)], max: [Math.max(x0, x1), Math.max(y0, y1), Math.max(z0, z1)] });
  }

  // ---------------------------------------------------------------- charts
  newChart(surf, dens, abs = false) {
    return { P: [], N: [], U: [], C: [], I: [], surface: this.S(surf), zone: this.zone, dens, abs, w: 0, h: 0 };
  }
  // Rigid transform applied to everything emitted until popped (decor only: colliders are NOT
  // transformed, so pass collide:false inside). M = rotation (row-major 3x3), t = translation.
  pushTransform(M, t) { this.xform = { M, t }; this._realCol = this.col; this.col = new CollisionWorld(8); }
  popTransform() { this.xform = null; if (this._realCol) this.col = this._realCol; this._realCol = null; }

  endChart(ch) {
    if (ch.I.length === 0) return null;
    if (this.xform) {
      const { M, t } = this.xform;
      ch.P = ch.P.map((p) => add(apply(M, p), t));
      ch.N = ch.N.map((n) => apply(M, n));
      ch.U = ch.P.map((p, i) => ch.U[i]);
    }
    let mnu = Infinity, mnv = Infinity, mxu = -Infinity, mxv = -Infinity;
    for (const c of ch.C) { mnu = Math.min(mnu, c[0]); mxu = Math.max(mxu, c[0]); mnv = Math.min(mnv, c[1]); mxv = Math.max(mxv, c[1]); }
    for (const c of ch.C) { c[0] -= mnu; c[1] -= mnv; }
    ch.w = Math.max(0.02, mxu - mnu); ch.h = Math.max(0.02, mxv - mnv);
    this.triCount += ch.I.length / 3;
    this.charts.push(ch);
    return ch;
  }
  densFor(pts, o) {
    let d = (o && o.dens != null ? o.dens : 1) * this.densMul;
    let mny = Infinity, mxy = -Infinity;
    for (const p of pts) { mny = Math.min(mny, p[1]); mxy = Math.max(mxy, p[1]); }
    if (mny >= this.farY - 1e-6 || mxy <= this.lowY + 1e-6) d *= this.farDens;
    return d;
  }

  // Convex planar polygon. If o.n given, winding is fixed to face it. Splits big faces.
  poly(pts, surf, o = {}) {
    let n = newellNormal(pts);
    if (o.n && dot(n, o.n) < 0) { pts = pts.slice().reverse(); n = mul(n, -1); }
    if (polyArea(pts) < 1e-5) return;
    let pieces = [pts];
    // far split (vertical-ish faces spanning farY)
    for (const Y of [this.farY, this.lowY]) {
      if (!(Math.abs(Y) < Infinity) || Math.abs(n[1]) >= 0.5) continue;
      const next = [];
      for (const pc of pieces) {
        const [lo, hi] = splitPoly(pc, UP, Y);
        if (lo) next.push(lo); if (hi) next.push(hi);
      }
      pieces = next;
    }
    // max edge split along world axes
    const me = o.maxEdge || this.maxEdge;
    for (const ax of [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
      const next = [];
      for (const pc of pieces) {
        let mn = Infinity, mx = -Infinity;
        for (const p of pc) { const v = dot(p, ax); mn = Math.min(mn, v); mx = Math.max(mx, v); }
        const k = Math.ceil((mx - mn) / me - 1e-6);
        if (k <= 1) { next.push(pc); continue; }
        let rest = pc;
        for (let i = 1; i < k && rest; i++) {
          const [lo, hi] = splitPoly(rest, ax, mn + (mx - mn) * i / k);
          if (lo) next.push(lo);
          rest = hi;
        }
        if (rest) next.push(rest);
      }
      pieces = next;
    }
    for (const pc of pieces) this.emitPoly(pc, n, surf, this.densFor(pc, o), o);
  }

  emitPoly(pts, n, surf, dens, o = {}) {
    const ch = this.newChart(surf, dens, !!o.abs);
    let T = Math.abs(n[1]) < 0.999 ? norm(cross(UP, n)) : [1, 0, 0];
    if (Math.abs(n[1]) >= 0.999) T = norm(sub([1, 0, 0], mul(n, n[0])));
    const B = cross(n, T);
    for (const p of pts) {
      ch.P.push(p); ch.N.push(n); ch.U.push(o.uvFn ? o.uvFn(p) : worldUV(p, n));
      ch.C.push([dot(p, T), dot(p, B)]);
    }
    for (let i = 1; i + 1 < pts.length; i++) ch.I.push(0, i, i + 1);
    this.endChart(ch);
  }

  quad(a, b, c, d, surf, o = {}) { this.poly([a, b, c, d], surf, o); }

  // Grid surface: P[i][j], N[i][j] (smooth normals), C[i][j] chart coords (m), U optional pattern uv.
  grid(P, N, C, U, surf, o = {}) {
    const ni = P.length - 1, nj = P[0].length - 1;
    let mny = Infinity, mxy = -Infinity;
    for (const row of P) for (const p of row) { mny = Math.min(mny, p[1]); mxy = Math.max(mxy, p[1]); }
    let dens = (o.dens != null ? o.dens : 1) * (o.abs ? 1 : this.densMul);
    if (!o.abs && (mny >= this.farY || mxy <= this.lowY)) dens *= this.farDens;
    const ch = this.newChart(surf, dens, !!o.abs);
    // orientation from a representative non-degenerate cell
    let flip3 = false, mirror = false, found = false;
    const cand = [[ni >> 1, nj >> 1], [0, 0], [ni - 1, nj - 1], [ni >> 1, 0], [0, nj >> 1]];
    for (let t = 0; t < cand.length && !found; t++) {
      for (let di = 0; di < ni && !found; di++) {
        const i = (cand[t][0] + di) % ni, j = cand[t][1];
        const a = P[i][j], b = P[i + 1][j], d = P[i][j + 1];
        const g = cross(sub(b, a), sub(d, a));
        if (len(g) < 1e-9) continue;
        const ca = C[i][j], cb = C[i + 1][j], cd = C[i][j + 1];
        const c2 = (cb[0] - ca[0]) * (cd[1] - ca[1]) - (cb[1] - ca[1]) * (cd[0] - ca[0]);
        if (Math.abs(c2) < 1e-12) continue;
        const nn = add(add(N[i][j], N[i + 1][j]), N[i][j + 1]);
        flip3 = dot(g, nn) < 0;
        mirror = flip3 ? c2 > 0 : c2 < 0;
        found = true;
      }
    }
    const idx = [];
    for (let i = 0; i <= ni; i++) {
      idx.push([]);
      for (let j = 0; j <= nj; j++) {
        idx[i].push(ch.P.length);
        ch.P.push(P[i][j]); ch.N.push(N[i][j]);
        ch.U.push(U ? U[i][j] : [C[i][j][0], C[i][j][1]]);
        ch.C.push(mirror ? [-C[i][j][0], C[i][j][1]] : [C[i][j][0], C[i][j][1]]);
      }
    }
    const tri = (a, b, c) => {
      const A = ch.P[a], Bp = ch.P[b], Cp = ch.P[c];
      if (len(cross(sub(Bp, A), sub(Cp, A))) < 1e-9) return;
      if (flip3) ch.I.push(a, c, b); else ch.I.push(a, b, c);
    };
    for (let i = 0; i < ni; i++) for (let j = 0; j < nj; j++) {
      const a = idx[i][j], b = idx[i + 1][j], c = idx[i + 1][j + 1], d = idx[i][j + 1];
      tri(a, b, c); tri(a, c, d);
    }
    return this.endChart(ch);
  }

  // ------------------------------------------------------------ primitives
  // Box centered at (cx,cy,cz), size (sx,sy,sz). o.yaw rotates about Y (collider follows).
  // o.m = full rotation matrix (decor; collider = AABB only if o.collide === 'aabb').
  // o.skip: letters of faces to omit: t b x X z Z (x = -x face, X = +x face ...)
  box(cx, cy, cz, sx, sy, sz, surf, o = {}) {
    const hx = sx / 2, hy = sy / 2, hz = sz / 2;
    const M = o.m || matYaw(o.yaw || 0);
    const C0 = [cx, cy, cz];
    const W = (x, y, z) => add(C0, apply(M, [x, y, z]));
    const skip = o.skip || '';
    const F = [
      ['t', [0, 1, 0], [[-hx, hy, -hz], [hx, hy, -hz], [hx, hy, hz], [-hx, hy, hz]]],
      ['b', [0, -1, 0], [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, -hy, hz], [-hx, -hy, hz]]],
      ['X', [1, 0, 0], [[hx, -hy, -hz], [hx, hy, -hz], [hx, hy, hz], [hx, -hy, hz]]],
      ['x', [-1, 0, 0], [[-hx, -hy, -hz], [-hx, hy, -hz], [-hx, hy, hz], [-hx, -hy, hz]]],
      ['Z', [0, 0, 1], [[-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]]],
      ['z', [0, 0, -1], [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz]]],
    ];
    const surfOf = (f) => (o.surfs && o.surfs[f] != null ? o.surfs[f] : surf);
    // Small boxes: unroll the 4 lateral faces into ONE strip chart (saves padding + seams)
    const perim = 2 * (sx + sz);
    const wrap = o.wrap !== false && !o.surfs && !o.fdens && perim <= (o.wrapMax || 30) && sy <= 26 &&
      !(this.farY < Infinity && cy - hy < this.farY && cy + hy > this.farY && !o.m && Math.abs(M[4] - 1) < 1e-6);
    let wrapped = '';
    if (wrap) {
      const lat = [
        ['X', [1, 0, 0], [hx, -hz], [hx, hz]], ['Z', [0, 0, 1], [hx, hz], [-hx, hz]],
        ['x', [-1, 0, 0], [-hx, hz], [-hx, -hz]], ['z', [0, 0, -1], [-hx, -hz], [hx, -hz]],
      ].filter((f) => !skip.includes(f[0]));
      if (lat.length >= 2) {
        const pts0 = [W(-hx, -hy, -hz), W(hx, hy, hz)];
        const ch = this.newChart(surf, this.densFor(pts0, o), !!o.abs);
        let s0 = 0;
        for (const [f, n, a, bq] of lat) {
          const nn = apply(M, n);
          const L = Math.hypot(bq[0] - a[0], bq[1] - a[1]);
          const q = [W(a[0], -hy, a[1]), W(bq[0], -hy, bq[1]), W(bq[0], hy, bq[1]), W(a[0], hy, a[1])];
          const cc = [[s0, -hy], [s0 + L, -hy], [s0 + L, hy], [s0, hy]];
          const bi = ch.P.length;
          for (let k = 0; k < 4; k++) { ch.P.push(q[k]); ch.N.push(nn); ch.U.push(worldUV(q[k], nn)); ch.C.push(cc[k]); }
          const g = cross(sub(q[1], q[0]), sub(q[3], q[0]));
          if (dot(g, nn) >= 0) ch.I.push(bi, bi + 1, bi + 2, bi, bi + 2, bi + 3);
          else ch.I.push(bi, bi + 2, bi + 1, bi, bi + 3, bi + 2);
          s0 += L;
          wrapped += f;
        }
        this._fixChartMirror(ch);
        this.endChart(ch);
      }
    }
    for (const [f, n, cs] of F) {
      if (skip.includes(f) || wrapped.includes(f)) continue;
      const fo = { ...o, n: apply(M, n) };
      if (o.fdens && o.fdens[f] != null) fo.dens = o.fdens[f];
      this.poly(cs.map((c) => W(c[0], c[1], c[2])), surfOf(f), fo);
    }
    if (o.collide === false) return;
    if (o.m) {
      if (o.collide === 'aabb') {
        let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
        for (const sxx of [-hx, hx]) for (const syy of [-hy, hy]) for (const szz of [-hz, hz]) {
          const p = W(sxx, syy, szz);
          for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], p[k]); mx[k] = Math.max(mx[k], p[k]); }
        }
        this.col.addBox((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2, (mx[0] - mn[0]) / 2, (mx[1] - mn[1]) / 2, (mx[2] - mn[2]) / 2, 0);
      }
      return;
    }
    this.col.addBox(cx, cy, cz, hx, hy, hz, o.yaw || 0);
  }

  // Axis-aligned box from min/max corners (convenience)
  slab(x0, y0, z0, x1, y1, z1, surf, o = {}) {
    this.box((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2, Math.abs(x1 - x0), Math.abs(y1 - y0), Math.abs(z1 - z0), surf, o);
  }

  // invisible collider
  solid(x0, y0, z0, x1, y1, z1) {
    this.col.addBox((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2, Math.abs(x1 - x0) / 2, Math.abs(y1 - y0) / 2, Math.abs(z1 - z0) / 2, 0);
  }

  // Wedge ramp: centered (cx,cz), local x length L (rises from yLow at -L/2 to yHigh at +L/2), width Wd.
  ramp(cx, cz, L, Wd, yLow, yHigh, yaw, surf, o = {}) {
    const base = o.base != null ? o.base : Math.min(yLow, yHigh) - (o.thick || 0.4);
    const M = matYaw(yaw), C0 = [cx, 0, cz];
    const W = (x, y, z) => add(C0, apply(M, [x, y, z]));
    const h = L / 2, w = Wd / 2;
    const up = cross(sub(W(h, yHigh, 0), W(-h, yLow, 0)), sub(W(-h, yLow, -w), W(-h, yLow, w)));
    this.poly([W(-h, yLow, -w), W(h, yHigh, -w), W(h, yHigh, w), W(-h, yLow, w)], surf, { ...o, n: up[1] > 0 ? up : mul(up, -1) });
    const side = o.sideSurf || surf;
    for (const s of [-1, 1]) {
      const pts = [W(-h, base, s * w), W(h, base, s * w), W(h, yHigh, s * w), W(-h, yLow, s * w)];
      this.poly(pts, side, { ...o, n: apply(M, [0, 0, s]) });
    }
    if (yHigh > base + 0.01) this.poly([W(h, base, -w), W(h, base, w), W(h, yHigh, w), W(h, yHigh, -w)], side, { ...o, n: apply(M, [1, 0, 0]) });
    if (yLow > base + 0.01) this.poly([W(-h, base, -w), W(-h, base, w), W(-h, yLow, w), W(-h, yLow, -w)], side, { ...o, n: apply(M, [-1, 0, 0]) });
    if (!o.noBottom) this.poly([W(-h, base, -w), W(h, base, -w), W(h, base, w), W(-h, base, w)], side, { ...o, n: [0, -1, 0] });
    if (o.collide !== false) this.col.addRamp(cx, cz, h, w, yaw, yLow, yHigh, base);
  }

  // Straight stairs. Starts at (x0,z0) (middle of the first riser), climbs along yaw direction
  // (local +x) from y0 to y1. Returns [x,z] at the top edge. o.solid: fill to o.base.
  stairs(x0, z0, yaw, width, y0, y1, surf, o = {}) {
    const rise0 = o.rise || 0.19, run = o.run || 0.36;
    const n = Math.max(1, Math.round(Math.abs(y1 - y0) / rise0));
    if (y1 < y0) {
      // descending: build it ascending from the far end
      const L0 = n * run;
      const e = add([x0, 0, z0], apply(matYaw(yaw), [L0, 0, 0]));
      this.stairs(e[0], e[2], yaw + Math.PI, width, y1, y0, surf, o);
      return [e[0], e[2]];
    }
    const rise = (y1 - y0) / n;
    const M = matYaw(yaw), C0 = [x0, 0, z0];
    const W = (x, y, z) => add(C0, apply(M, [x, y, z]));
    const w = width / 2;
    const L = n * run;
    const base = o.base != null ? o.base : null;
    // treads + risers: one strip chart (u across, v along profile)
    const P = [], N = [], C = [], U = [];
    let s = 0;
    const push = (x, y, nrm) => {
      P.push([W(x, y, -w), W(x, y, w)]);
      N.push([nrm, nrm]); C.push([[s, 0], [s, width]]);
      U.push([[x, -w], [x, w]]);
    };
    const nUp = apply(M, [0, 1, 0]), nFront = apply(M, [-1, 0, 0]);
    for (let i = 0; i < n; i++) {
      const xa = i * run, ya = y0 + i * rise, yb = ya + rise;
      // riser (vertical) from ya to yb at x = xa
      push(xa, ya, nFront); s += Math.abs(rise); push(xa, yb, nFront);
      // tread from xa to xa+run at yb
      push(xa, yb, nUp); s += run; push(xa + run, yb, nUp);
    }
    // grid expects P[i][j]; we built pairs, reorganise into separate quads per segment
    const ch = this.newChart(surf, this.densFor([[0, Math.min(y0, y1), 0]], o) * (o.dens != null ? 1 : 1), false);
    for (let k = 0; k + 1 < P.length; k += 2) {
      const a = P[k][0], b = P[k][1], c = P[k + 1][1], d = P[k + 1][0];
      const nrm = N[k][0];
      const base_i = ch.P.length;
      ch.P.push(a, b, c, d); ch.N.push(nrm, nrm, nrm, nrm);
      ch.C.push(C[k][0], C[k][1], C[k + 1][1], C[k + 1][0]);
      ch.U.push(worldUV(a, nrm), worldUV(b, nrm), worldUV(c, nrm), worldUV(d, nrm));
      // orientation: want CCW seen from nrm side
      const g = cross(sub(b, a), sub(d, a));
      if (dot(g, nrm) >= 0) ch.I.push(base_i, base_i + 1, base_i + 2, base_i, base_i + 2, base_i + 3);
      else ch.I.push(base_i, base_i + 2, base_i + 1, base_i, base_i + 3, base_i + 2);
    }
    this._fixChartMirror(ch);
    this.endChart(ch);
    // sides: one chart per side made of column quads
    const yTop = (i) => y0 + (i + 1) * rise;
    for (const sd of [-1, 1]) {
      const cs = this.newChart(o.sideSurf || surf, this.densFor([[0, Math.min(y0, y1), 0]], o));
      const nrm = apply(M, [0, 0, sd]);
      for (let i = 0; i < n; i++) {
        const xa = i * run, xb = xa + run;
        const yb = base != null ? base : (rise > 0 ? y0 + i * rise - 0.35 : yTop(i) - 0.35 + rise);
        const yt = yTop(i);
        const bot = Math.min(yb, yt - 0.05);
        const q = [W(xa, bot, sd * w), W(xb, bot, sd * w), W(xb, yt, sd * w), W(xa, yt, sd * w)];
        const bi = cs.P.length;
        const T = norm(cross(UP, nrm)), Bv = cross(nrm, T);
        for (const p of q) { cs.P.push(p); cs.N.push(nrm); cs.U.push(worldUV(p, nrm)); cs.C.push([dot(p, T), dot(p, Bv)]); }
        const g = cross(sub(q[1], q[0]), sub(q[3], q[0]));
        if (dot(g, nrm) >= 0) cs.I.push(bi, bi + 1, bi + 2, bi, bi + 2, bi + 3);
        else cs.I.push(bi, bi + 2, bi + 1, bi, bi + 3, bi + 2);
      }
      // side columns overlap in chart space (xa..xb columns are disjoint in x, but y ranges nest) -> fine: disjoint x
      this.endChart(cs);
    }
    // underside / back
    if (base == null) {
      const off = 0.35;
      const a = W(0, y0 - off, -w), b = W(0, y0 - off, w), c = W(L, y1 - off, w), d = W(L, y1 - off, -w);
      this.poly([a, b, c, d], o.sideSurf || surf, { ...o, n: mul(cross(sub(d, a), sub(b, a)), 1)[1] < 0 ? cross(sub(d, a), sub(b, a)) : cross(sub(b, a), sub(d, a)) });
    } else if (Math.max(y0, y1) > base + 0.05) {
      const xe = rise > 0 ? L : 0, ye = Math.max(y0 + rise * n, y0);
      const nb = apply(M, [rise > 0 ? 1 : -1, 0, 0]);
      this.poly([W(xe, base, -w), W(xe, base, w), W(xe, ye, w), W(xe, ye, -w)], o.sideSurf || surf, { ...o, n: nb });
    }
    // colliders: one box per step
    if (o.collide !== false) {
      for (let i = 0; i < n; i++) {
        const yt = yTop(i);
        const yb = base != null ? base : yt - 0.4;
        const cxl = (i + 0.5) * run;
        const wc = W(cxl, 0, 0);
        this.col.addBox(wc[0], (yt + yb) / 2, wc[2], run / 2 + 0.01, (yt - yb) / 2, w, yaw);
      }
    }
    const e = W(L, 0, 0);
    return [e[0], e[2]];
  }

  _fixChartMirror(ch) {
    // ensure CCW in chart space for the (already 3D-correct) triangles; mirror u if needed
    let area = 0;
    for (let k = 0; k < ch.I.length; k += 3) {
      const a = ch.C[ch.I[k]], b = ch.C[ch.I[k + 1]], c = ch.C[ch.I[k + 2]];
      area += (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    }
    if (area < 0) for (let i = 0; i < ch.C.length; i++) ch.C[i] = [-ch.C[i][0], ch.C[i][1]];
  }

  // Spiral stair around (cx,cz): steps between radii rIn..rOut, starting angle a0 at height y0,
  // climbing to y1, angle increasing (dir=+1) or decreasing (dir=-1). Angle convention:
  // position = (cx + r cos a, cz + r sin a). Returns the end angle.
  spiral(cx, cz, rIn, rOut, a0, y0, y1, surf, o = {}) {
    const dir = o.dir || 1, rise0 = o.rise || 0.18, run = o.run || 0.42, thick = o.thick || 0.35;
    const rMid = (rIn + rOut) / 2;
    const n = Math.max(1, Math.round((y1 - y0) / rise0));
    const rise = (y1 - y0) / n;
    const da = dir * run / rMid;
    const pos = (r, a, y) => [cx + r * Math.cos(a), y, cz + r * Math.sin(a)];
    const dens = this.densFor([[0, y0, 0]], o);
    // treads + risers chart
    const ch = this.newChart(surf, dens);
    const quadInto = (c, pts, nrm, cc) => {
      const bi = c.P.length;
      const ns = Array.isArray(nrm[0]) ? nrm : [nrm, nrm, nrm, nrm];
      for (let k = 0; k < 4; k++) { c.P.push(pts[k]); c.N.push(ns[k]); c.C.push(cc[k]); c.U.push(cc[k]); }
      const nn = ns[0];
      const g = cross(sub(pts[1], pts[0]), sub(pts[3], pts[0]));
      if (dot(g, nn) >= 0) c.I.push(bi, bi + 1, bi + 2, bi, bi + 2, bi + 3);
      else c.I.push(bi, bi + 2, bi + 1, bi, bi + 3, bi + 2);
    };
    let s = 0;
    const W = rOut - rIn;
    for (let i = 0; i < n; i++) {
      const a = a0 + i * da, b = a + da, ya = y0 + i * rise, yb = ya + rise;
      // riser at angle a, facing backwards along the path
      const tdir = [-Math.sin(a) * dir, 0, Math.cos(a) * dir];
      const rn = mul(tdir, -1);
      quadInto(ch, [pos(rIn, a, ya), pos(rOut, a, ya), pos(rOut, a, yb), pos(rIn, a, yb)], rn, [[0, s], [W, s], [W, s + rise], [0, s + rise]]);
      s += rise;
      quadInto(ch, [pos(rIn, a, yb), pos(rOut, a, yb), pos(rOut, b, yb), pos(rIn, b, yb)], [0, 1, 0], [[0, s], [W, s], [W, s + run], [0, s + run]]);
      s += run;
    }
    this._fixChartMirror(ch);
    this.endChart(ch);
    // inner & outer rims, underside
    for (const [r, sgn] of [[rIn, -1], [rOut, 1]]) {
      const cr = this.newChart(o.sideSurf || surf, dens);
      let ss = 0;
      for (let i = 0; i < n; i++) {
        const a = a0 + i * da, b = a + da, yt = y0 + (i + 1) * rise, ybt = yt - thick - rise;
        const na = [Math.cos(a) * sgn, 0, Math.sin(a) * sgn], nb = [Math.cos(b) * sgn, 0, Math.sin(b) * sgn];
        const step = Math.abs(da) * r;
        quadInto(cr, [pos(r, a, ybt), pos(r, b, ybt), pos(r, b, yt), pos(r, a, yt)], [na, nb, nb, na], [[ss, ybt], [ss + step, ybt], [ss + step, yt], [ss, yt]]);
        ss += step + 0.02;
      }
      this._fixChartMirror(cr);
      this.endChart(cr);
    }
    const cu = this.newChart(o.sideSurf || surf, dens);
    let su = 0;
    for (let i = 0; i < n; i++) {
      const a = a0 + i * da, b = a + da, ya = y0 + i * rise - thick, yb = ya + rise;
      const step = Math.abs(da) * rMid;
      quadInto(cu, [pos(rIn, a, ya), pos(rOut, a, ya), pos(rOut, b, yb), pos(rIn, b, yb)], [0, -1, 0], [[0, su], [W, su], [W, su + step], [0, su + step]]);
      su += step + 0.02;
    }
    this._fixChartMirror(cu);
    this.endChart(cu);
    // parapet on inner edge (continuous helical band)
    if (o.parapet) {
      const ph = o.parapet, pt = 0.14;
      const rows = [];
      const segs = n;
      const Pp = [[], [], []], Np = [[], [], []], Cp = [[], [], []];
      for (let i = 0; i <= segs; i++) {
        const a = a0 + i * da, yb = y0 + i * rise + rise * 0.5;
        const ri = rIn, ro = rIn + pt;
        Pp[0].push([pos(ri, a, yb - 0.3), pos(ri, a, yb + ph)]);
        Np[0].push([[-Math.cos(a), 0, -Math.sin(a)], [-Math.cos(a), 0, -Math.sin(a)]]);
        Pp[1].push([pos(ro, a, yb), pos(ro, a, yb + ph)]);
        Np[1].push([[Math.cos(a), 0, Math.sin(a)], [Math.cos(a), 0, Math.sin(a)]]);
        Pp[2].push([pos(ri, a, yb + ph), pos(ro, a, yb + ph)]);
        Np[2].push([[0, 1, 0], [0, 1, 0]]);
        const sArc = i * Math.abs(da) * rIn;
        Cp[0].push([[sArc, 0], [sArc, ph + 0.3]]);
        Cp[1].push([[sArc, 0], [sArc, ph]]);
        Cp[2].push([[sArc, 0], [sArc, pt]]);
      }
      for (let k = 0; k < 3; k++) this.grid(Pp[k], Np[k], Cp[k], null, o.parapetSurf || surf, { dens: 1 });
      for (let i = 0; i < n; i++) {
        const am = a0 + (i + 0.5) * da, yt = y0 + (i + 1) * rise;
        const p = pos(rIn + pt / 2, am, 0);
        this.col.addBox(p[0], yt + ph / 2 - 0.3, p[2], pt / 2 + 0.02, (ph + 0.6) / 2, (Math.abs(da) * rIn) / 2 + 0.03, -am);
      }
    }
    // colliders
    for (let i = 0; i < n; i++) {
      const am = a0 + (i + 0.5) * da, yt = y0 + (i + 1) * rise;
      const p = pos(rMid, am, 0);
      // local x = radial, local z = tangential ; yaw so that local +x maps to radial dir (cos a, sin a)
      // world dx = lx*cos(yaw) ; dz = -lx*sin(yaw)  => yaw = -a
      this.col.addBox(p[0], yt - (thick + rise) / 2, p[2], (rOut - rIn) / 2, (thick + rise) / 2, (Math.abs(da) * rOut) / 2 * 1.08, -am);
    }
    return a0 + n * da;
  }

  // Vertical prism/cylinder. o.sides, o.inward, o.caps ('t','b','tb'), o.collide
  cyl(cx, cz, r, y0, y1, surf, o = {}) {
    const n = o.sides || (r < 0.6 ? 10 : r < 2 ? 14 : 24);
    const a0 = o.a0 || 0;
    const P = [], N = [], C = [], U = [];
    const circ = 2 * Math.PI * r;
    for (let i = 0; i <= n; i++) {
      const a = a0 + (i / n) * Math.PI * 2;
      const ca = Math.cos(a), sa = Math.sin(a);
      const nr = o.inward ? [-ca, 0, -sa] : [ca, 0, sa];
      const x = cx + r * ca, z = cz + r * sa;
      const s = (i / n) * circ;
      P.push([[x, y0, z], [x, y1, z]]); N.push([nr, nr]);
      C.push([[s, y0], [s, y1]]); U.push([[s, y0], [s, y1]]);
    }
    this.grid(P, N, C, U, surf, o);
    const caps = o.caps == null ? 't' : o.caps;
    const ring = (y) => { const pts = []; for (let i = 0; i < n; i++) { const a = a0 + (i / n) * Math.PI * 2; pts.push([cx + r * Math.cos(a), y, cz + r * Math.sin(a)]); } return pts; };
    if (caps.includes('t')) this.poly(ring(y1), o.capSurf || surf, { ...o, n: o.inward ? [0, -1, 0] : [0, 1, 0] });
    if (caps.includes('b')) this.poly(ring(y0), o.capSurf || surf, { ...o, n: o.inward ? [0, 1, 0] : [0, -1, 0] });
    if (o.collide !== false && !o.inward) this.col.addCyl(cx, cz, r, y0, y1);
  }

  // Polygonal prism sides with individual [y0,y1] per side, all in ONE chart (exposed faces only).
  prismSides(cx, cz, r, n, a0, ranges, surf, o = {}) {
    const ch = this.newChart(surf, (o.dens != null ? o.dens : 1) * this.densMul);
    let s0 = 0;
    const side = 2 * r * Math.sin(Math.PI / n);
    for (let k = 0; k < n; k++) {
      const [y0, y1] = ranges[k] || [0, 0];
      if (y1 - y0 < 0.02) continue;
      const aa = a0 + (k / n) * Math.PI * 2, ab = a0 + ((k + 1) / n) * Math.PI * 2, am = (aa + ab) / 2;
      const nn = [Math.cos(am), 0, Math.sin(am)];
      const pa = [cx + r * Math.cos(aa), 0, cz + r * Math.sin(aa)], pb = [cx + r * Math.cos(ab), 0, cz + r * Math.sin(ab)];
      const q = [[pa[0], y0, pa[2]], [pb[0], y0, pb[2]], [pb[0], y1, pb[2]], [pa[0], y1, pa[2]]];
      const cc = [[s0, y0], [s0 + side, y0], [s0 + side, y1], [s0, y1]];
      const bi = ch.P.length;
      for (let i = 0; i < 4; i++) { ch.P.push(q[i]); ch.N.push(nn); ch.U.push(worldUV(q[i], nn)); ch.C.push(cc[i]); }
      const g = cross(sub(q[1], q[0]), sub(q[3], q[0]));
      if (dot(g, nn) >= 0) ch.I.push(bi, bi + 1, bi + 2, bi, bi + 2, bi + 3);
      else ch.I.push(bi, bi + 2, bi + 1, bi, bi + 3, bi + 2);
      s0 += side + 0.02;
    }
    this._fixChartMirror(ch);
    return this.endChart(ch);
  }

  // A column with base + capital, classical-ish
  column(cx, cz, r, y0, y1, surf, o = {}) {
    const bh = Math.min(0.6, (y1 - y0) * 0.06), ch = Math.min(0.7, (y1 - y0) * 0.06);
    const col = o.collide !== false;
    this.box(cx, y0 + bh / 2, cz, r * 2.6, bh, r * 2.6, o.baseSurf || surf, { skip: 'b', collide: col });
    this.cyl(cx, cz, r, y0 + bh, y1 - ch, surf, { ...o, caps: '' });
    this.box(cx, y1 - ch / 2, cz, r * 2.8, ch, r * 2.8, o.baseSurf || surf, { collide: col });
  }

  // Sphere section (lat in radians). o.inward for skies/domes. o.sectors splits longitude charts.
  sphere(cx, cy, cz, r, surf, o = {}) {
    const lat0 = o.lat0 != null ? o.lat0 : -Math.PI / 2, lat1 = o.lat1 != null ? o.lat1 : Math.PI / 2;
    const sectors = o.sectors || 1;
    const nLon = o.lon || 32, nLat = o.lat || 12;
    const latRef = lat0 <= 0 && lat1 >= 0 ? 0 : Math.min(Math.abs(lat0), Math.abs(lat1));
    const perSec = Math.ceil(nLon / sectors);
    for (let sct = 0; sct < sectors; sct++) {
      const P = [], N = [], C = [];
      for (let i = 0; i <= perSec; i++) {
        const lon = ((sct * perSec + i) / (perSec * sectors)) * Math.PI * 2 + (o.lon0 || 0);
        const row = [], nrow = [], crow = [];
        for (let j = 0; j <= nLat; j++) {
          const lat = lat0 + (lat1 - lat0) * j / nLat;
          const d = [Math.cos(lat) * Math.cos(lon), Math.sin(lat), Math.cos(lat) * Math.sin(lon)];
          const sc = o.scale || [1, 1, 1];
          const rr = o.rfn ? r * o.rfn(lon, lat) : r;
          row.push([cx + rr * d[0] * sc[0], cy + rr * d[1] * sc[1], cz + rr * d[2] * sc[2]]);
          const nd = norm([d[0] / sc[0], d[1] / sc[1], d[2] / sc[2]]);
          nrow.push(o.inward ? mul(nd, -1) : nd);
          const sc2 = o.scale || [1, 1, 1];
          crow.push([lon * r * Math.cos(latRef) * Math.max(sc2[0], sc2[2]), lat * r * sc2[1]]);
        }
        P.push(row); N.push(nrow); C.push(crow);
      }
      this.grid(P, N, C, null, surf, o);
    }
  }

  // Barrel vault: cross-section archCurve(w, pointed) at spring height ys, spanning along
  // local z from z0..z1 in a frame at (cx, cz) rotated by yaw. Inward normals.
  vault(cx, cz, yaw, w, ys, z0, z1, surf, o = {}) {
    const curve = archCurve(w, o.pointed || 0, o.segs || 16);
    const M = matYaw(yaw), C0 = [cx, 0, cz];
    const Wt = (x, y, z) => add(C0, apply(M, [x, y, z]));
    const nz = Math.max(1, Math.ceil(Math.abs(z1 - z0) / (o.maxLen || 24)));
    const rise = archRise(w, o.pointed || 0);
    for (let k = 0; k < nz; k++) {
      const za = z0 + (z1 - z0) * k / nz, zb = z0 + (z1 - z0) * (k + 1) / nz;
      const P = [], N = [], C = [];
      let s = 0;
      for (let i = 0; i < curve.length; i++) {
        const [x, y] = curve[i];
        if (i > 0) s += Math.hypot(x - curve[i - 1][0], y - curve[i - 1][1]);
        const prev = curve[Math.max(0, i - 1)], next = curve[Math.min(curve.length - 1, i + 1)];
        const tx = next[0] - prev[0], ty = next[1] - prev[1];
        let nx = -ty, ny = tx; // rotate tangent
        // point normal toward interior (toward (0, rise*0.3))
        if (nx * (0 - x) + ny * (rise * 0.3 - y) < 0) { nx = -nx; ny = -ny; }
        const nl = Math.hypot(nx, ny) || 1;
        const nw = apply(M, [nx / nl, ny / nl, 0]);
        P.push([Wt(x, ys + y, za), Wt(x, ys + y, zb)]);
        N.push([nw, nw]);
        C.push([[s, za], [s, zb]]);
      }
      this.grid(P, N, C, null, surf, o);
    }
  }

  // Tympanum / arch-shaped planar face (fan) at local frame: fills archCurve above spring line
  archFill(cx, cz, yaw, w, ys, surf, o = {}) {
    const curve = archCurve(w, o.pointed || 0, o.segs || 16);
    const M = matYaw(yaw), C0 = [cx, 0, cz];
    const pts = curve.map(([x, y]) => add(C0, apply(M, [x, ys + y, 0])));
    this.poly(pts, surf, { ...o, n: apply(M, [0, 0, o.flip ? -1 : 1]) });
  }

  // Wall with an arched opening. Wall runs from (x0,z0) along local +x (yaw) for Wd meters,
  // thickness T centred on the line, from y0 to y0+H. Opening width ow centred at o.at (default Wd/2),
  // spring height hs above y0, o.pointed. o.noJambs: opening reaches y0 (door) - default true.
  archWall(x0, z0, yaw, Wd, H, T, y0, ow, hs, surf, o = {}) {
    const M = matYaw(yaw), C0 = [x0, y0, z0];
    const Wt = (u, v, d) => add(C0, apply(M, [u, v, d]));
    const at = o.at != null ? o.at : Wd / 2;
    const a = at - ow / 2, b = at + ow / 2;
    const pointed = o.pointed || 0;
    const curve = archCurve(ow, pointed, o.segs || 14).map(([x, y]) => [x + at, y + hs]); // right->left
    const rise = archRise(ow, pointed);
    const top = H;
    // front/back faces
    const angOf = (p) => Math.atan2(p[1] - hs, p[0] - at);
    const rayCurve = (phi) => {
      // intersect ray from (at,hs) at angle phi with curve polyline
      const dx = Math.cos(phi), dy = Math.sin(phi);
      for (let i = 0; i + 1 < curve.length; i++) {
        const p = curve[i], q = curve[i + 1];
        const a1 = angOf(p), a2 = angOf(q);
        if (phi >= Math.min(a1, a2) - 1e-9 && phi <= Math.max(a1, a2) + 1e-9) {
          // solve (at,hs)+t(dx,dy) = p + s(q-p)
          const ex = q[0] - p[0], ey = q[1] - p[1];
          const den = dx * ey - dy * ex;
          if (Math.abs(den) < 1e-12) return p;
          const t = ((p[0] - at) * ey - (p[1] - hs) * ex) / den;
          return [at + t * dx, hs + t * dy];
        }
      }
      return phi < Math.PI / 2 ? curve[0] : curve[curve.length - 1];
    };
    const rayRect = (phi) => {
      const dx = Math.cos(phi), dy = Math.sin(phi);
      let t = Infinity;
      if (dx > 1e-9) t = Math.min(t, (Wd - at) / dx);
      if (dx < -1e-9) t = Math.min(t, (0 - at) / dx);
      if (dy > 1e-9) t = Math.min(t, (top - hs) / dy);
      return [at + t * dx, hs + t * dy];
    };
    const angs = new Set(curve.map(angOf).map((v) => Math.max(0, Math.min(Math.PI, v))));
    angs.add(0); angs.add(Math.PI);
    angs.add(Math.atan2(top - hs, Wd - at)); angs.add(Math.atan2(top - hs, 0 - at));
    const A = [...angs].sort((p, q) => p - q);
    const faces2d = []; // list of quads (2D, CCW)
    if (hs > 0.01 && !o.noJambs) {
      faces2d.push([[0, 0], [a, 0], [a, hs], [0, hs]]);
      faces2d.push([[b, 0], [Wd, 0], [Wd, hs], [b, hs]]);
    } else if (hs > 0.01) {
      faces2d.push([[0, 0], [a, 0], [a, hs], [0, hs]]);
      faces2d.push([[b, 0], [Wd, 0], [Wd, hs], [b, hs]]);
    }
    for (let i = 0; i + 1 < A.length; i++) {
      const c1 = rayCurve(A[i]), c2 = rayCurve(A[i + 1]);
      const o1 = rayRect(A[i]), o2 = rayRect(A[i + 1]);
      faces2d.push([c1, o1, o2, c2]);
    }
    const dens = this.densFor([Wt(0, 0, 0)], o) * (o.dens != null ? 1 : 1);
    for (const side of [1, -1]) {
      const nrm = apply(M, [0, 0, side]);
      const ch = this.newChart(surf, o.dens != null ? o.dens * this.densMul : dens);
      for (const q of faces2d) {
        const bi = ch.P.length;
        const pts = q.map(([u, v]) => Wt(u, v, side * T / 2));
        for (let k = 0; k < 4; k++) {
          ch.P.push(pts[k]); ch.N.push(nrm); ch.U.push(worldUV(pts[k], nrm));
          ch.C.push(side > 0 ? [q[k][0], q[k][1]] : [Wd - q[k][0], q[k][1]]);
        }
        const g = cross(sub(pts[1], pts[0]), sub(pts[3], pts[0]));
        const good = dot(g, nrm) >= 0;
        const tri = (i0, i1, i2) => {
          const P0 = pts[i0], P1 = pts[i1], P2 = pts[i2];
          if (len(cross(sub(P1, P0), sub(P2, P0))) < 1e-9) return;
          if (good) ch.I.push(bi + i0, bi + i1, bi + i2); else ch.I.push(bi + i0, bi + i2, bi + i1);
        };
        tri(0, 1, 2); tri(0, 2, 3);
      }
      this._fixChartMirror(ch);
      this.endChart(ch);
    }
    // intrados strip: left jamb up, curve left->right reversed..., right jamb down
    const bnd = [];
    if (hs > 0.01) bnd.push([a, 0, [1, 0]]);
    const cr = curve.slice().reverse(); // left -> right
    for (let i = 0; i < cr.length; i++) {
      const [x, y] = cr[i];
      let nx = at - x, ny = hs + rise * 0.25 - y; // toward opening interior
      if (y <= hs + 1e-6) { nx = x < at ? 1 : -1; ny = 0; }
      const nl = Math.hypot(nx, ny) || 1;
      bnd.push([x, y, [nx / nl, ny / nl]]);
    }
    if (hs > 0.01) bnd.push([b, 0, [-1, 0]]);
    const P = [], N = [], C = [];
    let s = 0;
    for (let i = 0; i < bnd.length; i++) {
      if (i > 0) s += Math.hypot(bnd[i][0] - bnd[i - 1][0], bnd[i][1] - bnd[i - 1][1]);
      const nw = apply(M, [bnd[i][2][0], bnd[i][2][1], 0]);
      P.push([Wt(bnd[i][0], bnd[i][1], -T / 2), Wt(bnd[i][0], bnd[i][1], T / 2)]);
      N.push([nw, nw]); C.push([[s, 0], [s, T]]);
    }
    this.grid(P, N, C, null, o.soffitSurf || surf, { dens: o.dens });
    // top and ends
    if (!o.noTop) this.poly([Wt(0, top, -T / 2), Wt(Wd, top, -T / 2), Wt(Wd, top, T / 2), Wt(0, top, T / 2)], surf, { ...o, n: [0, 1, 0] });
    if (o.ends) {
      this.poly([Wt(0, 0, -T / 2), Wt(0, 0, T / 2), Wt(0, top, T / 2), Wt(0, top, -T / 2)], surf, { ...o, n: apply(M, [-1, 0, 0]) });
      this.poly([Wt(Wd, 0, -T / 2), Wt(Wd, 0, T / 2), Wt(Wd, top, T / 2), Wt(Wd, top, -T / 2)], surf, { ...o, n: apply(M, [1, 0, 0]) });
    }
    // colliders: piers + lintel
    if (o.collide !== false) {
      const addLocal = (u0, u1, v0, v1) => {
        if (u1 - u0 < 0.01 || v1 - v0 < 0.01) return;
        const c = Wt((u0 + u1) / 2, 0, 0);
        this.col.addBox(c[0], y0 + (v0 + v1) / 2, c[2], (u1 - u0) / 2, (v1 - v0) / 2, T / 2, yaw);
      };
      addLocal(0, a, 0, top);
      addLocal(b, Wd, 0, top);
      addLocal(a, b, hs + rise * 0.55, top);
    }
  }

  // Hexagonal crystal from base point along direction dir (unit), length L, radius r, pointed tip
  crystal(base, dir, L, r, surf, o = {}) {
    const d = norm(dir);
    const tmp = Math.abs(d[1]) < 0.9 ? UP : [1, 0, 0];
    const e1 = norm(cross(d, tmp)), e2 = cross(d, e1);
    const tip = o.tip != null ? o.tip : r * 1.6;
    const sides = o.sides || 6;
    const ring = (h, rr) => {
      const out = [];
      for (let i = 0; i < sides; i++) {
        const a = (i / sides) * Math.PI * 2 + (o.rot || 0);
        out.push(add(add(base, mul(d, h)), add(mul(e1, Math.cos(a) * rr), mul(e2, Math.sin(a) * rr))));
      }
      return out;
    };
    const r0 = ring(-0.2, r * (o.baseScale || 0.85)), r1 = ring(L, r);
    const apex = add(base, mul(d, L + tip));
    const ch = this.newChart(surf, (o.dens != null ? o.dens : 1) * this.densMul);
    const sideW = 2 * r * Math.sin(Math.PI / sides);
    for (let i = 0; i < sides; i++) {
      const j = (i + 1) % sides;
      const a = r0[i], b = r0[j], c = r1[j], e = r1[i];
      const n = norm(cross(sub(b, a), sub(e, a)));
      const outward = sub(mul(add(a, b), 0.5), add(base, mul(d, dot(sub(mul(add(a, b), 0.5), base), d))));
      const nn = dot(n, outward) >= 0 ? n : mul(n, -1);
      const u0 = i * (sideW + 0.05), u1 = u0 + sideW;
      const bi = ch.P.length;
      const pts = [a, b, c, e, apex];
      const cc = [[u0, 0], [u1, 0], [u1, L + 0.2], [u0, L + 0.2], [(u0 + u1) / 2, L + 0.2 + tip]];
      const tn = norm(cross(sub(c, e), sub(apex, e)));
      const tnn = dot(tn, sub(mul(add(c, e), 0.5), add(base, mul(d, L)))) >= 0 ? tn : mul(tn, -1);
      for (let k = 0; k < 5; k++) { ch.P.push(pts[k]); ch.N.push(k === 4 ? tnn : nn); ch.C.push(cc[k]); ch.U.push(cc[k]); }
      // side quad (a,b,c,e) and tip tri (e,c,apex): orient each against its normal
      const orient = (i0, i1, i2, nrm) => {
        const g = cross(sub(pts[i1], pts[i0]), sub(pts[i2], pts[i0]));
        if (dot(g, nrm) >= 0) ch.I.push(bi + i0, bi + i1, bi + i2); else ch.I.push(bi + i0, bi + i2, bi + i1);
      };
      orient(0, 1, 2, nn); orient(0, 2, 3, nn); orient(3, 2, 4, tnn);
    }
    // tip triangles share apex vertex normal with side normals of the column; acceptable facet look
    this._fixChartMirrorPerTri(ch);
    this.endChart(ch);
    if (o.collide) {
      const c = add(base, mul(d, L * 0.5));
      this.col.addCyl(c[0], c[2], r * 0.9, Math.min(base[1], apex[1]), Math.max(base[1], apex[1]));
    }
  }

  // For hand-built charts where each triangle's chart winding must be CCW: if a chart has mixed
  // winding (shouldn't), mirror whole chart when the majority is CW.
  _fixChartMirrorPerTri(ch) { this._fixChartMirror(ch); }

  // flat disc facing normal n
  disc(c, n, r, surf, o = {}) {
    n = norm(n);
    const tmp = Math.abs(n[1]) < 0.9 ? UP : [1, 0, 0];
    const e1 = norm(cross(n, tmp)), e2 = cross(n, e1);
    const pts = [];
    const k = o.sides || 24;
    for (let i = 0; i < k; i++) { const a = (i / k) * Math.PI * 2; pts.push(add(c, add(mul(e1, Math.cos(a) * r), mul(e2, Math.sin(a) * r)))); }
    this.poly(pts, surf, { ...o, n });
  }
}
