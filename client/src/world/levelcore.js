// HYPNAGOGIA level authoring. Pure JS (no three.js): node tools import this directly.
//
// World plan (x east, y up, -z north; meters):
//   0 Vestibule   z +48..+2   twisting gilded frames on a causeway floating in the void (spawn)
//   1 Nave        z 0..-70    flooded gothic cathedral, 30 m vault, causeway + transept, rose window
//   2 Stacks      c(0,-92)    50 m circular library shaft, 3 spiral flights, landings, bridge, oculus
//   3 Geode       west        basalt-column terraces under a rock dome with giant glowing crystals
//   4 Baths       east        four cascading neon pools under a barrel vault
//   5 Garden      far east    sunken parterre in a box whose walls and ceiling are painted sky
//   6 Atrium      north, high floating concrete monoliths, bridges and ramps over an abyss
//   7 Desert      far north   colossal arches receding under two moons

import { Builder, matEuler, matYaw, apply, add, archRise } from './geom.js';
import { RNG, hashSeed } from './rng.js';
import { SURFACES, ZONES } from './zones.js';
import { packAtlas, flatten } from './atlas.js';

const PI = Math.PI;
// yaw about Y, then lay the (vertical) local Y axis down along local X
function matMulYawRoll(yaw) { return matEuler(0, yaw, PI / 2); }

export const Z = { VEST: 0, NAVE: 1, STACKS: 2, GEODE: 3, BATHS: 4, GARDEN: 5, ATRIUM: 6, DESERT: 7 };

// ------------------------------------------------------------ shared helpers
function annulus(b, cx, cz, rIn, rOut, a0, a1, yTop, thick, surf, o = {}) {
  const n = Math.max(1, Math.ceil(Math.abs(a1 - a0) / 0.09));
  const rm = (rIn + rOut) / 2;
  const at = (r, a, y) => [cx + r * Math.cos(a), y, cz + r * Math.sin(a)];
  const rows = (fn) => { const R = []; for (let i = 0; i <= n; i++) R.push(fn(a0 + (a1 - a0) * i / n, i)); return R; };
  const yb = yTop - thick;
  // top / bottom
  for (const [y, ny] of [[yTop, 1], [yb, -1]]) {
    if (ny < 0 && o.noBottom) continue;
    b.grid(rows((a) => [at(rIn, a, y), at(rOut, a, y)]), rows(() => [[0, ny, 0], [0, ny, 0]]),
      rows((a) => [[(a - a0) * rm, 0], [(a - a0) * rm, rOut - rIn]]), null, surf, o);
  }
  // rims
  for (const [r, sg] of [[rIn, -1], [rOut, 1]]) {
    if (sg > 0 && o.noOuter) continue;
    b.grid(rows((a) => [at(r, a, yb), at(r, a, yTop)]), rows((a) => { const nn = [sg * Math.cos(a), 0, sg * Math.sin(a)]; return [nn, nn]; }),
      rows((a) => [[(a - a0) * r, yb], [(a - a0) * r, yTop]]), null, o.rimSurf || surf, o);
  }
  if (!o.noEnds) {
    for (const [a, sg] of [[a0, -1], [a1, 1]]) {
      const dirSign = Math.sign(a1 - a0) * sg;
      const tn = [-Math.sin(a) * dirSign, 0, Math.cos(a) * dirSign];
      b.poly([at(rIn, a, yb), at(rOut, a, yb), at(rOut, a, yTop), at(rIn, a, yTop)], o.rimSurf || surf, { n: tn });
    }
  }
  if (o.collide !== false) {
    for (let i = 0; i < n; i++) {
      const am = a0 + (a1 - a0) * (i + 0.5) / n, da = Math.abs(a1 - a0) / n;
      const p = at(rm, am, 0);
      b.col.addBox(p[0], yTop - thick / 2, p[2], (rOut - rIn) / 2, thick / 2, rOut * da / 2 * 1.06, -am);
    }
  }
}

function candle(b, x, y, z, color = [1.0, 0.72, 0.42], intensity = 0.9, radius = 7) {
  b.box(x, y + 0.14, z, 0.09, 0.28, 0.09, 'glow_warm', { collide: false, skip: 'b' });
  b.light([x, y + 0.45, z], color, intensity, radius, true);
}

// ------------------------------------------------------------ 0: vestibule
function buildVestibule(b, r) {
  b.zone = Z.VEST; b.farY = 7; b.lowY = -2; b.farDens = 0.5;
  b.slab(-1.4, -0.7, 1.5, 1.4, 0, 47, 'marble');
  b.cyl(0, 48.5, 3.4, -0.7, 0, 'marble', { caps: 'tb', sides: 28 });
  b.cyl(0, 48.5, 3.6, -1.6, -0.7, 'gold', { caps: 'b', sides: 28, collide: false });
  b.slab(-1.65, 0, 1.5, -1.4, 0.75, 45.2, 'marble');
  b.slab(1.4, 0, 1.5, 1.65, 0.75, 45.2, 'marble');
  // twisting tunnel of gilded frames
  for (let k = 0; k < 18; k++) {
    const z = 45 - k * 2.4;
    const roll = k * 0.105 + r.range(-0.02, 0.02);
    const big = k % 3 === 1;
    const h = big ? 3.3 : 2.65, t = big ? 0.24 : 0.32, dpt = big ? 0.3 : 0.55;
    const M = matEuler(0, 0, roll);
    for (const [lx, ly, sx, sy] of [[0, h + t / 2, 2 * (h + t), t], [0, -h - t / 2, 2 * (h + t), t], [-h - t / 2, 0, t, 2 * h], [h + t / 2, 0, t, 2 * h]]) {
      const p = apply(M, [lx, ly, 0]);
      b.box(p[0], 2.1 + p[1], z + p[2], sx, sy, dpt, 'gold', { m: M, collide: false, dens: 0.7 });
    }
  }
  // floating paintings in the void, facing the causeway
  for (let i = 0; i < 9; i++) {
    const side = i % 2 ? 1 : -1;
    const x = side * r.range(6, 16), y = r.range(-3, 9), z = r.range(6, 44);
    const w = r.range(2.8, 6.5), h = w * r.range(0.6, 1.3);
    const yaw = -side * PI / 2 + r.range(-0.35, 0.35);
    const M = matEuler(r.range(-0.15, 0.15), yaw, r.range(-0.12, 0.12));
    b.box(x, y, z, w + 0.5, h + 0.5, 0.25, 'gold', { m: M, collide: false, dens: 0.6 });
    const f = apply(M, [0, 0, 0.14]);
    b.box(x + f[0], y + f[1], z + f[2], w, h, 0.04, 'canvas', { m: M, collide: false, skip: 'zbtxX', dens: 0.9 });
  }
  for (let z = 40, s = 1; z > 4; z -= 7.5, s = -s) candle(b, s * 1.52, 0.75, z);
  // impossible fragments adrift in the void: an inverted arcade and a tilted stair to nowhere
  const d0 = b.densMul; b.densMul = 0.3;
  b.pushTransform(matEuler(0.1, 0.35, PI + 0.12), [-34, 26, 20]);
  for (let k = 0; k < 3; k++) b.archWall(-10.5 + k * 7, 0, 0, 7, 12, 1.4, 0, 5.2, 3.5, 'stone', { pointed: 0.5, collide: false, noTop: k !== 1 });
  for (let k = 0; k < 4; k++) b.column(-10.5 + k * 7, 0, 0.6, -9, 0, 'stone', { collide: false });
  b.popTransform();
  b.pushTransform(matEuler(-0.35, -0.6, 0.5), [30, -4, 30]);
  b.stairs(0, 0, 0, 3, 0, 9, 'marble', { collide: false, run: 0.4 });
  b.box(5, 8.7, 0, 3, 0.6, 3.4, 'marble', { collide: false });
  b.popTransform();
  b.pushTransform(matEuler(0.2, 1.2, -0.25), [24, 18, 8]);
  b.archWall(-6, 0, 0, 12, 16, 2, 0, 6, 7, 'darkstone', { pointed: 0.6, collide: false, ends: true });
  b.popTransform();
  b.densMul = d0;
  b.light([0, 3, 3], [1.0, 0.8, 0.55], 1.4, 12, false);
  b.zoneBox(-60, -100, 1.5, 60, 100, 90, Z.VEST);
}

// ------------------------------------------------------------ 1: drowned nave
const NAVE = { X: 14, ZN: -70, T: 1.5, Y0: -7.7, YC: -7.3, YW: -7.45 };
function buildNave(b, r) {
  b.zone = Z.NAVE; b.farY = 4; b.lowY = -Infinity; b.farDens = 0.5;
  const { X, ZN, T, Y0, YC, YW } = NAVE;
  b.solid(-X, -9, ZN, X, Y0, 0);
  b.quad([-X, YW, 0], [X, YW, 0], [X, YW, ZN], [-X, YW, ZN], 'water', { n: [0, 1, 0] });
  // outer walls (exterior faces are seen from the vestibule: lower density)
  b.slab(-X - T, -9, ZN - T, -X, 16.5, T, 'stone', { skip: 'bxtzZ' });
  b.slab(X, -9, ZN - T, X + T, 16.5, T, 'stone', { skip: 'bXtzZ' });
  // south wall with the entry door (at y=0 from the vestibule causeway)
  b.slab(-X - T, -9, 0, -3, 33, T, 'stone', { skip: 'b', fdens: { Z: 0.35, t: 0.3 } });
  b.slab(3, -9, 0, X + T, 33, T, 'stone', { skip: 'b', fdens: { Z: 0.35, t: 0.3 } });
  b.slab(-3, -9, 0, 3, 0, T, 'stone', { skip: 'b' });
  b.archWall(-3, T / 2, 0, 6, 8, T, 0, 3, 3.0, 'darkstone', { pointed: 0.5 });
  b.slab(-3, 8, 0, 3, 33, T, 'stone', { skip: 'b', fdens: { Z: 0.35 } });
  // exterior facade (the first landmark seen from the vestibule): towers, spires, rose, portal
  for (const sx of [-1, 1]) {
    b.slab(sx * 9.8, -9, T, sx * 15.8, 38, T + 5.5, 'stone', { skip: 'b', dens: 0.4 });
    b.crystal([sx * 12.8, 38, T + 2.75], [0, 1, 0], 0.01, 4.1, 'darkstone', { sides: 4, tip: 16, rot: PI / 4, baseScale: 1, dens: 0.3 });
    for (const y of [8, 20, 30]) {
      b.box(sx * 12.8, y, T + 5.52, 1.2, 4.2, 0.06, 'glow_warm', { collide: false, skip: 'z', dens: 0.5 });
      b.archFill(sx * 12.8, T + 5.55, 0, 1.2, y + 2.1, 'glow_warm', { pointed: 0.6, dens: 0.5 });
    }
    b.light([sx * 12.8, 20, T + 8], [1.0, 0.75, 0.45], 1.0, 14, true);
  }
  b.disc([0, 17, T + 0.06], [0, 0, 1], 6, 'fresco', { sides: 36, dens: 0.6 });
  for (let i = 0; i < 12; i++) b.box(0, 17, T + 0.22, 12.2, 0.24, 0.3, 'darkstone', { m: matEuler(0, 0, (i / 12) * PI), collide: false, dens: 0.4 });
  b.archWall(-4.2, T + 0.35, 0, 8.4, 11.5, 0.7, 0, 3.2, 3.0, 'darkstone', { pointed: 0.5, collide: false, ends: true, dens: 0.6 });
  b.crystal([0, 33, T / 2], [0, 1, 0], 0.01, 3.2, 'darkstone', { sides: 4, tip: 9, rot: PI / 4, baseScale: 1, dens: 0.3 });
  // north wall with the exit door at causeway level
  b.slab(-X - T, -9, ZN - T, -3, 33, ZN, 'stone', { skip: 'bzt' });
  b.slab(3, -9, ZN - T, X + T, 33, ZN, 'stone', { skip: 'bzt' });
  b.slab(-3, -9, ZN - T, 3, YC, ZN, 'stone', { skip: 'b' });
  b.archWall(-3, ZN - T / 2, 0, 6, 9, T, YC, 3.2, 3.2, 'darkstone', { pointed: 0.5 });
  b.slab(-3, YC + 9, ZN - T, 3, 33, ZN, 'stone', { skip: 'b' });
  // rose window + tracery
  b.disc([0, 12, ZN + 0.06], [0, 0, 1], 5.2, 'fresco', { sides: 32 });
  for (let i = 0; i < 12; i++) {
    const M = matEuler(0, 0, (i / 12) * PI);
    b.box(0, 12, ZN + 0.25, 10.6, 0.22, 0.3, 'darkstone', { m: M, collide: false });
  }
  b.cyl(0, ZN + 0.4, 0.2, 11.6, 12.4, 'darkstone', { collide: false, caps: '' });
  b.light([0, 12, ZN + 4], [0.85, 0.7, 1.0], 2.2, 30, false);
  // columns + pointed arcades
  for (let k = 1; k <= 9; k++) for (const sx of [-1, 1]) b.column(sx * 7, -7 * k, 0.72, Y0, 8, 'stone');
  for (let k = 0; k < 10; k++) for (const sx of [-1, 1]) {
    b.archWall(sx * 7, -7 * k, PI / 2, 7, 14, 1.4, 8, 5.6, 0, 'stone', { pointed: 0.55, collide: false, noTop: true });
  }
  // vaults
  b.vault(0, 0, 0, 14, 22, 0, ZN, 'stone', { pointed: 0.6, segs: 18 });
  for (const sx of [-1, 1]) b.vault(sx * 10.85, 0, 0, 6.3, 15, 0, ZN, 'stone', { pointed: 0.5, segs: 12 });
  // lancet windows + cold light shafts
  for (let k = 0; k < 10; k++) for (const sx of [-1, 1]) {
    const z = -3.5 - 7 * k;
    b.box(sx * 13.96, 8.5, z, 0.08, 7, 1.5, 'glow_cool', { collide: false, skip: sx > 0 ? 'X' : 'x' });
    b.archFill(sx * 13.9, z, sx > 0 ? -PI / 2 : PI / 2, 1.5, 12, 'glow_cool', { pointed: 0.6 });
    if ((k + (sx > 0 ? 1 : 0)) % 2 === 0) b.light([sx * 11, 8, z], [0.62, 0.8, 1.0], 1.1, 13, false);
  }
  // balcony + grand stair + causeway + transept crossing
  b.slab(-4, -0.8, -5, 4, 0, 0, 'marble');
  b.slab(-4, 0, -5, -2.1, 0.95, -4.75, 'marble');
  b.slab(2.1, 0, -5, 4, 0.95, -4.75, 'marble');
  b.slab(-4.25, 0, -5, -4, 0.95, 0, 'marble');
  b.slab(4, 0, -5, 4.25, 0.95, 0, 'marble');
  const end = b.stairs(0, -5, PI / 2, 4, 0, YC, 'marble', { base: -8.5, run: 0.36 });
  b.slab(-2, -8.5, ZN, 2, YC, end[1], 'marble');
  b.slab(-12.5, -8.5, -36.6, -2, YC, -33.4, 'marble');
  b.slab(2, -8.5, -36.6, 12.5, YC, -33.4, 'marble');
  // fallen column drums half-sunk in the aisles
  for (let i = 0; i < 4; i++) {
    const sx = i % 2 ? 1 : -1;
    const x = sx * r.range(9.5, 11.5), z = -12 - i * 13 - r.range(0, 5), yaw = r.range(-0.9, 0.9), L = r.range(3.5, 6);
    const rr = 0.7;
    b.pushTransform(matMulYawRoll(yaw), [x, Y0 + rr * 0.8, z]);
    b.cyl(0, 0, rr, -L / 2, L / 2, 'stone', { caps: 'tb', sides: 12, collide: false });
    b.popTransform();
    b.col.addBox(x, Y0 + rr * 0.4, z, L / 2, rr * 0.8, rr * 0.9, yaw);
  }
  // votive candles floating on the water
  for (let i = 0; i < 14; i++) {
    const sx = r.chance(0.5) ? -1 : 1;
    const x = sx * r.range(3, 12.5), z = r.range(-66, -10);
    b.box(x, YW + 0.06, z, 0.22, 0.12, 0.22, 'glow_warm', { collide: false, skip: 'b' });
    if (i % 2 === 0) b.light([x, YW + 0.7, z], [1.0, 0.66, 0.36], 0.8, 6, true);
  }
  b.light([0, -4, -20], [1.0, 0.75, 0.5], 0.9, 10, true);
  b.zoneBox(-X - T, -12, ZN - T, X + T, 40, T, Z.NAVE);
}

// ------------------------------------------------------------ 2: stacks
const STACKS = { CX: 0, CZ: -92, R: 12, NP: 20, YB: -8, YT: 42, T: 1, rIn: 8.6, rOut: 11.6 };
function buildStacks(b, r, doors) {
  b.zone = Z.STACKS; b.farY = doors.N ? doors.N + 7 : 34; b.lowY = -Infinity; b.farDens = 0.5;
  const { CX, CZ, R, NP, YB, YT, T, rIn, rOut } = STACKS;
  const YC = NAVE.YC;
  // corridor from the nave
  b.slab(-2, -8.5, -81, 2, YC, NAVE.ZN - NAVE.T, 'marble');
  b.slab(-3, -8.5, -81, -2, -0.9, NAVE.ZN - NAVE.T, 'darkstone', { skip: 'b' });
  b.slab(2, -8.5, -81, 3, -0.9, NAVE.ZN - NAVE.T, 'darkstone', { skip: 'b' });
  b.slab(-3, -1.3, -81, 3, -0.5, NAVE.ZN - NAVE.T, 'darkstone', { skip: '' });
  b.light([0, -3.5, -76], [1.0, 0.7, 0.4], 0.9, 8, true);
  // --- spiral flights with landings
  const rise = 0.18, run = 0.42, rMid = (rIn + rOut) / 2, da = run / rMid;
  const flights = [];
  // flight 1: from just west of the entrance up to the west landing (angle PI - 0.2)
  const n1 = Math.round((doors.W - YC) / rise);
  const a1s = PI - 0.2 - n1 * da;
  flights.push([a1s, YC, doors.W]);
  const nE = Math.round((2 * PI - 0.2 - (PI + 0.3)) / da);
  doors.E = doors.W + nE * rise;
  flights.push([PI + 0.3, doors.W, doors.E]);
  const nN = Math.round((3.5 * PI - 0.2 - (2 * PI + 0.3)) / da);
  doors.N = doors.E + nN * rise;
  flights.push([2 * PI + 0.3, doors.E, doors.N]);
  for (const [a0, y0, y1] of flights) {
    b.spiral(CX, CZ, rIn, rOut, a0, y0, y1, 'wood', { rise, run, thick: 0.32, parapet: 1.0, parapetSurf: 'gold', sideSurf: 'wood' });
  }
  // landings (with parapets, gaps where bridges meet)
  const landing = (a0, a1, y, gaps = []) => {
    annulus(b, CX, CZ, rIn, rOut + 0.35, a0, a1, y, 0.5, 'wood', { noOuter: true });
    let s = a0;
    const segs = [];
    for (const [g0, g1] of gaps) { segs.push([s, g0]); s = g1; }
    segs.push([s, a1]);
    for (const [p, q] of segs) if (q - p > 0.02) annulus(b, CX, CZ, rIn, rIn + 0.14, p, q, y + 1.0, 1.0, 'gold', { noBottom: true });
  };
  landing(PI - 0.2, PI + 0.3, doors.W);
  const gE = 1.25 / rIn;
  landing(2 * PI - 0.2, 2 * PI + 0.3, doors.E, [[2 * PI - gE, 2 * PI + gE]]);
  landing(3.5 * PI - 0.2, 3.5 * PI + 0.3, doors.N);
  // bridge across the shaft at the east landing height + west balcony
  const yE = doors.E;
  b.slab(-rIn - 0.2, yE - 0.45, CZ - 1.1, rIn + 0.2, yE, CZ + 1.1, 'wood');
  b.slab(-rIn, yE, CZ - 1.1, rIn, yE + 1.0, CZ - 0.96, 'gold');
  b.slab(-rIn, yE, CZ + 0.96, rIn, yE + 1.0, CZ + 1.1, 'gold');
  landing(PI - 0.4, PI + 0.4, yE, [[PI - gE, PI + gE]]);
  b.box(-10.6, yE + 0.55, CZ, 0.6, 1.1, 0.8, 'wood', {});
  b.sphere(-10.6, yE + 1.7, CZ, 0.35, 'glow_warm', { lon: 10, lat: 6 });
  b.light([-10.2, yE + 2.2, CZ], [1.0, 0.75, 0.45], 1.4, 10, true);
  // an inverted spiral stair hanging from the oculus down the middle of the shaft (Escher)
  b.pushTransform(matEuler(PI, 0, 0), [CX, YT - 0.2, CZ]);
  b.spiral(0, 0, 3.4, 5.4, 0.4, 0, 21, 'wood', { rise: 0.2, run: 0.44, thick: 0.28, parapet: 0.9, parapetSurf: 'gold', sideSurf: 'wood', dens: 0.6 });
  b.popTransform();
  // --- shaft wall panels (doors cut into some)
  const pw = 2 * (R + T) * Math.tan(PI / NP) * 1.005;
  const doorAt = { 0: YC, 5: doors.W, 15: doors.E, 10: doors.N };
  for (let k = 0; k < NP; k++) {
    const a = PI / 2 + k * 2 * PI / NP;
    const yaw = PI / 2 - a;
    const cx = CX + (R + T / 2) * Math.cos(a), cz = CZ + (R + T / 2) * Math.sin(a);
    const seg = (y0, y1) => {
      if (y1 - y0 < 0.01) return;
      b.box(cx, (y0 + y1) / 2, cz, pw, y1 - y0, T, 'books', { yaw, skip: 'tbxX', surfs: { Z: 'darkstone' }, fdens: { Z: 0.14 } });
    };
    if (doorAt[k] == null) { seg(YB, YT); continue; }
    const dy = doorAt[k], H = 6.5;
    seg(YB, dy);
    const t = apply(matYaw(yaw), [1, 0, 0]);
    b.archWall(cx - t[0] * pw / 2, cz - t[2] * pw / 2, yaw, pw, H, T, dy, 2.4, 2.6, 'darkstone', { pointed: 0.4, noTop: true, dens: 1 });
    seg(dy + H, YT);
    // door threshold floor through the wall thickness
    b.box(cx, dy - 0.25, cz, 2.5, 0.5, T + 0.02, 'darkstone', { yaw, skip: 'b' });
  }
  // wooden pilasters between panels
  for (let k = 0; k < NP; k++) {
    const a = PI / 2 + (k + 0.5) * 2 * PI / NP;
    const px = CX + (R - 0.05) * Math.cos(a), pz = CZ + (R - 0.05) * Math.sin(a);
    b.box(px, (YB + YT) / 2, pz, 0.45, YT - YB, 0.5, 'wood', { yaw: PI / 2 - a, skip: 'tbZ', dens: 0.45 });
  }
  // shelf ledges (visual)
  for (const y of [4.6, 17.5, 21.5, 35.5, 39.5]) b.cyl(CX, CZ, R - 0.3, y, y + 0.16, 'wood', { inward: true, sides: NP, caps: '', a0: PI / 2 + PI / NP, collide: false });
  // floor + basin + orb
  const ring = (rad, y) => { const p = []; for (let k = 0; k < NP; k++) { const a = PI / 2 + (k + 0.5) * 2 * PI / NP; p.push([CX + rad * Math.cos(a), y, CZ + rad * Math.sin(a)]); } return p; };
  b.poly(ring(R / Math.cos(PI / NP), YC), 'marble', { n: [0, 1, 0] });
  b.solid(CX - R - 0.5, -9, CZ - R - 0.5, CX + R + 0.5, YC, CZ + R + 0.5);
  b.cyl(CX, CZ, 3.6, YC, YC + 0.6, 'stone', { caps: '', sides: 24 });
  b.disc([CX, YC + 0.45, CZ], [0, 1, 0], 3.4, 'pool', { sides: 24 });
  b.sphere(CX, YC + 4.2, CZ, 0.9, 'glow_warm', { lon: 14, lat: 8 });
  b.light([CX, YC + 4.2, CZ], [1.0, 0.72, 0.4], 2.0, 16, false);
  // ceiling + oculus
  b.poly(ring(R / Math.cos(PI / NP), YT), 'wood', { n: [0, -1, 0] });
  b.disc([CX, YT - 0.05, CZ], [0, -1, 0], 3.6, 'glow_warm', { sides: 24 });
  b.light([CX, YT - 4, CZ], [1.0, 0.85, 0.6], 2.4, 34, false);
  // lamps along the spiral
  for (let i = 0; i < 14; i++) {
    const a = a1s + 0.3 + i * 0.62;
    let y = null;
    for (const [f0, y0, y1] of flights) {
      const n = Math.round((y1 - y0) / rise), aEnd = f0 + n * da;
      if (a >= f0 && a <= aEnd) y = y0 + (a - f0) / da * rise;
    }
    if (y == null) continue;
    candle(b, CX + (rIn + 0.07) * Math.cos(a), y + 1.1, CZ + (rIn + 0.07) * Math.sin(a), [1.0, 0.7, 0.4], 0.9, 8);
  }
  b.zoneBox(CX - R - 1, -12, CZ - R - 1, CX + R + 1, YT + 2, CZ + R + 1, Z.STACKS, 1);
  b.zoneBox(-3, -12, -81, 3, 0, NAVE.ZN - NAVE.T, Z.STACKS, 2);
}

// ------------------------------------------------------------ 3: geode
function buildGeode(b, r, doorY) {
  b.zone = Z.GEODE; b.farY = doorY + 9; b.lowY = -Infinity;
  const CX = -36, CZ = -92, RAD = 21;
  const X0 = -13; // shaft wall outer face (west)
  // tunnel
  b.slab(-17, doorY - 0.6, CZ - 1.6, X0, doorY, CZ + 1.6, 'darkstone');
  b.slab(-17, doorY - 0.6, CZ - 2.3, X0, doorY + 5.4, CZ - 1.6, 'rock', { skip: 'b' });
  b.slab(-17, doorY - 0.6, CZ + 1.6, X0, doorY + 5.4, CZ + 2.3, 'rock', { skip: 'b' });
  b.slab(-17, doorY + 4.4, CZ - 2.3, X0, doorY + 5.4, CZ + 2.3, 'rock', {});
  // basalt column terraces (hex grid, flat-top hexes)
  const hr = 1.0, dx = 1.5 * hr, dz = Math.sqrt(3) * hr;
  const entry = [-17, CZ];
  const tops = [];
  const hmap = new Map();
  const hkey = (x, z) => `${Math.round(x * 4)},${Math.round(z * 4)}`;
  for (let i = -16; i <= 16; i++) for (let k = -13; k <= 13; k++) {
    const x = CX + i * dx, z = CZ + k * dz + (i & 1 ? dz / 2 : 0);
    const d = Math.hypot(x - CX, z - CZ);
    if (d > RAD - 1.2 || x > -16.4) continue;
    const t = Math.hypot(x - entry[0], z - entry[1]);
    const wob = 0.9 * Math.sin(z * 0.35 + 1.2) + 0.7 * Math.sin(x * 0.27 - 0.4);
    let h = doorY - 0.36 * Math.max(0, Math.floor((t + wob) / 2.4));
    h = Math.max(h, -7.4);
    let deco = false;
    if (t > 6 && r.chance(0.07)) { h += r.range(1.5, 5.5); deco = true; }
    tops.push([x, z, h, deco, r.chance(0.2) ? 'rock' : 'darkstone']);
    hmap.set(hkey(x, z), h);
  }
  const rv = hr * 0.965;
  for (const [x, z, h, deco, surf] of tops) {
    // only the exposed parts of each side (neighbor lower / missing)
    const ranges = [];
    for (let s = 0; s < 6; s++) {
      const am = (s + 0.5) * PI / 3;
      const nh = hmap.get(hkey(x + Math.sqrt(3) * hr * Math.cos(am), z + Math.sqrt(3) * hr * Math.sin(am)));
      const lo = nh == null ? Math.min(h - 1.6, -7.2) : nh;
      ranges.push(lo < h - 0.02 ? [Math.max(lo, -8.6), h] : [0, 0]);
    }
    b.prismSides(x, z, rv, 6, 0, ranges, surf, { dens: 0.75 });
    const cap = [];
    for (let s = 0; s < 6; s++) cap.push([x + rv * Math.cos(s * PI / 3), h, z + rv * Math.sin(s * PI / 3)]);
    b.poly(cap, surf, { n: [0, 1, 0], dens: 0.8 });
    b.col.addCyl(x, z, hr * 0.98, -9, h);
  }
  b.poly([[CX - RAD, -8.5, CZ - RAD], [CX + RAD, -8.5, CZ - RAD], [CX + RAD, -8.5, CZ + RAD], [CX - RAD, -8.5, CZ + RAD]], 'rock', { n: [0, 1, 0], dens: 0.25 });
  // rock dome
  const rfn = (lon, lat) => 1 + Math.cos(lat) * (0.06 * Math.sin(3 * lon + 1.3) * Math.cos(2 * lat) + 0.04 * Math.sin(7 * lon + 4 * lat) + 0.025 * Math.sin(13 * lon - 3 * lat));
  b.sphere(CX, -8.5, CZ, RAD, 'rock', { inward: true, lat0: 0, lat1: 0.62, lon: 24, lat: 4, sectors: 3, scale: [1, 1.28, 1], rfn });
  b.sphere(CX, -8.5, CZ, RAD, 'rock', { inward: true, lat0: 0.62, lat1: PI / 2, lon: 24, lat: 5, sectors: 2, scale: [1, 1.28, 1], rfn, dens: 0.45 });
  // exterior shell (seen from the sky-bridge): outward dome, low density
  b.sphere(CX, -8.5, CZ, RAD + 0.8, 'rock', { lat0: 0.05, lat1: PI / 2, lon: 24, lat: 6, sectors: 2, scale: [1, 1.26, 1], rfn, dens: 0.12 });
  // giant central crystal + clusters
  b.crystal([CX - 1, -7, CZ + 1], [0.12, 1, -0.08], 15, 2.1, 'crystal', { tip: 4.5, collide: true });
  b.light([CX, 3, CZ], [0.72, 0.45, 1.0], 2.6, 28, false);
  const walk = tops.filter((t) => !t[3]);
  for (let c = 0; c < 13; c++) {
    const t = walk[Math.floor(r.next() * walk.length)];
    if (Math.hypot(t[0] - entry[0], t[1] - entry[1]) < 7) continue;
    const surf = c % 3 === 0 ? 'crystal2' : 'crystal';
    const m = r.int(3, 6);
    for (let j = 0; j < m; j++) {
      const dir = [r.range(-0.7, 0.7), 1, r.range(-0.7, 0.7)];
      b.crystal([t[0] + r.range(-0.4, 0.4), t[2] - 0.2, t[1] + r.range(-0.4, 0.4)], dir, r.range(1.2, j === 0 ? 5.5 : 3.2), r.range(0.22, j === 0 ? 0.7 : 0.45), surf, { collide: j === 0, rot: r.range(0, 1) });
    }
    b.light([t[0], t[2] + 2.2, t[1]], surf === 'crystal2' ? [0.35, 0.9, 1.0] : [0.75, 0.45, 1.0], 1.3, 10, false);
  }
  // crystals hanging from the dome
  for (let c = 0; c < 9; c++) {
    const lon = r.range(0, 2 * PI), lat = r.range(0.75, 1.3);
    const rr = RAD * rfn(lon, lat) * 1.0;
    const base = [CX + rr * Math.cos(lat) * Math.cos(lon), -8.5 + rr * Math.sin(lat) * 1.28 + 0.6, CZ + rr * Math.cos(lat) * Math.sin(lon)];
    const inward = [(CX - base[0]) * 0.05, 0, (CZ - base[2]) * 0.05];
    const surf = c % 2 ? 'crystal2' : 'crystal';
    const m = r.int(2, 4);
    for (let j = 0; j < m; j++) {
      const dir = [inward[0] + r.range(-0.35, 0.35), -1, inward[2] + r.range(-0.35, 0.35)];
      b.crystal([base[0] + r.range(-0.8, 0.8), base[1], base[2] + r.range(-0.8, 0.8)], dir, r.range(j ? 2.0 : 4.5, j ? 4.5 : 8), r.range(0.3, j ? 0.6 : 1.0), surf, { rot: r.range(0, 1), dens: 0.7 });
    }
  }
  b.zoneBox(-60, -12, -120, X0, 30, -64, Z.GEODE);
}

// ------------------------------------------------------------ 4: baths
function buildBaths(b, r, Y0) {
  b.zone = Z.BATHS; b.farY = Y0 + 7.6; b.lowY = -Infinity;
  const ZA = -104, ZB = -80, CZ = -92, X0 = 14, TL = 12, NT = 4, XE = X0 + NT * TL; // XE = 62
  const yk = (k) => Y0 - 1.2 * k;
  const YV = Y0 + 7.5; // vault spring
  const ylow = yk(NT - 1);
  // west wall with door (matches the shaft's east door)
  b.archWall(13.5, ZB, PI / 2, ZB - ZA, YV + 7.6 - Y0 + 1, 1, Y0 - 0.01, 2.4, 2.6, 'tile', { pointed: 0.4, at: 12 });
  b.slab(13, ylow - 1.5, ZA, 14, Y0, ZB, 'tile', { skip: 'b' });
  // terraces
  for (let k = 0; k < NT; k++) {
    const xs = X0 + k * TL, xe = xs + TL, y = yk(k), yb = y - 1.3;
    const sx = k > 0 ? xs + 2.28 : xs;
    b.slab(sx, yb, ZA, xe, y, -99, 'tile', { skip: 'b' });
    b.slab(xs, yb, -99, xe, y, -97.5, 'tile', { skip: 'b' });
    b.slab(sx, yb, -85, xe, y, ZB, 'tile', { skip: 'b' });
    b.slab(xs, yb, -86.5, xe, y, -85, 'tile', { skip: 'b' });
    b.slab(xs, yb, -97.5, xs + 1.5, y, -86.5, 'tile', { skip: 'b' });
    b.slab(xe - 1.5, yb, -97.5, xe, y, -86.5, 'tile', { skip: 'b' });
    b.slab(xs + 1.5, yb, -97.5, xe - 1.5, y - 0.45, -86.5, 'tile_dark', { skip: 'b' });
    b.quad([xs + 1.5, y - 0.15, -86.5], [xe - 1.5, y - 0.15, -86.5], [xe - 1.5, y - 0.15, -97.5], [xs + 1.5, y - 0.15, -97.5], 'pool', { n: [0, 1, 0] });
    if (k > 0) {
      b.stairs(xs, ZA + 2.5, 0, 5, yk(k - 1), y, 'tile', { base: yb, run: 0.38 });
      b.stairs(xs, ZB - 2.5, 0, 5, yk(k - 1), y, 'tile', { base: yb, run: 0.38 });
    }
    if (k < NT - 1) {
      // glowing cascade on the terrace riser
      b.quad([xe + 0.02, yk(k + 1) - 0.1, -88.5], [xe + 0.02, yk(k + 1) - 0.1, -95.5], [xe + 0.02, y - 0.1, -95.5], [xe + 0.02, y - 0.1, -88.5], 'pool', { n: [1, 0, 0] });
    }
    // arcades between walkways and pools
    for (let j = 0; j < 3; j++) {
      b.archWall(xs + j * 4, -99.25, 0, 4, 6.5, 0.5, y, 2.8, 3.1, 'tile', { pointed: 0 });
      b.archWall(xs + j * 4, -84.75, 0, 4, 6.5, 0.5, y, 2.8, 3.1, 'tile', { pointed: 0 });
    }
    // neon strips + light pools
    b.box((xs + xe) / 2, y + 2.4, ZA + 0.07, TL - 0.4, 0.1, 0.1, 'neon_pink', { collide: false });
    b.box((xs + xe) / 2, y + 2.4, ZB - 0.07, TL - 0.4, 0.1, 0.1, 'neon_cyan', { collide: false });
    b.light([(xs + xe) / 2, y + 3.5, CZ], k % 2 ? [0.3, 0.95, 1.0] : [1.0, 0.35, 0.8], 1.5, 13, false);
    b.light([(xs + xe) / 2, y + 2.6, -101.5], [1.0, 0.3, 0.75], 0.9, 8, false);
    b.light([(xs + xe) / 2, y + 2.6, -82.5], [0.25, 0.9, 1.0], 0.9, 8, false);
  }
  // outer walls, walkway ceilings, vault
  b.slab(X0, ylow - 1.5, ZA - 1, XE, YV + 0.5, ZA, 'tile', { skip: 'b', fdens: { z: 0.3 } });
  b.slab(X0, ylow - 1.5, ZB, XE, YV + 0.5, ZB + 1, 'tile', { skip: 'b', fdens: { Z: 0.3 } });
  b.slab(X0, YV, ZA, XE, YV + 0.5, -99, 'tile', { collide: false });
  b.slab(X0, YV, -85, XE, YV + 0.5, ZB, 'tile', { collide: false });
  b.vault(0, CZ, PI / 2, 14, YV, X0, XE, 'tile', { segs: 14 });
  // roof (outside of the vault), seen from the sky-bridge
  b.ramp((X0 + XE) / 2, CZ - 6, 12, XE - X0, YV + 0.5, YV + 8.2, -PI / 2, 'tile_dark', { collide: false, noBottom: true, dens: 0.12, base: YV + 0.4 });
  b.ramp((X0 + XE) / 2, CZ + 6, 12, XE - X0, YV + 0.5, YV + 8.2, PI / 2, 'tile_dark', { collide: false, noBottom: true, dens: 0.12, base: YV + 0.4 });
  b.box((X0 + XE) / 2, YV - 0.1, -98.9, XE - X0, 0.14, 0.14, 'neon_pink', { collide: false });
  b.box((X0 + XE) / 2, YV - 0.1, -85.1, XE - X0, 0.14, 0.14, 'neon_cyan', { collide: false });
  // east wall with a great arch onto the garden
  b.archWall(XE + 0.5, ZB, PI / 2, ZB - ZA, YV + 7.6 - ylow + 0.5, 1, ylow, 8, 4.2, 'tile', { pointed: 0 });
  b.slab(XE, ylow - 1.5, ZA, XE + 1, ylow, ZB, 'tile', { skip: 'b' });
  b.zoneBox(13, -12, ZA - 1, XE + 1, 30, ZB + 1, Z.BATHS);
  return ylow;
}

// ------------------------------------------------------------ 5: garden
function buildGarden(b, r, YB) {
  b.zone = Z.GARDEN; b.farY = 6.5; b.lowY = -Infinity;
  const X0 = 63, X1 = 111, Z0 = -116, Z1 = -68, CX = 87, CZ = -92;
  const GA = 0.5, GB = -2.8, GC = -6.1, BOT = GC - 1, TOP = 26;
  // west wall (around the baths' arch), other walls: stone below, painted sky above
  const wall = (x0, z0, x1, z1, skip) => {
    const ext = x1 - x0 < 2 ? (x0 > X0 ? 'X' : 'x') : (z0 < Z0 ? 'z' : 'Z');
    b.slab(x0, BOT, z0, x1, GA + 5.5, z1, 'stone', { skip: 'b' + skip, fdens: { [ext]: 0.06 } });
    b.slab(x0, GA + 5.5, z0, x1, TOP, z1, 'fresco', { skip: 'b' + skip, surfs: { [ext]: 'stone' }, fdens: { x: 0.25, X: 0.25, z: 0.25, Z: 0.25, [ext]: 0.06 } });
  };
  b.slab(X0 - 1, BOT, Z0 - 1, X0, TOP, -104, 'stone', { skip: 'b' });
  b.slab(X0 - 1, BOT, -80, X0, TOP, Z1 + 1, 'stone', { skip: 'b' });
  b.slab(X0 - 1, BOT, -104, X0, YB - 1.5, -80, 'stone', { skip: 'b' });
  wall(X1, Z0 - 1, X1 + 1, Z1 + 1, 'zZt');
  wall(X0, Z0 - 1, X1, Z0, 'xXt');
  wall(X0, Z1, X1, Z1 + 1, 'xXt');
  b.slab(X0 - 1, TOP, Z0 - 1, X1 + 1, TOP + 0.6, Z1 + 1, 'fresco', { collide: false, fdens: { t: 0.1 } });
  b.disc([CX, TOP - 0.05, CZ], [0, -1, 0], 4.5, 'glow_warm', { sides: 28 });
  b.light([CX, TOP - 5, CZ], [1.0, 0.88, 0.62], 2.6, 48, false);
  // terraces A (perimeter) / B / C (parterre)
  b.slab(X0, BOT, Z0, X1, GA, Z0 + 6, 'marble', { skip: 'bzxX' });
  b.slab(X0, BOT, Z1 - 6, X1, GA, Z1, 'marble', { skip: 'bZxX' });
  b.slab(X0, BOT, Z0 + 6, X0 + 6, GA, Z1 - 6, 'marble', { skip: 'bxzZ' });
  b.slab(X1 - 6, BOT, Z0 + 6, X1, GA, Z1 - 6, 'marble', { skip: 'bXzZ' });
  b.slab(X0 + 6, BOT, Z0 + 6, X1 - 6, GB, Z0 + 12, 'stone', { skip: 'bzxX' });
  b.slab(X0 + 6, BOT, Z1 - 12, X1 - 6, GB, Z1 - 6, 'stone', { skip: 'bZxX' });
  b.slab(X0 + 6, BOT, Z0 + 12, X0 + 12, GB, Z1 - 12, 'stone', { skip: 'bxzZ' });
  b.slab(X1 - 12, BOT, Z0 + 12, X1 - 6, GB, Z1 - 12, 'stone', { skip: 'bXzZ' });
  b.slab(X0 + 12, BOT, Z0 + 12, X1 - 12, GC, Z1 - 12, 'hedge', { skip: 'bxXzZ', dens: 0.8 });
  // balcony from the baths + double stair down to A
  b.slab(X0, YB - 0.6, -96, X0 + 4, YB, -88, 'marble');
  b.slab(X0 + 3.8, YB, -96, X0 + 4, YB + 0.95, -88, 'marble');
  b.stairs(X0 + 2, -96, PI / 2, 4, YB, GA, 'marble', { base: GA - 0.5, run: 0.36 });
  b.stairs(X0 + 2, -88, -PI / 2, 4, YB, GA, 'marble', { base: GA - 0.5, run: 0.36 });
  // stairs A->B and B->C on all four sides
  const sides = [[0, [X0 + 6, CZ]], [PI, [X1 - 6, CZ]], [PI / 2, [CX, Z1 - 6]], [-PI / 2, [CX, Z0 + 6]]];
  for (const [yaw, [x, z]] of sides) {
    b.stairs(x, z, yaw, 4, GA, GB, 'marble', { base: GB - 0.5, run: 0.3 });
    const d = apply(matYaw(yaw), [6, 0, 0]);
    b.stairs(x + d[0], z + d[2], yaw, 4, GB, GC, 'marble', { base: GC - 0.5, run: 0.3 });
  }
  // fountain
  b.cyl(CX, CZ, 5, GC, GC + 0.6, 'marble', { caps: '', sides: 28 });
  b.disc([CX, GC + 0.42, CZ], [0, 1, 0], 4.8, 'pool', { sides: 28 });
  b.solid(CX - 4.6, GC, CZ - 4.6, CX + 4.6, GC + 0.3, CZ + 4.6);
  b.cyl(CX, CZ, 0.5, GC + 0.3, GC + 4.2, 'marble', { caps: '' });
  b.cyl(CX, CZ, 2.1, GC + 3.0, GC + 3.35, 'marble', { caps: 'tb', collide: false });
  b.sphere(CX, GC + 4.9, CZ, 0.7, 'gold', { lon: 14, lat: 8 });
  b.light([CX, GC + 6.5, CZ], [1.0, 0.8, 0.55], 1.4, 14, false);
  // hedge parterres in the four quadrants of C
  for (const qx of [-1, 1]) for (const qz of [-1, 1]) {
    const cx = CX + qx * 6.5, cz = CZ + qz * 6.5;
    for (const [rr, gap] of [[4.4, 1.2], [2.4, 1.0]]) {
      const ht = rr > 3 ? 1.5 : 1.1;
      b.slab(cx - rr, GC, cz - rr, cx - gap, GC + ht, cz - rr + 0.7, 'hedge', { skip: 'b' });
      b.slab(cx + gap, GC, cz - rr, cx + rr, GC + ht, cz - rr + 0.7, 'hedge', { skip: 'b' });
      b.slab(cx - rr, GC, cz + rr - 0.7, cx - gap, GC + ht, cz + rr, 'hedge', { skip: 'b' });
      b.slab(cx + gap, GC, cz + rr - 0.7, cx + rr, GC + ht, cz + rr, 'hedge', { skip: 'b' });
      b.slab(cx - rr, GC, cz - rr + 0.7, cx - rr + 0.7, GC + ht, cz - gap, 'hedge', { skip: 'b' });
      b.slab(cx - rr, GC, cz + gap, cx - rr + 0.7, GC + ht, cz + rr - 0.7, 'hedge', { skip: 'b' });
      b.slab(cx + rr - 0.7, GC, cz - rr + 0.7, cx + rr, GC + ht, cz - gap, 'hedge', { skip: 'b' });
      b.slab(cx + rr - 0.7, GC, cz + gap, cx + rr, GC + ht, cz + rr - 0.7, 'hedge', { skip: 'b' });
    }
    b.sphere(cx, GC + 0.9, cz, 0.9, 'leaf', { lon: 12, lat: 6, scale: [1, 1, 1] });
    b.col.addCyl(cx, cz, 0.9, GC, GC + 1.8);
  }
  // cypress trees on terrace B
  const tree = (x, z, y) => {
    const hgt = r.range(5.5, 8.5);
    b.cyl(x, z, 0.22, y, y + 1.2, 'wood', { caps: '', sides: 8, collide: false });
    b.sphere(x, y + 1.0 + hgt / 2, z, 1.15, 'leaf', { lon: 12, lat: 8, scale: [1, hgt / 2.3, 1], dens: 0.8 });
    b.col.addCyl(x, z, 0.7, y, y + hgt);
  };
  for (let i = 0; i < 8; i++) {
    const t = -15 + i * 4.3;
    if (Math.abs(t) < 3) continue;
    tree(X0 + 9, CZ + t, GB); tree(X1 - 9, CZ + t, GB);
    tree(CX + t, Z0 + 9, GB); tree(CX + t, Z1 - 9, GB);
  }
  for (const [x, z] of [[X0 + 3, Z0 + 3], [X1 - 3, Z0 + 3], [X0 + 3, Z1 - 3], [X1 - 3, Z1 - 3]]) {
    b.cyl(x, z, 0.35, GA, GA + 2.2, 'marble', { caps: 't' });
    candle(b, x, GA + 2.2, z, [1.0, 0.78, 0.5], 1.0, 9);
  }
  b.zoneBox(X0 - 1, -12, Z0 - 1, X1 + 1, TOP + 1, Z1 + 1, Z.GARDEN);
}

// ------------------------------------------------------------ 6: atrium
function buildAtrium(b, r, yN) {
  b.zone = Z.ATRIUM; b.farY = yN + 9; b.lowY = yN - 5; b.farDens = 0.3;
  const X0 = -30, X1 = 30, ZS = -116, ZN = -176, T = 1.5, YB = -10, YT = 50;
  // sky-bridge from the library's north door
  b.slab(-1.5, yN - 0.6, ZS, 1.5, yN, -104.9, 'concrete');
  b.slab(-1.5, yN, ZS, -1.3, yN + 1.0, -105.2, 'concrete');
  b.slab(1.3, yN, ZS, 1.5, yN + 1.0, -105.2, 'concrete');
  // shell: south wall with door, north wall with the portal, sides
  b.slab(X0 - T, YB, ZS, -1.6, YT, ZS + T, 'concrete', { skip: 'b', fdens: { Z: 0.25, t: 0.2 } });
  b.slab(1.6, YB, ZS, X1 + T, YT, ZS + T, 'concrete', { skip: 'b', fdens: { Z: 0.25, t: 0.2 } });
  b.slab(-1.6, YB, ZS, 1.6, yN - 0.6, ZS + T, 'concrete', { skip: 'b' });
  b.slab(-1.6, yN + 3.6, ZS, 1.6, YT, ZS + T, 'concrete', { skip: 'b' });
  const PX0 = -9, PX1 = 5, PY0 = 22, PY1 = 46;
  b.slab(X0 - T, YB, ZN - T, PX0, YT, ZN, 'concrete', { skip: 'b', fdens: { z: 0.3, t: 0.2 } });
  b.slab(PX1, YB, ZN - T, X1 + T, YT, ZN, 'concrete', { skip: 'b', fdens: { z: 0.3, t: 0.2 } });
  b.slab(PX0, YB, ZN - T, PX1, PY0 - 0.6, ZN, 'concrete', { skip: 'b', fdens: { z: 0.3 } });
  b.slab(PX0, PY1, ZN - T, PX1, YT, ZN, 'concrete', { skip: 'b' });
  b.slab(X0 - T, YB, ZN, X0, YT, ZS, 'concrete', { skip: 'bzZ', fdens: { x: 0.12, t: 0.15 } });
  b.slab(X1, YB, ZN, X1 + T, YT, ZS, 'concrete', { skip: 'bzZ', fdens: { X: 0.12, t: 0.15 } });
  b.slab(X0 - T, YT, ZN - T, X1 + T, YT + 1, ZS + T, 'concrete', { collide: false, fdens: { t: 0.15 } });
  b.quad([X0, YB + 0.5, ZS], [X1, YB + 0.5, ZS], [X1, YB + 0.5, ZN], [X0, YB + 0.5, ZN], 'water', { n: [0, 1, 0], dens: 0.35 });
  // light slits + skylights
  for (let z = ZS - 5; z > ZN + 3; z -= 7) {
    for (const sx of [-1, 1]) b.box(sx * (X1 - 0.04), 28, z, 0.08, 34, 0.5, 'glow_warm', { collide: false, skip: sx > 0 ? 'X' : 'x', dens: 0.5 });
  }
  for (let x = -20; x <= 20; x += 13.3) for (let z = ZS - 10; z > ZN; z -= 15) {
    b.box(x, YT - 0.05, z, 4, 0.1, 4, 'glow_cool', { collide: false, skip: 't', dens: 0.4 });
  }
  // walkable monoliths
  const mono = (x0, z0, x1, z1, top, bot) => b.slab(x0, bot, z0, x1, top, z1, 'concrete', { fdens: { b: 0.3 } });
  const P3 = yN + 5.5, P5 = 22;
  mono(-5, -126, 5, ZS, yN, yN - 14);
  b.slab(-0.9, yN - 0.6, -138, 0.9, yN, -126, 'concrete');
  mono(-6, -146, 6, -138, yN, yN - 18);
  b.ramp(15, -142, 18, 2.2, yN, P3, 0, 'concrete', { thick: 0.7 });
  mono(24, -146, 29, -138, P3, P3 - 10);
  b.slab(25.6, P3 - 0.6, -160, 27.4, P3, -146, 'concrete');
  mono(22, -166, 29, -160, P3, P3 - 22);
  b.ramp(12, -163, 20, 2.2, P5, P3, 0, 'concrete', { thick: 0.7 });
  mono(-6, -168, 2, -158, P5, P5 - 26);
  b.slab(-6, P5 - 0.8, -182, 2, P5, -168, 'concrete');
  // floating decor monoliths (not on the path)
  const pathBoxes = [[-7, -128, 7, -114], [-2, -140, 2, -124], [-8, -148, 8, -136], [4, -145, 26, -139], [22, -168, 31, -136], [0, -166, 24, -160], [-8, -184, 4, -156]];
  let placed = 0;
  for (let tries = 0; tries < 200 && placed < 16; tries++) {
    const x = r.range(X0 + 4, X1 - 4), z = r.range(ZN + 5, ZS - 5);
    if (pathBoxes.some(([a, c, bb, d]) => x > a - 4 && x < bb + 4 && z > c - 4 && z < d + 4)) continue;
    const w = r.range(2.5, 6), h = r.range(10, 28), dpt = r.range(2.5, 7);
    const y = r.range(-2, 36);
    const tilt = r.chance(0.35);
    const M = tilt ? matEuler(r.range(-0.25, 0.25), r.range(0, PI), r.range(-0.25, 0.25)) : matYaw(r.range(0, PI));
    b.box(x, y, z, w, h, dpt, 'concrete', { m: M, collide: false, dens: 0.3 });
    placed++;
  }
  // pillars rising out of the abyss water (some carry the walkable monoliths)
  for (const [x, z, top, w] of [[0, -121, yN - 14, 3], [0, -142, yN - 18, 3.5], [26.5, -142, P3 - 10, 2.5], [25.5, -163, P3 - 22, 3], [-2, -163, P5 - 26, 3.2]]) {
    b.box(x, (YB + top) / 2, z, w, top - YB, w, 'concrete', { collide: false, dens: 0.35 });
  }
  for (let i = 0; i < 9; i++) {
    const x = r.range(X0 + 3, X1 - 3), z = r.range(ZN + 4, ZS - 4);
    if (pathBoxes.some(([a, c, bb, d]) => x > a - 3 && x < bb + 3 && z > c - 3 && z < d + 3)) continue;
    const top = r.range(-2, yN - 6), w = r.range(1.6, 3.2);
    b.box(x, (YB + top) / 2, z, w, top - YB, w, 'concrete', { yaw: r.range(0, PI), collide: false, dens: 0.3 });
  }
  // small floating cubes high above, tumbling
  for (let i = 0; i < 14; i++) {
    const x = r.range(X0 + 3, X1 - 3), z = r.range(ZN + 4, ZS - 4), y = r.range(yN + 9, YT - 5), e = r.range(0.8, 2.6);
    b.box(x, y, z, e, e, e, 'concrete', { m: matEuler(r.range(0, PI), r.range(0, PI), r.range(0, PI)), collide: false, dens: 0.35 });
  }
  // a colossal slab hanging from the ceiling + a band of light at walkway level
  b.box(-17, YT - 12, -150, 7, 24, 7, 'concrete', { collide: false, skip: 't', dens: 0.3 });
  b.box(-17, YT - 24.4, -150, 5, 0.3, 5, 'glow_warm', { collide: false, skip: 't' });
  b.light([-17, YT - 27, -150], [1.0, 0.8, 0.55], 1.6, 22, false);
  for (const [x0, z0, x1, z1] of [[X0 + 0.02, ZN, X0 + 0.1, ZS], [X1 - 0.1, ZN, X1 - 0.02, ZS]]) {
    b.slab(x0, yN - 3.2, z0, x1, yN - 3.0, z1, 'glow_cool', { collide: false, skip: 'tb', dens: 0.3 });
  }
  b.light([-2, 34, ZN - 6], [1.0, 0.72, 0.52], 2.6, 45, false);
  b.light([0, yN + 4, -121], [1.0, 0.95, 0.85], 1.2, 16, false);
  b.light([26, P3 + 4, -150], [1.0, 0.9, 0.75], 1.3, 20, false);
  b.light([-20, 30, -150], [0.8, 0.85, 1.0], 1.5, 30, false);
  b.light([20, 12, -130], [1.0, 0.8, 0.6], 1.2, 30, false);
  b.zoneBox(X0 - T, -14, ZN - T, X1 + T, YT + 2, ZS + T, Z.ATRIUM);
  b.zoneBox(-3, yN - 6, ZS, 3, yN + 8, -104.9, Z.ATRIUM, 3);
  return { portal: [-2, P5, -182] };
}

// ------------------------------------------------------------ 7: desert
function buildDesert(b, r, portal) {
  b.zone = Z.DESERT; b.farY = 14; b.lowY = -Infinity; b.farDens = 0.5;
  const [px, py, pz] = portal;
  // grand stair down from the portal platform
  const end = b.stairs(px, pz, PI / 2, 8, py, 0, 'sandstone', { base: -1.5, run: 0.4 });
  b.light([px, 3, end[1] - 4], [1.0, 0.7, 0.5], 1.4, 26, false);
  // floor tiles: denser near the processional axis
  const XS = 120, ZA = -178, ZB = -460, TS = 20;
  for (let x = -XS; x < XS; x += TS) for (let z = ZA; z > ZB; z -= TS) {
    const cx = x + TS / 2, cz = z - TS / 2;
    const near = Math.abs(cx - px) < 32 && cz > -420;
    const dens = near ? 0.42 : Math.abs(cx) < 70 ? 0.2 : 0.11;
    b.quad([x, 0, z], [x + TS, 0, z], [x + TS, 0, z - TS], [x, 0, z - TS], 'sand', { n: [0, 1, 0], dens });
  }
  b.solid(-XS, -3, ZB, XS, 0, ZA);
  // colossal arches receding along the axis
  let z = -250;
  for (let i = 0; i < 6; i++) {
    const s = 1 + i * 0.12 + r.range(-0.05, 0.1);
    const W = 34 * s, H = 38 * s, Tk = 6.5 * s, ow = 19 * s, hs = 15 * s;
    const yaw = r.range(-0.18, 0.18);
    const sink = -r.range(0, 5) * (i > 1 ? 1 : 0);
    const c = [px + r.range(-6, 6), z];
    const t = apply(matYaw(yaw), [1, 0, 0]);
    b.archWall(c[0] - t[0] * W / 2, c[1] - t[2] * W / 2, yaw, W, H, Tk, sink, ow, hs, 'sandstone', { pointed: i % 3 === 2 ? 0.35 : 0, ends: true, dens: i < 2 ? 0.42 : 0.24 });
    z -= r.range(34, 46) * s;
  }
  // distant arches off-axis for depth
  for (let i = 0; i < 5; i++) {
    const sx = i % 2 ? 1 : -1;
    const x = sx * r.range(55, 100), zz = r.range(-240, -430);
    const s = r.range(0.8, 1.5), yaw = r.range(-0.8, 0.8);
    const t = apply(matYaw(yaw), [1, 0, 0]);
    b.archWall(x - t[0] * 17 * s, zz - t[2] * 17 * s, yaw, 34 * s, 38 * s, 6 * s, -r.range(0, 8), 19 * s, 15 * s, 'sandstone', { ends: true, dens: 0.14 });
  }
  // obelisks + a half-buried marble sphere
  for (let i = 0; i < 7; i++) {
    const x = px + (i % 2 ? 1 : -1) * r.range(14, 45), zz = r.range(-200, -400), h = r.range(9, 22);
    if (r.chance(0.4)) b.box(x, h / 2 - 1, zz, 2.2, h, 2.2, 'sandstone', { m: matEuler(r.range(-0.25, 0.25), r.range(0, PI), r.range(-0.25, 0.25)), collide: 'aabb', dens: 0.5 });
    else b.box(x, h / 2 - 0.5, zz, 2.2, h, 2.2, 'sandstone', { yaw: r.range(0, PI), dens: 0.5 });
  }
  b.sphere(px + 34, -4, -318, 11, 'marble', { lat0: -0.3, lat1: PI / 2, lon: 24, lat: 10, dens: 0.4 });
  b.col.addCyl(px + 34, -318, 10, -4, 7);
  // a colossal inverted arch hanging in the sky (Magritte weather)
  b.pushTransform(matEuler(0.05, 0.3, PI), [px + 20, 120, -380]);
  b.archWall(-20, 0, 0, 40, 44, 7, 0, 22, 17, 'sandstone', { collide: false, ends: true, dens: 0.16 });
  b.popTransform();
  // two moons
  const eye = [px, 20, -260];
  for (const [p, rad, s] of [[[-120, 175, -610], 46, 'moon'], [[140, 95, -585], 24, 'moon2']]) {
    b.disc(p, [eye[0] - p[0], eye[1] - p[1], eye[2] - p[2]], rad, s, { sides: 40, abs: true, dens: 1.4 });
  }
  b.zoneBox(-700, -80, -1200, 700, 600, -177.5, Z.DESERT, -5);
}

// ------------------------------------------------------------ sky
function buildSky(b) {
  b.zone = Z.DESERT; b.farY = Infinity; b.lowY = -Infinity;
  b.sphere(0, 0, -170, 520, 'sky', { inward: true, sectors: 4, lon: 40, lat: 14, abs: true, dens: 0.42 });
}

// ------------------------------------------------------------ assemble
export function buildLevelData(seed = 1, opts = {}) {
  seed = (seed >>> 0) || 1;
  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const b = new Builder(SURFACES);
  const R = (name) => new RNG(hashSeed(seed, name));
  const doors = { W: -3.4 };
  buildVestibule(b, R('vestibule'));
  buildNave(b, R('nave'));
  buildStacks(b, R('stacks'), doors);
  buildGeode(b, R('geode'), doors.W);
  const ylow = buildBaths(b, R('baths'), doors.E);
  buildGarden(b, R('garden'), ylow);
  const { portal } = buildAtrium(b, R('atrium'), doors.N);
  buildDesert(b, R('desert'), portal);
  buildSky(b);
  b.col.killY = -30;

  const atlasSize = opts.atlasSize || 4096;
  const pack = packAtlas(b.charts, atlasSize, opts.pad || 4, opts.fill || 0.75);
  const flat = flatten(b.charts, pack);
  const zoneVols = b.zoneVols.slice().sort((p, q) => q.prio - p.prio);
  const eye = 1.6;
  const tour = [
    [0, eye, 47, 0, -0.02], [0, eye + 0.2, 24, 0, 0], [0, eye + 0.3, 5, 0, 0.02],
    [0, 3.2, -3, 0, -0.18], [0, -2, -22, 0, -0.05], [0, -4.8, -50, 0.12, 0.05],
    [0, -5.4, -76, 0, 0.1], [0, -4.2, -88, 0, 0.55], [2, 2.5, -92, -0.6, 0.35],
    [-6, 0.5, -92, 1.4, -0.05], [-20, -0.8, -92, 1.57, -0.12], [-30, -1.5, -98, 2.3, 0.1],
    [-22, 0.5, -88, -1.0, 0.05], [-6, 5, -92, -1.57, 0.1], [6, 11.5, -92, -1.57, 0.0],
    [20, 10, -92, -1.57, -0.12], [40, 8, -90, -1.57, -0.1], [58, 6.5, -92, -1.57, -0.15],
    [70, 5.5, -92, -1.57, -0.25], [86, 0, -84, -2.2, 0.05], [96, 2.5, -104, -3.6, 0.12],
    [80, 8, -95, -4.4, 0.1], [40, 18, -100, -4.71, 0.0], [8, 24, -103, -3.3, 0.0],
    [0, doors.N + 1.7, -110, -3.14, 0.02], [0, doors.N + 2, -132, -3.14, -0.1],
    [14, doors.N + 4, -142, -4.2, 0.0], [26, doors.N + 7.5, -152, -3.14, -0.05],
    [12, 26, -163, -1.57, -0.05], [-2, 24, -172, -3.14, 0.0], [-2, 18, -198, -3.14, -0.08],
    [-2, 6, -230, -3.14, 0.02], [-2, 4, -300, -3.14, 0.06],
  ];
  const t1 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  return {
    seed,
    atlas: { size: atlasSize, texelsPerMeter: pack.tpm, fill: pack.fill, used: pack.used, charts: b.charts.length, pad: pack.pad },
    flat,
    charts: b.charts,
    pack,
    surfaceTypes: SURFACES,
    zones: ZONES,
    lights: b.lights,
    zoneVols,
    spawn: { position: [0, 1.6, 47.5], yaw: 0 },   // eye position (feet at y=0)
    collision: b.col,
    tour,
    doors,
    buildMs: t1 - t0,
  };
}

export function zoneIndexAt(zoneVols, x, y, z) {
  for (const v of zoneVols) {
    if (x >= v.min[0] && x <= v.max[0] && y >= v.min[1] && y <= v.max[1] && z >= v.min[2] && z <= v.max[2]) return v.zone;
  }
  let best = 0, bd = Infinity;
  for (const v of zoneVols) {
    const dx = Math.max(v.min[0] - x, 0, x - v.max[0]), dy = Math.max(v.min[1] - y, 0, y - v.max[1]), dz = Math.max(v.min[2] - z, 0, z - v.max[2]);
    const d = dx * dx + dy * dy + dz * dz;
    if (d < bd) { bd = d; best = v.zone; }
  }
  return best;
}
