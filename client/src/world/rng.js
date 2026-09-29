// Tiny deterministic RNG (mulberry32) + helpers. No dependencies (runs in node too).

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function hashSeed(seed, salt) {
  let h = (seed ^ 0x9e3779b9) >>> 0;
  const s = String(salt);
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 0x01000193) >>> 0;
  h ^= h >>> 13; h = Math.imul(h, 0x5bd1e995) >>> 0; h ^= h >>> 15;
  return h >>> 0;
}

export class RNG {
  constructor(seed) { this.f = mulberry32(seed >>> 0); }
  next() { return this.f(); }
  range(a, b) { return a + (b - a) * this.f(); }
  int(a, b) { return a + Math.floor((b - a + 1) * this.f()); }
  pick(arr) { return arr[Math.floor(this.f() * arr.length) % arr.length]; }
  chance(p) { return this.f() < p; }
  jitter(v, amt) { return v + (this.f() * 2 - 1) * amt; }
  // independent stream so edits in one zone don't reshuffle another
  fork(salt) { return new RNG(hashSeed(Math.floor(this.f() * 4294967296), salt)); }
}
