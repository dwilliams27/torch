// HYPNAGOGIA audio — low-level DSP helpers (pure WebAudio, no samples).
// Everything here is generated once per AudioContext and shared/reused.

export const mtof = (m) => 440 * Math.pow(2, (m - 69) / 12);

export function rand(a, b) { return a + Math.random() * (b - a); }
export function pick(arr) { return arr[(Math.random() * arr.length) | 0]; }
export function expRand(mean) { return -Math.log(1 - Math.random() * 0.999) * mean; }

// Tiny deterministic PRNG so generated buffers (IR, noise) are the same every run.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function hashString(s) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
  return h >>> 0;
}

/**
 * Seamlessly looping stereo noise buffer. Channels are decorrelated (different
 * seeds) so a single looping source already sounds wide.
 * color: 'white' | 'pink' | 'brown'
 */
export function makeNoiseBuffer(ctx, color, seconds = 6, seed = 1) {
  const sr = ctx.sampleRate;
  const n = Math.floor(sr * seconds);
  const fade = Math.floor(sr * 0.25);
  const buf = ctx.createBuffer(2, n, sr);
  for (let ch = 0; ch < 2; ch++) {
    const rnd = mulberry32(seed * 977 + ch * 131 + color.length);
    const raw = new Float32Array(n + fade);
    let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0, last = 0;
    for (let i = 0; i < raw.length; i++) {
      const w = rnd() * 2 - 1;
      if (color === 'pink') {
        // Paul Kellet's refined pink filter
        b0 = 0.99886 * b0 + w * 0.0555179; b1 = 0.99332 * b1 + w * 0.0750759;
        b2 = 0.96900 * b2 + w * 0.1538520; b3 = 0.86650 * b3 + w * 0.3104856;
        b4 = 0.55000 * b4 + w * 0.5329522; b5 = -0.7616 * b5 - w * 0.0168980;
        raw[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362) * 0.11;
        b6 = w * 0.115926;
      } else if (color === 'brown') {
        last = (last + 0.02 * w) / 1.02;
        raw[i] = last * 3.5;
      } else {
        raw[i] = w * 0.5;
      }
    }
    // remove DC (matters for brown noise)
    let mean = 0;
    for (let i = 0; i < raw.length; i++) mean += raw[i];
    mean /= raw.length;
    const out = buf.getChannelData(ch);
    for (let i = 0; i < n; i++) out[i] = raw[i] - mean;
    // crossfade tail into head -> click-free loop
    for (let i = 0; i < fade; i++) {
      const w = i / fade;
      out[i] = (raw[i] - mean) * w + (raw[n + i] - mean) * (1 - w);
    }
    // normalise to RMS ~0.25
    let ss = 0;
    for (let i = 0; i < n; i++) ss += out[i] * out[i];
    const k = 0.25 / Math.sqrt(ss / n + 1e-12);
    for (let i = 0; i < n; i++) out[i] *= k;
  }
  return buf;
}

/**
 * Long, dark, diffuse hall impulse response. Exponentially decaying stereo noise
 * whose spectrum darkens over time (time-varying one-pole lowpass), a short
 * pre-delay and a handful of early reflections. Normalised to unit energy per
 * channel so the wet level is set purely by the return gain.
 */
export function makeImpulseResponse(ctx, { seconds = 6, rt60 = 5.2, preDelay = 0.025, bright = 7000, dark = 650, seed = 7 } = {}) {
  const sr = ctx.sampleRate;
  const n = Math.floor(sr * seconds);
  const buf = ctx.createBuffer(2, n, sr);
  const pre = Math.floor(preDelay * sr);
  const tau = rt60 / 6.908; // e^-6.908 = -60 dB
  for (let ch = 0; ch < 2; ch++) {
    const rnd = mulberry32(seed + ch * 7919);
    const d = buf.getChannelData(ch);
    let lp = 0, a = 0, env = 0, dEnv = 1;
    const tailStart = n * 0.9;
    const decayPerSample = Math.exp(-1 / (tau * sr));
    for (let i = pre; i < n; i++) {
      if (((i - pre) & 63) === 0) {
        // control-rate update (every 64 samples) of the darkening filter
        const t = (i - pre) / sr;
        const frac = Math.min(1, t / (seconds * 0.8));
        const fc = bright * Math.pow(dark / bright, Math.sqrt(frac));
        a = 1 - Math.exp(-2 * Math.PI * fc / sr);
        env = Math.exp(-t / tau);
        dEnv = decayPerSample;
      } else env *= dEnv;
      lp += a * ((rnd() * 2 - 1) - lp);
      const t = (i - pre) / sr;
      let e = env * (t < 0.06 ? t / 0.06 : 1); // soft onset, no crack
      if (i > tailStart) e *= 1 - (i - tailStart) / (n - tailStart); // fade the very tail to zero
      d[i] = lp * e;
    }
    // sparse early reflections (differ per channel -> width)
    for (let k = 0; k < 7; k++) {
      const at = pre + Math.floor((0.008 + rnd() * 0.07) * sr);
      if (at < n) d[at] += (rnd() < 0.5 ? -1 : 1) * (0.25 - k * 0.025);
    }
    let ss = 0;
    for (let i = 0; i < n; i++) ss += d[i] * d[i];
    const k = 1 / Math.sqrt(ss + 1e-12);
    for (let i = 0; i < n; i++) d[i] *= k;
  }
  return buf;
}

// Additive timbres for sustained voices (sine-phase harmonic amplitudes).
const TIMBRES = {
  sine:   [0, 1],
  soft:   [0, 1, 0.22, 0.07, 0.025],
  warm:   [0, 1, 0.45, 0.22, 0.12, 0.07, 0.04, 0.025, 0.015],
  organ:  [0, 1, 0.55, 0.12, 0.32, 0.04, 0.14, 0, 0.09, 0, 0.04],
  hollow: [0, 1, 0, 0.14, 0, 0.05, 0, 0.022, 0, 0.011],
  reed:   Array.from({ length: 20 }, (_, i) => (i === 0 ? 0 : Math.pow(i, -1.25))),
  glass:  [0, 1, 0, 0.3, 0.0, 0.0, 0.12, 0, 0, 0.05],
  choir:  [0, 1, 0.7, 0.45, 0.3, 0.2, 0.14, 0.1, 0.07, 0.05, 0.035, 0.025, 0.018],
};

export function makeWaves(ctx) {
  const waves = {};
  for (const [name, amps] of Object.entries(TIMBRES)) {
    const imag = new Float32Array(amps);
    const real = new Float32Array(amps.length);
    waves[name] = ctx.createPeriodicWave(real, imag);
  }
  return waves;
}

export function makePanner(ctx) {
  if (ctx.createStereoPanner) return ctx.createStereoPanner();
  // very old Safari fallback: equal-power PannerNode driven via a shim .pan param-less object
  const p = ctx.createPanner();
  p.panningModel = 'equalpower';
  p.pan = null; // no AudioParam; callers check for null
  p.setPanStatic = (x) => p.setPosition(x, 0, 1 - Math.abs(x));
  return p;
}

export function setPan(p, x, t) {
  if (p.pan) p.pan.setValueAtTime(x, t);
  else if (p.setPanStatic) p.setPanStatic(x);
}

/** Safe exponential approach: clamps and uses setTargetAtTime. */
export function glide(param, value, t, tau) {
  param.setTargetAtTime(value, t, Math.max(0.001, tau));
}
