// HYPNAGOGIA audio — short-lived event voices (bells, glass, drips, ...).
// Each call builds a tiny graph, schedules it at time `t`, and tears it down
// when it ends. All are tuned for soft attacks (>= 2 ms), no content above
// ~9 kHz, and exponential tails that fall into the shared reverb.

import { rand, pick, makePanner, setPan } from './dsp.js';

const NYQ_SAFE = 9000;

function voiceOut(ctx, out, t, pan) {
  const p = makePanner(ctx);
  setPan(p, pan, t);
  p.connect(out);
  return p;
}

function env(ctx, t, peak, attack, decay, dest) {
  // decay = time to ~ -40 dB
  const g = ctx.createGain();
  g.gain.setValueAtTime(0, t);
  g.gain.linearRampToValueAtTime(peak, t + attack);
  g.gain.setTargetAtTime(0, t + attack, decay / 4.6);
  g.connect(dest);
  return g;
}

function osc(ctx, type, f, t, stop, dest, detune = 0) {
  const o = ctx.createOscillator();
  o.type = type;
  o.frequency.setValueAtTime(f, t);
  if (detune) o.detune.setValueAtTime(detune, t);
  o.connect(dest);
  o.start(t);
  o.stop(stop);
  return o;
}

function cleanup(src, nodes) {
  src.onended = () => { for (const n of nodes) { try { n.disconnect(); } catch (_) { /* ignore */ } } };
}

/** FM bell. ratio 1.41 = metallic/drowned, 2.0 = tonal church bell. */
export function bell(ctx, out, t, f, gain, o = {}) {
  const ratio = o.ratio ?? 1.41, index = o.index ?? 1.5, decay = o.decay ?? 7;
  const stop = t + decay * 1.15;
  const p = voiceOut(ctx, out, t, rand(-0.6, 0.6));
  const a = env(ctx, t, gain, 0.006, decay, p);
  const mg = ctx.createGain();
  mg.gain.setValueAtTime(f * index, t);
  mg.gain.setTargetAtTime(f * index * 0.15, t, decay / 10);
  const m = osc(ctx, 'sine', f * ratio, t, stop, mg);
  const c = osc(ctx, 'sine', f, t, stop, a);
  mg.connect(c.frequency);
  // quiet "hum" partial an octave below, slower decay: warmth
  const h = env(ctx, t, gain * 0.35, 0.02, decay * 1.1, p);
  const ho = osc(ctx, 'sine', f * 0.5, t, stop, h);
  cleanup(c, [a, mg, m, h, ho, p]);
}

/** Struck glass / free bar: inharmonic partials, higher ones die quicker. */
export function glass(ctx, out, t, f, gain, o = {}) {
  const decay = o.decay ?? 4.5;
  const p = voiceOut(ctx, out, t, rand(-0.75, 0.75));
  const parts = [[1, 1, 1], [1.0023, 0.5, 1.2], [2.756, 0.28, 0.45], [5.404, 0.1, 0.22]];
  const nodes = [p];
  let last = null, lastEnd = 0;
  for (const [r, a, d] of parts) {
    if (f * r > NYQ_SAFE) continue;
    const g = env(ctx, t, gain * a, 0.003, decay * d, p);
    const end = t + decay * d * 1.15;
    const o = osc(ctx, 'sine', f * r, t, end, g);
    if (end > lastEnd) { last = o; lastEnd = end; }
    nodes.push(g);
  }
  if (last) cleanup(last, nodes);
}

/** Wind-chime: brief, bright-ish, two partials. */
export function chime(ctx, out, t, f, gain) {
  glass(ctx, out, t, f, gain, { decay: 2.2 });
}

/** Kalimba / celesta-like soft pluck. */
export function pluck(ctx, out, t, f, gain, o = {}) {
  const decay = o.decay ?? 2.6;
  const p = voiceOut(ctx, out, t, rand(-0.5, 0.5));
  const g1 = env(ctx, t, gain, 0.004, decay, p);
  const g2 = env(ctx, t, gain * 0.28, 0.003, decay * 0.35, p);
  const g3 = env(ctx, t, gain * 0.07, 0.002, decay * 0.12, p);
  const o1 = osc(ctx, 'sine', f, t, t + decay * 1.1, g1);
  osc(ctx, 'sine', f * 2, t, t + decay * 0.5, g2);
  if (f * 3 < NYQ_SAFE) osc(ctx, 'sine', f * 3, t, t + decay * 0.2, g3);
  cleanup(o1, [g1, g2, g3, p]);
}

/** FM electric piano (Rhodes-ish), soft. */
export function epiano(ctx, out, t, f, gain) {
  const decay = 3.2, stop = t + decay * 1.1;
  const p = voiceOut(ctx, out, t, rand(-0.5, 0.5));
  const a = env(ctx, t, gain, 0.005, decay, p);
  const mg = ctx.createGain();
  mg.gain.setValueAtTime(f * 1.3, t);
  mg.gain.setTargetAtTime(f * 0.12, t, 0.18);
  const m = osc(ctx, 'sine', f, t, stop, mg);
  const c = osc(ctx, 'sine', f, t, stop, a);
  mg.connect(c.frequency);
  // slight detuned twin for chorus
  const b = env(ctx, t, gain * 0.5, 0.005, decay * 0.8, p);
  osc(ctx, 'sine', f, t, stop, b, 7);
  cleanup(c, [a, mg, m, b, p]);
}

/** Water drop: rising "plip". */
export function drip(ctx, out, t, f, gain) {
  const p = voiceOut(ctx, out, t, rand(-0.85, 0.85));
  const g = env(ctx, t, gain, 0.002, 0.16, p);
  const o = ctx.createOscillator();
  o.type = 'sine';
  o.frequency.setValueAtTime(f * 0.7, t);
  o.frequency.exponentialRampToValueAtTime(f * 1.35, t + 0.07);
  o.connect(g);
  o.start(t);
  o.stop(t + 0.3);
  cleanup(o, [g, p]);
}

/** Singing bowl: beating fundamental pair + inharmonic partials, long decay. */
export function bowl(ctx, out, t, f, gain) {
  const decay = 11;
  const p = voiceOut(ctx, out, t, rand(-0.4, 0.4));
  const parts = [[1, 1, 1], [1.0035, 0.8, 1], [2.71, 0.3, 0.6], [5.1, 0.07, 0.3]];
  const nodes = [p];
  let first = null;
  for (const [r, a, d] of parts) {
    if (f * r > NYQ_SAFE) continue;
    const g = env(ctx, t, gain * a, 0.025, decay * d, p);
    const o = osc(ctx, 'sine', f * r, t, t + decay * d * 1.1, g);
    if (!first) first = o;
    nodes.push(g);
  }
  cleanup(first, nodes);
}

/** Distant heavy thud (huge door / monolith settling). */
export function boom(ctx, out, t, f, gain, o = {}) {
  const noise = o.noise;
  const p = voiceOut(ctx, out, t, rand(-0.3, 0.3));
  const g = env(ctx, t, gain, 0.05, 3.2, p);
  const s = ctx.createOscillator();
  s.type = 'sine';
  s.frequency.setValueAtTime(52, t);
  s.frequency.exponentialRampToValueAtTime(38, t + 2.5);
  s.connect(g);
  s.start(t);
  s.stop(t + 3.8);
  const nodes = [g, p];
  if (noise) {
    const src = ctx.createBufferSource();
    src.buffer = noise;
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass'; lp.frequency.value = 140; lp.Q.value = 0.7;
    const ng = env(ctx, t, gain * 1.6, 0.08, 2.2, p);
    src.connect(lp); lp.connect(ng);
    src.start(t, rand(0, 3));
    src.stop(t + 3);
    nodes.push(lp, ng);
  }
  cleanup(s, nodes);
}

/** Library clock "tock": a resonant woody click. */
export function tick(ctx, out, t, f, gain, o = {}) {
  const noise = o.white || o.noise;
  if (!noise) return;
  const p = voiceOut(ctx, out, t, rand(-0.7, 0.7));
  const nodes = [p];
  let lastSrc = null;
  for (let k = 0; k < 2; k++) {
    const tt = t + k * 0.52;
    const src = ctx.createBufferSource();
    src.buffer = noise;
    const bp = ctx.createBiquadFilter();
    bp.type = 'bandpass'; bp.frequency.value = k ? 1450 : 1900; bp.Q.value = 9;
    const g = env(ctx, tt, gain * (k ? 0.7 : 1), 0.002, 0.06, p);
    src.connect(bp); bp.connect(g);
    src.start(tt, rand(0, 5));
    src.stop(tt + 0.12);
    nodes.push(bp, g);
    lastSrc = src;
  }
  cleanup(lastSrc, nodes);
}

/** Dream glitter: tiny high sine sparkle. */
export function sparkle(ctx, out, t, f, gain) {
  const d = rand(0.7, 1.6);
  const p = voiceOut(ctx, out, t, rand(-0.9, 0.9));
  const g = env(ctx, t, gain, 0.004, d, p);
  const o1 = osc(ctx, 'sine', f, t, t + d * 1.1, g);
  if (f * 2.01 < NYQ_SAFE) {
    const g2 = env(ctx, t, gain * 0.18, 0.003, d * 0.4, p);
    osc(ctx, 'sine', f * 2.01, t, t + d * 0.5, g2);
    cleanup(o1, [g, g2, p]);
  } else cleanup(o1, [g, p]);
}

export const EVENT_VOICES = { bell, glass, chime, pluck, epiano, drip, bowl, boom, tick, sparkle };

// ---------------------------------------------------------------- footsteps

const FOOT = {
  //          noise filter            dur    thump Hz/gain   ring?  splash
  stone:    { type: 'lowpass',  f: 1100, q: 0.8, dur: 0.09, thump: [75, 0.6] },
  concrete: { type: 'lowpass',  f: 750,  q: 0.7, dur: 0.1,  thump: [58, 0.8] },
  wood:     { type: 'bandpass', f: 520,  q: 2.5, dur: 0.07, thump: [160, 0.5] },
  grass:    { type: 'bandpass', f: 2600, q: 0.8, dur: 0.16, thump: [70, 0.2], grains: 3 },
  sand:     { type: 'bandpass', f: 1500, q: 0.5, dur: 0.2,  thump: [60, 0.25], attack: 0.025 },
  water:    { type: 'bandpass', f: 1200, q: 1.1, dur: 0.24, thump: [65, 0.3], splash: true },
  tile:     { type: 'bandpass', f: 2100, q: 1.8, dur: 0.05, thump: [95, 0.4] },
  crystal:  { type: 'lowpass',  f: 2400, q: 0.8, dur: 0.05, thump: [80, 0.3], ring: true },
  void:     { type: 'lowpass',  f: 600,  q: 0.5, dur: 0.15, thump: [90, 0.5], attack: 0.015 },
};

export function footstep(ctx, out, noise, t, material, intensity, ringFreq) {
  const m = FOOT[material] || FOOT.stone;
  const k = Math.max(0, Math.min(1, intensity));
  const gain = 0.05 + 0.1 * k;
  const p = voiceOut(ctx, out, t, rand(-0.12, 0.12));
  const nodes = [p];
  const grains = m.grains || 1;
  let lastSrc = null;
  for (let i = 0; i < grains; i++) {
    const tt = t + i * rand(0.025, 0.05);
    const src = ctx.createBufferSource();
    src.buffer = noise;
    src.playbackRate.value = rand(0.85, 1.15);
    const flt = ctx.createBiquadFilter();
    flt.type = m.type;
    flt.frequency.value = m.f * rand(0.85, 1.15) * (0.8 + 0.4 * k);
    flt.Q.value = m.q;
    const g = env(ctx, tt, gain / Math.sqrt(grains), m.attack || 0.004, m.dur, p);
    src.connect(flt); flt.connect(g);
    src.start(tt, rand(0, 5));
    src.stop(tt + m.dur * 1.4 + 0.25);
    nodes.push(flt, g);
    lastSrc = src;
  }
  if (m.thump) {
    const [hz, a] = m.thump;
    const g = env(ctx, t, gain * a, 0.004, 0.12, p);
    const o = ctx.createOscillator();
    o.frequency.setValueAtTime(hz * 1.4, t);
    o.frequency.exponentialRampToValueAtTime(hz, t + 0.05);
    o.connect(g); o.start(t); o.stop(t + 0.18);
    nodes.push(g);
  }
  if (m.splash) drip(ctx, out, t + 0.03, rand(500, 800), gain * 0.35);
  if (m.ring && ringFreq) glass(ctx, out, t, ringFreq, gain * 0.12, { decay: 1.5 });
  cleanup(lastSrc, nodes);
}

// ------------------------------------------------ mood-specific signature voices

/** Whale-like swell: slow triangle glide up a fourth and sagging back, very wet. */
export function whale(ctx, out, t, f, gain) {
  const dur = rand(4.5, 7);
  const p = voiceOut(ctx, out, t, rand(-0.8, 0.8));
  const lp = ctx.createBiquadFilter();
  lp.type = 'lowpass'; lp.frequency.value = 700; lp.Q.value = 1.2;
  const g = ctx.createGain();
  g.gain.setValueAtTime(0, t);
  g.gain.linearRampToValueAtTime(gain, t + dur * 0.35);
  g.gain.setTargetAtTime(0, t + dur * 0.55, dur * 0.12);
  const o = ctx.createOscillator();
  o.type = 'triangle';
  const up = pick([4 / 3, 3 / 2, 6 / 5]);
  o.frequency.setValueAtTime(f, t);
  o.frequency.exponentialRampToValueAtTime(f * up, t + dur * 0.4);
  o.frequency.exponentialRampToValueAtTime(f * 0.94, t + dur);
  const vib = ctx.createOscillator(); vib.frequency.value = rand(3.5, 5);
  const vg = ctx.createGain(); vg.gain.value = f * 0.006;
  vib.connect(vg); vg.connect(o.frequency);
  o.connect(lp); lp.connect(g); g.connect(p);
  o.start(t); vib.start(t); o.stop(t + dur + 0.6); vib.stop(t + dur + 0.6);
  cleanup(o, [lp, g, vg, vib, p]);
}

/** Slow heartbeat "lub-dub", felt more than heard. */
export function heart(ctx, out, t, f, gain, o = {}) {
  const p = voiceOut(ctx, out, t, 0);
  const nodes = [p];
  let last = null;
  for (const [dt, a, hz] of [[0, 1, 62], [0.27, 0.65, 54]]) {
    const tt = t + dt;
    const g = env(ctx, tt, gain * a, 0.012, 0.22, p);
    const s = ctx.createOscillator();
    s.frequency.setValueAtTime(hz * 1.5, tt);
    s.frequency.exponentialRampToValueAtTime(hz, tt + 0.06);
    s.connect(g); s.start(tt); s.stop(tt + 0.35);
    nodes.push(g);
    if (o.noise) {
      const src = ctx.createBufferSource(); src.buffer = o.noise;
      const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 220;
      const ng = env(ctx, tt, gain * a * 0.7, 0.01, 0.15, p);
      src.connect(lp); lp.connect(ng); src.start(tt, rand(0, 4)); src.stop(tt + 0.3);
      nodes.push(lp, ng);
    }
    last = s;
  }
  cleanup(last, nodes);
}

/** "Birdsong in reverse": swelling chirp that cuts off abruptly. */
export function rbird(ctx, out, t, f, gain) {
  const dur = rand(0.25, 0.55);
  const p = voiceOut(ctx, out, t, rand(-0.9, 0.9));
  const g = ctx.createGain();
  g.gain.setValueAtTime(0, t);
  g.gain.linearRampToValueAtTime(gain, t + dur);        // reversed envelope: slow rise...
  g.gain.linearRampToValueAtTime(0, t + dur + 0.025);   // ...abrupt (but click-free) stop
  const o = ctx.createOscillator();
  o.frequency.setValueAtTime(f * rand(1.2, 1.5), t);
  o.frequency.exponentialRampToValueAtTime(f, t + dur);
  const m = ctx.createOscillator(); m.frequency.value = rand(18, 34);
  const mg = ctx.createGain(); mg.gain.value = f * 0.06;
  m.connect(mg); mg.connect(o.frequency);
  o.connect(g); g.connect(p);
  o.start(t); m.start(t); o.stop(t + dur + 0.05); m.stop(t + dur + 0.05);
  cleanup(o, [g, m, mg, p]);
}

/** Fountain trickle: a quick cluster of tiny drops. */
export function trickle(ctx, out, t, f, gain) {
  const n = 5 + ((Math.random() * 8) | 0);
  let tt = t;
  for (let i = 0; i < n; i++) {
    drip(ctx, out, tt, f * rand(0.8, 1.9), gain * rand(0.4, 1));
    tt += rand(0.04, 0.13);
  }
}

/** Soft felt piano: slightly inharmonic partials, hammer-soft attack. */
export function piano(ctx, out, t, f, gain) {
  const decay = rand(4.5, 6.5);
  const p = voiceOut(ctx, out, t, rand(-0.4, 0.4));
  const B = 0.0004;
  const nodes = [p];
  let first = null;
  const amps = [1, 0.42, 0.2, 0.1, 0.05];
  amps.forEach((a, i) => {
    const n = i + 1;
    const fn = f * n * Math.sqrt(1 + B * n * n);
    if (fn > NYQ_SAFE * 0.6) return;
    const g = env(ctx, t, gain * a, 0.005, decay / (1 + i * 0.7), p);
    const o = osc(ctx, 'sine', fn, t, t + decay / (1 + i * 0.7) * 1.1, g, i === 0 ? 0 : rand(-2, 2));
    if (!first) first = o;
    nodes.push(g);
  });
  cleanup(first, nodes);
}

/** Lonely theremin: one sine voice gliding through the motif notes with vibrato. */
export function theremin(ctx, out, t, f, gain, o = {}) {
  const notes = o.notes && o.notes.length ? o.notes : [f];
  const gap = o.motifGap || 1.4;
  const dur = notes.length * gap + 1.2;
  const p = voiceOut(ctx, out, t, rand(-0.5, 0.5));
  const g = ctx.createGain();
  g.gain.setValueAtTime(0, t);
  g.gain.linearRampToValueAtTime(gain, t + 0.7);
  g.gain.setTargetAtTime(0, t + dur - 1.0, 0.35);
  const s = ctx.createOscillator();
  s.frequency.setValueAtTime(notes[0], t);
  notes.forEach((nf, i) => { if (i) s.frequency.setTargetAtTime(nf, t + i * gap, 0.12); });
  const vib = ctx.createOscillator();
  vib.frequency.value = rand(4.8, 5.8);
  const vg = ctx.createGain();
  vg.gain.setValueAtTime(0, t);
  vg.gain.linearRampToValueAtTime(notes[0] * 0.009, t + 1.4); // vibrato blooms in
  vib.connect(vg); vg.connect(s.frequency);
  const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 2200;
  s.connect(lp); lp.connect(g); g.connect(p);
  s.start(t); vib.start(t); s.stop(t + dur + 1.5); vib.stop(t + dur + 1.5);
  cleanup(s, [g, vib, vg, lp, p]);
}
theremin.wholeMotif = true;

/** Page rustle: a couple of airy noise grains. */
export function rustle(ctx, out, t, f, gain, o = {}) {
  const noise = o.white || o.noise;
  if (!noise) return;
  const p = voiceOut(ctx, out, t, rand(-0.8, 0.8));
  const nodes = [p];
  let last = null;
  const n = 2 + ((Math.random() * 3) | 0);
  for (let i = 0; i < n; i++) {
    const tt = t + i * rand(0.06, 0.2);
    const src = ctx.createBufferSource(); src.buffer = noise;
    src.playbackRate.value = rand(0.8, 1.2);
    const bp = ctx.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = rand(2200, 4200); bp.Q.value = 0.9;
    const g = env(ctx, tt, gain * rand(0.4, 1), rand(0.01, 0.04), rand(0.08, 0.25), p);
    src.connect(bp); bp.connect(g); src.start(tt, rand(0, 4)); src.stop(tt + 0.4);
    nodes.push(bp, g);
    last = src;
  }
  cleanup(last, nodes);
}

/** Reverse-swell "breath" of filtered noise that rises and dissolves: zone arrival. */
export function swell(ctx, out, t, noise, gain, dur = 3) {
  const src = ctx.createBufferSource();
  src.buffer = noise;
  src.loop = true;
  const bp = ctx.createBiquadFilter();
  bp.type = 'bandpass'; bp.Q.value = 0.8;
  bp.frequency.setValueAtTime(300, t);
  bp.frequency.exponentialRampToValueAtTime(2200, t + dur);
  const g = ctx.createGain();
  g.gain.setValueAtTime(0, t);
  g.gain.linearRampToValueAtTime(gain * 0.3, t + dur * 0.6);
  g.gain.linearRampToValueAtTime(gain, t + dur);
  g.gain.setTargetAtTime(0, t + dur, 0.35);
  const p = voiceOut(ctx, out, t, 0);
  src.connect(bp); bp.connect(g); g.connect(p);
  src.start(t, rand(0, 3));
  src.stop(t + dur + 2.5);
  cleanup(src, [bp, g, p]);
}

Object.assign(EVENT_VOICES, { whale, heart, rbird, trickle, piano, theremin, rustle });
