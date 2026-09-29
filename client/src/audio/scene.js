// HYPNAGOGIA audio — one zone's sound world (drone + pad + noise bed + events).
// A Scene owns a handful of persistent oscillators (~14-18) that are retuned
// rather than recreated, plus short event voices spawned with lookahead.
// Scenes fade in/out through three "fade" gains (dry / reverb send / delay
// send) so crossfades also crossfade the wet signal.

import { PRESETS, SCALES } from './presets.js';
import { mtof, rand, pick, makePanner, setPan, glide } from './dsp.js';
import { EVENT_VOICES } from './voices.js';

// Global mix balance applied on top of the per-preset values (keeps the
// spectrum from collapsing into the sub-bass, which laptop/phone speakers
// can't reproduce anyway).
export const MIX = { drone: 0.62, fifth: 0.9, sub: 0.36, hum: 1.0, pad: 1.9, noise: 1.0, events: 1.45, padCutoff: 1.3 };

const FADE_IN_TAU = 2.6;   // ~8 s to settle
const FADE_OUT_TAU = 3.0;  // ~9 s to near silence

// Filters whose cutoff is modulated (LFO / glides) run their params at k-rate:
// coefficients are then computed once per 128-frame block instead of per sample,
// which roughly halves the scene's CPU. Inaudible for these slow sweeps.
function kfilter(ctx, type, f, q) {
  const b = ctx.createBiquadFilter();
  b.type = type; b.frequency.value = f; b.Q.value = q;
  try { b.frequency.automationRate = 'k-rate'; b.Q.automationRate = 'k-rate'; b.detune.automationRate = 'k-rate'; b.gain.automationRate = 'k-rate'; } catch (_) { /* older Safari */ }
  return b;
}

function voiceLead(prev, pcs, lo, hi) {
  const used = new Set();
  const out = [];
  for (const p of prev) {
    let best = p, bestD = 1e9;
    for (let m = lo; m <= hi; m++) {
      const pc = ((m % 12) + 12) % 12;
      if (!pcs.includes(pc)) continue;
      const d = Math.abs(m - p) + (used.has(pc) ? 4 : 0) + (out.includes(m) ? 99 : 0);
      if (d < bestD) { bestD = d; best = m; }
    }
    out.push(best);
    used.add(((best % 12) + 12) % 12);
  }
  return out;
}

export class Scene {
  constructor(amb, presetName, transpose = 0, key = presetName) {
    const ctx = amb.ctx;
    const P = PRESETS[presetName];
    this.amb = amb;
    this.ctx = ctx;
    this.P = P;
    this.name = presetName;
    this.key = key;
    this.root = P.root + transpose;
    this.scale = SCALES[P.scale];
    this.active = true;
    this.deadAt = Infinity;
    this.dead = false;
    this.sources = [];
    this.nodes = [];
    this.dream = 0;

    const t0 = ctx.currentTime + 0.03;
    this.t0 = t0;

    // --- fade + send structure
    const G = (v = 1) => { const g = ctx.createGain(); g.gain.value = v; this.nodes.push(g); return g; };
    this.fadeDry = G(0); this.fadeRev = G(0); this.fadeDly = G(0);
    this.fadeDry.connect(amb.dryIn);
    this.fadeRev.connect(amb.revIn);
    this.fadeDly.connect(amb.dlyIn);

    const layer = (rev, dly) => {
      const bus = G(1);
      bus.connect(this.fadeDry);
      if (rev > 0) { const s = G(rev); bus.connect(s); s.connect(this.fadeRev); }
      if (dly > 0) { const s = G(dly); bus.connect(s); s.connect(this.fadeDly); }
      return bus;
    };
    const S = P.send;
    this.droneBus = layer(S.drone, 0);
    this.padBus = layer(S.pad, S.delay * 0.35);
    this.noiseBus = layer(S.noise, 0);
    this.evtBus = layer(S.events, S.delay);

    const osc = (type, f, detune = 0) => {
      const o = ctx.createOscillator();
      if (typeof type === 'string' && ['sine', 'square', 'sawtooth', 'triangle'].includes(type)) o.type = type;
      else o.setPeriodicWave(amb.waves[type] || amb.waves.soft);
      o.frequency.value = f;
      o.detune.value = detune;
      o.start(t0);
      this.sources.push(o);
      return o;
    };
    const lfo = (rate, depth, target) => {
      const o = osc('sine', rate);
      const g = G(depth);
      o.connect(g); g.connect(target);
      return g;
    };

    // --- drone
    const D = P.drone;
    const droneRoot = this.root - (D.octDown ? 12 : 0);
    this.droneFilter = kfilter(ctx, 'lowpass', D.cutoff, D.q);
    this.nodes.push(this.droneFilter);
    this.droneFilter.connect(this.droneBus);
    lfo(D.lfoRate, Math.min(D.lfoDepth, D.cutoff * 0.7), this.droneFilter.frequency);
    this.droneOscs = [];
    // Two voices, unequal level: the slow beating adds life without full-depth
    // amplitude pulsing (equal levels would "wub" all the way to silence).
    for (const side of [-1, 1]) {
      const o = osc(D.wave, mtof(droneRoot), side < 0 ? 0 : D.detune * 1.3);
      const g = G(D.gain * MIX.drone * (side < 0 ? 1.3 : 0.7));
      const pan = makePanner(ctx); setPan(pan, side * 0.35, t0); this.nodes.push(pan);
      o.connect(g); g.connect(pan); pan.connect(this.droneFilter);
      this.droneOscs.push({ o, mult: 1 });
    }
    if (D.fifth > 0) {
      const o = osc('sine', mtof(droneRoot + 7), 3);
      const g = G(D.fifth * MIX.fifth);
      o.connect(g); g.connect(this.droneFilter);
      // the fifth slowly breathes in and out
      lfo(0.013 + Math.random() * 0.01, D.fifth * MIX.fifth * 0.8, g.gain);
      this.droneOscs.push({ o, mult: 1.5 });
    }
    if (D.sub > 0) {
      const o = osc('sine', mtof(droneRoot - 12));
      const g = G(D.sub * MIX.sub);
      o.connect(g); g.connect(this.droneBus);
      this.droneOscs.push({ o, mult: 0.5 });
    }

    // --- hum (electrical / room)
    if (P.hum) {
      const H = P.hum;
      const o = osc('sawtooth', mtof(this.root) * H.mult);
      const f = kfilter(ctx, 'lowpass', H.cutoff, H.q);
      const g = G(H.gain);
      o.connect(f); f.connect(g); g.connect(this.droneBus);
      this.nodes.push(f);
      lfo(H.trem, H.gain * 0.45, g.gain);
    }

    // --- pad
    const PD = P.pad;
    this.padFilter = kfilter(ctx, 'lowpass', PD.cutoff * MIX.padCutoff, PD.q);
    this.nodes.push(this.padFilter);
    lfo(PD.lfoRate, Math.min(PD.lfoDepth, PD.cutoff * 0.7) * MIX.padCutoff, this.padFilter.frequency);
    let padOut = this.padFilter;
    if (PD.formant) {
      // "choir": dry lowpass + two parallel vowel formants that drift between ah/oh
      const F = PD.formant;
      const sum = G(1);
      const dry = G(0.45);
      this.padFilter.connect(dry); dry.connect(sum);
      for (const [fc, ph] of [[F.f1, 1], [F.f2, -1]]) {
        const bp = kfilter(ctx, 'bandpass', fc, F.q);
        const g = G(F.gain);
        this.padFilter.connect(bp); bp.connect(g); g.connect(sum);
        lfo(F.sweep, ph * F.depth * (fc / F.f1) * 0.6, bp.frequency);
        this.nodes.push(bp);
      }
      padOut = sum;
    }
    if (PD.tremRate) {
      const trem = G(1 - PD.tremDepth * 0.5);
      padOut.connect(trem);
      lfo(PD.tremRate, PD.tremDepth * 0.5, trem.gain);
      padOut = trem;
    }
    padOut.connect(this.padBus);

    // pan motion: two slow LFOs in antiphase across the four voices
    const panA = G(0.55), panB = G(-0.55);
    const panL = osc('sine', 0.037 + Math.random() * 0.02);
    panL.connect(panA); panL.connect(panB);
    const panA2 = G(0.4), panB2 = G(-0.4);
    const panL2 = osc('sine', 0.023 + Math.random() * 0.015);
    panL2.connect(panA2); panL2.connect(panB2);

    const firstPcs = this._chordPcs(P.prog[0]);
    const lo = this.root + PD.lo, hi = this.root + PD.hi;
    const seed = [lo + 5, lo + 9, lo + 12, lo + 16].map((m) => Math.min(hi, m));
    this.padNotes = voiceLead(seed, firstPcs, lo, hi);
    this.chordPcs = firstPcs;
    this.chordIdx = 1;
    this.padVoices = this.padNotes.map((m, i) => {
      const f = mtof(m);
      const o1 = osc(PD.wave, f, -PD.detune);
      const o2 = osc(PD.wave, f, PD.detune * (0.8 + 0.4 * Math.random()));
      const g = G(PD.gain * MIX.pad * rand(0.7, 1));
      const g2 = G(0.6); // partial-depth chorus beating
      const pan = makePanner(ctx); this.nodes.push(pan);
      if (pan.pan) {
        pan.pan.value = 0;
        try { pan.pan.automationRate = 'k-rate'; } catch (_) { /* */ }
        [panA, panB, panA2, panB2][i].connect(pan.pan);
      }
      o1.connect(g); o2.connect(g2); g2.connect(g); g.connect(pan); pan.connect(this.padFilter);
      return { o1, o2, g };
    });

    // --- noise bed
    const N = P.noise;
    const src = ctx.createBufferSource();
    src.buffer = amb.noise[N.color];
    src.loop = true;
    src.playbackRate.value = rand(0.93, 1.07);
    this.noiseFilter = kfilter(ctx, N.type, N.freq, N.q);
    this.noiseGain = G(N.gain * MIX.noise);
    src.connect(this.noiseFilter); this.noiseFilter.connect(this.noiseGain); this.noiseGain.connect(this.noiseBus);
    this.nodes.push(this.noiseFilter);
    src.start(t0, rand(0, src.buffer.duration));
    this.sources.push(src);

    // --- schedulers
    this.nextChord = t0 + rand(P.chordSec[0] * 0.5, P.chordSec[0]);
    this.nextGust = t0 + rand(1, 3);
    this.events = P.events.map((e) => ({ ...e, next: t0 + rand(1.5, e.every[0] * 0.8 + 1.5) }));
  }

  _deg2midi(d) {
    const s = this.scale;
    return this.root + s[((d % 7) + 7) % 7] + 12 * Math.floor(d / 7);
  }

  _chordPcs(deg) {
    const n = this.P.chordSize || 4;
    const pcs = [];
    for (let k = 0; k < n; k++) pcs.push(((this._deg2midi(deg + 2 * k) % 12) + 12) % 12);
    return pcs;
  }

  _scalePcs() { return this.scale.map((x) => (this.root + x) % 12); }

  fadeIn(t) {
    const L = this.P.level;
    for (const g of [this.fadeDry, this.fadeRev, this.fadeDly]) {
      g.gain.cancelScheduledValues(t);
      g.gain.setTargetAtTime(L, t, FADE_IN_TAU);
    }
    this.active = true;
    this.deadAt = Infinity;
  }

  fadeOut(t, tau = FADE_OUT_TAU) {
    for (const g of [this.fadeDry, this.fadeRev, this.fadeDly]) {
      g.gain.cancelScheduledValues(t);
      g.gain.setTargetAtTime(0, t, tau);
    }
    this.active = false;
    this.deadAt = t + tau * 6.5;
  }

  destroy(t) {
    if (this.dead) return;
    this.dead = true;
    for (const s of this.sources) { try { s.stop(t); } catch (_) { /* already stopped */ } }
    const nodes = this.nodes;
    const last = this.sources[0];
    if (last) last.onended = () => { for (const n of nodes) { try { n.disconnect(); } catch (_) { /* */ } } };
  }

  setDream(a, t) {
    this.dream = a;
    glide(this.padFilter.frequency, this.P.pad.cutoff * MIX.padCutoff * (1 + 0.7 * a), t, 1.2);
    glide(this.droneFilter.frequency, this.P.drone.cutoff * (1 + 0.35 * a), t, 1.5);
  }

  /** Schedule everything whose time falls before `horizon`. */
  schedule(horizon) {
    if (this.dead) return;
    const P = this.P;
    while (this.nextChord < horizon) {
      this._applyChord(this.nextChord);
      this.nextChord += rand(P.chordSec[0], P.chordSec[1]);
    }
    while (this.nextGust < horizon) {
      const N = P.noise;
      const t = this.nextGust;
      const span = rand(N.gust[0], N.gust[1]);
      const g = N.gain * MIX.noise * (1 - N.gustDepth + N.gustDepth * Math.random());
      glide(this.noiseGain.gain, g, t, span / 3);
      const [f0, f1] = N.freqRange;
      glide(this.noiseFilter.frequency, f0 * Math.pow(f1 / f0, Math.random()), t, span / 3);
      this.nextGust += span;
    }
    for (const e of this.events) {
      while (e.next < horizon) {
        if (this.active) this._fire(e, e.next);
        e.next += rand(e.every[0], e.every[1]);
      }
    }
  }

  _applyChord(t) {
    const P = this.P, PD = P.pad;
    const deg = P.prog[this.chordIdx % P.prog.length];
    this.chordIdx++;
    const pcs = this._chordPcs(deg);
    this.chordPcs = pcs;
    const next = voiceLead(this.padNotes, pcs, this.root + PD.lo, this.root + PD.hi);
    next.forEach((m, i) => {
      const v = this.padVoices[i];
      const tt = t + i * rand(0.4, 1.4);
      if (m !== this.padNotes[i]) {
        const f = mtof(m);
        glide(v.o1.frequency, f, tt, PD.glide);
        glide(v.o2.frequency, f, tt, PD.glide * 1.15);
      }
      glide(v.g.gain, PD.gain * MIX.pad * rand(0.55, 1), tt, 3);
    });
    this.padNotes = next;
    if (P.bassFollows) {
      let b = this._deg2midi(deg);
      while (b > this.root + 6) b -= 12;
      while (b < this.root - 5) b += 12;
      const base = mtof(b - (P.drone.octDown ? 12 : 0));
      for (const d of this.droneOscs) glide(d.o.frequency, base * d.mult, t, 2.5);
    }
  }

  _eventNotes(e) {
    const base = this.root + 12 * e.oct;
    const pcs = e.scaleOnly ? this._scalePcs() : (Math.random() < 0.7 ? this.chordPcs : this._scalePcs());
    const out = [];
    for (let m = base; m <= base + 19; m++) if (pcs.includes(((m % 12) + 12) % 12) && mtof(m) < 4200) out.push(m);
    return out.length ? out : [base];
  }

  _fire(e, t) {
    const voice = EVENT_VOICES[e.type];
    if (!voice) return;
    const cands = this._eventNotes(e);
    const opts = { ...e, noise: this.amb.noise.brown, white: this.amb.noise.white };
    const motif = Math.random() < (e.motif || 0);
    const n = motif ? (e.motifLen || 3) + ((Math.random() * ((e.motifVar || 0) + 1)) | 0) : 1;
    let idx = (Math.random() * cands.length) | 0;
    const dir = Math.random() < 0.5 ? -1 : 1;
    const notes = [];
    for (let k = 0; k < n; k++) {
      notes.push(mtof(cands[Math.max(0, Math.min(cands.length - 1, idx))]));
      idx += dir * (Math.random() < 0.7 ? 1 : 2);
      if (idx < 0 || idx >= cands.length) idx = Math.max(0, Math.min(cands.length - 1, idx - 2 * dir));
    }
    const g0 = e.gain * MIX.events;
    if (voice.wholeMotif) {
      voice(this.ctx, this.evtBus, t, notes[0], g0 * rand(0.8, 1), { ...opts, notes });
      return;
    }
    let tt = t;
    notes.forEach((f, k) => {
      voice(this.ctx, this.evtBus, tt, f, g0 * (1 - 0.12 * k) * rand(0.75, 1), opts);
      tt += (e.motifGap || 0.5) * rand(0.85, 1.2);
    });
  }

  /** A pitch for the dream sparkles / crystal footsteps: high chord tone. */
  highNote(oct = 4) {
    const base = this.root + 12 * oct;
    const c = [];
    for (let m = base; m <= base + 19; m++) if (this.chordPcs.includes(((m % 12) + 12) % 12)) c.push(m);
    let f = mtof(c.length ? pick(c) : base);
    while (f > 4200) f /= 2;
    return f;
  }
}
