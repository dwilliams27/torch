// HYPNAGOGIA audio — generative, zone-reactive ambience in pure WebAudio.
//
//   const amb = new Ambience();
//   button.onclick = () => amb.start();   // must be inside a user gesture (iOS)
//   amb.setZone(level.zones[i]);          // crossfades over ~6-8 s; cheap to call every frame
//   amb.setDream(activity0to1);           // glitter + brightness when diffusion results land
//   amb.setVolume(0.6); amb.toggle();
//   amb.footstep(0.5);  /  import { footstep } from './ambience.js'
//
// Signal flow:
//   Scene(s) ──dry──────────────────────────────┐
//            ──rev send─► HP/LP ► Convolver ────┤
//            ──dly send─► ping-pong delay ──────┤ (+ a bit into reverb)
//   sparkles / footsteps ─► (dry, rev, dly) ────┤
//                                               ▼
//              DC-block HP ► high-shelf cut ► limiter ► volume ► destination
//
// Design notes: at most 2 (briefly 3) scenes are alive at once; each has ~16
// persistent oscillators that are retuned, never recreated. Short event voices
// are scheduled from a 200 ms timer with 1.5 s lookahead (so background-tab
// timer throttling doesn't cause gaps). No AudioWorklet. Offline-renderable
// (pass {context: new OfflineAudioContext(...)} and call scheduleUntil()).

import { makeNoiseBuffer, makeImpulseResponse, makeWaves, rand, expRand } from './dsp.js';
import { Scene } from './scene.js';
import { PRESETS, presetForZone } from './presets.js';
import { sparkle as sparkleVoice, footstep as footVoice, swell as swellVoice, glass as glassVoice } from './voices.js';
import { mtof } from './dsp.js';

const LOOKAHEAD = 1.5;
const TICK_MS = 200;
const TRANSPOSE_VARIANTS = [0, 5, -3, 2, -5, 3];

let lastInstance = null;

export class Ambience {
  /**
   * @param {object} [opts]
   * @param {BaseAudioContext} [opts.context]  use an existing (or Offline) context
   * @param {number} [opts.volume=0.55]
   * @param {boolean} [opts.lite]  smaller buffers / shorter reverb (auto on phones)
   */
  constructor(opts = {}) {
    this.ctx = opts.context || null;
    this.volume = opts.volume ?? 0.55;
    this.lite = opts.lite ?? (typeof navigator !== 'undefined' && /iPhone|iPad|Android/i.test(navigator.userAgent || ''));
    this.enabled = true;
    this.started = false;
    this.scenes = [];
    this.current = null;
    this._pendingZone = null;
    this._zoneIdent = null;
    this._zoneAssign = new Map();   // zone identity -> {name, transpose, key}
    this._presetUse = new Map();    // preset name -> count of zones using it
    this._dreamTarget = 0;
    this._dream = 0;
    this._lastTick = 0;
    this._nextSparkle = 0;
    this._timer = null;
    this._built = false;
    this._offline = false;
  }

  get state() {
    return {
      started: this.started,
      enabled: this.enabled,
      context: this.ctx ? this.ctx.state : 'none',
      preset: this.current ? this.current.name : null,
      label: this.current ? this.current.P.label : null,
      scenes: this.scenes.length,
      dream: +this._dream.toFixed(3),
    };
  }

  // ------------------------------------------------------------------ lifecycle

  async start() {
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) { console.warn('[audio] WebAudio unavailable'); return false; }
      try { if (navigator.audioSession) navigator.audioSession.type = 'playback'; } catch (_) { /* iOS 17+ only */ }
      try { this.ctx = new AC({ latencyHint: 'balanced' }); } catch (_) { this.ctx = new AC(); }
    }
    const ctx = this.ctx;
    this._offline = typeof OfflineAudioContext !== 'undefined' && ctx instanceof OfflineAudioContext;

    // Gesture-bound work first (synchronously, before any await or heavy work):
    // iOS only unlocks output for calls made directly inside the user gesture.
    let resumed = null;
    if (!this._offline) {
      try {
        const b = ctx.createBuffer(1, 1, ctx.sampleRate);
        const s = ctx.createBufferSource();
        s.buffer = b; s.connect(ctx.destination); s.start(0);
      } catch (_) { /* */ }
      if (ctx.state !== 'running') resumed = ctx.resume().catch(() => {});
    }
    if (!this._built) this._build(); // ~50-90 ms of buffer synthesis (noise + reverb IR)
    if (resumed) {
      // resume() can hang on iOS if not in a gesture — don't block the caller forever
      await Promise.race([resumed, new Promise((r) => setTimeout(r, 600))]);
    }

    if (!this.started) {
      this.started = true;
      lastInstance = this;
      const t = ctx.currentTime;
      this.master.gain.setValueAtTime(0, t);
      this.master.gain.setTargetAtTime(this.enabled ? this.volume : 0, t + 0.05, 1.2);
      this._lastTick = t;
      this._nextSparkle = t + 1;
      this.setZone(this._pendingZone || { name: 'void', mood: 'void' });
      if (!this._offline) {
        this._timer = setInterval(() => this._tick(), TICK_MS);
        this._tick();
        this._installLifecycle();
      }
    }
    return true;
  }

  _installLifecycle() {
    const ctx = this.ctx;
    this._onVis = () => {
      if (document.hidden) ctx.suspend().catch(() => {});
      else if (this.enabled) ctx.resume().catch(() => {});
    };
    document.addEventListener('visibilitychange', this._onVis);
    // iOS may "interrupt" the context (calls, Siri, lock screen): resume on next gesture.
    this._onGesture = () => {
      if (this.enabled && !document.hidden && ctx.state !== 'running') ctx.resume().catch(() => {});
    };
    for (const ev of ['pointerdown', 'touchend', 'keydown']) {
      window.addEventListener(ev, this._onGesture, { passive: true, capture: true });
    }
  }

  /** Stop everything and release the context. */
  async dispose() {
    if (this._timer) clearInterval(this._timer);
    this._timer = null;
    if (this._onVis) document.removeEventListener('visibilitychange', this._onVis);
    if (this._onGesture) for (const ev of ['pointerdown', 'touchend', 'keydown']) window.removeEventListener(ev, this._onGesture, { capture: true });
    if (this.ctx && !this._offline && this.ctx.close) await this.ctx.close().catch(() => {});
    this.started = false;
    if (lastInstance === this) lastInstance = null;
  }

  _build() {
    const ctx = this.ctx;
    this._built = true;
    const secs = this.lite ? 3 : 4;
    this.noise = {
      pink: makeNoiseBuffer(ctx, 'pink', secs, 11),
      brown: makeNoiseBuffer(ctx, 'brown', secs, 23),
      white: makeNoiseBuffer(ctx, 'white', secs, 37),
    };
    this.waves = makeWaves(ctx);

    const G = (v) => { const g = ctx.createGain(); g.gain.value = v; return g; };
    const F = (type, f, q = 0.707, gain = 0) => {
      const b = ctx.createBiquadFilter();
      b.type = type; b.frequency.value = f; b.Q.value = q; b.gain.value = gain;
      return b;
    };

    // --- output chain
    this.mix = G(1);
    const dcBlock = F('highpass', 28, 0.6);
    const shelf = F('highshelf', 8000, 0.707, -2.5);
    const lim = ctx.createDynamicsCompressor();
    lim.threshold.value = -6; lim.knee.value = 0; lim.ratio.value = 20;
    lim.attack.value = 0.002; lim.release.value = 0.2;
    this.limiter = lim;
    this.master = G(0);
    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 2048;
    this.mix.connect(dcBlock); dcBlock.connect(shelf); shelf.connect(lim); lim.connect(this.master);
    this.master.connect(ctx.destination);
    this.master.connect(this.analyser);

    // --- buses
    this.dryIn = G(1);
    this.dryIn.connect(this.mix);

    // reverb
    this.revIn = G(1);
    const revHP = F('highpass', 150, 0.5);
    const revLP = F('lowpass', 6500, 0.5);
    const conv = ctx.createConvolver();
    conv.normalize = false;
    conv.buffer = makeImpulseResponse(ctx, this.lite ? { seconds: 3, rt60: 3.2 } : { seconds: 4.6, rt60: 4.9 });
    this.revReturn = G(0.55);
    this.revIn.connect(revHP); revHP.connect(revLP); revLP.connect(conv); conv.connect(this.revReturn);
    this.revReturn.connect(this.mix);

    // ping-pong delay with darkening feedback
    this.dlyIn = G(1);
    const tone = F('lowpass', 2200, 0.5);
    const dL = ctx.createDelay(3), dR = ctx.createDelay(3);
    dL.delayTime.value = 0.6; dR.delayTime.value = 0.6;
    const fbL = G(0.4), fbR = G(0.4);
    const fbTone = F('lowpass', 2200, 0.5);
    const merge = ctx.createChannelMerger(2);
    this.dlyIn.connect(tone); tone.connect(dL);
    dL.connect(fbL); fbL.connect(dR);
    dR.connect(fbR); fbR.connect(fbTone); fbTone.connect(dL);
    dL.connect(merge, 0, 0); dR.connect(merge, 0, 1);
    this.dlyReturn = G(0.55);
    merge.connect(this.dlyReturn);
    this.dlyReturn.connect(this.mix);
    const dly2rev = G(0.25);
    this.dlyReturn.connect(dly2rev); dly2rev.connect(this.revIn);
    this.delay = { dL, dR, fbL, fbR, tone, fbTone };
    // slow "tape wow" on the echoes (opposite phase L/R): lush, never seasick
    const wow = ctx.createOscillator();
    wow.frequency.value = 0.11;
    const wowL = G(0.0025), wowR = G(-0.0025);
    wow.connect(wowL); wow.connect(wowR);
    wowL.connect(dL.delayTime); wowR.connect(dR.delayTime);
    wow.start();

    // global event bus (sparkles, footsteps)
    this.fxBus = G(1);
    this.fxBus.connect(this.dryIn);
    const fxRev = G(0.9); this.fxBus.connect(fxRev); fxRev.connect(this.revIn);
    const fxDly = G(0.35); this.fxBus.connect(fxDly); fxDly.connect(this.dlyIn);
    this.footBus = G(1);
    this.footBus.connect(this.dryIn);
    const footRev = G(0.3); this.footBus.connect(footRev); footRev.connect(this.revIn);
  }

  // ------------------------------------------------------------------ controls

  /** Accepts a Level zone object (or anything with mood/name/prompt). Cheap to call every frame. */
  setZone(zone) {
    if (!zone) return;
    if (!this.started) { this._pendingZone = zone; return; }
    const ident = String(zone.id ?? '') + '|' + (zone.name || '') + '|' + (zone.mood || '');
    if (ident === this._zoneIdent) return;
    this._zoneIdent = ident;

    let a = this._zoneAssign.get(ident);
    if (!a) {
      const p = presetForZone(zone);
      const n = this._presetUse.get(p.name) || 0;
      this._presetUse.set(p.name, n + 1);
      const transpose = p.transpose + TRANSPOSE_VARIANTS[n % TRANSPOSE_VARIANTS.length];
      a = { name: p.name, transpose, key: p.name + ':' + transpose };
      this._zoneAssign.set(ident, a);
    }
    this._switchTo(a);
  }

  /** Force a preset by name (debug / demo). */
  setPreset(name, transpose = 0) {
    if (!PRESETS[name]) return;
    this._zoneIdent = 'preset:' + name + ':' + transpose;
    if (!this.started) { this._pendingZone = { name, mood: name, audio: { preset: name, transpose } }; return; }
    this._switchTo({ name, transpose, key: name + ':' + transpose });
  }

  _switchTo(a) {
    const ctx = this.ctx;
    if (this.current && this.current.key === a.key && this.current.active) return;
    const t = ctx.currentTime;
    let next = this.scenes.find((s) => s.key === a.key && !s.dead);
    if (!next) {
      next = new Scene(this, a.name, a.transpose, a.key);
      next.setDream(this._dream, t);
      this.scenes.push(next);
      this._arrival(next, t);
    }
    for (const s of this.scenes) if (s !== next && s.active) s.fadeOut(t);
    next.fadeIn(t);
    this.current = next;

    // keep the graph light: never more than 3 live scenes
    const live = this.scenes.filter((s) => !s.dead);
    if (live.length > 3) {
      const victims = live.filter((s) => s !== next).sort((x, y) => x.deadAt - y.deadAt);
      // quick fade + hard stop right away (robust even if the clock is frozen)
      for (const v of victims.slice(0, live.length - 3)) { v.fadeOut(t, 0.08); v.destroy(t + 0.6); }
      this.scenes = this.scenes.filter((s) => !s.dead);
    }

    // delay character follows the zone, slowly (the drifting delay time is a gentle tape-warble)
    const D = next.P.delay;
    const dl = this.delay;
    dl.dL.delayTime.setTargetAtTime(D.time, t, 2.5);
    dl.dR.delayTime.setTargetAtTime(D.time * 1.0, t, 2.5);
    dl.fbL.gain.setTargetAtTime(D.feedback, t, 2);
    dl.fbR.gain.setTargetAtTime(D.feedback, t, 2);
    dl.tone.frequency.setTargetAtTime(D.tone, t, 2);
    dl.fbTone.frequency.setTargetAtTime(D.tone, t, 2);
    if (!this._offline) this._tick();
  }

  /** Entering a new zone: a breath of rising air that resolves into a soft
   * strummed chord of the new key (lines up with the zone title card).
   * Rate-limited so border-hopping doesn't spam it. */
  _arrival(scene, t) {
    if (t - (this._lastArrival ?? -1e9) < 20) return;
    this._lastArrival = t;
    const ctx = this.ctx, dur = 2.6;
    swellVoice(ctx, this.fxBus, t + 0.2, this.noise.pink, 0.05, dur);
    const pcs = scene.chordPcs;
    const base = scene.root + 24;
    const notes = [];
    for (let m = base; m < base + 24 && notes.length < 4; m++) if (pcs.includes(((m % 12) + 12) % 12)) notes.push(m);
    notes.forEach((m, i) => glassVoice(ctx, this.fxBus, t + 0.2 + dur + i * 0.11, mtof(m), 0.028 * (1 - i * 0.12), { decay: 6 }));
  }

  /** activity 0..1 (e.g. diffusion results arriving). Cheap; smoothed internally. */
  setDream(activity) {
    const a = Number.isFinite(activity) ? Math.max(0, Math.min(1, activity)) : 0;
    this._dreamTarget = a;
  }

  setVolume(v) {
    this.volume = Math.max(0, Math.min(1, v));
    if (this.started && this.enabled) this.master.gain.setTargetAtTime(this.volume, this.ctx.currentTime, 0.15);
  }

  /** Mute/unmute (suspends the context while muted to save CPU). Returns enabled state. */
  toggle() {
    this.enabled = !this.enabled;
    if (!this.started) return this.enabled;
    const ctx = this.ctx, t = ctx.currentTime;
    if (this.enabled) {
      if (!this._offline) ctx.resume().catch(() => {});
      this.master.gain.setTargetAtTime(this.volume, t, 0.4);
    } else {
      this.master.gain.setTargetAtTime(0, t, 0.25);
      if (!this._offline) setTimeout(() => { if (!this.enabled) ctx.suspend().catch(() => {}); }, 1800);
    }
    return this.enabled;
  }

  /** Soft zone-appropriate footfall. intensity 0..1 (walk ~0.4, run ~0.8, landing 1). */
  footstep(intensity = 0.5) {
    if (!this.started || !this.enabled || !this.current || !this.ctx) return;
    if (!this._offline && this.ctx.state !== 'running') return;
    const t = this.ctx.currentTime + 0.01;
    footVoice(this.ctx, this.footBus, this.noise.white, t, this.current.P.foot, intensity, this.current.highNote(3));
  }

  // ------------------------------------------------------------------ scheduling

  _tick(nowOverride) {
    const ctx = this.ctx;
    if (!ctx) return;
    const now = nowOverride ?? ctx.currentTime;
    const dt = Math.max(0, Math.min(1, now - this._lastTick));
    this._lastTick = now;

    // dream smoothing (~1.5 s time constant), applied only when it moves
    const k = 1 - Math.exp(-dt / 1.5);
    this._dream += (this._dreamTarget - this._dream) * k;
    if (Math.abs(this._dream - (this._dreamApplied ?? -1)) > 0.015) {
      this._dreamApplied = this._dream;
      for (const s of this.scenes) if (!s.dead) s.setDream(this._dream, now);
      this.revReturn.gain.setTargetAtTime(0.55 + 0.12 * this._dream, now, 1.5);
    }

    const horizon = now + LOOKAHEAD;
    for (const s of this.scenes) s.schedule(horizon);

    // dream glitter
    const d = this._dream;
    while (this._nextSparkle < horizon) {
      const t = this._nextSparkle;
      if (d > 0.03 && this.current) {
        sparkleVoice(ctx, this.fxBus, t, this.current.highNote(this.current.P.sparkleOct), 0.006 + 0.02 * d * rand(0.4, 1));
        // glitter comes in slow waves (~23 s) so steady diffusion activity never becomes a constant hiss of bells
        const wave = 0.3 + 0.7 * (0.5 + 0.5 * Math.sin(t * 2 * Math.PI / 23));
        this._nextSparkle += Math.max(0.09, expRand(1 / ((0.1 + 1.6 * d * d) * wave)));
      } else {
        this._nextSparkle = horizon + 0.25;
      }
    }

    // retire faded scenes
    for (const s of this.scenes) if (!s.active && !s.dead && now > s.deadAt) s.destroy(now + 0.05);
    this.scenes = this.scenes.filter((s) => !s.dead);
  }

  /** Offline / testing: run the scheduler up to context time T (seconds). */
  scheduleUntil(T, step = TICK_MS / 1000) {
    let now = this._lastTick;
    while (now < T) { now = Math.min(T, now + step); this._tick(now); }
  }
}

/** Optional helper: footfall on the most recently started Ambience. */
export function footstep(intensity = 0.5) {
  if (lastInstance) lastInstance.footstep(intensity);
}

export { PRESETS, presetForZone };
export default Ambience;
