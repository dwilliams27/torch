// HYPNAGOGIA audio — zone mood presets and the zone -> preset mapper.
//
// A preset is a small, declarative description of a sound world:
//   root/scale/prog  harmony (pad voice-leads between chords built on `prog` degrees)
//   drone            sustained bass pedal (2 detuned osc + optional sub + optional fifth)
//   pad              4 gliding voices (2 detuned osc each) through a shared filter
//   noise            looping filtered noise bed with slow random "gusts"
//   hum              optional low electrical/room hum
//   events           sparse struck/rung tones scheduled with lookahead
//   send             reverb/delay send per layer; delay = ping-pong echo settings
//   foot             footstep material
// Levels are roughly calibrated so every preset lands near the same loudness
// (see demo.html?test=1 which renders each preset offline and prints peak/RMS).

export const SCALES = {
  ionian:     [0, 2, 4, 5, 7, 9, 11],
  dorian:     [0, 2, 3, 5, 7, 9, 10],
  aeolian:    [0, 2, 3, 5, 7, 8, 10],
  lydian:     [0, 2, 4, 6, 7, 9, 11],
  mixolydian: [0, 2, 4, 5, 7, 9, 10],
  phrygdom:   [0, 1, 4, 5, 7, 8, 10], // "hijaz"
  phrygian:   [0, 1, 3, 5, 7, 8, 10],
  wholeish:   [0, 2, 4, 6, 7, 9, 10], // lydian-dominant: floaty, unresolved
};

export const PRESETS = {
  drowned: {
    label: 'Drowned', level: 1.1,
    root: 38, scale: 'dorian', prog: [0, 5, 3, 0, 6, 3], chordSec: [16, 24], chordSize: 4,
    drone: { wave: 'organ', gain: 0.08, cutoff: 380, q: 0.7, detune: 7, sub: 0.06, fifth: 0.035, lfoRate: 0.021, lfoDepth: 140 },
    pad: { wave: 'warm', gain: 0.028, cutoff: 900, q: 0.9, detune: 9, glide: 2.8, lo: 12, hi: 31, lfoRate: 0.017, lfoDepth: 420 },
    noise: { color: 'pink', type: 'lowpass', freq: 700, q: 0.4, gain: 0.10, gust: [3, 7], gustDepth: 0.55, freqRange: [380, 1150] },
    events: [
      { type: 'drip', every: [2.5, 7], gain: 0.05, oct: 3, motif: 0.35 },
      { type: 'whale', every: [13, 26], gain: 0.05, oct: 2, motif: 0 },
      { type: 'bell', every: [14, 26], gain: 0.05, oct: 2, ratio: 1.41, index: 1.4, decay: 7, motif: 0.2 },
    ],
    send: { drone: 0.25, pad: 0.55, noise: 0.25, events: 0.85, delay: 0.28 },
    delay: { time: 0.62, feedback: 0.42, tone: 1600 },
    foot: 'water', sparkleOct: 4,
  },

  cathedral: {
    label: 'Cathedral', level: 1.36,
    root: 41, scale: 'mixolydian', prog: [0, 3, 4, 0, 5, 3, 6, 4], chordSec: [14, 20], chordSize: 3,
    drone: { wave: 'organ', gain: 0.06, cutoff: 700, q: 0.5, detune: 3, sub: 0.07, fifth: 0.04, lfoRate: 0.013, lfoDepth: 200 },
    pad: { wave: 'choir', gain: 0.024, cutoff: 1400, q: 0.6, detune: 6, glide: 3.4, lo: 12, hi: 29, lfoRate: 0.011, lfoDepth: 300,
           formant: { f1: 560, f2: 1050, q: 5, gain: 0.8, sweep: 0.03, depth: 180 }, tremRate: 0.09, tremDepth: 0.2 },
    noise: { color: 'pink', type: 'lowpass', freq: 380, q: 0.3, gain: 0.07, gust: [6, 12], gustDepth: 0.3, freqRange: [260, 520] },
    events: [
      { type: 'bell', every: [11, 22], gain: 0.085, oct: 1, ratio: 2.0, index: 2.2, decay: 9, motif: 0.5, motifLen: 3, motifGap: 1.6 },
      { type: 'glass', every: [7, 15], gain: 0.03, oct: 4, motif: 0.1 },
    ],
    send: { drone: 0.45, pad: 0.8, noise: 0.2, events: 1.0, delay: 0.0 },
    delay: { time: 0.9, feedback: 0.3, tone: 1400 },
    foot: 'stone', sparkleOct: 4,
  },

  library: {
    label: 'Library', level: 0.97,
    root: 45, scale: 'aeolian', prog: [0, 5, 2, 6, 3, 5, 4], chordSec: [12, 18], chordSize: 4, bassFollows: true,
    drone: { wave: 'soft', gain: 0.08, cutoff: 420, q: 0.6, detune: 5, sub: 0.03, fifth: 0.0, lfoRate: 0.02, lfoDepth: 120 },
    pad: { wave: 'hollow', gain: 0.03, cutoff: 1100, q: 0.8, detune: 7, glide: 2.2, lo: 7, hi: 26, lfoRate: 0.025, lfoDepth: 350 },
    noise: { color: 'pink', type: 'bandpass', freq: 900, q: 0.5, gain: 0.05, gust: [4, 9], gustDepth: 0.45, freqRange: [500, 1400] },
    events: [
      { type: 'pluck', every: [5, 11], gain: 0.06, oct: 2, motif: 0.75, motifLen: 4, motifVar: 3, motifGap: 0.19, decay: 3.6 },
      { type: 'tick', every: [12, 26], gain: 0.05, oct: 0, motif: 0 },
      { type: 'rustle', every: [6, 15], gain: 0.03, oct: 0, motif: 0 },
    ],
    send: { drone: 0.25, pad: 0.5, noise: 0.3, events: 0.6, delay: 0.25 },
    delay: { time: 0.84, feedback: 0.36, tone: 2200 },
    foot: 'wood', sparkleOct: 4,
  },

  garden: {
    label: 'Garden', level: 1.22,
    root: 43, scale: 'lydian', prog: [0, 1, 4, 0, 2, 1], chordSec: [13, 20], chordSize: 4, bassFollows: false,
    drone: { wave: 'warm', gain: 0.05, cutoff: 520, q: 0.5, detune: 6, sub: 0.045, fifth: 0.03, lfoRate: 0.017, lfoDepth: 160 },
    pad: { wave: 'soft', gain: 0.034, cutoff: 1700, q: 0.5, detune: 8, glide: 2.6, lo: 12, hi: 31, lfoRate: 0.02, lfoDepth: 600 },
    noise: { color: 'pink', type: 'bandpass', freq: 1500, q: 0.6, gain: 0.045, gust: [2.5, 6], gustDepth: 0.7, freqRange: [900, 2600] },
    events: [
      { type: 'chime', every: [11, 22], gain: 0.035, oct: 3, motif: 0.8, motifLen: 5, motifGap: 0.23, scaleOnly: true },
      { type: 'rbird', every: [5, 12], gain: 0.022, oct: 4, motif: 0.7, motifLen: 2, motifVar: 3, motifGap: 0.55, scaleOnly: true },
      { type: 'trickle', every: [3.5, 8], gain: 0.03, oct: 3, motif: 0 },
      { type: 'pluck', every: [8, 16], gain: 0.05, oct: 2, motif: 0.3, motifLen: 2, motifGap: 0.6 },
    ],
    send: { drone: 0.2, pad: 0.45, noise: 0.2, events: 0.6, delay: 0.2 },
    delay: { time: 0.46, feedback: 0.33, tone: 3000 },
    foot: 'grass', sparkleOct: 4,
  },

  crystal: {
    label: 'Crystal', level: 0.95,
    root: 40, scale: 'lydian', prog: [0, 4, 1, 5], chordSec: [15, 24], chordSize: 4,
    drone: { wave: 'sine', gain: 0.10, cutoff: 900, q: 0.4, detune: 4, sub: 0.11, fifth: 0.04, lfoRate: 0.015, lfoDepth: 200 },
    pad: { wave: 'glass', gain: 0.022, cutoff: 2200, q: 1.2, detune: 5, glide: 3.5, lo: 19, hi: 36, lfoRate: 0.013, lfoDepth: 900, tremRate: 0.12, tremDepth: 0.25 },
    noise: { color: 'white', type: 'bandpass', freq: 3200, q: 1.4, gain: 0.018, gust: [5, 10], gustDepth: 0.6, freqRange: [2400, 4200] },
    events: [
      { type: 'glass', every: [3, 7], gain: 0.05, oct: 3, motif: 0.4, motifLen: 3, motifGap: 0.9 },
      { type: 'bowl', every: [16, 28], gain: 0.05, oct: 1, motif: 0 },
    ],
    send: { drone: 0.4, pad: 0.8, noise: 0.5, events: 1.0, delay: 0.3 },
    delay: { time: 0.73, feedback: 0.45, tone: 3200 },
    foot: 'crystal', sparkleOct: 4,
  },

  neon: {
    label: 'Neon', level: 1.74,
    root: 37, scale: 'dorian', prog: [0, 3, 6, 2, 5, 1], chordSec: [11, 16], chordSize: 4,
    drone: { wave: 'reed', gain: 0.035, cutoff: 260, q: 2.5, detune: 9, sub: 0.07, fifth: 0.0, lfoRate: 0.03, lfoDepth: 110 },
    pad: { wave: 'reed', gain: 0.02, cutoff: 1000, q: 2.2, detune: 11, glide: 1.6, lo: 12, hi: 30, lfoRate: 0.045, lfoDepth: 550, tremRate: 0.25, tremDepth: 0.3 },
    noise: { color: 'pink', type: 'bandpass', freq: 800, q: 0.7, gain: 0.07, gust: [1.4, 3.2], gustDepth: 0.8, freqRange: [450, 1300] },
    hum: { mult: 1, gain: 0.02, cutoff: 240, q: 5, trem: 0.13 },
    events: [
      { type: 'epiano', every: [4, 10], gain: 0.055, oct: 2, motif: 0.55, motifLen: 3, motifGap: 0.375 },
      { type: 'drip', every: [5, 12], gain: 0.035, oct: 3, motif: 0.3 },
    ],
    send: { drone: 0.2, pad: 0.45, noise: 0.4, events: 0.7, delay: 0.55 },
    delay: { time: 0.375, feedback: 0.5, tone: 2000 },
    foot: 'tile', sparkleOct: 4,
  },

  desert: {
    label: 'Desert', level: 1.2,
    root: 45, scale: 'phrygdom', prog: [0, 1, 0, 6, 0, 3], chordSec: [16, 26], chordSize: 3,
    drone: { wave: 'warm', gain: 0.06, cutoff: 480, q: 0.9, detune: 3, sub: 0.05, fifth: 0.05, lfoRate: 0.012, lfoDepth: 180, octDown: true },
    pad: { wave: 'reed', gain: 0.02, cutoff: 1200, q: 0.8, detune: 13, glide: 4.0, lo: 7, hi: 26, lfoRate: 0.016, lfoDepth: 450 },
    noise: { color: 'pink', type: 'bandpass', freq: 700, q: 0.9, gain: 0.11, gust: [2, 6], gustDepth: 0.8, freqRange: [350, 1600] },
    events: [
      { type: 'theremin', every: [15, 28], gain: 0.045, oct: 2, motif: 1, motifLen: 3, motifVar: 2, motifGap: 1.5, scaleOnly: true },
      { type: 'bowl', every: [14, 26], gain: 0.06, oct: 1, motif: 0 },
      { type: 'glass', every: [11, 20], gain: 0.025, oct: 3, motif: 0.3, motifLen: 2, motifGap: 1.1 },
    ],
    send: { drone: 0.3, pad: 0.6, noise: 0.25, events: 0.9, delay: 0.25 },
    delay: { time: 1.1, feedback: 0.35, tone: 1800 },
    foot: 'sand', sparkleOct: 4,
  },

  brutalist: {
    label: 'Brutalist', level: 0.97,
    root: 34, scale: 'aeolian', prog: [0, 5, 0, 1, 0, 4], chordSec: [18, 28], chordSize: 3,
    drone: { wave: 'warm', gain: 0.07, cutoff: 300, q: 0.8, detune: 4, sub: 0.07, fifth: 0.04, lfoRate: 0.009, lfoDepth: 110 },
    pad: { wave: 'hollow', gain: 0.022, cutoff: 1100, q: 0.7, detune: 5, glide: 5.0, lo: 12, hi: 27, lfoRate: 0.01, lfoDepth: 260 },
    noise: { color: 'pink', type: 'bandpass', freq: 600, q: 0.5, gain: 0.07, gust: [6, 13], gustDepth: 0.5, freqRange: [330, 1000] },
    hum: { mult: 2, gain: 0.012, cutoff: 180, q: 2, trem: 0.05 },
    events: [
      { type: 'boom', every: [14, 30], gain: 0.10, oct: 0, motif: 0 },
      { type: 'piano', every: [8, 17], gain: 0.07, oct: 2, motif: 0.45, motifLen: 2, motifVar: 1, motifGap: 1.5 },
      { type: 'bell', every: [18, 34], gain: 0.04, oct: 2, ratio: 1.414, index: 1.1, decay: 8, motif: 0 },
    ],
    send: { drone: 0.4, pad: 0.7, noise: 0.35, events: 1.0, delay: 0.1 },
    delay: { time: 1.4, feedback: 0.3, tone: 1200 },
    foot: 'concrete', sparkleOct: 4,
  },

  threshold: {
    label: 'Threshold', level: 1.46,
    root: 39, scale: 'aeolian', prog: [0, 5, 3, 4, 0, 2], chordSec: [16, 26], chordSize: 3,
    drone: { wave: 'soft', gain: 0.09, cutoff: 420, q: 0.5, detune: 3, sub: 0.05, fifth: 0.03, lfoRate: 0.011, lfoDepth: 120 },
    pad: { wave: 'choir', gain: 0.02, cutoff: 1000, q: 0.5, detune: 7, glide: 4.5, lo: 12, hi: 29, lfoRate: 0.012, lfoDepth: 250,
           formant: { f1: 480, f2: 900, q: 4, gain: 0.7, sweep: 0.021, depth: 140 }, tremRate: 0.07, tremDepth: 0.3 },
    noise: { color: 'pink', type: 'bandpass', freq: 1800, q: 0.7, gain: 0.025, gust: [5, 11], gustDepth: 0.6, freqRange: [1100, 2600] },
    events: [
      { type: 'heart', every: [22, 40], gain: 0.09, oct: 0, motif: 1, motifLen: 6, motifVar: 4, motifGap: 1.15 },
      { type: 'glass', every: [9, 19], gain: 0.03, oct: 3, motif: 0.3, motifLen: 2, motifGap: 1.8 },
    ],
    send: { drone: 0.35, pad: 0.95, noise: 0.5, events: 0.7, delay: 0.25 },
    delay: { time: 1.05, feedback: 0.4, tone: 1600 },
    foot: 'stone', sparkleOct: 4,
  },

  void: {
    label: 'Void', level: 0.87,
    root: 36, scale: 'wholeish', prog: [0, 1, 5, 3], chordSec: [20, 32], chordSize: 3,
    drone: { wave: 'sine', gain: 0.12, cutoff: 600, q: 0.3, detune: 2.5, sub: 0.04, fifth: 0.05, lfoRate: 0.008, lfoDepth: 100 },
    pad: { wave: 'soft', gain: 0.026, cutoff: 1500, q: 0.5, detune: 4, glide: 6.0, lo: 14, hi: 34, lfoRate: 0.009, lfoDepth: 500, tremRate: 0.06, tremDepth: 0.35 },
    noise: { color: 'brown', type: 'lowpass', freq: 200, q: 0.3, gain: 0.06, gust: [8, 16], gustDepth: 0.5, freqRange: [120, 300] },
    events: [
      { type: 'glass', every: [8, 18], gain: 0.035, oct: 4, motif: 0.25, motifLen: 2, motifGap: 2.2 },
      { type: 'bowl', every: [18, 34], gain: 0.05, oct: 1, motif: 0 },
    ],
    send: { drone: 0.5, pad: 0.9, noise: 0.4, events: 1.0, delay: 0.35 },
    delay: { time: 1.25, feedback: 0.5, tone: 1500 },
    foot: 'void', sparkleOct: 4,
  },
};

// Known world zone ids (client/src/world/zones.js) -> preset. Keyword scoring is the fallback.
const ZONE_IDS = {
  vestibule: 'threshold', nave: 'drowned', stacks: 'library', geode: 'crystal',
  baths: 'neon', garden: 'garden', atrium: 'brutalist', desert: 'desert',
};

const KEYWORDS = {
  threshold: ['threshold', 'vestibule', 'heartbeat', 'corridor', 'picture frame', 'antechamber', 'hallway', 'foyer', 'waking'],
  drowned:   ['drown', 'water', 'sunken', 'flood', 'underwater', 'ocean', 'sea=', 'seas=', 'tide', 'submerg', 'aquatic', 'rain', 'lake', 'grotto', 'reef', 'wet=', 'abyssal'],
  cathedral: ['cathedral', 'nave', 'chapel', 'choir', 'church', 'sacred', 'temple', 'basilica', 'altar', 'organ', 'holy', 'gothic', 'reliquary', 'sanctum', 'hymn'],
  library:   ['library', 'book', 'stair', 'archive', 'scroll', 'study', 'shelf', 'shelves', 'reading', 'manuscript', 'vertical', 'scholar', 'index', 'catalog'],
  garden:    ['garden', 'grove', 'moss', 'overgrown', 'forest', 'bloom', 'flower', 'leaf', 'leaves', 'orchard', 'jungle', 'greenhouse', 'botanic', 'false sky', 'conservatory', 'vine', 'green'],
  crystal:   ['crystal', 'cavern', 'ice=', 'icy=', 'glass', 'prism', 'geode', 'quartz', 'frozen', 'cave', 'gem', 'mineral', 'shard', 'amethyst', 'glacier'],
  neon:      ['neon', 'bath', 'cyber', 'pool', 'electric', 'vapor', 'synth', 'arcade', 'tile', 'steam', 'spa', 'hologram', 'fluorescent', 'motel', 'laundromat'],
  desert:    ['desert', 'dune', 'sand', 'moon', 'arch', 'sun=', 'suns=', 'mirage', 'oasis', 'dust', 'canyon', 'wasteland', 'salt', 'arid', 'heat='],
  brutalist: ['brutal', 'monolith', 'concrete', 'atrium', 'slab', 'megastructure', 'bunker', 'tower', 'cement', 'plaza', 'modernist', 'massive'],
  void:      ['void', 'abyss', 'star=', 'stars=', 'nothing', 'empty', 'cosmos', 'limbo', 'infinite', 'null', 'space', 'astral', 'ether', 'blueprint', 'undreamt'],
};

const PRESET_NAMES = Object.keys(PRESETS);

// Keywords match at word starts ("drown" hits "drowned"); a trailing "=" means
// whole word only (so "sun" doesn't fire on "sunken", "ice" not on "office").
const KEY_RE = {};
for (const [p, words] of Object.entries(KEYWORDS)) {
  KEY_RE[p] = words.map((w) => (w.endsWith('=') ? new RegExp('\\b' + w.slice(0, -1) + '\\b') : new RegExp('\\b' + w)));
}

function count(text, res) {
  let n = 0;
  for (const re of res) if (re.test(text)) n++;
  return n;
}

/**
 * Decide which preset a Level zone should use. Honors an explicit
 * `zone.audio` ('crystal' or {preset:'crystal', transpose:2}) if present,
 * otherwise scores keywords in mood (x4), name (x2), subtitle and prompt (x1).
 * Returns {name, transpose, key} where key identifies the resulting sound.
 */
export function presetForZone(zone) {
  if (!zone) return { name: 'void', transpose: 0 };
  const explicit = zone.audio;
  if (typeof explicit === 'string' && PRESETS[explicit]) return { name: explicit, transpose: 0 };
  if (explicit && typeof explicit === 'object' && PRESETS[explicit.preset]) {
    return { name: explicit.preset, transpose: explicit.transpose | 0 };
  }
  if (typeof zone.id === 'string' && ZONE_IDS[zone.id]) return { name: ZONE_IDS[zone.id], transpose: 0 };
  const low = (v) => (typeof v === 'string' ? v.toLowerCase() : '');
  const mood = low(zone.mood), name = low(zone.name), sub = low(zone.subtitle), prompt = low(zone.prompt);
  let best = null, bestScore = 0;
  for (const p of PRESET_NAMES) {
    const k = KEY_RE[p];
    // exact mood match wins outright
    if (mood === p) return { name: p, transpose: 0 };
    const s = count(mood, k) * 4 + count(name, k) * 2 + count(sub, k) + count(prompt, k);
    if (s > bestScore) { bestScore = s; best = p; }
  }
  if (!best) {
    // unknown: deterministic choice from the zone's identity
    let h = 0;
    const id = String(zone.id ?? '') + name + mood;
    for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0;
    best = PRESET_NAMES[h % PRESET_NAMES.length];
  }
  return { name: best, transpose: 0 };
}
