// Looks: named sets of dream settings to switch between while playing (keys 1-6, or the
// buttons at the top of the settings panel). Each changes only the keys it names; the rest
// stay at their defaults. A setting given in the URL always wins over a look.
export const LOOKS = [
  { id: 'waking', hint: 'the default dream', set: {} },
  { id: 'drifting', hint: 'slow, soft changes', set: { liveTau: 0.8, liveFade: 0.7, captureRate: 8, sharpen: 0.35 } },
  { id: 'lucid', hint: 'the architecture shows through', set: { strength: 0.55, depthSep: 1.0, relight: 0.6, capSep: 1.2, edgeKeep: 0.6 } },
  { id: 'fever', hint: 'vivid and restless', set: { strength: 0.74, liveTau: 0.12, liveFade: 0.15, sharpen: 0.85 } },
  { id: 'lens', hint: 'the centre lens of 29 September, to compare', set: { strength: 0.6, fovea: 1.7, liveTau: 0.2 } },
  // walking, each new view gets a full pass instead of reusing the last view's deep features
  // (the engine's DeepCache): 5% more walking detail, 17% nearer what the view becomes when you
  // stop, walking flicker unchanged; 12% fewer dreams a second at the tour's pace, a third fewer at a
  // full walk, where every view is new (M116)
  { id: 'fresh', hint: 'each new view gets a full pass: more detail while walking', set: { plainWalk: 1 } },
];

// the URL parameter behind each setting a look can change
export const LOOK_PARAMS = {
  strength: 'strength', liveTau: 'livetau', liveFade: 'livefade', captureRate: 'rate', sharpen: 'sharpen',
  depthSep: 'sep', relight: 'relight', capSep: 'capsep', edgeKeep: 'edgekeep', fovea: 'fovea', warp: 'warp',
  plainWalk: 'plainwalk',
};
