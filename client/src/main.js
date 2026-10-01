// HYPNAGOGIA - bootstrap + main loop (render owner).
import * as THREE from 'three';
import { createSharedUniforms, createLevelMaterials, LightDriver, ZoneAtmos, buildChunks } from './render/materials.js';
import { Post } from './render/post.js';
import { Hud } from './render/hud.js';
import { Painter } from './dream/painter.js';
import { Dream } from './dream/dream.js';
import { Link, MockLink } from './dream/link.js';
import { FlyCam } from './dream/flycam.js';
import { DreamMemory } from './dream/memory.js';
import { LOOKS, LOOK_PARAMS } from './looks.js';

const params = new URLSearchParams(location.search);
const P = (k, d) => (params.has(k) ? params.get(k) : d);
// numeric parameter; empty or non-numeric falls back to the default instead of 0 / NaN
const num = (k, d) => { const v = parseFloat(P(k, '')); return Number.isFinite(v) ? v : d; };
const isTouch = matchMedia('(pointer: coarse)').matches || 'ontouchstart' in window;
const isMobile = isTouch && Math.min(screen.width, screen.height) < 900;
const PERF = P('perf', '0') === '1';
const AUTOPILOT = P('autopilot', '0') !== '0';
const $ = (id) => document.getElementById(id);
if (isTouch) {
  document.documentElement.classList.add('touch', 'stats-off');
  const en = document.querySelector('#title .enter');
  if (en) en.textContent = 'tap to enter';
}

const settings = {
  strength: num('strength', 0.66),       // (0.6 before the engine painted with depth; depth holds the geometry at 0.66)
  feedback: num('feedback', 0.55),
  feedbackMoving: num('fbmove', 1.3),      // walking multiplies feedback by this (0.55 -> ~0.72): each new view starts from the refined dream
  feedbackAnchor: num('fbanchor', 0.35),   // feedback luminance pulled toward the raw render
  feedbackSat: num('fbsat', 0.8),          // feedback saturation kept (the rest leaks away per trip)
  paintRate: num('paint', 0.65),
  captureRate: num('rate', 20),
  maxInFlight: num('inflight', 2),
  paintSpread: num('spread', 3),
  captureFov: num('capfov', 0),   // capture vertical field of view; 0 = fitted to the screen (dream.js)
  jpegQuality: num('q', 0.9),
  relaxHalfLife: num('memory', 300),
  seed: num('dseed', 20260928),
  lead: num('lead', 1),
  saccade: num('saccade', 0),
  // A capture framing is held until the view moves this far (m / rad). Standing still it
  // holds, so one framing converges over many passes; walking, a new one is taken nearly
  // every capture, because a result projected from where you were half a second ago
  // stretches along your motion (radial smears on floors, streaks on walls).
  kfDist: num('kfdist', 0.2),
  plainWalk: num('plainwalk', 0),         // also send framing ids as `kf`, so an engine with held_only paints each new framing afresh (the fresh look)
  kfAngle: num('kfangle', 0.06),
  glance: num('glance', 0.62),
  live: num('live', isMobile ? 3 : 5),   // newest results projected at display time (max 5, 0 = atlas only)
  liveTau: num('livetau', 0.35),          // a held framing's picture follows its newest result with this time constant (s; 0 = each result replaces)
  liveFade: num('livefade', 0.25),        // seconds a new result takes to fade in
  sharpen: Math.min(1, Math.max(0, num('sharpen', 0.6))), // contrast-adaptive sharpening of live pixels, 0..1 (0 = off)
  fovea: num('fovea', 1),                // >1: narrow captures with this many times the pixels per degree (a visible lens; 1 = off)
  warp: num('warp', 1),                   // >1: foveal warp, the middle of each capture gets this many times the pixels per degree (measured calmer but less detailed, even in the middle; 2.25x the capture's pixels at 1.5)
  motionCalm: num('calm', 0),             // moving, strength drops by up to this fraction (off: it muted the dream)
  // Geometry cues. The engine paints with each capture's depth when it can (/api/info depth);
  // these add to that. Strong display cues show the real geometry through paint that ignores
  // it (a ghost), so they are off by default; the lucid look turns them on.
  depthSep: num('sep', 0),                // display: darken what sits just behind a nearer silhouette (0 = off)
  relight: num('relight', 0),             // display: surfaces turning away from the eye darken (0 = off)
  capSep: num('capsep', 0),               // the same cue in what the model sees, on the raw render (0 = off)
  edgeKeep: num('edgekeep', 0),           // at silhouettes the model sees this much less of its own last dream
  promptOverride: P('prompt', ''),
  eyesStrength: num('eyestrength', 0.9),  // behind closed eyes a depth engine re-dreams the room this hard (depth holds its shape)
  eyesFeedback: num('eyefeedback', 0.1),  // ...and shows it little of its old dream
  paused: false,
};

function showError(msg) {
  const el = $('err');
  el.style.display = 'block';
  el.textContent += msg + '\n';
}
window.addEventListener('error', (e) => showError(String(e.message || e)));
window.addEventListener('unhandledrejection', (e) => showError('unhandled: ' + (e.reason?.stack || e.reason)));

async function loadLevel(seed, atlasSize) {
  if (P('level', '') !== 'test') {
    try {
      const m = await import('./world/level.js');
      const level = m.generateLevel(seed, { atlasSize });
      if (level && (level.geometry || level.geometries)) return { level, real: true };
    } catch (e) {
      console.warn('[hypnagogia] world level unavailable, using test level:', e);
    }
  }
  const m = await import('./dream/testlevel.js');
  return { level: m.generateTestLevel(seed, { atlasSize }), real: false };
}

async function loadPlayer(level, camera, dom, real) {
  if (real && level.collision && P('fly', '0') !== '1') {
    try {
      const m = await import('./world/player.js');
      return new m.Player(level, camera, dom);
    } catch (e) {
      console.warn('[hypnagogia] world player unavailable, using fly camera:', e);
    }
  }
  return new FlyCam(level, camera, dom);
}

async function loadAmbience() {
  try {
    const m = await import('./audio/ambience.js');
    return new m.Ambience();
  } catch (e) {
    console.warn('[hypnagogia] no ambience module:', e?.message || e);
    return null;
  }
}

// If the page was served by the dream server itself (any port), talk to it;
// otherwise (static dev server) assume the dream server is on :8765 of the same host.
async function defaultServerUrl() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  try {
    const ctl = new AbortController();
    const to = setTimeout(() => ctl.abort(), 1500);
    const r = await fetch('api/info', { signal: ctl.signal, cache: 'no-store' });
    clearTimeout(to);
    if (r.ok && (r.headers.get('content-type') || '').includes('json')) return `${proto}//${location.host}/ws`;
  } catch {}
  return `${proto}//${location.hostname || 'localhost'}:8765/ws`;
}

async function boot() {
  const ambienceReady = loadAmbience();   // early: the entering tap must find it built
  const canvas = $('view');
  const renderer = new THREE.WebGLRenderer({
    canvas, antialias: false, alpha: false, stencil: false, depth: true, powerPreference: 'high-performance',
  });
  renderer.autoClear = false;
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  const halfFloat = renderer.extensions.has('EXT_color_buffer_float') || renderer.extensions.has('EXT_color_buffer_half_float');

  const seed = num('seed', 7);
  const atlasSize = num('atlas', isMobile ? 2048 : 4096);
  const { level, real } = await loadLevel(seed, atlasSize);
  console.log(`[hypnagogia] level: ${real ? 'world' : 'test'} seed=${seed} atlas=${level.atlas?.size} tpm=${level.atlas?.texelsPerMeter?.toFixed?.(1)}`);

  const camera = new THREE.PerspectiveCamera(num('fov', 72), innerWidth / innerHeight, 0.05, 1000);
  camera.layers.enable(1);
  const scene = new THREE.Scene();

  const shared = createSharedUniforms(level);
  const materials = createLevelMaterials(shared);
  const geos = level.geometries || [level.geometry];
  // Main view + capture: the whole level in one draw call per geometry (tens of
  // thousands of triangles is nothing for the GPU; per-object culling costs more CPU).
  for (const g of geos) {
    const m = new THREE.Mesh(g, materials.display);
    m.matrixAutoUpdate = false;
    m.frustumCulled = false;
    scene.add(m);
  }
  // Paint pass: spatial chunks so only surfaces inside the capture frustum are
  // rasterized into the atlas.
  const chunkGeos = [];
  for (const g of geos) chunkGeos.push(...buildChunks(g, num('chunk', 36)));
  console.log(`[hypnagogia] ${chunkGeos.length} paint chunks, ${geos.reduce((a, g) => a + (g.index ? g.index.count : g.attributes.position.count) / 3, 0)} triangles`);
  if (level.decor) {
    level.decor.traverse((o) => o.layers.set(1));
    scene.add(level.decor);
  }
  const painter = new Painter(renderer, level, chunkGeos, materials, {
    atlasSize: level.atlas?.size || atlasSize, halfFloat: halfFloat && P('atlas8', '0') !== '1',
  });
  shared.uAtlas.value = painter.texture;
  shared.uLiveSharpen.value = settings.sharpen;
  // the dream survives a reload (IndexedDB; ?persist=0 starts from a blank dream every time)
  let memory = null;
  if (P('persist', '1') !== '0') {
    memory = new DreamMemory(renderer, painter, level);
    if (await memory.open()) {
      const n = await memory.restore();
      if (n) console.log(`[hypnagogia] the dream remembers: ${n} atlas tiles restored`);
      document.addEventListener('visibilitychange', () => { if (document.hidden) memory.flush(); });
    } else memory = null;
  }
  const lightsDrv = new LightDriver(level, shared);
  const atmos = new ZoneAtmos(level, shared);
  const post = new Post(renderer, { halfFloat });

  // ---- dream link
  const dreamMode = P('dream', 'server');
  let link = null;
  if (dreamMode === 'mock') link = new MockLink(renderer, { latency: num('mocklat', 150) });
  else if (dreamMode !== 'off') link = new Link(params.has('server') ? P('server') : await defaultServerUrl());

  const statusEl = $('status');
  let entered = false;
  let hud = null;
  let engineInfo = null, linkStatus = link ? link.status : 'off', serverStats = null;
  function renderStatus() {
    if (!link) { statusEl.innerHTML = 'dreaming disabled &middot; the world stays a blueprint'; return; }
    if (engineInfo && (linkStatus === 'connected')) {
      const fps = engineInfo.fps_estimate ? ` &middot; ~${(+engineInfo.fps_estimate).toFixed(1)} dreams/s` : '';
      statusEl.innerHTML = `dream engine <b>${engineInfo.engine}</b> &middot; ${engineInfo.width}&times;${engineInfo.height} &middot; ${engineInfo.device || ''}${fps}`;
    } else if (linkStatus === 'connected') statusEl.innerHTML = 'dream engine connected&hellip;';
    else statusEl.innerHTML = `no dream engine at <b>${link.url || ''}</b> &middot; retrying &middot; the world stays a blueprint until it wakes`;
  }
  renderStatus();

  const dream = new Dream({
    renderer, scene, level, materials, shared, painter, link, settings,
    onEvent: (ev, a) => {
      if (ev === 'info') { engineInfo = a; renderStatus(); }
      else if (ev === 'status') { linkStatus = a; renderStatus(); if (a === 'connected' && entered) hud?.toast('the dream engine wakes'); }
      else if (ev === 'stats') serverStats = a;
    },
  });
  dream.memory = memory;

  // ---- player / controls
  const player = await loadPlayer(level, camera, canvas, real);
  const spawnPos = new THREE.Vector3(...(level.spawn?.position || [0, 1.6, 0]));
  const spawnYaw = level.spawn?.yaw || 0;
  const startPose = { pos: spawnPos.clone(), yaw: spawnYaw, pitch: 0 };
  if (params.has('pos')) {
    const [x, y, z] = P('pos', '0,1.6,0').split(',').map(Number);
    startPose.pos.set(x, y, z);
  }
  if (params.has('yaw')) startPose.yaw = num('yaw', 0);
  if (params.has('pitch')) startPose.pitch = num('pitch', 0);
  // The world Player owns spawn/URL pose and its own scripted tour; the fly cam
  // (test level) gets the pose from us.
  const hasTour = typeof player.startAutopilot === 'function' && (level.tour?.length || 0) > 1;
  if (player instanceof FlyCam) player.setPose(startPose.pos.clone(), startPose.yaw, startPose.pitch);
  const initialPose = { pos: (player.position || camera.position).clone(), yaw: player.yaw ?? startPose.yaw, pitch: player.pitch ?? 0 };

  // Sound: the module is built at boot so the entering tap can start it synchronously
  // (iOS only unlocks audio for calls made inside the gesture itself).
  let ambience = null, audioStarted = false;
  ambienceReady.then((a) => { ambience = a; });
  function startAudio() {
    if (audioStarted || !ambience) return;
    audioStarted = true;
    ambience.start().then(() => ambience.setZone(level.zones[zoneIndex]))
      .catch((e) => { console.warn('[hypnagogia] ambience start failed', e); ambience = null; });
  }
  // entered without a tap (?title=0) or before the module arrived: the next gesture starts it
  for (const ev of ['pointerdown', 'touchend', 'keydown']) {
    window.addEventListener(ev, () => { if (entered) startAudio(); }, { capture: true, passive: true });
  }
  player.onStep = (intensity) => { if (audioStarted) ambience?.footstep(intensity); };
  let compare = false;
  let dreamMix = 1;

  // ---- close your eyes (hold E; "blink" in the panel): the view sinks into a warm eyelid
  // dark while the dream keeps capturing the room with a sister prompt, harder and with little
  // feedback, so it re-dreams what is there. Open: the same architecture wears a new dream,
  // and nobody saw it change. The new dream stays until the next zone or the next blink.
  const EYE_DREAMS = [
    'overgrown with flowering vines and moss', 'in deep winter, snow and frost',
    'carved from translucent glass and crystal', 'at golden dawn, warm light and long shadows',
    'half flooded with glowing turquoise water', 'at night, lit by moonlight and candles',
    'a faded fresco of cracked plaster and gold leaf', 'in a storm of drifting autumn leaves',
  ];
  // The variant is chosen once the lid is really shut (lid > 0.85), so a quick tap of E dips
  // the view and changes nothing; each zone keeps its own variant until the next blink there
  // or "forget".
  const eyes = { closed: false, lid: 0, n: 0, until: 0, held: false, dreams: {} };
  function closeEyes(on, forS = 0) {
    on = !!on;
    eyes.until = on && forS > 0 ? performance.now() + forS * 1000 : 0;
    eyes.closed = on;
  }

  // ---- looks (looks.js): defaults as parsed from the URL, then the chosen look on top
  const lookBase = { ...settings };
  function applyLook(id, announce) {
    const look = LOOKS.find((l) => l.id === id) || LOOKS[0];
    for (const l of LOOKS) for (const k of Object.keys(l.set)) settings[k] = lookBase[k];
    for (const [k, v] of Object.entries(look.set)) if (!params.has(LOOK_PARAMS[k])) settings[k] = v;
    settings.look = look.id;
    dream.live.tau = settings.liveTau;
    dream.live.fadeIn = Math.max(1e-3, settings.liveFade);
    shared.uLiveSharpen.value = settings.sharpen;
    try { localStorage.setItem('hypnagogia.look', look.id); } catch {}
    hud?.syncPanel();
    // (the fresh look needs an engine that tells new views from held ones: /api/info held_only)
    const inert = look.set.plainWalk && dream.info && !dream.info.held_only;
    if (announce) hud?.toast(inert ? `${look.id} \u00b7 this dream engine can't do it: no change` : `${look.id} \u00b7 ${look.hint}`);
  }
  let storedLook = null;
  try { storedLook = localStorage.getItem('hypnagogia.look'); } catch {}
  applyLook(P('look', storedLook || 'waking'), false);

  hud = new Hud(settings, {
    blink: () => { if (!entered) return; hud.togglePanel(false); closeEyes(true, 4); },
    looks: LOOKS,
    look: (id) => applyLook(id, true),
    forget: () => { forgetDream(); },
    compare: () => toggleCompare(),
    audio: () => toggleAudio(),
    onPanel: (open) => { settings.panelOpen = open; },
    onSetting: () => {},
  });
  if (P('hud', '1') === '0') hud.toggleVisible();

  function toggleCompare() {
    compare = !compare;
    hud.toast(compare ? 'the blueprint' : 'the dream');
  }
  function toggleAudio() {
    if (!ambience) { hud.toast('no sound module'); return; }
    // the capture-phase gesture listener may have just called startAudio() for this very
    // key press; until start() resolves, M means "start", never "toggle off"
    if (!ambience.started) { startAudio(); hud.toast('sound'); return; }
    try { ambience.toggle(); hud.toast('sound'); } catch (e) { console.warn(e); }
  }

  function lockPointer() {
    try { const r = canvas.requestPointerLock?.(); r?.catch?.(() => {}); } catch {}
  }
  function enter(gesture = true) {
    if (entered) return;
    entered = true;
    if (gesture) startAudio();
    document.body.classList.remove('in-title');
    document.body.classList.add('entered');
    $('title').classList.add('gone');
    if (!isTouch) lockPointer();
    // hand the attract-mode pose to the player so there's no jump
    if (hasTour) {
      // attract-mode tour ends where the visitor begins (unless ?autopilot keeps it running)
      if (!AUTOPILOT && player.autopilot) {
        player.stopAutopilot();
        player.setPose(initialPose.pos.clone(), initialPose.yaw, initialPose.pitch);
      }
    } else if (!AUTOPILOT) player.setPose?.(camera.position.clone(), currentYaw(), 0);
    setTimeout(() => hud.zoneCard(level.zones[zoneIndex]), 900);
  }
  $('title').addEventListener('click', () => enter());
  $('title').addEventListener('touchend', (e) => { e.preventDefault(); enter(); });
  canvas.addEventListener('click', () => {
    if (entered && !hud.panelOpen && !isTouch && document.pointerLockElement !== canvas) lockPointer();
  });

  function forgetDream() {
    dream.forget();
    eyes.dreams = {};
    settings.promptSuffix = '';
    hud.toast('the dream forgets');
  }
  window.addEventListener('keydown', (e) => {
    if (e.target.closest?.('textarea,input')) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;   // Cmd+R reloads; it must not also forget
    const k = e.code;
    if (k === 'Tab' || k === 'KeyP') { e.preventDefault(); hud.togglePanel(); }
    else if (k === 'KeyH') hud.toggleVisible();
    else if (k === 'KeyR') forgetDream();
    else if (k === 'KeyF') toggleCompare();
    else if (k === 'BracketLeft') { settings.strength = Math.max(0.1, +(settings.strength - 0.05).toFixed(2)); hud.toast(`strength ${settings.strength.toFixed(2)}`); hud.syncPanel(); }
    else if (k === 'BracketRight') { settings.strength = Math.min(0.95, +(settings.strength + 0.05).toFixed(2)); hud.toast(`strength ${settings.strength.toFixed(2)}`); hud.syncPanel(); }
    else if (k === 'KeyM') toggleAudio();
    else if (/^Digit[1-9]$/.test(k) && LOOKS[+k.slice(5) - 1]) applyLook(LOOKS[+k.slice(5) - 1].id, true);
    else if (k === 'KeyE' && !e.repeat && entered && !(player instanceof FlyCam)) { eyes.held = true; closeEyes(true); }
    else if (k === 'Enter' && !entered) enter();
  });
  window.addEventListener('keyup', (e) => { if (e.code === 'KeyE' && eyes.held) { eyes.held = false; closeEyes(false); } });
  // a held E never sees its keyup if focus leaves (Cmd-Tab, another tab): open then
  window.addEventListener('blur', () => { if (eyes.held) { eyes.held = false; closeEyes(false); } });

  // ---- sizing (adaptive internal resolution)
  const pixelBudget = num('pixels', isMobile ? 1.1e6 : 2.6e6);
  let resScale = num('res', 1);
  const adaptive = P('adaptive', '1') === '1';
  function resize() {
    const w = innerWidth, h = innerHeight;
    let pr = Math.min(devicePixelRatio || 1, isMobile ? 2 : 1.5);
    const px = w * h * pr * pr;
    if (px > pixelBudget) pr *= Math.sqrt(pixelBudget / px);
    pr *= resScale;
    renderer.setPixelRatio(pr);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    post.setSize(canvas.width, canvas.height);
  }
  window.addEventListener('resize', resize);
  resize();

  // ---- autopilot / attract mode
  const apFwd = new THREE.Vector3(-Math.sin(startPose.yaw), 0, -Math.cos(startPose.yaw));
  const apRight = new THREE.Vector3(Math.cos(startPose.yaw), 0, -Math.sin(startPose.yaw));
  let apYaw = startPose.yaw;
  function currentYaw() { return apYaw; }
  function autopilotPose(t) {
    const pos = startPose.pos.clone()
      .addScaledVector(apFwd, 1.6 * Math.sin(t * 0.05) + (P('autopilot', '1') === '2' ? t * 0.35 : 0))
      .addScaledVector(apRight, 1.1 * Math.sin(t * 0.037));
    apYaw = startPose.yaw + 0.75 * Math.sin(t * 0.085) + 0.25 * Math.sin(t * 0.031);
    const pitch = 0.1 * Math.sin(t * 0.063) + 0.06;
    return { pos, yaw: apYaw, pitch };
  }

  // ---- idle gaze: after ?idle seconds (default 90, 0 = never) without input the view
  // drifts slowly around where you stand, so the dream keeps painting what it turns to;
  // any input hands the view straight back from wherever the drift left it.
  const IDLE = num('idle', 90);
  let lastInputAt = performance.now(), lastActivity = -1, gaze = null;
  for (const ev of ['keydown', 'pointerdown', 'touchstart', 'wheel']) {
    window.addEventListener(ev, () => { lastInputAt = performance.now(); }, { capture: true, passive: true });
  }
  function idleGaze(now, time) {
    const act = player.input?.activity ?? 0;
    if (act !== lastActivity) { lastActivity = act; lastInputAt = now; }
    const idle = IDLE > 0 && entered && !player.autopilot && !hud.panelOpen && typeof player.look === 'function'
      && (now - lastInputAt) / 1000 > IDLE;
    if (!idle) { gaze = null; return; }
    if (!gaze) gaze = { t0: time, yaw: player.yaw, pitch: player.pitch };
    const t = time - gaze.t0, u = Math.min(1, t / 8), e = u * u * (3 - 2 * u);   // 8 s ease-in
    player.look(gaze.yaw + e * (0.55 * Math.sin(t * 0.09) + 0.18 * Math.sin(t * 0.23)),
      gaze.pitch + e * (0.06 * Math.sin(t * 0.13) - gaze.pitch));
  }

  // ---- main loop
  let zoneIndex = level.zoneAt(camera.position) | 0;
  mockTint();
  function mockTint() { if (link?.setTint) link.setTint(level.zones[zoneIndex]?.light || [1, 0.8, 0.6]); }
  atmos.update(zoneIndex, 1);
  let last = performance.now();
  const t0 = last;
  let frames = 0, fpsT = last, fps = 60, frameMsAvg = 16, lowFor = 0, highFor = 0, lastStats = 0, lastAudio = 0;
  const lineColor = new THREE.Vector3();
  let perfAcc = { n: 0, cpu: 0 };
  // Frame-exact snapshots for tools/shoot.mjs: resolves with the canvas PNG of the first
  // frame rendered once the tour clock (or scene clock) reaches t. Nothing is drawn on screen.
  const snaps = [];
  function serviceSnaps(time) {
    // at most one per frame (the earliest due), so a stall never hands two snaps one image
    const tour = player.autopilot ? player.autopilot.t : null;
    let best = -1;
    for (let i = 0; i < snaps.length; i++) {
      const s = snaps[i], clock = s.clock === 'scene' ? time : tour;
      if (clock != null && clock >= s.t && (best < 0 || s.t < snaps[best].t)) best = i;
    }
    if (best < 0) return;
    const s = snaps.splice(best, 1)[0];
    const e = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ');
    const out = { png: canvas.toDataURL('image/png'), t: s.clock === 'scene' ? time : tour, pos: camera.position.toArray(),
      yaw: e.y, pitch: e.x, zone: level.zones[zoneIndex]?.id };
    // with depth: [w, h], also the view's linear depth (m) on a w x h grid, top row first,
    // and the camera, so the tool can warp one frame onto the next
    if (s.depth) Object.assign(out, { depth: snapDepth(...s.depth), proj: camera.projectionMatrix.toArray(), world: camera.matrixWorld.toArray() });
    s.resolve(out);
  }
  let depthPass = null;
  function snapDepth(w, h) {
    if (!depthPass) {
      const mat = new THREE.ShaderMaterial({
        uniforms: { uDepth: { value: null }, uNear: { value: 0.05 }, uFar: { value: 1000 } },
        vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
        fragmentShader: `uniform sampler2D uDepth; uniform float uNear; uniform float uFar; varying vec2 vUv;
          void main(){
            float d = texture(uDepth, vUv).r;
            float v = clamp((uNear * uFar) / (uFar - (uFar - uNear) * d) / uFar, 0.0, 0.9999);   // needed: a cleared pixel would wrap to a tiny depth
            vec3 enc = fract(v * vec3(1.0, 255.0, 65025.0));
            enc -= enc.yzz * vec3(1.0 / 255.0, 1.0 / 255.0, 0.0);
            gl_FragColor = vec4(enc, 1.0);
          }`,
        depthTest: false, depthWrite: false,
      });
      const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat);
      quad.frustumCulled = false;
      depthPass = { mat, scene: new THREE.Scene().add(quad), cam: new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1), rt: null };
    }
    const P = depthPass;
    if (!P.rt || P.rt.width !== w || P.rt.height !== h) {
      P.rt?.dispose();
      P.rt = new THREE.WebGLRenderTarget(w, h, { depthBuffer: false });
      P.buf = new Uint8Array(w * h * 4);
    }
    P.mat.uniforms.uDepth.value = post.sceneRT.depthTexture;
    P.mat.uniforms.uNear.value = camera.near; P.mat.uniforms.uFar.value = camera.far;
    const prev = renderer.getRenderTarget();
    renderer.setRenderTarget(P.rt);
    renderer.render(P.scene, P.cam);
    // (an async capture readback may still hold the pixel-pack buffer: unbind, or this read is a no-op)
    const gl = renderer.getContext();
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    renderer.readRenderTargetPixels(P.rt, 0, 0, w, h, P.buf);
    renderer.setRenderTarget(prev);
    const d = new Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = ((h - 1 - y) * w + x) * 4, b = P.buf;   // GL rows run bottom-up
      d[y * w + x] = (b[i] / 255 + b[i + 1] / 65025 + b[i + 2] / 16581375) * camera.far;
    }
    return d;
  }

  function frame(now) {
    requestAnimationFrame(frame);
    const dt = Math.min(0.1, Math.max(0, (now - last) / 1000));
    last = now;
    const time = (now - t0) / 1000;
    const cpu0 = performance.now();

    // camera
    if (hasTour) {
      if (!entered && !player.autopilot && P('attract', '1') === '1') player.startAutopilot(0);
      idleGaze(now, time);
      if (!hud.panelOpen || isTouch || player.autopilot) player.update(dt);
    } else if (AUTOPILOT || !entered) {
      const p = autopilotPose(time);
      camera.position.copy(p.pos);
      camera.rotation.set(p.pitch, p.yaw, 0, 'YXZ');
    } else if (!hud.panelOpen || isTouch) {
      player.update(dt);
    }
    camera.updateMatrixWorld();

    // zone
    const zi = level.zoneAt(camera.position) | 0;
    if (zi !== zoneIndex && level.zones[zi]) {
      settings.promptSuffix = eyes.dreams[zi] || '';   // each zone keeps its own re-dream
      zoneIndex = zi;
      if (entered) hud.zoneCard(level.zones[zi]);
      mockTint();
      try { ambience?.setZone(level.zones[zi]); } catch {}
    }
    atmos.update(zoneIndex, dt);
    level.decor?.userData?.update?.(time, canvas.height);
    lightsDrv.update(camera.position, time);
    shared.uTime.value = time;
    const zl = level.zones[zoneIndex]?.light || [0.5, 0.7, 1];
    lineColor.set(zl[0], zl[1], zl[2]).multiplyScalar(0.55);

    // compare crossfade
    dreamMix += ((compare ? 0 : 1) - dreamMix) * (1 - Math.exp(-dt * 4));
    materials.display.uniforms.uDreamMix.value = dreamMix;
    // Standing still, many passes refine one framing, so modest feedback keeps the level's
    // structure (more makes the loop simplify and drift). Walking, each framing gets about
    // one pass, so more feedback carries the refined dream across viewpoints instead of
    // starting each from the misty first take. Scaled, so the panel's slider still rules.
    const moving = dream.motion;
    materials.capture.uniforms.uFeedback.value = Math.min(0.95, Math.max(0, settings.feedback * (1 + (settings.feedbackMoving - 1) * moving)));
    // eyes: the lid closes in ~0.2 s and opens in ~0.4 s; once it is mostly shut the room is re-dreamt
    if (eyes.until && now > eyes.until) closeEyes(false);
    eyes.lid += ((eyes.closed ? 1 : 0) - eyes.lid) * (1 - Math.exp(-dt / (eyes.closed ? 0.2 : 0.4)));
    const shut = eyes.closed && eyes.lid > 0.85;
    if (shut && !settings.eyesClosed) {   // just shut: the room will open dreamt as something else
      eyes.dreams[zoneIndex] = settings.promptSuffix = EYE_DREAMS[eyes.n++ % EYE_DREAMS.length];
    }
    settings.eyesClosed = shut;
    if (settings.eyesClosed) materials.capture.uniforms.uFeedback.value = settings.eyesFeedback;
    dream.live.tau = settings.eyesClosed ? Math.min(settings.liveTau, 0.12) : settings.liveTau;
    materials.capture.uniforms.uFeedbackAnchor.value = settings.feedbackAnchor;
    materials.capture.uniforms.uFeedbackSat.value = settings.feedbackSat;
    materials.display.uniforms.uRelight.value = settings.relight;

    // dream: capture / paint
    dream.update(now, dt, camera, zoneIndex, level.zones[zoneIndex]);

    // main render
    renderer.setRenderTarget(post.sceneRT);
    const fc = shared.uFogColor.value;
    renderer.setClearColor(new THREE.Color(fc.x, fc.y, fc.z), 1);
    renderer.clear(true, true, false);
    renderer.render(scene, camera);
    post.compMat.uniforms.uFade.value = Math.min(1, time / 2.5);
    post.render(camera, time, { lineColor, depthSep: settings.depthSep, lid: eyes.lid });
    if (snaps.length) serviceSnaps(time);

    // timing + adaptive resolution
    frames++;
    perfAcc.n++; perfAcc.cpu += performance.now() - cpu0;
    frameMsAvg = frameMsAvg * 0.95 + dt * 1000 * 0.05;
    if (now - fpsT > 1000) {
      fps = frames * 1000 / (now - fpsT); frames = 0; fpsT = now;
      if (adaptive && time > 4) {
        if (fps < 48) { lowFor++; highFor = 0; } else if (fps > 58) { highFor++; lowFor = 0; } else { lowFor = 0; highFor = 0; }
        if (lowFor >= 2 && resScale > 0.5) { resScale = Math.max(0.5, resScale * 0.85); resize(); lowFor = 0; }
        if (highFor >= 6 && resScale < 1) { resScale = Math.min(1, resScale * 1.1); resize(); highFor = 0; }
      }
      if (PERF) {
        const s = dream.stats;
        console.log(`[perf] fps=${fps.toFixed(1)} cpu=${(perfAcc.cpu / perfAcc.n).toFixed(2)}ms res=${canvas.width}x${canvas.height} scale=${resScale.toFixed(2)} dream=${s.dreamFps.toFixed(1)}/s lat=${s.latency.toFixed(0)}ms enc=${s.encodeMs.toFixed(1)}ms paint=${s.paintMs.toFixed(2)}ms caps=${s.captures} res=${s.results} drop=${s.dropped}`);
        perfAcc = { n: 0, cpu: 0 };
      }
    }
    if (now - lastStats > 250) {
      lastStats = now;
      const s = dream.stats;
      const eng = engineInfo ? `${engineInfo.engine} ${engineInfo.width}x${engineInfo.height}` : (link ? linkStatus : 'off');
      hud.setStats([
        `${fps.toFixed(0)} fps  ${canvas.width}x${canvas.height}`,
        `dream ${s.dreamFps.toFixed(1)}/s  ${s.latency ? s.latency.toFixed(0) + 'ms' : '--'}`,
        `${eng}${serverStats ? `  q${serverStats.queue}` : ''}`,
        `paint ${s.paintMs.toFixed(1)}ms  enc ${s.encodeMs.toFixed(0)}ms`,
        `strength ${(settings.eyesClosed && dream.info?.depth ? settings.eyesStrength : settings.strength).toFixed(2)}  fb ${materials.capture.uniforms.uFeedback.value.toFixed(2)}`,
      ]);
    }
    if (ambience && now - lastAudio > 300) {
      lastAudio = now;
      try { ambience.setDream(Math.min(1, dream.stats.dreamFps / 8) * (compare ? 0.2 : 1) * (1 - 0.7 * eyes.lid)); } catch {}
    }
  }
  requestAnimationFrame(frame);
  if (P('title', '1') === '0') enter(false);

  window.__hyp = {
    THREE, renderer, scene, camera, level, painter, dream, settings, post, materials, shared, link, player, memory,
    snapAt: (t, clock = 'tour', opts = {}) => new Promise((resolve) => snaps.push({ t, clock, resolve, depth: opts.depth })),
    perf: () => ({ fps, res: [canvas.width, canvas.height], dream: { ...dream.stats } }),
    audio: () => ({ started: audioStarted, ambience }),
    eyes: (on, forS) => closeEyes(on, forS),
  };
}

boot().catch((e) => { console.error(e); showError('boot failed: ' + (e?.stack || e)); });
