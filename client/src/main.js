// HYPNAGOGIA - bootstrap + main loop (render owner).
import * as THREE from 'three';
import { createSharedUniforms, createLevelMaterials, LightDriver, ZoneAtmos, buildChunks } from './render/materials.js';
import { Post } from './render/post.js';
import { Hud } from './render/hud.js';
import { Painter } from './dream/painter.js';
import { Dream } from './dream/dream.js';
import { Link, MockLink } from './dream/link.js';
import { FlyCam } from './dream/flycam.js';

const params = new URLSearchParams(location.search);
const P = (k, d) => (params.has(k) ? params.get(k) : d);
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
  strength: +P('strength', 0.6),
  feedback: +P('feedback', 0.55),
  paintRate: +P('paint', 0.65),
  captureRate: +P('rate', 20),
  maxInFlight: +P('inflight', 2),
  paintSpread: +P('spread', 3),
  captureFov: +P('capfov', 96),
  jpegQuality: +P('q', 0.9),
  relaxHalfLife: +P('memory', 300),
  seed: +P('dseed', 20260928),
  lead: +P('lead', 1),
  saccade: +P('saccade', 0),
  kfDist: +P('kfdist', 0.75),
  kfAngle: +P('kfangle', 0.16),
  glance: +P('glance', 0.62),
  promptOverride: P('prompt', ''),
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
  const canvas = $('view');
  const renderer = new THREE.WebGLRenderer({
    canvas, antialias: false, alpha: false, stencil: false, depth: true, powerPreference: 'high-performance',
  });
  renderer.autoClear = false;
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  const halfFloat = renderer.extensions.has('EXT_color_buffer_float') || renderer.extensions.has('EXT_color_buffer_half_float');

  const seed = +P('seed', 7);
  const atlasSize = +P('atlas', isMobile ? 2048 : 4096);
  const { level, real } = await loadLevel(seed, atlasSize);
  console.log(`[hypnagogia] level: ${real ? 'world' : 'test'} seed=${seed} atlas=${level.atlas?.size} tpm=${level.atlas?.texelsPerMeter?.toFixed?.(1)}`);

  const camera = new THREE.PerspectiveCamera(+P('fov', 72), innerWidth / innerHeight, 0.05, 1000);
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
  for (const g of geos) chunkGeos.push(...buildChunks(g, +P('chunk', 36)));
  console.log(`[hypnagogia] ${chunkGeos.length} paint chunks, ${geos.reduce((a, g) => a + (g.index ? g.index.count : g.attributes.position.count) / 3, 0)} triangles`);
  if (level.decor) {
    level.decor.traverse((o) => o.layers.set(1));
    scene.add(level.decor);
  }
  const painter = new Painter(renderer, level, chunkGeos, materials, {
    atlasSize: level.atlas?.size || atlasSize, halfFloat: halfFloat && P('atlas8', '0') !== '1',
  });
  shared.uAtlas.value = painter.texture;
  const lightsDrv = new LightDriver(level, shared);
  const atmos = new ZoneAtmos(level, shared);
  const post = new Post(renderer, { halfFloat });

  // ---- dream link
  const dreamMode = P('dream', 'server');
  let link = null;
  if (dreamMode === 'mock') link = new MockLink(renderer, { latency: +P('mocklat', 150) });
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
    renderer, scene, level, materials, painter, link, settings,
    onEvent: (ev, a) => {
      if (ev === 'info') { engineInfo = a; renderStatus(); }
      else if (ev === 'status') { linkStatus = a; renderStatus(); if (a === 'connected' && entered) hud?.toast('the dream engine wakes'); }
      else if (ev === 'stats') serverStats = a;
    },
  });

  // ---- player / controls
  const player = await loadPlayer(level, camera, canvas, real);
  const spawnPos = new THREE.Vector3(...(level.spawn?.position || [0, 1.6, 0]));
  const spawnYaw = level.spawn?.yaw || 0;
  const startPose = { pos: spawnPos.clone(), yaw: spawnYaw, pitch: 0 };
  if (params.has('pos')) {
    const [x, y, z] = P('pos', '0,1.6,0').split(',').map(Number);
    startPose.pos.set(x, y, z);
  }
  if (params.has('yaw')) startPose.yaw = +P('yaw', 0);
  if (params.has('pitch')) startPose.pitch = +P('pitch', 0);
  // The world Player owns spawn/URL pose and its own scripted tour; the fly cam
  // (test level) gets the pose from us.
  const hasTour = typeof player.startAutopilot === 'function' && (level.tour?.length || 0) > 1;
  if (player instanceof FlyCam) player.setPose(startPose.pos.clone(), startPose.yaw, startPose.pitch);
  const initialPose = { pos: (player.position || camera.position).clone(), yaw: player.yaw ?? startPose.yaw, pitch: player.pitch ?? 0 };

  let ambience = null;
  let compare = false;
  let dreamMix = 1;

  hud = new Hud(settings, {
    forget: () => { dream.forget(); hud.toast('the dream forgets'); },
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
    try { ambience.toggle(); hud.toast('sound'); } catch (e) { console.warn(e); }
  }

  function lockPointer() {
    try { const r = canvas.requestPointerLock?.(); r?.catch?.(() => {}); } catch {}
  }
  async function enter() {
    if (entered) return;
    entered = true;
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
    ambience = await loadAmbience();
    if (ambience) {
      try { await ambience.start(); ambience.setZone(level.zones[zoneIndex]); } catch (e) { console.warn('[hypnagogia] ambience start failed', e); ambience = null; }
    }
    setTimeout(() => hud.zoneCard(level.zones[zoneIndex]), 900);
  }
  $('title').addEventListener('click', enter);
  $('title').addEventListener('touchend', (e) => { e.preventDefault(); enter(); });
  canvas.addEventListener('click', () => {
    if (entered && !hud.panelOpen && !isTouch && document.pointerLockElement !== canvas) lockPointer();
  });

  window.addEventListener('keydown', (e) => {
    if (e.target.closest?.('textarea,input')) return;
    const k = e.code;
    if (k === 'Tab' || k === 'KeyP') { e.preventDefault(); hud.togglePanel(); }
    else if (k === 'KeyH') hud.toggleVisible();
    else if (k === 'KeyR') { dream.forget(); hud.toast('the dream forgets'); }
    else if (k === 'KeyF') toggleCompare();
    else if (k === 'BracketLeft') { settings.strength = Math.max(0.1, +(settings.strength - 0.05).toFixed(2)); hud.toast(`strength ${settings.strength.toFixed(2)}`); hud.syncPanel(); }
    else if (k === 'BracketRight') { settings.strength = Math.min(0.95, +(settings.strength + 0.05).toFixed(2)); hud.toast(`strength ${settings.strength.toFixed(2)}`); hud.syncPanel(); }
    else if (k === 'KeyM') toggleAudio();
    else if (k === 'Enter' && !entered) enter();
  });

  // ---- sizing (adaptive internal resolution)
  const pixelBudget = +P('pixels', isMobile ? 1.1e6 : 2.6e6);
  let resScale = +P('res', 1);
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

  function frame(now) {
    requestAnimationFrame(frame);
    const dt = Math.min(0.1, Math.max(0, (now - last) / 1000));
    last = now;
    const time = (now - t0) / 1000;
    const cpu0 = performance.now();

    // camera
    if (hasTour) {
      if (!entered && !player.autopilot && P('attract', '1') === '1') player.startAutopilot(0);
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
      zoneIndex = zi;
      if (entered) hud.zoneCard(level.zones[zi]);
      mockTint();
      try { ambience?.setZone(level.zones[zi]); } catch {}
    }
    atmos.update(zoneIndex, dt);
    try {
      level.update?.(dt, time, camera);
      level.decor?.userData?.update?.(dt, time, camera);
    } catch (e) { if (!frame._decorWarned) { console.warn('[hypnagogia] level/decor update failed', e); frame._decorWarned = true; } }
    lightsDrv.update(camera.position, time);
    shared.uTime.value = time;
    const zl = level.zones[zoneIndex]?.light || [0.5, 0.7, 1];
    lineColor.set(zl[0], zl[1], zl[2]).multiplyScalar(0.55);

    // compare crossfade
    dreamMix += ((compare ? 0 : 1) - dreamMix) * (1 - Math.exp(-dt * 4));
    materials.display.uniforms.uDreamMix.value = dreamMix;
    materials.capture.uniforms.uFeedback.value = settings.feedback;

    // dream: capture / paint
    dream.update(now, dt, camera, zoneIndex, level.zones[zoneIndex]);

    // main render
    renderer.setRenderTarget(post.sceneRT);
    const fc = shared.uFogColor.value;
    renderer.setClearColor(new THREE.Color(fc.x, fc.y, fc.z), 1);
    renderer.clear(true, true, false);
    renderer.render(scene, camera);
    post.compMat.uniforms.uFade.value = Math.min(1, time / 2.5);
    post.render(camera, time, { lineColor });

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
        `strength ${settings.strength.toFixed(2)}  fb ${settings.feedback.toFixed(2)}`,
      ]);
    }
    if (ambience && now - lastAudio > 300) {
      lastAudio = now;
      try { ambience.setDream(Math.min(1, dream.stats.dreamFps / 8) * (compare ? 0.2 : 1)); } catch {}
    }
  }
  requestAnimationFrame(frame);
  if (P('title', '1') === '0') enter();

  window.__hyp = { THREE, renderer, scene, camera, level, painter, dream, settings, post, materials, shared, link, player };
}

boot().catch((e) => { console.error(e); showError('boot failed: ' + (e?.stack || e)); });
