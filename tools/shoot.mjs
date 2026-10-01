#!/usr/bin/env node
// Frame-exact screenshots of the game on a fixed walk through each zone, for before/after
// comparisons of the dreamt look. Needs a running server (./run.sh ...) and Playwright.
//
//   PLAYWRIGHT=/path/to/node_modules/playwright/index.mjs \
//   node tools/shoot.mjs --url http://127.0.0.1:PORT/ [--out DIR] [--zones nave,stacks]
//                        [--browser chromium|webkit] [--size 1280x720] [--speed 3.4]
//                        [--device 'iPhone 15']   (Playwright device: touch, DPR, phone settings)
//                        [--settle 5] [--params 'strength=0.5&...'] [--perf 20]
//
// Per zone it loads the page at a tour waypoint, lets the dream paint that view for
// --settle seconds after the first result, then walks the tour at --speed m/s and grabs
// the frame at tour time 2, 4 and 6 s ("walk"), then stands still 3 s ("rest"). Frames are
// read from the canvas in the same frame they were drawn (window.__hyp.snapAt), so two
// runs give the same poses to within a frame. Writes <zone>_<phase>.png and shots.json.
// --flicker instead measures temporal stability: per zone, 40 frames at 10 Hz standing
// still and then 40 walking (1.7 m/s), each reduced to a 160x90 luma grid in the page;
// shots.json gets the mean |change| between consecutive frames (0-255) as series + summary,
// raw and "warped" (after reprojecting the previous frame with depth: see changeSeries).
// --strip instead saves a walking filmstrip per zone (see stripZone).
// --perf N instead rides the nave tour (?autopilot=1&tour=3) for N s after the first dream
// and reports frame-time percentiles, each frame timed from its start to gl.finish() so
// GPU work counts. It needs no hooks in the page, so it measures any client version
// (serve an old one with ./run.sh --client-dir DIR).
//
// Headless Chromium on a Mac without a display session throttles requestAnimationFrame
// to ~5 Hz, so the page gets a 60 Hz setTimeout-paced rAF and the browser runs with
// vsync and the frame-rate limit off.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? argv[i + 1] : d; };
const base = opt('url');
if (!base) { console.error('usage: node tools/shoot.mjs --url http://127.0.0.1:PORT/ [--out DIR] ...'); process.exit(2); }
const out = opt('out', path.join(os.tmpdir(), 'hypnagogia-shots'));
const browserName = opt('browser', 'chromium');
const [W, H] = opt('size', '1280x720').split('x').map(Number);
const speed = +opt('speed', 3.4);
const settle = +opt('settle', 5);
const extra = opt('params', '');
const perfSeconds = +opt('perf', 0);
const device = opt('device', '');
const flicker = argv.includes('--flicker');
const strip = argv.includes('--strip');
const pw = await import(process.env.PLAYWRIGHT || 'playwright');

// Zone -> tour waypoint index (client/src/world/levelcore.js `tour`, seed 7).
const ZONES = { vestibule: 0, nave: 3, stacks: 6, geode: 9, baths: 15, garden: 18, atrium: 24, desert: 30 };
const zones = (opt('zones', Object.keys(ZONES).join(','))).split(',');
const WALK_T = [2, 4, 6];

const launchArgs = browserName === 'chromium'
  ? ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--disable-gpu-vsync', '--disable-frame-rate-limit']
  : [];
const browser = await pw[browserName].launch({ headless: true, args: launchArgs, chromiumSandbox: true });
fs.mkdirSync(out, { recursive: true });

const wsUrl = (() => { const w = new URL('/ws', base); w.protocol = w.protocol === 'https:' ? 'wss:' : 'ws:'; return encodeURIComponent(w.href); })();   // never the page's :8765 fallback
async function openAt(k, params) {
  const page = await browser.newPage(device ? { ...pw.devices[device] } : { viewport: { width: W, height: H } });
  const logs = [];
  page.on('console', (m) => logs.push(`${m.type()}: ${m.text()}`));
  page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));
  await page.addInitScript(() => {
    // 60 Hz rAF; with window.__ft set, each frame is timed to gl.finish() on the game canvas
    let last = 0, gl = null;
    const getContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, attrs) {
      const c = getContext.call(this, type, attrs);
      if (this.id === 'view' && /webgl/.test(type)) gl = c;
      return c;
    };
    window.requestAnimationFrame = (cb) => {
      const wait = Math.max(0, 1000 / 60 - (performance.now() - last));
      return setTimeout(() => {
        last = performance.now();
        cb(last);
        if (window.__ft && gl) { gl.finish(); window.__ft.push(performance.now() - last); window.__ftAt.push(last); }
      }, wait);
    };
    window.cancelAnimationFrame = (id) => clearTimeout(id);
  });
  const url = new URL(base);
  url.search = `title=0&hud=0&adaptive=0&server=${wsUrl}&${params}`;
  await page.goto(url.href);
  if (k == null) return { page, logs };
  // Stand at waypoint k: the level module is imported by the page, so ask it for the tour.
  await page.waitForFunction(() => window.__hyp?.level?.tour, null, { timeout: 60000 });
  await page.evaluate((k) => {
    const h = window.__hyp, w = h.level.tour[k];
    h.player.setPose([w[0], w[1], w[2]], w[3], w[4]);
    h.player.frozen = true;
    h.painter.clear();
  }, k);
  return { page, logs };
}

const png = (dataUrl, file) => fs.writeFileSync(file, Buffer.from(dataUrl.split(',')[1], 'base64'));

async function shootZone(zone) {
  const k = ZONES[zone];
  const { page, logs } = await openAt(k, `perf=1&${extra}`);
  const dreaming = !/(^|&)dream=off/.test(extra);
  if (dreaming) await page.waitForFunction(() => window.__hyp.dream.stats.results >= 1, null, { timeout: 90000 });
  await page.waitForTimeout(settle * 1000);
  const shots = [];
  const settled = await page.evaluate(() => window.__hyp.snapAt(0, 'scene'));
  png(settled.png, path.join(out, `${zone}_settled.png`));
  shots.push({ zone, phase: 'settled', t: 0, pos: settled.pos, file: `${zone}_settled.png` });
  await page.evaluate(({ k, speed }) => {
    const p = window.__hyp.player;
    p.startAutopilot(k);
    p.autopilot.speed = speed;
  }, { k, speed });
  const walk = await page.evaluate((ts) => Promise.all(ts.map((t) => window.__hyp.snapAt(t))), WALK_T);
  walk.forEach((s, i) => {
    const file = `${zone}_walk${i + 1}.png`;
    png(s.png, path.join(out, file));
    shots.push({ zone, phase: `walk${i + 1}`, t: s.t, pos: s.pos, file });
  });
  await page.evaluate(() => { window.__hyp.player.autopilot.speed = 0; });
  const tRest = walk[walk.length - 1].t + 3;
  const rest = await page.evaluate((t) => window.__hyp.snapAt(t), tRest);
  png(rest.png, path.join(out, `${zone}_rest.png`));
  shots.push({ zone, phase: 'rest', t: rest.t, pos: rest.pos, file: `${zone}_rest.png` });
  const perf = await page.evaluate(() => window.__hyp.perf());
  const capture = await page.evaluate(() => { const d = window.__hyp.dream; return { size: [d.width, d.height], fov: d._fit?.fov ?? null }; });
  const errors = logs.filter((l) => /^(error|pageerror|warning)/.test(l));
  await page.close();
  return { shots, perf: { fps: perf.fps, res: perf.res, dream: perf.dream }, capture, errors };
}

// A filmstrip: after --settle, walk the tour at --speed and grab 8 frames 0.15 s apart from
// tour time 3 s (<zone>_strip0..7.png), for judging by eye whether things keep their identity.
async function stripZone(zone) {
  const k = ZONES[zone];
  const { page, logs } = await openAt(k, `perf=1&${extra}`);
  await page.waitForFunction(() => window.__hyp.dream.stats.results >= 1, null, { timeout: 90000 });
  await page.waitForTimeout(settle * 1000);
  await page.evaluate(({ k, speed }) => { const p = window.__hyp.player; p.startAutopilot(k); p.autopilot.speed = speed; }, { k, speed });
  const snaps = await page.evaluate((ts) => Promise.all(ts.map((t) => window.__hyp.snapAt(t))), Array.from({ length: 8 }, (_, i) => 3 + i * 0.15));
  const shots = snaps.map((sn, i) => {
    const file = `${zone}_strip${i}.png`;
    png(sn.png, path.join(out, file));
    return { zone, phase: `strip${i}`, t: sn.t, pos: sn.pos, file };
  });
  const errors = logs.filter((l) => /^(error|pageerror|warning)/.test(l));
  await page.close();
  return { shots, errors };
}

// Mean |luma change| between consecutive frames, 40 frames 0.1 s apart on `clock`, on a
// 160x90 grid (exact area averages of the frame): raw, and after warping the previous frame
// onto the current one with the current depth and both cameras ("warped"). Warping cancels
// most detail that only moved with the camera; what stays is paint that changed
// (re-invention, boiling) plus resampling at edges and view-dependent light. Grain and
// vignette are off while measuring (grain changes every frame; the vignette is fixed to
// the screen). Pixels hidden in the previous frame or off its edge are skipped; pairs with
// no valid pixel are counted in `dropped`. `detail` is the mean |luma gradient| of the full
// frame (and `...Centre` of its middle half), so a build cannot look steadier just by
// painting softer without it showing.
// `sil` / `tex` are how much the grid's luma steps between neighbouring cells across a depth
// silhouette (depths differ by more than 30%) and within a surface (within 2%; nothing past
// 300 m, which drops the sky dome):
// a pillar painted into the wall behind it has sil near tex. `dreamFps` is the dream rate over
// the last 2 s of each series.
async function changeSeries(page, clock) {
  const W = 160, H = 90;
  const cur = await page.evaluate(({ c, W, H }) => {
    const u = window.__hyp.post.compMat.uniforms;
    u.uGrain.value = 0; u.uVignette.value = 0;
    return window.__hyp.snapAt(0, c, { depth: [W, H] });   // also builds the depth pass
  }, { c: clock, W, H });
  return page.evaluate(async ({ t0, clock, W, H }) => {
    const c = document.createElement('canvas');
    const g = c.getContext('2d', { willReadFrequently: true });
    const shots = await Promise.all(Array.from({ length: 40 }, (_, i) => window.__hyp.snapAt(t0 + 0.5 + i * 0.1, clock, { depth: [W, H] })));
    const grid = (img) => {
      c.width = img.width; c.height = img.height; g.drawImage(img, 0, 0);
      const d = g.getImageData(0, 0, c.width, c.height).data, cw = c.width, ch = c.height;
      const full = new Float32Array(cw * ch);
      for (let i = 0; i < full.length; i++) full[i] = 0.2126 * d[i * 4] + 0.7152 * d[i * 4 + 1] + 0.0722 * d[i * 4 + 2];
      let grad = 0, gradC = 0, nC = 0;
      for (let y = 0; y < ch - 1; y++) for (let x = 0; x < cw - 1; x++) {
        const i = y * cw + x, g = Math.abs(full[i + 1] - full[i]) + Math.abs(full[i + cw] - full[i]);
        grad += g;
        if (x >= cw / 4 && x < cw * 3 / 4 && y >= ch / 4 && y < ch * 3 / 4) { gradC += g; nC++; }
      }
      const y = new Float32Array(W * H);
      for (let gy = 0; gy < H; gy++) {
        const y0 = Math.floor(gy * ch / H), y1 = Math.floor((gy + 1) * ch / H);
        for (let gx = 0; gx < W; gx++) {
          const x0 = Math.floor(gx * cw / W), x1 = Math.floor((gx + 1) * cw / W);
          let sum = 0;
          for (let yy = y0; yy < y1; yy++) for (let xx = x0; xx < x1; xx++) sum += full[yy * cw + xx];
          y[gy * W + gx] = sum / ((y1 - y0) * (x1 - x0));
        }
      }
      return { y, detail: grad / ((cw - 1) * (ch - 1)), detailC: gradC / Math.max(1, nC) };
    };
    const view = (m) => {   // inverse of a rigid camera matrix (column-major)
      const v = [m[0], m[4], m[8], 0, m[1], m[5], m[9], 0, m[2], m[6], m[10], 0, 0, 0, 0, 1];
      for (let r = 0; r < 3; r++) v[12 + r] = -(v[r] * m[12] + v[4 + r] * m[13] + v[8 + r] * m[14]);
      return v;
    };
    const warped = (a, b) => {   // a = previous frame, b = current
      const Pc = b.proj, M = b.world, Pp = a.proj, V = view(a.world);
      let sum = 0, n = 0;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const z = b.depth[y * W + x];
        if (!(z > 0.06) || z > 900) continue;
        const xv = ((x + 0.5) / W * 2 - 1) * z / Pc[0], yv = (1 - (y + 0.5) / H * 2) * z / Pc[5], zv = -z;
        const wx = M[0] * xv + M[4] * yv + M[8] * zv + M[12], wy = M[1] * xv + M[5] * yv + M[9] * zv + M[13], wz = M[2] * xv + M[6] * yv + M[10] * zv + M[14];
        const px = V[0] * wx + V[4] * wy + V[8] * wz + V[12], py = V[1] * wx + V[5] * wy + V[9] * wz + V[13], zp = -(V[2] * wx + V[6] * wy + V[10] * wz + V[14]);
        if (zp < 0.05) continue;
        const sx = Math.min(W - 1, Math.max(0, (px * Pp[0] / zp + 1) / 2 * W - 0.5)), sy = Math.min(H - 1, Math.max(0, (1 - py * Pp[5] / zp) / 2 * H - 0.5));
        const ex = (px * Pp[0] / zp + 1) / 2 * W - 0.5, ey = (1 - py * Pp[5] / zp) / 2 * H - 0.5;
        if (!(ex > -0.5 && ey > -0.5 && ex < W - 0.5 && ey < H - 0.5)) continue;
        if (Math.abs(a.depth[Math.round(sy) * W + Math.round(sx)] - zp) > 0.05 * zp + 0.05) continue;
        const x0 = Math.floor(sx), y0 = Math.floor(sy), x1 = Math.min(W - 1, x0 + 1), y1 = Math.min(H - 1, y0 + 1), fx = sx - x0, fy = sy - y0;
        const L = a.y;
        const l = (L[y0 * W + x0] * (1 - fx) + L[y0 * W + x1] * fx) * (1 - fy) + (L[y1 * W + x0] * (1 - fx) + L[y1 * W + x1] * fx) * fy;
        sum += Math.abs(b.y[y * W + x] - l); n++;
      }
      return { mean: n ? sum / n : null, valid: n / (W * H) };
    };
    const edges = (y, z) => {   // mean |luma step| across silhouettes and within surfaces
      let sa = 0, sn = 0, ta = 0, tn = 0;
      const pair = (i, j) => {
        const a = z[i], b = z[j];
        if (!(a > 0.06 && b > 0.06 && a < 300 && b < 300)) return;
        const r = Math.max(a, b) / Math.min(a, b), d = Math.abs(y[i] - y[j]);
        if (r > 1.3) { sa += d; sn++; } else if (r < 1.02) { ta += d; tn++; }
      };
      for (let gy = 0; gy < H; gy++) for (let gx = 0; gx < W; gx++) {
        const i = gy * W + gx;
        if (gx + 1 < W) pair(i, i + 1);
        if (gy + 1 < H) pair(i, i + W);
      }
      return { sil: sn ? sa / sn : null, tex: tn ? ta / tn : null, n: sn };
    };
    let prev = null, dropped = 0; const raw = [], warp = [], valid = [], detail = [], detailC = [], sil = [], tex = [];
    for (const s of shots) {
      const img = new Image(); img.src = s.png; await img.decode();
      const { y, detail: dt, detailC: dc } = grid(img);
      detail.push(+dt.toFixed(3));
      detailC.push(+dc.toFixed(3));
      const e = edges(y, s.depth);
      if (e.sil != null && e.tex != null && e.n >= 20) { sil.push(+e.sil.toFixed(3)); tex.push(+e.tex.toFixed(3)); }
      const f = { y, depth: s.depth, proj: s.proj, world: s.world };
      if (prev) {
        let a = 0; for (let i = 0; i < y.length; i++) a += Math.abs(y[i] - prev.y[i]);
        raw.push(+(a / y.length).toFixed(3));
        const w = warped(prev, f);
        valid.push(+w.valid.toFixed(3));
        if (w.mean != null) warp.push(+w.mean.toFixed(3)); else dropped++;
      }
      prev = f;
    }
    return { raw, warp, valid, dropped, detail, detailC, sil, tex, dreamFps: window.__hyp.dream.stats.dreamFps };
  }, { t0: cur.t, clock, W, H });
}
const summary = (d) => {
  const f = d.slice().sort((a, b) => a - b);
  return { mean: +(d.reduce((a, b) => a + b, 0) / d.length).toFixed(2), p90: f[Math.floor(0.9 * f.length)], max: f[f.length - 1] };
};
async function flickerZone(zone) {
  const k = ZONES[zone];
  const { page } = await openAt(k, `persist=0&${extra}`);
  await page.waitForFunction(() => window.__hyp.dream.stats.results >= 1, null, { timeout: 90000 });
  await page.waitForTimeout(6000);
  const still = await changeSeries(page, 'scene');
  await page.evaluate((k) => { const p = window.__hyp.player; p.startAutopilot(k); p.autopilot.speed = 1.7; }, k);
  const walking = await changeSeries(page, 'tour');
  await page.close();
  const mean = (v) => +(v.reduce((a, b) => a + b, 0) / Math.max(1, v.length)).toFixed(3);
  return {
    still: summary(still.raw), walking: summary(walking.raw),
    warped: { still: summary(still.warp), walking: summary(walking.warp), validStill: mean(still.valid), validWalking: mean(walking.valid),
      droppedStill: still.dropped, droppedWalking: walking.dropped },
    detail: { still: mean(still.detail), walking: mean(walking.detail), stillCentre: mean(still.detailC), walkingCentre: mean(walking.detailC) },
    edges: { silStill: mean(still.sil), texStill: mean(still.tex), silWalking: mean(walking.sil), texWalking: mean(walking.tex) },
    dreamFps: { still: +still.dreamFps.toFixed(2), walking: +walking.dreamFps.toFixed(2) },
    series: { still: still.raw, walking: walking.raw, stillWarped: still.warp, walkingWarped: walking.warp },
  };
}

async function perfRun(seconds) {
  const { page, logs } = await openAt(null, `autopilot=1&tour=${ZONES.nave}&perf=1&${extra}`);
  // any client version logs "[perf] ... caps=N res=N" once a second; wait for a first result
  const t0 = Date.now();
  while (Date.now() - t0 < 90000 && !logs.some((l) => /\[perf\].*caps=\d+ res=[1-9]/.test(l))) await page.waitForTimeout(250);
  await page.evaluate(() => { window.__ftAt = []; window.__ft = []; });
  await page.waitForTimeout(seconds * 1000);
  const [ft, at] = await page.evaluate(() => [window.__ft, window.__ftAt]);
  await page.close();
  // the five slowest frames and when they happened (s after measuring began): a one-off
  // (shader compile) reads differently from a periodic hitch
  const worst = ft.map((ms, i) => ({ ms: +ms.toFixed(2), at: +((at[i] - at[0]) / 1000).toFixed(2) }))
    .sort((a, b) => b.ms - a.ms).slice(0, 5);
  const f = ft.slice().sort((a, b) => a - b);
  const q = (p) => +f[Math.min(f.length - 1, Math.floor(p * f.length))].toFixed(2);
  const perfLines = logs.filter((l) => /\[perf\]/.test(l));
  return { frames: f.length, fps: +(f.length / seconds).toFixed(1), p50: q(0.5), p95: q(0.95), p99: q(0.99),
    max: +f[f.length - 1].toFixed(2), worst, lastPerf: perfLines[perfLines.length - 1] || null,
    errors: logs.filter((l) => /^(error|pageerror)/.test(l)) };
}

// only what describes the engine (the full /api/info can carry error text with local paths)
const info = await fetch(new URL('api/info', base)).then((r) => r.json()).catch(() => null);
const engine = info && Object.fromEntries(['engine', 'model', 'width', 'height', 'flexible', 'xframe', 'depth', 'depth_graft', 'lora', 'held_only', 'device']
  .map((k) => [k, k === 'model' && typeof info[k] === 'string' ? info[k].split('/').slice(-2).join('/') : info[k]]));   // (a local model dir: its last part)
// the client served: a short hash of the modules that decide what is captured and sent, so runs
// of different builds can be told apart
const client = await Promise.all(['src/main.js', 'src/looks.js', 'src/dream/dream.js', 'src/dream/live.js']
  .map((f) => fetch(new URL(f, base)).then((r) => r.text()))).then(async (ts) => {
  const h = new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(ts.join('\0'))));
  return Array.from(h.slice(0, 6), (b) => b.toString(16).padStart(2, '0')).join('');
}).catch(() => null);
const meta = { date: new Date().toISOString(), browser: browserName, device: device || null, size: device ? null : [W, H],
  speed: flicker ? 1.7 : speed, settle, params: extra, engine, client };
if (perfSeconds > 0) {
  meta.perf = await perfRun(perfSeconds);
  console.log(JSON.stringify(meta.perf));
} else if (strip) {
  meta.zones = {};
  for (const z of zones) { meta.zones[z] = await stripZone(z); console.log(`${z}: strip of ${meta.zones[z].shots.length}`); }
} else if (flicker) {
  meta.flicker = {};
  for (const z of zones) {
    meta.flicker[z] = await flickerZone(z);
    const r = meta.flicker[z];
    console.log(`${z}: still mean ${r.still.mean} p90 ${r.still.p90} max ${r.still.max} | walking mean ${r.walking.mean} p90 ${r.walking.p90} max ${r.walking.max} | warped still ${r.warped.still.mean} walking ${r.warped.walking.mean} (valid ${r.warped.validWalking}, dropped ${r.warped.droppedStill}/${r.warped.droppedWalking}) | detail ${r.detail.still}/${r.detail.walking} | sil/tex still ${r.edges.silStill}/${r.edges.texStill} | dreams/s ${r.dreamFps.still}/${r.dreamFps.walking}`);
  }
} else {
  meta.zones = {};
  for (const z of zones) {
    meta.zones[z] = await shootZone(z);
    const r = meta.zones[z];
    console.log(`${z}: ${r.shots.length} shots, ${r.perf.fps.toFixed(0)} fps, dream ${r.perf.dream.dreamFps.toFixed(1)}/s, capture ${r.capture.size.join('x')} @ ${r.capture.fov?.toFixed(0)}°${r.errors.length ? `, ${r.errors.length} console errors` : ''}`);
  }
}
// the engine's running share of cheap DeepCache passes, as the run left it (null if not reported)
meta.dcReuseAfter = await fetch(new URL('api/info', base)).then((r) => r.json()).then((i) => i.dc_reuse ?? null).catch(() => null);
fs.writeFileSync(path.join(out, 'shots.json'), JSON.stringify(meta, null, 1));
await browser.close();
