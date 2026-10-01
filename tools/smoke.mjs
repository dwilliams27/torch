#!/usr/bin/env node
// Smoke test against a running server, in Chromium and WebKit (Safari's engine), desktop
// and an emulated iPhone. Needs Playwright with both browsers installed.
//
//   PLAYWRIGHT=/path/to/node_modules/playwright/index.mjs \
//   node tools/smoke.mjs --url http://127.0.0.1:PORT/ [--out DIR] [--browsers chromium,webkit]
//
// Per desktop browser: boots without page or console errors; dreams (results arrive, the
// frame isn't blank); a keyboard walk moves the player and plays footsteps once the
// entering click has started the sound; every capture carries depth when the engine asks
// for it (/api/info depth); keys 1-6 switch looks and the panel offers all six;
// every look sends framing ids (header `fid`), only the fresh look as `kf` too; the dream
// survives a reload (IndexedDB tiles are restored with dreaming off) and "forget" drops them. On the emulated iPhone (WebKit,
// touch): the title says "tap", a tap enters and starts the sound, a left-thumb drag walks
// and a right-thumb drag looks. Prints one PASS/FAIL line per check; exit code 1 on any FAIL.
// The sound is muted before it starts (volume 0), so a run never plays out of the host's
// speakers; the checks look at the audio context's state instead. The emulated iPhone is
// WebKit with touch, not iOS: it does not enforce iOS's audio-unlock rules.
//
// Headless browsers on a Mac without a display session throttle requestAnimationFrame, so
// pages get a 60 Hz setTimeout-paced rAF (as in tools/shoot.mjs).
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? argv[i + 1] : d; };
const base = opt('url');
if (!base) { console.error('usage: node tools/smoke.mjs --url http://127.0.0.1:PORT/ [--out DIR] [--browsers ...]'); process.exit(2); }
const out = opt('out', path.join(os.tmpdir(), 'hypnagogia-smoke'));
const browsers = opt('browsers', 'chromium,webkit').split(',');
const pw = await import(process.env.PLAYWRIGHT || 'playwright');
fs.mkdirSync(out, { recursive: true });

const results = [];
function check(name, ok, detail = '') {
  results.push({ name, ok: !!ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
}
const wsUrl = (() => { const w = new URL('/ws', base); w.protocol = w.protocol === 'https:' ? 'wss:' : 'ws:'; return encodeURIComponent(w.href); })();   // never the page's :8765 fallback
const url = (q) => { const u = new URL(base); u.search = `server=${wsUrl}&${q}`; return u.href; };
// mute before the entering gesture starts the sound (the master gain then ramps to 0)
const mute = (page) => page.waitForFunction(() => { const a = window.__hyp?.audio?.().ambience; if (a) a.setVolume(0); return !!a; }, null, { timeout: 60000 });
const audioState = (page) => page.evaluate(() => { const a = window.__hyp.audio(); return { started: a.started, context: a.ambience?.state?.context ?? 'none' }; });
const shim = () => {
  let last = 0;
  window.requestAnimationFrame = (cb) => {
    const wait = Math.max(0, 1000 / 60 - (performance.now() - last));
    return setTimeout(() => { last = performance.now(); cb(last); }, wait);
  };
  window.cancelAnimationFrame = (id) => clearTimeout(id);
};
async function launch(name) {
  const args = name === 'chromium'
    ? ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--disable-gpu-vsync', '--disable-frame-rate-limit']
    : [];
  return pw[name].launch({ headless: true, args, chromiumSandbox: true });
}
function watch(page) {
  const errors = [];
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error' || (m.type() === 'warning' && /hypnagogia|dream|memory/.test(m.text()))) errors.push(`${m.type()}: ${m.text()}`); });
  return errors;
}
async function frameStats(page, file) {
  const s = await page.evaluate(() => window.__hyp.snapAt(0, 'scene'));
  fs.writeFileSync(file, Buffer.from(s.png.split(',')[1], 'base64'));
  // luma mean / spread of the frame, measured in the page on a small copy
  return page.evaluate(async (png) => {
    const img = new Image(); img.src = png; await img.decode();
    const c = document.createElement('canvas'); c.width = 160; c.height = 90;
    const g = c.getContext('2d'); g.drawImage(img, 0, 0, 160, 90);
    const d = g.getImageData(0, 0, 160, 90).data;
    let s = 0, s2 = 0; const n = d.length / 4;
    for (let i = 0; i < d.length; i += 4) { const y = 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2]; s += y; s2 += y * y; }
    const m = s / n; return { mean: m, std: Math.sqrt(Math.max(0, s2 / n - m * m)) };
  }, s.png);
}

async function desktop(name) {
  const browser = await launch(name);
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  await ctx.addInitScript(shim);
  const page = await ctx.newPage();
  const errors = watch(page);
  const pose = 'pos=0,-2.2,-24&yaw=0&pitch=0';
  await page.goto(url(`hud=0&adaptive=0&${pose}`));
  await page.waitForFunction(() => window.__hyp?.level, null, { timeout: 60000 });
  await mute(page);
  await page.mouse.click(640, 360);                       // the entering click (a real gesture)
  const t0 = Date.now();
  const dreamt = await page.waitForFunction(() => window.__hyp.dream.stats.results >= 3, null, { timeout: 60000 }).then(() => true, () => false);
  check(`${name}: dreams`, dreamt, dreamt ? `${((Date.now() - t0) / 1000).toFixed(1)} s to 3 results` : 'no results in 60 s');
  await page.waitForTimeout(6000);
  const fs1 = await frameStats(page, path.join(out, `${name}_desktop.png`));
  check(`${name}: frame is painted, not blank`, fs1.mean > 12 && fs1.std > 6, `luma mean ${fs1.mean.toFixed(0)}, spread ${fs1.std.toFixed(0)}`);
  const audio = await audioState(page);
  check(`${name}: entering click starts the sound`, audio.started && audio.context === 'running', `context ${audio.context}`);

  // keyboard walk: position changes, footsteps play
  await page.evaluate(() => {
    const a = window.__hyp.audio().ambience;
    window.__steps = 0;
    if (a) { const f = a.footstep.bind(a); a.footstep = (i) => { window.__steps++; return f(i); }; }
  });
  const p0 = await page.evaluate(() => window.__hyp.camera.position.toArray());
  await page.keyboard.down('KeyW'); await page.waitForTimeout(2500); await page.keyboard.up('KeyW');
  const p1 = await page.evaluate(() => window.__hyp.camera.position.toArray());
  const moved = Math.hypot(p1[0] - p0[0], p1[2] - p0[2]);
  const steps = await page.evaluate(() => window.__steps);
  check(`${name}: W walks`, moved > 3, `${moved.toFixed(1)} m in 2.5 s`);
  check(`${name}: footsteps`, steps >= 3, `${steps} steps`);

  // an engine that paints with depth gets it with every capture
  const dep = await page.evaluate(async () => {
    const d = window.__hyp.dream, link = d.link, send = link.send.bind(link);
    let n = 0, withDepth = 0, bad = 0;
    link.send = (h, p) => {
      n++;
      if (h.depth) { withDepth++; if (atob(h.depth.data).length !== h.depth.w * h.depth.h) bad++; }
      return send(h, p);
    };
    await new Promise((r) => setTimeout(r, 1500));
    link.send = send;
    return { wants: !!d.info?.depth, n, withDepth, bad };
  });
  check(`${name}: captures carry depth when the engine asks`, dep.n > 0 && dep.bad === 0 && (dep.wants ? dep.withDepth === dep.n : dep.withDepth === 0), JSON.stringify(dep));

  // looks: a key switches the whole set and the panel offers all of them
  await page.keyboard.press('Digit2');
  const lk = await page.evaluate(() => {
    const S = window.__hyp.settings;
    return { look: S.look, tau: S.liveTau, rate: S.captureRate, buttons: document.querySelectorAll('#panel .row.looks button').length };
  });
  await page.keyboard.press('Digit1');
  const back = await page.evaluate(() => ({ look: window.__hyp.settings.look, tau: window.__hyp.settings.liveTau }));
  // framing ids: `fid` in every look, `kf` (the same id) only in the fresh look (6)
  const kfs = async () => page.evaluate(async () => {
    const link = window.__hyp.dream.link, send = link.send.bind(link);
    let n = 0, kf = 0, fid = 0;
    const ids = new Set();
    link.send = (h, p) => { if (h.type === 'frame') { n++; if (Number.isInteger(h.fid)) fid++; if (Number.isInteger(h.kf) && h.kf === h.fid) { kf++; ids.add(h.kf); } } return send(h, p); };
    await new Promise((r) => setTimeout(r, 1200));
    link.send = send;
    return { n, kf, fid, distinct: ids.size };
  });
  const kfWaking = await kfs();
  await page.keyboard.press('Digit6');
  const kfFresh = { look: await page.evaluate(() => window.__hyp.settings.look), ...(await kfs()) };
  await page.keyboard.press('Digit1');
  // closed eyes: once the lid is shut the capture re-dreams (feedback 0.1, fast turnover, a
  // variant in front of the prompt); opening restores it, and losing focus opens held eyes
  const ey = await page.evaluate(async () => {
    const h = window.__hyp, U = h.materials.capture.uniforms, wait = (ms) => new Promise((r) => setTimeout(r, ms));
    h.eyes(true); await wait(900);
    const shut = { closed: h.settings.eyesClosed, fb: U.uFeedback.value, tau: h.dream.live.tau, lid: h.post.compMat.uniforms.uLid.value, variant: h.settings.promptSuffix };
    h.eyes(false); await wait(1200);
    const open = { closed: h.settings.eyesClosed, lid: h.post.compMat.uniforms.uLid.value, tau: h.dream.live.tau, variant: h.settings.promptSuffix };
    return { shut, open };
  });
  await page.keyboard.down('KeyE'); await page.waitForTimeout(400);
  await page.evaluate(() => window.dispatchEvent(new Event('blur')));
  await page.waitForTimeout(900);
  const afterBlur = await page.evaluate(() => window.__hyp.post.compMat.uniforms.uLid.value);
  await page.keyboard.up('KeyE');
  check(`${name}: closing the eyes re-dreams and opening restores`, ey.shut.closed && ey.shut.fb <= 0.1 + 1e-6 && ey.shut.tau <= 0.12 && ey.shut.lid > 0.9
    && !!ey.shut.variant && !ey.open.closed && ey.open.lid < 0.1 && ey.open.tau > 0.12 && ey.open.variant === ey.shut.variant && afterBlur < 0.2,
    `${JSON.stringify(ey)} lid after blur ${afterBlur.toFixed(2)}`);
  check(`${name}: looks switch with keys 1-6`, lk.look === 'drifting' && lk.tau === 0.8 && lk.rate === 8 && lk.buttons === 6 && back.look === 'waking' && back.tau !== 0.8,
    `${JSON.stringify(lk)} -> ${JSON.stringify(back)}`);
  // standing still, the ids must repeat (a new id every capture would make fresh a full pass every frame)
  check(`${name}: every look sends framing ids (fid), only the fresh look as kf`, kfWaking.n > 0 && kfWaking.fid === kfWaking.n && kfWaking.kf === 0
    && kfFresh.look === 'fresh' && kfFresh.n > 0 && kfFresh.fid === kfFresh.n && kfFresh.kf === kfFresh.n
    && kfFresh.distinct <= 4,
    `waking ${JSON.stringify(kfWaking)}, fresh ${JSON.stringify(kfFresh)}`);

  // the dream survives a reload, and forgets on request
  await page.evaluate(() => { window.__hyp.settings.paused = true; });
  await page.waitForTimeout(1500);
  await page.waitForFunction(() => { const m = window.__hyp.memory; if (!m) return true; m.flush(); return !m.busy && m.dirty.size === 0; }, null, { timeout: 60000, polling: 250 });
  const saved = await page.evaluate(() => window.__hyp.memory?.stats?.saved ?? 0);
  await page.goto(url(`title=0&hud=0&adaptive=0&dream=off&${pose}`));
  await page.waitForFunction(() => window.__hyp?.level, null, { timeout: 60000 });
  await page.waitForTimeout(1500);
  const restored = await page.evaluate(() => window.__hyp.memory?.stats?.restored ?? 0);
  const fs2 = await frameStats(page, path.join(out, `${name}_restored.png`));
  check(`${name}: the dream survives a reload`, saved > 0 && restored > 0 && fs2.std > 6, `${saved} tiles saved, ${restored} restored`);
  await page.evaluate(() => window.__hyp.dream.forget());
  await page.waitForTimeout(500);
  await page.reload();
  await page.waitForFunction(() => window.__hyp?.level, null, { timeout: 60000 });
  await page.waitForTimeout(800);
  const after = await page.evaluate(() => window.__hyp.memory?.stats?.restored ?? -1);
  check(`${name}: forget clears the stored dream`, after === 0, `${after} tiles after forget`);
  check(`${name}: no page or console errors`, errors.length === 0, errors.slice(0, 3).join(' | '));
  await browser.close();
}

async function iphone() {
  const browser = await launch('webkit');
  const ctx = await browser.newContext({ ...pw.devices['iPhone 15'] });
  await ctx.addInitScript(shim);
  const page = await ctx.newPage();
  const errors = watch(page);
  await page.goto(url('hud=0'));
  await page.waitForFunction(() => window.__hyp?.level, null, { timeout: 60000 });
  await mute(page);
  const label = await page.textContent('#title .enter');
  check('iphone: title says tap', /tap/.test(label || ''), `"${label}"`);
  const size = page.viewportSize();
  await page.touchscreen.tap(size.width / 2, size.height / 2);
  await page.waitForTimeout(600);
  const st = { entered: await page.evaluate(() => document.body.classList.contains('entered')), ...(await audioState(page)) };
  check('iphone: tap enters and starts the sound', st.entered && st.started && st.context === 'running', JSON.stringify(st));
  await page.waitForFunction(() => window.__hyp.dream.stats.results >= 2, null, { timeout: 60000 }).catch(() => {});
  // synthetic touches on the canvas: left half is the stick, right half looks
  const drag = (x0, y0, x1, y1, ms) => page.evaluate(async ({ x0, y0, x1, y1, ms }) => {
    const el = document.getElementById('view');
    const mk = (type, x, y) => {
      const t = { identifier: 7, clientX: x, clientY: y, pageX: x, pageY: y, target: el };
      const e = new Event(type, { bubbles: true, cancelable: true });
      Object.defineProperty(e, 'changedTouches', { value: [t] });
      Object.defineProperty(e, 'touches', { value: type === 'touchend' ? [] : [t] });
      el.dispatchEvent(e);
    };
    mk('touchstart', x0, y0);
    const n = Math.max(2, Math.round(ms / 16));
    for (let i = 1; i <= n; i++) { mk('touchmove', x0 + (x1 - x0) * Math.min(1, i / 4), y0 + (y1 - y0) * Math.min(1, i / 4)); await new Promise((r) => setTimeout(r, 16)); }
    mk('touchend', x1, y1);
  }, { x0, y0, x1, y1, ms });
  const p0 = await page.evaluate(() => ({ p: window.__hyp.camera.position.toArray(), yaw: window.__hyp.player.yaw }));
  await drag(size.width * 0.25, size.height * 0.7, size.width * 0.25, size.height * 0.7 - 60, 1500);
  const p1 = await page.evaluate(() => ({ p: window.__hyp.camera.position.toArray(), yaw: window.__hyp.player.yaw }));
  const walked = Math.hypot(p1.p[0] - p0.p[0], p1.p[2] - p0.p[2]);
  check('iphone: left thumb walks', walked > 1, `${walked.toFixed(1)} m`);
  await drag(size.width * 0.75, size.height * 0.5, size.width * 0.75 + 80, size.height * 0.5, 400);
  const p2 = await page.evaluate(() => ({ yaw: window.__hyp.player.yaw }));
  check('iphone: right thumb looks', Math.abs(p2.yaw - p1.yaw) > 0.1, `yaw ${(p2.yaw - p1.yaw).toFixed(2)} rad`);
  await page.waitForTimeout(3000);
  const fs1 = await frameStats(page, path.join(out, 'iphone.png'));
  check('iphone: frame is painted, not blank', fs1.mean > 12 && fs1.std > 6, `luma mean ${fs1.mean.toFixed(0)}, spread ${fs1.std.toFixed(0)}`);
  const info = await page.evaluate(() => {
    const h = window.__hyp, d = h.dream;
    return { atlas: h.painter.size, live: d.live.count, capture: [d.width, d.height], flexible: !!d.info?.flexible, res: h.perf().res };
  });
  const portrait = !info.flexible || info.capture[1] > info.capture[0];
  check('iphone: phone settings (2048 atlas, 3 live views, upright capture)', info.atlas === 2048 && info.live === 3 && portrait, JSON.stringify(info));
  check('iphone: no page or console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  await browser.close();
}

for (const b of browsers) await desktop(b);
if (browsers.includes('webkit')) await iphone();
fs.writeFileSync(path.join(out, 'smoke.json'), JSON.stringify({ date: new Date().toISOString(), url: base, results }, null, 1));
const failed = results.filter((r) => !r.ok).length;
console.log(`${results.length - failed}/${results.length} passed`);
process.exit(failed ? 1 : 0);
