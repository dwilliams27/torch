#!/usr/bin/env node
// Training pairs for a one-pass LoRA (M116): what the model is shown while you walk, and what
// the same view becomes when you stand there. Needs a running server and Playwright.
//
//   PLAYWRIGHT=/path/to/node_modules/playwright/index.mjs \
//   node tools/harvest_pairs.mjs --url http://127.0.0.1:PORT/ --out DIR
//        [--every 4] [--hold 5] [--speed 3.4] [--from 0] [--to LAST] [--phase both|walk|hold]
//        [--min-passes 15] [--stride 8] [--allow-lora] [--params 'look=fresh&...']
//
// Walk: the autopilot walks the tour from waypoint --from to --to, by default at the player's
// walking speed (3.4 m/s; the attract tour's 1.7 gives captures less feedback). After its
// first 3 s, every --every-th capture of a new framing (a centre view, not a glance) is kept as
// the game sent it: its JPEG (feedback from the walk included), its header (depth, prompt,
// seed, strength), the capture camera, the feedback and motion at the time, and the result
// that came back (what the walk showed).
// Hold: the player stands at each kept capture's camera for --hold seconds, as a player who
// stopped there would. The live views are cleared at each stop (and again as counting starts,
// once the last stop's results are in) and stops go in a strided order, so one stop's
// converged picture doesn't seed the next; the atlas still keeps what earlier stops painted,
// and a `--phase hold` resume starts from a blank one. The newest result on the capture's
// stream (its seed) is the target, the view
// converged at rest; the capture that result answered is kept too (at the loop's fixed point
// a LoRA should change nothing, so that is a pair as well). Dropped: a hold whose framing
// drifted from the walk's (more than 2 cm or 0.3 degrees), or that got under --min-passes
// results.
//
// Writes DIR/info.json (the server's engine settings; a server with a LoRA needs --allow-lora,
// and a resume refuses a server whose settings differ),
// DIR/NNNN/{input.jpg, walk.jpg, meta.json}, then rest.jpg and meta.hold, and target.jpg
// last (it marks the stop done; empty when dropped). Standing still paints the world, so
// the holds run after the whole walk. `--phase hold` resumes: kept captures without a target.
import fs from 'node:fs';
import path from 'node:path';

const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? argv[i + 1] : d; };
const base = opt('url'), out = opt('out');
if (!base || !out) { console.error('usage: node tools/harvest_pairs.mjs --url URL --out DIR [...]'); process.exit(2); }
const every = +opt('every', 4), hold = +opt('hold', 5), speed = +opt('speed', 3.4), from = +opt('from', 0);
const phase = opt('phase', 'both'), minPasses = +opt('min-passes', 15), stride = +opt('stride', 8), extra = opt('params', '');
const KEYS = ['engine', 'model', 'width', 'height', 'xframe', 'depth', 'depth_graft', 'lora', 'held_only', 'device'];
const info = Object.fromEntries(Object.entries(await (await fetch(new URL('/api/info', base))).json()).filter(([k]) => KEYS.includes(k))
  .map(([k, v]) => [k, k === 'model' && typeof v === 'string' ? v.split('/').slice(-2).join('/') : v]));   // (a local model dir: its last part)
const infoPath = path.join(out, 'info.json');
if (phase === 'hold' && fs.existsSync(infoPath)) {
  const was = JSON.parse(fs.readFileSync(infoPath, 'utf8'));
  const diff = KEYS.filter((k) => JSON.stringify(was[k] ?? null) !== JSON.stringify(info[k] ?? null));
  if (diff.length) { console.error(`this server differs from the walk's in ${diff.join(', ')}: its targets wouldn't match`); process.exit(2); }
}
if (info.lora && !argv.includes('--allow-lora')) { console.error(`the server merges a LoRA (${info.lora}); pass --allow-lora to harvest from it`); process.exit(2); }
if (phase !== 'hold' && fs.existsSync(out) && fs.readdirSync(out).some((d) => /^\d{4}$/.test(d))) {
  console.error(`${out} already has captures: a new walk would renumber over them (use a new --out, or --phase hold)`); process.exit(2);
}
const pw = await import(process.env.PLAYWRIGHT || 'playwright');
const browser = await pw.webkit.launch({ headless: true });
fs.mkdirSync(out, { recursive: true });
if (phase !== 'hold') fs.writeFileSync(infoPath, JSON.stringify({ ...info, speed, every, hold, params: extra, date: new Date().toISOString() }));
const dirOf = (n) => path.join(out, String(n).padStart(4, '0'));
const put = (n, name, b64) => { fs.mkdirSync(dirOf(n), { recursive: true }); fs.writeFileSync(path.join(dirOf(n), name), Buffer.from(b64, 'base64')); };
const log = (...a) => console.log(new Date().toTimeString().slice(0, 8), ...a);

const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
await page.addInitScript(() => {   // headless WebKit: a 60 Hz frame clock
  let last = 0;
  window.requestAnimationFrame = (cb) => setTimeout(() => { last = performance.now(); cb(last); }, Math.max(0, 1000 / 60 - (performance.now() - last)));
  window.cancelAnimationFrame = (id) => clearTimeout(id);
});
await page.exposeFunction('__keep', (n, rec) => {
  put(n, 'input.jpg', rec.jpeg);
  const { prompt, negative, seed, strength, depth, width, height, id } = rec.header;
  fs.writeFileSync(path.join(dirOf(n), 'meta.json'), JSON.stringify({ prompt, negative, seed, strength, depth, width, height, id, zone: rec.zone, cam: rec.cam,
    feedback: rec.feedback, motion: rec.motion, tour: rec.tour, speed }));
});
await page.exposeFunction('__walkOut', (n, b64) => put(n, 'walk.jpg', b64));
const wsUrl = (() => { const w = new URL('/ws', base); w.protocol = w.protocol === 'https:' ? 'wss:' : 'ws:'; return encodeURIComponent(w.href); })();   // never the page's :8765 fallback
const u = new URL(base); u.search = `title=0&hud=0&adaptive=0&persist=0&attract=0&idle=0&server=${wsUrl}${extra ? '&' + extra : ''}`;   // idle: no gaze drift during long holds
await page.goto(u.href);
await page.waitForFunction(() => window.__hyp?.level?.tour && window.__hyp.dream.link, null, { timeout: 60000 });
const to = +opt('to', await page.evaluate(() => window.__hyp.level.tour.length - 1));
await page.evaluate(() => {
  const h = window.__hyp, d = h.dream, link = d.link, send = link.send.bind(link);
  const b64 = (u8) => { let s = ''; for (let i = 0; i < u8.length; i += 8192) s += String.fromCharCode(...u8.subarray(i, i + 8192)); return btoa(s); };
  const H = window.__h = { mode: 'off', every: 1, fresh: 0, n: 0, lastKf: null, walkIds: new Map(), seedOf: new Map(), hold: null, skipUntil: 0 };
  link.send = (hdr, payload) => {
    if (hdr.type === 'frame') {
      H.seedOf.set(hdr.id, hdr.seed);
      const s = d.slots.find((x) => x.id === hdr.id);
      const c = s?.camera;
      const cam = c && { pos: c.position.toArray(), yaw: c.rotation.y, pitch: c.rotation.x, fov: c.fov };
      if (s && s.side === 0 && !s.narrow) {
        const fresh = s.kfId !== H.lastKf;
        H.lastKf = s.kfId;
        if (H.mode === 'walk' && fresh && hdr.depth && performance.now() > H.skipUntil && H.fresh++ % H.every === 0) {
          const n = H.n++;
          H.walkIds.set(hdr.id, n);
          const A = h.player.autopilot;   // metres along the tour (what two harvests of it share)
          const tour = A ? A.segLen.slice(0, A.seg).reduce((x, y) => x + y, 0) + A.u * A.segLen[A.seg] : null;
          window.__keep(n, { header: { ...hdr }, jpeg: b64(payload), cam, zone: h.level.zones[s.zone]?.id,
            feedback: h.materials?.capture?.uniforms?.uFeedback?.value ?? null, motion: d.motion, tour });
        }
      }
      const K = H.hold;
      if (H.mode === 'hold' && K && hdr.seed === K.seed && hdr.id >= K.since) K.caps.set(hdr.id, { header: { ...hdr }, jpeg: b64(payload), cam });
    }
    return send(hdr, payload);
  };
  link.on('result', (res) => {
    const id = res.header.id, K = H.hold;
    if (!res.bytes) return;
    if (H.walkIds.has(id)) { window.__walkOut(H.walkIds.get(id), b64(res.bytes)); H.walkIds.delete(id); }
    if (H.mode === 'hold' && K && H.seedOf.get(id) === K.seed && id >= K.since) K.results.set(id, b64(res.bytes));
  });
});

const stand = async (pos, yaw, pitch) => page.evaluate(({ pos, yaw, pitch }) => {
  const h = window.__hyp, p = h.player, d = h.dream;
  if (p.autopilot) p.stopAutopilot();
  p.setPose(pos, yaw, pitch);
  p.frozen = true;
  p.bobAmp = 0; p.dip = 0; p.lean = 0;
  // a teleport is not motion: no lead from a velocity spike, and the next capture starts a framing
  d._motion = null;
  if (d._kf) d._kf.pos.set(1e9, 1e9, 1e9);
  d.live?.clear();   // start from the atlas, not the last stop's converged views
}, { pos, yaw, pitch });

if (phase !== 'hold') {
  const w = await page.evaluate((k) => window.__hyp.level.tour[k], from);
  await stand([w[0], w[1], w[2]], w[3], w[4]);
  await page.waitForFunction(() => window.__hyp.dream.stats.results >= 1, null, { timeout: 90000 });
  await page.waitForTimeout(5000);
  await page.evaluate(({ k, speed, every }) => {
    const p = window.__hyp.player, H = window.__h;
    H.every = every; H.mode = 'walk'; H.maxSeg = k; H.skipUntil = performance.now() + 3000;   // up to walking pace first
    p.frozen = false; p.startAutopilot(k); p.autopilot.speed = speed;
  }, { k: from, speed, every });
  const t0 = Date.now(), r0 = await page.evaluate(() => window.__hyp.dream.stats.results);
  // until the tour reaches --to (or wraps back to the start)
  await page.waitForFunction((to) => {
    const A = window.__hyp.player.autopilot, H = window.__h;
    if (!A) return true;
    if (A.seg < H.maxSeg) return true;
    H.maxSeg = A.seg;
    return A.seg >= to;
  }, to, { timeout: 0, polling: 100 });
  const [n, r1] = await page.evaluate(() => { window.__h.mode = 'off'; return [window.__h.n, window.__hyp.dream.stats.results]; });
  const secs = (Date.now() - t0) / 1000;
  fs.writeFileSync(path.join(out, 'walk.json'), JSON.stringify({ seconds: secs, kept: n, results: r1 - r0, dreamsPerSecond: (r1 - r0) / secs }));
  await page.waitForTimeout(1500);   // the last results
  log(`walk: ${n} captures kept in ${secs.toFixed(0)} s, ${((r1 - r0) / secs).toFixed(1)} dreams/s`);
}

if (phase !== 'walk') {
  const left = fs.readdirSync(out).filter((d) => /^\d{4}$/.test(d) && fs.existsSync(path.join(out, d, 'meta.json'))
    && fs.existsSync(path.join(out, d, 'walk.jpg')) && !fs.existsSync(path.join(out, d, 'target.jpg'))).sort();
  const todo = [], pass = {};   // strided: consecutive stops far apart along the tour
  const S = Math.max(1, stride);
  for (let r = 0; r < S; r++) for (let i = r; i < left.length; i += S) { todo.push(left[i]); pass[left[i]] = r; }
  if (phase === 'hold') await page.waitForFunction(() => window.__hyp.dream.stats.results >= 1, null, { timeout: 90000 });
  log(`hold: ${todo.length} framings`);
  let kept = 0, dropped = 0;
  for (const name of todo) {
    const metaPath = path.join(out, name, 'meta.json');
    const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
    await stand(meta.cam.pos, meta.cam.yaw, meta.cam.pitch);
    // let captures already in flight from the last pose come back, then count from here
    await page.waitForTimeout(500);
    await page.evaluate((seed) => {
      const H = window.__h, d = window.__hyp.dream;
      d.live?.clear();   // the last stop's late results are in by now: drop them too
      H.mode = 'hold'; H.hold = { seed, since: d.nextId, caps: new Map(), results: new Map() };
    }, meta.seed);
    await page.waitForTimeout(hold * 1000);
    const rec = await page.evaluate(() => {
      const H = window.__h, K = H.hold;
      H.mode = 'off'; H.hold = null;
      const ids = [...K.results.keys()].sort((a, b) => a - b);
      const id = ids[ids.length - 1];
      if (id == null || !K.caps.has(id)) return { passes: ids.length };
      return { passes: ids.length, id, target: K.results.get(id), rest: K.caps.get(id) };
    });
    const c = rec.rest?.cam, m = meta.cam;
    const drift = c ? { pos: Math.hypot(c.pos[0] - m.pos[0], c.pos[1] - m.pos[1], c.pos[2] - m.pos[2]),
      ang: Math.max(Math.abs(Math.atan2(Math.sin(c.yaw - m.yaw), Math.cos(c.yaw - m.yaw))), Math.abs(c.pitch - m.pitch)) * 180 / Math.PI,
      fov: Math.abs(c.fov - m.fov) } : null;
    if (!rec.target || !drift || drift.pos > 0.02 || drift.ang > 0.3 || drift.fov > 0.01 || !rec.rest.header.depth || rec.passes < minPasses) {
      dropped++;
      meta.hold = { dropped: true, passes: rec.passes, drift, pass: pass[name], resumed: phase === 'hold' };
      fs.writeFileSync(metaPath, JSON.stringify(meta));
      fs.writeFileSync(path.join(out, name, 'target.jpg'), '');   // marks it done (empty: no pair)
      continue;
    }
    put(+name, 'rest.jpg', rec.rest.jpeg);
    const r = rec.rest.header;
    meta.hold = { passes: rec.passes, id: rec.id, seconds: hold, drift, pass: pass[name], resumed: phase === 'hold',
      rest: { strength: r.strength, depth: r.depth, prompt: r.prompt, seed: r.seed } };
    fs.writeFileSync(metaPath, JSON.stringify(meta));
    put(+name, 'target.jpg', rec.target);   // last: it marks the stop done
    kept++;
    if ((kept + dropped) % 25 === 0) log(`hold: ${kept + dropped}/${todo.length} (${dropped} dropped)`);
  }
  log(`hold: ${kept} pairs, ${dropped} dropped`);
}
await browser.close();
