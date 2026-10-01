// Close-your-eyes experiment: at a zone waypoint, settle, hold E for --hold s, release, and
// film the reopening. Writes <zone>_<phase>.png and a strip per zone, plus eyes.json with the
// mean |luma change| between +2 s and +4 s after opening (rest calm) and before vs +4 s (how
// different the new dream is).
import fs from 'node:fs'; import path from 'node:path';
const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? argv[i + 1] : d; };
const base = opt('url'), out = opt('out'), extra = opt('params', ''), hold = +opt('hold', 5);
const zones = opt('zones', 'nave,baths,garden,desert').split(',');
const ZONES = { vestibule: 0, nave: 3, stacks: 6, geode: 9, baths: 15, garden: 18, atrium: 24, desert: 30 };
const pw = await import(process.env.PLAYWRIGHT);
const browser = await pw.webkit.launch({ headless: true });
fs.mkdirSync(out, { recursive: true });
const res = {};
for (const zone of zones) {
  const page = await browser.newPage({ viewport: { width: 960, height: 540 } });
  const logs = []; page.on('pageerror', (e) => logs.push(e.message));
  await page.addInitScript(() => { let last = 0; window.requestAnimationFrame = (cb) => setTimeout(() => { last = performance.now(); cb(last); }, Math.max(0, 1000 / 60 - (performance.now() - last))); });
  const w = new URL('/ws', base); w.protocol = w.protocol === 'https:' ? 'wss:' : 'ws:';   // named: never the page's :8765 fallback
  const u = new URL(base); u.search = `title=0&hud=0&adaptive=0&persist=0&attract=0&server=${encodeURIComponent(w.href)}&${extra}`;
  await page.goto(u.href);
  await page.waitForFunction(() => window.__hyp?.level?.tour, null, { timeout: 60000 });
  await page.evaluate((k) => { const h = window.__hyp, w = h.level.tour[k]; h.player.setPose([w[0], w[1], w[2]], w[3], w[4]); h.player.frozen = true; h.painter.clear(); h.post.compMat.uniforms.uGrain.value = 0; }, ZONES[zone]);
  await page.waitForFunction(() => window.__hyp.dream.stats.results >= 1, null, { timeout: 90000 });
  await page.waitForTimeout(8000);
  const shot = async (name) => { const s = await page.evaluate(() => window.__hyp.snapAt(0, 'scene')); fs.writeFileSync(path.join(out, `${zone}_${name}.png`), Buffer.from(s.png.split(',')[1], 'base64')); return s.png; };
  const frames = { before: await shot('before') };
  await page.keyboard.down('KeyE');
  await page.waitForTimeout(hold * 1000 - 500);
  frames.closed = await shot('closed');
  await page.waitForTimeout(500);
  await page.keyboard.up('KeyE');
  const t0 = Date.now();
  for (const t of [0.25, 1, 2, 4]) { await page.waitForTimeout(Math.max(0, t0 + t * 1000 - Date.now())); frames['open' + t] = await shot('open' + t); }
  // a second blink: the next variant
  await page.keyboard.down('KeyE'); await page.waitForTimeout(hold * 1000); await page.keyboard.up('KeyE');
  await page.waitForTimeout(4000);
  frames.second = await shot('second');
  const stats = await page.evaluate(async (fr) => {
    const c = document.createElement('canvas'), g = c.getContext('2d', { willReadFrequently: true });
    const luma = async (src) => { const im = new Image(); im.src = src; await im.decode(); c.width = 240; c.height = 135; g.drawImage(im, 0, 0, 240, 135); const d = g.getImageData(0, 0, 240, 135).data; const y = new Float32Array(240 * 135); for (let i = 0; i < y.length; i++) y[i] = 0.2126 * d[4 * i] + 0.7152 * d[4 * i + 1] + 0.0722 * d[4 * i + 2]; return y; };
    const diff = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]); return s / a.length; };
    const L = {}; for (const k of Object.keys(fr)) L[k] = await luma(fr[k]);
    return { newDream: diff(L.before, L.open4), settle24: diff(L.open2, L.open4), settle12: diff(L.open1, L.open2), closedLuma: L.closed.reduce((a, b) => a + b, 0) / L.closed.length };
  }, frames);
  res[zone] = { ...stats, errors: logs };
  console.log(zone, JSON.stringify(stats));
  await page.close();
}
fs.writeFileSync(path.join(out, 'eyes.json'), JSON.stringify({ date: new Date().toISOString(), hold, params: extra, zones: res }, null, 1));
await browser.close();
