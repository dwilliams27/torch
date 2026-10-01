#!/usr/bin/env node
// Framing probe (M112): standing still for 5 s, then walking the tour at 1.7 m/s (as shoot.mjs --flicker
// walks) for 5 s, what share of frames repeat their stream's (seed's) last framing id (header fid)?
// Those are the frames a pool with --pool-held carry keeps on the GPU. WebKit, one page per room.
//
//   PLAYWRIGHT=/path/to/node_modules/playwright/index.mjs node tools/framing_probe.mjs [PORT] [PARAMS]
const pw = await import(process.env.PLAYWRIGHT || 'playwright');
const port = process.argv[2] || '8785', params = process.argv[3] || 'inflight=3';
const base = `http://127.0.0.1:${port}/`, ws = encodeURIComponent(`ws://127.0.0.1:${port}/ws`);
const ZONES = { vestibule: 0, nave: 3, stacks: 6, geode: 9, baths: 15, garden: 18, atrium: 24, desert: 30 };   // shoot.mjs's
const browser = await pw.webkit.launch({ headless: true });
const res = {};
for (const [name, k] of Object.entries(ZONES)) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  await page.addInitScript(() => { let last = 0; window.requestAnimationFrame = (cb) => setTimeout(() => { last = performance.now(); cb(last); }, Math.max(0, 1000 / 60 - (performance.now() - last))); window.cancelAnimationFrame = (id) => clearTimeout(id); });
  await page.goto(`${base}?title=0&hud=0&adaptive=0&server=${ws}&persist=0&${params}`);
  await page.waitForFunction(() => window.__hyp?.level?.tour, null, { timeout: 60000 });
  await page.evaluate((k) => { const h = window.__hyp, w = h.level.tour[k]; h.player.setPose([w[0], w[1], w[2]], w[3], w[4]); h.player.frozen = true; h.painter.clear(); }, k);
  await page.waitForFunction(() => window.__hyp.dream.stats.results >= 1, null, { timeout: 90000 });
  await page.waitForTimeout(3000);
  const probe = async (walk) => page.evaluate(async ({ k, walk }) => {
    const link = window.__hyp.dream.link, send = link.send.bind(link), last = new Map();
    let n = 0, rep = 0;
    link.send = (h, p) => { if (h.type === 'frame') { n++; if (last.get(h.seed) === h.fid) rep++; last.set(h.seed, h.fid); } return send(h, p); };
    if (walk) { const p = window.__hyp.player; p.startAutopilot(k); p.autopilot.speed = 1.7; }
    await new Promise((r) => setTimeout(r, 5000));
    link.send = send;
    return { n, rep };
  }, { k, walk });
  res[name] = { still: await probe(false), walking: await probe(true) };
  console.log(name, JSON.stringify(res[name]));
  await page.close();
}
await browser.close();
const tot = (w) => Object.values(res).reduce((a, r) => [a[0] + r[w].rep, a[1] + r[w].n], [0, 0]);
for (const w of ['still', 'walking']) { const [r, n] = tot(w); console.log(`${w}: ${r}/${n} frames repeat their stream's framing (${(100 * r / n).toFixed(0)}%)`); }
console.log(JSON.stringify({ date: new Date().toISOString(), params, rooms: res }));
