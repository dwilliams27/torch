# Waking world: before and after

Evidence for the live layer, foveated and screen-shaped captures, dream memory and seam
fixes (2026-09-28/29). Everything here was produced by the tools in `tools/` against a
local server running `torch_turbo` (SD-Turbo on MPS, 384x384 budget) on a Mac mini (M4 Pro,
48 GB), with headless WebKit and Chromium through Playwright. "Before" is the client at
4abe4538 plus only the frame-exact snapshot hook; "after" is the client named in each
file's `note`.

| File | What | Made by |
|---|---|---|
| `walk-before-after.jpg` | 8 zones, 4 s into the same walk at 3.4 m/s | `tools/shoot.mjs --browser webkit`, `tools/sheet.py --phases walk2 --scale 0.3` |
| `nave-before-after.jpg` | the nave early, late, and 3 s after stopping | `tools/sheet.py --zones nave --phases walk1,walk3,rest --scale 0.5` |
| `zones-after.jpg`, `hero.jpg` | every zone after stopping, new client | the `rest` frames of `data/shots-after.json`'s run |
| `phone-after.jpg`, `data/shots-phone.json` | four zones at rest on an emulated iPhone (WebKit, upright) | `tools/shoot.mjs --browser webkit --device "iPhone 15"` |
| `data/shots-before.json`, `data/shots-after.json` | the two walk runs (WebKit, 1280x720) | `tools/shoot.mjs --browser webkit` |
| `data/flicker-before.json`, `data/flicker-after.json`, `data/flicker-after-fovea-off.json` | mean luma change between frames 0.1 s apart, standing and walking (WebKit, 1280x720) | `tools/shoot.mjs --browser webkit --flicker` (`--params fovea=1` for the last) |
| `data/perf-before-*.json`, `data/perf-after-*.json` | frame time to `gl.finish()`, Chromium 1920x1080, 25 s on the nave tour; the new runs also record when the five slowest frames came | `tools/shoot.mjs --perf 25 --size 1920x1080`, the old client served with `./run.sh --client-dir` |
| `data/smoke.json` | 23 checks: Chromium, WebKit, emulated iPhone | `tools/smoke.mjs` |
| `data/judge.json` | a blind comparison of the two walk runs by a reviewer who was not told which build was which, with the prompt it was given | a fresh-context model session |
| `data/*.metrics.json` | per-image contrast / detail / saturation for the two sheets | `tools/sheet.py` |
| `brief.py` | builds the one-page proof brief (HTML, inline SVG) from `data/` | `python3 docs/shots/m111/brief.py --out DIR/page.html` |

Caveats: a headless browser on a Mac without a display session throttles
requestAnimationFrame, so the tools pace pages at 60 Hz with setTimeout. The browser and
the model share the machine's GPU, so frames delivered per second are low and per-frame
cost is the comparison that means something. Headless Chromium also spends about 200 ms
reading back and encoding each capture (WebKit about 14 ms), so walks and stability were
measured in WebKit, at about 7.5 dreams a second. The walking stability number rises when
the picture is sharper (detail moving across the screen counts as change). Real Safari, a
real iPhone, and a laptop client were not measured.
