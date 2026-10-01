# Steady dream: measurements

Evidence for keeping the dream's shapes while walking (2026-09-29). Server `torch_turbo`
(SD-Turbo on MPS, 384x384 budget) on a Mac mini (M4 Pro, 48 GB); pages in headless WebKit
at 1280x720 through Playwright.

**Warped change** (`tools/shoot.mjs --flicker`): frames 0.1 s apart are reduced to a 160x90
grid of exact area averages (grain and vignette off), then the previous frame is
reprojected onto the current one with the current depth and both cameras before the mean
|luma change| is taken. That cancels most detail that only moved with the camera; what
remains is paint that changed (re-invention, boiling), plus resampling at edges and
view-dependent light. `detail` (mean |luma gradient| of the full frame) is reported beside
it, because painting softer would also lower the change. Two runs of one build agree on
the 8-zone mean to about 1%, but single zones differ by up to ~17%.

| File | What |
|---|---|
| `brief.py` | builds the proof page `showcase/hypnagogia-steady-dream.html` from `data/` (and M111's `shots-after.json`, `perf-after-*.json`): `python3 docs/shots/m113/brief.py --out ../../showcase/hypnagogia-steady-dream.html` |
| `data/judge-final.json`, `final-strips.jpg` | the closing blind review: walking filmstrips of all 8 zones, M111's final build vs the build M113 closed with (frames 0, 2, 4, 6 of each strip in the image) |
| `data/shots-final.json`, `data/flicker-final.json`, `data/perf-final-1.json`, `-2.json` | the closing build: walk shots with the dream rate, and warped change and detail, 8 zones, one run each; frame time, Chromium 1920x1080, 2 runs |
| `data/flicker-baths-prompt-old-*.json`, `-new-*.json`, `baths-prompt.jpg`, `data/judge-baths.json` | the baths with the old prompt and with one that names the vaulted hall, terraced pool and tiled arcades (2 runs each; strip frames 0, 2, 4, 6, old above) |
| `data/flicker-nave-prompt-old-2.json`, `-new-1.json`, `-new-2.json` | the nave with a prompt without mist and light shafts (not adopted: walking detail +9%, change after motion compensation +7%, both runs each way; the shaft stays); run 1 of the old prompt is the nave in `flicker-final.json` |
| `data/flicker-livemix-0.25.json`, `data/flicker-trial-ease-0.5.json` | why standing change rose (0.66 before per-stream DeepCache, 0.85 in `flicker-final.json`): a smaller rest blend (`?livemix=0.25`) gave 0.89 and a slower hand-over to the oldest live view (0.5 s ease, temporary client copy) 1.16 |
| `data/flicker-rate-8.json` | captures capped at 8 a second (`?rate=8`, default 20), one run: standing change 0.58, walking change 3.04, walking detail 2.59, so the extra dreams look like the cause; dream rate not recorded; not adopted (M112) |
| `data/flicker-xframe-on-1.json`, `-on-2.json` | current client, cross-frame attention on (the default) |
| `data/flicker-xframe-off-1.json`, `-off-2.json` | same client, server `--engine-arg xframe=0` |
| `data/shots-dc-streams.json`, `data/flicker-dc-streams.json` | after DeepCache got one cache per stream: dream rate 10.6-12.1 per second (M111's final shots: 7.3-8.9), cheap passes 0.65 of frames (`dcReuseAfter`), warped change and detail |
| `data/judge-end-to-end.json` | blind filmstrip comparison of the whole build, M111 final vs now: steadier in all 8 zones, more coherent on average, less detailed in the nave, baths and desert |
| `data/flicker-xframe-bias-1.5.json` | anchor tokens weakened (`--engine-arg xframe_bias=-1.5`): walking change 3.24 vs 3.22, detail 2.40 vs 2.43, i.e. no measurable change in the game (the engine bench moves: 5.97 between 5.44 on and 6.51 off) |
| `data/judge-xframe.json` | blind filmstrip comparison, cross-frame attention on vs off: on is steadier (4.0 vs 3.25), a little less detailed (3.1 vs 3.5), much less in the baths; the reviewer prefers on |
| `data/judge-deepcache.json` | blind filmstrip comparison, DeepCache off vs per stream: per stream steadier in 6 of 8 zones, less unpainted geometry, detail about equal |
| `data/perf-1.json`, `data/perf-2.json` | frame time to `gl.finish()`, Chromium 1920x1080, 25 s (`--perf 25 --size 1920x1080`) |
| `data/judge.json` | a blind comparison of walking filmstrips, M111's final build vs now, by a reviewer not told which was which |
| `walk-strips-before-after.jpg` | frames 0.3 s apart while walking at 3.4 m/s, M111 vs now (`--strip`) |
| `vestibule-prompt.jpg`, `data/judge-vestibule.json` | vestibule walking, old prompt (top) vs the prompt that names the tunnel of rotated frames (bottom, adopted after a blind comparison) |
| `desert-prompt.jpg`, `data/judge-desert.json` | desert walking, old prompt (top) vs the prompt that names the staircase and tower (bottom, adopted after a blind comparison) |
| `calm-strips.jpg` | lowering strength while walking (`?calm`): steadier, but grey and flat; rejected by eye |
| `temporal-walk.jpg`, `temporal-turn.jpg` | engine alone on synthetic sequences (`bench/temporal.py --cfgs '{"xframe":0}' '{"xframe":1}'`): input row, then off, then on |

Not in the numbers: the occlusion test uses the nearest previous depth only, so silhouette
edges leak a little; walking pairs are 0.1 s of tour time and still pairs 0.1 s of page
time.
