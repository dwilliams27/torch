# Diffusion engine bench — `torch_turbo` (PyTorch / MPS)

All numbers are **end-to-end `Engine.process()`** (numpy uint8 in → numpy uint8 out: upload,
encode, noise, UNet, decode, GPU-side quantize, readback) measured by
`bench/bench_engines.py` over a 24-frame walking-camera sequence (`bench/scenes.py`), under the
shared `bench.lock`, fp16, torch 2.14.0 / diffusers 0.40.0, 2026-09-28. Stage columns come from
a second pass with a device sync after every stage (so they sum slightly above e2e). With
DeepCache on, "UNet" is the *average* over full + cheap passes (`dc_reuse` = share of cheap
passes on that sequence). The mini was busy with other work during some runs: single runs vary ±5%, outliers were
re-measured.

## Current state, 2026-09-30 (M112): chart-first brief

`showcase/hypnagogia-speed.html`, rebuilt by `docs/shots/m112/brief.py` from
`docs/shots/m112/data/` (`engines.json` from this bench, two runs of each config, the
commands in its `by`; `batch.json` from `bench/batch_probe.py`; the experiment data).
`bench_engines.py --sizes 384,512x320 --iters 40` on the mini, background priority, torch
2.14.0:

| config | 384 (two runs) | 512x320 (two runs) |
|---|---|---|
| stock-equivalent (`{"deepcache":0,"attn":"sdpa","vae":"taesd","xframe":0}`) | 123.9 / 123.6 ms | 131.7 / 132.4 ms |
| defaults, no graft (`{}`) | 75.2 / 75.1 ms | 81.5 / 80.6 ms |
| as served (`{"depth_graft":0.8}`, flat depth) | 74.8 / 74.9 ms | 79.8 / 80.5 ms |

The graft build (on flat depth) ran about 1% faster than defaults in every run; it always ran last in its
round, so an order effect is the likely cause, and its cost is below what this bench resolves. In the game (WebKit harness) the
dream rate is about 11 a second (`docs/shots/m116/data/flicker-waking-*.json`). Measured no
this week: smaller captures, batching, reprojected DeepCache, world-anchored noise (the page
and M112's log).

**GPU + Neural Engine pool** (same day; `bench/coreml/build_models.sh graft`, then
`./run.sh --engine torch_turbo,coreml_turbo --width 512 --height 320`): `ws_client` 18.3 / 19.0 fps
at 2 / 3 in flight against 13.2 for the GPU alone at 2; in the pool a frame takes 127 ms on the
ANE and 89 on the GPU (76 alone). Data `docs/shots/m112/data/pool-*.json`, one run each; the page
has the game's dream rate, frame time and the extra shimmer, and `--pool-held carry`, which keeps
a view held still on the GPU (`data/flicker-carry-*.json`). The sections below are history: other days and account priorities, so
compare rows within one table.

## 2026-09-29 re-baseline: the capture shapes clients now send

`python bench/bench_engines.py --engine torch_turbo --sizes 384,512x320,320x512 --iters 40`
on the mini (M4 Pro), from an account that runs at background priority, under
`bench.lock`, torch 2.14.0. Rows as printed by the bench (e2e = mean
`process()` over the 24-frame walking sequence; main stages from the profiled pass):

| engine | size | e2e | FPS | stages (ms) |
|---|---|---|---|---|
| torch-sd-turbo | 384 | 72.1 ms | 13.9 | upload 4.4, encode 6.8, unet 59.6, decode 9.3, dc_reuse 0.5 |
| torch-sd-turbo | 512x320 | 77.8 ms | 12.9 | upload 4.8, encode 7.9, unet 62.7, decode 10.2, dc_reuse 0.5 |
| torch-sd-turbo | 320x512 | 72.0 ms | 13.9 | upload 4.9, encode 7.5, unet 57.3, decode 9.8, dc_reuse 0.6 |

512x320 (what a 16:9 client now sends to a 384x384 engine) cost ~8% more than 384x384 in
this one run (single runs vary about ±5%). 320x512 has the same pixel count but reused
DeepCache slightly more on this sequence (`dc_reuse` is a share, not ms), so it is not a
clean shape comparison. The 2026-09-28 table below came from a normal-priority account,
probably why it reads faster (63.5 ms at 384): compare rows within one table.

## 2026-09-29: cross-frame attention (`xframe`, now on by default)

Self-attention also attends to the previous frame's K/V (per prompt + seed). Cost, same
protocol as the re-baseline above: 384 = 73.8 ms (+2.4%), 512x320 = 79.4 ms (+2.1%).
Frame-to-frame change the engine adds on the synthetic sequences
(`bench/temporal.py --size 384 --cfgs '{"xframe":0}' '{"xframe":1}' --seq walk|turn`):

| sequence | input change | output, off | output, on |
|---|---|---|---|
| walk | 8.07 | 6.51 | 5.44 |
| turn | 14.39 | 16.72 | 13.64 |

In the game (WebKit, 8 zones, two runs each way) it cuts change after motion
compensation by 12% on average at equal detail (`docs/shots/m113/`).

## Recommendation

| machine | engine config | size | e2e | FPS |
|---|---|---|---|---|
| **mini** — M4 Pro 16-core GPU, 48 GB | `torch_turbo` **defaults**: sd-turbo, TAESD-lite, ToDo, guarded DeepCache N=3 | **384** | 63.5 ms | **15.7** |
| mini, quality | same, `--width 448 --height 448` | 448 | 84.6 ms | 11.8 |
| mini, native res | same, `--width 512 --height 512` | 512 | 101.5 ms | 9.8 |
| mini, max fps | same, `--width 320 --height 320` | 320 | 44.3 ms | 22.6 |
| **laptop** — M2 Pro, 16 GB | same defaults | **384** | 79.4 ms | **12.6** |
| laptop, smoother | `--width 320 --height 320` | 320 | 62.2 ms | 16.1 |

`python -m server --engine torch_turbo` picks 384 automatically on both machines (512 only on
Max/Ultra chips). The guard makes fast camera motion fall back to full passes, so the rate
breathes between ~11 FPS (fast look-around) and ~17 FPS (still / slow walk) at 384 on the mini;
without the motion guard (`--engine-arg dc_max_shift=99`) the same sequence measures
512 → 10.8, 448 → 13.1, 384 → 17.0, 320 → 22.5, 256 → 34.8 FPS.

Baseline: the old project's naive diffusers SD-Turbo pipeline ran ~0.25–3 FPS. The
diffusers-equivalent path (same model, TAESD, stock attention, no caching) measures
5.2 FPS @512 / 8.1 FPS @384 on the mini → defaults are **~1.9× at 512 and ~1.9× at 384 at
equal model**, 3× if you compare 384-defaults with 512-stock.

## Mini (M4 Pro, 48 GB) — the hill climb, in order

| # | engine / model | variant | size | encode | UNet | decode | e2e ms | FPS |
|---|---|---|---|---|---|---|---|---|
| 1 | sdxs-512-dreamshaper + own tiny AE | eager | 512 | 20.3 | 64.4 | 22.5 | 107.2 | 9.3 |
| 1 | 〃 | 〃 | 384 | 11.9 | 44.9 | 13.2 | 69.7 | 14.3 |
| 2 | sd-turbo + TAESD | stock SDPA attention | 512 | 20.6 | 148.3 | 22.8 | 191.1 | 5.2 |
| 2 | 〃 | 〃 | 384 | 11.8 | 98.0 | 13.3 | 122.9 | 8.1 |
| 3 | sd-turbo + TAESD | ToDo K/V ×2 at 64²+32² | 512 | 20.3 | 128.2 | 22.6 | 170.8 | 5.9 |
| 3 | 〃 | 〃 | 384 | 11.9 | 90.0 | 13.1 | 114.9 | 8.7 |
| 4 | sd-turbo + TAESD | ToDo 64² only | 512 | 20.4 | 131.0 | 22.7 | 174.0 | 5.7 |
| 4 | 〃 | 〃 | 384 | 12.0 | 90.9 | 13.3 | 116.0 | 8.6 |
| 5 | sd-turbo + TAESD | + DeepCache N=2 | 512 | 20.4 | 85.5 | 22.8 | 128.3 | 7.8 |
| 5 | 〃 | 〃 | 384 | 11.9 | 57.1 | 13.0 | 81.6 | 12.3 |
| 5b | 〃 | + DeepCache N=2, branch 2 | 384 | 12.2 | 71.4 | 13.6 | 95.6 | 10.5 |
| 6 | sd-turbo + TAESD | + DeepCache N=3 | 512 | 20.6 | 76.8 | 22.8 | 113.7 | 8.8 |
| 6 | 〃 | 〃 | 384 | 11.7 | 50.2 | 13.1 | 70.6 | 14.2 |
| 7 | sd-turbo + *linear* 8×8-patch encoder | ToDo, no DeepCache | 384 | 0.2 | 91.1 | 13.3 | 104.4 | 9.6 |
| 8 | sd-turbo + **TAESD-lite** | ToDo + DeepCache N=3 (no guard) | 512 | 10.4 | 76.7 | 11.2 | 92.2 | 10.8 |
| 8 | 〃 | 〃 | 448 | 8.2 | 64.6 | 8.8 | 76.2 | 13.1 |
| 8 | 〃 | 〃 | 384 | 6.2 | 50.3 | 6.7 | 58.7 | 17.0 |
| 8 | 〃 | 〃 | 320 | 4.7 | 38.8 | 5.1 | 44.4 | 22.5 |
| 8 | 〃 | 〃 | 256 | 3.1 | 25.1 | 3.4 | 28.7 | 34.8 |
| **9** | **sd-turbo + TAESD-lite** | **+ motion/cut guard (default)** | 512 | 10.5 | 85.8 | 11.2 | 101.5 | 9.8 |
| **9** | 〃 | 〃 | 448 | 8.4 | 72.5 | 9.0 | 84.6 | 11.8 |
| **9** | 〃 | 〃 | **384** | 6.3 | 57.6 | 6.8 | **63.5** | **15.7** |
| **9** | 〃 | 〃 | 320 | 4.6 | 38.6 | 4.9 | 44.3 | 22.6 |
| **9** | 〃 | 〃 | 256 | 3.1 | 25.0 | 3.5 | 29.3 | 34.1 |

UNet-only ablations (`bench/ablate_unet.py`, sd-turbo, ms per UNet eval):

| size | eager (stock SDPA) | channels_last | ToDo 64² | ToDo 64²+32² | no self-attn at 64² (bound) | torch.compile (inductor-MPS) |
|---|---|---|---|---|---|---|
| 512 | 146.3 | 194.6 | 129.3 | 126.5 | 124.2 | 187.6 |
| 384 | 96.8 | 144.2 | ~97 (noisy) | 88.5 | 87.7 | — |

DeepCache pass costs and CPU-dispatch check (`bench/cpu_bound.py`, lean forward + ToDo):

| size | full pass: CPU enqueue / GPU wall | cheap pass: CPU enqueue / GPU wall |
|---|---|---|
| 512 | 9.9 / 130.9 ms | 3.3 / 41.2 ms |
| 384 | 9.4 / 90.7 ms | 2.8 / 23.5 ms |
| 256 | 8.8 / 45.3 ms | 2.7 / 10.7 ms |

→ GPU-bound (≈4.6 TFLOP/s effective on the 339-GMAC UNet at 512, ~70% of the M4 Pro GPU's
fp16 peak), so wins have to come from doing less math, not from dispatch tricks.
`PYTORCH_MPS_FAST_MATH=1` / `PYTORCH_MPS_PREFER_METAL=1`: no measurable change.

## Laptop (M2 Pro, 16 GB)

| engine config | size | encode | UNet | decode | e2e ms | FPS |
|---|---|---|---|---|---|---|
| sd-turbo + TAESD, ToDo + DeepCache N=3 | 384 | 15.6 | 66.5 | 16.8 | 91.1 | 11.0 |
| 〃 | 320 | 11.2 | 51.8 | 12.0 | 71.8 | 13.9 |
| 〃 | 256 | 8.1 | 39.5 | 8.8 | 53.9 | 18.5 |
| **+ TAESD-lite (defaults, before the motion guard)** | **384** | 8.9 | 64.1 | 9.3 | **79.4** | **12.6** |
| 〃 | 320 | 7.0 | 51.1 | 7.4 | 62.2 | 16.1 |
| 〃 | 448 | 11.4 | 81.6 | 11.9 | 99.1 | 10.1 |
| 〃 | 512 | 13.5 | 94.6 | 14.3 | 117.8 | 8.5 |

The laptop GPU is shared with the desktop (WindowServer, other apps, headless Chrome); a re-run at load-avg 5 gave 8.7 FPS @384, so treat these as ±30%.

**Through the real server** (`python -m server --engine torch_turbo`, WebSocket, JPEG both ways,
scripted client keeping 2 frames in flight): laptop @384 sustained **10.2 FPS including the cold
first frame** (steady ms_infer 62–65 ms). On the mini the same test gave only 8 FPS because it
shared the GPU with a Core ML conversion and a second hypnagogia server (port 8797) — the
engine-only numbers above were taken under the bench lock.

## What worked, what didn't (and why)

* **Hand-rolled 1-step img2img** (no pipeline): encode → q_sample at t = 999·strength with a
  *fixed* noise tensor per (seed, size) → one UNet eval → exact ε→x̂₀ → decode → quantize on GPU
  → single 3·H·W-byte readback. Prompt embeddings LRU-cached; cross-attn K/V cached per
  conditioning tensor; prompt changes glide old→new embedding over `morph`=1 s (per seed).
* **SD-Turbo is the model.** SDXS-512 (both variants) is ~2× faster per UNet eval but was
  distilled only at t = 999: at any img2img strength it returns grey, structureless mud
  (`samples/torch-sdxs_512.jpg`). SD-Turbo (ADD-trained on 4 timesteps) keeps perspective,
  arches and light pools at 0.3–0.5 and dreams hard but coherently at 0.7.
* **Resolution:** 384 keeps interior structure well at 0.5–0.7; at 320 the model starts
  turning halls into outdoor vistas at 0.7 (`samples/torch-sd-turbo_320_default.jpg`).
* **TAESD-lite** (`distill_taesd_lite.py`): TAESD's full-resolution 64-ch stages were ~50% of
  its FLOPs and TAESD was ~35% of the frame. Replaced both with half-res pixel-(un)shuffle
  stems (161k params, 316 KB, trained 5 min on the mini against the TAESD teacher on nave
  renders + SD-Turbo paintings of them + txt2img images). Encode+decode 43 → 22 ms @512;
  round-trip PSNR 31.2 dB vs TAESD 31.1 dB; img2img outputs are indistinguishable at
  thumbnail scale, marginally softer at 1:1 (`samples/torch-sd-turbo_512_dc3.jpg` = TAESD vs
  `..._512_default.jpg` = lite). Falls back to TAESD if the weights file is missing.
* **ToDo** (token-downsampled self-attention K/V, Smith et al. 2024) at the 64² level: −12% UNet.
  At 32² too gains ~2% more but visibly hazes low-strength outputs → 64² only.
* **Temporal DeepCache** (Ma et al. 2023, applied *across video frames* rather than across
  denoising steps): every 3rd frame runs the full UNet and caches the deep features entering
  the last up-block; the frames in between run only conv_in + the 64² down level + the last
  up-block (cheap pass 23 ms vs 91 ms at 384). It calms the image (semantics stop re-rolling
  every frame; `temporal.py` "added flicker" on the walk: no cache +1.86, N=3 −1.56) but a
  cheap pass carries some of the full-pass frame's structure: edge correlation of cheap-pass
  outputs with current vs stale input (`alignment.py`) is 0.204 / 0.169 on a 7°/frame turn,
  vs 0.300 / 0.108 for full passes. Hence the **guard**, computed on CPU thumbnails (no GPU
  sync, ~1 ms): phase-correlated global shift since the full pass > 3 latent px or low-freq
  change > 0.06 (scene cut) → full pass. Walking/still = cheap passes; fast look-around =
  full passes (turn sequence: identical to no-cache output).
* **Motion-compensated DeepCache** (`dc_motion=true`: translate the cached deep features by the
  phase-correlation shift) lifts turn alignment to 0.264 / 0.107, but newly revealed borders
  get replicated deep features → orange/red smears at the entering edge
  (`samples/temporal_turn_default_384.jpg` rows are no-cache / unguarded / default). Opt-in.
  With the guard on, MC only ever sees shifts ≤ 3 latent px: walk alignment 0.232 → 0.245 (full-pass
  reference 0.258) but occasional streaks at the entering border remain visible, so it stays off.
* **Linear 8×8-patch encoder** (`fit_linear_encoder.py`, R² 0.35–0.6): free but blurs structure
  at strength ≤ 0.5 → opt-in `encoder=linear` only.
* **Didn't help:** channels_last (+33% slower on MPS), torch.compile / inductor-MPS (+43%
  slower), MPS fast-math / prefer-metal env knobs (±0).

## Samples (`bench/samples/`)

Contact sheets: top row = procedural input (`bench/scenes.py`), then strength 0.3 / 0.5 / 0.7
with three zone prompts (drowned cathedral / overgrown garden / neon bathhouse).

* `torch-sd-turbo_{512,448,384,320,256}_default.jpg` — the shipped engine
* `torch-sd-turbo_512.jpg` — stock attention + TAESD reference; `_512_dc3.jpg` — TAESD + ToDo + DC
* `torch-sd-turbo_512_todo2.jpg` (ToDo at two levels: hazier at 0.3), `_512_linenc.jpg` (linear
  encoder: blurry at 0.3), `torch-sdxs_512.jpg` (SDXS img2img failure)
* `temporal_walk_default_384.jpg` / `temporal_turn_default_384.jpg` — top = input frames 8–13,
  rows: no cache, DeepCache N=3 unguarded, default (guarded)
* `coreml_*` — see RESULTS-coreml.md

## Reproduce

```bash
# mini: venv ~/hypnagogia-cache/venv-gpu, HF_HOME=~/hypnagogia-cache/hf
python bench/bench_engines.py --sizes 512,448,384,320,256 --iters 30 --samples --tag _default
python bench/bench_engines.py --sizes 512,384 --cfg '{"deepcache":0,"vae":"taesd","attn":"sdpa"}'   # ~stock
python bench/bench_engines.py --model sdxs --sizes 512,384 --samples
python bench/ablate_unet.py --sizes 512,384 --variants eager,cl,todo1,todo2_32,noattn64
python bench/temporal.py --size 384 --cfgs '{"deepcache":0}' '{"deepcache":3,"dc_max_shift":99}' '{"deepcache":3}'
python bench/temporal.py --size 384 --seq turn --cfgs '{"deepcache":0}' '{"deepcache":3,"dc_max_shift":99}' '{"deepcache":3}'
python bench/alignment.py --seq turn --max-shift 99            # cheap-pass alignment (add --deepcache 0 for reference)
python bench/cpu_bound.py --sizes 512,384,256
python bench/distill_taesd_lite.py --steps 1500                 # regenerates server/engines/taesd_lite.safetensors
# serve it
python -m server --engine torch_turbo                                      # 384 default
python -m server --engine torch_turbo --width 448 --height 448             # quality
python -m server --engine torch_turbo --engine-arg deepcache=0             # no temporal caching
```
