# Core ML / Apple Neural Engine engine: results

**Verdict: yes. For the same algorithm the Neural Engine is 1.2× faster than tuned
PyTorch-MPS, and it runs on separate silicon, so the GPU stays free.**

On the mini (M4 Pro, 16-core ANE), SD-Turbo 1-step img2img runs end to end on the ANE at:

- **84 ms/frame at 512², 11.9 FPS through the real server.** The torch-MPS engine with the
  same ToDo trick takes 102 ms (9.8 FPS). It took 174 ms before the other engineer's tuning.
- **68 ms at 384², 14.8 FPS, with exact attention.**
  - The same tuned torch-MPS engine takes 64 ms at 384, but uses ToDo there.
  - For the same algorithm (kv2), the ANE takes 55 ms.
  - Torch numbers come from `bench/out/results.jsonl`, 19:07.

Speed is roughly at parity, so the ANE's decisive advantage is that **it is a second
accelerator**.

- **Nothing runs on the GPU.** Every op runs on the ANE, per `MLComputePlan`.
- **It keeps full speed next to a GPU load.** An MPS UNet running at the same time changes
  the ANE's throughput by less than 1%.
- **It leaves the renderer alone.** Safari's 4K WebGL renderer on the same box keeps the GPU.
- **Throughput adds up in a pool.** An ANE + MPS pool gets roughly the sum of both.
- **Outputs match the torch engine.** Same noise, timestep and morph conventions; see the
  pool-consistency result below.

`server/engines/coreml_turbo.py` is the engine. It is first in `DEFAULT_PREFERENCE`, so on
the mini `--engine auto` picks it as long as the compiled models are present.

**The laptop (M2 Pro, 16 GB) also runs it well:** SD-Turbo at 384² with int8 weights,
all on the ANE, gives **84 ms e2e and 11.9 FPS through the real server**. The mini's
compiled `.mlmodelc` were copied to `~/hypnagogia-cache/coreml` there, and the output is
identical (`coreml_sdturbo_384_w8_m2pro.jpg`).

If `--width/--height` aren't given, the engine picks the largest size (512 → 384 → 256)
whose models exist in the model directory. So the mini serves 512 and the laptop serves 384
automatically.

## Headline numbers (mini, M4 Pro, macOS 26.5, coremltools 9.0; all timed runs held `bench.lock`)

End-to-end `engine.process()`: uint8 RGB in → uint8 RGB out, 24-frame walking-camera
sequence, median of 30–48 iterations. Everything runs on the ANE (`CPU_AND_NE`).

torch-MPS columns use the other engineer's `bench_engines.py` on the same box. The
"exact" column is the untuned SD-Turbo from 18:25; the "tuned" column is their 19:07 default,
which includes ToDo 2×2 at level 0.

| model | size | UNet variant | enc ms | UNet ms | dec ms | **e2e ms** | **FPS** | torch-MPS e2e: exact / tuned |
|---|---|---|---|---|---|---|---|---|
| SD-Turbo | 512 | exact fp16 | 8.2 | 132.9 | 8.3 | 149.9 | 6.7 | 174.0 / — |
| SD-Turbo | 512 | **kv2w8** (default at 512) | 8.2 | 67.1 | 8.4 | **84.5** | **11.8** | — / 102.4 (9.8 FPS) |
| SD-Turbo | 384 | exact fp16 | 4.6 | 63.3 | 4.8 | 73.0 | 13.7 | 115.0 / — |
| SD-Turbo | 384 | **w8** (default below 512) | 4.6 | 57.9 | 4.8 | **67.6** | **14.8** | — |
| SD-Turbo | 384 | kv2 (not recommended) | 4.6 | 45.5 | 4.7 | 55.1 | 18.1 | — / 64.4 (15.5 FPS) |
| SD-Turbo | 384 | kv2 on down-blocks only + w8 (not recommended) | 4.8 | 52.5 | 5.0 | 62.6 | 16.0 | — |
| SD-Turbo | 448 | w8 | 6.2 | 90.5 | 6.3 | 103.4 | 9.7 | 145.7 / — |
| SD-Turbo | 448 | kv2w8 (not recommended) | 6.3 | 53.6 | 6.4 | 66.7 | 15.0 | — / 84.6 |
| SDXS-512-0.9 | 512 | exact | 8.2 | 39.8 | 8.3 | 56.8 | 17.6 | 107 (sdxs-dreamshaper) |
| SDXS-512-0.9 | 384 | exact | 4.6 | 25.4 | 4.8 | 35.1 | 28.5 | 70 (sdxs-dreamshaper) |
| SDXS-512-0.9 | 256 | exact | 2.1 | 10.0 | 2.2 | 14.5 | 69 | — |

Through the real server (`python -m server --engine coreml_turbo`, measured with
`bench/ws_client.py`, 15–20 s, 1 connection, 2 frames in flight, JPEG both ways):

| machine | config | system FPS | engine FPS | latency p50 / p99 |
|---|---|---|---|---|
| mini M4 Pro | SD-Turbo 512 kv2w8 (default on the mini) | **11.90** | 11.91 | 168 / 177 ms |
| mini M4 Pro | SD-Turbo 384 exact fp16 | **13.65** | 13.67 | 147 / 152 ms |
| laptop M2 Pro | SD-Turbo 384 w8 (default on the laptop) | **11.87** | 11.98 | 167 / 253 ms |

On the laptop, e2e `process()` at 384 w8 is enc 6.9 + UNet 69.9 + dec 6.5 = **84.0 ms**. The
first load there compiled for the ANE in about 40 s; after that the server starts in 4 s.

Server efficiency is 0.999, so JPEG coding and the websocket overlap cleanly with inference.
Startup takes about 2.5 s once the OS has cached the ANE compilation. The **first ever** load of each
UNet runs the ANE compiler, which takes 20–40 s. The server loads engines in the background,
so this only delays the first frames.

## UNet alone, by compute unit (512², median of 15–20 runs)

| UNet | CPU_AND_NE | CPU_AND_GPU (Core ML) | ALL |
|---|---|---|---|
| SDXS-512-0.9 | **39.7 ms** | 82.8 ms | 39.7 ms (Core ML puts everything on the ANE) |
| SD-Turbo, exact | **134.2 ms** | 189.7 ms | – |
| SD-Turbo, int8 weights (`_w8`) | 126.1 ms | | |
| SD-Turbo, ToDo K/V 2×2 at the 64×64 level (`_kv2`) | 75.4 ms | | |
| SD-Turbo, kv2 + int8 (`_kv2w8`) | 67.1 ms | | |
| SD-Turbo, attention query chunk 256 / 512 / 1024 / none | 131.9 / 134.2 / 137.2 / 137.3 ms | | |

TAESD at 512² (encoder / decoder): **ANE 7.7 / 8.2 ms**, Core ML GPU 18.5 / 21.6 ms,
ALL 10.5 / 10.9 ms. For comparison, the torch-MPS TAESD takes 20 / 23 ms.

`MLComputePlan` (`bench/coreml/plan.py`) reports **100% of ops on the Neural Engine** for
every model: SD-Turbo UNet 6639 ops, SDXS UNet 3800, TAESD enc 76, dec 85. Nothing falls back
to the CPU or GPU.

Accuracy of the fp16 ANE UNet against the fp32 torch reference (same inputs): cosine
similarity 0.9999 for SDXS and 0.997 for SD-Turbo. Adding int8 weights doesn't reduce it
further (0.9968). The sample sheets show no visible difference.

## ANE ↔ GPU overlap (the "pipelining" question)

- **Across processes the overlap is essentially perfect** (`bench/coreml/concurrency.py`,
  SD-Turbo UNet at 384²):
  - The ANE Core ML UNet runs at 15.66 it/s alone and 15.59 it/s concurrently.
  - The torch-MPS fp16 UNet runs at 8.84 it/s alone and 8.66 it/s concurrently.
  - Combined that's **24.3 UNet evals/s**.
  - So ANE diffusion takes almost nothing from the GPU. Safari's 4K WebGL, or a second engine
    on MPS, keeps the GPU to itself.
- **Inside one process, threads don't overlap.** UNet(ANE) in one thread and TAESD(GPU) in
  another take 80.0 ms per frame, the same as running them one after the other.
  coremltools' `predict` holds the GIL for most of the call: a spinning Python thread gets
  only 29% of its solo rate while a predict runs. The server is unaffected (efficiency 0.999).
- Because `process()` is synchronous, per-frame pipelining can't happen inside the engine.
  The real opportunity is at the server level: run `coreml_turbo` (ANE) and `torch_turbo`
  (MPS) as **two worker processes and hand frames to whichever is free**. Both use the same
  model, math and seed noise, so their results match closely (PSNR 30–33 dB, see below).
  Throughput is roughly the sum, about 1.6–1.9× a single engine on this box.

## What the engine does

`RGB → TAESD enc → z0 → z_t = √ᾱ_t·z0 + √(1-ᾱ_t)·ε_seed → UNet (1 step, ε-pred) → x0 = (z_t − √(1-ᾱ_t)·ε̂)/√ᾱ_t → TAESD dec → RGB`

- **Timestep:** `t = round(clamp(strength, 0.02, 1)·999)`.
- **Noise:** `ε_seed` is cached per (seed, size) and generated with `torch.Generator("cpu")`.
- **Prompt changes:** the embedding glides to the new prompt with a time constant of `morph`
  seconds (default 1 s).
- **Matching torch_turbo:** the timestep mapping, noise generator and prompt glide are the same
  as in `server/engines/torch_turbo.py`. So a server **pool that alternates frames between ANE
  and MPS produces matching images**. `bench/coreml/pool_consistency.py` at 512, strength 0.5,
  seed 1234 gives mean |diff| 4–6/255 and PSNR 30–33 dB, with the same composition and detail
  (`bench/samples/coreml_vs_torch_512.jpg`, rows: input / torch / coreml / |diff|×4).
- **Text embeddings:** computed by the model's own CLIP text encoder (torch, CPU, 0.2–0.5 s to
  load from the HF cache) and cached per prompt. It's 1-step and CFG-free, so `negative` is
  ignored.
- **UNet conversion:** diffusers' `Transformer2DModel` is rewritten into the ANE-native
  layout (`bench/coreml/convert.py`), following Apple's ml-stable-diffusion recipe:
  - (B, C, 1, S) tensors
  - Linear → 1×1 Conv
  - channel LayerNorm
  - per-head split-einsum attention, with the query chunked in 512s
- **Model format:** fixed shapes, fp16 ML Program (macOS 15 target), compiled to
  `.mlmodelc` with `xcrun coremlcompiler`.
- **`_kv2` variant:** ToDo-style token downsampling (arXiv 2402.13573). Self-attention keys and
  values at the highest-resolution level (≥ 2048 tokens) come from a 2×2 average-pooled copy.
  That's 4× fewer keys, and it removes 59 ms (44%) of the SD-Turbo UNet's 134 ms of ANE time.
  The 4096-token self-attention is the ANE's bottleneck. Chunking the query doesn't help,
  because the key count is what matters.
- **`_w8` variant:** per-channel symmetric int8 weights (`linear_quantize_weights`). The UNet
  drops from 1.6 GB to 0.83 GB and gets ~6–9% faster.
- **Default variant:** `_kv2w8` at ≥ 512²; `_w8` below that (see quality). Override with
  `--engine-arg variant=""` for the exact fp16 model. A depth-grafted build (`_kv2w8_d08`,
  `build_models.sh graft`, 512x320; M112) comes first when `depth_graft` is 0.8, the server's
  default: it copies torch_turbo, whose ToDo acts at every size, although its pooled keys
  (32x20) are under the rule of thumb below.

## Quality (sample sheets in `bench/samples/`, rows = input, strength 0.3 / 0.5 / 0.7)

- `coreml_sdturbo_512.jpg` (exact) and `coreml_sdturbo_384_w8.jpg`: crisp and faithful to
  the input.
  - Strength 0.3–0.5 keeps the architecture exactly (columns, arches, floor grid) and
    re-materialises the surfaces.
  - Strength 0.7 re-imagines strongly: a flooded gothic nave, an overgrown temple garden, a
    neon corridor.
  - The ANE fp16 output can't be told apart from the torch output
    (`torch-sd-turbo_512.jpg`).
- `coreml_sdturbo_512_kv2w8.jpg`: very close to exact. Composition is the same; it's slightly
  softer and a little hazier on the neon prompt. This is the recommended speed/quality point
  at 512.
- `coreml_sdturbo_384_kv2.jpg`: **visibly worse**. There's haze, and vertical "curtain" streaks
  at strength 0.3–0.5, because 48×48 → 24×24 keys is too coarse. That's why kv2 is not used
  below 512 (except in the graft build, which copies the served torch engine).
  - The same holds at 448 (`coreml_sdturbo_448_kv2w8.jpg`: hazy at 0.3) and for kv2 on only
    the down blocks at 384 (`coreml_sdturbo_384_kv2dw8.jpg`).
  - Exact-architecture int8 (`coreml_sdturbo_448_w8.jpg`, `coreml_sdturbo_384_w8.jpg`) stays
    crisp.
  - Rule of thumb: pooled keys need a grid of at least 32×32.
- `coreml_sdxs_512.jpg`: SDXS-512-0.9 is 3× faster than SD-Turbo but **hazy and blurry for
  img2img** at strength ≤ 0.5. It was distilled for t=999 only, and the torch SDXS sample shows
  the same. It works only as a speed fallback (`--model sdxs`).

## How to run / reproduce

One-shot: `bench/coreml/build_models.sh [default|384|512|graft|all]` creates the converter venv
and builds the models into `~/hypnagogia-cache/coreml`. The `384` set took 59 s on the mini.
Manual steps:

```bash
# on the mini: venv with converter deps (python 3.12; torch pinned to 2.7.1 for coremltools 9.0)
uv venv ~/hypnagogia-cache/venv-coreml --python 3.12
uv pip install --python ~/hypnagogia-cache/venv-coreml/bin/python \
    torch==2.7.1 diffusers transformers accelerate safetensors coremltools pillow numpy aiohttp simplejpeg
P=~/hypnagogia-cache/venv-coreml/bin/python; export HF_HOME=~/hypnagogia-cache/hf
# build the models the engine uses (about 1 min each; output goes to ~/hypnagogia-cache/coreml)
$P bench/coreml/convert.py --model sdturbo --res 512 --kv-down 2 --w8 --suffix _kv2w8 --vae taesd
$P bench/coreml/convert.py --model sdturbo --res 384 --w8 --suffix _w8 --vae taesd
$P bench/coreml/convert.py --model sdturbo --res 512 384 --vae taesd          # exact fp16 variants
$P bench/coreml/convert.py --model sdxs --res 512 384 256 --vae sdxs          # SDXS
# serve (any venv that has coremltools; the runtime needs no specific torch version)
python -m server --engine coreml_turbo [--width 384 --height 384] [--model sdxs] \
       [--engine-arg variant='""'] [--engine-arg compute_units=GPU]   # variant "" = exact fp16
# benchmarks
$P bench/coreml/bench.py --unet ~/hypnagogia-cache/coreml/sdturbo_unet_512_ane.mlmodelc --cu NE GPU ALL
$P bench/coreml/sample.py --model sdturbo --sizes 512 384            # e2e + contact sheet
$P bench/coreml/concurrency.py --size 384                            # ANE || MPS
$P bench/coreml/plan.py ~/hypnagogia-cache/coreml/sdturbo_unet_512_ane_kv2w8.mlmodelc NE
```

The compiled `.mlmodelc` files are device-independent. Copy `~/hypnagogia-cache/coreml/` to
another Mac, such as the M2 Pro laptop, to run the engine there. That Mac pays the one-time ANE
compile on first load.

## Known issues / caveats

- Model shapes are fixed. The server resizes captures to the engine's width×height. Other
  sizes, including non-square ones like `640x384` (multiples of 64), need `convert.py --res WxH`.
- The first load of a new `.mlmodelc` on a machine takes 20–60 s (ANE compile, cached by the
  OS afterwards).
- Conversion needs torch ≤ 2.7-ish. coremltools 9.0 is tested only up to torch 2.7, and the
  server venv has torch 2.14. The *runtime* only needs `coremltools`. The torch there is used
  just for the CLIP text encoder.
- The exact-fp16 UNet converted via plain `diffusers` modules (`--attn plain`) fails to trace
  (`aten::Int` on traced shapes). It isn't needed, since the ANE layout also runs on the GPU.

## Next steps (ranked)

1. **Server: ANE + MPS pool.** It's measured at about +60% throughput, and the outputs
   match (see the pool-consistency numbers above). The server owner already has a
   `torch_turbo+coreml_turbo` pool running. Note that the torch engine's default ToDo
   (`attn="todo"`, levels=1) is the same trick as `_kv2`. Below 512² it causes haze and streaks
   at low strength (see the kv2 sheets at 384 and 448), so both engines should drop it there.
2. Fuse enc → UNet → dec into one Core ML program with fp16 I/O. This saves 3 dispatches and
   fp32 conversions, an estimated 3–5 ms/frame.
3. Try milder token merging at 384 (2×1 K/V, or ToMe on the query path) to get about 18 FPS
   without kv2's streaks.
4. Try int8 activations (W8A8) for the M4 ANE's int8 path. It could be 1.3–1.8× faster on
   the conv-heavy blocks, and it's wired up already (`convert.py --w8 --a8`).
   - I tried it: `cto.experimental.linear_quantize_activations` calibrates on the CPU and
     took **7.3 min per calibration sample** for the 384² UNet. I aborted it (6 samples would
     have taken about 45 min).
   - Run it overnight, or with a larger `calibration_op_group_size`, then check quality.
5. Try a Swift runner with `MLModel` async predictions for cross-frame pipelining. This only
   makes sense if the protocol allows a one-frame result lag, which it currently doesn't.
