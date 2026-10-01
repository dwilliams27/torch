# Hypnagogia ideas, lens: RAW SPEED — Claude Opus 5.5, 2026-09-30

One of five independent idea sessions (M115). The lens: make the dream pipeline end to end
(model, runtime, I/O, scheduling, client capture, server) 2-5x faster on the M4 Pro Mac mini,
or as fast on a fraction of the GPU, which the browser also needs. This was a read-only
session: nothing below has been run. "Estimate" means arithmetic on the repo's measured
numbers; "guess" means judgement.

## Where the time goes today

- **Engine**, 512x320 (the 16:9 capture), `torch_turbo` on MPS: 77.8 ms per dream: upload
  4.8, TAESD-lite encode 7.9, UNet 62.7 (averaged over full and DeepCache cheap passes at a
  0.5 cheap share), decode 10.2 (`bench/RESULTS.md`, 2026-09-29, background priority;
  cross-frame attention adds 2%). Stages come from a synced pass, so they sum slightly high.
- **The UNet is compute-bound**: a full pass is 90.7 ms of GPU time against 9.4 ms of CPU
  enqueue at 384²; the DeepCache cheap pass (conv_in, the top level, the last up block) is
  23.5 ms (`bench/cpu_bound.py`). It runs at ~70% of the GPU's peak, hence RESULTS'
  conclusion that wins must come from doing less math.
- **In the game**: 9.1-12.3 dreams/s in the WebKit harness; round trip 182-250 ms; capture
  readback and JPEG encode 14-28 ms in WebKit, 150-230 ms in headless Chromium (which then
  dreams only 4.6-5.1 times a second).
- **Neural Engine** (Core ML, same machine, 2026-09-28): SD-Turbo UNet 63.3 ms exact and
  57.9 ms with int8 weights at 384², 132.9 ms at 512², where exact attention at the top level
  is the bottleneck (pooling its K/V removed 59 ms); TAESD 7.7 / 8.2 ms at 512². ANE and MPS
  overlap almost perfectly as separate processes (15.66→15.59 and 8.84→8.66 it/s) and not at
  all as threads of one process (coremltools holds the GIL). None of these models are built
  in the current cache; `bench/coreml/build_models.sh` is the recipe.

## Already tried, so the ideas below avoid it or say why it's different now

SDXS (a 1-step student distilled only at t=999: grey mud for img2img); channels_last (+33%)
and torch.compile (+43%); MPS fast-math knobs (no change); the linear 8x8-patch encoder
(blurs structure); ToDo at the 32x32 level too (haze); K/V pooling on the ANE below 512²
(streaks); translate-only motion-compensated DeepCache (smears at the entering border);
world-anchored noise, twice (less stable than screen-fixed noise); W8A8 calibration (aborted
at 7.3 min per sample on CPU); ANE and GPU as threads of one process (GIL); `?calm`,
`xframe_bias`, more feedback for narrow captures, `?liveref` (half-way).

## The three bets

1. **Two-headed UNet** (idea 1): split SD-Turbo's UNet between the GPU (top level) and the
   Neural Engine (deep levels), pipelined across frames. About 1.6-2x the dreams per second,
   with a full pass on every frame, or today's rate on about a third of the GPU.
2. **A student cut from SD-Turbo** (idea 2), distilled on the game's own captures at its one
   timestep: 1.4-2x that compounds with everything else, and the only road to the laptop and
   the phone.
3. **Foveal warp** (idea 3): one capture with a fovea replaces the narrow/wide alternation.
   Each dream paints the whole screen at the fovea's density (1.5-1.8x more screen painted per
   second), the engine is untouched, and the narrow view's border goes away.

They stack: 1 and 3 change where and how often the UNet runs; 2 changes the UNet.

Each idea below gives the idea, why it could be big here, the cost, the smallest experiment
that would tell us in two hours or less, and what would kill it.

---

## 1. Two-headed UNet: the GPU draws the top level, the Neural Engine thinks the deep levels (BET)

**Idea.** Cut SD-Turbo's UNet where DeepCache already cuts it. The GPU (torch on MPS) runs
conv_in and `down_blocks[0]` (the 64x40 grid at 512x320, 320 channels) and later
`up_blocks[3]` and conv_out. A second process runs everything in between (`down_blocks[1-3]`,
the mid block, `up_blocks[0-2]`) as a fixed-shape Core ML program on the Neural Engine. Frames
are pipelined: while the ANE works on frame k's deep levels, the GPU runs frame k+1's shallow
down pass and frame k-1's up pass and decode. Every frame gets fresh deep features (a full
pass, not a DeepCache cheap pass) at the price of one pipeline stage of latency (~30 ms).

**Why here.** Each half lands on the silicon that suits it. The top level is where the ANE is
weakest (its exact 4096-token attention is the bottleneck at 512²) and it is exactly the
cheap pass the GPU already runs in 23.5 ms at 384² (~26 ms at 512x320 by interpolation). The
deep levels are convolution-heavy with at most 640 tokens. Estimate: once exact top-level
attention is set aside, the 384² ANE numbers imply ~8-9 TFLOP/s on the convolution-heavy
parts, and the deep path is ~64% of the ~0.36 TFLOP of a 512x320 pass, so ~30 ms on the ANE.
Budget: GPU per dream = upload 5 + encode 8 + top level 26 + decode 10 ≈ 49 ms (78 now), so
~20 dreams/s. With TAESD-lite's encode and decode also on the ANE (7.7 / 8.2 ms at 512² for
full TAESD), the GPU needs ~28-30 ms and the ANE ~40 ms: ~25 dreams/s, ANE-bound. That is
1.6-2x today's ~12.8/s engine ceiling at 512x320, and every frame is a full-quality pass
(turn alignment 0.300 for full passes against 0.204 for cheap ones, RESULTS). Or hold today's
~12/s with the GPU busy about a third of the time, which matters when the game is played on
the Mac mini's own 4K display. Precedent: AsyncDiff (Chen et al., NeurIPS 2024) split SD
2.1's denoiser into components on separate GPUs and ran them in parallel on slightly stale
inputs: 1.8x on two GPUs for a 0.01 drop in CLIP score. Here frames are pipelined instead of
denoising steps, so nothing is stale.

**Cost.** 3-6 days. A converter venv (torch 2.7.1 + coremltools 9, ~2 GB); one deep-path
model per capture shape (512x320 and 320x512; the deep path holds most of the 865M weights,
~0.8 GB each with int8 weights, twice that with the compiled copy); shared memory between the
torch process and the ANE process (`ProcEngine` already isolates engines); a two-frame
pipeline that uses `torch.mps.Event` so reading the 410 KB handoff tensor doesn't drain the
GPU queue. Cross-frame attention in the deep levels is lost unless its K/V ride along as
Core ML state (stateful models, macOS 15+), so first measure how much of xframe's 12% gain
lives in the deep levels. Risk: medium. Per-call overhead and copies run on ~4 efficiency
cores, and fp16 ANE numerics may shift the style slightly (but on every frame alike).

**Smallest experiment (≤2 h).** Add a `--part deep` wrapper to `bench/coreml/convert.py`
(inputs: `down_blocks[0]`'s output after its downsampler, 1x320x20x32, plus the time and
text embeddings; output: `up_blocks[2]`'s output, 1x640x40x64). Build it at 512x320 with
`--w8`, time `predict()` on CPU_AND_NE, confirm 100% ANE with `bench/coreml/plan.py`, and
compare with the torch deep path on ten real inputs. Then run the torch top level on the
ANE's deep tensor and compare the decoded image with a pure-torch full pass. Success: ≤35 ms
on the ANE, cosine ≥ 0.995, decoded PSNR ≥ 30 dB (whole-UNet ANE/MPS swaps measured 30-33).

**Kill it if** the deep path takes more than 45 ms on the ANE, the handoff costs more than
8 ms a frame on the efficiency cores, or the hybrid's output drifts visibly from torch's. The
fallback is asynchronous: the ANE runs whole-UNet passes at its own rate and refills
DeepCache's cache on the GPU (staler features, still ~1.6x).

## 2. A student cut from SD-Turbo itself, trained on the game's own traffic (BET)

**Idea.** Remove blocks from SD-Turbo's UNet the way BK-SDM did (the mid block, one
resnet+attention pair per down stage, one of three per up stage), keep SD-Turbo's weights for
everything else, and distill with output and per-block feature losses on exactly what the
game sends: captures that include feedback, at t≈599 (strength 0.6), with the eight zone
prompts, at 512x320 and 320x512. Then iterate DAgger-style: play with the student in the
loop, label its captures with the teacher, retrain, because the feedback loop feeds the
student its own paint. Later rounds can thin the top level's channels too.

**Why here.** The UNet is ~80% of engine time (62.7 of 77.8 ms) and the GPU is compute-bound.
BK-SDM (Kim et al., ECCV 2024) removed blocks from SD v1.4 and v2.1-base for 30-50% less
size, MACs and latency, recovering quality with feature distillation on 0.22M LAION pairs in
13 A100-days. Our task is far narrower (one timestep, one level family, eight prompts,
img2img), so far less data should do (guess: 100-300k teacher-labelled samples). SDXS failed
here only because it was distilled for t=999; a student trained at our timestep on our inputs
doesn't have that gap. Expected 1.4-2x on the UNet (BK-SDM's range). It compounds with
DeepCache (the cheap pass thins too), with idea 1 and with the laptop, which does 12.6/s at
384 today.

**Cost.** 1-3 weeks. Data: a server `--dump` option that saves decoded captures and headers
during harness autopilot runs across all zones (~36k an hour at ~10/s), stored as TAESD-lite
latents (20 KB each: 100k ≈ 2 GB). Training on MPS (guess: ~5 samples/s with the teacher run
online for feature targets, ~18k an hour, so a 200k-step round takes ~11 hours overnight,
niced, paused whenever someone plays); ~10-16 GB of the 48. Risk: medium-high. Small students
lose richness, and errors can compound through the feedback loop.

**Smallest experiment (≤2 h).** Training-free first: add a `skip` set to
`lean_unet_forward` (keeping the skip-connection bookkeeping consistent), time candidates
with `bench/ablate_unet.py` and look at `bench_engines.py --samples` sheets. Then fine-tune
the best cut for one hour (only the blocks next to each cut trainable) on ~5k teacher pairs
(`bench/scenes.py` naves plus harness captures), tracking PSNR against the teacher with
identical noise on held-out frames. Success: at least 35% less UNet time, and PSNR climbing
from the untrained cut to ≥27 dB within the hour, still rising.

**Kill it if** an hour of tuning recovers less than 1.5 dB, or arches and pillars melt at
strength 0.6 in the sheets.

## 3. Foveal warp: one capture with a fovea instead of narrow and wide captures (BET)

**Idea.** Render every centre capture through a smooth radial (or separable) magnification:
pixel density about 1.6x in the middle falling to ~0.6x at the edges, so one 512x320 dream
has the narrow capture's density where you look and wide coverage around it. The client
renders the capture ~1.7x larger and resamples colour and depth through the warp. `live.js`
and the paint shader project a world point with the capture's linear view-projection, then
apply the same closed-form warp before sampling; the pixel-footprint term uses the warp's
local magnification.

**Why here.** Walking, captures go two narrow to one wide (`dream.js`), so two dreams in three
paint only the middle 35% of the screen: at 12 dreams/s that is 4 x 1 + 8 x 0.35 = 6.8
screens a second. Warped, every dream paints the whole screen at the fovea's density: 12
screens a second (1.8x; 1.5x at rest, where the pattern is 1:1). One stream instead of two
also means consecutive frames of the stream are closer in time, so DeepCache's motion guard
trips less and the cross-frame anchors are fresher (both are kept per seed, and each capture
kind has its own seed). Density now falls off continuously, so the border of the narrow
view, the "lens" in the middle of the screen from the 2026-09-29 play test that M114 is
fixing, has nothing to show. Precedent for warping inputs to give a network more pixels where
they matter: Learning to Zoom (Recasens et al., ECCV 2018), a saliency-driven sampling layer
that beat uniform downsampling at a fixed input size. Whether SD-Turbo paints a warped image
well is the untested part.

**Cost.** 2-4 days, client only (a capture resample pass, the warp in `GLSL_LIVE` and the
paint shader, the footprint term). Rendering a 0.47-megapixel capture costs little. The edges
drop to ~0.6x of today's wide density. Risk: medium.

**Smallest experiment (≤2 h).** Offline in Python: render `bench/scenes.walk` frames at
870x544, warp them to 512x320 (controls: plain wide 512x320, and a 1.7x narrow crop), dream
each with `torch_turbo` at 0.6, unwarp the warped result to screen space, and sheet them
(`tools/sheet.py`). Score edge correlation (`alignment.py`'s `edges()`) against the
full-resolution input, centre 35% and periphery separately. Success: the centre within ~5%
of the narrow capture's score, the periphery within ~10% of the wide one's, and no bent
columns after unwarping.

**Kill it if** the model paints the distortion itself (curved pillars after unwarping), or
the centre gains less than half of what a narrow capture gains.

## 4. Geometry on the wire: warp the caches by the real reprojection

**Idea.** Send the capture's view-projection (16 floats) and a 64x40 linear-depth map
(5-10 KB) with each frame; the frozen protocol allows extra fields. The server keeps each
stream's last full-pass pose and depth and backward-warps `cache['deep']` (1x640x40x64) and
the cross-frame K/V into the new frame with `grid_sample`. The DeepCache guard becomes exact
(the share of deep cells disoccluded) instead of phase correlation on thumbnails, and the
result header carries a validity mask for regions whose features had to be invented; the
client doesn't project those (live views and the atlas already handle partial coverage).

**Why here.** Turns force full passes today (the guard trips at a 3-latent-pixel shift), so
the dream rate drops to ~11/s exactly when new scenery floods in, and in play only 0.5-0.65
of frames take the cheap pass. Translate-only compensation lifted turn alignment from 0.204
to 0.264 but was dropped for smears at the entering border; with a validity mask those
features never reach the screen, and depth makes the warp right under parallax. The same
warp lets narrow captures borrow the wide stream's deep features (a crop plus a 1.7x zoom),
so narrow dreams need no full passes of their own. Guess: a 0.8+ cheap share with a full pass
every 4-5 frames, about 1.3x, plus alignment that follows silhouettes (a pillar against its
wall). It is also the plumbing for ideas 5 and 14.

**Cost.** 2-4 days: a 64x40 depth pass and readback in the client, header fields, a per-stream
pose and depth store, a `grid_sample` of a 3.3 MB tensor (~1 ms on MPS), mask plumbing in
`live.js`. Risk: low-medium. Deep features have frame-wide receptive fields, so they only
approximately warp, worst under zoom (walking forward).

**Smallest experiment (≤2 h).** Offline. `bench/scenes.py`'s ray caster already computes each
pixel's hit distance (`best`); return it as depth. On `scenes.turn` (pure rotation, so the
warp is a homography) and `scenes.walk`, run `alignment.py` variants: stale cheap pass,
integer-shift compensation, and exact reprojection, scoring only the valid region. Success:
turn alignment ≥ 0.27 with the current frame and ≤ 0.12 with the stale one (full-pass
reference 0.300 / 0.108), and `temporal.py` walk flicker no worse than the guarded default.

**Kill it if** warped cheap passes still follow the stale frame (the deep features carry a
global layout that no warp fixes), or after two frames of a typical turn less than 80% of the
frame is valid.

## 5. Clockwork across frames: predict the deep features instead of computing them

**Idea.** Train a small adaptor that predicts this frame's deep features from the last full
pass's (warped as in idea 4), this frame's fresh top-level features and the camera delta, so
full passes are needed only at scene cuts or every 8-10 frames. Clockwork Diffusion
(Habibian et al., CVPR 2024) did this across denoising steps: it replaced the low-resolution
UNet levels with a light adaptor fed the previous step's features and saved 32% of FLOPs on
SD 1.5 at 8 steps with negligible FID and CLIP change, trained in a day on one GPU.

**Why here.** Cheap passes cost about a quarter of full ones; the full passes keep the
average up. With full passes on ~10% of frames: 0.1 x 100 + 0.9 x (26 + ~3) ≈ 36 ms of UNet
against ~60 now, about 1.4-1.5x end to end (estimate). An adaptor can also learn to fill the
disocclusions that a pure warp can't.

**Cost.** 3-5 days. Each training pair needs two full passes and the deep tensors are 3.3 MB,
so train online with the teacher running. Risk: medium.

**Smallest experiment (≤2 h).** First measure the headroom: on `scenes.walk` and
`scenes.turn`, PSNR of cheap-pass output against full-pass output as a function of the gap
since the full pass (1-8 frames). Then fit a linear predictor (a 1x1 conv, least squares as in
`fit_linear_encoder.py`) from [warped stale deep features, pooled fresh top-level features] to
fresh deep features on ~2k pairs. Success: ≥2 dB over plain reuse at a 4-frame gap.

**Kill it if** the linear fit gains less than 1 dB at a 4-frame gap, or plain warped reuse is
already within 1 dB of a full pass (then idea 4 alone is enough).

## 6. A zero-GPU mode: DeepCache and cross-frame attention inside Core ML

**Idea.** Port the torch engine's temporal state to the ANE engine: a multifunction Core ML
model (coremltools 8 and macOS 15+; functions share deduplicated weights) with a `full`
function that also outputs the deep features and a `cheap` function for the top level, and
cross-frame K/V as Core ML state or explicit inputs and outputs. Pool it with MPS by stream
(wide captures on one engine, narrow on the other) and give both engines the same int8 weight
grid (dequantize the ANE's int8 weights into the torch model), so that only fp16 accumulation
differs.

**Why here.** `coreml_turbo` carries no state between frames, so a pool with it loses
DeepCache's calm and cross-frame attention's 12% steadier walking, and it shimmers against
MPS (known issue 7; pool PSNR 30-33 dB). Alone on the ANE it ran 67.6 ms end to end at 384²
with int8 weights (14.8 FPS engine-only; 13.65 FPS through the server with exact fp16), with
the GPU untouched; ANE plus MPS measured 24.3 UNet evaluations a second at 384². This is the configuration for play on the Mac mini's own 4K
display, where Safari's renderer wants the whole GPU.

**Cost.** 1-2 weeks; 1-3 GB of models per shape. Risk: medium. State and multifunction
programs may push some ops off the ANE (check `MLComputePlan`), and the ANE's cheap pass is
all top-level attention, its weak spot.

**Smallest experiment (≤2 h).** Build two plain models at 512x320 with int8 weights, a full
UNet with a deep-feature output and a top-level-only model; time both on CPU_AND_NE;
`plan.py` must show 100% ANE. Success: the ANE's cheap pass costs at most 40% of its full
pass.

**Kill it if** the cheap pass costs more than 60% of the full pass on the ANE. Then the ANE
should only ever run the deep levels (idea 1).

## 7. W8A8 on the M4 Neural Engine, calibrated on game captures

**Idea.** Quantize activations as well as weights to int8 for the ANE's convolution-heavy work
(idea 1's deep path), keeping softmax and normalisation in fp16 (mixed precision by op type),
calibrated on 32-64 real captures rather than generic images. Apple's coremltools guide says
the A17 Pro and M4 Neural Engines have a faster int8-int8 compute path and that W8A8 "can lead
to considerable latency benefits" for compute-bound models.

**Why here.** By my arithmetic on RESULTS-coreml, the ANE runs SD-Turbo at ~6 TFLOP/s
effective (≈370 GFLOP at 384² in 63 ms) against the 38 TOPS Apple quotes for the M4
generation's Neural Engine, and
int8 weights alone bought only 6-9%, so the ANE is not starved for weight bandwidth; the
compute path is the lever. If the deep path drops from ~30 to ~18 ms (guess), idea 1's
pipeline becomes GPU-bound near 33 dreams/s.

**Cost.** 1-2 days of work. Calibration took 7.3 minutes per sample for the whole 384² UNet on
CPU (the reason it was aborted); the deep path is smaller, so 32 samples should take ~2-4
hours niced overnight. ~1 GB of disk. Risk: medium. A 1-step model's activations may have
outliers, and the int8 path may not pay off for this op mix.

**Smallest experiment (≤2 h).** Calibrate a proxy (one 1280-channel down block with its
resnets and attention, or the whole deep path if idea 1's model exists) on four samples, time
W8A8 against W8 on CPU_AND_NE, and compare outputs. Success: ≥1.3x at cosine ≥ 0.99.

**Kill it if** it gains 1.1x or less, or the compute plan moves quantized ops off the ANE.

## 8. Fold the constant channels: pruning for one timestep and one world, without training

**Idea.** The engine always runs at t≈599, without guidance, on one family of renders with
eight prompts. Record every convolution and linear channel's output statistics over ~500
captures at that timestep; channels that barely vary fold into the next layer's bias and are
physically removed, greedily, under an output-PSNR budget. Diff-Pruning (Fang, Ma and Wang,
NeurIPS 2023) cut about half the FLOPs of diffusion models at 10-20% of the original
training cost; here the input distribution is a sliver of SD's, so some of that may come
free.

**Why here.** It needs no training, compounds with every other idea and maps where idea 2
should cut. Guess: 1.1-1.4x.

**Cost.** 1-2 days, no disk. Risk: low (at worst, little is constant).

**Smallest experiment (≤2 h).** Forward hooks on the UNet; 200 captures (scenes plus harness
frames); replace the lowest-variance 10, 20 and 30% of each layer's channels with their means
(masking, not removal yet) and measure output PSNR against the unmasked engine. Success: at
least 20% of channels masked at ≥32 dB.

**Kill it if** fewer than 10% of channels can go at 30 dB.

## 9. Latent wire: the capture ends in latent space

**Idea.** Finish the capture pass with a distilled micro-encoder in a fragment shader (an 8x
pixel-unshuffle, then two or three small layers with 32 channels, down to the four latent
channels), trained against TAESD-lite latents on game captures the way
`distill_taesd_lite.py` trained its stems, optionally fed the G-buffer (depth, normal,
surface type). The client reads back 64x40x4 half floats (20 KB, instead of a 655 KB RGBA
readback plus a JPEG encode) and sends them in a new `format` (a protocol extension); the
server skips the JPEG decode, the upload and the VAE encode. A crazier cousin renders the
latent directly: each material's texture pre-encoded into latent mip maps and shaded in
latent space.

**Why here.** On the server it saves upload and encode, 12.7 of 77.8 ms (16%). On the client
it removes the readback and the JPEG encode: 14-28 ms in WebKit and 150-230 ms in headless
Chromium, which is why Chromium dreams only ~5 times a second. The feedback loop also stops
losing detail to two JPEGs and a VAE round trip on every pass (TAESD-lite's round trip is
31.2 dB). The linear encoder blurred structure (R² 0.35-0.6); a small nonlinear net with
G-buffer inputs, judged by its img2img output rather than R², is a different bet. The return
trip should stay JPEG: by my count the TAESD-lite decoder is ~45-50 GFLOP per 512x320 frame
(consistent with its 10.2 ms on MPS), too heavy for WebGL at 20 frames a second.

**Cost.** 3-5 days. Risk: medium (half-float readback in Safari's WebGL; micro-encoder
fidelity).

**Smallest experiment (≤2 h).** Train the micro-encoder offline (5-10 minutes) on scenes naves
and painted frames, plug it into `torch_turbo` as `encoder="micro"`, and run
`bench_engines.py --samples` and `temporal.py` against TAESD-lite. Success: sheets that look
the same at thumbnail size and img2img output within ~1 dB PSNR of TAESD-lite's.

**Kill it if** structure blurs at strength 0.6, as it did with the linear encoder.

## 10. A guided super-resolving decoder, and later quarter-token dreams

**Idea.** Train a TAESD-lite decoder head that outputs twice the resolution and takes the
capture's full-resolution render (colour and depth edges) as a guide: the model decides
colour and material, the renderer supplies the silhouettes. Two operating points: (a) keep
the UNet at 512x320 and decode 1024x640, sharp enough to drop narrow captures; (b) run the
UNet at 256x160, a quarter of the tokens, and decode 512x320.

**Why here.** UNet time scales roughly with tokens: 25.0 ms at 256² against 57.6 at 384²
(RESULTS), so 256x160 might take ~16 ms and a whole dream ~25-30 ms instead of 78, 2.5-3x
(estimate). The guide carries exactly the geometry the play test asked the dream to respect
(a pillar in front of a wall). Point (a) gets idea 3's gain another way. Point (b) runs into
SD-Turbo's resolution floor (at 320² it already turns halls into outdoor vistas at strength
0.7), so it probably needs a student trained at low resolution (idea 2).

**Cost.** 2-4 days; head training like TAESD-lite's (minutes to an hour). Risk: medium for
(a), high for (b).

**Smallest experiment (≤2 h).** `bench_engines.py --sizes 256x160,512x320 --samples` at 0.6 on
scenes and harness frames; upsample the 256x160 dreams bicubically and with a joint bilateral
filter guided by the 512x320 input (thirty lines of numpy); sheet plus edge correlation.
Success: the architecture survives at 256x160 and the guided upsample reaches at least 90% of
the real 512x320 dream's edge correlation.

**Kill it if** halls turn into vistas or mush at 256x160; then only point (a) remains.

## 11. Beat the roofline: Winograd convolutions in custom Metal kernels

**Idea.** Replace the UNet's 3x3 convolutions at the 64x40 and 32x20 grids with a fused
Winograd F(2x2,3x3) kernel (input transform, batched simdgroup-matrix GEMM, output
transform, bias and SiLU in the epilogue), compiled at runtime with
`torch.mps.compile_shader`. Winograd computes a 3x3 convolution with 2.25x fewer multiplies
(Lavin and Gray, CVPR 2016).

**Why here.** RESULTS put the UNet at ~70% of the GPU's peak and concluded that wins must come
from doing less math; channels_last and torch.compile were slower. Winograd is less math, the
one kernel-level trick that can beat that roofline. Guess: 1.2-1.4x on the UNet if 3x3
convolutions are ~60% of its time. MPS may already use Winograd for some shapes; the test
settles that.

**Cost.** 1-2 weeks. Risk: medium-high (fp16 is fine for F(2,3) but not F(4,3), and MPS's
convolutions are well tuned).

**Smallest experiment (≤2 h).** Time each UNet module with syncs to get the convolution share,
then write a naive F(2x2,3x3) kernel for one 320→320 convolution on the 64x40 grid and time it
against `F.conv2d`. Success: max error under 1e-2 in fp16 and at least 1.3x faster for that
shape.

**Kill it if** it is slower than MPS for the best shape, or convolutions are less than 40% of
UNet time.

## 12. Dream only what's missing: a coverage-driven capture planner

**Idea.** The client already knows, for every screen pixel, how much the live views cover and
how fine their pixels are there (`liveComposite`'s coverage and footprint). Use that to choose
each capture: an off-axis sub-rectangle aimed at the worst-covered region (the entering edge
while turning, the widening border while walking), in one of a few fixed engine shapes
(512x320, 320x320, 256x320), and fewer captures when coverage and resolution are already
good. StreamDiffusion's Stochastic Similarity Filter (Kodaira et al., 2023) skipped
near-identical input frames and cut GPU use 2.39x on an RTX 3060 on static input; this is the
geometry-aware version.

**Why here.** What counts is fresh paint per GPU-second, not dreams per second. At rest the
extra dreams measurably hurt (standing change 0.58 with `?rate=8` against 0.85 uncapped,
known issue 1), and while turning much of each full-width capture re-dreams pixels that a
fresh live view already covers (guess). It also frees the GPU for Safari when the game is
played on the Mac mini itself.

**Cost.** 3-5 days. Risk: medium. Sub-rectangle dreams lose global context and can seam
against their neighbours (the lens problem in another form), and each shape needs a warmup and
its own DeepCache stream.

**Smallest experiment (≤2 h).** Client only: during `tools/shoot.mjs` walks and turns, read
back a 64x40 mask from a capture-shader variant marking pixels the live layer already covers
at equal or finer resolution, and log the covered share of each capture. Success: at least
40% already finely covered while walking or turning (room for 1.5x or more).

**Kill it if** less than 20% is covered: captures are mostly new and there is nothing to save.

## 13. CRAZY: a dreamer small enough to live in the page

**Idea.** Distill a 20-60M-parameter one-step image-to-image generator (a conditional GAN
student trained on paired teacher outputs with a latent perceptual loss and an adversarial
loss, with the G-buffer as extra input) and run it in the browser with WebGPU compute, which
Safari 26 turns on by default on macOS and iOS, half floats included. No round trip, no JPEG,
and a phone dreams on its own GPU.

**Why here.** Round trips of 182-250 ms are why results land where you were a moment ago. A
local student of 10-20 GFLOP at ~30% shader efficiency would take ~5-10 ms per dream on an
M-series GPU (guess). Diffusion2GAN (Kang et al., ECCV 2024) distilled a diffusion model into
a one-step conditional GAN from paired noise-to-image outputs, with a cheap latent
perceptual loss (E-LatentLPIPS), and beat one-step SDXL-Turbo and SDXL-Lightning, but its
student was as big as its teacher. Tiny real-time image-to-image networks exist for fixed
styles; the narrowness of this game's domain (eight prompts, one level family, one strength)
is the only reason a tiny dreamer is plausible.

**Cost.** Weeks to months: 200k+ teacher pairs (~4 GB as latents), many overnight training
runs, and a hand-written WGSL inference path in a client that has no build step. Risk: very
high. Small generators collapse to an average look, and the dream becomes a style filter.

**Smallest experiment (≤2 h).** A capacity probe: train a ~5M-parameter, pix2pix-sized U-Net
on MPS for an hour against the teacher in one zone (2k pairs, one prompt, fixed noise) with an
L1 loss plus a TAESD-latent perceptual proxy; compare held-out PSNR with a colour-graded-input
baseline and look at the sheet. Success: ≥25 dB with teacher-like texture rather than blur,
and the loss still falling.

**Kill it if** it plateaus in blurry averages below ~22 dB.

## 14. CRAZY: an eventful UNet that recomputes only the tokens the world says changed

**Idea.** Keep every layer's activations from a stream's previous frame. For a new frame,
warp them with the client's exact reprojection (idea 4), mark as dirty the tokens that are
disoccluded, newly on screen, or whose input changed beyond a threshold, and push only those
through the network: sparse gathered convolutions (SIGE-style tiles) and attention in which
clean tokens reuse cached K/V (the token gating of Eventful Transformers), with a periodic
full refresh against drift.

**Why here.** SIGE (Li et al., NeurIPS 2022) measured 4.6x for DDPM on an M1 Pro GPU and 7.2x
for Stable Diffusion on an RTX 3090 when edits touch a small region; Eventful Transformers
(Dutson et al., ICCV 2023) cut video transformer cost 2-4x with little accuracy loss. Here a
turn at ~1 rad/s at 12 dreams a second brings roughly 5% new columns per dream (estimate), and
turning is exactly the case that forces full passes today. Maybe 3-5x in motion (guess), if
warped activations stay valid.

**Cost.** 4-8 weeks: custom Metal gather/scatter convolutions and a rewrite of
`lean_unet_forward`. Risk: very high. Receptive fields grow the dirty set level by level (at
the 8x5 grid nearly everything is dirty), warped activations drift, and sparse kernels may
not beat dense MPS at these sizes.

**Smallest experiment (≤2 h).** Measure the dirty share per level: on `scenes.turn` and
`scenes.walk` with exact warps, run the UNet on consecutive frames and count, per level, the
tokens whose warped previous activation differs from the fresh one by more than a threshold,
dilated by that level's receptive field. Success: at most 30% dirty on the 64x40 and 32x20
grids for typical deltas.

**Kill it if** more than 60% is dirty on those grids.

## 15. CRAZY: leave SD's 8x latent for a deep-compression backbone

**Idea.** Swap SD-Turbo and TAESD for a few-step transformer on a deep-compression
autoencoder. DC-AE (Chen et al., ICLR 2025) compresses 32x per side, so 512x320 becomes
16x10 = 160 tokens instead of SD's 2560. SANA-Sprint (Chen et al., ICCV 2025) is a 0.6B or
1.6B model on that latent, distilled to work at any of 1-4 steps (continuous-time consistency
plus adversarial distillation). The zone prompts' embeddings can be computed once offline, so
its large text encoder never loads at runtime.

**Why here.** Token count is the root of the UNet's cost. A 0.6B transformer over 160 tokens
is ~0.19 TFLOP (2 x parameters x tokens) against ~0.36-0.4 TFLOP for SD-Turbo at 512x320
(estimate from RESULTS' 339 GMAC at 512²): ~2x on the backbone before any student, more for a
student distilled on that latent. Unknown: the DC-AE decoder's cost at this size, which could
eat the gain.

**Cost.** Weeks; 2-6 GB of downloads (the owner approves anything over ~5 GB); the weights'
licence must be checked before anything ships publicly. Risk: very high. One-step students
can fail at intermediate timesteps exactly as SDXS did (being trained for several step counts
may help, which is the thing to test), and a 32x latent may not keep a thin pillar's edge.

**Smallest experiment (≤2 h).** The autoencoder first (a small download): encode and decode 50
game captures at 512x320 through DC-AE f32, score PSNR, sheet the thin structures, and time
the decoder on MPS. Success: pillar and arch edges intact at ≥27 dB and a decode of 15 ms or
less. Only then try SDEdit at t≈0.6 with the transformer.

**Kill it if** the round trip smears pillars and arches, or the decode costs more than 25 ms.

## 16. CRAZY: a dream light field that never re-dreams what was already dreamt

**Idea.** Keep every centre result (image, capture depth and camera, which is what a live view
holds) in a spatial index instead of dropping all but the newest five. At display time pick
the best few for the current pose with Unstructured Lumigraph weights (angle, distance,
resolution; Buehler et al., SIGGRAPH 2001) and project them exactly as the live layer does.
Idle and at-rest dream cycles go to speculative views (behind you, down the corridor, round
the corner), so a turn lands on sharp paint. The live model's job shrinks to what's new and to
slow evolution.

**Why here.** Today a place you walked through falls back to the soft atlas (~13 texels/m)
once its live views age out (2.5 s), so every one of those ~80 ms dreams is thrown away. With
a view cache, revisits and turn-arounds show paint at capture resolution for no model time.
Storage: ~100-150 KB per view with full-resolution depth (10k views ≈ 1-1.5 GB in IndexedDB,
as `memory.js` already stores the atlas); 20 resident views ≈ 26 MB of GPU memory.

**Cost.** 2-4 weeks. Risk: high, for taste. Views dreamt at different times disagree and pop
when the selection changes, the dream may feel less alive, and stale views need slow
re-dreaming.

**Smallest experiment (≤2 h).** In the harness, log every result's pose, replay a tour that
returns to its start, and count for each frame of the return leg how many stored views cover
at least 80% of the screen at no worse than 1.5x the current live resolution. Success: at
least half the return frames fully covered.

**Kill it if** fewer than 20% are covered: poses rarely repeat closely enough.

---

## Considered, not proposed

- **Token merging (ToMe).** With ToDo on, top-level self-attention is a small share of the
  GPU UNet: dropping it entirely saved ~5 ms of ~129 at 512² (`ablate_unet.py` bound).
  ToMe for SD's 2x (Bolya and Hoffman, 2023) came from 50-step 512² runs where attention
  dominates.
- **An MLX or hand-built runtime.** The UNet already runs at ~70% of peak on MPS, so a better
  runtime tops out near 1.4x without less math (idea 11 is the less-math version).
- **Batching wide and narrow captures into one call.** Compute-bound, about 1.1x at best.
- **Splitting the image between ANE and GPU** (DistriFusion-style patches). The engines'
  styles differ at 30-33 dB, which would put a seam down the middle of the screen.
- **Hardware JPEG or video codecs on the wire.** The CPU isn't the bottleneck; the server
  measured 0.999 efficiency.
- **Core AI** (WWDC26's successor to Core ML). It needs macOS 27, and upgrading the machine
  is the owner's call.

## How they fit together

Ideas 1, 3 and 4 stack: fresh deep features from the ANE, one capture stream, and caches
that follow the geometry. Idea 2 compounds with all of them and leads to 13. Idea 7 speeds up
idea 1's ANE half, and idea 8 shows idea 2 where to cut. If only one thing gets built this
week, build idea 1's two-hour experiment: it decides whether the Neural Engine can carry
most of the model while the GPU keeps rendering.
