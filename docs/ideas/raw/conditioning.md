# Lens: model conditioning and consistency · Claude Opus 5.5 (`claude-opus-5-5`) · 2026-09-30

Raw output of one of five idea generators for the M115 idea session. The subject is how
SD-Turbo is told what to paint: paint that respects the geometry (a pillar should stand
clear of the wall behind it), keeps its identity across time and viewpoints, and stays
intense. Engine numbers come from `docs/STATUS.md` and `bench/RESULTS*.md`. Anything
marked *guess* has not been measured. Nothing here was run; every idea ends with the
smallest test that would tell. Code references are to the build the owner played
(`d4cf3a1c`). M114 was changing the capture shader in the same checkout while this was
written (uncommitted: a depth-separation cue, edge-aware feedback, and a quarter-resolution
inverse-depth pass of each capture); ideas 1 and 4 note where that overlaps.

## What the code says

1. **The model is never told depth, and the nave's pillars share the wall's material.**
   `levelcore.js` builds the nave's columns, arcade walls and vaults all as `stone`
   (`b.column(..., 'stone')`, `b.archWall(..., 'stone')`, `b.vault(..., 'stone')`). In
   `CAPTURE_FRAG`, only point-light falloff, the rim term (`albedo * fres * amb * 0.8`)
   and fog at 0.4x density separate a pillar from the wall behind it.
2. **At rest, the feedback loop erodes edges.** The capture is `mix(raw, painted, fb)`
   with fb ≈ 0.55 everywhere (`fb = uFeedback * smoothstep(0.05, 0.5, conf)`). Once
   one pass merges the pillar into the wall, every later pass is shown the merge.
   Walking hides the problem because parallax moves the pillar's paint and the wall's
   paint differently, which matches the owner's "it just completely blends into the wall
   behind it unless I move".
3. **Painted structure that isn't in the geometry swims.** Paint is projected onto real
   surfaces, so an arch painted on a flat wall slides under parallax. The comments in
   `zones.js` record this for the atrium: the model "painted beams onto the flat planes,
   and they drifted as you walked". Fidelity to the geometry therefore steadies motion
   as well as improving the look.
4. **Strength trades geometry for dreaming.** RESULTS: SD-Turbo "keeps perspective,
   arches and light pools at 0.3–0.5 and dreams hard but coherently at 0.7", and at 320
   px it "starts turning halls into outdoor vistas at 0.7". A geometry input would
   separate the two, so strength could rise while the depth input holds the pillar in
   place.
5. **Budget and placement.** The engine is GPU-bound at ~70% of fp16 peak: 77.8 ms per
   512×320 frame, 62.7 ms of it UNet, so wins have to come from doing less math. A
   DeepCache cheap pass runs only `conv_in`, the top down level and the last up block.
   Conditioning that enters at `conv_in` or at the top level is fresh on every frame;
   conditioning that enters deep blocks lands only on full passes (about 2 frames in 5
   in play, where `dc_reuse` ≈ 0.6).
6. **The lens is partly a conditioning split.** Narrow captures use seed + 1, so they get
   their own noise, DeepCache state and cross-frame anchor. At rest they converge to
   their own picture over the middle ~35% of the screen (1/1.7²).
7. **Lineage.** The cached `stabilityai/sd-turbo` is epsilon-prediction with
   `sample_size` 64 (checked in its UNet and scheduler configs), so it descends from SD
   2.1-*base* (512). Weights made for SD 2.1-768 (v-prediction) are a worse fit. Any
   engine-side change must also go into `coreml_turbo` before an ANE + MPS pool runs, or
   the pool shimmers (M112's bar).

## What exists for SD-Turbo's lineage (SD 2.x-base, 512, epsilon)

| Conditioning | Weights for SD 2.1-base at 512? | Source and notes |
|---|---|---|
| Depth-conditioned full model | **Yes** (SD 2.0-base) | `sd2-community/stable-diffusion-2-depth`, a community mirror (Stability deprecated its SD 2.x repos). Resumed from 512-base-ema and fine-tuned 200k steps with an extra zero-initialized input channel for MiDaS (dpt_hybrid) depth. fp16 UNet 1.73 GB. |
| ControlNet-XS depth / canny | **Yes**, 14.2M params | `UmerHA/ConrolNetXS-SD2.1-depth` (`sample_size` 64); diffusers' example pairs the SD 2.1 XS adapters with `stable-diffusion-2-1-base`. Zavadski, Feiden & Rother, 2023. |
| Full ControlNet | Probably not base | `thibaud/controlnet-sd21-{depth,zoedepth,normalbae,ade20k,...}-diffusers`, ~700 MB each. The depth config has `upcast_attention: true`, which is the SD 2.1-768 setting (2.1-base's config lacks it), so it was probably trained on the v-model. That is an inference from the config and unverified. |
| T2I-Adapter | No | TencentARC released SD 1.4/1.5 and SDXL adapters only. |
| IP-Adapter | No | `h94/IP-Adapter`: SD 1.5 and SDXL only. |
| Control-LoRA | No | `stabilityai/control-lora`: SDXL only. |
| One-step translation recipe | Built on SD-Turbo itself | pix2pix-turbo / CycleGAN-Turbo (`GaParmar/img2img-turbo`, Parmar et al., 2024). Task checkpoints only (edge_to_image, day_to_night, ...). |
| Monocular depth (referee, labels) | Model-agnostic | Depth Anything V2 Small (Yang et al., NeurIPS 2024): 24.8M params, Apache-2.0, `depth-anything/Depth-Anything-V2-Small-hf`. Apple's Core ML build is `apple/coreml-depth-anything-v2-small` (49.8 MB fp16; its card gives 24.6 ms on an M3 Max, mostly on the Neural Engine). |

## Shared instruments (build once; most tests below reuse them)

- **G-buffer bench.** `bench/scenes.py` already computes each pixel's ray distance
  (`best`), normal (`n`) and material (`mat`, where 3 = pillar) for its nave. Returning
  them takes about 10 lines, and every engine-side test below can then run without a
  browser. View depth is `best` times the ray's cosine to the forward axis. Storing the
  hit cylinder's index also gives pillar instance ids.
- **Silhouette contrast ratio (SCR).** Mean luma difference 3 px either side of
  G-buffer depth edges (a log-depth jump > 0.15 over 3 px), divided by the mean luma
  difference at 3 px inside smooth regions. This is M114's geometry measure. *Pillar-SCR*
  restricts it to edges whose near side is a column. Report Laplacian variance beside it
  as "detail", so a method can't win by painting outlines on a blur.
- **Depth agreement.** Depth Anything V2 Small on the output, scale-and-shift fitted to
  inverse G-buffer depth. Report Pearson r and depth-edge recall (idea 18).
- **Game corpus.** For tests that need the real capture: a harness script that stands at
  the `tools/shoot.mjs` tour waypoints with `?feedback=0` and dumps the capture camera's
  render, depth, normals and (after idea 3) part ids. About 1–2 h to write, once.
- **Aux data to the server.** The frozen protocol allows extra header fields. A base64
  field of 7–30 KB (inverse depth or ids at 64×40 or 128×80) fits under the 1 MB header
  limit in `app.unpack`.

## Ideas

### 1. Turbo-Depth by weight arithmetic ★ BET

**Idea.** Make SD-Turbo depth-conditioned without training: UNet = SD-Turbo +
λ·(SD2-depth − SD2-base), with `conv_in` widened to 5 input channels. The fifth channel's
kernel comes entirely from SD2-depth. Feed the game's exact depth in that channel, in
the format SD2-depth learned: relative inverse depth at latent resolution, normalized
per frame to [−1, 1] (what diffusers' depth2img pipeline does with MiDaS output). The SD
community makes inpainting versions of fine-tunes with this "add difference" merge; this
applies it across the ADD distillation, and whether it survives is a guess.

**Why here.** Depth enters at `conv_in`, which every DeepCache cheap pass runs, so it is
fresh on every frame and costs ~0 ms. One more input channel in a 3×3 conv is
~7M MACs, against ~210 GMACs for the UNet at 512×320. It tells the model the pillar is a
separate object at another depth, which is exactly what the capture fails to say (note
1), and it should reduce painted structure that swims (note 3). It might also let
strength rise from 0.6 toward 0.75–0.8 for more intensity, or let a smaller capture
(448×256) hold the architecture that only 512×320 holds now (note 4) for more dreams per
second. TEXTure (Richardson et al., SIGGRAPH 2023) used SD2-depth to paint meshes view by
view for this reason: depth-conditioned paint lands on the geometry.

**Cost.** Test: 2 h and a 3.5 GB download (two fp16 UNets, `sd2-community/stable-diffusion-2-depth`
and `sd2-community/stable-diffusion-2-base`, which is the base SD2-depth was resumed
from; delete both after merging). The merged UNet is 1.7 GB. Integration: 1–2 days
(depth aux field, a small client depth readback at 64×40 or 128×80 — M114's
quarter-resolution inverse-depth capture pass may already provide it — the `conv_in`
concat, and a Core ML rebuild with a 5-channel `conv_in` for the pool). Per frame: ~0 ms on the
server, ~0.2–0.5 ms on the client. Risk: medium–high. SD-Turbo is 2.1-base + ADD and
SD2-depth is 2.0-base + depth, so the merge crosses three fine-tunes. The one-step output
may blur or turn grey (the SDXS failure in RESULTS), or the channel may be ignored at
t ≈ 600. Fallback: a short "healing" LoRA trained against stock SD-Turbo outputs plus a
depth term (setup as in idea 12).

**≤2 h test.** `bench/depth_graft.py`: merge tensor by tensor in fp32 on the CPU from the
fp16 sources, with λ ∈ {0.5, 0.75, 1.0} on the body and the depth kernel at full weight.
In the forward pass, `sample = cat([z_t, depth_lat], 1)`. On the G-buffer bench (3
test_set frames plus the 24-frame walk), at strengths 0.5, 0.65 and 0.8, run four arms:
stock SD-Turbo; graft with true depth; graft with flat depth; graft with horizontally
flipped depth. The flipped arm checks whether the model follows the channel: pillars
should appear where the flipped depth puts them. Report pillar-SCR, detail, time and a
sheet.

**Kill.** Detail below 70% of stock at every λ, grey mud, or true, flat and flipped depth
looking the same at 0.65 (the channel is ignored).

### 2. ControlNet-XS depth (14M params) on SD-Turbo

**Idea.** ControlNet-XS (Zavadski, Feiden & Rother, 2023) is a small ControlNet in which
the control network and the base encoder exchange features at every block. With about 1%
of the base model's parameters, its authors report better FID than full ControlNet. The
SD 2.1 depth model has 14.2M params and targets 512 (`sample_size` 64). Attach it to
SD-Turbo's UNet (`UNetControlNetXSModel.from_unet`) and feed G-buffer depth. A doubtful
second arm is thibaud's full SD 2.1 depth/ade20k ControlNets (see the table on their
base model).

**Why here.** This is the standard conditioning path, with published weights at
SD-Turbo's resolution and lineage, and small enough for the frame budget. Unlike idea 1
it leaves SD-Turbo's weights alone, so at scale 0 the look cannot change.

**Cost.** Test: 2 h, 57 MB. Integration: 2–3 days, because the XS adapter has to be
threaded through `lean_unet_forward` and `_down_no_ds`; on cheap passes only the top
level's exchange runs. Per frame: *guess* +5–10% (4–8 ms). Risk: medium. The adapter was
trained against 2.1-base's encoder features, and ADD moved them.

**≤2 h test.** Use the diffusers path without porting: one-step img2img on the G-buffer
bench with true depth as a 3-channel [0, 1] image, conditioning scale {0.5, 1.0, 1.5} ×
strength {0.5, 0.65, 0.8}, beside idea 1's arms. Report pillar-SCR, detail, and time per
call. The time is unported, so it is only an upper bound.

**Kill.** Broken or noisy output at the scales where control shows, or no visible effect
below scale 1.5.

### 3. Semantic G-buffer and attention modulation

**Idea.** The level builder already knows what every triangle is (`b.column`,
`b.archWall`, `b.vault`, `b.stairs`, `b.slab` in `geom.js` / `levelcore.js`). Add a
per-vertex part class and instance id, render them into a 64×40 map with each capture,
and send it as an aux field. In `_FastAttn`, bias self-attention logits so tokens attend
within their own instance (−β across instances, or −β·|Δ log depth|). Bias
cross-attention so column pixels attend to the prompt's "columns" tokens, water pixels to
"water", and so on. DenseDiffusion (Kim, Lee, Kim, Ha & Zhu, ICCV 2023) is training-free:
it modulates self- and cross-attention scores by a segmentation layout, and its authors
report results close to layout-trained models.

**Why here.** The pillar's tokens and the wall's tokens look alike (both `stone`), and at
t ≈ 600 self-attention copies wall texture across the boundary; an instance bias cuts
that path. Binding words to regions also addresses known issue 3 (ghost geometry where
the prompt describes something else), which so far has taken four hand-rewritten zone
prompts (M113).

**Cost.** 1–2 days for the client attribute and id pass and for server plumbing. The
existing `xframe_bias` mask is per key; a per-query mask (Nq × Nk: 2560 × 1280 at level
0) costs more. Per frame: *guess* +3–10% for a float mask in MPS SDPA; measure it. No
weights. Risk: medium; too large a β gives a cut-out "sticker" look.

**≤2 h test.** On the bench, use pillar instance ids from the ray tracer. Add the level-0
self-attention bias and a cross-attention boost for the nave prompt's column, arch and
water tokens; try β ∈ {0, 2, 4, 8} at strengths 0.65 and 0.8. Report pillar-SCR, detail,
time per frame and a sheet.

**Kill.** No SCR gain before halos or cut-out edges appear, or more than 15% frame time.

### 4. Geometry-aware feedback and an edge-contrast floor in the capture

**Idea.** Make the capture's feedback mix depend on geometry. Near depth silhouettes
(found with a depth unsharp mask, |z − blur(z)| / z, as in Luft, Colditz & Deussen,
SIGGRAPH 2006, who used it to add depth cues to images) and on thin near objects, drop
feedback toward ~0.2 so the raw render reasserts the edge on every pass. Keep 0.55+ on
large flat regions. Add a contrast floor: where the capture's luma contrast across a
depth edge is below a target, darken the far side just enough to reach it. If the
fed-back paint already carries the edge, the floor adds nothing, so it cannot compound
around the loop (M114's constraint). Optionally add a per-zone key light (n·L) to the raw
render so round columns read as round against flat walls.

**Why here.** This targets the rest-time mechanism in note 2 directly, at no server
cost. Overlap: M114's uncommitted capture work already darkens depth silhouettes on the
raw render and lowers feedback near them (`uCapDepthSep`, `uCapEdgeKeep`). What this idea
adds is the idempotent contrast floor, the key light, and the SCR test below as the way
to judge both.

**Cost.** Hours to a day. Client: +0.2–0.4 ms per capture (depth prepass plus a
half-resolution blur at 512×320). No disk. Risk: low–medium; outlines or halos may get
painted in, bringing back the blueprint look.

**≤2 h test.** Add `?fbedge=` and `?edgefloor=` flags. Take `tools/shoot.mjs` rest shots
at the nave pillar pose and run 8-zone `--flicker` twice each; compute SCR from the
shots' depth (`snapAt(..., {depth})` returns it). Success: rest SCR up in at least 6 of 8
zones, standing-still change ≤ 0.85, detail not lower.

**Kill.** Halos visible in a blind look, or standing-still change rises.

### 5. One-step trimap: per-pixel strength

**Idea.** TEXTure's keep/refine/generate trimap and Differential Diffusion (Levin & Fried,
2023: a per-pixel change map for img2img, training-free) give each pixel its own
strength. In one step: build z_t with a per-pixel noise amplitude s(x) = base
− a·(depth-edge strength) − b·(converged confidence) + c·(fresh disocclusion), give the
UNet the mean timestep, and invert per pixel: x̂0 = (z_t − σ(x)·ε̂) / α(x). The noise
tensor stays screen-fixed per stream, so the convergence at rest that M112 found
important is untouched; only its amplitude varies. The UNet was trained on uniform
noise, so whether it tolerates a ±0.15 strength field is a *guess*.

**Why here.** Strength currently does two jobs (note 4). Splitting it per pixel keeps
edges at 0.5 while flat walls dream at 0.8. It also lets fresh disocclusions (known issue
4: soft atlas, blueprint showing through) paint hard straight away.

**Cost.** About 30 engine lines and a 64×40 aux byte map; ~0 ms. Risk: medium; ringing or
blotches where the level changes.

**≤2 h test.** On the bench, compare uniform 0.8, uniform 0.55, and a depth-edge map
(0.55 at edges, 0.8 elsewhere, three smoothing radii). Report pillar-SCR, detail in flat
areas, and a close look at the transition band.

**Kill.** Seams or blotches at every smoothing radius that still raises SCR.

### 6. Geometry-anchored cross-frame attention and a location memory ★ BET

**Idea.** `xframe` lets each self-attention layer see the previous frame's keys and
values. They are matched by content only, since SD's self-attention has no positional
encoding. Give each latent token its world position (unprojected from the capture depth;
for ToDo-pooled keys use the nearest sample's position) and bias the cross-frame logits by
3D distance, −‖p_q − p_k‖² / 2σ², with σ scaled to the token's footprint. Each surface
point then attends to what was painted at that point before. FRESCO (Yang, Zhou, Liu &
Loy, CVPR 2024) constrains attention with correspondences in the same way, but it has to
estimate them with optical flow; the game has them exactly. Then replace "last frame per
stream" with a bank keyed by world position and shared by all streams, so narrow and
wide captures and return visits read the same memory. StreamV2V (Liang et al., ICLR 2025)
keeps a merged bank of past keys and values, training-free, at 20 FPS on one A100.

**Why here.** Content-only xframe already cut walking change after motion compensation
by 12% (3.59 → 3.15) for +2% cost; a correspondence prior is the next step toward
"less nauseating". A shared bank works on the lens from the model's side (narrow and wide
paint from one memory). It also keeps the world's identity beyond the five live views,
each of which lives about 0.5 s at rest.

**Cost.** 3–5 days: aux positions (or depth plus camera, ~15 KB per frame), a bank in
`_cross_frame`, and biases at levels 0–1. Memory: one 512×320 frame's keys and values
across the 16 self-attention layers come to ~16.6 MB in fp16 (computed from the layer
shapes with ToDo at level 0), so a 64-frame bank is ~1 GB of the 48 GB. Per frame:
*guess* +3–6%. Risk: medium. Too small a σ freezes the paint; too large a σ gives today's
behaviour. Masked SDPA speed on MPS is unknown.

**≤2 h test.** In `bench/temporal.py` (walk and turn; exact cameras and depth from
`scenes.py`), compare xframe off, content-only, geometry-biased (σ ∈ {0.3, 1, 3} m), and
geometry-biased over the last two frames. Frame-to-frame output change today (RESULTS):
walk 6.51 off / 5.44 on against an input change of 8.07; turn 16.72 / 13.64 against
14.39. Success: at least 15% below content-only on both sequences, with detail within 5%.

**Kill.** Less than 5% gain, or gain only with detail loss.

### 7. Cross-stream anchoring for the lens

**Idea.** Let the narrow stream attend to the latest wide stream's keys and values as well
as its own (optionally only the tokens inside the narrow frustum), and the wide stream to
the narrow's. Keep the separate seeds, so each stream still converges at rest.

**Why here.** This is the owner's first complaint ("a pretty clear rectangle artifact
centered in the middle"). Compositing fixes (M114) hide the seam; this makes the two
paintings agree, so the seam has less to hide. Cost is ~+1–2% (one more 640-token set at
level 0) and a few lines in `_cross_frame`. Caveat (*guess*): the model paints texture at
its own pixel frequency, so bricks may still come out 1.7× finer in the narrow view even
with shared keys and values. The test measures this.

**Cost.** Hours; ~+1–2%; no disk. Risk: low.

**≤1 h test.** On the bench at a fixed pose, render the nave wide (fitted FOV) and narrow
(tan / 1.7). Run 10 alternating passes of each with feedback emulated (previous output
mixed 0.55 with the raw render), with and without cross-stream anchors. Measure the
low-pass colour difference between the narrow output and the upsampled centre crop of the
wide output (Lab, Gaussian σ = 6 px), and the ratio of high-frequency energy between them.
Success: the low-pass difference halves while the high-frequency ratio stays ≥ 1.2.

**Kill.** The narrow view loses its extra detail, or the difference doesn't move.

### 8. Reference-attention looks and zone anchors

**Idea.** Compute a reference image's self-attention keys and values once, at the dream's
timestep, and append them to every frame's self-attention the way xframe does. Add
StyleAligned's AdaIN of queries and keys toward the reference's statistics (Hertz,
Voynov, Fruchter & Cohen-Or, CVPR 2024: shared attention to a reference makes
generations share a style, training-free; the "reference-only" mode of the
sd-webui-controlnet extension does something similar for one image). There are two uses.
A *zone anchor* (the zone's best converged frame) pins the zone's identity against slow
colour drift. *Looks* are reference images in different styles, switched by a header
field.

**Why here.** M114 owes the owner at least four switchable named looks while staying as
intense as `d4cf3a1c`. Looks built from reference keys and values switch instantly, need
no prompt rewrites, and cost about what xframe does (*guess* +2–3%). A zone anchor also
counters DESIGN's warning that more feedback "makes the loop simplify the level and drift
in colour".

**Cost.** 1–2 days; ~16 MB of keys and values per look; no downloads (references can be
SD-Turbo's own txt2img images of style prompts, or frames from `docs/shots/`). Risk:
content leakage, where the reference's objects appear. AdaIN and restricting the keys to
decoder levels reduce it.

**≤2 h test.** Take four references (two converged frames from `docs/shots/m113/` and two
SD-Turbo style images) and apply them to bench frames at 0.65. Measure palette distance
to the reference (Lab histogram EMD), structure kept (edge correlation with the input,
relative to stock), and leakage by eye. Success: 3 of 4 give distinct looks at ≥ 90% of
stock's edge correlation, with no leakage.

**Kill.** Leakage in most looks, or shifts too small to call a look.

### 9. Peeled captures: paint what's behind the pillar before you see it

**Idea.** At rest, spend some captures on a "peel": the same pose rendered without the
occluder classes (columns, fallen drums, balustrades; needs idea 3's tags), so the model
paints the wall behind the pillar. The live layer uses a peel only where the normal view
was occluded; its depth test already rejects the pillar itself. When the player steps
sideways, the revealed wall is already painted, in its surroundings' style.

**Why here.** Known issue 4: fresh disocclusions fall back to the soft atlas and show the
blueprint, and they are the least consistent pixels while walking. The dream rate at rest
is spare capacity. Capping captures at 8/s made standing still calmer (0.85 → 0.58), so
peels would use capacity that currently makes rest busier.

**Cost.** 2–3 days of client work (tagged culling, a peel flag in `live.js`); no server
change; a share of the dream rate at rest only. Risk: medium; peel paint may disagree with
the normal view along the pillar's outline.

**≤2 h test.** Hard-code the nave columns as the peel set. From the pillar rest pose,
record captures with `?peel=1`, then step 0.5 m sideways and measure what share of newly
visible wall pixels are covered by live paint rather than soft atlas, with and without
peels. Take a screenshot pair.

**Kill.** Visible seams around peeled regions, or a coverage gain below 30%.

### 10. Latent feedback loop, with ε feedback as a long shot

**Idea.** Feed the model its own previous latents instead of a re-encoded JPEG of a
reprojected painting. The server returns the output latent (64×40×4 fp16, 20 KB) along
with the JPEG. The client projects it into an RGBA16F live layer (SD's four latent
channels fit one texel), and the capture renders that layer at 1/8 resolution, so z0 is
the reprojected previous latent mixed with the encoded raw render. The long-shot variant
also feeds back the previous pass's predicted ε, reprojected the same way, as part of the
noise. This is a cheap stand-in for inversion; ReNoise (Garibi et al., ECCV 2024) inverts
few-step models like SDXL-Turbo by fixed-point iteration.

**Why here.** Every trip around the loop pays a JPEG encode and a TAESD round trip (31 dB
PSNR, RESULTS), so the model refines a slightly different image from the one it painted.
Latent feedback takes that noise out of the loop (relevant to known issue 1, rest busier
than it should be). If the raw render's share at rest can drop to zero, it also skips
the encoder there (7.9 of 77.8 ms). In M112's two
tries, noise warped with the camera changed the noise–image alignment every frame and
the model re-invented. The ε variant might let the noise follow the picture without that
failure, since ε̂ is aligned with the image by construction (*guess*).

**Cost.** 1–2 weeks (float layer, projection, protocol). A 2048² RGBA16F latent atlas is
32 MB and holds 8×8 pixels of detail per texel. Risk: high. Latents don't resample like
pixels, so bilinear warps under perspective may blotch; ε feedback may burn in patterns
or bring back M112's neon blotches (noise that is no longer white).

**≤2 h test.** On the bench walk with exact reprojection, run 24 frames of three loops:
A, the current path (decode → warp RGB → mix 0.55 with raw → encode); B, warp the latent
at 1/8 and mix with encode(raw); and B plus ε feedback (0.3 mix, renormalized). Report the
detail trend, colour drift and frame-to-frame output change.

**Kill.** Blotches or checkerboards from latent warps at walking speed; for ε, burn-in
within 24 frames.

### 11. Learned zone tokens trained on a geometry objective

**Idea.** Learn 2–4 new token embeddings per zone (Textual Inversion, Gal et al., ICLR
2023) and append them to the zone prompt. Train by gradient through the frozen text
encoder and the one-step UNet on the zone's own captures. The loss is a geometry term
(differentiable edge contrast at G-buffer depth edges, or idea 18's depth agreement) plus
an anchor to the stock prompt's outputs (L2 at quarter resolution, or LPIPS) so the look
stays.

**Why here.** Prompts have been the strongest geometry lever so far (M113: naming what
the walk passes won blind reviews in four zones), but they are tuned by hand and judged by
eye. Learned tokens keep that lever and make it measurable, at 0 ms per frame, since
embeddings are cached per prompt.

**Cost.** Days in total; *guess* 15–30 min of training per zone at ~1 s/step; under 1 MB
per zone. Risk: low–medium; tokens that fall off the text manifold, or overfitting to the
training poses.

**≤2 h test.** 32 nave frames (bench or game corpus), 2 tokens, 500 steps. On 8 held-out
poses report pillar-SCR, distance from the anchor, and a sheet. Success: +20% SCR on
held-out poses with the look intact.

**Kill.** Gains only on training poses, or visible artifacts.

### 12. Jump to the fixed point: a geometry-filtered self-distillation LoRA ★ BET · crazy

**Idea.** Standing still, a held framing converges over ~30 passes to a crisp result.
Walking, each view gets about one pass, "the model's misty first take" (DESIGN). Collect
pairs of (the first capture of a new framing, the converged result after ~3 s), with N
candidates per framing (varying seed, feedback, strength, and idea 4's edge-aware
feedback). Keep the best by a geometry score (pillar-SCR plus idea 18's depth agreement)
and stability. Then train a LoRA on SD-Turbo so that one pass lands there. This is
reward-ranked fine-tuning (RAFT, Dong et al., 2023) with the pix2pix-turbo recipe
(Parmar, Park, Narasimhan & Zhu, 2024: LoRA plus a retrained first conv on SD-Turbo
itself, one step, L2 + LPIPS; the authors report edge-to-image results on par with
ControlNet).

**Why here.** Walking would look like standing still (the charter's "sharper paint while
moving"). Feedback could drop, which means less of the loop's simplification and drift.
The geometry filter bakes respect for pillars into the weights. A merged LoRA costs 0 ms,
and the training data comes from the game itself.

**Cost.** 1–3 weeks end to end. Data: ~3 s per framing at ~10 dreams/s, so 1,000
framings × 4 candidates ≈ 3–4 h of harness time and ~0.5 GB. Training: *guess* 0.5–1.5
s/step on MPS at 512×320, so 5k steps take 1–2 h, and sweeps run overnight. LoRA
20–100 MB; the ANE build must be rebuilt to match. Risk: medium. The targets carry the
loop's own drift, and MPS backward speed and memory are unmeasured.

**≤2 h test.** 64 nave framings; a rank-8 LoRA on level-0 attention plus `conv_in`, 500
steps. On 16 held-out framings, compare stock and LoRA one-pass outputs against the
held-out converged targets (L2 at quarter resolution). Log the training step time, which
sizes the real run. Success: at least 25% closer to target, with no colour shift.

**Kill.** No gain at 500 steps, or a step time above 5 s (a real run would not fit in a
night).

### 13. Depth as a differentiable reward (DRaFT for a one-step model) · crazy

**Idea.** For a one-step model, reward fine-tuning is backprop through a single UNet call.
DRaFT (Clark, Vicol, Swersky & Fleet, ICLR 2024) fine-tuned SD with LoRA by
backpropagating differentiable rewards through sampling, and documented reward hacking.
Here the reward is agreement between Depth Anything V2's depth of the output and the exact
inverse G-buffer depth (scale-and-shift invariant), plus edge agreement. The regularizer
is distance to the stock output.

**Why here.** "Does the painting read as a pillar in front of a wall" is a perception
question, and a monocular depth network is a differentiable stand-in for it. Unlike idea
12, it doesn't need the loop to have found a good answer first. A merged LoRA costs 0 ms.

**Cost.** 1–3 weeks. Depth Anything V2 Small (99 MB fp32) sits in the graph; *guess* 2–4
s/step. Risk: high. Reward hacking (dark outlines, vignettes, fog gradients that fool the
depth network) and a pull toward photographic cues.

**≤2 h test.** 20 nave frames, a rank-4 LoRA, 200 steps. On 10 held-out frames report
depth r, pillar-SCR, distance from stock, and a sheet checked for outline hacks.

**Kill.** Gains only through visible outlines or vignettes, or a step time above 10 s.

### 14. Warp-consistency LoRA from exact correspondences · crazy

**Idea.** The game has exact dense correspondences between any two captures (depth plus
cameras), which video models trained on real footage never get. Fine-tune a LoRA with a
two-frame loss: the output at t+1 should match the output at t warped through the
G-buffer, wherever both see the same surface. Anchor it to the stock model's detail
(LPIPS to stock plus matched high-frequency energy) so it can't win by blurring, the usual
failure of temporal losses (Lai et al., ECCV 2018, "Learning Blind Video Temporal
Consistency"). Train it through the real capture pipeline so it learns to be a stable
iteration.

**Why here.** The queasiness comes from re-invention between views. Walking change after
motion compensation is 3.18 (on a 0–255 scale), 3.7× the standing-still figure of 0.85. A
merged LoRA costs 0 ms.

**Cost.** 2–4 weeks; two UNet passes per step; data from harness walks with depth. Risk:
high; collapse to low detail, or a dream that stops dreaming.

**≤2 h test.** Bench walk pairs with exact warps, a rank-8 LoRA, 300 steps. On a held-out
walk with another palette, report `temporal.py`'s frame-to-frame output change and
detail. Success: −20% change with at most 5% detail loss.

**Kill.** Flicker drops only when detail does.

### 15. Our own G-buffer adapter, trained on self-generated data · crazy

**Idea.** Build a one-way, T2I-Adapter-style adapter (Mou et al., AAAI 2024; here 2–15M
params). It takes inverse depth, normals, part classes and an edge map, and adds
residuals to the UNet's encoder features at 1/8 and 1/16 resolution. No dataset download
is needed: generate 20–50k SD-Turbo txt2img images from architecture prompts (the zone
prompts and variants) and label them with Depth Anything V2, deriving normals from depth
gradients. Train with a one-step objective (noise to t ∈ [500, 999], predict x0, L2 plus a
perceptual term). Because it is one-way, a larger version could run on the Neural Engine
one frame ahead: RESULTS-coreml measured that an MPS UNet running alongside changes ANE
throughput by less than 1%.

**Why here.** This would be a conditioning path built for this model, this speed and this
game's inputs. No published model takes normals and part classes together. *Guess*: +1–3%
per frame.

**Cost.** Weeks; 1–3 GB of data; overnight training runs (*guess* 1–2 s/step at 256²,
batch 4). Risk: high. One-step L2 training blurs, and Depth Anything's depth of generated
images is softer than the game's exact depth.

**≤2 h test.** Generate 2k images (~5 min) and label them (~2 min); train a 1M-param
adapter at 1/8 resolution only, for 45 min at 256². On held-out images at t = 750, check
whether the adapter raises the correlation between the output's estimated depth and the
input depth. Log the step time.

**Kill.** A step time above 5 s (a real run won't fit), or no gain in correlation.

### 16. Switch lineage to SD 1.5 plus a one-step consistency LoRA

**Idea.** SD-Turbo's lineage has one small depth adapter (idea 2) and full ControlNets of
doubtful fit. SD 1.5 has much more: ControlNet 1.1 depth, normal and ADE20K segmentation
(ADE20K has a "column, pillar" class); T2I-Adapters
(`TencentARC/t2iadapter_depth_sd15v2`); IP-Adapter (`h94`) for image-prompt looks; and
one-step consistency LoRAs such as Hyper-SD (Ren et al., 2024), whose SD 1.5 one-step
unified LoRA ByteDance lists as ControlNet-compatible (tested on scribble and canny).
Consistency models are trained to map any timestep to x0, so img2img at t ≈ 600 should
work, unlike SDXS, which is valid only at t = 999 (RESULTS: grey mud); that is a *guess*
to test. SD 1.x and 2.x share TAESD's latent space, so TAESD-lite carries over.

**Why here.** One switch brings depth, normals, segmentation driven by the game's part
map, and image-prompt looks, all with published weights, at adapter cost (*guess*
+2–5%).

**Cost.** Test: 2 h, ~2.5 GB (SD 1.5 fp16 UNet and text encoder from the
`stable-diffusion-v1-5` mirror, the LoRA, and the depth adapter). A new engine plus
retuned prompts: about a week. Risk: medium–high; the one-step look may be worse than
SD-Turbo's.

**≤2 h test.** Bench frames at 0.5, 0.65 and 0.8: SD 1.5 + Hyper-SD one-step, with and
without the depth adapter, beside SD-Turbo. Report UNet time at 512×320, pillar-SCR, and
a blind pick.

**Kill.** Mud at intermediate t, or a clearly worse look in the blind pick.

### 17. Bake the dream into a neural texture, with the model as teacher · crazy

**Idea.** Per zone, fit a learned feature atlas and a tiny deferred renderer (Deferred
Neural Rendering, Thies, Zollhöfer & Nießner, SIGGRAPH 2019) to the model's outputs from
many poses. Iterate the way Instruct-NeRF2NeRF does (Haque et al., ICCV 2023: re-edit the
training views with the 2D model while fitting the 3D one, so per-view inconsistencies
average into one consistent scene), with the model conditioned on the current neural
render through the capture feedback. In play, the neural texture renders at display rate
in the client, and live diffusion nudges it.

**Why here.** The result would be one consistent world instead of five live views over a
13 texels/m atlas. The pillar would look the same from every angle, and the phone's soft
2048 atlas (5.4 texels/m, known issue 8) would get learned detail. It is also the
consolidation step that the EMA atlas only approximates.

**Cost.** Weeks; hours of fitting per zone; *guess* 1–3 ms per frame for the client
shader at 1080p. Risk: high. A tiny renderer holds far less detail than the model paints,
and the result may feel baked rather than dreamt.

**≤2 h test.** No differentiable rasterizer is needed: have the harness dump per-pixel
atlas UVs with each (pose, dream) pair. Fit a feature atlas and a three-layer 1×1-conv
decoder to 200 pairs in one room, then report held-out-view LPIPS and render a walk.
Success: held-out LPIPS below 0.3 and a steady walk. If it can't fit, that failure
measures how inconsistent the views are.

**Kill.** Mush at every atlas resolution.

### 18. A monocular-depth referee: a metric first, then a controller

**Idea.** Run Depth Anything V2 Small on dream outputs and compare the result with the
capture's exact depth. First use it as the geometry metric the ideas above need. Later
run it in the loop at ~2 Hz on the Neural Engine through Apple's Core ML build, and return
a low-resolution fidelity map marking where the painting doesn't read as the geometry.
The client then lowers feedback or raises cue strength there (ideas 4 and 5).

**Why here.** The loop has no geometry sensor. M114's geometry bar needs a measure, and
ideas 12 and 13 need a score. On the ANE it stays out of the GPU budget.

**Cost.** Metric: hours, ~100 MB. Controller: days. Risk: the network may read
painted-on arches as real geometry, in which case it cannot referee painted structure.

**≤2 h test.** On bench outputs at strengths 0.4, 0.6 and 0.8 (higher strength loses
pillars), check whether depth r and depth-edge recall fall monotonically with strength,
and whether they flag the frames where pillars vanish (judged by eye). Also run it on
frames with painted arches, to see whether it reports arches that aren't in the G-buffer.

**Kill.** No separation between strengths, or it rewards painted structure that isn't
there.

### 19. Negative prompts that work in one step (Value Sign Flip)

**Idea.** SD-Turbo runs without CFG, so each zone's negative list is ignored (known issue
6: ghostly figures in the vestibule, dark figures in the baths' far arch). VSF (Guo & Du,
2025) is negative guidance for few-step models: it appends the negative prompt's
cross-attention keys and values with the values' sign flipped. NASA (Nguyen et al., 2024)
subtracts a scaled negative attention output instead. `_FastAttn` already handles
cross-attention and caches its keys and values per conditioning, so this is ~30 lines and
77 extra tokens.

**Why here.** It removes figures without rewriting prompts, for ~+1–2% (cross-attention is
small next to self-attention). VSF was shown on SD 3.5 Turbo and Wan, not on SD 2.x UNets,
so it is untested here.

**Cost.** Hours; no disk. Risk: low, unless the look degrades at the scales that remove
figures.

**≤2 h test.** Use frames where figures appear (the vestibule and baths far arch from
`shoot.mjs`, or bench prompts that invite figures at 0.8). Try VSF scales {0.5, 1, 2},
count figures blind over 40 frames, and check the look.

**Kill.** The look degrades before the figures go.

## Summary

| # | Idea | Targets | Per frame | Build | Risk | Mark |
|---|---|---|---|---|---|---|
| 1 | Turbo-Depth by weight arithmetic | pillar, intensity, drift | ~0 ms | 2 h test; 1–2 d | med–high | ★ bet |
| 2 | ControlNet-XS depth | pillar, intensity | +5–10% *guess* | 2 h; 2–3 d | med | |
| 3 | Semantic G-buffer + attention modulation | pillar, ghost geometry | +3–10% *guess* | 1–2 d | med | |
| 4 | Geometry-aware feedback + contrast floor (feedback part in M114) | pillar at rest | client +0.2–0.4 ms/capture | hours–1 d | low–med | |
| 5 | One-step trimap (per-pixel strength) | pillar, disocclusions | ~0 | ~1 d | med | |
| 6 | Geometry-anchored xframe + location memory | nausea, identity, lens | +3–6% *guess* | 3–5 d | med | ★ bet |
| 7 | Cross-stream anchoring | lens | +1–2% | hours | low | |
| 8 | Reference-attention looks, zone anchors | looks, colour drift | +2–3% *guess* | 1–2 d | low–med | |
| 9 | Peeled captures | disocclusions while walking | rest dream share | 2–3 d | med | |
| 10 | Latent (and ε) feedback | rest calm, loop noise | −8 to +1 ms | 1–2 wk | high | |
| 11 | Learned zone tokens | pillar, ghost geometry | 0 | days | low–med | |
| 12 | Geometry-filtered fixed-point LoRA | walking look, pillar | 0 | 1–3 wk | med | ★ bet · crazy |
| 13 | Depth-reward LoRA | pillar | 0 | 1–3 wk | high | crazy |
| 14 | Warp-consistency LoRA | nausea | 0 | 2–4 wk | high | crazy |
| 15 | Own G-buffer adapter | pillar, part classes | +1–3% *guess* | weeks | high | crazy |
| 16 | SD 1.5 lineage switch | pillar, looks | +2–5% *guess* | ~1 wk | med–high | |
| 17 | Neural-texture bake | identity, phone | client 1–3 ms *guess* | weeks | high | crazy |
| 18 | Monocular-depth referee | metric, pillar | 0 GPU (ANE) | hours / days | low–med | |
| 19 | One-step negatives (VSF) | figures | +1–2% | hours | low | |

## The three bets

1. **Idea 1, Turbo-Depth by weight arithmetic.** It is the cheapest path to a real depth
   input (0 ms per frame, 3.5 GB to test, 2 hours on the bench). If it holds, it changes
   the pillar, the swimming of painted structure, and the intensity trade-off at once. If
   the raw merge blurs, idea 2 answers the same question in the same session, and a
   healing LoRA is the fallback.
2. **Idea 6, geometry-anchored cross-frame attention and location memory (idea 7 is its
   first hour).** It is training-free and extends the one consistency mechanism that
   has already measured well (−12% walking change). It uses the game's exact
   correspondences, which no video method gets, and it works on the lens and on
   identity across visits.
3. **Idea 12, geometry-filtered self-distillation.** The weeks-scale bet. Its training
   data is free, it is supervised rather than reward-hacked, and it goes after the
   biggest remaining gap between walking and standing still while baking in geometry at
   0 ms.

## Considered and dropped

- **Noise that moves with the world, in any form that warps a noise field with the
  camera.** Two M112 attempts (nearest-warped, and integral noise warping after Chang,
  Tang, Gross & Azevedo, ICLR 2024) were less stable than screen-fixed noise, because the
  model converges when it sees identical inputs under identical noise. Idea 5 varies only
  the noise's amplitude, and idea 10's ε variant is labelled a long shot.
- **Depth of field in the capture as a depth cue.** The blur would be painted onto the far
  surfaces and projected into the atlas.
- **A G-buffer-aware decoder that sharpens silhouettes.** The live layer already cuts
  paint at silhouettes per display pixel with its PCF depth test. The pillar's problem is
  what gets painted on it, not where the paint stops.
- **IP-Adapter, T2I-Adapter or Control-LoRA for SD 2.1.** No such weights exist (see the
  table); ideas 8, 15 and 16 cover the same ground.
