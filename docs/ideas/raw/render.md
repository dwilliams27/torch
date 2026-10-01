# Rendering lens: Claude Opus 5.5 (claude-opus-5-5), 2026-09-30

*Raw ideas for the M115 idea session. The lens: how the game turns dreams into what you see,
and how it could use the scene's geometry (depth, normals, materials, lights, parallax,
occlusion, time) far more fully. Nothing here has been run. Figures from the repo cite their
source. Anything marked* guess *is one.*

M114 is building a lens fix, a depth-aware separation pass and a running average per held
framing right now. Ideas 1-8 are concrete candidates for those slots, each with an experiment
that can choose between them. Ideas 9-17 go further, and 13-17 would take weeks.

**Overlap with M114's working tree.** As of this writing, M114's uncommitted changes include
several of these ideas:

- a per-axis foveal warp (`liveWarp` in `live.js`, with a resample pass in `dream.js`): idea 8;
- a two-scale inverse-depth separation, on the display (`post.js`) and on the raw capture
  before feedback: idea 3;
- the simplest form of idea 2 (`uRelight`, limb darkening by |n·v|) and a capture-side lantern;
- less feedback at silhouettes (`uCapEdgeKeep`).

Entries 3 and 8 therefore serve as review and evaluation plans. Entry 2's new part is the
light-aware ratio, AO and a lantern on the display.

## First: the exchange rate, and some GPU to take back

- **Display milliseconds cost dreams.** The display is cheap (frame p50 ~1.2 ms at 1080p,
  STATUS), but the browser and PyTorch MPS share one GPU, and the UNet already runs at ~70% of
  the GPU's fp16 peak (`bench/RESULTS.md`). *Guess:* each 1 ms of display GPU per frame at
  60 Hz takes ~6% of the GPU, so up to ~6% of the dream rate. At the 4K display's 2.6 MP pixel
  budget, per-pixel costs are ~1.25x the 1080p figures. Calibrate once (~30 min): a
  `?gpuburn=ms` pass that spins a fragment loop for a fixed time, then the harness dream rate
  at 0, 2 and 4 ms. Every cost below is in frame ms, and that slope converts it to dreams/s.
- **Atlas mips are rebuilt on every paint.** The vendored three.js regenerates a mipmapped
  render target's whole mip chain at the end of every `render()` into it
  (`updateRenderTargetMipmap`). The atlas is a 4096² half-float target with mipmaps (128 MiB
  at level 0). Each paint call rebuilds all of it, and there are up to 3 a frame (~30 a
  second at 10 dreams/s, since each result is spread over 3 frames), plus one per relax pass.
  The display reads atlas mips only for fill and blur, so a rebuild every ~100 ms would do:
  turn mipmaps off on the target and call `generateMipmap` on a timer. *Guess:* a few tenths
  of a ms per rebuild. Measure it with `--perf`, and check it against known issue 5 (a
  10-15 ms frame about a second after the first dream).
- **Texture units.** The display shader uses 13 of the 16 guaranteed units (6 live views x
  image + depth, plus the atlas). Extra mip levels of an existing texture are free; new
  textures are not.

## The ideas

### 1. Tone from the world, detail from the view [BET]

**Idea.** Split every painted pixel into two bands. The low band (colour, brightness,
anything coarser than ~0.4 m on the surface) comes from one consistent source: the newest wide
view, or better, the atlas at a coarse mip, which fuses every result per surface. The high
band comes from the live views, each minus its own low-pass at the same footprint. A narrow
(centre) view can then add only detail, never its own tone, and a new dream can change large
areas only slowly.

**Why it could be big here.** The lens is two pictures with two tones. Narrow captures have
their own seed, DeepCache stream and cross-frame anchor ("each width keeps following its own
last picture", DESIGN), and in `liveComposite` a wide view composited over a narrow one gets
~10% weight. So the middle third of the screen shows the narrow stream's picture, faded
into the wide one over a band. Panorama stitching solves the same problem (overlapping photos
with different exposure) with gain compensation and multi-band blending (Brown & Lowe 2007).
Burt & Adelson (1983) showed that blending each band over a transition width matched to its
wavelength hides seams that a single feathered mask cannot. It also targets the queasiness:
people are most sensitive to flicker in large areas (low spatial frequency) changing at
roughly 5-15 Hz, and less to fine detail changing at the same rate (Robson 1966; Kelly 1979).
9-12 dreams/s sits in that band. A bonus of the atlas version: charts are per surface, so the
low band never bleeds from a pillar onto the wall behind it, as a screen-space blur would.

**Cost.** 1-3 days: a flag, tuning, then a slow "tone atlas" (1024², painted at ~0.1 per
result, time constant ~1 s; the main atlas takes ~0.65 per result, which is fast). GPU:
mipmaps on 512x320 results (negligible) and one `textureLod` per live view. *Guess:*
+0.1-0.3 ms at 1080p. Risk: medium.

**Smallest experiment (≤2 h).** `?bands=1`, display shader only. Result textures get mipmaps
(`dream.js _onResult`), and `liveComposite` also accumulates `textureLod(img, iuv, lod_i)` with
the same weights, where `lod_i = log2(F / fp_i)` and F ≈ 0.44 m (atlas mip 2.5 at 12.85
texels/m). Variant A: low band from the newest wide view (pure gain compensation, ~30 lines).
Variant B: low band from the atlas. Its fast EMA may flip between the two streams' tones,
which is why the slow tone atlas comes next. Shoot WebKit rest frames in 8 zones for the
default, A, B and `?fovea=1` (the lens-free control), and lay them out as a sheet. Run
`--flicker` twice. Success: in ≥6/8 zones where the default shows the lens, you can't find the
centre region by eye; standing change after motion compensation ≤0.72 (now 0.85) with walking
detail within 5% of 2.43; p95 +≤0.3 ms.

**What kills it.** Ghost edges: one picture's detail laid on another's tone, such as a
window's outline floating on a plain wall while walking, where the low band lags. Or detail
falls >10% because the matched low-pass removes mid frequencies that the low source can't
give back.

### 2. The world relights the dream [BET]

**Idea.** Keep the model's colours, but put back the shading that the geometry says must be
there and that the model flattened. Per display fragment, multiply the dream by
(shading with the real normal ÷ shading with the normal turned toward the camera)^β. The ratio
cancels distance falloff and light colour and keeps the part that depends on orientation.
Add ambient occlusion, and optionally a faint "lantern" carried by the player: a light at the
camera whose inverse-square falloff gives a pillar 3 m away ~7x the lantern light of a wall
8 m behind it. A pillar gets its cylinder's lit flank and dark limb and a contact shadow at
its foot. The wall keeps its light pools, and flickering or moving lights slide real shading
across painted surfaces.

**Why it could be big here.** It's the direct answer to "use the geometry of what I'm looking
at more fully". Today painted surfaces get no geometric shading beyond the flicker ratio
(`uLivingLight`: lit ÷ steady). A pillar whose paint continues the wall's texture has nothing
left to separate it until parallax does, which matches the owner's "unless I move". Every
input is already in the display shader (`accumulateLights`, `hemi`, the normal), so the
orientation term needs no extra pass. Precedents: flash/no-flash photography moves one
image's detail and shading layer onto another's colours (Petschnigg et al. 2004; Eisemann &
Durand 2004). StyLit (Fišer et al. 2016) made example-based stylisation follow rendered
illumination channels, so the output reads as the 3D scene. Enhancing Photorealism
Enhancement (Richter, AlHaija & Koltun 2021) fed G-buffers (normals, depth, materials) into
its network to keep output consistent with the game's geometry.

**Cost.** Orientation term and lantern: hours; *guess* ~0.05 ms, since it sits inside the
existing light loop. AO: screen-space GTAO at half resolution (Jimenez et al. 2016), *guess*
0.3-0.5 ms. Or bake per-vertex AO at level generation (the world is static, 38k triangles) and
pay nothing at runtime. Risk: medium. Double shading (the model's painted shading plus ours)
can turn creases muddy or make surfaces look CG.

**Smallest experiment (≤2 h).** `?relight=β&lantern=k` in DISPLAY_FRAG. S is
`luma(amb*hemi + lit)` with the real normal, S0 the same with the normal set to the view
direction, and `painted *= pow(clamp(S/S0, 0.5, 1.6), β)`. The lantern is a point light at the
camera (radius ~6 m). Take rest shots at pillar poses (nave columns, vestibule columns, garden
cypresses, baths arcades) and in all 8 zones, for β in {0, 0.5, 1}, lantern on and off, plus
M114's silhouette-contrast measure. Blind review: "which reads more 3D" and "which looks
better". Success: preferred for depth in ≥6/8 and not worse-looking in ≥5/8, with clear
separation on the pillar sheet at β = 0.5.

**What kills it.** The reviewer calls it muddy, plastic or CG. Or separation doesn't rise
because the model's flat colour dominates (then idea 3).

### 3. Painterly depth halos (unsharp-masking the depth buffer)

**Idea.** In the post pass, compare each pixel's linear depth with a blurred copy (kernel
~3-5% of the screen). Pixels behind their neighbourhood, like the wall right beside a pillar's
silhouette, get pushed a little toward the fog colour and darkened. Pixels in front get a
slight contrast lift. The pillar gets a soft air gap against the wall, the separation a
painter adds by hand, and it holds while the camera is still.

**Why it could be big here.** Luft, Colditz & Deussen (SIGGRAPH 2006) showed that the
difference between the depth buffer and its low-pass marks spatially important areas.
Modulating contrast and colour there adds depth cues that improved perception of complex
scenes. It needs only depth: post.js already reads the depth texture for contour lines, and
the fill chain shows how to build the pyramid. The same pass can give the looks menu two
variants: contour ink in dream colour at depth discontinuities (Saito & Takahashi 1990's
G-buffer lines), and depth of field focused at the centre depth.

**Cost.** Half a day; a 3-level linear-depth pyramid and one lookup, *guess* 0.15-0.3 ms.
Risk: low technically. The aesthetic risk is SSAO's dark-halo look if overdone, so push
toward fog-coloured haze rather than black.

**Smallest experiment (≤2 h).** `?dusm=λ`. Add a linear-depth downsample chain (1/2, 1/4,
1/8) in post.js and take ΔD = blurred − own depth, relative to depth. Where ΔD < 0, mix toward
0.8 x the fog colour by up to 0.4λ; where ΔD > 0, lift contrast by up to 10%. Take rest shots
at pillar poses and in 8 zones for λ in {0, 0.5, 1}, then run the same blind review as idea 2.
Success: preferred for depth in ≥6/8 and not worse-looking in ≥6/8. Run it on the same sheet
as idea 2, since the two are rivals for M114's separation pass.

**What kills it.** The reviewer names halos or outlines as artefacts.

### 4. Geometry-snapped paint (joint-bilateral lookups)

**Idea.** Wherever a result image is sampled (display live views, the atlas paint pass, the
capture's feedback), replace the bilinear colour fetch with the four texels it blends. Weight
each texel by whether that capture pixel actually saw this surface. `liveVis` already
computes that per tap from the capture depth; the colour fetch ignores it. Wall colour stops
landing on the pillar's rim and vice versa, silhouettes get the display's own 1-px edge
instead of a blend of capture pixels, and every feedback pass shows the model edges aligned
to the geometry.

**Why it could be big here.** A wide capture pixel spans ~4 screen pixels at 1080p (DESIGN),
so every silhouette carries a 4-8 px band where pillar and wall colours are averaged. With
the model already painting across the edge, that band erases what separation is left. Joint
bilateral upsampling (Kopf et al. 2007) showed that a low-resolution result upsampled with a
high-resolution guide takes on the guide's edges; here the guide (depth per surface) is exact.
In the feedback loop it shouldn't compound the way sharpening did (M111). A geometry-weighted
average applied to its own output changes nothing further, so it can't run away.

**Cost.** Hours; +3 colour fetches per view per fragment (+18 across 6 views), *guess*
+0.1-0.2 ms. Risk: low. Too-binary weights would give jaggies of capture-pixel size along
silhouettes, so keep the smoothstep visibility.

**Smallest experiment (≤2 h).** `?snap=1` in `GLSL_LIVE` and `PAINT_FRAG`: fetch the 4 texels at
the depth taps' positions (same flip), weight by bilinear weight x per-tap visibility,
normalise, and weight the CAS neighbours the same way. With M114's foveal warp, image texels
and depth texels no longer coincide, so map each image tap back through the inverse warp
before its depth test. Take 4x crops of pillar edges at rest
(nave, garden, baths) before and after, and run `--flicker` twice. Success: cleaner edges in
the crops, silhouette contrast +≥10% at equal detail, and walking change not up >3%. Do this
first: it's cheap, and it makes the comparisons for ideas 2 and 3 cleaner.

**What kills it.** Edges crawl while walking.

### 5. A change budget [BET]

**Idea.** Keep a full-resolution history of the painted image (before bloom and grain) and
reproject it each frame with depth. The world is static, so reprojection is exact except
where something is newly revealed. Move each pixel toward the new dream composite with a time
constant τ and a slew cap r: no pixel's brightness or colour moves faster than r per second,
measured in a perceptual space (Oklab; Ottosson 2020). Newly revealed pixels pass straight
through. The looks menu gets a slider from "calm" (τ ~0.5 s, low r) to "vivid" (off).

**Why it could be big here.** Perceived change stops depending on how often dreams arrive.
Known issue 1 is exactly that: faster dreams made standing still busier (0.66 to 0.85).
Capping captures at 8/s bought calm (0.58) by discarding dreams, and reweighting the stack
(`?liveref`) got halfway (0.72). This calms rest and walking, including live-view hand-overs,
and bounds the worst case. *Guess:* the queasiness comes more from occasional large, fast
swings over big areas than from the average change. The machinery is temporal
anti-aliasing's (Karis 2014; survey: Yang, Liu & Salvi 2020), used the other way round: to
limit change instead of accumulating samples.

**Cost.** A day; one RGBA16F history at display resolution (~16 MB at 1080p) and one
reprojection pass, *guess* 0.2-0.3 ms. Risk: medium. Old paint can trail while walking, and
wrong depth smears. Fog is included in the history, and changes smoothly, so it only lags a
little.

**Smallest experiment (≤2 h).** `?budget=τ,r` as a pass between the main render and post,
keeping the scene's undreamt alpha out of it. Run `--flicker` twice each for off, a fast
setting (0.3 s) and a slow one (0.6 s), and shoot walking strips. Success: standing change
≤0.6 at the full capture rate, walking change down ≥10%, detail within 5%, and a blind
reviewer finds no trails in the strips. Caution: the flicker metric itself reprojects frames
with depth, so this pass lowers it by construction. The strips and the blind review are the
real judge.

**What kills it.** The reviewer names trails, or detail drops >10%.

### 6. Consensus paint (the median of a held framing)

**Idea.** Standing still, the live layer holds 2-3 pixel-aligned results of each held
framing. Show their per-pixel median, choosing a whole dream per pixel by luminance so colours
stay intact. Today's stack gives the newest dream 0.35 and its weights depend on how many
views happen to be alive. With the median, something only one dream invented never reaches
the screen, and where the dreams agree the output is exactly one of them, so nothing is
averaged soft.

**Why it could be big here.** One change aimed at two known issues. Rest got busier with
faster dreams (issue 1), and ghostly figures still appear in the vestibule (issue 6; SD-Turbo
ignores the negative prompt), in some dreams and not others. Temporal medians are the classic
fix for impulsive noise. Burst photography's robust merge down-weights frames that disagree
with the rest (Hasinoff et al. 2016). *Guess:* at a held framing, SD-Turbo's variation is
partly impulsive (occasional re-inventions) rather than Gaussian jitter. The experiment
checks.

**Cost.** Hours to a day. A median of 3 is a few min/max operations per pixel, plus the
bookkeeping to place the three newest same-framing views in known slots (`LiveViews.update`
already groups views by `kfId` and width). GPU cost ~0. Risk: mottled patchwork where the three
disagree at fine scale. Take the choice from a coarse mip and blend.

**Smallest experiment (≤2 h).** `?median=1`. For the newest held framing with ≥3 results, pick
per pixel the view whose mip-2 luminance is the median, blended with a smoothstep; everything
else composites as today. Run `--flicker` twice, and count dark figures by eye in 20 rest
frames in the vestibule. Success: standing change ≤0.65 at the full rate, detail ≥ baseline,
fewer figures.

**What kills it.** Visible mottling in rest crops, or no calm gain. That would mean the
variation is Gaussian, and the running average M114 plans is the right tool.

### 7. Jittered dreams, super-resolved at rest

**Idea.** While a framing is held, offset each capture's projection by a sub-pixel Halton
jitter (±½ capture pixel). Each result is then a slightly shifted sample of nearly the same
dream. Splat them into a 2x-resolution accumulation for that framing at their true positions.
Eight dreams at rest could give roughly twice one dream's linear resolution and average out
the per-dream jitter. Walking uses today's path.

**Why it could be big here.** Rest is when the owner looks closely, and a wide capture pixel
is ~4 screen pixels across at 1080p. A held framing that gets many passes (DESIGN) is the
setting of temporal upsampling (TAAU, DLSS) and of burst super-resolution: Wronski et al.
(2019) recovered detail beyond a single frame's from sub-pixel-shifted handheld bursts. The
DeepCache guard tolerates shifts up to 3 latent px, so sub-pixel jitter keeps the cheap
passes. This is the running average per held framing that M112 and M114 plan, made to add
resolution as well as calm.

**Cost.** 2-3 days; a 1024x640 RGBA16F accumulation per held framing (~5 MB) and one splat
pass per result. Risk: medium-high. CNNs are not shift-equivariant (Zhang 2019), so a
half-pixel input shift may move the dream's content rather than re-sample it, and the
accumulation would then blur or double edges.

**Smallest experiment (offline, ≤2 h, under the bench lock).** Render one held pose of
`bench/scenes.py`'s nave 8 times with Halton sub-pixel offsets of the ray grid. Run the engine
as the game would (same seed, cross-frame attention and DeepCache on). Shift each output back
by its jitter and accumulate at 2x with a Gaussian splat. Compare with one output upsampled
2x and with the mean of 8 unjittered outputs, measuring high-frequency energy at 2x and edge
correlation with the input (as `bench/alignment.py` does). Success: ≥30% more high-frequency
energy than the unjittered mean, better edge correlation than one upsampled output, no
doubled edges.

**What kills it.** Doubled or smeared edges, meaning the content follows the jitter.

### 8. Foveation by warping one capture

**Idea.** Replace the alternating narrow and wide captures with one capture whose pixel
density falls off smoothly from the middle. Render the capture at 2x with a normal
projection, then resample it through a separable warp, x' = g(x) and y' = g(y). A separable
warp keeps vertical and horizontal lines straight, so columns stay straight in what the model
sees. The live-view, paint and fill shaders look results up through the same g and scale the
pixel footprint by g'. Visibility tests keep the linear 2x depth.

**Why it could be big here.** There is no boundary anywhere, so the lens can't come back in
another form. Each framing needs one seed, one DeepCache stream and one cross-frame anchor
instead of two. Every dream refreshes the whole view, where today a narrow dream refreshes
~35% of the screen and the periphery gets half the captures at rest (a third while walking).
The warp can also widen the capture's field of view past the screen at no pixel cost, which
helps the screen edges while turning (known issue 4). The honest trade: with the same pixel
count, g' = 1 + 0.4 cos(πx) gives 1.4x linear density in the middle (the narrow gives 1.7x
today, half the time) and 0.6x at the edges, so corners get softer at rest. The flexible
budget (up to 1.25x the engine's pixels) buys back a little. Precedents: Kernel Foveated
Rendering (Meng et al. 2018) renders in a warped (log-polar with a kernel) space, with 2-3x
speedups at 4K and little perceived loss. Learning to Zoom (Recasens et al. 2018) showed that
networks do better when their fixed-size input is resampled to magnify what matters. *Guess:*
SD-Turbo tolerates a mild separable warp, much like a wide-angle lens's distortion.

**Cost.** 2-4 days: a capture resample pass, one warp function in three shaders, and retiring
the narrow scheduling and the superellipse fade. GPU: the 2x capture is 0.65 MP, *guess*
<0.3 ms per capture. Risk: medium. Beyond the softer corners, the model may straighten warped
diagonals, which would make perspective lines bend once unwarped.

**Smallest experiment (offline, ≤2 h).** Ray-trace `bench/scenes.py`'s nave directly through
the warp (bend the ray grid, no resampling), plus a linear wide and a linear narrow (fov/1.7)
at 512x320. Run the engine at strength 0.6 with the nave prompt and unwarp. Compose wide and
narrow the current way (superellipse band). Measure centre detail (gradient energy in the
middle 35%), a radial profile of local contrast (a step marks a lens), and edge correlation
with the input. Success: centre detail ≥0.8x the narrow's, no step in the radial profile,
edge correlation ≥ the wide's, columns straight after unwarping. It is the fallback if idea 1
leaves a seam in detail scale.

**What kills it.** Bent lines, or the model paints things at a different scale depending on
radius.

### 9. Dehaze the dream, re-haze it with true depth

**Idea.** The model paints its own mist and light shafts into each result, flat in 2D: the
haze sits on a pillar as thickly as on the wall 5 m behind it. Estimate the painted haze in
each result with the dark channel prior, using the zone's fog colour as the airlight, and
remove it. Then add atmosphere back at display time from true depth, and later add true light
shafts from the real windows. Near surfaces get crisp, far ones hazy in proportion to real
distance, and the haze shifts correctly with parallax as you move.

**Why it could be big here.** Known issue 2: the nave, stacks and desert are softer than
M111. The nave "is a fixed shaft of light and haze", and a prompt without mist changed little
(+9% detail). Today the display trusts the model's flat atmosphere and cuts real fog to 35%
where dreamt (`fog *= mix(1.0, 0.35, k)`). Dark channel prior (He, Sun & Tang, CVPR 2009): in
haze-free patches at least one colour channel is near zero, which yields a per-pixel
transmission estimate, refined with a guided filter (He et al. 2010). We know roughly what
the transmission should be from depth, so we can clamp the estimate, which the original
method can't.

**Cost.** 1-2 days. Per result at 512x320, a separable 15x15 min filter and a guided filter,
*guess* <0.2 ms; the display fog already exists. Risk: medium-high. The prior fails on
bright, low-saturation surfaces (pale stone, fog-coloured plaster), which is the nave's
palette, giving blotchy dark patches and frame-to-frame instability. The model's mist colour
may also differ from the fog colour; fit the airlight per zone as a check.

**Smallest experiment (offline, ≤1.5 h).** Patch a scratch copy of `bench/scenes.py` to
return depth as well. Run nave frames through the engine with the nave prompt, dehaze using
the fog colour as the airlight, and re-haze with exp(-β·depth). Put 5 frames side by side.
Success by eye: near columns gain contrast and stand off the far wall, the far nave stays
misty, and ≥4/5 frames have no blotches. Then build it in game as `?rehaze=β`.

**What kills it.** Blotches on pale stone, or a noisy transmission map that flickers.

### 10. Reflections that obey the geometry

**Idea.** The nave floor and the baths pool are flat water. The model paints reflections into
them, and the projection glues those reflections to the floor, so while you walk the
reflected columns slide along with the floor instead of staying under their columns. On
water (and more weakly on tile and metal, whose patterns already return gloss), fade the
dream toward its low band. Then add screen-space reflections that march the depth buffer and
fetch the dreamt colour of whatever is reflected. A later step: give water pixels in the
capture the depth of the mirrored scene, so the model's own reflections are projected onto
the virtual mirrored geometry.

**Why it could be big here.** Two of the eight zones are about water ("The Drowned Nave",
"Lethe Baths"), and their prompts ask for reflections. Correct reflections are the strongest
geometry cue still water has. They also update with every dream, because they sample the
dreamt image. Screen-space ray tracing: McGuire & Mara (JCGT 2014), DDA marching of the depth
buffer. Rendering the scene with a mirrored camera is the standard technique for flat water.

**Cost.** SSR: 2-3 days at half resolution, *guess* 0.5-1 ms, the priciest idea here per
frame, so it pays the exchange rate. Mirrored-depth capture: ~1 week. Risk: medium. Rays miss
at screen edges, and reflections double (painted plus SSR) unless the painted one fades.

**Smallest experiment (≤2 h).** `?ssr=1`, on water only: a half-resolution linear march (32
steps, then a binary refine) of scene colour and depth in post, weighted by Fresnel, with
water's dream colour mixed toward its mip-3 low band where the ray hits. Shoot walking strips
(8 frames, 0.15 s apart) in the nave and baths before and after, and check `--perf` p95.
Success: a blind reviewer prefers the water in both zones; p95 +<1 ms.

**What kills it.** The reviewer names streaks or holes, or p95 rises >1.5 ms.

### 11. Parallax probes: dream what's behind the pillar before you step

**Idea.** Standing still, the strips of wall hidden behind pillars and door jambs are never
painted. They fall back to the atlas, often to the blueprint, so the first step sideways
reveals soft slivers, and the pillar separates from the wall only once motion shows them.
Spend a few rest captures on probes offset in position (0.3-0.6 m to the side with more depth
discontinuities) instead of yaw glances, painting those strips into the atlas and a live slot.
A variant: a capture that leaves out the nearest occluders, which the level generator can
flag.

**Why it could be big here.** Known issue 4 (fresh disocclusions fall back to the atlas). The
held-framing glance sequence (`[0,0,0,1,0,-1,...]` in `_capture`) only rotates yaw, which
never sees behind anything. Layered depth images (Shade et al. 1998) keep more than one depth
layer per pixel so that a nearby viewpoint renders without holes; the probes build that
second layer in paint.

**Cost.** 1-2 days; spends ~1 in 8 rest dreams. Risk: medium. Probe paint comes from another
pose and may seam against the main view (idea 1 helps).

**Smallest experiment (≤2 h).** `?probe=0.4`: two of the yaw glances in the held sequence
become ±x position offsets (alternate sides first; choose smarter later). Script it: rest
6 s, step 0.5 m sideways, grab a frame 0.3 s later. Count undreamt pixels (scene alpha
> 0.35) in 8 zones and look at the revealed strips. Success: ≥50% fewer undreamt pixels in
the revealed strips in ≥5/8 zones, no visible seam, and standing change up ≤10%.

**What kills it.** Seams, or a busier rest.

### 12. Sparse virtual paint (a sharp atlas near you)

**Idea.** Add a second "near atlas" that holds only the charts within ~20 m of the player, at
2-4x the density, re-packed as you move. Least-recently-used eviction writes a downsampled
copy back into the base atlas. A per-chart page table (chart id to rectangle) tells the
display, painter and capture where each surface's near paint lives.

**Why it could be big here.** The atlas holds ~12.85 texels/m (seed 7). Everything the live
views don't cover (screen edges, disocclusions, what's behind you when you turn back; known
issues 4 and 8) is ~4.5x softer at 3 m than the model's own pixels (the `live.js` header).
A near atlas makes the dream's memory as sharp as its newest pixels. It's also where
"best view wins" paint can live, a lever listed since M111 and never built. Sparse virtual
texturing (Barrett, GDC 2008) and id Tech 5's virtual textures (van Waveren 2009) serve huge
unique textures from a small page cache. Ptex (Burley & Lacewell 2008) stores per-face
textures with no UV-atlas pressure.

**Cost.** 1-2 weeks; +128 MiB (4096² half-float) and a second paint pass, with mip rebuilds
batched (see the top). Risk: medium. Re-packing can hitch, and chart ids need a vertex
attribute; `flatten` already computes `chartOf`.

**Smallest experiment (≤2 h).** Measure the upper bound first. Run `?atlas=8192&persist=0`
(the packer takes the size, so texels per metre double). Check that it allocates and runs in
WebKit and Chromium, and record frame p95 and paint ms. Screenshot the screen edges mid-turn
and the views after turning back, at 4096 vs 8192. Success: clearly sharper edges and
turn-backs in ≥5/8 zones. Then build the sparse version, which gets that without 512 MiB.

**What kills it.** No visible difference, because the live views already cover what matters.

### 13. The dream lights the world (weeks)

**Idea.** Treat what the model paints as light. Where a result is much brighter than the real
shading predicts (painted windows, neon, candles, crystals), mark it emissive, and let it
cast light onto the real geometry with a screen-space one-bounce gather. Later, a coarse
world-space irradiance cache fed from an emission atlas can light what's off screen. A
painted lancet window behind a pillar rims the pillar in its light; painted neon tints the
real pool and vault. The world visibly answers what the model dreams.

**Why it could be big here.** It's the strongest version of "use the geometry": the model
invents light, and the geometry decides where that light falls, with real occlusion. A
backlit rim is also the classic way to separate a pillar from a bright wall. The charter's
roadmap already wants things that respond to what the model paints (item 5, for sound).
Screen-space directional occlusion (Ritschel, Grosch & Seidel 2009) produced plausible
one-bounce colour bleeding from screen-space samples at interactive rates.

**Cost.** Weeks for a stable, good-looking version. GPU: a half-resolution gather with 8-16
samples plus temporal accumulation (idea 5's history), *guess* 1-2 ms, so the exchange rate
applies. Risk: high. Noise, light leaking through thin walls, and a runaway loop if it ever
feeds the capture; keep it display-only at first.

**Smallest experiment (offline, ≤2 h).** Grab `snapAt` frames with depth (display PNG, linear
depth, camera). Build an emitter mask (luminance and saturation above a threshold, sky
excluded) and reconstruct normals from depth. For each pixel, gather 32 random samples within
15% of the screen, weighted by emitter luminance x cosine / distance², with a visibility test
that marches the depth. Add the result at a few strengths, and look at the nave (windows
behind columns), the baths (neon) and the geode (crystals). Success: plausible rims and
bounce light without blotches, reading as the space being lit.

**What kills it.** It reads as random glows, or the noise would need heavy denoising.

### 14. The dream's own depth: a probe inside the UNet (weeks)

**Idea.** Chen, Viégas & Wattenberg (2023) found that linear probes on Stable Diffusion's
internal activations recover depth and a foreground/background split early in denoising, and
interventions showed those representations are causal. Fit a linear probe on SD-Turbo's
decoder features that predicts the capture's geometric depth, over many frames. Then, per
result, compare the probe's depth with the geometry's: where a pillar reads at the wall's
depth, the model has flattened it. That map puts fixes where they're needed: less feedback
there on the next capture, stronger separation from ideas 2-3 there. The probe's depth also
gives painted but unmodelled detail (a painted window recess, bookshelves) its own relief for
bump and parallax-occlusion shading. Stage 0 needs no probe: luminance as height, "dark
means deep" (Langer & Bülthoff 2000).

**Why it could be big here.** It goes after the pillar at its source, and it makes flattening
measurable per frame, a number to hill-climb instead of a judged sheet. At runtime the probe
is a 1x1 convolution on features the UNet computes anyway, so it's free. Its depth can ride
back as a small extra image, since the protocol allows extra fields.

**Cost.** Weeks to exploit fully (a detector wired into feedback control, relief shading,
POM); fitting the probe takes about an hour. Risk: medium-high. At strength 0.6 (t ≈ 600),
the one-step features may encode the depth of the render the model was shown rather than of
what it painted, and then the probe can't see flattening.

**Smallest experiment (offline, ≤2 h, bench lock).** Hook the outputs of two up blocks in
`lean_unet_forward` (DeepCache off). Run ~300 frames of `bench/scenes.py` walks and turns
(depth from a scratch patch) at strength 0.6, ridge-regress log depth from the features, and
hold out 20%. Then look at frames where the output visibly merged a column into the wall, as
in `docs/shots/render_capture_vs_model_nave.jpg`: does the probe put the column near wall
depth? Success: held-out R² ≥0.6, and the probe follows the painting in the flattened frames.

**What kills it.** The probe only reproduces the input's depth, with no sensitivity to what
was painted, or R² < 0.4.

### 15. The renderer hands the model exact correspondences (weeks)

**Idea.** The game knows, for every pixel of a new capture, exactly where that surface point
was in the previous capture of the same stream (both cameras plus depth). Send that as a
small flow field at latent resolution with the frame (64x40 x 2 half floats ≈ 10 KB). On the
server, use it to warp DeepCache's cached deep features, and to bias cross-frame attention
toward each token's true predecessor. Today the options are a global phase-correlation shift
or content matching alone. Newly revealed regions get fresh computation.

**Why it could be big here.** Consistency and speed at once. DeepCache's guard forces full
passes when the global shift exceeds 3 latent px (turning). Its motion-compensated mode, a
global translation, lifted turn alignment to 0.264 but smeared the entering borders
(`bench/RESULTS.md`). Exact, parallax-aware per-pixel warps could let cheap passes survive
turning and walking, more dreams per second exactly where they matter, and line cross-frame
anchors up with the geometry. TokenFlow (Geyer et al., ICLR 2024) showed that propagating
diffusion features along inter-frame correspondences gives consistent video edits, and
Rerender-A-Video (Yang et al. 2023) fused frames along optical flow. Here the correspondences
are exact and free, which no video method has. Why it could work where motion-compensated
DeepCache didn't: the smear came from replicate-padding a global shift into newly revealed
borders. Exact flow knows which pixels are new and can hand them to fresh computation instead
of copying.

**Cost.** Weeks: a protocol extension, warping at each cached level, disocclusion masks,
partial recompute. Risk: high. Deep features at 1/8-1/32 resolution straddle depth edges, so
warping them across silhouettes can smear meaning.

**Smallest experiment (offline, ≤2 h, bench lock).** On `bench/scenes.py` turn and walk
sequences (depth from a scratch patch), compute exact flow from each full-pass frame to the
following cheap frames. Run the cheap passes three ways: (a) the cache as is, (b) global
shift (`dc_motion`), (c) `cache['deep']` warped with `grid_sample` by the exact flow at its
resolution. Measure `bench/alignment.py` edge correlation with the current input (on the
turn today: cheap 0.204, full 0.300, global shift 0.264) and look at the borders. Success:
(c) ≥0.27 on the turn with no border smear, enough to justify relaxing the shift guard.

**What kills it.** No gain over (b), or smears at silhouettes.

### 16. Brushstrokes anchored to the world (weeks)

**Idea.** Render the dream as tens of thousands of brush strokes seeded on the world's
surfaces: Poisson-disk points per atlas chart, denser nearby. Each stroke is oriented by the
geometry (along a pillar's axis, around arches, across depth edges) and coloured from the
live composite or atlas at its anchor. Strokes never move on the surface. A new dream only
recolours them over a fraction of a second, and stroke size follows the source resolution,
so narrow and wide paint look alike.

**Why it could be big here.** It turns three problems into style. The lens goes, because no
pixel-level resolution change shows under strokes. Boiling goes, because strokes persist and
only their colour eases. Flat pillars go, because stroke direction follows the cylinder and
reads as shape. Meier (SIGGRAPH 1996) showed that strokes attached to particles on 3D
surfaces give frame-to-frame coherent painterly animation, without the "shower door" effect.
It fits the visual direction ("slow, breathing, painterly").

**Cost.** 1-2 weeks. *Guess* 1-3 ms for 100-300k instanced strokes at 1080p (overdraw), so
the exchange rate applies. Risk: high aesthetic risk of a Photoshop-filter look, and strokes
popping as the level of detail changes.

**Smallest experiment (offline, ≤2 h).** From a rest `snapAt` frame (colour, depth, camera),
place 40k strokes in screen space, oriented by normals and curvature from depth (vertical on
columns), draw them with PIL and look. Then reproject the same anchors across a 12-frame walk
and draw each frame to check coherence. Success: it reads as a painting of the place, with
columns shaped by their strokes, and holds together through the walk.

**What kills it.** It looks like a filter, or it loses the dream's detail.

### 17. The dream as a shader: an online-trained neural paint field (weeks to months)

**Idea.** Continuously fit a small multiresolution hash grid plus MLP that maps (world
position, normal, view direction) to paint colour. Train it on every result's pixels at their
capture-depth positions. Render by evaluating it per pixel, or only where the live views
don't reach (edges, disocclusions, distance). It fuses many dreams into one paint, a
least-squares consensus over views, with no texel cap near surfaces and no atlas charts, and
it can learn mild view dependence such as painted reflections on water.

**Why it could be big here.** It would replace the atlas plus live-view stack, the source of
the lens, the seams and the softness, with one continuous representation, and it would make
disocclusions and turn-backs crisp. Instant-NGP (Müller et al. 2022) trains hash-grid
primitives in seconds. Real-time Neural Radiance Caching (Müller et al. 2021) trains a tiny
MLP online inside a 16.6 ms real-time frame. Safari 26 ships WebGPU on by default (WebKit),
so compute-shader training in the browser is possible on the target machine.

**Cost.** Weeks to months. *Guess* 2-5 ms a frame for training and inference, so the exchange
rate bites hard. The WebGL2 renderer can't share textures with a WebGPU context, so this
means a port. Risk: high: the field may average inconsistent hallucinations into mush.

**Smallest experiment (offline, ≤2 h, PyTorch on MPS, niced, no server running).** Take a
held-framing sequence (8 results) and a walking one (8 results) with depth from the bench
scenes. Fit a pure-PyTorch hash grid (12 levels x 2^16 x 2 features) and a 2-layer MLP for 2
minutes. Render an intermediate pose and compare it with reprojecting the nearest result:
sharpness, holes. Success: as sharp as the nearest-result reprojection, with no holes.

**What kills it.** Visibly blurrier (the mean of the hallucinations), or more than a minute
per zone to converge.

## Also considered (one line each)

- **Flow-morphed hand-overs** (frame generation between dreams): turns blinks into glides,
  but gliding texture is a vection source for a player who is already queasy. Worth it only
  as a "liquid" look.
- **Idle micro-parallax**: a slow 1-2 cm sway reveals depth through motion parallax (Rogers &
  Graham 1979). Cheap as a look, but it's camera motion the player didn't ask for.
- **Depth of field at the centre depth**: strong separation, but without eye tracking the
  focus pulls as you look around feel wrong. It's a variant in idea 3.
- **Texture-space dreams** (img2img on flattened atlas charts with a material prompt, low
  strength): no perspective and high texel density, but charts are small fragments with no
  context, and the model paints scenes, not materials. Worth one offline test on a big nave
  wall chart.
- **Gaussian splats fitted to the dreams**: idea 17's promise with view-dependent colour, but
  heavier to train online.
- **Nested noise between narrow and wide**: make the narrow stream's latent noise a sub-pixel
  refinement of the wide stream's where they overlap (Chang et al. 2024's conditional
  upsampling). It's static at rest, unlike the world-anchored noise that was dropped, which
  changed every frame. That's for the model lens; noted here as a lens fix.

## Ranking

| # | Idea | Answers | Build | Frame cost (guess) | Works? (guess) |
|---|---|---|---|---|---|
| 1 [BET] | Tone from the world, detail from the view | lens, rest calm | 1-3 days | +0.1-0.3 ms | 0.6 |
| 2 [BET] | The world relights the dream | pillar, "use the geometry" | hours to 2 days | ~0.05 ms (+0.3-0.5 with GTAO) | 0.5 |
| 3 | Painterly depth halos (in M114's tree) | pillar | half a day | 0.15-0.3 ms | 0.7 |
| 4 | Geometry-snapped paint | silhouettes, smear | hours | 0.1-0.2 ms | 0.7 |
| 5 [BET] | A change budget | queasiness, rest calm | 1 day | 0.2-0.3 ms | 0.5 |
| 6 | Consensus paint (median) | rest calm, ghost figures | hours to 1 day | ~0 | 0.35 |
| 7 | Jittered dreams, super-resolved | rest sharpness | 2-3 days | small | 0.3 |
| 8 | Foveation by warping one capture (in M114's tree) | lens at the root, refresh, wider capture | 2-4 days | <0.3 ms per capture | 0.4 |
| 9 | Dehaze, re-haze with true depth | nave softness, aerial depth | 1-2 days | <0.2 ms per result | 0.35 |
| 10 | Reflections that obey the geometry | nave, baths | 2-3 days (+1 week) | 0.5-1 ms | 0.5 |
| 11 | Parallax probes | disocclusions | 1-2 days | ~1 in 8 rest dreams | 0.45 |
| 12 | Sparse virtual paint | edges, turn-backs, memory | 1-2 weeks | +128 MiB, a paint pass | 0.5 |
| 13 | The dream lights the world | a clear "wow", backlit pillars | weeks | 1-2 ms | 0.3 |
| 14 | The dream's own depth | pillar at the source, a flattening number | weeks (probe: 1 h) | ~0 | 0.35 |
| 15 | Exact correspondences for the model | consistency, cheap passes while turning | weeks | server-side | 0.3 |
| 16 | Brushstrokes anchored to the world | a look that hides the lens and the boil | 1-2 weeks | 1-3 ms | 0.3 |
| 17 | Neural paint field | replaces atlas and live stack | weeks to months | 2-5 ms | 0.15 |

**The three bets, one per complaint in the owner's 09-29 play note:** 1 for the lens (the
tool panorama stitching uses for exactly this seam, and it damps large-area boiling too),
2 for the pillar (the geometry's own lights and normals put back what the model flattened),
and 5 for the queasiness (a hard bound on change per second, independent of the dream rate).
All three are display-side, with no bet on the model's behaviour, and each has an experiment
that settles it in under two hours.

**Order:** 4 first (hours, and it makes every later comparison cleaner). Then put 2's
light-aware ratio against M114's depth separation (3) and its |n·v| relight on the same
pillar sheet. Then 1, then 5. Judge M114's warp (8) with idea 8's radial-profile test, and
use idea 1 as the fallback if the warp leaves a seam in detail scale.
**Weeks-scale picks for M115:** 14 (settled in an hour offline, and it goes after the pillar
at its source) and 15 (the renderer's unique asset, exact correspondences, turned into more
cheap passes: a real optimization). 13 is the one most likely to make the owner say wow.
**Cheap looks for M114's menu from this list:** Calm (5), Lantern (2), Air (3), Ink (3's
contour variant); later Wet (10) and Oil (16).
