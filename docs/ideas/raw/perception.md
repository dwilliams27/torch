# Perception and comfort: ideas for Hypnagogia (Claude Opus 5.5, 2026-09-30)

*Lens: perception and comfort. Author: Claude Opus 5.5, one of five fresh-context idea
generators in the 2026-09-30 idea session. Raw and unranked; nothing here has been run.
"Computed" means derived from the code's defaults; "estimate" and "guess" mean what they say.
Sources are named inline and listed at the end.*

## The owner's play notes, read through vision science

**The lens.** At `?fovea=1.7` the narrow capture has full weight over the central ~23% of
a 16:9 screen, and its fade band ends on a contour enclosing ~46% (computed from the
capture fit with its 4° margin and the superellipse band 0.7-0.98), which matches the
owner's "probably 40% of the screen area". On a
27-inch monitor at 65 cm its band sits about 8-11° above and below the point of gaze and
12-17° to either side (estimate). A wide capture pixel spans about 6.5 arcmin there
(DESIGN's "~4 screen pixels at 1080p"), a narrow one about 4.3. Peripheral acuity at 8-17°
is in the same range, about 4-8 arcmin by letter measures (Strasburger, Rentschler &
Jüttner 2011; grating measures run finer). So the resolution step alone is
probably near threshold when you look at the centre. Three other differences between the
two paintings are more likely what shows: their texture statistics (own seed, own stream,
a different reading of the scene), their temporal behaviour (walking, the centre is
re-dreamt twice as often as the edges), and possibly their perceived speed (lower contrast
looks slower, Thompson 1982; blurred periphery lowers perceived speed, Tariq & Didyk 2024).
On top of that comes the blend band, where two unrelated paintings are averaged, and a
border that stays fixed to the screen while the world slides through it. The owner's word
for it, "glasses", fits a known effect: progressive spectacle lenses make the world "swim"
through their distorted zones, and the swim goes with dizziness and nausea (Sauer et al.
2021).

**The pillar.** "Unless I move" names the cue that was doing the work: motion parallax.
Paint is anchored to the world, so moving makes the pillar's paint slide against the
wall's and the edge appears (Rogers & Graham 1979). Standing still leaves only pictorial
cues, chiefly shading and luminance contrast at the occluding edge, and those are what a
one-step img2img model flattens.

**The nausea.** Not measured. Suspects in the code: about 100-105° of scene shown on a
screen that spans roughly 35-50° of the eye, so the left and right edges are stretched
about 2.4-2.7x and move that much faster in turns (computed, for a laptop at 50 cm and a
27-inch monitor at 65 cm); head bob; the dream's own motion signals (features jumping at
the dream rate, crossfades between misaligned results, a drifting shimmer on fresh paint,
an outward-rolling wave on the blueprint lines); and the head-locked lens. Idea 7 is a
way to find out which.

**A method note.** Our blind reviewers are language models looking at stills. They see
every region at foveal resolution and see no motion, so they over-report peripheral
softness and under-report flicker and swim. Perception experiments should show them what
a fixating eye gets (frames blurred by eccentricity, or questions about the centre only)
and should turn temporal effects into pictures (difference maps, spectra). The only human
sensor is the owner; idea 7 makes his reports count.

## The three bets

1. **Retina capture (idea 1).** Removes the lens by construction: one continuous density
   gradient, one capture stream, so there is no border for the eye to find.
2. **Change under cover (idea 5).** Keeps the dream's intensity and spends its visible
   change where the eye can't see it, graded by a meter weighted like the eye's temporal
   sensitivity rather than by raw pixel change.
3. **Close your eyes (idea 13).** One key lets the model rewrite the room while the change
   is guaranteed unseen. It is the most dreamlike idea here and has no motion in it.

Runner-up for the pillar: idea 10, dream in colour, see in luminance.

---

## Ideas

### 1. Retina capture [BET]

**Idea.** Replace the alternating narrow and wide captures with one capture whose pixel
density falls smoothly from the centre, as the retina's does. Render the capture view
rectilinear at 2x (1024x640), resample it through a separable warp (x' = g(x), y' = g(y))
into the engine's 512x320, dream that, and undo the warp at projection time: `liveView()`
and the paint shader map a world point to rectilinear capture coordinates and then through
g to the image, while depth tests stay in the 2x rectilinear depth. A Gaussian density
profile (σ = 0.35 of the half-width) that peaks at today's 1.7x in the middle falls to
about 0.7x at the edges for the same pixel count (computed, per axis). A separable warp
keeps vertical and horizontal lines straight, so columns and door frames stay straight in
the model's input; only diagonals bend.

**Science.** Cortical magnification: visual cortex gives the field area roughly in
proportion to 1/(E + E2) (Daniel & Whitteridge 1961; Rovamo & Virsu 1979; log-polar
retinotopy, Schwartz 1977). Texture borders are found by gradients of pooled filter
responses (Bergen & Adelson 1988; Malik & Perona 1990). A two-level density field puts its
whole gradient on one closed contour, which contour integration picks out (Field, Hayes &
Hess 1993); a smooth field spreads the same change thinly everywhere. That this is enough
to hide it is a guess, and the experiment tests it.

**Why it matters here.** It removes the lens at its source, in all three channels at once:
one painting, one update rate, one seed. Every capture joins one stream, so cross-frame
attention and DeepCache see consecutive frames instead of every second or third, and the
model sees the whole scene every time (zoomed-in views are where small things get
re-invented, like the journal's statue that became a fountain). Looking at the centre, the
softer edges should stay under the eye's acuity out there (estimate: ~8-9 arcmin paint
against ~11 arcmin letter acuity at the screen edge). Looking at the edges shows softer
paint, but no border.

**Cost.** 1-2 days: a 2x capture target and a warp pass in `dream.js`; a warp function and
a separate depth texel size in `live.js` and the paint shader; the narrow scheduling goes.
The 2x capture raster is probably well under a millisecond of GPU (guess; check `--perf`).

**Two-hour experiment.** Offline, with the engine in-process as `bench/temporal.py` does.
Extend `bench/scenes.py` (a numpy ray tracer of a nave with columns) to trace the warped
image directly, rays through g⁻¹. For 8 poses make (a) wide 512x320; (b) a 1.7x narrow view
with its own seed, composited over (a) with the client's superellipse band; (c) retina,
unwarped. Run each through six feedback passes (input = mix(raw, last result, 0.55)) to
stand in for a held framing. Measure centre and edge detail (band-pass energy) and the
lens-o-meter ring (idea 2); ask a blind reviewer, panels shuffled, whether any region looks
painted or filtered differently, and where. Success: (b) flagged in at least 5 of 8 (the
positive control), (c) in at most 1; centre detail of (c) at least 90% of (b); columns
still straight after unwarping.

**Kill.** Columns bow or strokes smear after unwarping in 3 of 8 or more; or the reviewer
still finds a different centre in (c), which would mean magnification itself changes the
model's style (then see C2).

### 2. Lens-o-meter [tool]

**Idea.** Anything tied to the screen survives averaging over many poses and zones, while
the world's content averages away. Per frame, compute maps of local band-pass energy at
three scales, local mean luminance and chroma, change after motion compensation, and local
motion energy. Average each map in screen coordinates over a few hundred frames from all 8
zones, at rest and walking, and plot it against the narrow band's radius. A lens shows as a
step or a ring in one or more channels; a clean build is flat apart from the vignette.

**Science.** Texture segregation follows gradients of pooled filter energy (Bergen &
Adelson 1988; Malik & Perona 1990). Perceived speed drops with contrast below ~8 Hz
(Thompson 1982) and with lost peripheral detail (Tariq & Didyk 2024). Distortions fixed to
the head are experienced as swim (Sauer et al. 2021).

**Why it matters here.** M114's lens bar rests on a blind reviewer, who can't see the
temporal or speed channels. This gives a repeatable number per channel and says which one
the owner is seeing, which decides between ideas 1, 3 and 4.

**Cost.** Half a day: a 30 Hz dump mode in `tools/shoot.mjs` (320x180 luma, chroma and
depth; `--flicker` already reduces frames in the page, at 10 Hz) and a numpy script.

**Two-hour experiment.** Default against `?fovea=1`, two runs each, rest and walking.
Success: the default shows a ring at the band's radius at least 3x the run-to-run noise in
some channel, and `?fovea=1` does not.

**Kill.** The known positive control doesn't separate from `?fovea=1` in any channel.

### 3. Seams the eye can't find: multiband, variance-preserving blends

**Idea.** Where two unrelated paintings are alpha-blended 50/50, only about 71% (1/√2) of
their contrast survives and their features double. That happens across the lens band and
at the midpoint of every 0.25 s fade-in. Blend in two or three frequency bands instead: the
low band (tone and colour) from a wide, slow blend of all views and the atlas; the high band
(detail) from the finest view over a short transition, renormalised by
1/√(Σw² + 2ρ·w₁w₂), where ρ is the local correlation of the two views' detail (near 1 for
converged views of one framing, near 0 for unrelated ones), so contrast neither dips nor
doubles. Each band's transition is about one of its own wavelengths wide.

**Science.** The multiresolution spline hides seams by blending each band over a width
matched to its wavelength (Burt & Adelson 1983). Linear blending of stochastic textures
loses contrast and ghosts; preserving mean and variance fixes it (Heitz & Neyret 2018;
Burley 2019). Lost peripheral contrast reads as tunnel vision, and restoring it let people
accept about twice the blur (Patney et al. 2016).

**Why it matters here.** It goes after the exact thing the owner described ("a transition
zone on its outskirts that blends awkwardly") and the contrast pulse of every fade, in the
shader alone, whatever the capture scheme becomes. (More feedback for narrow captures was
tried and dropped, so this leaves the capture alone.)

**Cost.** Half a day to a day: mipmaps for the live result textures (created today with
`generateMipmaps = false`), a base/detail split and a few correlation taps in
`liveComposite`.

**Two-hour experiment.** A flag; rest frames in 8 zones with and without; band-pass RMS
contrast inside, across and outside the band, and mid-fade. Blind reviewer: find a
washed-out or doubled region. Success: the band's contrast deficit halves, and flagged
zones fall to 2 of 8 or fewer.

**Kill.** Renormalised detail sparkles where ρ is misjudged (grain in strips), or frame
p95 rises by more than 0.5 ms.

### 4. A fovea with no fixed address

**Idea.** If the fovea stays a separate capture, never put its border in the same screen
place twice. Give each narrow view a random zoom (1.4-2.0) and a centre offset of up to
~15% of the screen from a blue-noise sequence. At rest, cycle through 3-4 such framings,
each converging on its own; walking, draw a new one per capture, biased toward the heading
point. Each border is then fixed to the world (it stays on the wall as you turn), and the
average of many borders over time is a smooth gradient.

**Science.** Contour integration needs a consistent path (Field, Hayes & Hess 1993).
Repeated identical signals are easier to detect (probability summation over time, Watson
1979), so a border that never repeats should be harder to see (guess by extension).
Players fixate the screen centre most of the time (Kenny et al. 2005; central bias, Tatler
2007), and walkers look along their path (Land & Lee 1994; Matthis, Yates & Hayhoe 2018).

**Why it matters here.** A cheap fallback if idea 1 fails; the live layer already accepts
any view-projection matrix.

**Cost.** Half a day: an off-axis projection and randomised zoom in `_capture`.

**Two-hour experiment.** `?fovjitter=1`; lens-o-meter and rest change on 8 zones; walking
strips for a blind reviewer. Success: ring strength down 60% or more, centre detail within
10%, rest change up by no more than 10%.

**Kill.** Rest change climbs (each re-framing re-invents), or the reviewer now sees
scattered patches of a different style.

### 5. Change under cover [BET]

**Idea.** Treat visible change as a budget and spend it where the eye can't see it: slowly
at rest, freely during motion, first where busy texture masks it, and never as a sudden
onset in the periphery. Grade it with a visible-change meter that weights
motion-compensated change by the eye's temporal sensitivity and by eccentricity, reported
beside "evolution per 3 s". The target is less visible change per moment at the same
evolution per minute.

**How.** (1) At rest, accumulate results per held framing and let the displayed paint
follow with a 1-2 s time constant, which moves the dream's change below ~0.3 Hz, where
sensitivity to luminance change is low; the dream still evolves over a minute. (2) In
motion, loosen the reins: fast hand-overs, even cuts, and big re-framings or prompt steps
committed in the first 150 ms of a fast turn. (3) Fill fast, replace slow: painting undreamt
surfaces can be instant, since there is nothing to compare against, while replacing
existing paint is paced, faster where local contrast is high and slower on flat, bright or
central areas. (4) The meter: motion-compensated luma and chroma series at 30 Hz, their
temporal spectra weighted by the temporal contrast sensitivity function (luminance
band-pass peaking near 5-10 Hz, chroma low-pass) and by eccentricity; "evolution per 3 s" is
the motion-compensated difference between frames 3 s apart. Today's `--flicker` samples at
10 Hz, the rate dreams arrive, so it can't tell a slow drift from a 10 Hz shimmer of the
same total size.

**Science.** Large changes go unnoticed when gradual (Simons, Franconeri & Reimer 2000).
Motion silences awareness of change (Suchow & Alvarez 2011), modelled as a flicker-detector
effect (Choi, Bovik & Cormack 2014). Busy texture masks change (Legge & Foley 1980).
Spatiotemporal contrast sensitivity (Kelly 1979), and its low-pass chromatic counterpart
(Kelly 1983). Abrupt onsets capture attention (Yantis & Jonides 1984), so each peripheral
fade-in pulls the eye onto the artifact. Attention held on a central task lowers
peripheral sensitivity (Krajancich, Kellnhofer & Wetzstein 2023), which is the state of
walking.

**Why it matters here.** It addresses "disorienting" and M114's gentler bar without muting
the dream (`?calm` lowered strength and was dropped for draining it), and gives M112's rest
problem a target defined by the eye instead of by mean pixel change. Note the constraint in
today's live layer: with five views at ~10 results a second each view lives ~0.5 s, so any
fade longer than that needs the per-framing accumulator M112 already plans.

**Cost.** Meter: half a day. Scheduler: 1-2 days (the accumulator, motion-gated fade
times, a contrast-aware fade rate in `liveComposite`).

**Two-hour experiment.** Build the meter; measure default and `?rate=8` at rest in 8 zones
(WebKit harness, 2 runs), plus the accumulator if M112 has it by then. Success: the meter
ranks `?rate=8` calmer, as the committed data do, and reports how much of that calm is lost
evolution; the accumulator (when available) cuts visible change by 30% or more while
evolution per 3 s drops by less than 20%.

**Kill.** The meter disagrees with the committed blind steadiness verdicts (rerun the two
builds behind `judge-final.json`; it should agree in at least 6 of 8 zones), or slow rest
blending visibly softens detail.

### 6. Align, then let go

**Idea.** When a new result fades in over one whose features sit a few pixels elsewhere,
the eye sees each edge jump, and ten jumps a second across the whole view is apparent
motion. Before a view fades in, estimate its displacement from the paint on screen
(coarse-to-fine block matching at quarter resolution on the GPU, with a confidence mask),
warp the new view onto the old paint, then relax the warp to zero over about a second.
Jumps become slow drifts. Where the content really changed (low confidence), keep the
normal fade.

**Science.** Two displaced frames are seen as motion (Wertheimer 1912), crossfades included
(Anstis 1970). A 3-pixel jump ten times a second is jitter far above motion thresholds; the
same 3 pixels spread over a second is a drift near the threshold for relative motion
(estimate). Whole-field visual motion drives sway and vection (Lee & Lishman 1975; Brandt,
Dichgans & Koenig 1973), which makes this a candidate for the nausea (guess, unmeasured).

**Why it matters here.** It aims at the likeliest dream-made cause of queasiness while
walking, and an aligned blend is also a sharper blend.

**Cost.** 3-5 days: GPU flow at 128x80 for ~10 results a second, per-view warp textures,
and care in the capture loop (the capture should read the aligned views).

**Two-hour experiment.** Offline on idea 2's dumps: reproject the last result into the new
one's view with depth, block-match, and report the share of the picture with confident,
coherent shifts of one capture pixel or more, and the residual before and after
alignment. Success: 30% or more of the picture shifts coherently and alignment cuts the
residual by 30% or more.

**Kill.** The shifts are mostly incoherent: re-invention rather than sliding. Then
crossfade motion isn't the mechanism, and idea 5 is the fix.

### 7. Nausea triage kit

**Idea.** Separate the suspects and let the one person who can feel sick decide between
them. Give each suspect a frame-based proxy and a look in the planned looks menu, and add an
opt-in misery prompt (`?misc=1`): every 3 minutes a small "0-9?" for two seconds, answered
with one key, logged per look and summarised in the panel.

**Suspects.** (a) Field of view: 100-105° of scene on a screen spanning ~35-50° of the eye,
edges stretched ~2.4-2.7x (computed). (b) Head bob, 2.6 Hz vertical and 1.3 Hz lateral when
walking, and the ~90 ms velocity ease (computed). (c) Paint slip, the dream moving relative
to the geometry, measured as an illusory-flow index: residual optical flow after depth
reprojection, its size and its coherence per screen sector (a sheet sliding one way should
be worse than boil). (d) Rest boil (idea 5's meter). (e) The head-locked lens (idea 2).
(f) Small coherent motions the shaders add on purpose: the blueprint lines' brightness wave
rolls outward at ~1.7 m/s (`sin(uTime*0.6 - dist*0.35)`), the fresh-paint shimmer noise
drifts ~16 cm/s, mostly downward, and the chromatic breathing pulses radially at 0.05 Hz
(computed from the shaders).

**Science.** Sensory conflict (Reason & Brand 1975; Oman 1990). Visually induced sickness
peaks for 0.2-0.4 Hz oscillation of the flow (Diels & Howarth 2013); the bob and the idle
drift (under 0.04 Hz) sit outside that band. Image-scale mismatch (Draper et al. 2001) and
wider fields of view (Lin et al. 2002) raise sickness in headsets; for desktop play this is a
guess. Bob-like viewpoint jitter increases vection (Palmisano, Gillam & Blackburn 2000),
and vection and sickness are related without being the same thing (Keshavarz et al. 2015).
One-number ratings: the fast motion sickness scale tracks the SSQ well (Keshavarz & Hecht
2011); the 0-10 misery scale is used in vehicle studies (Bos, MacKinnon & Patterson 2005).
A "breath" look could pace the ambience's swells at ~6 per minute: guided slow breathing
and pleasant music each reduced sickness (Russell et al. 2014; Keshavarz & Hecht 2014),
though unguided entrainment is a guess.

**Why it matters here.** The owner said "a bit" nauseating. Without this we will optimise
whichever suspect we can measure rather than the one that makes him queasy.

**Cost.** About a day, plus the looks.

**Two-hour experiment.** The frame-based half: illusory-flow index and edge flow speed on
the 8-zone walk for default, `?fov=60` and a no-bob flag (new), 2 runs each, and the same
walk with the dream paused. Success: the proxies separate the suspects, for example
peripheral slip coherence above 0.5 with the dream running and near zero when paused. The
human half is the owner: ten minutes per look with the misery prompt on.

**Kill.** His ratings don't differ between looks. That is still an answer: for him it is
seated first-person play, not the dream.

### 8. A still periphery

**Idea.** Self-motion is sensed mostly in the periphery, and flicker is most visible
there; detail is judged in the centre. Make the dream eccentricity-aware: outside the
central ~40% of the screen, existing paint changes slowly (replaced only when a new view is
clearly finer, over a second or more) while undreamt surfaces still fill fast. While
turning or walking briskly, the outer 10-15% eases into a soft luminous mist in the zone's
colour and clears within half a second of stopping: a dynamic field-of-view restrictor
dressed as a tunnel, one of Klüver's form constants. Variant: a faint static canvas weave
over the frame, a background that stays still when you move.

**Science.** Peripheral vision dominates vection (Brandt, Dichgans & Koenig 1973). Subtle
dynamic field-of-view restriction reduced VR sickness and mostly went unnoticed (Fernandes
& Feiner 2016). A background that stays still with the body reduces simulator sickness
(rest frames: Prothero 1998; Prothero et al. 1999). Critical flicker frequency rises with
eccentricity for large targets (Hartmann, Lachenmayr & Brettel 1979). The canvas as a rest
frame is a guess.

**Why it matters here.** It works on the nausea and also hides a known weakness, screen
edges and fresh disocclusions falling back to the soft atlas or the blueprint, in the
dream's own style.

**Cost.** About a day: an eccentricity term in the live fade weights (slow peripheral
replacement needs views that outlive their fades; see idea 5), a motion uniform and the
mist in `post.js`, a procedural weave.

**Two-hour experiment.** The mist alone behind a flag (it doesn't need the accumulator);
walking and turning strips in 8 zones; visible change in the outer 30% (idea 5's meter) and
the share of edge pixels showing blueprint or atlas-only paint. Blind reviewer: deliberate
effect or a pumping vignette? Success: peripheral visible change down 40%, edge blueprint
halved, called deliberate in 6 of 8, no "pumping".

**Kill.** It reads as tunnel vision or pumping, to the reviewer or the owner; or the weave
reads as a dirty screen.

### 9. Panini dream

**Idea.** Render and capture with a Panini projection: vertical lines stay straight, the
left and right edges are compressed, and in a turn the edge-to-centre speed ratio falls
from ~2.6x to ~1.2x at today's width (computed, Panini d = 1); the centre looks about 1.3x
larger at the same width. For the dream, warp the capture into Panini before the model
sees it and unwarp at projection time, with the same machinery as idea 1; the two can be
one mapping.

**Science.** Rectilinear projection can't render natural-looking views much wider than
~70° (Sharpless, Postle & German 2010). Peripheral flow drives vection (Brandt et al.
1973). The model's training photographs are mostly much narrower than 110° (guess), so
stretched edges are outside its experience.

**Why it matters here.** Possibly less sickness in turns, and edges the model can read (some
of the flat, smeared edge paint seen in reviews may be stretch; guess).

**Cost.** Display only: half a day (render wider, warp in the composite pass, keep the
depth-based post effects in warped coordinates). Capture: 1-2 days, shared with idea 1.

**Two-hour experiment.** A display-only Panini flag; rest and turning strips in 8 zones;
blind reviewer: which looks more natural and less distorted? Success: Panini preferred in
6 of 8. Then a misery-prompt A/B for the owner.

**Kill.** Horizontals curve and wobble as you look up and down (reviewer on pitch strips, or
the owner).

### 10. Dream in colour, see in luminance

**Idea.** Let the geometry own the picture's low-frequency luminance, the shading that
tells a pillar from the wall, and let the dream own colour and fine texture. In the display
shader: out = dream × (formShade / dreamTone)^k, where formShade is a smooth shading term
from the real normals (light from above, the point lights, a rim term; most of it is already
computed there) and dreamTone is the dream's own low-passed luminance (the atlas's blurred
mip or a low mip of the live view), each normalised by a zone-level mean, with k around
0.3-0.6. The luminance layout then stays as steady as the geometry, while colour, which the
eye follows less well at the dream's rate, carries the fast part of the dream.

**Science.** Depth, shape from shading and motion weaken at isoluminance (Livingstone &
Hubel 1987, whose strict channel split is disputed; isoluminant gratings look slower,
Cavanagh, Tyler & Favreau 1984). Shape from shading with a light-from-above prior
(Ramachandran 1988). Chromatic temporal sensitivity is low-pass, luminance's reaches higher
frequencies (Kelly 1983).

**Why it matters here.** The most direct perceptual answer to "the pillar blends into the
wall unless I move", and a second route to calmer rest paint that doesn't drain colour.

**Cost.** About a day in the display shader. A capture-side twin could anchor the model's
input luminance at low frequency only; the existing `uFeedbackAnchor` (0.35) does it at
full frequency.

**Two-hour experiment.** `?formlight=k` with k = 0, 0.35, 0.6; 8 zones at rest, 2 runs.
Measure the planned silhouette contrast, rest change in CIELAB ΔE (a luma-only measure
would reward luminance anchoring for nothing), and detail. Blind reviewer: which shows the
shapes most clearly, and which looks more like a painting than a render? Success:
silhouette contrast up 25% in 6 of 8 at k = 0.35, the painting rating no worse in 5 of 8.

**Kill.** The reviewer calls it "CG" or "plastic", or it cancels the dream's own light (the
nave's shafts) in 3 zones or more.

### 11. Depth you feel when parallax stops

**Idea.** At rest the eye loses motion parallax, which is exactly when the pillar merged.
So fade pictorial depth in as the camera settles and out as it moves, over about a second:
a depth-buffer unsharp mask (surfaces in front of their surroundings lifted a few percent,
the background beside them lowered, over about half a degree, a Cornsweet profile, so the
whole pillar reads as separate while the paint inside it is untouched), plus aerial
perspective (contrast falling with distance, a little fog hue beyond ~6 m). Walking,
parallax does this job and the image stays clean. Leave out depth of field: without gaze
tracking the blur lands where you look (Mauderer et al. 2014).

**Science.** Depth cues are weighted by reliability (Landy, Maloney, Johnston & Young
1995); parallax (Rogers & Graham 1979); unsharp masking the depth buffer (Luft, Colditz &
Deussen 2006); the Craik-O'Brien-Cornsweet effect (Cornsweet 1970); contrast as a depth cue
(O'Shea, Blackburn & Ono 1994).

**Why it matters here.** Cheap, and it adds two rules to M114's planned separation pass:
when (only at rest) and how (faint edge profiles instead of outlines).

**Cost.** Half a day to a day in `post.js`: a 2-3 level blurred depth chain and a motion
uniform.

**Two-hour experiment.** A flag; pillar-and-wall rest shots in 8 zones; the planned
silhouette measure; blind reviewer: any halos or outlines, and what is in front of what?
Plus a strip across a stop to catch the cue popping in. Success: silhouette measure up 20%
in 6 of 8, halos flagged in at most 1.

**Kill.** Halos, or a visible pop when it fades in.

### 12. Pillars that breathe on their own beat (speculative)

**Idea.** Things that change together are seen as one thing. At rest, give each surface
chart its own slow phase: modulate its paint by ±1-2% at ~0.5 Hz, with phases anchored to
the world (a hash of the chart), so a pillar and the wall behind it brighten out of step.
The phase must be world-anchored and the effect rest-only: a pattern keyed to distance from
the camera would slide along surfaces as you walk, a new motion signal.

**Science.** Shape from temporal structure alone (Lee & Blake 1999), disputed as a
low-level artifact (Farid & Adelson 2001); common fate (Wertheimer 1923). Whether 1-2% is
enough to separate a pillar is a guess.

**Why it matters here.** If it works, the pillar separates at rest with no outlines and no
camera motion, and the room gains a breathing quality.

**Cost.** An hour in the display shader.

**Two-hour experiment.** Headless reviewers can't judge grouping in time, so the headless
part only checks safety: idea 5's meter within 10% of default. Then a look for the owner:
does the pillar stand out at rest with it on and not off, with no pulse noticed?

**Kill.** A visible pulse, or no difference to the owner.

### 13. Close your eyes [BET]

**Idea.** One new verb: hold a key (a two-finger hold on a phone) to close your eyes, and
the view sinks over ~0.6 s into the warm red-brown of light through eyelids, with faint
slow phosphenes and form constants drifting in it, while the sound dulls and a heartbeat
comes forward. Behind the lids the dream keeps working on the room: higher strength, a
sister prompt, a sweep of side glances, perhaps paint borrowed from zones visited earlier.
Open your eyes and the room is different, and you never saw it change. After a long idle
the eyes drift shut by themselves for a few seconds (a hypnagogic nod), so a passive player
gets it too.

**Science.** Changes made during blinks or blank intervals go unnoticed (O'Regan, Deubel,
Clark & Rensink 2000; Rensink, O'Regan & Clark 1997); memory across an interruption keeps
little detail (Irwin 1991). Sleep-onset imagery replays recent experience (Stickgold et al.
2000) and often begins with geometric, entoptic forms (Klüver 1966; Mavromatis 1987).
Things that change when you look away and back are a classic sign of dreaming, used by
lucid dreamers as a reality check (LaBerge & Rheingold 1990).

**Why it matters here.** It turns the model's weakness, re-invention, into the game's
signature moment, with no motion in it and a change nobody can see happen. It delivers the
charter's "rooms that reconfigure while you're away" without touching geometry, for one key
of UI.

**Cost.** 2-3 days for a tasteful version: an eyelid and phosphene layer in `post.js`, an
audio duck, an eyes-closed mode in `dream.js` (strength, prompt, glances; the live views
cleared on opening so the new paint shows), the idle trigger.

**Two-hour experiment.** A key toggles `closed`: the display fades to a flat eyelid colour;
captures run at strength +0.15 and lower feedback, with the zone prompt plus a sister
clause; open after 4 s. In 8 zones save the frame before closing and 1 s after opening,
plus a 5 Hz strip of the whole close and open. Blind reviewer: same place? what changed?
any harsh or strobe-like moment? which frame is more beautiful? Success: a clear, nameable
change in 6 of 8, no harsh moment, the after-frame at least as beautiful in 5 of 8.

**Kill.** Four seconds can't move the picture far past the feedback anchor, or the rewrite
turns to mush.

### 14. The room deepens while you stand still

**Idea.** Standing still should feel like drifting off. Over the first minute at rest,
blend the prompt embedding from the zone's waking prompt toward a deeper sister prompt, in
steps too small to see (one every 2 s, so DeepCache and the cross-frame anchor keep working
between steps), while idea 5's rest accumulation keeps visible change under the meter's
threshold. You never see anything change, but after a minute the nave is half coral.
Walking glides it back over ~10 s, under motion's cover.

**Science.** Gradual change blindness (Simons, Franconeri & Reimer 2000); motion silencing
(Suchow & Alvarez 2011).

**Why it matters here.** More intensity with no visible change: the "was it always like
that?" feeling of dreams, at no cost in comfort.

**Cost.** About a day: optional frame-header fields (`prompt2`, `mix`; the protocol allows
extra fields) and an embedding blend in the engine; a rest timer in the client.

**Two-hour experiment.** 60 s holds in 4 zones, ramped and not; 5 Hz strips. Measures:
visible change per second (within 15% of the unramped hold) and the difference between
t = 0 and t = 60 (3x the control or more). Blind reviewer on consecutive pairs ("did
anything change?") and first/last pairs. Success: consecutive pairs judged unchanged as
often as the control's, first and last clearly different in 3 of 4.

**Kill.** The feedback loop holds on to the old dream, the deep prompt turns to mush, or
the steps show as jumps.

### 15. Threshold mist

**Idea.** At a zone border the whole view is re-dreamt under a new prompt, the largest
change the game makes, in plain sight. Cross through a luminous mist instead: over ~1.5 s
the view dissolves into a near-uniform glow in the next zone's fog colour (full density for
at most half a second) while the capture, which sees through the mist, repaints with the new
prompt, and the new zone condenses out of it. While the view is uniform, the model can be
shown the mist itself at high strength so faint shapes surface in it before the
architecture returns. A side note from the code: the capture's prompt follows the camera's
zone, so looking back across a border repaints the previous zone with the new prompt.

**Science.** A blank between two views hides the change between them (Rensink, O'Regan &
Clark 1997). A homogeneous field (Ganzfeld) invites imagery (Wackermann, Pütz & Allefeld
2008), though in people it takes minutes, so this borrows only the blank and lets the model
supply the imagery.

**Why it matters here.** The most visible change becomes the most dreamlike moment, next to
the zone title card that already marks it.

**Cost.** About a day: a mist term in the display fog driven by time since the zone changed
(or distance to the border), and the optional mist capture.

**Two-hour experiment.** Three crossings on the tour, strips with and without; the
visible-change meter across the crossing; blind reviewer: harsh? abrupt? which is better?
Success: peak visible change halved, the mist preferred in 2 of 3.

**Kill.** It reads as a loading fade, or walking blind for half a second is unpleasant to
the owner.

### 16. Things at the corner of your eye

**Idea.** Peripheral vision can tell that something is there but not what it is. Let the
side glances, which today paint only the atlas at rest, dream at higher strength with
presence words in their prompt (statues, lanterns, plants, doorways), so the edges of a room
fill with suggestive shapes. Turn to look, and the centre capture's more faithful paint
replaces them: they dissolve under your gaze. Because it is anchored to the world, it reads
as a richer corner of the room that stays put when you turn, which a lens never does.

**Science.** Crowding limits identification in the periphery (Pelli & Tillman 2008);
peripheral vision keeps summary statistics (Rosenholtz 2016); presences at the edge of
vision are a hypnagogic staple (Mavromatis 1987).

**Why it matters here.** It turns the known ghostly figures into a controlled effect,
placed where it is felt rather than examined.

**Cost.** Half a day to a day: a per-kind strength and prompt for glance captures (already
their own stream and seed); centre views overwrite glance paint quickly.

**Two-hour experiment.** Rest 20 s in 8 zones with it on, then turn 60°. Blind reviewer:
describe the objects at the edges of the first frame, then the same place once it is in
front. Success: different, plausible objects in 5 of 8, none grotesque.

**Kill.** People or faces appear (SD-Turbo ignores negative prompts, and the charter wants
beauty, not horror), or the dissolve pops.

### 17. Peripheral metamers

**Idea.** The periphery registers summary statistics, so it can be given detail that is
statistically right rather than true. Harvest small exemplars per surface type from recent
central results (one 128² patch a second), and where paint is coarse (atlas-only or
wide-only surfaces, screen edges, fresh disocclusions) add their high-pass detail, laid out
in world space with histogram-preserving random tiling and scaled to the local contrast.
It is a learned version of today's procedural detail map, fed by the model's own brushwork.

**Science.** Metamers of the ventral stream (Freeman & Simoncelli 2011); summary
statistics in peripheral vision (Balas, Nakano & Rosenholtz 2009; Rosenholtz 2016), with
the caveat that structured scenes are harder to fool than textures (Wallis et al. 2019);
contrast restoration hides foveation (Patney et al. 2016); histogram-preserving tiling
(Heitz & Neyret 2018).

**Why it matters here.** Sharper-looking edges and disocclusions, and a smaller detail gap
between wide and narrow paint, at no diffusion cost.

**Cost.** 2-3 days.

**Two-hour experiment.** Offline on rest frames, in the centre where real narrow paint
exists: compare wide paint, wide paint plus synthesised detail, and the real narrow paint.
Show a reviewer each version blurred as a fixating eye 15° away would see it, and
unblurred. Success: at the simulated 15°, synthesised and real are told apart no better
than chance over 8 pairs, while plain wide is told apart.

**Kill.** The synthetic detail looks pasted on (repeats, wrong scale) even in the blurred
presentation.

---

## Crazy ideas (weeks, if they work)

### C1. The machine eye [CRAZY]

**Idea.** Build or vendor a foveated spatiotemporal visibility model of the FovVideoVDP or
ColorVideoVDP kind, with an eccentricity map for central fixation plus a gaze-sweep mode,
and put it in the loop. Every harness run reports the predicted visibility of flicker, the
lens and seams in JOD units; an optimiser tunes a few dozen knobs (fades, mixes, fovea,
feedback, sharpening) to minimise visible artifacts at fixed evolution and detail; and blind
reviewers get what a fixating eye sees (eccentricity-blurred, crowded renderings) instead of
stills that are foveal everywhere.

**Science.** FovVideoVDP (Mantiuk et al. 2021) and ColorVideoVDP (Mantiuk et al. 2024)
predict visible differences in wide-field video, accounting for eccentricity and temporal
frequency; attention-aware sensitivity (Krajancich, Kellnhofer & Wetzstein 2023); peripheral
appearance from summary statistics (Balas, Nakano & Rosenholtz 2009).

**Why it matters here.** Most of this list turns on whether the eye can see a given
change. Today each answer costs a blind review or a playtest; a machine eye would answer per
commit, overnight, for hundreds of variants.

**Cost.** Weeks: implementation or vendoring (ColorVideoVDP is open source; check licence
and install size first, and record any toolchain step in the charter), validation, the
optimiser, GPU time shared with the dream server.

**Two-hour experiment.** A numpy stand-in: three spatial bands times a temporal band-pass,
eccentricity-weighted, on default, `?rate=8`, `?fovea=1` and M111's final build. Success: it
orders steadiness as the committed blind reviews did in 6 of 8 zones, and shows the lens
ring at fovea 1.7 and not at 1.

**Kill.** It doesn't agree with the blind reviews; then invest in the full VDP or drop it.

### C2. Retina-native dreaming [CRAZY]

**Idea.** If idea 1 works but the model mis-paints warped images (bowed lines, stretched
strokes), teach it. Distil a warp-equivariant SD-Turbo: a LoRA student trained so that
student(warp(x)) ≈ warp(teacher(x)) over thousands of level captures across zones and
poses. Then go further, to a log-polar capture on the retina's own sampling grid, where
brushstrokes grow with eccentricity as receptive fields do, with a Panini outer mapping
(idea 9) so the edges aren't stretched.

**Science.** Log-polar retinotopy (Schwartz 1977); acuity and receptive-field size grow
roughly linearly with eccentricity (Rovamo & Virsu 1979; Strasburger et al. 2011). That a
one-step diffusion model can learn warp-equivariance cheaply is a guess.

**Why it matters here.** The pixel budget follows the eye in every frame, and a phone could
run a smaller budget at the same perceived quality.

**Cost.** Weeks: bulk capture dumps from the harness, days of LoRA training on the shared
GPU, evaluation.

**Two-hour experiment.** SD-Turbo (strength 0.6, four feedback passes) on 20 captures from
`bench/scenes.py`, rectilinear against warped-then-unwarped at two warp strengths; measure
column straightness (line fits, rectilinear result against unwarped result) and detail
against eccentricity. If deviations stay under ~1°, the training is unnecessary, which is
good news for idea 1. If they are large, a short LoRA pilot (under an hour, ~200 pairs)
shows whether they shrink.

**Kill.** Large deviations that the pilot doesn't reduce.

### C3. Eyes as input [CRAZY]

**Idea.** With consent, entirely on the device: a face-landmark model in the page tracks the
eyelids and a coarse gaze (a few degrees). Blinks become free change windows (pending swaps
commit during the ~100-200 ms of blink suppression); the retina capture's centre follows the
gaze; a long eye closure triggers idea 13's rewrite; blink rate and eyelid droop give a
drowsiness signal, so the dream deepens as you do.

**Science.** Visual suppression during blinks (Volkmann, Riggs & Moore 1980); blink change
blindness (O'Regan et al. 2000); webcam gaze (WebGazer, Papoutsaki et al. 2016); blink
detection from landmarks (Soukupová & Čech 2016). Foveated rendering tolerates 50-70 ms of
total latency (Albert et al. 2017), which a webcam pipeline may miss, so the fovea must stay
broad.

**Why it matters here.** The only way to know where the player is looking, and the purest
form of changing the dream when the eye can't see it.

**Cost.** Weeks: a vendored landmark model (a download that needs the owner's approval),
calibration, WebKit performance, privacy. A blocker to settle first: browsers allow camera
access only in a secure context (HTTPS or localhost), and the game is played over plain
HTTP from another machine.

**Two-hour experiment.** Nothing with a camera yet. Idea 13's key already tests whether
rewrites behind closed lids are worth having; meanwhile measure frame time with a stub
inference load in a worker (6 ms every 33 ms). Success: frame p95 within 0.5 ms, and the
owner likes idea 13.

**Kill.** The owner doesn't want a camera on, or idea 13 falls flat.

### C4. Change-blind architecture [CRAZY]

**Idea.** The geometry changes while nobody watches: a door you came through becomes a
wall, a corridor grows longer, a stair now climbs somewhere else. Swap only when every
affected surface has been outside the view, the capture frusta and the live views for over
a second, or during idea 13's closed eyes. The level generator builds the alternates with
their own atlas charts, so the dream paints them like everything else and the memory keeps
each variant.

**Science.** In VR, doors moved behind people's backs; 1 of 77 participants noticed (Suma
et al. 2011). Self-overlapping "impossible spaces" also pass unnoticed up to a point (Suma
et al. 2012). Scene representations are sparse (Rensink 2000).

**Why it matters here.** It is the charter's "rooms that reconfigure while you're away" and
"non-Euclidean passages", uncanny rather than jarring because nobody sees the change.

**Cost.** Weeks (world owner): alternates in `levelcore.js`, collision swaps, atlas space
for alternates, live-view invalidation, memory per variant, and reachability guarantees so
nobody gets trapped.

**Two-hour experiment.** One hand-made alternate (a wall against a doorway) in the test
level; a scripted walk that looks away and back. Check for paint leaks (a live view
projecting the old wall's paint onto the new doorway), the collision swap and frame time.
Success: a clean swap in 10 of 10 runs with no old paint visible.

**Kill.** Old paint can't be kept off the new geometry without clearing the whole dream, or
reachability can't be guaranteed.

### C5. A visual cortex in the loop [CRAZY]

**Idea.** Simulate a small neural field on a cortical grid (excitatory and inhibitory
activity with Mexican-hat coupling, whose instabilities form stripes and hexagons) and map
it to the visual field through the inverse log-polar map, where the stripes become tunnels,
spirals and lattices: Klüver's form constants. Use them as seed structure in the dream's
hypnagogic states, as faint luminance in the eyelid view (13), the threshold mist (15) and
undreamt blueprint, so the model grows arches, vaults and rose windows along the shapes real
hallucinations take. Coupling and drive become the depth of sleep. The patterns must stay
still or morph slowly: a rotating spiral produces illusory motion and roll vection, and the
flicker that drives lab-induced hallucinations is strobing, which the charter rules out.

**Science.** Form constants (Klüver 1966); their derivation from pattern formation in V1
and the retinotopic map (Ermentrout & Cowan 1979; Bressloff, Cowan, Golubitsky, Thomas &
Wiener 2001).

**Why it matters here.** It gives the dream states a source of imagery that is hypnagogic
by construction instead of stock prompt words.

**Cost.** Weeks for a tuned version; the GPU neural field is about a day, the art direction
and integration are the rest.

**Two-hour experiment.** A numpy neural field on a 128² grid through the log-polar map
gives six patterns (tunnel, spiral, lattice, cobweb and two mixed); overlay them at 10-20%
luminance on six blueprint or mist captures; one SD-Turbo pass each with the zone prompts
(engine in-process). Blind reviewer: describe the architecture. Success: in half or more,
the described structure follows the pattern ("concentric arches", "a spiral stair", "a
lattice vault").

**Kill.** The model ignores faint patterns (strong ones would show as stripes), or paints
them as wallpaper rather than architecture.

---

## Sources

- Albert, Patney, Luebke & Kim 2017. Latency requirements for foveated rendering in virtual reality. ACM Trans. Applied Perception 14(4).
- Anstis 1970. Phi movement as a subtraction process. Vision Research 10.
- Balas, Nakano & Rosenholtz 2009. A summary-statistic representation in peripheral vision explains visual crowding. Journal of Vision 9(12).
- Bergen & Adelson 1988. Early vision and texture perception. Nature 333.
- Bos, MacKinnon & Patterson 2005. Motion sickness symptoms in a ship motion simulator (the MISC scale). Aviation, Space, and Environmental Medicine 76.
- Brandt, Dichgans & Koenig 1973. Differential effects of central versus peripheral vision on egocentric and exocentric motion perception. Experimental Brain Research 16.
- Bressloff, Cowan, Golubitsky, Thomas & Wiener 2001. Geometric visual hallucinations, Euclidean symmetry and the functional architecture of striate cortex. Phil. Trans. R. Soc. B 356.
- Burley 2019. On histogram-preserving blending for randomized texture tiling. JCGT 8(4).
- Burt & Adelson 1983. A multiresolution spline with application to image mosaics. ACM Trans. Graphics 2(4).
- Cavanagh, Tyler & Favreau 1984. Perceived velocity of moving chromatic gratings. JOSA A 1(8).
- Choi, Bovik & Cormack 2014. Spatiotemporal flicker detector model of motion silencing. Perception 43.
- Cornsweet 1970. Visual Perception. Academic Press.
- Daniel & Whitteridge 1961. The representation of the visual field on the cerebral cortex in monkeys. J. Physiology 159.
- Diels & Howarth 2013. Frequency characteristics of visually induced motion sickness. Human Factors 55(3).
- Draper, Viirre, Furness & Gawron 2001. Effects of image scale and system time delay on simulator sickness within head-coupled virtual environments. Human Factors 43(1).
- Ermentrout & Cowan 1979. A mathematical theory of visual hallucination patterns. Biological Cybernetics 34.
- Farid & Adelson 2001. Synchrony does not promote grouping in temporally structured displays. Nature Neuroscience 4.
- Fernandes & Feiner 2016. Combating VR sickness through subtle dynamic field-of-view modification. IEEE 3DUI.
- Field, Hayes & Hess 1993. Contour integration by the human visual system: evidence for a local "association field". Vision Research 33.
- Freeman & Simoncelli 2011. Metamers of the ventral stream. Nature Neuroscience 14.
- Hartmann, Lachenmayr & Brettel 1979. The peripheral critical flicker frequency. Vision Research 19.
- Heitz & Neyret 2018. High-performance by-example noise using a histogram-preserving blending operator. Proc. ACM CGIT (HPG).
- Irwin 1991. Information integration across saccadic eye movements. Cognitive Psychology 23.
- Kelly 1979. Motion and vision II: stabilized spatio-temporal threshold surface. JOSA 69.
- Kelly 1983. Spatiotemporal variation of chromatic and achromatic contrast thresholds. JOSA 73(6).
- Kenny, Koesling, Delaney, McLoone & Ward 2005. A preliminary investigation into eye gaze data in a first person shooter game. ECMS.
- Keshavarz & Hecht 2011. Validating an efficient method to quantify motion sickness (the FMS). Human Factors 53(4).
- Keshavarz & Hecht 2014. Pleasant music as a countermeasure against visually induced motion sickness. Applied Ergonomics 45(3).
- Keshavarz, Riecke, Hettinger & Campos 2015. Vection and visually induced motion sickness: how are they related? Frontiers in Psychology 6.
- Klüver 1966. Mescal and Mechanisms of Hallucinations. University of Chicago Press.
- Krajancich, Kellnhofer & Wetzstein 2023. Towards attention-aware foveated rendering. ACM Trans. Graphics 42(4).
- LaBerge & Rheingold 1990. Exploring the World of Lucid Dreaming. Ballantine.
- Land & Lee 1994. Where we look when we steer. Nature 369.
- Landy, Maloney, Johnston & Young 1995. Measurement and modeling of depth cue combination: in defense of weak fusion. Vision Research 35.
- Lee & Blake 1999. Visual form created solely from temporal structure. Science 284.
- Lee & Lishman 1975. Visual proprioceptive control of stance. J. Human Movement Studies 1.
- Legge & Foley 1980. Contrast masking in human vision. JOSA 70.
- Lin, Duh, Parker, Abi-Rached & Furness 2002. Effects of field of view on presence, enjoyment, memory, and simulator sickness in a virtual environment. IEEE VR.
- Livingstone & Hubel 1987. Psychophysical evidence for separate channels for the perception of form, color, movement, and depth. J. Neuroscience 7.
- Luft, Colditz & Deussen 2006. Image enhancement by unsharp masking the depth buffer. ACM Trans. Graphics 25(3).
- Malik & Perona 1990. Preattentive texture discrimination with early vision mechanisms. JOSA A 7.
- Mantiuk et al. 2021. FovVideoVDP: a visible difference predictor for wide field-of-view video. ACM Trans. Graphics 40(4).
- Mantiuk et al. 2024. ColorVideoVDP: a visual difference predictor for image, video and display distortions. ACM Trans. Graphics 43(4).
- Matthis, Yates & Hayhoe 2018. Gaze and the control of foot placement when walking in natural terrain. Current Biology 28.
- Mauderer, Conte, Nacenta & Vishwanath 2014. Depth perception with gaze-contingent depth of field. CHI.
- Mavromatis 1987. Hypnagogia: The Unique State of Consciousness Between Wakefulness and Sleep. Routledge.
- O'Regan, Deubel, Clark & Rensink 2000. Picture changes during blinks: looking without seeing and seeing without looking. Visual Cognition 7.
- O'Shea, Blackburn & Ono 1994. Contrast as a depth cue. Vision Research 34.
- Oman 1990. Motion sickness: a synthesis and evaluation of the sensory conflict theory. Can. J. Physiology and Pharmacology 68.
- Palmisano, Gillam & Blackburn 2000. Global-perspective jitter improves vection in central vision. Perception 29.
- Papoutsaki et al. 2016. WebGazer: scalable webcam eye tracking using user interactions. IJCAI.
- Patney et al. 2016. Towards foveated rendering for gaze-tracked virtual reality. ACM Trans. Graphics 35(6).
- Pelli & Tillman 2008. The uncrowded window of object recognition. Nature Neuroscience 11.
- Prothero 1998. The role of rest frames in vection, presence and motion sickness. PhD thesis, University of Washington.
- Prothero, Draper, Furness, Parker & Wells 1999. The use of an independent visual background to reduce simulator side-effects. Aviation, Space, and Environmental Medicine 70(3).
- Ramachandran 1988. Perception of shape from shading. Nature 331.
- Reason & Brand 1975. Motion Sickness. Academic Press.
- Rensink 2000. The dynamic representation of scenes. Visual Cognition 7.
- Rensink, O'Regan & Clark 1997. To see or not to see: the need for attention to perceive changes in scenes. Psychological Science 8.
- Rogers & Graham 1979. Motion parallax as an independent cue for depth perception. Perception 8.
- Rosenholtz 2016. Capabilities and limitations of peripheral vision. Annual Review of Vision Science 2.
- Rovamo & Virsu 1979. An estimation and application of the human cortical magnification factor. Experimental Brain Research 37.
- Russell, Hoffman, Stromberg & Carlson 2014. Use of controlled diaphragmatic breathing for the management of motion sickness in a virtual reality environment. Applied Psychophysiology and Biofeedback 39.
- Sauer, Scherff, Lappe, Rifai, Stein & Wahl 2021. Self-motion illusions from distorted optic flow in multifocal glasses. iScience.
- Schwartz 1977. Spatial mapping in the primate sensory projection: analytic structure and relevance to perception. Biological Cybernetics 25.
- Sharpless, Postle & German 2010. Pannini: a new projection for rendering wide angle perspective images. Computational Aesthetics.
- Simons, Franconeri & Reimer 2000. Change blindness in the absence of a visual disruption. Perception 29.
- Soukupová & Čech 2016. Real-time eye blink detection using facial landmarks. CVWW.
- Stickgold, Malia, Maguire, Roddenberry & O'Connor 2000. Replaying the game: hypnagogic images in normals and amnesics. Science 290.
- Strasburger, Rentschler & Jüttner 2011. Peripheral vision and pattern recognition: a review. Journal of Vision 11(5).
- Suchow & Alvarez 2011. Motion silences awareness of visual change. Current Biology 21(2).
- Suma, Clark, Krum, Finkelstein, Bolas & Wartell 2011. Leveraging change blindness for redirection in virtual environments. IEEE VR.
- Suma, Lipps, Finkelstein, Krum & Bolas 2012. Impossible spaces: maximizing natural walking in virtual environments with self-overlapping architecture. IEEE TVCG 18(4).
- Tariq & Didyk 2024. Towards motion metamers for foveated rendering. ACM Trans. Graphics 43(4).
- Tatler 2007. The central fixation bias in scene viewing. Journal of Vision 7(14).
- Thompson 1982. Perceived rate of movement depends on contrast. Vision Research 22.
- Volkmann, Riggs & Moore 1980. Eyeblinks and visual suppression. Science 207.
- Wackermann, Pütz & Allefeld 2008. Ganzfeld-induced hallucinatory experience, its phenomenology and cerebral electrophysiology. Cortex 44.
- Wallis et al. 2019. Image content is more important than Bouma's law for scene metamers. eLife 8.
- Watson 1979. Probability summation over time. Vision Research 19.
- Wertheimer 1912. Experimentelle Studien über das Sehen von Bewegung. Zeitschrift für Psychologie 61.
- Wertheimer 1923. Untersuchungen zur Lehre von der Gestalt II. Psychologische Forschung 4.
- Yantis & Jonides 1984. Abrupt visual onsets and selective attention. J. Exp. Psychology: HPP 10.
