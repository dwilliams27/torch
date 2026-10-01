# HYPNAGOGIA — design contract

> A first-person walk through impossible architecture that is being *dreamt into
> existence* by a local diffusion model as you look at it.

This document is the shared contract between every part of the system. If you
own one part, you may change anything *inside* your part, but the interfaces in
this file are frozen unless the orchestrator changes them.

## The core idea (why this isn't strobe-y)

The old project ran img2img on the whole screen and blitted the result: every
diffusion frame was a brand-new image → flicker, and the game ran at the
diffusion rate.

New approach: **world-space dream painting.**

1. The browser renders a true 3D level at 60+ FPS with Three.js.
2. A few times per second it captures its current view (a *capture*: color at
   diffusion resolution + depth + the camera matrices) and sends the color to a
   local diffusion server.
3. The server returns a stylized image (SD-Turbo/SDXS-class 1-step img2img).
4. The client **projects that image back onto the world geometry** and blends it
   into a persistent per-surface texture — the *dream atlas* — with an
   exponential moving average and a visibility (shadow-map style) test against
   the capture's depth.
5. The newest few results (five on desktop, three on phones) are also kept whole,
   image plus capture depth and camera, as *live views* (`client/src/dream/live.js`).
   They are projected onto the geometry at display time, newest over older over the
   atlas, so near surfaces show the model's own pixels rather than the atlas's ~13
   texels/m. A result from a new framing takes over as it fades in. A held framing (pose
   and width, standing still) keeps one view whose picture follows each newer result of
   that framing with a time constant (`?livetau`, 0.35 s), frame by frame: a running
   average, so the model's per-result jitter averages away without the jumps a stack of
   the last few results makes when its oldest is dropped, and as calmly at 12 dreams a
   second as at 7. **No foveation by default.** Separate narrow captures (`?fovea=1.7`:
   centre captures alternating narrow and wide) showed as a lens over ~40% of the view
   (2026-09-29): the narrow views are painted by their own seed and stream and differ from
   the wide paint around them. A foveal warp inside one capture (`?warp=1.5`: per axis a
   perspective coordinate p lands on the image at q = m p / (1 + (m - 1)|p|); everything
   that projects a result goes through the same map) has no border, but measured calmer
   and less detailed, even in the middle of the view, for 2.25x the capture's pixels, so it
   is off too. Where views overlap, a view coarser than the paint already there mostly
   lets the finer paint show through, without uncovering the atlas.
6. The final frame renders every surface with that paint (falling back to a raw
   "undreamt" look where nothing is painted yet; undreamt slivers next to dreamt
   pixels borrow their colour in screen space).

Consequences: the game always renders at display rate; paint is spatially
anchored so it doesn't swim when you turn; the atlas only nudges what each result
saw (EMA), so the world *breathes and evolves* instead of strobing; unseen areas
are visibly "not yet dreamt" and wash in as you look; walking back through a zone
shows what the machine dreamt there, even after a reload (the atlas is kept in
IndexedDB, `client/src/dream/memory.js`). Diffusion FPS only controls how fast the
paint evolves, not how smooth the game feels.

Feedback: the image sent to the server is the *already painted* view (the live views
where they reach, else the atlas, mixed with the raw render by a `feedback` factor),
so the model refines its own prior dream → strong temporal coherence, with slow
artistic drift. Standing still, one held framing gets many passes and converges crisp at
modest feedback (0.55; more makes the loop simplify the level and drift in colour).
Walking, a view gets two or three passes (below), so feedback rises
with motion (x1.3, 0.55 to ~0.72; `?fbmove`) and each new view refines the reprojected
dream. A framing is held until the view moves `?kfdist=0.2` m or turns `?kfangle=0.06` rad,
so walking is not free of them: at 1.7 m/s with the pool (3 sent, ~16 captures a second), 58%
of walking captures repeat their stream's framing (M112, `tools/framing_probe.mjs`). A result
projected from where you were half a second ago stretches along your motion, so walking
captures are aimed where you will be when the result shows (capture-to-send time plus
round trip plus ~0.12 s, position lead capped at 0.4 s). A capture pixel spans ~4 screen
pixels across at 1080p, so live pixels get contrast-adaptive sharpening on display
(`?sharpen`, 0..1); the capture reads them unsharpened, since sharpening inside the
feedback loop would compound. (`?calm=0.25` drops strength while moving instead; it
visibly mutes the dream and is off by default.)

Closed eyes: holding E (or the panel's "close your eyes", which blinks for 4 s) sinks the
view into a warm eyelid glow. Once the lid is shut (a quick tap changes nothing), captures
continue with a sister prompt put in front of the zone's own (overgrown, winter, glass,
dawn, ...), feedback 0.1, a running average of 0.12 s, and, on a depth engine, strength
0.9; each such frame carries `cut`, so the engine switches prompts at once instead of
gliding. The depth graft keeps the architecture, so the room opens dreamt again in the same
shape. Each zone keeps its variant until its next blink or "forget".

Geometry: stock SD-Turbo at strength 0.6 keeps an input's tonal layout but paints over
structure whose tone matches its background (a stone pillar in front of a stone wall
became part of the wall), and no input-side cue fixed that. So the engine takes depth:
with `torch_turbo`'s depth graft (Engine interface below) each frame carries the capture's
relative inverse depth (averaged from the capture's own depth buffer over each latent
pixel), and the model paints what the geometry separates; strength rose from 0.6 to 0.66,
since depth now holds the shapes. On the display, `?sep` darkens what sits just behind a
nearer silhouette (unsharp masking the depth buffer, Luft et al. 2006) and `?relight`
darkens surfaces turning away from the eye. Both are off by default and on in the `lucid`
look: where the paint disagrees with the geometry, a strong cue shows the real geometry
through it as a ghost.

## Repo layout

```
run.sh                 one-command launcher (creates a venv with uv, starts server)
server/                Python: diffusion engines + HTTP/WebSocket server
  __main__.py          `python -m server --engine auto --port 8765`
  app.py               aiohttp app: static client, /api/info, /ws
  engines/<name>.py    each exposes create_engine(**cfg) -> Engine
  requirements.txt
client/                the game — plain ES modules, NO build step
  index.html           importmap: "three" -> ./vendor/three.module.min.js,
                       "three/addons/" -> ./vendor/addons/
  vendor/              vendored three.js (r17x) module + needed addons
  src/main.js          bootstrap + main loop                (render owner)
  src/world/*          level gen, zones, player, input      (world owner)
  src/dream/*          capture, link (ws), painter, live views,
                       dream memory (IndexedDB)             (render owner)
  src/render/*         materials, post-fx, hud              (render owner)
  src/audio/*          generative ambience                  (audio owner)
bench/                 benchmark scripts + RESULTS.md (engine owners)
tools/                 headless checks: smoke, frame-exact shots, sheets
docs/                  this file, notes
```

## Machines

- **Primary:** Mac mini M4 Pro (16-core GPU, 16-core ANE), 48 GB, macOS 26, 4K
  display, Safari as the browser (the client must work in Safari).
- **Secondary:** MacBook Pro M2 Pro, 16 GB. Must run, may be slower.
- **Phone:** opens the page over the local network; renders locally, diffusion runs
  on the Mac. Touch controls.

Caches, venvs and Core ML builds go in `~/hypnagogia-cache/`.

## Server protocol (frozen)

HTTP on one port (default 8765, bind 127.0.0.1; `--host 0.0.0.0` for phones):
- `GET /` and static files → `client/`
- `GET /api/info` → `{engine, model, width, height, flexible, xframe, depth, depth_graft, lora, fps_estimate, device, ...}`
  (`depth`: an engine paints with the depth a frame carries; send it. `depth_graft`: its weight.
  `lora`: the merged LoRA as run/file@sha256 prefix, plus " x<scale>" when not 1, or null.
  `held_only`: "" (off), "both", "xframe" or "deepcache", see `kf` below)
- `GET /ws` → WebSocket

WebSocket messages:

Client → server **binary** frame:
`[uint32 LE headerLen][header: UTF-8 JSON][payload: JPEG bytes]`
header = `{"type":"frame","id":int,"width":int,"height":int,
"prompt":str,"negative":str|null,"strength":float,"seed":int,"format":"jpeg",
"depth":{"w":int,"h":int,"data":base64}}` (`depth` optional: w x h bytes, rows top first,
at the latent size (image / 8), byte b = relative inverse depth b / 127.5 - 1, near = 1; the
client scales each frame so its 2nd..98th percentiles span the range, with a span of at least
15% of the median so a flat view isn't stretched; 1..256 per side). `"cut":true` (optional):
the prompt changed where nobody sees it, so an engine that glides between prompts
(`torch_turbo`, `coreml_turbo`: `takes_cut`) switches at once. `"fid":int` (optional; the client sends it in every look while `?kfdist` > 0, the default): the capture's
framing id; captures on one seed with equal `fid` show the same framing. A pool of engines
(with `--pool-held carry`, the default) gives a frame that repeats its stream's last framing only to engines with
`takes_held`, so a view held still is painted by one engine. `"kf":int` (optional; the fresh
look sends it, equal to `fid`; a frame with `kf` and no `fid` is routed by `kf`): the
capture's framing id for the engine. The server's GPU worker tells
an engine with `takes_held` whether a frame repeats the last framing it painted for that
client and seed (`held`); `torch_turbo`'s `held_only` (default "deepcache") then skips
DeepCache reuse on a new framing, and "both" skips cross-frame attention too)

Server → client **binary** result, same framing:
header = `{"type":"result","id":int,"width":int,"height":int,
"ms_infer":float,"ms_total":float}` + JPEG payload.

Server → client **text** JSON:
- on connect: `{"type":"info", ...same as /api/info}`
- `{"type":"dropped","id":int}` when a queued frame was superseded (so the
  client can release that capture's depth snapshot)
- every ~1 s: `{"type":"stats","fps":float,"ms_infer":float,"queue":int}`

Frame size: `width`/`height` of the engine are its preferred capture size. When `flexible`
is true (every serving engine takes any size: `torch_turbo`, `mock`), a frame whose
header size is a multiple of 64 on each side, within 1.25x the engine's pixel count and
2x its longer side, is processed at that size; the client picks a shape matching its
screen (`client/src/dream/dream.js` `_fitCapture`; for a 384x384 engine: 512x320 on
16:9, 448x320 on 4:3, 320x512 or 256x512 on an upright phone). Other frames are resized
to the engine's size. Results always come back at the header's size. Header and JPEG
sides must lie in 8..2048, or the frame is refused (`error` + `dropped`).

Scheduling: one GPU worker. Per connection, **latest-wins**: a new frame
replaces that connection's not-yet-started frame (the replaced one gets
`dropped`). Multiple connections are served round-robin. Client keeps ≤ 2
frames in flight so network/encode overlaps compute.

`strength` ∈ [0,1] is "how much the model may change the image"; the engine maps
it to its own timestep/steps. `seed` should produce *identical noise* for
identical (seed, size) — engines cache the noise tensor; fixed noise across
frames is a major temporal-coherence win. The client adds 1 to a zone's seed for narrow
(foveated) captures, so each width has its own noise, and on engines that carry state
between frames (`torch_turbo`'s cross-frame attention, keyed by prompt + seed) its own
anchor: each width keeps following its own last picture.

## Engine interface (frozen)

```python
# server/engines/<name>.py
def create_engine(**cfg) -> "Engine": ...

class Engine:
    name: str            # e.g. "torch-sdturbo"
    model: str           # HF id or description
    width: int           # preferred capture size (multiple of 64 or 8 as needed)
    height: int
    device: str
    flexible: bool       # optional, default False: process() takes any multiple-of-64
                         # size within the budget above (see "Frame size")
    def warmup(self) -> None: ...
    def process(self, image, prompt: str, strength: float, seed: int,
                negative: str | None = None):
        """image: np.uint8 (H, W, 3) RGB -> np.uint8 (H, W, 3) RGB, same size.
        Called from a single worker thread. Must cache prompt embeddings."""
    wants_depth: bool    # optional, default False: process() also takes depth=
                         # (np.float32 (h, w) in [-1, 1], the header's `depth`, or None)
    takes_cut: bool      # optional: process() also takes cut= (the header's `cut`)
    takes_held: bool     # optional: process() also takes held= (bool: the frame repeats the last
                         # framing this engine painted on its client and seed; from header `kf`)
```

`torch_turbo`'s depth graft (`depth_graft`, 0.8; `--engine-arg depth_graft=0` for stock; `coreml_turbo`
reads the same argument and loads a grafted `_d08` build only when it is 0.8)
makes the UNet SD-Turbo + 0.8 x (SD2-depth - SD2-base), with SD2-depth's fifth `conv_in`
channel taking the frame's depth; nothing is trained, it costs ~0 ms a frame, and the
first load downloads two 1.7 GB fp16 UNets into the Hugging Face cache. A frame without
depth gets flat depth, which paints muddier than stock (`bench/depth_graft.py`).
`--engine-arg lora=PATH` (with `lora_scale`, default 1) merges a LoRA trained by
`bench/one_pass.py` into the UNet at load, at no cost per frame. M116's, trained to pull one
walking pass toward the view at rest, came 3% closer on unseen views and moved the rest
picture about 3 times as far as a plain pass: off, and not in git.

Engines may carry state between frames as long as each frame stays a valid result on its
own: `torch_turbo` reuses deep UNet features (DeepCache) and, by default, lets
self-attention also attend to the previous frame's keys and values (`xframe`), which cuts
the re-invention of objects between dreams. DeepCache keeps its state per seed and size,
cross-frame attention per prompt, seed and strength band (the client gives each kind of
capture its own seed), so interleaved capture kinds don't clear each other's. Both help a
held framing converge; on a new framing they carry the last view's features into the new
one: the largest part of a walking frame's gap to the same view at rest that a single pass can
remove (M116: LPIPS 0.34 to 0.23 with both off on new framings; two rest views of one framing
already differ by 0.18). For a
client that sends framing ids as `kf` (the fresh look), `held_only` (default "deepcache") gives each
new framing a full pass; cross-frame attention stays, since it holds walking flicker down.

`--engine auto` picks the fastest engine that imports and loads on this machine;
`--engine mock` is a CPU-only stand-in (posterize / hue shift / blur + fake
latency) for client development.

## Client contracts

### Level (world owner → render owner)

`client/src/world/level.js` exports `generateLevel(seed:number) -> Level`:

```js
Level = {
  seed,
  atlas: { size: 4096, texelsPerMeter: number },   // size may be 2048 on mobile via generateLevel(seed, {atlasSize})
  geometry: THREE.BufferGeometry, // ONE merged static geometry (or `geometries: [...]` if >65k verts matters - use Uint32 indices, prefer one)
     // attributes:
     //  position  vec3 (meters, y up)
     //  normal    vec3
     //  uv        vec2  world-scale coords in meters for procedural base patterns
     //  atlasUv   vec2  unique non-overlapping chart coords in [0,1] of the dream atlas;
     //                  charts padded by >= 4 texels at the given size; texel density ≈ texelsPerMeter
     //  surface   float index into surfaceTypes
     //  zone      float index into zones
  surfaceTypes: [{ name, color:[r,g,b] (0..1), pattern:'stone'|'tile'|'brick'|'metal'|'wood'|'crystal'|'plaster'|'water'|'sky'|'glow', emissive: 0..1 }],
  zones: [{ id, name, subtitle, prompt, negative, fog:[r,g,b], fogDensity,
            light:[r,g,b], ambient:[r,g,b], sky:[r,g,b], mood: 'string for audio' }],
  zoneAt(position /*THREE.Vector3*/) -> zoneIndex,
  lights: [{ position:[x,y,z], color:[r,g,b], intensity, radius, flicker:bool }],
  spawn: { position:[x,y,z], yaw },
  collision: opaque object consumed by Player,
  decor: THREE.Object3D | null   // optional NON-painted animated things (motes, floating shards...)
}
```

The sky/void is real geometry (e.g. a large inward-facing dome / far planes)
with `pattern:'sky'` and atlas charts, so the model paints the sky too.

`decor.userData.update(timeSeconds, drawingBufferHeightPx)` is called once per frame.

`client/src/world/player.js` exports `class Player { constructor(level, camera, domElement); update(dt); get position(); get yaw(); get pitch(); setPose(pos, yaw, pitch); look(yaw, pitch); onStep /* optional (intensity 0..1) callback: footfalls, hard landings = 1 */ }`
— first-person controller with gravity, jumping, collision, step-up, desktop
(pointer-lock mouse + WASD/arrows, Shift run, Space jump) and touch (left thumb
virtual stick, right side drag to look). Camera eye height ≈ 1.6 m.

### Audio (audio owner → render owner)

`client/src/audio/ambience.js` exports `class Ambience { constructor(); async start() /* call synchronously inside the first user gesture (iOS) */; setZone(zone /* Level zone object */); setDream(activity /*0..1*/); setVolume(v); toggle(); footstep(intensity /*0..1*/) }`.

### Debug URL params (all owners honor what applies)

`?seed=N` level seed · `?pos=x,y,z&yaw=r&pitch=r` start pose · `?dream=off|mock|server` (default server; `mock` = in-browser fake diffusion, no server needed) · `?server=ws://host:port/ws` · `?atlas=2048` · `?autopilot=1` scripted camera tour (`&tour=N` picks a waypoint) · `?title=0` skip the title screen · `?perf=1` frame-time logging · `?level=test` test level · `?hud=0` · `?live=N` live views (5 desktop, 3 phone, max 5, 0 = atlas only) · `?livetau=0.35` a held framing's picture follows its newest result with this time constant, s (0 = each result replaces) · `?livefade=0.25` seconds a result takes to fade in · `?sharpen=0.6` live-pixel sharpening · `?warp=1` foveal warp (>1 on, at most 2) · `?rate=20` captures per second at most · `?fovea=1` narrow captures' density gain (>1 brings back the lens) · `?strength=0.66` · `?sep=0` `?relight=0` display depth cues (>0 on) · `?capsep=0` `?edgekeep=0` the depth cue in what the model sees, and less feedback at silhouettes · `?look=waking|drifting|lucid|fever|lens|fresh` start in a look (keys 1-6 and the panel switch looks; a URL setting beats a look) · `?inflight=2` captures sent ahead of their results (3 feeds a GPU + Neural Engine pool) · `?plainwalk=1` also send framing ids as header `kf`, so each new framing gets a plain pass on an engine with `held_only` (the fresh look; header `fid` is sent in every look) · `?eyestrength=0.9` `?eyefeedback=0.1` how hard a room is re-dreamt behind closed eyes (hold E; the strength only on a depth engine) · `?feedback=0.55` feedback standing still (the panel's slider) · `?fbmove=1.3` walking multiplies it (capped at 0.95) · `?fbanchor` `?fbsat` feedback leak · `?calm=0.25` strength drop while moving (default 0 = off) · `?capfov=N` capture vertical field of view (0 = fitted to the screen) · `?persist=0` start from a blank dream and don't store it · `?idle=90` seconds without input before the view starts drifting (0 = never).

Headless checks use Playwright (Chromium with `--use-angle=metal` for WebGL on Metal, and
WebKit for Safari's engine): `tools/smoke.mjs` (boot, dream, sound, walking, touch on an
emulated iPhone, reload memory), `tools/shoot.mjs` (frame-exact shots on a fixed walk per
zone; `--flicker` temporal stability, raw and "warped" (the previous frame reprojected
with depth first, so most detail that only moved cancels; reported beside a detail figure,
since softer paint also changes less; plus the luma step across depth silhouettes and
within surfaces, and the dream rate), `--strip` walking filmstrips,
`--perf` frame time), `tools/harvest_pairs.mjs` (a walking frame and the same view after
standing still, scored by `bench/walk_gap.py`) and `tools/sheet.py` (side-by-side sheets).
Point `PLAYWRIGHT=` at Playwright's `index.mjs` and pass `--url`; every tool names that
server in the page (`?server=`), since the client's own fallback is port 8765.
A headless browser on a Mac without a display session throttles requestAnimationFrame
to about 5 Hz, so the tools pace pages with a 60 Hz setTimeout instead.

## Visual direction

Taste over noise. Never harsh on the eyes: slow, breathing, painterly, luminous.
- The raw ("undreamt") world: dark, clean, architectural — soft
  graybox-with-light, faint glowing edge/contour lines, so it reads as a
  blueprint of a place not yet imagined. Paint washes in over it.
- The raw render sent to the model must have strong, readable structure and
  lighting (the model follows it): good contrast, colored light pools, fog for
  depth, distinct materials per surface type.
- Dreamt world: rich color, soft bloom, mild film grain and vignette, gentle
  chromatic breathing at the edges — never frantic.
- Zones (6–8) each with a strong identity + prompt: e.g. a drowned cathedral
  nave, a vertical library of endless stairs, a sunken garden under a false sky,
  a brutalist atrium of floating monoliths, a crystal cavern, a neon bathhouse,
  a desert of arches under two moons... Transitions announce the zone name with
  a slow elegant title card.
- Architecture should be *crazy*: multi-level, tall volumes, bridges over voids,
  spiral stairs, colonnades, ramps, floating platforms, windows onto impossible
  vistas, rooms at different scales (tiny corridors opening into vast halls).
