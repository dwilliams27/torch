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
5. The final frame renders every surface with its dream-atlas paint (falling back
   to a raw "undreamt" look where nothing is painted yet).

Consequences: the game always renders at display rate; paint is spatially
anchored so it doesn't swim when you turn; each new diffusion result only nudges
the surfaces it saw (EMA), so the world *breathes and evolves* instead of
strobing; unseen areas are visibly "not yet dreamt" and wash in as you look;
walking back through a zone shows what the machine dreamt there. Diffusion FPS
only controls how fast the paint evolves, not how smooth the game feels.

Feedback: the image sent to the server is the *already painted* view (mixed with
the raw render by a `feedback` factor), so the model refines its own prior
dream → strong temporal coherence, with slow artistic drift.

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
  src/dream/*          capture, link (ws), painter          (render owner)
  src/render/*         materials, post-fx, hud              (render owner)
  src/audio/*          generative ambience                  (audio owner)
bench/                 benchmark scripts + RESULTS.md (engine owners)
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
- `GET /api/info` → `{engine, model, width, height, fps_estimate, device}`
- `GET /ws` → WebSocket

WebSocket messages:

Client → server **binary** frame:
`[uint32 LE headerLen][header: UTF-8 JSON][payload: JPEG bytes]`
header = `{"type":"frame","id":int,"width":int,"height":int,
"prompt":str,"negative":str|null,"strength":float,"seed":int,"format":"jpeg"}`

Server → client **binary** result, same framing:
header = `{"type":"result","id":int,"width":int,"height":int,
"ms_infer":float,"ms_total":float}` + JPEG payload.

Server → client **text** JSON:
- on connect: `{"type":"info", ...same as /api/info}`
- `{"type":"dropped","id":int}` when a queued frame was superseded (so the
  client can release that capture's depth snapshot)
- every ~1 s: `{"type":"stats","fps":float,"ms_infer":float,"queue":int}`

Scheduling: one GPU worker. Per connection, **latest-wins**: a new frame
replaces that connection's not-yet-started frame (the replaced one gets
`dropped`). Multiple connections are served round-robin. Client keeps ≤ 2
frames in flight so network/encode overlaps compute.

`strength` ∈ [0,1] is "how much the model may change the image"; the engine maps
it to its own timestep/steps. `seed` should produce *identical noise* for
identical (seed, size) — engines cache the noise tensor; fixed noise across
frames is a major temporal-coherence win.

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
    def warmup(self) -> None: ...
    def process(self, image, prompt: str, strength: float, seed: int,
                negative: str | None = None):
        """image: np.uint8 (H, W, 3) RGB -> np.uint8 (H, W, 3) RGB, same size.
        Called from a single worker thread. Must cache prompt embeddings."""
```

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

`client/src/world/player.js` exports `class Player { constructor(level, camera, domElement); update(dt); get position(); get yaw(); get pitch(); setPose(pos, yaw, pitch) }`
— first-person controller with gravity, jumping, collision, step-up, desktop
(pointer-lock mouse + WASD/arrows, Shift run, Space jump) and touch (left thumb
virtual stick, right side drag to look). Camera eye height ≈ 1.6 m.

### Audio (audio owner → render owner)

`client/src/audio/ambience.js` exports `class Ambience { constructor(); async start() /* call on first user gesture */; setZone(zone /* Level zone object */); setDream(activity /*0..1*/); setVolume(v); toggle() }`.

### Debug URL params (all owners honor what applies)

`?seed=N` level seed · `?pos=x,y,z&yaw=r&pitch=r` start pose · `?dream=off|mock|server` (default server; `mock` = in-browser fake diffusion, no server needed) · `?server=ws://host:port/ws` · `?atlas=2048` · `?autopilot=1` scripted camera tour (`&tour=N` picks a waypoint) · `?title=0` skip the title screen · `?perf=1` frame-time logging · `?level=test` test level · `?hud=0`.

Headless screenshots (WebGL via Metal) with any Chromium, e.g. Playwright's:

```bash
export PLAYWRIGHT_BROWSERS_PATH=~/hypnagogia-cache/ms-playwright
npx -y playwright@1.49.1 install chromium webkit      # once
CHROME="$(find "$PLAYWRIGHT_BROWSERS_PATH" -path '*MacOS/Chromium' | head -1)"
"$CHROME" --headless=new --use-angle=metal --window-size=1280,720 --virtual-time-budget=8000 \
  --screenshot=out.png "http://127.0.0.1:PORT/?dream=mock&autopilot=1"
```

Serve `client/` with the real server, or with `python3 -m http.server` for `?dream=mock`.
WebKit smoke tests use Playwright's WebKit build.

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
