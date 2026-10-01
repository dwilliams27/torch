# Lens: PLAY AND WORLD (what the dream could do) · Claude Opus 5.5 (claude-opus-5-5) · 2026-09-30

These are sixteen ideas for what a world dreamt live by SD-Turbo could *do*. **[BET]** marks my
three bets and **[CRAZY]** marks the ideas that would take weeks if they pan out. Paths are relative
to `projects/hypnagogia/`. Numbers come from `docs/STATUS.md` and committed data unless marked
*guess*.

Several of these ideas share one thread. The owner's 2026-09-29 playtest found the dream
"disorienting and nauseating a bit" and asked for "different options I could try out". A
diffusion world can change a great deal while nothing is on screen: behind closed eyes, in a room
you left, or under the water. Change made there shows no visible boil, and it is what dreams do
anyway.

| # | Idea | In one line | Build | Per frame |
|---|---|---|---|---|
| 1 | **Close Your Eyes** [BET] | hold a key; the dream re-dreams the room around you while the screen is dark | 1-2 days | ~0 |
| 2 | **Semantic Paint** [BET] | the model's own word maps say where it painted lamps and water; those become light and sound | 3-5 days | small |
| 3 | **The Memory Gallery** [BET] | where you lingered becomes an oil painting hung in the Vestibule's empty frames | 1-2 days | ~0 |
| 4 | Painted Doors [CRAZY] | an opening the dream keeps painting on a wall becomes a way through | 1-3 weeks | tiny |
| 5 | Through the Water [CRAZY] | reflections show a second, separately dreamt world; stare long enough and you swap | 2-3 weeks | +1 half-res pass near water |
| 6 | Shared Dream [CRAZY] | everyone connected paints everyone's world; a phone can ride along as a second eye | 1-2 weeks | small |
| 7 | Leviathans [CRAZY] | big slow proxies whose identity the model decides, with paint that sticks as they move | 1-2 weeks | tiny |
| 8 | Associative Drift [CRAZY] | each room is dreamt with words captioned from the last thing you saw; endless rooms | days / 3+ weeks | none |
| 9 | Quantum Rooms | rooms change only while unobserved and keep every version | days to 1 week | ~0 |
| 10 | Dwell Blooms | standing still slowly deepens a place, and it stays deepened | 1-2 days | none |
| 11 | Desire Lines | where you walk, the dream paints paths | ~1 day | ~0 |
| 12 | Dream Bridges | some structures are intangible until you have looked long enough to dream them | 2-3 days | tiny |
| 13 | Fossils | the dream's recurring inventions harden into permanent features | 2-4 days | tiny |
| 14 | Dream Polaroids | keep a view and project it onto another place; later, as real geometry | days / weeks | 1 live slot |
| 15 | Sleepwalker | when idle, the dreamer walks toward what the dream painted; a morning slideshow | ~2 days | none |
| 16 | Hypnic Fall | stepping into the void is travel: a slow fall, a fade, another zone | hours | none |

**Shared building blocks** (build once):

- **Unseen captures:** captures from a camera that isn't the view, written only to the atlas, each
  with its own seed. Side glances already work this way. Used by 1, 5, 6 and 9. At rest the view
  gets about 9-12 dreams a second, but the `?rate=8` runs were calmer standing still (0.58 vs
  0.85 change; `docs/shots/m112/data/`). So unseen captures can spend the dreams that today only
  add boil.
- **Side channels from the server:** small per-dream maps sent next to the JPEG as a new message
  type, which the protocol allows. Used by 2, 4 and 13.
- **Fade-and-arrive links:** 4, 9c and 16.
- **Offscreen renders from the atlas:** 3 and 15.
- **Painted objects with their own charts and capture-time transforms:** 7, 9b and 14.

---

### 1. Close Your Eyes [BET]

**Idea.** You close your eyes by holding E. On a phone you hold two fingers on the look side, or
lay the phone face down. The screen sinks into a warm eyelid dark where the dream's newest
pictures glow faintly, blurred and averaged over about a second, like the patterns seen behind
closed eyes at sleep onset, and the sound goes muffled. Meanwhile the dream keeps working:
captures sweep all around you with the zone's next prompt variant at a little more strength,
painting only the atlas. When you open your eyes the architecture is the same but dreamt again (a
snowbound nave, the library at noon), and the live layer sharpens it within a second.

**Why only here.** Closing your eyes is the natural verb of hypnagogia (Before Your Eyes, 2021,
built a whole game on webcam-detected blinks). Most of the plumbing exists:

- `_capture` in `dream/dream.js` already aims atlas-only captures away from the view. Side glances
  have `slot.side != 0` and never enter the live layer.
- Each kind of capture has its own seed and stream.
- The server glides the prompt embedding when the prompt changes (`morph`, 1 s).

It also answers both playtest notes. The biggest changes happen while nothing is on screen, which
is Outer Wilds' rule for its quantum objects and gentler than any visible transition. And every
blink is a new option to try, with no menu.

**Cost.** 1-2 days, client only:

- An eyelid term in the post composite: a slow average of a 64x40 downsample of each new result,
  heavily blurred, about 5% brightness, shifted red.
- A sweep schedule in `dream.js`: 8 yaws x 2 pitches, atlas-only, its own seed, feedback about 0.3
  and strength about 0.7.
- 3-4 prompt variants per zone in `world/zones.js`.
- An `Ambience.setEyes()` lowpass, with `_arrival`'s chord reused on opening.

Per frame it costs about nothing, since captures keep their rate. It should be gentler on comfort
than today. The risk is opening onto soft atlas paint. To avoid it, make the last sweep capture
the forward view and hold the fade until that result lands (about 0.3 s).

**Smallest experiment (≤2 h).** No eyelid is needed yet. Use headless WebKit against a local
server, in the nave, stacks and garden:

1. Settle 10 s at a rest pose (`?pos=...&yaw=...`) and take a snapshot.
2. Through `window.__hyp`, set `settings.promptOverride` to a variant, strength 0.7 and feedback
   0.3. Turn the view with `player.look()` in 45° steps every 0.75 s for 6 s. Take shots only
   before and after.
3. Turn back and restore strength and feedback. Take snapshots at +0.25, +0.5, +1, +2 and +4 s.
4. Take 20 frames at 10 Hz to measure rest change.

It succeeds if the +1 s frame is a coherent, clearly different dream of the same room (a
`tools/sheet.py` sheet, or a blind "same place, new dream?" question) and rest change after +2 s
is at or below the closing build's 0.85.

**Kills it.** The swept atlas is a soft patchwork that re-boils visibly for seconds after the eyes
open. In that case a slow visible morph is better. It also fails if the variants drift so far that
the zones lose their identity.

### 2. Semantic Paint [BET]

**Idea.** Along with each dream, the server returns a small map for each chosen noun in the zone
prompt ("candlelight", "water", "moss", "books", "crystals", "neon", "moons"). Each map shows where
in the picture the model put that word, read from its own cross-attention. The client turns the
maps into facts about the world:

- Lamps the model painted cast real light, soft and slow to change.
- Painted water drips and laps from where it was painted.
- Footsteps splash on painted water and go soft on painted moss.

**Why only here.** The model computes this attention on every dream and then throws it away:

- DeepCache's cheap pass still runs `up_blocks[3]`. In SD-Turbo that block is cross-attention at
  full latent resolution (the cached UNet config says `CrossAttnUpBlock2D`, 320 channels, 5
  heads).
- `_FastAttn` sends that attention through fused SDPA. An explicit softmax over 5 heads x 2,560
  latent pixels (at 512x320) x 77 tokens is about 1M logits per layer, over three layers. My
  *guess* is well under a millisecond.
- Each map is 2.5 KB.

DAAM (Tang et al., ACL 2023) showed that per-word cross-attention maps localize nouns in Stable
Diffusion. It pooled every layer and every step, though, so whether one layer at one step is
enough is the open question. The capture's own depth turns the peaks of a map into world
positions, so a first version needs no new atlas, only a short list along the lines of "a candle
was painted here". This is charter item 5 ("ambience that responds to what the model paints")
taken literally: a lamp that exists only because the model imagined it, lighting the room.

**Cost.** Half a day on the server: attention processors that know their layer name, token ids
from the tokenizer's offsets, and a new `maps` message. Two to four days on the client:

- Map peaks become world points, from a few depth samples of the capture.
- Dream lights go into `LightDriver`, fading over at least 2 s.
- Spatial sound emitters use HRTF `PannerNode`s driving `audio/voices.js`.
- Footsteps get variants.

The per-frame cost is small. `MAX_LIGHTS` is 8, so dream lights either compete with the authored
ones or the cap rises to about 12. The comfort risk is flicker: commit a light only after the maps
agree over several dreams, cap them at about 4, and fade over seconds. A painted lamp also lights
the capture, so the model keeps painting it. That is either lovely or a plague of lamps, so cap
it.

A first step needs no server change. Let the newest result's mean hue, brightness and detail steer
the pad and chord colour, instead of `setDream`'s dream rate alone.

**Smallest experiment (≤2 h).** Server only:

1. Save one capture input per zone at rest (`__hyp.dream.lastJpeg`).
2. Behind an env flag, make `_FastAttn` keep softmax(QK^T) for chosen token ids in the `attn2`
   layers of `up_blocks.3.attentions.*`.
3. Run the 8 inputs at 512x320 with their zone prompts through a ten-line `create_engine()`
   script (the module's `__main__` shows how).
4. Write heat overlays for 3-4 nouns per zone, and time the extra work.

It succeeds if, in at least 5 of 8 zones, at least two nouns peak on the matching painted thing.
The overhead must also stay under 2 ms, and a DeepCache cheap pass must give the same maps as a
full pass (correlation above 0.9).

**Kills it.** The maps are diffuse, or they jump around from frame to frame. My *guess* is that a
one-step model anchored this strongly to its input image may barely consult the text per pixel.
The fallback is colour heuristics (warm, bright blobs are lamps), which is dumber but free and
keeps most of the magic.

### 3. The Memory Gallery [BET]

**Idea.** Wherever you stood still longest, the dream makes a painting. The game remembers where
you lingered, keeping the best pose per zone. It renders each of those views from the dream atlas
and has the model repaint it as an oil painting. The nine blank canvases floating in the
Vestibule's void show them. Each time you come back to the start of the dream, or reload the next
day, you walk past a gallery of your own dream, painted by the dream.

**Why only here.** The Vestibule ("where the waking world thins") already hangs nine gilded
frames whose `canvas` faces look toward the causeway (`buildVestibule`, 2.8-6.5 m wide). Each
canvas is a single atlas chart: charts carry `surface` and `zone`, and `flatten()` records
`chartOf`. The repaint is the usual img2img call with a painting prompt at about 0.45 strength,
one dream per memory, off the critical path. This is memory built out of the dream itself, and it
accumulates: the charter's "a dream that remembers where you have been".

**Cost.** 1-2 days. The atlas is too coarse for pictures: at about 13 texels/m, a 4 m canvas gets
roughly 50 texels across. So the paintings live in their own texture instead:

- A 2048x1024 texture holds 12 slots of 512x320.
- The display and capture shaders sample it on `canvas` fragments. Each fragment's painting is
  the one whose centre is nearest, from nine centre uniforms.
- The JPEGs are kept in IndexedDB beside the atlas.
- Paint jobs and live views skip the canvases.
- Dwell detection reuses `dream.motion`.

Per frame it costs about nothing. On legibility, a 4 m canvas 10 m away is about 300 px wide at
1080p, about a sixth of the screen at the game's 72° vertical field of view.

**Smallest experiment (≤2 h).**

1. Snapshot three tour frames (nave, geode, garden) with `snapAt` and downscale them to 512x320.
2. Repaint each with a short engine script at strength 0.45, using the prompt "a baroque oil
   painting of <the zone's first clause>, gilded frame, museum light, craquelure".
3. In a Vestibule session, patch the display shader through `__hyp.materials` so the three
   nearest canvases sample the paintings. Screenshot two poses on the causeway.

It succeeds if a blind reviewer names the place in at least 2 of the 3 paintings, and they read as
paintings in frames rather than mush.

**Kills it.** The lingering poses turn out to be dull (waiting in corridors); if so, choose poses by
the painted detail in view instead. It also fails if the paintings clash with the Vestibule's
palette.

### 4. Painted Doors [CRAZY]

**Idea.** Sometimes the dream keeps painting an opening where the geometry is solid: an arch, a lit
doorway, a window onto elsewhere. When that happens, the world honours it. Walk into it, the screen
fades to its colour, and you arrive somewhere else in the dream. LSD: Dream Emulator (1998) "links"
you to another dream whenever you walk into a wall, fading through a single colour. Destinations
are other painted doors, or the place you lingered longest in another zone. The version that would
take weeks grows a real room behind a door the dream keeps painting.

**Why only here.** Paint is anchored to real geometry with a depth buffer, so an opening the model
invents has a size and a position in metres. Compare the depth the model *painted* with the
capture's true depth.

- **Depth estimate:** Depth Anything V2 Small (24.8M parameters) runs in 32.8 ms under Core ML on
  an M1 Max, mostly on the Neural Engine, per Apple's model card. The Neural Engine sits idle
  while `torch_turbo` dreams on the GPU. My *guess* is that it could estimate depth for every
  other dream at almost no GPU cost.
- **Fit:** match its relative depth to the true depth, per image.
- **Accumulate:** project "painted deeper than the wall" into a low-res openness atlas, the same
  way colour is projected.
- **Decide:** call a region a door when it stays open over many dreams and is door-sized: at least
  1.8 m tall, 0.8-4 m wide, with its bottom on a walkable floor.

Semantic Paint's maps for "arch", "doorway" and "window" would be a cheaper detector. The same
depth signal gives painted relief for free: parallax wherever the model painted depth.

**Cost.** The link version takes about a week:

- a depth service on the Neural Engine, in the server;
- the openness atlas;
- the door detector;
- fade-and-arrive, plus rules for where each door leads.

Real rooms add 1-2 weeks. The annex rooms are prebuilt with reserved atlas charts and moved into
place. The wall is cut by a discard box in the three level shaders, plus a split collision box.
The per-frame cost is tiny. For comfort, a colour fade is the standard gentle teleport; trigger it
only after about 0.6 s of pushing into the wall.

**Smallest experiment (≤2 h).** Offline:

1. `snapAt(t, 'tour', {depth: [160, 90]})` gives the displayed dream plus linear depth. Take rest
   and walk frames in 8 zones.
2. Run Depth Anything V2 Small on those frames (PyTorch on MPS is fine for a test; about 100 MB of
   weights).
3. Fit scale and shift in disparity against the true depth.
4. Save overlays of regions at least 1.5x deeper than the geometry and at least 1 m² in area.

It succeeds if each zone has a few door-shaped regions, recurring across frames, that a person
would call a painted opening.

**Kills it.** The disagreement turns up only in fog, sky and glare. Or the work that keeps paint
on the geometry (feedback, cross-frame attention, the depth cues planned for the capture) leaves
almost no invented openings. Seeding doors on purpose would kill the discovery.

### 5. Through the Water [CRAZY]

**Idea.** Water gets real reflections, but the world in them is dreamt separately, with an
"under" prompt. The nave's floodwater reflects the same cathedral at night, lit by
bioluminescence; the baths reflect a snowbound bathhouse. Look down into the water long enough and
the dream flips: you stand in the reflected world, and the water shows the one you came from. It
is A Link to the Past's (1991) Light World and Dark World, with the Dark World dreamt.

**Why only here.** A reflection is just one more camera, and both the capture and the paint pass
work from any camera:

- **Mirror camera:** it renders the room from below the water, with its position reflected in the
  water plane, pitch negated, and anything under the plane discarded.
- **Under-atlas:** the model paints that view with the under-prompt, and the results go into a
  second atlas (2048² RGBA8, 16 MB) through that camera's `viewProj`.
- **Display:** water fragments sample a half-res reflection render that reads the under-atlas,
  rippled by the water pattern.
- **The flip:** swap the two atlases and prompts under a slow fade.

You can already stand in the nave's water, since the aisles have an invisible floor 0.25 m below
the surface (`NAVE.YW`). One thing to check along the way: with a reflection in the capture, the
model may paint water *as water*. Today the nave's floodwater comes out as green moss.

**Cost.** 2-3 weeks:

- a planar reflection for the dominant water plane in view;
- a painter and memory for the under-atlas;
- scheduling, at about 1 capture in 4 while you rest with water in view;
- the flip itself, plus under-variants of the ambience.

Per frame it adds one half-res scene render while water is visible. My *guess* is 0.3-0.8 ms on a
desktop; phones use the atlas alone or skip it.

**Smallest experiment (≤2 h, tight).**

1. At a nave rest pose, take the mirrored pose: y' = 2·(-7.45) - y, pitch' = -pitch.
2. Patch the capture shader to hide the water quad, and dream from the mirrored pose with an
   under-prompt.
3. Flip the result vertically and composite it, with a ripple, into the water pixels of the
   real-pose frame. The water mask comes from `snapAt` depth: pixels whose world y is about -7.45.

It succeeds if the composite reads as the reflection of another world and is beautiful.

**Kills it.** It fails if reflected views are all vault and sky, so the under-world comes out
generic, or if the capture tax visibly slows the main dream. It is also dead as a play idea if the
reflection alone turns out to be the whole win; that is a rendering feature.

### 6. Shared Dream [CRAZY]

**Idea.** Everyone connected to the server dreams into everyone else's world. You never see an
avatar. You notice the other dreamer in three ways: a soft light where they stand, their
footsteps in the space, and the world washing into existence wherever they look. A phone can join
as a second eye riding along with the desktop player, using the desktop's position and the
phone's gyro: hold it up to look behind you, and what it sees gets dreamt.

**Why only here.** The server already serves several connections round-robin. Clients with the
same prompt and seed already share a cross-frame attention anchor, so dreamers in the same zone
already subtly steer each other's pictures today. The work is three steps:

1. Add a `pose` field to frame headers.
2. Relay each result with its pose as a new `peer_result` message.
3. The receiving client renders capture depth from that camera into a free slot (one depth-only
   draw) and runs an ordinary paint job.

The GPU splits dreams between clients, but every client paints all of them, so nobody's world
fills more slowly. There is prior art. Journey (2012) gives strangers anonymous co-presence with
no names and no chat. Dark Souls (2011) leaves traces of other players.

**Cost.** 3-5 days for peers: the relay, peer painting, a presence light in the decor layer, and
spatial footsteps. The phone eye adds about 3 days (DeviceOrientation, where iOS asks permission
once). Per frame, paint jobs double, but they are cheap and spread over frames. My bandwidth
*guess* is about 0.3 MB/s per peer. No new service is needed: it rides the game's own server and
protocol.

**Smallest experiment (≤2 h).** No server change is needed:

1. Open two headless pages in one browser context and relay results between them over a
   `BroadcastChannel` (the JPEG, `viewProj` and position).
2. Page B renders depth from A's camera and pushes a paint job.
3. Hold two nave poses for 30 s. Measure B's atlas coverage (mean confidence from a small mip
   readback) with and without A, and look at B's view of what only A saw.

It succeeds if A's region appears in B's world without misprojection or seams.

**Kills it.** Nobody plays with two devices or two people, so it stays a demo. It also fails if
pose or latency skew misplaces the peer paint, or if the two streams leave a patchwork of styles.

### 7. Leviathans [CRAZY]

**Idea.** A few big, slow things drift through the largest volumes: a whale-sized shape under the
nave's vault, paper birds circling the stacks' oculus, koi in the baths, a caravan on the desert
horizon, a monolith turning in the atrium. They are plain proxy meshes, and what they *are* is up
to the dream: the nave's whale might come out as a ship, a cloud or a sleeping saint. Their paint
sticks to them as they move.

**Why only here.** This is charter item 5, painted things that move:

- Movers get their own atlas charts and render on layer 0, so the model sees them. Decor stays on
  layer 1 and is never painted.
- Each capture slot stores the movers' transforms. The paint pass renders each mover as it stood
  at capture time, so the projection stays exact after it has moved on.
- Live views skip movers, since a projection about 200 ms old would slide off them. For slow,
  distant things, atlas paint is enough.

The nave's audio mood already says "whale-like swells", so the whale can get a spatial voice.
Prior art: Abzû (2016), where whales and schools of fish keep you ambient company.

**Cost.** 1-2 weeks: a mover API in the level contract, per-slot transforms, painter and memory
support, prompts that name the movers, and audio. Per frame it is a few meshes. The comfort risk
is vection, the illusion of self-motion caused by big shapes moving in the periphery, and a classic
trigger of motion sickness. Keep movers slow (at most 0.5 m/s), far away, and under about 15% of
the view; those thresholds are *guesses* to test.

**Smallest experiment (≤2 h).** Start static:

1. Through `__hyp`, add one whale-like proxy on layer 0: a scaled ellipsoid with two fin slabs.
2. Append "a pale whale drifting under the vault" to the nave prompt.
3. Save the model's own results (`dream.lastResultBytes`) from three rest poses.

It succeeds if something alive appears in at least 2 of the 3, with no extra flicker. Only then
move it 2 m between captures and test that the paint follows the object.

**Kills it.** SD-Turbo paints the proxy as architecture, a stone lump. It also fails if the proxy
needs a detailed sculpt to read as a creature, because then the dream is no longer deciding.

### 8. Associative Drift [CRAZY]

**Idea.** Rooms follow from each other the way scenes in a dream do. When you cross into a new
zone, the game captions what you last looked at in the old one and folds its nouns into the new
zone's prompt. The library is dreamt "with the moss and pale light of the cathedral" you just
left, and over a night the zones drift into each other. The version that would take weeks is an
endless chain: new zones assembled from the level generator's parts, each prompt born from the
previous room's caption.

**Why only here.** Prompts are data (`world/zones.js`), and the server glides between embeddings
when a prompt changes. The zone's title card could whisper the borrowed words as its subtitle. The
captioner is a small local model, for example Florence-2-base (about 0.23B parameters). My *guess*
is 100-300 ms per caption on MPS, once per zone change. Prior art: the Surrealists' exquisite
corpse, and Yume Nikki's (2004) Nexus of doors.

**Cost.** The drift takes 3-5 days. Generated zones take 3 weeks or more: new generators, atlas
packing per zone, memory keys and audio presets. There is no per-frame cost.

**Smallest experiment (≤2 h).**

1. Caption 8 zone frames, for example the tiles of `docs/shots/m113/final-strips.jpg`.
2. Build merged prompts for 3 pairs of zones.
3. Dream the next zone's saved capture input with the plain prompt and with the merged prompt, and
   put the results side by side on a sheet.

It succeeds if the captions are specific ("green moss over flooded stone, pale shafts of light")
and the merged dreams carry the last room's motifs while keeping the zone's own architecture.

**Kills it.** Captions of the dream come out generic ("a painting of a room"). Or CLIP's 77-token
limit squeezes out the zone's geometry nouns, which brings ghost geometry back (known issue 3 in
STATUS).

### 9. Quantum Rooms

**Idea.** Rooms change only while nobody is looking, and the dream keeps every version. There are
three forms:

- **(a) Remote dreaming.** While you are in the geode, some dreams are spent inside the stacks you
  just left: captures at the tour's stacks poses, with a variant prompt, written only to the atlas.
  You come back to a library that was dreamt anew while you were away.
- **(b) Rooms with several memories.** A room owns 2-3 layer-masked copies of its surfaces, each
  with its own charts. You see one copy per visit, and each keeps its own dream.
- **(c) The endless stair.** Climb past the top of the stacks and you are seamlessly back at the
  bottom, one variant further on. P.T. (2014) did this with a corridor that changes on every lap.

**Why only here.** A world model such as Oasis (2024) can't keep a world at all: turn around and it
has changed. This game has real geometry and a persistent atlas, so it can do impermanence on
purpose and by rules, the way Outer Wilds' quantum objects move only when unobserved. The capture
frusta and the view say exactly what is being observed. Remote captures are ordinary captures from
other cameras, and they spend the dreams the resting view doesn't need. Layer copies need a
per-vertex attribute, a discard mask in the three level shaders, and a layer tag on collision
prims, as in Antichamber's impossible spaces.

**Cost.** (a) takes 1-2 days, (b) 3-5 days, and (c) about a week (a repeating stair segment and a
seamless teleport). Per frame it costs about nothing. For comfort, nothing is ever seen changing.

**Smallest experiment (≤2 h).** This tests (a), on the tour's stacks → geode → stacks leg. While
the player is in the geode, aim every third capture at the tour's stacks poses with a variant
prompt (patch `_capture` through `__hyp`). Take snapshots of the return at +0, +1 and +3 s. The
control run switches the prompt at the same moment with no remote captures. It succeeds if the
remote run shows the new library from the first frame back while the control shows a 1-3 s morph.

**Kills it.** The variants are either too close to notice or so far apart that the world stops
feeling like a place. Both are tunable.

### 10. Dwell Blooms

**Idea.** Standing still and looking slowly pulls a place deeper into its zone's own logic. The
garden blooms, the stacks' shelves multiply and lean in, the geode's crystals grow, and the nave's
water rises with fish passing. The drift takes 30-90 s, driven by how long you dwell rather than by
a timer, and it pauses when you walk away. The atlas keeps whatever bloomed, so the places where
you lingered become the strangest places in your world.

**Why only here.** teamLab's "Flowers and People, Cannot be Controlled but Live Together" (2017):
more flowers are born while people stay still, and moving scatters them. Here the flowers are a
prompt. The engine already interpolates between embeddings when the prompt changes
(`_conditioning`). A `mix` field in the frame header would set lerp(prompt, deep prompt, mix).
Strength stays fixed, so the meaning drifts without adding change per dream. That claim is exactly
what needs testing, since the work on calm at rest is already fighting boil.

**Cost.** Half a day on the server. A day on the client, to track dwell per 2 m cell and yaw sector
and persist it. No per-frame cost.

**Smallest experiment (≤2 h).** Run the server with `--engine-arg morph=40`. Hold a nave rest pose
for 60 s, switching to the deep variant at 5 s. Measure rest change the way `--flicker` does, and
take a 10-frame filmstrip over the minute. It succeeds if rest change stays within 10% of a plain
hold and the transformation is visible and smooth.

**Kills it.** At strength 0.6 with feedback, the input dominates and the drift barely shows.
Raising the strength enough to show it brings the boil back.

### 11. Desire Lines

**Idea.** Where you walk, the floor remembers. Each footfall adds a soft mark to a wear map, and
the capture shows the model a slightly warmer, smoother floor there. The dream paints whatever a
worn path means in that zone: a carpet runner in the stacks, trodden moss in the nave, a sand
track in the desert, stepping stones in the garden. The wear persists with the atlas, so over
weeks your habits become paths.

**Why only here.** Only a diffusion world with a feedback loop turns a faint statistic into an
invented feature. Footfalls already exist (`player.onStep`). The painter projects any image from
any camera, so a top-down orthographic footprint splat can paint the wear map in atlas space
(2048² R8, 4 MB). The cue goes into the raw render before the feedback mix, where cues don't
compound. Prior art: desire paths in urban planning, and Death Stranding (2019), where trails wear
in wherever many players walk.

**Cost.** About a day. Per frame, one texture fetch in the capture shader and one splat per
footstep.

**Smallest experiment (≤2 h).** Fake the wear first. Patch the capture shader to warm a 1.2 m band
along one garden terrace. Dream at rest from two poses, at three cue strengths, with the cue on and
off. It succeeds if the dream paints a path along the band, and the weakest cue that works shows no
visible stripe of its own.

**Kills it.** The model ignores any cue too weak to show as a stripe in the raw render. It also
fails if the paths read as stains.

### 12. Dream Bridges

**Idea.** A few structures exist only as blueprint until they are dreamt, such as a bridge across
the atrium's abyss or the stair to nowhere in the Vestibule's void. You can't stand on their faint
lines, and their edge acts as a wall, so you never fall. When you look long enough for the dream to
paint them, they take your weight. You stand at the brink and watch the way across being dreamt.

**Why only here.** Atlas confidence becomes game state. Render the object's triangles in atlas
space into a 1x1 target, with an async readback every 0.5 s. The mean confidence then switches
collision prims on and off, via an `enabled` flag in `world/collision.js`. Prior art: The
Unfinished Swan (2012) reveals a white world with thrown paint, and Scanner Sombre (2017) reveals a
cave with LIDAR points. The stacks' inverted spiral stair and the Vestibule's tilted stair already
exist as scenery. Walking the inverted stair would need gravity that flips, as in Manifold Garden
(2019); that is weeks of work with real nausea risk.

**Cost.** 2-3 days. The per-frame cost is tiny.

**Smallest experiment (≤2 h).** Log one object's mean confidence over its charts while the view
looks at it and away (`memory`'s pack pass can read the tile back). It succeeds if confidence
crosses 0.6 within 2-6 s of looking and never while the object is unseen, and if the blueprint
reads in a filmstrip as something that could become a bridge.

**Kills it.** Confidence saturates within half a second, leaving no moment. Or the blueprint is too
faint to invite the walk.

### 13. Fossils

**Idea.** The dream keeps its best inventions. After each dream, the client measures where the
output departs from its input in structure (the dark figures in the baths' far arch, the moss over
the nave's water) and projects that "surprise" into the world. Inventions that come back to the
same spot over many dreams fossilize: the atlas stops relaxing them and paints over them more
slowly. Over weeks, the world gathers its own recurring hallucinations as permanent features, and
the first time one hardens a single whispered word names it.

**Why only here.** Both images are already on the GPU: the capture target and the result texture.
Surprise is a low-passed difference normalized for luminance, painted into a small pin map in atlas
space. The paint shader and the relax pass both consult that map. The name comes from Semantic
Paint's nouns or a small CLIP vocabulary match. DeepDream (2015) amplified a network's pareidolia;
this keeps it instead.

**Cost.** 2-4 days, plus more for naming. The per-frame cost is tiny.

**Smallest experiment (≤2 h).** At rest in each zone, save 40 input/result pairs
(`dream.lastJpeg`, `dream.lastResultBytes`). In numpy, average the low-passed surprise per zone and
overlay it. It succeeds if at least 3 zones have stable, nameable hotspots rather than texture
noise everywhere.

**Kills it.** Surprise is uniform fine detail, because the model invents texture everywhere. It
also fails if pinned patches stop responding to light and read as stickers. Unwanted figures could
also fossilize, since SD-Turbo ignores the negative prompt, so `R` (forget) must clear pins.

### 14. Dream Polaroids

**Idea.** Press a key to keep what you see: the newest live view (its image, its exact capture
depth and its camera) lifts out of the world as a Polaroid. Later, anywhere, you hold it up. The
old place's paint lands on the surfaces of the room you stand in, and the model folds the collage
into the local dream, so the geode ends up wearing the garden. The tier that would take weeks uses
the fact that capture depth is exact geometry: a kept view becomes a real 3D fragment (a mesh from
its depth buffer, carrying its paint) set down somewhere else, such as a slice of the nave's arcade
standing in the desert.

**Why only here.** A live view is already an image, a depth and a camera, projected at display
time (`dream/live.js`); a Polaroid is one that never expires. Viewfinder (2023) places a photo into
the world and makes its contents real. Here the photo is a dream.

**Cost.** The collage takes 1-2 days. Fragments take weeks: depth to mesh with cuts at depth edges,
plus collision and atlas charts. The per-frame cost is one extra live slot.

**Smallest experiment (≤2 h).** Save a garden live view's texture and camera. In the geode, render
fresh depth from the current camera into a scratch slot. Paint the saved image through it with one
paint job via `__hyp`, and film 5 s. It succeeds if the collage is striking and the dream resolves
it into a hybrid within about 5 s, rather than leaving a smear.

**Kills it.** Collage paint smears across depth edges and the model can't reconcile it. It also
fails if it plays like a toy and pulls against the ambient feel.

### 15. Sleepwalker

**Idea.** After the idle timeout, the dreamer does more than drift its gaze: it sleepwalks slowly
along the tour, turning toward whatever the dream painted most vividly and pausing where it's
beautiful. It keeps dreaming if you leave it overnight, or on a phone by the bed with the screen
dim and the sound on. The next title screen shows what you dreamt while you slept, as a slideshow
of its pauses rendered from the atlas.

**Why only here.** The idle gaze (`idleGaze` in `main.js`) and the tour (`level.tour` with its
spline autopilot) already exist. The saliency comes from the dream itself: local contrast times
brightness of the newest result, on a 32x20 downsample read back asynchronously. The slideshow
reuses the Memory Gallery's offscreen renders. Prior art: Brian Eno's 77 Million Paintings (2006),
generative painting meant to be left running.

**Cost.** About 2 days, plus battery tuning on the phone (a cap on the capture rate). There is no
per-frame cost. For comfort, passive motion is the worst kind for nausea, so walk at most 0.8 m/s,
with no head bob and turns of at most 10° per second.

**Smallest experiment (≤2 h).** Mark the saliency peak on 20 saved rest results across zones. Check
whether it picks what a person would look at, or blown-out lamps and the stacks' orb. Then run a
3-minute headless sleepwalk, keeping one frame every 10 s.

**Kills it.** Saliency locks onto the brightest light, or the walk gets stuck against walls.

### 16. Hypnic Fall

**Idea.** Falling is travel. If you step off the Vestibule's causeway or into the atrium's abyss,
you don't respawn. You fall slowly while the view fades to the fog colour, then land softly in
another zone, facing a view. (A hypnic jerk is that sensation of falling at sleep onset; Alice fell
slowly past cupboards and bookshelves.) If you land somewhere undreamt, the blueprint washes into
dream around you.

**Why only here.** `player.respawn()` below `killY` is the hook, and the tour's poses make good
landing spots. Landing somewhere undreamt turns the usual wash-in into the reveal. It is cheap
non-Euclidean glue between the zones.

**Cost.** Hours to a day, with no per-frame cost. Falling causes vertical vection, so show at most
about 1.5 s of the fall before the fade, and never tumble the camera.

**Smallest experiment (≤2 h).** Patch `player.respawn` through `__hyp` to arrive at a geode pose
under a 1.2 s fade, and film the fall and the landing. It succeeds if an eye check finds it gentle
and the wash-in on arrival looks intended.

**Kills it.** Accidental falls become teleports nobody wanted. Allow it only at the void edges.

---

**With four ≤2 h experiment slots,** I would run ideas 1, 2 and 3 as written, plus the
depth-disagreement census from idea 4 as the cheapest test of a crazy idea.
