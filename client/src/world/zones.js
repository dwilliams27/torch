// Zone identities + surface types. Pure data (node-safe).
// Prompts are written for 1-step SD-Turbo/SDXS-class img2img: concrete nouns + light + medium.

export const SURFACES = [
  { name: 'stone',     color: [0.56, 0.54, 0.50], pattern: 'stone',   emissive: 0 },
  { name: 'darkstone', color: [0.28, 0.29, 0.31], pattern: 'stone',   emissive: 0 },
  { name: 'marble',    color: [0.84, 0.82, 0.78], pattern: 'tile',    emissive: 0 },
  { name: 'brick',     color: [0.54, 0.36, 0.28], pattern: 'brick',   emissive: 0 },
  { name: 'wood',      color: [0.40, 0.25, 0.14], pattern: 'wood',    emissive: 0 },
  { name: 'books',     color: [0.46, 0.22, 0.16], pattern: 'wood',    emissive: 0.02 },
  { name: 'concrete',  color: [0.64, 0.63, 0.60], pattern: 'formwork', emissive: 0 },
  { name: 'metal',     color: [0.46, 0.48, 0.52], pattern: 'metal',   emissive: 0 },
  { name: 'gold',      color: [0.86, 0.66, 0.30], pattern: 'metal',   emissive: 0.04 },
  { name: 'rock',      color: [0.30, 0.27, 0.33], pattern: 'stone',   emissive: 0 },
  { name: 'crystal',   color: [0.62, 0.46, 0.98], pattern: 'crystal', emissive: 0.45 },
  { name: 'crystal2',  color: [0.36, 0.86, 0.96], pattern: 'crystal', emissive: 0.45 },
  { name: 'tile',      color: [0.86, 0.90, 0.92], pattern: 'tile',    emissive: 0 },
  { name: 'tile_dark', color: [0.16, 0.20, 0.30], pattern: 'tile',    emissive: 0 },
  { name: 'water',     color: [0.08, 0.20, 0.24], pattern: 'water',   emissive: 0.04 },
  { name: 'pool',      color: [0.10, 0.62, 0.68], pattern: 'water',   emissive: 0.22 },
  { name: 'sand',      color: [0.80, 0.63, 0.45], pattern: 'plaster', emissive: 0 },
  { name: 'sandstone', color: [0.74, 0.50, 0.36], pattern: 'stone',   emissive: 0 },
  { name: 'hedge',     color: [0.18, 0.36, 0.17], pattern: 'plaster', emissive: 0 },
  { name: 'leaf',      color: [0.26, 0.46, 0.20], pattern: 'plaster', emissive: 0 },
  { name: 'plaster',   color: [0.80, 0.76, 0.70], pattern: 'plaster', emissive: 0 },
  { name: 'canvas',    color: [0.52, 0.40, 0.30], pattern: 'plaster', emissive: 0.12 },
  { name: 'sky',       color: [0.22, 0.20, 0.34], pattern: 'sky',     emissive: 1 },
  { name: 'fresco',    color: [0.46, 0.62, 0.86], pattern: 'sky',     emissive: 0.55 },
  { name: 'glow_warm', color: [1.00, 0.74, 0.42], pattern: 'glow',    emissive: 1 },
  { name: 'glow_cool', color: [0.60, 0.82, 1.00], pattern: 'glow',    emissive: 1 },
  { name: 'neon_pink', color: [1.00, 0.28, 0.72], pattern: 'glow',    emissive: 1 },
  { name: 'neon_cyan', color: [0.22, 0.96, 1.00], pattern: 'glow',    emissive: 1 },
  { name: 'moon',      color: [0.98, 0.93, 0.84], pattern: 'glow',    emissive: 1 },
  { name: 'moon2',     color: [0.98, 0.62, 0.52], pattern: 'glow',    emissive: 1 },
];

const NEG = 'blurry, lowres, text, watermark, signature, people, faces, deformed, jpeg artifacts, oversaturated';

export const ZONES = [
  {
    id: 'vestibule', name: 'The Vestibule', subtitle: 'where the waking world thins',
    // name the geometry (a tunnel of rotated square frames around a walkway): "picture
    // frames" got a gallery of paintings, and the real frames showed through it as panes
    prompt: 'a tunnel of enormous square gilded baroque frames, one behind another, each rotated a little further, twisting around a narrow walkway through a starry indigo void, candlelight glow, surreal dream, baroque oil painting, rich detail',
    negative: NEG,
    fog: [0.05, 0.04, 0.10], fogDensity: 0.018, light: [1.0, 0.78, 0.48], ambient: [0.10, 0.09, 0.16], sky: [0.06, 0.05, 0.14],
    mood: 'hushed threshold, distant choir pad, slow heartbeat, dust',
  },
  {
    id: 'nave', name: 'The Drowned Nave', subtitle: 'a cathedral the sea remembered',
    prompt: 'a flooded gothic cathedral nave, towering stone columns and pointed arches reflected in still dark water, shafts of pale light through tall lancet windows, moss, mist, cinematic matte painting, highly detailed',
    negative: NEG,
    fog: [0.10, 0.17, 0.19], fogDensity: 0.022, light: [0.70, 0.88, 1.0], ambient: [0.10, 0.15, 0.17], sky: [0.30, 0.45, 0.52],
    mood: 'vast reverberant stone, dripping water, low organ drone, whale-like swells',
  },
  {
    id: 'stacks', name: 'The Stacks', subtitle: 'every book you forgot to write',
    prompt: 'an impossibly tall circular library shaft, endless wooden bookshelves full of old books, spiral staircase and narrow bridges, warm lamplight, dust motes, glowing oculus far above, baroque, painterly, intricate',
    negative: NEG,
    fog: [0.16, 0.10, 0.06], fogDensity: 0.016, light: [1.0, 0.72, 0.40], ambient: [0.16, 0.11, 0.08], sky: [0.95, 0.80, 0.55],
    mood: 'warm paper hush, ticking clocks, page rustle, glassy harp arpeggios',
  },
  {
    id: 'geode', name: 'The Geode', subtitle: 'light, slowed until it hardened',
    prompt: 'a glowing crystal cavern, giant amethyst and quartz crystals, bioluminescent violet and cyan light, glittering reflections, underground, fantasy concept art, ethereal, luminous',
    negative: NEG,
    fog: [0.10, 0.05, 0.16], fogDensity: 0.03, light: [0.70, 0.50, 1.0], ambient: [0.10, 0.06, 0.16], sky: [0.20, 0.10, 0.30],
    mood: 'crystalline shimmer, glass bells, deep sub bass, slow resonant chimes',
  },
  {
    id: 'baths', name: 'Lethe Baths', subtitle: 'the water here forgets for you',
    // name the vaulted hall and its arcades: with cross-frame attention on, the old prompt
    // ("a neon-lit bathhouse ... arches") left the walls bare stucco while walking
    prompt: 'a long barrel-vaulted roman bathhouse hall, a turquoise pool stepping down in terraces between round-arched arcades of white mosaic tile, pink and cyan neon tubes along the vault, a glowing arch at the far end, steam, glossy wet reflections, vaporwave, cinematic lighting, intricate tilework',
    negative: NEG,
    fog: [0.16, 0.07, 0.16], fogDensity: 0.024, light: [1.0, 0.40, 0.80], ambient: [0.10, 0.10, 0.16], sky: [0.35, 0.15, 0.40],
    mood: 'wet synth pads, slow chorus, lapping water, 80s dream pop haze',
  },
  {
    id: 'garden', name: 'The Sunken Garden', subtitle: 'a sky someone painted on the ceiling',
    prompt: 'a sunken garden courtyard under a painted fresco sky dome, overgrown hedges, tall cypress trees, marble fountain, terraces with stone balustrades, golden hour light, lush, romantic landscape painting',
    negative: NEG,
    fog: [0.20, 0.22, 0.14], fogDensity: 0.014, light: [1.0, 0.86, 0.60], ambient: [0.18, 0.20, 0.14], sky: [0.50, 0.66, 0.86],
    mood: 'birdsong in reverse, fountain trickle, pastoral woodwinds, bright major pad',
  },
  {
    id: 'atrium', name: 'Monolith Atrium', subtitle: 'concrete that learned to float',
    // describe what the walk actually passes (a walkway to a doorway in a huge wall): asked
    // for monoliths and ramps over an abyss, the model painted beams onto the flat planes,
    // and they drifted as you walked
    prompt: 'a narrow concrete walkway across a vast brutalist hall toward a tall arched doorway in a colossal board-formed concrete wall, distant floating concrete monoliths, soft daylight from above, long shadows, monumental scale, architectural photography, detailed concrete texture',
    negative: NEG,
    fog: [0.24, 0.23, 0.22], fogDensity: 0.009, light: [1.0, 0.92, 0.78], ambient: [0.16, 0.15, 0.15], sky: [0.80, 0.78, 0.72],
    mood: 'monumental silence, sub drones, distant wind, sparse piano',
  },
  {
    id: 'desert', name: 'Desert of Two Moons', subtitle: 'the arches hold up nothing',
    // the walk climbs a staircase to a windowed tower between walls; asked for arches across
    // an empty desert, the model painted a distant mesa and the stairs became a flat slab
    prompt: 'a colossal sandstone staircase climbing toward a tall windowed tower between towering sandstone walls, endless desert at dusk, two enormous moons in a violet sky, surreal de chirico landscape, long shadows, cinematic matte painting',
    negative: NEG,
    fog: [0.34, 0.22, 0.30], fogDensity: 0.006, light: [1.0, 0.70, 0.55], ambient: [0.20, 0.15, 0.24], sky: [0.42, 0.28, 0.52],
    mood: 'wide open dusk, wind, detuned slow strings, lonely theremin',
  },
];
