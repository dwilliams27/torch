# Project: diffused-rays

## Overview

Build a Wolfenstein-style 3D dungeon crawler where every rendered frame is processed through Stable Diffusion img2img in real-time. The player navigates a simple maze, but instead of seeing traditional graphics, they see SD's "interpretation" of each frame. Target platform is macOS with Apple Silicon.

## Tech Stack

- Python 3.11+
- pygame: game loop, input handling, display
- numpy: software raycaster, image array manipulation
- PyTorch with MPS backend: Apple Silicon GPU acceleration
- diffusers (Hugging Face): Stable Diffusion inference, specifically SDXL Turbo
- Pillow: image conversion between numpy and SD pipeline

## Architecture

The game runs a synchronous pipeline each frame:

1. GAME STATE: Player has position (x, y as floats), angle (radians), and a reference to a 2D grid map where each cell is either empty (0) or wall (1+, different values can mean different wall types)

2. RAYCASTER: A software renderer that reads game state, casts rays from player position across the field of view, calculates wall distances, and outputs a 128x128 RGB numpy array with flat-colored walls (no textures needed, just solid colors based on wall type and distance for shading)

3. SD IMG2IMG: Takes the raycaster output as the input image, runs SDXL Turbo with approximately 2 inference steps, strength around 0.3-0.5, guidance_scale=0.0 (SDXL Turbo requirement), outputs a stylized 128x128 image

4. DISPLAY: The SD output is scaled up to the window size (512x512 or 640x480) and displayed via pygame

5. INPUT: Keyboard input is captured (W/S for forward/back, A/D for strafe or turn, mouse or arrow keys for looking), game state is updated, loop repeats

## File Structure
```
diffused-rays/
├── main.py              # Entry point, game loop, pygame setup
├── game_state.py        # Player class, Map class, movement logic
├── raycaster.py         # Software raycaster, outputs numpy array
├── stylizer.py          # SD pipeline wrapper, img2img logic
├── config.py            # Constants: resolution, SD params, movement speed
├── maps/
│   └── test_map.py      # Simple hardcoded map for testing
└── requirements.txt
```

## Milestones

Build these in order. Each should be runnable and testable before moving on.

### Milestone 1: Raycaster Only

Create a working raycaster that renders to pygame without any SD processing. Player should be able to move around a simple map with WASD. Walls should be flat colored (different colors for N/S vs E/W facing walls is a nice touch). Output resolution is 128x128, scaled up to 512x512 for display. Target 60fps.

Definition of done: Player can navigate a maze, walls render with distance-based shading, runs smoothly.

### Milestone 2: SD Pipeline Standalone

Create the stylizer module that can take any 128x128 numpy array, run it through SDXL Turbo img2img, and return the result. Test with a static image first. Verify MPS acceleration is working (check torch.backends.mps.is_available()).

Definition of done: A test script that loads an image, stylizes it, saves the output. Runs in under 1 second on Apple Silicon.

### Milestone 3: Integration

Connect raycaster output to SD input. Expect low fps initially (2-5 fps). Add an FPS counter to the display. Use a simple prompt like "dark dungeon corridor, stone walls, torchlight, fantasy art".

Definition of done: Player can walk around and see SD-stylized frames updating in real-time, even if slow.

### Milestone 4: Optimization

Improve framerate to 10+ fps through:
- Reducing inference steps (try 1-2)
- Tuning strength parameter
- Ensuring no unnecessary CPU-GPU transfers
- Optional: async frame generation (render frame N+1 while displaying frame N)

Definition of done: Playable at 10+ fps with acceptable visual quality.

## Key Technical Details

Raycaster algorithm (DDA style):
- For each x column of the output image, cast a ray from player position
- Step through the 2D grid until a wall is hit
- Calculate perpendicular distance to avoid fisheye
- Wall height on screen is inversely proportional to distance
- Draw vertical line for that column

SDXL Turbo specifics:
- Model: "stabilityai/sdxl-turbo"
- Use torch_dtype=torch.float16 and variant="fp16"
- guidance_scale must be 0.0
- num_inference_steps: 1-4 (start with 2)
- strength: 0.3-0.6 (controls how much SD changes the input)

MPS (Metal Performance Shaders):
- Use pipe.to("mps") to move pipeline to GPU
- After operations, may need torch.mps.synchronize() for timing accuracy

## Constraints

- Target resolution is 128x128 for SD processing (can experiment with 256x256 later)
- Display window should be 512x512
- Keep the raycaster simple: no textures, no sprites, just colored walls
- Single-threaded is fine for initial implementation
- No external game assets needed, everything is procedural or generated

## Starting Point

Begin with Milestone 1. Create a minimal pygame window with the raycaster working. Use this test map:
```python
MAP = [
    [1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,0,1,1,0,0,1],
    [1,0,1,0,0,0,1,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,1,0,0,0,1,0,0,1],
    [1,0,1,1,0,1,1,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1],
]
PLAYER_START = (5.0, 5.0)
PLAYER_START_ANGLE = 0.0
```

## Notes

- FOV: 60 degrees
- A/D should rotate, not strafe (keep it simple like original Wolfenstein)
- Wall colors: pick sensible defaults, expose in config
- Collision: simple, don't let player center enter wall cells, no need for sliding
- During SD processing: just show previous frame (will look like low fps, which it is)
- Prompt: hardcoded in config.py for now
- If MPS unavailable: warn and fall back to CPU