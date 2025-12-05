# torch

Wolfenstein-style raycaster with real-time Stable Diffusion stylization. Features flickering wall torches and dynamic firelight.

## Requirements

- Python 3.11+
- macOS with Apple Silicon (MPS)

## Setup

```bash
pip install pygame numpy torch diffusers transformers accelerate pillow
```

## Run

```bash
python main.py
```

## Controls

- **WASD / Arrows** - Move and turn
- **Space** - Toggle SD processing (first press loads model)
- **[ ]** - Cycle visual styles
- **Esc** - Quit
