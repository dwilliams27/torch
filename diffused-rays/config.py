"""Configuration constants for diffused-rays."""

import math

# Display settings
RENDER_WIDTH = 128
RENDER_HEIGHT = 128
DISPLAY_WIDTH = 512
DISPLAY_HEIGHT = 512
FPS_CAP = 60

# Player settings
MOVE_SPEED = 3.0  # units per second
TURN_SPEED = 2.0  # radians per second
FOV = math.radians(60)  # 60 degrees field of view

# Raycaster settings
MAX_DEPTH = 20.0  # maximum ray distance

# Wall colors (RGB) - base colors before distance shading
WALL_COLORS = {
    1: (180, 0, 0),      # Red walls
    2: (0, 180, 0),      # Green walls
    3: (0, 0, 180),      # Blue walls
    4: (180, 180, 0),    # Yellow walls
}
DEFAULT_WALL_COLOR = (128, 128, 128)  # Gray for undefined wall types

# N/S vs E/W wall shading (multiply factor for E/W walls to distinguish them)
EW_SHADE_FACTOR = 0.7

# Floor and ceiling colors
FLOOR_COLOR = (50, 50, 50)
CEILING_COLOR = (30, 30, 40)

# Stable Diffusion settings (for later milestones)
# Using SD Turbo (non-XL) for better MPS compatibility and faster inference
SD_MODEL = "stabilityai/sd-turbo"
SD_PROMPT = "dark dungeon corridor, stone walls, torchlight, fantasy art"
SD_NEGATIVE_PROMPT = ""
SD_NUM_STEPS = 2  # 2 steps for img2img, 1 step requires strength=1.0
SD_STRENGTH = 0.5  # How much to transform (0.3-0.6 for img2img)
SD_GUIDANCE_SCALE = 0.0  # Required for SD Turbo
