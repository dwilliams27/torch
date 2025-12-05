"""Configuration constants for diffused-rays."""

import math

# Display settings
RENDER_WIDTH = 256
RENDER_HEIGHT = 256
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 800
FPS_CAP = 60

# Player settings
MOVE_SPEED = 3.0  # units per second
TURN_SPEED = 2.0  # radians per second
FOV = math.radians(60)  # 60 degrees field of view

# Raycaster settings
MAX_DEPTH = 25.0  # maximum ray distance (increased for larger map)

# Wall colors (RGB) - vibrant base colors before distance shading
WALL_COLORS = {
    1: (140, 130, 120),  # Stone - warm gray
    2: (190, 80, 60),    # Brick - terracotta red
    3: (60, 160, 80),    # Moss stone - forest green
    4: (100, 200, 220),  # Ice/crystal - cyan
    5: (220, 180, 50),   # Gold/treasure - rich yellow
    6: (120, 60, 160),   # Obsidian - deep purple
    7: (140, 40, 50),    # Blood stone - dark crimson
    8: (70, 100, 180),   # Ancient - mystic blue
}
DEFAULT_WALL_COLOR = (100, 100, 100)  # Gray for undefined wall types

# N/S vs E/W wall shading (multiply factor for E/W walls to distinguish them)
EW_SHADE_FACTOR = 0.7

# Floor and ceiling colors - darker for dungeon atmosphere
FLOOR_COLOR = (40, 35, 30)
CEILING_COLOR = (20, 20, 30)

# Stable Diffusion settings
# Using SD Turbo (non-XL) for better MPS compatibility and faster inference
SD_MODEL = "stabilityai/sd-turbo"
SD_PROMPT = "dark medieval dungeon, burning torches on walls, flickering firelight, warm orange glow, ancient stone corridors, shadows dancing, atmospheric"
SD_NEGATIVE_PROMPT = ""
SD_NUM_STEPS = 2  # 2 steps minimum for img2img on MPS (1 step causes errors)
SD_STRENGTH = 0.5  # How much to transform (0.3-0.6 for img2img)
SD_GUIDANCE_SCALE = 0.0  # Required for SD Turbo
