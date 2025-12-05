"""Software raycaster using DDA algorithm with torch lighting."""

import math
import time
from typing import Optional

import numpy as np

import config
from game_state import Player, Map

# Torch configuration
TORCH_SPACING = 3  # Place torch every N wall units
TORCH_HEIGHT = 0.3  # Height on wall (0=bottom, 1=top), 0.3 = upper-mid
TORCH_WIDTH = 0.08  # Width of torch on wall (fraction of column)
TORCH_COLOR = (255, 150, 50)  # Orange flame
TORCH_GLOW_COLOR = (255, 100, 20)  # Deeper orange glow
TORCH_LIGHT_RADIUS = 4.0  # How far torch light reaches


def get_flicker(t: float, seed: float = 0) -> float:
    """Generate flickering intensity based on time and seed."""
    # Combine multiple sine waves for organic flicker
    flicker = (
        math.sin(t * 8 + seed) * 0.1 +
        math.sin(t * 13 + seed * 2) * 0.08 +
        math.sin(t * 21 + seed * 3) * 0.05
    )
    return 0.85 + flicker  # Range roughly 0.6-1.0


def has_torch(map_x: int, map_y: int, side: int) -> bool:
    """Determine if a wall cell should have a torch."""
    # Place torches on a grid pattern
    if side == 0:  # N/S wall
        return map_y % TORCH_SPACING == 1
    else:  # E/W wall
        return map_x % TORCH_SPACING == 1


def cast_rays(player: Player, game_map: Map, texture_manager=None) -> np.ndarray:
    """
    Cast rays and render a frame with torch lighting.

    Args:
        player: Player object with position and angle.
        game_map: Map object with wall data.
        texture_manager: Optional TextureManager for per-pixel texture sampling.

    Returns:
        numpy array of shape (RENDER_HEIGHT, RENDER_WIDTH, 3) with RGB values.
    """
    width = config.RENDER_WIDTH
    height = config.RENDER_HEIGHT
    current_time = time.time()

    # Create output buffer
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Fill ceiling and floor with slight gradient
    for y in range(height // 2):
        # Ceiling gets slightly lighter toward horizon
        ceil_factor = 0.7 + 0.3 * (y / (height // 2))
        frame[y, :] = [int(c * ceil_factor) for c in config.CEILING_COLOR]

    for y in range(height // 2, height):
        # Floor gets slightly lighter toward horizon
        floor_factor = 0.7 + 0.3 * (1 - (y - height // 2) / (height // 2))
        frame[y, :] = [int(c * floor_factor) for c in config.FLOOR_COLOR]

    # Store torch info for glow pass
    torch_columns = []

    # Cast a ray for each column
    for x in range(width):
        # Calculate ray direction
        camera_x = 2 * x / width - 1
        ray_angle = player.angle + camera_x * (config.FOV / 2)
        ray_dir_x = math.cos(ray_angle)
        ray_dir_y = math.sin(ray_angle)

        # Current map cell
        map_x = int(player.x)
        map_y = int(player.y)

        # Length of ray from one x/y side to next
        delta_dist_x = abs(1 / ray_dir_x) if ray_dir_x != 0 else float('inf')
        delta_dist_y = abs(1 / ray_dir_y) if ray_dir_y != 0 else float('inf')

        # Direction to step in x/y (+1 or -1)
        if ray_dir_x < 0:
            step_x = -1
            side_dist_x = (player.x - map_x) * delta_dist_x
        else:
            step_x = 1
            side_dist_x = (map_x + 1.0 - player.x) * delta_dist_x

        if ray_dir_y < 0:
            step_y = -1
            side_dist_y = (player.y - map_y) * delta_dist_y
        else:
            step_y = 1
            side_dist_y = (map_y + 1.0 - player.y) * delta_dist_y

        # DDA loop
        hit = False
        side = 0
        wall_type = 0

        while not hit:
            if side_dist_x < side_dist_y:
                side_dist_x += delta_dist_x
                map_x += step_x
                side = 0
            else:
                side_dist_y += delta_dist_y
                map_y += step_y
                side = 1

            wall_type = game_map.get_cell(map_x, map_y)
            if wall_type != 0:
                hit = True

        # Calculate perpendicular distance
        if side == 0:
            perp_dist = (map_x - player.x + (1 - step_x) / 2) / ray_dir_x if ray_dir_x != 0 else config.MAX_DEPTH
            # Calculate where on the wall we hit (0-1)
            wall_x = player.y + perp_dist * ray_dir_y
            wall_x -= math.floor(wall_x)
        else:
            perp_dist = (map_y - player.y + (1 - step_y) / 2) / ray_dir_y if ray_dir_y != 0 else config.MAX_DEPTH
            wall_x = player.x + perp_dist * ray_dir_x
            wall_x -= math.floor(wall_x)

        perp_dist = max(0.001, min(perp_dist, config.MAX_DEPTH))

        # Calculate wall height on screen
        wall_height = int(height / perp_dist)
        draw_start = max(0, height // 2 - wall_height // 2)
        draw_end = min(height - 1, height // 2 + wall_height // 2)

        # Get wall color
        base_color = config.WALL_COLORS.get(wall_type, config.DEFAULT_WALL_COLOR)

        # Base distance shading
        shade = max(0.15, 1.0 - perp_dist / config.MAX_DEPTH)
        if side == 1:
            shade *= config.EW_SHADE_FACTOR

        # Check for nearby torches and add warm light
        torch_on_wall = has_torch(map_x, map_y, side)
        torch_seed = map_x * 7 + map_y * 13  # Unique flicker per torch
        flicker = get_flicker(current_time, torch_seed)

        # Add torch glow to nearby walls
        warm_light = 0.0
        if torch_on_wall and perp_dist < TORCH_LIGHT_RADIUS:
            light_intensity = (1 - perp_dist / TORCH_LIGHT_RADIUS) * flicker
            warm_light = light_intensity * 0.5

        # Draw wall column - either with textures or solid color
        use_textures = texture_manager is not None and texture_manager.has_textures()

        if use_textures:
            # Per-pixel texture sampling
            for y in range(draw_start, draw_end + 1):
                # Calculate V coordinate (0 at top, 1 at bottom)
                v = (y - draw_start) / max(1, draw_end - draw_start)
                tex_color = texture_manager.sample(wall_type, wall_x, v)

                # Apply lighting
                color = []
                for i, c in enumerate(tex_color):
                    lit = c * shade
                    if warm_light > 0:
                        warmth = [1.3, 0.9, 0.5][i]
                        lit = lit * (1 + warm_light * warmth)
                    color.append(int(min(255, lit)))
                frame[y, x] = color
        else:
            # Solid color (original behavior)
            color = []
            for i, c in enumerate(base_color):
                lit = c * shade
                # Add warm orange tint from torches
                if warm_light > 0:
                    warmth = [1.3, 0.9, 0.5][i]  # R, G, B multipliers for warm light
                    lit = lit * (1 + warm_light * warmth)
                color.append(int(min(255, lit)))

            frame[draw_start:draw_end + 1, x] = color

        # Draw torch on wall if present
        if torch_on_wall and perp_dist < TORCH_LIGHT_RADIUS * 1.5:
            # Check if we're at the torch position on the wall
            torch_center = 0.5
            if abs(wall_x - torch_center) < TORCH_WIDTH:
                # Calculate torch vertical position
                torch_y_center = draw_start + int((draw_end - draw_start) * (1 - TORCH_HEIGHT))
                torch_h = max(3, int((draw_end - draw_start) * 0.15))

                torch_top = max(draw_start, torch_y_center - torch_h)
                torch_bottom = min(draw_end, torch_y_center + torch_h // 2)

                # Draw torch body (darker base)
                for y in range(torch_y_center, torch_bottom + 1):
                    if 0 <= y < height:
                        frame[y, x] = (80, 50, 30)  # Brown torch handle

                # Draw flame with flicker
                flame_intensity = flicker
                for y in range(torch_top, torch_y_center + 1):
                    if 0 <= y < height:
                        # Flame gets brighter toward top
                        flame_pos = 1 - (y - torch_top) / max(1, torch_y_center - torch_top)
                        r = int(min(255, TORCH_COLOR[0] * flame_intensity * (0.8 + 0.2 * flame_pos)))
                        g = int(min(255, TORCH_COLOR[1] * flame_intensity * flame_pos))
                        b = int(min(255, TORCH_COLOR[2] * flame_intensity * flame_pos * 0.5))
                        frame[y, x] = (r, g, b)

                # Record torch for glow pass
                torch_columns.append((x, torch_top, torch_y_center, perp_dist, flicker))

    # Add bloom/glow around torches
    for tx, t_top, t_mid, dist, flicker in torch_columns:
        glow_radius = max(2, int(8 / (dist + 0.5)))
        glow_intensity = flicker * (1 - dist / (TORCH_LIGHT_RADIUS * 1.5))

        for dx in range(-glow_radius, glow_radius + 1):
            for dy in range(-glow_radius, glow_radius + 1):
                px, py = tx + dx, t_top + dy
                if 0 <= px < width and 0 <= py < height:
                    d = math.sqrt(dx * dx + dy * dy)
                    if d <= glow_radius and d > 0:
                        falloff = (1 - d / glow_radius) * glow_intensity * 0.3
                        current = frame[py, px].astype(float)
                        glow = np.array([255, 120, 40]) * falloff
                        frame[py, px] = np.clip(current + glow, 0, 255).astype(np.uint8)

    return frame
