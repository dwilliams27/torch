"""Software raycaster using DDA algorithm."""

import math
import numpy as np

import config
from game_state import Player, Map


def cast_rays(player: Player, game_map: Map) -> np.ndarray:
    """
    Cast rays and render a frame.

    Returns:
        numpy array of shape (RENDER_HEIGHT, RENDER_WIDTH, 3) with RGB values.
    """
    width = config.RENDER_WIDTH
    height = config.RENDER_HEIGHT

    # Create output buffer
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Fill ceiling and floor
    frame[:height // 2, :] = config.CEILING_COLOR
    frame[height // 2:, :] = config.FLOOR_COLOR

    # Cast a ray for each column
    for x in range(width):
        # Calculate ray direction
        # Camera x coordinate: -1 (left) to +1 (right)
        camera_x = 2 * x / width - 1

        # Ray direction from player angle and FOV
        ray_angle = player.angle + camera_x * (config.FOV / 2)
        ray_dir_x = math.cos(ray_angle)
        ray_dir_y = math.sin(ray_angle)

        # Current map cell
        map_x = int(player.x)
        map_y = int(player.y)

        # Length of ray from one x/y side to next
        # Avoid division by zero
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
        side = 0  # 0 for N/S wall, 1 for E/W wall
        wall_type = 0

        while not hit:
            # Jump to next map cell
            if side_dist_x < side_dist_y:
                side_dist_x += delta_dist_x
                map_x += step_x
                side = 0  # Hit vertical (N/S facing) wall
            else:
                side_dist_y += delta_dist_y
                map_y += step_y
                side = 1  # Hit horizontal (E/W facing) wall

            # Check for wall hit
            wall_type = game_map.get_cell(map_x, map_y)
            if wall_type != 0:
                hit = True

        # Calculate perpendicular distance (avoids fisheye)
        if side == 0:
            perp_dist = (map_x - player.x + (1 - step_x) / 2) / ray_dir_x if ray_dir_x != 0 else config.MAX_DEPTH
        else:
            perp_dist = (map_y - player.y + (1 - step_y) / 2) / ray_dir_y if ray_dir_y != 0 else config.MAX_DEPTH

        # Clamp distance
        perp_dist = max(0.001, min(perp_dist, config.MAX_DEPTH))

        # Calculate wall height on screen
        wall_height = int(height / perp_dist)

        # Calculate start and end y for this wall slice
        draw_start = max(0, height // 2 - wall_height // 2)
        draw_end = min(height - 1, height // 2 + wall_height // 2)

        # Get wall color based on type
        base_color = config.WALL_COLORS.get(wall_type, config.DEFAULT_WALL_COLOR)

        # Apply distance shading (darker = further)
        shade = max(0.2, 1.0 - perp_dist / config.MAX_DEPTH)

        # Apply E/W wall shading to distinguish wall orientations
        if side == 1:
            shade *= config.EW_SHADE_FACTOR

        # Calculate final color
        color = tuple(int(c * shade) for c in base_color)

        # Draw wall column
        frame[draw_start:draw_end + 1, x] = color

    return frame
