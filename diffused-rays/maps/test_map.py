"""Test map for development."""

# Simple 10x10 maze
# 0 = empty, 1 = red wall, 2 = green wall, 3 = blue wall
MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 2, 2, 0, 3, 3, 0, 0, 1],
    [1, 0, 2, 0, 0, 0, 3, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 2, 0, 0, 0, 3, 0, 0, 1],
    [1, 0, 2, 2, 0, 3, 3, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# Player starting position (center of the map)
PLAYER_START_X = 5.0
PLAYER_START_Y = 5.0
PLAYER_START_ANGLE = 0.0  # Facing east (+X direction)
