"""Game state management: Player and Map classes."""

import math
from dataclasses import dataclass
from typing import List

import config


@dataclass
class Player:
    """Player state with position and viewing angle."""
    x: float
    y: float
    angle: float  # in radians, 0 = facing +X (east)

    def move_forward(self, distance: float, game_map: 'Map') -> None:
        """Move player forward in facing direction."""
        new_x = self.x + math.cos(self.angle) * distance
        new_y = self.y + math.sin(self.angle) * distance
        self._try_move(new_x, new_y, game_map)

    def move_backward(self, distance: float, game_map: 'Map') -> None:
        """Move player backward from facing direction."""
        new_x = self.x - math.cos(self.angle) * distance
        new_y = self.y - math.sin(self.angle) * distance
        self._try_move(new_x, new_y, game_map)

    def turn_left(self, angle: float) -> None:
        """Turn player left by given angle in radians."""
        self.angle -= angle
        # Keep angle in [0, 2*pi) range
        self.angle = self.angle % (2 * math.pi)

    def turn_right(self, angle: float) -> None:
        """Turn player right by given angle in radians."""
        self.angle += angle
        # Keep angle in [0, 2*pi) range
        self.angle = self.angle % (2 * math.pi)

    def _try_move(self, new_x: float, new_y: float, game_map: 'Map') -> None:
        """Attempt to move to new position with collision detection."""
        # Simple collision: don't allow movement if new position is in a wall
        # Check with a small margin to prevent getting too close to walls
        margin = 0.2

        # Try X movement first
        if not game_map.is_wall(new_x, self.y):
            # Also check corners
            if (not game_map.is_wall(new_x + margin, self.y + margin) and
                not game_map.is_wall(new_x + margin, self.y - margin) and
                not game_map.is_wall(new_x - margin, self.y + margin) and
                not game_map.is_wall(new_x - margin, self.y - margin)):
                self.x = new_x

        # Try Y movement
        if not game_map.is_wall(self.x, new_y):
            if (not game_map.is_wall(self.x + margin, new_y + margin) and
                not game_map.is_wall(self.x + margin, new_y - margin) and
                not game_map.is_wall(self.x - margin, new_y + margin) and
                not game_map.is_wall(self.x - margin, new_y - margin)):
                self.y = new_y


class Map:
    """2D grid map where 0 = empty, 1+ = wall types."""

    def __init__(self, grid: List[List[int]]):
        """Initialize map from 2D grid."""
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0

    def get_cell(self, x: int, y: int) -> int:
        """Get the value at grid position (x, y). Returns 1 if out of bounds."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return 1  # Treat out-of-bounds as wall

    def is_wall(self, x: float, y: float) -> bool:
        """Check if world position (x, y) is inside a wall."""
        grid_x = int(x)
        grid_y = int(y)
        return self.get_cell(grid_x, grid_y) != 0
