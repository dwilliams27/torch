"""Trippy frame blending for smooth SD transitions.

Blends between SD-generated frames with psychedelic effects.
"""

import numpy as np
from collections import deque
import time


class FrameBlender:
    """Blends SD frames over time for smoother, trippier visuals."""

    def __init__(self,
                 crossfade_frames: int = 8,
                 ghost_decay: float = 0.85,
                 ghost_layers: int = 4,
                 chromatic_strength: float = 0.015,
                 color_bleed: float = 0.1):
        """Initialize the frame blender.

        Args:
            crossfade_frames: Number of display frames to crossfade over
            ghost_decay: How quickly ghost layers fade (0-1)
            ghost_layers: Number of ghost/echo layers
            chromatic_strength: Chromatic aberration amount (0-0.05)
            color_bleed: Color bleeding/smearing amount (0-0.3)
        """
        self.crossfade_frames = crossfade_frames
        self.ghost_decay = ghost_decay
        self.ghost_layers = ghost_layers
        self.chromatic_strength = chromatic_strength
        self.color_bleed = color_bleed

        # Frame history for effects
        self.frame_history = deque(maxlen=ghost_layers + 1)

        # Crossfade state
        self.prev_sd_frame = None
        self.curr_sd_frame = None
        self.blend_progress = 1.0  # 0.0 = prev_frame, 1.0 = curr_frame
        self.last_new_frame_time = 0.0

        # For motion detection
        self.prev_frame_for_motion = None

    def new_sd_frame(self, frame: np.ndarray):
        """Register a new SD-generated frame.

        Args:
            frame: New SD frame (H, W, 3) uint8
        """
        # Shift frames for crossfade
        if self.curr_sd_frame is not None:
            self.prev_sd_frame = self.curr_sd_frame.copy()
            # Add to history for ghost effect
            self.frame_history.append(self.curr_sd_frame.copy())

        self.curr_sd_frame = frame.copy()
        self.blend_progress = 0.0
        self.last_new_frame_time = time.time()

    def get_blended_frame(self, raw_frame: np.ndarray, dt: float = 0.016) -> np.ndarray:
        """Get the current blended/effected frame.

        Args:
            raw_frame: Current raw raycaster frame (fallback)
            dt: Delta time since last call

        Returns:
            Blended frame with effects
        """
        if self.curr_sd_frame is None:
            return raw_frame

        # Advance crossfade progress
        # Aim to complete crossfade over crossfade_frames worth of time
        fade_speed = 1.0 / max(1, self.crossfade_frames)
        self.blend_progress = min(1.0, self.blend_progress + fade_speed)

        # Start with crossfade between prev and current SD frames
        if self.prev_sd_frame is not None and self.blend_progress < 1.0:
            # Smooth easing function
            t = self._ease_in_out(self.blend_progress)
            result = self._blend_frames(self.prev_sd_frame, self.curr_sd_frame, t)
        else:
            result = self.curr_sd_frame.copy()

        # Apply ghost/trail effect
        if len(self.frame_history) > 0 and self.ghost_decay < 1.0:
            result = self._apply_ghost_effect(result)

        # Apply chromatic aberration
        if self.chromatic_strength > 0:
            result = self._apply_chromatic_aberration(result)

        # Apply color bleeding
        if self.color_bleed > 0:
            result = self._apply_color_bleed(result)

        # Store for motion detection
        self.prev_frame_for_motion = result.copy()

        return result

    def _ease_in_out(self, t: float) -> float:
        """Smooth easing function for crossfade."""
        # Cubic ease in-out
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2

    def _blend_frames(self, frame_a: np.ndarray, frame_b: np.ndarray, t: float) -> np.ndarray:
        """Blend two frames with factor t (0 = frame_a, 1 = frame_b)."""
        return (
            frame_a.astype(np.float32) * (1 - t) +
            frame_b.astype(np.float32) * t
        ).astype(np.uint8)

    def _apply_ghost_effect(self, frame: np.ndarray) -> np.ndarray:
        """Apply ghosting/trail effect from frame history."""
        result = frame.astype(np.float32)

        # Blend in ghost layers with decreasing opacity
        ghost_weight = 0.15  # Base weight for ghost layers
        for i, ghost_frame in enumerate(reversed(list(self.frame_history))):
            if ghost_frame is None:
                continue
            # Each older layer is more faded
            layer_weight = ghost_weight * (self.ghost_decay ** (i + 1))
            if layer_weight < 0.01:
                continue
            result = result * (1 - layer_weight) + ghost_frame.astype(np.float32) * layer_weight

        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_chromatic_aberration(self, frame: np.ndarray) -> np.ndarray:
        """Apply chromatic aberration (RGB channel offset)."""
        h, w, _ = frame.shape

        # Calculate pixel offset based on strength
        offset = int(max(1, w * self.chromatic_strength))

        result = frame.copy()

        # Offset red channel to the right
        result[:, offset:, 0] = frame[:, :-offset, 0]
        result[:, :offset, 0] = frame[:, 0:1, 0]  # Fill edge

        # Offset blue channel to the left
        result[:, :-offset, 2] = frame[:, offset:, 2]
        result[:, -offset:, 2] = frame[:, -1:, 2]  # Fill edge

        # Green stays centered

        return result

    def _apply_color_bleed(self, frame: np.ndarray) -> np.ndarray:
        """Apply color bleeding/smearing effect."""
        # Simple horizontal color smear using weighted average
        h, w, c = frame.shape
        result = frame.astype(np.float32)

        # Create slightly offset versions and blend
        for offset in [1, 2]:
            weight = self.color_bleed / offset
            if offset < w:
                # Blend with left-shifted version
                left_shift = np.zeros_like(result)
                left_shift[:, :-offset] = result[:, offset:]
                left_shift[:, -offset:] = result[:, -1:]

                # Blend with right-shifted version
                right_shift = np.zeros_like(result)
                right_shift[:, offset:] = result[:, :-offset]
                right_shift[:, :offset] = result[:, 0:1]

                result = result * (1 - weight) + (left_shift + right_shift) * (weight / 2)

        return np.clip(result, 0, 255).astype(np.uint8)

    def set_crossfade_frames(self, frames: int):
        """Set number of frames to crossfade over."""
        self.crossfade_frames = max(1, frames)

    def set_ghost_decay(self, decay: float):
        """Set ghost trail decay (0 = instant, 1 = no decay)."""
        self.ghost_decay = max(0.0, min(1.0, decay))

    def set_chromatic_strength(self, strength: float):
        """Set chromatic aberration strength."""
        self.chromatic_strength = max(0.0, min(0.1, strength))

    def set_color_bleed(self, bleed: float):
        """Set color bleeding amount."""
        self.color_bleed = max(0.0, min(0.5, bleed))

    def reset(self):
        """Reset all state."""
        self.frame_history.clear()
        self.prev_sd_frame = None
        self.curr_sd_frame = None
        self.blend_progress = 1.0
        self.prev_frame_for_motion = None
