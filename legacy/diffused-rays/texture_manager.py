"""Texture manager for SD-animated wall textures."""

import threading
from queue import Queue, Empty
from typing import Optional, Dict, Tuple

import numpy as np

import config


class TextureManager:
    """
    Manages wall textures as a single atlas that gets stylized by SD.

    The atlas is a 256x128 image containing 8 textures (64x64 each) in a 4x2 grid.
    This allows all textures to be stylized together in one SD pass.
    """

    TILE_SIZE = 64
    ATLAS_WIDTH = 256   # 4 tiles
    ATLAS_HEIGHT = 128  # 2 tiles

    def __init__(self):
        self.textures: Dict[int, np.ndarray] = {}  # wall_type -> 64x64x3 array
        self.base_atlas: Optional[np.ndarray] = None

    def _get_tile_position(self, wall_type: int) -> Tuple[int, int]:
        """Get (x, y) position of a wall type in the atlas."""
        idx = wall_type - 1  # wall_type is 1-indexed
        x = (idx % 4) * self.TILE_SIZE
        y = (idx // 4) * self.TILE_SIZE
        return x, y

    def _generate_base_pattern(self, base_color: Tuple[int, int, int]) -> np.ndarray:
        """Generate a base pattern tile with noise to give SD something to work with."""
        tile = np.zeros((self.TILE_SIZE, self.TILE_SIZE, 3), dtype=np.uint8)

        # Fill with base color
        tile[:, :] = base_color

        # Add noise/variation for texture
        noise = np.random.randint(-30, 30, (self.TILE_SIZE, self.TILE_SIZE, 3))
        tile = np.clip(tile.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add some simple brick/stone pattern
        for y in range(0, self.TILE_SIZE, 16):
            # Horizontal lines
            tile[y:y+2, :] = np.clip(tile[y:y+2].astype(np.int16) - 40, 0, 255).astype(np.uint8)

        for x in range(0, self.TILE_SIZE, 32):
            # Vertical lines (offset every other row)
            for y in range(0, self.TILE_SIZE, 32):
                offset = 16 if (y // 16) % 2 else 0
                x_pos = (x + offset) % self.TILE_SIZE
                tile[y:y+16, x_pos:x_pos+2] = np.clip(
                    tile[y:y+16, x_pos:x_pos+2].astype(np.int16) - 40, 0, 255
                ).astype(np.uint8)

        return tile

    def create_base_atlas(self) -> np.ndarray:
        """Create the base atlas with patterns for all wall types."""
        atlas = np.zeros((self.ATLAS_HEIGHT, self.ATLAS_WIDTH, 3), dtype=np.uint8)

        for wall_type in range(1, 9):
            x, y = self._get_tile_position(wall_type)
            base_color = config.WALL_COLORS.get(wall_type, config.DEFAULT_WALL_COLOR)
            atlas[y:y+self.TILE_SIZE, x:x+self.TILE_SIZE] = self._generate_base_pattern(base_color)

        self.base_atlas = atlas
        return atlas

    def split_atlas(self, styled_atlas: np.ndarray):
        """Split a styled atlas back into individual textures."""
        # Handle size mismatch (SD might return different size)
        if styled_atlas.shape[:2] != (self.ATLAS_HEIGHT, self.ATLAS_WIDTH):
            from PIL import Image
            img = Image.fromarray(styled_atlas)
            img = img.resize((self.ATLAS_WIDTH, self.ATLAS_HEIGHT), Image.Resampling.LANCZOS)
            styled_atlas = np.array(img)

        for wall_type in range(1, 9):
            x, y = self._get_tile_position(wall_type)
            self.textures[wall_type] = styled_atlas[y:y+self.TILE_SIZE, x:x+self.TILE_SIZE].copy()

    def sample(self, wall_type: int, u: float, v: float) -> Tuple[int, int, int]:
        """
        Sample texture at UV coordinates.

        Args:
            wall_type: Wall type (1-8)
            u: Horizontal coordinate (0-1)
            v: Vertical coordinate (0-1)

        Returns:
            RGB tuple
        """
        tex = self.textures.get(wall_type)
        if tex is None:
            # Fallback to solid color
            return config.WALL_COLORS.get(wall_type, config.DEFAULT_WALL_COLOR)

        # Nearest neighbor sampling (fast)
        tx = int(u * (self.TILE_SIZE - 1)) % self.TILE_SIZE
        ty = int(v * (self.TILE_SIZE - 1)) % self.TILE_SIZE
        return tuple(tex[ty, tx])

    def has_textures(self) -> bool:
        """Check if textures have been generated."""
        return len(self.textures) > 0


class AsyncTextureStylizer:
    """
    Async texture atlas stylizer.

    Runs SD img2img on the texture atlas in a background thread,
    similar to AsyncStylizer but for the texture atlas.
    """

    def __init__(self):
        self.input_queue: Queue = Queue(maxsize=1)
        self.output_queue: Queue = Queue(maxsize=1)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frames_processed = 0

    def start(self):
        """Start the background processing thread."""
        if self.running:
            return

        # Pre-load the SD pipeline
        from stylizer import load_pipeline
        load_pipeline()

        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the background processing thread."""
        self.running = False
        if self.thread:
            try:
                self.input_queue.get_nowait()
            except Empty:
                pass
            self.thread.join(timeout=1.0)
            self.thread = None

    def _process_loop(self):
        """Background thread that stylizes texture atlases."""
        from stylizer import stylize_frame
        import time

        while self.running:
            try:
                atlas, prompt = self.input_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                # Stylize the atlas
                start = time.time()
                styled = stylize_frame(atlas, prompt=prompt)
                elapsed = time.time() - start
                print(f"Texture stylize took {elapsed:.2f}s")

                # Validate output
                if styled is None or (styled == 0).all():
                    print("Warning: texture stylization returned black/empty")
                    continue

                # Put result (replace old if exists)
                try:
                    self.output_queue.get_nowait()
                except Empty:
                    pass
                self.output_queue.put(styled)
                self.frames_processed += 1

            except Exception as e:
                print(f"Texture stylizer error: {e}")
                import traceback
                traceback.print_exc()

    def submit(self, atlas: np.ndarray, prompt: str):
        """Submit an atlas for stylization. Non-blocking, drops old pending."""
        try:
            try:
                self.input_queue.get_nowait()
            except Empty:
                pass
            self.input_queue.put_nowait((atlas.copy(), prompt))
        except:
            pass

    def get_result(self) -> Optional[np.ndarray]:
        """Get the latest styled atlas if available. Non-blocking."""
        try:
            return self.output_queue.get_nowait()
        except Empty:
            return None
