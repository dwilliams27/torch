"""Stable Diffusion img2img stylizer using SD Turbo with async support."""

import warnings
import threading
from typing import Optional
from queue import Queue, Empty

import numpy as np
from PIL import Image

import config

# Lazy imports for torch/diffusers to avoid slow startup
_pipe = None
_device = None

# Processing size - 256x256 is optimal for speed (~3 FPS vs ~1 FPS at 512)
SD_PROCESS_SIZE = 256


def get_device() -> str:
    """Get the best available device for inference."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        warnings.warn("MPS/CUDA not available, falling back to CPU. Performance will be slow.")
        return "cpu"


def load_pipeline():
    """Load and cache the SD Turbo pipeline."""
    global _pipe, _device

    if _pipe is not None:
        return _pipe

    import torch
    from diffusers import AutoPipelineForImage2Image

    _device = get_device()
    print(f"Loading SD Turbo on {_device}...")

    _pipe = AutoPipelineForImage2Image.from_pretrained(
        config.SD_MODEL,
        torch_dtype=torch.float16,
    )
    _pipe = _pipe.to(_device)

    # Disable progress bar for faster inference
    _pipe.set_progress_bar_config(disable=True)

    print("Pipeline loaded successfully!")
    return _pipe


def stylize_frame(
    frame: np.ndarray,
    prompt: Optional[str] = None,
    strength: Optional[float] = None,
    num_steps: Optional[int] = None,
) -> np.ndarray:
    """
    Apply Stable Diffusion img2img to a frame (synchronous).

    Args:
        frame: Input numpy array of shape (H, W, 3) with RGB uint8 values.
        prompt: Text prompt for generation. Uses config default if None.
        strength: How much to transform the image (0-1). Uses config default if None.
        num_steps: Number of inference steps. Uses config default if None.

    Returns:
        Stylized numpy array of same shape as input.
    """
    import torch

    pipe = load_pipeline()

    # Use defaults from config if not specified
    if prompt is None:
        prompt = config.SD_PROMPT
    if strength is None:
        strength = config.SD_STRENGTH
    if num_steps is None:
        num_steps = config.SD_NUM_STEPS

    # Store original size
    orig_h, orig_w = frame.shape[:2]

    # Convert numpy to PIL and resize to processing size
    input_image = Image.fromarray(frame).resize(
        (SD_PROCESS_SIZE, SD_PROCESS_SIZE),
        Image.Resampling.LANCZOS
    )

    # Run inference
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=input_image,
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=config.SD_GUIDANCE_SCALE,
        ).images[0]

    # Resize back to original size
    result = result.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

    # Convert back to numpy
    output_frame = np.array(result)

    # Ensure MPS operations are complete
    if _device == "mps":
        torch.mps.synchronize()

    return output_frame


class AsyncStylizer:
    """
    Async frame stylizer that processes frames in a background thread.

    This allows the game to continue rendering while SD processes the previous frame,
    effectively hiding latency and improving perceived smoothness.
    """

    def __init__(self):
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)
        self.running = False
        self.thread = None
        self.last_output = None
        self.frames_processed = 0

    def start(self):
        """Start the async processing thread."""
        if self.running:
            return

        # Pre-load the pipeline
        load_pipeline()

        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the async processing thread."""
        self.running = False
        if self.thread:
            # Clear queues to unblock thread
            try:
                self.input_queue.get_nowait()
            except Empty:
                pass
            self.thread.join(timeout=1.0)
            self.thread = None

    def _process_loop(self):
        """Background thread that processes frames."""
        import torch

        while self.running:
            try:
                # Wait for input (frame, prompt) tuple
                frame, prompt = self.input_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                # Process the frame with specified prompt
                output = stylize_frame(frame, prompt=prompt)

                # Put result in output queue (replace old if exists)
                try:
                    self.output_queue.get_nowait()
                except Empty:
                    pass
                self.output_queue.put(output)
                self.frames_processed += 1

            except Exception as e:
                print(f"Async stylizer error: {e}")

    def submit_frame(self, frame: np.ndarray, prompt: str = None):
        """Submit a frame for processing. Non-blocking, drops old frames."""
        if prompt is None:
            prompt = config.SD_PROMPT
        try:
            # Remove old pending frame if any
            try:
                self.input_queue.get_nowait()
            except Empty:
                pass
            # Add new frame with prompt
            self.input_queue.put_nowait((frame, prompt))
        except:
            pass  # Queue full, skip this frame

    def get_result(self) -> Optional[np.ndarray]:
        """Get the latest processed frame. Non-blocking, returns None if not ready."""
        try:
            self.last_output = self.output_queue.get_nowait()
        except Empty:
            pass
        return self.last_output

    def get_latest(self, fallback: np.ndarray) -> np.ndarray:
        """Get latest result or fallback if none available."""
        result = self.get_result()
        return result if result is not None else fallback


def test_stylizer():
    """Test the stylizer with a synthetic image."""
    import time

    # Create a simple test image (gradient)
    print("Creating test image...")
    test_image = np.zeros((config.RENDER_HEIGHT, config.RENDER_WIDTH, 3), dtype=np.uint8)

    # Red to blue gradient
    for x in range(config.RENDER_WIDTH):
        r = int(255 * (1 - x / config.RENDER_WIDTH))
        b = int(255 * x / config.RENDER_WIDTH)
        test_image[:, x] = [r, 128, b]

    # Add some structure
    test_image[40:88, 40:88] = [100, 100, 100]  # Gray square
    test_image[50:78, 50:78] = [50, 50, 50]  # Darker inner square

    # Save input
    input_pil = Image.fromarray(test_image)
    input_pil.save("test_input.png")
    print("Saved test_input.png")

    # Test synchronous
    print("\nTesting synchronous stylizer...")
    load_pipeline()  # Pre-load
    _ = stylize_frame(test_image)  # Warm up

    times = []
    for i in range(5):
        start_time = time.time()
        output = stylize_frame(test_image)
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")

    avg = sum(times) / len(times)
    print(f"  Average: {avg:.2f}s ({1/avg:.1f} FPS)")

    # Save output
    output_pil = Image.fromarray(output)
    output_pil.save("test_output.png")
    print("Saved test_output.png")

    # Test async
    print("\nTesting async stylizer...")
    async_stylizer = AsyncStylizer()
    async_stylizer.start()

    # Submit frames and measure throughput
    start_time = time.time()
    for i in range(10):
        async_stylizer.submit_frame(test_image)
        time.sleep(0.05)  # Simulate game loop doing other work

    # Wait for processing to complete
    time.sleep(2.0)
    elapsed = time.time() - start_time
    frames = async_stylizer.frames_processed
    print(f"  Processed {frames} frames in {elapsed:.1f}s")
    print(f"  Throughput: {frames/elapsed:.1f} FPS")

    async_stylizer.stop()

    return avg


if __name__ == "__main__":
    test_stylizer()
