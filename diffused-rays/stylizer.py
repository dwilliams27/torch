"""Stable Diffusion img2img stylizer using SDXL Turbo."""

import warnings
from typing import Optional

import numpy as np
from PIL import Image

import config

# Lazy imports for torch/diffusers to avoid slow startup
_pipe = None
_device = None


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
    """Load and cache the SDXL Turbo pipeline."""
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


# Processing size for SDXL Turbo (must be at least 512x512)
SD_PROCESS_SIZE = 512


def stylize_frame(
    frame: np.ndarray,
    prompt: Optional[str] = None,
    strength: Optional[float] = None,
    num_steps: Optional[int] = None,
) -> np.ndarray:
    """
    Apply Stable Diffusion img2img to a frame.

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

    # Convert numpy to PIL
    input_image = Image.fromarray(frame)

    # SDXL Turbo requires larger images (at least 512x512)
    # Upscale input to processing size
    input_image = input_image.resize(
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

    # Stylize
    print("Stylizing...")
    start_time = time.time()
    output = stylize_frame(test_image)
    elapsed = time.time() - start_time
    print(f"Stylization took {elapsed:.2f} seconds")

    # Save output
    output_pil = Image.fromarray(output)
    output_pil.save("test_output.png")
    print("Saved test_output.png")

    return elapsed


if __name__ == "__main__":
    test_stylizer()
