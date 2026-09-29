"""CPU-only fake diffusion for client development (no GPU, no model download).

Looks vaguely like a painterly img2img pass so the dream-painting pipeline can be
judged visually:  seeded low-frequency warp ("brush wobble") -> median-filter
blotching -> prompt-derived gradient map (each zone prompt gets its own palette)
-> soft posterize -> ink contour lines -> mix with the input by ``strength``.

Deterministic for identical (image, prompt, strength, seed) like a real engine
with cached noise, and pads its runtime up to ``latency_ms`` to mimic the GPU.
"""
from __future__ import annotations

import colorsys
import hashlib
import time

import numpy as np
from PIL import Image, ImageFilter


class MockEngine:
    def __init__(self, width: int = 512, height: int = 512, latency_ms: float = 120.0,
                 jitter_ms: float = 15.0, model: str | None = None, **_ignored):
        self.name = "mock"
        self.model = model or "mock-painterly (CPU, no diffusion)"
        self.width = int(width)
        self.height = int(height)
        self.device = "cpu"
        self.latency_ms = float(latency_ms)
        self.jitter_ms = float(jitter_ms)
        self._warp_cache: dict[tuple, tuple] = {}
        self._palette_cache: dict[str, np.ndarray] = {}
        self._rng = np.random.default_rng(1234)

    # -- cached "noise" --------------------------------------------------------
    def _warp(self, seed: int, h: int, w: int):
        key = (seed, h, w)
        if key not in self._warp_cache:
            rng = np.random.default_rng(seed & 0xFFFFFFFF)
            fields = []
            for _ in range(2):
                coarse = rng.standard_normal((max(2, h // 48), max(2, w // 48))).astype(np.float32)
                img = Image.fromarray(coarse, mode="F").resize((w, h), Image.BICUBIC)
                fields.append(np.asarray(img, dtype=np.float32))
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            self._warp_cache[key] = (yy, xx, fields[0], fields[1])
            if len(self._warp_cache) > 8:
                self._warp_cache.pop(next(iter(self._warp_cache)))
        return self._warp_cache[key]

    def _palette(self, prompt: str) -> np.ndarray:
        """256-entry RGB gradient map derived from the prompt text."""
        pal = self._palette_cache.get(prompt)
        if pal is None:
            hsh = hashlib.sha1(prompt.encode("utf-8", "ignore")).digest()
            h0 = hsh[0] / 255.0
            spread = 0.12 + hsh[1] / 255.0 * 0.35
            stops = [
                colorsys.hsv_to_rgb((h0 + 0.55) % 1, 0.75, 0.10),           # deep shadow
                colorsys.hsv_to_rgb((h0 + spread) % 1, 0.70, 0.45),         # mid
                colorsys.hsv_to_rgb(h0, 0.55, 0.85),                        # light
                colorsys.hsv_to_rgb((h0 - spread * 0.5) % 1, 0.18, 1.0),    # highlight
            ]
            stops = np.array(stops, dtype=np.float32) * 255.0
            t = np.linspace(0, len(stops) - 1, 256)
            i = np.minimum(t.astype(int), len(stops) - 2)
            f = (t - i)[:, None]
            f = f * f * (3 - 2 * f)
            pal = stops[i] * (1 - f) + stops[i + 1] * f
            self._palette_cache[prompt] = pal.astype(np.float32)
            if len(self._palette_cache) > 64:
                self._palette_cache.pop(next(iter(self._palette_cache)))
        return pal

    # -- engine API --------------------------------------------------------------
    def warmup(self) -> None:
        img = np.zeros((self.height, self.width, 3), np.uint8)
        self.process(img, "warmup", 0.5, 0)

    def process(self, image: np.ndarray, prompt: str, strength: float, seed: int,
                negative: str | None = None) -> np.ndarray:
        t0 = time.perf_counter()
        h, w, _ = image.shape
        s = float(np.clip(strength, 0.0, 1.0))

        # 1. seeded brush wobble (identical for identical seed -> temporally stable)
        yy, xx, fy, fx = self._warp(int(seed), h, w)
        amp = 1.5 + 7.0 * s
        sy = np.clip(yy + fy * amp, 0, h - 1).astype(np.int32)
        sx = np.clip(xx + fx * amp, 0, w - 1).astype(np.int32)
        warped = image[sy, sx]

        # 2. paint blotches: median at half res, upsampled soft
        small = Image.fromarray(warped).resize((w // 2, h // 2), Image.BILINEAR)
        small = small.filter(ImageFilter.MedianFilter(5))
        paint = np.asarray(small.resize((w, h), Image.BICUBIC), dtype=np.float32)

        # 3. prompt gradient map on luminance, mixed with the original hue
        lum = paint @ np.array([0.299, 0.587, 0.114], np.float32)
        lum = np.clip((lum - 8.0) * 1.15, 0, 255)
        graded = self._palette(prompt or "")[lum.astype(np.uint8)]
        dream = graded * 0.62 + paint * 0.38

        # 4. soft posterize (quantize then blend back a bit)
        levels = 7.0
        post = np.round(dream / 255.0 * levels) / levels * 255.0
        dream = dream * 0.45 + post * 0.55

        # 5. ink contours from luminance gradient
        gy = np.abs(np.diff(lum, axis=0, prepend=lum[:1]))
        gx = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
        edge = np.clip((gx + gy - 18.0) / 40.0, 0, 1)[..., None]
        dream = dream * (1.0 - 0.55 * edge)

        # 6. glow: blurred highlights added back
        hi = np.clip(dream - 170.0, 0, None).astype(np.uint8)
        glow = np.asarray(Image.fromarray(hi).filter(ImageFilter.GaussianBlur(6)), np.float32)
        dream = dream + glow * 0.8

        out = image.astype(np.float32) * (1.0 - s) + dream * s
        out = np.clip(out, 0, 255).astype(np.uint8)

        # 7. pretend to be a GPU
        target = self.latency_ms + (self._rng.uniform(-1, 1) * self.jitter_ms if self.jitter_ms else 0.0)
        remaining = target / 1000.0 - (time.perf_counter() - t0)
        if remaining > 0:
            _precise_sleep(remaining)
        return out


def _measure_sleep_slop() -> float:
    worst = 0.0
    for _ in range(3):
        t = time.perf_counter()
        time.sleep(0.002)
        worst = max(worst, time.perf_counter() - t - 0.002)
    return worst


_SLOP = None


def _precise_sleep(dt: float) -> None:
    """time.sleep can overshoot by 20-150 ms when macOS coalesces timers (seen on the mini for
    processes started over ssh), so sleep coarsely and spin-yield the tail."""
    global _SLOP
    if _SLOP is None:
        _SLOP = _measure_sleep_slop()
    end = time.perf_counter() + dt
    margin = 0.15 if _SLOP > 0.001 else 0.002  # observed overshoot is far larger than measured slop
    while True:
        rem = end - time.perf_counter()
        if rem <= 0:
            return
        if rem > margin:
            time.sleep(rem - margin)
        else:
            time.sleep(0)


def create_engine(**cfg) -> MockEngine:
    return MockEngine(**cfg)
