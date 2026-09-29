#!/usr/bin/env python3
"""Do coreml_turbo (ANE) and torch_turbo (MPS) agree for identical (image, prompt, strength, seed)?
Matters for a server pool that alternates frames between them. Writes a sheet
(rows: input, torch_turbo, coreml_turbo, |diff|x4) to bench/samples/coreml_vs_torch_<S>.jpg.

  ~/hypnagogia-cache/venv-run/bin/python bench/coreml/pool_consistency.py --size 512
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(BENCH))
sys.path.insert(0, BENCH)
import scenes  # noqa: E402
from server import engines  # noqa: E402

PROMPTS = [
    "vast drowned gothic cathedral, bioluminescent, volumetric light, oil painting",
    "overgrown sunken garden temple, moss and flowers, god rays, lush fantasy concept art",
    "neon bathhouse, wet tiles, cyan and magenta glow, cinematic, moody",
]

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--strength", type=float, default=0.5)
a = ap.parse_args()
S = a.size
stills = scenes.test_set(S, S)
outs = {}
for name in ("torch_turbo", "coreml_turbo"):
    eng = engines.load(name, width=S, height=S)
    eng.warmup()
    outs[name] = [eng.process(im, PROMPTS[i], a.strength, 1234) for i, im in enumerate(stills)]
    print(name, getattr(eng, "model", "?"), flush=True)
    del eng
rows = [stills, outs["torch_turbo"], outs["coreml_turbo"]]
diffs = []
for x, y in zip(outs["torch_turbo"], outs["coreml_turbo"]):
    d = np.abs(x.astype(np.float32) - y.astype(np.float32))
    mse = float((d ** 2).mean())
    diffs.append((float(d.mean()), 10 * np.log10(255 ** 2 / max(mse, 1e-9))))
    rows.append(None)
rows[3] = [np.clip(np.abs(x.astype(np.int16) - y.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
           for x, y in zip(outs["torch_turbo"], outs["coreml_turbo"])]
rows = rows[:4]
cell = 256
sheet = np.concatenate([np.concatenate([np.asarray(Image.fromarray(im).resize((cell, cell), Image.LANCZOS))
                                        for im in r], 1) for r in rows], 0)
out = os.path.join(BENCH, "samples", f"coreml_vs_torch_{S}.jpg")
Image.fromarray(sheet).save(out, quality=85)
for i, (mad, psnr) in enumerate(diffs):
    print(f"still {i}: mean |diff| {mad:.1f}/255, PSNR {psnr:.1f} dB")
print("sheet ->", out)
