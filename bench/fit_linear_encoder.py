#!/usr/bin/env python3
"""Fit a linear 8x8-patch -> 4-channel latent encoder that imitates the TAESD encoder.

The img2img path immediately drowns the latent in noise (sqrt(1-a_t) ~ 0.85 at strength 0.5),
so the encoder only has to get low/mid frequencies right; a least-squares linear map from
pixel-unshuffled 8x8x3 patches (192 -> 4) costs ~0 ms instead of 12-20 ms.

    python bench/fit_linear_encoder.py      # writes server/engines/taesd_linear_enc.npz
Training data: procedural naves (3 palettes, walking camera) + their SD-Turbo stylizations
(covers the painted-feedback distribution the client sends).
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import scenes  # noqa: E402
from bench_engines import PROMPTS  # noqa: E402


def main():
    import torch
    import torch.nn.functional as F

    from server.engines import torch_turbo as tt

    eng = tt.create_engine(model="sd-turbo", width=512, height=512, morph=0, vae="taesd", deepcache=0)
    dev = eng.device
    imgs = []
    for pal in ("blue", "amber", "violet"):
        imgs += scenes.walk(12, 512, 512, palette=pal)
    imgs += scenes.test_set(512, 512)
    styl = []
    for i, im in enumerate(imgs[::2]):
        styl.append(eng.process(im, PROMPTS[i % 3], 0.3 + 0.4 * ((i * 7) % 5) / 4, i))
    imgs += styl
    X, Y = [], []
    with torch.inference_mode():
        for im in imgs:
            x = torch.from_numpy(im).to(dev).permute(2, 0, 1)[None].float().div(127.5).sub(1)
            z = eng.vae.encode(x.to(eng.dtype)).latents.float()            # (1,4,64,64)
            p = F.pixel_unshuffle(x, 8)                                    # (1,192,64,64)
            X.append(p.flatten(2)[0].T.cpu())
            Y.append(z.flatten(2)[0].T.cpu())
    X = torch.cat(X).double()
    Y = torch.cat(Y).double()
    Xb = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1)
    # ridge-regularized least squares
    A = Xb.T @ Xb + 1e-3 * torch.eye(Xb.shape[1], dtype=X.dtype)
    Wb = torch.linalg.solve(A, Xb.T @ Y)                                   # (193,4)
    pred = Xb @ Wb
    r2 = 1 - ((pred - Y) ** 2).mean(0) / Y.var(0)
    print("samples", len(X), "R2 per latent channel:", [round(float(v), 3) for v in r2])
    out = os.path.join(ROOT, "server", "engines", "taesd_linear_enc.npz")
    np.savez(out, weight=Wb[:-1].T.float().numpy(), bias=Wb[-1].float().numpy())  # weight (4,192)
    print("wrote", out)


if __name__ == "__main__":
    main()
