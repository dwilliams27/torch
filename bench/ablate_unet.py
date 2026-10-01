#!/usr/bin/env python3
"""UNet-only ablations on MPS: memory format, torch.compile, attention variants, resolution.

    python bench/ablate_unet.py --model sd-turbo --sizes 512,384 --variants eager,cl,todo2,noattn64

Timed sections hold ~/hypnagogia-cache/bench.lock (see bench_engines.py).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from bench_engines import bench_lock  # noqa: E402
from server.engines import torch_turbo as tt  # noqa: E402


def timeit(fn, iters, dev):
    fn()
    tt._sync(dev)
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    tt._sync(dev)
    return (time.perf_counter() - t) / iters * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sd-turbo")
    ap.add_argument("--sizes", default="512,384")
    ap.add_argument("--variants", default="eager,cl,todo2,todo2_32,noattn64")
    ap.add_argument("--iters", type=int, default=15)
    a = ap.parse_args()
    eng = tt.create_engine(model=a.model, width=512, height=512, morph=0, depth_graft=0)
    dev, dt = eng.device, eng.dtype
    unet = eng.unet
    emb = eng._embed("vast drowned gothic cathedral, bioluminescent, volumetric light, oil painting")
    t = torch.tensor(499.0, device=dev)
    default_procs = unet.attn_processors
    for S in [int(s) for s in a.sizes.split(",")]:
        z = torch.randn(1, 4, S // 8, S // 8, device=dev, dtype=dt)
        for v in a.variants.split(","):
            unet.set_attn_processor(dict(default_procs))
            unet.to(memory_format=torch.contiguous_format)
            zz = z
            fwd = unet
            if v == "cl":
                unet.to(memory_format=torch.channels_last)
                zz = z.contiguous(memory_format=torch.channels_last)
            elif v.startswith("todo") or v == "noattn64":
                mode = "skip" if v == "noattn64" else "todo"
                levels = 2 if v.endswith("_32") else 1
                tt.install_fast_attention(unet, (S // 8, S // 8), mode=mode, levels=levels)
            elif v == "compile":
                fwd = torch.compile(unet)
            f = lambda: fwd(zz, t, encoder_hidden_states=emb, return_dict=False)[0]  # noqa: E731
            with torch.inference_mode():
                for _ in range(3):
                    f()
                tt._sync(dev)
                with bench_lock():
                    ms = timeit(f, a.iters, dev)
            print(f"| unet {a.model} | {S} | {v} | {ms:.1f} ms |", flush=True)
    unet.set_attn_processor(dict(default_procs))


if __name__ == "__main__":
    main()
