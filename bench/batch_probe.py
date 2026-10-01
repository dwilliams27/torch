#!/usr/bin/env python3
"""UNet time for one full pass at batch 1, 2 and 4: would the server gain by painting two or
four frames in one call? (M112)

    python bench/batch_probe.py [--size 512x320] [--out batch.json]

The UNet alone (`lean_unet_forward`: depth graft, fp16, ToDo attention; no DeepCache, no
cross-frame attention, no autoencoder), random latents, median of 12 after 3 warm-up calls,
synchronised, under the bench lock (`bench_engines.bench_lock`).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_engines import bench_lock, machine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="512x320")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import torch
    from server.engines import torch_turbo as tt
    W, H = (int(v) for v in a.size.split("x"))
    eng = tt.create_engine(model="sd-turbo", width=W, height=H, morph=0, deepcache=0, xframe=0)
    cond = eng._conditioning("a drowned gothic nave", 1)
    t_, _, _, _ = eng._timestep(0.66)
    res = {}
    lock = bench_lock()
    lock.__enter__()
    for b in (1, 2, 4):
        zin = torch.randn(b, 5 if eng.wants_depth else 4, H // 8, W // 8, device=eng.device, dtype=eng.dtype)
        c = cond.expand(b, -1, -1).contiguous()
        for _ in range(3):
            with torch.inference_mode():
                tt.lean_unet_forward(eng.unet, zin, t_, c)
        ts = []
        for _ in range(12):
            if eng.device == "mps":
                torch.mps.synchronize()
            s = time.perf_counter()
            with torch.inference_mode():
                tt.lean_unet_forward(eng.unet, zin, t_, c)
            if eng.device == "mps":
                torch.mps.synchronize()
            ts.append((time.perf_counter() - s) * 1000)
        res[b] = round(statistics.median(ts), 1)
        print(f"batch {b}: {res[b]:.1f} ms, {res[b] / b:.1f} ms a frame", flush=True)
    lock.__exit__(None, None, None)
    out = {"date": time.strftime("%Y-%m-%d"), "size": [W, H], "device": eng.device, "depth_graft": eng.depth_graft,
           "cfg": {"deepcache": 0, "xframe": 0, "unet_only": True}, "warmup": 3, "iters": 12, "machine": machine(),
           "torch": torch.__version__, "unet_ms": res, "per_frame_ms": {b: round(v / b, 1) for b, v in res.items()}}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
