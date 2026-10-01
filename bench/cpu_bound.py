#!/usr/bin/env python3
"""Is the MPS UNet CPU-dispatch-bound? Compare enqueue-only time vs synced wall time.

    python bench/cpu_bound.py --sizes 512,384,256
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="512,384,256")
    a = ap.parse_args()
    eng = tt.create_engine(width=512, height=512, morph=0, depth_graft=0)
    emb = eng._embed("test prompt")
    t = torch.tensor(499.0, device=eng.device)
    for S in [int(s) for s in a.sizes.split(",")]:
        eng._attn_ref["hw"] = (S // 8, S // 8)
        z = torch.randn(1, 4, S // 8, S // 8, device=eng.device, dtype=eng.dtype)
        cache = {}
        with torch.inference_mode():
            for reuse in (False, True):
                f = lambda: tt.lean_unet_forward(eng.unet, z, t, emb, cache=cache, reuse=reuse)  # noqa: E731
                for _ in range(3):
                    f()
                torch.mps.synchronize()
                with bench_lock():
                    enq, wall = [], []
                    for _ in range(10):
                        torch.mps.synchronize()
                        t0 = time.perf_counter()
                        f()
                        t1 = time.perf_counter()
                        torch.mps.synchronize()
                        t2 = time.perf_counter()
                        enq.append((t1 - t0) * 1000)
                        wall.append((t2 - t0) * 1000)
                print(f"| {S} | {'cheap' if reuse else 'full'} | enqueue {sum(enq)/len(enq):.1f} ms | wall {sum(wall)/len(wall):.1f} ms |", flush=True)


if __name__ == "__main__":
    main()
