#!/usr/bin/env python3
"""Does ANE diffusion overlap with GPU work?  Two *processes*:

  A: Core ML SD-Turbo UNet on the Neural Engine (CPU_AND_NE)
  B: a GPU load - torch-MPS SD-Turbo UNet fp16 (the competing engine) or the
     same Core ML UNet on CPU_AND_GPU (--gpu coreml)

Each runs alone, then both at once; reports per-process throughput.  Holds the
bench lock for the whole timed section.

  ~/hypnagogia-cache/venv-coreml/bin/python bench/coreml/concurrency.py --size 384
"""
import argparse
import contextlib
import fcntl
import json
import multiprocessing as mp
import os
import time

LOCK = os.path.expanduser("~/hypnagogia-cache/bench.lock")
C = os.path.expanduser("~/hypnagogia-cache/coreml")


@contextlib.contextmanager
def bench_lock():
    f = open(LOCK, "a+")
    fcntl.flock(f, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def worker(kind, size, go, stop, q):
    import numpy as np
    if kind in ("ane", "coreml-gpu"):
        import coremltools as ct
        cu = ct.ComputeUnit.CPU_AND_NE if kind == "ane" else ct.ComputeUnit.CPU_AND_GPU
        m = ct.models.CompiledMLModel(f"{C}/sdturbo_unet_{size}_ane.mlmodelc", compute_units=cu)
        r = np.load(f"{C}/sdturbo_unet_{size}_ane_ref.npz")
        feeds = {"sample": r["sample"], "timestep": r["timestep"], "encoder_hidden_states": r["ehs"]}

        def step():
            m.predict(feeds)
    else:  # torch mps
        import torch
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet", variant="fp16",
                                                    torch_dtype=torch.float16).to("mps").eval()
        lat = size // 8
        x = torch.randn(1, 4, lat, lat, device="mps", dtype=torch.float16)
        t = torch.tensor([499.0], device="mps")
        e = torch.randn(1, 77, 1024, device="mps", dtype=torch.float16)

        def step():
            with torch.inference_mode():
                unet(x, t, e, return_dict=False)
            torch.mps.synchronize()
    for _ in range(3):
        step()
    q.put(("ready", kind))
    while True:
        cmd = go.get()
        if cmd is None:
            break
        dur = cmd
        n = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < dur:
            step()
            n += 1
        dt = time.perf_counter() - t0
        q.put((kind, n / dt, dt / n * 1000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--gpu", default="torch", choices=["torch", "coreml"])
    ap.add_argument("--dur", type=float, default=8.0)
    a = ap.parse_args()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    gk = "torch-mps" if a.gpu == "torch" else "coreml-gpu"
    kinds = ["ane", gk]
    gos = {k: ctx.Queue() for k in kinds}
    procs = {k: ctx.Process(target=worker, args=(k, a.size, gos[k], None, q)) for k in kinds}
    for p in procs.values():
        p.start()
    for _ in kinds:
        print("ready:", q.get(), flush=True)
    res = {}
    with bench_lock():
        for k in kinds:
            gos[k].put(a.dur)
            kind, fps, ms = q.get()
            res[f"{kind}_alone_fps"] = round(fps, 2)
            res[f"{kind}_alone_ms"] = round(ms, 1)
        for k in kinds:
            gos[k].put(a.dur)
        for _ in kinds:
            kind, fps, ms = q.get()
            res[f"{kind}_concurrent_fps"] = round(fps, 2)
            res[f"{kind}_concurrent_ms"] = round(ms, 1)
    for k in kinds:
        gos[k].put(None)
    for p in procs.values():
        p.join()
    res["size"] = a.size
    res["combined_fps"] = round(res[f"ane_concurrent_fps"] + res[f"{gk}_concurrent_fps"], 2)
    print(json.dumps(res), flush=True)
    with open(os.path.join(C, "concurrency.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
