#!/usr/bin/env python3
"""End-to-end check of server/engines/coreml_turbo.py on the mini: e2e timing of
engine.process() (walking-camera sequence, under the bench lock) + a quality
contact sheet (rows: input, then strengths) -> bench/samples/coreml_<model>_<size>.jpg

  ~/hypnagogia-cache/venv-coreml/bin/python bench/coreml/sample.py --model sdxs --sizes 512 384
"""
import argparse
import contextlib
import fcntl
import json
import os
import statistics
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(BENCH)
sys.path.insert(0, ROOT)
sys.path.insert(0, BENCH)
import scenes  # noqa: E402  (shared procedural test scenes)
from server.engines import coreml_turbo  # noqa: E402

PROMPTS = [
    "vast drowned gothic cathedral, bioluminescent, volumetric light, oil painting",
    "overgrown sunken garden temple, moss and flowers, god rays, lush fantasy concept art",
    "neon bathhouse, wet tiles, cyan and magenta glow, cinematic, moody",
]
LOCK = os.path.expanduser("~/hypnagogia-cache/bench.lock")


@contextlib.contextmanager
def bench_lock():
    f = open(LOCK, "a+")
    fcntl.flock(f, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sdxs")
    ap.add_argument("--sizes", type=int, nargs="+", default=[512])
    ap.add_argument("--cu", default="NE")
    ap.add_argument("--vae-cu", default=None)
    ap.add_argument("--strengths", default="0.3,0.5,0.7")
    ap.add_argument("--iters", type=int, default=48)
    ap.add_argument("--cell", type=int, default=320)
    ap.add_argument("--tag", default="")
    ap.add_argument("--variant", default=None, help='UNet variant suffix, e.g. "" or "_kv2" (default: engine auto)')
    a = ap.parse_args()
    strengths = [float(s) for s in a.strengths.split(",")]
    for S in a.sizes:
        t = time.time()
        eng = coreml_turbo.create_engine(model=a.model, width=S, height=S, compute_units=a.cu,
                                         vae_compute_units=a.vae_cu, variant=a.variant)
        t_create = time.time() - t
        t = time.time()
        eng.warmup()
        t_warm = time.time() - t
        frames = scenes.walk(24, S, S)
        for p in PROMPTS:
            eng.process(frames[0], p, 0.5, 1234)  # embed prompts outside the timed loop
        stage = {k: [] for k in ("enc", "unet", "dec", "total")}
        with bench_lock():
            ts = []
            for i in range(a.iters):
                f = frames[i % len(frames)]
                t = time.perf_counter()
                eng.process(f, PROMPTS[0], 0.5, 1234)
                ts.append((time.perf_counter() - t) * 1000)
                for k in stage:
                    stage[k].append(eng.last_timings[k])
        med = statistics.median(ts)
        r = {"engine": eng.name, "variant": eng.variant, "size": S, "cu": a.cu, "vae_cu": a.vae_cu or a.cu,
             "create_s": round(t_create, 1), "warmup_s": round(t_warm, 1),
             "e2e_ms": round(med, 1), "fps": round(1000 / med, 1),
             **{f"{k}_ms": round(statistics.median(v), 1) for k, v in stage.items()}}
        print(json.dumps(r), flush=True)
        with open(os.path.expanduser("~/hypnagogia-cache/coreml/e2e.jsonl"), "a") as fh:
            fh.write(json.dumps(r) + "\n")
        # contact sheet
        stills = scenes.test_set(S, S)
        cell = a.cell
        rows = [np.concatenate([np.asarray(Image.fromarray(im).resize((cell, cell), Image.LANCZOS))
                                for im in stills], 1)]
        for s in strengths:
            outs = [eng.process(im, PROMPTS[j % len(PROMPTS)], s, 1234) for j, im in enumerate(stills)]
            rows.append(np.concatenate([np.asarray(Image.fromarray(o).resize((cell, cell), Image.LANCZOS))
                                        for o in outs], 1))
        os.makedirs(os.path.join(BENCH, "samples"), exist_ok=True)
        name = f"coreml_{a.model}_{S}{eng.variant}{a.tag}.jpg"
        Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(BENCH, "samples", name), quality=85)
        print(f"  sheet -> bench/samples/{name} (rows: input, strengths {strengths})", flush=True)
        del eng


if __name__ == "__main__":
    main()
