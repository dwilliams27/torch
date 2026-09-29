#!/usr/bin/env python3
"""Reproducible throughput + quality bench for HYPNAGOGIA diffusion engines.

    python bench/bench_engines.py --engine torch_turbo --model sdxs --sizes 512,384 --iters 40
    python bench/bench_engines.py --engine torch_turbo --model sd-turbo --sizes 512 --samples

Timing protocol
  * engine is created + warmed (per size) OUTSIDE the lock;
  * the timed sections take an exclusive fcntl.flock on ~/hypnagogia-cache/bench.lock
    (/tmp/hypnagogia-bench.lock if that dir doesn't exist; override: HYPNAGOGIA_BENCH_LOCK)
    so two engine engineers never time on a shared GPU;
  * "e2e" = wall time of process() over a 24-frame walking camera sequence (numpy uint8 in,
    numpy uint8 out, i.e. exactly what the server worker sees minus JPEG);
  * "stages" = a second pass with engine.profile=True (device sync after every stage),
    so stage sums are slightly larger than e2e.
Results are appended as JSON lines to bench/out/results.jsonl and printed as markdown rows.
`--samples` writes a small contact sheet jpg per (model,size) to bench/samples/.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import scenes  # noqa: E402

PROMPTS = [
    "vast drowned gothic cathedral, bioluminescent, volumetric light, oil painting",
    "overgrown sunken garden temple, moss and flowers, god rays, lush fantasy concept art",
    "neon bathhouse, wet tiles, cyan and magenta glow, cinematic, moody",
]


def machine() -> str:
    try:
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()) // 2**30
        return f"{platform.node().split('.')[0]} ({chip}, {mem}GB)"
    except Exception:
        return platform.node()


@contextlib.contextmanager
def bench_lock():
    path = os.path.expanduser(os.environ.get("HYPNAGOGIA_BENCH_LOCK", "~/hypnagogia-cache/bench.lock"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        t = time.time()
        fcntl.flock(f, fcntl.LOCK_EX)
        waited = time.time() - t
        if waited > 1:
            print(f"  (waited {waited:.1f}s for bench lock)", flush=True)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def resize(img, w, h):
    from PIL import Image

    return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))


def run(args):
    mod = importlib.import_module(f"server.engines.{args.engine}")
    extra = json.loads(args.cfg) if args.cfg else {}
    sizes = [int(s) for s in args.sizes.split(",")]
    t0 = time.time()
    eng = mod.create_engine(model=args.model, width=sizes[0], height=sizes[0], **extra)
    load_s = time.time() - t0
    print(f"# {eng.name} model={eng.model} device={eng.device} load={load_s:.1f}s cfg={extra}", flush=True)
    base_seq = scenes.walk(24, 512, 512, palette="blue")
    stills = scenes.test_set(512, 512)
    rows = []
    for S in sizes:
        seq = [resize(f, S, S) for f in base_seq]
        # warm up this shape outside the lock (kernel compilation / allocator growth)
        for i in range(args.warmup):
            eng.process(seq[i % len(seq)], PROMPTS[0], args.strength, 7)
        with bench_lock():
            eng.profile = False
            t = time.perf_counter()
            for i in range(args.iters):
                eng.process(seq[i % len(seq)], PROMPTS[0], args.strength, 7)
            e2e = (time.perf_counter() - t) / args.iters * 1000
            eng.profile = True
            acc: dict = {}
            n_prof = max(8, args.iters // 3)
            for i in range(n_prof):
                eng.process(seq[i % len(seq)], PROMPTS[0], args.strength, 7)
                for k, v in eng.last_timings.items():
                    acc[k] = acc.get(k, 0.0) + v
            eng.profile = False
        stages = {k: round(v / n_prof, 2) for k, v in acc.items()}
        row = dict(engine=eng.name, model=eng.model, size=S, e2e_ms=round(e2e, 2), fps=round(1000 / e2e, 2),
                   stages=stages, machine=machine(), device=eng.device, cfg=extra, strength=args.strength,
                   torch=_torch_version(), when=time.strftime("%Y-%m-%d %H:%M"))
        rows.append(row)
        st = " ".join(f"{k}={v:.1f}" for k, v in stages.items() if k != "total")
        print(f"| {eng.name} | {S} | {e2e:.1f} ms | {1000 / e2e:.1f} FPS | {st} |", flush=True)
        if args.samples:
            sheet(eng, stills, S, args)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    with open(os.path.join(HERE, "out", "results.jsonl"), "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def _torch_version():
    try:
        import torch

        return torch.__version__
    except Exception:
        return None


def sheet(eng, stills, S, args):
    """Contact sheet: rows = strengths, cols = (input, output) per still/prompt pair."""
    from PIL import Image

    strengths = [float(s) for s in args.sheet_strengths.split(",")]
    cell = args.cell
    rows = []
    ins = [resize(im, S, S) for im in stills]
    top = [np.asarray(Image.fromarray(im).resize((cell, cell))) for im in ins]
    rows.append(np.concatenate(top, 1))
    for s in strengths:
        outs = []
        for j, im in enumerate(ins):
            o = eng.process(im, PROMPTS[j % len(PROMPTS)], s, 1234)
            outs.append(np.asarray(Image.fromarray(o).resize((cell, cell), Image.LANCZOS)))
        rows.append(np.concatenate(outs, 1))
    tag = args.tag or ""
    os.makedirs(os.path.join(HERE, "samples"), exist_ok=True)
    name = f"{eng.name}_{S}{tag}.jpg".replace("/", "_")
    Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(HERE, "samples", name), quality=82)
    print(f"  sample sheet -> bench/samples/{name} (rows: input, strengths {strengths})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="torch_turbo")
    ap.add_argument("--model", default="sd-turbo")
    ap.add_argument("--sizes", default="512,448,384,320")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--cfg", default="", help='extra JSON cfg for create_engine, e.g. \'{"channels_last":true}\'')
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--sheet-strengths", default="0.3,0.5,0.7")
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--tag", default="")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
