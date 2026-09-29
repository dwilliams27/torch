#!/usr/bin/env python3
"""Temporal-quality check: run a walking-camera sequence through an engine config and write
a filmstrip (input row + output row per config) plus a flicker metric.

    python bench/temporal.py --size 384 --cfgs '{}' '{"deepcache":3}' --frames 6

flicker = mean |out[i] - out[i-1]| minus mean |in[i] - in[i-1]| (in 0..255 units): how much
frame-to-frame change the engine adds on top of the real camera motion. Lower is calmer.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import scenes  # noqa: E402
from bench_engines import PROMPTS, resize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sd-turbo")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--strength", type=float, default=0.55)
    ap.add_argument("--frames", type=int, default=6, help="frames shown in the strip")
    ap.add_argument("--cell", type=int, default=192)
    ap.add_argument("--cfgs", nargs="+", default=["{}", '{"deepcache":3}'])
    ap.add_argument("--out", default=os.path.join(HERE, "samples", "temporal.jpg"))
    ap.add_argument("--seq", default="walk", choices=["walk", "turn"])
    a = ap.parse_args()
    from PIL import Image

    from server.engines import torch_turbo as tt

    frames = scenes.walk(24, 512, 512, palette="amber") if a.seq == "walk" else scenes.turn(24, 512, 512)
    seq = [resize(f, a.size, a.size) for f in frames]
    rows = [np.concatenate([resize(f, a.cell, a.cell) for f in seq[8:8 + a.frames]], 1)]
    base = None
    for c in a.cfgs:
        cfg = json.loads(c)
        if base is None:
            base = tt.create_engine(model=a.model, width=a.size, height=a.size, morph=0, **cfg)
        else:  # reuse loaded weights, just change runtime knobs
            base.deepcache = int(cfg.get("deepcache", 3))
            base.dc_branch = int(cfg.get("dc_branch", 1))
            base._dc.clear()
            base.dc_thresh = float(cfg.get("dc_thresh", 0.06))
            base.dc_motion = bool(cfg.get("dc_motion", False))
            base.dc_max_shift = int(cfg.get("dc_max_shift", 3))
        outs, diffs = [], []
        for f in seq:
            outs.append(base.process(f, PROMPTS[0], a.strength, 7))
            diffs.append(round(base.last_dc_diff, 3))
        if base.deepcache:
            print("  dc diffs:", diffs, flush=True)
        d_in = np.mean([np.abs(seq[i].astype(int) - seq[i - 1]).mean() for i in range(1, len(seq))])
        d_out = np.mean([np.abs(outs[i].astype(int) - outs[i - 1]).mean() for i in range(1, len(outs))])
        print(f"cfg={cfg} frame-delta in={d_in:.2f} out={d_out:.2f} added={d_out - d_in:.2f}", flush=True)
        rows.append(np.concatenate([resize(o, a.cell, a.cell) for o in outs[8:8 + a.frames]], 1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    Image.fromarray(np.concatenate(rows, 0)).save(a.out, quality=82)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
