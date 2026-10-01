#!/usr/bin/env python3
"""Does a DeepCache cheap pass follow the *current* frame's geometry (what the client's
projection assumes) or the stale full-pass frame's? Edge-map correlation of each output with
the current input vs with the input of the frame whose deep features it reused.

    python bench/alignment.py --size 384 --seq turn
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import scenes  # noqa: E402
from bench_engines import PROMPTS, resize  # noqa: E402


def edges(im):
    g = im.astype(np.float32).mean(2)
    gx = np.abs(np.diff(g, axis=1))[:-1]
    gy = np.abs(np.diff(g, axis=0))[:, :-1]
    e = gx + gy
    # blur a little so 1-2 px misregistration isn't all-or-nothing
    k = np.ones(5) / 5
    e = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, e)
    e = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, e)
    return (e - e.mean()) / (e.std() + 1e-6)


def corr(a, b):
    return float((a * b).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--seq", default="turn", choices=["walk", "turn"])
    ap.add_argument("--strength", type=float, default=0.55)
    ap.add_argument("--deepcache", type=int, default=3)
    ap.add_argument("--motion", type=int, default=0)
    ap.add_argument("--max-shift", type=int, default=99)
    ap.add_argument("--branch", type=int, default=1)
    a = ap.parse_args()
    from server.engines import torch_turbo as tt

    frames = scenes.walk(24, 512, 512, palette="amber") if a.seq == "walk" else scenes.turn(24, 512, 512)
    seq = [resize(f, a.size, a.size) for f in frames]
    eng = tt.create_engine(width=a.size, height=a.size, morph=0, depth_graft=0, deepcache=a.deepcache, dc_thresh=99,
                           dc_motion=bool(a.motion), dc_branch=a.branch, dc_max_shift=a.max_shift)
    E_in = [edges(f) for f in seq]
    rows = []
    for i, f in enumerate(seq):
        o = eng.process(f, PROMPTS[0], a.strength, 7)
        k = i % max(a.deepcache, 3)  # frames since the full pass (deepcache=0: same frames, reference)
        if k == 0:
            continue
        eo = edges(o)
        rows.append((corr(eo, E_in[i]), corr(eo, E_in[i - k])))
    cur, stale = np.mean([r[0] for r in rows]), np.mean([r[1] for r in rows])
    print(f"{a.seq} N={a.deepcache} motion={a.motion} branch={a.branch}: {'cheap-pass' if a.deepcache >= 2 else 'full-pass (reference)'} output edge-corr with CURRENT input {cur:.3f} vs STALE input {stale:.3f} "
          f"({sum(r[0] > r[1] for r in rows)}/{len(rows)} frames closer to current)")


if __name__ == "__main__":
    main()
