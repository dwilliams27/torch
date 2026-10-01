#!/usr/bin/env python3
"""How far what you see while walking is from what the same view becomes when you stop.

    python bench/walk_gap.py DIR [DIR ...] [--out gap.json]
    python bench/walk_gap.py --floor DIR REHOLD_DIR   # the same framings held twice

DIR is a tools/harvest_pairs.mjs run: per framing, walk.jpg (the result the game showed for
a walking capture) and target.jpg (the newest result after standing at that capture's camera
for a few seconds). Per run: LPIPS (Zhang et al. 2018, AlexNet) and mean |difference| (0-255)
from walk to target, detail (mean luma gradient) of both, per zone and overall. Runs on the
CPU, so it can score one harvest while the GPU serves the next.

--floor compares the targets of framings held twice with different histories (REHOLD_DIR
from `tools/harvest_pairs.mjs --phase hold` on copies of DIR's captures): how far a target
depends on what came before it, a floor under any "closer to the target" figure.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from fixed_point import detail  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--floor", action="store_true")
    a = ap.parse_args()
    import lpips
    import torch

    from server.app import jpeg_decode   # decoded as the game's server decodes (bench/one_pass.py too)
    read = lambda p: np.array(jpeg_decode(open(p, "rb").read()))

    torch.set_num_threads(4)
    net = lpips.LPIPS(net="alex", verbose=False).eval()
    t = lambda im: torch.from_numpy(im).permute(2, 0, 1)[None].float().div(127.5).sub(1.0)
    res = {}
    if a.floor:
        base, again = a.runs
        rows = []
        for d in sorted(glob.glob(os.path.join(again, "[0-9]" * 4))):
            t1, t2 = os.path.join(base, os.path.basename(d), "target.jpg"), os.path.join(d, "target.jpg")
            if not (os.path.exists(t1) and os.path.getsize(t1) and os.path.exists(t2) and os.path.getsize(t2)):
                continue
            x, y = read(t1), read(t2)
            with torch.no_grad():
                rows.append({"pair": os.path.basename(d), "lpips": float(net(t(x), t(y)).mean()),
                             "absdiff": float(np.abs(x.astype(np.float32) - y.astype(np.float32)).mean())})
        res = {"base": os.path.basename(os.path.normpath(base)), "again": os.path.basename(os.path.normpath(again)), "n": len(rows), "lpips": float(np.mean([r["lpips"] for r in rows])),
               "absdiff": float(np.mean([r["absdiff"] for r in rows])), "pairs": rows}
        print(f"target held twice: {res['n']} framings, LPIPS {res['lpips']:.3f}, |diff| {res['absdiff']:.1f}")
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
        return
    for run in a.runs:
        rows = []
        for d in sorted(glob.glob(os.path.join(run, "[0-9]" * 4))):
            m = json.load(open(os.path.join(d, "meta.json")))
            tp, wp = os.path.join(d, "target.jpg"), os.path.join(d, "walk.jpg")
            if (m.get("hold") or {}).get("dropped") or not os.path.exists(wp) or not os.path.exists(tp) or not os.path.getsize(tp):
                continue
            w, g = read(wp), read(tp)
            with torch.no_grad():
                lp = float(net(t(w), t(g)).mean())
            rows.append({"pair": os.path.basename(d), "zone": m.get("zone"), "lpips": lp,
                         "absdiff": float(np.abs(w.astype(np.float32) - g.astype(np.float32)).mean()),
                         "detail_walk": detail(w), "detail_target": detail(g), "passes": m["hold"].get("passes")})
        if not rows:
            raise SystemExit(f"{run}: no pairs")
        agg = lambda rs: {k: float(np.mean([r[k] for r in rs])) for k in ("lpips", "absdiff", "detail_walk", "detail_target")} | {"n": len(rs)}
        zones = sorted({r["zone"] for r in rows if r["zone"]})
        name = os.path.basename(os.path.normpath(run))   # (results keep no local paths: the project is published)
        side = lambda f: json.load(open(os.path.join(run, f))) if os.path.exists(os.path.join(run, f)) else None
        info = side("info.json")   # the harvest's engine settings, speed and hold (whitelisted by the harvester)
        res[name] = {"all": agg(rows), "zones": {z: agg([r for r in rows if r["zone"] == z]) for z in zones},
                     "passes_median": float(np.median([r["passes"] for r in rows if r["passes"] is not None])),
                     "walk": side("walk.json"), "harvest": info and {k: info.get(k) for k in (
                         "engine", "depth_graft", "xframe", "held_only", "lora", "speed", "every", "hold", "params", "date")},
                     "pairs": rows}
        s = res[name]["all"]
        print(f"{name}: {s['n']} pairs, LPIPS walk->target {s['lpips']:.3f}, |diff| {s['absdiff']:.1f}, "
              f"detail walk {s['detail_walk']:.2f} target {s['detail_target']:.2f}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
