#!/usr/bin/env python3
"""Do coreml_turbo (ANE) and torch_turbo (MPS) agree for identical (image, depth, prompt, strength,
seed)? Matters for a server pool that alternates frames between them. Writes a sheet (rows: input,
torch_turbo, coreml_turbo, |diff|x4) to bench/samples/coreml_vs_torch_<size>[_graft].jpg.

torch_turbo runs one full pass per frame (deepcache=0, xframe=0): its stream features (DeepCache's
reuse, cross-frame attention) have no ANE counterpart, and this measures the engines, not a stream.
Both run with morph=0, so each still gets its own prompt at once.
--graft compares the depth-grafted builds (torch's default graft 0.8 against a `_d08` Core ML UNet,
both given the procedural scenes' depth); without it, stock SD-Turbo on both.

  ~/hypnagogia-cache/venv-run/bin/python bench/coreml/pool_consistency.py --size 512x320 --graft
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(BENCH))
sys.path.insert(0, BENCH)
import scenes  # noqa: E402
from server import engines  # noqa: E402

PROMPTS = [
    "vast drowned gothic cathedral, bioluminescent, volumetric light, oil painting",
    "overgrown sunken garden temple, moss and flowers, god rays, lush fantasy concept art",
    "neon bathhouse, wet tiles, cyan and magenta glow, cinematic, moody",
]

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="512", help="512 or WxH")
ap.add_argument("--strength", type=float, default=0.5)
ap.add_argument("--graft", action="store_true", help="depth-grafted builds on both engines")
ap.add_argument("--json", default=None, help="also write the numbers here")
a = ap.parse_args()
W, H = (int(v) for v in a.size.lower().split("x")) if "x" in a.size else (int(a.size),) * 2
pairs = scenes.test_set(W, H, with_depth=True)
stills = [im for im, _ in pairs]
# morph=0: every still shares the seed, and a prompt glide would blend each still's prompt with the last one's
cfg = {
    "torch_turbo": dict(deepcache=0, xframe=0, morph=0, **({} if a.graft else {"depth_graft": 0})),
    "coreml_turbo": dict(morph=0, **({} if a.graft else {"depth_graft": 0})),
}
outs, models = {}, {}
for name, extra in cfg.items():
    eng = engines.load(name, width=W, height=H, **extra)
    if bool(getattr(eng, "wants_depth", False)) != a.graft:
        sys.exit(f"{name}: depth graft {'missing' if a.graft else 'present'} ({getattr(eng, 'model', '?')})")
    eng.warmup()
    kw = lambda d: {"depth": d} if a.graft else {}  # noqa: E731
    outs[name] = [eng.process(im, PROMPTS[i], a.strength, 1234, **kw(d)) for i, (im, d) in enumerate(pairs)]
    models[name] = getattr(eng, "model", "?")
    print(name, models[name], flush=True)
    del eng
res = []
for x, y in zip(outs["torch_turbo"], outs["coreml_turbo"]):
    d = np.abs(x.astype(np.float32) - y.astype(np.float32))
    mse = float((d ** 2).mean())
    res.append({"mean_abs_diff": round(float(d.mean()), 2), "psnr_db": round(10 * np.log10(255 ** 2 / max(mse, 1e-9)), 2)})
rows = [stills, outs["torch_turbo"], outs["coreml_turbo"],
        [np.clip(np.abs(x.astype(np.int16) - y.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
         for x, y in zip(outs["torch_turbo"], outs["coreml_turbo"])]]
cw, ch = (256, 256 * H // W) if W >= H else (256 * W // H, 256)
sheet = np.concatenate([np.concatenate([np.asarray(Image.fromarray(im).resize((cw, ch), Image.LANCZOS))
                                        for im in r], 1) for r in rows], 0)
tag = f"{W}x{H}" if W != H else str(W)
out = os.path.join(BENCH, "samples", f"coreml_vs_torch_{tag}{'_graft' if a.graft else ''}.jpg")
Image.fromarray(sheet).save(out, quality=85)
for i, r in enumerate(res):
    print(f"still {i}: mean |diff| {r['mean_abs_diff']:.1f}/255, PSNR {r['psnr_db']:.1f} dB")
print("sheet ->", out)
if a.json:
    import subprocess
    import time
    try:
        machine = subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:  # noqa: BLE001
        machine = None
    val = lambda x: x.startswith("--js") and "=" not in x   # (--json PATH; --json=PATH is one token)  # noqa: E731
    args = [x for i, x in enumerate(sys.argv[1:]) if not x.startswith("--js") and not val(sys.argv[i])]
    with open(a.json, "w") as f:   # (no output path in `by`: results get published)
        json.dump({"by": " ".join(["bench/coreml/pool_consistency.py"] + args), "date": time.strftime("%Y-%m-%d"),
                   "machine": machine, "size": [W, H],
                   "strength": a.strength, "graft": a.graft, "engines": models, "config": cfg, "stills": res,
                   "sheet": os.path.relpath(out, os.path.dirname(BENCH))}, f, indent=1)
