#!/usr/bin/env python3
"""Contact sheets and look metrics for tools/shoot.mjs output (needs numpy + Pillow; the
server venv has both).

    python tools/sheet.py RUN_DIR [RUN_DIR2 ...] --out sheet.jpg [--zones nave,stacks]
                          [--phases settled,walk1,walk2,walk3,rest] [--scale 0.5]

One run: a grid, zones down, phases across. Several runs: for each zone and phase the runs
sit side by side (before | after), labelled with the run directory's name.

Metrics per image, printed as a table and written next to the sheet as <out>.metrics.json:
  detail    mean |Laplacian| of luma after a 2x2 box downsample (0-255 scale): brush detail
            vs mush; the downsample keeps per-pixel film grain from dominating it
  contrast  RMS contrast, std of luma (0-255): milkiness drops it
  sat       mean HSV saturation (0-1)
  luma      mean luma (0-255)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw


def metrics(img: Image.Image) -> dict:
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    y = a @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    h, w = (y.shape[0] // 2) * 2, (y.shape[1] // 2) * 2
    y2 = y[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    lap = np.abs(4 * y2[1:-1, 1:-1] - y2[:-2, 1:-1] - y2[2:, 1:-1] - y2[1:-1, :-2] - y2[1:-1, 2:])
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = np.where(mx > 1e-3, (mx - mn) / np.maximum(mx, 1e-3), 0)
    return {"detail": round(float(lap.mean()), 2), "contrast": round(float(y.std()), 2),
            "sat": round(float(sat.mean()), 3), "luma": round(float(y.mean()), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--zones", default=None)
    ap.add_argument("--phases", default="settled,walk1,walk2,walk3,rest")
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()
    metas = [json.load(open(os.path.join(r, "shots.json"))) for r in a.runs]
    zones = a.zones.split(",") if a.zones else list(metas[0]["zones"].keys())
    phases = a.phases.split(",")
    names = [os.path.basename(os.path.normpath(r)) for r in a.runs]

    table = {}
    cells = []  # rows of images
    for z in zones:
        row = []
        for ph in phases:
            for r, name in zip(a.runs, names):
                p = os.path.join(r, f"{z}_{ph}.png")
                if not os.path.exists(p):   # keep runs paired: a blank cell stands in
                    row.append(None)
                    continue
                im = Image.open(p).convert("RGB")
                table.setdefault(z, {}).setdefault(ph, {})[name] = metrics(im)
                w, h = im.size
                im = im.resize((int(w * a.scale), int(h * a.scale)), Image.LANCZOS)
                d = ImageDraw.Draw(im)
                label = f"{z} {ph}" + (f"  [{name}]" if len(a.runs) > 1 else "")
                d.rectangle([0, 0, 8 + 7 * len(label), 16], fill=(0, 0, 0))
                d.text((4, 2), label, fill=(235, 230, 218))
                row.append(im)
        if any(row):
            cells.append(row)
    if not cells:
        raise SystemExit("no images found")
    cw, ch = next(im for row in cells for im in row if im).size
    gap = 4
    per_row = max(len(r) for r in cells)
    if len(a.runs) > 1:  # one phase per row keeps pairs readable
        cells = [r[i:i + len(a.runs)] for r in cells for i in range(0, len(r), len(a.runs))]
        per_row = len(a.runs)
    sheet = Image.new("RGB", (per_row * (cw + gap) - gap, len(cells) * (ch + gap) - gap), (0, 0, 0))
    for j, row in enumerate(cells):
        for i, im in enumerate(row):
            if im is not None:
                sheet.paste(im, (i * (cw + gap), j * (ch + gap)))
    sheet.save(a.out, quality=88)
    json.dump(table, open(os.path.splitext(a.out)[0] + ".metrics.json", "w"), indent=1)

    print(f"{'zone':10s} {'phase':8s} " + " ".join(f"{n[:22]:>32s}" for n in names))
    for z, phs in table.items():
        for ph, per in phs.items():
            vals = []
            for n in names:
                m = per.get(n)
                vals.append(f"{'-':>32s}" if not m else f"d{m['detail']:5.2f} c{m['contrast']:5.1f} s{m['sat']:.2f} l{m['luma']:5.1f}".rjust(32))
            print(f"{z:10s} {ph:8s} " + " ".join(vals))


if __name__ == "__main__":
    main()
