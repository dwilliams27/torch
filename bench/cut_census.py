#!/usr/bin/env python3
"""Student-cut census: which parts of SD-Turbo's UNet can the game do without?

    python bench/cut_census.py --frames DIR [DIR ...] --out sheet.jpg

The first step toward a smaller, faster student (BK-SDM, Kim et al. 2023, removed the mid
block and one resnet-attention pair per stage, then distilled). No training here: each cut
replaces some sub-blocks with the identity, keeping every skip connection the up path
expects, and the engine runs as served (depth graft 0.8, the frame's depth), one frame at a
time (no DeepCache, no cross-frame attention). Per cut: UNet milliseconds (synchronised,
median of 5), and against the uncut model on the same frames: PSNR, and the silhouette
contrast `bench/depth_graft.py` uses. Frame dirs as in depth_graft.py (capture.jpg,
depth.json); the depth is packed as the client packs it.

Cut names: `mid` (the whole mid block), `mida` (its attention only), `dI.J` (down block I's
J-th resnet-attention pair; only where the resnet keeps its channel count), `dI.Ja` (that
pair's attention only), `uI.Ja` (up block I's J-th attention).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from depth_graft import sil_contrast  # noqa: E402

CUTS = {
    "none": [],
    "mida": ["mida"],
    "mid": ["mid"],
    "d0.1a": ["d0.1a"], "d1.1a": ["d1.1a"], "d2.1a": ["d2.1a"],
    "d0.1": ["d0.1"], "d1.1": ["d1.1"], "d2.1": ["d2.1"], "d3.1": ["d3.1"],
    "u1.2a": ["u1.2a"], "u2.2a": ["u2.2a"], "u3.2a": ["u3.2a"],
    "u3.*a": ["u3.0a", "u3.1a", "u3.2a"],
    # BK-SDM-like: no mid block, second pair of every down stage, last attention of every up stage
    "bk": ["mid", "d0.1", "d1.1", "d2.1", "d3.1", "u1.2a", "u2.2a", "u3.2a"],
}


def cut_forward(unet, sample, timestep, ehs, skip):
    import torch
    t_emb = unet.get_time_embed(sample=sample, timestep=timestep)
    emb = unet.time_embedding(t_emb, None)
    if unet.time_embed_act is not None:
        emb = unet.time_embed_act(emb)
    h = unet.conv_in(sample)
    res = (h,)
    for bi, blk in enumerate(unet.down_blocks):
        attns = getattr(blk, "attentions", None)
        for j, resnet in enumerate(blk.resnets):
            pair = f"d{bi}.{j}"
            if pair in skip and resnet.in_channels == resnet.out_channels:
                pass   # identity: the skip connection carries h through unchanged
            else:
                h = resnet(h, emb)
                if attns is not None and f"{pair}a" not in skip:
                    h = attns[j](h, encoder_hidden_states=ehs, return_dict=False)[0]
            res += (h,)
        if blk.downsamplers is not None:
            for ds in blk.downsamplers:
                h = ds(h)
            res += (h,)
    mb = unet.mid_block
    if mb is not None and "mid" not in skip:
        h = mb.resnets[0](h, emb)
        for a, r in zip(mb.attentions, mb.resnets[1:]):
            if "mida" not in skip:
                h = a(h, encoder_hidden_states=ehs, return_dict=False)[0]
            h = r(h, emb)
    for bi, up in enumerate(unet.up_blocks):
        k = len(up.resnets)
        rs, res = res[-k:], res[:-k]
        attns = getattr(up, "attentions", None)
        for j, resnet in enumerate(up.resnets):
            h = torch.cat([h, rs[-1]], dim=1)
            rs = rs[:-1]
            h = resnet(h, emb)
            if attns is not None and f"u{bi}.{j}a" not in skip:
                h = attns[j](h, encoder_hidden_states=ehs, return_dict=False)[0]
        if up.upsamplers is not None:
            size = res[-1].shape[2:] if res else None
            for us in up.upsamplers:
                h = us(h, size)
    if unet.conv_norm_out is not None:
        h = unet.conv_act(unet.conv_norm_out(h))
    return unet.conv_out(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cuts", nargs="+", default=list(CUTS))
    ap.add_argument("--strength", type=float, default=0.66)
    ap.add_argument("--seed", type=int, default=20268847)
    a = ap.parse_args()
    a.cuts = ["none"] + [c for c in a.cuts if c != "none"]   # the uncut baseline, first
    import re

    import torch
    from PIL import Image, ImageDraw
    from server.engines import torch_turbo as tt

    src = open(os.path.join(os.path.dirname(HERE), "client", "src", "world", "zones.js")).read()
    prompts = dict(re.findall(r"id: '(\w+)'.*?prompt: '([^']+)'", src, re.S))
    prompts["pillar"] = prompts["nave"]
    frames = []
    for d in a.frames:
        img = np.asarray(Image.open(os.path.join(d, "capture.jpg")).convert("RGB"))
        dj = json.load(open(os.path.join(d, "depth.json")))
        inv = 1.0 / np.maximum(np.asarray(dj["depth"], np.float32).reshape(dj["H"], dj["W"]), 0.05)
        H, W = inv.shape
        lat = inv.reshape(H // 8, 8, W // 8, 8).mean((1, 3))           # as the client packs it
        lo, hi = np.percentile(lat, [2, 98])
        hi = max(hi, lo + 0.15 * np.median(lat))
        frames.append((os.path.basename(os.path.normpath(d)), img, inv, np.clip(2 * (lat - lo) / (hi - lo) - 1, -1, 1)))
    H, W = frames[0][1].shape[:2]
    eng = tt.create_engine(model="sd-turbo", width=W, height=H, morph=0, deepcache=0, xframe=0)
    eng.profile = True
    ref = eng.process(frames[0][1], prompts[frames[0][0]], a.strength, a.seed, depth=frames[0][3])   # the engine's own forward
    state = {"skip": set()}
    eng._unet_call = lambda u, zt, t, cond, **kw: cut_forward(u, zt, t, cond, state["skip"])
    same = eng.process(frames[0][1], prompts[frames[0][0]], a.strength, a.seed, depth=frames[0][3])
    err = float(np.abs(same.astype(np.int16) - ref.astype(np.int16)).max())
    print(f"uncut forward vs the engine's: max |diff| {err:g} (0-255)", flush=True)
    assert err <= 2, "cut_forward with no cuts must match lean_unet_forward"
    for _ in range(3):   # MPS kernels
        eng.process(frames[0][1], prompts[frames[0][0]], a.strength, a.seed, depth=frames[0][3])

    def psnr(x, y):
        mse = np.mean((x.astype(np.float32) - y.astype(np.float32)) ** 2)
        return float(10 * np.log10(255 ** 2 / max(mse, 1e-6)))

    results, outs = {}, {}
    for cut in a.cuts:
        state["skip"] = set(CUTS[cut])
        ms, ps, sil = [], [], []
        for name, img, inv, dl in frames:
            runs = []
            for _ in range(5):
                out = eng.process(img, prompts[name], a.strength, a.seed, depth=dl)
                runs.append(eng.last_timings["unet"])
            ms.append(statistics.median(runs))
            outs[(cut, name)] = out
            if cut != "none":
                ps.append(psnr(out, outs[("none", name)]))
            r = sil_contrast(out, inv)[2]
            if r is not None:
                sil.append(r)
        results[cut] = {"skip": CUTS[cut], "unet_ms": round(float(np.median(ms)), 1),
                        "psnr_vs_none": round(float(np.mean(ps)), 2) if ps else None,
                        "sil_ratio": round(float(np.mean(sil)), 3) if sil else None}
        print(cut, results[cut], flush=True)
    base = results["none"]["unet_ms"]
    for r in results.values():
        r["saving"] = round(1 - r["unet_ms"] / base, 3)
    rows = [np.concatenate([outs[(c, n)] for c in a.cuts], 1) for n, _, _, _ in frames]
    sheet = Image.fromarray(np.concatenate(rows, 0))
    dr = ImageDraw.Draw(sheet)
    for i, c in enumerate(a.cuts):
        dr.rectangle((i * W + 4, 4, i * W + 12 + 8 * len(c), 22), fill=(0, 0, 0))
        dr.text((i * W + 8, 7), c, fill=(255, 255, 140))
    sheet.resize((sheet.width // 3, sheet.height // 3)).save(a.out, quality=86)
    json.dump({"date": time.strftime("%Y-%m-%d"), "strength": a.strength, "size": [W, H], "frames": [f[0] for f in frames],
               "engine": {"depth_graft": eng.depth_graft, "deepcache": 0, "xframe": 0, "device": eng.device}, "cuts": results},
              open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
