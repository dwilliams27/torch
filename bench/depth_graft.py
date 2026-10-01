#!/usr/bin/env python3
"""Depth graft probe: can SD-Turbo take the game's depth as a fifth input channel?

    python bench/depth_graft.py --frames DIR [DIR ...] --out sheet.jpg [--lambdas 0.5 1.0]

The UNet becomes SD-Turbo + lambda * (SD2-depth - SD2-base), with SD2-depth's fifth
`conv_in` input channel (MiDaS-style relative inverse depth at latent size, scaled per frame
to [-1, 1]) taken whole: the "add difference" merge the community uses to make inpainting
fine-tunes, here across SD-Turbo's distillation. Nothing is trained. Weights (fp16 UNets,
~1.7 GB each): sd2-community/stable-diffusion-2-depth and sd2-community/stable-diffusion-2-base
(what SD2-depth was resumed from).

Each frame DIR holds `capture.jpg` (what the model gets; capture with feedback=0 for the pure
render) and `depth.json` ({W, H, depth: view depth in m, top row first}). Arms per frame:
stock SD-Turbo; the graft with the true depth; with flat depth; with the depth flipped
left-right (if the model follows the channel, structure moves with the flipped depth).
Plain engine (no DeepCache, no cross-frame attention) so every arm sees the same thing.

Metric per output: silhouette contrast, the mean |luma step| between pixels 2 px apart across
a depth silhouette (inverse depths differ by more than 30%) over the same within a surface
(within 2%), 2x2-box luma. Higher: the picture separates what the geometry separates. With
the flipped arm it is measured against the flipped depth too (`flip-match`): the arm should
score higher against the depth it was given.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def luma(img):
    y = img[..., :3].astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    h, w = y.shape[0] // 2 * 2, y.shape[1] // 2 * 2
    return y[:h, :w].reshape(h // 2, 2, w // 2, 2).mean((1, 3))


def sil_contrast(img, inv):
    """(silhouette step, surface step, ratio) on a 2x2-box grid, pairs 1 cell (2 px) apart."""
    y = luma(img)
    h, w = y.shape
    iv = inv[:h * 2, :w * 2].reshape(h, 2, w, 2).mean((1, 3))
    sil, tex = [], []
    for dy, dx in ((0, 1), (1, 0)):
        a, b = iv[:h - dy, :w - dx], iv[dy:, dx:]
        d = np.abs(y[:h - dy, :w - dx] - y[dy:, dx:])
        r = np.maximum(a, b) / np.maximum(np.minimum(a, b), 1e-6)
        ok = (a > 1e-3) & (b > 1e-3)
        sil.append(d[ok & (r > 1.3)])
        tex.append(d[ok & (r < 1.02)])
    s, t = np.concatenate(sil), np.concatenate(tex)
    if len(s) < 50 or len(t) < 50:
        return None, None, None
    return float(s.mean()), float(t.mean()), float(s.mean() / max(t.mean(), 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.5, 1.0])
    ap.add_argument("--strengths", nargs="+", type=float, default=[0.6])
    ap.add_argument("--seed", type=int, default=20268847)
    ap.add_argument("--norm", default="pct", choices=["minmax", "pct"],
                    help="depth scaling to [-1, 1]: 2nd-98th percentile clipped (what the client sends), or min-max")
    ap.add_argument("--no-controls", action="store_true", help="skip the flat and flipped depth arms")
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(HERE), "client", "src", "world", "zones.js"))
    a = ap.parse_args()
    import torch
    from PIL import Image
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from server.engines import torch_turbo as tt

    # zone prompts straight from the client (frame dir name = zone id; 'pillar' = nave)
    import re
    src = open(a.prompts).read()
    prompts = dict(re.findall(r"id: '(\w+)'.*?prompt: '([^']+)'", src, re.S))
    prompts["pillar"] = prompts["nave"]

    frames = []
    for d in a.frames:
        img = np.asarray(Image.open(os.path.join(d, "capture.jpg")).convert("RGB"))
        dj = json.load(open(os.path.join(d, "depth.json")))
        z = np.asarray(dj["depth"], np.float32).reshape(dj["H"], dj["W"])
        frames.append((os.path.basename(os.path.normpath(d)), img, 1.0 / np.maximum(z, 0.05)))
    H, W = frames[0][1].shape[:2]
    assert all(f[1].shape[:2] == (H, W) for f in frames), "all frames must share one size"

    # stock SD-Turbo in, grafted here arm by arm (the engine's own graft stays off)
    eng = tt.create_engine(model="sd-turbo", width=W, height=H, morph=0, deepcache=0, xframe=0, depth_graft=0)
    assert not eng.wants_depth
    eng.process(frames[0][1], "warmup", 0.6, 0)   # MPS kernels, so the first timed frame isn't a compile
    unet, dev, dt = eng.unet, eng.device, eng.dtype
    stock = {k: v.detach().to("cpu", torch.float16).clone() for k, v in unet.state_dict().items()}
    f = "unet/diffusion_pytorch_model.fp16.safetensors"   # the revisions the engine pins
    sd_depth = load_file(hf_hub_download(tt.DEPTH_REPO, f, revision=tt.DEPTH_REV))
    sd_base = load_file(hf_hub_download(tt.DEPTH_BASE_REPO, f, revision=tt.DEPTH_BASE_REV))
    conv4 = unet.conv_in
    conv5 = torch.nn.Conv2d(5, conv4.out_channels, 3, padding=1).to(dev, dt)
    state = {"dz": None}
    orig_call = eng._unet_call

    def call(u, zt, t, cond, **kw):
        if state["dz"] is not None:
            zt = torch.cat([zt, state["dz"]], 1)
        return orig_call(u, zt, t, cond, **kw)
    eng._unet_call = call

    def graft(lam):
        if lam is None:
            unet.conv_in = conv4
            unet.load_state_dict({k: v.to(dev, dt) for k, v in stock.items()})
            return
        merged = {}
        for k, v in stock.items():
            if k.startswith("conv_in."):
                continue
            merged[k] = (v.float() + lam * (sd_depth[k].float() - sd_base[k].float())).to(dev, dt)
        unet.load_state_dict(merged, strict=False)
        w = sd_depth["conv_in.weight"].float()
        with torch.no_grad():
            conv5.weight[:, :4] = (stock["conv_in.weight"].float() + lam * (w[:, :4] - sd_base["conv_in.weight"].float())).to(dev, dt)
            conv5.weight[:, 4:] = w[:, 4:].to(dev, dt)
            conv5.bias[:] = (stock["conv_in.bias"].float() + lam * (sd_depth["conv_in.bias"].float() - sd_base["conv_in.bias"].float())).to(dev, dt)
        unet.conv_in = conv5

    def depth_latent(inv, mode):
        h, w = H // 8, W // 8
        d = inv.reshape(h, 8, w, 8).mean((1, 3))          # area mean of inverse depth
        if mode == "flip":
            d = d[:, ::-1]
        lo, hi = (d.min(), d.max()) if a.norm == "minmax" else np.percentile(d, [2, 98])
        d = np.clip(2.0 * (d - lo) / max(hi - lo, 1e-6) - 1.0, -1.0, 1.0)
        if mode == "flat":
            d = np.zeros_like(d)
        return torch.from_numpy(np.ascontiguousarray(d)).to(dev, dt)[None, None]

    arms = [("stock", None, None)] + [(f"graft{lam:g}", lam, "true") for lam in a.lambdas]
    lam_max = max(a.lambdas)
    if not a.no_controls:
        arms += [(f"graft{lam_max:g}-flat", lam_max, "flat"), (f"graft{lam_max:g}-flip", lam_max, "flip")]
    results = {}
    rows = []
    for s in a.strengths:
        for name, lam, mode in arms:
            graft(lam)
            for fname, img, inv in frames:
                state["dz"] = depth_latent(inv, mode) if lam is not None else None
                eng.reset_temporal()
                t0 = time.perf_counter()
                out = eng.process(img, prompts[fname], s, a.seed)
                ms = (time.perf_counter() - t0) * 1000
                key = f"{fname}|{name}|s{s:g}"
                sc = sil_contrast(out, inv)
                rec = {"sil": sc[0], "tex": sc[1], "ratio": sc[2], "ms": round(ms, 1)}
                if mode == "flip":
                    rec["flip_match"] = sil_contrast(out, inv[:, ::-1])[2]
                results[key] = rec
                results.setdefault(f"{fname}|input", {"ratio": sil_contrast(img, inv)[2]})
                print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
                results[key]["img"] = out
    # sheet: one row per frame and strength: input | each arm
    from PIL import ImageDraw
    cols = ["input"] + [n for n, _, _ in arms]
    for s in a.strengths:
        for fname, img, inv in frames:
            tiles = [img] + [results[f"{fname}|{n}|s{s:g}"]["img"] for n, _, _ in arms]
            rows.append(np.concatenate(tiles, 1))
    sheet = Image.fromarray(np.concatenate(rows, 0))
    dr = ImageDraw.Draw(sheet)
    for i, c in enumerate(cols):
        dr.text((i * W + 6, 4), c, fill=(255, 255, 120))
    sheet = sheet.resize((sheet.width // 2, sheet.height // 2))
    sheet.save(a.out, quality=88)
    for v in results.values():
        v.pop("img", None)
    json.dump({"arms": cols, "strengths": a.strengths, "results": results}, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
