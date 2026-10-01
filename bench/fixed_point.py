#!/usr/bin/env python3
"""Fixed-point LoRA pilot: can one pass land where thirty passes converge?

    python bench/fixed_point.py --frames DIR [DIR ...] --out RUN_DIR [--steps 500] [--hold 16]

Standing still, the game shows the model its last result (the capture shader's feedback)
and a held framing converges over many passes; walking, a view gets one pass, the model's
misty first take. This pilot measures whether a small LoRA can teach SD-Turbo (with the
depth graft, as served) to jump there in one pass (idea 9 in docs/ideas/2026-09-30.md;
RAFT-style self-distillation with the pix2pix-turbo recipe: LoRA on the UNet, one step).

1. Targets: for each framing (capture.jpg taken with feedback off + depth.json), the loop
   at rest is run offline for `--passes` passes: capture_k = mix(raw, leak(result_k-1), 0.55),
   with the capture shader's saturation leak (0.8) and luminance anchor (0.35); no
   reprojection is needed at a held framing. Target = the last result; first take = pass 1.
2. LoRA, hand-rolled (no peft here): rank `--rank` on to_q/to_k/to_v/to_out of every
   transformer at the top UNet level (both down and up), trained on MSE in latent space
   between the one-pass x0 prediction from the raw capture and the target's latent.
3. Held-out framings: latent L2 to target for stock vs LoRA, detail (mean luma gradient),
   mean colour; a sheet (raw | first take | LoRA one pass | target). Step time logged:
   it sizes a real run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def luma(a):
    return a[..., :3].astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def detail(img):
    y = luma(img)
    return float(np.abs(np.diff(y, axis=1)).mean() + np.abs(np.diff(y, axis=0)).mean())


def feedback_mix(raw, prev, fb=0.55, sat=0.8, anchor=0.35):
    """The capture shader's feedback at a held framing (0..1 sRGB floats)."""
    lp0 = luma(prev)[..., None]
    p = lp0 + (prev - lp0) * sat
    lr = luma(raw)[..., None] + 0.02
    p = p * (1.0 + (np.clip(lr / (lp0 + 0.02), 0.5, 2.0) - 1.0) * anchor)
    return np.clip(raw + (p - raw) * fb, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", type=int, default=30)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--hold", type=int, default=16, help="framings held out for evaluation")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--strength", type=float, default=0.66)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--loss", default="l2", choices=["l2", "l1grad", "lpips"],
                    help="l2 (regresses to the mean: blurs); L1 plus L1 on the latent's spatial gradients; or "
                         "LPIPS on the decoded image (Zhang et al. 2018, AlexNet; the pix2pix-turbo recipe) plus 0.5 x latent L1")
    ap.add_argument("--targets", default=None, help="an .npz of first takes and targets from an earlier run (skips the loop)")
    a = ap.parse_args()
    import torch
    import torch.nn as nn
    from PIL import Image
    from server.engines import torch_turbo as tt

    os.makedirs(a.out, exist_ok=True)
    src = open(os.path.join(os.path.dirname(HERE), "client", "src", "world", "zones.js")).read()
    eng = tt.create_engine(model="sd-turbo", width=512, height=320, morph=0, deepcache=0, xframe=0)
    dev, dt = eng.device, eng.dtype

    # --- framings: raw capture, packed depth, and the zone (depth.json's `zone`) for the prompt
    frames = []
    for d in a.frames:
        img = np.asarray(Image.open(os.path.join(d, "capture.jpg")).convert("RGB"))
        dj = json.load(open(os.path.join(d, "depth.json")))
        inv = 1.0 / np.maximum(np.asarray(dj["depth"], np.float32).reshape(dj["H"], dj["W"]), 0.05)
        H, W = inv.shape
        lat = inv.reshape(H // 8, 8, W // 8, 8).mean((1, 3))
        lo, hi = np.percentile(lat, [2, 98])
        hi = max(hi, lo + 0.15 * np.median(lat))
        zone = dj.get("zone")
        frames.append({"name": os.path.basename(os.path.normpath(d)), "raw": img, "depth": np.clip(2 * (lat - lo) / (hi - lo) - 1, -1, 1),
                       "zone": zone})
    zmap = {z: p for z, p in re.findall(r"id: '(\w+)'.*?prompt: '([^']+)'", src, re.S)}
    missing = [f["name"] for f in frames if f["zone"] not in zmap]
    if missing:
        raise SystemExit(f"no zone (depth.json `zone`) for {missing}")

    def prompt_of(f):
        return zmap[f["zone"]]

    # --- 1. targets: the loop at rest, offline (or from an earlier run's cache)
    t0 = time.time()
    cache = np.load(a.targets) if a.targets else None
    for f in frames:
        if cache is not None and f"{f['name']}_target" in cache:
            f["first"], f["target"] = cache[f"{f['name']}_first"], cache[f"{f['name']}_target"]
            continue
        raw = f["raw"].astype(np.float32) / 255.0
        out = eng.process(f["raw"], prompt_of(f), a.strength, a.seed, depth=f["depth"])
        f["first"] = out
        for _ in range(a.passes - 1):
            cap = (feedback_mix(raw, out.astype(np.float32) / 255.0) * 255.0 + 0.5).astype(np.uint8)
            out = eng.process(cap, prompt_of(f), a.strength, a.seed, depth=f["depth"])
        f["target"] = out
    t_loop = time.time() - t0
    print(f"targets: {len(frames)} framings x {a.passes} passes in {t_loop:.0f}s", flush=True)
    np.savez_compressed(os.path.join(a.out, "targets.npz"), **{f"{f['name']}_{k}": f[k] for f in frames for k in ("first", "target")})

    # --- 2. LoRA on the top level's attention projections
    class LoRA(nn.Module):
        def __init__(self, base: nn.Linear, r: int):
            super().__init__()
            self.base = base
            self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev) / base.in_features ** 0.5)
            self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))

        def forward(self, x, *args, **kw):
            y = self.base(x, *args, **kw)
            return y + ((x.float() @ self.A.t()) @ self.B.t()).to(y.dtype)

    unet = eng.unet
    loras = []
    for blk in (unet.down_blocks[0], unet.up_blocks[-1]):
        for tr in blk.attentions:
            for m in tr.modules():
                for name in ("to_q", "to_k", "to_v"):
                    lin = getattr(m, name, None)
                    if isinstance(lin, nn.Linear):
                        lo_ = LoRA(lin, a.rank)
                        setattr(m, name, lo_)
                        loras.append(lo_)
                out = getattr(m, "to_out", None)
                if isinstance(out, nn.ModuleList) and isinstance(out[0], nn.Linear):
                    lo_ = LoRA(out[0], a.rank)
                    out[0] = lo_
                    loras.append(lo_)
    params = [p for l in loras for p in (l.A, l.B)]
    print(f"LoRA: {len(loras)} projections, {sum(p.numel() for p in params) / 1e6:.2f}M params", flush=True)

    # one-pass x0 prediction, differentiable in the LoRA
    tt_, sa, s1a, t_int = eng._timestep(a.strength)

    def encode(img):
        x = torch.from_numpy(np.ascontiguousarray(img)).to(dev).permute(2, 0, 1)[None].to(dt).div(255.0)
        with torch.no_grad():
            if eng._lite_enc is not None:
                return eng._lite_enc(x)
            return eng.vae.encode(x.mul(2).sub(1)).latents

    def decode(z):
        with torch.no_grad():
            y = eng._lite_dec(torch.tanh(z / 3.0) * 3.0).mul(2).sub(1) if eng._lite_dec is not None else eng.vae.decode(z).sample
        return y[0].clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()

    # (the engine builds these under inference mode; clones are ordinary tensors autograd can use)
    for f in frames:
        f["z_raw"], f["z_tgt"] = encode(f["raw"]).clone(), encode(f["target"]).clone()
        f["cond"] = eng._conditioning(prompt_of(f), a.seed).clone()
        f["dz"] = eng._depth_latent(f["depth"], f["z_raw"].shape, f["z_raw"].shape[2:], ("fp", 0)).clone()
    noise = eng._noise(a.seed, frames[0]["z_raw"].shape).clone()
    tt_ = tt_.clone() if hasattr(tt_, "clone") else tt_

    def x0(f):
        zt = f["z_raw"] * sa + noise * s1a
        eps = tt.lean_unet_forward(unet, torch.cat([zt, f["dz"]], 1), tt_, f["cond"])
        return (zt - eps * s1a) / sa

    # every k-th framing held out, so the held-out set spans the zones along the tour
    k = max(2, len(frames) // max(1, a.hold))
    hold = frames[k - 1::k][:a.hold]
    train = [f for f in frames if not any(f is h for h in hold)]

    def evaluate(tag):
        rows, stats = [], []
        for f in hold:
            with torch.no_grad():
                z = x0(f)
            img = decode(z)
            # compared after the same decode + encode round trip the targets went through
            l2 = float(torch.mean((encode(img).float() - f["z_tgt"].float()) ** 2))
            l2_first = float(torch.mean((encode(f["first"]).float() - f["z_tgt"].float()) ** 2))
            stats.append({"frame": f["name"], "l2": l2, "l2_first": l2_first, "detail": detail(img), "detail_target": detail(f["target"]),
                          "detail_first": detail(f["first"]), "mean_rgb": [float(v) for v in img.reshape(-1, 3).mean(0)],
                          "mean_rgb_target": [float(v) for v in f["target"].reshape(-1, 3).mean(0)]})
            rows.append(np.concatenate([f["raw"], f["first"], img, f["target"]], 1))
        Image.fromarray(np.concatenate(rows[:6], 0)).resize((1024, 320 * min(6, len(rows)) // 2)).save(os.path.join(a.out, f"sheet-{tag}.jpg"), quality=86)
        return stats

    if a.loss == "lpips":
        import lpips
        perceptual = lpips.LPIPS(net="alex", verbose=False).to(dev).eval().requires_grad_(False)
        for f in frames:   # targets as the decoder's range, [-1, 1]
            f["img_tgt"] = torch.from_numpy(f["target"]).to(dev).permute(2, 0, 1)[None].float().div(127.5).sub(1.0)

    def decode_grad(z):   # the frozen tiny decoder, differentiable (as decode(), without no_grad)
        y = eng._lite_dec(torch.tanh(z / 3.0) * 3.0).mul(2).sub(1) if eng._lite_dec is not None else eng.vae.decode(z).sample
        return y.float().clamp(-1, 1)

    stock = evaluate("stock")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    unet.train(False)
    times, losses = [], []
    for step in range(a.steps):
        f = train[step % len(train)]
        s = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        z = x0(f)
        d = z.float() - f["z_tgt"].float()
        if a.loss == "l2":
            loss = torch.mean(d ** 2)
        elif a.loss == "l1grad":   # L1, plus L1 on horizontal and vertical differences of the error (sharp edges must match)
            loss = d.abs().mean() + (d[..., :, 1:] - d[..., :, :-1]).abs().mean() + (d[..., 1:, :] - d[..., :-1, :]).abs().mean()
        else:   # perceptual: any plausible detail is fine, a blur is not
            loss = perceptual(decode_grad(z), f["img_tgt"]).mean() + 0.5 * d.abs().mean()
        if not torch.isfinite(loss):
            print(f"step {step}: loss is {float(loss)}; stopping (fp16 backward overflowed?)", flush=True)
            break
        loss.backward()
        opt.step()
        if dev == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - s)
        losses.append(float(loss))
        if step % 50 == 0 or step == a.steps - 1:
            print(f"step {step} loss {np.mean(losses[-50:]):.4f} {np.median(times[-50:]):.2f}s/step", flush=True)
    tuned = evaluate("lora")
    res = {"date": time.strftime("%Y-%m-%d"), "framings": len(frames), "held_out": len(hold), "passes": a.passes, "steps": a.steps,
           "rank": a.rank, "lr": a.lr, "strength": a.strength, "loss": a.loss, "lora_projections": len(loras),
           "lora_params_m": round(sum(p.numel() for p in params) / 1e6, 3), "target_loop_s": round(t_loop, 1),
           "steps_run": len(losses), "step_s_median": round(float(np.median(times)), 3) if times else None,
           "loss_first50": round(float(np.mean(losses[:50])), 5) if losses else None,
           "loss_last50": round(float(np.mean(losses[-50:])), 5) if losses else None,
           "held_out_l2": {"first_take": float(np.mean([s["l2_first"] for s in stock])), "stock_one_pass": float(np.mean([s["l2"] for s in stock])),
                           "lora_one_pass": float(np.mean([s["l2"] for s in tuned]))},
           "detail": {"first_take": float(np.mean([s["detail_first"] for s in stock])), "stock_one_pass": float(np.mean([s["detail"] for s in stock])),
                      "lora_one_pass": float(np.mean([s["detail"] for s in tuned])), "target": float(np.mean([s["detail_target"] for s in stock]))},
           "per_frame": {"stock": stock, "lora": tuned}}
    json.dump(res, open(os.path.join(a.out, "fixed_point.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_frame"}, indent=1))


if __name__ == "__main__":
    main()
