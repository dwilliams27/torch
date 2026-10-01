#!/usr/bin/env python3
"""Distilled-student pilot (M112): can a UNet with blocks cut out learn back what it lost?

    python bench/student_pilot.py --pairs DIR [DIR ...] --out RUN_DIR [--steps 800] [--cut bk]

Teacher: the engine as served (SD-Turbo + depth graft 0.8), one full pass. Student: the same
UNet with the cut from `bench/cut_census.py` (default `bk`: no mid block, the second
resnet-attention pair of every down stage, the last attention of every up stage; 26% less
time a full pass, and untrained it shatters the picture, M115). The student's up blocks and
`conv_out` train in fp32 (Adam) to reproduce the teacher's x0 prediction (latent MSE) on
captures harvested from the game (`tools/harvest_pairs.mjs`: each pair's walking capture
and its rest capture, with their depth, prompt, seed and strength). Held out: the same
stretches of the tour as `bench/one_pass.py`. Reported: PSNR of the decoded student output
against the decoded teacher output on held-out captures, before and after, the loss curve
and step time. A pilot of the method, not a product: BK-SDM (Kim et al. 2023) distils for
days, with feature-level losses as well.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from cut_census import CUTS, cut_forward  # noqa: E402
from one_pass import load_pairs, split  # noqa: E402


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 10 * np.log10(255 ** 2 / max(mse, 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--cut", default="bk", choices=[c for c in CUTS if c != "none"])
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    import torch
    from server.engines import torch_turbo as tt

    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    pairs = [p for d in a.pairs for p in load_pairs(d)]
    train, hold = split(pairs)
    H, W = pairs[0]["input"].shape[:2]
    eng = tt.create_engine(model="sd-turbo", width=W, height=H, morph=0, deepcache=0, xframe=0)
    dev, dt = eng.device, eng.dtype
    unet = eng.unet
    unet.requires_grad_(False)
    for m in (eng.vae, eng._lite_enc, eng._lite_dec):
        if m is not None:
            m.requires_grad_(False)
    skip = set(CUTS[a.cut])

    # every capture (walking and rest) is a training input; the teacher's x0 is the target
    conds, steps_t = {}, {}

    @torch.no_grad()
    def prep(img, strength, depth, prompt, seed):
        x = torch.from_numpy(np.ascontiguousarray(img)).to(dev).permute(2, 0, 1)[None].to(dt).div(255.0)
        z0 = (eng._lite_enc(x) if eng._lite_enc is not None else eng.vae.encode(x.mul(2).sub(1)).latents).clone()
        if prompt not in conds:
            conds[prompt] = eng._conditioning(prompt, 0).clone()
        if strength not in steps_t:
            t_, sa, s1a, _ = eng._timestep(strength)
            steps_t[strength] = (t_.clone(), sa, s1a)
        t_, sa, s1a = steps_t[strength]
        zt = z0 * sa + eng._noise(seed, z0.shape).clone() * s1a
        dz = eng._depth_latent(depth, z0.shape, z0.shape[2:], ("sp", 0)).clone()
        zin = torch.cat([zt, dz], 1) if eng.wants_depth else zt
        return {"zin": zin, "zt": zt, "cond": conds[prompt], "t": steps_t[strength]}

    def items(ps):
        out = []
        for p in ps:
            out.append(prep(p["input"], p["strength"], p["depth"], p["prompt"], p["seed"]))
            out.append(prep(p["rest"], p["rest_strength"], p["rest_depth"], p["rest_prompt"], p["seed"]))
        return out

    t0 = time.time()
    tr, ho = items(train), items(hold)

    def x0(it, student):
        t_, sa, s1a = it["t"]
        c = it["cond"].clone()   # a fresh tensor: the attention caches cross-attention K/V by identity
        eps = cut_forward(unet, it["zin"], t_, c, skip) if student else tt.lean_unet_forward(unet, it["zin"], t_, c)
        return (it["zt"] - eps * s1a) / sa

    with torch.no_grad():
        for it in tr + ho:
            it["target"] = x0(it, False).float().clone()
    print(f"{len(tr)} training and {len(ho)} held-out captures ready in {time.time() - t0:.0f}s", flush=True)

    def decode(z):
        with torch.no_grad():
            y = eng._lite_dec(torch.tanh(z.to(dt) / 3.0) * 3.0).mul(2).sub(1) if eng._lite_dec is not None else eng.vae.decode(z.to(dt)).sample
        return y[0].clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()

    def evaluate():
        with torch.no_grad():
            return float(np.mean([psnr(decode(x0(it, True).float()), decode(it["target"])) for it in ho]))

    # the student's trainable part: everything after the cut middle, in fp32
    trainable = list(unet.up_blocks) + [unet.conv_norm_out, unet.conv_out]
    params = []
    for m in trainable:
        if m is None:
            continue
        m.float().requires_grad_(True)
        params += [p for p in m.parameters()]
    n_params = sum(p.numel() for p in params)
    print(f"cut {a.cut}: training {n_params / 1e6:.0f}M parameters (up blocks, output) in fp32", flush=True)

    # the up blocks now run in fp32: their inputs are cast, their outputs cast back
    def cast_hooks(mod):
        mod.register_forward_pre_hook(lambda m, args, kw: (tuple(x.float() if torch.is_tensor(x) and x.is_floating_point() else x for x in args),
                                                            {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in kw.items()}),
                                      with_kwargs=True)
        mod.register_forward_hook(lambda m, args, out: out.to(dt) if torch.is_tensor(out) else
                                  tuple(o.to(dt) if torch.is_tensor(o) and o.is_floating_point() else o for o in out))
    # (cut_forward calls the up blocks' resnets, attentions and upsamplers one by one, so the
    # casts go on those, and on the output layers)
    called = [unet.conv_norm_out, unet.conv_out]
    for up in unet.up_blocks:
        called += list(up.resnets) + list(getattr(up, "attentions", None) or []) + list(up.upsamplers or [])
    for m in called:
        if m is not None:
            cast_hooks(m)

    before = evaluate()
    print(f"held-out PSNR against the teacher, untrained: {before:.2f} dB", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    times, losses, curve = [], [], []
    for step in range(a.steps):
        it = random.choice(tr)
        s = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = torch.mean((x0(it, True).float() - it["target"]) ** 2)
        if not torch.isfinite(loss):
            print(f"step {step}: loss {float(loss)}; stopping", flush=True)
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if dev == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - s)
        losses.append(float(loss))
        if step % 100 == 0 or step == a.steps - 1:
            ev = evaluate() if step % 200 == 0 or step == a.steps - 1 else None
            if ev is not None:
                curve.append((step, round(ev, 2)))
            print(f"step {step} loss {np.mean(losses[-100:]):.4f} {np.median(times[-100:]):.2f}s/step"
                  + (f" held-out PSNR {ev:.2f} dB" if ev is not None else ""), flush=True)
    after = evaluate()
    res = {"date": time.strftime("%Y-%m-%d"), "cut": a.cut, "skip": sorted(skip), "sources": [os.path.basename(os.path.normpath(d)) for d in a.pairs],
           "train_captures": len(tr), "held_out_captures": len(ho), "steps": len(losses), "lr": a.lr, "trainable_m": round(n_params / 1e6, 1),
           "step_s_median": round(float(np.median(times)), 3) if times else None,
           "loss_first100": round(float(np.mean(losses[:100])), 5) if losses else None,
           "loss_last100": round(float(np.mean(losses[-100:])), 5) if losses else None,
           "psnr_untrained": round(before, 2), "psnr_trained": round(after, 2), "curve": curve,
           "note": "teacher and student on the plain forward (no DeepCache, no cross-frame attention); PSNR of decoded images, held-out stretches"}
    json.dump(res, open(os.path.join(a.out, "student_pilot.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
