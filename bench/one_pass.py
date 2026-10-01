#!/usr/bin/env python3
"""One-pass LoRA (M116): teach SD-Turbo, depth graft and all, to paint in one pass what a
view converges to when you stand still, from the game's own walking captures.

    python bench/one_pass.py --pairs DIR [DIR ...] --out RUN_DIR [--steps 3000] [--rank 8]
                             [--layers top] [--rest 0.3] [--eval-only LORA]

Pairs come from tools/harvest_pairs.mjs: input.jpg (a walking capture as the game sent it,
feedback included), walk.jpg (what the served engine returned for it), target.jpg (the newest
result after holding that framing), rest.jpg (the capture that result answered), meta.json
(prompt, seed, strength and packed depth of both captures).

Training: the one-pass x0 prediction from input, through the LoRA'd UNet, against the target:
LPIPS on the decoded image (Zhang et al. 2018, AlexNet) plus 0.5 x L1 on the latent, the
loss M115's pilot (bench/fixed_point.py) found keeps detail. With probability --rest a step
trains the fixed-point pair instead (rest -> target): at rest the loop already sits there,
and the LoRA must not push it anywhere new. The LoRA (hand-rolled, no peft) sits on the
to_q/k/v/out projections of every transformer in the chosen UNet levels: `top` (the pilot's),
`top2` (and the next level down), or `all`.

Held out: every 6th 16 m stretch of the tour, with 8 m either side kept out of training too;
positions are metres along the tour (meta `tour`, recorded by the harvester, so harvests at
different speeds agree), or for older harvests metres walked from their first capture (use
one such harvest at a time). The views look far ahead, so held-out views still show rooms the
training saw, from new places. A harvest made from a server with a LoRA is refused, and each
harvest's engine settings (DIR/info.json) are copied into the results. Reported on the held-out
views: latent L2 (after a TAESD-lite encode of both sides) and LPIPS to the target for the stock one pass, the
served walking result, and the LoRA one pass, with one-pass images put through the server's
JPEG encode and decode as served results are; detail (mean luma gradient); and for the rest
pairs, how far the LoRA moves the loop's fixed point.
Writes RUN_DIR/{lora.safetensors, one_pass.json, sheet-*.jpg}.
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from fixed_point import detail  # noqa: E402


def unpack_depth(d):
    """The header's packed depth -> [-1, 1] floats, as the server decodes it (app._depth_field)."""
    raw = np.frombuffer(base64.b64decode(d["data"]), np.uint8)
    return raw.reshape(int(d["h"]), int(d["w"])).astype(np.float32) / 127.5 - 1.0


def load_pairs(root, min_passes=15):
    # decoded as the server decodes what it is sent, so the model sees the same pixels
    from server.app import jpeg_decode
    pairs, walked, last = [], 0.0, None
    for d in sorted(glob.glob(os.path.join(root, "[0-9]" * 4))):
        m = json.load(open(os.path.join(d, "meta.json")))
        pos = np.asarray(m["cam"]["pos"], np.float64)
        walked += 0.0 if last is None else float(np.linalg.norm(pos - last))   # distance along the walk
        last = pos
        where = m["tour"] if m.get("tour") is not None else walked
        h = m.get("hold") or {}
        tp = os.path.join(d, "target.jpg")
        if h.get("dropped") or not h or not os.path.exists(tp) or not os.path.getsize(tp) \
                or (h.get("passes") or 0) < min_passes or not os.path.exists(os.path.join(d, "walk.jpg")):
            continue
        img = lambda n: np.array(jpeg_decode(open(os.path.join(d, n), "rb").read()))
        pairs.append({"name": os.path.basename(os.path.normpath(root)) + "/" + os.path.basename(d), "walked": where,
                      "zone": m.get("zone"), "prompt": m["prompt"], "seed": int(m["seed"]) & 0x7FFFFFFF,
                      "strength": float(m["strength"]), "depth": unpack_depth(m["depth"]), "input": img("input.jpg"),
                      "walk": img("walk.jpg"),
                      "target": img("target.jpg"), "rest": img("rest.jpg"), "rest_strength": float(h["rest"]["strength"]),
                      "rest_depth": unpack_depth(h["rest"]["depth"]), "rest_prompt": h["rest"]["prompt"], "passes": h.get("passes")})
    return pairs


def split(pairs, stretch=16.0, every=6, guard=8.0):
    """Held-out stretches of the tour (every `every`-th `stretch` metres walked), and the
    training set without them or anything within `guard` metres of one."""
    held = lambda w: int(w // stretch) % every == every - 1
    hold = [p for p in pairs if held(p["walked"])]
    train = [p for p in pairs if not any(held(p["walked"] + dw) for dw in np.arange(-guard, guard + 0.01, 1.0))]
    return train, hold


def lora_targets(unet, layers):
    import torch.nn as nn
    blocks = {"top": (unet.down_blocks[0], unet.up_blocks[-1]),
              "top2": (unet.down_blocks[0], unet.down_blocks[1], unet.up_blocks[-2], unet.up_blocks[-1]),
              "all": tuple(unet.down_blocks) + (unet.mid_block,) + tuple(unet.up_blocks)}[layers]
    names = {id(m): n for n, m in unet.named_modules()}
    out = []
    for blk in blocks:
        for tr in getattr(blk, "attentions", None) or []:
            for m in tr.modules():
                for name in ("to_q", "to_k", "to_v"):
                    lin = getattr(m, name, None)
                    if isinstance(lin, nn.Linear):
                        out.append((names[id(lin)], m, name, None))
                to = getattr(m, "to_out", None)
                if isinstance(to, nn.ModuleList) and isinstance(to[0], nn.Linear):
                    out.append((names[id(to[0])], to, 0, None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="tools/harvest_pairs.mjs runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--layers", default="top", choices=["top", "top2", "all"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rest", type=float, default=0.3, help="share of steps on fixed-point pairs")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval-only", default=None, help="a saved LoRA: evaluate it, no training")
    ap.add_argument("--stretch", type=float, default=16.0, help="held-out stretch length (m walked); every 6th is held out")
    ap.add_argument("--min-passes", type=int, default=15, help="drop holds that got fewer results")
    a = ap.parse_args()
    import lpips
    import torch
    import torch.nn as nn
    from PIL import Image
    from safetensors import safe_open
    from safetensors.torch import save_file
    from server.app import jpeg_decode, jpeg_encode
    from server.engines import torch_turbo as tt

    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    engines = {}
    for d in a.pairs:   # what served each harvest
        ip = os.path.join(d, "info.json")
        info = json.load(open(ip)) if os.path.exists(ip) else None
        if info and info.get("lora"):
            raise SystemExit(f"{d} was harvested from a server with a LoRA ({info['lora']}): its targets aren't the stock loop's")
        engines[os.path.basename(os.path.normpath(d))] = info and {k: info.get(k) for k in ("depth_graft", "xframe", "held_only", "lora", "speed", "hold", "every")}
    pairs = [p for d in a.pairs for p in load_pairs(d, a.min_passes)]
    train, hold = split(pairs, stretch=a.stretch)
    if not hold or not train:
        raise SystemExit(f"{len(pairs)} pairs: too few to hold out every 6th {a.stretch:g} m stretch (try a shorter --stretch)")
    by_zone = lambda ps: {str(z): sum(p["zone"] == z for p in ps) for z in sorted({p["zone"] or "" for p in pairs})}
    print(f"{len(pairs)} pairs: {len(train)} train, {len(hold)} held out {by_zone(hold)}", flush=True)
    H, W = pairs[0]["input"].shape[:2]
    eng = tt.create_engine(model="sd-turbo", width=W, height=H, morph=0, deepcache=0, xframe=0)
    dev, dt = eng.device, eng.dtype
    unet = eng.unet
    # only the LoRA learns (the TAESD-lite stems are built unfrozen)
    for m in (unet, eng.vae, eng._lite_enc, eng._lite_dec):
        if m is not None:
            m.requires_grad_(False)

    class LoRA(nn.Module):
        def __init__(self, base: nn.Linear, r: int):
            super().__init__()
            self.base = base
            self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev) / base.in_features ** 0.5)
            self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))

        def forward(self, x, *args, **kw):
            y = self.base(x, *args, **kw)
            return y + ((x.float() @ self.A.t()) @ self.B.t()).to(y.dtype)

    if a.eval_only:   # the file says what it is, and what it was trained on
        with safe_open(a.eval_only, "pt") as f:
            md = f.metadata() or {}
        a.layers, a.rank = md.get("layers", a.layers), int(md.get("rank", a.rank))
        if "stretch" in md and float(md["stretch"]) != a.stretch:
            train, hold = split(pairs, stretch=float(md["stretch"]))
        trained = set(json.loads(md.get("train_pairs", "[]")))
        if trained & {p["name"] for p in hold}:
            raise SystemExit(f"{a.eval_only} was trained on some of these held-out views")
    loras = {}
    for name, parent, attr, _ in lora_targets(unet, a.layers):
        base = parent[attr] if isinstance(attr, int) else getattr(parent, attr)
        lo = LoRA(base, a.rank)
        if isinstance(attr, int):
            parent[attr] = lo
        else:
            setattr(parent, attr, lo)
        loras[name] = lo
    if a.eval_only:
        with safe_open(a.eval_only, "pt") as f:
            if set(f.keys()) != {n + x for n in loras for x in (".A", ".B")}:
                raise SystemExit(f"{a.eval_only}: its layers don't match --layers {a.layers}")
            for name, lo in loras.items():
                lo.A.data.copy_(f.get_tensor(name + ".A").to(dev))
                lo.B.data.copy_(f.get_tensor(name + ".B").to(dev))
    params = [p for lo in loras.values() for p in (lo.A, lo.B)]
    print(f"LoRA: {len(loras)} projections ({a.layers}, rank {a.rank}), {sum(p.numel() for p in params) / 1e6:.2f}M params", flush=True)

    def to_x(img):
        return torch.from_numpy(np.ascontiguousarray(img)).to(dev).permute(2, 0, 1)[None].to(dt).div(255.0)

    @torch.no_grad()
    def encode(img):
        x = to_x(img)
        return (eng._lite_enc(x) if eng._lite_enc is not None else eng.vae.encode(x.mul(2).sub(1)).latents).clone()

    def decode_grad(z):   # the frozen tiny decoder, differentiable; [-1, 1]
        y = eng._lite_dec(torch.tanh(z / 3.0) * 3.0).mul(2).sub(1) if eng._lite_dec is not None else eng.vae.decode(z).sample
        return y.float().clamp(-1, 1)

    def to_img(y):
        """a decoded latent as the game gets it: through the server's JPEG encode and decode"""
        img = y[0].clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return np.array(jpeg_decode(jpeg_encode(img)))

    def as_target(img):
        return torch.from_numpy(img).to(dev).permute(2, 0, 1)[None].float().div(127.5).sub(1.0)

    conds, steps_t = {}, {}

    def prep(p):
        z = {}
        for k, im, s, dpt, pr in (("in", p["input"], p["strength"], p["depth"], p["prompt"]),
                                  ("rest", p["rest"], p["rest_strength"], p["rest_depth"], p["rest_prompt"])):
            if pr not in conds:
                conds[pr] = eng._conditioning(pr, 0).clone()
            if s not in steps_t:
                t_, sa, s1a, _ = eng._timestep(s)
                steps_t[s] = (t_.clone(), sa, s1a)
            z0 = encode(im)
            z[k] = dict(z0=z0, dz=eng._depth_latent(dpt, z0.shape, z0.shape[2:], ("op", 0)).clone(), cond=conds[pr], t=steps_t[s],
                        noise=eng._noise(p["seed"], z0.shape).clone())
        z["tgt"] = encode(p["target"])
        return z

    t0 = time.time()
    for p in pairs:
        p["z"] = prep(p)
    print(f"encoded in {time.time() - t0:.0f}s", flush=True)

    def x0(zz):
        t_, sa, s1a = zz["t"]
        zt = zz["z0"] * sa + zz["noise"] * s1a
        # a fresh conditioning tensor per call: the engine's attention caches cross-attention K/V
        # by the conditioning's identity, which would hand this step the last step's graph
        zin = torch.cat([zt, zz["dz"]], 1) if eng.wants_depth else zt
        eps = tt.lean_unet_forward(unet, zin, t_, zz["cond"].clone())
        return (zt - eps * s1a) / sa

    perceptual = lpips.LPIPS(net="alex", verbose=False).to(dev).eval().requires_grad_(False)

    @torch.no_grad()
    def scores(img, p):
        """latent L2 (after a decode/encode round trip, as the targets had) and LPIPS, to the target"""
        l2 = float(torch.mean((encode(img).float() - p["z"]["tgt"].float()) ** 2))
        lp = float(perceptual(as_target(img), as_target(p["target"])).mean())
        return l2, lp

    def evaluate(tag, on):
        rows, stats = [], []
        for i, p in enumerate(on):
            with torch.no_grad():
                one, rest = to_img(decode_grad(x0(p["z"]["in"]))), to_img(decode_grad(x0(p["z"]["rest"])))
            s = {"pair": p["name"], "zone": p["zone"], "passes": p["passes"]}
            s["l2"], s["lpips"] = scores(one, p)
            s["rest_l2"], s["rest_lpips"] = scores(rest, p)
            s["detail"], s["detail_rest"], s["detail_target"] = detail(one), detail(rest), detail(p["target"])
            s["mean_rgb"] = [float(v) for v in one.reshape(-1, 3).mean(0)]
            s["walk_l2"], s["walk_lpips"] = scores(p["walk"], p)
            s["detail_walk"] = detail(p["walk"])
            stats.append(s)
            if i % max(1, len(on) // 6) == 0 and len(rows) < 6:
                rows.append(np.concatenate([p["input"], p["walk"], one, p["target"]], 1))
        Image.fromarray(np.concatenate(rows, 0)).resize((1024, 160 * len(rows))).save(os.path.join(a.out, f"sheet-{tag}.jpg"), quality=86)
        return stats

    # stock first: the LoRA's B is zero, or restore it after
    saved = [(lo.B.data.clone()) for lo in loras.values()]
    for lo in loras.values():
        lo.B.data.zero_()
    stock = evaluate("stock", hold)
    for lo, b in zip(loras.values(), saved):
        lo.B.data.copy_(b)

    times, losses = [], []
    if not a.eval_only:
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
        unet.train(False)
        for p in train:
            p["img_tgt"] = as_target(p["target"])
        # dynamic loss scaling: the UNet runs in fp16, where small gradients underflow to zero
        # (Adam's steps don't depend on the scale; a step whose gradients overflow is skipped)
        scale, good, skipped = 1024.0, 0, 0
        for step in range(a.steps):
            p = random.choice(train)
            src = p["z"]["rest"] if random.random() < a.rest else p["z"]["in"]
            s = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            z = x0(src)
            loss = perceptual(decode_grad(z), p["img_tgt"]).mean() + 0.5 * (z.float() - p["z"]["tgt"].float()).abs().mean()
            if not torch.isfinite(loss):
                print(f"step {step}: loss is {float(loss)}; stopping", flush=True)
                break
            (loss * scale).backward()
            if not bool(torch.stack([torch.isfinite(q.grad).all() for q in params if q.grad is not None]).all()):
                scale, good, skipped = scale / 2, 0, skipped + 1
                continue
            for q in params:
                if q.grad is not None:
                    q.grad.div_(scale)
            opt.step()
            good += 1
            if good % 200 == 0 and scale < 65536:
                scale *= 2
            if dev == "mps":
                torch.mps.synchronize()
            times.append(time.perf_counter() - s)
            losses.append(float(loss))
            if step % 100 == 0 or step == a.steps - 1:
                print(f"step {step} loss {np.mean(losses[-100:]):.4f} {np.median(times[-100:]):.2f}s/step scale {scale:g} skipped {skipped}", flush=True)
        meta = {"rank": str(a.rank), "layers": a.layers, "steps": str(len(losses)), "depth_graft": str(eng.depth_graft),
                "pairs": str(len(train)), "trained": time.strftime("%Y-%m-%d"), "by": "bench/one_pass.py",
                "stretch": str(a.stretch), "train_pairs": json.dumps(sorted(p["name"] for p in train))}
        save_file({k: v for name, lo in loras.items() for k, v in ((name + ".A", lo.A.detach().float().cpu().contiguous()),
                                                                     (name + ".B", lo.B.detach().float().cpu().contiguous()))},
                  os.path.join(a.out, "lora.safetensors"), metadata=meta)
    tuned = evaluate("lora", hold)

    m = lambda xs, k: float(np.mean([x[k] for x in xs if k in x]))
    res = {"date": time.strftime("%Y-%m-%d"), "sources": [os.path.basename(os.path.normpath(d)) for d in a.pairs], "pairs": len(pairs), "train": len(train), "held_out_n": len(hold),
           "held_out_zones": by_zone(hold), "train_zones": by_zone(train), "harvest_engines": engines,
           "note": "scored as an unmerged fp32 LoRA on the plain forward (no DeepCache, no cross-frame attention)",
           "passes_median": float(np.median([p["passes"] for p in pairs])), "steps": len(losses), "rank": a.rank, "layers": a.layers,
           "lr": a.lr, "rest_share": a.rest, "lora_projections": len(loras), "lora_params_m": round(sum(p.numel() for p in params) / 1e6, 3),
           "step_s_median": round(float(np.median(times)), 3) if times else None, "eval_only": a.eval_only and "/".join(os.path.normpath(a.eval_only).split(os.sep)[-2:]),
           "skipped_steps": skipped if not a.eval_only else None,
           "loss_first100": round(float(np.mean(losses[:100])), 5) if losses else None,
           "loss_last100": round(float(np.mean(losses[-100:])), 5) if losses else None,
           "held_out": {"l2": {"stock": m(stock, "l2"), "walk": m(stock, "walk_l2"), "lora": m(tuned, "l2")},
                        "lpips": {"stock": m(stock, "lpips"), "walk": m(stock, "walk_lpips"), "lora": m(tuned, "lpips")},
                        "detail": {"stock": m(stock, "detail"), "walk": m(stock, "detail_walk"), "lora": m(tuned, "detail"),
                                   "target": m(stock, "detail_target")},
                        "rest_l2": {"stock": m(stock, "rest_l2"), "lora": m(tuned, "rest_l2")},
                        "rest_lpips": {"stock": m(stock, "rest_lpips"), "lora": m(tuned, "rest_lpips")},
                        "rest_detail": {"stock": m(stock, "detail_rest"), "lora": m(tuned, "detail_rest")}},
           "engine": {"depth_graft": eng.depth_graft, "deepcache": 0, "xframe": 0, "device": dev},
           "per_pair": {"stock": stock, "lora": tuned}}
    res["closer_l2"] = round(1 - res["held_out"]["l2"]["lora"] / res["held_out"]["l2"]["stock"], 4)
    res["closer_lpips"] = round(1 - res["held_out"]["lpips"]["lora"] / res["held_out"]["lpips"]["stock"], 4)
    json.dump(res, open(os.path.join(a.out, "one_pass.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_pair"}, indent=1))


if __name__ == "__main__":
    main()
