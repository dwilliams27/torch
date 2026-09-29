#!/usr/bin/env python3
"""Distill "TAESD-lite": TAESD with its two full-resolution stages replaced by half-res
pixel-(un)shuffle stems. TAESD's full-res 64-channel convs are ~50% of its FLOPs and TAESD
was ~35% of an engine frame; this removes most of that.

  encoder: [conv3->64 @1x, Block @1x, conv s2]  ->  [unshuffle2, conv12->64, relu, conv, relu, conv] @1/2x
  decoder: [up2, conv @1x, Block @1x, conv64->3] -> [conv, relu, conv, relu, conv64->12, shuffle2]   @1/2x

Only the new stem/head are trained (everything else stays frozen TAESD weights), against the
TAESD teacher, on procedural naves + SD-Turbo paintings of them + SD-Turbo txt2img images
(i.e. the raw renders and the painted-feedback frames the client actually sends).

    python bench/distill_taesd_lite.py --steps 1500     # -> server/engines/taesd_lite.safetensors
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import scenes  # noqa: E402
from bench_engines import PROMPTS, bench_lock  # noqa: E402

EXTRA_PROMPTS = PROMPTS + [
    "crystal cavern, prismatic refractions, glowing geodes, fantasy matte painting",
    "brutalist atrium with floating monoliths, soft fog, cinematic concrete",
    "desert of arches under two moons, dusk, dreamy watercolor",
    "vertical library of endless stairs, warm candlelight, intricate etching",
    "portrait of an old sailor, dramatic lighting, oil on canvas",
    "lush rainforest waterfall, sunlight, highly detailed photograph",
    "abstract colorful fluid painting, swirls, high contrast",
    "neon city street at night in the rain, reflections",
]


def make_data(eng, n_scenes=64, seed=0):
    rng = np.random.default_rng(seed)
    imgs = []
    for i in range(n_scenes):
        imgs.append(scenes.nave(512, 512, cam=(rng.uniform(-2, 2), rng.uniform(1.2, 2.5), rng.uniform(0, 30)),
                                yaw=rng.uniform(-1.2, 1.2), pitch=rng.uniform(-0.2, 0.5), fov=rng.uniform(55, 100),
                                palette=["blue", "amber", "violet"][i % 3], seed=i))
    painted = []
    for i, im in enumerate(imgs):
        painted.append(eng.process(im, EXTRA_PROMPTS[i % len(EXTRA_PROMPTS)], rng.uniform(0.3, 0.8), int(rng.integers(1 << 30))))
    dreams = []
    noise = (np.random.default_rng(1).random((512, 512, 3)) * 255).astype(np.uint8)
    for i in range(32):
        dreams.append(eng.process(noise, EXTRA_PROMPTS[i % len(EXTRA_PROMPTS)], 1.0, 1000 + i))
    return imgs + painted + dreams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default=os.path.join(ROOT, "server", "engines", "taesd_lite.safetensors"))
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file

    from server.engines import torch_turbo as tt

    eng = tt.create_engine(width=512, height=512, morph=0, deepcache=0, vae="taesd")
    dev = eng.device
    t0 = time.time()
    data = make_data(eng)
    print(f"data: {len(data)} images in {time.time() - t0:.1f}s", flush=True)
    data = torch.from_numpy(np.stack(data)).permute(0, 3, 1, 2).contiguous()  # uint8 N,3,512,512 (cpu)

    teacher = eng.vae.float()   # train in fp32
    for p in teacher.parameters():
        p.requires_grad_(False)
    stem, head = tt.make_taesd_lite_modules()
    stem, head = stem.to(dev).float(), head.to(dev).float()
    enc_rest = teacher.encoder.layers[3:]
    dec_body = teacher.decoder.layers[:15]
    params = list(stem.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    g = torch.Generator().manual_seed(0)
    N = data.shape[0]

    def batch():
        idx = torch.randint(0, N, (a.batch,), generator=g)
        ys = torch.randint(0, 512 - a.crop + 1, (a.batch,), generator=g)
        xs = torch.randint(0, 512 - a.crop + 1, (a.batch,), generator=g)
        crops = torch.stack([data[i, :, y:y + a.crop, x:x + a.crop] for i, y, x in zip(idx, ys, xs)])
        if torch.rand(1, generator=g) < 0.5:
            crops = crops.flip(3)
        return crops.to(dev).float().div(255.0)  # [0,1]

    with bench_lock():  # not a timed bench, but keeps the shared GPU quiet for others' timings
        t0 = time.time()
        for step in range(a.steps):
            x01 = batch()
            with torch.no_grad():
                z_t = teacher.encoder.layers(x01)                    # teacher latents
                feat = dec_body(torch.tanh(z_t / 3) * 3)              # teacher decoder features @1/2
                y_t = teacher.decoder.layers[15:](feat)               # teacher pixels [0,1]
            z_s = enc_rest(stem(x01))
            y_s = head(feat)
            loss_e = F.mse_loss(z_s, z_t)
            loss_d = F.l1_loss(y_s, y_t) + F.mse_loss(y_s, y_t)
            loss = loss_e + loss_d
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 100 == 0 or step == a.steps - 1:
                print(f"step {step:5d}  enc_mse {loss_e.item():.4f}  dec_l1+l2 {loss_d.item():.4f}  "
                      f"{time.time() - t0:.0f}s", flush=True)
    sd = {f"stem.{k}": v.detach().half().cpu().contiguous() for k, v in stem.state_dict().items()}
    sd.update({f"head.{k}": v.detach().half().cpu().contiguous() for k, v in head.state_dict().items()})
    save_file(sd, a.out)
    print("wrote", a.out, sum(v.numel() for v in sd.values()), "params")


if __name__ == "__main__":
    main()
