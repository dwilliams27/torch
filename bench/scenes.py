"""Procedural test inputs for the diffusion benches: dark architectural interiors.

A tiny vectorized numpy ray tracer (planes + vertical cylinders) that renders
something shaped like what the game sends the model: a dark nave with columns,
arched side openings, colored light pools, a luminous far window, and fog.
`nave(...)` renders one frame; `walk(n)` renders a short forward/turning camera
path so benches exercise realistic, temporally coherent input.
"""
from __future__ import annotations

import numpy as np


def _norm(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def nave(w=512, h=512, cam=(0.0, 1.6, 0.0), yaw=0.0, pitch=0.08, fov=75.0,
         lights=None, palette="blue", seed=0, with_depth=False):
    """uint8 (h, w, 3); with_depth: also a depth channel shaped like the client's (latent size
    (h/8, w/8), [-1, 1], near = 1), but min-max scaled inverse ray distance, where the client sends
    the 2nd-98th percentile of inverse view depth in 8 bits."""
    rng = np.random.default_rng(seed)
    W, Hc, Zf, SP, PR, PX = 4.0, 9.0, 60.0, 6.0, 0.45, 2.6  # half-width, ceiling, far wall, spacing, pillar r, pillar x
    if lights is None:
        cols = {"blue": [(0.2, 0.55, 1.0), (1.0, 0.55, 0.2), (0.3, 1.0, 0.7)],
                "amber": [(1.0, 0.6, 0.25), (0.9, 0.3, 0.2), (1.0, 0.85, 0.5)],
                "violet": [(0.7, 0.3, 1.0), (0.2, 0.8, 1.0), (1.0, 0.3, 0.6)]}[palette]
        lights = []
        for k in range(8):
            z = 3.0 + k * SP
            side = -1 if k % 2 else 1
            lights.append(((side * (PX - 0.8), 2.8, z), cols[k % 3], 7.0))
        lights.append(((0.0, 6.0, Zf - 3), (1.0, 0.9, 0.8), 30.0))
    o = np.array(cam, dtype=np.float64)
    fwd = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    right = _norm(np.cross(np.array([0, 1.0, 0]), fwd))
    up = np.cross(fwd, right)
    th = np.tan(np.radians(fov) / 2)
    xs = (np.arange(w) + 0.5) / w * 2 - 1
    ys = 1 - (np.arange(h) + 0.5) / h * 2
    U, V = np.meshgrid(xs * th * (w / h), ys * th)
    d = _norm(U[..., None] * right + V[..., None] * up + fwd)
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    inf = np.full(dx.shape, np.inf)
    best = inf.copy()
    mat = np.zeros(dx.shape, np.int8)  # 0 floor 1 ceil 2 wall 3 pillar 4 far 5 aisle-glow
    with np.errstate(divide="ignore", invalid="ignore"):
        for t, m in (((0 - o[1]) / dy, 0), ((Hc - o[1]) / dy, 1), ((W - o[0]) / dx, 2), ((-W - o[0]) / dx, 2),
                     ((Zf - o[2]) / dz, 4)):
            ok = (t > 1e-4) & (t < best)
            best = np.where(ok, t, best)
            mat = np.where(ok, m, mat)
        # pillars: vertical cylinders
        a = dx * dx + dz * dz
        for k in range(12):
            for sx in (-1, 1):
                cx, cz = sx * PX, 3.0 + k * SP
                ox, oz = o[0] - cx, o[2] - cz
                b = 2 * (ox * dx + oz * dz)
                c = ox * ox + oz * oz - PR * PR
                disc = b * b - 4 * a * c
                t = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
                ok = (disc > 0) & (t > 1e-4) & (t < best)
                best = np.where(ok, t, best)
                mat = np.where(ok, 3, mat)
    P = o + d * best[..., None]
    px, py, pz = P[..., 0], P[..., 1], P[..., 2]
    n = np.zeros_like(P)
    n[mat == 0] = (0, 1, 0)
    n[mat == 1] = (0, -1, 0)
    n[mat == 4] = (0, 0, -1)
    wm = mat == 2
    n[wm] = np.stack([-np.sign(px[wm]), 0 * px[wm], 0 * px[wm]], -1)
    pm = mat == 3
    cx = np.sign(px[pm]) * PX
    cz = 3.0 + np.round((pz[pm] - 3.0) / SP) * SP
    n[pm] = _norm(np.stack([px[pm] - cx, 0 * cx, pz[pm] - cz], -1))
    # arched openings in the side walls -> glowing aisles
    zz = np.mod(pz - 3.0 - SP / 2, SP) - SP / 2
    arch = wm & (np.abs(zz) < 1.6) & (py < 3.6 + np.sqrt(np.maximum(1.6 ** 2 - zz ** 2, 0)))
    mat = np.where(arch, 5, mat)
    # albedo / patterns
    alb = np.full(P.shape, 0.5)
    tile = ((np.floor(px) + np.floor(pz)) % 2)
    grout = (np.abs(np.mod(px, 1) - 0.5) > 0.47) | (np.abs(np.mod(pz, 1) - 0.5) > 0.47)
    alb[mat == 0] = (0.55 + 0.15 * tile[mat == 0])[:, None] * np.array([0.8, 0.78, 0.74])
    alb[(mat == 0) & grout] = (0.15, 0.14, 0.13)
    row = np.floor(py / 0.4)
    brick = (np.abs(np.mod(pz + row * 0.4, 0.8) - 0.4) > 0.37) | (np.abs(np.mod(py, 0.4) - 0.2) > 0.18)
    alb[mat == 2] = (0.45, 0.43, 0.46)
    alb[(mat == 2) & brick] = (0.2, 0.19, 0.2)
    ang = np.arctan2(n[..., 2], n[..., 0])
    alb[mat == 3] = (0.6 + 0.2 * np.cos(ang[mat == 3] * 16))[:, None] * np.array([0.85, 0.82, 0.78])
    alb[mat == 1] = (0.25, 0.25, 0.3)
    alb[mat == 4] = (0.3, 0.3, 0.32)
    grain = np.mod(np.sin(np.floor(px * 24) * 12.9898 + np.floor(py * 24) * 78.233 + np.floor(pz * 24) * 37.719) * 43758.5453, 1.0)
    alb *= (0.9 + 0.2 * grain)[..., None]  # world-space grain: stable under camera motion
    # lighting
    col = np.zeros_like(P)
    for lp, lc, li in lights:
        L = np.array(lp) - P
        dist2 = np.sum(L * L, -1)
        L = L / np.sqrt(dist2)[..., None]
        ndl = np.clip(np.sum(n * L, -1), 0, 1)
        col += alb * np.array(lc) * (li * ndl / (1 + dist2))[..., None]
    col += alb * 0.03
    # emissives: far window (tall lancet) and aisle glow
    win = (mat == 4) & (np.abs(px) < 1.6) & (py > 1.5) & (py < 6.5 + np.sqrt(np.maximum(1.6 ** 2 - px ** 2, 0)))
    col[win] = np.array([1.6, 1.3, 1.0]) * (1.0 + 0.3 * np.sin(px[win] * 9) * np.sin(py[win] * 7))[:, None]
    ai = mat == 5
    glow = {"blue": (0.1, 0.35, 0.8), "amber": (0.9, 0.4, 0.1), "violet": (0.5, 0.1, 0.8)}[palette]
    col[ai] = np.array(glow) * (0.4 + 0.6 * np.clip(1 - py[ai] / 6, 0, 1))[:, None]
    # contour lines (the "blueprint" look of the undreamt world)
    edge = (np.abs(np.mod(py, 3.0)) < 0.03) & (mat == 2)
    col[edge] += 0.15
    # fog
    fogc = {"blue": (0.03, 0.06, 0.12), "amber": (0.12, 0.07, 0.03), "violet": (0.08, 0.03, 0.12)}[palette]
    f = np.exp(-np.minimum(best, 200) * 0.045)[..., None]
    col = col * f + np.array(fogc) * (1 - f)
    col = col / (1 + col)  # reinhard
    img = np.clip(col, 0, 1) ** (1 / 2.2)
    img = (img * 255 + 0.5).astype(np.uint8)
    if not with_depth:
        return img
    inv = 1.0 / np.maximum(np.minimum(best, 200.0), 0.1)
    inv = inv[: h // 8 * 8, : w // 8 * 8].reshape(h // 8, 8, w // 8, 8).mean((1, 3))
    lo, hi = float(inv.min()), float(inv.max())
    return img, ((inv - lo) / max(hi - lo, 1e-6) * 2 - 1).astype(np.float32)


def walk(n=24, w=512, h=512, palette="blue"):
    """A short camera path: walk forward while slowly turning (temporally coherent input)."""
    frames = []
    for i in range(n):
        s = i / max(n - 1, 1)
        frames.append(nave(w, h, cam=(0.6 * np.sin(s * 3), 1.6, s * 5.0), yaw=0.35 * np.sin(s * 2.5),
                           pitch=0.08 + 0.05 * np.sin(s * 4), palette=palette, seed=i))
    return frames


def turn(n=24, w=512, h=512, palette="amber", rate=0.12):
    """Fast yaw turn (rate rad/frame; 0.12 ~ 7 deg/frame ~ 100 deg/s at 14 FPS): stresses temporal reuse."""
    return [nave(w, h, cam=(0.0, 1.6, 6.0), yaw=-0.8 + rate * i, pitch=0.1, palette=palette, seed=i) for i in range(n)]


def test_set(w=512, h=512, with_depth=False):
    """Three stills with distinct palettes/views for quality contact sheets (with_depth: (img, depth) pairs)."""
    return [
        nave(w, h, palette="blue", with_depth=with_depth),
        nave(w, h, cam=(-1.5, 1.6, 8.0), yaw=0.7, pitch=0.25, palette="amber", seed=1, with_depth=with_depth),
        nave(w, h, cam=(1.0, 1.6, 20.0), yaw=-0.4, pitch=0.35, fov=90, palette="violet", seed=2, with_depth=with_depth),
    ]


if __name__ == "__main__":
    import sys

    from PIL import Image

    out = sys.argv[1] if len(sys.argv) > 1 else "scenes.jpg"
    ims = test_set(384, 384)
    Image.fromarray(np.concatenate(ims, 1)).save(out, quality=88)
    print("wrote", out)
