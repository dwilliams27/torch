"""torch_turbo -- 1-step img2img diffusion engine for PyTorch (MPS / CUDA / CPU).

Hand-rolled single-step img2img: no diffusers pipeline objects at runtime, just

    uint8 HWC  --upload-->  [-1,1] NCHW fp16
               --tiny VAE encode-->  latent x0            (TAESD / SDXS tiny AE)
               --q_sample(x0, t, fixed_noise[seed])-->  x_t
               --UNet(x_t, t, cached_text_emb)-->  eps     (lean forward, ToDo attention, DeepCache)
               --x0_hat = (x_t - sqrt(1-a_t) eps) / sqrt(a_t)-->
               --tiny VAE decode-->  uint8 HWC  (quantized on the GPU, one readback)

Measured (bench/RESULTS.md), end to end, M4 Pro mini: 384 -> 60 ms = 16.6 FPS, 512 -> 93 ms =
10.7 FPS, 320 -> 22.4 FPS (the plain diffusers-equivalent path: 5.2 FPS @512). M2 Pro laptop @384: see RESULTS.

Speed tricks, all measured on MPS:
  * TAESD-lite (default when server/engines/taesd_lite.safetensors exists): TAESD with its two
    full-resolution stages replaced by distilled half-res pixel-(un)shuffle stems (161k params,
    bench/distill_taesd_lite.py): encode+decode 43 -> 22 ms @512 at equal round-trip PSNR.
  * ToDo (token-downsampled self-attention K/V at the 64x64 latent level): -12% UNet.
  * Temporal DeepCache: every `deepcache`-th frame runs the whole UNet and caches the deep
    features that enter the last up block; the frames in between only run conv_in + the top
    down level + the last up block (~25% of the FLOPs). Skip connections are always fresh, so
    fine structure tracks the current frame; only deep semantics lag <= N-1 frames -- which
    measurably *reduces* frame-to-frame flicker (bench/temporal.py).
  * lean UNet forward (no diffusers bookkeeping), cross-attn K/V cached per conditioning.
  * Tried and rejected: channels_last (+33%), torch.compile/inductor-MPS (+43%), SDXS (fast
    but its 1-step UNet can't do img2img), linear patch encoder (free but blurs structure).

Why each piece:
  * 1 UNet eval, no CFG (turbo/SDXS models are trained guidance-free; `negative`
    is accepted and ignored).
  * Tiny autoencoder instead of the 83M-param KL VAE: the KL decoder alone costs
    more than the whole SDXS UNet.
  * Prompt embeddings are LRU-cached; when the prompt changes (zone transition)
    the conditioning glides from the old embedding to the new one over
    `morph` seconds, so styles melt into each other instead of snapping.
  * Noise is a fixed tensor per (seed, latent size): identical noise on every
    frame is the single biggest temporal-coherence win for img2img video.
  * strength -> continuous timestep t = strength * 999 with exact DDPM
    q_sample / x0 math (matches diffusers Euler 'trailing' 1-step img2img).
  * Inputs whose size is not a multiple of 64 are reflect-padded on the GPU and
    cropped back, so output pixels stay aligned with input pixels (the client
    projects the result back onto geometry, so alignment is sacred).

Model presets (cfg key `model`):
  sd-turbo    stabilityai/sd-turbo         (default; 865M SD2.1 ADD-distilled UNet -- the only one
                                            that does real img2img at intermediate timesteps)
  sdxs        IDKiro/sdxs-512-dreamshaper  (tiny 1-step UNet, 2x faster, but only valid at t=999:
                                            img2img comes out as grey mud -- see bench/RESULTS.md)
  sdxs-0.9    IDKiro/sdxs-512-0.9
  any other string is treated as a diffusers repo id / local path.

cfg keys: model, width, height, device, dtype('fp16'|'fp32'|'bf16'),
          vae('auto'|'taesd'|'lite'|'lite-enc'|'lite-dec'|'model'|'full'; lite = TAESD with distilled
              half-res stems, bench/distill_taesd_lite.py),
          channels_last(bool, slower on MPS), compile(bool), morph(float seconds), t_snap(bool),
          attn('todo' token-downsampled self-attn K/V [default] | 'fast' | 'sdpa' diffusers default),
          attn_levels(int, how many top UNet resolutions get ToDo; default 1),
          deepcache(int N: full UNet every N frames, cheap shallow-only passes in between; default 3; 0 = off),
          dc_branch(int, how many top UNet levels the cheap pass recomputes; default 1),
          dc_thresh(float, scene-cut guard: mean |pool4(thumb) - pool4(thumb at last full pass)| in 0..1
                    RGB above which a cheap pass is upgraded to a full one; default 0.06),
          dc_max_shift(int, motion guard: global shift (latent px = 8 image px) since the full pass above
                    which a full pass is forced -- fast look-around recomputes; default 3),
          dc_motion(bool, translate the cached deep features by that shift instead; better edge alignment
                    but smears newly revealed borders -- off by default),
          encoder('taesd' | 'linear' = least-squares 8x8-patch encoder, ~free, see bench/fit_linear_encoder.py)
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections import OrderedDict

import numpy as np

try:  # torch is optional at import time so `--engine auto` can probe cheaply
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None

PRESETS = {
    "sdxs": dict(repo="IDKiro/sdxs-512-dreamshaper", vae="model", native=512),
    "sdxs-dreamshaper": dict(repo="IDKiro/sdxs-512-dreamshaper", vae="model", native=512),
    "sdxs-0.9": dict(repo="IDKiro/sdxs-512-0.9", vae="model", native=512),
    "sd-turbo": dict(repo="stabilityai/sd-turbo", vae="taesd", native=512, variant="fp16"),
}
TAESD_REPO = "madebyollin/taesd"


def _pick_device(req: str | None) -> str:
    if req and req != "auto":
        return req
    if torch is not None and torch.backends.mps.is_available():
        return "mps"
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _machine_default_size() -> int:
    """Default capture size per machine class (measured, see bench/RESULTS.md):
    M4 Pro mini 384 -> ~16 FPS, M2 Pro laptop 384 -> ~12 FPS; Max/Ultra-class GPUs can afford 512."""
    try:
        import subprocess

        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode()
    except Exception:
        return 384
    if any(k in chip for k in ("Max", "Ultra")):
        return 512
    return 384


def nn_seq(*mods):
    import torch.nn as nn

    return nn.Sequential(*mods)


def _sync(device: str):
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


class _FastAttn:
    """Attention processor for UNet transformer blocks.

    * self-attention at the top `levels` resolutions uses token-downsampled K/V
      (ToDo, Smith et al. 2024): K,V computed from a 2x2 avg-pooled feature map,
      so the 4096x4096 attention at 64x64 latents becomes 4096x1024. Queries keep
      full resolution, so structure/alignment is untouched.
    * cross-attention K/V (77 text tokens) are cached per conditioning tensor.
    """

    def __init__(self, shape_ref: dict, mode: str = "todo", levels: int = 1, factor: int = 2):
        self.shape_ref, self.mode, self.levels, self.factor = shape_ref, mode, levels, factor
        self._ehs = None
        self._kv = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *a, **kw):
        if hidden_states.ndim != 3 or attention_mask is not None or attn.group_norm is not None \
                or getattr(attn, "spatial_norm", None) is not None or attn.norm_cross:
            raise RuntimeError("_FastAttn: unsupported attention configuration")
        B, N, C = hidden_states.shape
        heads = attn.heads
        q = attn.to_q(hidden_states)
        if encoder_hidden_states is not None:
            if self._ehs is encoder_hidden_states:
                k, v = self._kv
            else:
                k, v = attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)
                self._ehs, self._kv = encoder_hidden_states, (k, v)
        else:
            src = hidden_states
            h0, w0 = self.shape_ref["hw"]
            lvl = int(round(math.log(max(h0 * w0 / N, 1.0), 4)))
            h, w = h0 >> lvl, w0 >> lvl
            if self.mode != "none" and lvl < self.levels and h * w == N and h % self.factor == 0 and w % self.factor == 0:
                if self.mode == "skip":  # upper bound only (not a real mode)
                    o = attn.to_out[1](attn.to_out[0](attn.to_v(hidden_states)))
                    return o / attn.rescale_output_factor
                x = hidden_states.transpose(1, 2).reshape(B, C, h, w)
                src = F.avg_pool2d(x, self.factor).flatten(2).transpose(1, 2)
            k, v = attn.to_k(src), attn.to_v(src)
        hd = q.shape[-1] // heads
        q = q.view(B, -1, heads, hd).transpose(1, 2)
        k = k.view(k.shape[0], -1, heads, hd).transpose(1, 2)
        v = v.view(v.shape[0], -1, heads, hd).transpose(1, 2)
        if k.shape[0] != B:
            k, v = k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1)
        o = F.scaled_dot_product_attention(q, k, v, scale=attn.scale)
        o = o.transpose(1, 2).reshape(B, -1, heads * hd)
        o = attn.to_out[1](attn.to_out[0](o))
        if attn.residual_connection:
            raise RuntimeError("_FastAttn: residual_connection unsupported")
        return o / attn.rescale_output_factor


def install_fast_attention(unet, latent_hw, mode: str = "todo", levels: int = 1, factor: int = 2) -> dict:
    """Swap every UNet attention processor for _FastAttn. Returns the shared shape dict
    (update ['hw'] when the latent size changes)."""
    ref = {"hw": tuple(latent_hw)}
    unet.set_attn_processor({name: _FastAttn(ref, mode, levels, factor) for name in unet.attn_processors})
    return ref


def _down_no_ds(blk, h, emb, ehs):
    """Run a UNet down block's resnet/attention pairs but not its downsampler."""
    out = ()
    attns = getattr(blk, "attentions", None)
    for i, resnet in enumerate(blk.resnets):
        h = resnet(h, emb)
        if attns is not None:
            h = attns[i](h, encoder_hidden_states=ehs, return_dict=False)[0]
        out += (h,)
    return h, out


def lean_unet_forward(unet, sample, timestep, ehs, cache: dict | None = None, reuse: bool = False,
                      branch: int = 1, shift: tuple[int, int] = (0, 0)):
    """Minimal UNet2DConditionModel forward for SD1/SD2-family configs (no CFG, masks,
    adapters, ...), with temporal DeepCache (Ma et al. 2023, applied across video frames):

      full pass  (reuse=False): normal UNet; stores cache['deep'] = the input of the first
                                of the last `branch` up blocks (deep, low-res semantics).
      cheap pass (reuse=True):  conv_in + the top `branch` down levels (no downsampler) +
                                the last `branch` up blocks fed with cache['deep'].
    Skip connections at the kept levels are always fresh, so fine structure tracks the
    current frame exactly; only the deep semantic features lag by <= N-1 frames.
    """
    t_emb = unet.get_time_embed(sample=sample, timestep=timestep)
    emb = unet.time_embedding(t_emb, None)
    if unet.time_embed_act is not None:
        emb = unet.time_embed_act(emb)
    h = unet.conv_in(sample)
    res = (h,)
    n = len(unet.down_blocks)
    for i, blk in enumerate(unet.down_blocks):
        if reuse and i == branch - 1:
            h, r = _down_no_ds(blk, h, emb, ehs)
            res += r
            break
        if getattr(blk, "has_cross_attention", False):
            h, r = blk(hidden_states=h, temb=emb, encoder_hidden_states=ehs)
        else:
            h, r = blk(hidden_states=h, temb=emb)
        res += r
    if reuse:
        h = cache["deep"]
        if shift != (0, 0):  # motion-compensate the stale deep features (global translation)
            f = h.shape[-1] // sample.shape[-1] if h.shape[-1] >= sample.shape[-1] else 0
            if f:
                h = _shift2d(h, shift[0] * f, shift[1] * f)
            else:  # deep map coarser than the latent: shift by the rounded fraction
                g = sample.shape[-1] // h.shape[-1]
                h = _shift2d(h, int(round(shift[0] / g)), int(round(shift[1] / g)))
        first_up = n - branch
    else:
        first_up = 0
        if unet.mid_block is not None:
            if getattr(unet.mid_block, "has_cross_attention", False):
                h = unet.mid_block(h, emb, encoder_hidden_states=ehs)
            else:
                h = unet.mid_block(h, emb)
    for j in range(first_up, len(unet.up_blocks)):
        up = unet.up_blocks[j]
        if cache is not None and not reuse and j == n - branch:
            cache["deep"] = h
        k = len(up.resnets)
        rs, res = res[-k:], res[:-k]
        if getattr(up, "has_cross_attention", False):
            h = up(hidden_states=h, temb=emb, res_hidden_states_tuple=rs, encoder_hidden_states=ehs)
        else:
            h = up(hidden_states=h, temb=emb, res_hidden_states_tuple=rs)
    if unet.conv_norm_out is not None:
        h = unet.conv_act(unet.conv_norm_out(h))
    return unet.conv_out(h)


def make_taesd_lite_modules():
    """Half-res replacements for TAESD's full-res stages (trained by bench/distill_taesd_lite.py).
    stem: [0,1] RGB @1x -> features @1/2x (replaces encoder.layers[:3])
    head: decoder features @1/2x -> [0,1] RGB @1x (replaces decoder.layers[15:])"""
    import torch.nn as nn

    stem = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(12, 64, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1))
    head = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(64, 12, 3, padding=1), nn.PixelShuffle(2))
    return stem, head


def _phase_shift(a: "np.ndarray", b: "np.ndarray") -> tuple[int, int]:
    """Integer (dy, dx) such that b ~= a shifted by (dy, dx); phase correlation, (C,H,W) float arrays."""
    A = np.fft.rfft2(a)
    B = np.fft.rfft2(b)
    R = (B * np.conj(A)).sum(0)
    R /= np.abs(R) + 1e-6
    r = np.fft.irfft2(R, s=a.shape[-2:])
    iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)
    H, W = r.shape
    return (iy - H if iy > H // 2 else iy), (ix - W if ix > W // 2 else ix)


def _thumb(img: "np.ndarray") -> "np.ndarray":
    """(3, H/8, W/8) float32 [0,1] block-mean thumbnail on the CPU (one value per latent pixel)."""
    H, W = img.shape[0] // 8 * 8, img.shape[1] // 8 * 8
    t = img[:H:2, :W:2].reshape(H // 8, 4, W // 8, 4, 3).mean((1, 3), dtype=np.float32)
    return (t * (1.0 / 255.0)).transpose(2, 0, 1)


def _pool4(t: "np.ndarray") -> "np.ndarray":
    C, H, W = t.shape
    return t[:, :H // 4 * 4, :W // 4 * 4].reshape(C, H // 4, 4, W // 4, 4).mean((2, 4))


def _shift2d(x, dy: int, dx: int):
    """Translate a (N,C,H,W) map by (dy, dx) pixels with edge replication."""
    if dy == 0 and dx == 0:
        return x
    H, W = x.shape[-2:]
    dy, dx = max(-H + 1, min(H - 1, dy)), max(-W + 1, min(W - 1, dx))
    x = F.pad(x, (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0)), mode="replicate")
    return x[..., max(-dy, 0):max(-dy, 0) + H, max(-dx, 0):max(-dx, 0) + W]


def _from_pretrained(cls, repo, dtype, **kw):
    """diffusers/transformers renamed torch_dtype->dtype at some point; try both."""
    try:
        return cls.from_pretrained(repo, torch_dtype=dtype, **kw)
    except TypeError:
        return cls.from_pretrained(repo, dtype=dtype, **kw)


class TorchTurboEngine:
    def __init__(self, model: str = "sd-turbo", width: int | None = None, height: int | None = None,
                 device: str | None = None, dtype: str = "fp16", vae: str = "auto",
                 channels_last: bool = False, compile: bool = False, morph: float = 1.0,
                 t_snap: bool = False, max_prompts: int = 64, attn: str = "todo", attn_levels: int = 1,
                 deepcache: int = 3, dc_branch: int = 1, dc_thresh: float = 0.06, dc_max_shift: int = 3,
                 dc_motion: bool = False,
                 encoder: str = "taesd",
                 **_ignored):
        if torch is None:
            raise RuntimeError("torch not installed")
        from diffusers import AutoencoderKL, AutoencoderTiny, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        self.device = _pick_device(device)
        dev = self.device
        if dtype == "fp32" or dev == "cpu":
            self.dtype = torch.float32
        elif dtype == "bf16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16
        model = model or "sd-turbo"
        preset = PRESETS.get(model, dict(repo=model, vae="taesd", native=512))
        self.repo = preset["repo"]
        self.model = self.repo
        self.name = "torch-" + (model if model in PRESETS else os.path.basename(model.rstrip("/")))
        size = _machine_default_size()
        self.width = int(width or size)
        self.height = int(height or width or size)
        self.channels_last = bool(channels_last)
        self.morph = float(morph)
        self.t_snap = bool(t_snap)
        self.profile = False          # bench sets this: sync + time every stage
        self.last_timings: dict = {}
        self._lock = threading.Lock()
        variant = preset.get("variant")

        t0 = time.time()
        kw = dict(variant=variant) if variant else {}
        self.tokenizer = CLIPTokenizer.from_pretrained(self.repo, subfolder="tokenizer")
        self.text_encoder = _from_pretrained(CLIPTextModel, self.repo, self.dtype, subfolder="text_encoder", **kw).to(dev).eval()
        self.unet = _from_pretrained(UNet2DConditionModel, self.repo, self.dtype, subfolder="unet", **kw).to(dev).eval()
        vae_kind = preset["vae"] if vae == "auto" else vae
        _lite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taesd_lite.safetensors")
        if vae == "auto" and vae_kind == "taesd" and os.path.exists(_lite_path):
            vae_kind = "lite"          # distilled half-res TAESD stems: -50% VAE time, ~same quality
        if vae_kind == "model":        # the model repo ships its own tiny AE (SDXS)
            self.vae = _from_pretrained(AutoencoderTiny, self.repo, self.dtype, subfolder="vae")
        elif vae_kind == "full":       # KL VAE -- quality reference only, slow
            sub = "vae_large" if "sdxs" in self.repo.lower() else "vae"
            self.vae = _from_pretrained(AutoencoderKL, self.repo, self.dtype, subfolder=sub, **({} if sub == "vae_large" else kw))
        else:  # 'taesd', or 'lite'/'lite-enc'/'lite-dec' = TAESD with distilled half-res stems
            self.vae = _from_pretrained(AutoencoderTiny, TAESD_REPO, self.dtype)
        self.vae = self.vae.to(dev).eval()
        self._lite_enc = self._lite_dec = None
        if vae_kind.startswith("lite"):
            try:
                from safetensors.torch import load_file

                sd = load_file(_lite_path)
                stem, head = make_taesd_lite_modules()
                stem.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("stem.")})
                head.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("head.")})
                if vae_kind in ("lite", "lite-enc"):
                    self._lite_enc = nn_seq(stem.to(dev, self.dtype).eval(), self.vae.encoder.layers[3:])
                if vae_kind in ("lite", "lite-dec"):
                    self._lite_dec = nn_seq(self.vae.decoder.layers[:15], head.to(dev, self.dtype).eval())
            except Exception as e:  # missing/corrupt weights -> plain TAESD
                print(f"[torch_turbo] TAESD-lite unavailable ({e}); using TAESD")
                vae_kind = "taesd"
        self.vae_kind = vae_kind
        self.vae_is_tiny = isinstance(self.vae, AutoencoderTiny)
        self.vae_scale = 1.0 if self.vae_is_tiny else float(self.vae.config.scaling_factor)
        self.encoder = "taesd"
        if encoder == "linear" and self.vae_is_tiny:
            p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taesd_linear_enc.npz")
            if os.path.exists(p):
                d = np.load(p)
                self._lin_w = torch.from_numpy(d["weight"]).reshape(4, -1, 1, 1).to(dev, self.dtype)
                self._lin_b = torch.from_numpy(d["bias"]).to(dev, self.dtype)
                self.encoder = "linear"
            else:
                print(f"[torch_turbo] {p} missing; using the TAESD encoder")
        self.load_s = time.time() - t0

        for m in (self.unet, self.vae, self.text_encoder):
            m.requires_grad_(False)
        if self.channels_last:
            self.unet.to(memory_format=torch.channels_last)
            self.vae.to(memory_format=torch.channels_last)

        # noise schedule (scaled_linear, as in every SD1/2-family scheduler config)
        from diffusers import EulerDiscreteScheduler

        cfg = EulerDiscreteScheduler.load_config(self.repo, subfolder="scheduler")
        self.prediction_type = cfg.get("prediction_type", "epsilon")
        betas = torch.linspace(cfg.get("beta_start", 0.00085) ** 0.5, cfg.get("beta_end", 0.012) ** 0.5,
                               cfg.get("num_train_timesteps", 1000), dtype=torch.float64) ** 2
        self.alphas_cumprod = torch.cumprod(1.0 - betas, 0)
        self.num_train_timesteps = int(cfg.get("num_train_timesteps", 1000))

        self.attn = attn
        self._attn_ref = None
        if attn in ("todo", "fast", "skip"):
            self._attn_ref = install_fast_attention(self.unet, (self.height // 8, self.width // 8),
                                                    mode={"fast": "none"}.get(attn, attn), levels=int(attn_levels))
        c = self.unet.config
        self._lean_ok = not (c.get("addition_embed_type") or c.get("class_embed_type") or c.get("center_input_sample")
                             or c.get("time_cond_proj_dim") or c.get("encoder_hid_dim_type"))
        self.deepcache = int(deepcache) if self._lean_ok else 0
        self.dc_branch = int(dc_branch)
        self.dc_thresh = float(dc_thresh)
        self.dc_motion = bool(dc_motion)
        self.dc_max_shift = int(dc_max_shift)
        self.last_dc_shift = (0, 0)
        self.last_dc_diff = 0.0
        self._dc: dict = {}
        self._unet_call = lean_unet_forward if self._lean_ok else None
        if compile:
            try:
                self._unet_call = torch.compile(lean_unet_forward) if self._lean_ok else None
                self._compiled_unet = None if self._lean_ok else torch.compile(self.unet)
            except Exception as e:  # pragma: no cover
                print(f"[torch_turbo] torch.compile failed, eager: {e}")

        self._emb_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._max_prompts = max_prompts
        self._noise_cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
        self._t_cache: dict = {}
        self._morph_state: dict = {}   # seed -> dict(key, emb, t)
        self.fps_estimate = None

    # ------------------------------------------------------------------ caches
    @torch.inference_mode()
    def _embed(self, prompt: str) -> "torch.Tensor":
        e = self._emb_cache.get(prompt)
        if e is not None:
            self._emb_cache.move_to_end(prompt)
            return e
        ids = self.tokenizer(prompt or "", padding="max_length", max_length=self.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids.to(self.device)
        e = self.text_encoder(ids)[0].to(self.dtype)
        self._emb_cache[prompt] = e
        if len(self._emb_cache) > self._max_prompts:
            self._emb_cache.popitem(last=False)
        return e

    def _conditioning(self, prompt: str, seed: int) -> "torch.Tensor":
        target = self._embed(prompt)
        if self.morph <= 0:
            return target
        now = time.monotonic()
        st = self._morph_state.get(seed)
        if st is None or st["emb"].shape != target.shape:
            self._morph_state[seed] = dict(key=prompt, emb=target, t=now, settled=True)
            if len(self._morph_state) > 16:
                self._morph_state.pop(next(iter(self._morph_state)))
            return target
        dt = now - st["t"]
        st["t"] = now
        if st["key"] != prompt:
            st["key"] = prompt
            st["settled"] = False
            st["w"] = 0.0
        if st["settled"]:
            st["emb"] = target
            return target
        a = 1.0 - math.exp(-max(dt, 1e-3) / self.morph)
        st["w"] = st.get("w", 0.0) + (1.0 - st.get("w", 0.0)) * a
        if st["w"] > 0.985:
            st["emb"], st["settled"] = target, True
            return target
        st["emb"] = torch.lerp(st["emb"], target, a)
        return st["emb"]

    def _noise(self, seed: int, shape) -> "torch.Tensor":
        key = (int(seed), tuple(shape))
        n = self._noise_cache.get(key)
        if n is None:
            g = torch.Generator("cpu").manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
            n = torch.randn(shape, generator=g, dtype=torch.float32).to(self.device, self.dtype)
            if self.channels_last:
                n = n.contiguous(memory_format=torch.channels_last)
            self._noise_cache[key] = n
            if len(self._noise_cache) > 8:
                self._noise_cache.popitem(last=False)
        return n

    def _timestep(self, strength: float):
        s = float(min(max(strength, 0.02), 1.0))
        t = s * (self.num_train_timesteps - 1)
        if self.t_snap:  # ADD student timesteps (sd-turbo was distilled on these)
            t = min((999, 749, 499, 249), key=lambda v: abs(v - t))
        t = int(round(t))
        c = self._t_cache.get(t)
        if c is None:
            ab = float(self.alphas_cumprod[t])
            c = (torch.tensor(t, device=self.device, dtype=torch.float32), math.sqrt(ab), math.sqrt(1.0 - ab), t)
            self._t_cache[t] = c
        return c

    # ------------------------------------------------------------------ core
    def _mark(self, name, t_prev):
        if not self.profile:
            return t_prev
        _sync(self.device)
        now = time.perf_counter()
        self.last_timings[name] = (now - t_prev) * 1000.0
        return now

    @torch.inference_mode()
    def process(self, image, prompt: str, strength: float, seed: int, negative: str | None = None):
        with self._lock:
            return self._process(image, prompt, strength, seed)

    def _process(self, image, prompt, strength, seed):
        dev, dt = self.device, self.dtype
        t0 = time.perf_counter()
        tp = t0
        if self.profile:
            self.last_timings = {}
        img = np.ascontiguousarray(image[..., :3] if image.ndim == 3 and image.shape[2] > 3 else image)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if not img.flags.writeable:  # torch.from_numpy warns on read-only arrays (e.g. np.asarray(PIL))
            img = img.copy()
        H, W = img.shape[:2]
        cond = self._conditioning(prompt, seed)
        tp = self._mark("prompt", tp)

        thumb = _thumb(img) if self.deepcache >= 2 else None  # CPU-side motion/cut guard input
        x = torch.from_numpy(img).to(dev)                      # uint8 HWC, one small upload
        x = x.permute(2, 0, 1).unsqueeze(0).to(dt).mul_(2.0 / 255.0).sub_(1.0)
        ph, pw = (-H) % 64, (-W) % 64
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect" if (ph < H and pw < W) else "replicate")
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        tp = self._mark("upload", tp)

        if self.encoder == "linear":   # 8x8 patch -> latent least-squares fit of TAESD (bench/fit_linear_encoder.py)
            z0 = F.conv2d(F.pixel_unshuffle(x, 8), self._lin_w, self._lin_b)
        elif self._lite_enc is not None:
            z0 = self._lite_enc(x.add(1.0).mul_(0.5))
        elif self.vae_is_tiny:
            z0 = self.vae.encode(x).latents
        else:
            z0 = self.vae.encode(x).latent_dist.mean * self.vae_scale
        tp = self._mark("encode", tp)

        tt, sa, s1a, t_int = self._timestep(strength)
        if self._attn_ref is not None:
            self._attn_ref["hw"] = (z0.shape[2], z0.shape[3])
        noise = self._noise(seed, z0.shape)
        zt = z0 * sa + noise * s1a
        if self._unet_call is None:
            out = (getattr(self, "_compiled_unet", None) or self.unet)(zt, tt, encoder_hidden_states=cond, return_dict=False)[0]
        else:
            reuse, shift = False, (0, 0)
            if self.deepcache >= 2:
                st, key = self._dc, (int(seed), tuple(z0.shape))
                # strength may be animated by the client: deep features stay reusable within +-0.04 strength
                if st.get("key") == key and st.get("cond") is cond and st.get("age", 1 << 30) < self.deepcache - 1 \
                        and abs(t_int - st["t"]) <= 40:
                    # motion / scene-cut guards, computed on CPU thumbnails (no GPU sync): global shift
                    # since the full pass (phase correlation, latent px) and low-frequency change.
                    dy, dx = _phase_shift(st["thumb"], thumb)
                    self.last_dc_diff = float(np.abs(_pool4(thumb) - st["thumb4"]).mean())
                    reuse = max(abs(dy), abs(dx)) <= self.dc_max_shift and self.last_dc_diff < self.dc_thresh
                    if self.dc_motion:
                        shift = (dy, dx)
                if reuse:
                    st["age"] += 1
                else:
                    st.clear()
                    st.update(key=key, cond=cond, age=0, t=t_int, thumb=thumb, thumb4=_pool4(thumb))
            if self.profile:
                self.last_timings["dc_reuse"] = float(reuse)
            self.last_dc_shift = shift
            out = self._unet_call(self.unet, zt, tt, cond, cache=self._dc if self.deepcache >= 2 else None,
                                  reuse=reuse, branch=self.dc_branch, shift=shift)
        if self.prediction_type == "epsilon":
            z0h = (zt - out * s1a) / sa
        elif self.prediction_type == "v_prediction":
            z0h = zt * sa - out * s1a
        else:
            z0h = out
        tp = self._mark("unet", tp)

        if self._lite_dec is not None:
            y = self._lite_dec(torch.tanh(z0h / 3.0) * 3.0).mul_(2.0).sub_(1.0)
        elif self.vae_is_tiny:
            y = self.vae.decode(z0h).sample
        else:
            y = self.vae.decode(z0h / self.vae_scale).sample
        tp = self._mark("decode", tp)

        if ph or pw:
            y = y[:, :, :H, :W]
        y = y[0].clamp_(-1.0, 1.0).add_(1.0).mul_(127.5).round_().to(torch.uint8).permute(1, 2, 0).contiguous()
        res = y.cpu().numpy()
        t1 = time.perf_counter()
        if self.profile:
            self.last_timings["readback"] = (t1 - tp) * 1000.0
        self.last_timings["total"] = (t1 - t0) * 1000.0
        ms = self.last_timings["total"]
        self.fps_estimate = 1000.0 / ms if self.fps_estimate is None else 0.9 * self.fps_estimate + 100.0 / ms
        return res

    def warmup(self, iters: int = 4) -> None:
        """Compile/cache MPS kernels for this size: full + cheap (DeepCache) UNet paths."""
        rng = np.random.default_rng(0)
        img = (rng.random((self.height, self.width, 3)) * 255).astype(np.uint8)
        for _ in range(max(iters, self.deepcache)):
            self.process(img, "warmup", 0.5, 0)
        self._morph_state.clear()
        self._dc.clear()
        self.fps_estimate = None

    def close(self) -> None:
        """Drop model weights so the server can swap engines without leaking GPU memory."""
        with self._lock:
            self._dc.clear()
            self._emb_cache.clear()
            self._noise_cache.clear()
            self._morph_state.clear()
            for k in ("unet", "vae", "text_encoder", "_unet_call"):
                setattr(self, k, None)
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def info(self) -> dict:
        return dict(engine=self.name, model=self.model, width=self.width, height=self.height,
                    fps_estimate=self.fps_estimate, device=self.device)


def create_engine(**cfg) -> TorchTurboEngine:
    return TorchTurboEngine(**cfg)


if __name__ == "__main__":  # quick smoke test:  python -m server.engines.torch_turbo in.png out.jpg
    import sys

    from PIL import Image

    eng = create_engine(model=os.environ.get("MODEL", "sd-turbo"))
    src = Image.open(sys.argv[1]).convert("RGB").resize((eng.width, eng.height))
    eng.warmup()
    out = eng.process(np.asarray(src), sys.argv[3] if len(sys.argv) > 3 else "oil painting", 0.5, 42)
    Image.fromarray(out).save(sys.argv[2], quality=90)
    print(eng.info(), eng.last_timings)
