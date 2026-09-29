#!/usr/bin/env python3
"""Convert 1-step diffusion UNets + TAESD to Core ML for HYPNAGOGIA.

Produces fixed-shape fp16 ML Programs and compiles them to .mlmodelc:

  <out>/<model>_unet_<res>_<attn>.mlmodelc   sample(1,4,h,w) f32, timestep(1,) f32,
                                               encoder_hidden_states(1,77,C) f32
                                               -> noise_pred(1,4,h,w)
  <out>/taesd_enc_<res>.mlmodelc              image(1,3,H,W) in [0,1] -> latent(1,4,h,w)
  <out>/taesd_dec_<res>.mlmodelc              latent(1,4,h,w) -> image(1,3,H,W) in [0,255]

attn:
  plain  - diffusers modules traced as-is (GPU-friendly baseline)
  ane    - Transformer2D blocks rewritten to the ANE-friendly (B, C, 1, S) layout:
           Linear -> 1x1 Conv, channel LayerNorm, per-head split-einsum attention
           (Apple "Deploying Transformers on the Apple Neural Engine" recipe),
           query sequence chunked for long sequences (SPLIT_EINSUM_V2 style).

Usage (on the mini):
  HF_HOME=~/hypnagogia-cache/hf ~/hypnagogia-cache/venv-coreml/bin/python \
      bench/coreml/convert.py --model sdxs --res 512 --attn ane
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("HF_HOME", os.path.expanduser("~/hypnagogia-cache/hf"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPOS = {"sdxs": "IDKiro/sdxs-512-0.9", "sdturbo": "stabilityai/sd-turbo"}
DEFAULT_OUT = os.path.expanduser("~/hypnagogia-cache/coreml")


# --------------------------------------------------------------------------------------
# ANE-friendly rewrite of diffusers' Transformer2DModel (weights shared/copied)
# --------------------------------------------------------------------------------------

def conv_from_linear(lin: nn.Linear) -> nn.Conv2d:
    conv = nn.Conv2d(lin.in_features, lin.out_features, 1, bias=lin.bias is not None)
    conv.weight.data = lin.weight.data.detach().clone()[:, :, None, None]
    if lin.bias is not None:
        conv.bias.data = lin.bias.data.detach().clone()
    return conv


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel axis of a (B, C, 1, S) tensor."""

    def __init__(self, ln: nn.LayerNorm):
        super().__init__()
        self.eps = ln.eps
        self.weight = nn.Parameter(ln.weight.detach().clone().view(1, -1, 1, 1))
        self.bias = nn.Parameter(ln.bias.detach().clone().view(1, -1, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        xc = x - mean
        var = (xc * xc).mean(dim=1, keepdim=True)
        return xc * torch.rsqrt(var + self.eps) * self.weight + self.bias


CHUNK = 512
KV_DOWN = 1          # >1: downsample self-attention K/V spatially (ToDo, arXiv 2402.13573)
KV_DOWN_MIN_S = 2048  # ... only where the query sequence is at least this long (highest-res level)


def split_einsum_attention(q, k, v, heads, dh, scale, sq):
    """q: (B, H*dh, 1, Sq); k, v: (B, H*dh, 1, Sk) -> (B, H*dh, 1, Sq).
    sq: static python int (query length) so chunking is resolved at trace time."""
    kt = k.transpose(1, 3)  # (B, Sk, 1, C)
    outs = []
    n_chunks = max(1, sq // CHUNK) if sq > CHUNK else 1
    bounds = [round(i * sq / n_chunks) for i in range(n_chunks + 1)]
    for h in range(heads):
        qh = q[:, h * dh:(h + 1) * dh]
        kh = kt[:, :, :, h * dh:(h + 1) * dh]
        vh = v[:, h * dh:(h + 1) * dh]
        parts = []
        for c in range(n_chunks):
            qc = qh[..., bounds[c]:bounds[c + 1]]
            w = torch.einsum("bchq,bkhc->bkhq", [qc, kh]) * scale  # (B, Sk, 1, csz)
            w = w.softmax(dim=1)
            parts.append(torch.einsum("bkhq,bchk->bchq", [w, vh]))  # (B, dh, 1, csz)
        outs.append(parts[0] if len(parts) == 1 else torch.cat(parts, dim=3))
    return torch.cat(outs, dim=1)


class ANEAttention(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.heads = attn.heads
        inner = attn.to_q.out_features
        self.dh = inner // self.heads
        self.scale = float(attn.scale)
        self.to_q = conv_from_linear(attn.to_q)
        self.to_k = conv_from_linear(attn.to_k)
        self.to_v = conv_from_linear(attn.to_v)
        self.to_out = conv_from_linear(attn.to_out[0])

    def forward(self, x, ctx=None, hw=None):
        if not torch.jit.is_tracing():  # eager pre-pass records the static query length
            self.sq_static = int(x.shape[3])
            self.bc_static = (int(x.shape[0]), int(x.shape[1]))
        c = x if ctx is None else ctx
        if (ctx is None and KV_DOWN > 1 and hw is not None and self.sq_static >= KV_DOWN_MIN_S
                and getattr(self, "kv_enable", True)):
            # ToDo-style token downsampling: self-attention keys/values from a KV_DOWN x KV_DOWN
            # average-pooled copy of the tokens (4x fewer keys at KV_DOWN=2).
            h, w = hw
            b, ch = self.bc_static
            xs = F.avg_pool2d(x.reshape(b, ch, h, w), KV_DOWN)
            c = xs.reshape(b, ch, 1, (h // KV_DOWN) * (w // KV_DOWN))
        q = self.to_q(x)
        k = self.to_k(c)
        v = self.to_v(c)
        return self.to_out(split_einsum_attention(q, k, v, self.heads, self.dh, self.scale,
                                                  self.sq_static))


class ANEFeedForward(nn.Module):
    def __init__(self, ff):
        super().__init__()
        geglu = ff.net[0]
        self.approx = getattr(geglu, "approximate", "none")
        self.proj = conv_from_linear(geglu.proj)
        self.out = conv_from_linear(ff.net[2])

    def forward(self, x):
        h, gate = self.proj(x).chunk(2, dim=1)
        return self.out(h * F.gelu(gate, approximate=self.approx))


class ANEBlock(nn.Module):
    def __init__(self, blk):
        super().__init__()
        assert getattr(blk, "norm_type", "layer_norm") == "layer_norm", blk.norm_type
        self.only_cross = bool(blk.only_cross_attention)
        self.norm1 = ChannelLayerNorm(blk.norm1)
        self.attn1 = ANEAttention(blk.attn1)
        self.has_attn2 = blk.attn2 is not None
        if self.has_attn2:
            self.norm2 = ChannelLayerNorm(blk.norm2)
            self.attn2 = ANEAttention(blk.attn2)
        self.norm3 = ChannelLayerNorm(blk.norm3)
        self.ff = ANEFeedForward(blk.ff)

    def forward(self, x, ctx, hw=None):
        x = self.attn1(self.norm1(x), ctx if self.only_cross else None, hw) + x
        if self.has_attn2:
            x = self.attn2(self.norm2(x), ctx) + x
        return self.ff(self.norm3(x)) + x


class ANETransformer2D(nn.Module):
    """Drop-in for diffusers Transformer2DModel (continuous input). Expects
    encoder_hidden_states already in (B, C, 1, S) layout."""

    def __init__(self, t):
        super().__init__()
        self.norm = t.norm
        self.use_linear = bool(t.use_linear_projection)
        self.proj_in = conv_from_linear(t.proj_in) if self.use_linear else t.proj_in
        self.proj_out = conv_from_linear(t.proj_out) if self.use_linear else t.proj_out
        self.blocks = nn.ModuleList([ANEBlock(b) for b in t.transformer_blocks])

    def forward(self, hidden_states, encoder_hidden_states=None, *args, **kwargs):
        if not torch.jit.is_tracing():  # eager pre-pass records static shapes
            self.static_shape = tuple(int(v) for v in hidden_states.shape)
        b, c, h, w = self.static_shape
        x = self.proj_in(self.norm(hidden_states))
        inner = self.proj_in.out_channels if self.use_linear else x.shape[1]
        x = x.reshape(b, inner, 1, h * w)
        for blk in self.blocks:
            x = blk(x, encoder_hidden_states, (h, w))
        x = x.reshape(b, inner, h, w)
        x = self.proj_out(x) + hidden_states
        return (x,)


KV_DOWN_WHERE = "all"  # "all" | "up_blocks" | "down_blocks": which UNet half gets K/V downsampling


def aneify_unet(unet):
    from diffusers.models.transformers.transformer_2d import Transformer2DModel
    n = 0
    for name, mod in list(unet.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, Transformer2DModel):
                setattr(mod, cname, ANETransformer2D(child))
                n += 1
    for name, mod in unet.named_modules():
        if isinstance(mod, ANEAttention):
            mod.kv_enable = KV_DOWN_WHERE == "all" or name.startswith(KV_DOWN_WHERE)
    return n


class UNetWrap(nn.Module):
    def __init__(self, unet, ane_layout: bool):
        super().__init__()
        self.unet = unet
        self.ane_layout = ane_layout

    def forward(self, sample, timestep, encoder_hidden_states):
        ehs = encoder_hidden_states
        if self.ane_layout:
            ehs = ehs.transpose(1, 2).unsqueeze(2)  # (B, C, 1, 77)
        return self.unet(sample, timestep, ehs, return_dict=False)[0]


class TAEEnc(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.layers = vae.encoder.layers

    def forward(self, image):  # [0,1]
        return self.layers(image)


class TAEDec(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.layers = vae.decoder.layers

    def forward(self, latent):
        x = torch.tanh(latent / 3.0) * 3.0
        return (self.layers(x).clamp(0.0, 1.0) * 255.0)


# --------------------------------------------------------------------------------------

def compile_mlpackage(pkg, out_dir):
    """xcrun coremlcompiler compile pkg out_dir -> out_dir/<name>.mlmodelc"""
    t = time.time()
    subprocess.run(["xcrun", "coremlcompiler", "compile", pkg, out_dir], check=True,
                   stdout=subprocess.DEVNULL)
    name = os.path.splitext(os.path.basename(pkg))[0] + ".mlmodelc"
    print(f"  compiled {name} in {time.time() - t:.1f}s", flush=True)
    return os.path.join(out_dir, name)


def calibration_data(model_key, w, h, n=6):
    """Realistic UNet inputs for activation calibration: TAESD latents of the procedural test
    scenes, noised to t in [250, 750], with real prompt embeddings."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import scenes
    from diffusers import AutoencoderTiny
    from transformers import CLIPTextModel, CLIPTokenizer
    repo = REPOS[model_key]
    tok = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer")
    kw = {"variant": "fp16"} if model_key == "sdturbo" else {}
    te = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder", torch_dtype=torch.float32, **kw).eval()
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float32).eval()
    prompts = ["vast drowned gothic cathedral, bioluminescent, volumetric light, oil painting",
               "overgrown sunken garden temple, moss and flowers, god rays, lush fantasy concept art",
               "neon bathhouse, wet tiles, cyan and magenta glow, cinematic, moody"]
    embs = [te(tok([p], padding="max_length", max_length=77, truncation=True,
                   return_tensors="pt").input_ids)[0].numpy() for p in prompts]
    frames = scenes.walk(n, w, h)
    betas = np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2
    abar = np.cumprod(1 - betas)
    rng = np.random.default_rng(0)
    out = []
    for i, f in enumerate(frames):
        x = torch.from_numpy(f).permute(2, 0, 1)[None].float() / 255.0
        z0 = vae.encoder.layers(x).numpy()
        t = int(rng.integers(250, 750))
        zt = np.sqrt(abar[t]) * z0 + np.sqrt(1 - abar[t]) * rng.standard_normal(z0.shape)
        out.append({"sample": zt.astype(np.float32), "timestep": np.array([t], np.float32),
                    "encoder_hidden_states": embs[i % len(embs)].astype(np.float32)})
    return out


def to_coreml(module, example_inputs, input_specs, output_names, pkg_path, target, w8=False,
              a8_data=None):
    import coremltools as ct
    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(module, example_inputs, check_trace=False, strict=False)
    print(f"  traced in {time.time() - t:.1f}s", flush=True)
    t = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=input_specs,
        outputs=[ct.TensorType(name=n) for n in output_names],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=target,
        compute_units=ct.ComputeUnit.ALL,
        skip_model_load=True,
    )
    print(f"  converted in {time.time() - t:.1f}s", flush=True)
    if a8_data is not None:
        import coremltools.optimize.coreml as cto
        t = time.time()
        acfg = cto.OptimizationConfig(global_config=cto.experimental.OpActivationLinearQuantizerConfig(
            mode="linear_symmetric"))
        mlmodel = cto.experimental.linear_quantize_activations(mlmodel, acfg, a8_data)
        print(f"  int8 activation calibration ({len(a8_data)} samples) in {time.time() - t:.1f}s", flush=True)
    if w8:
        import coremltools.optimize.coreml as cto
        t = time.time()
        cfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_channel", weight_threshold=2048))
        mlmodel = cto.linear_quantize_weights(mlmodel, config=cfg)
        print(f"  int8 weight quantization in {time.time() - t:.1f}s", flush=True)
    if os.path.exists(pkg_path):
        shutil.rmtree(pkg_path)
    mlmodel.save(pkg_path)
    return pkg_path


def parse_size(tok):
    if "x" in str(tok):
        w, h = (int(v) for v in str(tok).lower().split("x"))
    else:
        w = h = int(tok)
    return w, h


def size_tag(w, h):
    return str(w) if w == h else f"{w}x{h}"


def convert_unet(model_key, w, h, attn, out_dir, target, suffix="", w8=False, a8=False):
    import coremltools as ct
    from diffusers import UNet2DConditionModel
    repo = REPOS[model_key]
    kw = {"variant": "fp16"} if model_key == "sdturbo" else {}
    unet = UNet2DConditionModel.from_pretrained(repo, subfolder="unet", torch_dtype=torch.float32, **kw).eval()
    if attn == "plain":
        from diffusers.models.attention_processor import AttnProcessor
        unet.set_attn_processor(AttnProcessor())
    ane = attn == "ane"
    if ane:
        n = aneify_unet(unet)
        print(f"  rewrote {n} Transformer2D blocks to ANE layout (chunk={CHUNK})", flush=True)
    wrap = UNetWrap(unet, ane).eval()
    lh, lw = h // 8, w // 8
    ctx_dim = unet.config.cross_attention_dim
    ex = (torch.randn(1, 4, lh, lw), torch.tensor([499.0]), torch.randn(1, 77, ctx_dim))
    with torch.no_grad():
        ref = wrap(*ex)
    print(f"  torch ref out std {ref.std().item():.4f}", flush=True)
    specs = [
        ct.TensorType(name="sample", shape=(1, 4, lh, lw), dtype=np.float32),
        ct.TensorType(name="timestep", shape=(1,), dtype=np.float32),
        ct.TensorType(name="encoder_hidden_states", shape=(1, 77, ctx_dim), dtype=np.float32),
    ]
    name = f"{model_key}_unet_{size_tag(w, h)}_{attn}{suffix}"
    a8_data = calibration_data(model_key, w, h) if a8 else None
    pkg = to_coreml(wrap, ex, specs, ["noise_pred"], os.path.join(out_dir, name + ".mlpackage"), target, w8=w8,
                    a8_data=a8_data)
    np.savez(os.path.join(out_dir, name + "_ref.npz"), sample=ex[0].numpy(), timestep=ex[1].numpy(),
             ehs=ex[2].numpy(), out=ref.numpy())
    return compile_mlpackage(pkg, out_dir)


def convert_taesd(w, h, out_dir, target, repo="IDKiro/sdxs-512-0.9"):
    import coremltools as ct
    from diffusers import AutoencoderTiny
    if repo == "IDKiro/sdxs-512-0.9":
        vae = AutoencoderTiny.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32).eval()
        tag = "sdxs"
    else:
        vae = AutoencoderTiny.from_pretrained(repo, torch_dtype=torch.float32).eval()
        tag = "taesd"
    lh, lw = h // 8, w // 8
    st = size_tag(w, h)
    enc = TAEEnc(vae).eval()
    dec = TAEDec(vae).eval()
    out = []
    pkg = to_coreml(enc, (torch.rand(1, 3, h, w),),
                    [ct.TensorType(name="image", shape=(1, 3, h, w), dtype=np.float32)],
                    ["latent"], os.path.join(out_dir, f"{tag}_tae_enc_{st}.mlpackage"), target)
    out.append(compile_mlpackage(pkg, out_dir))
    pkg = to_coreml(dec, (torch.randn(1, 4, lh, lw),),
                    [ct.TensorType(name="latent", shape=(1, 4, lh, lw), dtype=np.float32)],
                    ["image"], os.path.join(out_dir, f"{tag}_tae_dec_{st}.mlpackage"), target)
    out.append(compile_mlpackage(pkg, out_dir))
    return out


def main():
    global CHUNK, KV_DOWN, KV_DOWN_MIN_S, KV_DOWN_WHERE
    import coremltools as ct
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sdturbo", choices=list(REPOS) + ["none"])
    ap.add_argument("--res", nargs="+", default=["512"], help="sizes: 512 or WxH (multiples of 64)")
    ap.add_argument("--attn", default="ane", choices=["ane", "plain"])
    ap.add_argument("--vae", default="none", choices=["none", "sdxs", "taesd"])
    ap.add_argument("--chunk", type=int, default=CHUNK, help="attention query chunk (0 = none)")
    ap.add_argument("--w8", action="store_true", help="int8 per-channel weight quantization (UNet)")
    ap.add_argument("--a8", action="store_true", help="int8 activations (calibrated; experimental, W8A8 with --w8)")
    ap.add_argument("--kv-down", type=int, default=1, help="self-attn K/V spatial downsample factor (ToDo)")
    ap.add_argument("--kv-down-min-s", type=int, default=2048, help="apply kv-down where query len >= this")
    ap.add_argument("--kv-down-where", default="all", choices=["all", "up_blocks", "down_blocks"])
    ap.add_argument("--suffix", default="", help="name suffix for experimental variants")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--target", default="macOS15", choices=["macOS14", "macOS15"])
    a = ap.parse_args()
    CHUNK = a.chunk if a.chunk > 0 else 1 << 30
    KV_DOWN, KV_DOWN_MIN_S, KV_DOWN_WHERE = a.kv_down, a.kv_down_min_s, a.kv_down_where
    os.makedirs(a.out, exist_ok=True)
    target = getattr(ct.target, a.target)
    torch.set_grad_enabled(False)
    for tok in a.res:
        w, h = parse_size(tok)
        assert w % 64 == 0 and h % 64 == 0, "sizes must be multiples of 64"
        if a.model in REPOS:
            print(f"== UNet {a.model} {w}x{h} attn={a.attn} chunk={a.chunk} w8={a.w8} kv_down={a.kv_down}", flush=True)
            print("  ->", convert_unet(a.model, w, h, a.attn, a.out, target, a.suffix, a.w8, a.a8), flush=True)
        if a.vae != "none":
            print(f"== TAESD ({a.vae}) {w}x{h}", flush=True)
            repo = "IDKiro/sdxs-512-0.9" if a.vae == "sdxs" else "madebyollin/taesd"
            print("  ->", convert_taesd(w, h, a.out, target, repo), flush=True)


if __name__ == "__main__":
    main()
