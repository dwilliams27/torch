"""Core ML / Apple Neural Engine 1-step img2img engine.

Pipeline (all stages are fixed-shape fp16 Core ML programs, default: all on the ANE,
so the GPU stays free for the game's WebGL renderer):

    RGB uint8 --TAESD enc--> z0 --(noise to t)--> z_t --UNet 1 step--> eps --x0--> TAESD dec --> RGB

    z_t = sqrt(abar_t) * z0 + sqrt(1 - abar_t) * eps_seed      (eps_seed cached per seed/size)
    x0  = (z_t - sqrt(1 - abar_t) * eps_pred) / sqrt(abar_t)   (epsilon-prediction, Euler == DDIM for 1 step)
    t   = round(clamp(strength, 0.02, 1) * 999)

Seed noise (torch CPU generator), timestep mapping and the prompt-morph glide deliberately
match server/engines/torch_turbo.py, so a server pool that alternates frames between this
engine (ANE) and torch_turbo (MPS) produces matching images.

Models are produced by ``bench/coreml/convert.py`` into ``~/hypnagogia-cache/coreml``:
    <model>_unet_<size>_ane.mlmodelc, <vae>_tae_enc_<size>.mlmodelc, <vae>_tae_dec_<size>.mlmodelc
where <size> is "512" (square) or "640x384".  The UNet uses an ANE-native layout
(1x1 convs, channel LayerNorm, split-einsum attention).  Text embeddings come from
the model's own CLIP text encoder (torch, CPU) and are cached per prompt.

cfg (all optional):  coreml_dir, model ("sdturbo" (default) | "sdxs" | HF id), width, height
(default: largest of 512/384/256 whose models exist; a pool passes the game's 512x320), variant
(UNet file suffix; default: the depth-grafted "_kv2w8_d08" when depth_graft is 0.8, else "_kv2w8"
at >=512 = ToDo 2x2 K/V downsampling at the 64x64 level + int8 weights, "_w8" below 512, "" = exact
fp16), depth_graft (0.8, torch_turbo's default; 0 = a stock build), compute_units ("NE" | "GPU" |
"ALL" | "CPU") for the UNet, vae_compute_units, text_device ("cpu"), hf_cache (dir). A grafted
UNet takes each frame's depth (process(depth=)), like torch_turbo; there is no DeepCache,
cross-frame attention, `held` or LoRA here. Measured numbers: bench/RESULTS-coreml.md and M112's
page (docs/shots/m112/).
If the compiled models or coremltools are missing, create_engine raises so that
``--engine auto`` falls back to the next engine.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time

import numpy as np

log = logging.getLogger("hypnagogia.coreml")

DEFAULT_DIR = os.environ.get("HYPNAGOGIA_COREML_DIR", os.path.expanduser("~/hypnagogia-cache/coreml"))

MODELS = {
    # key: (HF repo for text encoder/tokenizer, vae tag used for TAESD files, description)
    "sdxs": ("IDKiro/sdxs-512-0.9", "sdxs", "IDKiro/sdxs-512-0.9 (1-step) + tiny VAE, Core ML/ANE"),
    "sdturbo": ("stabilityai/sd-turbo", "taesd", "stabilityai/sd-turbo + TAESD, Core ML/ANE"),
}
SIZE_PREFERENCE = [512, 384, 256]
VARIANT_PREFERENCE = ["_kv2w8_d08", "_kv2w8", "_kv2", "_w8", ""]   # _d08: torch_turbo's depth graft (0.8)
DEPTH_KEEP = 12   # engine frames a stream's last depth stands in for a frame without one (as torch_turbo)
SMALL_VARIANT_PREFERENCE = ["_kv2w8_d08", "_w8", ""]   # (the graft build matches torch_turbo, whose ToDo acts at every size)
ALIASES = {"IDKiro/sdxs-512-0.9": "sdxs", "stabilityai/sd-turbo": "sdturbo", "sd-turbo": "sdturbo",
           "sdxs-512": "sdxs", "sdxs-512-0.9": "sdxs"}


def _graft_of(variant: str) -> float:
    """The depth graft a UNet variant was built with: "_kv2w8_d08" -> 0.8, no "_dNN" -> 0."""
    import re
    m = re.search(r"_d(\d\d)$", variant or "")
    return int(m.group(1)) / 10 if m else 0.0


def _size_tag(w: int, h: int) -> str:
    return str(w) if w == h else f"{w}x{h}"


def _alphas_cumprod(n=1000, beta_start=0.00085, beta_end=0.012):
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, n, dtype=np.float64) ** 2
    return np.cumprod(1.0 - betas)


class CoreMLTurboEngine:
    def __init__(self, coreml_dir: str | None = None, model: str | None = None, width: int | None = None,
                 height: int | None = None, compute_units: str = "NE", vae_compute_units: str | None = None,
                 text_device: str = "cpu", hf_cache: str | None = None, attn: str = "ane",
                 variant: str | None = None, morph: float = 1.0, depth_graft: float = 0.8, **_ignored):
        try:
            import coremltools as ct  # noqa: F401
        except Exception as e:  # pragma: no cover - platform dependent
            raise RuntimeError(f"coreml_turbo needs coremltools on macOS ({e})") from e
        import coremltools as ct

        key = ALIASES.get(model or "sdturbo", model or "sdturbo")
        if key not in MODELS:
            raise ValueError(f"coreml_turbo: unknown model {model!r}; choose one of {list(MODELS)}")
        repo, vae_tag, desc = MODELS[key]
        d = os.path.expanduser(coreml_dir or DEFAULT_DIR)
        if width is None and height is None:
            # no explicit size: largest preferred size whose models were built on this machine
            def built(sz):
                t = _size_tag(sz, sz)
                return (os.path.isdir(os.path.join(d, f"{vae_tag}_tae_enc_{t}.mlmodelc"))
                        and any(f.startswith(f"{key}_unet_{t}_{attn}") and f.endswith(".mlmodelc")
                                for f in (os.listdir(d) if os.path.isdir(d) else [])))
            width = height = next((sz for sz in SIZE_PREFERENCE if built(sz)), SIZE_PREFERENCE[0])
        width = int(width if width is not None else height)
        height = int(height if height is not None else width)
        self.width, self.height = width, height
        tag = _size_tag(self.width, self.height)
        # UNet variants (see bench/RESULTS-coreml.md): "_kv2w8" = ToDo 2x2 K/V downsampling in the
        # highest-res self-attention + int8 weights (fastest), "_w8" = int8 weights, "" = exact fp16,
        # "..._d08" = torch_turbo's depth graft at 0.8 (convert.py --depth-graft), used only when the
        # config's depth_graft (torch_turbo's default, 0.8) is that value. Stock K/V downsampling
        # only pays off at >= 512x512; the graft build has it at every size, as torch_turbo does.
        graft = float(depth_graft or 0.0)
        if variant is None:
            prefs = VARIANT_PREFERENCE if self.width * self.height >= 512 * 512 else SMALL_VARIANT_PREFERENCE
            variant = next((v for v in prefs if abs(_graft_of(v) - graft) < 1e-6
                            and os.path.isdir(os.path.join(d, f"{key}_unet_{tag}_{attn}{v}.mlmodelc"))), "")
        self.variant = variant
        paths = {
            "unet": os.path.join(d, f"{key}_unet_{tag}_{attn}{variant}.mlmodelc"),
            "enc": os.path.join(d, f"{vae_tag}_tae_enc_{tag}.mlmodelc"),
            "dec": os.path.join(d, f"{vae_tag}_tae_dec_{tag}.mlmodelc"),
        }
        missing = [p for p in paths.values() if not os.path.isdir(p)]
        if missing:
            have = sorted(f for f in os.listdir(d) if f.endswith(".mlmodelc")) if os.path.isdir(d) else []
            raise FileNotFoundError(
                "coreml_turbo: compiled Core ML models not found: " + ", ".join(missing)
                + f" (have: {have or 'nothing'}). Build them with: python bench/coreml/convert.py "
                f"--model {key} --res {tag} --attn {attn} --vae {vae_tag}"
                + (" (the depth-grafted build: bench/coreml/build_models.sh graft)" if graft > 0 else ""))

        cu = {"NE": ct.ComputeUnit.CPU_AND_NE, "GPU": ct.ComputeUnit.CPU_AND_GPU,
              "ALL": ct.ComputeUnit.ALL, "CPU": ct.ComputeUnit.CPU_ONLY}
        ucu = cu[compute_units.upper()]
        vcu = cu[(vae_compute_units or compute_units).upper()]
        t0 = time.time()
        # First load of an ANE model triggers the ANE compiler (~15 s for the UNet);
        # the OS caches the result, later loads are faster.
        self._unet = ct.models.CompiledMLModel(paths["unet"], compute_units=ucu)
        self._enc = ct.models.CompiledMLModel(paths["enc"], compute_units=vcu)
        self._dec = ct.models.CompiledMLModel(paths["dec"], compute_units=vcu)
        # a depth-grafted UNet (convert.py --depth-graft) takes the frame's depth as a fifth channel
        try:
            import json
            with open(os.path.join(paths["unet"], "metadata.json")) as f:
                inputs = {i.get("name") for i in json.load(f)[0].get("inputSchema", [])}
        except (OSError, ValueError, IndexError, KeyError):
            inputs = set()
        self.wants_depth = "depth" in inputs     # app.py sends a frame's depth only to engines that want it
        self.depth_graft = _graft_of(variant) if self.wants_depth else 0.0   # (as built; reported to app.py)
        self.takes_cut = True                     # process(cut=True): switch prompts at once, as torch_turbo
        self._depths: dict = {}
        self._frame = 0
        log.info("coreml_turbo: loaded %s%s in %.1fs (unet=%s vae=%s)", tag, variant, time.time() - t0,
                 compute_units, vae_compute_units or compute_units)

        self.name = f"coreml-{key}"
        self.model = desc + (f" [{variant.strip('_')}]" if variant else "")
        self.device = "ane" if ucu == ct.ComputeUnit.CPU_AND_NE else f"coreml-{compute_units.lower()}"
        self.key = key
        self._repo = repo
        self._hf_cache = hf_cache
        self._text_device = text_device
        self._tok = None
        self._te = None
        self._emb_cache: dict[str, np.ndarray] = {}
        self._noise_cache: dict[tuple, np.ndarray] = {}
        self.morph = float(morph)
        self._morph_state: dict = {}
        self._abar = _alphas_cumprod()
        self._lock = threading.Lock()
        self.last_timings: dict[str, float] = {}
        self._load_text_encoder()

    # -- text conditioning ---------------------------------------------------------------
    def _load_text_encoder(self):
        import torch
        from transformers import CLIPTextModel, CLIPTokenizer
        cache_dir = self._hf_cache
        if cache_dir is None and not os.environ.get("HF_HOME"):
            alt = os.path.expanduser("~/hypnagogia-cache/hf/hub")
            if os.path.isdir(os.path.join(alt, "models--" + self._repo.replace("/", "--"))):
                cache_dir = alt
        kw = {"cache_dir": cache_dir} if cache_dir else {}
        t0 = time.time()
        te_kw = dict(kw)
        if self.key == "sdturbo":
            te_kw["variant"] = "fp16"
        for local_only in (True, False):  # cached files first (no network round-trips), then download
            try:
                self._tok = CLIPTokenizer.from_pretrained(self._repo, subfolder="tokenizer",
                                                          local_files_only=local_only, **kw)
                self._te = CLIPTextModel.from_pretrained(self._repo, subfolder="text_encoder",
                                                         torch_dtype=torch.float32, local_files_only=local_only,
                                                         **te_kw).eval()
                break
            except Exception:
                if not local_only:
                    raise
        self._te.to(self._text_device)
        log.info("coreml_turbo: text encoder %s loaded in %.1fs", self._repo, time.time() - t0)

    def _embed(self, prompt: str) -> np.ndarray:
        e = self._emb_cache.get(prompt)
        if e is not None:
            return e
        import torch
        ids = self._tok([prompt], padding="max_length", max_length=self._tok.model_max_length,
                        truncation=True, return_tensors="pt").input_ids.to(self._text_device)
        with torch.inference_mode():
            e = self._te(ids)[0].float().cpu().numpy().astype(np.float32)
        if len(self._emb_cache) > 64:
            self._emb_cache.pop(next(iter(self._emb_cache)))
        self._emb_cache[prompt] = e
        return e

    def _conditioning(self, prompt: str, seed: int, cut: bool = False) -> np.ndarray:
        """Prompt embedding with the same exponential cross-fade on prompt change as torch_turbo
        (time constant `morph` seconds, per seed), so zone styles melt instead of snapping; `cut`
        (a change nobody sees, e.g. behind closed eyes) switches at once."""
        prompt = prompt or ""
        target = self._embed(prompt)
        if self.morph <= 0:
            return target
        if cut:
            self._morph_state[seed] = dict(key=prompt, emb=target, t=time.monotonic(), settled=True)
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
            st["key"], st["settled"], st["w"] = prompt, False, 0.0
        if st["settled"]:
            st["emb"] = target
            return target
        a = 1.0 - math.exp(-max(dt, 1e-3) / self.morph)
        st["w"] = st.get("w", 0.0) + (1.0 - st.get("w", 0.0)) * a
        if st["w"] > 0.985:
            st["emb"], st["settled"] = target, True
            return target
        st["emb"] = (st["emb"] + (target - st["emb"]) * a).astype(np.float32)
        return st["emb"]

    def _noise(self, seed: int) -> np.ndarray:
        k = (int(seed), self.height, self.width)
        n = self._noise_cache.get(k)
        if n is None:
            import torch  # same generator as torch_turbo -> identical noise for identical seeds
            g = torch.Generator("cpu").manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
            n = torch.randn((1, 4, self.height // 8, self.width // 8), generator=g,
                            dtype=torch.float32).numpy()
            if len(self._noise_cache) > 16:
                self._noise_cache.pop(next(iter(self._noise_cache)))
            self._noise_cache[k] = n
        return n

    # -- engine API ------------------------------------------------------------------------
    def warmup(self) -> None:
        img = np.full((self.height, self.width, 3), 96, np.uint8)
        img[self.height // 3:, :, :] = 40
        for _ in range(2):
            self.process(img, "a dim stone hall, warm lantern light", 0.5, 0)

    def _depth(self, depth, seed: int) -> np.ndarray:
        """(1, 1, h/8, w/8) float32: the client's relative inverse depth in [-1, 1] (near = 1),
        resized to latent size if needed. No depth: the stream's last one if recent, else flat
        (as torch_turbo._depth_latent)."""
        lh, lw = self.height // 8, self.width // 8
        if depth is None:
            last = self._depths.get(int(seed))
            if last is None or self._frame - last[0] > DEPTH_KEEP:
                return np.zeros((1, 1, lh, lw), np.float32)
            return last[1]
        d = np.asarray(depth, np.float32)
        if d.shape != (lh, lw):   # (a resized frame) the same resampling as torch_turbo._depth_latent
            import torch
            import torch.nn.functional as F
            t = torch.from_numpy(np.ascontiguousarray(d))[None, None]
            big = d.shape[0] >= lh and d.shape[1] >= lw
            t = F.interpolate(t, size=(lh, lw), mode="area") if big else F.interpolate(t, size=(lh, lw), mode="bilinear", align_corners=False)
            d = t[0, 0].numpy()
        d = np.clip(d, -1.0, 1.0)[None, None].astype(np.float32)
        self._depths[int(seed)] = (self._frame, d)
        while len(self._depths) > 16:
            self._depths.pop(next(iter(self._depths)))
        return d

    def process(self, image, prompt: str, strength: float, seed: int, negative: str | None = None, depth=None,
                cut=False):
        """image: uint8 (H, W, 3) RGB at (height, width) -> uint8 (H, W, 3). negative is ignored
        (1-step turbo models run without classifier-free guidance). depth: see _depth (grafted UNets)."""
        with self._lock:
            self._frame += 1
            h, w = image.shape[:2]
            src = image
            if (w, h) != (self.width, self.height):
                from PIL import Image
                src = np.asarray(Image.fromarray(image).resize((self.width, self.height), Image.BILINEAR))
            t0 = time.perf_counter()
            emb = self._conditioning(prompt, seed, cut)
            t1 = time.perf_counter()
            x = np.ascontiguousarray(src.transpose(2, 0, 1)[None], dtype=np.float32) * (1.0 / 255.0)
            z0 = self._enc.predict({"image": x})["latent"]
            t2 = time.perf_counter()
            t = int(round(float(min(max(strength, 0.02), 1.0)) * 999))
            a = float(np.sqrt(self._abar[t]))
            b = float(np.sqrt(1.0 - self._abar[t]))
            zt = (a * z0 + b * self._noise(seed)).astype(np.float32)
            feed = {"sample": zt, "timestep": np.array([t], np.float32), "encoder_hidden_states": emb}
            if self.wants_depth:
                feed["depth"] = self._depth(depth, seed)
            eps = self._unet.predict(feed)["noise_pred"]
            t3 = time.perf_counter()
            x0 = ((zt - b * eps) * (1.0 / a)).astype(np.float32)
            img = self._dec.predict({"latent": x0})["image"]
            t4 = time.perf_counter()
            out = np.clip(img[0].transpose(1, 2, 0) + 0.5, 0, 255).astype(np.uint8)
            if (w, h) != (self.width, self.height):
                from PIL import Image
                out = np.asarray(Image.fromarray(out).resize((w, h), Image.BILINEAR))
            self.last_timings = {"text": (t1 - t0) * 1e3, "enc": (t2 - t1) * 1e3,
                                 "unet": (t3 - t2) * 1e3, "dec": (t4 - t3) * 1e3,
                                 "total": (time.perf_counter() - t0) * 1e3}
            return out


def create_engine(**cfg) -> CoreMLTurboEngine:
    return CoreMLTurboEngine(**cfg)
