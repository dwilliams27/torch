"""HYPNAGOGIA server: static client, /api/info, /ws (binary frame protocol).

Threading model (the event loop never blocks):

    event loop --recv--> codec pool: JPEG decode (+resize)  --+
        ^                                                    v
        |                             per-connection latest-wins pending slot
        |                                                    |  (lock-guarded)
        |                                                    v
        +--send-- codec pool: resize + JPEG encode <-- GPU worker thread
                                                        pulls frames round-robin,
                                                        runs engine.process

Decode of frame N+1 and encode of frame N-1 overlap inference of frame N, and the
GPU thread takes the next frame itself the moment an inference ends, so with the
client keeping 2 frames in flight the GPU never waits on I/O or the event loop.
"""
from __future__ import annotations

import asyncio
import base64
import collections
import fcntl
import gc
import io
import itertools
import json
import logging
import mimetypes
import os
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from aiohttp import WSCloseCode, WSMsgType, web

from . import engines as engines_pkg

log = logging.getLogger("hypnagogia")

ROOT = Path(__file__).resolve().parent.parent
CLIENT_DIR = ROOT / "client"
JPEG_QUALITY = 88
MAX_MSG = 32 * 1024 * 1024

for ext, mime in {
    ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
    ".map": "application/json", ".wasm": "application/wasm", ".css": "text/css",
    ".html": "text/html", ".svg": "image/svg+xml", ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json", ".ktx2": "image/ktx2", ".webp": "image/webp",
    ".woff2": "font/woff2", ".ogg": "audio/ogg", ".mp3": "audio/mpeg", ".wav": "audio/wav",
    ".glsl": "text/plain", ".txt": "text/plain",
}.items():
    mimetypes.add_type(mime, ext)

# ----------------------------------------------------------------------------
# JPEG codec: simplejpeg (libjpeg-turbo, releases the GIL, ~1.7x PIL) w/ PIL fallback
# ----------------------------------------------------------------------------
MAX_SIDE = 2048   # frames (header size and JPEG size) larger than this are refused


def _check_side(w: int, h: int):
    if not (8 <= w <= MAX_SIDE and 8 <= h <= MAX_SIDE):
        raise ValueError(f"frame size {w}x{h} outside 8..{MAX_SIDE}")


try:
    import simplejpeg

    def jpeg_decode(buf: bytes) -> np.ndarray:
        h, w, _, _ = simplejpeg.decode_jpeg_header(buf)   # size first: never inflate a huge image
        _check_side(w, h)
        return simplejpeg.decode_jpeg(buf, colorspace="RGB", fastdct=True, fastupsample=True)

    def jpeg_encode(img: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
        return simplejpeg.encode_jpeg(np.ascontiguousarray(img), quality=quality,
                                      colorspace="RGB", colorsubsampling="420", fastdct=True)
    JPEG_LIB = "simplejpeg"
except Exception:  # pragma: no cover
    simplejpeg = None

    def jpeg_decode(buf: bytes) -> np.ndarray:
        from PIL import Image
        with Image.open(io.BytesIO(buf)) as im:
            _check_side(*im.size)
            return np.asarray(im.convert("RGB"))

    def jpeg_encode(img: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
        from PIL import Image
        b = io.BytesIO()
        Image.fromarray(img).save(b, "JPEG", quality=quality)
        return b.getvalue()
    JPEG_LIB = "pillow"


def takes_size(engine, w: int, h: int) -> bool:
    """Engines with `flexible = True` take frames at the client's own capture size (it picks
    one matching its screen's aspect) within their pixel budget; others get every frame
    resized to their fixed width x height. Multiples of 64, so no two sizes share a padded
    latent shape (torch_turbo's DeepCache keys on it) and no padding is hidden."""
    if engine is None or not getattr(engine, "flexible", False) or w % 64 or h % 64:
        return False
    ew, eh = int(engine.width), int(engine.height)
    return min(w, h) >= 64 and max(w, h) <= 2 * max(ew, eh) and w * h <= 1.25 * ew * eh


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[1] == w and img.shape[0] == h:
        return img
    from PIL import Image
    return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))


DEPTH_PREFERENCE = 1.5   # --engine auto: how much faster a depth-less engine must be to win


def _depth_field(d) -> np.ndarray:
    """Header `depth` -> float32 array in [-1, 1]: {"w", "h", "data": base64 of w*h bytes, rows
    top first, byte b = relative inverse depth (b / 127.5 - 1, near = 1)}."""
    if not isinstance(d, dict) or not {"w", "h", "data"} <= d.keys():
        raise ValueError("depth needs w, h and data")
    w, h, data = int(d["w"]), int(d["h"]), d["data"]
    if not (1 <= w <= 256 and 1 <= h <= 256):
        raise ValueError(f"depth size {w}x{h} out of range")
    if not isinstance(data, str) or len(data) != 4 * ((w * h + 2) // 3):   # checked before decoding
        raise ValueError(f"depth data is not base64 of {w}x{h} bytes")
    raw = base64.b64decode(data, validate=True)
    if len(raw) != w * h:
        raise ValueError(f"depth has {len(raw)} bytes for {w}x{h}")
    return np.frombuffer(raw, np.uint8).reshape(h, w).astype(np.float32) / 127.5 - 1.0


def pack(header: dict, payload: bytes = b"") -> bytes:
    hb = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<I", len(hb)) + hb + payload


def unpack(data: bytes) -> tuple[dict, bytes]:
    if len(data) < 4:
        raise ValueError("short frame")
    (n,) = struct.unpack_from("<I", data, 0)
    if n > len(data) - 4 or n > 1 << 20:
        raise ValueError("bad header length")
    return json.loads(data[4:4 + n].decode("utf-8")), data[4 + n:]


# ----------------------------------------------------------------------------
# Single GPU worker thread. Everything that touches the engine runs here (load,
# warmup, bench, process) so MPS/CoreML only ever see one thread.
#
# The worker *pulls* frames itself: when an inference finishes it immediately
# takes the next pending frame under the same lock the event loop uses to swap
# pending slots, so there is no event-loop hop between two inferences (that hop
# costs ~1-3 ms, which is 5%+ of a 40 ms engine).
# ----------------------------------------------------------------------------
class GpuWorker:
    """One thread per engine. Several workers (an engine *pool*, e.g. Core ML on the ANE
    plus torch on the GPU) share one Condition and pull from the same pending slots."""

    def __init__(self, cv: threading.Condition, frame_source, frame_done, name: str = "gpu"):
        self.cv = cv
        self.name = name
        self._ctl: collections.deque = collections.deque()
        self._stopping = False
        self.frame_source = frame_source   # frame_source(worker), cv held -> (engine, frame) | None
        self.frame_done = frame_done       # frame_done(worker, ...) on this thread after each frame
        self.engine = None                 # set when this worker serves frames
        self.ms = float("inf")             # benchmarked ms/frame of self.engine
        self.serving = False
        self.waiting = False
        self.busy = False
        self.last_end = 0.0
        self.last_kf: dict = {}   # (client, seed) -> the framing id of the last frame painted (header `kf`)
        self.gap_ms_ema = 0.0
        self.ms_ema = 0.0
        self.done = 0
        self.thread = threading.Thread(target=self._run, name=f"engine-{name}", daemon=True)
        self.thread.start()

    def submit(self, fn, *args, **kw) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        with self.cv:
            self._ctl.append((loop, fut, fn, args, kw))
            self.cv.notify_all()
        return fut

    def poke(self):
        with self.cv:
            self.cv.notify_all()

    def stop(self):
        with self.cv:
            self._stopping = True
            self.serving = False
            self.cv.notify_all()

    def _run(self):
        while True:
            with self.cv:
                while True:
                    if self._stopping:
                        return
                    if self._ctl:
                        ctl, job = self._ctl.popleft(), None
                        break
                    job = self.frame_source(self)
                    if job is not None:
                        ctl = None
                        break
                    self.waiting = True
                    self.cv.wait()
                    self.waiting = False
                self.busy = True
            try:
                if ctl is not None:
                    loop, fut, fn, args, kw = ctl
                    try:
                        res, err = fn(*args, **kw), None
                    except BaseException as e:  # noqa: BLE001 - forwarded to the awaiting coroutine
                        res, err = None, e
                    try:
                        loop.call_soon_threadsafe(_resolve, fut, res, err)
                    except RuntimeError:  # loop closed during shutdown
                        return
                else:
                    engine, frame = job
                    if frame.kf is not None:
                        # held: the last frame this engine painted on the stream (client, seed) had
                        # the same framing id; judged here, in the order the engine paints, so drops
                        # and a pool's other engines can't make it stale
                        k = (frame.conn.cid, frame.seed)
                        frame.held = self.last_kf.pop(k, None) == frame.kf
                        self.last_kf[k] = frame.kf
                        while len(self.last_kf) > 64:
                            self.last_kf.pop(next(iter(self.last_kf)))
                    t0 = time.perf_counter()
                    gap = t0 - self.last_end if self.last_end and frame.t_decoded < self.last_end else None
                    try:
                        out, err = DreamServer._infer(engine, frame), None
                    except BaseException as e:  # noqa: BLE001
                        out, err = None, e
                    self.last_end = t1 = time.perf_counter()
                    self.frame_done(self, frame, out, err, t0, t1, gap)
            finally:
                self.busy = False


def _resolve(fut: asyncio.Future, res, err):
    if fut.cancelled():
        return
    if err is not None:
        fut.set_exception(err)
    else:
        fut.set_result(res)


# ----------------------------------------------------------------------------
# Engine selection
# ----------------------------------------------------------------------------
def _bench_image(w: int, h: int) -> np.ndarray:
    """Structured test image (gradients, pillars, a light) so engines do real work."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([x / w * 180 + 30, y / h * 120 + 40, (1 - x / w) * 160 + 40], -1)
    img[(x // (w / 8)).astype(int) % 2 == 0] *= 0.55
    d = np.hypot(x - w * 0.6, y - h * 0.3)
    img += (np.clip(1 - d / (0.25 * w), 0, 1) ** 2 * 200)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


class _BenchLock:
    """Advisory flock on ~/hypnagogia-cache/bench.lock (shared-GPU etiquette on the mini)."""

    def __init__(self, timeout: float = 30.0):
        self.path = Path.home() / "hypnagogia-cache" / "bench.lock"
        self.timeout = timeout
        self.fd = None
        self.held = False

    def __enter__(self):
        if not self.path.parent.is_dir():
            return self
        self.fd = open(self.path, "a+")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.held = True
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    log.warning("bench.lock busy for %.0fs; timing without it", self.timeout)
                    break
                time.sleep(0.1)
        return self

    def __exit__(self, *exc):
        if self.fd:
            if self.held:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.fd.close()


# ----------------------------------------------------------------------------
# Process isolation. coremltools' predict() holds the GIL for most of an ANE call
# (a spinning Python thread gets ~27% of its normal rate), which serialises a
# CoreML engine against a torch engine, the event loop and the JPEG codecs. Running
# an engine in its own process removes all GIL coupling; the round trip costs
# ~1 ms of pickling for a 512x512 frame.
# ----------------------------------------------------------------------------
def _engine_child(conn, name: str, cfg: dict, level: int):
    logging.basicConfig(level=level, format="%(asctime)s %(levelname).1s %(name)s[" + name + "]: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "urllib3", "huggingface_hub", "filelock", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        eng = engines_pkg.load(name, **cfg)
        meta = {k: getattr(eng, k, None) for k in ("name", "model", "width", "height", "device", "flexible", "xframe", "wants_depth", "depth_graft", "takes_cut", "takes_held", "held_only", "lora")}
        conn.send(("ok", meta))
    except BaseException as e:  # noqa: BLE001
        conn.send(("err", f"{type(e).__name__}: {e}"))
        return
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        op, args = msg
        try:
            if op == "process":
                args, kw = args
                out = eng.process(*args, **kw)
                if not isinstance(out, np.ndarray):
                    out = np.asarray(out)
                conn.send(("ok", out))
            elif op == "warmup":
                eng.warmup()
                conn.send(("ok", None))
            else:
                conn.send(("err", f"unknown op {op}"))
        except BaseException as e:  # noqa: BLE001
            conn.send(("err", f"{type(e).__name__}: {e}"))


class ProcEngine:
    """Engine proxy running the real engine in a spawned child process."""

    def __init__(self, name: str, cfg: dict):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_engine_child, name=f"engine-{name}",
                                 args=(child, name, cfg, logging.getLogger().level), daemon=True)
        self._proc.start()
        child.close()
        status, meta = self._recv(timeout=None)
        if status != "ok":
            self.close()
            raise RuntimeError(meta)
        for k, v in meta.items():
            setattr(self, k, v)
        self.device = f"{self.device} (proc)"

    def _recv(self, timeout):
        try:
            return self._conn.recv()
        except (EOFError, OSError) as e:
            raise RuntimeError(f"engine process died (exit code {self._proc.exitcode})") from e

    def _call(self, op, *args):
        self._conn.send((op, args))
        status, val = self._recv(None)
        if status != "ok":
            raise RuntimeError(val)
        return val

    def warmup(self):
        self._call("warmup")

    def process(self, image, prompt, strength, seed, negative=None, **kw):
        return self._call("process", (image, prompt, strength, seed, negative), {k: v for k, v in kw.items() if v is not None})

    def close(self):
        try:
            self._conn.send(None)
        except Exception:  # noqa: BLE001
            pass
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.kill()


def load_and_warm(name: str, cfg: dict, isolate: bool = False) -> Any:
    t0 = time.perf_counter()
    eng = ProcEngine(name, cfg) if isolate else engines_pkg.load(name, **cfg)
    t1 = time.perf_counter()
    eng.warmup()
    t2 = time.perf_counter()
    log.info("engine %-14s loaded in %.1fs, warmed in %.1fs (%s, %s, %dx%d on %s)", name,
             t1 - t0, t2 - t1, getattr(eng, "name", name), getattr(eng, "model", "?"),
             eng.width, eng.height, getattr(eng, "device", "?"))
    return eng


def bench_engine(eng, runs: int = 5, strength: float = 0.5) -> float:
    img = _bench_image(eng.width, eng.height)
    prompt = "a luminous impossible cathedral, oil painting"
    for _ in range(3):  # MPS/ANE clocks and caches need a few calls to settle
        eng.process(img, prompt, strength, 1234)
    with _BenchLock():
        times = []
        for i in range(runs):
            frame = np.roll(img, 2 * i, axis=1)  # slow pan: engines with motion-gated caches must pay for it
            t = time.perf_counter()
            eng.process(frame, prompt, strength, 1234)
            times.append((time.perf_counter() - t) * 1000)
    # mean, not median: engines with feature caching (DeepCache) run a full pass every Nth
    # frame, and a median would hide exactly that cost
    return float(np.mean(times))


def release_engine(eng):
    close = getattr(eng, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001
            log.exception("engine close failed")
    del eng
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass


# ----------------------------------------------------------------------------
# Connections / frames
# ----------------------------------------------------------------------------
@dataclass
class Frame:
    conn: "Conn"
    id: int
    prompt: str
    negative: str | None
    strength: float
    seed: int
    out_w: int
    out_h: int
    jpeg: bytes
    t_recv: float
    image: np.ndarray | None = None
    t_decoded: float = 0.0
    engine: str = ""
    depth: np.ndarray | None = None   # the capture's relative inverse depth in [-1, 1] (header `depth`)
    cut: bool = False                 # the prompt changed where nobody sees: no morph (header `cut`)
    kf: int | None = None             # the capture's framing id (header `kf`; sent in the fresh look)
    fid: int | None = None            # the framing id the pool routes by (header `fid`, sent in every look; else `kf`)
    held: bool | None = None          # set by the worker: it repeats the last framing this engine painted on its stream


@dataclass
class Conn:
    cid: int
    ws: web.WebSocketResponse
    peer: str
    pending: Frame | None = None
    received: int = 0
    completed: int = 0
    dropped: int = 0
    done_times: collections.deque = field(default_factory=lambda: collections.deque(maxlen=64))
    closed: bool = False
    want_started: bool = False   # opt-in: notify when a frame starts on the GPU
    depth_warned: bool = False   # a bad depth field was logged once

    async def send_json(self, obj: dict):
        if self.closed or self.ws.closed:
            return
        try:
            await self.ws.send_str(json.dumps(obj, separators=(",", ":")))
        except (ConnectionError, RuntimeError):
            self.closed = True

    async def send_bytes(self, data: bytes):
        if self.closed or self.ws.closed:
            return
        try:
            await self.ws.send_bytes(data)
        except (ConnectionError, RuntimeError):
            self.closed = True


# ----------------------------------------------------------------------------
# The server
# ----------------------------------------------------------------------------
class DreamServer:
    def __init__(self, args):
        self.args = args
        self.engine_name = "loading"
        self.status = "loading"          # loading -> warming -> ready | error
        self.status_detail = ""
        self.fallback_reason = ""
        self.candidates: list[dict] = []
        self.fps_estimate = 0.0
        self.bench_ms = 0.0
        self.conns: collections.OrderedDict[int, Conn] = collections.OrderedDict()
        self._cid = itertools.count(1)
        self._bg: set[asyncio.Task] = set()
        self._tasks: list[asyncio.Task] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self.lock = threading.Condition()  # guards conns / pending slots / workers
        self.workers: list[GpuWorker] = []  # serving engines, fastest first
        self._loading: list[GpuWorker] = []
        self._pool_mode = False
        self._pool_kf: collections.OrderedDict = collections.OrderedDict()   # (client, seed) -> last framing id handed out
        self.codec = ThreadPoolExecutor(max_workers=max(2, min(4, (os.cpu_count() or 4) // 3)),
                                        thread_name_prefix="codec")
        # stats
        self.done_times: collections.deque = collections.deque(maxlen=256)
        self.ms_infer_ema = 0.0
        self.ms_total_ema = 0.0
        self.total_done = 0
        self.total_dropped = 0
        self.gpu_busy_s = 0.0
        self._busy_window: collections.deque = collections.deque(maxlen=256)  # (t_end, busy_s)
        self.gap_ms_ema = 0.0              # GPU idle time between back-to-back frames
        self.started = time.time()

    @property
    def engine(self):
        """Primary (fastest) serving engine; defines the capture size."""
        ws = self.workers
        return ws[0].engine if ws else None

    @property
    def busy(self) -> bool:
        return any(w.busy for w in self.workers)

    @property
    def flexible(self) -> bool:
        """Every serving engine takes the client's own capture size (see takes_size)."""
        return bool(self.workers) and all(getattr(w.engine, "flexible", False) for w in self.workers)

    # -- info ------------------------------------------------------------------
    @property
    def wants_depth(self) -> bool:
        return any(bool(getattr(w.engine, "wants_depth", False)) for w in self.workers)

    def info(self) -> dict:
        e = self.engine
        w = e.width if e else (self.args.width or 512)
        h = e.height if e else (self.args.height or 512)
        return {
            "engine": getattr(e, "name", self.engine_name) if e else self.engine_name,
            "model": getattr(e, "model", "") if e else (self.args.model or ""),
            "width": int(w), "height": int(h),
            "flexible": self.flexible,
            "xframe": bool(getattr(e, "xframe", False)) if e else False,
            # the client should send each capture's depth (header `depth`): an engine paints with it
            "depth": self.wants_depth,
            "depth_graft": float(getattr(e, "depth_graft", 0.0) or 0.0) if e else 0.0,
            "lora": getattr(e, "lora", None) if e else None,
            "held_only": (getattr(e, "held_only", "") or "") if e else "",
            # share of cheap DeepCache passes (in-process torch engines only; None elsewhere)
            "dc_reuse": round(float(e.dc_reuse), 3) if e is not None and isinstance(getattr(e, "dc_reuse", None), float) else None,
            "fps_estimate": round(self.live_fps() or self.fps_estimate, 2),
            "device": getattr(e, "device", "") if e else "",
            "warming": self.status != "ready",
            "status": self.status,
            "status_detail": self.status_detail,
            "fallback_reason": self.fallback_reason,
            "module": self.engine_name,
            "candidates": self.candidates,
            "ms_infer": round(self.ms_infer_ema or self.bench_ms, 1),
            "clients": len(self.conns),
            "gap_ms": round(self.gap_ms_ema, 2),
            "pool": [{"module": w.name, "engine": getattr(w.engine, "name", w.name),
                      "device": getattr(w.engine, "device", ""), "ms": round(w.ms, 1),
                      "ms_live": round(w.ms_ema, 1), "done": w.done} for w in self.workers],
            # frames a client should keep in flight to keep every engine busy
            "inflight_hint": len(self.workers) + 1 if self.workers else 2,
            "jpeg": JPEG_LIB,
        }

    def _conn_list(self) -> list[Conn]:
        with self.lock:
            return list(self.conns.values())

    def live_fps(self) -> float:
        now = time.perf_counter()
        recent = [t for t in self.done_times if now - t < 2.0]
        if len(recent) < 2:
            return 0.0
        return (len(recent) - 1) / max(1e-3, recent[-1] - recent[0])

    # -- lifecycle -------------------------------------------------------------
    async def on_startup(self, app):
        self.loop = asyncio.get_running_loop()
        self._tasks = [asyncio.create_task(self._load_engine(), name="engine-loader"),
                       asyncio.create_task(self._stats_loop(), name="stats")]

    async def on_shutdown(self, app):
        log.info("shutting down: closing %d websocket(s)", len(self.conns))
        for c in self._conn_list():
            c.closed = True
            try:
                await c.ws.close(code=WSCloseCode.GOING_AWAY, message=b"server shutdown")
            except Exception:  # noqa: BLE001
                pass

    async def on_cleanup(self, app):
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self.codec.shutdown(wait=False, cancel_futures=True)
        for w in self.workers + self._loading:
            w.stop()
            proc = getattr(w.engine, "_proc", None)   # isolated engine: don't wait for it
            if proc is not None and proc.is_alive():
                proc.terminate()

    # -- engine loading ----------------------------------------------------------
    def _engine_cfg(self, name: str) -> dict:
        cfg: dict[str, Any] = {}
        if self.args.width:
            cfg["width"] = self.args.width
        if self.args.height:
            cfg["height"] = self.args.height
        if self.args.model and name != "mock":
            cfg["model"] = self.args.model
        for kv in self.args.engine_arg or []:
            k, _, v = kv.partition("=")
            try:
                cfg[k] = json.loads(v)
            except json.JSONDecodeError:
                cfg[k] = v
        return cfg

    def _isolate(self, name: str) -> bool:
        iso = self.args.isolate
        if name == "mock" and iso != "always":
            return False
        if iso == "auto":  # a pool shares the GIL otherwise; single engines stay in-process
            return self._pool_mode
        return iso == "always"

    def _new_worker(self, name: str) -> GpuWorker:
        w = GpuWorker(self.lock, self._next_job, self._frame_done, name)
        self._loading.append(w)
        return w

    async def _try_engine(self, name: str, size: tuple[int, int] | None):
        """Load + warm + bench `name` on its own worker thread. Returns worker or None."""
        rec = {"name": name}
        if not self.workers:  # a pool keeps serving (status ready) while later members load
            self.engine_name, self.status = name, "warming"
        self.status_detail = f"loading {name}"
        await self._broadcast_info()
        w = self._new_worker(name)
        cfg = self._engine_cfg(name)
        if size and "width" not in cfg:
            cfg["width"], cfg["height"] = size
        try:
            eng = await w.submit(load_and_warm, name, cfg, self._isolate(name))
            self.status_detail = f"benchmarking {name}"
            ms = await w.submit(bench_engine, eng, self.args.bench_runs)
        except Exception as e:  # noqa: BLE001
            rec.update(ok=False, error=f"{type(e).__name__}: {e}"[:300])
            self.candidates.append(rec)
            log.warning("engine %s unavailable: %s", name, rec["error"])
            log.debug("traceback", exc_info=True)
            w.stop()
            self._loading.remove(w)
            return None
        rec.update(ok=True, ms=round(ms, 1), fps=round(1000 / ms, 2), size=[eng.width, eng.height],
                   device=getattr(eng, "device", ""))
        self.candidates.append(rec)
        log.info("engine %-14s %.1f ms/frame (%.2f fps) at %dx%d", name, ms, 1000 / ms, eng.width, eng.height)
        w.engine, w.ms = eng, ms
        return w

    async def _drop_worker(self, w: GpuWorker):
        eng, w.engine = w.engine, None
        await w.submit(release_engine, eng)
        w.stop()
        if w in self._loading:
            self._loading.remove(w)

    async def _load_engine(self):
        a = self.args
        present = set(engines_pkg.available())
        mode = a.auto_select
        if a.engine == "auto":
            prefs = a.prefer or os.environ.get("HYPNAGOGIA_ENGINES", "")
            prefs = [p.strip() for p in prefs.split(",") if p.strip()] or engines_pkg.DEFAULT_PREFERENCE
            names = [n for n in prefs if n in present and n != "mock"]
            log.info("auto: candidate engines %s, select=%s (present: %s)", names, mode, sorted(present))
        else:
            names = [n.strip() for n in a.engine.replace("+", ",").split(",") if n.strip()]
            mode = "pool" if len(names) > 1 else "first"

        self._pool_mode = mode == "pool"
        kept: list[GpuWorker] = []
        size = (a.width, a.height) if a.width and a.height else None
        for name in names:
            w = await self._try_engine(name, size)
            if w is None:
                continue
            if mode == "pool":  # pool members must share the capture size; bench mode lets each
                # engine run at its own machine-tuned default and simply keeps the highest fps
                size = size or (w.engine.width, w.engine.height)
                kept.append(w)
                if len(kept) == 1:  # start serving right away with the first engine
                    await self._serve(kept)
                    self.status, self.engine_name = "ready", name
                    await self._broadcast_info()
            elif mode == "bench":
                # earlier preference wins unless the newcomer is clearly faster: the startup bench
                # is noisy on a shared machine, and the default order puts the ANE engine (which
                # leaves the GPU to the browser's renderer) first. An engine that paints with the
                # capture's depth counts as DEPTH_PREFERENCE times faster: without depth the model
                # paints over the geometry (a pillar becomes the wall behind it)
                eff = lambda x: x.ms / (DEPTH_PREFERENCE if getattr(x.engine, "wants_depth", False) else 1.0)  # noqa: E731
                if kept and eff(kept[0]) <= eff(w) * (1.0 + a.prefer_margin):
                    await self._drop_worker(w)
                else:
                    for old in kept:
                        await self._drop_worker(old)
                    kept = [w]
            else:  # first
                kept = [w]
                break

        if mode == "pool" and kept:
            best = min(w.ms for w in kept)
            for w in [w for w in kept if w.ms > best * a.pool_slack]:
                log.info("pool: dropping %s (%.0f ms, > %.1fx the fastest)", w.name, w.ms, a.pool_slack)
                kept.remove(w)
                with self.lock:
                    w.serving = False
                await self._drop_worker(w)

        if not kept:
            if a.engine != "auto" and a.strict:
                log.critical("engine %s failed and --strict given; exiting", a.engine)
                os._exit(2)
            why = "; ".join(f"{c['name']}: {c.get('error')}" for c in self.candidates) or "no engine modules found"
            self.fallback_reason = why
            bar = "!" * 78
            log.warning("\n%s\n!!  NO REAL DIFFUSION ENGINE LOADED — FALLING BACK TO THE CPU MOCK ENGINE\n"
                        "!!  %s\n%s", bar, why, bar)
            w = await self._try_engine("mock", size)
            kept = [w]

        await self._serve(kept)
        best = self.workers[0]
        self.bench_ms = best.ms
        self.fps_estimate = sum(1000.0 / w.ms for w in self.workers if w.ms > 0)
        self.status, self.status_detail = "ready", ""
        self.engine_name = "+".join(w.name for w in self.workers)
        e = best.engine
        log.info("READY: engine=%s model=%s %dx%d device=%s  ~%.1f ms/frame  (%.2f fps engine-only%s)",
                 self.engine_name, getattr(e, "model", "?"), e.width, e.height,
                 "+".join(str(getattr(w.engine, "device", "?")) for w in self.workers), best.ms,
                 self.fps_estimate, f", pool of {len(self.workers)}" if len(self.workers) > 1 else "")
        grafted = {bool(getattr(w.engine, "wants_depth", False)) for w in self.workers}
        if len(grafted) > 1:
            log.warning("the pool mixes engines that paint with depth and engines that don't: their frames will "
                        "look different (--engine-arg depth_graft=0 makes torch_turbo match a stock pool)")
        await self._broadcast_info()

    async def _serve(self, kept: list[GpuWorker]):
        with self.lock:
            for w in kept:
                w.serving = True
                if w in self._loading:
                    self._loading.remove(w)
            self.workers = sorted(kept, key=lambda w: w.ms)
            self.lock.notify_all()

    async def _broadcast_info(self):
        msg = {"type": "info", **self.info()}
        await asyncio.gather(*(c.send_json(msg) for c in self._conn_list()),
                             return_exceptions=True)

    # -- scheduling (runs on the GPU thread, under self.lock) ----------------------
    def _next_job(self, worker: GpuWorker):
        """Round-robin over connections with a decoded pending frame. In a pool, a slower
        idle engine leaves a frame to a faster idle one that would take it."""
        if not worker.serving or worker.engine is None:
            return None
        faster_idle = []
        for other in self.workers:
            if other is worker:
                break
            if other.waiting and other.serving:
                faster_idle.append(other)
        # --pool-held carry: a frame repeating its stream's last framing (header fid, else kf) goes only to an engine
        # that carries the stream (takes_held: cross-frame attention, DeepCache), so a view held still
        # doesn't alternate engines; off unless such an engine serves
        carry = (getattr(self.args, "pool_held", "carry") == "carry" and len(self.workers) > 1
                 and any(o.serving and getattr(o.engine, "takes_held", False) for o in self.workers))

        def refuses(w, f, k):
            return (carry and f.fid is not None and not getattr(w.engine, "takes_held", False)
                    and self._pool_kf.get(k) == f.fid)
        for cid, conn in list(self.conns.items()):
            f = conn.pending
            if f is None or conn.closed:
                continue
            k = (cid, f.seed)
            if refuses(worker, f, k):
                continue
            if any(not refuses(o, f, k) for o in faster_idle):
                return None
            self.conns.move_to_end(cid)   # the served connection goes to the back of the round
            if f.fid is not None:
                self._pool_kf[k] = f.fid
                self._pool_kf.move_to_end(k)
                while len(self._pool_kf) > 64:
                    self._pool_kf.popitem(last=False)
            conn.pending = None
            f.engine = worker.name
            if conn.want_started:
                try:
                    self.loop.call_soon_threadsafe(self._spawn, conn.send_json(
                        {"type": "started", "id": f.id, "ms_infer": round(self.ms_infer_ema or self.bench_ms, 1)}))
                except RuntimeError:
                    pass
            return worker.engine, f
        return None

    def _frame_done(self, worker: GpuWorker, frame: Frame, res, err, t0: float, t1: float,
                    gap: float | None):
        """Engine thread: record timing and hand the result to the event loop."""
        self._busy_window.append((t1, (t1 - t0) / max(1, len(self.workers))))
        worker.done += 1
        proc = getattr(worker.engine, "_proc", None)
        if err is not None and proc is not None and not proc.is_alive():
            # an isolated engine's process died: take it out of the pool while another engine serves,
            # so frames (and --pool-held carry, which then turns itself off) go to the engines left
            with self.lock:
                if worker.serving and any(o.serving for o in self.workers if o is not worker):
                    log.error("pool: %s's process died; no longer serving", worker.name)
                    worker.serving = False
                    self.lock.notify_all()
        if res is not None:
            worker.ms_ema = res[1] if not worker.ms_ema else 0.85 * worker.ms_ema + 0.15 * res[1]
        if gap is not None and gap < 0.25:
            g = gap * 1000.0
            worker.gap_ms_ema = g if not worker.gap_ms_ema else 0.9 * worker.gap_ms_ema + 0.1 * g
            self.gap_ms_ema = g if not self.gap_ms_ema else 0.9 * self.gap_ms_ema + 0.1 * g
        try:
            self.loop.call_soon_threadsafe(self._on_frame_done, frame, res, err, t0)
        except RuntimeError:
            pass

    def _on_frame_done(self, frame: Frame, res, err, t_start: float):
        if err is not None:
            log.error("engine.process failed on frame %s: %s", frame.id, err, exc_info=err)
            self._spawn(self._send_failed(frame, str(err)))
        else:
            out, ms_infer = res
            self._spawn(self._finish(frame, out, ms_infer, t_start))

    def _spawn(self, coro):
        t = asyncio.get_running_loop().create_task(coro)
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)

    async def _send_failed(self, frame: Frame, message: str):
        await frame.conn.send_json({"type": "error", "id": frame.id, "message": message[:300]})
        await frame.conn.send_json({"type": "dropped", "id": frame.id})

    @staticmethod
    def _infer(engine, frame: Frame):
        img = frame.image
        if (img.shape[1], img.shape[0]) != (engine.width, engine.height) and not takes_size(engine, img.shape[1], img.shape[0]):
            img = resize(img, engine.width, engine.height)
        t = time.perf_counter()
        kw = {}
        if frame.depth is not None and getattr(engine, "wants_depth", False):
            kw["depth"] = frame.depth
        if frame.cut and getattr(engine, "takes_cut", False):
            kw["cut"] = True
        if frame.held is not None and getattr(engine, "takes_held", False):
            kw["held"] = frame.held
        out = engine.process(img, frame.prompt, frame.strength, frame.seed, frame.negative, **kw)
        if not isinstance(out, np.ndarray):
            out = np.asarray(out.convert("RGB") if hasattr(out, "convert") else out)
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
        return out, (time.perf_counter() - t) * 1000.0

    async def _finish(self, frame: Frame, out: np.ndarray, ms_infer: float, t_start: float):
        loop = asyncio.get_running_loop()
        conn = frame.conn

        def encode():
            return jpeg_encode(resize(out, frame.out_w, frame.out_h), self.args.quality)
        try:
            payload = await loop.run_in_executor(self.codec, encode)
        except RuntimeError:  # executor shut down
            return
        except Exception as ex:  # noqa: BLE001
            await self._send_failed(frame, f"result encode: {ex}")
            return
        now = time.perf_counter()
        ms_total = (now - frame.t_recv) * 1000.0
        hdr = {"type": "result", "id": frame.id, "width": frame.out_w, "height": frame.out_h,
               "ms_infer": round(ms_infer, 2), "ms_total": round(ms_total, 2),
               "ms_queue": round((t_start - frame.t_decoded) * 1000.0, 2),
               "ms_decode": round((frame.t_decoded - frame.t_recv) * 1000.0, 2), "engine": frame.engine}
        await conn.send_bytes(pack(hdr, payload))
        conn.completed += 1
        conn.done_times.append(now)
        self.done_times.append(now)
        self.total_done += 1
        a = 0.15
        self.ms_infer_ema = ms_infer if not self.ms_infer_ema else (1 - a) * self.ms_infer_ema + a * ms_infer
        self.ms_total_ema = ms_total if not self.ms_total_ema else (1 - a) * self.ms_total_ema + a * ms_total

    # -- frames in -----------------------------------------------------------------
    async def _on_frame(self, conn: Conn, data: bytes):
        t_recv = time.perf_counter()
        hdr = None
        try:
            hdr, payload = unpack(data)
            if hdr.get("type") != "frame":
                raise ValueError(f"unexpected binary message type {hdr.get('type')!r}")
            fid = int(hdr["id"])
            fmt = hdr.get("format", "jpeg")
            if fmt not in ("jpeg", "jpg"):
                raise ValueError(f"unsupported format {fmt!r}")
            neg = hdr.get("negative")
            frame = Frame(conn=conn, id=fid, prompt=str(hdr.get("prompt") or ""),
                          negative=str(neg) if neg else None,
                          strength=float(min(1.0, max(0.0, float(hdr.get("strength", 0.5))))),
                          seed=int(hdr.get("seed", 0)) & 0x7FFFFFFF,
                          out_w=int(hdr.get("width") or 0), out_h=int(hdr.get("height") or 0),
                          jpeg=payload, t_recv=t_recv)
            if frame.out_w or frame.out_h:
                _check_side(frame.out_w, frame.out_h)
            frame.cut = hdr.get("cut") is True
            kf, framing = hdr.get("kf"), hdr.get("fid")
            frame.kf = kf if isinstance(kf, int) and not isinstance(kf, bool) else None
            frame.fid = framing if isinstance(framing, int) and not isinstance(framing, bool) else frame.kf
            if hdr.get("depth") is not None and self.wants_depth:
                try:
                    frame.depth = _depth_field(hdr["depth"])
                except (ValueError, TypeError, KeyError) as e:   # the image is still good: dream without it
                    if not conn.depth_warned:
                        conn.depth_warned = True
                        log.warning("client %d sent a bad depth field (%s); ignoring it", conn.cid, e)
        except Exception as e:  # noqa: BLE001
            fid = hdr.get("id") if isinstance(hdr, dict) else None
            err = {"type": "error", "message": f"bad frame: {e}"[:300]}
            if isinstance(fid, int):
                err["id"] = fid
            await conn.send_json(err)
            if isinstance(fid, int):  # let the client release that capture
                await conn.send_json({"type": "dropped", "id": fid})
            return
        conn.received += 1
        e = self.engine
        tw = e.width if e else (self.args.width or 0)
        th = e.height if e else (self.args.height or 0)
        if self.flexible and takes_size(e, frame.out_w, frame.out_h):
            tw, th = frame.out_w, frame.out_h   # the client's own capture size

        def decode():
            img = jpeg_decode(frame.jpeg)
            h, w = img.shape[:2]
            return img, w, h, (resize(img, tw, th) if tw and th and (w, h) != (tw, th) else img)
        try:
            img, w, h, img_in = await asyncio.get_running_loop().run_in_executor(self.codec, decode)
        except RuntimeError:
            return
        except Exception as ex:  # noqa: BLE001
            await self._send_failed(frame, f"jpeg decode: {ex}")
            return
        frame.out_w, frame.out_h = frame.out_w or w, frame.out_h or h
        # engine may have become ready (with another size) during decode
        e = self.engine
        if e is not None and (img_in.shape[1], img_in.shape[0]) != (e.width, e.height) \
                and not (self.flexible and takes_size(e, img_in.shape[1], img_in.shape[0])):
            img_in = resize(img, e.width, e.height)
        frame.image, frame.jpeg = img_in, b""
        frame.t_decoded = time.perf_counter()
        if conn.closed:
            return
        # latest-wins: only a *decoded* frame replaces the pending one, so the GPU always
        # has something ready to start while a newer capture is still being decoded.
        with self.lock:
            old, conn.pending = conn.pending, frame
            self.lock.notify_all()
        if old is not None:
            conn.dropped += 1
            self.total_dropped += 1
            await conn.send_json({"type": "dropped", "id": old.id})

    # -- websocket -----------------------------------------------------------------
    async def ws_handler(self, request: web.Request):
        ws = web.WebSocketResponse(max_msg_size=MAX_MSG, heartbeat=20.0, compress=False)
        await ws.prepare(request)
        conn = Conn(cid=next(self._cid), ws=ws, peer=request.remote or "?",
                    want_started=request.query.get("started") in ("1", "true"))
        with self.lock:
            self.conns[conn.cid] = conn
        log.info("client #%d connected from %s (%d total)", conn.cid, conn.peer, len(self.conns))
        await conn.send_json({"type": "info", **self.info()})
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    await self._on_frame(conn, msg.data)
                elif msg.type == WSMsgType.TEXT:
                    try:
                        obj = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    t = obj.get("type")
                    if t == "ping":
                        await conn.send_json({"type": "pong", "t": obj.get("t"), "server_time": time.time()})
                    elif t == "info":
                        await conn.send_json({"type": "info", **self.info()})
                    elif t == "hello":  # optional capabilities handshake
                        conn.want_started = "started" in (obj.get("want") or [])
                elif msg.type == WSMsgType.ERROR:
                    log.warning("client #%d ws error: %s", conn.cid, ws.exception())
        finally:
            with self.lock:
                conn.closed = True
                conn.pending = None
                self.conns.pop(conn.cid, None)
            log.info("client #%d disconnected (recv %d, done %d, dropped %d)", conn.cid,
                     conn.received, conn.completed, conn.dropped)
        return ws

    # -- stats -----------------------------------------------------------------------
    async def _stats_loop(self):
        last_log = 0.0
        while True:
            await asyncio.sleep(1.0)
            if len(self.workers) > 1:  # startup bench is noisy; prefer the engine that is fast *now*
                with self.lock:
                    self.workers = sorted(self.workers, key=lambda w: w.ms_ema or w.ms)
            if not self.conns and not self.done_times:
                continue
            now = time.perf_counter()
            busy = sum(b for t, b in self._busy_window if now - t < 2.0)
            fps = self.live_fps()
            q = sum(1 for c in self._conn_list() if c.pending is not None)
            base = {"type": "stats", "fps": round(fps, 2), "ms_infer": round(self.ms_infer_ema, 1),
                    "queue": q, "ms_total": round(self.ms_total_ema, 1),
                    "gpu_util": round(min(1.0, busy / 2.0), 3), "clients": len(self.conns),
                    "gap_ms": round(self.gap_ms_ema, 2),
                    "status": self.status}
            for c in self._conn_list():
                recent = [t for t in c.done_times if now - t < 2.0]
                cfps = (len(recent) - 1) / max(1e-3, recent[-1] - recent[0]) if len(recent) > 1 else 0.0
                await c.send_json({**base, "conn_fps": round(cfps, 2), "dropped": c.dropped})
            if fps > 0 and time.time() - last_log > self.args.log_every:
                last_log = time.time()
                log.info("%.2f fps | infer %.1f ms | gap %.2f ms | e2e(server) %.1f ms | gpu %.0f%% | "
                         "clients %d | done %d dropped %d",
                         fps, self.ms_infer_ema, self.gap_ms_ema, self.ms_total_ema, base["gpu_util"] * 100,
                         len(self.conns), self.total_done, self.total_dropped)


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------
@web.middleware
async def headers_mw(request: web.Request, handler):
    resp = await handler(request)
    if not isinstance(resp, web.WebSocketResponse):
        resp.headers.setdefault("Cache-Control", "no-cache")
        resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    return resp


def build_app(args) -> web.Application:
    srv = DreamServer(args)
    app = web.Application(middlewares=[headers_mw], client_max_size=MAX_MSG)
    app["dream"] = srv
    client_dir = Path(args.client_dir).resolve() if args.client_dir else CLIENT_DIR

    async def api_info(request):
        return web.json_response(srv.info())

    async def static(request: web.Request):
        rel = request.match_info.get("path", "") or "index.html"
        p = (client_dir / rel).resolve()
        if not p.is_relative_to(client_dir):
            raise web.HTTPForbidden()
        if p.is_dir():
            p = p / "index.html"
        if not p.is_file():
            if rel == "index.html":
                return web.Response(status=404, content_type="text/html",
                                    text="<h1>HYPNAGOGIA</h1><p>client/index.html not found. "
                                         "Server is up: <a href='/api/info'>/api/info</a></p>")
            raise web.HTTPNotFound()
        return web.FileResponse(p, headers={"Cache-Control": "no-cache"})

    app.router.add_get("/api/info", api_info)
    app.router.add_get("/ws", srv.ws_handler)
    app.router.add_get("/", static)
    app.router.add_get("/{path:.+}", static)
    app.on_startup.append(srv.on_startup)
    app.on_shutdown.append(srv.on_shutdown)
    app.on_cleanup.append(srv.on_cleanup)
    return app
