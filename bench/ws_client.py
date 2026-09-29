#!/usr/bin/env python3
"""End-to-end load test for the HYPNAGOGIA /ws protocol.

Sends real JPEG frames at max rate with <= --inflight frames outstanding per
connection (like the game client), and reports whole-system throughput and
latency — network + JPEG decode + queueing + inference + encode — alongside the
server-reported inference time, so the gap between "engine FPS" and "system FPS"
is visible.

    python bench/ws_client.py                                 # ws://localhost:8765/ws, 20 s
    python bench/ws_client.py ws://localhost:8765/ws -d 30 --json
    python bench/ws_client.py --connections 3 --inflight 3    # exercise round-robin + drops
    python bench/ws_client.py --rate 15                       # game-like fixed capture cadence
    python bench/ws_client.py --images shots/ --save out/     # real captures in, results out

Needs: aiohttp, numpy, pillow (all in server/requirements.txt).
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import struct
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np
from PIL import Image


def pack(header: dict, payload: bytes) -> bytes:
    hb = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<I", len(hb)) + hb + payload


def unpack(data: bytes):
    (n,) = struct.unpack_from("<I", data, 0)
    return json.loads(data[4:4 + n]), data[4 + n:]


def synth_frames(w: int, h: int, n: int, quality: int) -> list[bytes]:
    """A slowly moving, lit colonnade: structured enough that the model does real work."""
    out = []
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = x / w - 0.5, y / h - 0.5
    for i in range(n):
        t = i / n * 2 * math.pi
        yaw = 0.15 * math.sin(t)
        horizon = 0.02 * math.sin(2 * t)
        vv = v - horizon
        depth = 0.35 / np.maximum(np.abs(vv), 0.02)                    # floor/ceiling distance
        wx = (u + yaw) * depth
        floor = ((np.floor(wx * 2) + np.floor(depth * 2 + i * 0.05)) % 2) * 0.25 + 0.35
        pillars = (np.abs(((u + yaw) * 9 + 0.5) % 1 - 0.5) < 0.12) & (np.abs(vv) < 0.28)
        light = np.exp(-((u + yaw - 0.15) ** 2 + (vv + 0.1) ** 2) * 18)
        fog = np.exp(-depth * 0.12)
        base = floor * fog
        r = base * 0.9 + light * 0.9 + pillars * 0.35
        g = base * 0.75 + light * 0.6 + pillars * 0.30
        b = base * 1.1 + light * 0.3 + pillars * 0.45
        img = np.clip(np.stack([r, g, b], -1) * 200, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=quality)
        out.append(buf.getvalue())
    return out


def load_frames(d: Path, w: int, h: int, quality: int) -> list[bytes]:
    out = []
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            im = Image.open(p).convert("RGB").resize((w, h), Image.BICUBIC)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality)
            out.append(buf.getvalue())
    if not out:
        sys.exit(f"no images in {d}")
    return out


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


class Client:
    def __init__(self, idx: int, args, frames_cache: dict):
        self.idx, self.args, self.frames_cache = idx, args, frames_cache
        self.sent: dict[int, float] = {}
        self.lat: list[float] = []
        self.ms_infer: list[float] = []
        self.ms_total: list[float] = []
        self.results = 0
        self.dropped = 0
        self.errors = 0
        self.done_times: list[float] = []
        self.info: dict = {}
        self.stats: dict = {}
        self.saved = 0
        self.closed_early = 0.0
        self.by_engine: dict[str, list[float]] = {}

    async def run(self, session: aiohttp.ClientSession, t_measure: float, t_end: float, ready_evt):
        a = self.args
        url = a.url + (("&" if "?" in a.url else "?") + "started=1" if a.jit else "")
        async with session.ws_connect(url, max_msg_size=64 << 20, compress=0) as ws:
            # wait for a ready info
            deadline = time.perf_counter() + a.ready_timeout
            while True:
                msg = await asyncio.wait_for(ws.receive(), timeout=max(0.1, deadline - time.perf_counter()))
                if msg.type == aiohttp.WSMsgType.TEXT:
                    obj = json.loads(msg.data)
                    if obj.get("type") == "info":
                        self.info = obj
                        if not obj.get("warming", False):
                            break
                        print(f"[c{self.idx}] server warming ({obj.get('status_detail') or obj.get('engine')})…",
                              file=sys.stderr)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    raise RuntimeError("socket closed before ready")
            ready_evt.set()
            w = a.width or self.info["width"]
            h = a.height or self.info["height"]
            key = (w, h)
            if key not in self.frames_cache:
                self.frames_cache[key] = (load_frames(Path(a.images), w, h, a.quality) if a.images
                                          else synth_frames(w, h, a.frames, a.quality))
            frames = self.frames_cache[key]
            slots = asyncio.Semaphore(a.inflight)
            fid = self.idx * 10_000_000
            stop = asyncio.Event()
            # --jit: send the next capture so it lands just as the GPU frees up
            can_send = asyncio.Event()
            can_send.set()
            send_at = [0.0]
            lead = [0.010]   # EMA of client-side transit (latency - server ms_total), seconds

            async def reader():
                async for msg in ws:
                    now = time.perf_counter()
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        hdr, payload = unpack(msg.data)
                        if hdr.get("type") != "result":
                            continue
                        t0 = self.sent.pop(hdr["id"], None)
                        slots.release()
                        if t0 is not None and "ms_total" in hdr:
                            tr = max(0.0, (now - t0) - hdr["ms_total"] / 1000.0)
                            lead[0] = 0.8 * lead[0] + 0.2 * tr
                        if a.jit and not self.sent:
                            send_at[0] = now
                            can_send.set()
                        if t0 is not None and t_measure <= now <= t_end:  # count by arrival time
                            self.results += 1
                            self.done_times.append(now)
                            self.lat.append((now - t0) * 1000)
                            self.ms_infer.append(hdr.get("ms_infer", float("nan")))
                            self.by_engine.setdefault(hdr.get("engine", "?"), []).append(hdr.get("ms_infer", float("nan")))
                            self.ms_total.append(hdr.get("ms_total", float("nan")))
                        if a.save and self.saved < a.save_count:
                            Path(a.save).mkdir(parents=True, exist_ok=True)
                            (Path(a.save) / f"c{self.idx}_{hdr['id']}_{hdr.get('engine', '')}.jpg").write_bytes(payload)
                            self.saved += 1
                        elif self.results == 1 and self.idx == 0:
                            Image.open(io.BytesIO(payload)).verify()  # sanity: valid JPEG
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        obj = json.loads(msg.data)
                        t = obj.get("type")
                        if t == "started" and a.jit:
                            send_at[0] = now + max(0.0, obj.get("ms_infer", 0) / 1000.0 - lead[0] - a.jit_margin / 1000.0)
                            can_send.set()
                        elif t == "dropped":
                            t0 = self.sent.pop(obj["id"], None)
                            slots.release()
                            if a.jit and not self.sent:
                                send_at[0] = now
                                can_send.set()
                            if t0 is not None and t_measure <= now <= t_end:
                                self.dropped += 1
                        elif t == "stats":
                            self.stats = obj
                        elif t == "error":
                            self.errors += 1
                            print(f"[c{self.idx}] server error: {obj}", file=sys.stderr)
                            if "id" not in obj:
                                slots.release()
                    else:
                        break
                    if stop.is_set() and not self.sent:
                        break

            rtask = asyncio.create_task(reader())
            i = 0
            next_t = time.perf_counter()
            while time.perf_counter() < t_end and not rtask.done():
                try:
                    await asyncio.wait_for(slots.acquire(), timeout=5.0)
                except asyncio.TimeoutError:
                    if rtask.done() or ws.closed:
                        break
                    print(f"[c{self.idx}] no response for 5 s (in flight: {list(self.sent)})", file=sys.stderr)
                    self.sent.clear()
                    slots = asyncio.Semaphore(a.inflight)
                    continue
                if a.jit:
                    try:
                        await asyncio.wait_for(can_send.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        can_send.set()
                        continue
                    can_send.clear()
                    await asyncio.sleep(max(0.0, send_at[0] - time.perf_counter()))
                if a.rate > 0:  # fixed capture cadence, like a game client (latest-wins does the rest)
                    next_t = max(next_t + 1.0 / a.rate, time.perf_counter())
                    await asyncio.sleep(max(0.0, next_t - time.perf_counter()))
                fid += 1
                hdr = {"type": "frame", "id": fid, "width": w, "height": h, "prompt": a.prompt,
                       "negative": None, "strength": a.strength, "seed": a.seed, "format": "jpeg"}
                self.sent[fid] = time.perf_counter()
                await ws.send_bytes(pack(hdr, frames[i % len(frames)]))
                i += 1
            stop.set()
            if rtask.done() or ws.closed:
                self.closed_early = time.perf_counter()
                print(f"[c{self.idx}] server closed the connection", file=sys.stderr)
            try:
                await asyncio.wait_for(rtask, timeout=5.0)
            except asyncio.TimeoutError:
                rtask.cancel()


async def main_async(args):
    frames_cache: dict = {}
    clients = [Client(i, args, frames_cache) for i in range(args.connections)]
    ready = [asyncio.Event() for _ in clients]
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # connect everything first; start the clock once all are ready
        t_holder = {}

        async def one(c, evt):
            await evt_start.wait()
            await c.run(session, t_holder["measure"], t_holder["end"], evt)

        # Two-phase: first wait for readiness via a throwaway info probe
        async with session.ws_connect(args.url) as ws:
            deadline = time.perf_counter() + args.ready_timeout
            info = {}
            while time.perf_counter() < deadline:
                msg = await asyncio.wait_for(ws.receive(), timeout=max(0.1, deadline - time.perf_counter()))
                if msg.type != aiohttp.WSMsgType.TEXT:
                    raise SystemExit(f"unexpected message {msg.type}")
                info = json.loads(msg.data)
                if info.get("type") == "info" and not info.get("warming"):
                    break
                if info.get("type") == "info":
                    print(f"server warming: {info.get('status_detail') or info.get('status')}", file=sys.stderr)
        print(f"server: engine={info.get('engine')} model={info.get('model')} "
              f"{info.get('width')}x{info.get('height')} device={info.get('device')} "
              f"fps_estimate={info.get('fps_estimate')}", file=sys.stderr)

        lock_fd = None
        if args.lock:
            import fcntl
            lp = Path(args.lock).expanduser()
            lp.parent.mkdir(parents=True, exist_ok=True)
            lock_fd = open(lp, "a+")
            t_wait = time.perf_counter()
            await asyncio.get_running_loop().run_in_executor(None, fcntl.flock, lock_fd, fcntl.LOCK_EX)
            print(f"holding {lp} (waited {time.perf_counter() - t_wait:.1f}s)", file=sys.stderr)
        now = time.perf_counter()
        t_holder["measure"] = now + args.warmup
        t_holder["end"] = now + args.warmup + args.duration
        evt_start = asyncio.Event()
        evt_start.set()
        try:
            await asyncio.gather(*(one(c, e) for c, e in zip(clients, ready)))
        finally:
            if lock_fd is not None:
                lock_fd.close()  # releases the flock

    dur = args.duration
    early = [c.closed_early for c in clients if c.closed_early]
    if early:  # measure only the time the server was actually there
        dur = max(1e-3, min(early) - t_holder["measure"])
    lat = [x for c in clients for x in c.lat]
    mi = [x for c in clients for x in c.ms_infer]
    mt = [x for c in clients for x in c.ms_total]
    res = sum(c.results for c in clients)
    by_engine: dict[str, list[float]] = {}
    for c in clients:
        for k, v in c.by_engine.items():
            by_engine.setdefault(k, []).extend(v)
    summary = {
        "url": args.url, "engine": info.get("engine"), "model": info.get("model"),
        "size": [args.width or info.get("width"), args.height or info.get("height")],
        "connections": args.connections, "inflight": args.inflight, "rate": args.rate, "jit": args.jit, "duration_s": round(dur, 2),
        "results": res, "dropped": sum(c.dropped for c in clients), "errors": sum(c.errors for c in clients),
        "fps_total": round(res / dur, 2),
        "fps_per_conn": [round(c.results / dur, 2) for c in clients],
        "latency_ms": {"p50": round(pct(lat, 50), 1), "p90": round(pct(lat, 90), 1),
                       "p99": round(pct(lat, 99), 1), "max": round(max(lat), 1) if lat else None},
        "server_ms_infer_mean": round(float(np.mean(mi)), 1) if mi else None,
        "server_ms_total_mean": round(float(np.mean(mt)), 1) if mt else None,
        # engines of a pool run concurrently: capacity is the sum of each one's rate
        "engine_only_fps": round(sum(1000.0 / float(np.mean(v)) for v in by_engine.values()), 2) if mi else None,
        "per_engine": {k: {"frames": len(v), "ms_infer": round(float(np.mean(v)), 1)} for k, v in by_engine.items()},
        "last_server_stats": clients[0].stats,
    }
    if summary["engine_only_fps"]:
        summary["system_efficiency"] = round(summary["fps_total"] / summary["engine_only_fps"], 3)
    if args.json:
        print(json.dumps(summary))
    else:
        L = summary["latency_ms"]
        print(f"\n  engine        {summary['engine']}  ({summary['model']})  {summary['size'][0]}x{summary['size'][1]}")
        print(f"  system FPS    {summary['fps_total']:.2f}   ({res} results in {dur:.0f}s, "
              f"{args.connections} conn x {args.inflight} in flight; per-conn {summary['fps_per_conn']})")
        pe = "  ".join(f"{k}: {v['frames']} @ {v['ms_infer']} ms" for k, v in summary["per_engine"].items())
        print(f"  engine FPS    {summary['engine_only_fps']}   (sum over engines of 1000 / mean ms_infer; {pe})"
              f"  -> efficiency {summary.get('system_efficiency')}")
        print(f"  latency ms    p50 {L['p50']}  p90 {L['p90']}  p99 {L['p99']}  max {L['max']}   "
              f"(server-side mean {summary['server_ms_total_mean']})")
        print(f"  dropped {summary['dropped']}  errors {summary['errors']}\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url", nargs="?", default="ws://localhost:8765/ws")
    p.add_argument("-d", "--duration", type=float, default=20.0)
    p.add_argument("--warmup", type=float, default=3.0, help="seconds excluded from measurement")
    p.add_argument("--inflight", type=int, default=2)
    p.add_argument("--rate", type=float, default=0.0,
                   help="send at most this many frames/s per connection (0 = as fast as slots allow). "
                        "With --rate above the server fps, latest-wins keeps the started frame fresh: "
                        "latency ~ infer + 1/rate instead of 2 x infer")
    p.add_argument("-c", "--connections", type=int, default=1)
    p.add_argument("--jit", action="store_true",
                   help="just-in-time mode: connect with ?started=1 and send the next capture timed to "
                        "arrive as the GPU frees up (latency ~ 1 x infer instead of 2 x)")
    p.add_argument("--jit-margin", type=float, default=8.0, help="ms of safety margin for --jit")
    p.add_argument("--width", type=int, default=0, help="frame size (default: server's preferred)")
    p.add_argument("--height", type=int, default=0)
    p.add_argument("--frames", type=int, default=48, help="distinct synthetic frames to cycle")
    p.add_argument("--images", default=None, help="directory of images to send instead of synthetic frames")
    p.add_argument("--quality", type=int, default=85, help="JPEG quality of sent frames")
    p.add_argument("--prompt", default="a drowned cathedral nave, luminous, oil painting, volumetric light")
    p.add_argument("--strength", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", default=None, help="directory to save some results into")
    p.add_argument("--save-count", type=int, default=8)
    p.add_argument("--ready-timeout", type=float, default=600.0, help="max seconds to wait for warmup")
    p.add_argument("--lock", nargs="?", const="~/hypnagogia-cache/bench.lock", default=None,
                   help="hold an exclusive flock on this file (default ~/hypnagogia-cache/bench.lock) "
                        "for the timed run -- shared-GPU etiquette")
    p.add_argument("--json", action="store_true", help="print one JSON summary line")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
