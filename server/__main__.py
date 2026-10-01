"""python -m server [--engine auto|mock|torch_turbo|coreml_turbo|...] [--port 8765]"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading

from aiohttp import web

from . import engines
from .app import build_app


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m server", description="HYPNAGOGIA diffusion server")
    p.add_argument("--engine", default=os.environ.get("HYPNAGOGIA_ENGINE", "auto"),
                   help="engine module, 'auto', or a comma list to run a POOL of engines concurrently "
                        "(e.g. coreml_turbo,torch_turbo: ANE + GPU in parallel) "
                        f"(available: {', '.join(engines.available())})")
    p.add_argument("--prefer", default=None,
                   help="comma-separated preference list for --engine auto "
                        f"(default $HYPNAGOGIA_ENGINES or {','.join(engines.DEFAULT_PREFERENCE)})")
    p.add_argument("--auto-select", choices=["bench", "first", "pool"], default="bench",
                   help="auto: benchmark every loadable candidate and keep the fastest (bench), "
                        "take the first that loads (first; less memory/startup time), or keep all "
                        "that are within --pool-slack of the fastest and run them concurrently (pool)")
    p.add_argument("--prefer-margin", type=float, default=0.15,
                   help="auto/bench: a later candidate must be this much faster to displace an earlier one")
    p.add_argument("--pool-slack", type=float, default=2.5,
                   help="pool: drop engines slower than this factor x the fastest")
    p.add_argument("--pool-held", choices=["any", "carry"], default="carry",
                   help="pool: 'carry' (default) keeps a frame that repeats its stream's last framing (header fid, "
                        "or kf from older fresh-look clients) off engines that don't carry a stream from frame to "
                        "frame (no takes_held, e.g. coreml_turbo), so a view held still is painted by one engine; "
                        "'any' lets every engine take every frame")
    p.add_argument("--bench-runs", type=int, default=12, help="timed runs per candidate for auto selection")
    p.add_argument("--isolate", choices=["auto", "always", "never"], default="auto",
                   help="run engines in their own processes (no GIL sharing). auto = only for pools")
    p.add_argument("--strict", action="store_true", help="exit instead of falling back to mock")
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address; 0.0.0.0 to let phones and other machines connect")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--width", type=int, default=None, help="override engine capture width")
    p.add_argument("--height", type=int, default=None, help="override engine capture height")
    p.add_argument("--model", default=None, help="override the engine's model id/path")
    p.add_argument("--engine-arg", action="append", metavar="KEY=VALUE",
                   help="extra create_engine kwarg (VALUE parsed as JSON if possible); repeatable, "
                        "e.g. --engine-arg latency_ms=60")
    p.add_argument("--quality", type=int, default=88, help="result JPEG quality")
    p.add_argument("--client-dir", default=None, help="serve a different client directory")
    p.add_argument("--log-every", type=float, default=5.0, help="seconds between throughput log lines")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def lan_addresses() -> list[str]:
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("aiohttp.access", "PIL", "urllib3", "httpx", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger("hypnagogia")

    app = build_app(args)
    host = socket.gethostname().split(".")[0]
    urls = [f"http://localhost:{args.port}", f"http://{host}:{args.port}"] + \
           [f"http://{ip}:{args.port}" for ip in lan_addresses()]
    log.info("serving client + /ws on %s  (engine=%s; loading in background)", "  ".join(urls), args.engine)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:  # same flags as aiohttp
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((args.host, args.port))
            probe.listen(1)
    except OSError as e:
        log.error("cannot listen on %s:%d (%s) - another server running? pick one with --port",
                  args.host, args.port, e.strerror)
        sys.exit(1)
    try:
        web.run_app(app, host=args.host, port=args.port, print=None, access_log=None,
                    reuse_address=True, shutdown_timeout=3.0, handle_signals=True)
    finally:
        # A model load may still be running in the (daemon) GPU thread, and torch/CoreML
        # leave non-daemon helper threads around; don't let either hang the exit.
        others = [t for t in threading.enumerate() if t is not threading.main_thread() and not t.daemon]
        if others or app["dream"].busy:
            logging.shutdown()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
