"""Play the legacy pygame game (diffused-rays, Dec 2025) in a browser.

The original is a desktop pygame app. This wrapper runs it headless (SDL dummy
video driver), streams the window as MJPEG, and forwards browser key presses
into pygame. The game code itself is untouched.

    python legacy/web_play.py --port 8766      # then open http://localhost:8766
                                               # (--host 0.0.0.0 to reach it from other devices)
"""
import argparse
import io
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

GAME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diffused-rays")
sys.path.insert(0, GAME_DIR)

import pygame  # noqa: E402
from PIL import Image  # noqa: E402

_held = set()
_frame = {"jpeg": b"", "seq": 0}
_cond = threading.Condition()
_last_grab = [0.0]


class _Pressed:
    def __getitem__(self, k):
        return k in _held


pygame.key.get_pressed = lambda: _Pressed()

_orig_flip = pygame.display.flip


def _flip():
    _orig_flip()
    now = time.monotonic()
    if now - _last_grab[0] < 1 / 30:
        return
    _last_grab[0] = now
    surf = pygame.display.get_surface()
    if surf is None:
        return
    img = Image.frombytes("RGB", surf.get_size(), pygame.image.tobytes(surf, "RGB"))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=82)
    with _cond:
        _frame["jpeg"] = buf.getvalue()
        _frame["seq"] += 1
        _cond.notify_all()


pygame.display.flip = _flip

KEYMAP = {
    "ArrowUp": "K_UP", "ArrowDown": "K_DOWN", "ArrowLeft": "K_LEFT", "ArrowRight": "K_RIGHT",
    " ": "K_SPACE", "[": "K_LEFTBRACKET", "]": "K_RIGHTBRACKET", "-": "K_MINUS", "=": "K_EQUALS",
    ",": "K_COMMA", ".": "K_PERIOD",
}
KEYMAP.update({"+": "K_EQUALS", "_": "K_MINUS"})  # shifted variants
for ch in "wasdtzb":  # G (depth) is left out: it pulls a multi-GB SD1.5 + ControlNet download
    KEYMAP[ch] = "K_" + ch

PAGE = b"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Diffused Rays (Dec 2025)</title>
<style>html,body{margin:0;height:100%;background:#111;color:#aaa;font:13px ui-monospace,monospace}
body{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px}
img{max-width:96vw;max-height:86vh;image-rendering:pixelated;border:1px solid #333}</style></head>
<body><img src="/stream" alt="game">
<div>legacy: diffused-rays (Dec 2025) &middot; WASD/arrows move &middot; Space SD view &middot; T texture &middot;
B trippy &middot; Z zones &middot; [ ] style &middot; - = blend</div>
<script>
const send=(t,k)=>fetch('/key?type='+t+'&key='+encodeURIComponent(k),{method:'POST'});
const down=new Set();
addEventListener('keydown',e=>{if(e.key==='Escape')return;e.preventDefault();
  const k=e.key.length===1?e.key.toLowerCase():e.key;if(down.has(k))return;down.add(k);send('down',k)});
addEventListener('keyup',e=>{const k=e.key.length===1?e.key.toLowerCase():e.key;down.delete(k);send('up',k)});
addEventListener('blur',()=>{for(const k of down)send('up',k);down.clear()});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
        elif path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = -1
            try:
                while True:
                    with _cond:
                        _cond.wait_for(lambda: _frame["seq"] != seq, timeout=5)
                        seq, jpg = _frame["seq"], _frame["jpeg"]
                    if not jpg:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        q = parse_qs(urlparse(self.path).query)
        name = KEYMAP.get(q.get("key", [""])[0])
        if name:
            k = getattr(pygame, name)
            if q.get("type", [""])[0] == "down":
                _held.add(k)
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode="", scancode=0))
            else:
                _held.discard(k)
                pygame.event.post(pygame.event.Event(pygame.KEYUP, key=k, mod=0, unicode="", scancode=0))
        self.send_response(204)
        self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"legacy diffused-rays on http://{args.host}:{args.port}", flush=True)
    os.chdir(GAME_DIR)
    # The game loads SD-Turbo with torch_dtype=fp16 but no variant, which pulls the
    # multi-GB fp32 files only to cast them down. Ask for the fp16 files instead:
    # same weights, same behaviour, a fraction of the download.
    try:
        import diffusers
        cls = diffusers.AutoPipelineForImage2Image
        _orig = cls.from_pretrained.__func__

        def _fp16(c, name, *a, **kw):
            if name == "stabilityai/sd-turbo":
                kw.setdefault("variant", "fp16")
            return _orig(c, name, *a, **kw)

        cls.from_pretrained = classmethod(_fp16)
    except Exception as e:  # diffusers missing: the game reports it itself
        print("fp16 variant patch skipped:", e, flush=True)
    import main as game  # the Dec 2025 game, unmodified
    game.main()


if __name__ == "__main__":
    main()
