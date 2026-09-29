# legacy: diffused-rays (Dec 2025)

The game Hypnagogia replaced: a pygame DDA raycaster that ran whole-frame SD-Turbo
img2img (Space), a stylized texture atlas (T), or ControlNet depth (G). Kept unmodified
as the "before" picture.
It was built with Claude Code in this repo on 2025-12-04..08 (the commits record no model).

Play in a browser (headless pygame, streamed as MJPEG, keys forwarded):

    python web_play.py --port 8766        # needs pygame, torch, diffusers, transformers, pillow

Or natively, with a window and its synth music: `cd diffused-rays && python main.py`
(adds `sounddevice`). `HYPNAGOGIA_MINI=user@host ../tools/play-on-mini.sh` runs both games on a remote Mac.
