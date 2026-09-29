#!/usr/bin/env bash
# Build the Core ML models used by server/engines/coreml_turbo.py (macOS, Apple Silicon).
#
#   bench/coreml/build_models.sh            # default set: SD-Turbo 512 (kv2w8) + 384 (w8) + TAESD
#   bench/coreml/build_models.sh 384        # only 384 (e.g. on a 16 GB laptop)
#   bench/coreml/build_models.sh all        # + exact fp16 variants + SDXS
#
# Creates its own converter venv (torch pinned to 2.7.1, the newest torch coremltools 9.0 is
# tested with) in $CACHE/venv-coreml, downloads weights to $CACHE/hf, writes compiled
# .mlmodelc to $CACHE/coreml (the engine's default location). ~1 min per UNet on an M4 Pro.
# The first server start afterwards spends 20-60 s in the ANE compiler (cached by the OS).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CACHE="${HYPNAGOGIA_CACHE:-$HOME/hypnagogia-cache}"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
VENV="$CACHE/venv-coreml"
PY="$VENV/bin/python"
WHAT="${1:-default}"

mkdir -p "$CACHE/coreml" "$CACHE/hf"
if [ ! -x "$PY" ]; then
  "$UV" venv "$VENV" --python 3.12
fi
"$UV" pip install --python "$PY" -q "torch==2.7.1" diffusers transformers accelerate safetensors \
  "coremltools>=9.0" pillow numpy
export HF_HOME="$CACHE/hf"
conv() { nice -n 10 "$PY" "$ROOT/bench/coreml/convert.py" "$@"; }

case "$WHAT" in
  384)
    conv --model sdturbo --res 384 --w8 --suffix _w8 --vae taesd ;;
  512)
    conv --model sdturbo --res 512 --kv-down 2 --w8 --suffix _kv2w8 --vae taesd ;;
  default)
    conv --model sdturbo --res 512 --kv-down 2 --w8 --suffix _kv2w8 --vae taesd
    conv --model sdturbo --res 384 --w8 --suffix _w8 --vae taesd ;;
  all)
    conv --model sdturbo --res 512 --kv-down 2 --w8 --suffix _kv2w8 --vae taesd
    conv --model sdturbo --res 384 --w8 --suffix _w8 --vae taesd
    conv --model sdturbo --res 512 384
    conv --model sdturbo --res 512 --w8 --suffix _w8
    conv --model sdxs --res 512 384 256 --vae sdxs ;;
  *) echo "usage: $0 [default|384|512|all]"; exit 2 ;;
esac
# the .mlpackage sources are not needed at runtime
rm -rf "$CACHE"/coreml/*.mlpackage
ls "$CACHE/coreml" | grep mlmodelc
