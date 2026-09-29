#!/usr/bin/env bash
# Owner-only, run from a laptop: copy this project to a Mac over ssh and (re)start both
# games there, reachable from other devices on the network.
#
#   HYPNAGOGIA_MINI=user@host tools/play-on-mini.sh         # new game :8765, legacy :8766
#   HYPNAGOGIA_MINI=user@host tools/play-on-mini.sh stop    # stop both
set -euo pipefail
[[ "$(id -un)" == _sink ]] && { echo "this script is for the repo owner, not sink lanes" >&2; exit 1; }
HOST="${HYPNAGOGIA_MINI:?set HYPNAGOGIA_MINI=user@host for the machine that runs the games}"
NAME="${HOST#*@}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ssh "$HOST" bash -s <<'EOF'
pkill -f 'hypnagogia/run.sh' 2>/dev/null || true
for p in 8765 8766; do lsof -tiTCP:$p -sTCP:LISTEN 2>/dev/null | xargs kill 2>/dev/null || true; done
sleep 2
for p in 8765 8766; do lsof -tiTCP:$p -sTCP:LISTEN 2>/dev/null | xargs kill -9 2>/dev/null || true; done
true
EOF
[[ "${1:-}" == "stop" ]] && { echo "stopped"; exit 0; }

rsync -a --delete --exclude .venv --exclude __pycache__ --exclude 'bench/out' "$ROOT"/ "$HOST":hypnagogia/
rsync -a --delete --exclude __pycache__ "$ROOT"/legacy/ "$HOST":hypnagogia-legacy/

ssh "$HOST" bash -s <<'EOF'
set -euo pipefail
mkdir -p ~/hypnagogia-cache/logs ~/hypnagogia-cache/hf
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"
VENV="${HYPNAGOGIA_VENV:-$HOME/hypnagogia-cache/venv-run}"
cd ~/hypnagogia
HYPNAGOGIA_VENV="$VENV" nohup ./run.sh --host 0.0.0.0 --port 8765 > ~/hypnagogia-cache/logs/play.log 2>&1 &
# run.sh writes this stamp only after every requirement is installed.
for i in $(seq 1 300); do [[ -f "$VENV/.hypnagogia-req.sha256" ]] && break; sleep 2; done
[[ -f "$VENV/.hypnagogia-req.sha256" ]] || { echo "venv not ready after 10 min; see ~/hypnagogia-cache/logs/play.log" >&2; exit 1; }
PY="$VENV/bin/python"
"$PY" -c "import pygame" 2>/dev/null || uv pip install --python "$PY" -q pygame
cd ~/hypnagogia-legacy
HF_HOME="$HOME/hypnagogia-cache/hf" nohup "$PY" web_play.py --host 0.0.0.0 --port 8766 > ~/hypnagogia-cache/logs/legacy.log 2>&1 &
sleep 4
lsof -tiTCP:8766 -sTCP:LISTEN >/dev/null || { echo "legacy failed to start; see ~/hypnagogia-cache/logs/legacy.log" >&2; exit 1; }
EOF
echo "new:    http://$NAME:8765   (engine picks and warms up first; log: ~/hypnagogia-cache/logs/play.log)"
echo "legacy: http://$NAME:8766   (Space = SD view; first use fetches SD-Turbo's VAE, ~160 MB)"
