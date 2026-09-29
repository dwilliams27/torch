#!/usr/bin/env bash
# HYPNAGOGIA one-command launcher.
#
#   ./run.sh                      # --engine auto on port 8765
#   ./run.sh --engine mock        # CPU fake diffusion, no model download
#   ./run.sh --engine torch_turbo --width 384 --height 384
#
# Env overrides:
#   HYPNAGOGIA_VENV=/path/to/venv   venv location (default: .venv next to this script)
#   HYPNAGOGIA_PYTHON=3.12          python version for a new venv
#   HYPNAGOGIA_SKIP_INSTALL=1       never touch the venv's packages
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# If ~/hypnagogia-cache exists, the venv lives there (outside the repo tree).
if [[ -n "${HYPNAGOGIA_VENV:-}" ]]; then VENV="$HYPNAGOGIA_VENV"
elif [[ -d "$HOME/hypnagogia-cache" && ! -d "$ROOT/.venv" ]]; then
  VENV="$HOME/hypnagogia-cache/venv-run"
else VENV="$ROOT/.venv"; fi
PYVER="${HYPNAGOGIA_PYTHON:-3.12}"
REQ="$ROOT/server/requirements.txt"

say()  { printf '\033[38;5;183m[hypnagogia]\033[0m %s\n' "$*"; }
warn() { printf '\033[38;5;209m[hypnagogia]\033[0m %s\n' "$*" >&2; }

# ---- locate uv -------------------------------------------------------------
UV=""
for cand in "$(command -v uv 2>/dev/null || true)" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" ; do
  if [[ -n "$cand" && -x "$cand" ]]; then UV="$cand"; break; fi
done

# ---- venv -------------------------------------------------------------------
if [[ ! -x "$VENV/bin/python" ]]; then
  if [[ -z "$UV" ]]; then
    warn "uv not found (looked in PATH, ~/.local/bin, ~/.cargo/bin)."
    warn "Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh"
    warn "or point HYPNAGOGIA_VENV at an existing python $PYVER venv."
    exit 1
  fi
  say "creating venv at $VENV (python $PYVER)"
  "$UV" venv --python "$PYVER" "$VENV"
fi
PY="$VENV/bin/python"

# ---- deps (skipped when requirements.txt is unchanged) ----------------------
if [[ "${HYPNAGOGIA_SKIP_INSTALL:-0}" != "1" ]]; then
  HASH="$( (cat "$REQ"; "$PY" -c 'import sys;print(sys.version)') | shasum -a 256 | cut -d' ' -f1)"
  STAMP="$VENV/.hypnagogia-req.sha256"
  if [[ ! -f "$STAMP" || "$(cat "$STAMP")" != "$HASH" ]]; then
    say "installing server/requirements.txt into $VENV"
    if [[ -n "$UV" ]]; then
      "$UV" pip install --python "$PY" -r "$REQ"
    else
      "$PY" -m pip install -r "$REQ"
    fi
    echo "$HASH" > "$STAMP"
  fi
fi

# ---- where to point a browser ------------------------------------------------
PORT=8765; BIND=127.0.0.1
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[$i]}" in
    --port) PORT="${args[$((i + 1))]:-$PORT}" ;;
    --port=*) PORT="${args[$i]#--port=}" ;;
    --host) BIND="${args[$((i + 1))]:-$BIND}" ;;
    --host=*) BIND="${args[$i]#--host=}" ;;
  esac
done

echo
say "open http://localhost:$PORT"
if [[ "$BIND" == "0.0.0.0" ]]; then
  IPS="$( (ifconfig 2>/dev/null || ip -4 addr 2>/dev/null) | awk '/inet / && $2 !~ /^127\./ {sub("/.*", "", $2); print $2}' | sort -u | tr '\n' ' ')"
  say "   or http://$(hostname -s 2>/dev/null || hostname):$PORT"
  for ip in $IPS; do say "   or http://$ip:$PORT"; done
  say "phones on the same network can open any of those."
else
  say "phones: restart with --host 0.0.0.0 to listen on the network."
fi
say "        no server needed for a look-around: add ?dream=mock"
echo

# Model caches: keep them out of the repo tree.
if [[ -z "${HF_HOME:-}" && -d "$HOME/hypnagogia-cache/hf" ]]; then
  export HF_HOME="$HOME/hypnagogia-cache/hf"
fi
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

exec "$PY" -m server "$@"
