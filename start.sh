#!/usr/bin/env bash
# One-command local start on macOS / Linux: creates .venv, installs, opens the web UI.
#   ./start.sh          UI + local OCR + cloud VLM clients (Claude / GPT / Gemini)
#   ./start.sh full     also local HuggingFace VLMs and the YOLO detector (large download)
#   ./start.sh update   reinstall after pulling new code
# API keys: put lines like  ANTHROPIC_API_KEY=sk-...  in a file named .env next to this script.
set -euo pipefail
cd "$(dirname "$0")"

EXTRAS="ui,ocr,anthropic,openai,gemini"
[ "${1:-}" = "full" ] && EXTRAS="all"

PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10+ is required (macOS: brew install python; Ubuntu: sudo apt install python3 python3-venv)."
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "Creating the virtual environment in .venv ..."
  "$PY" -m venv .venv || { echo "Could not create .venv (Ubuntu/Debian: sudo apt install python3-venv)"; exit 1; }
fi
VPY=.venv/bin/python
STAMP=".venv/drawid_installed_${EXTRAS//,/_}.txt"
[ "${1:-}" = "update" ] && rm -f .venv/drawid_installed_*.txt
if [ ! -f "$STAMP" ]; then
  echo "Installing drawing-identifier [$EXTRAS] - the first time takes a few minutes ..."
  "$VPY" -m pip install --upgrade pip
  if [ "$(uname -s)" = "Linux" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
    # no NVIDIA GPU: the CPU build of PyTorch is ~10x smaller than the default CUDA one
    "$VPY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || echo "  (CPU PyTorch index unreachable: using the default build)"
  fi
  "$VPY" -m pip install -e ".[$EXTRAS]"
  "$VPY" -m drawing_identifier.cli fonts || echo "  (fonts not downloaded: the synthetic generator will use built-in fonts)"
  echo ok > "$STAMP"
fi

if [ -f .env ]; then set -a; . ./.env; set +a; fi

if ! "$VPY" -c 'from drawing_identifier.transcription.engines import find_tesseract as f; import sys; sys.exit(0 if f() else 1)' 2>/dev/null; then
  echo "[optional] Tesseract not found: brew install tesseract  /  sudo apt install tesseract-ocr"
fi
command -v ollama >/dev/null 2>&1 || echo "[optional] Ollama not found: https://ollama.com/download, then: ollama pull qwen2.5vl:3b"
if [ -z "${ANTHROPIC_API_KEY:-}${OPENAI_API_KEY:-}${GEMINI_API_KEY:-}" ]; then
  echo "[optional] No cloud API key set: put ANTHROPIC_API_KEY=... in .env to use Claude."
fi
echo
echo "Starting the UI on http://127.0.0.1:7860 - Ctrl+C to stop."
exec "$VPY" -m drawing_identifier.cli ui --open
