#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Default to physical GPU 7 unless user overrides externally.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
echo "[run_dynamic_moe] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python - <<'PY'
import importlib.util, sys
missing = [m for m in ("trl", "peft", "datasets") if importlib.util.find_spec(m) is None]
if missing:
    print("Missing dependencies:", ", ".join(missing))
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)
PY
python main_server_sim.py
