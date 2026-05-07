#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# 默认只暴露一张 GPU，除非用户在外部显式覆盖。
# Default to a single visible GPU unless the user overrides it externally.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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
