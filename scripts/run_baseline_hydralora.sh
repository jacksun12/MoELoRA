#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 用法：
# Usage:
#   ./scripts/run_baseline_hydralora.sh            # 使用环境变量中的默认 GPU / use the default GPU from the environment
#   CUDA_VISIBLE_DEVICES=6 ./scripts/run_baseline_hydralora.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python scripts/train_trl_peft_baseline.py
