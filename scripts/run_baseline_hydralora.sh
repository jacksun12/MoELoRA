#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Usage:
#   ./scripts/run_baseline_hydralora.sh            # default GPU from env
#   CUDA_VISIBLE_DEVICES=6 ./scripts/run_baseline_hydralora.sh
python scripts/train_trl_peft_baseline.py
