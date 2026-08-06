#!/usr/bin/env bash
# Compare Google Gemma 4 E2B vs Sunbird Sunflower-Gemma4-E2B on Ndizi (+ optional FLEURS).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m gemini_asr_bench.cli \
  --env-file .env \
  --model google/gemma-4-E2B-it \
  --model Sunbird/Sunflower-Gemma4-E2B \
  --dataset smutuvi/ndizi-1 \
  --dataset smutuvi/ndizi-1-2025 \
  --experiment-name gemma4_e2b_ndizi \
  "$@"
