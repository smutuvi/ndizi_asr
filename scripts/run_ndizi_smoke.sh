#!/usr/bin/env bash
# Smoke: 2 samples per Ndizi dataset × Gemini 2.5 Flash + Sunbird.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m ndizi_asr.cli \
  --config configs/ndizi_gemini_sunbird.yaml \
  --env-file .env \
  --max-samples 2 \
  --experiment-name ndizi_smoke_gemini25_vs_sunbird \
  "$@"
