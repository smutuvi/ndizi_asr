#!/usr/bin/env bash
# Smoke test: Sunbird Whisper-51 on FLEURS (2 samples / language).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m ndizi_asr.cli \
  --config configs/fleurs_sunbird.yaml \
  --env-file .env \
  --model Sunbird/asr-whisper-51-african-languages \
  --max-samples 2 \
  --experiment-name fleurs_smoke_sunbird \
  "$@"
