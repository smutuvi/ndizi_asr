#!/usr/bin/env bash
# Full FLEURS test set with Sunbird Whisper-51 (local).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m gemini_asr_bench.cli \
  --config configs/fleurs_sunbird.yaml \
  --env-file .env \
  "$@"
