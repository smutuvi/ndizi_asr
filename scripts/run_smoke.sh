#!/usr/bin/env bash
# Smoke test: 2 samples per language, single model.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m gemini_asr_bench.cli \
  --config configs/fleurs_afro.yaml \
  --env-file .env \
  --model google/gemini-2.5-flash \
  --max-samples 2 \
  --experiment-name fleurs_smoke \
  --concurrency 2 \
  "$@"
