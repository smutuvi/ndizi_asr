#!/usr/bin/env bash
# Compare Gemini 2.5 / 3.1 Flash Lite / 3.5 / 3.6 on the same sample cap.
# Datasets: FLEURS (sw/am/om/ha) + WaxalNLP (am/om) + Ndizi ×2
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f .env ]]; then
  echo "Missing .env — copy .env.example and set OPENROUTER_API_KEY" >&2
  exit 1
fi

MAX_SAMPLES="${MAX_SAMPLES:-100}"

python -m gemini_asr_bench.cli \
  --env-file .env \
  --model google/gemini-2.5-flash \
         google/gemini-3.1-flash-lite \
         google/gemini-3.5-flash \
         google/gemini-3.6-flash \
  --dataset smutuvi/ndizi-1 smutuvi/ndizi-1-2025 google/fleurs google/WaxalNLP \
  --max-samples "${MAX_SAMPLES}" \
  --streaming \
  --experiment-name "flash_compare_n${MAX_SAMPLES}" \
  --concurrency 4 \
  "$@"
