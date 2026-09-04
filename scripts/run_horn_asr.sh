#!/usr/bin/env bash
# Gemini 2.5 Flash on LesanAI/Horn-ASR (Amharic + Oromo).
# Smoke: ./scripts/run_horn_asr.sh --max-samples 20
# Full lang sets (1000 each): ./scripts/run_horn_asr.sh --max-samples 1000
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f .env ]]; then
  echo "Missing .env — copy .env.example and set OPENROUTER_API_KEY" >&2
  exit 1
fi

python -m gemini_asr_bench.cli \
  --config configs/horn_asr_gemini25.yaml \
  --env-file .env \
  "$@"
