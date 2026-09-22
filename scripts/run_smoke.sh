#!/usr/bin/env bash
# Smoke test: 2 samples per language × Gemini 2.5 / 3.5 / 3.6 Flash.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f .env ]]; then
  echo "Missing .env — copy .env.example and set OPENROUTER_API_KEY" >&2
  exit 1
fi

python -m ndizi_asr.cli \
  --config configs/fleurs_afro.yaml \
  --env-file .env \
  --model google/gemini-2.5-flash google/gemini-3.5-flash google/gemini-3.6-flash \
  --language swahili=sw_ke amharic=am_et oromo=om_et hausa=ha_ng \
  --max-samples 2 \
  --experiment-name fleurs_smoke_flash_compare \
  --concurrency 2 \
  "$@"
