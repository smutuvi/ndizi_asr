#!/usr/bin/env bash
# Gemini 3.1 Flash Lite on FLEURS (sw/am/om/ha) + WaxalNLP (am/om) + Ndizi.
# Override model with: ./scripts/run_gemini31.sh --model google/gemini-3.1-pro-preview
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f .env ]]; then
  echo "Missing .env — copy .env.example and set OPENROUTER_API_KEY" >&2
  exit 1
fi

python -m ndizi_asr.cli \
  --config configs/gemini31_afro.yaml \
  --env-file .env \
  "$@"
