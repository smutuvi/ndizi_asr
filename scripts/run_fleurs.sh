#!/usr/bin/env bash
# Run default FLEURS Afro evaluation across Gemini Flash models via OpenRouter.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "Missing .env — copy .env.example and set OPENROUTER_API_KEY" >&2
  exit 1
fi

python -m gemini_asr_bench.cli \
  --config configs/fleurs_afro.yaml \
  --env-file .env \
  "$@"
