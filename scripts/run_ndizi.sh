#!/usr/bin/env bash
# Full Ndizi eval: Gemini 2.5 Flash + Sunbird on ndizi-1 and ndizi-1-2025.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f .env ]]; then
  echo "Missing .env — set OPENROUTER_API_KEY and HF_TOKEN" >&2
  exit 1
fi

python -m gemini_asr_bench.cli \
  --config configs/ndizi_gemini_sunbird.yaml \
  --env-file .env \
  "$@"
