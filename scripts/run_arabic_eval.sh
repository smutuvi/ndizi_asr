#!/usr/bin/env bash
# Arabic eval: 2.5 / 3.1 / 3.5 / 3.6 on FLEURS + CV + MGB-2 + MGB-3 (400/source).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f .env ]]; then
  echo "Missing .env — copy .env.example and set OPENROUTER_API_KEY" >&2
  exit 1
fi

python -m ndizi_asr.cli \
  --config configs/arabic_eval_n400.yaml \
  --env-file .env \
  "$@"
