#!/usr/bin/env bash
# Multi-model CropConcept Oromo smoke/compare from ndizi_asr.
#
# Defaults target the GPU box layout:
#   EVAL_ROOT=/local/ndizi/4.Scripts/ndizi_eval
# Override with env vars as needed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

EVAL_ROOT="${EVAL_ROOT:-/local/ndizi/4.Scripts/ndizi_eval}"
CSV="${CSV:-${EVAL_ROOT}/data/manifest_cropconcept_oromo.csv}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"
EXPERIMENT="${EXPERIMENT:-cropconcept_oromo_n${MAX_SAMPLES}}"

# Default: same set as ndizi_eval --models oromo
MODELS=(
  gemini-2.5-flash
  omnilingualasr
  omnilingualasr-ctc
  ethio-asr-oromo
  ethio-asr-multilingual-1b
  sunbird-whisper-51
)

if [[ ! -f "$CSV" ]]; then
  echo "CSV not found: $CSV" >&2
  echo "Set CSV=... or EVAL_ROOT=... to your ndizi_eval tree." >&2
  exit 2
fi

echo "CSV=$CSV"
echo "MAX_SAMPLES=$MAX_SAMPLES"
echo "EXPERIMENT=$EXPERIMENT"
echo "MODELS=${MODELS[*]}"

python -m ndizi_asr.cli \
  --env-file .env \
  --model "${MODELS[@]}" \
  --csv "$CSV" \
  --data-root "$EVAL_ROOT" \
  --language-filter om \
  --normalize ethio \
  --max-samples "$MAX_SAMPLES" \
  --output-dir results \
  --experiment-name "$EXPERIMENT" \
  "$@"
