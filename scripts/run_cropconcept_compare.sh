#!/usr/bin/env bash
# Multi-model CropConcept Oromo smoke/compare from ndizi_asr.
#
# Usage:
#   ./scripts/run_cropconcept_compare.sh              # run default models, n=10
#   ./scripts/run_cropconcept_compare.sh list          # list models used by this script
#   ./scripts/run_cropconcept_compare.sh list-all      # list full ndizi_asr registry
#   ./scripts/run_cropconcept_compare.sh run [flags…]  # same as default run
#   MAX_SAMPLES=10 ./scripts/run_cropconcept_compare.sh
#
# Defaults target the GPU box layout:
#   EVAL_ROOT=/local/ndizi/4.Scripts/ndizi_eval
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

list_run_models() {
  echo "Models in this script (${#MODELS[@]}):"
  local i=1
  for m in "${MODELS[@]}"; do
    printf "  %d. %s\n" "$i" "$m"
    i=$((i + 1))
  done
  echo
  echo "Group alias: oromo  (same set)"
}

CMD="${1:-run}"
case "$CMD" in
  list|models)
    list_run_models
    exit 0
    ;;
  list-all|list-registry)
    python -m ndizi_asr.cli --list-models
    exit 0
    ;;
  run|compare)
    shift || true
    ;;
  -h|--help|help)
    sed -n '2,12p' "$0"
    exit 0
    ;;
  -*)
    # Flags passed directly → treat as run with those extra flags
    ;;
  *)
    # Unknown first token: pass through as extra CLI flags to ndizi_asr
    ;;
esac

if [[ ! -f "$CSV" ]]; then
  echo "CSV not found: $CSV" >&2
  echo "Set CSV=... or EVAL_ROOT=... to your ndizi_eval tree." >&2
  exit 2
fi

echo "CSV=$CSV"
echo "MAX_SAMPLES=$MAX_SAMPLES"
echo "EXPERIMENT=$EXPERIMENT"
list_run_models

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
