#!/usr/bin/env bash
# Multi-model CropConcept Oromo compare from ndizi_asr.
#
# Ethiopia-style invocation (same flags as ethiopia_asr_eval run):
#
#   ./scripts/run_cropconcept_compare.sh \
#     --models \
#       gemini-2.5-flash \
#       omnilingualasr-ctc \
#       ethio-asr-oromo \
#       ethio-asr-multilingual-1b \
#       sunbird-whisper-51 \
#     --csv /local/ndizi/4.Scripts/ndizi_eval/data/manifest_cropconcept_oromo.csv \
#     --languages om \
#     --max-samples 10
#
# Or use the group alias:
#   ./scripts/run_cropconcept_compare.sh --models oromo --csv ... --languages om
#
# Helpers:
#   ./scripts/run_cropconcept_compare.sh list
#   ./scripts/run_cropconcept_compare.sh list-all
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

EVAL_ROOT="${EVAL_ROOT:-/local/ndizi/4.Scripts/ndizi_eval}"
DEFAULT_CSV="${EVAL_ROOT}/data/manifest_cropconcept_oromo.csv"
MAX_SAMPLES="${MAX_SAMPLES:-}"
EXPERIMENT="${EXPERIMENT:-cropconcept_oromo_all}"

DEFAULT_MODELS=(
  our-transcript
  gemini-2.5-flash
  gemini-3.5-flash
  omnilingualasr-ctc
  ethio-asr-oromo
  ethio-asr-multilingual-1b
  whisper
  sunbird-whisper-51
  sunflower-gemma4-e2b
  ndizi-gemma4-merged
)

list_run_models() {
  echo "Default models for this script (${#DEFAULT_MODELS[@]}):"
  local i=1
  for m in "${DEFAULT_MODELS[@]}"; do
    printf "  %d. %s\n" "$i" "$m"
    i=$((i + 1))
  done
  echo
  echo "Same as: --models oromo"
  echo
  echo "Example:"
  echo "  ./scripts/run_cropconcept_compare.sh \\"
  echo "    --models \\"
  for m in "${DEFAULT_MODELS[@]}"; do
    echo "      $m \\"
  done
  echo "    --csv ${DEFAULT_CSV} \\"
  echo "    --languages om \\"
  echo "    --max-samples ${MAX_SAMPLES}"
}

CMD="${1:-}"
case "$CMD" in
  list|models)
    list_run_models
    exit 0
    ;;
  list-all|list-registry)
    python -m ndizi_asr.cli --list-models
    exit 0
    ;;
  -h|--help|help)
    sed -n '2,25p' "$0"
    exit 0
    ;;
  run|compare)
    shift
    ;;
esac

# Detect whether the user already passed models / csv / languages.
USER_ARGS=("$@")
HAS_MODELS=0
HAS_CSV=0
HAS_LANG=0
HAS_MAX=0
HAS_NORM=0
HAS_OUT=0
HAS_EXP=0
i=0
while [[ $i -lt ${#USER_ARGS[@]} ]]; do
  case "${USER_ARGS[$i]}" in
    --model|--models) HAS_MODELS=1 ;;
    --csv) HAS_CSV=1 ;;
    --languages|--language-filter) HAS_LANG=1 ;;
    --max-samples) HAS_MAX=1 ;;
    --normalize) HAS_NORM=1 ;;
    --output-dir) HAS_OUT=1 ;;
    --experiment-name) HAS_EXP=1 ;;
  esac
  i=$((i + 1))
done

ARGS=(--env-file .env --data-root "$EVAL_ROOT")

if [[ $HAS_MODELS -eq 0 ]]; then
  ARGS+=(--models "${DEFAULT_MODELS[@]}")
fi
if [[ $HAS_CSV -eq 0 ]]; then
  ARGS+=(--csv "$DEFAULT_CSV")
fi
if [[ $HAS_LANG -eq 0 ]]; then
  ARGS+=(--languages om)
fi
if [[ $HAS_NORM -eq 0 ]]; then
  ARGS+=(--normalize oromo_fair)
fi
if [[ $HAS_MAX -eq 0 && -n "$MAX_SAMPLES" ]]; then
  ARGS+=(--max-samples "$MAX_SAMPLES")
fi
if [[ $HAS_OUT -eq 0 ]]; then
  ARGS+=(--output-dir results)
fi
if [[ $HAS_EXP -eq 0 ]]; then
  ARGS+=(--experiment-name "$EXPERIMENT")
fi

# Resolve relative --csv against EVAL_ROOT if needed (best-effort preview)
CSV_PREVIEW="$DEFAULT_CSV"
i=0
while [[ $i -lt ${#USER_ARGS[@]} ]]; do
  if [[ "${USER_ARGS[$i]}" == "--csv" && $((i + 1)) -lt ${#USER_ARGS[@]} ]]; then
    CSV_PREVIEW="${USER_ARGS[$((i + 1))]}"
  fi
  i=$((i + 1))
done
if [[ "$CSV_PREVIEW" != /* && ! -f "$CSV_PREVIEW" && -f "${EVAL_ROOT}/${CSV_PREVIEW}" ]]; then
  # Rewrite relative csv path to absolute under EVAL_ROOT for the CLI call
  NEW_ARGS=()
  i=0
  while [[ $i -lt ${#USER_ARGS[@]} ]]; do
    if [[ "${USER_ARGS[$i]}" == "--csv" && $((i + 1)) -lt ${#USER_ARGS[@]} ]]; then
      NEW_ARGS+=(--csv "${EVAL_ROOT}/${USER_ARGS[$((i + 1))]}")
      i=$((i + 2))
      continue
    fi
    NEW_ARGS+=("${USER_ARGS[$i]}")
    i=$((i + 1))
  done
  USER_ARGS=("${NEW_ARGS[@]}")
  CSV_PREVIEW="${EVAL_ROOT}/${CSV_PREVIEW#./}"
fi

if [[ ! -f "$CSV_PREVIEW" ]]; then
  # Only enforce when we can resolve a concrete path
  if [[ $HAS_CSV -eq 0 || "$CSV_PREVIEW" == /* || "$CSV_PREVIEW" == */* ]]; then
    if [[ ! -f "$CSV_PREVIEW" ]]; then
      echo "CSV not found: $CSV_PREVIEW" >&2
      echo "Pass --csv PATH or set EVAL_ROOT=... to your ndizi_eval tree." >&2
      exit 2
    fi
  fi
fi

echo "=== CropConcept compare ==="
echo "EVAL_ROOT=$EVAL_ROOT"
list_run_models
echo
echo "Running: python -m ndizi_asr.cli ${ARGS[*]} ${USER_ARGS[*]}"
echo

python -m ndizi_asr.cli "${ARGS[@]}" "${USER_ARGS[@]}"
