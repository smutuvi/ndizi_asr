#!/usr/bin/env bash
# Multi-ASR → Opus fusion on the full CropConcept Oromo counted set (61 clips).
#
# Few-shot 8 + retrieve (seed 42) ⇒ eval n≈53. No Sagalee-FT required.
#
#   cd /local/ndizi/4.Scripts/ndizi_asr && git pull
#   EVAL_ROOT=/local/ndizi/4.Scripts/ndizi_eval ./scripts/run_cropconcept_fusion_multi.sh
#
# Optional:
#   WITH_OMNI_CTC=1 WITH_OMNI_LLM=1 ./scripts/run_cropconcept_fusion_multi.sh
#   EDITOR=gemini-4-argon ./scripts/run_cropconcept_fusion_multi.sh
#   REUSE_HYPS=results/.../fusion_predictions.jsonl ./scripts/run_cropconcept_fusion_multi.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

EVAL_ROOT="${EVAL_ROOT:-/local/ndizi/4.Scripts/ndizi_eval}"
CSV="${CSV:-${EVAL_ROOT}/data/manifest_cropconcept_oromo.csv}"
EDITOR="${EDITOR:-opus-5.5}"
FEW_SHOT="${FEW_SHOT:-8}"
SEED="${SEED:-42}"
GLOSSARY="${GLOSSARY:-${ROOT}/configs/glossary_cropconcept_oromo.txt}"
EXPERIMENT="${EXPERIMENT:-cropconcept_oromo_counted_full61_multi_asr_opus_fs${FEW_SHOT}}"

ASR_MODELS=(
  ethio-asr-oromo
  ethio-asr-multilingual-1b
  sunbird-whisper-51
)
if [[ "${WITH_OMNI_CTC:-0}" == "1" ]]; then
  ASR_MODELS+=(omnilingualasr-ctc)
fi
if [[ "${WITH_OMNI_LLM:-0}" == "1" ]]; then
  ASR_MODELS+=(omnilingualasr)
fi

if [[ ! -f "$CSV" ]]; then
  echo "CSV not found: $CSV" >&2
  echo "Set EVAL_ROOT=/path/to/ndizi_eval" >&2
  exit 2
fi

ARGS=(
  --env-file .env
  --asr-models "${ASR_MODELS[@]}"
  --editor "$EDITOR"
  --csv "$CSV"
  --data-root "$EVAL_ROOT"
  --languages om
  --normalize ethio
  --few-shot "$FEW_SHOT"
  --few-shot-retrieve
  --seed "$SEED"
  --output-dir results
  --experiment-name "$EXPERIMENT"
)

if [[ -f "$GLOSSARY" ]]; then
  ARGS+=(--glossary "$GLOSSARY")
fi
if [[ -n "${REUSE_HYPS:-}" && -f "$REUSE_HYPS" ]]; then
  ARGS+=(--reuse-hyps "$REUSE_HYPS")
  echo "Reusing ASR hyps: $REUSE_HYPS"
fi

echo "=== Multi-ASR → Opus CropConcept fusion ==="
echo "CSV=$CSV"
echo "ASR: ${ASR_MODELS[*]}"
echo "Editor=$EDITOR  few-shot=$FEW_SHOT  seed=$SEED"
echo "Experiment=$EXPERIMENT"
echo

python -m ndizi_asr.ensemble "${ARGS[@]}" "$@"

OUT="results/${EXPERIMENT}"
echo
echo "=== Done → $OUT ==="
if [[ -f "${OUT}/fusion_predictions.jsonl" ]]; then
  python -m ndizi_asr.error_sheet \
    --predictions "${OUT}/fusion_predictions.jsonl" \
    --output-dir "$OUT" || true
fi
