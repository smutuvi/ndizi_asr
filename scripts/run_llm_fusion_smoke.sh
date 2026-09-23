#!/usr/bin/env bash
# Smart fusion: Ethio CTC ×2 + Sunbird Whisper-51 + LLM editor.
#
# ASR hyp cache:
#   results/llm_fusion_full3/fusion_predictions.jsonl   # Ethio×2 + Sunbird (preferred)
#   results/llm_fusion_full/fusion_predictions.jsonl    # Ethio×2 only (fallback)
#
# Save 3-model cache from a completed sunbird fusion run (no LLM/ASR re-run):
#   python -m ndizi_asr.save_hyp_cache \
#     --from-dir results/llm_fusion_opus55_fs8_sunbird \
#     --from-file results/llm_fusion_full/fusion_predictions.jsonl \
#     -o results/llm_fusion_full3/fusion_predictions.jsonl
#
# 3-model Opus few-shot (auto-reuses full3 when present):
#   EXPERIMENT=llm_fusion_opus55_fs8_sunbird \
#     ./scripts/run_llm_fusion_smoke.sh --few-shot 8 --seed 42 --no-resume
#
# Override hyp cache / disable reuse:
#   REUSE_HYPS=/path/to/fusion_predictions.jsonl ./scripts/run_llm_fusion_smoke.sh ...
#   REUSE_HYPS= ./scripts/run_llm_fusion_smoke.sh ...
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

EVAL_ROOT="${EVAL_ROOT:-/local/ndizi/4.Scripts/ndizi_eval}"
CSV="${CSV:-${EVAL_ROOT}/data/manifest_cropconcept_oromo.csv}"
# MAX_SAMPLES=all  or empty  → full counted manifest
MAX_SAMPLES="${MAX_SAMPLES:-}"
EDITOR="${EDITOR:-opus-5.5}"
ASR_MODELS="${ASR_MODELS:-ethio-asr-multilingual-1b ethio-asr-oromo sunbird-whisper-51}"

# Default hyp cache: prefer 3-model (Ethio×2+Sunbird), else Ethio-only full run
if [[ -z "${REUSE_HYPS+x}" ]]; then
  if [[ -f "${ROOT}/results/llm_fusion_full3/fusion_predictions.jsonl" ]]; then
    REUSE_HYPS="${ROOT}/results/llm_fusion_full3/fusion_predictions.jsonl"
  else
    REUSE_HYPS="${ROOT}/results/llm_fusion_full/fusion_predictions.jsonl"
  fi
else
  REUSE_HYPS="${REUSE_HYPS}"
fi

SAMPLE_ARGS=()
if [[ -z "$MAX_SAMPLES" || "$MAX_SAMPLES" == "all" || "$MAX_SAMPLES" == "0" || "$MAX_SAMPLES" == "-1" ]]; then
  EXPERIMENT="${EXPERIMENT:-llm_fusion_full}"
else
  EXPERIMENT="${EXPERIMENT:-llm_fusion_n${MAX_SAMPLES}}"
  SAMPLE_ARGS=(--max-samples "$MAX_SAMPLES")
fi

REUSE_ARGS=()
# Auto-attach --reuse-hyps unless:
#   - REUSE_HYPS is empty
#   - caller already passed --reuse-hyps
#   - this run IS a full cache rebuild (llm_fusion_full / llm_fusion_full3)
has_reuse_flag=0
for arg in "$@"; do
  if [[ "$arg" == "--reuse-hyps" ]]; then
    has_reuse_flag=1
    break
  fi
done
if [[ "$has_reuse_flag" -eq 0 && -n "${REUSE_HYPS}" \
   && "$EXPERIMENT" != "llm_fusion_full" && "$EXPERIMENT" != "llm_fusion_full3" ]]; then
  if [[ -f "$REUSE_HYPS" ]]; then
    REUSE_ARGS=(--reuse-hyps "$REUSE_HYPS")
    echo "Reusing ASR hyps (missing models will be run): $REUSE_HYPS"
  else
    echo "WARNING: REUSE_HYPS not found ($REUSE_HYPS); will run ASR." >&2
  fi
fi

# shellcheck disable=SC2086
python -m ndizi_asr.ensemble \
  --env-file .env \
  --asr-models $ASR_MODELS \
  --editor "$EDITOR" \
  --csv "$CSV" \
  --data-root "$EVAL_ROOT" \
  --languages om \
  --normalize ethio \
  "${SAMPLE_ARGS[@]}" \
  "${REUSE_ARGS[@]}" \
  --output-dir results \
  --experiment-name "$EXPERIMENT" \
  "$@"
