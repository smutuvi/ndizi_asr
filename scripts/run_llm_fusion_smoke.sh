#!/usr/bin/env bash
# Smart fusion: best-2 Ethio CTC + LLM editor.
#
# Baseline (text-only fusion, all counted clips):
#   ./scripts/run_llm_fusion_smoke.sh
#
# Few-shot (8 gold examples, score held-out ~22):
#   ./scripts/run_llm_fusion_smoke.sh --few-shot 8 --no-resume \
#     --reuse-hyps results/llm_fusion_full/fusion_predictions.jsonl \
#     --experiment-name llm_fusion_fs8
#
# Audio + hyps (use Gemini editor for multimodal):
#   ./scripts/run_llm_fusion_smoke.sh --audio --editor gemini --no-resume \
#     --reuse-hyps results/llm_fusion_full/fusion_predictions.jsonl \
#     --experiment-name llm_fusion_audio
#
# Few-shot + audio:
#   ./scripts/run_llm_fusion_smoke.sh --few-shot 8 --audio --editor gemini --no-resume \
#     --reuse-hyps results/llm_fusion_full/fusion_predictions.jsonl \
#     --experiment-name llm_fusion_fs8_audio
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

EVAL_ROOT="${EVAL_ROOT:-/local/ndizi/4.Scripts/ndizi_eval}"
CSV="${CSV:-${EVAL_ROOT}/data/manifest_cropconcept_oromo.csv}"
# MAX_SAMPLES=all  or empty  → full counted manifest
MAX_SAMPLES="${MAX_SAMPLES:-}"
EDITOR="${EDITOR:-opus}"

SAMPLE_ARGS=()
if [[ -z "$MAX_SAMPLES" || "$MAX_SAMPLES" == "all" || "$MAX_SAMPLES" == "0" || "$MAX_SAMPLES" == "-1" ]]; then
  EXPERIMENT="${EXPERIMENT:-llm_fusion_full}"
else
  EXPERIMENT="${EXPERIMENT:-llm_fusion_n${MAX_SAMPLES}}"
  SAMPLE_ARGS=(--max-samples "$MAX_SAMPLES")
fi

python -m ndizi_asr.ensemble \
  --env-file .env \
  --asr-models ethio-asr-multilingual-1b ethio-asr-oromo \
  --editor "$EDITOR" \
  --csv "$CSV" \
  --data-root "$EVAL_ROOT" \
  --languages om \
  --normalize ethio \
  "${SAMPLE_ARGS[@]}" \
  --output-dir results \
  --experiment-name "$EXPERIMENT" \
  "$@"
