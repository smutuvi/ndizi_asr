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
# Glossary from few-shot only (no eval pollution), then Opus A/B:
#   # Prefer an existing fs8 run that has few_shot_ids.json, e.g.:
#   #   results/llm_fusion_opus55_fs8  or  results/llm_fusion_gemini_fs8
#   python -m ndizi_asr.build_glossary \
#     --run-dir results/llm_fusion_opus55_fs8 \
#     --predictions results/llm_fusion_full/fusion_predictions.jsonl \
#     -o results/glossaries/cropconcept_fs8_seed42.txt
#   # After a NEW fs8 fusion (writes few_shot_examples.jsonl), you can omit --predictions:
#   #   python -m ndizi_asr.build_glossary \
#   #     --run-dir results/llm_fusion_opus55_fs8_glossary \
#   #     -o results/glossaries/cropconcept_fs8_seed42.txt
#   ./scripts/run_llm_fusion_smoke.sh --few-shot 8 --editor opus-5.5 --seed 42 --no-resume \
#     --glossary results/glossaries/cropconcept_fs8_seed42.txt \
#     --reuse-hyps results/llm_fusion_full/fusion_predictions.jsonl \
#     --experiment-name llm_fusion_opus55_fs8_glossary
#
# Error sheet (Opus fs8 baseline → prompt tweaks; no glossary):
#   python -m ndizi_asr.error_sheet \
#     --predictions results/llm_fusion_opus55_fs8/fusion_predictions.jsonl
#   # read results/llm_fusion_opus55_fs8/error_sheet.md  then edit prompts in ensemble.py
#   ./scripts/run_llm_fusion_smoke.sh --few-shot 8 --editor opus-5.5 --seed 42 --no-resume \
#     --reuse-hyps results/llm_fusion_full/fusion_predictions.jsonl \
#     --experiment-name llm_fusion_opus55_fs8_promptv2
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
