#!/usr/bin/env bash
# Smart fusion: best-2 Ethio CTC + LLM editor.
#
# ASR hyp cache (keep this; all editor A/Bs reuse it):
#   results/llm_fusion_full/fusion_predictions.jsonl
#
# Build / refresh that cache once (runs ASR + editor on all counted clips):
#   ./scripts/run_llm_fusion_smoke.sh
#   # → results/llm_fusion_full/{fusion_predictions.jsonl,fusion_metrics.json,fusion_summary.csv}
#
# Opus 5.5 few-shot baseline (auto --reuse-hyps when cache exists):
#   EDITOR=opus-5.5 EXPERIMENT=llm_fusion_opus55_fs8 \
#     ./scripts/run_llm_fusion_smoke.sh --few-shot 8 --seed 42 --no-resume
#
# Error sheet → prompt tweak → re-run:
#   python -m ndizi_asr.error_sheet \
#     --predictions results/llm_fusion_opus55_fs8/fusion_predictions.jsonl
#   # (also accepts legacy fusion_predictions_fs8.jsonl in that dir)
#   # edit FUSION_* prompts in src/ndizi_asr/ensemble.py
#   EDITOR=opus-5.5 EXPERIMENT=llm_fusion_opus55_fs8_promptv2 \
#     ./scripts/run_llm_fusion_smoke.sh --few-shot 8 --seed 42 --no-resume
#
# Override hyp cache path:
#   REUSE_HYPS=/path/to/fusion_predictions.jsonl ./scripts/run_llm_fusion_smoke.sh ...
# Disable reuse:
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

# Default ASR hypothesis cache from the full counted run
REUSE_HYPS="${REUSE_HYPS:-${ROOT}/results/llm_fusion_full/fusion_predictions.jsonl}"

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
#   - this run IS the full cache rebuild (llm_fusion_full)
has_reuse_flag=0
for arg in "$@"; do
  if [[ "$arg" == "--reuse-hyps" ]]; then
    has_reuse_flag=1
    break
  fi
done
if [[ "$has_reuse_flag" -eq 0 && -n "${REUSE_HYPS}" && "$EXPERIMENT" != "llm_fusion_full" ]]; then
  if [[ -f "$REUSE_HYPS" ]]; then
    REUSE_ARGS=(--reuse-hyps "$REUSE_HYPS")
    echo "Reusing ASR hyps: $REUSE_HYPS"
  else
    echo "WARNING: REUSE_HYPS not found ($REUSE_HYPS); will run ASR." >&2
  fi
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
  "${REUSE_ARGS[@]}" \
  --output-dir results \
  --experiment-name "$EXPERIMENT" \
  "$@"
