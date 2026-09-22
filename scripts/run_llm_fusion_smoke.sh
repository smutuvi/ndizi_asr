#!/usr/bin/env bash
# Smart fusion smoke: best-2 Ethio CTC + LLM editor (Opus by default).
#
#   ./scripts/run_llm_fusion_smoke.sh
#   MAX_SAMPLES=10 ./scripts/run_llm_fusion_smoke.sh --editor opus
#   ./scripts/run_llm_fusion_smoke.sh --editor sonnet --max-samples 5
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

EVAL_ROOT="${EVAL_ROOT:-/local/ndizi/4.Scripts/ndizi_eval}"
CSV="${CSV:-${EVAL_ROOT}/data/manifest_cropconcept_oromo.csv}"
# MAX_SAMPLES=all  or 0  → full manifest; default 10 for smoke
MAX_SAMPLES="${MAX_SAMPLES:-10}"
EDITOR="${EDITOR:-opus}"

if [[ "$MAX_SAMPLES" == "all" || "$MAX_SAMPLES" == "0" || "$MAX_SAMPLES" == "-1" ]]; then
  EXPERIMENT="${EXPERIMENT:-llm_fusion_full}"
  SAMPLE_ARGS=()
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
