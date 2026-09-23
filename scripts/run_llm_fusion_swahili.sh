#!/usr/bin/env bash
# Smart fusion on Ndizi Swahili:
#   ASR: smutuvi/ndizi-whisper-large-v3-turbo-optimized
#        badrex/w2v-bert-2.0-swahili-asr
#   Editor: Gemini 2.5 Flash (OpenRouter)  [alias: gemini / gemma-2.5]
#   Data: 50 clips from smutuvi/ndizi-1 + 50 from smutuvi/ndizi-1-2025
#
# First run (build hyps + fuse; ~100 utts):
#   ./scripts/run_llm_fusion_swahili.sh
#
# Few-shot 8 (score ~92 held-out):
#   ./scripts/run_llm_fusion_swahili.sh --few-shot 8 --seed 42 --no-resume \
#     --experiment-name llm_fusion_swahili_fs8
#
# Reuse hyps after first full run:
#   REUSE_HYPS=results/llm_fusion_swahili_full/fusion_predictions.jsonl \
#     ./scripts/run_llm_fusion_swahili.sh --few-shot 8 --seed 42 --no-resume \
#     --experiment-name llm_fusion_swahili_fs8
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

EDITOR="${EDITOR:-gemini}"
EXPERIMENT="${EXPERIMENT:-llm_fusion_swahili_full}"
CONFIG="${CONFIG:-${ROOT}/configs/swahili_fusion.yaml}"
ASR_MODELS="${ASR_MODELS:-ndizi-whisper-turbo w2v-bert-swahili}"

REUSE_ARGS=()
if [[ -n "${REUSE_HYPS:-}" ]]; then
  if [[ -f "$REUSE_HYPS" ]]; then
    REUSE_ARGS=(--reuse-hyps "$REUSE_HYPS")
    echo "Reusing ASR hyps: $REUSE_HYPS"
  else
    echo "WARNING: REUSE_HYPS not found ($REUSE_HYPS)" >&2
  fi
elif [[ -f "${ROOT}/results/llm_fusion_swahili_full/fusion_predictions.jsonl" \
     && "$EXPERIMENT" != "llm_fusion_swahili_full" ]]; then
  REUSE_ARGS=(--reuse-hyps "${ROOT}/results/llm_fusion_swahili_full/fusion_predictions.jsonl")
  echo "Reusing ASR hyps: ${REUSE_ARGS[1]}"
fi

# shellcheck disable=SC2086
python -m ndizi_asr.ensemble \
  --env-file .env \
  --config "$CONFIG" \
  --asr-models $ASR_MODELS \
  --editor "$EDITOR" \
  --normalize default \
  "${REUSE_ARGS[@]}" \
  --output-dir results \
  --experiment-name "$EXPERIMENT" \
  "$@"
