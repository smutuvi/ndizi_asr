#!/usr/bin/env bash
# Transcribe sample_ndizi_CC_pilot with Gemini 2.5 Pro, Gemini 4 Argon,
# and deploy sunflower51-lora-qc-v2 (~14.60% Fair WER on ndizi-1-sw).
#
#   ./scripts/run_cc_pilot_transcribe.sh
#   CUSTOM_MODEL=hf:outputs/sunflower51_ndizi1sw_lora_qc_v2/merged ./scripts/run_cc_pilot_transcribe.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

PILOT_DIR="${PILOT_DIR:-${ROOT}/sample_ndizi_CC_pilot}"
CSV="${CSV:-${PILOT_DIR}/manifest.csv}"
CUSTOM_MODEL="${CUSTOM_MODEL:-sunflower51-lora-qc-v2}"
EXPERIMENT="${EXPERIMENT:-cc_pilot_gemini_pro_argon_sunflower_v2}"

if [[ ! -f "$CSV" ]]; then
  echo "Missing $CSV — regenerate with a short Python write of audio_path/reference/language rows." >&2
  exit 2
fi

echo "=== CC pilot transcribe ==="
echo "CSV=$CSV"
echo "Models: gemini-2.5-pro  gemini-4-argon  $CUSTOM_MODEL"
echo "Experiment=$EXPERIMENT"
echo

python -m ndizi_asr.cli \
  --env-file .env \
  --models \
    gemini-2.5-pro \
    gemini-4-argon \
    "$CUSTOM_MODEL" \
  --csv "$CSV" \
  --languages sw \
  --normalize default \
  --concurrency 1 \
  --output-dir results \
  --experiment-name "$EXPERIMENT" \
  "$@"

echo
echo "Done → results/${EXPERIMENT}/"
echo "Predictions per model under results/${EXPERIMENT}/*/predictions.jsonl"
