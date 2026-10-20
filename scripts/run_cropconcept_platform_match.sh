#!/usr/bin/env bash
# Re-transcribe CropConcept Oromo using the exact colleague recipe:
#   - google/gemini-2.5-pro via OpenRouter chat/completions
#   - Original .m4a/.aac file bytes sent as input_audio with "format": "mp4" (no re-encoding/resampling)
#   - Single user message: [input_audio, text] in that exact order
#   - temperature: 0.0, no max_tokens, no reasoning parameter
#   - Exact Oromo JSON prompt (63 clips) or English JSON prompt (9 clips via sidecar hint_given_to_transcriber)
#   - Extract JSON["text"].strip()
#
#   EVAL_ROOT=/Users/mutuvi/Documents/ndizi/Ethiopia_crop_concept/ndizi_eval ./scripts/run_cropconcept_platform_match.sh --concurrency 2
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXPERIMENT="${EXPERIMENT:-cropconcept_oromo_exact_recipe}" \
  "$ROOT/scripts/run_cropconcept_compare.sh" \
  --models our-transcript gemini-2.5-pro \
  --normalize oromo_fair \
  --cropconcept-recipe \
  --timeout-seconds 300 \
  "$@"
