#!/usr/bin/env bash
# Re-transcribe CropConcept Oromo with gemini-2.5-flash using platform-style
# settings and score it next to our-transcript in the same experiment.
#
# Platform-style settings (vs. the default compare run):
#   - whole clip in one request (no 30 s VAD chunking; 53/61 clips are > 30 s)
#   - max_tokens 4096 (longest our_transcript is ~350 words)
#   - plain-transcript prompt (no labels / diarization)
#   - oromo_fair normalization, same 61 counted clips and strata as our-transcript
#
#   ./scripts/run_cropconcept_platform_match.sh
#   EVAL_ROOT=/local/ndizi/4.Scripts/ndizi_eval ./scripts/run_cropconcept_platform_match.sh --concurrency 4
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PLATFORM_PROMPT="${PLATFORM_PROMPT:-Generate a verbatim transcript of the speech in this audio. Do not include speaker labels, timestamps, translations, or any additional text or comments. Just the transcript.}"

EXPERIMENT="${EXPERIMENT:-cropconcept_oromo_platform_match}" \
  "$ROOT/scripts/run_cropconcept_compare.sh" \
  --models our-transcript gemini-2.5-flash \
  --normalize oromo_fair \
  --max-chunk-seconds 0 \
  --max-tokens 4096 \
  --temperature 0 \
  --timeout-seconds 300 \
  --prompt "$PLATFORM_PROMPT" \
  "$@"
