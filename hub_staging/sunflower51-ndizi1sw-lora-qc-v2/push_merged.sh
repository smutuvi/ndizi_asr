#!/usr/bin/env bash
# Run on the machine that has the merged checkpoint (GPU box).
# Loads HF_TOKEN from ndizi_asr/.env and uploads merged weights + model card.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
set -a
# shellcheck disable=SC1091
source "${ROOT}/.env"
set +a
: "${HF_TOKEN:?HF_TOKEN missing in .env}"

REPO="${HF_REPO:-smutuvi/sunflower51-ndizi1sw-lora-qc-v2}"
MERGED="${MERGED_DIR:-${ROOT}/outputs/sunflower51_ndizi1sw_lora_qc_v2/merged}"
CARD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "$MERGED" ]]; then
  echo "Missing merged dir: $MERGED" >&2
  exit 1
fi

# Keep model card next to weights for a single upload-folder call
cp -f "${CARD_DIR}/README.md" "${MERGED}/README.md"

huggingface-cli upload "$REPO" "$MERGED" . --repo-type model --token "$HF_TOKEN"
echo "Uploaded -> https://huggingface.co/$REPO"
