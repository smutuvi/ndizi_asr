#!/usr/bin/env bash
# Ethiopia CropConcept / Ethio Hub eval helpers (merged from ndizi_eval).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MODE="${1:-hub}"
shift || true

case "$MODE" in
  hub)
    python -m ndizi_asr.cli \
      --config configs/ethio_hub.yaml \
      --env-file .env \
      --max-samples "${MAX_SAMPLES:-20}" \
      --experiment-name "ethio_hub_n${MAX_SAMPLES:-20}" \
      "$@"
    ;;
  cropconcept)
    python -m ndizi_asr.cli \
      --config configs/cropconcept_oromo.yaml \
      --env-file .env \
      "$@"
    ;;
  list)
    python -m ndizi_asr.cli --list-models
    ;;
  *)
    echo "Usage: $0 {hub|cropconcept|list} [extra ndizi-asr flags...]" >&2
    exit 2
    ;;
esac
