#!/usr/bin/env bash
# Permanently prefer system ffmpeg inside a conda env (fixes omniasr libx264 crashes).
#
# Usage:
#   ./scripts/setup_ffmpeg_env.sh              # current $CONDA_PREFIX
#   ./scripts/setup_ffmpeg_env.sh omniasr      # named env
#   ./scripts/setup_ffmpeg_env.sh ndizi
set -euo pipefail

ENV_NAME="${1:-}"
if [[ -n "$ENV_NAME" ]]; then
  PREFIX="$(conda info --base)/envs/${ENV_NAME}"
else
  PREFIX="${CONDA_PREFIX:-}"
fi

if [[ -z "$PREFIX" || ! -d "$PREFIX" ]]; then
  echo "No conda env found. Activate one or pass a name: $0 omniasr" >&2
  exit 2
fi

ACTIVATE_DIR="${PREFIX}/etc/conda/activate.d"
DEACTIVATE_DIR="${PREFIX}/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

cat > "${ACTIVATE_DIR}/ffmpeg_system.sh" <<'EOF'
# Prefer system ffmpeg over broken conda builds (libx264 / LD_LIBRARY_PATH).
export FFMPEG_BINARY="${FFMPEG_BINARY:-/usr/bin/ffmpeg}"
# Remember prior PATH ffmpeg shadowing is fine; ndizi_asr uses FFMPEG_BINARY.
EOF

cat > "${DEACTIVATE_DIR}/ffmpeg_system.sh" <<'EOF'
unset FFMPEG_BINARY
EOF

echo "Installed conda activate hook:"
echo "  ${ACTIVATE_DIR}/ffmpeg_system.sh"
echo
echo "Re-activate the env to apply:"
echo "  conda deactivate && conda activate $(basename "$PREFIX")"
echo "  echo \$FFMPEG_BINARY   # → /usr/bin/ffmpeg"
