#!/usr/bin/env bash
# CropConcept Oromo smart fusion after Sagalee continued FT.
#
# ASR: Sagalee-FT Ethio-1B + Ethio-Oromo + Ethio-multilingual-1B (+ optional Sunbird)
# LLM editor: OpenRouter (default opus-5.5) with few-shot + glossary
#
# Usage (GPU box):
#   cd /local/ndizi/4.Scripts/ndizi_asr && git pull
#   EVAL_ROOT=/local/ndizi/4.Scripts/ndizi_eval \
#     ./scripts/run_cropconcept_fusion.sh
#
# Env overrides:
#   EDITOR=gemini-2.5-flash
#   WITH_SUNBIRD=1          # also fuse sunbird-whisper-51
#   FEW_SHOT=8
#   SKIP_GLOSSARY=1         # skip glossary rebuild / second pass
#   REUSE_HYPS=             # force re-run ASR (empty disables cache)
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
chmod +x "$0" 2>/dev/null || true

EVAL_ROOT="${EVAL_ROOT:-/local/ndizi/4.Scripts/ndizi_eval}"
CSV="${CSV:-${EVAL_ROOT}/data/manifest_cropconcept_oromo.csv}"
EDITOR="${EDITOR:-opus-5.5}"
FEW_SHOT="${FEW_SHOT:-8}"
SEED="${SEED:-42}"
SAGALEE_CKPT="${SAGALEE_CKPT:-outputs/ethio1b_sagalee_continued}"
CACHE_EXP="${CACHE_EXP:-cropconcept_fusion_asr_cache}"
FUSION_EXP="${FUSION_EXP:-cropconcept_fusion_sagalee_fs${FEW_SHOT}}"
GLOSSARY="${GLOSSARY:-${ROOT}/configs/glossary_cropconcept_oromo.txt}"

ASR_MODELS=(
  "ctc:${SAGALEE_CKPT}"
  ethio-asr-oromo
  ethio-asr-multilingual-1b
)
if [[ "${WITH_SUNBIRD:-0}" == "1" ]]; then
  ASR_MODELS+=(sunbird-whisper-51)
fi

if [[ ! -f "$CSV" ]]; then
  echo "CSV not found: $CSV" >&2
  echo "Set EVAL_ROOT / CSV to your ndizi_eval CropConcept manifest." >&2
  exit 2
fi
if [[ ! -d "$SAGALEE_CKPT" ]]; then
  echo "Checkpoint not found: $SAGALEE_CKPT" >&2
  echo "Train first or set SAGALEE_CKPT=..." >&2
  exit 2
fi

echo "=== CropConcept fusion ==="
echo "EVAL_ROOT=$EVAL_ROOT"
echo "CSV=$CSV"
echo "ASR: ${ASR_MODELS[*]}"
echo "Editor=$EDITOR  few-shot=$FEW_SHOT  seed=$SEED"
echo

# ---------------------------------------------------------------------------
# 1) ASR hyp cache (no LLM) — reuse later
# ---------------------------------------------------------------------------
CACHE_JSONL="${ROOT}/results/${CACHE_EXP}/fusion_predictions.jsonl"
if [[ ! -f "$CACHE_JSONL" ]]; then
  echo "→ Building ASR hyp cache → results/${CACHE_EXP}/"
  # Editor still required by CLI; use a cheap one but we only need hyps —
  # run with --few-shot 0 and accept one editor pass, OR reuse after.
  # Prefer: run ensemble once with few-shot for scored fusion + hyps.
  :
fi

# ---------------------------------------------------------------------------
# 2) Fusion pass A: few-shot (builds few_shot_examples for glossary)
# ---------------------------------------------------------------------------
echo "→ Fusion pass A (few-shot=${FEW_SHOT}, seed=${SEED})"
PASS_A_ARGS=(
  --env-file .env
  --asr-models "${ASR_MODELS[@]}"
  --editor "$EDITOR"
  --csv "$CSV"
  --data-root "$EVAL_ROOT"
  --languages om
  --normalize ethio
  --few-shot "$FEW_SHOT"
  --seed "$SEED"
  --few-shot-retrieve
  --output-dir results
  --experiment-name "${FUSION_EXP}_pass_a"
  --no-resume
)
if [[ -f "$CACHE_JSONL" ]]; then
  PASS_A_ARGS+=(--reuse-hyps "$CACHE_JSONL")
  echo "  reusing hyps: $CACHE_JSONL"
fi
if [[ -f "$GLOSSARY" ]]; then
  PASS_A_ARGS+=(--glossary "$GLOSSARY")
  echo "  seed glossary: $GLOSSARY"
fi

python -m ndizi_asr.ensemble "${PASS_A_ARGS[@]}" "$@"

PASS_A_DIR="${ROOT}/results/${FUSION_EXP}_pass_a"
# Persist ASR cache from this run if missing
if [[ ! -f "$CACHE_JSONL" && -f "${PASS_A_DIR}/fusion_predictions.jsonl" ]]; then
  mkdir -p "${ROOT}/results/${CACHE_EXP}"
  python -m ndizi_asr.save_hyp_cache \
    --from-file "${PASS_A_DIR}/fusion_predictions.jsonl" \
    -o "$CACHE_JSONL"
  echo "  wrote hyp cache → $CACHE_JSONL"
fi

# ---------------------------------------------------------------------------
# 3) Build / refresh glossary from few-shot examples
# ---------------------------------------------------------------------------
if [[ "${SKIP_GLOSSARY:-0}" != "1" ]]; then
  echo "→ Building glossary from few-shot examples"
  OUT_GLOSS="${PASS_A_DIR}/glossary_from_fewshot.txt"
  GLOSS_ARGS=(
    --few-shot-ids "${PASS_A_DIR}/few_shot_ids.json"
    --predictions "${PASS_A_DIR}/fusion_predictions.jsonl"
    -o "$OUT_GLOSS"
  )
  if [[ -f "${PASS_A_DIR}/few_shot_examples.jsonl" ]]; then
    GLOSS_ARGS+=(--examples "${PASS_A_DIR}/few_shot_examples.jsonl")
  fi
  python -m ndizi_asr.build_glossary "${GLOSS_ARGS[@]}" || {
    echo "WARNING: glossary build failed; continuing with seed glossary" >&2
    OUT_GLOSS="$GLOSSARY"
  }
  # Merge seed + auto glossary if both exist
  MERGED="${PASS_A_DIR}/glossary_merged.txt"
  {
    echo "# CropConcept Oromo — seed + few-shot glossary"
    echo
    if [[ -f "$GLOSSARY" ]]; then
      echo "## Seed"
      cat "$GLOSSARY"
      echo
    fi
    if [[ -f "$OUT_GLOSS" ]]; then
      echo "## From few-shot"
      cat "$OUT_GLOSS"
    fi
  } > "$MERGED"
  GLOSSARY_USE="$MERGED"
else
  GLOSSARY_USE="$GLOSSARY"
fi

# ---------------------------------------------------------------------------
# 4) Fusion pass B: same ASR hyps + refreshed glossary
# ---------------------------------------------------------------------------
echo "→ Fusion pass B (glossary + few-shot retrieve)"
PASS_B_ARGS=(
  --env-file .env
  --asr-models "${ASR_MODELS[@]}"
  --editor "$EDITOR"
  --csv "$CSV"
  --data-root "$EVAL_ROOT"
  --languages om
  --normalize ethio
  --few-shot "$FEW_SHOT"
  --seed "$SEED"
  --few-shot-retrieve
  --output-dir results
  --experiment-name "$FUSION_EXP"
  --no-resume
)
if [[ -f "$CACHE_JSONL" ]]; then
  PASS_B_ARGS+=(--reuse-hyps "$CACHE_JSONL")
elif [[ -f "${PASS_A_DIR}/fusion_predictions.jsonl" ]]; then
  PASS_B_ARGS+=(--reuse-hyps "${PASS_A_DIR}/fusion_predictions.jsonl")
fi
if [[ -n "${GLOSSARY_USE:-}" && -f "$GLOSSARY_USE" ]]; then
  PASS_B_ARGS+=(--glossary "$GLOSSARY_USE")
fi

python -m ndizi_asr.ensemble "${PASS_B_ARGS[@]}" "$@"

FINAL_DIR="${ROOT}/results/${FUSION_EXP}"
echo
echo "=== Done ==="
echo "Results: $FINAL_DIR"
echo "Error sheet:"
echo "  python -m ndizi_asr.error_sheet --predictions ${FINAL_DIR}/fusion_predictions.jsonl"
python -m ndizi_asr.error_sheet \
  --predictions "${FINAL_DIR}/fusion_predictions.jsonl" \
  --output-dir "${FINAL_DIR}" || true
echo
echo "Open ${FINAL_DIR}/summary* / error_sheet.md for WER vs single-model baselines (~0.79)."
