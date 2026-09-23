"""Post-filter fusion text to tokens attested in ASR hypotheses (no LLM calls).

Use on an existing fusion_predictions.jsonl to measure how much invented
vocabulary is hurting WER before spending more editor tokens.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from .config import NormalizationConfig
from .metrics import normalize_text, summarize_predictions

_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def constrain_to_hyp_vocab(
    fusion: str,
    hyps: dict[str, str],
    *,
    max_edit: int = 1,
) -> str:
    """Keep fusion tokens attested in hyps; map near-misses (edit distance ≤ max_edit)."""
    allowed: dict[str, str] = {}
    for text in hyps.values():
        for tok in tokenize(str(text or "")):
            allowed.setdefault(tok, tok)
    if not allowed:
        return ""

    def _closest(tok: str) -> Optional[str]:
        if tok in allowed:
            return allowed[tok]
        if max_edit <= 0:
            return None
        best: Optional[str] = None
        best_d = max_edit + 1
        for a in allowed:
            if abs(len(a) - len(tok)) > max_edit:
                continue
            d = _edit_distance(tok, a, max_edit)
            if d is not None and d < best_d:
                best_d = d
                best = a
                if d == 0:
                    break
        return best if best_d <= max_edit else None

    kept: list[str] = []
    for tok in tokenize(fusion):
        hit = _closest(tok)
        if hit:
            kept.append(hit)
    return " ".join(kept)


def _edit_distance(a: str, b: str, max_d: int) -> Optional[int]:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_d:
        return None
    # Banded Levenshtein; early-exit when > max_d
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        row_min = i
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur[j] = min(ins, delete, sub)
            row_min = min(row_min, cur[j])
        if row_min > max_d:
            return None
        prev = cur
    return prev[-1] if prev[-1] <= max_d else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def apply_constraint(
    rows: list[dict[str, Any]],
    *,
    max_edit: int = 1,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        hyps = dict(row.get("hypotheses") or {})
        raw = str(row.get("fusion") or "")
        constrained = constrain_to_hyp_vocab(raw, hyps, max_edit=max_edit)
        new_row = dict(row)
        new_row["fusion_raw"] = raw
        new_row["fusion"] = constrained
        new_row["constrain_vocab"] = True
        new_row["constrain_max_edit"] = max_edit
        out.append(new_row)
    return out


def score_rows(rows: list[dict[str, Any]], *, label: str) -> dict[str, Any]:
    pred_rows = [
        {
            "reference": r.get("reference"),
            "prediction": r.get("fusion"),
            "error": r.get("fusion_error"),
            "language": r.get("language") or r.get("language_config"),
            "latency_seconds": r.get("fusion_latency_seconds"),
        }
        for r in rows
    ]
    summary = summarize_predictions(pred_rows, NormalizationConfig(mode="ethio"))
    return {
        "label": label,
        "wer_n": summary.get("wer_normalized"),
        "cer_n": summary.get("cer_normalized"),
        "wer": summary.get("wer"),
        "cer": summary.get("cer"),
        "n": summary.get("num_scored"),
        "errors": summary.get("num_errors"),
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Drop/map fusion tokens not attested in ASR hyps (offline, no LLM)."
    )
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write constrained jsonl (default: <pred_dir>/fusion_predictions_constrained.jsonl)",
    )
    p.add_argument(
        "--max-edit",
        type=int,
        default=1,
        help="Allow mapping fusion token → hyp token if edit distance ≤ N (default 1)",
    )
    p.add_argument(
        "--also-best-hyp",
        action="store_true",
        help="Also score oracle-free baselines: each ASR hyp alone",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    rows = _load_jsonl(args.predictions)
    if not rows:
        raise SystemExit(f"No rows in {args.predictions}")

    constrained = apply_constraint(rows, max_edit=args.max_edit)
    out_path = args.output or (
        args.predictions.parent / "fusion_predictions_constrained.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in constrained:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    baselines = [score_rows(rows, label="fusion_raw")]
    baselines.append(score_rows(constrained, label=f"fusion_constrained_ed≤{args.max_edit}"))

    if args.also_best_hyp:
        # Score each ASR system from hypotheses
        keys: list[str] = []
        for r in rows:
            for k in (r.get("hypotheses") or {}):
                if k not in keys:
                    keys.append(k)
        for k in keys:
            hyp_rows = []
            for r in rows:
                hyp_rows.append(
                    {
                        **r,
                        "fusion": str((r.get("hypotheses") or {}).get(k) or ""),
                        "fusion_error": (r.get("asr_errors") or {}).get(k),
                    }
                )
            baselines.append(score_rows(hyp_rows, label=f"hyp:{k}"))

    print(f"n={len(rows)}  wrote {out_path}")
    print(f"{'System':<40} | {'WER↓':>8} | {'CER':>8} | n")
    print("-" * 70)
    for b in sorted(baselines, key=lambda x: (x["wer_n"] is None, float(x["wer_n"] or 99))):
        wer = "—" if b["wer_n"] is None else f"{b['wer_n']:.4f}"
        cer = "—" if b["cer_n"] is None else f"{b['cer_n']:.4f}"
        print(f"{b['label']:<40} | {wer:>8} | {cer:>8} | {b['n']}")

    # Also write a tiny summary csv next to output
    summary_path = out_path.with_name("constrain_summary.csv")
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "wer_normalized", "cer_normalized", "n"])
        w.writeheader()
        for b in baselines:
            w.writerow(
                {
                    "system": b["label"],
                    "wer_normalized": b["wer_n"],
                    "cer_normalized": b["cer_n"],
                    "n": b["n"],
                }
            )
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
