#!/usr/bin/env python3
"""Compare baseline vs LoRA Ndizi evals: per-dataset metrics + best/worst hyps.

Example (on the GPU box):

  python scripts/analyze_ndizi_lora_results.py \\
    --baseline results/baseline_sunbird51_ndizi_test \\
    --lora results/sunbird51_ndizi_lora_test \\
    --top 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jiwer import wer as jiwer_wer


def _find_metrics(exp_dir: Path) -> Path:
    hits = sorted(exp_dir.glob("*/metrics.json"))
    if hits:
        return hits[0]
    summary = exp_dir / "summary.json"
    if summary.is_file():
        return summary
    raise FileNotFoundError(f"No metrics.json under {exp_dir}")


def _find_predictions(exp_dir: Path) -> Path:
    hits = sorted(exp_dir.glob("*/predictions.jsonl"))
    if not hits:
        raise FileNotFoundError(f"No predictions.jsonl under {exp_dir}")
    return hits[0]


def _load_metrics(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "overall" in raw:
        return raw
    if isinstance(raw, dict) and "models" in raw:
        # summary wrapper — take first model
        models = raw["models"]
        first = next(iter(models.values()))
        return first
    raise ValueError(f"Unrecognized metrics shape in {path}")


def _fmt_pct(x: Any) -> str:
    if x is None:
        return "   n/a"
    return f"{100.0 * float(x):6.2f}%"


def _score_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("error"):
                continue
            ref = str(r.get("reference_normalized") or r.get("reference") or "")
            hyp = str(r.get("prediction_normalized") or r.get("prediction") or "")
            if not ref.strip():
                continue
            try:
                w = float(jiwer_wer(ref, hyp))
            except Exception:
                w = 1.0
            r["_wer"] = w
            rows.append(r)
    return rows


def _print_examples(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n### {title}")
    for i, r in enumerate(rows, 1):
        ref = str(r.get("reference") or "")
        hyp = str(r.get("prediction") or "")
        print(
            f"\n{i}. WER={r['_wer']:.3f}  dataset={r.get('dataset_id')}  "
            f"id={r.get('sample_id')}"
        )
        print(f"   REF: {ref[:400]}{'…' if len(ref) > 400 else ''}")
        print(f"   HYP: {hyp[:400]}{'…' if len(hyp) > 400 else ''}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--lora", type=Path, required=True)
    p.add_argument("--top", type=int, default=5, help="Best/worst example count")
    args = p.parse_args()

    base_m = _load_metrics(_find_metrics(args.baseline))
    lora_m = _load_metrics(_find_metrics(args.lora))

    datasets = sorted(
        set(base_m.get("per_dataset") or {}) | set(lora_m.get("per_dataset") or {})
    )

    print("## Per-dataset comparison (normalized WER/CER preferred when present)\n")
    hdr = (
        f"{'dataset':28} {'n':>5}  "
        f"{'base WER':>9} {'lora WER':>9} {'ΔWER':>8}  "
        f"{'base CER':>9} {'lora CER':>9} {'ΔCER':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    def pick(block: dict, key: str) -> Any:
        # Prefer normalized metrics used in the comparison table (WER↓ column)
        return block.get(f"{key}_normalized", block.get(key))

    for ds in datasets + ["__overall__"]:
        if ds == "__overall__":
            b, l = base_m.get("overall") or {}, lora_m.get("overall") or {}
            label = "ALL"
        else:
            b = (base_m.get("per_dataset") or {}).get(ds) or {}
            l = (lora_m.get("per_dataset") or {}).get(ds) or {}
            label = ds
        bw, lw = pick(b, "wer"), pick(l, "wer")
        bc, lc = pick(b, "cer"), pick(l, "cer")
        n = l.get("num_scored") or b.get("num_scored") or l.get("n") or b.get("n") or ""
        dwer = (float(lw) - float(bw)) if bw is not None and lw is not None else None
        dcer = (float(lc) - float(bc)) if bc is not None and lc is not None else None
        print(
            f"{label:28} {str(n):>5}  "
            f"{_fmt_pct(bw):>9} {_fmt_pct(lw):>9} "
            f"{(_fmt_pct(dwer) if dwer is not None else '   n/a'):>8}  "
            f"{_fmt_pct(bc):>9} {_fmt_pct(lc):>9} "
            f"{(_fmt_pct(dcer) if dcer is not None else '   n/a'):>8}"
        )

    print(
        "\nNegative ΔWER = LoRA better. Overall numbers should match your "
        "0.4465 → 0.4146 WER comparison."
    )

    lora_rows = _score_rows(_find_predictions(args.lora))
    lora_rows.sort(key=lambda r: r["_wer"])
    k = max(1, int(args.top))
    _print_examples(f"LoRA best {k} (lowest WER)", lora_rows[:k])
    _print_examples(f"LoRA worst {k} (highest WER)", list(reversed(lora_rows[-k:])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
