#!/usr/bin/env python3
"""Merge per-model predictions.jsonl into one JSON + CSV (no WER/CER).

Usage:
  python scripts/combine_pilot_predictions.py \\
    --exp-dir results/cc_pilot_gemini_pro_argon_sunflower_v2

  # or after copying to home:
  python scripts/combine_pilot_predictions.py \\
    --exp-dir ~/cc_pilot_gemini_pro_argon_sunflower_v2 \\
    --out-dir ~/cc_pilot_gemini_pro_argon_sunflower_v2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _slug_from_dir(model_dir: Path) -> str:
    # predictions live in <exp>/<model_slug>/predictions.jsonl
    return model_dir.name


def combine(exp_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    by_id: dict[str, dict[str, Any]] = {}
    model_slugs: list[str] = []

    for pred_path in sorted(exp_dir.glob("*/predictions.jsonl")):
        slug = _slug_from_dir(pred_path.parent)
        model_slugs.append(slug)
        for row in _load_jsonl(pred_path):
            sid = str(row.get("sample_id") or "")
            if not sid:
                continue
            file_name = str(row.get("file_name") or "")
            if not file_name and ":" in sid:
                file_name = sid.split(":")[1]
            rec = by_id.setdefault(
                sid,
                {
                    "sample_id": sid,
                    "file_name": file_name or sid,
                    "language": row.get("language"),
                    "language_config": row.get("language_config"),
                    "audio_path": row.get("audio_path") or row.get("path"),
                    "predictions": {},
                    "errors": {},
                    "latencies": {},
                },
            )
            text = row.get("prediction")
            if text is None:
                text = row.get("hypothesis") or row.get("text") or ""
            rec["predictions"][slug] = text
            rec[f"prediction_{slug}"] = text
            err = row.get("error")
            if err:
                rec["errors"][slug] = err
                rec[f"error_{slug}"] = err
            lat = row.get("latency_seconds")
            if lat is not None:
                rec["latencies"][slug] = lat
                rec[f"latency_{slug}"] = lat

    return [by_id[k] for k in sorted(by_id)], model_slugs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--exp-dir",
        type=Path,
        default=Path("results/cc_pilot_gemini_pro_argon_sunflower_v2"),
        help="Experiment dir containing <model>/predictions.jsonl",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as --exp-dir)",
    )
    p.add_argument(
        "--out-stem",
        default="combined_predictions",
        help="Output basename without extension (default: combined_predictions)",
    )
    args = p.parse_args()

    exp_dir = args.exp_dir.expanduser().resolve()
    out_dir = (args.out_dir or args.exp_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not exp_dir.is_dir():
        raise SystemExit(f"Experiment dir not found: {exp_dir}")

    pred_files = list(exp_dir.glob("*/predictions.jsonl"))
    if not pred_files:
        raise SystemExit(f"No */predictions.jsonl under {exp_dir}")

    rows, model_slugs = combine(exp_dir)
    payload = {
        "experiment_dir": str(exp_dir),
        "models": model_slugs,
        "n_samples": len(rows),
        "note": "Transcription-only combine; no WER/CER (no references).",
        "samples": rows,
    }

    json_path = out_dir / f"{args.out_stem}.json"
    csv_path = out_dir / f"{args.out_stem}.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Flat CSV: one row per clip, one column per model
    fieldnames = [
        "sample_id",
        "file_name",
        "language",
        "language_config",
        "audio_path",
        *[f"prediction_{s}" for s in model_slugs],
        *[f"error_{s}" for s in model_slugs],
        *[f"latency_{s}" for s in model_slugs],
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            flat = {k: row.get(k, "") for k in fieldnames}
            w.writerow(flat)

    print(f"models ({len(model_slugs)}): {', '.join(model_slugs)}")
    print(f"samples: {len(rows)}")
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
