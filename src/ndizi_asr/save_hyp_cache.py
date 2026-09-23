"""Merge fusion/few-shot jsonl rows into a reusable ASR hyp cache.

Use after a 3-model run to save Ethio+Sunbird hyps for later --reuse-hyps
without re-running ASR or the LLM editor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _merge_row(dst: dict[str, Any], src: dict[str, Any]) -> None:
    if not dst.get("reference") and src.get("reference"):
        dst["reference"] = src["reference"]
    for key in ("language", "language_config", "dataset_id"):
        if not dst.get(key) and src.get(key):
            dst[key] = src[key]
    hyps = dict(dst.get("hypotheses") or {})
    for k, v in (src.get("hypotheses") or {}).items():
        if k not in hyps:
            hyps[k] = v
    dst["hypotheses"] = hyps
    errs = dict(dst.get("asr_errors") or {})
    for k, v in (src.get("asr_errors") or {}).items():
        if k not in errs:
            errs[k] = v
    dst["asr_errors"] = errs


def build_cache(sources: list[Path]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for path in sources:
        for row in _load_jsonl(path):
            sid = str(row.get("sample_id") or "")
            if not sid:
                continue
            if sid not in by_id:
                by_id[sid] = {
                    "sample_id": sid,
                    "dataset_id": row.get("dataset_id"),
                    "language": row.get("language"),
                    "language_config": row.get("language_config"),
                    "reference": row.get("reference"),
                    "hypotheses": {},
                    "asr_errors": {},
                }
            _merge_row(by_id[sid], row)
    return [by_id[k] for k in sorted(by_id)]


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Save a reusable ASR hyp cache from fusion / few-shot jsonl files."
    )
    p.add_argument(
        "--from-dir",
        type=Path,
        action="append",
        default=[],
        help="Experiment dir (reads fusion_predictions.jsonl + few_shot_examples.jsonl). "
        "Repeatable.",
    )
    p.add_argument(
        "--from-file",
        type=Path,
        action="append",
        default=[],
        help="Extra jsonl with hypotheses. Repeatable.",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("results/llm_fusion_full3/fusion_predictions.jsonl"),
        help="Output hyp cache path",
    )
    p.add_argument(
        "--require-models",
        nargs="*",
        default=[
            "ethio-asr-multilingual-1b",
            "ethio-asr-oromo",
            "sunbird-whisper-51",
        ],
        help="Warn if these ASR keys are missing on any row",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    sources: list[Path] = list(args.from_file)
    for d in args.from_dir:
        sources.append(d / "fusion_predictions.jsonl")
        sources.append(d / "few_shot_examples.jsonl")
        # legacy few-shot filename
        sources.append(d / "fusion_predictions_fs8.jsonl")

    sources = [s for s in sources if s.exists()]
    if not sources:
        raise SystemExit("No source jsonl files found. Pass --from-dir and/or --from-file.")

    rows = build_cache(sources)
    if not rows:
        raise SystemExit("No rows merged.")

    missing_counts = {m: 0 for m in (args.require_models or [])}
    for r in rows:
        hyps = r.get("hypotheses") or {}
        for m in missing_counts:
            if m not in hyps:
                missing_counts[m] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows → {args.output}")
    print(f"Sources: {', '.join(str(s) for s in sources)}")
    model_keys: set[str] = set()
    for r in rows:
        model_keys.update((r.get("hypotheses") or {}).keys())
    print(f"ASR keys present: {sorted(model_keys)}")
    for m, n in missing_counts.items():
        if n:
            print(f"WARNING: {m} missing on {n}/{len(rows)} rows", file=sys.stderr)
    meta = {
        "num_rows": len(rows),
        "asr_keys": sorted(model_keys),
        "sources": [str(s) for s in sources],
        "missing": missing_counts,
    }
    meta_path = args.output.with_name("hyp_cache_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
