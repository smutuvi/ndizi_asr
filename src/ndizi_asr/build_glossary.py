"""Build a fusion domain glossary from few-shot samples only (no eval pollution).

Preferred inputs (from a prior --few-shot run):
  results/<exp>/few_shot_examples.jsonl
  results/<exp>/few_shot_ids.json

Falls back to pulling shot IDs from few_shot_ids.json and looking up
reference/hypotheses in a full fusion_predictions.jsonl (e.g. llm_fusion_full).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_shot_ids(path: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    shot = [str(x) for x in meta.get("shot_ids") or []]
    eval_ids = [str(x) for x in meta.get("eval_ids") or []]
    return shot, eval_ids, meta


def _rows_for_shots(
    *,
    shot_ids: list[str],
    examples_path: Optional[Path],
    predictions_path: Optional[Path],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    if examples_path and examples_path.exists():
        for row in _load_jsonl(examples_path):
            by_id[str(row.get("sample_id"))] = row
    if predictions_path and predictions_path.exists():
        for row in _load_jsonl(predictions_path):
            sid = str(row.get("sample_id"))
            if sid in shot_ids and sid not in by_id:
                by_id[sid] = row
            elif sid in shot_ids:
                # Fill missing fields from predictions if examples row is thin
                cur = by_id[sid]
                if not cur.get("reference") and row.get("reference"):
                    cur["reference"] = row["reference"]
                if not cur.get("hypotheses") and row.get("hypotheses"):
                    cur["hypotheses"] = row["hypotheses"]

    missing = [sid for sid in shot_ids if sid not in by_id]
    if missing:
        raise SystemExit(
            "Missing few-shot rows for sample_id(s): "
            + ", ".join(missing[:8])
            + (f" (+{len(missing) - 8} more)" if len(missing) > 8 else "")
            + "\nProvide --examples (few_shot_examples.jsonl) or --predictions "
            "that contains those IDs (e.g. results/llm_fusion_full/fusion_predictions.jsonl)."
        )
    return [by_id[sid] for sid in shot_ids]


def build_glossary_text(
    rows: list[dict[str, Any]],
    *,
    meta: Optional[dict[str, Any]] = None,
    min_len: int = 3,
    max_terms: int = 80,
) -> str:
    """Extract preferred spellings from few-shot gold (+ ASR disagreement cues)."""
    gold_counts: Counter[str] = Counter()
    preferred_over_asr: Counter[str] = Counter()
    for row in rows:
        gold = str(row.get("reference") or "")
        gold_toks = tokenize(gold)
        gold_counts.update(t for t in gold_toks if len(t) >= min_len)
        hyp_toks: set[str] = set()
        for text in (row.get("hypotheses") or {}).values():
            hyp_toks.update(tokenize(str(text or "")))
        for t in gold_toks:
            if len(t) >= min_len and t not in hyp_toks:
                preferred_over_asr[t] += 1

    # Rank: disagreement-with-ASR first, then frequent gold terms
    ranked: list[str] = []
    seen: set[str] = set()
    for term, _ in preferred_over_asr.most_common():
        if term not in seen:
            ranked.append(term)
            seen.add(term)
    for term, _ in gold_counts.most_common():
        if term not in seen:
            ranked.append(term)
            seen.add(term)
    ranked = ranked[:max_terms]

    seed = (meta or {}).get("seed")
    n_shot = (meta or {}).get("few_shot") or len(rows)
    shot_ids = (meta or {}).get("shot_ids") or [r.get("sample_id") for r in rows]

    lines = [
        "# CropConcept / Afaan Oromo ASR fusion glossary",
        "# Built ONLY from few-shot gold (held-out eval IDs excluded).",
        f"# few_shot={n_shot} seed={seed} n_terms={len(ranked)}",
        f"# shot_ids={', '.join(str(x) for x in shot_ids)}",
        "#",
        "# Use with: ndizi-asr-fuse ... --glossary <this-file>",
        "# Edit freely: drop stopwords, add external agronomy spellings.",
        "",
        "Preferred spellings / domain terms:",
    ]
    for t in ranked:
        tag = "  # gold≠ASR" if t in preferred_over_asr else ""
        lines.append(f"- {t}{tag}")

    if preferred_over_asr:
        lines.append("")
        lines.append("# Terms attested in few-shot gold but missing from both CTC hyps:")
        for t, c in preferred_over_asr.most_common(40):
            lines.append(f"#   {t} (n={c})")

    lines.append("")
    return "\n".join(lines)


def resolve_inputs(
    *,
    run_dir: Optional[Path],
    few_shot_ids: Optional[Path],
    examples: Optional[Path],
    predictions: Optional[Path],
) -> tuple[list[str], dict[str, Any], list[dict[str, Any]]]:
    ids_path = few_shot_ids
    examples_path = examples
    if run_dir:
        if ids_path is None:
            cand = run_dir / "few_shot_ids.json"
            if cand.exists():
                ids_path = cand
        if examples_path is None:
            cand = run_dir / "few_shot_examples.jsonl"
            if cand.exists():
                examples_path = cand
        if predictions is None:
            cand = run_dir / "fusion_predictions.jsonl"
            if cand.exists():
                predictions = cand

    if ids_path is None or not ids_path.exists():
        raise SystemExit(
            "Need few_shot_ids.json (--few-shot-ids or --run-dir from a --few-shot run)."
        )

    shot_ids, eval_ids, meta = _load_shot_ids(ids_path)
    if not shot_ids:
        raise SystemExit(f"No shot_ids in {ids_path}")

    # Safety: never silently include eval rows when building from a predictions dump
    rows = _rows_for_shots(
        shot_ids=shot_ids,
        examples_path=examples_path,
        predictions_path=predictions,
    )
    leaked = [str(r.get("sample_id")) for r in rows if str(r.get("sample_id")) in set(eval_ids)]
    if leaked:
        raise SystemExit(f"Internal error: eval IDs leaked into glossary rows: {leaked}")

    meta = dict(meta)
    meta.setdefault("shot_ids", shot_ids)
    return shot_ids, meta, rows


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Build a domain glossary from few-shot samples only "
            "(avoids eval-set pollution)."
        )
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Fusion experiment dir containing few_shot_ids.json "
        "(and ideally few_shot_examples.jsonl)",
    )
    p.add_argument("--few-shot-ids", type=Path, default=None, help="Path to few_shot_ids.json")
    p.add_argument(
        "--examples",
        type=Path,
        default=None,
        help="few_shot_examples.jsonl (written by fusion when --few-shot > 0)",
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional fusion_predictions.jsonl to look up shot refs/hyps "
        "(use llm_fusion_full if examples file is missing)",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output glossary text path",
    )
    p.add_argument("--min-len", type=int, default=3, help="Min token length (default 3)")
    p.add_argument("--max-terms", type=int, default=80, help="Max glossary terms (default 80)")
    args = p.parse_args(list(argv) if argv is not None else None)

    _, meta, rows = resolve_inputs(
        run_dir=args.run_dir,
        few_shot_ids=args.few_shot_ids,
        examples=args.examples,
        predictions=args.predictions,
    )
    text = build_glossary_text(
        rows, meta=meta, min_len=args.min_len, max_terms=args.max_terms
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote glossary ({len(rows)} few-shot rows) → {args.output}")
    print(f"  seed={meta.get('seed')} few_shot={meta.get('few_shot')}")
    print("  Eval IDs were excluded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
