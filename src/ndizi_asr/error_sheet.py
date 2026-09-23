"""Error sheet for smart fusion: gold vs fusion vs ASR hyps (eval rows only).

Use this to find top failure modes before editing FUSION_SYSTEM / FUSION_USER_TEMPLATE.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Optional

from .config import NormalizationConfig
from .metrics import compute_wer, normalize_text

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


def resolve_predictions_path(path: Path) -> Path:
    """Accept fusion_predictions.jsonl or legacy fusion_predictions_fs8.jsonl."""
    if path.exists():
        return path
    if path.name == "fusion_predictions.jsonl" and path.parent.is_dir():
        legacy = sorted(path.parent.glob("fusion_predictions_*.jsonl"))
        if legacy:
            return legacy[0]
    raise FileNotFoundError(
        f"No predictions at {path}"
        + (
            f" (also checked {path.parent}/fusion_predictions_*.jsonl)"
            if path.parent.is_dir()
            else ""
        )
    )


def _norm(text: str, language: str = "om") -> str:
    return normalize_text(text, NormalizationConfig(mode="ethio"), language=language)


@dataclass
class UtteranceErrors:
    sample_id: str
    wer: float
    gold: str
    fusion: str
    hyps: dict[str, str]
    tags: list[str]
    invented: list[str]
    dropped: list[str]
    notes: str


def _tag_utterance(
    *,
    sample_id: str,
    gold: str,
    fusion: str,
    hyps: dict[str, str],
    language: str,
) -> UtteranceErrors:
    g = _norm(gold, language)
    f = _norm(fusion, language)
    hyp_norms = {k: _norm(v, language) for k, v in hyps.items()}
    wer = float(compute_wer([g], [f])) if g or f else 0.0

    g_toks = tokenize(g)
    f_toks = tokenize(f)
    hyp_tok_union: set[str] = set()
    for t in hyp_norms.values():
        hyp_tok_union.update(tokenize(t))

    g_set, f_set = set(g_toks), set(f_toks)
    invented = sorted(t for t in f_set - g_set - hyp_tok_union if len(t) >= 2)
    dropped = sorted(t for t in g_set - f_set if len(t) >= 2)

    # Numbers / digits dropped or invented
    g_nums = {t for t in g_toks if t.isdigit()}
    f_nums = {t for t in f_toks if t.isdigit()}
    num_drop = sorted(g_nums - f_nums)
    num_add = sorted(f_nums - g_nums)

    tags: list[str] = []
    if invented:
        tags.append("invented")
    if dropped:
        tags.append("dropped")
    if num_drop or num_add:
        tags.append("numbers")
    # Over-correction: fusion worse than best hyp on this utterance
    hyp_wers = {k: float(compute_wer([g], [v])) for k, v in hyp_norms.items()}
    best_hyp_wer = min(hyp_wers.values()) if hyp_wers else 1.0
    if wer > best_hyp_wer + 1e-9:
        tags.append("over_correction")
    # Fusion copied neither hyp closely and missed gold tokens
    if f_toks and not (f_set & hyp_tok_union) and dropped:
        tags.append("ignored_hyps")
    if not tags and wer > 0:
        tags.append("other")
    if wer == 0:
        tags = ["ok"]

    # Compact alignment hint
    sm = SequenceMatcher(a=g_toks, b=f_toks)
    ops = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        ops.append(f"{tag}:{' '.join(g_toks[i1:i2])!r}→{' '.join(f_toks[j1:j2])!r}")
    notes = "; ".join(ops[:6])
    if num_drop:
        notes = (notes + "; " if notes else "") + f"lost_nums={num_drop}"
    if num_add:
        notes = (notes + "; " if notes else "") + f"extra_nums={num_add}"

    return UtteranceErrors(
        sample_id=sample_id,
        wer=wer,
        gold=gold,
        fusion=fusion,
        hyps=hyps,
        tags=tags,
        invented=invented,
        dropped=dropped,
        notes=notes,
    )


def analyze(
    predictions: Path,
    *,
    asr_models: Optional[list[str]] = None,
) -> list[UtteranceErrors]:
    predictions = resolve_predictions_path(Path(predictions))
    rows = _load_jsonl(predictions)
    out: list[UtteranceErrors] = []
    for row in rows:
        # Eval predictions only (few-shot rows are not written here)
        hyps = dict(row.get("hypotheses") or {})
        if asr_models:
            hyps = {m: str(hyps.get(m) or "") for m in asr_models}
        else:
            hyps = {k: str(v or "") for k, v in hyps.items()}
        out.append(
            _tag_utterance(
                sample_id=str(row.get("sample_id")),
                gold=str(row.get("reference") or ""),
                fusion=str(row.get("fusion") or ""),
                hyps=hyps,
                language=str(row.get("language") or row.get("language_config") or "om"),
            )
        )
    out.sort(key=lambda e: (-e.wer, e.sample_id))
    return out


def write_tsv(path: Path, errors: list[UtteranceErrors]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "sample_id",
        "wer",
        "tags",
        "invented",
        "dropped",
        "gold",
        "fusion",
        "hypotheses",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for i, e in enumerate(errors, start=1):
            hyp_s = " || ".join(f"[{k}] {v}" for k, v in e.hyps.items())
            w.writerow(
                {
                    "rank": i,
                    "sample_id": e.sample_id,
                    "wer": f"{e.wer:.4f}",
                    "tags": ",".join(e.tags),
                    "invented": " ".join(e.invented),
                    "dropped": " ".join(e.dropped),
                    "gold": e.gold.replace("\t", " ").replace("\n", " "),
                    "fusion": e.fusion.replace("\t", " ").replace("\n", " "),
                    "hypotheses": hyp_s.replace("\t", " ").replace("\n", " "),
                    "notes": e.notes.replace("\t", " "),
                }
            )


def write_markdown(path: Path, errors: list[UtteranceErrors], *, top_n: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tag_counts: dict[str, int] = {}
    for e in errors:
        for t in e.tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    lines = [
        "# Fusion error sheet",
        "",
        f"Scored utterances: **{len(errors)}** (sorted worst WER first).",
        "",
        "## Tag counts",
        "",
    ]
    for t, c in sorted(tag_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- `{t}`: {c}")
    lines += ["", f"## Top {min(top_n, len(errors))} by WER", ""]
    for i, e in enumerate(errors[:top_n], start=1):
        lines.append(f"### {i}. `{e.sample_id}` — WER {e.wer:.3f} — tags: {', '.join(e.tags)}")
        lines.append("")
        lines.append(f"- **gold**: {e.gold}")
        lines.append(f"- **fusion**: {e.fusion}")
        for k, v in e.hyps.items():
            lines.append(f"- **{k}**: {v}")
        if e.invented:
            lines.append(f"- **invented** (not in gold∪hyps): {', '.join(e.invented)}")
        if e.dropped:
            lines.append(f"- **dropped** (in gold, missing in fusion): {', '.join(e.dropped)}")
        if e.notes:
            lines.append(f"- **align**: {e.notes}")
        lines.append("")
    lines += [
        "## How to use",
        "",
        "1. Read the top failures and tag counts.",
        "2. Edit `FUSION_SYSTEM` / `FUSION_USER_TEMPLATE` in `src/ndizi_asr/ensemble.py`",
        "   (e.g. stronger 'do not invent words not in hyps', preserve numbers).",
        "3. Re-run Opus fs8 with `--reuse-hyps` and **no** `--glossary`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(errors: list[UtteranceErrors], *, top_n: int = 8) -> None:
    tag_counts: dict[str, int] = {}
    for e in errors:
        for t in e.tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    mean_wer = sum(e.wer for e in errors) / len(errors) if errors else 0.0
    print(f"n={len(errors)}  mean_utt_wer={mean_wer:.4f}")
    print("tags:", ", ".join(f"{t}={c}" for t, c in sorted(tag_counts.items(), key=lambda x: -x[1])))
    print(f"\nWorst {min(top_n, len(errors))}:")
    for e in errors[:top_n]:
        print(f"  {e.wer:.3f}  {e.sample_id}  [{','.join(e.tags)}]")
        inv = f"  invented={e.invented}" if e.invented else ""
        drp = f"  dropped={e.dropped[:8]}" if e.dropped else ""
        print(f"           {inv}{drp}".rstrip())


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Build an error sheet from fusion_predictions.jsonl (eval rows)."
    )
    p.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to fusion_predictions.jsonl (Opus fs8 baseline, no glossary)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write error_sheet.md / error_sheet.tsv (default: alongside predictions)",
    )
    p.add_argument("--top", type=int, default=12, help="Top-N utterances in the markdown sheet")
    p.add_argument(
        "--asr-models",
        nargs="*",
        default=None,
        help="Optional ordered ASR keys to display (default: all in row)",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    errors = analyze(args.predictions, asr_models=args.asr_models)
    if not errors:
        raise SystemExit(f"No rows in {args.predictions}")

    pred_resolved = resolve_predictions_path(Path(args.predictions))
    out_dir = args.output_dir or pred_resolved.parent
    md_path = out_dir / "error_sheet.md"
    tsv_path = out_dir / "error_sheet.tsv"
    write_markdown(md_path, errors, top_n=args.top)
    write_tsv(tsv_path, errors)
    print_summary(errors, top_n=min(8, args.top))
    print(f"\nPredictions: {pred_resolved}")
    print(f"Wrote {md_path}")
    print(f"Wrote {tsv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
