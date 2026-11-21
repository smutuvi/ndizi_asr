#!/usr/bin/env python3
"""Figure 2: Fair WER of one system by group, with farmer-level bootstrap intervals.

    python scripts/make_fig_groups.py \
        --predictions results/<run>/<model>/predictions.jsonl \
        --inventory test_inventory.csv \
        --demographics private/speaker_demographics.csv \
        --out figures/fig_groups.png --draws 4000 --seed 7

Inputs
  predictions.jsonl  rows with sample_id, reference, prediction (as written by ndizi_asr's evaluation)
  test_inventory.csv  sample_id, speaker_id, duration_sec, language (first language), district
  demographics CSV    optional and private (never released): speaker_id, gender, age

Scoring: each clip is scored with the repository's Swahili Fair normalizer
(ndizi_asr.metrics.normalize_text in mode swahili_fair), applied to reference and prediction alike.
If ndizi_asr cannot be imported the script stops; it never falls back silently. --normalizer none
(lowercase and split) exists only for testing the plumbing.

Group WER is pooled: total word errors (substitutions + deletions + insertions) over total reference
words. Intervals resample the group's farmers with replacement (cluster bootstrap) and take the 2.5
and 97.5 percentiles; a group with one farmer gets no interval and is drawn as an open circle.
Clips listed in --exclude-sample-ids (for example clips that only record a consent note) are dropped.

Outputs: the figure (PNG and PDF) and a CSV next to it with every plotted number (group, clips,
farmers, WER, interval), so the paper's text can be checked against the figure.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

L1_NAMES = {"Kiiraq": "Iraqw", "unknown": "Unspecified", "": "Unspecified"}


def word_errors(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein distance on words: substitutions + deletions + insertions."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def get_normalizer(name: str):
    if name == "none":
        return lambda t: " ".join(str(t or "").lower().split())
    try:
        from ndizi_asr.config import NormalizationConfig
        from ndizi_asr.metrics import normalize_text
    except ImportError as e:
        raise SystemExit(f"Cannot import ndizi_asr ({e}). Run inside the ndizi_asr environment "
                         "(pip install -e .), or use --normalizer none only to test the plumbing.")
    cfg = NormalizationConfig(mode="swahili_fair")
    return lambda t: normalize_text(t or "", cfg, language="sw")


def age_band(a: str) -> str | None:
    try:
        a = float(a)
    except (TypeError, ValueError):
        return None
    return "Under 30" if a < 30 else "30-39" if a < 40 else "40-49" if a < 50 else "50+"


def dur_band(d: str) -> str:
    d = float(d)
    return "<=15 s" if d <= 15 else "15-30 s" if d <= 30 else ">30 s"


def group_stats(rows: list[dict], key: str, draws: int, rng) -> list[dict]:
    out = []
    groups = {}
    for r in rows:
        g = r.get(key)
        if g is None:
            continue
        groups.setdefault(g, []).append(r)
    for g, rs in groups.items():
        per = {}
        for r in rs:
            e, w = per.get(r["speaker"], (0, 0))
            per[r["speaker"]] = (e + r["err"], w + r["words"])
        err = np.array([v[0] for v in per.values()], dtype=float)
        words = np.array([v[1] for v in per.values()], dtype=float)
        wer = 100 * err.sum() / max(words.sum(), 1)
        lo = hi = None
        if len(per) > 1:
            idx = rng.integers(0, len(per), size=(draws, len(per)))
            boot = 100 * err[idx].sum(axis=1) / np.maximum(words[idx].sum(axis=1), 1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append({"panel": key, "group": g, "clips": len(rs), "farmers": len(per), "wer": wer, "low": lo, "high": hi})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--demographics", type=Path, default=None)
    ap.add_argument("--exclude-sample-ids", nargs="*", default=[])
    ap.add_argument("--normalizer", choices=["swahili_fair", "none"], default="swahili_fair")
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("figures/fig_groups.png"))
    args = ap.parse_args()

    norm = get_normalizer(args.normalizer)
    inv = {r["sample_id"]: r for r in csv.DictReader(args.inventory.open(encoding="utf-8-sig"))}
    demo = {}
    if args.demographics:
        demo = {r["speaker_id"]: r for r in csv.DictReader(args.demographics.open(encoding="utf-8-sig"))}
    excluded = set(args.exclude_sample_ids)

    rows, missing = [], 0
    for line in args.predictions.open(encoding="utf-8"):
        if not line.strip():
            continue
        p = json.loads(line)
        sid = str(p["sample_id"])
        if sid in excluded:
            continue
        m = inv.get(sid)
        if m is None:
            missing += 1
            continue
        ref, hyp = norm(p.get("reference", "")).split(), norm(p.get("prediction", "")).split()
        d = demo.get(m["speaker_id"], {})
        g = (d.get("gender") or "").strip().lower()
        rows.append({
            "speaker": m["speaker_id"], "err": word_errors(ref, hyp), "words": len(ref),
            "Duration": dur_band(m["duration_sec"]),
            "First language": L1_NAMES.get(m.get("language", ""), m.get("language", "")),
            "District": m.get("district") or None,
            "Gender": {"f": "Female", "female": "Female", "m": "Male", "male": "Male"}.get(g),
            "Age": age_band(d.get("age")),
        })
    if not rows:
        raise SystemExit("No prediction matched the inventory (check that sample_id values agree).")
    overall = 100 * sum(r["err"] for r in rows) / max(sum(r["words"] for r in rows), 1)
    print(f"{len(rows)} clips scored ({missing} predictions not in the inventory, {len(excluded)} ids excluded); "
          f"overall Fair WER {overall:.2f}%")

    rng = np.random.default_rng(args.seed)
    panels = ["Duration", "First language", "District"] + (["Gender", "Age"] if demo else [])
    order = {"Duration": ["<=15 s", "15-30 s", ">30 s"], "Age": ["Under 30", "30-39", "40-49", "50+"],
             "Gender": ["Female", "Male"]}
    stats = []
    for pnl in panels:
        st = group_stats(rows, pnl, args.draws, rng)
        if pnl in order:
            st.sort(key=lambda s: order[pnl].index(s["group"]) if s["group"] in order[pnl] else 99)
        else:
            st.sort(key=lambda s: -s["clips"])
        stats += st

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["panel", "group", "clips", "farmers", "wer", "low", "high"])
        w.writeheader()
        for s in stats:
            w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in s.items()})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    widths = [max(len([s for s in stats if s["panel"] == p]), 2) for p in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 3.1), sharey=True, gridspec_kw={"width_ratios": widths})
    axes = np.atleast_1d(axes)
    for ax, pnl in zip(axes, panels):
        st = [s for s in stats if s["panel"] == pnl]
        for i, s in enumerate(st):
            if s["low"] is None:
                ax.plot(i, s["wer"], "o", mfc="white", mec="black", ms=5)
            else:
                ax.errorbar(i, s["wer"], yerr=[[s["wer"] - s["low"]], [s["high"] - s["wer"]]],
                            fmt="o", color="black", ms=4, capsize=2, lw=0.9)
        ax.axhline(overall, ls="--", lw=0.8, color="gray")
        ax.set_xticks(range(len(st)))
        ax.set_xticklabels([f"{s['group']}\n({s['farmers']})" for s in st], rotation=60, ha="right", fontsize=7)
        ax.set_title(pnl, fontsize=9)
        ax.set_xlim(-0.6, len(st) - 0.4)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("Fair WER (%)", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    fig.savefig(args.out.with_suffix(".pdf"))
    print(f"Wrote {args.out}, {args.out.with_suffix('.pdf')} and {args.out.with_suffix('.csv')} "
          f"(numbers in parentheses under each label are farmers)")


if __name__ == "__main__":
    main()
