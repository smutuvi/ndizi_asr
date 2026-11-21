#!/usr/bin/env python3
"""Figure 2 from the existing group table (results/summaries/ndizi1sw_sunflower/breakdown_group_ci.csv).

    python scripts/plot_fig_groups_ci.py \
        --table results/summaries/ndizi1sw_sunflower/breakdown_group_ci.csv \
        --system "LoRA-QC-v2 (Deploy)" --overall 14.60 --out figures/fig_groups.png

The table has columns factor, group, n, speakers, system, fair_wer_pct, ci_low_pct, ci_high_pct, with
farmer-level intervals already computed. This script only draws them: one panel per factor (duration,
first language, district, gender, age, in that order when present), points with interval bars, an open
circle where a group has one farmer or no interval, and a dashed line at the overall WER. It prints the
factors and systems it finds, and writes the plotted rows to a CSV next to the figure.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

TITLES = {"dur": "Duration", "duration": "Duration", "l1": "First language", "language": "First language",
          "first_language": "First language", "district": "District", "gender": "Gender", "sex": "Gender",
          "age": "Age", "age_band": "Age"}
ORDER = ["Duration", "First language", "District", "Gender", "Age"]
GROUP_ORDER = {"Duration": ["<=15s", "15-30s", ">30s"],
               "Age": ["<30", "under 30", "Under 30", "30-39", "40-49", "50+", ">=50", "50 and over"],
               "Gender": ["Female", "F", "female", "Male", "M", "male"]}
NAMES = {"Kiiraq": "Iraqw", "unknown": "Unspecified"}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--system", default="LoRA-QC-v2 (Deploy)")
    ap.add_argument("--overall", type=float, default=14.60)
    ap.add_argument("--skip-groups", nargs="*", default=["unknown age", "Unknown", "unknown", "nan", ""],
                    help="group labels left out of Gender and Age panels (unknown cells)")
    ap.add_argument("--out", type=Path, default=Path("figures/fig_groups.png"))
    args = ap.parse_args()

    rows = list(csv.DictReader(args.table.open(encoding="utf-8-sig")))
    print("factors:", sorted({r["factor"] for r in rows}))
    print("systems:", sorted({r["system"] for r in rows}))
    rows = [r for r in rows if r["system"] == args.system]
    if not rows:
        raise SystemExit(f"No rows for system {args.system!r}; pick one of the systems printed above.")
    panels = {}
    for r in rows:
        t = TITLES.get(r["factor"].strip().lower(), r["factor"])
        if t in ("Gender", "Age") and r["group"].strip() in args.skip_groups:
            continue
        panels.setdefault(t, []).append({
            "group": NAMES.get(r["group"], r["group"]), "clips": r["n"], "farmers": r["speakers"],
            "wer": num(r["fair_wer_pct"]), "low": num(r["ci_low_pct"]), "high": num(r["ci_high_pct"])})
    names = [p for p in ORDER if p in panels] + [p for p in panels if p not in ORDER]
    for p in names:
        if p in GROUP_ORDER:
            o = GROUP_ORDER[p]
            panels[p].sort(key=lambda s: o.index(s["group"]) if s["group"] in o else 99)
        else:
            panels[p].sort(key=lambda s: -int(float(s["clips"] or 0)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["panel", "group", "clips", "farmers", "wer", "low", "high"])
        for p in names:
            for s in panels[p]:
                w.writerow([p, s["group"], s["clips"], s["farmers"], s["wer"], s["low"], s["high"]])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    widths = [max(len(panels[p]), 2) for p in names]
    fig, axes = plt.subplots(1, len(names), figsize=(11, 3.1), sharey=True, gridspec_kw={"width_ratios": widths})
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, names):
        for i, s in enumerate(panels[p]):
            single = (s["low"] is None or s["high"] is None or str(s["farmers"]).strip() in ("1", "1.0"))
            if single:
                ax.plot(i, s["wer"], "o", mfc="white", mec="black", ms=5)
            else:
                ax.errorbar(i, s["wer"], yerr=[[s["wer"] - s["low"]], [s["high"] - s["wer"]]],
                            fmt="o", color="black", ms=4, capsize=2, lw=0.9)
        ax.axhline(args.overall, ls="--", lw=0.8, color="gray")
        ax.set_xticks(range(len(panels[p])))
        ax.set_xticklabels([f"{s['group']}\n({s['farmers']})" for s in panels[p]], rotation=60, ha="right", fontsize=7)
        ax.set_title(p, fontsize=9)
        ax.set_xlim(-0.6, len(panels[p]) - 0.4)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("Fair WER (%)", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    fig.savefig(args.out.with_suffix(".pdf"))
    print(f"Wrote {args.out}, {args.out.with_suffix('.pdf')} and {args.out.with_suffix('.csv')} "
          f"for {args.system} (numbers in parentheses are farmers)")


if __name__ == "__main__":
    main()
