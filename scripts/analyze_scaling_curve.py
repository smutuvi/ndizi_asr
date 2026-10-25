#!/usr/bin/env python3
"""Evaluate and fit a neural data-scaling curve for SunflowerASR-51 Swahili LoRA-QC-v2.

Discovers test-set evaluations across sample budgets (e.g. N in {250, 500, 1000, 2000, 4000, 7608}),
computes Fixed WER, Swahili Fair WER, and Fair Clean WER via ``rescore_predictions_file``,
fits the power-law scaling model:

    WER(N) = WER_inf + A * N^(-alpha)

and extrapolates expected WER at N = 10,000, 15,000, and 30,000 segments.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ndizi_asr.data_qc import rescore_predictions_file


def _find_predictions_file(exp_dir: Path) -> Path:
    candidates = (
        sorted(exp_dir.glob("*/predictions.jsonl"))
        + sorted(exp_dir.glob("predictions.jsonl"))
        + sorted(exp_dir.glob("*/predictions.csv"))
        + sorted(exp_dir.glob("predictions.csv"))
        + sorted(exp_dir.glob("utterances*.csv"))
        + sorted(exp_dir.glob("*/utterances*.csv"))
    )
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    raise FileNotFoundError(f"No predictions.jsonl/csv or utterances.csv found under {exp_dir}")


def fit_power_law(ns: List[int], wers: List[float]) -> Optional[Tuple[float, float, float, float]]:
    """Grid + least-squares fit for WER(N) = wer_inf + A * N^(-alpha).

    Returns (wer_inf, A, alpha, r2) with fractions in [0, 1].
    """
    if len(ns) < 2:
        return None
    x = np.asarray(ns, dtype=np.float64)
    y = np.asarray(wers, dtype=np.float64)
    min_y = float(np.min(y))

    # With 2 points, fix a conservative Swahili irreducible floor (11.5%) and solve 2-parameter power law
    wer_inf_grid = (
        np.asarray([min(0.115, max(0.05, min_y - 0.025))], dtype=np.float64)
        if len(ns) == 2
        else np.linspace(0.04, max(0.045, min_y - 0.002), 120)
    )

    best_sse = float("inf")
    best_params: Optional[Tuple[float, float, float]] = None

    for wer_inf in wer_inf_grid:
        rem = y - wer_inf
        if np.any(rem <= 1e-5):
            continue
        for alpha in np.linspace(0.08, 0.85, 160):
            basis = np.power(x, -alpha)
            a_hat = float(np.dot(basis, rem) / max(np.dot(basis, basis), 1e-12))
            if a_hat <= 0:
                continue
            pred = wer_inf + a_hat * basis
            sse = float(np.sum((y - pred) ** 2))
            if sse < best_sse:
                best_sse = sse
                best_params = (float(wer_inf), a_hat, float(alpha))

    if best_params is None:
        return None
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (best_sse / max(tss, 1e-12))
    return (*best_params, r2)


def main() -> int:
    p = argparse.ArgumentParser(description="Analyze SunflowerASR-51 Swahili LoRA data-scaling curve.")
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root directory containing sunflower51_ndizi1sw_lora_qc_v2*_test folders.",
    )
    p.add_argument(
        "--full-n",
        type=int,
        default=7608,
        help="Segment count for the full training run (default: 7608).",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/summaries/ndizi1sw_sunflower/scaling_curve_summary.csv"),
        help="Output CSV path.",
    )
    args = p.parse_args()

    runs: Dict[int, Path] = {}
    for d in sorted(args.results_dir.glob("sunflower51_ndizi1sw_lora_qc_v2*")):
        if not d.is_dir():
            continue
        m = re.search(r"_n(\d+)(?:_test)?$", d.name)
        if m:
            runs[int(m.group(1))] = d
        elif d.name == "sunflower51_ndizi1sw_lora_qc_v2_test":
            runs[int(args.full_n)] = d

    if not runs:
        print(f"No scaling evaluation directories found under {args.results_dir}")
        return 1

    summary_rows: List[Dict[str, Any]] = []
    for n_train in sorted(runs.keys()):
        exp_dir = runs[n_train]
        try:
            pred_file = _find_predictions_file(exp_dir)
            m = rescore_predictions_file(pred_file)
            row = {
                "n_train_segments": n_train,
                "approx_interviews": int(round(n_train * 5572.0 / 7608.0)),
                "approx_hours": round(n_train * 42.0 / 7608.0, 2),
                "experiment_dir": exp_dir.name,
                "n_total": m["n_total"],
                "n_clean": m["n_clean_after_outlier_filter"],
                "wer_fixed": m["wer_fixed_punct_norm"],
                "cer_fixed": m["cer_fixed_punct_norm"],
                "wer_swahili_fair": m["wer_swahili_fair"],
                "cer_swahili_fair": m["cer_swahili_fair"],
                "wer_swahili_fair_clean": m["wer_swahili_fair_clean"],
                "cer_swahili_fair_clean": m["cer_swahili_fair_clean"],
            }
            summary_rows.append(row)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Skipping {exp_dir}: {exc}")

    if not summary_rows:
        print("No valid evaluation runs could be scored.")
        return 1

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary_rows[0].keys())
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n=== SunflowerASR-51 Swahili LoRA-QC-v2 Data Scaling Curve ===")
    print(
        f"{'N_segments':>10} | {'~Clips':>7} | {'~Hours':>6} | "
        f"{'Fixed WER':>9} | {'Fair WER':>9} | {'Fair Clean':>10} | {'Delta Fair':>10}"
    )
    print("-" * 78)
    prev_fair: Optional[float] = None
    for r in summary_rows:
        fair = r["wer_swahili_fair"]
        delta_str = f"{100.0 * (fair - prev_fair):+6.2f} pp" if prev_fair is not None else "    baseline"
        prev_fair = fair
        print(
            f"{r['n_train_segments']:10d} | {r['approx_interviews']:7d} | {r['approx_hours']:5.1f}h | "
            f"{100.0 * r['wer_fixed']:8.2f}% | {100.0 * fair:8.2f}% | "
            f"{100.0 * r['wer_swahili_fair_clean']:9.2f}% | {delta_str:>10}"
        )

    ns = [int(r["n_train_segments"]) for r in summary_rows]
    fairs = [float(r["wer_swahili_fair"]) for r in summary_rows]
    fit = fit_power_law(ns, fairs)
    if fit is not None:
        wer_inf, a_coeff, alpha, r2 = fit
        note = " (2-point estimate; add N=1000,2000 for full 3-param fit)" if len(ns) == 2 else ""
        print(f"\n=== Power-Law Scaling Fit: WER(N) = WER_inf + A * N^(-alpha){note} ===")
        print(
            f"  Irreducible floor (WER_inf): {100.0 * wer_inf:.2f}% "
            f"| A = {100.0 * a_coeff:.2f}% | alpha = {alpha:.3f} | R^2 = {r2:.4f}"
        )
        for target_n in (10000, 15000, 30000):
            pred_wer = wer_inf + a_coeff * (target_n ** (-alpha))
            hrs = target_n * 42.0 / 7608.0
            gain_vs_full = 100.0 * (pred_wer - fairs[-1])
            print(
                f"  Projected at N = {target_n:6d} (~{hrs:5.1f}h): "
                f"{100.0 * pred_wer:6.2f}% Fair WER ({gain_vs_full:+5.2f} pp vs N={ns[-1]})"
            )
    print(f"\nSaved CSV to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
