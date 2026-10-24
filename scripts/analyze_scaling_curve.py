#!/usr/bin/env python3
"""Evaluate and fit a neural data-scaling curve for SunflowerASR-51 Swahili LoRA-QC-v2.

Discovers test-set evaluations across sample budgets (e.g. N in {250, 500, 1000, 2000, 4000, 7608}),
computes Fixed WER, Swahili Fair WER, and Fair Clean WER (plus per-L1 / per-duration breakdowns),
fits the power-law scaling model:

    WER(N) = WER_inf + A * N^(-alpha)

and extrapolates expected WER at N = 10,000, 15,000, and 30,000 segments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from jiwer import cer as jiwer_cer, wer as jiwer_wer

from ndizi_asr.ethio_normalize import normalize_swahili_fair
from ndizi_asr.metrics import clean_glued_punctuation


def _load_Utterances_or_preds(exp_dir: Path) -> List[Dict[str, Any]]:
    """Load (id, reference, hypothesis) rows from utterances.csv or predictions.jsonl."""
    utt_csvs = sorted(exp_dir.glob("utterances*.csv")) + sorted(exp_dir.glob("*/utterances*.csv"))
    for p in utt_csvs:
        rows: List[Dict[str, Any]] = []
        with p.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                ref = (r.get("reference") or "").strip()
                hyp = (r.get("hypothesis") or "").strip()
                if ref:
                    rows.append(
                        {
                            "id": str(r.get("id") or len(rows)),
                            "reference": ref,
                            "hypothesis": hyp,
                        }
                    )
        if rows:
            return rows

    pred_files = sorted(exp_dir.glob("*/predictions.jsonl")) + sorted(exp_dir.glob("predictions.jsonl"))
    for p in pred_files:
        rows = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("error"):
                    continue
                ref = (obj.get("reference") or "").strip()
                hyp = (obj.get("hypothesis") or "").strip()
                if ref:
                    rows.append(
                        {
                            "id": str(obj.get("id") or len(rows)),
                            "reference": ref,
                            "hypothesis": hyp,
                        }
                    )
        if rows:
            return rows
    raise FileNotFoundError(f"No utterances.csv or predictions.jsonl found under {exp_dir}")


def score_run(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    refs_fixed: List[str] = []
    hyps_fixed: List[str] = []
    refs_fair: List[str] = []
    hyps_fair: List[str] = []
    refs_fair_clean: List[str] = []
    hyps_fair_clean: List[str] = []

    for r in rows:
        rf = clean_glued_punctuation(r["reference"])
        hf = clean_glued_punctuation(r["hypothesis"])
        if not rf:
            continue
        refs_fixed.append(rf)
        hyps_fixed.append(hf)

        r_fair = normalize_swahili_fair(r["reference"])
        h_fair = normalize_swahili_fair(r["hypothesis"])
        if not r_fair:
            continue
        refs_fair.append(r_fair)
        hyps_fair.append(h_fair)

        utt_wer_fixed = float(jiwer_wer([rf], [hf]))
        ref_words = len(rf.split())
        # Same outlier rule used across PAPER_REPORT_NDIZI_SWAHILI_ASR:
        # drop corrupted/truncated reference outliers (ref_words <= 2 and WER >= 1.0, or WER > 2.0)
        is_outlier = (ref_words <= 2 and utt_wer_fixed >= 1.0) or (utt_wer_fixed > 2.0)
        if not is_outlier:
            refs_fair_clean.append(r_fair)
            hyps_fair_clean.append(h_fair)

    return {
        "n_total": len(refs_fair),
        "n_clean": len(refs_fair_clean),
        "wer_fixed": float(jiwer_wer(refs_fixed, hyps_fixed)),
        "cer_fixed": float(jiwer_cer(refs_fixed, hyps_fixed)),
        "wer_swahili_fair": float(jiwer_wer(refs_fair, hyps_fair)),
        "cer_swahili_fair": float(jiwer_cer(refs_fair, hyps_fair)),
        "wer_swahili_fair_clean": float(jiwer_wer(refs_fair_clean, hyps_fair_clean)),
        "cer_swahili_fair_clean": float(jiwer_cer(refs_fair_clean, hyps_fair_clean)),
    }


def fit_power_law(ns: List[int], wers: List[float]) -> Optional[Tuple[float, float, float, float]]:
    """Grid + least-squares fit for WER(N) = wer_inf + A * N^(-alpha).

    Returns (wer_inf, A, alpha, r2) with percentages in [0, 1].
    """
    if len(ns) < 3:
        return None
    x = np.asarray(ns, dtype=np.float64)
    y = np.asarray(wers, dtype=np.float64)
    min_y = float(np.min(y))

    best_sse = float("inf")
    best_params: Optional[Tuple[float, float, float]] = None

    # Grid search over plausible asymptote wer_inf in [0.04, min_y - 0.002] and alpha in [0.10, 0.85]
    upper_inf = max(0.045, min_y - 0.002)
    for wer_inf in np.linspace(0.04, upper_inf, 120):
        rem = y - wer_inf
        if np.any(rem <= 1e-5):
            continue
        for alpha in np.linspace(0.10, 0.85, 150):
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
            utt_rows = _load_Utterances_or_preds(exp_dir)
            metrics = score_run(utt_rows)
            row = {
                "n_train_segments": n_train,
                "approx_interviews": int(round(n_train * 5572.0 / 7608.0)),
                "approx_hours": round(n_train * 42.0 / 7608.0, 2),
                "experiment_dir": exp_dir.name,
                **metrics,
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
        print("\n=== Power-Law Scaling Fit: WER(N) = WER_inf + A * N^(-alpha) ===")
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
