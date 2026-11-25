#!/usr/bin/env python3
"""List the training-subset runs for the scaling study as "n<N>=<predictions file>" arguments for rescore_systems.py.

    python scripts/scaling_runs.py --results-dir results

Finds the folders results/sunflower51_ndizi1sw_lora_qc_v2*_n<N>[_test] with the same rule as analyze_scaling_curve.py (predictions.jsonl or
predictions.csv, in the folder or one level down, else utterances*.csv) and prints one line per run, sorted by N. The full run
(sunflower51_ndizi1sw_lora_qc_v2_test) is left out because the main rescoring already covers it.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def find_predictions(d: Path):
    cands = (sorted(d.glob("*/predictions.jsonl")) + sorted(d.glob("predictions.jsonl")) + sorted(d.glob("*/predictions.csv"))
             + sorted(d.glob("predictions.csv")) + sorted(d.glob("utterances*.csv")) + sorted(d.glob("*/utterances*.csv")))
    return cands[0] if cands else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()
    runs = {}
    for d in sorted(args.results_dir.glob("sunflower51_ndizi1sw_lora_qc_v2*")):
        m = re.search(r"_n(\d+)(?:_test)?$", d.name)
        if m and d.is_dir():
            f = find_predictions(d)
            if f:
                runs[int(m.group(1))] = f
    if not runs:
        raise SystemExit(f"No scaling runs found under {args.results_dir} (expected folders like sunflower51_ndizi1sw_lora_qc_v2_n1000_test)")
    for n in sorted(runs):
        print(f"n{n}={runs[n]}")


if __name__ == "__main__":
    main()
