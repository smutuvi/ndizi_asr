#!/usr/bin/env python3
"""Experiments 1b, 2 and 6: rescore every system under each normalization step and variant list.

    PYTHONPATH=src python scripts/rescore_systems.py \
        --systems "LoRA-QC-v2=results/.../predictions.jsonl" "LoRA-30s=results/summaries/.../utterances.csv" ... \
        --mined-map results/variant_audit/mined_map.json \
        --compound-audit results/variant_audit/compound_audit.csv \
        --exclude-sample-ids "smutuvi_ndizi-1-sw:smutuvi_ndizi-1-sw_403:403" "smutuvi_ndizi-1-sw:smutuvi_ndizi-1-sw_829:829" \
        --out results/variant_audit/rescore.csv

Each system file is a predictions.jsonl (sample_id, reference, prediction) or a CSV with sample_id, reference and
prediction or hypothesis. For every system it reports corpus WER (total word errors over total reference words)
under:
  S0 base, S1 + backchannels, S2 + elongation, S3 + compounds, S4 Fair (repository lists)   (experiment 6)
  Fair-train: steps S0-S3 with only the compound rules attested in training+validation, plus the mined map (1b)
each on all clips and without the --exclude-sample-ids clips (experiment 2). Before scoring it checks that S4
reproduces the repository's clean_swahili_fair_transcription on every reference and hypothesis.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import fair_steps as fs


def load(path: Path):
    if path.suffix == ".jsonl":
        rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    else:
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    out = []
    for r in rows:
        hyp = r.get("prediction", r.get("hypothesis", r.get("fusion", "")))
        out.append((str(r["sample_id"]), r.get("reference") or "", hyp or ""))
    return out


def wer(pairs, cfg):
    e = n = 0
    for _, ref, hyp in pairs:
        r = fs.normalize(ref, **cfg).split()
        h = fs.normalize(hyp, **cfg).split()
        e += fs.word_errors(r, h)
        n += len(r)
    return 100 * e / max(n, 1), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", required=True, help='"Name=path" for each system')
    ap.add_argument("--mined-map", type=Path, default=None, help="mined_map.json, or the reviewed mined_variants.csv")
    ap.add_argument("--compound-audit", type=Path, default=None)
    ap.add_argument("--exclude-sample-ids", nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=Path("results/variant_audit/rescore.csv"))
    args = ap.parse_args()

    configs = dict(fs.STEPS)
    if args.mined_map:
        if args.mined_map.suffix == ".csv":      # the reviewed mined_variants.csv: use rows marked accepted=yes
            mined = {r["variant"]: r["canonical"] for r in csv.DictReader(args.mined_map.open(encoding="utf-8-sig"))
                     if r.get("accepted", "").strip().lower() == "yes"}
        else:
            mined = json.loads(args.mined_map.read_text(encoding="utf-8"))
        comps = fs.REPO_COMPOUNDS
        if args.compound_audit:
            keep = {int(r["index"]) for r in csv.DictReader(args.compound_audit.open(encoding="utf-8"))
                    if r["status"] == "trainval"}
            comps = [c for i, c in enumerate(fs.REPO_COMPOUNDS) if i in keep]
        configs["Fair-train (training-only lists)"] = dict(backch=True, elong=True, compounds=comps, variants=mined)
        print(f"Fair-train uses {len(mined)} mined variants and {len(comps)} of {len(fs.REPO_COMPOUNDS)} compound rules")

    excl = set(args.exclude_sample_ids)
    out = []
    for spec in args.systems:
        name, path = spec.split("=", 1)
        pairs = load(Path(path))
        bad = fs.check_identical([p[1] for p in pairs] + [p[2] for p in pairs])
        if bad:
            raise SystemExit(f"{name}: the step pipeline differs from the repository normalizer on {bad} texts; stopping.")
        kept = [p for p in pairs if p[0] not in excl]
        for step, cfg in configs.items():
            w_all, n_all = wer(pairs, cfg)
            w_k, n_k = wer(kept, cfg)
            out.append({"system": name, "normalization": step, "clips_all": len(pairs), "wer_all": f"{w_all:.2f}",
                        "clips_without_excluded": len(kept), "wer_without_excluded": f"{w_k:.2f}",
                        "ref_words_all": n_all})
            print(f"{name:22s} {step:40s} {len(pairs):5d} clips {w_all:6.2f}   {len(kept):5d} clips {w_k:6.2f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
