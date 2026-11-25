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


def wer(pairs, cfg, cer=False):
    """(WER, reference words, CER or None). CER = corpus character error rate on the normalized strings (jiwer)."""
    e = n = 0
    refs, hyps = [], []
    for _, ref, hyp in pairs:
        rs, hs = fs.normalize(ref, **cfg), fs.normalize(hyp, **cfg)
        r, h = rs.split(), hs.split()
        e += fs.word_errors(r, h)
        n += len(r)
        if cer and rs.strip():
            refs.append(rs); hyps.append(hs)
    c = None
    if cer:
        import jiwer
        c = 100 * jiwer.cer(refs, hyps)
    return 100 * e / max(n, 1), n, c


def extras(pairs, cfg):
    """Error-type shares, share of clips with a much longer hypothesis, and variety-letter recall on the normalized text.

    sub/del/ins: substitutions, deletions, insertions as % of reference words (jiwer alignment), summing to the WER.
    long_hyp_pct: % of clips whose hypothesis has at least twice the reference's words and at least 10 more words (repetition loops, insertions).
    vl_*: the standalone tokens a, b, c are the variety letters; per clip, matched = sum over letters of min(count in reference, count in hypothesis);
          recall = matched / reference letter tokens, precision = matched / hypothesis letter tokens (clip-level multiset match, not an alignment).
    """
    import jiwer
    from collections import Counter
    refs, hyps = [], []
    long_n = n_clips = vref = vhyp = vmatch = 0
    for _, ref, hyp in pairs:
        rs, hs = fs.normalize(ref, **cfg), fs.normalize(hyp, **cfg)
        if not rs.strip():
            continue
        refs.append(rs); hyps.append(hs)
        rw, hw = len(rs.split()), len(hs.split())
        n_clips += 1
        long_n += (hw >= 2 * rw and hw - rw >= 10)
        cr = Counter(t for t in rs.split() if t in ("a", "b", "c"))
        ch = Counter(t for t in hs.split() if t in ("a", "b", "c"))
        vref += sum(cr.values()); vhyp += sum(ch.values()); vmatch += sum(min(cr[k], ch[k]) for k in cr)
    o = jiwer.process_words(refs, hyps)
    n = o.hits + o.substitutions + o.deletions
    return {"sub_pct": f"{100 * o.substitutions / n:.2f}", "del_pct": f"{100 * o.deletions / n:.2f}", "ins_pct": f"{100 * o.insertions / n:.2f}",
            "long_hyp_pct": f"{100 * long_n / max(n_clips, 1):.2f}", "vl_ref_tokens": vref,
            "vl_recall": f"{100 * vmatch / vref:.2f}" if vref else "", "vl_precision": f"{100 * vmatch / vhyp:.2f}" if vhyp else ""}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", required=True, help='"Name=path" for each system')
    ap.add_argument("--mined-map", type=Path, default=None, help="mined_map.json, or the reviewed mined_variants.csv")
    ap.add_argument("--compound-audit", type=Path, default=None)
    ap.add_argument("--repo-audit", type=Path, default=None,
                    help="variant_audit.csv: also keep the repository's entries whose status is in --repo-statuses")
    ap.add_argument("--repo-statuses", nargs="*", default=["trainval", "both_common"])
    ap.add_argument("--lists", type=Path, default=None,
                    help="fair_train.json from freeze_fair_train.py: adds the row 'Fair-train FROZEN' (the paper's metric under option B)")
    ap.add_argument("--exclude-sample-ids", nargs="*", default=[])
    ap.add_argument("--clean-ids-file", type=Path, default=None,
                    help="file with one sample_id per line (the audited, flagged test references): adds the Clean columns "
                         "(all clips except these, and except --exclude-sample-ids)")
    ap.add_argument("--clean-ids-dir", type=Path, default=None,
                    help="folder with <system name>.txt (from outlier_flags.py): each system's Clean column excludes ITS OWN outlier ids, "
                         "as the repository's original Clean score did")
    ap.add_argument("--cer", action="store_true", help="also report the character error rate (needs jiwer, which ndizi_asr already uses)")
    ap.add_argument("--extras", action="store_true",
                    help="also report, for the final metric (the frozen lists if given, else S4), substitution/deletion/insertion shares, "
                         "the share of clips with a much longer hypothesis, and variety-letter recall")
    ap.add_argument("--out", type=Path, default=Path("results/variant_audit/rescore.csv"))
    args = ap.parse_args()

    configs = dict(fs.STEPS)
    if args.mined_map:
        if args.mined_map.suffix == ".csv":      # the reviewed mined_variants.csv: use rows marked accepted=yes
            mined = {r["variant"]: r["canonical"] for r in csv.DictReader(args.mined_map.open(encoding="utf-8-sig"))
                     if r.get("accepted", "").strip().lower() == "yes"}
        else:
            mined = json.loads(args.mined_map.read_text(encoding="utf-8"))
        n_mined = len(mined)
        if args.repo_audit:
            keep_repo = {}
            for r in csv.DictReader(args.repo_audit.open(encoding="utf-8")):
                v, t = r["variant"], r["target"]
                if r["status"] not in args.repo_statuses:
                    continue
                if ("kua" in v and "kuwa" in t) or ("kuwa" in v and "kua" in t):
                    continue                    # 'grow' vs 'become': different verbs
                keep_repo[v] = t
            mined = {**mined, **keep_repo}      # on a conflict the repository entry wins
            for k in list(mined):               # resolve chains (ndiyo->ndio->ndo becomes ndiyo->ndo)
                seen = {k}
                while mined[k] in mined and mined[mined[k]] != mined[k] and mined[k] not in seen:
                    seen.add(mined[k]); mined[k] = mined[mined[k]]
            print(f"Fair-train variants: {n_mined} mined + {len(keep_repo)} repository entries attested in training "
                  f"(statuses {args.repo_statuses}, kua/kuwa pairs removed) = {len(mined)} after merging")
        comps = fs.REPO_COMPOUNDS
        if args.compound_audit:
            keep = {int(r["index"]) for r in csv.DictReader(args.compound_audit.open(encoding="utf-8"))
                    if r["status"] == "trainval"}
            comps = [c for i, c in enumerate(fs.REPO_COMPOUNDS) if i in keep]
        configs["Fair-train (training-only lists)"] = dict(backch=True, elong=True, compounds=comps, variants=mined)
        print(f"Fair-train uses {len(mined)} mined variants and {len(comps)} of {len(fs.REPO_COMPOUNDS)} compound rules")

    if args.lists:
        frozen = fs.load_lists(args.lists)
        configs["Fair-train: + frozen compounds"] = dict(backch=True, elong=True, compounds=frozen["compounds"], variants={})
        configs["Fair-train FROZEN (reviewed lists)"] = frozen
    final_step = "Fair-train FROZEN (reviewed lists)" if args.lists else "S4 + variant map = Fair (repository)"
    excl = set(args.exclude_sample_ids)
    clean_excl = set()
    if args.clean_ids_file:
        clean_excl = {l.strip() for l in args.clean_ids_file.read_text(encoding="utf-8").splitlines() if l.strip()}
        print(f"Clean column: excluding {len(clean_excl)} flagged ids (the same set for every system)")
    out = []
    for spec in args.systems:
        name, path = spec.split("=", 1)
        if not path.strip() or Path(path).is_dir() or not Path(path).exists():
            raise SystemExit(f"{name}: the path {path!r} is empty, a folder, or does not exist. A shell variable such as $V2 is "
                             f"probably not set in this terminal: run echo \"$V2\" to check, or write the full path.")
        pairs = load(Path(path))
        bad = fs.check_identical([p[1] for p in pairs] + [p[2] for p in pairs])
        if bad:
            raise SystemExit(f"{name}: the step pipeline differs from the repository normalizer on {bad} texts; stopping.")
        kept = [p for p in pairs if p[0] not in excl]
        own = None
        if args.clean_ids_dir:
            f = args.clean_ids_dir / f"{name}.txt"
            if not f.exists():
                raise SystemExit(f"{f} not found: run outlier_flags.py with this system name first")
            own = {l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()}
        for step, cfg in configs.items():
            w_all, n_all, c_all = wer(pairs, cfg, args.cer)
            w_k, n_k, c_k = wer(kept, cfg, args.cer)
            row = {"system": name, "normalization": step, "clips_all": len(pairs), "wer_all": f"{w_all:.2f}",
                   "clips_without_excluded": len(kept), "wer_without_excluded": f"{w_k:.2f}", "ref_words_all": n_all}
            extra = ""
            if args.cer:
                row["cer_all"], row["cer_without_excluded"] = f"{c_all:.2f}", f"{c_k:.2f}"
                extra += f"   CER {c_all:5.2f} / {c_k:5.2f}"
            if clean_excl or own is not None:
                ce = own if own is not None else clean_excl
                clean = [p for p in pairs if p[0] not in ce and p[0] not in excl]
                w_c, _, c_c = wer(clean, cfg, args.cer)
                row["clips_clean"], row["wer_clean"] = len(clean), f"{w_c:.2f}"
                extra += f"   clean {len(clean):5d} clips WER {w_c:6.2f}"
                if args.cer:
                    row["cer_clean"] = f"{c_c:.2f}"; extra += f" CER {c_c:5.2f}"
            if args.extras and step == final_step:
                row.update(extras(kept, cfg))
                extra += f"   S/D/I {row['sub_pct']}/{row['del_pct']}/{row['ins_pct']} long-hyp {row['long_hyp_pct']}% letters recall {row['vl_recall']}"
            out.append(row)
            print(f"{name:22s} {step:40s} {len(pairs):5d} clips {w_all:6.2f}   {len(kept):5d} clips {w_k:6.2f}{extra}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        cols = []
        for r in out:
            cols += [k for k in r if k not in cols]
        w = csv.DictWriter(f, fieldnames=cols, restval=""); w.writeheader(); w.writerows(out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
