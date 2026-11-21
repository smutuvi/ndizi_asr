#!/usr/bin/env python3
"""Experiment 1a: where do the Fair normalizer's lists come from, and a replacement built from training data only.

    PYTHONPATH=src python scripts/variant_audit.py \
        --train response_coding/sampling/train_inventory.csv \
        --valid results/coding_gold/validation_inventory.csv \
        --test response_coding/sampling/test_inventory.csv \
        --out-dir results/variant_audit

Reads transcripts (column `transcript`) and normalizes them with every Fair step except the variant map.
Writes:
  variant_audit.csv   each entry of the repository's variant map: how often the variant and its target occur
                      in training+validation and in the test references, and a status:
                        trainval      the variant occurs in training or validation transcripts
                        test_only     it occurs only in test references (it could only have come from test)
                        unattested    it occurs nowhere in these transcripts
                        both_common   both forms are common in training data: possibly two different words
  compound_audit.csv  the same for each compound/split rule (matches in training+validation versus test)
  mined_variants.csv  candidate pairs found ONLY in training+validation by the variant rules below, with counts
  mined_map.json      the automatically accepted pairs (variant -> canonical). These are CANDIDATES: spelling rules
                      cannot tell a variant from a different word, so have a Swahili reader review mined_variants.csv
                      (set the accepted column to yes/no) and pass the reviewed CSV to rescore_systems.py
Mining rules: r<->l at one position, dropped initial h (only restoring h, never removing it, because
initial h- also marks negation), w/y glide after u/i, one doubled vowel. A pair is
accepted when both forms occur in training+validation, the variant at least --min-count times, the canonical
form (the more frequent one) at least --ratio times as often, the word has at least 3 letters, and the pair is
not in --protect (pairs that are different words, e.g. kuwa/kua). Test transcripts are only counted for the
audit; they never decide a mined pair. A Swahili reader should still look over mined_variants.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import fair_steps as fs
import ndizi_asr.ethio_normalize as en

VOWELS = set("aeiou")
DEFAULT_PROTECT = ["kuwa:kua", "ajiri:ajili", "kura:kula", "mara:mala", "elezea:eleza", "zilizo:zilizopo",
                   "pole:pore", "lima:rima", "hali:ali", "hasa:asa"]


def read_texts(paths):
    out = []
    for p in paths:
        if p and Path(p).exists():
            out += [r.get("transcript") or "" for r in csv.DictReader(open(p, encoding="utf-8-sig"))]
        elif p:
            print(f"warning: {p} not found, skipped")
    return out


def core_tokens(texts):
    c = Counter()
    for t in texts:
        c.update(fs.normalize(t, variants={}).split())
    return c


def candidates(tok: str):
    """(variant_form, rule) pairs one edit away under the Swahili variant rules."""
    out = []
    for i, ch in enumerate(tok):
        if ch in "rl":
            out.append((tok[:i] + ("l" if ch == "r" else "r") + tok[i + 1:], "r/l"))
    if tok.startswith("h") and len(tok) > 3:
        out.append((tok[1:], "initial h"))
    else:
        out.append(("h" + tok, "initial h"))
    for i in range(1, len(tok) - 1):
        if tok[i] in "wy" and tok[i - 1] in ("u" if tok[i] == "w" else "i") and tok[i + 1] in VOWELS:
            out.append((tok[:i] + tok[i + 1:], "glide"))
    if DOUBLE_VOWEL_RULE:
        for i in range(len(tok) - 1):
            if tok[i] in VOWELS and tok[i] == tok[i + 1]:
                out.append((tok[:i] + tok[i + 1:], "double vowel"))
    return out


DOUBLE_VOWEL_RULE = False   # off by default: word-internal double vowels are often lexical (kuuza 'sell' / kuza 'grow')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", default=None)
    ap.add_argument("--test", required=True)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--ratio", type=float, default=3.0)
    ap.add_argument("--protect", nargs="*", default=DEFAULT_PROTECT, help="pairs a:b that are different words")
    ap.add_argument("--double-vowel", action="store_true", help="also try the doubled-vowel rule (risky; review closely)")
    ap.add_argument("--out-dir", type=Path, default=Path("results/variant_audit"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    global DOUBLE_VOWEL_RULE
    DOUBLE_VOWEL_RULE = args.double_vowel

    tv_texts = read_texts([args.train, args.valid])
    te_texts = read_texts([args.test])
    tv, te = core_tokens(tv_texts), core_tokens(te_texts)
    print(f"training+validation: {len(tv_texts)} transcripts, {sum(tv.values())} tokens, {len(tv)} types | "
          f"test: {len(te_texts)} transcripts")

    # ---- audit of the repository's variant map
    rows, status = [], Counter()
    for var, can in fs.REPO_MAP.items():
        var, can = var.strip(), can.strip()
        if var == can:
            continue
        a, b, t = tv.get(var, 0), tv.get(can, 0), te.get(var, 0)
        s = ("both_common" if a >= 5 and b >= 5 and max(a, b) < 3 * min(a, b) else
             "trainval" if a > 0 else "test_only" if t > 0 else "unattested")
        status[s] += 1
        rows.append({"variant": var, "target": can, "count_trainval_variant": a, "count_trainval_target": b,
                     "count_test_variant": t, "status": s})
    with (args.out_dir / "variant_audit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ---- audit of the compound rules (on lowercased, punctuation-free text)
    def flat(texts):
        return [fs.normalize(t, compounds=[], variants={}) for t in texts]
    tvf, tef = flat(tv_texts), flat(te_texts)
    crow, cstat = [], Counter()
    for i, (pat, repl) in enumerate(fs.REPO_COMPOUNDS):
        a = sum(len(pat.findall(x)) for x in tvf)
        t = sum(len(pat.findall(x)) for x in tef)
        s = "trainval" if a > 0 else "test_only" if t > 0 else "unattested"
        cstat[s] += 1
        crow.append({"index": i, "pattern": pat.pattern, "replacement": repl, "matches_trainval": a,
                     "matches_test": t, "status": s})
    with (args.out_dir / "compound_audit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(crow[0])); w.writeheader(); w.writerows(crow)

    # ---- mining from training+validation only
    protect = {tuple(sorted(p.split(":"))) for p in args.protect}
    lexical = getattr(en, "_SWAHILI_LEXICAL_DOUBLE_VOWEL_WORDS", set())
    seen, mined = set(), []
    for tok in tv:
        if len(tok) < 3 or not tok.isalpha():
            continue
        for var, rule in candidates(tok):
            if var not in tv or len(var) < 3:
                continue
            pair = tuple(sorted((tok, var)))
            if pair in seen:
                continue
            seen.add(pair)
            a, b = tv[pair[0]], tv[pair[1]]
            can, vf = (pair[0], pair[1]) if a >= b else (pair[1], pair[0])
            cc, vc = tv[can], tv[vf]
            reason = ""
            if pair in protect:
                reason = "protected (different words)"
            elif any(("kuwa" in x and "kua" in y) or ("kua" in x and "kuwa" in y) for x, y in [pair, pair[::-1]]):
                reason = "kua 'grow' / kuwa 'be' family: different verbs"
            elif rule == "initial h" and not can.startswith("h"):
                # Dropping h is the first-language variant; initial h- also marks negation (haina 'has not' vs
                # aina 'type'), so only ever restore a missing h, never remove one.
                reason = "h-form is the rarer one: may be a negative form (haina/aina)"
            elif rule == "double vowel" and (can in lexical or vf in lexical):
                reason = "lexical double vowel"
            elif vc < args.min_count:
                reason = f"variant seen fewer than {args.min_count} times"
            elif cc < args.ratio * vc:
                reason = f"both forms common (ratio {cc / vc:.1f} < {args.ratio:g}): may be different words"
            mined.append({"variant": vf, "canonical": can, "rule": rule, "count_variant": vc, "count_canonical": cc,
                          "accepted": "yes" if not reason else "no", "reason": reason})
    mined.sort(key=lambda r: (r["accepted"] != "yes", -r["count_variant"]))
    with (args.out_dir / "mined_variants.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mined[0]) if mined else ["variant"]); w.writeheader(); w.writerows(mined)
    acc = {r["variant"]: r["canonical"] for r in mined if r["accepted"] == "yes"}
    for k in list(acc):                      # resolve chains a->b->c to a->c
        seen_chain = {k}
        while acc[k] in acc and acc[k] not in seen_chain:
            seen_chain.add(acc[k]); acc[k] = acc[acc[k]]
    (args.out_dir / "mined_map.json").write_text(json.dumps(acc, ensure_ascii=False, indent=1), encoding="utf-8")
    repo_pairs = {(k.strip(), v.strip()) for k, v in fs.REPO_MAP.items() if k.strip() != v.strip()}
    overlap = sum((k, v) in repo_pairs for k, v in acc.items())

    print(f"Repository variant map ({sum(status.values())} entries): {dict(status)}")
    print(f"Repository compound rules ({sum(cstat.values())}): {dict(cstat)}")
    print(f"Mined from training+validation: {len(mined)} candidate pairs, {len(acc)} accepted "
          f"({overlap} of them are also in the repository map)")
    print(f"Wrote variant_audit.csv, compound_audit.csv, mined_variants.csv and mined_map.json in {args.out_dir}")


if __name__ == "__main__":
    main()
