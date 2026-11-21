"""Swahili Fair normalization with switchable steps, built from the repository's own rules.

Each step reuses the compiled patterns and lists in ndizi_asr.ethio_normalize, so with every step on
and the repository's variant map, `normalize()` returns exactly what
clean_swahili_fair_transcription(..., drop_backchannels=True) returns. `check_identical()` verifies that.

Steps, in the repository's order:
  base        NFKC, encoding repair (mojibake), lowercase, glottal apostrophe, punctuation to space
  backch      remove backchannels (eeh, mmmh, ...) and stray one-letter hesitation tokens
  elong       collapse expressive letter repeats (labdaaa) and conversational word-final double vowels
  compounds   the listed compound and split spellings (kwa sababu / kwasababu), applied before and after
  variants    the spelling-variant map (r/l, dropped h, glides, contractions)
"""
from __future__ import annotations

import re
import unicodedata

import ndizi_asr.ethio_normalize as en

STRAY = {"m", "e", "u", "o", "hm", "mm", "eh", "ah", "uh", "oh"}
REPO_MAP = {k: v for k, v in en._SWAHILI_SPELLING_MAP.items()}
REPO_COMPOUNDS = list(en._SWAHILI_COMPOUND_PATTERNS)


def normalize(text, *, backch=True, elong=True, compounds=None, variants=None) -> str:
    compounds = REPO_COMPOUNDS if compounds is None else compounds
    vmap = REPO_MAP if variants is None else variants
    t = unicodedata.normalize("NFKC", str(text or ""))
    t = en._SWAHILI_MOJIBAKE_RE.sub(" ", t)
    t = t.lower()
    t = en._GLOTTAL_RE.sub("", t)
    t = en._PLATFORM_PUNCT_RE.sub(" ", t)
    if backch:
        t = en._SWAHILI_BACKCHANNEL_RE.sub(" ", t)
    if elong:
        t = en._SWAHILI_TRIPLE_CHAR_RE.sub(r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    for pat, repl in compounds:
        t = pat.sub(repl, t)
    toks = []
    for tok in t.split():
        if backch and tok in STRAY:
            continue
        tok = vmap.get(tok, tok)
        if elong:
            tok = en._strip_conversational_trailing_double_vowel(tok)
        tok = vmap.get(tok, tok)
        toks.append(tok)
    joined = " ".join(toks)
    for pat, repl in compounds:
        joined = pat.sub(repl, joined)
    return joined


STEPS = {
    "S0 base (punctuation, encoding)": dict(backch=False, elong=False, compounds=[], variants={}),
    "S1 + backchannels": dict(backch=True, elong=False, compounds=[], variants={}),
    "S2 + elongation": dict(backch=True, elong=True, compounds=[], variants={}),
    "S3 + compounds": dict(backch=True, elong=True, compounds=None, variants={}),
    "S4 + variant map = Fair (repository)": dict(backch=True, elong=True, compounds=None, variants=None),
}


def check_identical(texts) -> int:
    """Number of texts where the full pipeline differs from the repository function (should be 0)."""
    return sum(normalize(t) != en.clean_swahili_fair_transcription(t, drop_backchannels=True) for t in texts)


def word_errors(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]
