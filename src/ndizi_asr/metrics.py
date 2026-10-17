from __future__ import annotations

import re
import string
from typing import Iterable, Optional

from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer

from .config import NormalizationConfig
from .ethio_normalize import clean_ethio_transcription, clean_oromo_fair_transcription


_PUNCT_TABLE = str.maketrans("", "", string.punctuation + "።፣፤፥፦፧፨።،؛؟«»\"'„“”‘’…–—")


def normalize_text(
    text: str,
    cfg: NormalizationConfig,
    *,
    language: Optional[str] = None,
) -> str:
    mode = (cfg.mode or "default").lower()
    if mode in {"none", "raw", "off"}:
        return text or ""
    if mode in {"ethio", "ethiopia", "geez"}:
        return clean_ethio_transcription(text, language=language)
    if mode in {"oromo_fair", "oromo-fair", "qubee"}:
        return clean_oromo_fair_transcription(text)

    out = text or ""
    if cfg.lowercase:
        out = out.lower()
    if cfg.strip_punctuation:
        out = out.translate(_PUNCT_TABLE)
    if cfg.collapse_whitespace:
        out = re.sub(r"\s+", " ", out).strip()
    else:
        out = out.strip()
    return out


def compute_wer(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    refs = list(references)
    hyps = list(hypotheses)
    if not refs:
        return 0.0
    return float(jiwer_wer(refs, hyps))


def compute_cer(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    refs = list(references)
    hyps = list(hypotheses)
    if not refs:
        return 0.0
    return float(jiwer_cer(refs, hyps))


def summarize_predictions(
    rows: list[dict],
    normalization: NormalizationConfig,
) -> dict:
    """Compute pooled WER/CER on successful predictions only."""
    ok = [r for r in rows if not r.get("error") and r.get("prediction") is not None]
    if not ok:
        return {
            "num_samples": len(rows),
            "num_scored": 0,
            "num_errors": len(rows),
            "wer": None,
            "cer": None,
            "wer_normalized": None,
            "cer_normalized": None,
            "mean_latency_seconds": None,
        }

    refs = [str(r.get("reference") or "") for r in ok]
    hyps = [str(r.get("prediction") or "") for r in ok]
    refs_n = [
        normalize_text(
            x,
            normalization,
            language=str(r.get("language") or r.get("language_config") or ""),
        )
        for x, r in zip(refs, ok)
    ]
    hyps_n = [
        normalize_text(
            x,
            normalization,
            language=str(r.get("language") or r.get("language_config") or ""),
        )
        for x, r in zip(hyps, ok)
    ]
    latencies = [float(r["latency_seconds"]) for r in ok if r.get("latency_seconds") is not None]

    return {
        "num_samples": len(rows),
        "num_scored": len(ok),
        "num_errors": len(rows) - len(ok),
        "wer": compute_wer(refs, hyps),
        "cer": compute_cer(refs, hyps),
        "wer_normalized": compute_wer(refs_n, hyps_n),
        "cer_normalized": compute_cer(refs_n, hyps_n),
        "mean_latency_seconds": (sum(latencies) / len(latencies)) if latencies else None,
    }
