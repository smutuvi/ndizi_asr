"""Meta Omnilingual ASR backend. Ported from ndizi_eval."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..audio import join_chunk_transcripts, maybe_chunk
from .types import TranscriptionResult

logger = logging.getLogger(__name__)

MAX_CHUNK_S = 35.0

OMNI_LANG = {
    "am": "amh_Ethi",
    "amh": "amh_Ethi",
    "amharic": "amh_Ethi",
    "am_et": "amh_Ethi",
    "om": "orm_Latn",
    "orm": "orm_Latn",
    "oromo": "orm_Latn",
    "om_et": "orm_Latn",
    "sw": "swh_Latn",
    "swa": "swh_Latn",
    "swahili": "swh_Latn",
    "ha": "hau_Latn",
    "hau": "hau_Latn",
    "hausa": "hau_Latn",
}


def _maybe_add_omnilingual_root(explicit: Optional[str] = None) -> None:
    root = explicit or os.environ.get("OMNILINGUAL_ASR_ROOT")
    if root:
        root_path = Path(root).expanduser().resolve()
        if str(root_path) not in sys.path:
            sys.path.insert(0, str(root_path))


def resolve_omni_lang(language: Optional[str], language_config: Optional[str] = None) -> str:
    for candidate in (language, language_config):
        if not candidate:
            continue
        key = str(candidate).strip().lower()
        if key in OMNI_LANG:
            return OMNI_LANG[key]
        # Pass through already-qualified ids like amh_Ethi
        if "_" in key and len(key) > 3:
            return candidate
    return language or "eng_Latn"


class OmnilingualASRClient:
    """Local Meta Omnilingual ASR via ASRInferencePipeline."""

    def __init__(
        self,
        model_card: str = "omniASR_LLM_7B",
        *,
        omnilingual_root: Optional[str] = None,
        max_chunk_s: float = MAX_CHUNK_S,
    ) -> None:
        self.model_card = model_card
        self.max_chunk_s = max_chunk_s
        _maybe_add_omnilingual_root(omnilingual_root)
        try:
            from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        except ImportError as e:
            raise ImportError(
                "omnilingual-asr is not installed. Install with:\n"
                '  pip install "omnilingual-asr[data]"\n'
                "Or set OMNILINGUAL_ASR_ROOT to a local checkout.\n"
                f"Original error: {e}"
            ) from e
        logger.info("Loading Omnilingual ASR card: %s", model_card)
        self.pipeline = ASRInferencePipeline(model_card=model_card)

    def _transcribe_batch(self, waveforms: list[np.ndarray], lang: str) -> list[str]:
        items = [{"waveform": w.astype(np.float32), "sample_rate": 16000} for w in waveforms]
        try:
            return self.pipeline.transcribe(
                items, lang=[lang] * len(items), batch_size=max(1, len(items))
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Omnilingual batch failed: %s", e)
            return [""] * len(items)

    def transcribe_array(
        self,
        array: np.ndarray,
        sampling_rate: int,
        *,
        language: Optional[str] = None,
        language_config: Optional[str] = None,
    ) -> TranscriptionResult:
        t0 = time.perf_counter()
        try:
            wav = np.asarray(array, dtype=np.float32)
            if wav.ndim > 1:
                wav = wav.mean(axis=-1)
            if int(sampling_rate) != 16000:
                n_out = int(round(len(wav) * 16000 / float(sampling_rate)))
                x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
                x_new = np.linspace(0.0, 1.0, num=max(n_out, 1), endpoint=False)
                wav = np.interp(x_new, x_old, wav.astype(np.float64)).astype(np.float32)
                sampling_rate = 16000
            lang = resolve_omni_lang(language, language_config)
            chunks, _ = maybe_chunk(wav, sampling_rate, self.max_chunk_s)
            hyps = []
            for chunk in chunks:
                hyps.extend(self._transcribe_batch([chunk], lang))
            text = join_chunk_transcripts(hyps)
            return TranscriptionResult(
                text=text,
                latency_seconds=time.perf_counter() - t0,
                model=self.model_card,
            )
        except Exception as exc:  # noqa: BLE001
            return TranscriptionResult(
                text="",
                latency_seconds=time.perf_counter() - t0,
                model=self.model_card,
                error=str(exc),
            )

    def close(self) -> None:
        del self.pipeline
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
