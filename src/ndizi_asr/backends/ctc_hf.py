"""HuggingFace CTC backends (w2v-bert / Ethio-ASR / MMS). Ported from ndizi_eval."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional

import numpy as np

from ..audio import join_chunk_transcripts, maybe_chunk
from .types import TranscriptionResult

logger = logging.getLogger(__name__)

MAX_CHUNK_S = 60.0
_LID_PREFIX = re.compile(r"^\[[A-Za-z]{2,8}\]\|")


def _hf_token() -> Optional[str]:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return None


def _strip_lid_prefix(text: str) -> str:
    return _LID_PREFIX.sub("", text).strip()


class HuggingFaceCTCClient:
    """Local CTC ASR via transformers (Ethio-ASR, wav2vec2, MMS, …)."""

    def __init__(
        self,
        model_id: str,
        *,
        device: Optional[str] = None,
        max_chunk_s: float = MAX_CHUNK_S,
    ) -> None:
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        self.model_id = model_id
        self.max_chunk_s = max_chunk_s
        token = _hf_token()
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading CTC model %s on %s", model_id, self.device)

        kwargs = {"token": token} if token else {}
        self.processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        self.model = AutoModelForCTC.from_pretrained(model_id, **kwargs).to(self.device)
        self.model.eval()
        self.input_key = self._probe_input_key()

    def _probe_input_key(self) -> str:
        dummy = np.zeros(16000, dtype=np.float32)
        out = self.processor(dummy, sampling_rate=16000, return_tensors="pt")
        for k in ("input_features", "input_values"):
            if k in out:
                return k
        raise RuntimeError(
            f"Processor for {self.model_id} has neither input_features nor input_values"
        )

    def _decode_chunk(self, waveform: np.ndarray) -> str:
        import torch

        inputs = self.processor(
            waveform.astype(np.float32),
            sampling_rate=16000,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(pred_ids)[0]
        return _strip_lid_prefix(str(text).strip())

    def transcribe_array(
        self,
        array: np.ndarray,
        sampling_rate: int,
        *,
        language: Optional[str] = None,
        language_config: Optional[str] = None,
    ) -> TranscriptionResult:
        del language, language_config
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
            chunks, _ = maybe_chunk(wav, sampling_rate, self.max_chunk_s)
            text = join_chunk_transcripts([self._decode_chunk(c) for c in chunks])
            return TranscriptionResult(
                text=text,
                latency_seconds=time.perf_counter() - t0,
                model=self.model_id,
            )
        except Exception as exc:  # noqa: BLE001
            return TranscriptionResult(
                text="",
                latency_seconds=time.perf_counter() - t0,
                model=self.model_id,
                error=str(exc),
            )

    def close(self) -> None:
        import torch

        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
