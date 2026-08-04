from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from .sunbird_langs import LANGUAGE_TOKENS_WHISPER, resolve_sunbird_language_code
from .types import TranscriptionResult

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# Whisper encoder receptive field (~30s). Chunk longer audio with overlap-free joins.
MAX_SECONDS = 30.0


def _resample_linear(array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return np.asarray(array, dtype=np.float32)
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    n_out = int(round(len(arr) * target_sr / float(orig_sr)))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(arr), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, arr).astype(np.float32)


def _chunk_audio(array: np.ndarray, sampling_rate: int, max_seconds: float = MAX_SECONDS) -> list[np.ndarray]:
    max_samples = int(max_seconds * sampling_rate)
    if len(array) <= max_samples:
        return [array]
    chunks = []
    for start in range(0, len(array), max_samples):
        chunk = array[start : start + max_samples]
        if len(chunk) > 0:
            chunks.append(chunk)
    return chunks


class HuggingFaceWhisperClient:
    """Local Whisper ASR via transformers (supports Sunbird African Whisper)."""

    def __init__(
        self,
        model_id: str,
        *,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        num_beams: int = 1,
    ) -> None:
        try:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError as exc:
            raise ImportError(
                "Local Whisper backends require torch and transformers. "
                "Install with: pip install -e '.[local]'"
            ) from exc

        self.model_id = model_id
        self.num_beams = num_beams
        self.torch = torch

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        dtype_map = {
            None: None,
            "auto": None,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if torch_dtype not in dtype_map:
            raise ValueError(f"Unsupported torch_dtype={torch_dtype!r}")
        resolved_dtype = dtype_map[torch_dtype]
        if resolved_dtype is None and self.device.type == "cuda":
            resolved_dtype = torch.float16
        if resolved_dtype is None:
            resolved_dtype = torch.float32

        logger.info("Loading Whisper model %s on %s (%s)", model_id, self.device, resolved_dtype)
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=resolved_dtype,
        ).to(self.device)
        self.model.eval()

        self._transcribe_tok = self.processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        self._notimestamps_tok = self.processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")

    def transcribe_array(
        self,
        audio_array: np.ndarray,
        sampling_rate: int,
        *,
        language: Optional[str] = None,
        language_config: Optional[str] = None,
    ) -> TranscriptionResult:
        started = time.perf_counter()
        try:
            lang_code = resolve_sunbird_language_code(language, language_config)
            lang_tok = LANGUAGE_TOKENS_WHISPER[lang_code]
            audio = _resample_linear(np.asarray(audio_array), int(sampling_rate), SAMPLE_RATE)
            if audio.ndim > 1:
                audio = audio.mean(axis=-1)
            texts: list[str] = []
            for chunk in _chunk_audio(audio, SAMPLE_RATE):
                texts.append(self._transcribe_chunk(chunk, lang_tok))
            text = " ".join(t.strip() for t in texts if t.strip()).strip()
            return TranscriptionResult(
                text=text,
                latency_seconds=time.perf_counter() - started,
                model=self.model_id,
            )
        except Exception as exc:  # noqa: BLE001 - surface as row error
            logger.exception("Sunbird/Whisper transcription failed")
            return TranscriptionResult(
                text="",
                latency_seconds=time.perf_counter() - started,
                model=self.model_id,
                error=str(exc),
            )

    def _transcribe_chunk(self, audio: np.ndarray, lang_tok: int) -> str:
        torch = self.torch
        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            do_normalize=True,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device, dtype=self.model.dtype)
        forced_decoder_ids = [
            (1, lang_tok),
            (2, self._transcribe_tok),
            (3, self._notimestamps_tok),
        ]
        with torch.inference_mode():
            predicted_ids = self.model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                num_beams=self.num_beams,
                do_sample=False,
            )
        texts = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return texts[0] if texts else ""
