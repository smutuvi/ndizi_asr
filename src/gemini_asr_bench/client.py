from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import requests
import soundfile as sf

from .config import OpenRouterConfig

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    text: str
    latency_seconds: float
    model: str
    raw_response: Optional[dict[str, Any]] = None
    error: Optional[str] = None


def audio_array_to_wav_base64(array: np.ndarray, sampling_rate: int) -> tuple[str, str]:
    """Encode a float/int waveform as base64 WAV for OpenRouter input_audio."""
    arr = np.asarray(array)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=-1)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, arr, int(sampling_rate), format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii"), "wav"


class OpenRouterASRClient:
    """Chat-completions client that sends audio to multimodal Gemini models."""

    def __init__(
        self,
        api_key: str,
        config: Optional[OpenRouterConfig] = None,
        site_url: str = "https://github.com/smutuvi/ndizi_mlops_gemini",
        app_name: str = "ndizi_mlops_gemini",
    ) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        self.api_key = api_key
        self.config = config or OpenRouterConfig()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": site_url,
                "X-Title": app_name,
            }
        )

    def transcribe(
        self,
        *,
        model: str,
        audio_b64: str,
        audio_format: str,
        prompt: str,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        text_prompt = prompt
        if language:
            text_prompt = f"{prompt}\nLanguage: {language}."

        payload: dict[str, Any] = {
            "model": model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": audio_format,
                            },
                        },
                    ],
                }
            ],
        }

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        last_error: Optional[str] = None
        started = time.perf_counter()

        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = self.session.post(
                    url,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                if resp.status_code in {429, 500, 502, 503, 504}:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    sleep_for = self.config.retry_backoff_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "Retryable OpenRouter error (attempt %d/%d): %s; sleeping %.1fs",
                        attempt,
                        self.config.max_retries,
                        last_error,
                        sleep_for,
                    )
                    time.sleep(sleep_for)
                    continue

                resp.raise_for_status()
                data = resp.json()
                text = self._extract_text(data)
                return TranscriptionResult(
                    text=text,
                    latency_seconds=time.perf_counter() - started,
                    model=model,
                    raw_response=data,
                )
            except requests.RequestException as exc:
                last_error = str(exc)
                sleep_for = self.config.retry_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "OpenRouter request failed (attempt %d/%d): %s; sleeping %.1fs",
                    attempt,
                    self.config.max_retries,
                    last_error,
                    sleep_for,
                )
                time.sleep(sleep_for)

        return TranscriptionResult(
            text="",
            latency_seconds=time.perf_counter() - started,
            model=model,
            error=last_error or "unknown error",
        )

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif isinstance(item, str):
                    parts.append(item)
            return " ".join(p.strip() for p in parts if p).strip()
        return ""
