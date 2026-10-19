from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests
import soundfile as sf

from .backends.types import TranscriptionResult
from .config import OpenRouterConfig

logger = logging.getLogger(__name__)


_CROPCONCEPT_ORTHOGRAPHY = """
Orthography and style (Afaan Oromo / Qubee):
- Write every syllable you hear; do not truncate or drop word onsets (e.g. prefer full forms like "mijataa" over fragments like "jau").
- Prefer standard long vowels where the speaker lengthens (aa, ee, ii, oo, uu) and write geminate consonants when clearly doubled (bb, dd, gg, mm, nn, tt).
- Keep mid-word hudhaa as an apostrophe only when it is a clear glottal break (e.g. re'ee); otherwise omit it.
- Treat short fillers consistently: "ee", "eshi", "nu", "yoo" (do not invent long variants like "nuuf"/"eshii"/"yaauu" unless clearly distinct).
- Prefer common crop/farm spellings when spoken: qamadii (wheat), oomisha (yield/produce), sanyii (seed), oduu (news/info), deebbala (increase), waggaa (year), qulqulluu (clean/pure), akkaataa (way/manner).
- Keep numbers and units as spoken; do not summarize or paraphrase.
""".strip()

CROPCONCEPT_PROMPT_OROMO = f"""Transcribe the speech in this audio accurately and verbatim, in the language actually spoken. Do not translate. Keep each language's own script. If the speaker switches languages, transcribe each part as spoken.

Likely language: Afaan Oromo (om). This form is also used in Afarigna (aa), Amharic (am), English (en), Hadiya (hdy).

{_CROPCONCEPT_ORTHOGRAPHY}

Respond with a single JSON object and nothing else:
{{"language": "<code of the language mainly spoken>", "text": "<the transcript>"}}
Use one of these codes when it applies: om, aa, am, en, hdy. If there is no intelligible speech, respond with {{"language": null, "text": ""}}."""

CROPCONCEPT_PROMPT_ENGLISH = f"""Transcribe the speech in this audio accurately and verbatim, in the language actually spoken. Do not translate. Keep each language's own script. If the speaker switches languages, transcribe each part as spoken.

Likely language: English (en). This form is also used in Afaan Oromo (om), Afarigna (aa), Amharic (am), Hadiya (hdy).

When any stretch is Afaan Oromo, follow these rules:
{_CROPCONCEPT_ORTHOGRAPHY}

Respond with a single JSON object and nothing else:
{{"language": "<code of the language mainly spoken>", "text": "<the transcript>"}}
Use one of these codes when it applies: en, om, aa, am, hdy. If there is no intelligible speech, respond with {{"language": null, "text": ""}}."""


def _extract_json_transcript(raw_text: str) -> str:
    """Parse {"language": ..., "text": "..."} from CropConcept JSON response."""
    s = (raw_text or "").strip()
    if not s:
        return ""
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s, re.IGNORECASE)
    candidate = fence.group(1) if fence else s
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict) and "text" in obj:
            return str(obj.get("text") or "").strip()
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "text" in obj:
                return str(obj.get("text") or "").strip()
        except Exception:
            pass
    return s


def read_cropconcept_hint_language(audio_path: str | Path) -> str:
    """Read language.hint_given_to_transcriber from sidecar .json if present."""
    p = Path(audio_path)
    sidecar = p.with_suffix(".json")
    if sidecar.is_file():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            hint = str((data.get("language") or {}).get("hint_given_to_transcriber") or "").strip().lower()
            if hint in {"en", "om"}:
                return hint
        except Exception:
            pass
    return "om"


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

    def transcribe_cropconcept_recipe(
        self,
        *,
        model: str,
        audio_path: str | Path,
        hint_language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Exact CropConcept Ethiopia recipe: raw file bytes as format='mp4', audio first, JSON output."""
        path = Path(audio_path)
        audio_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        hint = (hint_language or read_cropconcept_hint_language(path)).lower()
        prompt = CROPCONCEPT_PROMPT_ENGLISH if hint == "en" else CROPCONCEPT_PROMPT_OROMO

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": "mp4",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.0,
        }

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        last_error: Optional[str] = None
        started = time.perf_counter()

        for attempt in range(1, self.config.max_retries + 1):
            req_payload = dict(payload)
            if attempt > 1:
                req_payload["reasoning"] = {"max_tokens": 1024}
            try:
                resp = self.session.post(
                    url,
                    json=req_payload,
                    timeout=max(self.config.timeout_seconds, 300.0),
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
                choices = data.get("choices") or []
                msg = (choices[0].get("message") or {}) if choices else {}
                raw_content = msg.get("content")
                if isinstance(raw_content, list):
                    raw_text = " ".join(
                        str(item.get("text") or "")
                        for item in raw_content
                        if isinstance(item, dict)
                    ).strip()
                else:
                    raw_text = str(raw_content or "").strip()

                if not raw_text or raw_text.startswith("**"):
                    last_error = "Empty content or leaked reasoning in response"
                    sleep_for = self.config.retry_backoff_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "Empty/reasoning-only response on %s (attempt %d/%d); retrying in %.1fs",
                        path.name,
                        attempt,
                        self.config.max_retries,
                        sleep_for,
                    )
                    time.sleep(sleep_for)
                    continue

                text = _extract_json_transcript(raw_text)
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

    def chat_text(
        self,
        *,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        reasoning: Optional[dict[str, Any]] = None,
    ) -> TranscriptionResult:
        """Text-only chat completion (LLM fusion / post-edit)."""
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._chat_completions(model=model, messages=messages, reasoning=reasoning)

    def chat_with_audio(
        self,
        *,
        model: str,
        prompt: str,
        audio_b64: str,
        audio_format: str = "wav",
        system: Optional[str] = None,
        reasoning: Optional[dict[str, Any]] = None,
    ) -> TranscriptionResult:
        """Multimodal chat: text prompt + input_audio (audio+hyp fusion)."""
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": audio_format},
                    },
                ],
            }
        )
        return self._chat_completions(model=model, messages=messages, reasoning=reasoning)

    def _chat_completions(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        reasoning: Optional[dict[str, Any]] = None,
    ) -> TranscriptionResult:
        payload: dict[str, Any] = {
            "model": model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": messages,
        }
        if reasoning:
            payload["reasoning"] = reasoning
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        last_error: Optional[str] = None
        started = time.perf_counter()
        # Thinking models can be slow
        timeout = self.config.timeout_seconds
        if reasoning:
            timeout = max(timeout, 300.0)

        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = self.session.post(
                    url,
                    json=payload,
                    timeout=timeout,
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

        def _from_content(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, list):
                parts: list[str] = []
                for item in value:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        if item.get("type") in {"text", "output_text"}:
                            parts.append(str(item.get("text") or ""))
                        elif "text" in item:
                            parts.append(str(item.get("text") or ""))
                return " ".join(p.strip() for p in parts if p).strip()
            return str(value).strip()

        text = _from_content(content)
        if text:
            return text
        # Some GPT-6 / OpenRouter shapes put usable text elsewhere
        for key in ("reasoning", "reasoning_content", "refusal"):
            alt = _from_content(message.get(key))
            if alt and key != "refusal":
                return alt
        # OpenRouter sometimes mirrors assistant text at top-level
        for key in ("output_text", "text"):
            alt = _from_content(data.get(key))
            if alt:
                return alt
        return ""
