from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


LOCAL_HF_PREFIXES = (
    "hf:",
    "Sunbird/",
    "openai/whisper-",
    "distil-whisper/",
)


@dataclass(frozen=True)
class ModelSpec:
    raw: str
    backend: str  # openrouter | hf_whisper
    model_id: str


def parse_model_spec(model: str) -> ModelSpec:
    raw = model.strip()
    if not raw:
        raise ValueError("Empty model id")

    lower = raw.lower()
    if lower.startswith("openrouter:"):
        return ModelSpec(raw=raw, backend="openrouter", model_id=raw.split(":", 1)[1].strip())
    if lower.startswith("hf:"):
        return ModelSpec(raw=raw, backend="hf_whisper", model_id=raw.split(":", 1)[1].strip())

    if raw.startswith(LOCAL_HF_PREFIXES) or raw.startswith("Sunbird/"):
        return ModelSpec(raw=raw, backend="hf_whisper", model_id=raw)

    # Default: OpenRouter chat multimodal models (e.g. google/gemini-*)
    return ModelSpec(raw=raw, backend="openrouter", model_id=raw)


def needs_openrouter(models: list[str]) -> bool:
    return any(parse_model_spec(m).backend == "openrouter" for m in models)
