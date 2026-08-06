from __future__ import annotations

from dataclasses import dataclass


LOCAL_WHISPER_PREFIXES = (
    "Sunbird/asr-whisper",
    "openai/whisper-",
    "distil-whisper/",
)

MODEL_ALIASES: dict[str, str] = {
    # Google Gemma 4 E2B (instruction-tuned multimodal / audio)
    "google/gemma-4-e2b": "google/gemma-4-E2B-it",
    "google/gemma-4-E2B": "google/gemma-4-E2B-it",
    "google/gemma-4-e2b-it": "google/gemma-4-E2B-it",
    "gemma-4-e2b": "google/gemma-4-E2B-it",
    "gemma4-e2b": "google/gemma-4-E2B-it",
    # Sunbird Gemma 4 E2B (Sunflower multimodal ASR/chat)
    "sunbird/gemma-4-e2b": "Sunbird/Sunflower-Gemma4-E2B",
    "Sunbird/gemma-4-e2b": "Sunbird/Sunflower-Gemma4-E2B",
    "sunbird/sunflower-gemma4-e2b": "Sunbird/Sunflower-Gemma4-E2B",
    "Sunbird/sunflower-gemma4-e2b": "Sunbird/Sunflower-Gemma4-E2B",
    "sunflower-gemma4-e2b": "Sunbird/Sunflower-Gemma4-E2B",
}


@dataclass(frozen=True)
class ModelSpec:
    raw: str
    backend: str  # openrouter | hf_whisper | hf_gemma
    model_id: str


def canonicalize_model_id(model: str) -> str:
    raw = model.strip()
    if raw.lower().startswith(("openrouter:", "hf:", "hf_whisper:", "hf_gemma:", "gemma:")):
        prefix, rest = raw.split(":", 1)
        rest = rest.strip()
        return MODEL_ALIASES.get(rest, MODEL_ALIASES.get(rest.lower(), rest))
    return MODEL_ALIASES.get(raw, MODEL_ALIASES.get(raw.lower(), raw))


def _is_gemma_model(model_id: str) -> bool:
    mid = model_id.lower()
    if "gemma-4" in mid or "gemma4" in mid:
        return True
    if mid.startswith("sunbird/sunflower-gemma"):
        return True
    return False


def _is_whisper_model(model_id: str) -> bool:
    return model_id.startswith(LOCAL_WHISPER_PREFIXES) or "/whisper" in model_id.lower()


def parse_model_spec(model: str) -> ModelSpec:
    raw = model.strip()
    if not raw:
        raise ValueError("Empty model id")

    lower = raw.lower()
    if lower.startswith("openrouter:"):
        model_id = canonicalize_model_id("openrouter:" + raw.split(":", 1)[1])
        return ModelSpec(raw=raw, backend="openrouter", model_id=model_id)

    if lower.startswith(("hf_gemma:", "gemma:")):
        model_id = canonicalize_model_id(raw)
        return ModelSpec(raw=raw, backend="hf_gemma", model_id=model_id)

    if lower.startswith(("hf:", "hf_whisper:")):
        model_id = canonicalize_model_id(raw)
        backend = "hf_gemma" if _is_gemma_model(model_id) else "hf_whisper"
        return ModelSpec(raw=raw, backend=backend, model_id=model_id)

    model_id = canonicalize_model_id(raw)
    if _is_whisper_model(model_id):
        return ModelSpec(raw=raw, backend="hf_whisper", model_id=model_id)
    if _is_gemma_model(model_id):
        return ModelSpec(raw=raw, backend="hf_gemma", model_id=model_id)

    # Default: OpenRouter chat multimodal models (e.g. google/gemini-*)
    return ModelSpec(raw=raw, backend="openrouter", model_id=model_id)


def needs_openrouter(models: list[str]) -> bool:
    return any(parse_model_spec(m).backend == "openrouter" for m in models)


def needs_hf_auth(models: list[str]) -> bool:
    return any(parse_model_spec(m).backend in {"hf_whisper", "hf_gemma"} for m in models)
