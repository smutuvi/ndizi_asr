from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


LOCAL_WHISPER_PREFIXES = (
    "Sunbird/asr-whisper",
    "openai/whisper-",
    "distil-whisper/",
)

LOCAL_CTC_PREFIXES = (
    "badrex/Ethio-ASR",
    "facebook/mms-1b",
    "facebook/wav2vec2",
)

MODEL_ALIASES: dict[str, str] = {
    # Google Gemma 4 E2B
    "google/gemma-4-e2b": "google/gemma-4-E2B-it",
    "google/gemma-4-E2B": "google/gemma-4-E2B-it",
    "google/gemma-4-e2b-it": "google/gemma-4-E2B-it",
    "gemma-4-e2b": "google/gemma-4-E2B-it",
    "gemma4-e2b": "google/gemma-4-E2B-it",
    # Sunbird Gemma 4
    "sunbird/gemma-4-e2b": "Sunbird/Sunflower-Gemma4-E2B",
    "Sunbird/gemma-4-e2b": "Sunbird/Sunflower-Gemma4-E2B",
    "sunbird/sunflower-gemma4-e2b": "Sunbird/Sunflower-Gemma4-E2B",
    "Sunbird/sunflower-gemma4-e2b": "Sunbird/Sunflower-Gemma4-E2B",
    "sunflower-gemma4-e2b": "Sunbird/Sunflower-Gemma4-E2B",
    # Ethiopia / CropConcept short keys (from ndizi_eval)
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "whisper": "openai/whisper-large-v3",
    "whisper-large-v3": "openai/whisper-large-v3",
    "ethio-asr-amharic": "badrex/Ethio-ASR-amharic",
    "ethio-asr-oromo": "badrex/Ethio-ASR-oromo",
    "ethio-asr-multilingual-1b": "badrex/Ethio-ASR-multilingual-1B",
    "sunbird-whisper-salt": "Sunbird/asr-whisper-large-v3-salt",
    "sunbird-whisper-51": "Sunbird/asr-whisper-51-african-languages",
    "omnilingualasr": "omniASR_LLM_7B",
    "omnilingualasr-ctc": "omniASR_CTC_1B",
}

# Optional language filter for monolingual Ethio models (canonical short codes).
LANGUAGE_FILTERS: dict[str, str] = {
    "badrex/Ethio-ASR-amharic": "am",
    "badrex/Ethio-ASR-oromo": "om",
    "ethio-asr-amharic": "am",
    "ethio-asr-oromo": "om",
}

MODEL_GROUPS: dict[str, list[str]] = {
    "ethio": [
        "ethio-asr-amharic",
        "ethio-asr-oromo",
        "ethio-asr-multilingual-1b",
    ],
    "oromo": [
        "gemini-2.5-flash",
        "omnilingualasr-ctc",
        "ethio-asr-oromo",
        "ethio-asr-multilingual-1b",
        "sunbird-whisper-51",
    ],
    "w2v-bert": ["ethio-asr-amharic", "ethio-asr-oromo"],
    "w2v-bert-family": ["ethio-asr-amharic", "ethio-asr-oromo"],
    "whisper-family": ["whisper", "sunbird-whisper-salt", "sunbird-whisper-51"],
    "local-ethio": [
        "omnilingualasr-ctc",
        "ethio-asr-amharic",
        "ethio-asr-oromo",
        "whisper",
        "sunbird-whisper-51",
    ],
}

_BACKEND_PREFIXES = (
    "openrouter:",
    "hf:",
    "hf_whisper:",
    "hf_gemma:",
    "gemma:",
    "ctc:",
    "hf_ctc:",
    "omnilingual:",
    "omni:",
    "gemini:",
)


@dataclass(frozen=True)
class ModelSpec:
    raw: str
    backend: str  # openrouter | hf_whisper | hf_gemma | hf_ctc | omnilingual
    model_id: str
    language_filter: Optional[str] = None


def canonicalize_model_id(model: str) -> str:
    raw = model.strip()
    lower = raw.lower()
    for prefix in _BACKEND_PREFIXES:
        if lower.startswith(prefix):
            rest = raw.split(":", 1)[1].strip()
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


def _is_ctc_model(model_id: str) -> bool:
    if model_id.startswith(LOCAL_CTC_PREFIXES):
        return True
    mid = model_id.lower()
    return "ethio-asr" in mid or "/wav2vec2" in mid or "/mms-" in mid


def _is_omnilingual_card(model_id: str) -> bool:
    mid = model_id.lower()
    return mid.startswith("omniasr") or mid.startswith("omnilingual")


def _language_filter_for(raw: str, model_id: str) -> Optional[str]:
    return LANGUAGE_FILTERS.get(raw) or LANGUAGE_FILTERS.get(model_id)


def parse_model_spec(model: str) -> ModelSpec:
    raw = model.strip()
    if not raw:
        raise ValueError("Empty model id")

    lower = raw.lower()
    if lower.startswith(("openrouter:", "gemini:")):
        model_id = canonicalize_model_id(raw)
        return ModelSpec(
            raw=raw,
            backend="openrouter",
            model_id=model_id,
            language_filter=_language_filter_for(raw, model_id),
        )

    if lower.startswith(("omnilingual:", "omni:")):
        model_id = canonicalize_model_id(raw)
        return ModelSpec(
            raw=raw,
            backend="omnilingual",
            model_id=model_id,
            language_filter=_language_filter_for(raw, model_id),
        )

    if lower.startswith(("ctc:", "hf_ctc:")):
        model_id = canonicalize_model_id(raw)
        return ModelSpec(
            raw=raw,
            backend="hf_ctc",
            model_id=model_id,
            language_filter=_language_filter_for(raw, model_id),
        )

    if lower.startswith(("hf_gemma:", "gemma:")):
        model_id = canonicalize_model_id(raw)
        return ModelSpec(
            raw=raw,
            backend="hf_gemma",
            model_id=model_id,
            language_filter=_language_filter_for(raw, model_id),
        )

    if lower.startswith(("hf:", "hf_whisper:")):
        model_id = canonicalize_model_id(raw)
        if _is_gemma_model(model_id):
            backend = "hf_gemma"
        elif _is_ctc_model(model_id):
            backend = "hf_ctc"
        else:
            backend = "hf_whisper"
        return ModelSpec(
            raw=raw,
            backend=backend,
            model_id=model_id,
            language_filter=_language_filter_for(raw, model_id),
        )

    model_id = canonicalize_model_id(raw)
    if _is_omnilingual_card(model_id) or lower in {"omnilingualasr", "omnilingualasr-ctc"}:
        return ModelSpec(
            raw=raw,
            backend="omnilingual",
            model_id=model_id,
            language_filter=_language_filter_for(raw, model_id),
        )
    if _is_ctc_model(model_id):
        return ModelSpec(
            raw=raw,
            backend="hf_ctc",
            model_id=model_id,
            language_filter=_language_filter_for(raw, model_id),
        )
    if _is_whisper_model(model_id):
        return ModelSpec(
            raw=raw,
            backend="hf_whisper",
            model_id=model_id,
            language_filter=_language_filter_for(raw, model_id),
        )
    if _is_gemma_model(model_id):
        return ModelSpec(
            raw=raw,
            backend="hf_gemma",
            model_id=model_id,
            language_filter=_language_filter_for(raw, model_id),
        )

    return ModelSpec(
        raw=raw,
        backend="openrouter",
        model_id=model_id,
        language_filter=_language_filter_for(raw, model_id),
    )


def expand_model_groups(models: Sequence[str]) -> list[str]:
    """Expand group names (ethio, oromo, …) into concrete model ids."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in models:
        key = raw.strip()
        if key in MODEL_GROUPS:
            for m in MODEL_GROUPS[key]:
                if m not in seen:
                    out.append(m)
                    seen.add(m)
            continue
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def list_known_models() -> str:
    lines = [
        "key | resolves to | backend",
        "--- | --- | ---",
    ]
    for key, model_id in sorted(MODEL_ALIASES.items()):
        try:
            spec = parse_model_spec(key)
            lines.append(f"{key} | {model_id} | {spec.backend}")
        except ValueError:
            lines.append(f"{key} | {model_id} | ?")
    lines.append("")
    lines.append("Groups:")
    for g, members in sorted(MODEL_GROUPS.items()):
        lines.append(f"  {g}: {', '.join(members)}")
    lines.append("")
    lines.append(
        "Prefixes: openrouter:/gemini:, hf:/hf_whisper:, hf_gemma:/gemma:, "
        "ctc:/hf_ctc:, omnilingual:/omni:"
    )
    return "\n".join(lines)


def needs_openrouter(models: list[str]) -> bool:
    return any(parse_model_spec(m).backend == "openrouter" for m in expand_model_groups(models))


def needs_hf_auth(models: list[str]) -> bool:
    return any(
        parse_model_spec(m).backend in {"hf_whisper", "hf_gemma", "hf_ctc"}
        for m in expand_model_groups(models)
    )


def needs_omnilingual(models: list[str]) -> bool:
    return any(parse_model_spec(m).backend == "omnilingual" for m in expand_model_groups(models))
