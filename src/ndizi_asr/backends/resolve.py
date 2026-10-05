from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

LOCAL_WHISPER_PREFIXES = (
    "Sunbird/asr-whisper",
    "openai/whisper-",
    "distil-whisper/",
    "smutuvi/ndizi-whisper",
)

LOCAL_CTC_PREFIXES = (
    "badrex/Ethio-ASR",
    "badrex/w2v-bert",
    "facebook/mms-1b",
    "facebook/wav2vec2",
)

_CTC_CONFIG_TYPES = frozenset(
    {
        "wav2vec2",
        "wav2vec2_bert",
        "wav2vec2-bert",
        "hubert",
        "data2vec-audio",
        "data2vec_audio",
        "unispeech",
        "unispeech-sat",
        "sew",
        "sew-d",
        "mms",
    }
)
_WHISPER_CONFIG_TYPES = frozenset({"whisper"})
_GEMMA_CONFIG_TYPES = frozenset({"gemma3", "gemma2", "gemma", "paligemma"})


def _peek_hf_model_type(model_id: str) -> Optional[str]:
    """Read model_type from a local checkpoint config.json if present."""
    path = Path(model_id).expanduser()
    cfg_path = path / "config.json" if path.is_dir() else None
    if cfg_path is None or not cfg_path.is_file():
        return None
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Could not read %s: %s", cfg_path, exc)
        return None
    mt = data.get("model_type") or data.get("architectures")
    if isinstance(mt, list) and mt:
        # e.g. ["Wav2Vec2ForCTC"] ? wav2vec2
        name = str(mt[0]).lower()
        if "whisper" in name:
            return "whisper"
        if "wav2vec2" in name or "mms" in name:
            return "wav2vec2"
        if "gemma" in name:
            return "gemma3"
        return name
    if isinstance(mt, str):
        return mt.lower().strip()
    return None


def _backend_from_local_config(model_id: str) -> Optional[str]:
    mt = _peek_hf_model_type(model_id)
    if not mt:
        return None
    if mt in _WHISPER_CONFIG_TYPES or "whisper" in mt:
        return "hf_whisper"
    if mt in _CTC_CONFIG_TYPES or "wav2vec" in mt or mt.startswith("mms"):
        return "hf_ctc"
    if mt in _GEMMA_CONFIG_TYPES or "gemma" in mt:
        return "hf_gemma"
    return None


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
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "gemini-3.5-flash": "google/gemini-3.5-flash",
    "gemini-3.6-flash": "google/gemini-3.6-flash",
    "gemini-3.8-flash": "google/gemini-3.8-flash",
    "gemini-3-flash": "google/gemini-3.5-flash",
    "gemini": "google/gemini-2.5-flash",
    "gemini-flash": "google/gemini-2.5-flash",
    "gemini-pro": "google/gemini-2.5-pro",
    "whisper": "openai/whisper-large-v3",
    "whisper-large-v3": "openai/whisper-large-v3",
    "ethio-asr-amharic": "badrex/Ethio-ASR-amharic",
    "ethio-asr-oromo": "badrex/Ethio-ASR-oromo",
    "ethio-asr-multilingual-1b": "badrex/Ethio-ASR-multilingual-1B",
    "sunbird-whisper-salt": "Sunbird/asr-whisper-large-v3-salt",
    "sunbird-whisper-51": "Sunbird/asr-whisper-51-african-languages",
    # Swahili / Ndizi
    "ndizi-whisper-turbo": "smutuvi/ndizi-whisper-large-v3-turbo-optimized",
    "ndizi-whisper": "smutuvi/ndizi-whisper-large-v3-turbo-optimized",
    "w2v-bert-swahili": "badrex/w2v-bert-2.0-swahili-asr",
    "swahili-asr": "badrex/w2v-bert-2.0-swahili-asr",
    "omnilingualasr": "omniASR_LLM_7B",
    "omnilingualasr-ctc": "omniASR_CTC_1B",
}

# Optional language filter for monolingual Ethio models (canonical short codes).
LANGUAGE_FILTERS: dict[str, str] = {
    "badrex/Ethio-ASR-amharic": "am",
    "badrex/Ethio-ASR-oromo": "om",
    "ethio-asr-amharic": "am",
    "ethio-asr-oromo": "om",
    "badrex/w2v-bert-2.0-swahili-asr": "sw",
    "w2v-bert-swahili": "sw",
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
    "swahili": [
        "ndizi-whisper-turbo",
        "w2v-bert-swahili",
        "sunbird-whisper-51",
        "gemini-2.5-flash",
    ],
    "w2v-bert": ["ethio-asr-amharic", "ethio-asr-oromo"],
    "w2v-bert-family": ["ethio-asr-amharic", "ethio-asr-oromo"],
    "whisper-family": ["whisper", "sunbird-whisper-salt", "sunbird-whisper-51", "ndizi-whisper-turbo"],
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


def _is_whisper_model(model_id: str) -> bool:
    local = _backend_from_local_config(model_id)
    if local == "hf_whisper":
        return True
    if local is not None:
        return False
    mid = model_id.lower()
    return (
        model_id.startswith(LOCAL_WHISPER_PREFIXES)
        or "/whisper" in mid
        or "ndizi-whisper" in mid
    )


def _is_ctc_model(model_id: str) -> bool:
    local = _backend_from_local_config(model_id)
    if local == "hf_ctc":
        return True
    if local is not None:
        return False
    if model_id.startswith(LOCAL_CTC_PREFIXES):
        return True
    mid = model_id.lower()
    return (
        "ethio-asr" in mid
        or "ethio1b" in mid
        or "/wav2vec2" in mid
        or "/mms-" in mid
        or "w2v-bert" in mid
        or "sagalee_continued" in mid
    )


def _is_gemma_model(model_id: str) -> bool:
    local = _backend_from_local_config(model_id)
    if local == "hf_gemma":
        return True
    if local is not None:
        return False
    mid = model_id.lower()
    if "gemma-4" in mid or "gemma4" in mid:
        return True
    if mid.startswith("sunbird/sunflower-gemma"):
        return True
    return False


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
        # Prefer local config.json when loading a fine-tuned checkpoint dir
        local_backend = _backend_from_local_config(model_id)
        if local_backend:
            backend = local_backend
        elif _is_gemma_model(model_id):
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
