"""Helpers to expand CLI dataset flags into loader sources."""

from __future__ import annotations

from typing import Any, Optional

FLEURS_DEFAULT_LANGUAGES = {
    "swahili": "sw_ke",
    "amharic": "am_et",
    "oromo": "om_et",
    "hausa": "ha_ng",
}

# Known Hub datasets with sensible ASR column defaults.
DATASET_PRESETS: dict[str, dict[str, Any]] = {
    "smutuvi/ndizi-1": {
        "language": "swahili",
        "audio_column": "audio",
        "text_column": "text",
        "split": "test",
    },
    "smutuvi/ndizi-1-2025": {
        "language": "swahili",
        "audio_column": "audio",
        "text_column": "text",
        "split": "test",
    },
    "google/fleurs": {
        "audio_column": "audio",
        "text_column": "transcription",
        "split": "test",
        "expand_fleurs_languages": True,
    },
    "fleurs": {
        "dataset_id": "google/fleurs",
        "audio_column": "audio",
        "text_column": "transcription",
        "split": "test",
        "expand_fleurs_languages": True,
    },
}


def _parse_dataset_token(token: str) -> tuple[str, Optional[str], Optional[str]]:
    """Parse DATASET or DATASET:SPLIT or DATASET:CONFIG:SPLIT."""
    parts = [p.strip() for p in token.split(":") if p.strip()]
    if not parts:
        raise ValueError(f"Empty dataset token: {token!r}")
    if len(parts) == 1:
        return parts[0], None, None
    if len(parts) == 2:
        # Could be split or config; treat as split for simple datasets,
        # and as config for fleurs-style ids when it looks like xx_yy.
        second = parts[1]
        if "_" in second and len(second) <= 8:
            return parts[0], second, None
        return parts[0], None, second
    return parts[0], parts[1], parts[2]


def expand_dataset_flags(
    tokens: list[str],
    *,
    default_split: str = "test",
    default_language: str = "swahili",
    languages: Optional[dict[str, str]] = None,
    audio_column: Optional[str] = None,
    text_column: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Expand CLI --dataset tokens into dataset.sources entries."""
    sources: list[dict[str, Any]] = []
    fleurs_langs = languages or FLEURS_DEFAULT_LANGUAGES

    for token in tokens:
        dataset_id, config_name, split = _parse_dataset_token(token)
        preset_key = dataset_id
        preset = dict(DATASET_PRESETS.get(preset_key) or DATASET_PRESETS.get(dataset_id.lower()) or {})
        if "dataset_id" in preset:
            dataset_id = str(preset["dataset_id"])

        resolved_split = split or str(preset.get("split") or default_split)
        resolved_audio = audio_column or preset.get("audio_column")
        resolved_text = text_column or preset.get("text_column")
        resolved_lang = str(preset.get("language") or default_language)

        if preset.get("expand_fleurs_languages") and not config_name:
            for lang_name, fleurs_cfg in fleurs_langs.items():
                sources.append(
                    {
                        "dataset_id": dataset_id,
                        "config_name": fleurs_cfg,
                        "split": resolved_split,
                        "language": lang_name,
                        "audio_column": resolved_audio or "audio",
                        "text_column": resolved_text or "transcription",
                        "max_samples": max_samples,
                    }
                )
            continue

        entry: dict[str, Any] = {
            "dataset_id": dataset_id,
            "split": resolved_split,
            "language": resolved_lang,
            "max_samples": max_samples,
        }
        if config_name:
            entry["config_name"] = config_name
            # If user passed fleurs config, map language name when known
            for lang_name, fleurs_cfg in fleurs_langs.items():
                if fleurs_cfg == config_name:
                    entry["language"] = lang_name
                    break
        if resolved_audio:
            entry["audio_column"] = resolved_audio
        if resolved_text:
            entry["text_column"] = resolved_text
        sources.append(entry)

    return sources
