from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np

from ..config import DatasetConfig


@dataclass
class AudioSample:
    sample_id: str
    language: str
    language_config: str
    reference: str
    audio_array: np.ndarray
    sampling_rate: int
    meta: dict[str, Any]


AUDIO_COLUMN_CANDIDATES = ["audio", "audio_path", "path", "file", "wav", "speech"]
TEXT_COLUMN_CANDIDATES = [
    "transcription",
    "raw_transcription",
    "sentence",
    "text",
    "transcript",
    "normalized_text",
    "label",
]
ID_COLUMN_CANDIDATES = ["id", "utt_id", "utterance_id", "path", "audio_id"]


def pick_column(columns: list[str], preferred: Optional[str], candidates: list[str]) -> str:
    cols_l = {c.lower(): c for c in columns}
    if preferred and preferred.lower() in cols_l:
        return cols_l[preferred.lower()]
    for cand in candidates:
        if cand.lower() in cols_l:
            return cols_l[cand.lower()]
    raise ValueError(f"Could not find column from {candidates} in {columns}")


def extract_audio(value: Any) -> tuple[np.ndarray, int]:
    if isinstance(value, dict):
        array = value.get("array")
        sr = value.get("sampling_rate") or value.get("sample_rate")
        if array is None or sr is None:
            path = value.get("path")
            if path:
                import soundfile as sf

                array, sr = sf.read(path)
            else:
                raise ValueError(f"Unsupported audio dict keys: {list(value.keys())}")
        return np.asarray(array), int(sr)

    if isinstance(value, str):
        import soundfile as sf

        array, sr = sf.read(value)
        return np.asarray(array), int(sr)

    raise ValueError(f"Unsupported audio value type: {type(value)}")


class BaseDatasetLoader:
    def iter_samples(self) -> Iterator[AudioSample]:
        raise NotImplementedError


def _load_hf_split(
    dataset_id: str,
    *,
    config_name: Optional[str],
    split: str,
    streaming: bool,
    revision: Optional[str],
    trust_remote_code: bool,
    load_kwargs: dict[str, Any],
):
    from datasets import Audio, load_dataset

    kwargs: dict[str, Any] = {
        "split": split,
        "streaming": streaming,
        "trust_remote_code": trust_remote_code,
        **(load_kwargs or {}),
    }
    if revision:
        kwargs["revision"] = revision

    if config_name:
        ds = load_dataset(dataset_id, config_name, **kwargs)
    else:
        ds = load_dataset(dataset_id, **kwargs)

    # Decode audio arrays when present
    cols = list(ds.column_names) if hasattr(ds, "column_names") else []
    if "audio" in cols:
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


class HuggingFaceDatasetLoader(BaseDatasetLoader):
    """Generic loader for any Hugging Face speech dataset with audio + text columns."""

    def __init__(
        self,
        cfg: DatasetConfig,
        *,
        language_name: str = "default",
        language_config: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.language_name = language_name
        self.language_config = language_config or cfg.config_name or "default"

    def iter_samples(self) -> Iterator[AudioSample]:
        config_name = self.cfg.config_name
        if not config_name and self.language_config and self.language_config != "default":
            config_name = self.language_config

        ds = _load_hf_split(
            self.cfg.dataset_id,
            config_name=config_name,
            split=self.cfg.split,
            streaming=self.cfg.streaming,
            revision=self.cfg.revision,
            trust_remote_code=self.cfg.trust_remote_code,
            load_kwargs=self.cfg.load_kwargs,
        )

        columns = self._resolve_columns(ds)
        audio_col = pick_column(columns, self.cfg.audio_column, AUDIO_COLUMN_CANDIDATES)
        text_col = pick_column(columns, self.cfg.text_column, TEXT_COLUMN_CANDIDATES)
        id_col = None
        try:
            id_col = pick_column(columns, self.cfg.id_column, ID_COLUMN_CANDIDATES)
        except ValueError:
            id_col = None

        count = 0
        for idx, row in enumerate(ds):
            if self.cfg.max_samples is not None and count >= self.cfg.max_samples:
                break
            reference = row.get(text_col)
            if reference is None:
                continue
            reference = str(reference).strip()
            if not reference:
                continue

            audio_array, sr = extract_audio(row[audio_col])
            sample_id = (
                str(row[id_col])
                if id_col and row.get(id_col) is not None
                else f"{self.language_config}_{idx}"
            )
            yield AudioSample(
                sample_id=sample_id,
                language=self.language_name,
                language_config=self.language_config,
                reference=reference,
                audio_array=audio_array,
                sampling_rate=sr,
                meta={k: v for k, v in row.items() if k not in {audio_col}},
            )
            count += 1

    @staticmethod
    def _resolve_columns(ds) -> list[str]:
        if hasattr(ds, "column_names") and ds.column_names is not None:
            return list(ds.column_names)
        if hasattr(ds, "features") and ds.features is not None:
            return list(ds.features.keys())
        raise ValueError("Could not resolve dataset columns")


class FleursDatasetLoader(BaseDatasetLoader):
    """Convenience wrapper around google/fleurs with multi-language support."""

    DEFAULT_LANGUAGES = {
        "swahili": "sw_ke",
        "amharic": "am_et",
        "oromo": "om_et",
        "hausa": "ha_ng",
    }

    def __init__(self, cfg: DatasetConfig) -> None:
        self.cfg = cfg
        langs = cfg.languages or self.DEFAULT_LANGUAGES
        self.languages = {str(k): str(v) for k, v in langs.items()}

    def iter_samples(self) -> Iterator[AudioSample]:
        for lang_name, config_name in self.languages.items():
            loader = HuggingFaceDatasetLoader(
                DatasetConfig(
                    type="huggingface",
                    dataset_id=self.cfg.dataset_id or "google/fleurs",
                    config_name=config_name,
                    split=self.cfg.split,
                    audio_column=self.cfg.audio_column or "audio",
                    text_column=self.cfg.text_column or "transcription",
                    id_column=self.cfg.id_column or "id",
                    max_samples=self.cfg.max_samples,
                    streaming=self.cfg.streaming,
                    revision=self.cfg.revision,
                    trust_remote_code=self.cfg.trust_remote_code,
                    load_kwargs=self.cfg.load_kwargs,
                ),
                language_name=lang_name,
                language_config=config_name,
            )
            yield from loader.iter_samples()


def build_loader(cfg: DatasetConfig) -> BaseDatasetLoader:
    dtype = (cfg.type or "fleurs").lower()
    if dtype in {"fleurs", "google/fleurs"}:
        return FleursDatasetLoader(cfg)
    if dtype in {"huggingface", "hf", "datasets"}:
        # Single config HF dataset (optionally multi-lang via languages map)
        if cfg.languages:
            return _MultiConfigHFLoader(cfg)
        return HuggingFaceDatasetLoader(cfg)
    raise ValueError(
        f"Unknown dataset.type={cfg.type!r}. Use 'fleurs' or 'huggingface'."
    )


class _MultiConfigHFLoader(BaseDatasetLoader):
    def __init__(self, cfg: DatasetConfig) -> None:
        self.cfg = cfg

    def iter_samples(self) -> Iterator[AudioSample]:
        for lang_name, config_name in self.cfg.languages.items():
            loader = HuggingFaceDatasetLoader(
                DatasetConfig(
                    type="huggingface",
                    dataset_id=self.cfg.dataset_id,
                    config_name=config_name,
                    split=self.cfg.split,
                    audio_column=self.cfg.audio_column,
                    text_column=self.cfg.text_column,
                    id_column=self.cfg.id_column,
                    max_samples=self.cfg.max_samples,
                    streaming=self.cfg.streaming,
                    revision=self.cfg.revision,
                    trust_remote_code=self.cfg.trust_remote_code,
                    load_kwargs=self.cfg.load_kwargs,
                ),
                language_name=lang_name,
                language_config=config_name,
            )
            yield from loader.iter_samples()
