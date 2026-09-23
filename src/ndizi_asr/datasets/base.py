from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np

from ..config import DatasetConfig, DatasetSource


@dataclass
class AudioSample:
    sample_id: str
    language: str
    language_config: str
    reference: str
    audio_array: np.ndarray
    sampling_rate: int
    meta: dict[str, Any]
    dataset_id: str = ""


AUDIO_COLUMN_CANDIDATES = ["audio", "audio_path", "path", "file_name", "file", "wav", "speech"]
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


def resolve_hub_audio_path(
    dataset_id: str,
    file_name: str,
    *,
    config_name: Optional[str] = None,
    split: str = "test",
) -> str:
    """Resolve a basename like am_0001.wav to a local Hub cache path (Horn-ASR style)."""
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    if Path(file_name).exists():
        return file_name

    candidates: list[str] = []
    if config_name:
        candidates.append(f"{config_name}/{split}/{file_name}")
        candidates.append(f"{config_name}/{file_name}")
    candidates.append(f"{split}/{file_name}")
    candidates.append(file_name)

    last_err: Optional[Exception] = None
    for rel in candidates:
        try:
            return hf_hub_download(repo_id=dataset_id, filename=rel, repo_type="dataset")
        except Exception as exc:  # noqa: BLE001 - try next candidate
            last_err = exc
    raise FileNotFoundError(
        f"Could not resolve audio {file_name!r} under {dataset_id} "
        f"(config={config_name!r}, split={split!r})"
    ) from last_err


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
        return _to_mono_float32(array), int(sr)

    if isinstance(value, str):
        import soundfile as sf

        array, sr = sf.read(value)
        return _to_mono_float32(array), int(sr)

    # Hugging Face datasets >= 4.x: torchcodec AudioDecoder
    if hasattr(value, "get_all_samples"):
        samples = value.get_all_samples()
        data = samples.data
        if hasattr(data, "detach"):
            data = data.detach()
        if hasattr(data, "cpu"):
            data = data.cpu().numpy()
        else:
            data = np.asarray(data)
        sr = int(getattr(samples, "sample_rate", None) or getattr(samples, "sampling_rate"))
        return _to_mono_float32(data), sr

    # Legacy / duck-typed objects with .array + .sampling_rate
    array = getattr(value, "array", None)
    sr = getattr(value, "sampling_rate", None) or getattr(value, "sample_rate", None)
    if array is not None and sr is not None:
        return _to_mono_float32(array), int(sr)

    raise ValueError(f"Unsupported audio value type: {type(value)}")


def _to_mono_float32(array: Any) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        # (channels, samples) from torchcodec, or (samples, channels) from soundfile
        if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
            arr = arr.mean(axis=0) if arr.shape[0] > 1 else arr[0]
        else:
            arr = arr.mean(axis=1) if arr.shape[1] > 1 else arr[:, 0]
    return np.ascontiguousarray(arr, dtype=np.float32)


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

    cols = list(ds.column_names) if hasattr(ds, "column_names") else []
    # Prefer 16 kHz decode for ASR; works on map-style and most streaming IterableDatasets
    if "audio" in cols:
        try:
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        except Exception:
            # Streaming backends occasionally reject cast; extract_audio still handles decode
            pass
    return ds


def _slug_dataset_id(dataset_id: str) -> str:
    return dataset_id.replace("/", "_")


class HuggingFaceDatasetLoader(BaseDatasetLoader):
    """Generic loader for any Hugging Face speech dataset with audio + text columns."""

    def __init__(
        self,
        cfg: DatasetConfig,
        *,
        language_name: str = "default",
        language_config: Optional[str] = None,
        dataset_id_override: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.language_name = language_name
        self.language_config = language_config or cfg.config_name or "default"
        self.dataset_id = dataset_id_override or cfg.dataset_id

    def iter_samples(self) -> Iterator[AudioSample]:
        config_name = self.cfg.config_name
        if not config_name and self.language_config and self.language_config not in {
            "default",
            self.dataset_id,
            _slug_dataset_id(self.dataset_id),
        }:
            # Only treat language_config as HF config when it looks like a subset name
            # (e.g. fleurs sw_ke), not a dataset slug used for labeling.
            if "/" not in str(self.language_config):
                config_name = self.language_config

        ds = _load_hf_split(
            self.dataset_id,
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

        ds_slug = _slug_dataset_id(self.dataset_id)
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

            audio_val = row[audio_col]
            if isinstance(audio_val, str):
                from pathlib import Path

                if not Path(audio_val).exists():
                    audio_val = resolve_hub_audio_path(
                        self.dataset_id,
                        audio_val,
                        config_name=config_name,
                        split=self.cfg.split,
                    )
            audio_array, sr = extract_audio(audio_val)
            raw_id = (
                str(row[id_col])
                if id_col and row.get(id_col) is not None
                else f"{self.language_config}_{idx}"
            )
            sample_id = f"{ds_slug}:{raw_id}:{idx}"
            yield AudioSample(
                sample_id=sample_id,
                language=self.language_name,
                language_config=self.language_config,
                reference=reference,
                audio_array=audio_array,
                sampling_rate=sr,
                meta={k: v for k, v in row.items() if k not in {audio_col}},
                dataset_id=self.dataset_id,
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


class MultiSourceHFLoader(BaseDatasetLoader):
    """Load and concatenate multiple HF dataset sources in one eval run."""

    def __init__(self, cfg: DatasetConfig) -> None:
        if not cfg.sources:
            raise ValueError("MultiSourceHFLoader requires dataset.sources")
        self.cfg = cfg

    def iter_samples(self) -> Iterator[AudioSample]:
        for source in self.cfg.sources:
            yield from self._iter_source(source)

    def _iter_source(self, source: DatasetSource) -> Iterator[AudioSample]:
        loader = HuggingFaceDatasetLoader(
            DatasetConfig(
                type="huggingface",
                dataset_id=source.dataset_id,
                config_name=source.config_name,
                split=source.split,
                language=source.language,
                audio_column=source.audio_column or self.cfg.audio_column or "audio",
                text_column=source.text_column or self.cfg.text_column or "text",
                id_column=source.id_column if source.id_column is not None else self.cfg.id_column,
                max_samples=(
                    source.max_samples
                    if source.max_samples is not None
                    else self.cfg.max_samples
                ),
                streaming=self.cfg.streaming,
                revision=source.revision if source.revision is not None else self.cfg.revision,
                trust_remote_code=(
                    bool(source.trust_remote_code)
                    if source.trust_remote_code is not None
                    else self.cfg.trust_remote_code
                ),
                load_kwargs=source.load_kwargs or self.cfg.load_kwargs,
            ),
            language_name=source.language,
            language_config=_slug_dataset_id(source.dataset_id),
            dataset_id_override=source.dataset_id,
        )
        yield from loader.iter_samples()


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


def build_loader(cfg: DatasetConfig) -> BaseDatasetLoader:
    dtype = (cfg.type or "fleurs").lower()
    if dtype in {"csv", "manifest"}:
        from .csv_manifest import CSVManifestLoader

        return CSVManifestLoader(cfg)
    if dtype in {"cropconcept", "crop_concept"}:
        from .csv_manifest import CropConceptLoader

        return CropConceptLoader(cfg)
    if cfg.sources:
        return MultiSourceHFLoader(cfg)
    if dtype in {"fleurs", "google/fleurs"}:
        return FleursDatasetLoader(cfg)
    if dtype in {"huggingface", "hf", "datasets"}:
        if cfg.languages:
            return _MultiConfigHFLoader(cfg)
        return HuggingFaceDatasetLoader(
            cfg,
            language_name=cfg.language or "default",
            language_config=cfg.config_name or "default",
        )
    raise ValueError(
        f"Unknown dataset.type={cfg.type!r}. "
        "Use 'fleurs', 'huggingface', 'csv', or 'cropconcept'."
    )
