from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class OpenRouterConfig:
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_seconds: float = 120.0
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass
class NormalizationConfig:
    lowercase: bool = True
    strip_punctuation: bool = True
    collapse_whitespace: bool = True


@dataclass
class DatasetSource:
    """One Hugging Face dataset split to include in an evaluation run."""

    dataset_id: str
    split: str = "test"
    config_name: Optional[str] = None
    language: str = "swahili"
    audio_column: Optional[str] = None
    text_column: Optional[str] = None
    id_column: Optional[str] = None
    max_samples: Optional[int] = None
    revision: Optional[str] = None
    trust_remote_code: Optional[bool] = None
    load_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    type: str = "fleurs"
    dataset_id: str = "google/fleurs"
    config_name: Optional[str] = None
    split: str = "test"
    languages: dict[str, str] = field(default_factory=dict)
    # Optional multi-dataset mode (e.g. ndizi-1 + ndizi-1-2025)
    sources: list[DatasetSource] = field(default_factory=list)
    # Default language label when using a single HF dataset (no languages map)
    language: str = "swahili"
    audio_column: str = "audio"
    text_column: str = "transcription"
    id_column: Optional[str] = None
    max_samples: Optional[int] = None
    streaming: bool = True
    revision: Optional[str] = None
    trust_remote_code: bool = False
    # Extra kwargs forwarded to datasets.load_dataset
    load_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalWhisperConfig:
    device: Optional[str] = None  # cuda | mps | cpu | null=auto
    torch_dtype: Optional[str] = "auto"  # auto | float16 | bfloat16 | float32
    num_beams: int = 1
    # Local GPU models are typically safer sequential; override if needed.
    concurrency: int = 1


@dataclass
class RunConfig:
    experiment_name: str = "gemini_asr_bench"
    models: list[str] = field(default_factory=list)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    local_whisper: LocalWhisperConfig = field(default_factory=LocalWhisperConfig)
    prompt: str = (
        "Transcribe the speech in this audio exactly. "
        "Return only the transcript text with no commentary or labels."
    )
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    concurrency: int = 4
    output_dir: str = "results"
    resume: bool = True
    save_every: int = 20


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _parse_sources(raw_sources: Any, defaults: dict[str, Any]) -> list[DatasetSource]:
    if not raw_sources:
        return []
    if not isinstance(raw_sources, list):
        raise ValueError("'dataset.sources' must be a list")
    out: list[DatasetSource] = []
    for item in raw_sources:
        if isinstance(item, str):
            # Accept "org/name" or "org/name:split"
            dataset_id, split = item, defaults.get("split", "test")
            if ":" in item:
                dataset_id, split = item.rsplit(":", 1)
            out.append(
                DatasetSource(
                    dataset_id=dataset_id.strip(),
                    split=str(split).strip() or "test",
                    language=str(defaults.get("language", "swahili")),
                    audio_column=defaults.get("audio_column"),
                    text_column=defaults.get("text_column"),
                    id_column=defaults.get("id_column"),
                    max_samples=defaults.get("max_samples"),
                    revision=defaults.get("revision"),
                    trust_remote_code=defaults.get("trust_remote_code"),
                )
            )
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Invalid dataset source entry: {item!r}")
        if not item.get("dataset_id"):
            raise ValueError(f"dataset source missing dataset_id: {item!r}")
        out.append(
            DatasetSource(
                dataset_id=str(item["dataset_id"]),
                split=str(item.get("split") or defaults.get("split") or "test"),
                config_name=item.get("config_name", defaults.get("config_name")),
                language=str(item.get("language") or defaults.get("language") or "swahili"),
                audio_column=item.get("audio_column", defaults.get("audio_column")),
                text_column=item.get("text_column", defaults.get("text_column")),
                id_column=item.get("id_column", defaults.get("id_column")),
                max_samples=item.get("max_samples", defaults.get("max_samples")),
                revision=item.get("revision", defaults.get("revision")),
                trust_remote_code=item.get("trust_remote_code", defaults.get("trust_remote_code")),
                load_kwargs=dict(item.get("load_kwargs") or defaults.get("load_kwargs") or {}),
            )
        )
    return out


def load_run_config(raw: dict[str, Any]) -> RunConfig:
    ds_raw = _as_dict(raw.get("dataset"))
    or_raw = _as_dict(raw.get("openrouter"))
    norm_raw = _as_dict(raw.get("normalization"))
    local_raw = _as_dict(raw.get("local_whisper"))

    languages = ds_raw.get("languages") or {}
    if not isinstance(languages, dict):
        raise ValueError("'dataset.languages' must be a mapping of name -> config")

    defaults_for_sources = {
        "split": ds_raw.get("split", "test"),
        "config_name": ds_raw.get("config_name"),
        "language": ds_raw.get("language", "swahili"),
        "audio_column": ds_raw.get("audio_column"),
        "text_column": ds_raw.get("text_column"),
        "id_column": ds_raw.get("id_column"),
        "max_samples": ds_raw.get("max_samples"),
        "revision": ds_raw.get("revision"),
        "trust_remote_code": ds_raw.get("trust_remote_code"),
        "load_kwargs": ds_raw.get("load_kwargs") or {},
    }
    sources = _parse_sources(ds_raw.get("sources"), defaults_for_sources)

    return RunConfig(
        experiment_name=str(raw.get("experiment_name", "gemini_asr_bench")),
        models=list(raw.get("models") or []),
        dataset=DatasetConfig(
            type=str(ds_raw.get("type", "fleurs")).lower(),
            dataset_id=str(ds_raw.get("dataset_id", "google/fleurs")),
            config_name=ds_raw.get("config_name"),
            split=str(ds_raw.get("split", "test")),
            languages={str(k): str(v) for k, v in languages.items()},
            sources=sources,
            language=str(ds_raw.get("language", "swahili")),
            audio_column=str(ds_raw.get("audio_column", "audio")),
            text_column=str(ds_raw.get("text_column", "transcription")),
            id_column=ds_raw.get("id_column"),
            max_samples=ds_raw.get("max_samples"),
            streaming=bool(ds_raw.get("streaming", True)),
            revision=ds_raw.get("revision"),
            trust_remote_code=bool(ds_raw.get("trust_remote_code", False)),
            load_kwargs=dict(ds_raw.get("load_kwargs") or {}),
        ),
        openrouter=OpenRouterConfig(
            base_url=str(or_raw.get("base_url", "https://openrouter.ai/api/v1")),
            timeout_seconds=float(or_raw.get("timeout_seconds", 120)),
            max_retries=int(or_raw.get("max_retries", 5)),
            retry_backoff_seconds=float(or_raw.get("retry_backoff_seconds", 2.0)),
            temperature=float(or_raw.get("temperature", 0.0)),
            max_tokens=int(or_raw.get("max_tokens", 1024)),
        ),
        local_whisper=LocalWhisperConfig(
            device=local_raw.get("device"),
            torch_dtype=local_raw.get("torch_dtype", "auto"),
            num_beams=int(local_raw.get("num_beams", 1)),
            concurrency=max(1, int(local_raw.get("concurrency", 1))),
        ),
        prompt=str(raw.get("prompt") or RunConfig.prompt).strip(),
        normalization=NormalizationConfig(
            lowercase=bool(norm_raw.get("lowercase", True)),
            strip_punctuation=bool(norm_raw.get("strip_punctuation", True)),
            collapse_whitespace=bool(norm_raw.get("collapse_whitespace", True)),
        ),
        concurrency=max(1, int(raw.get("concurrency", 4))),
        output_dir=str(raw.get("output_dir", "results")),
        resume=bool(raw.get("resume", True)),
        save_every=max(1, int(raw.get("save_every", 20))),
    )
