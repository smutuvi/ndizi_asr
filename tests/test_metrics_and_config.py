from __future__ import annotations

from gemini_asr_bench.backends.resolve import needs_openrouter, parse_model_spec
from gemini_asr_bench.backends.sunbird_langs import resolve_sunbird_language_code
from gemini_asr_bench.config import NormalizationConfig, load_run_config
from gemini_asr_bench.metrics import normalize_text, summarize_predictions


def test_load_run_config_defaults():
    cfg = load_run_config(
        {
            "models": ["google/gemini-2.5-flash"],
            "dataset": {
                "type": "fleurs",
                "languages": {"swahili": "sw_ke"},
            },
        }
    )
    assert cfg.models == ["google/gemini-2.5-flash"]
    assert cfg.dataset.type == "fleurs"
    assert cfg.dataset.languages["swahili"] == "sw_ke"
    assert cfg.local_whisper.concurrency == 1


def test_normalize_text():
    cfg = NormalizationConfig(lowercase=True, strip_punctuation=True, collapse_whitespace=True)
    assert normalize_text("  Habari, Dunia!  ", cfg) == "habari dunia"


def test_summarize_predictions():
    rows = [
        {"reference": "hello world", "prediction": "hello world", "latency_seconds": 1.0},
        {"reference": "foo", "prediction": "bar", "latency_seconds": 2.0, "error": None},
    ]
    summary = summarize_predictions(rows, NormalizationConfig())
    assert summary["num_scored"] == 2
    assert summary["wer"] is not None
    assert summary["mean_latency_seconds"] == 1.5


def test_parse_model_spec_backends():
    assert parse_model_spec("google/gemini-2.5-flash").backend == "openrouter"
    assert parse_model_spec("Sunbird/asr-whisper-51-african-languages").backend == "hf_whisper"
    assert parse_model_spec("hf:Sunbird/asr-whisper-51-african-languages").model_id == (
        "Sunbird/asr-whisper-51-african-languages"
    )
    assert needs_openrouter(["Sunbird/asr-whisper-51-african-languages"]) is False
    assert needs_openrouter(["google/gemini-2.5-flash"]) is True


def test_resolve_sunbird_language_code():
    assert resolve_sunbird_language_code("swahili", "sw_ke") == "swa"
    assert resolve_sunbird_language_code("amharic", "am_et") == "amh"
    assert resolve_sunbird_language_code("oromo", "om_et") == "orm"
    assert resolve_sunbird_language_code("hausa", "ha_ng") == "hau"
    assert resolve_sunbird_language_code("Swahili", None) == "swa"


def test_load_multi_source_config():
    cfg = load_run_config(
        {
            "models": [
                "google/gemini-2.5-flash",
                "Sunbird/asr-whisper-51-african-languages",
            ],
            "dataset": {
                "type": "huggingface",
                "language": "swahili",
                "audio_column": "audio",
                "text_column": "text",
                "sources": [
                    "smutuvi/ndizi-1:test",
                    {"dataset_id": "smutuvi/ndizi-1-2025", "split": "test"},
                ],
            },
        }
    )
    assert len(cfg.dataset.sources) == 2
    assert cfg.dataset.sources[0].dataset_id == "smutuvi/ndizi-1"
    assert cfg.dataset.sources[0].split == "test"
    assert cfg.dataset.sources[0].language == "swahili"
    assert cfg.dataset.sources[1].dataset_id == "smutuvi/ndizi-1-2025"
