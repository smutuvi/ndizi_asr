from __future__ import annotations

from gemini_asr_bench.backends.resolve import (
    canonicalize_model_id,
    needs_hf_auth,
    needs_openrouter,
    parse_model_spec,
)
from gemini_asr_bench.backends.sunbird_langs import resolve_sunbird_language_code
from gemini_asr_bench.config import NormalizationConfig, load_run_config
from gemini_asr_bench.datasets.presets import expand_dataset_flags
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
    assert parse_model_spec("google/gemma-4-E2B-it").backend == "hf_gemma"
    assert parse_model_spec("google/gemma-4-e2b").model_id == "google/gemma-4-E2B-it"
    assert parse_model_spec("sunbird/gemma-4-e2b").backend == "hf_gemma"
    assert parse_model_spec("sunbird/gemma-4-e2b").model_id == "Sunbird/Sunflower-Gemma4-E2B"
    assert parse_model_spec("Sunbird/Sunflower-Gemma4-E2B").backend == "hf_gemma"
    assert needs_openrouter(["sunbird/gemma-4-e2b"]) is False
    assert needs_hf_auth(["google/gemma-4-e2b", "Sunbird/asr-whisper-51-african-languages"]) is True
    assert canonicalize_model_id("gemma-4-e2b") == "google/gemma-4-E2B-it"


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


def test_expand_dataset_flags():
    sources = expand_dataset_flags(
        ["smutuvi/ndizi-1", "smutuvi/ndizi-1-2025", "google/fleurs"],
        max_samples=2,
    )
    assert sources[0]["dataset_id"] == "smutuvi/ndizi-1"
    assert sources[0]["text_column"] == "text"
    assert sources[1]["dataset_id"] == "smutuvi/ndizi-1-2025"
    fleurs = [s for s in sources if s["dataset_id"] == "google/fleurs"]
    assert len(fleurs) == 4
    assert {s["config_name"] for s in fleurs} == {"sw_ke", "am_et", "om_et", "ha_ng"}
    assert all(s["max_samples"] == 2 for s in sources)


def test_expand_waxalnlp_dataset_flags():
    sources = expand_dataset_flags(["google/WaxalNLP", "google/WaxalNLP:amh_asr"], max_samples=3)
    waxal = [s for s in sources if s["dataset_id"] == "google/WaxalNLP"]
    # Bare google/WaxalNLP expands to amh_asr + orm_asr only (no hau/swa ASR).
    expanded = [s for s in waxal if s.get("config_name") in {"amh_asr", "orm_asr"}]
    assert len(expanded) >= 2
    assert {s["config_name"] for s in expanded} == {"amh_asr", "orm_asr"}
    assert any(s["language"] == "amharic" and s["config_name"] == "amh_asr" for s in waxal)


def test_expand_horn_asr_dataset_flags():
    sources = expand_dataset_flags(["LesanAI/Horn-ASR", "LesanAI/Horn-ASR:am"], max_samples=5)
    horn = [s for s in sources if s["dataset_id"] == "LesanAI/Horn-ASR"]
    assert any(s["config_name"] == "am" and s["language"] == "amharic" for s in horn)
    assert any(s["config_name"] == "om" and s["language"] == "oromo" for s in horn)
    assert all(s.get("text_column") == "transcript" for s in horn)
    assert all(s.get("audio_column") == "file_name" for s in horn)
    assert all(s.get("max_samples") == 5 for s in horn)
