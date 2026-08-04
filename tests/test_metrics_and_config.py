from __future__ import annotations

from gemini_asr_bench.config import load_run_config
from gemini_asr_bench.metrics import normalize_text, summarize_predictions
from gemini_asr_bench.config import NormalizationConfig


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
