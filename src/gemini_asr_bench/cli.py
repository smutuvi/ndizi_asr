from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

from .backends.resolve import needs_openrouter
from .config import load_run_config
from .evaluate import BenchmarkRunner


def _parse_kv_pairs(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gemini-asr-bench",
        description=(
            "Benchmark ASR models on Hugging Face datasets via OpenRouter "
            "or local Whisper checkpoints (e.g. Sunbird)."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fleurs_afro.yaml"),
        help="YAML config path (default: configs/fleurs_afro.yaml)",
    )
    p.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help=(
            "Model id (repeatable). OpenRouter ids like google/gemini-2.5-flash, "
            "or local HF Whisper ids like Sunbird/asr-whisper-51-african-languages "
            "(prefix with hf: if needed)."
        ),
    )
    p.add_argument(
        "--dataset-id",
        default=None,
        help="Hugging Face dataset id, e.g. google/fleurs or mozilla-foundation/common_voice_17_0",
    )
    p.add_argument(
        "--dataset-type",
        choices=["fleurs", "huggingface"],
        default=None,
        help="Dataset loader type",
    )
    p.add_argument(
        "--config-name",
        default=None,
        help="HF config/subset name (for single-config huggingface datasets)",
    )
    p.add_argument(
        "--language",
        action="append",
        dest="languages",
        default=None,
        help="Language mapping NAME=CONFIG (repeatable), e.g. swahili=sw_ke",
    )
    p.add_argument("--split", default=None, help="Dataset split (default from config)")
    p.add_argument("--audio-column", default=None)
    p.add_argument("--text-column", default=None)
    p.add_argument("--max-samples", type=int, default=None, help="Cap samples per language/config")
    p.add_argument("--streaming", action="store_true", default=None)
    p.add_argument("--no-streaming", action="store_true", default=False)
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--device", default=None, help="Local Whisper device: cuda|mps|cpu")
    p.add_argument(
        "--torch-dtype",
        default=None,
        help="Local Whisper dtype: auto|float16|bfloat16|float32",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--experiment-name", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--resume", action="store_true", default=None)
    p.add_argument("--no-resume", action="store_true", default=False)
    p.add_argument("--env-file", type=Path, default=Path(".env"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def apply_overrides(raw: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    raw = dict(raw)
    ds = dict(raw.get("dataset") or {})
    local = dict(raw.get("local_whisper") or {})

    if args.experiment_name:
        raw["experiment_name"] = args.experiment_name
    if args.models:
        raw["models"] = args.models
    if args.output_dir:
        raw["output_dir"] = args.output_dir
    if args.concurrency is not None:
        raw["concurrency"] = args.concurrency
    if args.prompt:
        raw["prompt"] = args.prompt

    if args.resume:
        raw["resume"] = True
    if args.no_resume:
        raw["resume"] = False

    if args.dataset_type:
        ds["type"] = args.dataset_type
    if args.dataset_id:
        ds["dataset_id"] = args.dataset_id
    if args.config_name:
        ds["config_name"] = args.config_name
    if args.split:
        ds["split"] = args.split
    if args.audio_column:
        ds["audio_column"] = args.audio_column
    if args.text_column:
        ds["text_column"] = args.text_column
    if args.max_samples is not None:
        ds["max_samples"] = args.max_samples
    if args.streaming:
        ds["streaming"] = True
    if args.no_streaming:
        ds["streaming"] = False
    if args.languages:
        ds["languages"] = _parse_kv_pairs(args.languages)

    if args.device:
        local["device"] = args.device
    if args.torch_dtype:
        local["torch_dtype"] = args.torch_dtype

    raw["dataset"] = ds
    raw["local_whisper"] = local
    return raw


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.env_file.exists():
        load_dotenv(args.env_file)
    else:
        load_dotenv()

    if not args.config.exists():
        logging.error("Config not found: %s", args.config)
        return 2

    with args.config.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw = apply_overrides(raw, args)
    config = load_run_config(raw)

    env_base = os.getenv("OPENROUTER_BASE_URL")
    if env_base:
        config.openrouter.base_url = env_base

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip() or None
    if needs_openrouter(config.models) and not api_key:
        logging.error(
            "OPENROUTER_API_KEY not set, but OpenRouter models were requested. "
            "Copy .env.example to .env and add your key, or pass only local models "
            "like Sunbird/asr-whisper-51-african-languages."
        )
        return 2

    runner = BenchmarkRunner(config, api_key=api_key)
    summary = runner.run()

    print("\n=== Benchmark summary ===")
    for model, metrics in summary.get("models", {}).items():
        overall = metrics.get("overall") or {}
        print(
            f"{model}: WER={overall.get('wer_normalized')} "
            f"CER={overall.get('cer_normalized')} "
            f"scored={overall.get('num_scored')} errors={overall.get('num_errors')}"
        )
        for lang, lm in (metrics.get("per_language") or {}).items():
            print(
                f"  - {lang}: WER={lm.get('wer_normalized')} CER={lm.get('cer_normalized')} "
                f"n={lm.get('num_scored')}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
