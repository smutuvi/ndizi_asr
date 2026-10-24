from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

from .backends.resolve import (
    expand_model_groups,
    list_known_models,
    needs_hf_auth,
    needs_omnilingual,
    needs_openrouter,
    parse_model_spec,
)
from .config import load_run_config
from .datasets.presets import expand_dataset_flags
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
        prog="ndizi-asr",
        description=(
            "Ndizi ASR — train and evaluate speech recognition models "
            "(OpenRouter Gemini, local Whisper/Gemma/CTC, Omnilingual) "
            "on Hugging Face datasets, CSV manifests, or CropConcept sets."
        ),
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="Print known model aliases/groups and exit",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config path. If omitted, build the run from CLI flags.",
    )
    p.add_argument(
        "--model",
        "--models",
        nargs="+",
        dest="models",
        default=None,
        metavar="MODEL",
        help=(
            "Model ids or groups (space-separated). Alias: --models. Examples: "
            "--models gemini-2.5-flash omnilingualasr ethio-asr-oromo "
            "or --model oromo / ethio / whisper-family"
        ),
    )
    p.add_argument(
        "--dataset",
        nargs="+",
        dest="datasets",
        default=None,
        metavar="DATASET",
        help=(
            "One or more HF datasets (space-separated). Examples: "
            "--dataset smutuvi/ndizi-1 google/fleurs LesanAI/Horn-ASR"
        ),
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        dest="datasets",
        metavar="DATASET",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--dataset-id",
        default=None,
        help="Single Hugging Face dataset id (legacy; prefer --dataset).",
    )
    p.add_argument(
        "--source",
        nargs="+",
        dest="sources",
        default=None,
        metavar="SOURCE",
        help="Additional HF dataset sources. Format: org/name or org/name:split.",
    )
    p.add_argument(
        "--dataset-type",
        choices=["fleurs", "huggingface", "csv", "cropconcept"],
        default=None,
        help="Dataset loader type",
    )
    p.add_argument(
        "--csv",
        dest="csv_path",
        default=None,
        help="Local CSV manifest (columns: audio_path, reference, language)",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Root for resolving relative audio_path values in CSV manifests",
    )
    p.add_argument(
        "--language-filter",
        "--languages",
        nargs="+",
        dest="language_filter",
        default=None,
        metavar="LANG",
        help=(
            "Keep only these language codes for CSV/CropConcept (e.g. am om). "
            "Alias: --languages (ndizi_eval-compatible)."
        ),
    )
    p.add_argument(
        "--normalize",
        choices=["default", "ethio", "oromo_platform", "oromo_fair", "swahili_fair", "swahili_canon", "none"],
        default=None,
        help=(
            "Scoring text normalization: ethio = Ge'ez/Latin cleanup; "
            "oromo_platform = colleague tidy (NFKC, lower, punct→space, collapse); "
            "oromo_fair = platform + hudhaa strip + spelling map + gemination collapse; "
            "swahili_fair = compound words + Bantu R/L & H-drop map + backchannel strip"
        ),
    )
    p.add_argument("--omni-card", default=None, help="Override Omnilingual model card")
    p.add_argument(
        "--omnilingual-root",
        default=None,
        help="Optional path to omnilingual-asr checkout (or set OMNILINGUAL_ASR_ROOT)",
    )
    p.add_argument(
        "--config-name",
        default=None,
        help="HF config/subset name (for single-config huggingface datasets)",
    )
    p.add_argument(
        "--language",
        nargs="+",
        dest="languages",
        default=None,
        metavar="NAME=CONFIG",
        help="Language mappings (space-separated), e.g. --language swahili=sw_ke amharic=am_et",
    )
    p.add_argument("--split", default=None, help="Dataset split (default: test)")
    p.add_argument("--audio-column", default=None)
    p.add_argument("--text-column", default=None)
    p.add_argument("--max-samples", type=int, default=None, help="Cap samples per language/config")
    p.add_argument("--streaming", action="store_true", default=None)
    p.add_argument("--no-streaming", action="store_true", default=False)
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--device", default=None, help="Local model device: cuda|mps|cpu")
    p.add_argument(
        "--torch-dtype",
        default=None,
        help="Local model dtype: auto|float16|bfloat16|float32",
    )
    p.add_argument("--num-beams", type=int, default=None, help="Local Whisper beam search width (default: 1)")
    p.add_argument(
        "--gemma-mbr",
        action="store_true",
        default=False,
        help="Rerank N-best Whisper beams using EmbeddingGemma MBR + Ndizi train-corpus prototype similarity",
    )
    p.add_argument(
        "--rag-prompt",
        action="store_true",
        default=False,
        help="Condition Whisper <startofprev> prompt on nearest training utterance retrieved by EmbeddingGemma",
    )
    p.add_argument(
        "--canonicalize-output",
        action="store_true",
        default=False,
        help="Canonicalize Swahili compounds and R/L/H-drop variants in Whisper output",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--experiment-name", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument(
        "--max-chunk-seconds",
        type=float,
        default=None,
        help="OpenRouter VAD chunk length in seconds (default 30; 0 = send whole clip)",
    )
    p.add_argument("--max-tokens", type=int, default=None, help="OpenRouter max output tokens")
    p.add_argument("--temperature", type=float, default=None, help="OpenRouter temperature")
    p.add_argument("--timeout-seconds", type=float, default=None, help="OpenRouter request timeout")
    p.add_argument(
        "--cropconcept-recipe",
        action="store_true",
        default=False,
        help="Use exact CropConcept Ethiopia recipe (raw .m4a bytes as format=mp4, audio-first, sidecar language hint, JSON output)",
    )
    p.add_argument("--resume", action="store_true", default=None)
    p.add_argument("--no-resume", action="store_true", default=False)
    p.add_argument("--env-file", type=Path, default=Path(".env"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _default_raw_config() -> dict[str, Any]:
    return {
        "experiment_name": "ndizi_asr_eval",
        "models": [],
        "dataset": {
            "type": "huggingface",
            "split": "test",
            "language": "swahili",
            "audio_column": "audio",
            "text_column": "text",
            "streaming": True,
            "sources": [],
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "timeout_seconds": 120,
            "max_retries": 5,
            "retry_backoff_seconds": 2.0,
            "temperature": 0.0,
            "max_tokens": 1024,
        },
        "local_whisper": {
            "device": None,
            "torch_dtype": "auto",
            "num_beams": 1,
            "concurrency": 1,
        },
        "prompt": (
            "Transcribe the speech in this audio exactly. "
            "Return only the transcript text with no commentary or labels."
        ),
        "normalization": {
            "mode": "default",
            "lowercase": True,
            "strip_punctuation": True,
            "collapse_whitespace": True,
        },
        "concurrency": 4,
        "output_dir": "results",
        "resume": True,
        "save_every": 20,
    }


def apply_overrides(raw: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    raw = dict(raw)
    ds = dict(raw.get("dataset") or {})
    local = dict(raw.get("local_whisper") or {})
    norm = dict(raw.get("normalization") or {})

    if args.experiment_name:
        raw["experiment_name"] = args.experiment_name
    if args.models:
        raw["models"] = expand_model_groups(args.models)
    if args.output_dir:
        raw["output_dir"] = args.output_dir
    if args.concurrency is not None:
        raw["concurrency"] = args.concurrency
    if args.prompt:
        raw["prompt"] = args.prompt

    openrouter = dict(raw.get("openrouter") or {})
    if args.max_chunk_seconds is not None:
        openrouter["max_chunk_seconds"] = args.max_chunk_seconds
    if args.max_tokens is not None:
        openrouter["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        openrouter["temperature"] = args.temperature
    if args.timeout_seconds is not None:
        openrouter["timeout_seconds"] = args.timeout_seconds
    if getattr(args, "cropconcept_recipe", False):
        openrouter["cropconcept_recipe"] = True
    raw["openrouter"] = openrouter

    if args.resume:
        raw["resume"] = True
    if args.no_resume:
        raw["resume"] = False

    if args.normalize:
        norm["mode"] = args.normalize

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
        ds["max_samples"] = args.max_samples if args.max_samples > 0 else None
    if args.streaming:
        ds["streaming"] = True
    if args.no_streaming:
        ds["streaming"] = False
    if args.languages:
        ds["languages"] = _parse_kv_pairs(args.languages)
    if args.csv_path:
        ds["type"] = "csv"
        ds["csv_path"] = args.csv_path
    if args.data_root:
        ds["data_root"] = args.data_root
    if args.language_filter:
        ds["language_filter"] = list(args.language_filter)

    dataset_tokens: list[str] = []
    if args.datasets:
        dataset_tokens.extend(args.datasets)
    if args.sources:
        dataset_tokens.extend(args.sources)
    if dataset_tokens:
        langs = ds.get("languages") if isinstance(ds.get("languages"), dict) else None
        expanded = expand_dataset_flags(
            dataset_tokens,
            default_split=str(ds.get("split") or "test"),
            default_language=str(ds.get("language") or "swahili"),
            languages=langs,
            audio_column=ds.get("audio_column"),
            text_column=ds.get("text_column"),
            max_samples=ds.get("max_samples"),
        )
        ds["type"] = "huggingface"
        ds["sources"] = expanded

    if args.device:
        local["device"] = args.device
    if args.torch_dtype:
        local["torch_dtype"] = args.torch_dtype
    if getattr(args, "num_beams", None) is not None:
        local["num_beams"] = int(args.num_beams)
    if getattr(args, "gemma_mbr", False):
        local["gemma_mbr"] = True
    if getattr(args, "rag_prompt", False):
        local["rag_prompt"] = True
    if getattr(args, "canonicalize_output", False):
        local["canonicalize_output"] = True
    if args.omni_card:
        local["omni_card"] = args.omni_card
    if args.omnilingual_root:
        local["omnilingual_root"] = args.omnilingual_root

    raw["dataset"] = ds
    raw["local_whisper"] = local
    raw["normalization"] = norm
    return raw


def main(argv: Optional[list[str]] = None) -> int:
    # Before datasets/torchcodec import: force path-based audio decode
    os.environ.setdefault("HF_DATASETS_DISABLE_TORCHCODEC", "1")

    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.list_models:
        print(list_known_models())
        return 0

    if args.env_file.exists():
        load_dotenv(args.env_file)
    else:
        load_dotenv()

    if args.config is None:
        raw = _default_raw_config()
    else:
        if not args.config.exists():
            logging.error("Config not found: %s", args.config)
            return 2
        with args.config.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    raw = apply_overrides(raw, args)
    if raw.get("models"):
        raw["models"] = expand_model_groups(list(raw["models"]))
    config = load_run_config(raw)

    if not config.models:
        logging.error("No models specified. Pass --model ... or set models in --config.")
        return 2
    has_data = (
        bool(config.dataset.sources)
        or (config.dataset.type in {"fleurs", "google/fleurs"})
        or bool(config.dataset.dataset_id)
        or bool(config.dataset.csv_path)
        or (
            config.dataset.type in {"cropconcept", "crop_concept"}
            and config.dataset.cropconcept_media_map
        )
    )
    if not has_data:
        logging.error(
            "No datasets specified. Pass --dataset ..., --csv path, "
            "or provide a YAML --config."
        )
        return 2

    for m in config.models:
        spec = parse_model_spec(m)
        logging.info("Model %s → backend=%s id=%s", m, spec.backend, spec.model_id)

    env_base = os.getenv("OPENROUTER_BASE_URL")
    if env_base:
        config.openrouter.base_url = env_base

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip() or None
    if needs_openrouter(config.models) and not api_key:
        logging.error(
            "OPENROUTER_API_KEY not set, but OpenRouter models were requested. "
            "Copy .env.example to .env and add your key, or pass only local models."
        )
        return 2

    hf_token = (
        os.getenv("HF_TOKEN", "").strip()
        or os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip()
        or os.getenv("HUGGING_FACE_HUB_TOKEN", "").strip()
    )
    if needs_hf_auth(config.models) and not hf_token:
        logging.warning(
            "Local HF models usually need auth for gated repos (Gemma / Sunbird). "
            "Accept model terms on Hugging Face and set HF_TOKEN in .env."
        )
    if needs_omnilingual(config.models):
        logging.info(
            "Omnilingual backend requested — ensure omnilingual-asr is installed "
            '(`pip install "omnilingual-asr[data]"`) or OMNILINGUAL_ASR_ROOT is set.'
        )

    runner = BenchmarkRunner(config, api_key=api_key)
    summary = runner.run()
    _print_comparison_table(summary)
    return 0


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _print_comparison_table(summary: dict[str, Any]) -> None:
    """Print a markdown-style model comparison table (WER/CER)."""
    rows: list[dict[str, Any]] = []
    for model, metrics in (summary.get("models") or {}).items():
        overall = metrics.get("overall") or {}
        rows.append(
            {
                "model": model,
                "backend": metrics.get("backend") or "",
                "wer": overall.get("wer"),
                "cer": overall.get("cer"),
                "wer_n": overall.get("wer_normalized"),
                "cer_n": overall.get("cer_normalized"),
                "n": overall.get("num_scored"),
                "errors": overall.get("num_errors"),
            }
        )

    def sort_key(r: dict[str, Any]):
        w = r.get("wer_n")
        return (w is None, float(w) if w is not None else 0.0)

    rows.sort(key=sort_key)

    print("\n=== Model comparison (overall) ===")
    if not rows:
        print("(no models scored)")
        return

    headers = ("Model", "Backend", "WER↓", "CER", "WER_raw", "CER_raw", "n", "err")
    table_rows = [
        (
            str(r["model"]),
            str(r["backend"]),
            _fmt_metric(r["wer_n"]),
            _fmt_metric(r["cer_n"]),
            _fmt_metric(r["wer"]),
            _fmt_metric(r["cer"]),
            str(r["n"] if r["n"] is not None else "—"),
            str(r["errors"] if r["errors"] is not None else "—"),
        )
        for r in rows
    ]
    widths = [len(h) for h in headers]
    for tr in table_rows:
        for i, cell in enumerate(tr):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: tuple[str, ...]) -> str:
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for tr in table_rows:
        print(fmt_row(tr))

    # Per-language breakdown when more than one language was scored
    lang_keys: set[str] = set()
    for metrics in (summary.get("models") or {}).values():
        lang_keys.update((metrics.get("per_language") or {}).keys())
    if len(lang_keys) > 1:
        print("\n=== Per-language (normalized WER / CER) ===")
        for lang in sorted(lang_keys):
            print(f"\n[{lang}]")
            for r in rows:
                metrics = (summary.get("models") or {}).get(r["model"]) or {}
                lm = (metrics.get("per_language") or {}).get(lang) or {}
                print(
                    f"  {r['model']}: WER={_fmt_metric(lm.get('wer_normalized'))} "
                    f"CER={_fmt_metric(lm.get('cer_normalized'))} "
                    f"n={lm.get('num_scored')}"
                )

    out = summary.get("experiment_name")
    print(
        f"\nFull metrics: results/<experiment>/summary.csv "
        f"(experiment={out or 'see --experiment-name'})"
    )


if __name__ == "__main__":
    sys.exit(main())
