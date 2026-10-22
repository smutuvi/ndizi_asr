"""Training entrypoints scaffold for Ndizi ASR MLOps.

Fine-tuning Whisper / CTC models on Hub or CSV manifests. Full Trainer loops
land here; for now this validates config and prints the planned run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Whisper / CTC fine-tune configuration (scaffold)."""

    experiment_name: str = "asr_finetune"
    backend: str = "whisper"  # whisper | ctc
    model_id: str = "openai/whisper-large-v3"
    dataset_id: Optional[str] = None
    csv_path: Optional[str] = None
    config_name: Optional[str] = None
    split: str = "train"
    audio_column: str = "audio"
    text_column: str = "text"
    language: Optional[str] = None
    output_dir: str = "outputs"
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 8
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    logging_steps: int = 50
    save_steps: int = 500
    eval_split: Optional[str] = "validation"
    max_steps: Optional[int] = None
    lora: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def load_train_config(raw: dict[str, Any]) -> TrainConfig:
    return TrainConfig(
        experiment_name=str(raw.get("experiment_name", "asr_finetune")),
        backend=str(raw.get("backend", "whisper")).lower(),
        model_id=str(raw.get("model_id", "openai/whisper-large-v3")),
        dataset_id=raw.get("dataset_id"),
        csv_path=raw.get("csv_path"),
        config_name=raw.get("config_name"),
        split=str(raw.get("split", "train")),
        audio_column=str(raw.get("audio_column", "audio")),
        text_column=str(raw.get("text_column", "text")),
        language=raw.get("language"),
        output_dir=str(raw.get("output_dir", "outputs")),
        num_train_epochs=float(raw.get("num_train_epochs", 3.0)),
        per_device_train_batch_size=int(raw.get("per_device_train_batch_size", 8)),
        learning_rate=float(raw.get("learning_rate", 1e-5)),
        warmup_ratio=float(raw.get("warmup_ratio", 0.1)),
        fp16=bool(raw.get("fp16", True)),
        gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 1)),
        logging_steps=int(raw.get("logging_steps", 50)),
        save_steps=int(raw.get("save_steps", 500)),
        eval_split=raw.get("eval_split", "validation"),
        max_steps=raw.get("max_steps"),
        lora=bool(raw.get("lora", False)),
        extra=dict(raw.get("extra") or {}),
    )


def run_train(cfg: TrainConfig, *, dry_run: bool = False) -> dict[str, Any]:
    """Execute or preview a fine-tune run.

    Trainer implementation is staged: validate inputs, write a run plan under
    outputs/<experiment>/, then (when dry_run=False and deps present) call the
    backend-specific trainer.
    """
    extra_ids = cfg.extra.get("dataset_ids") or cfg.extra.get("train_datasets")
    has_hub = bool(cfg.dataset_id) or bool(extra_ids)
    if not has_hub and not cfg.csv_path:
        raise ValueError(
            "Provide dataset_id, extra.dataset_ids / extra.train_datasets (HF Hub), "
            "or csv_path for training data"
        )
    if cfg.backend not in {"whisper", "whisper_grpo", "grpo", "ctc"}:
        raise ValueError(f"Unsupported train backend {cfg.backend!r}; use whisper|whisper_grpo|ctc")

    out = Path(cfg.output_dir) / cfg.experiment_name
    out.mkdir(parents=True, exist_ok=True)
    plan = asdict(cfg)
    plan_path = out / "train_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    logger.info("Wrote train plan to %s", plan_path)

    if dry_run:
        logger.info("Dry-run only — skipping Trainer. Install .[train] and re-run without --dry-run.")
        return {"status": "dry_run", "plan_path": str(plan_path), "config": plan}

    # Deferred import so eval-only installs stay light.
    try:
        if cfg.backend in {"whisper_grpo", "grpo"} or str(cfg.extra.get("stage", "")).lower() == "grpo":
            from .whisper_grpo import train_whisper_grpo

            return train_whisper_grpo(cfg)
        if cfg.backend == "whisper":
            from .whisper_trainer import train_whisper

            return train_whisper(cfg)
        from .ctc_trainer import train_ctc

        return train_ctc(cfg)
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies missing. Install with:\n"
            '  pip install -e ".[train]"\n'
            f"Original error: {exc}"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ndizi-asr-train",
        description="Fine-tune Whisper or CTC ASR models (Ndizi MLOps).",
    )
    p.add_argument("--config", type=Path, required=True, help="YAML train config")
    p.add_argument("--dry-run", action="store_true", help="Validate and write plan only")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # Gated Sunbird / Gemma models need HF_TOKEN from .env (same as eval CLI).
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    if not args.config.exists():
        logging.error("Config not found: %s", args.config)
        return 2
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    cfg = load_train_config(raw)
    result = run_train(cfg, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
