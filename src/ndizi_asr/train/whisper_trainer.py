"""Whisper fine-tuning via Hugging Face Trainer (requires .[train])."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from . import TrainConfig


def train_whisper(cfg: "TrainConfig") -> dict[str, Any]:
    """Fine-tune a Whisper checkpoint."""
    raise NotImplementedError(
        "Whisper Trainer loop is scaffolded but not yet implemented. "
        "Use --dry-run to validate configs, or port your Fine_Tune_* Whisper notebook "
        f"into ndizi_asr.train.whisper_trainer (model={cfg.model_id})."
    )
