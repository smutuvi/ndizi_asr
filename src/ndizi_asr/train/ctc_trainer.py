"""CTC fine-tuning via Hugging Face Trainer (requires .[train])."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from . import TrainConfig


def train_ctc(cfg: "TrainConfig") -> dict[str, Any]:
    """Fine-tune a CTC / wav2vec2 / w2v-bert checkpoint."""
    raise NotImplementedError(
        "CTC Trainer loop is scaffolded but not yet implemented. "
        "Use --dry-run to validate configs, or port your wav2vec2 fine-tune notebooks "
        f"into ndizi_asr.train.ctc_trainer (model={cfg.model_id})."
    )
