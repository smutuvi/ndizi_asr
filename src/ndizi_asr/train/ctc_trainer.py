"""CTC continued fine-tuning (Ethio-ASR / MMS / wav2vec2) via Hugging Face Trainer."""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

os.environ.setdefault("HF_DATASETS_DISABLE_TORCHCODEC", "1")

import numpy as np
import torch
from datasets import Audio, load_dataset
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

if TYPE_CHECKING:
    from . import TrainConfig

logger = logging.getLogger(__name__)

# Ethio-ASR joint LID+ASR targets (Abdullah et al. 2026): [ORM] + graphemes
_LANG_TAG = {
    "oromo": "[ORM]",
    "om": "[ORM]",
    "orm": "[ORM]",
    "afaan oromo": "[ORM]",
    "amharic": "[AMH]",
    "am": "[AMH]",
    "amh": "[AMH]",
    "tigrinya": "[TIR]",
    "ti": "[TIR]",
    "tir": "[TIR]",
    "sidaama": "[SID]",
    "sidama": "[SID]",
    "sid": "[SID]",
    "wolaytta": "[WAL]",
    "wal": "[WAL]",
}

_LID_RE = re.compile(r"^\[(?:AMH|TIR|ORM|SID|WAL)\]\|?\s*", re.IGNORECASE)


def _hf_token() -> Optional[str]:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return None


def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def _lang_tag(language: Optional[str]) -> str:
    if not language:
        return "[ORM]"
    key = str(language).strip().lower()
    return _LANG_TAG.get(key, f"[{key.upper()[:3]}]")


def _hub_ids(cfg: "TrainConfig") -> List[str]:
    extra_ids = cfg.extra.get("dataset_ids") or cfg.extra.get("train_datasets")
    if extra_ids:
        return [str(x) for x in extra_ids]
    if cfg.dataset_id:
        return [str(cfg.dataset_id)]
    raise ValueError("Provide dataset_id or extra.dataset_ids for Hub training")


def _make_training_arguments(**kwargs: Any) -> TrainingArguments:
    """Build TrainingArguments, dropping/aliasing keys for older transformers."""
    params = inspect.signature(TrainingArguments.__init__).parameters
    # Older transformers: evaluation_strategy; newer: eval_strategy
    if "eval_strategy" in kwargs and "eval_strategy" not in params and "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    elif "evaluation_strategy" in kwargs and "evaluation_strategy" not in params and "eval_strategy" in params:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    unknown = [k for k in list(kwargs) if k not in params and k != "self"]
    for k in unknown:
        logger.warning("TrainingArguments: ignoring unsupported arg %s=%r", k, kwargs.pop(k))
    return TrainingArguments(**kwargs)


@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    input_key: str = "input_values"
    padding: bool | str = True

    def __call__(self, features: List[dict]) -> dict:
        input_features = [{self.input_key: f[self.input_key]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


def train_ctc(cfg: "TrainConfig") -> dict[str, Any]:
    """Continued fine-tune a CTC checkpoint (e.g. Ethio-ASR 1B on Sagalee)."""
    import evaluate

    hub_ids = _hub_ids(cfg)
    if len(hub_ids) != 1:
        raise ValueError(
            "CTC trainer currently supports a single Hub dataset_id "
            f"(got {hub_ids}). Merge offline or call once per corpus."
        )
    dataset_id = hub_ids[0]
    out_dir = Path(cfg.output_dir) / cfg.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_col = cfg.audio_column
    text_candidates = [cfg.text_column, "transcription", "transcript", "text", "sentence"]
    lang_tag = _lang_tag(cfg.language or cfg.extra.get("language") or "oromo")
    prepend_lid = bool(cfg.extra.get("prepend_lang_token", True))
    max_seconds = float(cfg.extra.get("max_input_seconds", 30))
    min_seconds = float(cfg.extra.get("min_input_seconds", 0.5))
    max_train = cfg.extra.get("max_train_samples")
    max_eval = cfg.extra.get("max_eval_samples")

    token = _hf_token()
    load_kw: dict[str, Any] = {"token": token} if token else {}
    logger.info("Loading CTC model %s", cfg.model_id)
    processor = AutoProcessor.from_pretrained(cfg.model_id, **load_kw)
    model = AutoModelForCTC.from_pretrained(cfg.model_id, **load_kw)

    train_split = cfg.split or "train"
    eval_split = cfg.eval_split or "dev"
    logger.info("Loading %s train=%s eval=%s", dataset_id, train_split, eval_split)
    raw_train = load_dataset(dataset_id, split=train_split, **load_kw)
    try:
        raw_eval = load_dataset(dataset_id, split=eval_split, **load_kw)
    except Exception:
        alt = "validation" if eval_split == "dev" else "dev"
        logger.warning("Split %s missing; trying %s", eval_split, alt)
        raw_eval = load_dataset(dataset_id, split=alt, **load_kw)

    text_col = _pick_col(list(raw_train.column_names), text_candidates)
    if text_col is None:
        raise ValueError(
            f"No text column among {text_candidates}; columns={raw_train.column_names}"
        )
    if audio_col not in raw_train.column_names:
        audio_col = _pick_col(list(raw_train.column_names), ["audio", "speech"]) or audio_col
    if audio_col not in raw_train.column_names:
        raise ValueError(f"No audio column; columns={raw_train.column_names}")

    raw_train = raw_train.cast_column(audio_col, Audio(sampling_rate=16000))
    raw_eval = raw_eval.cast_column(audio_col, Audio(sampling_rate=16000))

    if max_train:
        raw_train = raw_train.select(range(min(int(max_train), len(raw_train))))
    if max_eval:
        raw_eval = raw_eval.select(range(min(int(max_eval), len(raw_eval))))

    probe = processor(np.zeros(16000, dtype=np.float32), sampling_rate=16000)
    input_key = "input_values" if "input_values" in probe else "input_features"

    def prepare_fn(batch: dict) -> dict:
        audio = batch[audio_col]
        array = np.asarray(audio["array"], dtype=np.float32)
        if array.ndim > 1:
            array = array.mean(axis=-1)
        sr = int(audio.get("sampling_rate") or 16000)
        batch["_duration"] = float(len(array) / float(sr or 16000))
        feats = processor(array, sampling_rate=sr)
        batch[input_key] = feats[input_key][0]
        text = str(batch[text_col] or "").strip()
        if prepend_lid and text and not _LID_RE.match(text):
            text = f"{lang_tag} {text}"
        batch["labels"] = processor.tokenizer(text, return_attention_mask=False).input_ids
        return batch

    def _ok(example: dict) -> bool:
        d = float(example.get("_duration") or 0.0)
        labels = example.get("labels")
        return min_seconds <= d <= max_seconds and bool(labels)

    logger.info(
        "Preparing features (lid=%s tag=%s max_s=%.1f)…",
        prepend_lid,
        lang_tag,
        max_seconds,
    )
    train_ds = raw_train.map(prepare_fn, remove_columns=raw_train.column_names, desc="prepare-train")
    train_ds = train_ds.filter(_ok, desc="filter-train-duration")
    eval_ds = raw_eval.map(prepare_fn, remove_columns=raw_eval.column_names, desc="prepare-eval")
    eval_ds = eval_ds.filter(_ok, desc="filter-eval-duration")
    logger.info("Train rows=%d eval rows=%d", len(train_ds), len(eval_ds))

    wer_metric = evaluate.load("wer")
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = getattr(processor.tokenizer, "pad_token_id", None) or 0

    def compute_metrics(pred: Any) -> dict[str, float]:
        pred_ids = np.argmax(pred.predictions, axis=-1)
        label_ids = np.where(pred.label_ids == -100, pad_id, pred.label_ids)
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        pred_str = [_LID_RE.sub("", s).strip() for s in pred_str]
        label_str = [_LID_RE.sub("", s).strip() for s in label_str]
        return {"wer": float(wer_metric.compute(predictions=pred_str, references=label_str))}

    use_bf16 = bool(cfg.extra.get("bf16", False))
    args = _make_training_arguments(
        output_dir=str(out_dir),
        # Optional on some transformers builds; dropped automatically if unsupported
        group_by_length=bool(cfg.extra.get("group_by_length", False)),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=int(
            cfg.extra.get("per_device_eval_batch_size", cfg.per_device_train_batch_size)
        ),
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        eval_strategy="steps" if len(eval_ds) else "no",
        save_strategy="steps",
        num_train_epochs=cfg.num_train_epochs,
        max_steps=int(cfg.max_steps) if cfg.max_steps else -1,
        fp16=bool(cfg.fp16) and not use_bf16,
        bf16=use_bf16,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=int(cfg.extra.get("eval_steps", cfg.save_steps)),
        save_total_limit=int(cfg.extra.get("save_total_limit", 2)),
        load_best_model_at_end=bool(len(eval_ds)),
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        report_to=list(cfg.extra.get("report_to") or []),
        dataloader_num_workers=int(cfg.extra.get("dataloader_num_workers", 2)),
        gradient_checkpointing=bool(cfg.extra.get("gradient_checkpointing", True)),
        remove_unused_columns=False,
    )

    if bool(cfg.extra.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    trainer_kw: dict[str, Any] = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if len(eval_ds) else None,
        data_collator=DataCollatorCTCWithPadding(processor=processor, input_key=input_key),
        compute_metrics=compute_metrics if len(eval_ds) else None,
    )
    try:
        trainer = Trainer(**trainer_kw, processing_class=processor)
    except TypeError:
        trainer = Trainer(**trainer_kw, tokenizer=processor)

    logger.info(
        "Starting CTC continued FT: model=%s data=%s lr=%s epochs=%s → %s",
        cfg.model_id,
        dataset_id,
        cfg.learning_rate,
        cfg.num_train_epochs,
        out_dir,
    )
    train_result = trainer.train()
    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))

    metrics: Dict[str, Any] = dict(train_result.metrics)
    if len(eval_ds):
        metrics.update(trainer.evaluate())

    summary = {
        "status": "ok",
        "backend": "ctc",
        "model_id": cfg.model_id,
        "dataset_id": dataset_id,
        "language_tag": lang_tag,
        "output_dir": str(out_dir),
        "metrics": metrics,
        "eval_hint": (
            f"python -m ndizi_asr.cli --model hf:{out_dir} "
            f"--dataset {dataset_id} --normalize ethio "
            f"--experiment-name {cfg.experiment_name}_eval"
        ),
        "cropconcept_hint": (
            f"python -m ndizi_asr.cli --model hf:{out_dir} ethio-asr-multilingual-1b "
            f"--config configs/cropconcept_oromo.yaml "
            f"--experiment-name cropconcept_ethio1b_sagalee_ft"
        ),
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Saved continued-FT checkpoint to %s", out_dir)
    return summary
