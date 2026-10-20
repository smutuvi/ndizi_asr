"""Whisper / Sunbird LoRA fine-tuning via Hugging Face Seq2SeqTrainer."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

os.environ.setdefault("HF_DATASETS_DISABLE_TORCHCODEC", "1")

import numpy as np
import torch
from datasets import Audio, Dataset, concatenate_datasets, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from ndizi_asr.backends.sunbird_langs import (
    LANGUAGE_TOKENS_WHISPER,
    resolve_sunbird_language_code,
)

if TYPE_CHECKING:
    from . import TrainConfig

logger = logging.getLogger(__name__)


def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def _hub_ids(cfg: "TrainConfig") -> List[str]:
    extra_ids = cfg.extra.get("dataset_ids") or cfg.extra.get("train_datasets")
    if extra_ids:
        return [str(x) for x in extra_ids]
    if cfg.dataset_id:
        return [str(cfg.dataset_id)]
    raise ValueError("Provide dataset_id or extra.dataset_ids for Hub training")


def _hf_token() -> Optional[str]:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return None


def _gated_access_error(model_id: str, exc: Exception) -> OSError:
    token = _hf_token()
    auth_hint = (
        "HF_TOKEN is set, but access was still denied — accept terms at "
        f"https://huggingface.co/{model_id} with the same account."
        if token
        else (
            "No HF_TOKEN found. Put HF_TOKEN=hf_... in .env (after accepting model "
            "terms) or run: huggingface-cli login"
        )
    )
    return OSError(
        f"Cannot download gated model {model_id!r}.\n"
        f"1) Open https://huggingface.co/{model_id} while logged in and accept access.\n"
        f"2) {auth_hint}\n"
        f"Original error: {exc}"
    )


def _load_merged_split(hub_ids: List[str], split: str, *, audio_col: str) -> Dataset:
    parts = []
    for did in hub_ids:
        logger.info("Loading %s split=%s", did, split)
        ds = load_dataset(did, split=split)
        ds = ds.cast_column(audio_col, Audio(sampling_rate=16000))
        parts.append(ds)
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[dict]) -> dict:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def _set_language_forcing(model: WhisperForConditionalGeneration, processor: WhisperProcessor, lang_code: str) -> None:
    tok = processor.tokenizer
    lang_id = LANGUAGE_TOKENS_WHISPER.get(lang_code)
    if lang_id is None:
        try:
            lang_id = tok.convert_tokens_to_ids(f"<|{lang_code}|>")
        except Exception:
            lang_id = None
    if lang_id is None or lang_id < 0:
        logger.warning("No language token for %s; leaving decoder unconstrained", lang_code)
        return
    transcribe_id = tok.convert_tokens_to_ids("<|transcribe|>")
    notimestamps_id = tok.convert_tokens_to_ids("<|notimestamps|>")
    forced = [(1, lang_id), (2, transcribe_id), (3, notimestamps_id)]
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.forced_decoder_ids = forced
        model.generation_config.task = "transcribe"
    model.config.forced_decoder_ids = forced
    # Ensure training labels use the exact same language + transcribe prefix token IDs
    try:
        tok.prefix_tokens = [lang_id, transcribe_id, notimestamps_id]
    except Exception:  # noqa: BLE001
        pass


def _apply_lora(model: WhisperForConditionalGeneration, cfg: "TrainConfig"):
    from peft import LoraConfig, get_peft_model

    r = int(cfg.extra.get("lora_r", 16))
    alpha = int(cfg.extra.get("lora_alpha", 32))
    dropout = float(cfg.extra.get("lora_dropout", 0.05))
    target = cfg.extra.get("lora_target_modules") or [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
    ]
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target),
        bias="none",
    )
    logger.info("Applying LoRA r=%s alpha=%s targets=%s", r, alpha, target)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def train_whisper(cfg: "TrainConfig") -> Dict[str, Any]:
    """Fine-tune Whisper (optionally LoRA) on Hub datasets such as ndizi-1."""
    import evaluate

    hub_ids = _hub_ids(cfg)
    train_split = cfg.split or "train"
    eval_split = cfg.eval_split or "validation"
    max_input_s = float(cfg.extra.get("max_input_seconds", 30.0))
    label_max = int(cfg.extra.get("label_max_length", 448))
    lang_raw = cfg.language or cfg.extra.get("language") or "swahili"
    try:
        lang_code = resolve_sunbird_language_code(str(lang_raw))
    except ValueError:
        lang_code = str(lang_raw).strip().lower()

    probe = load_dataset(hub_ids[0], split=train_split)
    audio_col = cfg.audio_column or _pick_col(
        list(probe.column_names), ["audio", "audio_path", "path", "file", "wav", "speech"]
    )
    text_col = cfg.text_column or _pick_col(
        list(probe.column_names),
        ["text", "transcript", "sentence", "transcription", "normalized_text"],
    )
    if not audio_col or not text_col:
        raise ValueError(f"Need audio+text columns; have {probe.column_names}")
    logger.info("Columns audio=%s text=%s datasets=%s lang=%s", audio_col, text_col, hub_ids, lang_code)

    train_ds = _load_merged_split(hub_ids, train_split, audio_col=audio_col)
    eval_ds = None
    if eval_split:
        try:
            eval_ds = _load_merged_split(hub_ids, eval_split, audio_col=audio_col)
        except Exception as exc:
            logger.warning("Could not load eval split %s: %s", eval_split, exc)

    token = _hf_token()
    load_kw: Dict[str, Any] = {}
    if token:
        load_kw["token"] = token
    try:
        processor = WhisperProcessor.from_pretrained(cfg.model_id, **load_kw)
        model = WhisperForConditionalGeneration.from_pretrained(cfg.model_id, **load_kw)
    except Exception as exc:
        msg = str(exc).lower()
        if "gated" in msg or "401" in msg or "restricted" in msg or "unauthorized" in msg:
            raise _gated_access_error(cfg.model_id, exc) from exc
        raise
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.forced_decoder_ids = None
    _set_language_forcing(model, processor, lang_code)

    if bool(cfg.extra.get("freeze_encoder", False)) and hasattr(model, "freeze_encoder"):
        logger.info("Freezing Whisper acoustic encoder (decoder-only adaptation)")
        model.freeze_encoder()

    if cfg.lora:
        model = _apply_lora(model, cfg)
    if cfg.extra.get("gradient_checkpointing", True):
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    def prepare(batch: dict) -> dict:
        audio = batch[audio_col]
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        dur = len(arr) / max(sr, 1)
        batch["_drop"] = dur > max_input_s or not str(batch.get(text_col) or "").strip()
        batch["input_features"] = processor.feature_extractor(arr, sampling_rate=sr).input_features[0]
        batch["labels"] = processor.tokenizer(
            str(batch[text_col]),
            truncation=True,
            max_length=label_max,
        ).input_ids
        return batch

    cols = list(train_ds.column_names)
    train_ds = (
        train_ds.map(prepare, remove_columns=cols, num_proc=1)
        .filter(lambda x: not x["_drop"])
        .remove_columns(["_drop"])
    )
    if eval_ds is not None:
        ecols = list(eval_ds.column_names)
        eval_ds = (
            eval_ds.map(prepare, remove_columns=ecols, num_proc=1)
            .filter(lambda x: not x["_drop"])
            .remove_columns(["_drop"])
        )

    out_dir = str(Path(cfg.output_dir) / cfg.experiment_name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    from jiwer import cer as jiwer_cer, wer as jiwer_wer

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pairs = [(r.strip(), p.strip()) for r, p in zip(label_str, pred_str) if r.strip()]
        if not pairs:
            return {"wer": 0.0, "cer": 0.0}
        refs, hyps = zip(*pairs)
        return {
            "wer": float(jiwer_wer(list(refs), list(hyps))),
            "cer": float(jiwer_cer(list(refs), list(hyps))),
        }

    use_bf16 = bool(cfg.extra.get("bf16", False))
    use_fp16 = bool(cfg.fp16) and not use_bf16 and torch.cuda.is_available()
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=int(
            cfg.extra.get("per_device_eval_batch_size", cfg.per_device_train_batch_size)
        ),
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=-1 if cfg.max_steps is None else int(cfg.max_steps),
        fp16=use_fp16,
        bf16=use_bf16,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        logging_steps=cfg.logging_steps,
        predict_with_generate=True,
        generation_max_length=min(label_max, 444),
        label_smoothing_factor=float(cfg.extra.get("label_smoothing_factor", 0.0)),
        lr_scheduler_type=str(cfg.extra.get("lr_scheduler_type", "cosine")),
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="wer" if eval_ds is not None else None,
        greater_is_better=False if eval_ds is not None else None,
        save_total_limit=int(cfg.extra.get("save_total_limit", 2)),
        dataloader_num_workers=int(cfg.extra.get("dataloader_num_workers", 2)),
    )

    trainer_kwargs: Dict[str, Any] = dict(
        args=args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics if eval_ds is not None else None,
    )
    # transformers API: processing_class (new) vs tokenizer (old)
    try:
        trainer = Seq2SeqTrainer(**trainer_kwargs, processing_class=processor)
    except TypeError:
        trainer = Seq2SeqTrainer(**trainer_kwargs, tokenizer=processor)

    logger.info(
        "Starting train: n_train=%s n_eval=%s lora=%s model=%s",
        len(train_ds),
        len(eval_ds) if eval_ds is not None else 0,
        cfg.lora,
        cfg.model_id,
    )
    train_result = trainer.train()
    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)
    if cfg.lora and bool(cfg.extra.get("save_merged", True)) and hasattr(model, "merge_and_unload"):
        merged_dir = Path(out_dir) / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving merged standalone Whisper model to %s", merged_dir)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir))
        processor.save_pretrained(str(merged_dir))

    metrics = dict(train_result.metrics)
    if eval_ds is not None:
        metrics.update(trainer.evaluate())

    return {
        "status": "ok",
        "output_dir": out_dir,
        "model_id": cfg.model_id,
        "datasets": hub_ids,
        "language": lang_code,
        "lora": cfg.lora,
        "n_train": len(train_ds),
        "n_eval": len(eval_ds) if eval_ds is not None else 0,
        "metrics": metrics,
        "eval_hint": (
            f"python -m ndizi_asr.cli --model hf:{out_dir} "
            f"--dataset smutuvi/ndizi-1 --max-samples 50 "
            f"--experiment-name sunbird51_ndizi_lora_eval"
        ),
    }
