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


# Sunbird ISO codes → English names expected by Whisper.generate / tokenizer.set_prefix_tokens.
# (generate rejects "swa"; it wants "swahili".)
_WHISPER_GENERATE_LANG_NAME: Dict[str, str] = {
    "eng": "english",
    "fra": "french",
    "swa": "swahili",
    "sna": "shona",
    "yor": "yoruba",
    "som": "somali",
    "afr": "afrikaans",
    "amh": "amharic",
    "mlg": "malagasy",
    "lin": "lingala",
    "hau": "hausa",
    "xho": "xhosa",
    "zul": "zulu",
}


def _set_language_forcing(model: WhisperForConditionalGeneration, processor: WhisperProcessor, lang_code: str) -> None:
    """Force language at *generation* time; keep model.config.forced_decoder_ids cleared for training.

    Leaving forced_decoder_ids on ``model.config`` during Seq2SeqTrainer.train() makes newer
    transformers pass both decoder_input_ids (from labels) and decoder_inputs_embeds, which
    raises: ValueError: You cannot specify both decoder_input_ids and decoder_inputs_embeds.
    Language for training comes from tokenizer prefix tokens baked into labels instead.
    """
    tok = processor.tokenizer
    lang_id = LANGUAGE_TOKENS_WHISPER.get(lang_code)
    if lang_id is None:
        try:
            lang_id = tok.convert_tokens_to_ids(f"<|{lang_code}|>")
        except Exception:
            lang_id = None
    if lang_id is None or lang_id < 0:
        logger.warning("No language token for %s; leaving decoder unconstrained", lang_code)
        model.config.forced_decoder_ids = None
        return
    transcribe_id = tok.convert_tokens_to_ids("<|transcribe|>")
    notimestamps_id = tok.convert_tokens_to_ids("<|notimestamps|>")
    forced = [(1, lang_id), (2, transcribe_id), (3, notimestamps_id)]
    # Whisper.generate wants English names ("swahili"), not Sunbird ISO codes ("swa").
    whisper_lang_name = _WHISPER_GENERATE_LANG_NAME.get(lang_code)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.forced_decoder_ids = forced
        model.generation_config.language = whisper_lang_name  # None for Sunbird-only langs (e.g. orm)
        model.generation_config.task = "transcribe"
    # Critical: must be None during training forward (labels already carry the prefix).
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    prefix_lang = whisper_lang_name or lang_code
    try:
        tok.set_prefix_tokens(language=prefix_lang, task="transcribe", predict_timestamps=False)
    except Exception:  # noqa: BLE001
        try:
            tok.prefix_tokens = [lang_id, transcribe_id, notimestamps_id]
        except Exception:  # noqa: BLE001
            pass
    logger.info(
        "Whisper language forcing: sunbird=%s generate_lang=%s lang_token_id=%s",
        lang_code,
        whisper_lang_name,
        lang_id,
    )


def _drop_conflicting_decoder_embeds(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """PEFT/Trainer can inject decoder_inputs_embeds alongside decoder_input_ids (HF #39542)."""
    if kwargs.get("decoder_input_ids") is not None:
        kwargs = {k: v for k, v in kwargs.items() if k != "decoder_inputs_embeds"}
    return kwargs


def _guard_whisper_decoder_kwargs(model: Any) -> Any:
    """Wrap PeftModel / Whisper forwards so ids and embeds never both reach the decoder."""
    # 1) PeftModel.forward
    if hasattr(model, "get_base_model"):
        peft_fwd = model.forward

        def peft_forward(*args, **kwargs):
            return peft_fwd(*args, **_drop_conflicting_decoder_embeds(kwargs))

        model.forward = peft_forward  # type: ignore[method-assign]

        base = model.get_base_model()
    else:
        base = model

    # 2) WhisperForConditionalGeneration.forward
    if hasattr(base, "forward"):
        gen_fwd = base.forward

        def gen_forward(*args, **kwargs):
            return gen_fwd(*args, **_drop_conflicting_decoder_embeds(kwargs))

        base.forward = gen_forward  # type: ignore[method-assign]

    # 3) Inner WhisperModel.forward — also strip kwargs that leak into decoder(**kwargs)
    whisper_model = getattr(base, "model", None)
    if whisper_model is not None and hasattr(whisper_model, "forward"):
        inner_fwd = whisper_model.forward

        def inner_forward(*args, **kwargs):
            kwargs = _drop_conflicting_decoder_embeds(kwargs)
            kwargs.pop("inputs_embeds", None)
            # Prefer token ids when both somehow remain as explicit args on the model call
            if kwargs.get("decoder_input_ids") is not None:
                kwargs["decoder_inputs_embeds"] = None
            return inner_fwd(*args, **kwargs)

        whisper_model.forward = inner_forward  # type: ignore[method-assign]

    logger.info("Guarded Whisper forward against decoder_input_ids + decoder_inputs_embeds clash")
    return model


def _ensure_decoder_input_ids(model: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Whisper needs decoder_input_ids when Trainer label-smoothing pops ``labels``.

    In transformers≥4.x/5.x the decoder check is XOR-based, so *neither* ids nor embeds
    raises the misleading error: "cannot specify both decoder_input_ids and …embeds".
    """
    inputs = dict(inputs)
    inputs.pop("decoder_inputs_embeds", None)
    inputs.pop("inputs_embeds", None)
    if inputs.get("decoder_input_ids") is not None or "labels" not in inputs:
        return inputs
    from transformers.models.whisper.modeling_whisper import shift_tokens_right

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    cfg = base.config
    labels = inputs["labels"]
    # shift_tokens_right treats -100 poorly; map pad for the shift only
    shift_labels = labels.clone()
    shift_labels[shift_labels == -100] = cfg.pad_token_id
    inputs["decoder_input_ids"] = shift_tokens_right(
        shift_labels, cfg.pad_token_id, cfg.decoder_start_token_id
    )
    return inputs


class _WhisperSeq2SeqTrainer(Seq2SeqTrainer):
    """Whisper-safe Seq2SeqTrainer (decoder ids, generate kwargs)."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = _ensure_decoder_input_ids(model, _drop_conflicting_decoder_embeds(dict(inputs)))
        try:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        except TypeError:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        """Strip labels before generate — Whisper.generate rejects unused ``labels`` kwargs."""
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        # Seq2SeqTrainer keeps labels in generation_inputs; Whisper 5.x validate_model_kwargs errors.
        # Temporarily wrap generate to drop non-audio kwargs.
        base_generate = model.generate

        def _generate_no_labels(*args, **kwargs):
            for key in (
                "labels",
                "decoder_input_ids",
                "decoder_attention_mask",
                "decoder_inputs_embeds",
            ):
                kwargs.pop(key, None)
            return base_generate(*args, **kwargs)

        model.generate = _generate_no_labels  # type: ignore[method-assign]
        try:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                **gen_kwargs,
            )
        finally:
            model.generate = base_generate  # type: ignore[method-assign]

    def create_optimizer(self):
        """Support differential learning rate for Whisper encoder vs. decoder parameters."""
        enc_lr = getattr(self, "_encoder_learning_rate", None)
        if enc_lr is None or self.optimizer is not None:
            return super().create_optimizer()

        decay_parameters = self.get_decay_parameter_names(self.model)
        enc_decay, enc_nodecay = [], []
        dec_decay, dec_nodecay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_enc = ".encoder." in name or name.startswith("encoder.")
            in_decay = name in decay_parameters
            if is_enc and in_decay:
                enc_decay.append(param)
            elif is_enc:
                enc_nodecay.append(param)
            elif in_decay:
                dec_decay.append(param)
            else:
                dec_nodecay.append(param)

        wd = float(self.args.weight_decay)
        dec_lr = float(self.args.learning_rate)
        enc_lr_f = float(enc_lr)
        optimizer_grouped_parameters = [
            {"params": dec_decay, "weight_decay": wd, "lr": dec_lr},
            {"params": dec_nodecay, "weight_decay": 0.0, "lr": dec_lr},
            {"params": enc_decay, "weight_decay": wd, "lr": enc_lr_f},
            {"params": enc_nodecay, "weight_decay": 0.0, "lr": enc_lr_f},
        ]
        optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g["params"]]
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        logger.info(
            "Configured differential optimizer: decoder_lr=%.2e (%d tensors) | encoder_lr=%.2e (%d tensors)",
            dec_lr,
            len(dec_decay) + len(dec_nodecay),
            enc_lr_f,
            len(enc_decay) + len(enc_nodecay),
        )
        return self.optimizer

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        target_dir = Path(output_dir or self.args.output_dir)
        non_lora = {
            n: p.detach().cpu()
            for n, p in self.model.named_parameters()
            if p.requires_grad and "lora_" not in n
        }
        if non_lora:
            torch.save(non_lora, target_dir / "unfrozen_base_params.pt")

    def _load_best_model(self):
        super()._load_best_model()
        best_ckpt = getattr(self.state, "best_model_checkpoint", None)
        if best_ckpt:
            extra_pt = Path(best_ckpt) / "unfrozen_base_params.pt"
            if extra_pt.is_file():
                logger.info("Restoring unfrozen base/encoder parameters from %s", extra_pt)
                state = torch.load(extra_pt, map_location="cpu")
                self.model.load_state_dict(state, strict=False)

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        # Save before eval so a generate() crash does not lose the finished epoch.
        if getattr(self.control, "should_evaluate", False) and getattr(self.control, "should_save", False):
            trial = kwargs.get("trial", args[3] if len(args) > 3 else None)
            model = kwargs.get("model", args[2] if len(args) > 2 else self.model)
            self._save_checkpoint(model, trial)
            self.control.should_save = False
        return super()._maybe_log_save_evaluate(*args, **kwargs)


def _apply_lora(model: WhisperForConditionalGeneration, cfg: "TrainConfig"):
    from peft import LoraConfig, get_peft_model

    r = int(cfg.extra.get("lora_r", 16))
    alpha = int(cfg.extra.get("lora_alpha", 32))
    dropout = float(cfg.extra.get("lora_dropout", 0.05))
    unfreeze_top_k = int(cfg.extra.get("unfreeze_top_encoder_layers", 0))
    freeze_enc = bool(cfg.extra.get("freeze_encoder", False))

    target = cfg.extra.get("lora_target_modules") or [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
    ]
    # When unfreezing top-K encoder layers in full-rank mode (or strictly decoder-only LoRA),
    # restrict LoRA wrapping to decoder projections so encoder layers don't get dual-wrapped.
    if unfreeze_top_k > 0 or (freeze_enc and bool(cfg.extra.get("decoder_only_lora", False))):
        proj_pat = "|".join(str(x) for x in target)
        target_spec: Any = rf".*decoder\.layers\.\d+\..*({proj_pat})"
    else:
        target_spec = list(target)

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_spec,
        bias="none",
    )
    logger.info("Applying LoRA r=%s alpha=%s targets=%s", r, alpha, target_spec)
    model = get_peft_model(model, lora_cfg)

    if unfreeze_top_k > 0:
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        enc = getattr(getattr(base, "model", base), "encoder", None)
        dec = getattr(getattr(base, "model", base), "decoder", None)
        if enc is not None and hasattr(enc, "layers"):
            n_enc = len(enc.layers)
            k = min(unfreeze_top_k, n_enc)
            for layer in enc.layers[n_enc - k :]:
                for p in layer.parameters():
                    p.requires_grad = True
            if hasattr(enc, "layer_norm"):
                for p in enc.layer_norm.parameters():
                    p.requires_grad = True
            logger.info(
                "Unfroze top %d/%d Whisper encoder layers (layers %d..%d) + encoder.layer_norm",
                k,
                n_enc,
                n_enc - k,
                n_enc - 1,
            )
        if dec is not None and hasattr(dec, "layer_norm"):
            for p in dec.layer_norm.parameters():
                p.requires_grad = True

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

    out_dir = str(Path(cfg.output_dir) / cfg.experiment_name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    qc_dir = Path(out_dir) / "qc"
    chunk_long = bool(cfg.extra.get("chunk_long_audio", False))

    if bool(cfg.extra.get("apply_data_qc", True)):
        from ndizi_asr.data_qc import audit_and_filter_hf_split

        min_cps = float(cfg.extra.get("min_chars_per_sec", 2.5))
        max_cps = float(cfg.extra.get("max_chars_per_sec", 26.0))
        train_ds, _ = audit_and_filter_hf_split(
            train_ds,
            split_name="train",
            audio_col=audio_col,
            text_col=text_col,
            max_duration_s=max_input_s,
            min_chars_per_sec=min_cps,
            max_chars_per_sec=max_cps,
            chunk_long_audio=chunk_long,
            apply_filter=True,
            repair_glued_punctuation=True,
            report_dir=qc_dir,
        )
        if eval_ds is not None:
            eval_ds, _ = audit_and_filter_hf_split(
                eval_ds,
                split_name="validation",
                audio_col=audio_col,
                text_col=text_col,
                max_duration_s=max_input_s,
                min_chars_per_sec=min_cps,
                max_chars_per_sec=max_cps,
                chunk_long_audio=chunk_long and bool(cfg.extra.get("chunk_long_audio_eval", True)),
                apply_filter=bool(cfg.extra.get("filter_eval_outliers", False)),
                repair_glued_punctuation=True,
                report_dir=qc_dir,
            )

    if chunk_long:
        from .long_audio_chunker import expand_long_audio_dataset

        chunk_max_s = float(cfg.extra.get("chunk_max_seconds", 28.0))
        train_ds = expand_long_audio_dataset(
            train_ds,
            audio_col=audio_col,
            text_col=text_col,
            max_chunk_seconds=chunk_max_s,
            split_label="train",
        )
        if eval_ds is not None and bool(cfg.extra.get("chunk_long_audio_eval", True)):
            eval_ds = expand_long_audio_dataset(
                eval_ds,
                audio_col=audio_col,
                text_col=text_col,
                max_chunk_seconds=chunk_max_s,
                split_label="validation",
            )

    token = _hf_token()
    load_kw: Dict[str, Any] = {}
    if token:
        load_kw["token"] = token
    try:
        processor = WhisperProcessor.from_pretrained(cfg.model_id, **load_kw)
    except Exception as exc:
        msg = str(exc).lower()
        if "gated" in msg or "401" in msg or "restricted" in msg or "unauthorized" in msg:
            raise _gated_access_error(cfg.model_id, exc) from exc
        raise

    # Configure tokenizer prefix tokens before feature/label mapping
    whisper_lang_name = _WHISPER_GENERATE_LANG_NAME.get(lang_code)
    lang_id = LANGUAGE_TOKENS_WHISPER.get(lang_code)
    if lang_id is not None:
        transcribe_id = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        notimestamps_id = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        try:
            processor.tokenizer.set_prefix_tokens(
                language=whisper_lang_name or lang_code,
                task="transcribe",
                predict_timestamps=False,
            )
        except Exception:  # noqa: BLE001
            try:
                processor.tokenizer.prefix_tokens = [lang_id, transcribe_id, notimestamps_id]
            except Exception:  # noqa: BLE001
                pass

    from ndizi_asr.ethio_normalize import clean_swahili_training_target
    from ndizi_asr.metrics import clean_glued_punctuation

    canon_targets = bool(cfg.extra.get("canonicalize_train_text", False))

    def prepare(batch: dict) -> dict:
        audio = batch[audio_col]
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        dur = len(arr) / max(sr, 1)
        raw_txt = str(batch.get(text_col) or "")
        clean_txt = (
            clean_swahili_training_target(raw_txt)
            if canon_targets
            else clean_glued_punctuation(raw_txt)
        )
        batch["_drop"] = dur > max_input_s or not clean_txt
        batch["input_features"] = processor.feature_extractor(arr, sampling_rate=sr).input_features[0]
        batch["labels"] = processor.tokenizer(
            clean_txt,
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

    try:
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
        logger.info("Freezing Whisper acoustic encoder (before selective unfreeze / LoRA)")
        model.freeze_encoder()

    if cfg.lora:
        model = _apply_lora(model, cfg)
    # Re-assert after PEFT wrap (some peft/transformers versions restore forced ids).
    model.config.forced_decoder_ids = None
    if hasattr(model, "config"):
        model.config.suppress_tokens = []
        if bool(cfg.extra.get("apply_spec_augment", False)):
            model.config.apply_spec_augment = True
            model.config.mask_time_prob = float(cfg.extra.get("mask_time_prob", 0.05))
            model.config.mask_feature_prob = float(cfg.extra.get("mask_feature_prob", 0.04))
            logger.info(
                "Enabled Whisper SpecAugment (mask_time_prob=%.3f, mask_feature_prob=%.3f)",
                model.config.mask_time_prob,
                model.config.mask_feature_prob,
            )
    if cfg.extra.get("gradient_checkpointing", True):
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    # PEFT + Trainer can inject decoder_inputs_embeds alongside decoder_input_ids (HF #39542).
    model = _guard_whisper_decoder_kwargs(model)
    # Re-apply generate language after PEFT wrap (eval uses model.generate).
    _set_language_forcing(
        model.get_base_model() if hasattr(model, "get_base_model") else model,
        processor,
        lang_code,
    )
    model.config.forced_decoder_ids = None

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
        # Whisper + label_smoothing pops labels before forward; without decoder_input_ids
        # transformers 5.x raises a misleading "both ids and embeds" XOR error. Default off.
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
        trainer = _WhisperSeq2SeqTrainer(**trainer_kwargs, processing_class=processor)
    except TypeError:
        trainer = _WhisperSeq2SeqTrainer(**trainer_kwargs, tokenizer=processor)
    trainer._encoder_learning_rate = cfg.extra.get("encoder_learning_rate")  # type: ignore[attr-defined]

    logger.info(
        "Starting train: n_train=%s n_eval=%s lora=%s model=%s",
        len(train_ds),
        len(eval_ds) if eval_ds is not None else 0,
        cfg.lora,
        cfg.model_id,
    )
    resume_ckpt = cfg.extra.get("resume_from_checkpoint")
    if resume_ckpt is True or resume_ckpt == "auto":
        # Resume from latest checkpoint under output_dir if present
        resume_ckpt = True
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt if resume_ckpt else None)
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
