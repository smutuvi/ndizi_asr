"""Stage-2 Group Relative Policy Optimization (GRPO) for Whisper / SunflowerASR.

Optimizes sequence-level Word Error Rate (WER), Character Error Rate (CER),
domain glossary fidelity, and anti-hallucination length penalties directly on top
of a Stage-1 SFT / LoRA Whisper checkpoint.

Key efficiency features:
- Freezes the Whisper audio encoder and caches ``encoder_outputs`` once per clip,
  reusing them across all G rollouts, the policy forward pass, and the reference
  policy forward pass.
- Uses ``model.disable_adapter()`` on the PEFT LoRA model as the exact zero-memory
  reference policy pi_ref for KL regularization.
- Combines group-normalized GRPO policy gradient + Schulman KL penalty with a
  lightweight teacher-forced CE anchor (``sft_anchor_weight``) to prevent
  catastrophic forgetting.
"""

from __future__ import annotations

import logging
import math
import os
import random
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_DATASETS_DISABLE_TORCHCODEC", "1")

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Audio, concatenate_datasets, load_dataset
from jiwer import cer as jiwer_cer, wer as jiwer_wer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.modeling_outputs import BaseModelOutput

from ndizi_asr.backends.sunbird_langs import (
    LANGUAGE_TOKENS_WHISPER,
    resolve_sunbird_language_code,
)

from .whisper_trainer import (
    _gated_access_error,
    _hf_token,
    _hub_ids,
    _pick_col,
)

if TYPE_CHECKING:
    from . import TrainConfig

logger = logging.getLogger(__name__)

# High-value Ndizi Swahili survey terms & Bantu relative verbs whose exact
# orthography strongly impacts WER and downstream insight extraction.
DEFAULT_SWAHILI_SURVEY_TERMS: Tuple[str, ...] = (
    "unachokipenda",
    "unachokiona",
    "kisichokupendeza",
    "ninachokipenda",
    "nilichokipenda",
    "nisichokipenda",
    "usichokipenda",
    "kinachonipendeza",
    "kwa sababu",
    "kwa hiyo",
    "vile vile",
    "aina a",
    "aina b",
    "aina c",
    "maharage",
    "mvua",
    "ukame",
    "maua",
    "uzao",
    "imekauka",
    "imenyauka",
    "mbolea",
    "mbegu",
    "wadudu",
    "magonjwa",
    "afya",
    "majani",
    "mizizi",
    "kupanda",
    "kuvuna",
    "mavuno",
    "sokoni",
    "radha",
    "ladha",
)

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def _norm_for_reward(text: str) -> str:
    t = (text or "").lower().strip()
    t = _PUNCT_RE.sub(" ", t)
    return _SPACE_RE.sub(" ", t).strip()


def _glossary_f1(ref_norm: str, hyp_norm: str, terms: Tuple[str, ...]) -> float:
    """F1 on survey glossary terms/phrases present in the reference."""
    if not ref_norm:
        return 1.0
    ref_padded = f" {ref_norm} "
    hyp_padded = f" {hyp_norm} "
    ref_hits = {t for t in terms if f" {t} " in ref_padded}
    if not ref_hits:
        return 1.0
    hyp_hits = {t for t in terms if f" {t} " in hyp_padded}
    tp = len(ref_hits & hyp_hits)
    if tp == 0:
        return 0.0
    prec = tp / max(len(hyp_hits), 1)
    rec = tp / max(len(ref_hits), 1)
    return 2.0 * prec * rec / max(prec + rec, 1e-8)


def _ngram_repeat_penalty(hyp_norm: str, n: int = 4) -> float:
    words = hyp_norm.split()
    if len(words) < n * 2:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    unique_ratio = len(set(ngrams)) / float(len(ngrams))
    return max(0.0, 0.85 - unique_ratio) * 2.0


def compute_ndizi_reward(
    ref: str,
    hyp: str,
    *,
    wer_weight: float = 0.70,
    cer_weight: float = 0.30,
    glossary_weight: float = 0.15,
    length_penalty_weight: float = 0.20,
    terms: Tuple[str, ...] = DEFAULT_SWAHILI_SURVEY_TERMS,
) -> float:
    """Composite sequence-level reward in [-2.0, +1.15] (higher is better)."""
    ref_n = _norm_for_reward(ref)
    hyp_n = _norm_for_reward(hyp)
    if not ref_n:
        return 0.0
    if not hyp_n:
        return -1.5

    w = min(float(jiwer_wer(ref_n, hyp_n)), 2.0)
    c = min(float(jiwer_cer(ref_n, hyp_n)), 2.0)
    g_f1 = _glossary_f1(ref_n, hyp_n, terms)

    len_ratio = len(hyp_n) / float(max(len(ref_n), 1))
    over_len = max(0.0, len_ratio - 1.25)
    under_len = max(0.0, 0.55 - len_ratio)
    rep_pen = _ngram_repeat_penalty(hyp_n)

    reward = (
        1.0
        - (wer_weight * w + cer_weight * c)
        + glossary_weight * g_f1
        - length_penalty_weight * (over_len + under_len + rep_pen)
    )
    return float(reward)


class _SemanticEmbeddingScorer:
    """Dense semantic similarity scorer (EmbeddingGemma / Multilingual-E5) for GRPO rewards."""

    def __init__(
        self,
        model_id: str = "google/embeddinggemma-300m",
        fallback_model_id: str = "intfloat/multilingual-e5-base",
        device: Optional[torch.device] = None,
        token: Optional[str] = None,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = None
        self.tokenizer = None
        self.loaded_id = ""

        kw: Dict[str, Any] = {"token": token} if token else {}
        for candidate in (model_id, fallback_model_id):
            if not candidate:
                continue
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(candidate, **kw)
                self.model = AutoModel.from_pretrained(
                    candidate, torch_dtype=self.dtype, **kw
                ).to(self.device)
                self.model.eval()
                for p in self.model.parameters():
                    p.requires_grad = False
                self.loaded_id = candidate
                logger.info("Loaded semantic embedding reward model %s on %s", candidate, self.device)
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load semantic model %s (%s); trying fallback", candidate, exc)

    @property
    def available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    @torch.inference_mode()
    def score_candidates(self, ref: str, hyps: List[str]) -> np.ndarray:
        """Return cosine similarities in [0.0, 1.0] between ``ref`` and each hypothesis."""
        if not self.available or not hyps:
            return np.zeros(len(hyps), dtype=np.float32)
        ref_n = _norm_for_reward(ref)
        hyps_n = [_norm_for_reward(h) for h in hyps]
        if not ref_n:
            return np.zeros(len(hyps), dtype=np.float32)

        texts = [ref_n] + [h if h else " " for h in hyps_n]
        if "e5" in self.loaded_id.lower():
            texts = [f"passage: {t}" for t in texts]

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**enc)
        hidden = out.last_hidden_state.float()  # (1+G, T, D)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        normed = F.normalize(pooled, p=2, dim=-1)
        ref_vec = normed[0:1]  # (1, D)
        hyp_vecs = normed[1:]  # (G, D)
        cos = (hyp_vecs * ref_vec).sum(dim=-1).clamp(min=0.0, max=1.0)
        sims = cos.detach().cpu().numpy().astype(np.float32)
        for i, h in enumerate(hyps_n):
            if not h:
                sims[i] = 0.0
        return sims


def _forced_prefix_ids(processor: WhisperProcessor, lang_code: str) -> List[int]:
    tok = processor.tokenizer
    lang_id = LANGUAGE_TOKENS_WHISPER.get(lang_code)
    if lang_id is None:
        lang_id = tok.convert_tokens_to_ids(f"<|{lang_code}|>")
    transcribe_id = tok.convert_tokens_to_ids("<|transcribe|>")
    notimestamps_id = tok.convert_tokens_to_ids("<|notimestamps|>")
    return [int(tok.bos_token_id or 50258), int(lang_id), int(transcribe_id), int(notimestamps_id)]


def _sequence_logprobs(
    model: Any,
    encoder_last_hidden_state: torch.Tensor,
    seq_ids: torch.Tensor,
    *,
    prefix_len: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token log-probabilities for ``seq_ids[:, prefix_len:]``.

    Args:
        encoder_last_hidden_state: (1, T_enc, D) cached encoder output.
        seq_ids: (B, L) full decoder sequence starting with ``<|startoftranscript|>``.
        prefix_len: number of initial prompt tokens (e.g. 4: SOT, lang, task, notimestamps).
    Returns:
        token_logprobs: (B, L - prefix_len) log P(y_t | y_<t, x)
        valid_mask: (B, L - prefix_len) float mask (1.0 for non-pad tokens up to first EOS)
    """
    B = seq_ids.size(0)
    enc_expanded = encoder_last_hidden_state.expand(B, -1, -1)
    dec_in = seq_ids[:, :-1]
    targets = seq_ids[:, 1:]

    enc_out = BaseModelOutput(last_hidden_state=enc_expanded)
    outputs = model(
        encoder_outputs=enc_out,
        decoder_input_ids=dec_in,
        use_cache=False,
    )
    logits = outputs.logits  # (B, L-1, V)
    # Slice from (prefix_len - 1) so the first predicted target is seq_ids[:, prefix_len]
    start = max(prefix_len - 1, 0)
    logits_slice = logits[:, start:, :]
    targets_slice = targets[:, start:]

    log_probs = F.log_softmax(logits_slice.float(), dim=-1)
    tok_lp = torch.gather(log_probs, dim=-1, index=targets_slice.unsqueeze(-1)).squeeze(-1)

    # Mask out padding tokens after the first pad/eos token
    is_pad = targets_slice.eq(pad_token_id)
    # Include the first EOS token itself so the model learns when to stop, mask everything after
    eos_cumsum = is_pad.cumsum(dim=-1)
    valid_mask = (eos_cumsum <= 1).to(tok_lp.dtype)
    return tok_lp * valid_mask, valid_mask


def _evaluate_val(
    model: Any,
    processor: WhisperProcessor,
    eval_items: List[Dict[str, Any]],
    forced_decoder_ids: List[Tuple[int, int]],
    device: torch.device,
    model_dtype: torch.dtype,
    max_eval_samples: int = 256,
) -> Dict[str, float]:
    model.eval()
    refs: List[str] = []
    hyps: List[str] = []
    subset = eval_items[:max_eval_samples]
    with torch.inference_mode():
        for item in subset:
            feats = torch.tensor(item["input_features"], dtype=model_dtype, device=device).unsqueeze(0)
            pred_ids = model.generate(
                input_features=feats,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=224,
                num_beams=1,
                do_sample=False,
            )
            hyp = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
            r_n = _norm_for_reward(item["text"])
            h_n = _norm_for_reward(hyp)
            if r_n:
                refs.append(r_n)
                hyps.append(h_n)
    model.train()
    if not refs:
        return {"eval_wer": 0.0, "eval_cer": 0.0}
    return {
        "eval_wer": float(jiwer_wer(refs, hyps)),
        "eval_cer": float(jiwer_cer(refs, hyps)),
    }


def train_whisper_grpo(cfg: "TrainConfig") -> Dict[str, Any]:
    """Run Stage-2 GRPO on a Whisper / SunflowerASR checkpoint."""
    from peft import LoraConfig, PeftModel, get_peft_model

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

    # GRPO hyperparameters
    group_size = int(cfg.extra.get("group_size", 6))
    temperature = float(cfg.extra.get("temperature", 0.75))
    top_p = float(cfg.extra.get("top_p", 0.95))
    kl_beta = float(cfg.extra.get("kl_beta", 0.04))
    sft_anchor_weight = float(cfg.extra.get("sft_anchor_weight", 0.15))
    wer_weight = float(cfg.extra.get("wer_weight", 0.70))
    cer_weight = float(cfg.extra.get("cer_weight", 0.30))
    glossary_weight = float(cfg.extra.get("glossary_weight", 0.15))
    length_penalty_weight = float(cfg.extra.get("length_penalty_weight", 0.20))
    semantic_reward_weight = float(cfg.extra.get("semantic_reward_weight", 0.20))
    semantic_model_id = str(cfg.extra.get("semantic_model_id", "google/embeddinggemma-300m"))
    semantic_fallback_id = str(cfg.extra.get("semantic_fallback_model_id", "intfloat/multilingual-e5-base"))
    quality_gate = bool(cfg.extra.get("quality_gate", True))
    min_learnable_wer = float(cfg.extra.get("min_learnable_wer", 0.04))
    max_learnable_wer = float(cfg.extra.get("max_learnable_wer", 0.85))
    max_corrupted_cer = float(cfg.extra.get("max_corrupted_cer", 0.45))
    max_new_tokens = int(cfg.extra.get("max_new_tokens", 200))
    max_train_samples = cfg.extra.get("max_train_samples")
    max_eval_samples = int(cfg.extra.get("max_eval_samples", 256))
    eval_every_steps = int(cfg.extra.get("eval_every_steps", 250))
    seed = int(cfg.extra.get("seed", 42))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.float16 if (cfg.fp16 and device.type == "cuda") else torch.float32

    token = _hf_token()
    load_kw: Dict[str, Any] = {"token": token} if token else {}

    logger.info("Loading base/SFT Whisper model %s for Stage-2 GRPO on %s", cfg.model_id, device)
    local_dir = Path(cfg.model_id).expanduser()
    try:
        if local_dir.is_dir() and (local_dir / "adapter_config.json").is_file():
            import json

            adata = json.loads((local_dir / "adapter_config.json").read_text(encoding="utf-8"))
            base_id = str(adata.get("base_model_name_or_path") or "Sunbird/SunflowerASR-51-african-languages")
            processor = WhisperProcessor.from_pretrained(cfg.model_id, **load_kw)
            base_m = WhisperForConditionalGeneration.from_pretrained(
                base_id, torch_dtype=model_dtype, **load_kw
            )
            peft_m = PeftModel.from_pretrained(base_m, cfg.model_id, **load_kw)
            base_model = peft_m.merge_and_unload()
        else:
            processor = WhisperProcessor.from_pretrained(cfg.model_id, **load_kw)
            base_model = WhisperForConditionalGeneration.from_pretrained(
                cfg.model_id, torch_dtype=model_dtype, **load_kw
            )
    except Exception as exc:
        msg = str(exc).lower()
        if "gated" in msg or "401" in msg:
            raise _gated_access_error(cfg.model_id, exc) from exc
        raise

    # Clear generation_config.language/task so explicit forced_decoder_ids works cleanly
    if hasattr(base_model, "generation_config") and base_model.generation_config is not None:
        base_model.generation_config.language = None
        base_model.generation_config.task = None
        base_model.generation_config.forced_decoder_ids = None
    base_model.config.forced_decoder_ids = None
    base_model.config.suppress_tokens = []
    base_model.config.use_cache = False

    if hasattr(base_model, "freeze_encoder"):
        logger.info("Freezing Whisper encoder for Stage-2 GRPO (decoder policy optimization)")
        base_model.freeze_encoder()

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
    model = get_peft_model(base_model, lora_cfg).to(device)
    model.print_trainable_parameters()

    prefix_ids = _forced_prefix_ids(processor, lang_code)
    forced_decoder_ids = [
        (1, prefix_ids[1]),
        (2, prefix_ids[2]),
        (3, prefix_ids[3]),
    ]
    pad_token_id = int(processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id or 50257)
    eos_token_id = int(processor.tokenizer.eos_token_id or 50257)

    out_dir = Path(cfg.output_dir) / cfg.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    parts_tr = []
    for did in hub_ids:
        ds = load_dataset(did, split=train_split)
        ds = ds.cast_column(cfg.audio_column or "audio", Audio(sampling_rate=16000))
        parts_tr.append(ds)
    raw_train = parts_tr[0] if len(parts_tr) == 1 else concatenate_datasets(parts_tr)

    audio_col = cfg.audio_column or _pick_col(list(raw_train.column_names), ["audio", "path"])
    text_col = cfg.text_column or _pick_col(list(raw_train.column_names), ["transcript", "text"])
    chunk_long = bool(cfg.extra.get("chunk_long_audio", False))

    if bool(cfg.extra.get("apply_data_qc", True)):
        from ndizi_asr.data_qc import audit_and_filter_hf_split

        min_cps = float(cfg.extra.get("min_chars_per_sec", 2.5))
        max_cps = float(cfg.extra.get("max_chars_per_sec", 26.0))
        raw_train, _ = audit_and_filter_hf_split(
            raw_train,
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

    if chunk_long:
        from .long_audio_chunker import expand_long_audio_dataset

        chunk_max_s = float(cfg.extra.get("chunk_max_seconds", 28.0))
        raw_train = expand_long_audio_dataset(
            raw_train,
            audio_col=audio_col,
            text_col=text_col,
            max_chunk_seconds=chunk_max_s,
            split_label="train",
        )

    from ndizi_asr.metrics import clean_glued_punctuation

    def _prepare_list(ds_split, cap: Optional[int] = None) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for row_idx, row in enumerate(ds_split):
            txt = clean_glued_punctuation(str(row.get(text_col) or "").strip())
            if not txt:
                continue
            audio = row[audio_col]
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
            dur = len(arr) / max(sr, 1)
            if dur > max_input_s or dur < 0.4:
                continue
            feats = processor.feature_extractor(arr, sampling_rate=sr).input_features[0]
            tok_ids = processor.tokenizer(txt, truncation=True, max_length=label_max - 8).input_ids
            # Strip auto-prepended BOS/prefix if present so we can prepend exact prefix_ids
            clean_toks = [
                int(t)
                for t in tok_ids
                if int(t) not in set(prefix_ids) and int(t) != eos_token_id
            ]
            ref_seq = prefix_ids + clean_toks + [eos_token_id]
            items.append(
                {
                    "row_index": row_idx,
                    "input_features": feats,
                    "text": txt,
                    "ref_seq": ref_seq,
                }
            )
            if cap is not None and len(items) >= int(cap):
                break
        return items

    logger.info("Preparing GRPO training & validation examples...")
    train_items = _prepare_list(raw_train, cap=int(max_train_samples) if max_train_samples else None)
    eval_items: List[Dict[str, Any]] = []
    if eval_split:
        try:
            raw_eval = load_dataset(hub_ids[0], split=eval_split)
            raw_eval = raw_eval.cast_column(audio_col, Audio(sampling_rate=16000))
            if bool(cfg.extra.get("apply_data_qc", True)):
                from ndizi_asr.data_qc import audit_and_filter_hf_split

                raw_eval, _ = audit_and_filter_hf_split(
                    raw_eval,
                    split_name="validation",
                    audio_col=audio_col,
                    text_col=text_col,
                    max_duration_s=max_input_s,
                    chunk_long_audio=False,
                    apply_filter=False,
                    repair_glued_punctuation=True,
                    report_dir=qc_dir,
                )
            eval_items = _prepare_list(raw_eval, cap=max_eval_samples)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load eval_split %s: %s", eval_split, exc)

    semantic_scorer: Optional[_SemanticEmbeddingScorer] = None
    if semantic_reward_weight > 0.0:
        semantic_scorer = _SemanticEmbeddingScorer(
            model_id=semantic_model_id,
            fallback_model_id=semantic_fallback_id,
            device=device,
            token=token,
        )

    logger.info(
        "GRPO ready: n_train=%d n_eval=%d G=%d temp=%.2f kl_beta=%.3f sft_anchor=%.2f sem_w=%.2f (%s) qgate=%s",
        len(train_items),
        len(eval_items),
        group_size,
        temperature,
        kl_beta,
        sft_anchor_weight,
        semantic_reward_weight if (semantic_scorer and semantic_scorer.available) else 0.0,
        semantic_scorer.loaded_id if (semantic_scorer and semantic_scorer.available) else "off",
        quality_gate,
    )

    # Initial baseline validation before RL steps
    best_wer = math.inf
    if eval_items:
        init_metrics = _evaluate_val(
            model,
            processor,
            eval_items,
            forced_decoder_ids,
            device,
            model_dtype,
            max_eval_samples=max_eval_samples,
        )
        best_wer = init_metrics["eval_wer"]
        logger.info(
            "Initial pre-GRPO val (n=%d): WER=%.4f | CER=%.4f",
            len(eval_items),
            init_metrics["eval_wer"],
            init_metrics["eval_cer"],
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=0.01)

    grad_accum = max(int(cfg.gradient_accumulation_steps), 1)
    total_epochs = max(float(cfg.num_train_epochs), 1.0)
    steps_per_epoch = math.ceil(len(train_items) / grad_accum)
    total_opt_steps = (
        int(cfg.max_steps)
        if cfg.max_steps is not None
        else int(math.ceil(steps_per_epoch * total_epochs))
    )
    warmup_steps = int(total_opt_steps * float(cfg.warmup_ratio))

    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        progress = float(step - warmup_steps) / float(max(total_opt_steps - warmup_steps, 1))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and model_dtype == torch.float16))

    base_whisper = model.get_base_model()
    encoder_module = base_whisper.get_encoder()
    prefix_len = len(prefix_ids)

    global_step = 0
    micro_step = 0
    skipped_corrupted = 0
    skipped_trivial = 0
    runtime_corrupted_rows: List[Dict[str, Any]] = []
    running_reward = 0.0
    running_sem = 0.0
    running_wer = 0.0
    running_loss = 0.0
    running_kl = 0.0
    log_count = 0
    started_at = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    model.train()

    epoch_idx = 0
    while global_step < total_opt_steps:
        epoch_idx += 1
        indices = list(range(len(train_items)))
        random.shuffle(indices)
        made_progress_in_epoch = False

        for idx in indices:
            item = train_items[idx]

            feats = torch.tensor(
                item["input_features"], dtype=model_dtype, device=device
            ).unsqueeze(0)

            # 1) Encode audio once (encoder is frozen) and sample G rollouts
            with torch.no_grad():
                enc_out = encoder_module(feats)
                enc_hidden = enc_out.last_hidden_state  # (1, T_enc, D)

                model.eval()
                # 1 greedy rollout + (G-1) sampled rollouts
                greedy_seq = model.generate(
                    encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden),
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )
                n_sample = max(group_size - 1, 1)
                sampled_seqs = model.generate(
                    encoder_outputs=BaseModelOutput(
                        last_hidden_state=enc_hidden.expand(n_sample, -1, -1)
                    ),
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=1,
                )
                model.train()

            # Pad all G sequences to common length
            seq_list = [greedy_seq[0]] + [sampled_seqs[i] for i in range(sampled_seqs.size(0))]
            max_len = max(s.size(0) for s in seq_list)
            padded_seqs = torch.full(
                (len(seq_list), max_len),
                fill_value=pad_token_id,
                dtype=torch.long,
                device=device,
            )
            for i, s in enumerate(seq_list):
                padded_seqs[i, : s.size(0)] = s

            # 2) Decode texts & compute Ndizi lexical + semantic embedding rewards
            decoded_hyps = processor.batch_decode(padded_seqs, skip_special_tokens=True)
            ref_text = item["text"]
            ref_norm = _norm_for_reward(ref_text) or "a"
            greedy_norm = _norm_for_reward(decoded_hyps[0]) or ""
            greedy_wer = float(jiwer_wer(ref_norm, greedy_norm))
            greedy_cer = float(jiwer_cer(ref_norm, greedy_norm))

            # Quality gate 1: skip swapped/corrupted/unintelligible audio-reference pairs
            if quality_gate and greedy_wer > max_learnable_wer and greedy_cer > max_corrupted_cer:
                skipped_corrupted += 1
                runtime_corrupted_rows.append(
                    {
                        "row_index": item.get("row_index", idx),
                        "greedy_wer": round(greedy_wer, 4),
                        "greedy_cer": round(greedy_cer, 4),
                        "reference_preview": ref_text[:120],
                        "greedy_hyp_preview": decoded_hyps[0][:120],
                    }
                )
                continue

            rewards = np.array(
                [
                    compute_ndizi_reward(
                        ref_text,
                        h,
                        wer_weight=wer_weight,
                        cer_weight=cer_weight,
                        glossary_weight=glossary_weight,
                        length_penalty_weight=length_penalty_weight,
                    )
                    for h in decoded_hyps
                ],
                dtype=np.float32,
            )
            sem_mean = 0.0
            if semantic_scorer is not None and semantic_scorer.available and semantic_reward_weight > 0.0:
                sem_sims = semantic_scorer.score_candidates(ref_text, decoded_hyps)
                rewards = rewards + float(semantic_reward_weight) * sem_sims
                sem_mean = float(sem_sims.mean())

            r_mean = float(rewards.mean())
            r_std = float(rewards.std())

            # Quality gate 2: skip trivial clips where all G rollouts are identical and already mastered
            if quality_gate and r_std <= 1e-4 and greedy_wer < min_learnable_wer:
                skipped_trivial += 1
                continue

            if r_std > 1e-5:
                advantages = (rewards - r_mean) / (r_std + 1e-6)
            else:
                advantages = np.zeros_like(rewards)

            micro_step += 1
            made_progress_in_epoch = True
            adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)

            # 3) Reference policy log-probs (zero extra memory via disable_adapter)
            with torch.no_grad():
                with model.disable_adapter():
                    ref_tok_lp, valid_mask = _sequence_logprobs(
                        model,
                        enc_hidden,
                        padded_seqs,
                        prefix_len=prefix_len,
                        pad_token_id=pad_token_id,
                    )

            # 4) Current policy log-probs + GRPO + KL + SFT anchor loss
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and model_dtype == torch.float16)):
                pol_tok_lp, _ = _sequence_logprobs(
                    model,
                    enc_hidden,
                    padded_seqs,
                    prefix_len=prefix_len,
                    pad_token_id=pad_token_id,
                )
                # Per-token Schulman KL approx: exp(log_ref - log_pol) - (log_ref - log_pol) - 1
                log_diff = (ref_tok_lp - pol_tok_lp).clamp(min=-10.0, max=10.0)
                per_tok_kl = (torch.exp(log_diff) - log_diff - 1.0) * valid_mask

                seq_lengths = valid_mask.sum(dim=-1).clamp(min=1.0)
                seq_pol_lp = (pol_tok_lp * valid_mask).sum(dim=-1) / seq_lengths
                seq_kl = per_tok_kl.sum(dim=-1) / seq_lengths

                pg_loss = -(adv_tensor * seq_pol_lp).mean()
                kl_loss = seq_kl.mean()
                total_loss = pg_loss + kl_beta * kl_loss

                # Teacher-forced SFT anchor on ground-truth reference sequence
                if sft_anchor_weight > 0.0:
                    ref_seq_t = torch.tensor(
                        [item["ref_seq"]], dtype=torch.long, device=device
                    )
                    sft_lp, sft_mask = _sequence_logprobs(
                        model,
                        enc_hidden,
                        ref_seq_t,
                        prefix_len=prefix_len,
                        pad_token_id=pad_token_id,
                    )
                    sft_nll = -(sft_lp * sft_mask).sum() / sft_mask.sum().clamp(min=1.0)
                    total_loss = total_loss + sft_anchor_weight * sft_nll

                loss_scaled = total_loss / float(grad_accum)

            scaler.scale(loss_scaled).backward()

            running_reward += r_mean
            running_sem += sem_mean
            running_wer += min(greedy_wer, 2.0)
            running_loss += float(total_loss.detach().cpu().item())
            running_kl += float(kl_loss.detach().cpu().item())
            log_count += 1

            if micro_step % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % max(int(cfg.logging_steps), 1) == 0:
                    denom = max(log_count, 1)
                    logger.info(
                        "GRPO step %d/%d | loss=%.4f | reward=%.4f | sem_cos=%.4f | greedy_wer=%.4f | kl=%.4f | "
                        "skip_corrupt=%d | skip_triv=%d | lr=%.2e",
                        global_step,
                        total_opt_steps,
                        running_loss / denom,
                        running_reward / denom,
                        running_sem / denom,
                        running_wer / denom,
                        running_kl / denom,
                        skipped_corrupted,
                        skipped_trivial,
                        scheduler.get_last_lr()[0],
                    )
                    running_reward = 0.0
                    running_sem = 0.0
                    running_wer = 0.0
                    running_loss = 0.0
                    running_kl = 0.0
                    log_count = 0

                if eval_items and global_step % eval_every_steps == 0:
                    val_m = _evaluate_val(
                        model,
                        processor,
                        eval_items,
                        forced_decoder_ids,
                        device,
                        model_dtype,
                        max_eval_samples=max_eval_samples,
                    )
                    logger.info(
                        "GRPO val @ step %d: WER=%.4f | CER=%.4f (best_wer=%.4f)",
                        global_step,
                        val_m["eval_wer"],
                        val_m["eval_cer"],
                        best_wer,
                    )
                    if val_m["eval_wer"] <= best_wer:
                        best_wer = val_m["eval_wer"]
                        best_dir = out_dir / "best_adapter"
                        best_dir.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(str(best_dir))
                        processor.save_pretrained(str(best_dir))

                if global_step >= total_opt_steps:
                    break

        if not made_progress_in_epoch:
            logger.warning("No learnable clips remained in epoch %d; stopping GRPO early", epoch_idx)
            break

    if runtime_corrupted_rows:
        import csv as _csv

        corrupt_csv = qc_dir / "grpo_runtime_skipped_corrupted.csv"
        with corrupt_csv.open("w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=list(runtime_corrupted_rows[0].keys()))
            writer.writeheader()
            writer.writerows(runtime_corrupted_rows)
        logger.info("Wrote %d runtime-skipped corrupted GRPO rows to %s", len(runtime_corrupted_rows), corrupt_csv)

    # Final validation
    final_metrics: Dict[str, Any] = {
        "train_runtime": time.perf_counter() - started_at,
        "global_steps": global_step,
        "skipped_corrupted": skipped_corrupted,
        "skipped_trivial": skipped_trivial,
    }
    if eval_items:
        val_m = _evaluate_val(
            model,
            processor,
            eval_items,
            forced_decoder_ids,
            device,
            model_dtype,
            max_eval_samples=max_eval_samples,
        )
        final_metrics.update(val_m)
        logger.info(
            "Final GRPO val: WER=%.4f | CER=%.4f (best_wer=%.4f)",
            val_m["eval_wer"],
            val_m["eval_cer"],
            best_wer,
        )
        # If an earlier step was better on validation WER, reload best_adapter before merging
        best_dir = out_dir / "best_adapter"
        if val_m["eval_wer"] > best_wer and (best_dir / "adapter_model.safetensors").is_file():
            logger.info("Restoring best_adapter (WER=%.4f) before saving final merged model", best_wer)
            from safetensors.torch import load_file

            state = load_file(str(best_dir / "adapter_model.safetensors"))
            model.load_state_dict(state, strict=False)
            final_metrics["restored_best_wer"] = best_wer

    model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))

    if bool(cfg.extra.get("save_merged", True)) and hasattr(model, "merge_and_unload"):
        merged_dir = out_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving merged Stage-2 GRPO Whisper model to %s", merged_dir)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir))
        processor.save_pretrained(str(merged_dir))

    return {
        "status": "ok",
        "output_dir": str(out_dir),
        "model_id": cfg.model_id,
        "datasets": hub_ids,
        "language": lang_code,
        "stage": "grpo",
        "n_train": len(train_items),
        "n_eval": len(eval_items),
        "metrics": final_metrics,
        "eval_hint": (
            f"ndizi-asr --model hf:{out_dir}/merged "
            f"--dataset smutuvi/ndizi-1-sw --split test --text-column transcript "
            f"--device cuda --experiment-name sunflower51_ndizi1sw_grpo_test_full"
        ),
    }
