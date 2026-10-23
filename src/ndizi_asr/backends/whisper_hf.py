from __future__ import annotations

import logging
import os
import time
from typing import Optional

import numpy as np

from .sunbird_langs import LANGUAGE_TOKENS_WHISPER, resolve_sunbird_language_code
from .types import TranscriptionResult

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# Whisper encoder receptive field (~30s). Chunk longer audio with overlap-free joins.
MAX_SECONDS = 30.0

# Friendly / dataset labels → OpenAI Whisper language names (not Sunbird ISO-639-3).
_OPENAI_WHISPER_LANG: dict[str, str] = {
    "swahili": "swahili",
    "sw": "swahili",
    "swa": "swahili",
    "sw_ke": "swahili",
    "amharic": "amharic",
    "am": "amharic",
    "amh": "amharic",
    "am_et": "amharic",
    "hausa": "hausa",
    "ha": "hausa",
    "hau": "hausa",
    "ha_ng": "hausa",
    "english": "english",
    "en": "english",
    "eng": "english",
}


def _is_sunbird_whisper(model_id: str) -> bool:
    from pathlib import Path
    import json

    mid = model_id.lower()
    if "sunbird" in mid or "asr-whisper-51" in mid or "sunflower" in mid:
        return True
    p = Path(model_id).expanduser()
    if p.is_dir():
        for fname in ("adapter_config.json", "config.json"):
            f = p / fname
            if f.is_file():
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    base = str(
                        data.get("base_model_name_or_path") or data.get("_name_or_path") or ""
                    ).lower()
                    if "sunbird" in base or "asr-whisper-51" in base or "sunflower" in base:
                        return True
                except Exception:  # noqa: BLE001
                    pass
    return False


def _openai_whisper_language(language: Optional[str], language_config: Optional[str]) -> Optional[str]:
    for candidate in (language, language_config):
        if not candidate:
            continue
        key = str(candidate).strip().lower()
        if key in _OPENAI_WHISPER_LANG:
            return _OPENAI_WHISPER_LANG[key]
    return None


def _hf_token() -> Optional[str]:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return None


def _gated_access_error(model_id: str, exc: Exception) -> OSError:
    token = _hf_token()
    auth_hint = (
        "HF_TOKEN is set, but access was still denied — make sure this account "
        f"accepted the model terms at https://huggingface.co/{model_id}."
        if token
        else (
            "No HF_TOKEN found. Add a Hugging Face token to `.env` "
            "(HF_TOKEN=hf_...) after accepting the model terms, or run `huggingface-cli login`."
        )
    )
    return OSError(
        f"Cannot download gated model {model_id!r}.\n"
        f"1) Open https://huggingface.co/{model_id} while logged in and accept access.\n"
        f"2) {auth_hint}\n"
        f"Original error: {exc}"
    )


def _resample_linear(array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return np.asarray(array, dtype=np.float32)
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    n_out = int(round(len(arr) * target_sr / float(orig_sr)))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(arr), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, arr).astype(np.float32)


def _is_local_finetune(model_id: str) -> bool:
    from pathlib import Path

    p = Path(model_id).expanduser()
    if p.is_dir():
        return True
    mid = model_id.lower()
    return "ndizi-whisper" in mid or "lora" in mid or "merged" in mid


def _chunk_audio(array: np.ndarray, sampling_rate: int, max_seconds: float = MAX_SECONDS) -> list[np.ndarray]:
    max_samples = int(max_seconds * sampling_rate)
    if len(array) <= max_samples:
        return [array]
    try:
        from ..audio import chunk_waveform_by_vad

        vad_chunks = chunk_waveform_by_vad(
            np.asarray(array, dtype=np.float32),
            max_chunk_s=min(max_seconds, 28.0),
            sampling_rate=sampling_rate,
        )
        if vad_chunks:
            return vad_chunks
    except Exception as exc:  # noqa: BLE001
        logger.debug("VAD chunking unavailable (%s); falling back to fixed windows", exc)
    chunks = []
    for start in range(0, len(array), max_samples):
        chunk = array[start : start + max_samples]
        if len(chunk) > 0:
            chunks.append(chunk)
    return chunks


class _GemmaMBRAndRAGHelper:
    """Single-model inference booster using EmbeddingGemma-300M (or multilingual-e5 fallback).

    Provides two inference-time capabilities without any external LLM API:
    1. **N-Best Neural MBR + Prototype Reranking**:
       Scores N Whisper beam candidates using:
         - Acoustic sequence log-probability from Whisper
         - Pairwise EmbeddingGemma Minimum Bayes Risk (MBR) semantic consensus
         - Pairwise word-level lexical MBR consensus
         - Cosine similarity to the nearest in-domain training utterances (smutuvi/ndizi-1-sw)
         - Repetition / n-gram loop penalty
    2. **Retrieval-Augmented `<|startofprev|>` Whisper Prompting (RAG-Whisper)**:
       Retrieves the closest training transcript to a fast pass-1 decode and extracts
       domain terms/phrasing to condition Whisper's `<|startofprev|>` prompt tokens.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        model_id: str = "google/embeddinggemma-300m",
        fallback_model_id: str = "intfloat/multilingual-e5-base",
        corpus_dataset_id: str = "smutuvi/ndizi-1-sw",
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.loaded_id: Optional[str] = None
        self.corpus_texts: list[str] = []
        self.corpus_embs = None

        hf_tok = _hf_token()
        kw = {"token": hf_tok} if hf_tok else {}
        for cand in (model_id, fallback_model_id):
            if not cand:
                continue
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(cand, **kw)
                self.model = AutoModel.from_pretrained(cand, **kw).to(self.device)
                self.model.eval()
                for p in self.model.parameters():
                    p.requires_grad = False
                self.loaded_id = cand
                logger.info("Loaded EmbeddingGemma/semantic reranker %s on %s", cand, self.device)
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load semantic model %s (%s); trying fallback", cand, exc)

        if self.model is not None and corpus_dataset_id:
            self._load_train_corpus(corpus_dataset_id)

    @property
    def available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not self.available or not texts:
            return np.zeros((len(texts), 1), dtype=np.float32)
        torch = self.torch
        from ..ethio_normalize import clean_swahili_fair_transcription

        normed = [clean_swahili_fair_transcription(t) or t.lower().strip() or "a" for t in texts]
        all_vecs = []
        with torch.no_grad():
            for i in range(0, len(normed), batch_size):
                batch = normed[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(self.device)
                out = self.model(**enc)
                hidden = getattr(out, "last_hidden_state", None)
                if hidden is None:
                    vecs = out[0]
                else:
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    vecs = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                vecs = torch.nn.functional.normalize(vecs.float(), p=2, dim=-1)
                all_vecs.append(vecs.cpu().numpy())
        return np.vstack(all_vecs)

    def _load_train_corpus(self, dataset_id: str) -> None:
        import csv
        from huggingface_hub import hf_hub_download
        from ..ethio_normalize import clean_swahili_training_target

        texts: list[str] = []
        for cand_rel in ("data/train/metadata.csv", "train/metadata.csv"):
            try:
                local_csv = hf_hub_download(repo_id=dataset_id, filename=cand_rel, repo_type="dataset")
                with open(local_csv, "r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        t = clean_swahili_training_target(str(row.get("transcript") or row.get("text") or ""))
                        if len(t.split()) >= 4:
                            texts.append(t)
                if texts:
                    break
            except Exception:  # noqa: BLE001
                continue
        if texts:
            self.corpus_texts = texts
            self.corpus_embs = self.encode(texts, batch_size=128)
            logger.info(
                "Indexed %d training transcripts from %s with %s for MBR/RAG",
                len(texts),
                dataset_id,
                self.loaded_id,
            )

    def retrieve_prompt_context(self, draft_text: str, top_k: int = 2) -> str:
        """Retrieve nearest training utterances and build a concise domain prefix."""
        if not draft_text.strip() or self.corpus_embs is None or not self.corpus_texts:
            return ""
        q = self.encode([draft_text])[0]
        sims = self.corpus_embs @ q
        top_idx = np.argsort(-sims)[:top_k]
        # Take the opening question/context clause (up to 28 words) from top neighbor
        best_txt = self.corpus_texts[int(top_idx[0])]
        words = best_txt.split()[:28]
        return " ".join(words)

    def rerank_nbest(
        self,
        candidates: list[str],
        seq_scores: list[float],
    ) -> str:
        """Select best hypothesis via Acoustic LogProb + EmbeddingGemma MBR + Corpus Prior."""
        import re
        from ..ethio_normalize import clean_swahili_fair_transcription

        valid = [(c.strip(), float(s)) for c, s in zip(candidates, seq_scores) if c and c.strip()]
        if not valid:
            return ""
        if len(valid) == 1:
            return valid[0][0]

        hyps = [v[0] for v in valid]
        scores = np.asarray([v[1] for v in valid], dtype=np.float64)
        # Normalize acoustic scores across beams to [0, 1]
        s_std = float(scores.std())
        ac_norm = (scores - scores.mean()) / (s_std + 1e-5) if s_std > 1e-6 else np.zeros_like(scores)

        # 1) Lexical token-F1 MBR consensus across beams
        norm_hyps = [clean_swahili_fair_transcription(h) or h.lower() for h in hyps]
        tok_sets = [set(nh.split()) for nh in norm_hyps]
        lex_mbr = np.zeros(len(hyps), dtype=np.float64)
        for i in range(len(hyps)):
            if not tok_sets[i]:
                continue
            sims = []
            for j in range(len(hyps)):
                if not tok_sets[j]:
                    continue
                inter = len(tok_sets[i] & tok_sets[j])
                f1 = (2.0 * inter) / max(len(tok_sets[i]) + len(tok_sets[j]), 1)
                sims.append(f1)
            lex_mbr[i] = float(np.mean(sims)) if sims else 0.0

        # 2) EmbeddingGemma semantic MBR + in-domain train prototype similarity
        sem_mbr = np.zeros(len(hyps), dtype=np.float64)
        proto_sim = np.zeros(len(hyps), dtype=np.float64)
        if self.available:
            embs = self.encode(hyps)
            sim_mat = embs @ embs.T
            sem_mbr = sim_mat.mean(axis=1).astype(np.float64)
            if self.corpus_embs is not None and len(self.corpus_embs) > 0:
                c_sims = embs @ self.corpus_embs.T
                top5 = np.sort(c_sims, axis=1)[:, -5:]
                proto_sim = top5.mean(axis=1).astype(np.float64)

        # 3) Repetition / loop penalty
        rep_pen = np.zeros(len(hyps), dtype=np.float64)
        for i, nh in enumerate(norm_hyps):
            ws = nh.split()
            if len(ws) >= 6:
                bigrams = [f"{ws[k]}_{ws[k+1]}" for k in range(len(ws) - 1)]
                uniq_ratio = len(set(bigrams)) / float(len(bigrams))
                if uniq_ratio < 0.72:
                    rep_pen[i] += (0.72 - uniq_ratio) * 2.5
            if re.search(r"\b(\w+)(?:\s+\1){2,}\b", nh):
                rep_pen[i] += 0.5

        total = (
            0.45 * ac_norm
            + 0.85 * lex_mbr
            + 0.65 * sem_mbr
            + 0.35 * proto_sim
            - rep_pen
        )
        best_idx = int(np.argmax(total))
        return hyps[best_idx]


class HuggingFaceWhisperClient:
    """Local Whisper ASR via transformers (supports Sunbird African Whisper)."""

    def __init__(
        self,
        model_id: str,
        *,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        num_beams: int = 1,
        gemma_mbr: bool = False,
        rag_prompt: bool = False,
        canonicalize_output: bool = False,
        token: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError as exc:
            raise ImportError(
                "Local Whisper backends require torch and transformers. "
                "Install with: pip install -e '.[local]'"
            ) from exc

        self.model_id = model_id
        self.num_beams = max(1, int(num_beams))
        self.gemma_mbr = bool(gemma_mbr)
        self.rag_prompt = bool(rag_prompt)
        self.canonicalize_output = bool(canonicalize_output)
        if (self.gemma_mbr) and self.num_beams < 5:
            self.num_beams = 5
        self.torch = torch
        self._gemma_helper: Optional[_GemmaMBRAndRAGHelper] = None
        hf_token = token if token is not None else _hf_token()

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        dtype_map = {
            None: None,
            "auto": None,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if torch_dtype not in dtype_map:
            raise ValueError(f"Unsupported torch_dtype={torch_dtype!r}")
        resolved_dtype = dtype_map[torch_dtype]
        if resolved_dtype is None and self.device.type == "cuda":
            resolved_dtype = torch.float16
        if resolved_dtype is None:
            resolved_dtype = torch.float32

        logger.info(
            "Loading Whisper model %s on %s (%s); hf_token=%s",
            model_id,
            self.device,
            resolved_dtype,
            "set" if hf_token else "missing",
        )
        load_kwargs = {"token": hf_token} if hf_token else {}
        try:
            from pathlib import Path
            import json

            local_dir = Path(model_id).expanduser()
            adapter_cfg = local_dir / "adapter_config.json" if local_dir.is_dir() else None
            if adapter_cfg is not None and adapter_cfg.is_file():
                from peft import PeftModel

                adata = json.loads(adapter_cfg.read_text(encoding="utf-8"))
                base_id = str(adata.get("base_model_name_or_path") or "Sunbird/SunflowerASR-51-african-languages")
                try:
                    self.processor = WhisperProcessor.from_pretrained(model_id, **load_kwargs)
                except Exception:  # noqa: BLE001
                    self.processor = WhisperProcessor.from_pretrained(base_id, **load_kwargs)
                base_model = WhisperForConditionalGeneration.from_pretrained(
                    base_id,
                    torch_dtype=resolved_dtype,
                    **load_kwargs,
                )
                peft_model = PeftModel.from_pretrained(base_model, model_id, **load_kwargs)
                self.model = peft_model.merge_and_unload().to(self.device)
            else:
                self.processor = WhisperProcessor.from_pretrained(model_id, **load_kwargs)
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=resolved_dtype,
                    **load_kwargs,
                ).to(self.device)
        except Exception as exc:  # noqa: BLE001
            name = exc.__class__.__name__
            msg = str(exc).lower()
            if "gated" in msg or name in {"GatedRepoError", "RepositoryNotFoundError"} or "401" in msg:
                raise _gated_access_error(model_id, exc) from exc
            raise

        self.model.eval()
        self._sunbird = _is_sunbird_whisper(model_id)
        self._local_ft = _is_local_finetune(model_id)
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            # Clear language/task on generation_config so explicit forced_decoder_ids
            # in .generate() does not double-prepend prefix tokens in transformers 5.x.
            self.model.generation_config.language = None
            self.model.generation_config.task = None
        self._transcribe_tok = self.processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        self._notimestamps_tok = self.processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        if self.gemma_mbr or self.rag_prompt:
            self._gemma_helper = _GemmaMBRAndRAGHelper(device=str(self.device))

    def _forced_decoder_ids(
        self,
        language: Optional[str],
        language_config: Optional[str],
    ) -> list[tuple[int, int]]:
        if self._sunbird:
            lang_code = resolve_sunbird_language_code(language, language_config)
            lang_tok = LANGUAGE_TOKENS_WHISPER[lang_code]
            return [
                (1, lang_tok),
                (2, self._transcribe_tok),
                (3, self._notimestamps_tok),
            ]
        # openai/whisper-* and fine-tunes (e.g. ndizi-whisper-large-v3-turbo-*)
        lang = _openai_whisper_language(language, language_config)
        if not lang:
            # Language not natively supported by OpenAI Whisper (e.g. Oromo); auto-detect task=transcribe
            return list(self.processor.get_decoder_prompt_ids(task="transcribe"))
        return list(
            self.processor.get_decoder_prompt_ids(language=lang, task="transcribe")
        )

    def transcribe_array(
        self,
        audio_array: np.ndarray,
        sampling_rate: int,
        *,
        language: Optional[str] = None,
        language_config: Optional[str] = None,
    ) -> TranscriptionResult:
        started = time.perf_counter()
        try:
            from ..audio import join_chunk_transcripts
            from ..ethio_normalize import clean_swahili_training_target

            forced = self._forced_decoder_ids(language, language_config)
            audio = _resample_linear(np.asarray(audio_array), int(sampling_rate), SAMPLE_RATE)
            if audio.ndim > 1:
                audio = audio.mean(axis=-1)
            texts: list[str] = []
            prev_chunk_tail: str = ""
            for chunk in _chunk_audio(audio, SAMPLE_RATE):
                chunk_txt = self._transcribe_chunk(
                    chunk,
                    forced,
                    prev_context=prev_chunk_tail,
                )
                if chunk_txt.strip():
                    texts.append(chunk_txt.strip())
                    prev_chunk_tail = " ".join(chunk_txt.strip().split()[-18:])
            text = join_chunk_transcripts(texts)
            if self.canonicalize_output and text:
                text = clean_swahili_training_target(text)
            return TranscriptionResult(
                text=text,
                latency_seconds=time.perf_counter() - started,
                model=self.model_id,
            )
        except Exception as exc:  # noqa: BLE001 - surface as row error
            logger.exception("Sunbird/Whisper transcription failed")
            return TranscriptionResult(
                text="",
                latency_seconds=time.perf_counter() - started,
                model=self.model_id,
                error=str(exc),
            )

    def _transcribe_chunk(
        self,
        audio: np.ndarray,
        forced_decoder_ids: list[tuple[int, int]],
        *,
        prev_context: str = "",
    ) -> str:
        torch = self.torch
        # Match whisper_trainer.py feature extraction (do_normalize=False by default)
        # on local/Ndizi fine-tunes; keep do_normalize=True for zero-shot Sunbird hub models.
        do_norm = (
            bool(getattr(self.processor.feature_extractor, "do_normalize", False))
            if self._local_ft
            else True
        )
        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            do_normalize=do_norm,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device, dtype=self.model.dtype)

        # Optional Pass-1 RAG prompt lookup via EmbeddingGemma
        prompt_ids_tensor = None
        if self.rag_prompt and self._gemma_helper is not None and self._gemma_helper.available:
            with torch.inference_mode():
                draft_ids = self.model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=96,
                    num_beams=1,
                    do_sample=False,
                )
            draft_txt = self.processor.batch_decode(draft_ids, skip_special_tokens=True)[0].strip()
            ctx_str = self._gemma_helper.retrieve_prompt_context(draft_txt)
            if prev_context:
                ctx_str = f"{ctx_str} {prev_context}".strip()
            if ctx_str:
                try:
                    p_ids = self.processor.get_prompt_ids(ctx_str, return_tensors="pt")
                    prompt_ids_tensor = p_ids.to(self.device)
                except Exception:  # noqa: BLE001
                    prompt_ids_tensor = None

        gen_kwargs: dict = dict(
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=256,
            num_beams=self.num_beams,
            do_sample=False,
        )
        if prompt_ids_tensor is not None:
            gen_kwargs["prompt_ids"] = prompt_ids_tensor

        if self.gemma_mbr and self.num_beams > 1 and self._gemma_helper is not None:
            n_ret = min(self.num_beams, 6)
            with torch.inference_mode():
                out = self.model.generate(
                    input_features,
                    num_return_sequences=n_ret,
                    return_dict_in_generate=True,
                    output_scores=True,
                    length_penalty=1.0,
                    **gen_kwargs,
                )
            seqs = out.sequences
            cands = self.processor.batch_decode(
                seqs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            seq_scores = (
                out.sequences_scores.detach().float().cpu().tolist()
                if getattr(out, "sequences_scores", None) is not None
                else [0.0] * len(cands)
            )
            return self._gemma_helper.rerank_nbest(cands, seq_scores)

        with torch.inference_mode():
            predicted_ids = self.model.generate(
                input_features,
                **gen_kwargs,
            )
        texts = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return texts[0] if texts else ""
