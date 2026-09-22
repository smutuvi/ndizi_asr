"""Local Gemma 4 multimodal ASR (audio → text) via transformers."""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import numpy as np

from .types import TranscriptionResult

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MAX_SECONDS = 30.0


def _hf_token() -> Optional[str]:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return None


def _gated_access_error(model_id: str, exc: Exception) -> OSError:
    token = _hf_token()
    auth_hint = (
        "HF_TOKEN is set, but access was still denied — accept the model terms "
        f"at https://huggingface.co/{model_id}."
        if token
        else (
            "No HF_TOKEN found. Accept model terms on Hugging Face and set "
            "HF_TOKEN in .env (or run: huggingface-cli login)."
        )
    )
    return OSError(
        f"Cannot download gated model {model_id!r}.\n"
        f"1) Open https://huggingface.co/{model_id} and accept access.\n"
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


def _prepare_wave(audio_array: np.ndarray, sampling_rate: int) -> np.ndarray:
    wave = _resample_linear(np.asarray(audio_array), int(sampling_rate), SAMPLE_RATE)
    if wave.ndim > 1:
        wave = wave.mean(axis=-1)
    max_samples = int(MAX_SECONDS * SAMPLE_RATE)
    if len(wave) > max_samples:
        wave = wave[:max_samples]
    return np.ascontiguousarray(wave, dtype=np.float32)


def _default_instruction(language: Optional[str]) -> str:
    if language and language.lower() not in {"default", "unknown"}:
        return (
            f"Transcribe the following speech segment in {language} exactly. "
            "Return only the transcript text with no commentary or labels."
        )
    return (
        "Transcribe the following speech segment exactly. "
        "Return only the transcript text with no commentary or labels."
    )


class HuggingFaceGemmaASRClient:
    """Gemma 4 (and Sunbird Gemma 4) native-audio ASR via AutoModelForImageTextToText."""

    def __init__(
        self,
        model_id: str,
        *,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        max_new_tokens: int = 256,
        token: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Gemma ASR backends require torch and a recent transformers. "
                "Install with: pip install -e '.[local]'"
            ) from exc

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.torch = torch
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
        if resolved_dtype is None:
            if self.device.type == "cuda":
                resolved_dtype = torch.bfloat16
            else:
                resolved_dtype = torch.float32

        logger.info(
            "Loading Gemma ASR model %s on %s (%s); hf_token=%s",
            model_id,
            self.device,
            resolved_dtype,
            "set" if hf_token else "missing",
        )
        load_kwargs: dict = {"token": hf_token} if hf_token else {}
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, **load_kwargs)
            self.model = AutoModelForImageTextToText.from_pretrained(
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

    def transcribe_array(
        self,
        audio_array: np.ndarray,
        sampling_rate: int,
        *,
        language: Optional[str] = None,
        language_config: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        del language_config  # unused; instruction uses friendly language name
        started = time.perf_counter()
        try:
            wave = _prepare_wave(audio_array, sampling_rate)
            instruction = (prompt or "").strip() or _default_instruction(language)
            text = self._transcribe_wave(wave, instruction)
            return TranscriptionResult(
                text=text,
                latency_seconds=time.perf_counter() - started,
                model=self.model_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Gemma transcription failed")
            return TranscriptionResult(
                text="",
                latency_seconds=time.perf_counter() - started,
                model=self.model_id,
                error=str(exc),
            )

    def _transcribe_wave(self, wave: np.ndarray, instruction: str) -> str:
        torch = self.torch
        processor = self.processor

        # Prefer chat-template + audio placeholders when available (Gemma 4).
        if hasattr(processor, "feature_extractor") and hasattr(processor, "apply_chat_template"):
            inputs = self._build_gemma_inputs(wave, instruction)
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": wave},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

        inputs = self._to_device(inputs)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        new_tokens = out[:, input_len:]
        text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        return text

    def _build_gemma_inputs(self, wave: np.ndarray, instruction: str):
        processor = self.processor
        fe_out = processor.feature_extractor([wave], return_tensors="pt")
        n_soft = self._soft_token_count(fe_out)
        boa = getattr(processor, "boa_token", "<|audio|>")
        at = getattr(processor, "audio_token", "<|audio_soft_token|>")
        eoa = getattr(processor, "eoa_token", "<|audio|>")
        # Fallbacks if attribute names differ across transformers versions
        if not hasattr(processor, "audio_token"):
            at = processor.tokenizer.convert_ids_to_tokens(
                getattr(processor, "audio_token_id", processor.tokenizer.unk_token_id)
            )

        audio_block = f"{boa}{at * n_soft}{eoa}" if n_soft > 0 else ""
        user_content = [
            {"type": "audio", "audio": wave},
            {"type": "text", "text": instruction},
        ]
        messages = [{"role": "user", "content": user_content}]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if isinstance(prompt, list):
            prompt = prompt[0]
        if at in prompt and n_soft > 0:
            prompt = prompt.replace(at, audio_block, 1)

        inputs = processor(text=prompt, return_tensors="pt", return_mm_token_type_ids=True)
        inputs["input_features"] = fe_out["input_features"]
        if "input_features_mask" in fe_out:
            inputs["input_features_mask"] = fe_out["input_features_mask"]
        return inputs

    def _soft_token_count(self, fe_out) -> int:
        processor = self.processor
        if "input_features_mask" not in fe_out:
            return int(getattr(processor, "audio_seq_length", 0) or 0)
        mask = fe_out["input_features_mask"][0]
        if hasattr(mask, "numpy"):
            mask = mask.numpy()
        num_mel = int(np.asarray(mask, dtype=bool).sum())
        if num_mel <= 0:
            return 0
        t = num_mel
        for _ in range(2):
            t = (t + 2 - 3) // 2 + 1
        max_len = int(getattr(processor, "audio_seq_length", t) or t)
        return min(t, max_len)

    def _to_device(self, inputs):
        torch = self.torch
        moved = {}
        for key, val in inputs.items():
            if not isinstance(val, torch.Tensor):
                moved[key] = val
                continue
            tensor = val.to(self.device)
            if key not in ("input_features", "input_features_mask") and tensor.is_floating_point():
                tensor = tensor.to(dtype=self.model.dtype)
            elif key == "input_features" and tensor.is_floating_point():
                tensor = tensor.to(dtype=self.model.dtype)
            moved[key] = tensor
        return moved
