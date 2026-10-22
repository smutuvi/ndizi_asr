"""Word-aligned long-audio chunking (>30s -> <=28s) for Whisper training.

Uses ``torchaudio.pipelines.MMS_FA`` forced alignment (when available) to find
exact word start/end timestamps in clips longer than ``max_chunk_seconds``,
then packs words into <= ``max_chunk_seconds`` audio+transcript segments so
100% of long multi-turn survey interviews are kept for training instead of
being dropped by Whisper's 30-second window limit.

Falls back to Silero-VAD silence chunking + duration-proportional word slicing
if ``torchaudio`` MMS_FA is unavailable or fails on a noisy clip.
"""

from __future__ import annotations

import gc
import logging
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _word_spans_in_text(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", str(text or ""))]


def _slice_text_by_word_span(
    text: str,
    *,
    word_start: int,
    word_end_exclusive: int,
    n_aligned_words: int,
) -> str:
    raw = str(text or "").strip()
    if not raw or n_aligned_words <= 0 or word_end_exclusive <= word_start:
        return ""
    spans = _word_spans_in_text(raw)
    n_words = len(spans)
    if n_words == 0:
        return ""
    if n_words == n_aligned_words:
        i0 = max(0, min(word_start, n_words))
        i1 = max(i0, min(word_end_exclusive, n_words))
    else:
        i0 = int(round(word_start * n_words / n_aligned_words))
        i1 = int(round(word_end_exclusive * n_words / n_aligned_words))
        i1 = max(i1, i0 + 1) if word_end_exclusive > word_start else i0
        i0 = max(0, min(i0, n_words))
        i1 = max(i0, min(i1, n_words))
    if i0 >= n_words:
        return ""
    start = spans[i0][0]
    end = spans[i1 - 1][1]
    return raw[start:end].strip()


def _sanitize_words_for_mms_fa(text: str, allowed_chars: set[str]) -> List[str]:
    t = str(text or "").strip().lower()
    if not t:
        return []
    t = re.sub(r"[^a-z' ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return []
    out: List[str] = []
    for w in t.split():
        w2 = "".join(ch for ch in w if ch in allowed_chars)
        if w2:
            out.append(w2)
    return out


def _segment_by_word_times(
    word_times: List[Tuple[float, float, str]],
    *,
    chunk_seconds: float,
) -> List[Tuple[float, float, int, int]]:
    if not word_times:
        return []
    segs: List[Tuple[float, float, int, int]] = []
    cur_start = word_times[0][0]
    cur_end = cur_start
    word_start = 0
    cur_count = 0

    for idx, (w_start, w_end, _w) in enumerate(word_times):
        if cur_count == 0:
            cur_start = w_start
            cur_end = w_end
            word_start = idx
            cur_count = 1
            continue
        if w_end - cur_start <= chunk_seconds:
            cur_end = w_end
            cur_count += 1
            continue
        segs.append((cur_start, cur_end, word_start, idx))
        cur_start = w_start
        cur_end = w_end
        word_start = idx
        cur_count = 1

    if cur_count > 0:
        segs.append((cur_start, cur_end, word_start, len(word_times)))
    return segs


def _vad_proportional_segments(
    arr: np.ndarray,
    sr: int,
    text: str,
    *,
    chunk_seconds: float,
) -> List[Tuple[np.ndarray, str]]:
    """Fallback when MMS_FA fails: split audio via Silero-VAD and apportion words by duration."""
    from ndizi_asr.audio import chunk_waveform_by_vad

    spans = _word_spans_in_text(text)
    if not spans:
        return []
    try:
        chunks = chunk_waveform_by_vad(arr, max_chunk_s=chunk_seconds, sampling_rate=sr)
    except Exception:  # noqa: BLE001
        max_samples = int(chunk_seconds * sr)
        chunks = [arr[i : i + max_samples] for i in range(0, len(arr), max_samples)]
    chunks = [c for c in chunks if len(c) >= int(0.5 * sr)]
    if not chunks:
        return []

    total_samples = sum(len(c) for c in chunks)
    n_words = len(spans)
    out: List[Tuple[np.ndarray, str]] = []
    cum_samples = 0
    prev_w = 0
    for idx, ch in enumerate(chunks):
        cum_samples += len(ch)
        if idx == len(chunks) - 1:
            next_w = n_words
        else:
            next_w = max(prev_w + 1, min(n_words, int(round(cum_samples * n_words / max(total_samples, 1)))))
        sub_txt = _slice_text_by_word_span(
            text, word_start=prev_w, word_end_exclusive=next_w, n_aligned_words=n_words
        )
        if sub_txt:
            out.append((ch, sub_txt))
        prev_w = next_w
        if prev_w >= n_words:
            break
    return out


def expand_long_audio_dataset(
    dataset: Any,
    *,
    audio_col: str,
    text_col: str,
    max_chunk_seconds: float = 28.0,
    min_chunk_seconds: float = 0.5,
    split_label: str = "train",
) -> Any:
    """Expand clips longer than ``max_chunk_seconds`` into word-aligned segments.

    Tries ``torchaudio.pipelines.MMS_FA`` forced alignment first; falls back to
    Silero-VAD + proportional word slicing if MMS_FA fails on a given clip.
    Unloads MMS_FA from GPU memory before returning.
    """
    n_before = len(dataset)
    fa_model = None
    fa_tokenizer = None
    fa_aligner = None
    fa_sr = 16000
    fa_allowed: set[str] = set()
    fa_device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        import torchaudio
        from torchaudio.pipelines import MMS_FA as _MMS_FA_BUNDLE

        fa_model = _MMS_FA_BUNDLE.get_model().to(fa_device).eval()
        fa_tokenizer = _MMS_FA_BUNDLE.get_tokenizer()
        fa_aligner = _MMS_FA_BUNDLE.get_aligner()
        fa_sr = int(_MMS_FA_BUNDLE.sample_rate)
        fa_allowed = set(_MMS_FA_BUNDLE.get_dict().keys())
        logger.info("Loaded torchaudio MMS_FA aligner on %s for %s chunking", fa_device, split_label)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "torchaudio MMS_FA unavailable (%s); using Silero-VAD proportional chunking for long clips",
            exc,
        )

    stats = {"short_kept": 0, "long_mms_fa_clips": 0, "long_vad_fallback_clips": 0, "segments_created": 0}

    def _align_and_slice_mms(arr: np.ndarray, sr: int, txt: str) -> List[Tuple[np.ndarray, str]]:
        if fa_model is None or fa_tokenizer is None or fa_aligner is None:
            return []
        words = _sanitize_words_for_mms_fa(txt, fa_allowed)
        if len(words) < 2:
            return []
        import torchaudio

        wf = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        if sr != fa_sr:
            wf = torchaudio.functional.resample(wf, sr, fa_sr)
        with torch.inference_mode():
            emission, _ = fa_model(wf.to(fa_device))
        token_spans = fa_aligner(emission[0], fa_tokenizer(words))
        num_frames = int(emission.size(1))
        if num_frames <= 0:
            return []
        ratio = wf.size(1) / num_frames / float(fa_sr)
        word_times: List[Tuple[float, float, str]] = []
        for spans, word in zip(token_spans, words):
            if not spans:
                continue
            s0 = float(ratio * spans[0].start)
            s1 = float(ratio * spans[-1].end)
            if s1 > s0:
                word_times.append((s0, s1, word))
        segs = _segment_by_word_times(word_times, chunk_seconds=float(max_chunk_seconds))
        if not segs:
            return []
        n_fa = len(words)
        out_pairs: List[Tuple[np.ndarray, str]] = []
        for s0, s1, w0, w1 in segs:
            seg_txt = _slice_text_by_word_span(txt, word_start=w0, word_end_exclusive=w1, n_aligned_words=n_fa)
            if not seg_txt:
                continue
            # Add a small 0.12s pad around word boundaries
            i0 = max(0, int(round((s0 - 0.12) * sr)))
            i1 = min(len(arr), int(round((s1 + 0.12) * sr)))
            if i1 - i0 < int(min_chunk_seconds * sr):
                continue
            seg_arr = arr[i0:i1]
            # If a single word span still exceeds max_chunk_seconds, cap it
            max_samps = int(max_chunk_seconds * sr)
            if len(seg_arr) > max_samps:
                seg_arr = seg_arr[:max_samps]
            out_pairs.append((seg_arr, seg_txt))
        return out_pairs

    def _expand_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        out_audio: List[Dict[str, Any]] = []
        out_text: List[str] = []
        for aud, raw_txt in zip(batch[audio_col], batch[text_col]):
            txt = str(raw_txt or "").strip()
            if not txt:
                continue
            arr = np.asarray(aud["array"], dtype=np.float32)
            sr = int(aud["sampling_rate"])
            dur = len(arr) / max(sr, 1)
            if dur < min_chunk_seconds:
                continue
            if dur <= max_chunk_seconds:
                out_audio.append({"array": arr, "sampling_rate": sr})
                out_text.append(txt)
                stats["short_kept"] += 1
                continue

            # Long clip (> max_chunk_seconds): try MMS_FA word alignment first
            pairs: List[Tuple[np.ndarray, str]] = []
            try:
                pairs = _align_and_slice_mms(arr, sr, txt)
            except Exception as exc:  # noqa: BLE001
                logger.debug("MMS_FA failed on clip (dur=%.1fs): %s", dur, exc)
                pairs = []

            if pairs:
                stats["long_mms_fa_clips"] += 1
            else:
                pairs = _vad_proportional_segments(arr, sr, txt, chunk_seconds=max_chunk_seconds)
                stats["long_vad_fallback_clips"] += 1

            for seg_arr, seg_txt in pairs:
                out_audio.append({"array": seg_arr, "sampling_rate": sr})
                out_text.append(seg_txt)
                stats["segments_created"] += 1

        return {audio_col: out_audio, text_col: out_text}

    drop_cols = [c for c in dataset.column_names if c not in {audio_col, text_col}]
    dataset = dataset.map(
        _expand_batch,
        batched=True,
        batch_size=8,
        remove_columns=drop_cols,
        desc=f"chunk_long_{split_label}",
    )

    # Free MMS_FA model from GPU before loading Whisper
    if fa_model is not None:
        del fa_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(
        "[%s] Long-audio chunking (<=%.1fs): %d -> %d segments "
        "(short_kept=%d, long_mms_fa=%d, long_vad_fallback=%d, new_segments=%d)",
        split_label,
        max_chunk_seconds,
        n_before,
        len(dataset),
        stats["short_kept"],
        stats["long_mms_fa_clips"],
        stats["long_vad_fallback_clips"],
        stats["segments_created"],
    )
    return dataset
