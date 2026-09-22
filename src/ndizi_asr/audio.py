"""Audio I/O helpers and optional Silero-VAD chunking for long clips."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

TARGET_SR = 16000
SOUNDFILE_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
logger = logging.getLogger(__name__)

_vad_model = None
_vad_get_speech_timestamps = None


def _load_via_ffmpeg(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            f"Cannot decode {path!r}: ffmpeg not found on PATH "
            "(needed for .m4a / compressed audio)."
        )
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        path,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed for {path!r}: {err or proc.returncode}")
    if not proc.stdout:
        raise RuntimeError(f"ffmpeg returned empty audio for {path!r}")
    wav = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return wav, target_sr


def _load_via_soundfile(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    wav, sr = sf.read(path, always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if int(sr) != int(target_sr):
        try:
            import librosa

            wav = librosa.resample(wav, orig_sr=int(sr), target_sr=target_sr)
        except ImportError:
            # Linear resample fallback (no librosa)
            n_out = int(round(len(wav) * target_sr / float(sr)))
            x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=max(n_out, 1), endpoint=False)
            wav = np.interp(x_new, x_old, wav.astype(np.float64)).astype(np.float32)
        sr = target_sr
    return wav.astype(np.float32), int(sr)


def load_waveform(path: str, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """Load mono float32 waveform resampled to target_sr."""
    suffix = Path(path).suffix.lower()
    if suffix in SOUNDFILE_SUFFIXES:
        try:
            return _load_via_soundfile(path, target_sr)
        except Exception as exc:  # noqa: BLE001
            logger.debug("soundfile failed for %s (%s); trying ffmpeg", path, exc)
    return _load_via_ffmpeg(path, target_sr)


def duration_seconds(waveform: np.ndarray, sr: int) -> float:
    return float(len(waveform) / max(sr, 1))


def _get_vad():
    global _vad_model, _vad_get_speech_timestamps
    if _vad_model is None:
        from silero_vad import get_speech_timestamps, load_silero_vad

        _vad_model = load_silero_vad()
        _vad_get_speech_timestamps = get_speech_timestamps
    return _vad_model, _vad_get_speech_timestamps


def chunk_waveform_by_vad(
    waveform_16k: np.ndarray,
    max_chunk_s: float,
    *,
    sampling_rate: int = TARGET_SR,
) -> List[np.ndarray]:
    """Split long audio at Silero-VAD boundaries into chunks <= max_chunk_s."""
    import torch

    if len(waveform_16k) / sampling_rate <= max_chunk_s:
        return [waveform_16k]

    try:
        model, get_speech_timestamps = _get_vad()
        wav_tensor = torch.from_numpy(waveform_16k).float()
        timestamps = get_speech_timestamps(
            wav_tensor, model, sampling_rate=sampling_rate, return_seconds=True
        )
    except Exception:  # noqa: BLE001
        timestamps = []

    if not timestamps:
        n = int(max_chunk_s * sampling_rate)
        hop = max(1, n - int(0.5 * sampling_rate))
        return [
            waveform_16k[i : i + n]
            for i in range(0, len(waveform_16k), hop)
            if i < len(waveform_16k)
        ]

    chunks: List[np.ndarray] = []
    cur_start = timestamps[0]["start"]
    cur_end = timestamps[0]["start"]

    def emit(start_s: float, end_s: float) -> None:
        s, e = int(start_s * sampling_rate), int(end_s * sampling_rate)
        if e > s:
            chunks.append(waveform_16k[s:e])

    for seg in timestamps:
        seg_len = seg["end"] - seg["start"]
        if seg_len > max_chunk_s:
            if cur_end > cur_start:
                emit(cur_start, cur_end)
            t = seg["start"]
            while t < seg["end"]:
                emit(t, min(t + max_chunk_s, seg["end"]))
                t += max_chunk_s
            cur_start = cur_end = seg["end"]
            continue
        if seg["end"] - cur_start <= max_chunk_s:
            cur_end = seg["end"]
        else:
            emit(cur_start, cur_end)
            cur_start, cur_end = seg["start"], seg["end"]
    if cur_end > cur_start:
        emit(cur_start, cur_end)
    return chunks or [waveform_16k]


def maybe_chunk(
    waveform: np.ndarray,
    sr: int,
    max_chunk_s: Optional[float],
) -> Tuple[List[np.ndarray], bool]:
    if max_chunk_s is None or duration_seconds(waveform, sr) <= max_chunk_s:
        return [waveform], False
    return chunk_waveform_by_vad(waveform, max_chunk_s, sampling_rate=sr), True


def join_chunk_transcripts(parts: List[str]) -> str:
    return " ".join(p.strip() for p in parts if p and str(p).strip())
