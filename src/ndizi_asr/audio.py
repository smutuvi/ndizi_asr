"""Audio I/O helpers and optional Silero-VAD chunking for long clips."""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

TARGET_SR = 16000
SOUNDFILE_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
logger = logging.getLogger(__name__)

# Serialize ffmpeg/soundfile decode — conda ffmpeg segfaults under threaded load.
_FFMPEG_LOCK = threading.Lock()

_vad_model = None
_vad_get_speech_timestamps = None


def _is_conda_ffmpeg(path: str) -> bool:
    p = path.lower()
    return ("/envs/" in p) or ("/conda/" in p) or ("miniconda" in p) or ("miniforge" in p)


def _ffmpeg_candidates() -> list[str]:
    """Prefer system ffmpeg; conda builds often break under concurrent decode."""
    import os

    out: list[str] = []
    env_bin = os.environ.get("FFMPEG_BINARY", "").strip()
    if env_bin:
        if Path(env_bin).is_file():
            # When explicitly set, use only this binary (no conda fallback).
            return [env_bin]
        logger.warning("FFMPEG_BINARY=%s not found; falling back to discovery", env_bin)
    for fallback in ("/usr/bin/ffmpeg", "/bin/ffmpeg"):
        if Path(fallback).is_file():
            out.append(fallback)
    which = shutil.which("ffmpeg")
    if which and not _is_conda_ffmpeg(which):
        out.append(which)
    # Conda ffmpeg last — works solo but often segfaults with concurrency > 1
    if which and _is_conda_ffmpeg(which):
        out.append(which)
    seen: set[str] = set()
    uniq: list[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _ffmpeg_env(*, clean_libs: bool) -> Optional[dict]:
    import os

    if not clean_libs:
        return None
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("DYLD_LIBRARY_PATH", None)
    return env


def _load_via_ffmpeg(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    """Decode compressed audio via ffmpeg → temp WAV → soundfile (avoids pipe heap issues)."""
    with _FFMPEG_LOCK:
        return _load_via_ffmpeg_unlocked(path, target_sr)


def _load_via_ffmpeg_unlocked(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    import os
    import tempfile

    candidates = _ffmpeg_candidates()
    if not candidates:
        raise RuntimeError(
            f"Cannot decode {path!r}: ffmpeg not found on PATH "
            "(needed for .m4a / compressed audio). "
            "Install ffmpeg or set FFMPEG_BINARY=/usr/bin/ffmpeg."
        )

    errors: list[str] = []
    for ffmpeg in candidates:
        # System ffmpeg: always clean conda libs. Conda ffmpeg: try clean then env.
        is_system = ffmpeg.startswith("/usr/") or ffmpeg.startswith("/bin/")
        attempt_flags = (True,) if is_system else (True, False)

        for clean_libs in attempt_flags:
            label = "clean-libs" if clean_libs else "env"
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                cmd = [
                    ffmpeg,
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    path,
                    "-ac",
                    "1",
                    "-ar",
                    str(target_sr),
                    "-f",
                    "wav",
                    tmp_path,
                ]
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    check=False,
                    env=_ffmpeg_env(clean_libs=clean_libs),
                )
                if proc.returncode != 0 or not Path(tmp_path).is_file() or Path(tmp_path).stat().st_size == 0:
                    err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
                    errors.append(f"{ffmpeg} [{label}]: {err or proc.returncode}")
                    continue
                wav, sr = _load_via_soundfile(tmp_path, target_sr)
                logger.info("Decoded %s via ffmpeg=%s (%s)", path, ffmpeg, label)
                return wav, sr
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{ffmpeg} [{label}]: {exc}")
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    raise RuntimeError(
        f"ffmpeg failed for {path!r}. Tried: "
        + " | ".join(errors)
        + ". Tip: export FFMPEG_BINARY=/usr/bin/ffmpeg"
    )


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
