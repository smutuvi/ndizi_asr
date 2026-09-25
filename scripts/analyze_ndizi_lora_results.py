#!/usr/bin/env python3
"""Compare baseline vs LoRA Ndizi evals: per-dataset metrics + best/worst hyps.

Example (on the GPU box):

  python scripts/analyze_ndizi_lora_results.py \\
    --baseline results/baseline_sunbird51_ndizi_test \\
    --lora results/sunbird51_ndizi_lora_test \\
    --top 5

Play the worst LoRA samples (reloads audio from Hub):

  python scripts/analyze_ndizi_lora_results.py \\
    --baseline results/baseline_sunbird51_ndizi_test \\
    --lora results/sunbird51_ndizi_lora_test \\
    --top 5 --play worst --play-n 3

Or only export WAVs (good for remote / no speakers):

  python scripts/analyze_ndizi_lora_results.py \\
    --baseline results/baseline_sunbird51_ndizi_test \\
    --lora results/sunbird51_ndizi_lora_test \\
    --top 5 --export-audio /tmp/ndizi_worst_audio --play worst
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from jiwer import wer as jiwer_wer

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _find_metrics(exp_dir: Path) -> Path:
    hits = sorted(exp_dir.glob("*/metrics.json"))
    if hits:
        return hits[0]
    summary = exp_dir / "summary.json"
    if summary.is_file():
        return summary
    raise FileNotFoundError(f"No metrics.json under {exp_dir}")


def _find_predictions(exp_dir: Path) -> Path:
    hits = sorted(exp_dir.glob("*/predictions.jsonl"))
    if not hits:
        raise FileNotFoundError(f"No predictions.jsonl under {exp_dir}")
    return hits[0]


def _load_metrics(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "overall" in raw:
        return raw
    if isinstance(raw, dict) and "models" in raw:
        models = raw["models"]
        first = next(iter(models.values()))
        return first
    raise ValueError(f"Unrecognized metrics shape in {path}")


def _fmt_pct(x: Any) -> str:
    if x is None:
        return "   n/a"
    return f"{100.0 * float(x):6.2f}%"


def _score_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("error"):
                continue
            ref = str(r.get("reference_normalized") or r.get("reference") or "")
            hyp = str(r.get("prediction_normalized") or r.get("prediction") or "")
            if not ref.strip():
                continue
            try:
                w = float(jiwer_wer(ref, hyp))
            except Exception:
                w = 1.0
            r["_wer"] = w
            rows.append(r)
    return rows


def _print_examples(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n### {title}")
    for i, r in enumerate(rows, 1):
        ref = str(r.get("reference") or "")
        hyp = str(r.get("prediction") or "")
        print(
            f"\n{i}. WER={r['_wer']:.3f}  dataset={r.get('dataset_id')}  "
            f"id={r.get('sample_id')}"
        )
        print(f"   REF: {ref[:400]}{'…' if len(ref) > 400 else ''}")
        print(f"   HYP: {hyp[:400]}{'…' if len(hyp) > 400 else ''}")


def _row_index(row: dict[str, Any]) -> int:
    """Parse trailing :idx from sample_id like smutuvi_ndizi-1:<raw>:<idx>."""
    sid = str(row.get("sample_id") or "")
    try:
        return int(sid.rsplit(":", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Cannot parse row index from sample_id={sid!r}") from exc


def _load_waveform_for_row(row: dict[str, Any], *, split: str) -> tuple[Any, int]:
    """Reload audio from the Hub dataset using dataset_id + sample index."""
    import numpy as np
    from datasets import Audio, load_dataset

    dataset_id = str(row.get("dataset_id") or "").strip()
    if not dataset_id:
        raise ValueError(f"Row missing dataset_id: {row.get('sample_id')}")
    idx = _row_index(row)
    ds = load_dataset(dataset_id, split=split)
    # Prefer an audio feature column
    audio_col = None
    cols = list(getattr(ds, "column_names", []) or [])
    for cand in ("audio", "audio_path", "path", "file", "wav", "speech"):
        if cand in cols:
            audio_col = cand
            break
    if audio_col is None:
        raise ValueError(f"No audio column in {dataset_id}; columns={cols}")
    if audio_col == "audio":
        ds = ds.cast_column(audio_col, Audio(sampling_rate=16000))
    if idx < 0 or idx >= len(ds):
        raise IndexError(f"Index {idx} out of range for {dataset_id}:{split} (n={len(ds)})")
    item = ds[idx]
    audio = item[audio_col]
    if isinstance(audio, dict) and "array" in audio:
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio.get("sampling_rate") or 16000)
        return arr, sr
    if isinstance(audio, str):
        from ndizi_asr.audio import load_waveform

        return load_waveform(audio)
    raise TypeError(f"Unsupported audio payload type: {type(audio)}")


def _write_wav(path: Path, array: Any, sr: int) -> None:
    import numpy as np
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    sf.write(str(path), arr, sr)


def _pulse_available() -> bool:
    """True only if a PulseAudio/PipeWire server looks reachable."""
    import os
    import socket

    if os.environ.get("PULSE_SERVER"):
        return True
    # Default native UNIX socket
    for sock in (
        Path(os.environ.get("XDG_RUNTIME_DIR", "")) / "pulse" / "native",
        Path(f"/run/user/{os.getuid()}/pulse/native"),
    ):
        if sock.is_socket():
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(0.2)
                s.connect(str(sock))
                s.close()
                return True
            except OSError:
                continue
    return False


def _play_wav(path: Path) -> bool:
    """Play a WAV with the first available local player. Returns True on success."""
    players: list[list[str]] = []
    if shutil.which("ffplay"):
        players.append(["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(path)])
    if sys.platform == "darwin" and shutil.which("afplay"):
        players.append(["afplay", str(path)])
    if shutil.which("aplay"):
        players.append(["aplay", "-q", str(path)])
    if _pulse_available() and shutil.which("paplay"):
        players.append(["paplay", str(path)])

    if not players:
        print(
            "   No usable audio player (ffplay/afplay/aplay; paplay skipped — no PulseAudio). "
            f"File saved at: {path}"
        )
        return False

    for cmd in players:
        print(f"   playing with {cmd[0]} …")
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        except Exception as exc:
            print(f"   {cmd[0]} failed: {exc}")
            continue
        err = (proc.stderr or "") + (proc.stdout or "")
        if proc.returncode == 0 and "Connection refused" not in err:
            return True
        if err.strip():
            print(f"   {cmd[0]} exit={proc.returncode}: {err.strip()[:200]}")
        else:
            print(f"   {cmd[0]} exit={proc.returncode}")
    print(f"   Playback failed on this machine (likely headless). Keep the WAV: {path}")
    return False


def _safe_stem(row: dict[str, Any], rank: int, kind: str) -> str:
    sid = str(row.get("sample_id") or "unknown").replace("/", "_").replace(":", "_")
    wer = float(row.get("_wer") or 0.0)
    return f"{kind}_{rank:02d}_wer{wer:.3f}_{sid}"


def _handle_audio(
    rows: list[dict[str, Any]],
    *,
    kind: str,
    play: bool,
    export_dir: Optional[Path],
    split: str,
) -> None:
    if not rows:
        return
    # Always persist somewhere lasting — temp dirs are easy to lose on headless boxes.
    if export_dir is None:
        export_dir = Path("/tmp/ndizi_asr_play") / kind
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n### Audio ({kind}) → {export_dir.resolve()}")
    played_any = False
    for i, row in enumerate(rows, 1):
        print(
            f"\n{i}. loading audio for {row.get('sample_id')} "
            f"(dataset={row.get('dataset_id')}, WER={row['_wer']:.3f})"
        )
        try:
            arr, sr = _load_waveform_for_row(row, split=split)
        except Exception as exc:
            print(f"   FAILED to load audio: {exc}")
            continue
        wav_path = export_dir / f"{_safe_stem(row, i, kind)}.wav"
        _write_wav(wav_path, arr, sr)
        meta_path = wav_path.with_suffix(".txt")
        meta_path.write_text(
            "\n".join(
                [
                    f"sample_id: {row.get('sample_id')}",
                    f"dataset_id: {row.get('dataset_id')}",
                    f"wer: {row['_wer']:.6f}",
                    f"REF: {row.get('reference')}",
                    f"HYP: {row.get('prediction')}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"   wrote {wav_path}")
        if play:
            if _play_wav(wav_path):
                played_any = True
            if sys.stdin.isatty() and i < len(rows):
                try:
                    input("   [Enter] next clip  (Ctrl-C to stop) ")
                except KeyboardInterrupt:
                    print("\n   stopped.")
                    break

    print(f"\nAudio + transcripts under: {export_dir.resolve()}")
    if play and not played_any:
        print(
            "Playback did not work here (typical on GPU servers without PulseAudio).\n"
            "Copy to your laptop, then open the WAVs locally, e.g.:\n"
            f"  scp 'user@host:{export_dir.resolve()}/*.wav' ./ndizi_worst/"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--lora", type=Path, required=True)
    p.add_argument("--top", type=int, default=5, help="Best/worst example count")
    p.add_argument(
        "--play",
        choices=["none", "worst", "best", "both"],
        default="none",
        help="Play LoRA examples (reloads audio from Hub via dataset_id + sample index)",
    )
    p.add_argument(
        "--play-n",
        type=int,
        default=None,
        help="How many clips to play/export (default: --top)",
    )
    p.add_argument(
        "--export-audio",
        type=Path,
        default=None,
        help="Write WAVs (+ .txt REF/HYP) here; useful on remote servers",
    )
    p.add_argument(
        "--split",
        default="test",
        help="Hub split used when reloading audio (default: test)",
    )
    args = p.parse_args()

    base_m = _load_metrics(_find_metrics(args.baseline))
    lora_m = _load_metrics(_find_metrics(args.lora))

    datasets = sorted(
        set(base_m.get("per_dataset") or {}) | set(lora_m.get("per_dataset") or {})
    )

    print("## Per-dataset comparison (normalized WER/CER preferred when present)\n")
    hdr = (
        f"{'dataset':28} {'n':>5}  "
        f"{'base WER':>9} {'lora WER':>9} {'ΔWER':>8}  "
        f"{'base CER':>9} {'lora CER':>9} {'ΔCER':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    def pick(block: dict, key: str) -> Any:
        return block.get(f"{key}_normalized", block.get(key))

    for ds in datasets + ["__overall__"]:
        if ds == "__overall__":
            b, l = base_m.get("overall") or {}, lora_m.get("overall") or {}
            label = "ALL"
        else:
            b = (base_m.get("per_dataset") or {}).get(ds) or {}
            l = (lora_m.get("per_dataset") or {}).get(ds) or {}
            label = ds
        bw, lw = pick(b, "wer"), pick(l, "wer")
        bc, lc = pick(b, "cer"), pick(l, "cer")
        n = l.get("num_scored") or b.get("num_scored") or l.get("n") or b.get("n") or ""
        dwer = (float(lw) - float(bw)) if bw is not None and lw is not None else None
        dcer = (float(lc) - float(bc)) if bc is not None and lc is not None else None
        print(
            f"{label:28} {str(n):>5}  "
            f"{_fmt_pct(bw):>9} {_fmt_pct(lw):>9} "
            f"{(_fmt_pct(dwer) if dwer is not None else '   n/a'):>8}  "
            f"{_fmt_pct(bc):>9} {_fmt_pct(lc):>9} "
            f"{(_fmt_pct(dcer) if dcer is not None else '   n/a'):>8}"
        )

    print(
        "\nNegative ΔWER = LoRA better. Overall numbers should match your "
        "0.4465 → 0.4146 WER comparison."
    )

    lora_rows = _score_rows(_find_predictions(args.lora))
    lora_rows.sort(key=lambda r: r["_wer"])
    k = max(1, int(args.top))
    best = lora_rows[:k]
    worst = list(reversed(lora_rows[-k:]))
    _print_examples(f"LoRA best {k} (lowest WER)", best)
    _print_examples(f"LoRA worst {k} (highest WER)", worst)

    play_n = max(1, int(args.play_n if args.play_n is not None else k))
    want_play = args.play != "none"
    want_export = args.export_audio is not None
    if want_play or want_export:
        do_worst = args.play in {"worst", "both"} or (args.play == "none" and want_export)
        do_best = args.play in {"best", "both"} or (args.play == "none" and want_export)
        if do_worst:
            _handle_audio(
                worst[:play_n],
                kind="worst",
                play=args.play in {"worst", "both"},
                export_dir=(args.export_audio / "worst") if args.export_audio else None,
                split=args.split,
            )
        if do_best:
            _handle_audio(
                best[:play_n],
                kind="best",
                play=args.play in {"best", "both"},
                export_dir=(args.export_audio / "best") if args.export_audio else None,
                split=args.split,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
