"""Dataset & audio quality audit, glued-punctuation repair, and outlier filtering.

Provides:
1. ``audit_and_filter_hf_split`` — used inside ``whisper_trainer`` and ``whisper_grpo``
   to repair glued punctuation in-place, filter corrupted audio/transcript pairs
   (extreme speech-rate outliers, empty/1-word labels, sub-0.5s clips), and save
   per-split CSV + JSON quality reports (what was flagged, what was filtered, and
   full statistics of what remains).
2. ``ndizi-asr-qc`` CLI — runs a standalone quality audit across ``train``,
   ``validation``, and ``test`` splits of a Hugging Face dataset (e.g.
   ``smutuvi/ndizi-1-sw``) and optionally re-scores existing prediction CSV/JSONL
   files to show:
   - Old (glued-punctuation) WER/CER vs. New (space-separated punctuation) WER/CER
   - Full test set WER/CER vs. QC-clean test set WER/CER (excluding swapped/truncated
     reference outliers).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from jiwer import cer as jiwer_cer, process_words, wer as jiwer_wer

from .config import NormalizationConfig
from .metrics import clean_glued_punctuation, normalize_text

logger = logging.getLogger(__name__)

_GLUED_PUNCT_RE = re.compile(r"\w[.,?!;:]\w", flags=re.UNICODE)
_OLD_PUNCT_TABLE = str.maketrans("", "", string.punctuation + "።፣፤፥፦፧፨።،؛؟«»\"'„“”‘’…–—")


def _norm_legacy_glued(text: str) -> str:
    """Reproduce the legacy normalization where punctuation was deleted without spaces."""
    out = (text or "").lower().translate(_OLD_PUNCT_TABLE)
    return re.sub(r"\s+", " ", out).strip()


def _percentile(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def _dist_summary(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"count": 0, "sum": 0.0, "mean": 0.0, "median": 0.0, "p5": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "count": int(arr.size),
        "sum": round(float(arr.sum()), 2),
        "mean": round(float(arr.mean()), 2),
        "median": round(float(np.median(arr)), 2),
        "p5": round(float(np.percentile(arr, 5)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "min": round(float(arr.min()), 2),
        "max": round(float(arr.max()), 2),
    }


def inspect_row_quality(
    *,
    text: str,
    duration_sec: float,
    min_duration_s: float = 0.5,
    max_duration_s: float = 30.0,
    min_words: int = 2,
    min_chars_per_sec: float = 2.5,
    max_chars_per_sec: float = 26.0,
    chunk_long_audio: bool = False,
) -> Tuple[bool, str, List[str], Dict[str, Any]]:
    """Inspect a single (audio_duration, transcript) pair and return QC decision.

    Returns:
        (keep, repaired_text, issue_flags, metrics_dict)
    """
    raw_text = str(text or "").strip()
    has_glued = bool(_GLUED_PUNCT_RE.search(raw_text))
    repaired = clean_glued_punctuation(raw_text)
    norm_cfg = NormalizationConfig(mode="default", lowercase=True, strip_punctuation=True, collapse_whitespace=True)
    norm_txt = normalize_text(repaired, norm_cfg)
    words = norm_txt.split() if norm_txt else []
    n_words = len(words)
    n_chars = len(norm_txt)
    dur = float(duration_sec or 0.0)
    cps = n_chars / max(dur, 0.1) if dur > 0 else 0.0
    wps = n_words / max(dur, 0.1) if dur > 0 else 0.0

    flags: List[str] = []
    drop_reasons: List[str] = []

    if has_glued:
        flags.append("glued_punctuation_repaired")

    if n_words < min_words:
        flags.append("empty_or_tiny_transcript")
        drop_reasons.append("empty_or_tiny_transcript")

    if dur > 0 and dur < min_duration_s:
        flags.append("duration_too_short")
        drop_reasons.append("duration_too_short")

    if dur > max_duration_s:
        if chunk_long_audio:
            flags.append("duration_gt_30s_chunked")
        else:
            flags.append("duration_gt_30s_dropped")
            drop_reasons.append("duration_gt_30s_dropped")

    if dur >= 2.0 and n_words >= min_words:
        if cps < min_chars_per_sec:
            flags.append("speech_rate_too_low_truncated_or_silence")
            drop_reasons.append("speech_rate_too_low_truncated_or_silence")
        elif cps > max_chars_per_sec:
            flags.append("speech_rate_too_high_corrupted_or_paste")
            drop_reasons.append("speech_rate_too_high_corrupted_or_paste")

    keep = len(drop_reasons) == 0
    info = {
        "duration_sec": round(dur, 2),
        "word_count": n_words,
        "char_count": n_chars,
        "chars_per_sec": round(cps, 2),
        "words_per_sec": round(wps, 2),
        "had_glued_punctuation": has_glued,
        "drop_reasons": drop_reasons,
    }
    return keep, repaired, flags, info


def audit_and_filter_hf_split(
    dataset: Any,
    *,
    split_name: str,
    audio_col: str = "audio",
    text_col: str = "transcript",
    max_duration_s: float = 30.0,
    min_duration_s: float = 0.5,
    min_words: int = 2,
    min_chars_per_sec: float = 2.5,
    max_chars_per_sec: float = 26.0,
    chunk_long_audio: bool = False,
    apply_filter: bool = True,
    repair_glued_punctuation: bool = True,
    report_dir: Optional[Path] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Audit a Hugging Face dataset split, repair glued punctuation, and optionally filter outliers."""
    n_total = len(dataset)
    issue_counts: Counter = Counter()
    drop_reason_counts: Counter = Counter()
    lang_before: Counter = Counter()
    lang_after: Counter = Counter()
    district_after: Counter = Counter()

    durs_before: List[float] = []
    words_before: List[float] = []
    cps_before: List[float] = []

    durs_after: List[float] = []
    words_after: List[float] = []
    cps_after: List[float] = []

    flagged_rows: List[Dict[str, Any]] = []
    keep_indices: List[int] = []
    repaired_texts: List[str] = []

    # Prefer metadata duration_sec column if present for fast pass; else compute from audio array
    cols = set(dataset.column_names)
    has_dur_col = "duration_sec" in cols or "audio_duration" in cols
    dur_col = "duration_sec" if "duration_sec" in cols else ("audio_duration" if "audio_duration" in cols else None)

    # Read metadata columns without decoding audio if duration_sec is available
    select_cols = [c for c in [text_col, dur_col, "sample_id", "speaker_id", "language", "district", "stage"] if c and c in cols]
    meta_ds = dataset.select_columns(select_cols) if (dur_col and select_cols) else dataset

    for idx, row in enumerate(meta_ds):
        raw_txt = str(row.get(text_col) or "")
        if dur_col and row.get(dur_col) is not None:
            dur = float(row[dur_col])
        else:
            aud = row.get(audio_col)
            if isinstance(aud, dict) and aud.get("array") is not None:
                sr = int(aud.get("sampling_rate") or 16000)
                dur = len(aud["array"]) / max(sr, 1)
            else:
                dur = 10.0  # unknown duration fallback

        lang = str(row.get("language") or "unknown")
        dist = str(row.get("district") or "unknown")
        lang_before[lang] += 1

        keep, repaired, flags, info = inspect_row_quality(
            text=raw_txt,
            duration_sec=dur,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            min_words=min_words,
            min_chars_per_sec=min_chars_per_sec,
            max_chars_per_sec=max_chars_per_sec,
            chunk_long_audio=chunk_long_audio,
        )

        durs_before.append(dur)
        words_before.append(float(info["word_count"]))
        cps_before.append(float(info["chars_per_sec"]))

        for fl in flags:
            issue_counts[fl] += 1
        for dr in info["drop_reasons"]:
            drop_reason_counts[dr] += 1

        action = "kept"
        if not keep and apply_filter:
            action = "filtered_out"
        elif "glued_punctuation_repaired" in flags:
            action = "repaired_kept"

        if flags:
            flagged_rows.append(
                {
                    "split": split_name,
                    "row_index": idx,
                    "speaker_id": row.get("speaker_id", ""),
                    "language": lang,
                    "district": dist,
                    "stage": row.get("stage", ""),
                    "duration_sec": info["duration_sec"],
                    "word_count": info["word_count"],
                    "chars_per_sec": info["chars_per_sec"],
                    "action": action,
                    "issue_flags": "|".join(flags),
                    "drop_reasons": "|".join(info["drop_reasons"]),
                    "transcript_preview": raw_txt[:140],
                }
            )

        if keep or not apply_filter:
            keep_indices.append(idx)
            repaired_texts.append(repaired if repair_glued_punctuation else raw_txt)
            durs_after.append(dur)
            words_after.append(float(info["word_count"]))
            cps_after.append(float(info["chars_per_sec"]))
            lang_after[lang] += 1
            district_after[dist] += 1

    # Apply repair and filtering to dataset if requested
    out_ds = dataset
    if apply_filter and len(keep_indices) < n_total:
        out_ds = out_ds.select(keep_indices)
    if repair_glued_punctuation:
        out_ds = out_ds.map(
            lambda batch: {text_col: [clean_glued_punctuation(t) for t in batch[text_col]]},
            batched=True,
            desc=f"repair_glued_punct_{split_name}",
        )

    summary: Dict[str, Any] = {
        "split": split_name,
        "total_before": n_total,
        "total_after": len(keep_indices),
        "filtered_out_count": n_total - len(keep_indices),
        "kept_pct": round(100.0 * len(keep_indices) / max(n_total, 1), 2),
        "hours_before": round(sum(durs_before) / 3600.0, 2),
        "hours_after": round(sum(durs_after) / 3600.0, 2),
        "issues_identified": dict(issue_counts),
        "filtered_out_by_reason": dict(drop_reason_counts),
        "stats_before": {
            "duration_sec": _dist_summary(durs_before),
            "word_count": _dist_summary(words_before),
            "chars_per_sec": _dist_summary(cps_before),
            "languages": dict(lang_before),
        },
        "stats_remaining": {
            "duration_sec": _dist_summary(durs_after),
            "word_count": _dist_summary(words_after),
            "chars_per_sec": _dist_summary(cps_after),
            "languages": dict(lang_after),
            "districts": dict(district_after),
        },
    }

    if report_dir is not None:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        csv_path = report_dir / f"qc_flagged_{split_name}.csv"
        fieldnames = [
            "split",
            "row_index",
            "speaker_id",
            "language",
            "district",
            "stage",
            "duration_sec",
            "word_count",
            "chars_per_sec",
            "action",
            "issue_flags",
            "drop_reasons",
            "transcript_preview",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flagged_rows)

        json_path = report_dir / f"qc_summary_{split_name}.json"
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(
            "[%s] QC summary: %d -> %d kept (%.1f%%, %.2fh -> %.2fh) | glued_repaired=%d | filtered=%d | wrote %s",
            split_name,
            n_total,
            len(keep_indices),
            summary["kept_pct"],
            summary["hours_before"],
            summary["hours_after"],
            issue_counts.get("glued_punctuation_repaired", 0),
            n_total - len(keep_indices),
            csv_path,
        )

    return out_ds, summary


def rescore_predictions_file(
    pred_path: Path,
    *,
    report_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Re-score an existing predictions CSV/JSONL or utterances.csv with:
    1) Legacy glued-punctuation normalization vs. Fixed space-punctuation normalization
    2) All test clips vs. QC-clean test clips (excluding swapped/corrupted reference outliers).
    """
    rows: List[Dict[str, Any]] = []
    if pred_path.suffix.lower() == ".jsonl":
        with pred_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        with pred_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    norm_cfg = NormalizationConfig(mode="default", lowercase=True, strip_punctuation=True, collapse_whitespace=True)

    refs_raw, hyps_raw = [], []
    refs_legacy, hyps_legacy = [], []
    refs_fixed, hyps_fixed = [], []
    refs_clean, hyps_clean = [], []

    outlier_rows: List[Dict[str, Any]] = []
    glued_count = 0

    for idx, r in enumerate(rows):
        ref = str(r.get("reference") or "")
        hyp = str(r.get("prediction") if r.get("prediction") is not None else (r.get("hypothesis") or r.get("fusion") or ""))
        if not ref.strip():
            continue
        if _GLUED_PUNCT_RE.search(ref):
            glued_count += 1

        r_leg = _norm_legacy_glued(ref)
        h_leg = _norm_legacy_glued(hyp)
        r_fix = normalize_text(ref, norm_cfg)
        h_fix = normalize_text(hyp, norm_cfg)

        refs_raw.append(ref)
        hyps_raw.append(hyp)
        refs_legacy.append(r_leg)
        hyps_legacy.append(h_leg)
        refs_fixed.append(r_fix)
        hyps_fixed.append(h_fix)

        utt_wer = float(process_words(r_fix, h_fix).wer) if r_fix else 0.0
        utt_cer = float(jiwer_cer(r_fix, h_fix)) if r_fix else 0.0
        ref_w = len(r_fix.split())
        hyp_w = len(h_fix.split())
        len_ratio = hyp_w / float(max(ref_w, 1))

        # Identify extreme reference-mismatch / truncated-reference / corrupted-audio test rows
        outlier_reason = None
        if utt_wer >= 0.90 and utt_cer >= 0.45:
            if len_ratio >= 1.55:
                outlier_reason = "truncated_reference_hyp_much_longer"
            elif len_ratio <= 0.45:
                outlier_reason = "severe_audio_drop_or_unannotated_noise"
            else:
                outlier_reason = "swapped_reference_or_corrupted_audio"
        elif utt_wer >= 1.30 and len_ratio >= 1.60:
            outlier_reason = "truncated_reference_hyp_much_longer"

        if outlier_reason:
            outlier_rows.append(
                {
                    "row_index": idx,
                    "sample_id": r.get("sample_id") or r.get("audio_file") or f"row_{idx}",
                    "outlier_reason": outlier_reason,
                    "wer_fixed": round(utt_wer, 4),
                    "cer_fixed": round(utt_cer, 4),
                    "ref_words": ref_w,
                    "hyp_words": hyp_w,
                    "reference_preview": ref[:120],
                    "hypothesis_preview": hyp[:120],
                }
            )
        else:
            refs_clean.append(r_fix)
            hyps_clean.append(h_fix)

    res = {
        "file": str(pred_path),
        "n_total": len(refs_fixed),
        "glued_punctuation_refs": glued_count,
        "glued_punctuation_pct": round(100.0 * glued_count / max(len(refs_fixed), 1), 2),
        "wer_raw": round(float(jiwer_wer(refs_raw, hyps_raw)), 4),
        "cer_raw": round(float(jiwer_cer(refs_raw, hyps_raw)), 4),
        "wer_legacy_glued_norm": round(float(jiwer_wer(refs_legacy, hyps_legacy)), 4),
        "cer_legacy_glued_norm": round(float(jiwer_cer(refs_legacy, hyps_legacy)), 4),
        "wer_fixed_punct_norm": round(float(jiwer_wer(refs_fixed, hyps_fixed)), 4),
        "cer_fixed_punct_norm": round(float(jiwer_cer(refs_fixed, hyps_fixed)), 4),
        "delta_wer_from_punct_fix": round(
            float(jiwer_wer(refs_fixed, hyps_fixed)) - float(jiwer_wer(refs_legacy, hyps_legacy)), 4
        ),
        "n_clean_after_outlier_filter": len(refs_clean),
        "n_outliers_flagged": len(outlier_rows),
        "wer_clean_subset": round(float(jiwer_wer(refs_clean, hyps_clean)), 4) if refs_clean else 0.0,
        "cer_clean_subset": round(float(jiwer_cer(refs_clean, hyps_clean)), 4) if refs_clean else 0.0,
        "outliers_by_reason": dict(Counter(x["outlier_reason"] for x in outlier_rows)),
    }

    if report_dir is not None:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        out_csv = report_dir / f"test_outliers_{pred_path.parent.name}_{pred_path.stem}.csv"
        if outlier_rows:
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(outlier_rows[0].keys()))
                writer.writeheader()
                writer.writerows(outlier_rows)
        res["outliers_csv"] = str(out_csv)

    return res


def _print_split_table(summaries: List[Dict[str, Any]]) -> None:
    print("\n=== Dataset Quality Audit Summary (Before vs. After QC Filtering) ===")
    header = (
        f"{'Split':12} | {'Before':>6} | {'Kept':>6} | {'Kept%':>6} | "
        f"{'Hrs(B->A)':>12} | {'GluedFixed':>10} | {'Dropped(>30s)':>13} | {'Dropped(Rate/Tiny)':>18}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        iss = s["issues_identified"]
        drp = s["filtered_out_by_reason"]
        gt30 = drp.get("duration_gt_30s_dropped", 0)
        other_drp = s["filtered_out_count"] - gt30
        hrs = f"{s['hours_before']:.1f}->{s['hours_after']:.1f}h"
        print(
            f"{s['split']:12} | {s['total_before']:6d} | {s['total_after']:6d} | "
            f"{s['kept_pct']:5.1f}% | {hrs:>12} | "
            f"{iss.get('glued_punctuation_repaired', 0):10d} | {gt30:13d} | {other_drp:18d}"
        )

    print("\n=== Detailed Issues Identified & Remaining Split Statistics ===")
    for s in summaries:
        rem = s["stats_remaining"]
        d_stat = rem["duration_sec"]
        w_stat = rem["word_count"]
        c_stat = rem["chars_per_sec"]
        print(f"\n[{s['split'].upper()}]")
        print(f"  Issues identified   : {json.dumps(s['issues_identified'])}")
        print(f"  Filtered by reason  : {json.dumps(s['filtered_out_by_reason'])}")
        print(
            f"  Remaining duration  : mean={d_stat['mean']}s, median={d_stat['median']}s, "
            f"p5={d_stat['p5']}s, p95={d_stat['p95']}s (total={s['hours_after']}h)"
        )
        print(
            f"  Remaining word count: mean={w_stat['mean']}w, median={w_stat['median']}w, "
            f"p5={w_stat['p5']}w, p95={w_stat['p95']}w"
        )
        print(
            f"  Remaining speech rt : mean={c_stat['mean']} chars/s, median={c_stat['median']} chars/s, "
            f"p5={c_stat['p5']}, p95={c_stat['p95']}"
        )
        if rem.get("languages"):
            print(f"  Remaining languages : {json.dumps(rem['languages'])}")


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    p = argparse.ArgumentParser(
        prog="ndizi-asr-qc",
        description="Audit train/validation/test audio & transcript quality and re-score predictions with fixed punctuation.",
    )
    p.add_argument("--dataset", default="smutuvi/ndizi-1-sw", help="Hugging Face dataset ID (default: smutuvi/ndizi-1-sw)")
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Splits to audit (default: train validation test)",
    )
    p.add_argument("--text-column", default="transcript")
    p.add_argument("--audio-column", default="audio")
    p.add_argument("--max-duration", type=float, default=30.0)
    p.add_argument("--min-duration", type=float, default=0.5)
    p.add_argument("--min-chars-per-sec", type=float, default=2.5)
    p.add_argument("--max-chars-per-sec", type=float, default=26.0)
    p.add_argument(
        "--chunk-long-audio",
        action="store_true",
        help="Treat >30s clips as kept (chunked via MMS_FA/VAD) instead of dropped",
    )
    p.add_argument(
        "--rescore-predictions",
        nargs="*",
        default=[],
        help="Optional prediction CSV/JSONL files (or experiment dirs) to re-score with the punctuation fix & outlier audit",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qc_audit"),
        help="Directory to write qc_summary.json and per-split flagged CSVs",
    )
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: List[Dict[str, Any]] = []

    if args.dataset and args.dataset.lower() != "none":
        from huggingface_hub import hf_hub_download

        for sp in args.splits:
            ds_rows: Optional[Any] = None
            # Fast path: try downloading data/<split>/metadata.csv directly (0.5s vs downloading all MP3s)
            for cand_rel in (f"data/{sp}/metadata.csv", f"{sp}/metadata.csv", f"{sp}.csv"):
                try:
                    local_csv = hf_hub_download(
                        repo_id=args.dataset,
                        filename=cand_rel,
                        repo_type="dataset",
                    )
                    with open(local_csv, "r", encoding="utf-8") as f:
                        csv_list = list(csv.DictReader(f))
                    if csv_list:
                        from datasets import Dataset

                        ds_rows = Dataset.from_list(csv_list)
                        break
                except Exception:  # noqa: BLE001
                    continue

            if ds_rows is None:
                try:
                    from datasets import load_dataset

                    ds_rows = load_dataset(args.dataset, split=sp)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not load %s split=%s: %s", args.dataset, sp, exc)
                    continue

            _, summary = audit_and_filter_hf_split(
                ds_rows,
                split_name=sp,
                audio_col=args.audio_column,
                text_col=args.text_column,
                max_duration_s=args.max_duration,
                min_duration_s=args.min_duration,
                min_chars_per_sec=args.min_chars_per_sec,
                max_chars_per_sec=args.max_chars_per_sec,
                chunk_long_audio=args.chunk_long_audio,
                apply_filter=True,
                repair_glued_punctuation=False,
                report_dir=args.output_dir,
            )
            all_summaries.append(summary)

        if all_summaries:
            _print_split_table(all_summaries)

    rescored_list: List[Dict[str, Any]] = []
    for raw_p in args.rescore_predictions:
        cand = Path(raw_p)
        files_to_score: List[Path] = []
        if cand.is_file():
            files_to_score.append(cand)
        elif cand.is_dir():
            files_to_score.extend(sorted(cand.rglob("predictions.csv")))
            files_to_score.extend(sorted(cand.rglob("utterances.csv")))
            files_to_score.extend(sorted(cand.rglob("fusion_predictions.jsonl")))
        for fp in files_to_score:
            r_info = rescore_predictions_file(fp, report_dir=args.output_dir)
            rescored_list.append(r_info)

    if rescored_list:
        print("\n=== Re-scored Predictions: Glued-Punctuation Fix & Outlier Audit ===")
        hdr = (
            f"{'Prediction File':52} | {'n':>4} | {'Glued%':>6} | "
            f"{'Legacy WER':>10} | {'Fixed WER↓':>10} | {'Fixed CER↓':>10} | {'Clean WER (n)':>16}"
        )
        print(hdr)
        print("-" * len(hdr))
        for r in rescored_list:
            short_name = str(Path(r["file"]).parent.name + "/" + Path(r["file"]).name)[-52:]
            clean_str = f"{r['wer_clean_subset']:.4f} ({r['n_clean_after_outlier_filter']})"
            print(
                f"{short_name:52} | {r['n_total']:4d} | {r['glued_punctuation_pct']:5.1f}% | "
                f"{r['wer_legacy_glued_norm']:10.4f} | {r['wer_fixed_punct_norm']:10.4f} | "
                f"{r['cer_fixed_punct_norm']:10.4f} | {clean_str:>16}"
            )
            if r["outliers_by_reason"]:
                print(f"   -> Flagged test outliers ({r['n_outliers_flagged']}): {json.dumps(r['outliers_by_reason'])}")
                if r.get("outliers_csv"):
                    print(f"   -> Details CSV: {r['outliers_csv']}")

    combined_path = args.output_dir / "qc_audit_full_report.json"
    combined_path.write_text(
        json.dumps({"splits": all_summaries, "rescored_predictions": rescored_list}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nFull QC report written to: {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
