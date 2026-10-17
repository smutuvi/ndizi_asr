"""Local CSV / CropConcept manifest loaders (from ndizi_eval)."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterator, Optional, Sequence

from ..audio import load_waveform
from ..config import DatasetConfig
from .base import AudioSample, BaseDatasetLoader

logger = logging.getLogger(__name__)

LANG_ALIASES = {
    "am": "am",
    "amh": "am",
    "amharic": "am",
    "am_et": "am",
    "amh_asr": "am",
    "om": "om",
    "orm": "om",
    "oromo": "om",
    "om_et": "om",
    "orm_asr": "om",
    "sw": "sw",
    "swa": "sw",
    "swahili": "sw",
    "ha": "ha",
    "hau": "ha",
    "hausa": "ha",
}

LANG_DISPLAY = {
    "am": "amharic",
    "om": "oromo",
    "sw": "swahili",
    "ha": "hausa",
}

REQUIRED_COLUMNS = ("audio_path", "reference", "language")


def normalize_lang(code: str) -> str:
    key = str(code).strip().lower()
    if key in LANG_ALIASES:
        return LANG_ALIASES[key]
    return key


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "t"}


def _resolve_audio_path(raw: str, csv_dir: Path, data_root: Optional[Path]) -> Path:
    p = Path(raw)
    if p.is_file():
        return p.resolve()
    candidates = []
    if data_root is not None:
        candidates.append(data_root / raw)
        candidates.append(data_root / p.name)
    candidates.append(csv_dir / raw)
    candidates.append(csv_dir / p.name)
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        f"Audio not found for {raw!r} (searched {[str(c) for c in candidates]})"
    )


class CSVManifestLoader(BaseDatasetLoader):
    """Load evaluation rows from a CSV with audio_path, reference, language."""

    def __init__(self, cfg: DatasetConfig) -> None:
        if not cfg.csv_path:
            raise ValueError("CSVManifestLoader requires dataset.csv_path")
        self.cfg = cfg
        self.csv_path = Path(cfg.csv_path).expanduser().resolve()
        self.data_root = Path(cfg.data_root).expanduser().resolve() if cfg.data_root else None

    def iter_samples(self) -> Iterator[AudioSample]:
        with self.csv_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"{self.csv_path} is empty or missing a header")
            missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
            if missing:
                raise ValueError(
                    f"{self.csv_path} missing columns {missing}; have {list(reader.fieldnames)}"
                )

            lang_filter = None
            if self.cfg.languages:
                lang_filter = {normalize_lang(k) for k in self.cfg.languages.keys()} | {
                    normalize_lang(v) for v in self.cfg.languages.values()
                }
            if self.cfg.language_filter:
                lang_filter = {normalize_lang(x) for x in self.cfg.language_filter}

            count = 0
            for i, row in enumerate(reader):
                lang = normalize_lang(str(row["language"]))
                if lang_filter is not None and lang not in lang_filter:
                    continue
                audio = _resolve_audio_path(
                    str(row["audio_path"]), self.csv_path.parent, self.data_root
                )
                array, sr = load_waveform(str(audio))
                file_name = str(row.get("file_name") or audio.name)
                yield AudioSample(
                    sample_id=f"csv:{file_name}:{i}",
                    language=LANG_DISPLAY.get(lang, lang),
                    language_config=lang,
                    reference=str(row["reference"]),
                    audio_array=array,
                    sampling_rate=sr,
                    meta={
                        "domain": str(row.get("domain") or ""),
                        "audio_path": str(audio),
                        "file_name": file_name,
                        "media_id": str(row.get("media_id") or "").strip(),
                        "submission_id": str(row.get("submission_id") or "").strip(),
                        "question_id": str(row.get("question_id") or "").strip(),
                        "question": str(row.get("question") or "").strip(),
                        "our_transcript": str(row.get("our_transcript") or "").strip(),
                        "counted_in_accuracy": str(row.get("counted_in_accuracy") or "").strip(),
                    },
                    dataset_id=self.cfg.dataset_id or f"csv:{self.csv_path.name}",
                )
                count += 1
                if self.cfg.max_samples is not None and count >= self.cfg.max_samples:
                    break


class CropConceptLoader(BaseDatasetLoader):
    """Join CropConcept media map + transcript check CSVs on media_id."""

    def __init__(self, cfg: DatasetConfig) -> None:
        if not cfg.cropconcept_media_map or not cfg.cropconcept_transcripts:
            raise ValueError(
                "CropConceptLoader requires dataset.cropconcept_media_map and "
                "dataset.cropconcept_transcripts"
            )
        if not cfg.cropconcept_audio_dir:
            raise ValueError("CropConceptLoader requires dataset.cropconcept_audio_dir")
        self.cfg = cfg
        self.media_map = Path(cfg.cropconcept_media_map).expanduser().resolve()
        self.transcripts = Path(cfg.cropconcept_transcripts).expanduser().resolve()
        self.audio_dir = Path(cfg.cropconcept_audio_dir).expanduser().resolve()

    def iter_samples(self) -> Iterator[AudioSample]:
        if not self.audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")

        with self.media_map.open(encoding="utf-8-sig", newline="") as f:
            map_rows = list(csv.DictReader(f))
        with self.transcripts.open(encoding="utf-8-sig", newline="") as f:
            tx_rows = list(csv.DictReader(f))

        if not map_rows or not tx_rows:
            raise ValueError("CropConcept CSVs have no data rows")

        for required in ("media_id", "audio_file"):
            if required not in map_rows[0]:
                raise ValueError(
                    f"{self.media_map} missing column {required!r}; have {list(map_rows[0])}"
                )
        for required in ("media_id", "corrected_transcript", "language"):
            if required not in tx_rows[0]:
                raise ValueError(
                    f"{self.transcripts} missing column {required!r}; have {list(tx_rows[0])}"
                )

        map_by_id = {
            str(r["media_id"]).strip(): r
            for r in map_rows
            if str(r.get("media_id") or "").strip()
        }
        lang_filter = (
            {normalize_lang(x) for x in self.cfg.language_filter}
            if self.cfg.language_filter
            else None
        )
        require_counted = not self.cfg.cropconcept_include_not_counted

        count = 0
        skipped = {"not_counted": 0, "empty_ref": 0, "missing_map": 0, "missing_audio": 0, "lang": 0}
        for i, tx in enumerate(tx_rows):
            media_id = str(tx.get("media_id") or "").strip()
            if not media_id:
                skipped["missing_map"] += 1
                continue
            if (
                require_counted
                and "counted_in_accuracy" in tx
                and not _truthy(tx.get("counted_in_accuracy"))
            ):
                skipped["not_counted"] += 1
                continue
            reference = str(tx.get("corrected_transcript") or "")
            if not reference.strip():
                skipped["empty_ref"] += 1
                continue
            mapped = map_by_id.get(media_id)
            if mapped is None:
                skipped["missing_map"] += 1
                continue
            lang = normalize_lang(
                str(tx.get("language") or mapped.get("detected_language") or "om")
            )
            if lang_filter is not None and lang not in lang_filter:
                skipped["lang"] += 1
                continue
            audio_name = str(mapped.get("audio_file") or "").strip()
            audio_path = self.audio_dir / audio_name
            if not audio_path.is_file():
                skipped["missing_audio"] += 1
                continue
            array, sr = load_waveform(str(audio_path))
            yield AudioSample(
                sample_id=f"cropconcept:{media_id}:{i}",
                language=LANG_DISPLAY.get(lang, lang),
                language_config=lang,
                reference=reference,
                audio_array=array,
                sampling_rate=sr,
                meta={
                    "domain": "cropconcept",
                    "media_id": media_id,
                    "submission_id": str(
                        tx.get("submission_id") or mapped.get("submission_id") or ""
                    ).strip(),
                    "question_id": str(
                        tx.get("question_id") or mapped.get("widget_id") or ""
                    ).strip(),
                    "question": str(
                        tx.get("question") or mapped.get("question") or ""
                    ).strip(),
                    "our_transcript": str(tx.get("our_transcript") or "").strip(),
                    "counted_in_accuracy": str(
                        tx.get("counted_in_accuracy") or ""
                    ).strip(),
                    "audio_path": str(audio_path),
                    "file_name": audio_name,
                },
                dataset_id=self.cfg.dataset_id or "cropconcept",
            )
            count += 1
            if self.cfg.max_samples is not None and count >= self.cfg.max_samples:
                break

        logger.info(
            "CropConcept join: kept=%d skipped_not_counted=%d skipped_empty_ref=%d "
            "skipped_missing_map=%d skipped_missing_audio=%d skipped_lang=%d",
            count,
            skipped["not_counted"],
            skipped["empty_ref"],
            skipped["missing_map"],
            skipped["missing_audio"],
            skipped["lang"],
        )


def write_manifest_csv(
    rows: Sequence[dict],
    out_csv: Path,
    *,
    relative_to: Optional[Path] = None,
) -> Path:
    out_csv = out_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["audio_path", "reference", "language", "domain", "file_name"]
    extra_cols = (
        "media_id",
        "submission_id",
        "question_id",
        "question",
        "our_transcript",
        "counted_in_accuracy",
    )
    if any(row.get(col) for row in rows for col in extra_cols):
        fieldnames.extend(extra_cols)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            audio = Path(str(row["audio_path"]))
            audio_cell = (
                str(audio.relative_to(relative_to)) if relative_to is not None else str(audio)
            )
            out_row = {
                "audio_path": audio_cell,
                "reference": row.get("reference", ""),
                "language": row.get("language", ""),
                "domain": row.get("domain", ""),
                "file_name": row.get("file_name", audio.name),
            }
            for col in extra_cols:
                if col in fieldnames:
                    out_row[col] = row.get(col, "")
            writer.writerow(out_row)
    return out_csv
