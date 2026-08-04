from __future__ import annotations

import csv
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from .client import OpenRouterASRClient, audio_array_to_wav_base64
from .config import RunConfig
from .datasets import AudioSample, build_loader
from .metrics import normalize_text, summarize_predictions

logger = logging.getLogger(__name__)


def _slug(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class BenchmarkRunner:
    def __init__(self, config: RunConfig, api_key: str) -> None:
        self.config = config
        self.client = OpenRouterASRClient(api_key=api_key, config=config.openrouter)
        self.output_root = Path(config.output_dir) / config.experiment_name
        self.output_root.mkdir(parents=True, exist_ok=True)

    def run(self, models: Optional[list[str]] = None) -> dict[str, Any]:
        models = models or self.config.models
        if not models:
            raise ValueError("No models specified. Pass --model or set models in config.")

        loader = build_loader(self.config.dataset)
        samples = list(loader.iter_samples())
        if not samples:
            raise RuntimeError("No evaluation samples loaded from dataset.")

        logger.info("Loaded %d samples", len(samples))
        summary: dict[str, Any] = {
            "experiment_name": self.config.experiment_name,
            "created_at": _utc_now(),
            "dataset": asdict(self.config.dataset),
            "models": {},
        }

        for model in models:
            model_summary = self._run_model(model, samples)
            summary["models"][model] = model_summary

        summary_path = self.output_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        self._write_summary_csv(summary)
        logger.info("Wrote summary to %s", summary_path)
        return summary

    def _run_model(self, model: str, samples: list[AudioSample]) -> dict[str, Any]:
        model_dir = self.output_root / _slug(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = model_dir / "predictions.jsonl"
        metrics_path = model_dir / "metrics.json"

        done_ids = set()
        rows: list[dict[str, Any]] = []
        if self.config.resume and predictions_path.exists():
            with predictions_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    rows.append(row)
                    done_ids.add(str(row.get("sample_id")))
            logger.info("Resuming %s with %d existing predictions", model, len(done_ids))

        pending = [s for s in samples if s.sample_id not in done_ids]
        logger.info("Model %s: %d pending / %d total", model, len(pending), len(samples))

        buffer: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal buffer
            if not buffer:
                return
            with predictions_path.open("a", encoding="utf-8") as f:
                for row in buffer:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            buffer = []

        def process_one(sample: AudioSample) -> dict[str, Any]:
            audio_b64, fmt = audio_array_to_wav_base64(sample.audio_array, sample.sampling_rate)
            result = self.client.transcribe(
                model=model,
                audio_b64=audio_b64,
                audio_format=fmt,
                prompt=self.config.prompt,
                language=sample.language,
            )
            pred = result.text
            return {
                "sample_id": sample.sample_id,
                "language": sample.language,
                "language_config": sample.language_config,
                "reference": sample.reference,
                "prediction": pred,
                "reference_normalized": normalize_text(sample.reference, self.config.normalization),
                "prediction_normalized": normalize_text(pred, self.config.normalization),
                "latency_seconds": result.latency_seconds,
                "model": model,
                "error": result.error,
            }

        with ThreadPoolExecutor(max_workers=self.config.concurrency) as pool:
            futures = {pool.submit(process_one, s): s for s in pending}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=model):
                row = fut.result()
                rows.append(row)
                buffer.append(row)
                if len(buffer) >= self.config.save_every:
                    flush()
            flush()

        # Per-language metrics
        by_lang: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_lang.setdefault(str(row.get("language") or "unknown"), []).append(row)

        per_language = {
            lang: summarize_predictions(lang_rows, self.config.normalization)
            for lang, lang_rows in sorted(by_lang.items())
        }
        overall = summarize_predictions(rows, self.config.normalization)
        metrics = {
            "model": model,
            "overall": overall,
            "per_language": per_language,
            "predictions_path": str(predictions_path),
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        self._write_predictions_csv(model_dir / "predictions.csv", rows)
        return metrics

    @staticmethod
    def _write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        fields = [
            "sample_id",
            "language",
            "language_config",
            "reference",
            "prediction",
            "reference_normalized",
            "prediction_normalized",
            "latency_seconds",
            "model",
            "error",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _write_summary_csv(self, summary: dict[str, Any]) -> None:
        path = self.output_root / "summary.csv"
        fields = [
            "model",
            "language",
            "num_scored",
            "num_errors",
            "wer",
            "cer",
            "wer_normalized",
            "cer_normalized",
            "mean_latency_seconds",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for model, metrics in summary.get("models", {}).items():
                overall = metrics.get("overall") or {}
                writer.writerow(
                    {
                        "model": model,
                        "language": "ALL",
                        **{k: overall.get(k) for k in fields if k not in {"model", "language"}},
                    }
                )
                for lang, lang_metrics in (metrics.get("per_language") or {}).items():
                    writer.writerow(
                        {
                            "model": model,
                            "language": lang,
                            **{k: lang_metrics.get(k) for k in fields if k not in {"model", "language"}},
                        }
                    )
