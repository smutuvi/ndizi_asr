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

from .audio import join_chunk_transcripts, maybe_chunk
from .backends.ctc_hf import HuggingFaceCTCClient
from .backends.gemma_hf import HuggingFaceGemmaASRClient
from .backends.omnilingual import OmnilingualASRClient
from .backends.resolve import expand_model_groups, parse_model_spec
from .backends.types import TranscriptionResult
from .backends.whisper_hf import HuggingFaceWhisperClient
from .client import OpenRouterASRClient, audio_array_to_wav_base64
from .config import RunConfig
from .datasets import AudioSample, build_loader
from .datasets.csv_manifest import normalize_lang
from .metrics import normalize_text, summarize_predictions

logger = logging.getLogger(__name__)

_LOCAL_BACKENDS = frozenset({"hf_whisper", "hf_gemma", "hf_ctc", "omnilingual", "our_transcript"})

STRATA_DEFINITIONS: tuple[tuple[str, str], ...] = (
    ("strict_verbatim", "Strict Verbatim (ratio >= 0.85)"),
    ("verbatim", "Verbatim (ratio >= 0.70)"),
    ("near_verbatim", "Near-Verbatim (ratio >= 0.60)"),
    ("summarized", "Summarized (ratio < 0.60)"),
    ("all_counted", "All Counted"),
)


def _compute_ref_ratio(reference: str, our_transcript: str) -> Optional[float]:
    ref_w = len((reference or "").split())
    our_w = len((our_transcript or "").split())
    if our_w <= 0:
        return None
    return round(ref_w / our_w, 4)


def _row_matches_stratum(row: dict[str, Any], stratum_key: str) -> bool:
    if stratum_key == "all_counted":
        return True
    ratio = row.get("ref_ratio")
    if ratio is None:
        return False
    r = float(ratio)
    if stratum_key == "strict_verbatim":
        return r >= 0.85
    if stratum_key == "verbatim":
        return r >= 0.70
    if stratum_key == "near_verbatim":
        return r >= 0.60
    if stratum_key == "summarized":
        return r < 0.60
    return False


def _slug(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_").replace(":", "_")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _lang_matches_filter(sample: AudioSample, language_filter: Optional[str]) -> bool:
    if not language_filter:
        return True
    want = normalize_lang(language_filter)
    for candidate in (sample.language_config, sample.language):
        if candidate and normalize_lang(str(candidate)) == want:
            return True
    return False


class BenchmarkRunner:
    def __init__(self, config: RunConfig, api_key: Optional[str] = None) -> None:
        self.config = config
        self.api_key = api_key
        self.openrouter_client: Optional[OpenRouterASRClient] = None
        self._whisper_clients: dict[str, HuggingFaceWhisperClient] = {}
        self._gemma_clients: dict[str, HuggingFaceGemmaASRClient] = {}
        self._ctc_clients: dict[str, HuggingFaceCTCClient] = {}
        self._omni_clients: dict[str, OmnilingualASRClient] = {}
        self.output_root = Path(config.output_dir) / config.experiment_name
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _get_openrouter(self) -> OpenRouterASRClient:
        if self.openrouter_client is None:
            if not self.api_key:
                raise ValueError("OPENROUTER_API_KEY is required for OpenRouter models")
            self.openrouter_client = OpenRouterASRClient(
                api_key=self.api_key,
                config=self.config.openrouter,
            )
        return self.openrouter_client

    def _get_whisper(self, model_id: str) -> HuggingFaceWhisperClient:
        if model_id not in self._whisper_clients:
            lw = self.config.local_whisper
            self._whisper_clients[model_id] = HuggingFaceWhisperClient(
                model_id,
                device=lw.device,
                torch_dtype=lw.torch_dtype,
                num_beams=lw.num_beams,
                gemma_mbr=lw.gemma_mbr,
                rag_prompt=lw.rag_prompt,
                canonicalize_output=lw.canonicalize_output,
            )
        return self._whisper_clients[model_id]

    def _get_gemma(self, model_id: str) -> HuggingFaceGemmaASRClient:
        if model_id not in self._gemma_clients:
            lw = self.config.local_whisper
            self._gemma_clients[model_id] = HuggingFaceGemmaASRClient(
                model_id,
                device=lw.device,
                torch_dtype=lw.torch_dtype,
            )
        return self._gemma_clients[model_id]

    def _get_ctc(self, model_id: str) -> HuggingFaceCTCClient:
        if model_id not in self._ctc_clients:
            lw = self.config.local_whisper
            self._ctc_clients[model_id] = HuggingFaceCTCClient(
                model_id,
                device=lw.device,
            )
        return self._ctc_clients[model_id]

    def _get_omnilingual(self, model_id: str) -> OmnilingualASRClient:
        card = self.config.local_whisper.omni_card or model_id
        if card not in self._omni_clients:
            lw = self.config.local_whisper
            self._omni_clients[card] = OmnilingualASRClient(
                card,
                omnilingual_root=lw.omnilingual_root,
            )
        return self._omni_clients[card]

    def _transcribe_openrouter_chunked(
        self,
        *,
        model_id: str,
        array,
        sampling_rate: int,
        language: Optional[str],
    ) -> TranscriptionResult:
        chunks, _ = maybe_chunk(array, sampling_rate, self.config.openrouter.max_chunk_seconds)
        client = self._get_openrouter()
        parts: list[str] = []
        total_latency = 0.0
        last_error: Optional[str] = None
        for chunk in chunks:
            audio_b64, fmt = audio_array_to_wav_base64(chunk, sampling_rate)
            result = client.transcribe(
                model=model_id,
                audio_b64=audio_b64,
                audio_format=fmt,
                prompt=self.config.prompt,
                language=language,
            )
            total_latency += float(result.latency_seconds or 0.0)
            if result.error:
                last_error = result.error
            else:
                parts.append(result.text)
        return TranscriptionResult(
            text=join_chunk_transcripts(parts),
            latency_seconds=total_latency,
            model=model_id,
            error=last_error if not parts else None,
        )

    def run(self, models: Optional[list[str]] = None) -> dict[str, Any]:
        models = expand_model_groups(models or self.config.models)
        if not models:
            raise ValueError("No models specified. Pass --model or set models in config.")

        if self.config.dataset.streaming and self.config.dataset.type not in {
            "csv",
            "manifest",
            "cropconcept",
        }:
            logger.info("Streaming dataset samples (no bulk preload)")
        else:
            logger.info("Iterating dataset samples on demand")

        summary: dict[str, Any] = {
            "experiment_name": self.config.experiment_name,
            "created_at": _utc_now(),
            "dataset": asdict(self.config.dataset),
            "models": {},
        }

        for model in models:
            model_summary = self._run_model(model)
            summary["models"][model] = model_summary

        summary_path = self.output_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        self._write_summary_csv(summary)
        self._write_stratified_summary(summary)
        logger.info("Wrote summary to %s", summary_path)
        return summary

    def _iter_pending_samples(
        self,
        done_ids: set[str],
        language_filter: Optional[str],
        *,
        skip_waveform: bool = False,
    ):
        loader = build_loader(self.config.dataset)
        try:
            it = loader.iter_samples(skip_ids=done_ids, skip_waveform=skip_waveform)
        except TypeError:
            it = loader.iter_samples()
        for sample in it:
            if not _lang_matches_filter(sample, language_filter):
                continue
            if sample.sample_id not in done_ids:
                yield sample

    def _run_model(self, model: str) -> dict[str, Any]:
        spec = parse_model_spec(model)
        model_dir = self.output_root / _slug(spec.raw)
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

        skip_waveform = (
            spec.backend == "our_transcript"
            or (spec.backend == "openrouter" and self.config.openrouter.cropconcept_recipe)
        )
        pending = self._iter_pending_samples(
            done_ids, spec.language_filter, skip_waveform=skip_waveform
        )
        logger.info(
            "Model %s [%s]: streaming pending samples (skipping %d already done)%s",
            spec.model_id,
            spec.backend,
            len(done_ids),
            f" lang_filter={spec.language_filter}" if spec.language_filter else "",
        )

        if spec.backend == "hf_whisper":
            self._get_whisper(spec.model_id)
        elif spec.backend == "hf_gemma":
            self._get_gemma(spec.model_id)
        elif spec.backend == "hf_ctc":
            self._get_ctc(spec.model_id)
        elif spec.backend == "omnilingual":
            self._get_omnilingual(spec.model_id)

        buffer: list[dict[str, Any]] = []

        def flush() -> None:
            if not buffer:
                return
            with predictions_path.open("a", encoding="utf-8") as f:
                for row in buffer:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            buffer.clear()

        def process_one(sample: AudioSample) -> dict[str, Any]:
            lang_for_norm = sample.language_config or sample.language
            if spec.backend == "hf_whisper":
                result = self._get_whisper(spec.model_id).transcribe_array(
                    sample.audio_array,
                    sample.sampling_rate,
                    language=sample.language,
                    language_config=sample.language_config,
                )
            elif spec.backend == "hf_gemma":
                result = self._get_gemma(spec.model_id).transcribe_array(
                    sample.audio_array,
                    sample.sampling_rate,
                    language=sample.language,
                    language_config=sample.language_config,
                    prompt=self.config.prompt,
                )
            elif spec.backend == "hf_ctc":
                result = self._get_ctc(spec.model_id).transcribe_array(
                    sample.audio_array,
                    sample.sampling_rate,
                    language=sample.language,
                    language_config=sample.language_config,
                )
            elif spec.backend == "omnilingual":
                result = self._get_omnilingual(spec.model_id).transcribe_array(
                    sample.audio_array,
                    sample.sampling_rate,
                    language=sample.language,
                    language_config=sample.language_config,
                )
            elif spec.backend == "our_transcript":
                our_tx = str((sample.meta or {}).get("our_transcript") or "").strip()
                result = TranscriptionResult(
                    text=our_tx,
                    latency_seconds=0.0,
                    model="our_transcript",
                    error=None if our_tx else "Missing our_transcript in sample metadata",
                )
            else:
                audio_path_meta = str((sample.meta or {}).get("audio_path") or "").strip()
                if self.config.openrouter.cropconcept_recipe and audio_path_meta and Path(audio_path_meta).is_file():
                    result = self._get_openrouter().transcribe_cropconcept_recipe(
                        model=spec.model_id,
                        audio_path=audio_path_meta,
                    )
                else:
                    result = self._transcribe_openrouter_chunked(
                        model_id=spec.model_id,
                        array=sample.audio_array,
                        sampling_rate=sample.sampling_rate,
                        language=sample.language,
                    )
            pred = result.text
            our_tx_meta = str((sample.meta or {}).get("our_transcript") or "").strip()
            ref_ratio = _compute_ref_ratio(sample.reference, our_tx_meta) if our_tx_meta else None
            return {
                "sample_id": sample.sample_id,
                "dataset_id": sample.dataset_id,
                "language": sample.language,
                "language_config": sample.language_config,
                "reference": sample.reference,
                "prediction": pred,
                "reference_normalized": normalize_text(
                    sample.reference, self.config.normalization, language=lang_for_norm
                ),
                "prediction_normalized": normalize_text(
                    pred, self.config.normalization, language=lang_for_norm
                ),
                "latency_seconds": result.latency_seconds,
                "model": spec.raw,
                "backend": spec.backend,
                "error": result.error,
                "our_transcript": our_tx_meta,
                "ref_ratio": ref_ratio,
            }

        workers = (
            self.config.local_whisper.concurrency
            if spec.backend in _LOCAL_BACKENDS
            else self.config.concurrency
        )
        self._process_pending_samples(
            pending,
            process_one,
            workers=workers,
            desc=spec.raw,
            rows=rows,
            buffer=buffer,
            flush=flush,
        )
        flush()

        if not rows:
            raise RuntimeError("No evaluation samples loaded from dataset.")

        by_lang: dict[str, list[dict[str, Any]]] = {}
        by_dataset: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_lang.setdefault(str(row.get("language") or "unknown"), []).append(row)
            by_dataset.setdefault(str(row.get("dataset_id") or "unknown"), []).append(row)

        per_language = {
            lang: summarize_predictions(lang_rows, self.config.normalization)
            for lang, lang_rows in sorted(by_lang.items())
        }
        per_dataset = {
            ds_id: summarize_predictions(ds_rows, self.config.normalization)
            for ds_id, ds_rows in sorted(by_dataset.items())
        }
        overall = summarize_predictions(rows, self.config.normalization)
        per_stratum: dict[str, Any] = {}
        if any(row.get("ref_ratio") is not None for row in rows):
            for stratum_key, stratum_label in STRATA_DEFINITIONS:
                s_rows = [r for r in rows if _row_matches_stratum(r, stratum_key)]
                if s_rows:
                    sm = summarize_predictions(s_rows, self.config.normalization)
                    sm["label"] = stratum_label
                    per_stratum[stratum_key] = sm
        metrics = {
            "model": spec.raw,
            "backend": spec.backend,
            "model_id": spec.model_id,
            "overall": overall,
            "per_language": per_language,
            "per_dataset": per_dataset,
            "per_stratum": per_stratum,
            "predictions_path": str(predictions_path),
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        self._write_predictions_csv(model_dir / "predictions.csv", rows)
        return metrics

    def _process_pending_samples(
        self,
        pending,
        process_one,
        *,
        workers: int,
        desc: str,
        rows: list[dict[str, Any]],
        buffer: list[dict[str, Any]],
        flush,
    ) -> None:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            sample_iter = iter(pending)
            in_flight: set = set()
            pbar = tqdm(desc=desc)

            def submit_next() -> None:
                try:
                    sample = next(sample_iter)
                except StopIteration:
                    return
                in_flight.add(pool.submit(process_one, sample))

            for _ in range(workers):
                submit_next()

            while in_flight:
                for fut in as_completed(in_flight):
                    in_flight.remove(fut)
                    row = fut.result()
                    rows.append(row)
                    buffer.append(row)
                    pbar.update(1)
                    if len(buffer) >= self.config.save_every:
                        flush()
                    submit_next()
                    break

            pbar.close()

    @staticmethod
    def _write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        fields = [
            "sample_id",
            "dataset_id",
            "language",
            "language_config",
            "reference",
            "prediction",
            "reference_normalized",
            "prediction_normalized",
            "latency_seconds",
            "model",
            "backend",
            "error",
            "our_transcript",
            "ref_ratio",
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
            "dataset_id",
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
                        "dataset_id": "ALL",
                        "language": "ALL",
                        **{k: overall.get(k) for k in fields if k not in {"model", "dataset_id", "language"}},
                    }
                )
                for ds_id, ds_metrics in (metrics.get("per_dataset") or {}).items():
                    writer.writerow(
                        {
                            "model": model,
                            "dataset_id": ds_id,
                            "language": "ALL",
                            **{
                                k: ds_metrics.get(k)
                                for k in fields
                                if k not in {"model", "dataset_id", "language"}
                            },
                        }
                    )
                for lang, lang_metrics in (metrics.get("per_language") or {}).items():
                    writer.writerow(
                        {
                            "model": model,
                            "dataset_id": "ALL",
                            "language": lang,
                            **{
                                k: lang_metrics.get(k)
                                for k in fields
                                if k not in {"model", "dataset_id", "language"}
                            },
                        }
                    )
                for s_key, s_metrics in (metrics.get("per_stratum") or {}).items():
                    writer.writerow(
                        {
                            "model": model,
                            "dataset_id": f"STRATUM:{s_key}",
                            "language": "ALL",
                            **{
                                k: s_metrics.get(k)
                                for k in fields
                                if k not in {"model", "dataset_id", "language"}
                            },
                        }
                    )

    def _write_stratified_summary(self, summary: dict[str, Any]) -> None:
        models_dict = summary.get("models", {})
        if not any(m.get("per_stratum") for m in models_dict.values()):
            return
        path = self.output_root / "stratified_summary.csv"
        display_strata = ("verbatim", "near_verbatim", "summarized", "all_counted")
        fields = [
            "model",
            "verbatim_n",
            "verbatim_wer_pct",
            "verbatim_cer_pct",
            "near_verbatim_n",
            "near_verbatim_wer_pct",
            "near_verbatim_cer_pct",
            "summarized_n",
            "summarized_wer_pct",
            "summarized_cer_pct",
            "all_counted_n",
            "all_counted_wer_pct",
            "all_counted_cer_pct",
        ]
        table_rows: list[dict[str, Any]] = []
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for model, metrics in models_dict.items():
                ps = metrics.get("per_stratum") or {}
                row: dict[str, Any] = {"model": model}
                for sk in display_strata:
                    sm = ps.get(sk) or {}
                    wer = sm.get("wer_normalized", sm.get("wer"))
                    cer = sm.get("cer_normalized", sm.get("cer"))
                    row[f"{sk}_n"] = sm.get("num_scored", 0)
                    row[f"{sk}_wer_pct"] = f"{wer * 100:.2f}" if wer is not None else ""
                    row[f"{sk}_cer_pct"] = f"{cer * 100:.2f}" if cer is not None else ""
                writer.writerow(row)
                table_rows.append(row)

        print(f"\n=== Stratified Evaluation Summary (norm={self.config.normalization}) ===")
        print(
            "| Model | Verbatim (>=0.70) WER / CER | Near-Verbatim (>=0.60) WER / CER "
            "| Summarized (<0.60) WER / CER | All Counted WER / CER |"
        )
        print("| :--- | :---: | :---: | :---: | :---: |")
        for r in table_rows:
            def _cell(prefix: str) -> str:
                w = r.get(f"{prefix}_wer_pct", "")
                c = r.get(f"{prefix}_cer_pct", "")
                n = r.get(f"{prefix}_n", 0)
                if not w:
                    return "—"
                return f"{w}% / {c}% (n={n})"

            print(
                f"| `{r['model']}` | {_cell('verbatim')} | {_cell('near_verbatim')} "
                f"| {_cell('summarized')} | {_cell('all_counted')} |"
            )
        print()
