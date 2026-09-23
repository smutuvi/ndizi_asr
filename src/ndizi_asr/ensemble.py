"""Smart fusion: combine top ASR hypotheses with an LLM editor (language knowledge).

Pipeline per utterance:
  1) Run N ASR backends (default: best two Ethio CTC models)
  2) Ask an OpenRouter LLM to reconcile hypotheses
     - optional --few-shot N (gold examples; score held-out only)
     - optional --audio (multimodal: audio + hyps)
  3) Score each ASR system + the fused output (WER/CER) on the eval split
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from .backends.ctc_hf import HuggingFaceCTCClient
from .backends.omnilingual import OmnilingualASRClient
from .backends.resolve import expand_model_groups, parse_model_spec
from .backends.whisper_hf import HuggingFaceWhisperClient
from .client import OpenRouterASRClient, audio_array_to_wav_base64
from .config import NormalizationConfig, OpenRouterConfig, load_run_config
from .datasets import AudioSample, build_loader
from .metrics import summarize_predictions

logger = logging.getLogger(__name__)

DEFAULT_ASR_MODELS = (
    "ethio-asr-multilingual-1b",
    "ethio-asr-oromo",
    "sunbird-whisper-51",
)

EDITOR_ALIASES = {
    "opus": "anthropic/claude-opus-4.5",
    "opus-4.5": "anthropic/claude-opus-4.5",
    "opus4.5": "anthropic/claude-opus-4.5",
    "claude-opus": "anthropic/claude-opus-4.5",
    "claude-opus-4.5": "anthropic/claude-opus-4.5",
    # Claude Opus 5.5 (OpenRouter)
    "opus-5.5": "anthropic/claude-opus-5.5",
    "opus5.5": "anthropic/claude-opus-5.5",
    "opus-5": "anthropic/claude-opus-5.5",
    "opus5": "anthropic/claude-opus-5.5",
    "claude-opus-5.5": "anthropic/claude-opus-5.5",
    "sonnet": "anthropic/claude-sonnet-4.5",
    "gpt-4.1": "openai/gpt-4.1",
    "gpt4.1": "openai/gpt-4.1",
    # OpenAI GPT-6 family (OpenRouter)
    "gpt-6-luna": "openai/gpt-6-luna",
    "gpt6-luna": "openai/gpt-6-luna",
    "luna": "openai/gpt-6-luna",
    "leo": "openai/gpt-6-luna",  # common misspelling of Luna
    "gpt-6-leo": "openai/gpt-6-luna",
    "gpt-6-sol": "openai/gpt-6-sol",
    "gpt6-sol": "openai/gpt-6-sol",
    "sol": "openai/gpt-6-sol",
    "gpt-6-astra": "openai/gpt-6-astra",
    "gpt6-astra": "openai/gpt-6-astra",
    "astra": "openai/gpt-6-astra",
    "gemini": "google/gemini-2.5-flash",
    "gemini-flash": "google/gemini-2.5-flash",
    "gemini-pro": "google/gemini-2.5-pro",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
}

FUSION_SYSTEM = (
    "You are a conservative ASR fusion editor for Afaan Oromo / Amharic. "
    "Your job is to SELECT and lightly repair words from the given ASR hypotheses — "
    "not to paraphrase, translate, or rewrite into more fluent speech. "
    "Return ONLY the final transcript text — no quotes, labels, or commentary."
)

FUSION_USER_TEMPLATE = """Language: {language}
Domain: Ethiopian CropConcept agronomy / hybrid wheat farmer speech (Oromo or Amharic).
{few_shot_block}
ASR hypotheses for THIS utterance (may contain errors; these are your ONLY allowed vocabulary):
{hyp_block}
{audio_note}
Instructions (follow strictly):
- Stay close to the hypotheses. Prefer copying spans from one hypothesis over inventing new wording.
- Every content word in your output should appear in at least one hypothesis (minor spelling fixes OK: doubled letters, near-homophones already attested).
- Do NOT introduce synonyms, discourse fillers, or "more natural" Oromo/Amharic that is absent from both hypotheses.
- When hypotheses disagree, pick the clearer/more complete reading; do not blend into a third paraphrase.
- Preserve numbers, quantities, and named entities exactly as in the hypotheses (do not change digits).
- Delete only clear ASR garbage / duplicated fragments; do not drop long attested phrases to "clean up" style.
- If both hypotheses are empty, return an empty string.
{glossary_block}
Final transcript:"""


def resolve_editor_model(name: str) -> str:
    raw = name.strip()
    return EDITOR_ALIASES.get(raw, EDITOR_ALIASES.get(raw.lower(), raw))


def build_reasoning_config(
    *,
    thinking: bool,
    effort: str = "high",
    exclude: bool = True,
    editor_model: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """OpenRouter unified reasoning / thinking parameter."""
    mid = (editor_model or "").lower()
    # GPT-6 Luna/Sol often need an explicit effort; default to none for clean transcripts
    # unless the user asked for --thinking.
    if thinking:
        return {"enabled": True, "effort": effort, "exclude": exclude}
    if "gpt-6" in mid or mid.startswith("openai/gpt-6"):
        return {"effort": "none", "exclude": True}
    return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _print_table(rows: list[dict[str, Any]], *, title: str = "Smart fusion comparison") -> None:
    headers = ("System", "WER↓", "CER", "WER_raw", "CER_raw", "n", "err")
    table = [
        (
            str(r["name"]),
            _fmt_metric(r["wer_n"]),
            _fmt_metric(r["cer_n"]),
            _fmt_metric(r["wer"]),
            _fmt_metric(r["cer"]),
            str(r["n"] if r["n"] is not None else "—"),
            str(r["errors"] if r["errors"] is not None else "—"),
        )
        for r in rows
    ]
    widths = [len(h) for h in headers]
    for tr in table:
        for i, cell in enumerate(tr):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: tuple[str, ...]) -> str:
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    print(f"\n=== {title} ===")
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for tr in table:
        print(fmt_row(tr))


class _ASRPool:
    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device
        self._ctc: dict[str, HuggingFaceCTCClient] = {}
        self._whisper: dict[str, HuggingFaceWhisperClient] = {}
        self._omni: dict[str, OmnilingualASRClient] = {}

    def transcribe(self, model_key: str, sample: AudioSample) -> tuple[str, Optional[str]]:
        spec = parse_model_spec(model_key)
        try:
            if spec.backend == "hf_ctc":
                if spec.model_id not in self._ctc:
                    self._ctc[spec.model_id] = HuggingFaceCTCClient(
                        spec.model_id, device=self.device
                    )
                result = self._ctc[spec.model_id].transcribe_array(
                    sample.audio_array,
                    sample.sampling_rate,
                    language=sample.language,
                    language_config=sample.language_config,
                )
            elif spec.backend == "hf_whisper":
                if spec.model_id not in self._whisper:
                    self._whisper[spec.model_id] = HuggingFaceWhisperClient(
                        spec.model_id, device=self.device
                    )
                result = self._whisper[spec.model_id].transcribe_array(
                    sample.audio_array,
                    sample.sampling_rate,
                    language=sample.language,
                    language_config=sample.language_config,
                )
            elif spec.backend == "omnilingual":
                if spec.model_id not in self._omni:
                    self._omni[spec.model_id] = OmnilingualASRClient(spec.model_id)
                result = self._omni[spec.model_id].transcribe_array(
                    sample.audio_array,
                    sample.sampling_rate,
                    language=sample.language,
                    language_config=sample.language_config,
                )
            else:
                return "", f"ensemble ASR backends must be local (got {spec.backend})"
            return result.text or "", result.error
        except Exception as exc:  # noqa: BLE001
            return "", str(exc)


def _build_few_shot_block(examples: list[dict[str, Any]], asr_models: list[str]) -> str:
    if not examples:
        return ""
    parts = [
        "Here are worked examples of correcting ASR hypotheses to the gold transcript:",
        "",
    ]
    for i, ex in enumerate(examples, start=1):
        hyps = ex.get("hypotheses") or {}
        parts.append(f"Example {i}:")
        for m in asr_models:
            parts.append(f"  [{m}]: {hyps.get(m) or '(empty)'}")
        parts.append(f"  Gold: {ex.get('reference') or ''}")
        parts.append("")
    parts.append("Now correct the next utterance in the same way.")
    parts.append("")
    return "\n".join(parts)


def _build_fusion_prompt(
    *,
    language: str,
    hyps: dict[str, str],
    glossary: Optional[str],
    few_shot_block: str = "",
    with_audio: bool = False,
) -> str:
    lines = []
    for i, (name, text) in enumerate(hyps.items(), start=1):
        lines.append(f"{i}. [{name}]: {text or '(empty)'}")
    glossary_block = ""
    if glossary and glossary.strip():
        glossary_block = (
            "\nDomain glossary / preferred spellings (use when consistent with evidence):\n"
            f"{glossary.strip()}\n"
        )
    audio_note = (
        "\nAudio for THIS utterance is attached — use it together with the hypotheses.\n"
        if with_audio
        else ""
    )
    return FUSION_USER_TEMPLATE.format(
        language=language or "unknown",
        few_shot_block=few_shot_block,
        hyp_block="\n".join(lines),
        audio_note=audio_note,
        glossary_block=glossary_block,
    )


def _load_reuse_hyps(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row.get("sample_id"))] = row
    return out


def run_fusion(
    *,
    asr_models: list[str],
    editor_model: str,
    dataset_cfg: Any,
    normalization: NormalizationConfig,
    openrouter: OpenRouterConfig,
    api_key: str,
    output_dir: Path,
    device: Optional[str] = None,
    glossary: Optional[str] = None,
    resume: bool = True,
    few_shot: int = 0,
    seed: int = 42,
    use_audio: bool = False,
    reuse_hyps: Optional[Path] = None,
    thinking: bool = False,
    thinking_effort: str = "high",
) -> dict[str, Any]:
    asr_models = expand_model_groups(asr_models)
    editor_model = resolve_editor_model(editor_model)
    reasoning = build_reasoning_config(
        thinking=thinking, effort=thinking_effort, editor_model=editor_model
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_bits = []
    if few_shot > 0:
        mode_bits.append(f"fs{few_shot}")
    if use_audio:
        mode_bits.append("audio")
    if thinking:
        mode_bits.append(f"think-{thinking_effort}")
    mode_suffix = ("+" + "+".join(mode_bits)) if mode_bits else ""
    fusion_name = f"fusion:{editor_model}{mode_suffix}"
    # Stable name per experiment dir (modes are separated by --experiment-name).
    pred_path = output_dir / "fusion_predictions.jsonl"
    # Migrate older fs8/audio/think filenames if present
    if not pred_path.exists() and mode_bits:
        legacy = output_dir / f"fusion_predictions_{'_'.join(mode_bits)}.jsonl"
        if legacy.exists():
            legacy.rename(pred_path)
            logger.info("Renamed legacy predictions %s → %s", legacy.name, pred_path.name)

    done: dict[str, dict[str, Any]] = {}
    if resume and pred_path.exists():
        with pred_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                done[str(row.get("sample_id"))] = row
        logger.info("Resuming fusion with %d existing rows (%s)", len(done), pred_path.name)
    elif not resume and pred_path.exists():
        pred_path.unlink()
        logger.info("Starting fresh predictions file %s", pred_path.name)
    reused = _load_reuse_hyps(reuse_hyps) if reuse_hyps and reuse_hyps.exists() else {}
    if reused:
        logger.info("Reusing ASR hypotheses from %s (%d rows)", reuse_hyps, len(reused))

    pool = _ASRPool(device=device)
    or_cfg = openrouter
    is_gpt6 = "gpt-6" in editor_model.lower()
    # GPT-6 defaults to reasoning; 1024 often gets burned on hidden tokens → empty content.
    need_more_tokens = (thinking or is_gpt6) and or_cfg.max_tokens < 4096
    if need_more_tokens or (thinking and or_cfg.timeout_seconds < 300.0):
        or_cfg = OpenRouterConfig(
            base_url=openrouter.base_url,
            timeout_seconds=max(
                openrouter.timeout_seconds, 300.0 if thinking else openrouter.timeout_seconds
            ),
            max_retries=openrouter.max_retries,
            retry_backoff_seconds=openrouter.retry_backoff_seconds,
            temperature=openrouter.temperature,
            max_tokens=max(openrouter.max_tokens, 4096 if need_more_tokens else openrouter.max_tokens),
        )
    llm = OpenRouterASRClient(api_key=api_key, config=or_cfg)
    if reasoning:
        logger.info("OpenRouter reasoning config for %s: %s", editor_model, reasoning)

    # Materialize samples (CropConcept counted set is small).
    samples = list(build_loader(dataset_cfg).iter_samples())
    if not samples:
        raise RuntimeError("No evaluation samples loaded")

    # Ensure ASR hyps exist for every sample (reuse or run).
    # Reused rows may omit newer models (e.g. add sunbird on top of an Ethio-only cache).
    records: list[dict[str, Any]] = []
    for sample in tqdm(samples, desc="asr"):
        sid = sample.sample_id
        if sid in done and done[sid].get("hypotheses") is not None:
            base = {
                "sample_id": sid,
                "dataset_id": done[sid].get("dataset_id", sample.dataset_id),
                "language": done[sid].get("language", sample.language),
                "language_config": done[sid].get(
                    "language_config", sample.language_config
                ),
                "reference": done[sid].get("reference", sample.reference),
                "hypotheses": dict(done[sid].get("hypotheses") or {}),
                "asr_errors": dict(done[sid].get("asr_errors") or {}),
            }
        elif sid in reused and reused[sid].get("hypotheses") is not None:
            base = {
                "sample_id": sid,
                "dataset_id": sample.dataset_id,
                "language": sample.language,
                "language_config": sample.language_config,
                "reference": sample.reference,
                "hypotheses": dict(reused[sid].get("hypotheses") or {}),
                "asr_errors": dict(reused[sid].get("asr_errors") or {}),
            }
        else:
            base = {
                "sample_id": sid,
                "dataset_id": sample.dataset_id,
                "language": sample.language,
                "language_config": sample.language_config,
                "reference": sample.reference,
                "hypotheses": {},
                "asr_errors": {},
            }

        hyps = dict(base["hypotheses"])
        errors = dict(base["asr_errors"])
        for m in asr_models:
            if m not in hyps:
                text, err = pool.transcribe(m, sample)
                hyps[m] = text
                errors[m] = err
        base["hypotheses"] = hyps
        base["asr_errors"] = errors
        base["_sample"] = sample  # transient; stripped before write
        records.append(base)

    rng = random.Random(seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    n_shot = max(0, min(int(few_shot), len(records) - 1)) if few_shot > 0 else 0
    if few_shot > 0 and n_shot < 1:
        raise ValueError("--few-shot needs at least 2 samples (1 shot + 1 eval)")
    shot_idx = set(order[:n_shot]) if n_shot else set()
    eval_indices = [i for i in range(len(records)) if i not in shot_idx]

    few_shot_examples = [records[i] for i in order[:n_shot]] if n_shot else []
    few_shot_block = _build_few_shot_block(few_shot_examples, asr_models)

    if n_shot:
        logger.info(
            "Few-shot: %d examples, %d eval utterances (seed=%d)",
            n_shot,
            len(eval_indices),
            seed,
        )
        shot_ids = [records[i]["sample_id"] for i in order[:n_shot]]
        eval_ids = [records[i]["sample_id"] for i in eval_indices]
        (output_dir / "few_shot_ids.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "few_shot": n_shot,
                    "shot_ids": shot_ids,
                    "eval_ids": eval_ids,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        # Persist shot rows (refs + hyps) so glossary can be built without touching eval gold
        shot_path = output_dir / "few_shot_examples.jsonl"
        with shot_path.open("w", encoding="utf-8") as f:
            for i in order[:n_shot]:
                rec = records[i]
                f.write(
                    json.dumps(
                        {
                            "sample_id": rec["sample_id"],
                            "language": rec.get("language"),
                            "reference": rec.get("reference"),
                            "hypotheses": rec.get("hypotheses"),
                            "split": "few_shot",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        logger.info("Wrote few-shot examples → %s", shot_path)

    by_system: dict[str, list[dict[str, Any]]] = {m: [] for m in asr_models}
    by_system[fusion_name] = []
    buffer: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        with pred_path.open("a", encoding="utf-8") as f:
            for row in buffer:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        buffer = []

    for i in tqdm(eval_indices, desc="fusion-eval"):
        rec = records[i]
        sample: AudioSample = rec["_sample"]
        sid = rec["sample_id"]

        # Resume only if same fusion mode already stored
        prior = done.get(sid)
        mode_ok = (
            prior
            and prior.get("fusion_model") == editor_model
            and bool(prior.get("few_shot")) == (n_shot > 0)
            and bool(prior.get("use_audio")) == use_audio
            and bool(prior.get("thinking")) == thinking
            and prior.get("thinking_effort", "high") == thinking_effort
            and prior.get("fusion") is not None
        )
        if mode_ok:
            row = prior
        else:
            prompt = _build_fusion_prompt(
                language=sample.language or sample.language_config,
                hyps=rec["hypotheses"],
                glossary=glossary,
                few_shot_block=few_shot_block,
                with_audio=use_audio,
            )
            if use_audio:
                audio_b64, fmt = audio_array_to_wav_base64(
                    sample.audio_array, sample.sampling_rate
                )
                fused = llm.chat_with_audio(
                    model=editor_model,
                    prompt=prompt,
                    audio_b64=audio_b64,
                    audio_format=fmt,
                    system=FUSION_SYSTEM,
                    reasoning=reasoning,
                )
            else:
                fused = llm.chat_text(
                    model=editor_model,
                    prompt=prompt,
                    system=FUSION_SYSTEM,
                    reasoning=reasoning,
                )
            fusion_text = (fused.text or "").strip()
            # Strip accidental quotes / labels some models prepend
            if fusion_text.startswith("Final transcript:"):
                fusion_text = fusion_text.split(":", 1)[1].strip()
            fusion_text = fusion_text.strip().strip('"').strip("'")
            fusion_error = fused.error
            if not fusion_text and not fusion_error:
                fusion_error = "empty_fusion_output"
                raw = fused.raw_response or {}
                choice0 = (raw.get("choices") or [{}])[0]
                usage = raw.get("usage") or {}
                logger.warning(
                    "Empty fusion for %s (model=%s). finish_reason=%s "
                    "usage=%s message_keys=%s",
                    sid,
                    editor_model,
                    choice0.get("finish_reason"),
                    {
                        "completion_tokens": usage.get("completion_tokens"),
                        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                            "reasoning_tokens"
                        )
                        or usage.get("reasoning_tokens"),
                    },
                    list((choice0.get("message") or {}).keys()),
                )
            row = {
                "sample_id": sid,
                "dataset_id": rec.get("dataset_id"),
                "language": rec.get("language"),
                "language_config": rec.get("language_config"),
                "reference": rec.get("reference"),
                "hypotheses": rec.get("hypotheses"),
                "asr_errors": rec.get("asr_errors"),
                "fusion_model": editor_model,
                "few_shot": n_shot,
                "use_audio": use_audio,
                "thinking": thinking,
                "thinking_effort": thinking_effort if thinking else None,
                "seed": seed,
                "split": "eval",
                "fusion": fusion_text,
                "fusion_error": fusion_error,
                "fusion_latency_seconds": fused.latency_seconds,
            }
            buffer.append(row)
            if len(buffer) >= 5:
                flush()

        ref = str(row.get("reference") or "")
        hyps = row.get("hypotheses") or {}
        for m in asr_models:
            by_system[m].append(
                {
                    "reference": ref,
                    "prediction": str(hyps.get(m) or ""),
                    "error": (row.get("asr_errors") or {}).get(m),
                    "latency_seconds": None,
                }
            )
        by_system[fusion_name].append(
            {
                "reference": ref,
                "prediction": str(row.get("fusion") or ""),
                "error": row.get("fusion_error"),
                "latency_seconds": row.get("fusion_latency_seconds"),
            }
        )

    flush()

    table_rows: list[dict[str, Any]] = []
    metrics_out: dict[str, Any] = {
        "created_at": _utc_now(),
        "editor": editor_model,
        "few_shot": n_shot,
        "use_audio": use_audio,
        "thinking": thinking,
        "thinking_effort": thinking_effort if thinking else None,
        "seed": seed,
        "num_eval": len(eval_indices),
        "systems": {},
    }
    for name, rows in by_system.items():
        summary = summarize_predictions(rows, normalization)
        metrics_out["systems"][name] = summary
        table_rows.append(
            {
                "name": name,
                "wer": summary.get("wer"),
                "cer": summary.get("cer"),
                "wer_n": summary.get("wer_normalized"),
                "cer_n": summary.get("cer_normalized"),
                "n": summary.get("num_scored"),
                "errors": summary.get("num_errors"),
            }
        )

    table_rows.sort(key=lambda r: (r["wer_n"] is None, float(r["wer_n"] or 0.0)))
    title = "Smart fusion comparison"
    if n_shot or use_audio or thinking:
        title += f" (eval n={len(eval_indices)}"
        if n_shot:
            title += f", few-shot={n_shot}"
        if use_audio:
            title += ", audio"
        if thinking:
            title += f", thinking={thinking_effort}"
        title += ")"
    _print_table(table_rows, title=title)

    metrics_path = output_dir / "fusion_metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = output_dir / "fusion_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "system",
                "wer_normalized",
                "cer_normalized",
                "wer",
                "cer",
                "num_scored",
                "num_errors",
            ],
        )
        writer.writeheader()
        for r in table_rows:
            writer.writerow(
                {
                    "system": r["name"],
                    "wer_normalized": r["wer_n"],
                    "cer_normalized": r["cer_n"],
                    "wer": r["wer"],
                    "cer": r["cer"],
                    "num_scored": r["n"],
                    "num_errors": r["errors"],
                }
            )
    logger.info("Wrote %s and %s", metrics_path, csv_path)
    logger.info("Predictions → %s", pred_path)
    return metrics_out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ndizi-asr-fuse",
        description=(
            "Smart fusion: dual ASR + LLM editor. "
            "Flags: --few-shot N, --audio (multimodal), --reuse-hyps PATH."
        ),
    )
    p.add_argument("--config", type=Path, default=None, help="Optional YAML (dataset section used)")
    p.add_argument(
        "--asr-models",
        nargs="+",
        default=list(DEFAULT_ASR_MODELS),
        help=f"Local ASR systems to fuse (default: {' '.join(DEFAULT_ASR_MODELS)})",
    )
    p.add_argument(
        "--editor",
        default="opus",
        help="OpenRouter LLM editor (opus, sonnet, gpt-4.1, gemini, gemini-pro, …)",
    )
    p.add_argument("--csv", dest="csv_path", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--languages", "--language-filter", nargs="+", dest="language_filter", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--normalize", choices=["default", "ethio", "none"], default="ethio")
    p.add_argument("--glossary", type=Path, default=None, help="Optional text glossary for the editor")
    p.add_argument(
        "--few-shot",
        type=int,
        default=0,
        metavar="N",
        help="Use N gold examples in the prompt; score only the held-out utterances",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for few-shot split")
    p.add_argument(
        "--audio",
        action="store_true",
        help="Multimodal fusion: send audio + hypotheses to the editor (prefer --editor gemini)",
    )
    p.add_argument(
        "--reuse-hyps",
        type=Path,
        default=None,
        help="Reuse ASR hypotheses from a prior fusion_predictions.jsonl (skip ASR)",
    )
    p.add_argument(
        "--thinking",
        action="store_true",
        help="Enable OpenRouter reasoning/thinking tokens for the editor (Claude Opus)",
    )
    p.add_argument(
        "--thinking-effort",
        default="high",
        choices=["minimal", "low", "medium", "high", "xhigh", "max"],
        help="Reasoning effort when --thinking is set (default: high)",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--experiment-name", default=None)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true", default=False)
    p.add_argument("--env-file", type=Path, default=Path(".env"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.env_file.exists():
        load_dotenv(args.env_file)
    else:
        load_dotenv()

    raw: dict[str, Any]
    if args.config and args.config.exists():
        raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    else:
        raw = {
            "experiment_name": "llm_fusion",
            "dataset": {"type": "csv", "streaming": False},
            "normalization": {"mode": "ethio"},
        }

    ds = dict(raw.get("dataset") or {})
    if args.csv_path:
        ds["type"] = "csv"
        ds["csv_path"] = args.csv_path
    if args.data_root:
        ds["data_root"] = args.data_root
    if args.language_filter:
        ds["language_filter"] = list(args.language_filter)
    if args.max_samples is not None:
        ds["max_samples"] = args.max_samples
    raw["dataset"] = ds
    raw["normalization"] = {
        **dict(raw.get("normalization") or {}),
        "mode": args.normalize,
    }
    if args.experiment_name:
        raw["experiment_name"] = args.experiment_name
    if args.output_dir:
        raw["output_dir"] = args.output_dir

    cfg = load_run_config(raw)
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        logging.error("OPENROUTER_API_KEY required for the LLM editor")
        return 2
    if not cfg.dataset.csv_path and cfg.dataset.type not in {"cropconcept", "fleurs", "huggingface"}:
        logging.error("Provide --csv or a dataset config")
        return 2

    glossary = None
    if args.glossary and args.glossary.exists():
        glossary = args.glossary.read_text(encoding="utf-8")

    editor = resolve_editor_model(args.editor)
    if args.audio and "anthropic/" in editor:
        logging.warning(
            "Audio fusion with Anthropic editors may be unsupported on OpenRouter; "
            "prefer --editor gemini or gemini-pro if requests fail."
        )

    exp = cfg.experiment_name
    if not exp:
        bits = ["llm_fusion"]
        if args.few_shot:
            bits.append(f"fs{args.few_shot}")
        if args.audio:
            bits.append("audio")
        if args.thinking:
            bits.append(f"think-{args.thinking_effort}")
        exp = "_".join(bits)
        raw["experiment_name"] = exp
        cfg = load_run_config(raw)

    out = Path(cfg.output_dir) / cfg.experiment_name
    resume = False if args.no_resume else args.resume

    print("Smart fusion = dual ASR hypotheses + LLM language editor")
    print(f"  ASR:      {expand_model_groups(args.asr_models)}")
    print(f"  Editor:   {editor}")
    print(f"  Few-shot: {args.few_shot} (seed={args.seed})")
    print(f"  Audio:    {args.audio}")
    print(f"  Thinking: {args.thinking}" + (f" (effort={args.thinking_effort})" if args.thinking else ""))
    print(f"  Out:      {out}")

    run_fusion(
        asr_models=list(args.asr_models),
        editor_model=args.editor,
        dataset_cfg=cfg.dataset,
        normalization=cfg.normalization,
        openrouter=cfg.openrouter,
        api_key=api_key,
        output_dir=out,
        device=args.device,
        glossary=glossary,
        resume=resume,
        few_shot=args.few_shot,
        seed=args.seed,
        use_audio=args.audio,
        reuse_hyps=args.reuse_hyps,
        thinking=args.thinking,
        thinking_effort=args.thinking_effort,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
