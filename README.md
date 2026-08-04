# ndizi_mlops_gemini

Benchmark **Gemini** multimodal ASR models via [OpenRouter](https://openrouter.ai) on speech datasets hosted on [Hugging Face](https://huggingface.co/datasets).

Default evaluation covers **FLEURS** for:

| Language | FLEURS config |
|----------|---------------|
| Swahili  | `sw_ke`       |
| Amharic  | `am_et`       |
| Oromo    | `om_et`       |
| Hausa    | `ha_ng`       |

Default models:

- `google/gemini-2.5-flash`
- `google/gemini-3.5-flash`
- `google/gemini-3.6-flash`

## Setup

```bash
# In your conda/venv env (e.g. conda activate ndizi):
pip install -e .

cp .env.example .env
# edit .env and set OPENROUTER_API_KEY=...
```

If you skip `pip install -e .`, the `scripts/*.sh` helpers set `PYTHONPATH=src` so `python -m gemini_asr_bench.cli` still resolves — but you still need the Python deps from `requirements.txt` / `pip install -e .`.

Optional: set `HF_TOKEN` in `.env` for gated Hugging Face datasets.

## Quick start (FLEURS)

Smoke test (2 samples / language, Gemini 2.5 / 3.5 / 3.6 Flash):

```bash
chmod +x scripts/*.sh
./scripts/run_smoke.sh
```

Full FLEURS run:

```bash
./scripts/run_fleurs.sh
```

Or:

```bash
python -m gemini_asr_bench.cli --config configs/fleurs_afro.yaml
```

## Any Hugging Face dataset

Use `dataset.type: huggingface` (see `configs/huggingface_example.yaml`) or CLI flags:

```bash
python -m gemini_asr_bench.cli \
  --config configs/huggingface_example.yaml \
  --dataset-type huggingface \
  --dataset-id mozilla-foundation/common_voice_17_0 \
  --config-name sw \
  --split test \
  --audio-column audio \
  --text-column sentence \
  --max-samples 50 \
  --model google/gemini-2.5-flash
```

Multi-config HF datasets (same pattern as FLEURS):

```bash
python -m gemini_asr_bench.cli \
  --dataset-type huggingface \
  --dataset-id google/fleurs \
  --language swahili=sw_ke \
  --language amharic=am_et \
  --split test \
  --text-column transcription \
  --model google/gemini-3.5-flash \
  --max-samples 10
```

### Column auto-detection

If columns are omitted, the loader looks for:

- **Audio:** `audio`, `audio_path`, `path`, `file`, `wav`, `speech`
- **Text:** `transcription`, `raw_transcription`, `sentence`, `text`, `transcript`, `normalized_text`, `label`

## Outputs

Results land under `results/<experiment_name>/`:

```
results/fleurs_afro_gemini/
  summary.json
  summary.csv
  google_gemini-2.5-flash/
    predictions.jsonl
    predictions.csv
    metrics.json
  ...
```

Metrics include raw and normalized **WER** / **CER**, per language and overall. Runs resume from existing `predictions.jsonl` by default (`--no-resume` to disable).

## Configuration

See `configs/fleurs_afro.yaml`. Important knobs:

| Key | Meaning |
|-----|---------|
| `models` | OpenRouter model slugs |
| `dataset.type` | `fleurs` or `huggingface` |
| `dataset.dataset_id` | HF dataset id |
| `dataset.languages` | map of friendly name → HF config |
| `dataset.max_samples` | cap per language/config (`null` = all) |
| `concurrency` | parallel API workers |
| `prompt` | transcription instruction sent with audio |

## How transcription works

Gemini models are called through OpenRouter **chat completions** with multimodal `input_audio` (base64 WAV), matching [OpenRouter audio docs](https://openrouter.ai/docs/guides/overview/multimodal/audio). Audio arrays from Hugging Face are resampled/encoded to 16-bit WAV before upload.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

## License

MIT
