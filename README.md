# ndizi_mlops_gemini

Benchmark ASR models on Hugging Face speech datasets:

- **OpenRouter** multimodal chat models (Gemini Flash family)
- **Local Whisper** checkpoints (e.g. [`Sunbird/asr-whisper-51-african-languages`](https://huggingface.co/Sunbird/asr-whisper-51-african-languages))

Default FLEURS languages:

| Language | FLEURS config | Sunbird code |
|----------|---------------|--------------|
| Swahili  | `sw_ke`       | `swa`        |
| Amharic  | `am_et`       | `amh`        |
| Oromo    | `om_et`       | `orm`        |
| Hausa    | `ha_ng`       | `hau`        |

Default OpenRouter models:

- `google/gemini-2.5-flash`
- `google/gemini-3.5-flash`
- `google/gemini-3.6-flash`

## Setup

```bash
# In your conda/venv env (e.g. conda activate ndizi):
pip install -e .

# For local Sunbird / Whisper backends:
pip install -e ".[local]"

cp .env.example .env
# OPENROUTER_API_KEY=...   # only needed for Gemini/OpenRouter models
# HF_TOKEN=...             # needed to accept/download gated Sunbird weights
```

If you skip `pip install -e .`, the `scripts/*.sh` helpers set `PYTHONPATH=src` so `python -m gemini_asr_bench.cli` still resolves — but you still need the Python deps from `requirements.txt` / `pip install -e .`.

Accept the Sunbird model terms on Hugging Face, then set `HF_TOKEN` in `.env`.

## Quick start (FLEURS + Gemini)

Smoke test (2 samples / language, Gemini 2.5 / 3.5 / 3.6 Flash):

```bash
chmod +x scripts/*.sh
./scripts/run_smoke.sh
```

Full FLEURS run:

```bash
./scripts/run_fleurs.sh
```

## Sunbird Whisper-51

Smoke test (2 samples / language):

```bash
./scripts/run_sunbird_smoke.sh
```

Full FLEURS test set:

```bash
./scripts/run_sunbird.sh
```

## Ndizi (`smutuvi/ndizi-1` + `smutuvi/ndizi-1-2025`)

Compare Gemini 2.5 Flash and Sunbird on both Ndizi test sets:

```bash
# Smoke (2 samples per dataset)
./scripts/run_ndizi_smoke.sh

# Full test splits
./scripts/run_ndizi.sh
```

Or:

```bash
python -m gemini_asr_bench.cli \
  --config configs/ndizi_gemini_sunbird.yaml \
  --model google/gemini-2.5-flash \
  --model Sunbird/asr-whisper-51-african-languages
```

Requires `OPENROUTER_API_KEY` and `HF_TOKEN` in `.env`.

## Gemma 4 E2B (Google + Sunbird)

Local multimodal Gemma ASR (no OpenRouter). Accept Gemma / Sunbird model terms and set `HF_TOKEN`.

```bash
pip install -e ".[local]"

# Smoke on Ndizi
python -m gemini_asr_bench.cli \
  --model google/gemma-4-E2B-it Sunbird/Sunflower-Gemma4-E2B \
  --dataset smutuvi/ndizi-1 smutuvi/ndizi-1-2025 \
  --max-samples 2 \
  --experiment-name gemma4_e2b_ndizi_smoke

# Or use aliases + FLEURS
python -m gemini_asr_bench.cli \
  --model google/gemma-4-e2b sunbird/gemma-4-e2b \
  --dataset smutuvi/ndizi-1 google/fleurs \
  --max-samples 5
```

Helper script:

```bash
./scripts/run_gemma4_ndizi.sh --max-samples 2
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

### Column auto-detection

If columns are omitted, the loader looks for:

- **Audio:** `audio`, `audio_path`, `path`, `file`, `wav`, `speech`
- **Text:** `transcription`, `raw_transcription`, `sentence`, `text`, `transcript`, `normalized_text`, `label`

## Outputs

Results land under `results/<experiment_name>/`:

```
results/fleurs_sunbird_whisper51/
  summary.json
  summary.csv
  Sunbird_asr-whisper-51-african-languages/
    predictions.jsonl
    predictions.csv
    metrics.json
```

Metrics include raw and normalized **WER** / **CER**, per language and overall. Runs resume from existing `predictions.jsonl` by default (`--no-resume` to disable).

## Configuration

See `configs/fleurs_afro.yaml` and `configs/fleurs_sunbird.yaml`.

| Key | Meaning |
|-----|---------|
| `models` | OpenRouter slugs and/or local HF Whisper ids (`Sunbird/...`) |
| `dataset.type` | `fleurs` or `huggingface` |
| `dataset.max_samples` | cap per language/config (`null` = all) |
| `concurrency` | parallel OpenRouter workers |
| `local_whisper.device` | `cuda` / `mps` / `cpu` / `null` (auto) |
| `local_whisper.torch_dtype` | `auto` / `float16` / `bfloat16` / `float32` |

## How transcription works

- **Gemini / OpenRouter:** chat completions with multimodal `input_audio` (base64 WAV)
- **Sunbird / local Whisper:** `transformers.WhisperForConditionalGeneration` with Sunbird forced language decoder ids (`swa`, `amh`, `orm`, `hau`, …)

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

## License

MIT
