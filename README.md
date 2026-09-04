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
  --model google/gemini-2.5-flash Sunbird/asr-whisper-51-african-languages
```

Requires `OPENROUTER_API_KEY` and `HF_TOKEN` in `.env`.

## Gemini 3.1 Flash Lite (FLEURS + WaxalNLP + Ndizi)

Evaluate `google/gemini-3.1-flash-lite` on Swahili, Amharic, Oromo, and Hausa:

| Source | Languages |
|--------|-----------|
| `google/fleurs` | sw / am / om / ha |
| `google/WaxalNLP` | am (`amh_asr`) / om (`orm_asr`) only — no hau/swa ASR configs |
| `smutuvi/ndizi-1` + `ndizi-1-2025` | sw |

```bash
# Smoke (2 samples per source)
./scripts/run_gemini31.sh --max-samples 2

# Full test splits
./scripts/run_gemini31.sh

# Or Pro instead of Flash Lite
./scripts/run_gemini31.sh --model google/gemini-3.1-pro-preview
```

Equivalent CLI:

```bash
python -m gemini_asr_bench.cli \
  --model google/gemini-3.1-flash-lite \
  --dataset smutuvi/ndizi-1 smutuvi/ndizi-1-2025 google/fleurs google/WaxalNLP \
  --experiment-name gemini31_flash_lite_afro
```

## Flash model compare (Afro, cap 100/source)

Compare Gemini 2.5 / 3.1 Flash Lite / 3.5 / 3.6 on FLEURS (sw/am/om/ha), WaxalNLP (am/om), and Ndizi:

```bash
# Smoke (default cap 100/source)
./scripts/run_flash_compare.sh

# Smaller smoke
MAX_SAMPLES=10 ./scripts/run_flash_compare.sh
```

Results land in `results/flash_compare_n100/` (or `flash_compare_n10/` when overridden).

## Arabic eval (4 models, cap 400/source)

Compare the same four Flash models on Arabic read and broadcast speech:

| Source | Notes |
|--------|-------|
| `google/fleurs` `ar_eg` test | Egyptian read speech |
| `MohamedRashad/common-voice-18-arabic` test | Multi-dialect read |
| `Esra11/mgb2-arabic` train | Broadcast; relative WER only |
| `MightyStudent/Egyptian-ASR-MGB-3` train | Egyptian YouTube/broadcast |

```bash
# Smoke (10 samples per source)
./scripts/run_arabic_eval.sh --max-samples 10

# Medium smoke
./scripts/run_arabic_eval.sh --max-samples 100

# Full run (400 samples per source × 4 models ≈ 6,400 API calls)
./scripts/run_arabic_eval.sh
```

Requires `OPENROUTER_API_KEY` in `.env`. Results: `results/arabic_eval_n400/`.

## Horn-ASR (Amharic + Oromo)

LesanAI [Horn-ASR](https://huggingface.co/datasets/LesanAI/Horn-ASR): spontaneous interview speech, 1,000 test clips per language (`am` / `om`). Audio is stored as WAV + CSV (`file_name` / `transcript`).

```bash
# Quick smoke (20/lang)
./scripts/run_horn_asr.sh --max-samples 20

# Default in config (100/lang)
./scripts/run_horn_asr.sh

# Full test sets (1000/lang)
./scripts/run_horn_asr.sh --max-samples 1000 --experiment-name horn_asr_gemini25_n1000
```

Results: `results/horn_asr_gemini25_n100/` (or the overridden experiment name).

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
