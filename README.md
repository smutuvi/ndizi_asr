# ndizi_asr

**One-stop Ndizi ASR train + evaluate** for African speech recognition.

Formerly `gemini_asr_bench` / `ndizi_mlops_gemini`, now also absorbing [ndizi_eval](https://github.com/smutuvi/ndizi_eval) (Ethiopia CropConcept).

## What it covers

| Capability | Backends / data |
|------------|-----------------|
| **Eval — API** | OpenRouter Gemini Flash family |
| **Eval — local** | Whisper / Sunbird, Gemma 4 E2B, **Ethio CTC (w2v-bert)**, **Meta Omnilingual ASR** |
| **Datasets** | FLEURS, WaxalNLP, Horn-ASR, Ndizi Hub sets, Arabic eval, **CSV manifests**, **CropConcept** |
| **Scoring** | WER/CER; `default` or **`ethio`** (Geʼez) normalize |
| **Train** | Scaffold CLI (`ndizi-asr-train`) for Whisper/CTC fine-tunes |

## Setup

```bash
pip install -e .
# Local Whisper / CTC / VAD:
pip install -e ".[local]"
# Optional Omnilingual:
pip install -e ".[omnilingual]"
# Train deps (Trainer loops still scaffolded):
pip install -e ".[train]"

cp .env.example .env
# OPENROUTER_API_KEY=...   # Gemini / OpenRouter
# HF_TOKEN=...             # gated Sunbird / Gemma
# OMNILINGUAL_ASR_ROOT=... # optional local omnilingual checkout
```

CLI entrypoints:

```bash
ndizi-asr --list-models
python -m ndizi_asr.cli --list-models
ndizi-asr-train --config configs/train_whisper_scaffold.yaml --dry-run
```

(`gemini-asr-bench` / `python -m gemini_asr_bench.cli` still work as deprecated aliases.)

## Quick eval

```bash
chmod +x scripts/*.sh
./scripts/run_smoke.sh          # Gemini × FLEURS smoke
./scripts/run_horn_asr.sh       # Amharic + Oromo interviews
./scripts/run_ethio.sh hub      # Ethio Hub (FLEURS + Waxal) n=20
./scripts/run_ethio.sh list
```

Multi-model / group example:

```bash
python -m ndizi_asr.cli \
  --model gemini-2.5-flash ethio sunbird-whisper-51 \
  --dataset google/fleurs:am_et google/fleurs:om_et \
  --normalize ethio \
  --max-samples 10 \
  --experiment-name ethio_smoke
```

CSV manifest (ndizi_eval style):

```bash
python -m ndizi_asr.cli \
  --csv /path/to/manifest.csv \
  --data-root /path/to/audio_root \
  --model ethio-asr-oromo sunbird-whisper-51 \
  --normalize ethio \
  --language-filter om \
  --experiment-name cropconcept_csv
```

CropConcept YAML: `configs/cropconcept_oromo.yaml` (edit absolute paths).

## Training (scaffold)

```bash
ndizi-asr-train --config configs/train_whisper_scaffold.yaml --dry-run
# writes outputs/<experiment>/train_plan.json
```

Full `Seq2SeqTrainer` / CTC Trainer loops are stubs — port existing Ndizi fine-tune notebooks into `ndizi_asr.train.whisper_trainer` / `ctc_trainer`.

## Outputs

```
results/<experiment_name>/
  summary.json
  summary.csv
  <model_slug>/
    predictions.jsonl
    predictions.csv
    metrics.json
```

## Config tips

| Key | Meaning |
|-----|---------|
| `models` | OpenRouter ids, HF ids, short aliases (`ethio-asr-oromo`), or groups (`ethio`, `oromo`) |
| `dataset.type` | `fleurs` \| `huggingface` \| `csv` \| `cropconcept` |
| `normalization.mode` | `default` \| `ethio` \| `none` |
| `local_whisper.omni_card` | Override Omnilingual card (e.g. `omniASR_LLM_1B`) |

Prefixes: `gemini:` / `openrouter:`, `hf:` / `hf_whisper:`, `ctc:`, `omnilingual:` / `omni:`, `gemma:`.

## Migration from ndizi_eval

| ndizi_eval | ndizi_asr |
|------------|-----------|
| `python -m ethiopia_asr_eval run --models oromo` | `--model oromo` |
| `--normalize ethio` | `--normalize ethio` |
| `--csv` / CropConcept builders | `--csv` or `dataset.type: cropconcept` |
| Omnilingual / Ethio CTC | same backends, wired here |
| Hub `--dataset hub:config:split` | `--dataset` / YAML `sources` |

Keep `ndizi_eval` archived as a pointer once you confirm CropConcept WER/CER parity on a known run.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

## License

MIT
