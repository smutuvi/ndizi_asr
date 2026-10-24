---
language:
- sw
- swa
license: apache-2.0
base_model: Sunbird/SunflowerASR-51-african-languages
tags:
- whisper
- asr
- speech-recognition
- swahili
- lora
- agriculture
- africa
- sunbird
datasets:
- smutuvi/ndizi-1-sw
library_name: transformers
pipeline_tag: automatic-speech-recognition
---

# sunflower51-ndizi1sw-lora-qc-v2

Merged **rsLoRA** adaptation of [`Sunbird/SunflowerASR-51-african-languages`](https://huggingface.co/Sunbird/SunflowerASR-51-african-languages) for spontaneous **Swahili** farmer-survey speech (`smutuvi/ndizi-1-sw`).

This is the **best single-model / deploy** checkpoint from the Ndizi Swahili ASR experiments: dataset QC + `MMS_FA` long-audio chunking + 100% frozen base encoder/decoder + encoder+decoder rsLoRA (5 epochs).

| Metric (test, n=1,036) | Value |
|---|---:|
| **Swahili Fair WER** | **14.60%** |
| **Fair Clean WER** | **13.82%** |
| Fair CER | 6.10% |
| Fair Clean CER | 5.46% |
| Fixed WER | 24.11% |
| Fixed CER | 8.13% |
| Legacy WER | 33.49% |

Base weights stay frozen (`freeze_encoder: true`, `unfreeze_top_encoder_layers: 0`); only LoRA adapters were trained, then **merged** for inference.

---

## Quick start

```python
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="smutuvi/sunflower51-ndizi1sw-lora-qc-v2",
    chunk_length_s=28,
    device=0,  # or "cpu"
)
print(pipe("path/or/url/to/audio.wav")["text"])
```

With [`ndizi-asr`](https://github.com/) (recommended for paper-comparable scoring):

```bash
ndizi-asr \
  --model hf:smutuvi/sunflower51-ndizi1sw-lora-qc-v2 \
  --dataset smutuvi/ndizi-1-sw --split test --text-column transcript \
  --normalize swahili_fair \
  --device cuda --experiment-name sunflower51_lora_qc_v2_test
```

CropConcept Oromo regression check (zero-shot vs base SunflowerASR-51):

```bash
ndizi-asr --models sunbird-whisper-51 \
  --csv "$CSV" --languages om --normalize oromo_fair \
  --output-dir results/cropconcept_degrade_base

ndizi-asr --models hf:smutuvi/sunflower51-ndizi1sw-lora-qc-v2 \
  --csv "$CSV" --languages om --normalize oromo_fair \
  --output-dir results/cropconcept_degrade_lora_qc_v2
```

---

## Evaluation parameters explained

Paper and `ndizi-asr` tables report several WERs on the **same** hypotheses. They differ only in **how text is normalized before** `jiwer` and (for Clean) **which test rows are kept**. Always compare systems under the **same** column.

### 1. Legacy WER (`wer_raw` / Legacy)

- **What:** Minimal cleanup; historical baseline that **under-penalized glued punctuation** in references like `vizuri.ndio` or `aina B?nisichopenda`.
- **When to use:** Only for continuity with early runs. **Do not** use as the primary number in new work.
- **CLI:** not the default path in current scoring; produced by `ndizi-asr-qc` / `rescore_predictions_file`.

### 2. Default / Fixed WER (`--normalize default`, Fixed)

- **What:** Standard scorer used as the “honest” mechanical WER:
  - lowercase
  - sentence/clause punctuation and common MacRoman/UTF-8 mojibake (`‚Ä¶`, `√â`, …) → **space** (so words are not glued)
  - remaining punctuation stripped
  - whitespace collapsed
- **What it does *not* do:** No Swahili compound merging, no R/L or H-drop spelling map, no backchannel stripping.
- **When to use:** Cross-system apples-to-apples when you want orthography/spelling differences to still count as errors.
- **CLI:**

```bash
ndizi-asr ... --normalize default
```

In the paper table this column is **Fixed WER** / **Fixed CER** (`wer_fixed_punct_norm`).

### 3. Fair WER (`--normalize swahili_fair`)

- **What:** Linguistically fair Swahili normalization applied **identically** to reference and hypothesis (`clean_swahili_fair_transcription`):
  - NFKC + mojibake cleanup
  - punctuation → space; lowercase
  - strip oral backchannels (`eeh`, `mmmh`, `aah`, …) before collapsing elongations
  - collapse 3+ repeated letters (`labdaaa` → `labda`)
  - canonicalize compounds / locatives (e.g. `barabara ni` → `barabarani`)
  - Bantu L1 doublets: R/L and initial H-drop maps (`tayali`→`tayari`, `abari`→`habari`)
- **When to use:** **Primary headline metric** for Ndizi Swahili farmer interviews — reduces artificial error from annotation style and L1 phonology without changing the model.
- **CLI:**

```bash
ndizi-asr ... --normalize swahili_fair
```

Related modes:

| Mode | Meaning |
|---|---|
| `swahili_fair` | Fair + drop backchannels (**recommended**) |
| `swahili_canon` | Same maps but **keep** canonicalized backchannels (`mmh`/`eeh`/`aah`) |
| `oromo_fair` / `oromo_platform` / `ethio` | Analogous fair/platform/Geʼez normalizers for CropConcept / Ethio evals |

### 4. Clean / Fair Clean (subset filter, not a normalize mode)

- **What:** Same Fixed or Fair scorer, but **excluding** test utterances flagged as reference/audio outliers by QC, e.g.:
  - Fixed WER ≥ 0.90 and CER ≥ 0.45 with extreme hyp/ref length ratio → truncated reference, severe dropouts, or swapped/corrupted audio
  - or Fixed WER ≥ 1.30 with hyp ≫ ref
- **Counts:** Full test `n = 1,036`; Clean typically `n ≈ 1,028–1,029` depending on the prediction file.
- **Columns:**
  - **Fixed Clean WER** — Fixed norm on clean subset
  - **Fair Clean WER** — Fair norm on clean subset (**best “deploy quality” single-model number**: **13.82%** for this checkpoint; **5.46% CER**)
- **When to use:** Report alongside full-test Fair WER when arguing model quality without a handful of broken labels dominating the mean.
- **How:** Re-score an existing `predictions.csv` / utterances file with `ndizi-asr` QC rescoring (`rescore_predictions_file` in `data_qc.py`), or use the published paper summary tables.

### How the columns relate (intuition)

```text
Legacy  →  Fixed (default punct fix)  →  Fair (Swahili orthography)
                ↓                              ↓
         Fixed Clean                      Fair Clean
         (drop QC outliers)               (drop QC outliers)
```

Typical gap on this corpus: Legacy ≫ Fixed ≫ Fair; Clean is a small further drop (~0.5–1 pp) after removing ~7–8 bad test rows.

---

## Training recipe (summary)

| Setting | Value |
|---|---|
| Base | `Sunbird/SunflowerASR-51-african-languages` |
| Train data | `smutuvi/ndizi-1-sw` after QC + MMS_FA chunking (~7,608 train segments) |
| Adapters | rsLoRA `r=32`, `α=64`, dropout `0.05` on `q/k/v/out_proj`, `fc1`, `fc2` (encoder + decoder) |
| Frozen base | 100% encoder+decoder weights frozen |
| LR | encoder LoRA `1.5e-5`, decoder LoRA `4.0e-5`, cosine, warmup_ratio `0.08` |
| Epochs | 5 |
| SpecAugment | `mask_time_prob=0.05`, `mask_feature_prob=0.04` |
| Labels | `canonicalize_train_text: true` (Swahili training target cleanup) |
| Config | `configs/train_sunbird51_ndizi_lora.yaml` (`experiment_name: sunflower51_ndizi1sw_lora_qc_v2`) |

---

## Intended use & limitations

- **In-domain:** Tanzanian smallholder agricultural interview Swahili (mobile field audio, multi-L1 speakers).
- **Zero-shot other languages:** Base is 51 African languages; LoRA was Swahili-only. Prefer a CropConcept / held-out multilingual check before claiming no degradation.
- **Long audio:** Prefer chunked inference (≤~28–30s) with overlap stitching as in `ndizi-asr` Whisper backend.
- **License:** Follow the base model license and your dataset terms; model card marks `apache-2.0` for the adapter merge packaging—confirm base card before commercial redistribution.

---

## Citation

If you use this checkpoint, please cite the Ndizi / SunflowerASR-51 adaptation work and the upstream Sunbird model. Evaluation definitions (Fixed / Fair / Clean) match `ndizi_asr.metrics` and `ndizi_asr.data_qc.rescore_predictions_file`.
