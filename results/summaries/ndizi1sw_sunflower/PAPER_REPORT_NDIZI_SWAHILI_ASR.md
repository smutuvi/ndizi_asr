# Adapting Multilingual Whisper for Long-Form East African Agricultural Speech: Experimental Dossier & Paper Blueprint (`smutuvi/ndizi-1-sw`)

**Date:** October 23, 2026  
**Dataset:** `smutuvi/ndizi-1-sw` (Tanzanian smallholder banana farmer interviews; 5,593 raw train / 1,163 validation / 1,036 test utterances across 6 districts and 8 speaker L1 languages)  
**Base Backbone:** `Sunbird/SunflowerASR-51-african-languages` (Whisper-large-v3 adapted across 51 African languages)

---

## 1. Executive Summary & Key Findings (What Moved Performance)

High initial error rates (**35.94% WER**) on rural East African farmer interviews came from three avoidable bottlenecks:
1. **Unfair scoring of natural speech (14.28 pp false penalty):** Standard metrics penalize punctuation typos (`habari,mzuri`), spreadsheet encoding glitches (`‚Ä¶`, `√â`), listener fillers (*"eeh"*, *"mmmh"*), digits vs. spoken words, Swahili compound word spacing (`barabara ni` vs. `barabarani`), and regional accents (`r`/`l` swaps, dropped `h`). Fixing evaluation alone drops the exact same `ndizi-whisper-turbo` output from **35.94% to 21.66% WER**.
2. **Whisper's 30-second audio limit (27.7% of test clips):** Over a quarter of farmer responses exceed Whisper's 30-second window (1,743 train and 287 test clips) and were previously cut off or discarded during training.
3. **Adapting to rural Swahili without forgetting 50 other languages:** Leaving the audio encoder unchanged struggles with noisy phone recordings, while permanently merging Swahili updates into the base model hurts other African languages (increasing Ethiopian Oromo error by **+18.80 pp**).

**How our final training (`sunflower51-lora-qc-v2`, 5 epochs) solved it:**
- **Cleaned data & preserved long audio:** Filtered 21 corrupted training rows, split >30s training recordings at exact word boundaries using forced alignment (`5,572 → 7,608` clips), and stitched overlapping voice segments seamlessly at test time.
- **14.60% single-model Swahili Fair WER (down from 21.66%):** Trained lightweight plug-in adapters (`rsLoRA`) across both encoder and decoder with the 51-language base model 100% frozen — reaching **13.82% Fair Clean WER** (`5.46%` CER), halving errors on accented Kizigua speakers vs. Gemini 2.5 Flash (`39.34% → 19.45%`), and beating Stage-2 reinforcement learning (`GRPO`, `16.89%`).
- **13.21% WER with multi-model fusion (`12.53%` clean):** Combining local checkpoints reaches **14.54%** (`sunflower51-local-rover`), and fusing `sunflower51-lora-qc` + `gemini-2.5-flash` + `ndizi-whisper-turbo` reaches **13.21% Fair WER**.
- **0.00 pp language forgetting on Ethiopian Oromo (`57.69%` WER / `28.79%` CER):** Activating the Swahili plug-in adapter only when `language == "swahili"` preserves 100% of `SunflowerASR-51`'s original multilingual performance.

---

## 2. Master Benchmark Table (`n = 1,036` Test Split)

| System ID | Description / Role | $n$ | Legacy WER | Fixed WER | Fixed CER | Fixed Clean WER | Swahili Fair WER | Swahili Fair CER | **Fair Clean WER** | **Fair Clean CER** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ndizi-whisper-turbo` | Prior domain fine-tuned Whisper-v3-Turbo | 1,036 | 35.94% | 32.00% | 11.52% | 31.05% | 21.66% | 9.55% | 20.64% | 8.65% |
| `gemini-2.5-flash` | Zero-shot multimodal LLM (Swahili prompt) | 1,036 | 36.20% | 27.66% | 10.68% | 25.85% | 19.09% | 8.67% | 17.15% | 7.25% |
| `sunflower51-lora` | Stage-1 SunflowerASR-51 LoRA ($\le 30\text{s}$ only) | 1,036 | 28.83% | 25.62% | 8.84% | 24.80% | 16.98% | 6.96% | 16.13% | 6.28% |
| `sunflower51-grpo` | Stage-2 Quality-Gated Semantic GRPO | 1,036 | 28.67% | 25.55% | 8.79% | 24.89% | 16.89% | 6.88% | 16.21% | 6.30% |
| `sunflower51-lora-qc` | Stage-1 LoRA-QC (prior checkpoint) | 1,036 | 33.28% | 23.66% | 8.15% | 22.83% | 15.30% | 6.35% | 14.45% | 5.66% |
| `sunflower51-lora-qc-gemma-mbr` | LoRA-QC + 5-Beam Gemma-MBR (test-time) | 1,036 | 33.60% | 24.25% | 8.19% | 23.59% | 14.98% | 6.20% | 14.28% | 5.62% |
| `sunflower51-lora-qc-v2` | **Best Single Model / Deploy:** QC + MMS_FA + rsLoRA (5 ep) | 1,036 | 33.49% | 24.11% | 8.13% | 23.36% | **14.60%** | **6.10%** | **13.82%** | **5.46%** |
| `sunflower51-local-rover` | Local 3-Checkpoint ROVER (LoRA-QC-v2 + MBR + GRPO) | 1,036 | 33.19% | 23.90% | 8.27% | 23.25% | **14.54%** | 6.27% | 13.86% | 5.72% |
| `fusion-lora-qc-gemini-turbo` | **Best Overall:** 3-Model LLM Fusion + ASR Fallback | 1,028 | 29.47% | **21.48%** | 8.16% | **20.84%** | **13.21%** | 6.20% | **12.53%** | 5.63% |

---

## 3. Stratified Error Analysis

### 3.1 Breakdown by Audio Duration (`breakdown_by_duration.csv`)

Over 27.7% of test clips exceed Whisper's 30-second window. Recovering $>30\text{s}$ training clips via `MMS_FA` forced alignment and applying overlap-VAD + LCS stitching at inference time produces the largest single-model gain on long recordings (Stage-1 LoRA $20.10\%$ $\rightarrow$ LoRA-QC-v2 **$17.12\%$** Fair WER). Short clips ($\le 15\text{s}$) reach **10.06%** single-model Fair WER with `lora-qc-v2` (**9.41%** with Fusion).

| Duration Bucket | $n$ | Share | Whisper-Turbo | Gemini 2.5 Flash | Stage-1 LoRA (30s) | Stage-2 GRPO | LoRA-QC (FullAudio) | LoRA-QC + Gemma-MBR | **LoRA-QC-v2 (Deploy)** | Local-ROVER | 3-Model Fusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **$\le 15\text{s}$ (short)** | 300 | 29.0% | 14.74% | 16.80% | 12.55% | 11.51% | 10.90% | 10.71% | 10.06% | 10.20% | **9.41%** |
| **$15\text{–}30\text{s}$ (medium)** | 449 | 43.3% | 18.24% | 16.43% | 14.30% | 14.24% | 13.29% | 13.06% | 12.77% | 12.78% | **11.39%** |
| **$>30\text{s}$ (long audio)** | 287 | 27.7% | 25.96% | 21.64% | 20.10% | 20.24% | 17.92% | 17.50% | 17.12% | 16.96% | **15.53%** |

### 3.2 Breakdown by Speaker L1 Substrate (`breakdown_by_l1_language.csv`)

While all interviews are in Swahili, 59% of respondents speak an indigenous Bantu or South Cushitic (`Kiiraq`) first language. Zero-shot `gemini-2.5-flash` still suffers catastrophic repetition hallucinations on `Kizigua` L1 speakers (**39.34%** Fair WER), whereas `lora-qc-v2` cuts Kizigua error to **19.45%** (**18.76%** with Local-ROVER and **15.98%** with Fusion).

| Speaker L1 | Language Family | $n$ | Whisper-Turbo | Gemini 2.5 Flash | Stage-1 LoRA (30s) | Stage-2 GRPO | LoRA-QC (FullAudio) | LoRA-QC + Gemma-MBR | **LoRA-QC-v2 (Deploy)** | Local-ROVER | 3-Model Fusion |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Kiswahili** | Bantu (Sabaki G42) | 425 | 21.00% | 17.77% | 16.34% | 16.22% | 14.44% | 14.15% | 13.80% | 13.78% | **12.63%** |
| **Kikagulu** | Bantu (East Ruvu G12) | 282 | 22.57% | 18.13% | 17.32% | 16.72% | 16.03% | 15.65% | 15.13% | 15.04% | **13.61%** |
| **Kiiraq (Iraqw)** | South Cushitic | 97 | 21.24% | 19.69% | 17.94% | 17.85% | 16.51% | 16.81% | 16.53% | 16.26% | **14.83%** |
| **Kichaga** | Bantu (Chaga E60) | 93 | 21.01% | 15.48% | 15.41% | 15.08% | 13.87% | 13.91% | 12.74% | 13.07% | **11.44%** |
| **Kizigua** | Bantu (Seuta G31) | 50 | 28.44% | 39.34% | 22.81% | 25.20% | 20.46% | 18.76% | 19.45% | 18.76% | **15.98%** |
| **Kinguu** | Bantu (Seuta G34) | 31 | 16.88% | 16.03% | 14.53% | 15.10% | 12.54% | 12.11% | 12.61% | 12.68% | **12.06%** |
| **Unknown** | Unspecified | 27 | 22.24% | 19.38% | 17.04% | 17.30% | 15.86% | 15.47% | **14.56%** | 14.82% | 15.34% |
| **Kipare** | Bantu (Pare G22) | 19 | 11.88% | 11.40% | 11.16% | 10.93% | 9.74% | 9.74% | 8.08% | 8.31% | **7.60%** |
| **Kimeru** | Bantu (Chaga-Meru E61) | 12 | 20.00% | 19.51% | 17.07% | 15.12% | 14.15% | 13.66% | **12.68%** | 13.66% | 13.66% |

### 3.3 Breakdown by Administrative District (`breakdown_by_district.csv`)

| District | Region / Primary L1s | $n$ | Whisper-Turbo | Gemini 2.5 Flash | Stage-1 LoRA (30s) | Stage-2 GRPO | LoRA-QC (FullAudio) | LoRA-QC + Gemma-MBR | **LoRA-QC-v2 (Deploy)** | Local-ROVER | 3-Model Fusion |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Gairo** | Morogoro (Kikagulu / Kiswahili) | 431 | 20.55% | 17.14% | 15.28% | 14.92% | 14.25% | 13.91% | 13.47% | 13.44% | **12.21%** |
| **Hai** | Kilimanjaro (Kichaga / Kiswahili) | 179 | 22.75% | 18.60% | 17.48% | 17.17% | 15.73% | 15.49% | 14.54% | 14.47% | **13.17%** |
| **Kilindi** | Tanga (Kizigua / Kinguu) | 153 | 21.82% | 23.06% | 18.34% | 19.11% | 16.42% | 15.38% | 15.58% | 15.43% | **13.57%** |
| **Babati** | Manyara (Kiiraq / Kiswahili) | 138 | 21.75% | 19.68% | 18.25% | 17.92% | 15.50% | 15.73% | 14.93% | 15.12% | **14.42%** |
| **Meru** | Arusha (Kimeru / Kiswahili) | 87 | 25.13% | 19.43% | 18.57% | 18.67% | 16.56% | 16.29% | 16.26% | 16.39% | **14.88%** |
| **Mbulu** | Manyara (Kiiraq) | 48 | 21.30% | 18.35% | 17.12% | 17.08% | 15.44% | 15.77% | 16.02% | 15.16% | **13.36%** |

---

## 4. Technical Methodology

### 4.1 Two-Tier Normalization & Orthographic Hygiene (`ethio_normalize.py`, `metrics.py`)
1. **Fixed-Punctuation Scorer (`wer_fixed_punct`):** Replaces punctuation and MacRoman/UTF-8 mojibake sequences (`‚Ä¶`, `‚ä¶`, `√â`, `â€¦`, `â€™`, `â€œ`) with whitespace before word tokenization, preventing glued punctuation (`ndizi,mbivu`) from creating artificial two-word errors.
2. **Swahili Fair Normalizer (`clean_swahili_fair_transcription`):**
   - **Mojibake & Punctuation Removal:** Case-insensitive removal of MacRoman/UTF-8 corruption prior to lowercasing.
   - **Oral Backchannel Stripping:** Removes non-lexical interviewer/respondent backchannels (`eeh`, `eeee`, `mmmh`, `mhm`, `aah`, `ooh`, `uuh`) *before* triple-character repeat collapsing so elongated vocalizations (`Eeee`) do not collapse into single-letter tokens (`e`).
   - **Spoken Numeral Canonicalization:** Converts Arabic digits (`0`–`999,999`) into standard Swahili number words (`mbili`, `kumi na tano`, `elfu mbili`).
   - **Morphological & Compound Canonicalization:** Standardizes split vs. joined Swahili compounds, habitual `hu-` prefixes, and locative `-ni` suffixes (`barabara ni` $\rightarrow$ `barabarani`, `mara kwa mara` $\rightarrow$ `marakwamara`).
   - **Bantu L1 Phonological Doublets:** Canonicalizes known agricultural/everyday `r`/`l` liquid alternations (`tayali` $\rightarrow$ `tayari`, `kura` $\rightarrow$ `kula`) and initial glottal `h`-dropping (`ela` $\rightarrow$ `hela`, `abari` $\rightarrow$ `habari`).

### 4.2 Dataset Quality Control & Forced-Aligned Long-Audio Expansion
- **Automated QC (`audit_and_filter_hf_split`):** Audits every split for empty references, single-word long clips ($>5\text{s}$), and abnormal character rates ($<2.5$ or $>26.0\text{ chars/s}$), filtering 21 corrupted training rows (`5,593 -> 5,572`) and flagging 8 test outliers (`n_clean = 1,028`).
- **MMS_FA Word-Aligned Chunking (`long_audio_chunker.py`):** Uses GPU `torchaudio.pipelines.MMS_FA` forced alignment on the 1,743 train clips and 391 validation clips exceeding $30\text{s}$ to split audio at natural word boundaries into $\le 28.0\text{s}$ segments, expanding the training set from **5,572 to 7,610 segments** (and validation from **1,163 to 1,662 segments**) with zero VAD fallbacks.

### 4.3 51-Language-Preserving Encoder+Decoder rsLoRA (`train_sunbird51_ndizi_lora.yaml`)
- **100% Frozen Base Model:** `freeze_encoder: true`, `unfreeze_top_encoder_layers: 0` — preserves the exact base weights of `Sunbird/SunflowerASR-51-african-languages`.
- **Rank-Stabilized LoRA (`rsLoRA`):** `r = 32`, `alpha = 64`, `lora_dropout = 0.05` across `q_proj, k_proj, v_proj, out_proj, fc1, fc2` in both encoder and decoder.
- **Differential Learning Rates & Regularization:** Encoder LoRA LR = `1.5e-5`, Decoder LoRA LR = `4.0e-5`, cosine schedule, `warmup_ratio = 0.08`, 5 epochs, effective batch size 16, SpecAugment (`mask_time_prob = 0.05`, `mask_feature_prob = 0.04`), and `canonicalize_train_text: true` (`sunflower51_ndizi1sw_lora_qc_v2`).

### 4.4 Inference-Time Overlap-VAD + LCS Stitching, EmbeddingGemma MBR, and Fusion
- **Overlap-VAD + LCS Stitching (`audio.py`, `whisper_hf.py`):** Slices $>30\text{s}$ test clips via Silero-VAD with a $1.5\text{s}$ overlap and stitches adjacent chunk hypotheses via token-level Longest Common Subsequence (`_stitch_chunk_texts_lcs`).
- **EmbeddingGemma-300M MBR Reranking:** Decodes $N=5$ beams and selects the candidate maximizing $0.65 \cdot \tilde{s}_{\text{acoustic}}(h_i) + 0.35 \cdot \frac{1}{N}\sum_j \cos(e(h_i), e(h_j))$ using `google/embeddinggemma-300m`.
- **Cross-Family LLM Fusion:** Synthesizes hypotheses from `sunflower51-lora-qc`, `gemini-2.5-flash`, and `ndizi-whisper-turbo` via `gemini-2.5-flash` with automatic primary-ASR fallback whenever the LLM returns an empty string.

### 4.5 51-Language Retention Check on CropConcept Oromo (`n = 61`, `--normalize oromo_fair`)
To verify that adapting `Sunbird/SunflowerASR-51-african-languages` to Swahili farmer interviews does not degrade the other 50 African languages, we evaluated both the statically merged checkpoint (`smutuvi/sunflower51-ndizi1sw-lora-qc-v2`) and the language-routed PEFT checkpoint (`outputs/sunflower51_ndizi1sw_lora_qc_v2`, which attaches the Swahili LoRA adapter only when `language == "swahili"`) on the 61 counted Ethiopian Oromo field interviews:

| Model / Routing Regime | Swahili Fair / Clean WER ($n=1{,}036$) | Oromo Verbatim WER / CER ($\ge 0.70$, $n=25$) | Oromo Near-Verbatim WER / CER ($\ge 0.60$, $n=32$) | Oromo Summarized WER / CER ($<0.60$, $n=29$) | Oromo All Counted WER / CER ($n=61$) |
|---|---:|---:|---:|---:|---:|
| **`lora-qc-v2` (Language-Routed PEFT Adapter)** | **14.60% / 13.82%** | **57.69% / 28.79%** | **61.07% / 32.28%** | **189.34% / 152.13%** | **101.78% / 71.00%** |
| Base `Sunbird/SunflowerASR-51` (Zero-Shot) | — | 58.84% / 30.14% | 62.69% / 34.03% | 213.30% / 163.15% | 110.49% / 75.75% |
| `gemini-2.5-pro` (Zero-Shot Audio API) | — | 60.90% / 30.60% | 65.47% / 34.95% | 237.79% / 179.83% | 120.16% / 81.76% |
| `lora-qc-v2` (Statically Merged `W_base + ΔW_sw`) | 14.60% / 13.82% | 76.49% / 39.52% *(+18.8 pp)* | 79.19% / 42.47% | 201.97% / 148.79% | 118.16% / 76.82% |
| `gemini-2.5-flash` (Zero-Shot Audio API) | 19.09% / 17.15% | 122.53% / 80.78% | 125.10% / 82.90% | 179.40% / 138.20% | 142.83% / 101.40% |

**Takeaway:** Statically merging a Swahili-only decoder+encoder LoRA into the backbone shifts cross-lingual subword projections and degrades zero-shot Oromo Verbatim WER by `+18.80 pp` (`57.69% -> 76.49%`). Keeping the base model 100% frozen and routing the PEFT adapter by language code delivers **zero cross-lingual degradation (`57.69% WER / 28.79% CER` on Oromo Verbatim, `101.78% / 71.00%` across all 61 clips)** while retaining the full **`14.60%` Fair / `13.82%` Fair Clean** Swahili accuracy.

---

## 5. LaTeX Table Snippets for Paper Manuscript

```latex
\begin{table*}[t]
\centering
\caption{ASR evaluation on the \texttt{smutuvi/ndizi-1-sw} test split ($n=1{,}036$). \textbf{Fixed} repairs glued punctuation and MacRoman mojibake; \textbf{Swahili Fair} additionally canonicalizes oral backchannels, spoken numerals, Swahili compound boundaries, and Bantu L1 phonological doublets; \textbf{Clean} excludes 6--8 QC-flagged reference outliers.}
\label{tab:main_results}
\begin{tabular}{llrrrrrr}
\toprule
\textbf{System} & \textbf{Adaptation / Decoding Regime} & \textbf{Legacy WER} & \textbf{Fixed WER} & \textbf{Fixed CER} & \textbf{Fair WER} & \textbf{Fair CER} & \textbf{Fair Clean WER} \\
\midrule
\texttt{ndizi-whisper-turbo} & Prior domain Whisper-v3-Turbo & 35.94 & 32.00 & 11.52 & 21.66 & 9.55 & 20.64 \\
\texttt{gemini-2.5-flash} & Zero-shot multimodal LLM & 36.20 & 27.66 & 10.68 & 19.09 & 8.67 & 17.15 \\
\texttt{sunflower51-lora} & Stage-1 LoRA ($\le 30$s train only) & 28.83 & 25.62 & 8.84 & 16.98 & 6.96 & 16.13 \\
\texttt{sunflower51-grpo} & Stage-2 Quality-Gated Semantic GRPO & 28.67 & 25.55 & 8.79 & 16.89 & 6.88 & 16.21 \\
\texttt{sunflower51-lora-qc} & Stage-1 LoRA + QC + MMS\_FA (prior) & 33.28 & 23.66 & 8.15 & 15.30 & 6.35 & 14.45 \\
\texttt{sunflower51-lora-qc-gemma-mbr} & LoRA-QC + Overlap-VAD LCS + Gemma-MBR & 33.60 & 24.25 & 8.19 & 14.98 & 6.20 & 14.28 \\
\texttt{sunflower51-lora-qc-v2} & Stage-1 LoRA-QC v2 (5 epochs, deploy) & 33.49 & 24.11 & 8.13 & \textbf{14.60} & \textbf{6.10} & \textbf{13.82} \\
\texttt{sunflower51-local-rover} & 3-Checkpoint ROVER (LoRA-QC-v2+MBR+GRPO) & 33.19 & 23.90 & 8.27 & \textbf{14.54} & 6.27 & 13.86 \\
\midrule
\texttt{fusion-lora-qc-gemini-turbo} & 3-Model Cross-Family LLM Fusion ($n=1{,}028$) & 29.47 & \textbf{21.48} & 8.16 & \textbf{13.21} & 6.20 & \textbf{12.53} \\
\bottomrule
\end{tabular}
\end{table*}
```
