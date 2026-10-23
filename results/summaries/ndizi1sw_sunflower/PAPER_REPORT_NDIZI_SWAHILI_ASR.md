# Adapting Multilingual Whisper for Long-Form East African Agricultural Speech: Experimental Dossier & Paper Blueprint (`smutuvi/ndizi-1-sw`)

**Date:** October 23, 2026  
**Dataset:** `smutuvi/ndizi-1-sw` (Tanzanian smallholder banana farmer interviews; 5,593 raw train / 1,163 validation / 1,036 test utterances across 6 districts and 8 speaker L1 languages)  
**Base Backbone:** `Sunbird/SunflowerASR-51-african-languages` (Whisper-large-v3 adapted across 51 African languages)

---

## 1. Executive Summary & Headline Results

Spontaneous agricultural survey speech recorded in rural East Africa presents three compounding challenges that cause off-the-shelf and naively fine-tuned ASR models to appear to fail (>35% naive WER):
1. **Evaluation & Orthographic Noise (~8–12 pp artificial penalty):** Field transcripts contain glued punctuation (`habari,mzuri`), spreadsheet MacRoman/UTF-8 mojibake (`‚Ä¶`, `√â`), inconsistent interviewer/farmer backchannels (`eeh`, `mmmh`), mixed Arabic digits vs. spoken Swahili numerals, Swahili compound/locative boundary drift (`barabara ni` vs. `barabarani`), and Bantu L1 phonological variation (`r`/`l` substitution and initial `h`-dropping).
2. **The 30-Second Long-Audio Bottleneck (27.7% of corpus):** Over a quarter of farmer responses exceed Whisper's 30-second positional window (1,743 train clips and 287 test clips). Naive training drops clips >30s, starving the model of 31% of training data, while naive inference suffers boundary truncation.
3. **Multilingual Preservation vs. Acoustic Domain Shift:** Adapting solely the decoder leaves the acoustic encoder mismatched to noisy rural mobile recordings and non-native Swahili accents, whereas unfreezing the encoder destroys the 51-language representation of `SunflowerASR-51`.

By systematically addressing each bottleneck, our experimentation achieved:
- **35.94% $\rightarrow$ 23.86% (-12.08 pp)** on the baseline `ndizi-whisper-turbo` predictions purely by fixing glued-punctuation/mojibake tokenization (`32.00%` Fixed WER) and applying linguistically fair Swahili normalization (`23.86%` Fair WER).
- **23.86% $\rightarrow$ 16.90% (-6.96 pp)** single-model Swahili Fair WER (`16.05%` Fair Clean WER) via **Stage-1 LoRA-QC (`sunflower51-lora-qc`)**: automated dataset QC + `torchaudio` `MMS_FA` forced-aligned long-audio training segmentation (5,572 $\rightarrow$ 7,610 segments) + 100%-frozen-base encoder+decoder rsLoRA + SpecAugment.
- **16.90% $\rightarrow$ 16.59% (-0.31 pp)** single-model Swahili Fair WER (`15.91%` Fair Clean WER, `5.91%` CER) by adding inference-time **1.5s overlap-VAD + Longest Common Subsequence (LCS) boundary stitching** and **5-beam EmbeddingGemma-300M Minimum Bayes Risk (MBR) reranking**.
- **16.59% $\rightarrow$ 14.88% (-1.71 pp)** Swahili Fair WER (**`14.08%` Fair Clean WER**, `6.09%` CER) via **3-Model Cross-Family Fusion** combining `sunflower51-lora-qc`, `gemini-2.5-flash`, and `ndizi-whisper-turbo` with automatic ASR fallback on empty LLM responses.

---

## 2. Master Benchmark Table (`n = 1,036` Test Split)

| System ID | Description / Role | $n$ | Legacy WER | Fixed WER | Fixed CER | Fixed Clean WER | Swahili Fair WER | Swahili Fair CER | **Fair Clean WER** | **Fair Clean CER** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ndizi-whisper-turbo` | Prior domain fine-tuned Whisper-v3-Turbo | 1,036 | 35.94% | 32.00% | 11.52% | 31.05% | 23.86% | 9.86% | 22.87% | 8.97% |
| `gemini-2.5-flash` | Zero-shot multimodal LLM (Swahili prompt) | 1,036 | 36.20% | 27.66% | 10.68% | 25.85% | 20.53% | 8.96% | 18.61% | 7.54% |
| `sunflower51-lora` | Stage-1 SunflowerASR-51 LoRA ($\le 30\text{s}$ only) | 1,036 | 28.83% | 25.62% | 8.84% | 24.80% | 18.64% | 7.25% | 17.80% | 6.58% |
| `sunflower51-grpo` | Stage-2 Quality-Gated Semantic GRPO | 1,036 | 28.67% | 25.55% | 8.79% | 24.89% | 18.53% | 7.18% | 17.86% | 6.60% |
| `sunflower51-lora-qc` | **Deploy Baseline:** QC + MMS_FA Full-Audio rsLoRA | 1,036 | 33.28% | 23.66% | 8.15% | 22.83% | **16.90%** | 6.64% | **16.05%** | 5.95% |
| `sunflower51-local-rover` | Local 3-Checkpoint ROVER (LoRA-QC + LoRA + GRPO) | 1,036 | 32.99% | **23.37%** | **8.08%** | **22.54%** | 16.76% | 6.60% | **15.91%** | **5.91%** |
| `sunflower51-lora-qc-gemma-mbr` | **Best Single Model:** LoRA-QC + 5-Beam Gemma-MBR | 1,036 | 33.60% | 24.25% | 8.19% | 23.59% | **16.59%** | **6.49%** | **15.91%** | **5.91%** |
| `fusion-lora-qc-gemini-turbo` | **Best Overall:** 3-Model LLM Fusion + ASR Fallback | 1,028 | 29.54% | **21.69%** | 8.53% | **20.92%** | **14.88%** | 6.83% | **14.08%** | 6.09% |

---

## 3. Stratified Error Analysis

### 3.1 Breakdown by Audio Duration (`breakdown_by_duration.csv`)

Over 27.7% of test clips exceed Whisper's 30-second window. Recovering $>30\text{s}$ training clips via `MMS_FA` forced alignment and applying overlap-VAD + LCS stitching at inference time produces the largest single-model gain on long recordings (-2.61 pp from `21.83%` to `19.22%` Fair WER). Short clips ($\le 15\text{s}$) reach **12.33%** single-model Fair WER (**10.96%** with Fusion).

| Duration Bucket | $n$ | Share | Whisper-Turbo | Gemini 2.5 Flash | Stage-1 LoRA (30s) | Stage-2 GRPO | LoRA-QC (FullAudio) | LoRA-QC + Gemma-MBR | Local-ROVER | 3-Model Fusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **$\le 15\text{s}$ (short)** | 300 | 29.0% | 17.94% | 18.31% | 14.02% | 13.09% | 12.47% | **12.33%** | 12.42% | **10.96%** |
| **$15\text{–}30\text{s}$ (medium)** | 449 | 43.3% | 20.33% | 17.78% | 15.93% | 15.77% | 14.78% | **14.53%** | 14.78% | **12.83%** |
| **$>30\text{s}$ (long audio)** | 287 | 27.7% | 27.99% | 23.12% | 21.83% | 21.98% | 19.60% | **19.22%** | 19.35% | **17.39%** |

### 3.2 Breakdown by Speaker L1 Substrate (`breakdown_by_l1_language.csv`)

While all interviews are in Swahili, 59% of respondents speak an indigenous Bantu or South Cushitic (`Kiiraq`) first language. Zero-shot `gemini-2.5-flash` suffers catastrophic repetition hallucinations on `Kizigua` L1 speakers (**40.89%** Fair WER), whereas `LoRA-QC + Gemma-MBR` halves Kizigua error to **20.35%** (**17.67%** with Fusion).

| Speaker L1 | Language Family | $n$ | Whisper-Turbo | Gemini 2.5 Flash | Stage-1 LoRA (30s) | Stage-2 GRPO | LoRA-QC (FullAudio) | LoRA-QC + Gemma-MBR | Local-ROVER | 3-Model Fusion |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Kiswahili** | Bantu (Sabaki G42) | 425 | 22.95% | 18.86% | 17.76% | 17.66% | 15.83% | **15.53%** | 15.61% | **13.96%** |
| **Kikagulu** | Bantu (East Ruvu G12) | 282 | 25.07% | 20.21% | 19.65% | 18.96% | 18.13% | **17.77%** | 18.07% | **16.51%** |
| **Kiiraq (Iraqw)** | South Cushitic | 97 | 22.83% | 20.60% | 19.29% | 19.07% | 17.76% | 18.12% | **17.74%** | **14.99%** |
| **Kichaga** | Bantu (Chaga E60) | 93 | 24.14% | 17.98% | 17.47% | 17.22% | 16.06% | 16.17% | **15.98%** | **13.58%** |
| **Kizigua** | Bantu (Seuta G31) | 50 | 31.02% | 40.89% | 24.28% | 26.65% | 22.11% | **20.35%** | 22.07% | **17.67%** |
| **Kinguu** | Bantu (Seuta G34) | 31 | 18.45% | 16.82% | 15.19% | 15.83% | 13.48% | **12.99%** | 13.13% | **12.90%** |
| **Unknown** | Unspecified | 27 | 25.45% | 20.54% | 17.96% | 18.60% | **17.05%** | **17.05%** | **17.05%** | **16.54%** |
| **Kipare** | Bantu (Pare G22) | 19 | 13.44% | 12.03% | 11.79% | 11.08% | **9.67%** | 9.91% | **9.67%** | **8.25%** |
| **Kimeru** | Bantu (Chaga-Meru E61) | 12 | 26.83% | 20.98% | 19.51% | 17.56% | 16.10% | **15.61%** | 16.10% | **15.61%** |

### 3.3 Breakdown by Administrative District (`breakdown_by_district.csv`)

| District | Region / Primary L1s | $n$ | Whisper-Turbo | Gemini 2.5 Flash | Stage-1 LoRA (30s) | Stage-2 GRPO | LoRA-QC (FullAudio) | LoRA-QC + Gemma-MBR | Local-ROVER | 3-Model Fusion |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Gairo** | Morogoro (Kikagulu / Kiswahili) | 431 | 22.90% | 18.80% | 17.24% | 16.85% | 16.12% | **15.78%** | 15.99% | **14.48%** |
| **Hai** | Kilimanjaro (Kichaga / Kiswahili) | 179 | 25.24% | 20.49% | 19.07% | 18.74% | 17.40% | 17.28% | **17.18%** | **14.80%** |
| **Kilindi** | Tanga (Kizigua / Kinguu) | 153 | 23.94% | 24.39% | 19.94% | 20.66% | 17.92% | **16.84%** | 17.85% | **15.55%** |
| **Babati** | Manyara (Kiiraq / Kiswahili) | 138 | 23.87% | 20.89% | 19.97% | 19.66% | 17.10% | 17.29% | **16.93%** | **15.63%** |
| **Meru** | Arusha (Kimeru / Kiswahili) | 87 | 27.41% | 20.29% | 19.47% | 19.50% | 17.53% | 17.33% | **17.30%** | **15.74%** |
| **Mbulu** | Manyara (Kiiraq) | 48 | 22.24% | 18.86% | 17.88% | 17.92% | **16.17%** | 16.62% | **16.17%** | **12.38%** |

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
\texttt{ndizi-whisper-turbo} & Prior domain Whisper-v3-Turbo & 35.94 & 32.00 & 11.52 & 23.86 & 9.86 & 22.87 \\
\texttt{gemini-2.5-flash} & Zero-shot multimodal LLM & 36.20 & 27.66 & 10.68 & 20.53 & 8.96 & 18.61 \\
\texttt{sunflower51-lora} & Stage-1 LoRA ($\le 30$s train only) & 28.83 & 25.62 & 8.84 & 18.64 & 7.25 & 17.80 \\
\texttt{sunflower51-grpo} & Stage-2 Quality-Gated Semantic GRPO & 28.67 & 25.55 & 8.79 & 18.53 & 7.18 & 17.86 \\
\texttt{sunflower51-lora-qc} & Stage-1 LoRA + QC + MMS\_FA Full-Audio & 33.28 & 23.66 & 8.15 & 16.90 & 6.64 & 16.05 \\
\texttt{sunflower51-local-rover} & 3-Checkpoint ROVER (LoRA-QC+LoRA+GRPO) & 32.99 & \textbf{23.37} & \textbf{8.08} & 16.76 & 6.60 & \textbf{15.91} \\
\texttt{sunflower51-lora-qc-gemma-mbr} & LoRA-QC + Overlap-VAD LCS + Gemma-MBR & 33.60 & 24.25 & 8.19 & \textbf{16.59} & \textbf{6.49} & \textbf{15.91} \\
\midrule
\texttt{fusion-lora-qc-gemini-turbo} & 3-Model Cross-Family LLM Fusion ($n=1{,}028$) & 29.54 & \textbf{21.69} & 8.53 & \textbf{14.88} & 6.83 & \textbf{14.08} \\
\bottomrule
\end{tabular}
\end{table*}
```
