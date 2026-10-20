"""ASR → Oromo–English translation → LLM extraction benchmark (CropConcept Oromo).

Standalone module inside `ndizi_asr` so it runs directly where Sunbird & fusion
results live (`results/cropconcept_oromo_local_sunbird_rerun/...`).

Compares downstream agricultural insight extraction (13-tag taxonomy + numbers)
against the human reference across arms on all 61 counted clips (no few-shot leakage):

  0) sunbird_lexicon_only            : regex/lexicon baseline on Sunbird Oromo
  1) sunbird_direct_argon            : Sunbird Oromo → Argon extraction (no MT)
  2) sunbird_nllb_en_only_argon      : Sunbird Oromo → NLLB-200 (gaz_Latn→eng_Latn) → Argon
  3) sunbird_nllb_bilingual_argon    : Sunbird Oromo → NLLB English + Oromo → Argon
  4) sunbird_glossary_mt_argon       : Sunbird Oromo → questionnaire-glossary MT → Argon

Usage:
  PYTHONPATH=src python -m ndizi_asr.translate_extract \\
    --asr-predictions results/cropconcept_oromo_local_sunbird_rerun/sunbird-whisper-51/predictions.csv \\
    --manifest-csv ../ndizi_eval/data/manifest_cropconcept_oromo.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from .ethio_normalize import clean_oromo_fair_transcription

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ---------------------------------------------------------------------------
# 1. CropConcept Oromo 13-Tag Insight Taxonomy & Number Concepts
# ---------------------------------------------------------------------------

INSIGHT_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "rust_disease_resistance": {
        "question_focus": "audio_q1",
        "label_en": "Rust & Crop Disease Resistance (Waagii dandamachuu)",
        "patterns": [r"\bwaag[ie]+", r"\bwag[ie]+", r"\bkawaag", r"\bsataam", r"\bdhibee\b", r"\bdhukkub"],
    },
    "fungicide_cost_reduction": {
        "question_focus": "audio_q1",
        "label_en": "Reduced Fungicide / Chemical Spraying Cost (Baasii qorichaa hir'isuu)",
        "patterns": [r"\bqorich", r"\bkeemikaal", r"\bbiif", r"\bh?afuuf", r"\bdeddeebi", r"\bbaasii\b", r"\bwacii\b"],
    },
    "spraying_labor_relief": {
        "question_focus": "audio_q1",
        "label_en": "Reduced Knapsack Spraying Labor & Fatigue (Dadhabbii manfaa baachuu)",
        "patterns": [r"\bmanfaa\b", r"\bdadhab", r"\bdafqa", r"\bhumna\b", r"\bbaachuu", r"\bbaanu\b"],
    },
    "higher_yield_productivity": {
        "question_focus": "audio_q1",
        "label_en": "Higher Grain Yield / +25% Productivity (Callaa / oomisha guddaa)",
        "patterns": [
            r"\bcallaa\b",
            r"\boomisha\s+(?:guddaa|fooyya|gaarii|caal|dabal)",
            r"\bsanyiirratti\s+dabalata",
            r"\bmurtii\s+dabal",
            r"\bwuxee\b",
            r"\buttee",
            r"\bbarakaa",
            r"\bdigdamii\s+shan",
            r"\b25\s*%",
            r"\bparsantii",
            r"\bpersentii",
        ],
    },
    "climate_stress_tolerance": {
        "question_focus": "audio_q1",
        "label_en": "Heat, Frost, Cold, or Drought Tolerance (Hoo'isa / qorra dandamachuu)",
        "patterns": [r"\bhoo?'?i[ns]", r"\bqorra\b", r"\bhongee\b", r"\bcaama\b", r"\bqilleens", r"\bgoggog"],
    },
    "market_income_gain": {
        "question_focus": "audio_q1",
        "label_en": "Better Market Price & Household Income (Galii / gatii gabaa)",
        "patterns": [r"\bgalii\b", r"\bgabii\b", r"\bgabaa", r"\bgurgur", r"\bgatii\s+gaarii", r"\bqallaba"],
    },
    "demo_plot_or_small_trial": {
        "question_focus": "audio_q2",
        "label_en": "On-Farm Demo Plot or Small-Parcel Trial First (Yaalii maasaa / qarxii xiqqoo)",
        "patterns": [
            r"\byaalii\b",
            r"\byaalam",
            r"\bijaan\s+arg",
            r"\bmaasaa\s+irratti",
            r"\bmaasarratti",
            r"\blafa\s+(?:xiqqoo|tokko)",
            r"\bqar[tx]ii\b",
            r"\bbooyyee\b",
            r"\btest\b",
            r"\bagarsiis",
            r"\bargisiis",
        ],
    },
    "peer_farmer_experience": {
        "question_focus": "audio_q2",
        "label_en": "Learning from Peer Farmers / Prior Adopters (Muuxannoo warra duraan facaafate / ollaa)",
        "patterns": [
            r"\bmuuxannoo",
            r"\blimdii\b",
            r"\bnimdi",
            r"\bollaa",
            r"\bwarra\s+(?:duraan|facaas|facaaf)",
            r"\bwarroota\s+(?:duraan|facaas)",
            r"\bnamarraa",
        ],
    },
    "germination_flowering_seed_check": {
        "question_focus": "audio_q2",
        "label_en": "Inspecting Germination, Flowering, or Seed Grain Quality (Biqila / daraaraa / firee ilaaluu)",
        "patterns": [r"\bdaraar", r"\bbiqil", r"\bmagars", r"\bmagru", r"\bfir[ie]+", r"\bmadaal"],
    },
    "extension_training_guidance": {
        "question_focus": "audio_q3",
        "label_en": "Extension Agent Training & Agronomic Follow-up (Leenjii / gorsa ogeessa qonnaa / D.A)",
        "patterns": [
            r"\bleenjii\b",
            r"\bleenjis",
            r"\bgorsa\b",
            r"\bgorsan",
            r"\bbarums",
            r"\bbaruum",
            r"\bogeess",
            r"\bogeeyy",
            r"\bwaajjira\s+qonna",
            r"\bgib[i]?rinna",
            r"\bgabrinna",
            r"\bd\.?a\b",
            r"\bhubachiis",
            r"\bqajeelfam",
        ],
    },
    "timely_seed_delivery": {
        "question_focus": "audio_q3",
        "label_en": "Timely Seed Delivery Before Planting Season (Waqtiidhaan / yeroodhaan qophaa'uu)",
        "patterns": [
            r"\bwa[qk]tii",
            r"\bwaaqtii",
            r"\byeroodhaan",
            r"\byeoodhaan",
            r"\byeroo\s+(?:isaa|dhaan|faca)",
            r"\bsa'?aatii",
            r"\bwaggaa\s+waggaa",
            r"\bdafee\s+nuu",
            r"\bdafanii\s+nuu",
        ],
    },
    "fertilizer_herbicide_supply": {
        "question_focus": "audio_q3",
        "label_en": "Bundled Fertilizer & Herbicide Availability (Xaa'oo / madabaraa fi qoricha aramaa)",
        "patterns": [
            r"\bxaa['’‘ʼʻ`]?oo",
            r"\bxaaoo",
            r"\bmadabara",
            r"\bmedebara",
            r"\bmadaal",
            r"\baramaa\b",
            r"\bgurraach",
            r"\badiin",
        ],
    },
    "credit_or_deferred_payment": {
        "question_focus": "audio_q3",
        "label_en": "Credit Access or Post-Harvest Payment Plan (Liqiidhaan / yeroo kaffaltii)",
        "patterns": [
            r"\bliqii",
            r"\bliqeess",
            r"\blíqii",
            r"\bleekii",
            r"\bmallatteess",
            r"\bharkarraa\s+(?:dhab|qab)",
            r"\byeroo\s+qaxaroo",
        ],
    },
    "gov_coop_channel_not_traders": {
        "question_focus": "audio_q3",
        "label_en": "Direct Government / Kebele / Coop Supply Instead of Private Traders (Mootummaa / ganda vs daldaaltota)",
        "patterns": [
            r"\bmootummaa",
            r"\bdaldaal",
            r"\bnagga?ad",
            r"\bganda",
            r"\bgandum",
            r"\bqabalee",
            r"\byuuniyeen",
            r"\binstituut",
        ],
    },
}

ALL_TAG_KEYS: List[str] = list(INSIGHT_TAXONOMY.keys())

NUMBER_CONCEPTS: Dict[str, List[str]] = {
    "num_1_single_unit": [r"\btokko\b", r"\btakk", r"\b1\b"],
    "num_2_units": [r"\blama\b", r"\blamm", r"\b2\b"],
    "num_3_sprays_or_units": [r"\bsadi[i]?\b", r"\b3\b"],
    "num_4_sprays_or_thousands": [r"\bafur\b", r"\bafr", r"\b4\b"],
    "num_5_plus_sprays_or_thousands": [r"\bshan", r"\b5\b"],
    "num_6_to_7": [r"\bja['’‘ʼʻ`]?a\b", r"\btorba", r"\b6\b", r"\b7\b"],
    "num_25_percent": [r"\bdigdami[i]?\s+shan", r"\b25\b", r"\bparsantii", r"\bpersentii"],
    "num_thousands_birr": [r"\bkuma\b", r"\bdhibba\b", r"\bbirrii\b", r"\bqarshii\b"],
}

ENGLISH_NUM_PATTERNS: Dict[str, List[str]] = {
    "num_1_single_unit": [r"\bone\b", r"\bsingle\b", r"\b1\b"],
    "num_2_units": [r"\btwo\b", r"\b2\b", r"\btwice\b"],
    "num_3_sprays_or_units": [r"\bthree\b", r"\b3\b", r"\bthrice\b"],
    "num_4_sprays_or_thousands": [r"\bfour\b", r"\b4\b"],
    "num_5_plus_sprays_or_thousands": [r"\bfive\b", r"\b5\b"],
    "num_6_to_7": [r"\bsix\b", r"\bseven\b", r"\b6\b", r"\b7\b"],
    "num_25_percent": [r"\b25\s*%", r"\b25\s*percent\b", r"\btwenty[- ]five\s*percent\b"],
    "num_thousands_birr": [r"\bthousand\b", r"\bbirr\b", r"\b\d{4,5}\b"],
}

# Questionnaire-only glossary (NO terms mined from the 61 reference transcripts)
QUESTIONNAIRE_GLOSSARY_OM_EN = """Questionnaire & general Ethiopian agronomy terms (Afaan Oromo → English):
- qamadii / sanyii filatamaa: wheat / improved (hybrid) wheat seed
- waagii / sataama / dhibee: wheat rust / crop disease
- qoricha / keemikaala / biifuu: fungicide or herbicide chemical / spraying
- manfaa: knapsack sprayer
- oomisha / callaa / wuxee: harvest / grain yield / productivity
- xaa'oo / madabaraa: fertilizer (standard Oromo / common loanword)
- aramaa: weeds (qoricha aramaa = herbicide)
- ogeessa qonnaa / gibrinnaa / D.A.: agricultural extension agent (development agent)
- leenjii / gorsa: training / agronomic advice
- yaalii / maasaa: trial / farm plot
- ollaa: neighbor farmer
- waldaa / yuuniyeenii / qabalee / ganda: cooperative / union / kebele
- mootummaa: government
- daldaaltota / nagaadee: private traders (standard / loanword)
- liqii: credit / loan
- waqtii / yeroodhaan / rooba dura: on time / before the rains
- qarshii / birrii / gatii: money / Ethiopian birr / price"""


def extract_taxonomy_tags(text: str) -> Set[str]:
    norm = (text or "").lower().replace("’", "'").replace("‘", "'").replace("ʼ", "'")
    matched: Set[str] = set()
    for tag, spec in INSIGHT_TAXONOMY.items():
        for pat in spec["patterns"]:
            if re.search(pat, norm):
                matched.add(tag)
                break
    return matched


def extract_number_concepts(text: str) -> Set[str]:
    norm = (text or "").lower().replace("’", "'").replace("‘", "'")
    matched: Set[str] = set()
    for key, pats in NUMBER_CONCEPTS.items():
        for pat in pats:
            if re.search(pat, norm):
                matched.add(key)
                break
    return matched


def extract_number_concepts_bilingual(text_om: str, text_en: str = "") -> Set[str]:
    found = set(extract_number_concepts(text_om))
    norm_en = (text_en or "").lower()
    for key, pats in ENGLISH_NUM_PATTERNS.items():
        for pat in pats:
            if re.search(pat, norm_en):
                found.add(key)
                break
    return found


def score_set_agreement(
    ref_sets: Sequence[Set[str]],
    hyp_sets: Sequence[Set[str]],
    all_labels: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    tp = fp = fn = 0
    jaccards: List[float] = []
    for r, h in zip(ref_sets, hyp_sets):
        tp += len(r & h)
        fp += len(h - r)
        fn += len(r - h)
        union = len(r | h)
        jaccards.append((len(r & h) / union) if union > 0 else 1.0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    macro_f1 = micro_f1
    if all_labels:
        per_label_f1: List[float] = []
        for lbl in all_labels:
            l_tp = sum(1 for r, h in zip(ref_sets, hyp_sets) if lbl in r and lbl in h)
            l_fp = sum(1 for r, h in zip(ref_sets, hyp_sets) if lbl not in r and lbl in h)
            l_fn = sum(1 for r, h in zip(ref_sets, hyp_sets) if lbl in r and lbl not in h)
            if (l_tp + l_fp + l_fn) == 0:
                continue
            per_label_f1.append((2 * l_tp) / (2 * l_tp + l_fp + l_fn))
        if per_label_f1:
            macro_f1 = sum(per_label_f1) / len(per_label_f1)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "jaccard": round(sum(jaccards) / max(len(jaccards), 1), 4),
    }


def _allowed_tags_block() -> str:
    return "\n".join(f'- "{k}": {v["label_en"]}' for k, v in INSIGHT_TAXONOMY.items())


def _parse_json_object(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"tags": [], "numbers": [], "farmer_summary_en": ""}


# ---------------------------------------------------------------------------
# 2. Local NLLB-200 Oromo (gaz_Latn) → English (eng_Latn) Translator
# ---------------------------------------------------------------------------

class NLLBOromoEnglishTranslator:
    """Translates Afaan Oromo (`gaz_Latn`) to English (`eng_Latn`) with NLLB-200."""

    def __init__(
        self,
        model_id: str = "facebook/nllb-200-3.3B",
        device: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model_id = model_id
        logger.info("Loading NLLB translator %s on %s...", model_id, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang="gaz_Latn")
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
        self.model.eval()
        self.eng_token_id = self.tokenizer.convert_tokens_to_ids("eng_Latn")

    def translate_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 8,
        max_new_tokens: int = 256,
    ) -> List[str]:
        import torch

        cleaned = [clean_oromo_fair_transcription(t) or (t or "").strip() for t in texts]
        outputs: List[str] = []
        self.tokenizer.src_lang = "gaz_Latn"
        with torch.no_grad():
            for i in tqdm(range(0, len(cleaned), batch_size), desc=f"NLLB ({self.model_id})", unit="batch"):
                batch = cleaned[i : i + batch_size]
                non_empty_idx = [j for j, s in enumerate(batch) if s]
                batch_out = [""] * len(batch)
                if non_empty_idx:
                    sub = [batch[j] for j in non_empty_idx]
                    enc = self.tokenizer(
                        sub,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    ).to(self.device)
                    gen = self.model.generate(
                        **enc,
                        forced_bos_token_id=self.eng_token_id,
                        max_new_tokens=max_new_tokens,
                        num_beams=4,
                    )
                    decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
                    for j, d in zip(non_empty_idx, decoded):
                        batch_out[j] = d.strip()
                outputs.extend(batch_out)
        return outputs


# ---------------------------------------------------------------------------
# 3. OpenRouter Questionnaire-Glossary MT & Argon Insight Extraction
# ---------------------------------------------------------------------------

GLOSSARY_MT_PROMPT = """You are an Afaan Oromo to English agricultural translator.
Survey question context ({question_id}): "{question}"

{glossary}

Task:
Translate the following Afaan Oromo ASR transcript of a farmer interview into faithful, plain English.
- Keep all farmer facts, numbers, quantities, and opinions exactly as spoken.
- Ignore obvious interviewer backchannels ("Eshii", "Hayyee") or ASR stutters.
- Do NOT invent agronomy details that are not in the Oromo text.
- Return ONLY the English translation (no preamble, no notes).

Afaan Oromo ASR transcript:
\"\"\"
{transcript}
\"\"\"

English translation:"""

TRANSLATED_INSIGHT_JSON_PROMPT = """You are an expert Ethiopian agricultural economist analyzing a farmer interview about hybrid wheat.
Survey Question asked to the farmer ({question_id}):
"{question}"

Task:
Analyze the farmer's response (ignoring interviewer prompts or backchannels) and return ONLY a valid JSON object:
{{
  "tags": [list of applicable tag strings from the allowed taxonomy below],
  "numbers": [list of explicit numbers/quantities mentioned by the farmer, e.g. "3-4 sprays", "25%", "2000 birr"],
  "farmer_summary_en": "1-2 concise sentences in English summarizing ONLY the farmer's core answer"
}}

Allowed taxonomy tag strings:
{allowed_tags}

Return ONLY raw JSON (no markdown fences, no commentary)."""


def call_openrouter_glossary_translate(
    transcript: str,
    *,
    question_id: str,
    question: str,
    model_id: str,
    api_key: str,
    use_glossary: bool = True,
    timeout_s: float = 90.0,
) -> str:
    import requests

    if not (transcript or "").strip():
        return ""
    prompt = GLOSSARY_MT_PROMPT.format(
        question_id=question_id,
        question=question,
        glossary=QUESTIONNAIRE_GLOSSARY_OM_EN if use_glossary else "",
        transcript=transcript,
    )
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(4):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout_s)
            if resp.status_code == 429:
                time.sleep(min(2.0**attempt, 20.0))
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            return str(content or "").strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("Glossary MT attempt %d failed: %s", attempt + 1, e)
            time.sleep(min(2.0**attempt, 15.0))
    return ""


def call_openrouter_extract_insights(
    *,
    oromo_text: str,
    english_translation: Optional[str],
    question_id: str,
    question: str,
    model_id: str = "google/gemini-4-argon",
    api_key: str,
    include_oromo_with_translation: bool = True,
    timeout_s: float = 90.0,
) -> Dict[str, Any]:
    import requests

    base = TRANSLATED_INSIGHT_JSON_PROMPT.format(
        question_id=question_id,
        question=question,
        allowed_tags=_allowed_tags_block(),
    )
    if english_translation:
        if include_oromo_with_translation and oromo_text:
            body = (
                f"\n\nEnglish translation of ASR transcript:\n\"\"\"\n{english_translation}\n\"\"\""
                f"\n\nOriginal Afaan Oromo ASR transcript (for term/number verification):\n\"\"\"\n{oromo_text}\n\"\"\""
            )
        else:
            body = f"\n\nEnglish translation of ASR transcript:\n\"\"\"\n{english_translation}\n\"\"\""
    else:
        body = f"\n\nAfaan Oromo ASR transcript:\n\"\"\"\n{oromo_text}\n\"\"\""

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": base + body}],
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(4):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout_s)
            if resp.status_code == 429:
                time.sleep(min(2.0**attempt, 20.0))
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            parsed = _parse_json_object(str(content))
            llm_tags = {t for t in (parsed.get("tags") or []) if t in INSIGHT_TAXONOMY}
            en_sum = str(parsed.get("farmer_summary_en") or "")
            num_str = " ".join(str(x) for x in (parsed.get("numbers") or [])) + " " + en_sum
            nums = extract_number_concepts_bilingual(
                oromo_text,
                (english_translation or "") + " " + num_str,
            )
            return {
                "tags": sorted(llm_tags),
                "tags_with_lexicon": sorted(llm_tags | extract_taxonomy_tags(oromo_text)),
                "numbers": sorted(nums),
                "farmer_summary_en": en_sum,
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM insight extract attempt %d failed: %s", attempt + 1, e)
            time.sleep(min(2.0**attempt, 15.0))
    return {
        "tags": sorted(extract_taxonomy_tags(oromo_text)),
        "tags_with_lexicon": sorted(extract_taxonomy_tags(oromo_text)),
        "numbers": sorted(extract_number_concepts(oromo_text)),
        "farmer_summary_en": english_translation or "",
    }


def _alias_keys_for_sample(sample_id: str, file_name: str = "") -> List[str]:
    """Build lookup aliases so `csv:om_64_….m4a:0` joins to manifest `om_64_….m4a`."""
    keys: List[str] = []
    for raw in (sample_id, file_name):
        raw = (raw or "").strip()
        if not raw:
            continue
        keys.append(raw)
        # ndizi_asr CSV loader ids look like: csv:{file_name}:{row_index}
        m = re.match(r"^csv:(.+):\d+$", raw)
        if m:
            keys.append(m.group(1))
            raw = m.group(1)  # continue aliasing from the file_name portion
        else:
            m = re.match(r"^cropconcept:([^:]+):\d+$", raw)
            if m:
                keys.append(m.group(1))
                raw = m.group(1)
        name = Path(raw).name
        keys.append(name)
        stem = Path(name).stem
        if stem and stem != name:
            keys.append(stem)
            for ext in (".m4a", ".wav", ".mp3", ".flac", ".ogg"):
                keys.append(stem + ext)
    # preserve order, drop empties/dupes
    seen: Set[str] = set()
    out: List[str] = []
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _load_asr_map(
    asr_path: Path,
    asr_col: str = "prediction",
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """Return (id -> ASR transcript, id -> ref_ratio if present in predictions file).

    Indexes every row under several aliases (`sample_id`, bare `file_name`, stem)
    so `csv:om_64_….m4a:0` joins cleanly to the CropConcept manifest `file_name`.
    """
    out: Dict[str, str] = {}
    ratios: Dict[str, float] = {}

    def _store(keys: Sequence[str], text: str, ratio: Optional[float]) -> None:
        for k in keys:
            out[k] = text
            if ratio is not None:
                ratios[k] = ratio

    if asr_path.suffix.lower() == ".jsonl":
        for line in asr_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            sid = str(r.get("sample_id") or r.get("file_name") or "")
            fn = str((r.get("metadata") or {}).get("file_name") or r.get("file_name") or "")
            if not sid and not fn:
                continue
            text = ""
            if asr_col in r and isinstance(r[asr_col], str):
                text = r[asr_col]
            elif isinstance(r.get("hypotheses"), dict) and asr_col in r["hypotheses"]:
                text = str(r["hypotheses"][asr_col] or "")
            elif "prediction" in r:
                text = str(r.get("prediction") or "")
            ratio = float(r["ref_ratio"]) if r.get("ref_ratio") is not None else None
            _store(_alias_keys_for_sample(sid, fn), text, ratio)
        return out, ratios

    with asr_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sid = str(r.get("sample_id") or "")
            fn = str(r.get("file_name") or "")
            if not sid and not fn:
                continue
            val = r.get(asr_col)
            if val is None:
                for fallback in ("sunbird_whisper_pred", "prediction", "hypothesis", "our_transcript"):
                    if fallback in r:
                        val = r[fallback]
                        break
            ratio = None
            if r.get("ref_ratio"):
                try:
                    ratio = float(r["ref_ratio"])
                except ValueError:
                    pass
            _store(_alias_keys_for_sample(sid, fn), str(val or ""), ratio)
    return out, ratios


def run_translate_extract_pipeline(
    *,
    asr_predictions: Path,
    manifest_csv: Path,
    output_dir: Path,
    asr_col: str = "prediction",
    nllb_model: str = "facebook/nllb-200-3.3B",
    mt_llm_model: str = "google/gemini-4-argon",
    extractor_model: str = "google/gemini-4-argon",
    skip_nllb: bool = False,
    skip_glossary_mt: bool = False,
    env_file: str = ".env",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    load_dotenv(env_file)
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required in .env or environment.")

    output_dir.mkdir(parents=True, exist_ok=True)
    asr_map, asr_ratios = _load_asr_map(asr_predictions, asr_col=asr_col)
    if not asr_map:
        raise SystemExit(f"No ASR predictions found in {asr_predictions} (col={asr_col})")

    with manifest_csv.open(encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    records: List[Dict[str, Any]] = []
    matched = 0
    for m in manifest_rows:
        fn = m.get("file_name") or m.get("sample_id") or ""
        sid = m.get("sample_id") or fn
        hyp = ""
        for key in _alias_keys_for_sample(sid, fn):
            if key in asr_map and asr_map[key]:
                hyp = asr_map[key]
                break
        if hyp:
            matched += 1
        ref = m.get("reference") or m.get("corrected_transcript") or ""
        our = m.get("our_transcript") or hyp or ""
        ratio = None
        for key in _alias_keys_for_sample(sid, fn):
            if key in asr_ratios:
                ratio = asr_ratios[key]
                break
        if ratio is None:
            ratio = float(
                m.get("ref_ratio")
                or m.get("ratio")
                or (len(ref.split()) / max(len(our.split()), 1))
            )
        records.append(
            {
                "file_name": fn or sid,
                "sample_id": sid,
                "question_id": m.get("question_id") or "",
                "question": m.get("question") or "",
                "ratio": float(ratio),
                "reference": ref,
                "asr_oromo": hyp,
                "ref_tags": sorted(extract_taxonomy_tags(ref)),
                "ref_nums": sorted(extract_number_concepts(ref)),
                "sunbird_lexicon_tags": sorted(extract_taxonomy_tags(hyp)),
                "sunbird_lexicon_nums": sorted(extract_number_concepts(hyp)),
            }
        )

    logger.info(
        "Joined ASR predictions to manifest: %d / %d clips matched (non-empty hyp)",
        matched,
        len(records),
    )
    if matched == 0:
        raise SystemExit(
            "No ASR predictions joined to the manifest. "
            f"Predictions look like sample_id=`csv:om_….m4a:0` while manifest uses "
            f"file_name=`om_….m4a`. Check --asr-predictions / --asr-col. "
            f"First prediction keys sample: {list(asr_map.keys())[:3]}"
        )
    if matched < len(records):
        logger.warning(
            "Only %d / %d clips have ASR text — remaining will score as empty.",
            matched,
            len(records),
        )

    if not skip_nllb:
        nllb = NLLBOromoEnglishTranslator(model_id=nllb_model, device=device)
        nllb_translations = nllb.translate_batch([r["asr_oromo"] for r in records])
        for r, tr in zip(records, nllb_translations):
            r["nllb_en"] = tr
        del nllb

    if not skip_glossary_mt:
        for r in tqdm(records, desc=f"Glossary MT ({mt_llm_model})", unit="clip"):
            r["glossary_mt_en"] = call_openrouter_glossary_translate(
                r["asr_oromo"],
                question_id=r["question_id"],
                question=r["question"],
                model_id=mt_llm_model,
                api_key=api_key,
                use_glossary=True,
            )

    for r in tqdm(records, desc=f"Argon extract ({extractor_model})", unit="clip"):
        direct = call_openrouter_extract_insights(
            oromo_text=r["asr_oromo"],
            english_translation=None,
            question_id=r["question_id"],
            question=r["question"],
            model_id=extractor_model,
            api_key=api_key,
        )
        r["direct_argon_tags"] = direct["tags"]
        r["direct_argon_plus_lex_tags"] = direct["tags_with_lexicon"]
        r["direct_argon_nums"] = direct["numbers"]
        r["direct_argon_summary_en"] = direct["farmer_summary_en"]

        if not skip_nllb:
            nllb_en_only = call_openrouter_extract_insights(
                oromo_text=r["asr_oromo"],
                english_translation=r.get("nllb_en") or "",
                question_id=r["question_id"],
                question=r["question"],
                model_id=extractor_model,
                api_key=api_key,
                include_oromo_with_translation=False,
            )
            r["nllb_en_only_argon_tags"] = nllb_en_only["tags"]
            r["nllb_en_only_argon_nums"] = nllb_en_only["numbers"]

            nllb_bi = call_openrouter_extract_insights(
                oromo_text=r["asr_oromo"],
                english_translation=r.get("nllb_en") or "",
                question_id=r["question_id"],
                question=r["question"],
                model_id=extractor_model,
                api_key=api_key,
                include_oromo_with_translation=True,
            )
            r["nllb_bilingual_argon_tags"] = nllb_bi["tags"]
            r["nllb_bilingual_plus_lex_tags"] = nllb_bi["tags_with_lexicon"]
            r["nllb_bilingual_argon_nums"] = nllb_bi["numbers"]

        if not skip_glossary_mt:
            gmt = call_openrouter_extract_insights(
                oromo_text=r["asr_oromo"],
                english_translation=r.get("glossary_mt_en") or "",
                question_id=r["question_id"],
                question=r["question"],
                model_id=extractor_model,
                api_key=api_key,
                include_oromo_with_translation=True,
            )
            r["glossary_mt_argon_tags"] = gmt["tags"]
            r["glossary_mt_plus_lex_tags"] = gmt["tags_with_lexicon"]
            r["glossary_mt_argon_nums"] = gmt["numbers"]

    arms: List[Tuple[str, str, str]] = [
        ("sunbird_lexicon_only", "sunbird_lexicon_tags", "sunbird_lexicon_nums"),
        ("sunbird_direct_argon", "direct_argon_tags", "direct_argon_nums"),
        ("sunbird_direct_argon+lex", "direct_argon_plus_lex_tags", "direct_argon_nums"),
    ]
    if not skip_nllb:
        arms.extend(
            [
                ("sunbird_nllb_en_only_argon", "nllb_en_only_argon_tags", "nllb_en_only_argon_nums"),
                ("sunbird_nllb_bilingual_argon", "nllb_bilingual_argon_tags", "nllb_bilingual_argon_nums"),
                ("sunbird_nllb_bilingual_argon+lex", "nllb_bilingual_plus_lex_tags", "nllb_bilingual_argon_nums"),
            ]
        )
    if not skip_glossary_mt:
        arms.extend(
            [
                ("sunbird_glossary_mt_argon", "glossary_mt_argon_tags", "glossary_mt_argon_nums"),
                ("sunbird_glossary_mt_argon+lex", "glossary_mt_plus_lex_tags", "glossary_mt_argon_nums"),
            ]
        )

    strata = {
        "all_counted": records,
        "strict_verbatim_ge_070": [r for r in records if r["ratio"] >= 0.70],
        "near_verbatim_ge_060": [r for r in records if r["ratio"] >= 0.60],
        "summarized_lt_060": [r for r in records if r["ratio"] < 0.60],
    }

    summary: Dict[str, Any] = {"strata": {}}
    csv_rows: List[Dict[str, Any]] = []
    for s_name, sub in strata.items():
        if not sub:
            continue
        summary["strata"][s_name] = {"n": len(sub), "arms": {}}
        ref_t = [set(r["ref_tags"]) for r in sub]
        ref_n = [set(r["ref_nums"]) for r in sub]
        for arm_name, t_col, n_col in arms:
            hyp_t = [set(r.get(t_col) or []) for r in sub]
            hyp_n = [set(r.get(n_col) or []) for r in sub]
            tm = score_set_agreement(ref_t, hyp_t, all_labels=ALL_TAG_KEYS)
            nm = score_set_agreement(ref_n, hyp_n, all_labels=list(NUMBER_CONCEPTS.keys()))
            row = {
                "stratum": s_name,
                "arm": arm_name,
                "n": len(sub),
                "taxonomy_precision": tm["precision"],
                "taxonomy_recall": tm["recall"],
                "taxonomy_micro_f1": tm["micro_f1"],
                "taxonomy_macro_f1": tm["macro_f1"],
                "taxonomy_jaccard": tm["jaccard"],
                "number_micro_f1": nm["micro_f1"],
                "number_recall": nm["recall"],
            }
            summary["strata"][s_name]["arms"][arm_name] = row
            csv_rows.append(row)

    (output_dir / "translate_extract_per_clip.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "translate_extract_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if csv_rows:
        with (output_dir / "translate_extract_summary.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)

    print("\n=== ASR → Oromo–English MT → Argon Insight Extraction ===")
    print(f"{'Stratum':<24} | {'Arm':<34} | {'Prec':>6} | {'Rec':>6} | {'MicroF1':>7} | {'MacroF1':>7} | {'NumF1':>6}")
    print("-" * 104)
    for row in csv_rows:
        print(
            f"{row['stratum']:<24} | {row['arm']:<34} | "
            f"{100*row['taxonomy_precision']:5.1f}% | {100*row['taxonomy_recall']:5.1f}% | "
            f"{100*row['taxonomy_micro_f1']:6.1f}% | {100*row['taxonomy_macro_f1']:6.1f}% | "
            f"{100*row['number_micro_f1']:5.1f}%"
        )

    return summary


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m ndizi_asr.translate_extract",
        description="Run ASR → Oromo–English translation (NLLB-200 + questionnaire-glossary MT) → Argon insight extraction",
    )
    p.add_argument(
        "--asr-predictions",
        required=True,
        help="Path to Sunbird predictions.csv or fusion_predictions.jsonl",
    )
    p.add_argument(
        "--asr-col",
        default="prediction",
        help="Column or hypothesis key for ASR text (default: prediction)",
    )
    p.add_argument(
        "--manifest-csv",
        default="../ndizi_eval/data/manifest_cropconcept_oromo.csv",
        help="Path to manifest_cropconcept_oromo.csv",
    )
    p.add_argument(
        "--output-dir",
        default="results/cropconcept_oromo_sunbird_mt_argon",
    )
    p.add_argument(
        "--nllb-model",
        default="facebook/nllb-200-3.3B",
        help="HF NLLB model id (e.g. facebook/nllb-200-3.3B or facebook/nllb-200-distilled-1.3B)",
    )
    p.add_argument("--mt-llm-model", default="google/gemini-4-argon")
    p.add_argument("--extractor-model", default="google/gemini-4-argon")
    p.add_argument("--skip-nllb", action="store_true")
    p.add_argument("--skip-glossary-mt", action="store_true")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_translate_extract_pipeline(
        asr_predictions=Path(args.asr_predictions).expanduser().resolve(),
        manifest_csv=Path(args.manifest_csv).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        asr_col=args.asr_col,
        nllb_model=args.nllb_model,
        mt_llm_model=args.mt_llm_model,
        extractor_model=args.extractor_model,
        skip_nllb=args.skip_nllb,
        skip_glossary_mt=args.skip_glossary_mt,
        env_file=args.env_file,
        device=args.device,
    )


if __name__ == "__main__":
    main()
