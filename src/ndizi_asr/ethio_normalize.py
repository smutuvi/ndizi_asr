"""Ethio-ASR text normalization (Ge'ez + Latin). Ported from ndizi_eval."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional, Set

DEFAULT_ETHIO_CHARACTER_SET = (
    " !#$%&'*+,-.0123456789=?@abcdefghijklmnopqrstuvwxyz"
    "ሀሁሂሃሄህሆሇለሉሊላሌልሎሏሐሑሒሓሔሕሖሗመሙሚማሜምሞሟ"
    "ሠሡሢሣሤሥሦሧረሩሪራሬርሮሯሰሱሲሳሴስሶሷሸሹሺሻሼሽሾሿ"
    "ቀቁቂቃቄቅቆቇቈቊቋቌቍቐቑቒቓቔቕቖቘቚቛቜቝ"
    "በቡቢባቤብቦቧቨቩቪቫቬቭቮቯተቱቲታቴትቶቷቸቹቺቻቼችቾቿ"
    "ኀኁኂኃኄኅኆኇኈኊኋኌኍነኑኒናኔንኖኗኘኙኚኛኜኝኞኟ"
    "አኡኢኣኤእኦኧከኩኪካኬክኮኯኰኲኳኴኵኸኹኺኻኼኽኾ"
    "ዀዂዃዄዅወዉዊዋዌውዎዏዐዑዒዓዔዕዖዘዙዚዛዜዝዞዟ"
    "ዠዡዢዣዤዥዦዧየዩዪያዬይዮዯደዱዲዳዴድዶዷዸዹዺዻዼዽዾዿ"
    "ጀጁጂጃጄጅጆጇገጉጊጋጌግጎጏጐጒጓጔጕጘጙጚጛጜጝጞጟ"
    "ጠጡጢጣጤጥጦጧጨጩጪጫጬጭጮጯጰጱጲጳጴጵጶጷጸጹጺጻጼጽጾጿ"
    "ፀፁፂፃፄፅፆፇፈፉፊፋፌፍፎፏፐፑፒፓፔፕፖፗፘፙፚ"
    "፠፡።፣፤፥፦፧፨፩፪፫፬፭፮፯፰፱፲፳፴፵፶፷፸፹፺፻፼€"
)

PUNCTUATION_MARKS: Set[str] = set("-@<%]>?!.,|;:'’‘ʼʻ“”&+።፠፡፣፤፥፦፧፨()[]{}#*_~^=`/\\")

ALLOWED_CHARACTERS: Set[str] = set(DEFAULT_ETHIO_CHARACTER_SET) - PUNCTUATION_MARKS

_AMHARIC_LANG_CODES = frozenset(
    {"amh", "am", "amharic", "am_et", "amh_asr", "am-et"}
)


class GeezNormalizer:
    def __init__(self) -> None:
        self.char_map = {
            "ሀ": "ሃ",
            "ሐ": "ሃ",
            "ሓ": "ሃ",
            "ኅ": "ሃ",
            "ኻ": "ሃ",
            "ኃ": "ሃ",
            "ኁ": "ሁ",
            "ኂ": "ሂ",
            "ኄ": "ሄ",
            "ኆ": "ሆ",
            "ሑ": "ሁ",
            "ሒ": "ሂ",
            "ሔ": "ሄ",
            "ሕ": "ህ",
            "ሖ": "ሆ",
            "ሠ": "ሰ",
            "ሡ": "ሱ",
            "ሢ": "ሲ",
            "ሣ": "ሳ",
            "ሤ": "ሴ",
            "ሥ": "ስ",
            "ሦ": "ሶ",
            "ዐ": "አ",
            "ዑ": "ኡ",
            "ዒ": "ኢ",
            "ዓ": "አ",
            "ኣ": "አ",
            "ዔ": "ኤ",
            "ዕ": "እ",
            "ዖ": "ኦ",
        }
        self.labialized_map = [
            (r"ኰ", "ኮ"),
            (r"ዀ", "ኮ"),
            (r"ኵ", "ኩ"),
            (r"ዅ", "ኩ"),
        ]

    def normalize(self, text: str, remove_punctuation: bool = True) -> str:
        if remove_punctuation:
            for p in PUNCTUATION_MARKS:
                text = text.replace(p, " ")
        text = text.translate(str.maketrans(self.char_map))
        for pattern, replacement in self.labialized_map:
            text = re.sub(pattern, replacement, text)
        return re.sub(r"\s+", " ", text).strip()


_geez: Optional[GeezNormalizer] = None


def get_geez_normalizer() -> GeezNormalizer:
    global _geez
    if _geez is None:
        _geez = GeezNormalizer()
    return _geez


def process_text(text: str, *, allowed: Optional[Set[str]] = None) -> str:
    allowed_set = allowed if allowed is not None else ALLOWED_CHARACTERS
    for punc in PUNCTUATION_MARKS:
        text = text.replace(punc, " ")
    text = "".join(c for c in text if c.lower() in allowed_set)
    return re.sub(r"\s+", " ", text).strip().lower()


def clean_ethio_transcription(
    text: str,
    *,
    language: Optional[str] = None,
    apply_geez_normalizer: bool = True,
) -> str:
    t = str(text or "").strip()
    lang = str(language or "").strip().lower()
    if apply_geez_normalizer and lang in _AMHARIC_LANG_CODES:
        t = get_geez_normalizer().normalize(t, remove_punctuation=True)
    return process_text(t)


_GLOTTAL_RE = re.compile(r"(?<=[a-zA-Z])['’‘ʼʻ`](?=[a-zA-Z])")

# Colleague / platform scoring tidy: any punctuation or symbol → space.
_PLATFORM_PUNCT_RE = re.compile(
    r"""[
        !"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~
        ’‘ʼʻ“”„«»…–—؟،؛
        ፠፡።፣፤፥፦፧፨
    ]+""",
    re.VERBOSE,
)

# Scoring-only Qubee length collapse: models and annotators often disagree on
# vowel/consonant gemination (timirtii/timmirtii, qaba/qabaa, akkaataa/akkataa).
_GEMINATION_RE = re.compile(r"(.)\1+")

# High-frequency particles / fillers / crop orthography that differ by more
# than gemination alone. Applied after lowercase + hudhaa strip, before
# gemination collapse. Keep this conservative (no speculative content words).
_OROMO_SPELLING_MAP: dict[str, str] = {
    "nuuf": "nu",
    "nuu": "nu",
    "eshii": "eshi",
    "eeshi": "eshi",
    "eeh": "ee",
    "ehh": "ee",
    "eh": "ee",
    "ahh": "aa",
    "yaauu": "yoo",
    "yaaoo": "yoo",
    "yawuu": "yoo",
    "debalan": "deebbala",
    "deebalan": "deebbala",
    "debala": "deebbala",
    "omisha": "oomisha",
    "omishaa": "oomisha",
    "oomishaa": "oomisha",
    "sanyi": "sanyii",
    "sanyiin": "sanyii",
    "odoo": "oduu",
}


def _apply_oromo_spelling_map(text: str) -> str:
    tokens = text.split()
    mapped = [_OROMO_SPELLING_MAP.get(tok, tok) for tok in tokens]
    return " ".join(mapped)


def _collapse_qubee_gemination(text: str) -> str:
    """Collapse runs of identical letters for fair ASR scoring (length not scored)."""
    return _GEMINATION_RE.sub(r"\1", text)


def clean_oromo_platform_transcription(text: str) -> str:
    """Colleague/platform scoring tidy (identical on hyp and ref).

    Matches the shared CropConcept artifact preprocess:
    - standardise character encoding (Unicode NFKC)
    - lowercase everything
    - replace each punctuation mark with a space
    - collapse extra whitespace

    No Oromo spelling map and no gemination collapse — use ``oromo_fair`` for that.
    """
    t = unicodedata.normalize("NFKC", str(text or ""))
    t = t.lower()
    t = _PLATFORM_PUNCT_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def clean_oromo_fair_transcription(text: str) -> str:
    """Oromo-aware scoring normalize: hudhaa strip, spelling map, gemination collapse.

    Starts from the platform tidy, then:
    - Mid-word Qubee hudhaa (apostrophe) is removed without splitting the word
      (applied on the raw string before punct→space so ``re'ee`` stays one token).
    - A small particle/orthography map plus letter-run collapse reduce false
      errors from annotator vs model spelling of the same spoken form.
    """
    t = _GLOTTAL_RE.sub("", str(text or "").strip())
    t = process_text(t)
    t = _apply_oromo_spelling_map(t)
    t = _collapse_qubee_gemination(t)
    return re.sub(r"\s+", " ", t).strip()
