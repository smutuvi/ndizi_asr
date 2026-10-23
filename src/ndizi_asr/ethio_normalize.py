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


# ---------------------------------------------------------------------------
# Swahili Spoken-Survey Fair Normalizer (ndizi-1-sw / rural East African ASR)
# ---------------------------------------------------------------------------
# Canonicalizes orthographic / dialectal surface variants that denote the exact
# same spoken Swahili word across different human annotators:
#   1. Compound vs split conjunctions/adverbs (kwa sababu <-> kwasababu, etc.)
#   2. Bantu L1 R/L & H-drop survey variants (vizuli->vizuri, kuusu->kuhusu, etc.)
#   3. Expressive vowel/nasal lengthening (labdaaa->labda, chaaa->cha, mmmh->mmh)
#   4. Spoken backchannels/hesitations (mmh, eeh, aah, m) optionally stripped.

_SWAHILI_MOJIBAKE_RE = re.compile(r"(‚ä¶|‚Ä¶|√â|â€¦|â€™|â€œ|â€|\ufffd)", flags=re.IGNORECASE)
_SWAHILI_TRIPLE_CHAR_RE = re.compile(r"(.)\1{2,}")
_SWAHILI_BACKCHANNEL_RE = re.compile(
    r"\b(?:m+h+|h+m+|e+h+|a+h+|u+h+|o+h+|m{2,}|e{2,}|a{2,}|u{2,})\b"
)

_SWAHILI_COMPOUND_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bkwa\s+sababu\b"), "kwasababu"),
    (re.compile(r"\bkwa\s+kweli\b"), "kwakweli"),
    (re.compile(r"\bkwa\s+hiyo\b"), "kwahiyo"),
    (re.compile(r"\bkwa\s+hivyo\b"), "kwahivyo"),
    (re.compile(r"\bkwa\s+iyo\b"), "kwahiyo"),
    (re.compile(r"\bkwa\s+ivyo\b"), "kwahivyo"),
    (re.compile(r"\bvile\s+vile\b"), "vilevile"),
    (re.compile(r"\bmbali\s+mbali\b"), "mbalimbali"),
    (re.compile(r"\bmmoja\s+mmoja\b"), "mmojammoja"),
    (re.compile(r"\bkidogo\s+kidogo\b"), "kidogokidogo"),
    (re.compile(r"\bpole\s+pole\b"), "polepole"),
    (re.compile(r"\bsawa\s+sawa\b"), "sawa"),
    (re.compile(r"\bsasa\s+hivi\b"), "sasahivi"),
    (re.compile(r"\bsasa\s+ivi\b"), "sasahivi"),
    (re.compile(r"\bjuu\s+ya\b"), "juuya"),
    # Split agglutinated Swahili relative/auxiliary verbs in survey transcripts
    (re.compile(r"\bunacho\s+kipenda\b"), "unachokipenda"),
    (re.compile(r"\bninacho\s+kipenda\b"), "ninachokipenda"),
    (re.compile(r"\bnilicho\s+kipenda\b"), "nilichokipenda"),
    (re.compile(r"\bunacho\s+kiona\b"), "unachokiona"),
    (re.compile(r"\bninacho\s+kiona\b"), "ninachokiona"),
    (re.compile(r"\bunacho\s+penda\b"), "unachopenda"),
    (re.compile(r"\bninacho\s+penda\b"), "ninachopenda"),
    (re.compile(r"\bnisicho\s+kipenda\b"), "nisichokipenda"),
    (re.compile(r"\bnisicho\s+penda\b"), "nisichopenda"),
    (re.compile(r"\bkisicho\s+kupendeza\b"), "kisichokupendeza"),
    (re.compile(r"\bina\s+maana\b"), "inamaana"),
    (re.compile(r"\bina\s+zaa\b"), "inazaa"),
    (re.compile(r"\bina\s+kuwa\b"), "inakuwa"),
    (re.compile(r"\byana\s+kuwa\b"), "yanakuwa"),
    (re.compile(r"\bina\s+vutia\b"), "inavutia"),
    (re.compile(r"\byana\s+vutia\b"), "yanavutia"),
    (re.compile(r"\byan\s+vutia\b"), "yanavutia"),
]

_SWAHILI_SPELLING_MAP: dict[str, str] = {
    # L1 Bantu R/L alternation in rural Tanzania
    "vizuli": "vizuri",
    "nzuli": "nzuri",
    "mzuli": "mzuri",
    "wazuli": "wazuri",
    "pazuli": "pazuri",
    "kazuli": "kazuri",
    "uzuli": "uzuri",
    "mazuli": "mazuri",
    "mizuli": "mizuri",
    "zuli": "zuri",
    "kitaru": "kitalu",
    "malage": "maharage",
    "maalage": "maharage",
    "mahalage": "maharage",
    " tayali": "tayari",
    "tayali": "tayari",
    "habali": "habari",
    "hatali": "hatari",
    "sukali": "sukari",
    "daktali": "daktari",
    "serikali": "serikali",
    "angalau": "angalau",
    # Intervocalic / initial H-drop & glide spelling variants
    "apa": "hapa",
    "apo": "hapo",
    "ata": "hata",
    "ela": "hela",
    "ali": "hali",
    "awa": "hawa",
    "aya": "haya",
    "ii": "hii",
    "izi": "hizi",
    "uku": "huku",
    "uko": "huko",
    "uu": "huu",
    "uenda": "huenda",
    "kuusu": "kuhusu",
    "kusu": "kuhusu",
    "marage": "maharage",
    "maarage": "maharage",
    "maharagi": "maharage",
    "maragi": "maharage",
    "mauwa": "maua",
    "mvula": "mvua",
    "mvuwa": "mvua",
    "juwa": "jua",
    "kuwa": "kua",
    "hio": "hiyo",
    "hivo": "hivyo",
    "ivyo": "hivyo",
    "ivo": "hivyo",
    "iyo": "hiyo",
    "io": "hiyo",
    "kwaio": "kwahiyo",
    "kwaiyo": "kwahiyo",
    "kwahio": "kwahiyo",
    "kwaivo": "kwahivyo",
    "kwaivyo": "kwahivyo",
    "kwasabu": "kwasababu",
    "kwasaabu": "kwasababu",
    "alafu": "halafu",
    "amna": "hamna",
    "unaeza": "unaweza",
    "naeza": "naweza",
    "anaeza": "anaweza",
    "wanaeza": "wanaweza",
    "mda": "muda",
    "mana": "maana",
    "naa": "na",
    "yaa": "ya",
    "waa": "wa",
    "zaa": "za",
    "tuu": "tu",
    "inaa": "ina",
    "unaonae": "unaona",
    "vizurii": "vizuri",
    "mzurii": "mzuri",
    "nzurii": "nzuri",
    "sawasawa": "sawa",
    "mi": "mimi",
    "yanii": "yani",
    "nii": "ni",
    "icho": "hicho",
    "ivi": "hivi",
    "ivii": "hivi",
    "kadili": "kadri",
    "kadiri": "kadri",
    "saana": "sana",
    "sanaa": "sana",
    "kwenyee": "kwenye",
    "kwambaa": "kwamba",
    "unavosema": "unavyosema",
    "anavosema": "anavyosema",
    "navosema": "navyosema",
    "yamekua": "yamekuwa",
    "imekua": "imekuwa",
    "zimekua": "zimekuwa",
    "wamekua": "wamekuwa",
    "amekua": "amekuwa",
    "inakua": "inakuwa",
    "yanakua": "yanakuwa",
    "zinakua": "zinakuwa",
    "wanakua": "wanakuwa",
    "anakua": "anakuwa",
    "ilikuwepo": "iliyopo",
    "nachokipenda": "ninachokipenda",
    "nachopenda": "ninachopenda",
    "elezea": "eleza",
    "kuelezea": "kueleza",
    "anaelezea": "anaeleza",
    "unaelezea": "unaeleza",
    "naelezea": "naeleza",
    # Common colloquial / standard pairs in Ndizi survey prompts & responses
    "yaani": "yani",
    "ndio": "ndo",
    "ndiyo": "ndo",
    "sasaivi": "sasahivi",
    "sahivi": "sasahivi",
    "iliopo": "iliyopo",
    "ulio": "uliyo",
    "walio": "waliyo",
    "zilizo": "zilizopo",
    "ambazo": "ambazo",
    "naipendaa": "naipenda",
    "imesha": "imeisha",
    " teh": "eeh",
}


def clean_swahili_fair_transcription(
    text: str,
    *,
    drop_backchannels: bool = True,
) -> str:
    """Swahili survey fair normalization (compounds, R/L & H-drop, backchannels).

    Applied identically to reference and hypothesis so neither model nor
    annotator is penalized for purely orthographic choices in spontaneous
    farmer interviews.
    """
    t = unicodedata.normalize("NFKC", str(text or ""))
    t = _SWAHILI_MOJIBAKE_RE.sub(" ", t)
    t = t.lower()
    # Keep intra-word apostrophe in ng'ombe -> ngombe before punct->space
    t = _GLOTTAL_RE.sub("", t)
    t = _PLATFORM_PUNCT_RE.sub(" ", t)
    # Strip multi-letter backchannels (Eeee, Aaaaa, Mmmh) BEFORE collapsing 3+ repeated letters
    if drop_backchannels:
        t = _SWAHILI_BACKCHANNEL_RE.sub(" ", t)
    else:
        t = re.sub(r"\b(?:m+h+|h+m+|m{2,})\b", "mmh", t)
        t = re.sub(r"\b(?:e+h+|e{2,})\b", "eeh", t)
        t = re.sub(r"\b(?:a+h+|a{2,})\b", "aah", t)
    # Collapse 3+ repeated letters from expressive elongation (labdaaa -> labda)
    t = _SWAHILI_TRIPLE_CHAR_RE.sub(r"\1", t)
    # Normalize compound words before token mapping
    t = re.sub(r"\s+", " ", t).strip()
    for pat, repl in _SWAHILI_COMPOUND_PATTERNS:
        t = pat.sub(repl, t)
    tokens = []
    for tok in t.split():
        # Strip stray single-letter hesitation tokens at clause boundaries (m, e, u, o)
        if drop_backchannels and tok in {"m", "e", "u", "o", "hm", "mm", "eh", "ah", "uh", "oh"}:
            continue
        tokens.append(_SWAHILI_SPELLING_MAP.get(tok, tok))
    return " ".join(tokens)


def clean_swahili_training_target(text: str) -> str:
    """Canonicalize Swahili training labels while preserving sentence punctuation.

    Repairs glued punctuation, strips mojibake, collapses 3+ letter elongations
    (``labdaaa`` -> ``labda``), standardizes backchannels (``mmmh`` -> ``mmh``),
    and canonicalizes R/L, H-drop, and compound-word spelling variants so the
    Whisper decoder learns a single consistent orthography.
    """
    t = unicodedata.normalize("NFKC", str(text or ""))
    t = _SWAHILI_MOJIBAKE_RE.sub(" ", t)
    # Insert space after glued sentence punctuation: "vizuri.ndio" -> "vizuri. ndio"
    t = re.sub(r"([.,?!;:])([^\W\d_])", r"\1 \2", t, flags=re.UNICODE)
    # Collapse 3+ repeated letters (case-insensitive)
    t = re.sub(r"([a-zA-Z])\1{2,}", r"\1", t)
    # Canonicalize compound expressions (case-insensitive)
    for pat, repl in _SWAHILI_COMPOUND_PATTERNS:
        t = re.sub(pat.pattern, repl, t, flags=re.IGNORECASE)
    # Canonicalize word tokens while keeping trailing punctuation attached
    out_tokens: list[str] = []
    for raw_tok in t.split():
        m = re.match(r"^([^\w]*)([\w']+)([^\w]*)$", raw_tok, flags=re.UNICODE)
        if not m:
            continue
        lead, core, trail = m.groups()
        low = core.lower()
        if re.fullmatch(r"(?:m+h+|h+m+|m{2,})", low):
            canon = "mmh"
        elif re.fullmatch(r"(?:e+h+|e{2,})", low):
            canon = "eeh"
        elif re.fullmatch(r"(?:a+h+|a{2,})", low):
            canon = "aah"
        else:
            canon = _SWAHILI_SPELLING_MAP.get(low, core)
            if core and core[0].isupper() and canon and canon != core:
                canon = canon[0].upper() + canon[1:]
        out_tokens.append(f"{lead}{canon}{trail}")
    return re.sub(r"\s+", " ", " ".join(out_tokens)).strip()


