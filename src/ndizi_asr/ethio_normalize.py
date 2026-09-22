"""Ethio-ASR text normalization (Ge'ez + Latin). Ported from ndizi_eval."""

from __future__ import annotations

import re
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

PUNCTUATION_MARKS: Set[str] = set("-@<%]>?!.,|;:'&+።፠፡፣፤፥፦፧፨()[]{}#*_~^=`/\\")

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
