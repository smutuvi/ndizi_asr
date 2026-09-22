"""Sunbird Whisper language codes and forced decoder token ids.

Source: Sunbird/asr-whisper-51-african-languages model card.
"""

from __future__ import annotations

# Whisper decoder language token ids used by the Sunbird adaptation.
LANGUAGE_TOKENS_WHISPER: dict[str, int] = {
    # Existing Whisper language codes
    "eng": 50259,
    "fra": 50265,
    "swa": 50318,
    "sna": 50324,
    "yor": 50325,
    "som": 50326,
    "afr": 50327,
    "amh": 50334,
    "mlg": 50349,
    "lin": 50353,
    "hau": 50354,
    # Overwritten unused language tokens for additional African languages
    "ach": 50357,
    "aka": 50356,
    "bam": 50355,
    "bem": 50352,
    "ber": 50351,
    "cgg": 50350,
    "dag": 50348,
    "dga": 50347,
    "ewe": 50346,
    "ful": 50345,
    "ibo": 50344,
    "kab": 50343,
    "kau": 50342,
    "kik": 50341,
    "kin": 50340,
    "kln": 50339,
    "koo": 50338,
    "kpo": 50337,
    "led": 50336,
    "lgg": 50335,
    "lth": 50333,
    "lug": 50332,
    "luo": 50331,
    "luy": 50330,
    "myx": 50329,
    "nbl": 50328,
    "nya": 50323,
    "nyn": 50322,
    "orm": 50321,
    "pcm": 50320,
    "ruc": 50319,
    "rwm": 50317,
    "sot": 50316,
    "teo": 50315,
    "tsn": 50314,
    "ttj": 50313,
    "wol": 50312,
    "xho": 50311,
    "xog": 50310,
    "zul": 50309,
}

# Friendly / English names → ISO-639-3 style Sunbird codes
LANGUAGE_NAMES: dict[str, str] = {
    "Acholi": "ach",
    "Afrikaans": "afr",
    "Akan": "aka",
    "Amharic": "amh",
    "Ateso": "teo",
    "Bambara": "bam",
    "Bemba": "bem",
    "Berber": "ber",
    "Chichewa": "nya",
    "Dagaare": "dga",
    "Dagbani": "dag",
    "English": "eng",
    "Ewe": "ewe",
    "French": "fra",
    "Fulani": "ful",
    "Hausa": "hau",
    "Igbo": "ibo",
    "Ikposo": "kpo",
    "Kabyle": "kab",
    "Kalenjin": "kln",
    "Kanuri": "kau",
    "Kikuyu": "kik",
    "Kinyarwanda": "kin",
    "Kwamba": "rwm",
    "Lendu": "led",
    "Lingala": "lin",
    "Luganda": "lug",
    "Lugbara": "lgg",
    "Luhya": "luy",
    "Lumasaba": "myx",
    "Luo": "luo",
    "Lusoga": "xog",
    "Malagasy": "mlg",
    "Ndebele": "nbl",
    "Nigerian Pidgin": "pcm",
    "Oromo": "orm",
    "Rukiga": "cgg",
    "Rukonjo": "koo",
    "Runyankole": "nyn",
    "Ruruuli": "ruc",
    "Rutooro": "ttj",
    "Shona": "sna",
    "Somali": "som",
    "Sotho": "sot",
    "Swahili": "swa",
    "Thur": "lth",
    "Tswana": "tsn",
    "Wolof": "wol",
    "Xhosa": "xho",
    "Yoruba": "yor",
    "Zulu": "zul",
}

# Common aliases used in this repo / FLEURS configs
LANGUAGE_ALIASES: dict[str, str] = {
    "swahili": "swa",
    "amharic": "amh",
    "oromo": "orm",
    "hausa": "hau",
    "sw": "swa",
    "am": "amh",
    "om": "orm",
    "ha": "hau",
    "sw_ke": "swa",
    "am_et": "amh",
    "om_et": "orm",
    "ha_ng": "hau",
}


def resolve_sunbird_language_code(language: str | None, language_config: str | None = None) -> str:
    """Map friendly names / FLEURS codes to Sunbird language codes."""
    for candidate in (language, language_config):
        if not candidate:
            continue
        key = str(candidate).strip()
        if key in LANGUAGE_TOKENS_WHISPER:
            return key
        lower = key.lower()
        if lower in LANGUAGE_ALIASES:
            return LANGUAGE_ALIASES[lower]
        # Title-case English name
        titled = key[:1].upper() + key[1:].lower() if key else key
        if titled in LANGUAGE_NAMES:
            return LANGUAGE_NAMES[titled]
        # Case-insensitive English name lookup
        for name, code in LANGUAGE_NAMES.items():
            if name.lower() == lower:
                return code
    raise ValueError(
        f"Could not map language={language!r} config={language_config!r} "
        f"to a Sunbird language code. Known aliases: {sorted(LANGUAGE_ALIASES)}"
    )
