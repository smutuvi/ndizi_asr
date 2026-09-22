from .resolve import (
    MODEL_GROUPS,
    ModelSpec,
    canonicalize_model_id,
    expand_model_groups,
    list_known_models,
    needs_hf_auth,
    needs_omnilingual,
    needs_openrouter,
    parse_model_spec,
)
from .sunbird_langs import resolve_sunbird_language_code
from .types import TranscriptionResult

__all__ = [
    "MODEL_GROUPS",
    "ModelSpec",
    "TranscriptionResult",
    "canonicalize_model_id",
    "expand_model_groups",
    "list_known_models",
    "needs_hf_auth",
    "needs_omnilingual",
    "needs_openrouter",
    "parse_model_spec",
    "resolve_sunbird_language_code",
]
