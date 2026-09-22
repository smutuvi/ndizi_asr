"""Back-compat shim: prefer `import ndizi_asr` / `ndizi-asr` CLI."""

from __future__ import annotations

import warnings

warnings.warn(
    "gemini_asr_bench is renamed to ndizi_asr; update imports and use `ndizi-asr`.",
    DeprecationWarning,
    stacklevel=2,
)

from ndizi_asr import *  # noqa: F401,F403
from ndizi_asr import __version__

__all__ = ["__version__"]
