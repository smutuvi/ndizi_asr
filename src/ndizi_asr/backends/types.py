from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TranscriptionResult:
    text: str
    latency_seconds: float
    model: str
    raw_response: Optional[dict[str, Any]] = None
    error: Optional[str] = None
