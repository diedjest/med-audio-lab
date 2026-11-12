"""Recording session entity model representing metadata and stored audio file info."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class RecordingSession:
    """Represents a persisted audio or telemetry recording session."""

    id: int | None = None
    created_at: str = ""
    title: str = ""
    patient: str = ""
    notes: str = ""
    sample_rate: int = 44100
    duration_s: float = 0.0
    file_path: str = ""
    source: Literal["microphone", "serial"] | str = "microphone"


__all__: list[str] = ["RecordingSession"]
