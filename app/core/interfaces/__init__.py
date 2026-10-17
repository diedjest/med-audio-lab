"""Abstract interface specifications for MedAudioLab v2 core domain."""

from __future__ import annotations

from app.core.interfaces.audio_engine import (
    IAudioCaptureEngine,
    IAudioPlaybackEngine,
)
from app.core.interfaces.export_service import IExportService
from app.core.interfaces.serial_engine import ISerialEngine
from app.core.interfaces.storage_repository import IStorageRepository

__all__: list[str] = [
    "IAudioCaptureEngine",
    "IAudioPlaybackEngine",
    "ISerialEngine",
    "IStorageRepository",
    "IExportService",
]
