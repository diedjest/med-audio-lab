"""Abstract interface for application storage and session persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.models.recording_session import RecordingSession


class IStorageRepository(ABC):
    """Abstract persistence repository for recordings, analysis metrics, and settings."""

    @abstractmethod
    def ensure_schema(self) -> None:
        """Create database tables and indexes if they do not exist."""
        ...

    @abstractmethod
    def add_recording(self, session: RecordingSession) -> int:
        """Insert a new recording session and return its generated integer ID."""
        ...

    @abstractmethod
    def get_recording(self, id: int) -> RecordingSession | None:
        """Retrieve a recording session by ID, or return None if not found."""
        ...

    @abstractmethod
    def list_recordings(self) -> list[RecordingSession]:
        """Retrieve all saved recording sessions, ordered by creation date descending."""
        ...

    @abstractmethod
    def delete_recording(self, id: int) -> bool:
        """Delete a recording and its metrics; return False if the id does not exist."""
        ...

    @abstractmethod
    def save_metrics(self, recording_id: int, metrics: dict[str, str]) -> None:
        """Persist key-value analysis metrics associated with a recording ID."""
        ...

    @abstractmethod
    def get_metrics(self, recording_id: int) -> dict[str, str]:
        """Fetch all stored key-value analysis metrics for a given recording ID."""
        ...

    @abstractmethod
    def get_setting(self, key: str, default: str = "") -> str:
        """Retrieve a persistent setting value by key, falling back to default."""
        ...

    @abstractmethod
    def set_setting(self, key: str, value: str) -> None:
        """Store or update a persistent key-value configuration setting."""
        ...


__all__: list[str] = ["IStorageRepository"]
