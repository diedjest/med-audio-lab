"""Saving, loading, and listing recordings (WAV file + database record)."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np

from app.core.dsp.metrics import compute_signal_metrics, format_metrics
from app.core.interfaces.storage_repository import IStorageRepository
from app.core.models.recording_session import RecordingSession
from app.infrastructure.storage import file_io
from app.services.settings_service import VALID_SOURCES, SettingsService

_TIMESTAMP_FILE_FORMAT = "%Y%m%d_%H%M%S"
_TIMESTAMP_DB_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_SLUG_LENGTH = 80
_FALLBACK_SLUG = "record"
_RESERVED_WINDOWS_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)
# Peaks within this tolerance of full scale are still written as 16-bit PCM
_CLIP_TOLERANCE = 1.0 + 1e-4


class RecordingNotFoundError(LookupError):
    """Raised when a recording id is not in the database."""


def slugify(text: str, fallback: str = _FALLBACK_SLUG) -> str:
    """Make ``text`` safe for use in a file name (Unicode letters are kept).

    Runs of characters other than letters, digits, ``_`` and ``-`` become one
    underscore; the result is capped at 80 characters and never a reserved
    Windows device name.
    """
    slug = re.sub(r"[^\w-]+", "_", text.strip(), flags=re.UNICODE)
    slug = re.sub(r"_+", "_", slug).strip("_")[:_MAX_SLUG_LENGTH].strip("_")
    if not slug:
        return fallback
    if slug.lower() in _RESERVED_WINDOWS_NAMES:
        return f"{slug}_"
    return slug


class ProjectManager:
    """Coordinates the WAV file store and the recordings database.

    Note: audio files are read and written with ``app.infrastructure.storage.file_io``
    directly, since the repository interface only covers metadata.
    """

    def __init__(self, repository: IStorageRepository, settings: SettingsService) -> None:
        self._repository = repository
        self._settings = settings

    # ------------------------------------------------------------ filenames
    def generate_filename(self, title: str) -> str:
        """Return a timestamped, slugified WAV file name, e.g. ``20260101_120000_my_title.wav``."""
        stamp = datetime.now().strftime(_TIMESTAMP_FILE_FORMAT)
        return f"{stamp}_{slugify(title)}.wav"

    @staticmethod
    def _unique_path(directory: Path, filename: str) -> Path:
        """``directory/filename``, or with ``_2``, ``_3`` ... if it already exists."""
        candidate = directory / filename
        stem, suffix = candidate.stem, candidate.suffix
        counter = 2
        while candidate.exists():
            candidate = directory / f"{stem}_{counter}{suffix}"
            counter += 1
        return candidate

    # --------------------------------------------------------------- saving
    def save_recording(
        self,
        data: np.ndarray,
        sample_rate: int,
        title: str,
        patient: str,
        notes: str,
        source: str,
    ) -> RecordingSession:
        """Write ``data`` to a WAV file in the save directory and register it.

        Audio is stored as 16-bit PCM. If the signal exceeds full scale
        (|x| > 1, e.g. after filtering), 32-bit float is used instead so nothing
        is clipped. If the database insert fails the WAV file is removed again.

        Args:
            data: Samples, shape ``(n,)`` or ``(n, channels)``.
            sample_rate: Sample rate in Hz.
            title: Recording title (also used for the file name).
            patient: Patient / sample identifier.
            notes: Free-text notes.
            source: ``"microphone"`` or ``"serial"``.

        Returns:
            The stored ``RecordingSession`` with its database id and file path.

        Raises:
            ValueError: empty data, non-positive sample rate, or unknown source.
        """
        samples = np.asarray(data, dtype=np.float32)
        if samples.ndim not in (1, 2) or samples.shape[0] == 0:
            raise ValueError(f"No audio data to save (shape {samples.shape}).")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if source not in VALID_SOURCES:
            raise ValueError(f"source must be one of {VALID_SOURCES}, got {source!r}")

        finite = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
        subtype = self._settings.wav_bit_depth
        if float(np.max(np.abs(finite))) > _CLIP_TOLERANCE:
            subtype = "FLOAT_32"  # integer PCM would clip

        directory = self._settings.save_directory
        directory.mkdir(parents=True, exist_ok=True)
        clean_title = title.strip()
        path = self._unique_path(directory, self.generate_filename(clean_title))
        file_io.save_wav(finite, int(sample_rate), path, subtype)

        session = RecordingSession(
            id=None,
            created_at=datetime.now().strftime(_TIMESTAMP_DB_FORMAT),
            title=clean_title,
            patient=patient.strip(),
            notes=notes.strip(),
            sample_rate=int(sample_rate),
            duration_s=samples.shape[0] / float(sample_rate),
            file_path=str(path),
            source=source,
        )
        try:
            session.id = self._repository.add_recording(session)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return session

    # -------------------------------------------------------------- loading
    def load_recording(self, recording_id: int) -> tuple[RecordingSession, np.ndarray]:
        """Load a recording's metadata and audio.

        Returns:
            ``(session, samples)`` where samples are float32, exactly as saved (first
            channel for multi-channel files).

        Raises:
            RecordingNotFoundError: no such recording id.
            FileNotFoundError: the recording's WAV file no longer exists.
        """
        session = self._repository.get_recording(recording_id)
        if session is None:
            raise RecordingNotFoundError(f"Recording {recording_id} not found.")
        if not Path(session.file_path).is_file():
            raise FileNotFoundError(
                f"Audio file for recording {recording_id} is missing: {session.file_path}"
            )
        _, samples = file_io.read_wav(
            session.file_path, mono=True, normalize=True, keep_float_level=True
        )
        return session, samples

    def delete_recording(self, recording_id: int, delete_file: bool = False) -> None:
        """Remove a recording from the library, optionally deleting its audio file.

        The database record (and its metrics) is always removed.  The WAV file is only
        deleted when ``delete_file`` is True, it is a ``.wav`` file, and no other
        recording still points at it.

        Raises:
            RecordingNotFoundError: no such recording id.
            OSError: the record was removed but the file could not be deleted.
        """
        session = self._repository.get_recording(recording_id)
        if session is None or not self._repository.delete_recording(recording_id):
            raise RecordingNotFoundError(f"Recording {recording_id} not found.")
        if not delete_file:
            return

        path = Path(session.file_path)
        if path.suffix.lower() != ".wav" or not path.is_file():
            return
        if any(other.file_path == session.file_path for other in self._repository.list_recordings()):
            return  # another library entry still uses this file
        try:
            path.unlink()
        except OSError as exc:
            raise OSError(
                f"Recording {recording_id} was removed from the library, but its file "
                f"could not be deleted: {exc}"
            ) from exc

    def get_recording(self, recording_id: int) -> RecordingSession | None:
        """Metadata of one recording, or None."""
        return self._repository.get_recording(recording_id)

    def get_metrics(self, recording_id: int) -> dict[str, str]:
        """Stored analysis metrics of a recording (empty if none were saved)."""
        return self._repository.get_metrics(recording_id)

    def ensure_metrics(self, recording_id: int) -> dict[str, str]:
        """Stored metrics of a recording; computed from its audio and stored if missing.

        Raises:
            RecordingNotFoundError: no such recording id.
            FileNotFoundError: metrics are missing and the audio file is gone.
        """
        stored = self._repository.get_metrics(recording_id)
        if stored:
            return stored
        session, samples = self.load_recording(recording_id)
        metrics = format_metrics(compute_signal_metrics(samples, session.sample_rate))
        self._repository.save_metrics(recording_id, metrics)
        return metrics

    def list_recordings(self) -> list[RecordingSession]:
        """Return all recordings, newest first."""
        return self._repository.list_recordings()


__all__: list[str] = ["ProjectManager", "RecordingNotFoundError", "slugify"]
