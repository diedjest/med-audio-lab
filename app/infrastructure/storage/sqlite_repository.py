"""SQLite persistence for recordings, analysis metrics, and settings."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from app.core.interfaces.storage_repository import IStorageRepository
from app.core.models.recording_session import RecordingSession

DEFAULT_APP_DIR: Path = Path.home() / "Documents" / "MedAudioLab"
DEFAULT_DB_PATH: Path = DEFAULT_APP_DIR / "med_audio_lab.sqlite3"

#: Pass this as ``db_path`` for a private in-memory database (nothing touches the disk).
MEMORY_DB = ":memory:"

_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
_BUSY_TIMEOUT_S = 10.0

_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS recordings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        title TEXT NOT NULL,
        patient TEXT NOT NULL DEFAULT '',
        notes TEXT NOT NULL DEFAULT '',
        samplerate INTEGER NOT NULL,
        duration REAL NOT NULL,
        file_path TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'microphone'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS metrics (
        recording_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        value TEXT NOT NULL,
        PRIMARY KEY (recording_id, name),
        FOREIGN KEY (recording_id) REFERENCES recordings(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
)

_SELECT_RECORDING = (
    "SELECT id, created_at, title, patient, notes, samplerate, duration, file_path, source "
    "FROM recordings"
)


def _to_session(row: sqlite3.Row) -> RecordingSession:
    return RecordingSession(
        id=int(row["id"]),
        created_at=str(row["created_at"]),
        title=str(row["title"]),
        patient=str(row["patient"]),
        notes=str(row["notes"]),
        sample_rate=int(row["samplerate"]),
        duration_s=float(row["duration"]),
        file_path=str(row["file_path"]),
        source=str(row["source"]),
    )


class SqliteRepository(IStorageRepository):
    """SQLite-backed repository.

    For a database file, every operation opens its own short-lived connection, so the
    repository can be shared freely between threads.  A ``":memory:"`` database only
    exists inside one connection, so that case keeps a single connection open
    (guarded by a lock) until :meth:`close`.  The schema is created lazily on first use
    (and explicitly by :meth:`ensure_schema`).
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._in_memory = db_path is not None and str(db_path) == MEMORY_DB
        self._db_path = Path(MEMORY_DB) if self._in_memory else (Path(db_path) if db_path is not None else DEFAULT_DB_PATH)
        self._schema_ready = False
        self._schema_lock = threading.Lock()
        self._memory_con: sqlite3.Connection | None = None
        self._memory_lock = threading.RLock()

    @property
    def db_path(self) -> Path:
        """Location of the database file (``:memory:`` for an in-memory database)."""
        return self._db_path

    @property
    def is_memory(self) -> bool:
        """True for an in-memory database."""
        return self._in_memory

    @property
    def records_dir(self) -> Path:
        """Default directory for recorded WAV files."""
        return self._db_path.parent / "records"

    @property
    def exports_dir(self) -> Path:
        """Default directory for exported reports."""
        return self._db_path.parent / "exports"

    def _shared_connection(self) -> sqlite3.Connection:
        """The one connection of an in-memory database (created on first use)."""
        if self._memory_con is None:
            con = sqlite3.connect(MEMORY_DB, check_same_thread=False)
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA foreign_keys = ON")
            self._memory_con = con
        return self._memory_con

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A connection inside a transaction: commit on success, roll back on error."""
        if not self._schema_ready:
            self.ensure_schema()
        if self._in_memory:
            with self._memory_lock, self._shared_connection() as con:  # `with con` = transaction only
                yield con
            return
        con = sqlite3.connect(self._db_path, timeout=_BUSY_TIMEOUT_S)
        try:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA foreign_keys = ON")
            with con:
                yield con
        finally:
            con.close()

    @staticmethod
    def _create_schema(con: sqlite3.Connection) -> None:
        for statement in _SCHEMA:
            con.execute(statement)
        columns = {row[1] for row in con.execute("PRAGMA table_info(recordings)")}
        if "source" not in columns:  # database created before v2
            con.execute(
                "ALTER TABLE recordings ADD COLUMN source TEXT NOT NULL DEFAULT 'microphone'"
            )

    def ensure_schema(self) -> None:
        """Create directories, tables, and migrate databases from older versions."""
        with self._schema_lock:
            if self._in_memory:
                with self._memory_lock, self._shared_connection() as con:
                    self._create_schema(con)
            else:
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
                self.records_dir.mkdir(parents=True, exist_ok=True)
                self.exports_dir.mkdir(parents=True, exist_ok=True)
                con = sqlite3.connect(self._db_path, timeout=_BUSY_TIMEOUT_S)
                try:
                    with con:
                        self._create_schema(con)
                finally:
                    con.close()
            self._schema_ready = True

    def close(self) -> None:
        """Release the connection of an in-memory database (its content is lost).

        File databases hold no open connection between operations, so this is a no-op
        for them.
        """
        with self._memory_lock:
            if self._memory_con is not None:
                self._memory_con.close()
                self._memory_con = None
                self._schema_ready = False

    # -------------------------------------------------------------- recordings
    def add_recording(self, session: RecordingSession) -> int:
        """Insert a recording and return its new id (``session.id`` is ignored)."""
        created_at = session.created_at or datetime.now().strftime(_TIMESTAMP_FORMAT)
        with self._connect() as con:
            cursor = con.execute(
                """
                INSERT INTO recordings
                    (created_at, title, patient, notes, samplerate, duration, file_path, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    session.title,
                    session.patient,
                    session.notes,
                    int(session.sample_rate),
                    float(session.duration_s),
                    session.file_path,
                    session.source,
                ),
            )
            return int(cursor.lastrowid)

    def get_recording(self, id: int) -> RecordingSession | None:
        """Return the recording with the given id, or None."""
        with self._connect() as con:
            row = con.execute(f"{_SELECT_RECORDING} WHERE id = ?", (int(id),)).fetchone()
        return _to_session(row) if row is not None else None

    def list_recordings(self) -> list[RecordingSession]:
        """Return all recordings, newest first."""
        with self._connect() as con:
            rows = con.execute(f"{_SELECT_RECORDING} ORDER BY created_at DESC, id DESC").fetchall()
        return [_to_session(row) for row in rows]

    def delete_recording(self, id: int) -> bool:
        """Delete a recording together with its metrics (the audio file is untouched)."""
        with self._connect() as con:
            con.execute("DELETE FROM metrics WHERE recording_id = ?", (int(id),))
            cursor = con.execute("DELETE FROM recordings WHERE id = ?", (int(id),))
            return cursor.rowcount > 0

    # ----------------------------------------------------------------- metrics
    def save_metrics(self, recording_id: int, metrics: dict[str, str]) -> None:
        """Insert or update metrics for a recording.

        Raises:
            sqlite3.IntegrityError: the recording does not exist.
        """
        if not metrics:
            return
        with self._connect() as con:
            con.executemany(
                """
                INSERT INTO metrics (recording_id, name, value)
                VALUES (?, ?, ?)
                ON CONFLICT(recording_id, name) DO UPDATE SET value = excluded.value
                """,
                [(int(recording_id), str(name), str(value)) for name, value in metrics.items()],
            )

    def get_metrics(self, recording_id: int) -> dict[str, str]:
        """Return all stored metrics for a recording in insertion order."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT name, value FROM metrics WHERE recording_id = ? ORDER BY rowid",
                (int(recording_id),),
            ).fetchall()
        return {str(row["name"]): str(row["value"]) for row in rows}

    # ---------------------------------------------------------------- settings
    def get_setting(self, key: str, default: str = "") -> str:
        """Return a setting value, or ``default`` if unset."""
        with self._connect() as con:
            row = con.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row is not None else default

    def set_setting(self, key: str, value: str) -> None:
        """Insert or update a setting."""
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )


__all__: list[str] = ["SqliteRepository", "DEFAULT_DB_PATH", "DEFAULT_APP_DIR", "MEMORY_DB"]
