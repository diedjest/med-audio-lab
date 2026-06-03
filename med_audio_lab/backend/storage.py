from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

# Импортируем из нашего нового пакета утилит
from ..utils.helpers import slugify

APP_DIR = Path.home() / "Documents" / "MedAudioLab"
RECORDS_DIR = APP_DIR / "records"
EXPORTS_DIR = APP_DIR / "exports"
DB_PATH = APP_DIR / "med_audio_lab.sqlite3"

@dataclass
class RecordingInfo:
    id: int
    created_at: str
    title: str
    patient: str
    notes: str
    samplerate: int
    duration: float
    file_path: str

def ensure_storage() -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                title TEXT NOT NULL,
                patient TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                samplerate INTEGER NOT NULL,
                duration REAL NOT NULL,
                file_path TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                recording_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (recording_id, name),
                FOREIGN KEY(recording_id) REFERENCES recordings(id)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

def get_setting(key: str, default: str = "") -> str:
    ensure_storage()
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return str(row[0]) if row else default

def set_setting(key: str, value: str) -> None:
    ensure_storage()
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            INSERT INTO settings(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, value),
        )

def get_save_dir() -> Path:
    saved = get_setting("save_dir", str(RECORDS_DIR))
    path = Path(saved).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_wav(data: np.ndarray, samplerate: int, title: str, directory: str | Path | None = None) -> Path:
    ensure_storage()
    target_dir = Path(directory) if directory else get_save_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{slugify(title)}.wav"
    path = target_dir / name
    clipped = np.clip(np.asarray(data, dtype=np.float32), -1.0, 1.0)
    sf.write(path, clipped, int(samplerate), subtype="PCM_16")
    return path

def save_signal_csv(
    data: np.ndarray,
    samplerate: int,
    title: str,
    directory: str | Path | None = None,
    suffix: str = "samples",
) -> Path:
    ensure_storage()
    target_dir = Path(directory) if directory else get_save_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = target_dir / f"{stamp}_{slugify(title)}_{suffix}.csv"
    values = np.asarray(data, dtype=np.float32)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["index", "time_s", "value"])
        for i, value in enumerate(values):
            writer.writerow([i, f"{i / max(1, samplerate):.9f}", f"{float(value):.9g}"])
    return path

def add_recording(
    path: Path,
    title: str,
    patient: str,
    notes: str,
    samplerate: int,
    duration: float,
) -> int:
    ensure_storage()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            """
            INSERT INTO recordings
                (created_at, title, patient, notes, samplerate, duration, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (created_at, title, patient, notes, int(samplerate), float(duration), str(path)),
        )
        return int(cur.lastrowid)

def save_metrics(recording_id: int, metrics: dict[str, str]) -> None:
    ensure_storage()
    with sqlite3.connect(DB_PATH) as con:
        con.executemany(
            """
            INSERT INTO metrics(recording_id, name, value)
            VALUES (?, ?, ?)
            ON CONFLICT(recording_id, name) DO UPDATE SET value=excluded.value
            """,
            [(recording_id, key, str(value)) for key, value in metrics.items()],
        )

def list_recordings() -> list[RecordingInfo]:
    ensure_storage()
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT id, created_at, title, patient, notes, samplerate, duration, file_path
            FROM recordings
            ORDER BY id DESC
            """
        ).fetchall()
    return [RecordingInfo(**dict(row)) for row in rows]

def get_recording(recording_id: int) -> RecordingInfo | None:
    ensure_storage()
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            """
            SELECT id, created_at, title, patient, notes, samplerate, duration, file_path
            FROM recordings
            WHERE id = ?
            """,
            (recording_id,),
        ).fetchone()
    return RecordingInfo(**dict(row)) if row else None

def export_report_csv(recording: RecordingInfo, metrics: dict[str, str]) -> Path:
    ensure_storage()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = EXPORTS_DIR / f"{stamp}_{slugify(recording.title)}_report.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Параметр", "Значение"])
        writer.writerow(["ID", recording.id])
        writer.writerow(["Дата", recording.created_at])
        writer.writerow(["Название", recording.title])
        writer.writerow(["Пациент/образец", recording.patient])
        writer.writerow(["Комментарий", recording.notes])
        writer.writerow(["Частота дискретизации", recording.samplerate])
        writer.writerow(["Длительность, с", f"{recording.duration:.3f}"])
        writer.writerow(["Файл", recording.file_path])
        writer.writerow([])
        for key, value in metrics.items():
            writer.writerow([key, value])
    return path