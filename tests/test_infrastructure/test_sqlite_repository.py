"""Tests for app.infrastructure.storage.sqlite_repository.SqliteRepository.

Most tests use a private in-memory database (``":memory:"``), so each test starts from
nothing and nothing touches the disk.  The few tests that are about files - directories,
persistence between instances, migration of an old database - use ``tmp_path``.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.core.models.recording_session import RecordingSession
from app.infrastructure.storage.sqlite_repository import MEMORY_DB, SqliteRepository


@pytest.fixture
def repo() -> Iterator[SqliteRepository]:
    repository = SqliteRepository(MEMORY_DB)
    repository.ensure_schema()
    yield repository
    repository.close()


def session(title: str = "Recording", **overrides: object) -> RecordingSession:
    values: dict[str, object] = dict(
        id=None, created_at="2026-01-01 10:00:00", title=title, patient="P-1", notes="a note",
        sample_rate=192_000, duration_s=2.5, file_path="C:/rec/a.wav", source="microphone",
    )
    values.update(overrides)
    return RecordingSession(**values)  # type: ignore[arg-type]


def table_names(repository: SqliteRepository) -> set[str]:
    with repository._connect() as con:
        return {row["name"] for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def column_names(repository: SqliteRepository, table: str) -> list[str]:
    with repository._connect() as con:
        return [row["name"] for row in con.execute(f"PRAGMA table_info({table})")]


# -------------------------------------------------------------------- schema
def test_ensure_schema_creates_the_tables(repo: SqliteRepository) -> None:
    assert {"recordings", "metrics", "settings"} <= table_names(repo)


def test_recordings_table_has_every_column(repo: SqliteRepository) -> None:
    assert column_names(repo, "recordings") == [
        "id", "created_at", "title", "patient", "notes", "samplerate", "duration", "file_path", "source",
    ]
    assert column_names(repo, "metrics") == ["recording_id", "name", "value"]
    assert column_names(repo, "settings") == ["key", "value"]


def test_ensure_schema_is_idempotent_and_keeps_data(repo: SqliteRepository) -> None:
    rid = repo.add_recording(session())

    repo.ensure_schema()
    repo.ensure_schema()

    assert repo.get_recording(rid) is not None


def test_the_schema_is_created_on_first_use_without_asking(tmp_path: Path) -> None:
    fresh = SqliteRepository(MEMORY_DB)  # ensure_schema() never called

    assert fresh.list_recordings() == []
    assert fresh.get_setting("anything", "d") == "d"
    fresh.close()


def test_memory_databases_are_isolated_from_each_other() -> None:
    first, second = SqliteRepository(MEMORY_DB), SqliteRepository(MEMORY_DB)
    first.add_recording(session("only in the first"))

    assert len(first.list_recordings()) == 1
    assert second.list_recordings() == []
    first.close()
    second.close()


def test_memory_database_touches_no_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    repository = SqliteRepository(MEMORY_DB)

    repository.add_recording(session())
    repository.set_setting("k", "v")
    repository.close()

    assert os.listdir(tmp_path) == []  # in particular: no stray records/ or exports/ folders
    assert repository.is_memory


# ---------------------------------------------------- add_recording / get_recording
def test_add_and_get_recording_roundtrip(repo: SqliteRepository) -> None:
    original = session()

    rid = repo.add_recording(original)
    loaded = repo.get_recording(rid)

    assert rid == 1
    assert loaded == RecordingSession(**{**original.__dict__, "id": rid})


def test_ids_are_assigned_sequentially_and_the_given_id_is_ignored(repo: SqliteRepository) -> None:
    first = repo.add_recording(session("a", id=500))
    second = repo.add_recording(session("b"))

    assert (first, second) == (1, 2)


def test_get_unknown_recording_returns_none(repo: SqliteRepository) -> None:
    assert repo.get_recording(999) is None
    repo.add_recording(session())
    assert repo.get_recording(2) is None


def test_missing_timestamp_is_filled_in(repo: SqliteRepository) -> None:
    rid = repo.add_recording(session(created_at=""))

    stamp = repo.get_recording(rid).created_at  # type: ignore[union-attr]
    assert len(stamp) == 19 and stamp[4] == "-" and stamp[10] == " " and stamp[13] == ":"


@pytest.mark.parametrize("source", ["microphone", "serial"])
def test_source_is_stored(repo: SqliteRepository, source: str) -> None:
    rid = repo.add_recording(session(source=source))

    assert repo.get_recording(rid).source == source  # type: ignore[union-attr]


def test_awkward_text_is_stored_literally(repo: SqliteRepository) -> None:
    nasty = "Robert'); DROP TABLE recordings;--"
    rid = repo.add_recording(session(title=nasty, patient="Иванов Пётр", notes="line1\nline2\t\"quoted\"", file_path="D:\\a b\\ü.wav"))

    loaded = repo.get_recording(rid)

    assert loaded is not None
    assert (loaded.title, loaded.patient) == (nasty, "Иванов Пётр")
    assert loaded.notes == "line1\nline2\t\"quoted\"" and loaded.file_path == "D:\\a b\\ü.wav"
    assert "recordings" in table_names(repo)


def test_numbers_keep_their_types(repo: SqliteRepository) -> None:
    rid = repo.add_recording(session(sample_rate=768_000, duration_s=0.125))

    loaded = repo.get_recording(rid)

    assert loaded is not None
    assert loaded.sample_rate == 768_000 and isinstance(loaded.sample_rate, int)
    assert loaded.duration_s == 0.125 and isinstance(loaded.duration_s, float)


# ----------------------------------------------------------- list_recordings
def test_list_is_empty_at_first(repo: SqliteRepository) -> None:
    assert repo.list_recordings() == []


def test_list_is_newest_first(repo: SqliteRepository) -> None:
    repo.add_recording(session("middle", created_at="2026-02-01 00:00:00"))
    repo.add_recording(session("oldest", created_at="2025-12-31 23:59:59"))
    repo.add_recording(session("newest", created_at="2026-03-01 00:00:00"))

    assert [r.title for r in repo.list_recordings()] == ["newest", "middle", "oldest"]


def test_recordings_with_the_same_timestamp_list_latest_insert_first(repo: SqliteRepository) -> None:
    for title in ("first", "second", "third"):
        repo.add_recording(session(title, created_at="2026-01-01 10:00:00"))

    assert [r.title for r in repo.list_recordings()] == ["third", "second", "first"]


def test_list_returns_complete_sessions_with_ids(repo: SqliteRepository) -> None:
    ids = [repo.add_recording(session(f"r{i}", created_at=f"2026-01-0{i + 1} 00:00:00")) for i in range(3)]

    listed = repo.list_recordings()

    assert [r.id for r in listed] == ids[::-1]
    assert all(r.file_path and r.sample_rate and r.source for r in listed)


# ------------------------------------------------------------------ delete
def test_delete_removes_the_recording_and_its_metrics(repo: SqliteRepository) -> None:
    keep, drop = repo.add_recording(session("keep")), repo.add_recording(session("drop"))
    repo.save_metrics(keep, {"RMS": "1"})
    repo.save_metrics(drop, {"RMS": "2"})

    assert repo.delete_recording(drop) is True

    assert repo.get_recording(drop) is None and repo.get_metrics(drop) == {}
    assert repo.get_recording(keep) is not None and repo.get_metrics(keep) == {"RMS": "1"}


def test_deleting_an_unknown_recording_reports_false(repo: SqliteRepository) -> None:
    assert repo.delete_recording(42) is False


# ---------------------------------------------------------------- metrics
def test_metrics_roundtrip_and_empty(repo: SqliteRepository) -> None:
    rid = repo.add_recording(session())
    assert repo.get_metrics(rid) == {}

    repo.save_metrics(rid, {"RMS": "0.25", "Peak": "0.9", "Dominant frequency": "40000.0 Hz"})

    assert repo.get_metrics(rid) == {"RMS": "0.25", "Peak": "0.9", "Dominant frequency": "40000.0 Hz"}


def test_metrics_keep_their_insertion_order(repo: SqliteRepository) -> None:
    rid = repo.add_recording(session())
    names = ["Zeta", "Alpha", "Mid", "Beta"]

    repo.save_metrics(rid, {n: "1" for n in names})

    assert list(repo.get_metrics(rid)) == names


def test_saving_a_metric_again_updates_it(repo: SqliteRepository) -> None:
    rid = repo.add_recording(session())
    repo.save_metrics(rid, {"RMS": "0.1", "Peak": "0.9"})

    repo.save_metrics(rid, {"RMS": "0.2"})

    assert repo.get_metrics(rid) == {"RMS": "0.2", "Peak": "0.9"}


def test_metric_values_are_stored_as_text(repo: SqliteRepository) -> None:
    rid = repo.add_recording(session())

    repo.save_metrics(rid, {"Samples": 1234, "RMS": 0.5})  # type: ignore[dict-item]

    assert repo.get_metrics(rid) == {"Samples": "1234", "RMS": "0.5"}


def test_metrics_are_kept_per_recording(repo: SqliteRepository) -> None:
    a, b = repo.add_recording(session("a")), repo.add_recording(session("b"))

    repo.save_metrics(a, {"RMS": "1"})
    repo.save_metrics(b, {"RMS": "2"})

    assert repo.get_metrics(a) == {"RMS": "1"} and repo.get_metrics(b) == {"RMS": "2"}


def test_saving_no_metrics_is_a_no_op(repo: SqliteRepository) -> None:
    rid = repo.add_recording(session())

    repo.save_metrics(rid, {})

    assert repo.get_metrics(rid) == {}


def test_metrics_for_a_missing_recording_are_refused(repo: SqliteRepository) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        repo.save_metrics(777, {"RMS": "1"})

    assert repo.get_metrics(777) == {}  # and nothing was stored


# ----------------------------------------------------------------- settings
def test_setting_defaults(repo: SqliteRepository) -> None:
    assert repo.get_setting("missing") == ""
    assert repo.get_setting("missing", "fallback") == "fallback"


def test_setting_create_read_update(repo: SqliteRepository) -> None:
    repo.set_setting("save_dir", "C:/data")
    assert repo.get_setting("save_dir") == "C:/data"

    repo.set_setting("save_dir", "D:/other")

    assert repo.get_setting("save_dir") == "D:/other"


def test_an_empty_string_is_a_real_value_not_a_missing_one(repo: SqliteRepository) -> None:
    repo.set_setting("device", "")

    assert repo.get_setting("device", "default") == ""


def test_settings_are_independent_and_unicode_safe(repo: SqliteRepository) -> None:
    repo.set_setting("a", "1")
    repo.set_setting("b", "Привет, мир")

    assert (repo.get_setting("a"), repo.get_setting("b")) == ("1", "Привет, мир")


def test_settings_do_not_touch_recordings(repo: SqliteRepository) -> None:
    repo.set_setting("k", "v")

    assert repo.list_recordings() == []


# ------------------------------------------------------------------ threads
def test_the_repository_can_be_used_from_several_threads(repo: SqliteRepository) -> None:
    errors: list[BaseException] = []

    def work(worker: int) -> None:
        try:
            for i in range(10):
                rid = repo.add_recording(session(f"w{worker}-{i}"))
                repo.save_metrics(rid, {"worker": str(worker)})
                repo.set_setting(f"k{worker}", str(i))
                repo.list_recordings()
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    assert len(repo.list_recordings()) == 60
    assert sorted(r.id for r in repo.list_recordings()) == list(range(1, 61))


# -------------------------------------------------------- file-based databases
def test_file_database_creates_its_folders(tmp_path: Path) -> None:
    db = tmp_path / "deep" / "dir" / "lab.sqlite3"
    repository = SqliteRepository(db)

    repository.ensure_schema()

    assert db.is_file()
    assert repository.records_dir.is_dir() and repository.exports_dir.is_dir()
    assert repository.records_dir.parent == db.parent and not repository.is_memory


def test_file_database_survives_a_new_instance(tmp_path: Path) -> None:
    db = tmp_path / "lab.sqlite3"
    first = SqliteRepository(db)
    rid = first.add_recording(session("kept"))
    first.save_metrics(rid, {"RMS": "1"})
    first.set_setting("k", "v")

    second = SqliteRepository(db)

    assert second.get_recording(rid).title == "kept"  # type: ignore[union-attr]
    assert second.get_metrics(rid) == {"RMS": "1"} and second.get_setting("k") == "v"


def test_a_database_from_the_legacy_application_is_migrated(tmp_path: Path) -> None:
    db = tmp_path / "legacy.sqlite3"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE recordings (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, "
        "title TEXT NOT NULL, patient TEXT NOT NULL DEFAULT '', notes TEXT NOT NULL DEFAULT '', "
        "samplerate INTEGER NOT NULL, duration REAL NOT NULL, file_path TEXT NOT NULL)"
    )
    con.execute(
        "INSERT INTO recordings (created_at, title, samplerate, duration, file_path) "
        "VALUES ('2024-01-01 10:00:00', 'old recording', 44100, 1.5, 'a.wav')"
    )
    con.commit()
    con.close()

    repository = SqliteRepository(db)
    old = repository.list_recordings()

    assert len(old) == 1 and old[0].title == "old recording" and old[0].sample_rate == 44100
    assert old[0].source == "microphone"  # the column added by the migration
    new_id = repository.add_recording(session("new", source="serial"))
    assert repository.get_recording(new_id).source == "serial"  # type: ignore[union-attr]
