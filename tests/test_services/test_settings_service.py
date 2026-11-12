"""Tests for the calibration and channel settings of SettingsService."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.infrastructure.storage.sqlite_repository import MEMORY_DB, SqliteRepository
from app.services.settings_service import KEY_CALIBRATION_V_PER_FS, SettingsService


@pytest.fixture
def repo() -> Iterator[SqliteRepository]:
    repository = SqliteRepository(MEMORY_DB)
    repository.ensure_schema()
    yield repository
    repository.close()


@pytest.fixture
def settings(repo: SqliteRepository) -> SettingsService:
    return SettingsService(repo)


# --------------------------------------------------------------- calibration
def test_calibration_defaults_to_not_calibrated(settings: SettingsService) -> None:
    assert settings.calibration_v_per_fs == 0.0


def test_calibration_round_trips(settings: SettingsService) -> None:
    settings.calibration_v_per_fs = 2.5

    assert settings.calibration_v_per_fs == 2.5


def test_calibration_zero_clears_it(settings: SettingsService) -> None:
    settings.calibration_v_per_fs = 2.5
    settings.calibration_v_per_fs = 0.0

    assert settings.calibration_v_per_fs == 0.0


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_invalid_calibration_is_rejected(settings: SettingsService, bad: float) -> None:
    with pytest.raises(ValueError, match="calibration_v_per_fs"):
        settings.calibration_v_per_fs = bad


@pytest.mark.parametrize("stored", ["abc", "-3", "nan", "0"])
def test_corrupt_stored_calibration_reads_as_not_calibrated(
    repo: SqliteRepository, settings: SettingsService, stored: str
) -> None:
    repo.set_setting(KEY_CALIBRATION_V_PER_FS, stored)

    assert settings.calibration_v_per_fs == 0.0


# -------------------------------------------------------------------- channel
def test_channel_defaults_to_the_first(settings: SettingsService) -> None:
    assert settings.last_channel == 1


def test_channel_round_trips(settings: SettingsService) -> None:
    settings.last_channel = 3

    assert settings.last_channel == 3


@pytest.mark.parametrize("bad", [0, -1])
def test_channel_must_be_positive(settings: SettingsService, bad: int) -> None:
    with pytest.raises(ValueError, match="last_channel"):
        settings.last_channel = bad
