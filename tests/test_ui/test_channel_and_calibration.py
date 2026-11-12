"""Channel selection through the service and view model, and the calibration setting."""

from __future__ import annotations

from collections.abc import Iterator
from unittest import mock

import pytest
from PySide6.QtWidgets import QApplication

from app.infrastructure.storage.sqlite_repository import MEMORY_DB, SqliteRepository
from app.services.recording_service import RecordingService
from app.services.settings_service import SettingsService
from app.ui.dialogs.settings_dialog import SettingsDialog
from app.ui.viewmodels.recording_vm import RecordingViewModel


@pytest.fixture
def settings() -> Iterator[SettingsService]:
    repo = SqliteRepository(MEMORY_DB)
    repo.ensure_schema()
    yield SettingsService(repo)
    repo.close()


@pytest.fixture
def audio() -> mock.MagicMock:
    engine = mock.MagicMock()
    engine.sample_rate = 48000
    return engine


@pytest.fixture
def service(qapp: QApplication, audio: mock.MagicMock, settings: SettingsService) -> RecordingService:
    return RecordingService(audio, mock.MagicMock(), mock.MagicMock(), settings)


# -------------------------------------------------------------------- service
def test_service_records_the_chosen_channel_zero_based(service: RecordingService, audio: mock.MagicMock) -> None:
    assert service.start_audio(None, 48000, channel=2)

    assert audio.start.call_args.kwargs["channel"] == 1
    service.abort()


def test_service_defaults_to_the_first_channel(service: RecordingService, audio: mock.MagicMock) -> None:
    assert service.start_audio(None, 48000)

    assert audio.start.call_args.kwargs["channel"] == 0
    service.abort()


def test_service_remembers_the_channel(
    service: RecordingService, settings: SettingsService
) -> None:
    assert service.start_audio(None, 48000, channel=3)

    assert settings.last_channel == 3
    service.abort()


def test_failed_start_does_not_remember_the_channel(
    service: RecordingService, audio: mock.MagicMock, settings: SettingsService
) -> None:
    audio.start.side_effect = RuntimeError("busy")
    errors: list[str] = []
    service.error_occurred.connect(errors.append)

    assert not service.start_audio(None, 48000, channel=4)

    assert settings.last_channel == 1
    assert errors


# ----------------------------------------------------------------- view model
@pytest.fixture
def view_model(service: RecordingService, settings: SettingsService) -> RecordingViewModel:
    return RecordingViewModel(service, settings)


def test_view_model_starts_with_the_saved_channel(
    service: RecordingService, settings: SettingsService
) -> None:
    settings.last_channel = 2

    assert RecordingViewModel(service, settings).channel == 2


def test_view_model_channel_emits_a_change_once(view_model: RecordingViewModel) -> None:
    seen: list[int] = []
    view_model.channel_changed.connect(seen.append)

    view_model.channel = 2
    view_model.channel = 2

    assert seen == [2]


def test_view_model_rejects_channel_below_one(view_model: RecordingViewModel) -> None:
    with pytest.raises(ValueError, match="channel"):
        view_model.channel = 0


def test_view_model_passes_the_channel_to_the_service(
    view_model: RecordingViewModel, audio: mock.MagicMock
) -> None:
    view_model.channel = 2

    view_model.on_start_recording()

    assert audio.start.call_args.kwargs["channel"] == 1
    view_model._service.abort()


# ------------------------------------------------------------ settings dialog
def test_dialog_shows_and_saves_the_calibration(qapp: QApplication, settings: SettingsService) -> None:
    settings.calibration_v_per_fs = 1.25
    dialog = SettingsDialog(settings)
    assert dialog.calibration_spin.value() == 1.25

    dialog.calibration_spin.setValue(3.3)
    dialog.apply()

    assert settings.calibration_v_per_fs == 3.3


def test_dialog_zero_means_not_calibrated(qapp: QApplication, settings: SettingsService) -> None:
    settings.calibration_v_per_fs = 1.25
    dialog = SettingsDialog(settings)

    dialog.calibration_spin.setValue(0.0)
    dialog.apply()

    assert settings.calibration_v_per_fs == 0.0
    assert dialog.calibration_spin.specialValueText() == "не задана"
