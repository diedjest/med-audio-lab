"""Reactive state and commands for the recording tab.

All state is exposed as Qt properties with notify signals, so Python code can
connect to ``<name>_changed`` and QML / property bindings work too.  Everything
runs on the GUI thread; the service signals it listens to are delivered there.
"""

from __future__ import annotations

from PySide6.QtCore import Property, QObject, Signal, Slot

from app.core.dsp.live_spectrum import DB_FLOOR
from app.core.models.device_info import AudioDeviceInfo, SerialPortInfo
from app.core.models.recording_session import RecordingSession
from app.services.recording_service import (
    SOURCE_MICROPHONE,
    SOURCE_SERIAL,
    RecordingService,
)
from app.services.settings_service import VALID_SOURCES, SettingsService

STATUS_READY = "Готово"
LABEL_DEFAULT_INPUT = "Устройство ввода по умолчанию"
LABEL_NO_PORT = "COM-порт не выбран"


class RecordingViewModel(QObject):
    """Wraps :class:`RecordingService` for the UI.

    Observable state (read-only properties):
    ``is_recording``, ``is_paused``, ``current_source``, ``selected_device_label``,
    ``status_text``, ``peak_dbfs``, ``rms_dbfs`` and ``has_unsaved_recording``.

    Editable inputs (properties with setters): ``title``, ``patient``, ``notes``,
    ``sample_rate``, ``baudrate`` and ``serial_analysis_rate``.

    Commands (slots): ``on_start_recording``, ``on_pause``, ``on_resume``,
    ``on_stop``, ``on_source_changed``, ``on_device_selected`` and
    ``on_discard_unsaved``.

    Service signals ``recording_started``, ``recording_stopped``, ``live_waveform``,
    ``live_spectrum`` and ``error_occurred`` are forwarded unchanged so views only
    need to know this object.
    """

    # --- property notify signals
    is_recording_changed = Signal(bool)
    is_paused_changed = Signal(bool)
    current_source_changed = Signal(str)
    selected_device_label_changed = Signal(str)
    status_text_changed = Signal(str)
    peak_dbfs_changed = Signal(float)
    rms_dbfs_changed = Signal(float)
    has_unsaved_recording_changed = Signal(bool)
    title_changed = Signal(str)
    patient_changed = Signal(str)
    notes_changed = Signal(str)
    sample_rate_changed = Signal(int)
    channel_changed = Signal(int)
    baudrate_changed = Signal(int)
    serial_analysis_rate_changed = Signal(int)

    # --- forwarded service signals
    recording_started = Signal(int)
    recording_stopped = Signal(object)  # RecordingSession
    live_waveform = Signal(object)
    live_spectrum = Signal(object, object, object)
    error_occurred = Signal(str)

    def __init__(
        self,
        service: RecordingService,
        settings: SettingsService,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._settings = settings

        self._is_recording = service.is_recording
        self._is_paused = service.is_paused
        self._has_unsaved = service.has_unsaved_recording
        self._current_source = settings.last_source
        self._status_text = STATUS_READY
        self._peak_dbfs = DB_FLOOR
        self._rms_dbfs = DB_FLOOR
        self._title = ""
        self._patient = ""
        self._notes = ""

        self._device_index: int | None = settings.last_device_index
        self._audio_device: AudioDeviceInfo | None = None
        self._serial_port = settings.last_serial_port
        self._serial_info: SerialPortInfo | None = None
        self._sample_rate = settings.last_sample_rate
        self._channel = settings.last_channel
        self._baudrate = settings.last_baudrate
        self._analysis_rate = settings.last_serial_analysis_rate
        self._device_label = self._compute_label()

        service.recording_started.connect(self._on_started)
        service.recording_paused.connect(self._on_paused)
        service.recording_resumed.connect(self._on_resumed)
        service.recording_stopped.connect(self._on_stopped)
        service.level_updated.connect(self._on_levels)
        service.error_occurred.connect(self._on_error)
        service.live_waveform.connect(self.live_waveform)
        service.live_spectrum.connect(self.live_spectrum)

    # ------------------------------------------------------------ helpers
    def _compute_label(self) -> str:
        if self._current_source == SOURCE_SERIAL:
            if self._serial_info is not None:
                return self._serial_info.label
            return self._serial_port or LABEL_NO_PORT
        if self._audio_device is not None:
            return self._audio_device.label
        if self._device_index is None:
            return LABEL_DEFAULT_INPUT
        return f"Устройство {self._device_index}"

    def _set_status(self, text: str) -> None:
        if text != self._status_text:
            self._status_text = text
            self.status_text_changed.emit(text)

    def _set_levels(self, peak: float, rms: float) -> None:
        if peak != self._peak_dbfs:
            self._peak_dbfs = peak
            self.peak_dbfs_changed.emit(peak)
        if rms != self._rms_dbfs:
            self._rms_dbfs = rms
            self.rms_dbfs_changed.emit(rms)

    def _refresh_label(self) -> None:
        label = self._compute_label()
        if label != self._device_label:
            self._device_label = label
            self.selected_device_label_changed.emit(label)

    def _sync_state(self) -> None:
        """Pull recording / paused / unsaved flags from the service."""
        recording, paused = self._service.is_recording, self._service.is_paused
        unsaved = self._service.has_unsaved_recording
        if recording != self._is_recording:
            self._is_recording = recording
            self.is_recording_changed.emit(recording)
        if paused != self._is_paused:
            self._is_paused = paused
            self.is_paused_changed.emit(paused)
        if unsaved != self._has_unsaved:
            self._has_unsaved = unsaved
            self.has_unsaved_recording_changed.emit(unsaved)
        if not recording:
            self._set_levels(DB_FLOOR, DB_FLOOR)

    def _fail(self, message: str) -> None:
        """Report a problem detected by the view model itself."""
        self._set_status(f"Ошибка: {message}")
        self.error_occurred.emit(message)

    def _remember(self, **values: object) -> None:
        try:
            for name, value in values.items():
                setattr(self._settings, name, value)
        except Exception as exc:  # settings are a convenience; never block the UI
            self._fail(f"Не удалось сохранить настройки: {exc}")

    def _set_input(self, attr: str, value: object, signal: Signal) -> None:
        if getattr(self, attr) != value:
            setattr(self, attr, value)
            signal.emit(value)

    # ------------------------------------------------- read-only properties
    @Property(bool, notify=is_recording_changed)
    def is_recording(self) -> bool:
        return self._is_recording

    @Property(bool, notify=is_paused_changed)
    def is_paused(self) -> bool:
        return self._is_paused

    @Property(str, notify=current_source_changed)
    def current_source(self) -> str:
        return self._current_source

    @Property(str, notify=selected_device_label_changed)
    def selected_device_label(self) -> str:
        return self._device_label

    @Property(str, notify=status_text_changed)
    def status_text(self) -> str:
        return self._status_text

    @Property(float, notify=peak_dbfs_changed)
    def peak_dbfs(self) -> float:
        return self._peak_dbfs

    @Property(float, notify=rms_dbfs_changed)
    def rms_dbfs(self) -> float:
        return self._rms_dbfs

    @Property(bool, notify=has_unsaved_recording_changed)
    def has_unsaved_recording(self) -> bool:
        """True if a finished capture could not be saved and awaits a retry."""
        return self._has_unsaved

    # --------------------------------------------------- editable properties
    @Property(str, notify=title_changed)
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, value: str) -> None:
        self._set_input("_title", str(value), self.title_changed)

    @Property(str, notify=patient_changed)
    def patient(self) -> str:
        return self._patient

    @patient.setter
    def patient(self, value: str) -> None:
        self._set_input("_patient", str(value), self.patient_changed)

    @Property(str, notify=notes_changed)
    def notes(self) -> str:
        return self._notes

    @notes.setter
    def notes(self, value: str) -> None:
        self._set_input("_notes", str(value), self.notes_changed)

    @Property(int, notify=sample_rate_changed)
    def sample_rate(self) -> int:
        """Microphone sample rate in Hz used by the next recording."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int) -> None:
        if int(value) <= 0:
            raise ValueError(f"sample_rate must be positive, got {value}")
        self._set_input("_sample_rate", int(value), self.sample_rate_changed)

    @Property(int, notify=channel_changed)
    def channel(self) -> int:
        """Microphone input channel (counted from 1) recorded by the next recording."""
        return self._channel

    @channel.setter
    def channel(self, value: int) -> None:
        if int(value) < 1:
            raise ValueError(f"channel must be >= 1, got {value}")
        self._set_input("_channel", int(value), self.channel_changed)

    @Property(int, notify=baudrate_changed)
    def baudrate(self) -> int:
        """Serial baud rate used by the next serial recording."""
        return self._baudrate

    @baudrate.setter
    def baudrate(self, value: int) -> None:
        if int(value) <= 0:
            raise ValueError(f"baudrate must be positive, got {value}")
        self._set_input("_baudrate", int(value), self.baudrate_changed)

    @Property(int, notify=serial_analysis_rate_changed)
    def serial_analysis_rate(self) -> int:
        """Sample rate in Hz assumed for serial data."""
        return self._analysis_rate

    @serial_analysis_rate.setter
    def serial_analysis_rate(self, value: int) -> None:
        if int(value) <= 0:
            raise ValueError(f"serial_analysis_rate must be positive, got {value}")
        self._set_input("_analysis_rate", int(value), self.serial_analysis_rate_changed)

    # ---------------------------------------------------------------- slots
    @Slot()
    def on_start_recording(self) -> None:
        """Start capturing from the selected source and device."""
        if self._service.is_recording:
            self._fail("Запись уже идёт.")
            return
        if self._current_source == SOURCE_SERIAL:
            if not self._serial_port:
                self._fail("COM-порт не выбран.")
                return
            self._service.start_serial(self._serial_port, self._baudrate, self._analysis_rate)
        else:
            self._service.start_audio(self._device_index, self._sample_rate, self._channel)
        # state changes arrive through the service signals; failures through error_occurred

    @Slot()
    def on_pause(self) -> None:
        """Pause the running recording."""
        if self._service.is_recording:
            self._service.pause()

    @Slot()
    def on_resume(self) -> None:
        """Resume a paused recording."""
        if self._service.is_recording:
            self._service.resume()

    @Slot()
    def on_stop(self) -> None:
        """Stop and save with the current title / patient / notes.

        Also retries saving a previous capture that failed to save.  Blocks the GUI
        thread briefly (up to ~2 s) while the capture threads finish.
        """
        if not self._service.is_recording and not self._service.has_unsaved_recording:
            self._fail("Запись не идёт.")
            return
        self._service.stop_and_save(self._title, self._patient, self._notes)
        self._sync_state()

    @Slot()
    def on_discard_unsaved(self) -> None:
        """Drop a capture that could not be saved."""
        self._service.discard_unsaved()
        self._sync_state()
        self._set_status("Несохранённая запись отброшена.")

    @Slot()
    def on_abort_recording(self) -> None:
        """Stop the running recording and discard its audio (nothing is saved)."""
        self._service.abort()
        self._sync_state()
        self._set_status("Запись отброшена.")

    @Slot(str)
    def on_source_changed(self, source: str) -> None:
        """Switch between ``"microphone"`` and ``"serial"`` (not while recording)."""
        if source not in VALID_SOURCES:
            self._fail(f"Неизвестный источник {source!r}.")
            return
        if self._service.is_recording:
            self._fail("Нельзя менять источник во время записи.")
            return
        if source == self._current_source:
            return
        self._current_source = source
        self.current_source_changed.emit(source)
        self._refresh_label()
        self._remember(last_source=source)
        self._set_status(STATUS_READY)

    @Slot(object)
    def on_device_selected(self, device: AudioDeviceInfo | SerialPortInfo | None) -> None:
        """Select the input device (microphone source) or port (serial source).

        Choosing an audio device also adopts its default sample rate.  ``None``
        resets to the default device / no port.  Not allowed while recording.
        """
        if self._service.is_recording:
            self._fail("Нельзя менять устройство во время записи.")
            return

        if device is None:
            if self._current_source == SOURCE_SERIAL:
                self._serial_port, self._serial_info = "", None
            else:
                self._device_index, self._audio_device = None, None
        elif isinstance(device, AudioDeviceInfo):
            if self._current_source != SOURCE_MICROPHONE:
                self._fail("Аудиоустройство нельзя использовать с источником COM-порт.")
                return
            self._device_index, self._audio_device = device.index, device
            if device.default_sample_rate > 0:
                self.sample_rate = device.default_sample_rate
        elif isinstance(device, SerialPortInfo):
            if self._current_source != SOURCE_SERIAL:
                self._fail("COM-порт нельзя использовать с источником «микрофон».")
                return
            self._serial_port, self._serial_info = device.device, device
        else:
            self._fail(f"Неподдерживаемый тип устройства: {type(device).__name__}")
            return
        self._refresh_label()

    # ----------------------------------------------------- service handlers
    def _on_started(self, sample_rate: int) -> None:
        self._sync_state()
        channel = f", канал {self._channel}" if self._current_source == SOURCE_MICROPHONE else ""
        self._set_status(f"Запись с устройства «{self._device_label}»{channel}, {sample_rate} Гц")
        self.recording_started.emit(sample_rate)

    def _on_paused(self) -> None:
        self._sync_state()
        self._set_status("Пауза")

    def _on_resumed(self) -> None:
        self._sync_state()
        self._set_status("Запись")

    def _on_stopped(self, session: RecordingSession) -> None:
        self._sync_state()
        self._set_status(f"Сохранено: «{session.title}» ({session.duration_s:.2f} с)")
        self.recording_stopped.emit(session)

    def _on_levels(self, peak_dbfs: float, rms_dbfs: float) -> None:
        self._set_levels(peak_dbfs, rms_dbfs)

    def _on_error(self, message: str) -> None:
        self._sync_state()
        self._set_status(f"Ошибка: {message}")
        self.error_occurred.emit(message)


__all__: list[str] = ["RecordingViewModel"]
