"""Recording workflow orchestrator: capture, live monitoring, and saving.

Threading contract
------------------
* Public methods (``start_*``, ``pause``, ``resume``, ``stop_and_save``) must be
  called from the thread that owns the service (normally the GUI thread), because
  they start and stop its ``QTimer``.
* Signals are safe to emit from any thread.  Engine error callbacks arrive on
  capture worker threads and do nothing but emit ``error_occurred``; Qt queues the
  delivery to receivers living in other threads.
* ``_live_tick`` runs on the owning thread and only reads non-consuming
  snapshots, so it never blocks capture.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal

from app.constants import DEFAULT_BLOCK_SIZE, LIVE_UPDATE_INTERVAL_MS
from app.core.dsp.live_spectrum import (
    LiveSpectrum,
    MIN_SPECTRUM_SAMPLES,
    amplitude_to_dbfs,
    level_stats,
    window_sizes,
)
from app.core.interfaces.audio_engine import IAudioCaptureEngine
from app.core.interfaces.serial_engine import ISerialEngine
from app.core.models.recording_session import RecordingSession
from app.services.project_manager import ProjectManager
from app.services.settings_service import SettingsService

SOURCE_MICROPHONE = "microphone"
SOURCE_SERIAL = "serial"

SERIAL_FULL_SCALE: float = 2048.0  # ADC counts that map to 1.0 (centered 12-bit)
DEFAULT_TITLE: str = "Recording"


@dataclass(frozen=True)
class _Pending:
    """A finished capture that has not been saved yet."""

    data: np.ndarray
    sample_rate: int
    source: str
    info: str


class RecordingService(QObject):
    """Coordinates capture engines, live monitoring, and saving of recordings."""

    recording_started = Signal(int)  # sample_rate
    recording_paused = Signal()
    recording_resumed = Signal()
    recording_stopped = Signal(object)  # RecordingSession
    level_updated = Signal(float, float)  # peak_dbfs, rms_dbfs
    live_waveform = Signal(object)  # np.ndarray
    live_spectrum = Signal(object, object, object)  # freqs, magnitudes, peak_hz | None
    error_occurred = Signal(str)

    def __init__(
        self,
        audio_engine: IAudioCaptureEngine,
        serial_engine: ISerialEngine,
        project_manager: ProjectManager,
        settings: SettingsService,
        parent: QObject | None = None,
        *,
        serial_full_scale: float = SERIAL_FULL_SCALE,
    ) -> None:
        """
        Args:
            audio_engine: Microphone capture engine.
            serial_engine: Serial / ADC capture engine.
            project_manager: Saves finished recordings.
            settings: Remembers the last used devices and rates.
            parent: Optional Qt parent.
            serial_full_scale: Serial counts that correspond to 1.0 in all live
                displays and in the saved WAV (default 2048, a centered 12-bit ADC).
        """
        super().__init__(parent)
        if serial_full_scale <= 0:
            raise ValueError(f"serial_full_scale must be positive, got {serial_full_scale}")
        self._audio = audio_engine
        self._serial = serial_engine
        self._project_manager = project_manager
        self._settings = settings
        self._serial_full_scale = float(serial_full_scale)

        self._active: IAudioCaptureEngine | ISerialEngine | None = None
        self._source: str | None = None
        self._sample_rate = 0
        self._paused = False
        self._serial_context = ""
        self._error_reported = False
        self._pending: _Pending | None = None
        self._live_spectrum = LiveSpectrum()
        self._last_tick_ms = 0.0

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)  # coarse timers drift to ~47 ms on Windows
        self._timer.setInterval(LIVE_UPDATE_INTERVAL_MS)
        self._timer.timeout.connect(self._live_tick)

        for engine in (audio_engine, serial_engine):
            register = getattr(engine, "set_error_callback", None)
            if callable(register):
                register(self._on_engine_error)

    # ---------------------------------------------------------- properties
    @property
    def is_recording(self) -> bool:
        """True from a successful ``start_*`` until ``stop_and_save`` (paused included)."""
        return self._active is not None

    @property
    def is_paused(self) -> bool:
        """True while the active recording is paused."""
        return self._active is not None and self._paused

    @property
    def source(self) -> str | None:
        """``"microphone"``, ``"serial"``, or None when idle."""
        return self._source

    @property
    def sample_rate(self) -> int:
        """Sample rate of the active (or last started) recording in Hz."""
        return self._sample_rate

    @property
    def has_unsaved_recording(self) -> bool:
        """True if a finished capture could not be saved and is waiting for a retry."""
        return self._pending is not None

    @property
    def last_tick_ms(self) -> float:
        """Duration of the most recent live-monitoring tick in milliseconds."""
        return self._last_tick_ms

    # ------------------------------------------------------------- errors
    def _on_engine_error(self, error: Exception) -> None:
        """Engine error callback; may run on a capture worker thread."""
        self._error_reported = True
        self.error_occurred.emit(str(error))

    def _can_start(self) -> bool:
        if self._active is not None:
            self.error_occurred.emit("Запись уже идёт.")
            return False
        if self._pending is not None:
            self.error_occurred.emit(
                "Предыдущая запись не сохранена. Повторите сохранение "
                "или отбросьте её."
            )
            return False
        return True

    def _remember(self, **values: object) -> None:
        """Persist last-used settings; a settings failure must not stop recording."""
        try:
            for name, value in values.items():
                setattr(self._settings, name, value)
        except Exception as exc:
            self.error_occurred.emit(f"Не удалось сохранить настройки: {exc}")

    # -------------------------------------------------------------- start
    def start_audio(self, device_index: int | None, sample_rate: int, channel: int = 1) -> bool:
        """Start microphone capture and live monitoring.

        ``channel`` is the input channel to record, counted from 1; the recording is mono.

        Returns:
            True if capture started; otherwise False with ``error_occurred`` emitted.
        """
        if not self._can_start():
            return False
        try:
            self._audio.start(
                device_index, int(sample_rate), 1, "float32", DEFAULT_BLOCK_SIZE, channel=max(1, int(channel)) - 1
            )
        except Exception as exc:
            self.error_occurred.emit(f"Не удалось начать запись звука: {exc}")
            return False

        self._remember(
            last_source=SOURCE_MICROPHONE,
            last_device_index=device_index,
            last_sample_rate=int(sample_rate),
            last_channel=max(1, int(channel)),
        )
        self._serial_context = ""
        self._begin(self._audio, SOURCE_MICROPHONE, int(self._audio.sample_rate) or int(sample_rate))
        return True

    def start_serial(self, port: str, baudrate: int, analysis_rate: int) -> bool:
        """Start serial capture and live monitoring.

        Args:
            port: Serial port name.
            baudrate: Port speed.
            analysis_rate: Sample rate in Hz assumed for analysis, display, and the
                saved file (the port itself has no fixed sample rate).

        Returns:
            True if capture started; otherwise False with ``error_occurred`` emitted.
        """
        if not self._can_start():
            return False
        if int(analysis_rate) <= 0 or int(baudrate) <= 0:
            self.error_occurred.emit("Скорость и частота анализа должны быть положительными.")
            return False
        try:
            self._serial.start(port, int(baudrate))
        except Exception as exc:
            self.error_occurred.emit(f"Не удалось начать приём из COM-порта: {exc}")
            return False

        self._remember(
            last_source=SOURCE_SERIAL,
            last_serial_port=port,
            last_baudrate=int(baudrate),
            last_serial_analysis_rate=int(analysis_rate),
        )
        self._serial_context = f"{port}, {int(baudrate)} baud; analysis rate {int(analysis_rate)} Hz"
        self._begin(self._serial, SOURCE_SERIAL, int(analysis_rate))
        return True

    def _begin(
        self,
        engine: IAudioCaptureEngine | ISerialEngine,
        source: str,
        sample_rate: int,
    ) -> None:
        self._active = engine
        self._source = source
        self._sample_rate = sample_rate
        self._paused = False
        self._error_reported = False
        self._timer.start()
        self.recording_started.emit(sample_rate)

    # ------------------------------------------------------ pause / resume
    def pause(self) -> None:
        """Pause capture and live monitoring (no-op if idle or already paused)."""
        if self._active is None or self._paused:
            return
        try:
            self._active.pause()
        except Exception as exc:
            self.error_occurred.emit(f"Не удалось поставить на паузу: {exc}")
            return
        self._paused = True
        self._timer.stop()
        self.recording_paused.emit()

    def resume(self) -> None:
        """Resume capture and live monitoring (no-op if not paused)."""
        if self._active is None or not self._paused:
            return
        try:
            self._active.resume()
        except Exception as exc:
            self.error_occurred.emit(f"Не удалось продолжить запись: {exc}")
            return
        self._paused = False
        self._timer.start()
        self.recording_resumed.emit()

    # ----------------------------------------------------- stop and save
    def stop_and_save(
        self,
        title: str = "",
        patient: str = "",
        notes: str = "",
    ) -> RecordingSession | None:
        """Stop capture, save it through the ``ProjectManager``, and emit ``recording_stopped``.

        If a previous save failed, calling this again (while idle) retries saving the
        captured data with the new metadata.  Stopping can block briefly (up to ~2 s)
        while the capture worker threads finish.

        Returns:
            The saved ``RecordingSession``, or None if nothing was saved (the reason
            is reported through ``error_occurred``).
        """
        engine = self._active
        if engine is None:
            if self._pending is None:
                self.error_occurred.emit("Запись не идёт.")
                return None
            return self._save_pending(title, patient, notes)

        source = self._source or SOURCE_MICROPHONE
        rate = self._sample_rate
        self._timer.stop()
        try:
            raw = np.asarray(engine.stop())
            info = self._stop_info(engine, source, rate)
        except Exception as exc:
            self._reset()
            self.error_occurred.emit(f"Ошибка при остановке записи: {exc}")
            return None
        self._reset()

        x = raw if raw.ndim <= 1 else raw[:, 0]
        x = x.astype(np.float32, copy=False)
        if source == SOURCE_SERIAL:
            x = (x / np.float32(self._serial_full_scale)).astype(np.float32, copy=False)

        min_samples = 3 if source == SOURCE_SERIAL else max(256, rate // 20)
        if x.size < min_samples:
            self.error_occurred.emit(
                f"Записано слишком мало данных (отсчётов: {x.size}), запись не сохранена."
            )
            return None

        self._pending = _Pending(x, rate, source, info)
        return self._save_pending(title, patient, notes)

    def abort(self) -> None:
        """Stop capturing and throw the audio away (used when the application quits).

        Does nothing if no recording is running.  A capture that could not be saved
        is discarded too.
        """
        engine = self._active
        self._timer.stop()
        if engine is not None:
            try:
                engine.stop()
            except Exception as exc:
                self.error_occurred.emit(f"Ошибка при остановке записи: {exc}")
        self._reset()
        self._pending = None

    def discard_unsaved(self) -> None:
        """Drop a finished capture that could not be saved."""
        self._pending = None

    def _stop_info(
        self,
        engine: IAudioCaptureEngine | ISerialEngine,
        source: str,
        rate: int,
    ) -> str:
        if source != SOURCE_SERIAL:
            return ""
        measured = int(getattr(engine, "actual_sample_rate", 0))
        return f"Serial: {self._serial_context}; measured ~{measured} Hz"

    def _save_pending(self, title: str, patient: str, notes: str) -> RecordingSession | None:
        pending = self._pending
        if pending is None:
            return None
        full_notes = "\n".join(part for part in (notes.strip(), pending.info) if part)
        try:
            session = self._project_manager.save_recording(
                pending.data,
                pending.sample_rate,
                title.strip() or DEFAULT_TITLE,
                patient,
                full_notes,
                pending.source,
            )
        except Exception as exc:
            self.error_occurred.emit(
                f"Не удалось сохранить запись: {exc}. Записанные данные остались в памяти; "
                "повторите сохранение."
            )
            return None
        self._pending = None
        self.recording_stopped.emit(session)
        return session

    def _reset(self) -> None:
        self._active = None
        self._source = None
        self._paused = False
        self._serial_context = ""

    # --------------------------------------------------------- live monitor
    def _live_tick(self) -> None:
        """Timer callback: read a snapshot, emit levels, waveform, and spectrum."""
        engine = self._active
        if engine is None or self._paused:
            return

        started = time.perf_counter()
        if not engine.is_recording:
            # The device/port died; keep the captured data for stop_and_save().
            self._timer.stop()
            if not self._error_reported:
                self._error_reported = True
                self.error_occurred.emit(
                    "Запись неожиданно прервалась. Сохраните уже записанные данные."
                )
            return

        fs = self._sample_rate
        n_spec, n_wave = window_sizes(fs)
        raw = engine.snapshot(n_wave)
        if raw.size < MIN_SPECTRUM_SAMPLES:
            return

        x = raw if raw.ndim == 1 else raw[:, 0]
        if self._source == SOURCE_SERIAL:
            x = x / np.float32(self._serial_full_scale)

        tail = x[-n_spec:]
        peak, rms = level_stats(tail)
        self.level_updated.emit(amplitude_to_dbfs(peak), amplitude_to_dbfs(rms))
        self.live_waveform.emit(x)

        live = self._live_spectrum.compute(tail, fs, rms)
        if live is not None:
            self.live_spectrum.emit(*live)

        self._last_tick_ms = (time.perf_counter() - started) * 1000.0


__all__: list[str] = ["RecordingService"]
