"""Live-monitoring thread: samples a capture engine ~30 times per second."""

from __future__ import annotations

import math
import threading
import time

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from app.constants import LIVE_UPDATE_INTERVAL_MS
from app.core.dsp.live_spectrum import (
    MIN_SPECTRUM_SAMPLES,
    LiveSpectrum,
    amplitude_to_dbfs,
    level_stats,
    window_sizes,
)
from app.core.interfaces.audio_engine import IAudioCaptureEngine

_DEFAULT_STOP_TIMEOUT_MS = 2000


class CaptureMonitorWorker(QThread):
    """Reads snapshots from a capture engine and emits them for live display.

    Each tick (every 33 ms, catching up at most one tick when it falls behind)
    takes one non-consuming snapshot of the latest ~0.25 s, then computes peak/RMS
    over the last 80 ms and a windowed FFT spectrum in ``[5 kHz, Nyquist / 2]``.
    Nothing is emitted while the engine is idle, paused, or has lost its device.

    Signals (delivered to receivers in their own threads via Qt's queued
    connections):

    * ``waveform_ready(ndarray)`` - latest samples (a fresh array each tick).
    * ``spectrum_ready(freqs_hz, magnitudes, peak_hz)`` - ``peak_hz`` is NaN when
      the signal is too quiet to have a meaningful peak.  ``freqs_hz`` is shared
      and read-only.
    * ``levels_ready(peak_dbfs, rms_dbfs)``.
    * ``monitor_error(message)`` - the engine raised while being sampled; emitted
      once per failure streak, monitoring continues.

    Signals emitted just before :meth:`stop_monitoring` may still be delivered to
    receivers shortly afterwards, because queued connections are asynchronous.
    """

    waveform_ready = Signal(np.ndarray)
    spectrum_ready = Signal(np.ndarray, np.ndarray, float)
    levels_ready = Signal(float, float)
    monitor_error = Signal(str)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        interval_ms: int = LIVE_UPDATE_INTERVAL_MS,
    ) -> None:
        super().__init__(parent)
        if interval_ms <= 0:
            raise ValueError(f"interval_ms must be positive, got {interval_ms}")
        self._interval_s = interval_ms / 1000.0
        self._stop_event = threading.Event()
        self._engine: IAudioCaptureEngine | None = None
        self._sample_rate = 0
        self._spectrum = LiveSpectrum()
        self._last_tick_ms = 0.0
        self._max_tick_ms = 0.0
        self._tick_count = 0

    # ------------------------------------------------------------- stats
    @property
    def last_tick_ms(self) -> float:
        """Duration of the most recent tick in milliseconds."""
        return self._last_tick_ms

    @property
    def max_tick_ms(self) -> float:
        """Longest tick since monitoring started, in milliseconds."""
        return self._max_tick_ms

    @property
    def tick_count(self) -> int:
        """Ticks executed since monitoring started."""
        return self._tick_count

    # ----------------------------------------------------------- control
    def start_monitoring(self, engine: IAudioCaptureEngine, sample_rate: int) -> None:
        """Start (or restart) monitoring ``engine``.

        Args:
            engine: A capture engine; it need not be recording yet.
            sample_rate: Sample rate of the engine's stream in Hz.

        Raises:
            ValueError: non-positive sample rate.
        """
        if int(sample_rate) <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if self.isRunning():
            self.stop_monitoring()
        self._engine = engine
        self._sample_rate = int(sample_rate)
        self._spectrum = LiveSpectrum()
        self._last_tick_ms = self._max_tick_ms = 0.0
        self._tick_count = 0
        self._stop_event.clear()
        self.start()

    def stop_monitoring(self, timeout_ms: int = _DEFAULT_STOP_TIMEOUT_MS) -> bool:
        """Set the stop flag and wait for the thread to finish.

        Must not be called from the worker thread itself.

        Returns:
            True if the thread has finished (or was never running).
        """
        self._stop_event.set()
        if QThread.currentThread() is self:
            return False
        return bool(self.wait(timeout_ms))

    # --------------------------------------------------------------- loop
    def run(self) -> None:
        engine = self._engine
        if engine is None:
            return
        fs = self._sample_rate
        n_spec, n_wave = window_sizes(fs)
        interval = self._interval_s
        next_tick = time.perf_counter()
        failing = False

        while not self._stop_event.is_set():
            started = time.perf_counter()
            try:
                self._tick(engine, fs, n_spec, n_wave)
                failing = False
            except Exception as exc:  # a flaky engine must not kill the monitor
                if not failing:
                    self.monitor_error.emit(f"{type(exc).__name__}: {exc}")
                failing = True
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._last_tick_ms = elapsed_ms
            self._max_tick_ms = max(self._max_tick_ms, elapsed_ms)
            self._tick_count += 1

            next_tick += interval
            delay = next_tick - time.perf_counter()
            if delay < 0.0:  # fell behind: do not burst to catch up
                next_tick = time.perf_counter()
                delay = 0.0
            self._stop_event.wait(delay)

    def _tick(
        self,
        engine: IAudioCaptureEngine,
        fs: int,
        n_spec: int,
        n_wave: int,
    ) -> None:
        if not engine.is_recording or engine.is_paused:
            return
        raw = engine.snapshot(n_wave)
        if raw.size < MIN_SPECTRUM_SAMPLES:
            return

        x = raw if raw.ndim == 1 else raw[:, 0]
        tail = x[-n_spec:]
        peak, rms = level_stats(tail)
        self.levels_ready.emit(amplitude_to_dbfs(peak), amplitude_to_dbfs(rms))
        self.waveform_ready.emit(x)

        live = self._spectrum.compute(tail, fs, rms)
        if live is not None:
            freqs, magnitudes, peak_hz = live
            self.spectrum_ready.emit(
                freqs, magnitudes, math.nan if peak_hz is None else float(peak_hz)
            )


__all__: list[str] = ["CaptureMonitorWorker"]
