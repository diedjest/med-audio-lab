"""Background DSP orchestration: every analysis runs in a ``QThread`` worker.

Protocol
--------
Every ``analysis_started`` is followed by exactly one terminal signal, either
``analysis_complete`` or ``analysis_error`` (cancelled and superseded requests
report an error).  Signals from a request that is no longer current are dropped.

Concurrency
-----------
At most one worker thread runs at any time.  A new request cancels the current
one; the new worker starts as soon as the old thread has finished.  Only CWT is
interruptible mid-computation (between scale chunks); the other analyses are
single NumPy calls, so a superseded one finishes in the background and its
result is discarded.

Public methods must be called from the thread that owns the service (normally the
GUI thread).  Input arrays are copied at call time, so callers may reuse them.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from app.core.dsp.cwt import CwtEngine
from app.core.dsp.envelope import hilbert_envelope, moving_average_envelope
from app.core.dsp.fft import fft_spectrum
from app.core.dsp.filters import FilterPipeline, FilterSpec
from app.core.dsp.impulse_detector import ImpulseDetector
from app.core.dsp.metrics import autocorrelation, compute_signal_metrics
from app.core.dsp.stft import compute_stft, welch_psd
from app.core.models.impulse import ImpulseDetectorConfig

#: Progress reporter handed to analysis functions: ``report(fraction, message)``.
#: It raises ``_Cancelled`` when the request has been cancelled.
ProgressReporter = Callable[[float, str], None]


class AnalysisType(str, Enum):
    """Names emitted in the service signals."""

    FFT = "fft"
    PSD = "psd"
    STFT = "stft"
    CWT = "cwt"
    METRICS = "metrics"
    IMPULSES = "impulses"
    FILTER = "filter"
    ENVELOPE = "envelope"
    AUTOCORRELATION = "autocorrelation"


class _Cancelled(Exception):
    """Raised inside a worker to abort a cancelled request."""


@dataclass
class _Request:
    """One analysis job."""

    id: int
    kind: AnalysisType
    func: Callable[[ProgressReporter], object]
    cancel_event: threading.Event


class _AnalysisWorker(QThread):
    """Runs one request off the GUI thread and reports through signals."""

    progress = Signal(int, float, str)  # request id, fraction, message
    succeeded = Signal(int, object)  # request id, result
    failed = Signal(int, str)  # request id, message

    def __init__(self, request: _Request) -> None:
        super().__init__()
        self._request = request

    def run(self) -> None:
        request = self._request
        event = request.cancel_event

        def report(fraction: float, message: str) -> None:
            if event.is_set():
                raise _Cancelled()
            self.progress.emit(request.id, float(min(1.0, max(0.0, fraction))), message)

        try:
            report(0.0, f"Computing {request.kind.value}...")
            result = request.func(report)
            if event.is_set():
                raise _Cancelled()
            self.progress.emit(request.id, 1.0, f"{request.kind.value} complete")
            self.succeeded.emit(request.id, result)
        except _Cancelled:
            self.failed.emit(request.id, "Analysis cancelled.")
        except Exception as exc:  # reported to the UI, never crashes the thread
            self.failed.emit(request.id, str(exc) or type(exc).__name__)


def _as_signal(raw: np.ndarray) -> np.ndarray:
    """float32 1D signal from a copied input (first channel of multi-channel data)."""
    arr = np.asarray(raw)
    if arr.ndim > 1:
        arr = arr[:, 0]
    return np.asarray(arr, dtype=np.float32).ravel()


class AnalysisService(QObject):
    """Runs DSP computations in background threads and reports through signals."""

    analysis_started = Signal(str)  # analysis type
    analysis_progress = Signal(float, str)  # fraction 0-1, message
    analysis_complete = Signal(str, object)  # analysis type, result
    analysis_error = Signal(str, str)  # analysis type, message

    def __init__(
        self,
        cwt_engine: CwtEngine,
        filter_pipeline: FilterPipeline,
        impulse_detector: ImpulseDetector,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._cwt = cwt_engine
        self._filters = filter_pipeline
        self._detector = impulse_detector

        self._ids = itertools.count(1)
        self._current: _Request | None = None  # latest request without a terminal signal
        self._pending: _Request | None = None  # current request waiting for the old thread
        self._worker: _AnalysisWorker | None = None  # the one running thread, if any

    # ----------------------------------------------------------- state
    @property
    def is_busy(self) -> bool:
        """True while a request is pending or running."""
        return self._current is not None

    def cancel(self) -> None:
        """Cancel the current request; it reports ``analysis_error`` ("cancelled")."""
        self._retire_current("Analysis cancelled.")

    def shutdown(self, timeout_ms: int = 10_000) -> bool:
        """Cancel everything and wait for the worker thread (call before app exit).

        Returns:
            True if no thread is left running.
        """
        self.cancel()
        worker = self._worker
        if worker is None:
            return True
        return bool(worker.wait(timeout_ms))

    # -------------------------------------------------------- scheduling
    def _retire_current(self, reason: str) -> None:
        """Cancel the current request and report it as failed."""
        request = self._current
        if request is None:
            return
        request.cancel_event.set()
        self._current = None
        if self._pending is request:
            self._pending = None
        self.analysis_error.emit(request.kind.value, reason)

    def _submit(
        self,
        kind: AnalysisType,
        func: Callable[[ProgressReporter], object],
    ) -> None:
        self._retire_current("Analysis cancelled: superseded by a new request.")
        request = _Request(next(self._ids), kind, func, threading.Event())
        self._current = request
        self.analysis_started.emit(kind.value)
        if self._worker is None:
            self._launch(request)
        else:
            self._pending = request  # starts when the cancelled thread has finished

    def _launch(self, request: _Request) -> None:
        worker = _AnalysisWorker(request)
        worker.progress.connect(self._on_progress)
        worker.succeeded.connect(self._on_succeeded)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def _on_worker_finished(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.deleteLater()
        pending, self._pending = self._pending, None
        if pending is not None and pending is self._current:
            self._launch(pending)

    # ------------------------------------------------------ worker slots
    def _is_current(self, request_id: int) -> bool:
        return self._current is not None and self._current.id == request_id

    def _on_progress(self, request_id: int, fraction: float, message: str) -> None:
        if self._is_current(request_id):
            self.analysis_progress.emit(fraction, message)

    def _on_succeeded(self, request_id: int, result: object) -> None:
        if not self._is_current(request_id):
            return
        kind = self._current.kind  # type: ignore[union-attr]
        self._current = None
        self.analysis_complete.emit(kind.value, result)

    def _on_failed(self, request_id: int, message: str) -> None:
        if not self._is_current(request_id):
            return
        kind = self._current.kind  # type: ignore[union-attr]
        self._current = None
        self.analysis_error.emit(kind.value, message)

    # ---------------------------------------------------------- analyses
    def compute_fft(self, data: np.ndarray, fs: int, **kwargs: Any) -> None:
        """Single-sided FFT spectrum.

        Keyword args go to ``fft_spectrum`` (``window``, ``fmin_hz``, ``fmax_hz``).
        Result: ``(frequencies_hz, magnitudes)``.
        """
        raw = np.array(data, copy=True)
        self._submit(
            AnalysisType.FFT,
            lambda report: fft_spectrum(_as_signal(raw), int(fs), **kwargs),
        )

    def compute_psd(self, data: np.ndarray, fs: int, **kwargs: Any) -> None:
        """Welch power spectral density.

        Keyword args go to ``welch_psd`` (``nperseg``, ``window``).
        Result: ``(frequencies_hz, psd)``.
        """
        raw = np.array(data, copy=True)
        self._submit(
            AnalysisType.PSD,
            lambda report: welch_psd(_as_signal(raw), int(fs), **kwargs),
        )

    def compute_stft(self, data: np.ndarray, fs: int, **kwargs: Any) -> None:
        """STFT spectrogram.

        Keyword args go to ``compute_stft`` (``nperseg``, ``noverlap``, ``window``,
        ``fmin_hz``, ``fmax_hz``, ``db``).  Result: ``StftResult``.
        """
        raw = np.array(data, copy=True)
        self._submit(
            AnalysisType.STFT,
            lambda report: compute_stft(_as_signal(raw), int(fs), **kwargs),
        )

    def compute_cwt(self, data: np.ndarray, fs: int, **kwargs: Any) -> None:
        """Continuous wavelet transform with progress reporting and cancellation.

        Keyword args go to ``CwtEngine.compute`` (``fmin_hz``, ``fmax_hz``, ``scales``,
        ``max_seconds``, ``target_fs``, ``scale_stride``).  A ``progress_callback``
        given here is called in addition to the service's own, from the worker thread.
        Result: ``CwtResult``.  Use :meth:`estimate_cwt_load` first for large inputs.
        """
        raw = np.array(data, copy=True)
        user_callback = kwargs.pop("progress_callback", None)

        def run(report: ProgressReporter) -> object:
            def on_progress(fraction: float, message: str) -> None:
                report(fraction, message)  # raises _Cancelled to abort between chunks
                if user_callback is not None:
                    user_callback(fraction, message)

            return self._cwt.compute(
                _as_signal(raw), int(fs), progress_callback=on_progress, **kwargs
            )

        self._submit(AnalysisType.CWT, run)

    def compute_metrics(self, data: np.ndarray, fs: int) -> None:
        """Signal statistics and spectral metrics. Result: ``SignalMetrics``."""
        raw = np.array(data, copy=True)
        self._submit(
            AnalysisType.METRICS,
            lambda report: compute_signal_metrics(_as_signal(raw), int(fs)),
        )

    def detect_impulses(
        self,
        data: np.ndarray,
        fs: int,
        config: ImpulseDetectorConfig | None = None,
    ) -> None:
        """Impulse detection. Result: ``list[DetectedImpulse]``.

        ``config`` overrides the detector's own configuration for this call.
        """
        raw = np.array(data, copy=True)
        self._submit(
            AnalysisType.IMPULSES,
            lambda report: self._detector.detect(_as_signal(raw), int(fs), config),
        )

    def apply_filter(
        self,
        data: np.ndarray,
        fs: int,
        spec: FilterSpec | list[FilterSpec] | tuple[FilterSpec, ...],
    ) -> None:
        """Zero-phase filtering. Result: filtered ``np.ndarray``.

        ``spec`` may be one ``FilterSpec`` or a sequence applied as a chain.
        """
        raw = np.array(data, copy=True)

        def run(report: ProgressReporter) -> object:
            signal = _as_signal(raw)
            if isinstance(spec, (list, tuple)):
                return self._filters.apply_chain(signal, int(fs), list(spec))
            return self._filters.apply(signal, int(fs), spec)

        self._submit(AnalysisType.FILTER, run)

    def compute_envelope(
        self,
        data: np.ndarray,
        fs: int,
        *,
        method: str = "hilbert",
        window_ms: float = 12.0,
    ) -> None:
        """Amplitude envelope. Result: ``np.ndarray``.

        Args:
            method: ``"hilbert"`` (instantaneous amplitude, default) or
                ``"moving_average"`` (smoothed ``|x|``).
            window_ms: Smoothing window for ``"moving_average"``.
        """
        raw = np.array(data, copy=True)

        def run(report: ProgressReporter) -> object:
            signal = _as_signal(raw)
            if method == "hilbert":
                return hilbert_envelope(signal)
            if method == "moving_average":
                return moving_average_envelope(signal, int(fs), window_ms)
            raise ValueError(f"Unknown envelope method {method!r}; use 'hilbert' or 'moving_average'.")

        self._submit(AnalysisType.ENVELOPE, run)

    def compute_autocorrelation(self, data: np.ndarray, fs: int, **kwargs: Any) -> None:
        """Normalized autocorrelation.

        Keyword args go to ``autocorrelation`` (``max_lag_ms``).
        Result: ``(lags_ms, correlation)``.
        """
        raw = np.array(data, copy=True)
        self._submit(
            AnalysisType.AUTOCORRELATION,
            lambda report: autocorrelation(_as_signal(raw), int(fs), **kwargs),
        )

    # ---------------------------------------------------------- pre-flight
    def estimate_cwt_load(self, data: np.ndarray, fs: int, **kwargs: Any) -> dict[str, Any]:
        """Synchronously estimate the memory a CWT would need, without computing it.

        Accepts the same selection keywords as :meth:`compute_cwt`
        (``fmin_hz``, ``fmax_hz``, ``scales``, ``max_seconds``, ``target_fs``,
        ``scale_stride``).

        Returns:
            A dict with ``n_samples``, ``sample_rate``, ``duration_s``, ``n_scales``,
            ``n_fft``, ``estimated_mb``, ``budget_mb``, ``fits_budget`` and, when the
            job does not fit, ``suggested_max_seconds`` and ``suggested_scale_stride``
            (None where no such change would help).

        Raises:
            ValueError: signal too short or no scales in the requested band.
        """
        kwargs.pop("progress_callback", None)
        n_samples = int(np.shape(data)[0]) if np.ndim(data) else 0
        return dict(self._cwt.estimate_load(n_samples, int(fs), **kwargs))


__all__: list[str] = ["AnalysisService", "AnalysisType"]
