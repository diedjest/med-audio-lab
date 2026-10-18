"""Analysis state and commands: one loaded signal, many views of it.

``plot_ready(plot_type, payload)`` payloads (``payload`` is always a dict):

========================  ======================================================
``"waveform"``            ``data``, ``sample_rate``, ``selection`` ((start_s, end_s) or None),
                          ``filtered``
``"fft"``                 ``freqs_hz``, ``magnitudes``, ``peak_hz``, ``peak_magnitude``,
                          ``sample_rate``
``"psd"``                 ``freqs_hz``, ``psd`` (V^2/Hz), ``sample_rate``
``"stft"``                ``time``, ``frequencies_hz``, ``magnitude_db``, ``sample_rate``
``"cwt"``                 ``result`` (``CwtResult``), ``sample_rate``, ``max_seconds``
                          (None unless the input was truncated to fit memory)
``"envelope"``            ``data`` (envelope), ``signal``, ``sample_rate``
``"impulses"``            ``impulses``, ``data``, ``sample_rate``
``"autocorrelation"``     ``lags_ms``, ``correlation``
``"wavelet_slice"``       ``time_rel_s``, ``values``, ``frequency_hz`` (the row used)
========================  ======================================================

While a segment is selected (:meth:`AnalysisViewModel.set_selection`) every
analysis payload except ``"waveform"`` describes the segment only: its time axis
starts at 0 at the segment start.  The waveform always shows the whole signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import Property, QObject, Signal, Slot

from app.core.dsp.cwt import longitudinal_slice
from app.core.dsp.fft import dominant_peak
from app.core.dsp.filters import FilterSpec
from app.core.models.analysis_result import CwtResult, SignalMetrics, StftResult
from app.core.models.impulse import DetectedImpulse, ImpulseDetectorConfig
from app.services.analysis_service import AnalysisService, AnalysisType

_DEFAULT_OPTIONS: dict[str, dict[str, Any]] = {
    "fft": {"window": "hann"},
    "psd": {"nperseg": 4096, "window": "hann"},
    "stft": {"nperseg": 4096, "noverlap": 3072, "window": "hann"},
    "cwt": {},
    "envelope": {"method": "hilbert", "window_ms": 12.0},
    "impulses": {},
    "autocorrelation": {"max_lag_ms": 120.0},
}
_CWT_DEFAULT_FMAX_HZ = 100_000.0


@dataclass
class _Pending:
    """The one request this view model is waiting for."""

    kind: str
    generation: int
    context: dict[str, Any] = field(default_factory=dict)


class AnalysisViewModel(QObject):
    """Wraps :class:`AnalysisService` and keeps the state of the analysed signal.

    Read-only properties (each with a ``<name>_changed`` signal):
    ``current_data``, ``current_sample_rate``, ``current_metrics``,
    ``current_impulses``, ``cwt_result`` and ``is_busy``.

    A segment of the signal can be selected with :meth:`set_selection`; all analyses
    (and ``current_metrics`` / ``current_impulses``) then use that segment only, until
    :meth:`clear_selection` goes back to the full signal.  Filtering still acts on the
    whole signal.  ``analysis_data`` is the samples being analysed.

    Load a signal with :meth:`load_signal`; the ``show_*`` slots then compute and
    announce results through ``plot_ready``.  Per-analysis parameters are set with
    :meth:`set_options`.  Only this view model's own requests are reported: results
    for superseded requests or for a signal that has since been replaced are dropped.
    Basic metrics are computed in the background whenever the service is idle.
    """

    current_data_changed = Signal()
    current_sample_rate_changed = Signal(int)
    current_metrics_changed = Signal()
    current_impulses_changed = Signal()
    cwt_result_changed = Signal()
    selection_changed = Signal()
    is_busy_changed = Signal(bool)

    plot_ready = Signal(str, object)  # (plot_type, payload dict)
    analysis_progress = Signal(float, str)
    analysis_error = Signal(str, str)  # (analysis type, message)
    notice = Signal(str)  # non-fatal information, e.g. an automatic CWT truncation

    def __init__(self, service: AnalysisService, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._service = service
        self._data = np.zeros(0, dtype=np.float32)
        self._original: np.ndarray | None = None
        self._selection: tuple[int, int] | None = None  # sample indices [start, end)
        self._fs = 0
        self._metrics: SignalMetrics | None = None
        self._impulses: list[DetectedImpulse] = []
        self._cwt: CwtResult | None = None
        self._busy = service.is_busy

        self._options: dict[str, dict[str, Any]] = {k: dict(v) for k, v in _DEFAULT_OPTIONS.items()}
        self._generation = 0
        self._pending: _Pending | None = None
        self._auto_metrics = True
        self._metrics_failed = False

        service.analysis_started.connect(self._on_started)
        service.analysis_progress.connect(self._on_progress)
        service.analysis_complete.connect(self._on_complete)
        service.analysis_error.connect(self._on_error)

    # ----------------------------------------------------------- properties
    @Property(object, notify=current_data_changed)
    def current_data(self) -> np.ndarray:
        """The loaded signal (float32, 1D). Treat as read-only."""
        return self._data

    @Property(int, notify=current_sample_rate_changed)
    def current_sample_rate(self) -> int:
        return self._fs

    @Property(object, notify=current_metrics_changed)
    def current_metrics(self) -> SignalMetrics | None:
        return self._metrics

    @Property(object, notify=current_impulses_changed)
    def current_impulses(self) -> list[DetectedImpulse]:
        return self._impulses

    @Property(object, notify=cwt_result_changed)
    def cwt_result(self) -> CwtResult | None:
        return self._cwt

    @Property(bool, notify=is_busy_changed)
    def is_busy(self) -> bool:
        return self._busy

    @property
    def selection(self) -> tuple[float, float] | None:
        """The selected segment as ``(start_s, end_s)``; None when the full signal is analysed."""
        if self._selection is None:
            return None
        return self._selection[0] / self._fs, self._selection[1] / self._fs

    @property
    def analysis_data(self) -> np.ndarray:
        """The samples the analyses run on: the selected segment, else the whole signal."""
        if self._selection is None:
            return self._data
        return self._data[self._selection[0] : self._selection[1]]

    @property
    def has_original(self) -> bool:
        """True if a filter replaced the data and :meth:`restore_original` can undo it."""
        return self._original is not None

    # -------------------------------------------------------------- options
    def set_options(self, analysis: str, **options: Any) -> None:
        """Set keyword arguments used by one analysis.

        Args:
            analysis: ``"fft"``, ``"stft"``, ``"cwt"``, ``"envelope"``, ``"impulses"``
                (key ``config``: an ``ImpulseDetectorConfig``) or ``"autocorrelation"``.
            **options: Passed to the service call (e.g. ``nperseg=2048`` for STFT);
                a value of ``None`` removes that option again.

        Raises:
            KeyError: unknown analysis name.
        """
        if analysis not in self._options:
            raise KeyError(f"Unknown analysis {analysis!r}; choose from {sorted(self._options)}")
        for name, value in options.items():
            if value is None:
                self._options[analysis].pop(name, None)  # None removes an option
            else:
                self._options[analysis][name] = value

    def options(self, analysis: str) -> dict[str, Any]:
        """A copy of the current options of one analysis."""
        return dict(self._options[analysis])

    # --------------------------------------------------------------- state
    def _set_busy(self, busy: bool) -> None:
        if busy != self._busy:
            self._busy = busy
            self.is_busy_changed.emit(busy)

    def _replace_data(self, data: np.ndarray, fs: int) -> None:
        """Install a new signal and invalidate everything derived from the old one."""
        rate_changed = fs != self._fs
        self._data, self._fs = data, fs
        self._metrics_failed = False
        self.current_data_changed.emit()
        if rate_changed:
            self.current_sample_rate_changed.emit(fs)
        self._invalidate_results()

    def _invalidate_results(self) -> None:
        """Drop metrics, impulses and the CWT: they belong to a different input."""
        if self._metrics is not None:
            self._metrics = None
            self.current_metrics_changed.emit()
        if self._impulses:
            self._impulses = []
            self.current_impulses_changed.emit()
        if self._cwt is not None:
            self._cwt = None
            self.cwt_result_changed.emit()

    def load_signal(
        self,
        data: np.ndarray,
        sample_rate: int,
        *,
        compute_metrics: bool = True,
    ) -> None:
        """Make ``data`` the current signal (copied; first channel if multi-channel).

        Cancels any running analysis and clears all derived results.

        Args:
            data: Samples.
            sample_rate: Sample rate in Hz.
            compute_metrics: Compute ``current_metrics`` in the background.

        Raises:
            ValueError: non-positive sample rate.
        """
        fs = int(sample_rate)
        if fs <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        arr = np.asarray(data)
        if arr.ndim > 1:
            arr = arr[:, 0]
        signal = np.nan_to_num(np.array(arr, dtype=np.float32).ravel(), nan=0.0, posinf=0.0, neginf=0.0)

        self._generation += 1
        self._pending = None
        self._original = None
        self._auto_metrics = bool(compute_metrics)
        self._service.cancel()
        had_selection, self._selection = self._selection is not None, None
        self._replace_data(signal, fs)
        if had_selection:
            self.selection_changed.emit()
        self._maybe_request_metrics()

    # ------------------------------------------------------------ selection
    def set_selection(self, start_s: float, end_s: float) -> None:
        """Restrict all analyses to the segment ``[start_s, end_s]`` of the current signal.

        The bounds are clamped to the signal and rounded to samples.  A segment that
        covers the whole signal is the same as :meth:`clear_selection`.  Running
        analyses are cancelled and every derived result is cleared.

        Raises:
            ValueError: no signal, or fewer than two samples between the bounds.
        """
        if self._data.size == 0 or self._fs <= 0:
            raise ValueError("No signal loaded.")
        lo, hi = sorted((float(start_s), float(end_s)))
        first = min(self._data.size, max(0, int(round(lo * self._fs))))
        last = min(self._data.size, max(0, int(round(hi * self._fs))))
        if last - first < 2:
            raise ValueError("The selected segment must contain at least two samples.")
        self._apply_selection(None if (first, last) == (0, self._data.size) else (first, last))

    @Slot()
    def clear_selection(self) -> None:
        """Go back to analysing the full signal."""
        self._apply_selection(None)

    def _apply_selection(self, new: tuple[int, int] | None) -> None:
        if new == self._selection:
            return
        self._selection = new
        self._generation += 1
        self._pending = None
        self._service.cancel()
        self._metrics_failed = False
        self._invalidate_results()
        self.selection_changed.emit()
        self._maybe_request_metrics()

    def selected_segment(self) -> np.ndarray:
        """A copy of the analysed samples (the selection, else the whole signal)."""
        return np.array(self.analysis_data, copy=True)

    @Slot()
    def cancel_analysis(self) -> None:
        """Abort the analysis that is running (it reports a cancelled error)."""
        self._service.cancel()

    @Slot()
    def restore_original(self) -> None:
        """Undo :meth:`apply_filter`: go back to the signal as it was loaded."""
        if self._original is None:
            return
        self._generation += 1
        self._pending = None
        self._service.cancel()
        original, self._original = self._original, None
        self._replace_data(original, self._fs)
        self.plot_ready.emit("waveform", self._waveform_payload(original, filtered=False))
        self._maybe_request_metrics()

    def _waveform_payload(self, data: np.ndarray, *, filtered: bool) -> dict[str, Any]:
        return {"data": data, "sample_rate": self._fs, "selection": self.selection, "filtered": filtered}

    # ------------------------------------------------------------ requests
    def _require_data(self, kind: str) -> bool:
        if self._data.size == 0 or self._fs <= 0:
            self.analysis_error.emit(kind, "No signal loaded.")
            return False
        return True

    def _request(self, kind: str, start: Callable[[], None], **context: Any) -> None:
        """Start a service call and remember it as the request we are waiting for.

        ``_pending`` is cleared first so the "superseded" error the service emits for
        a previous request is ignored, and set again only after the new request has
        been registered (its result can only arrive later, through the event loop).
        """
        self._pending = None
        start()
        self._pending = _Pending(kind, self._generation, context)

    def _maybe_request_metrics(self) -> None:
        if (
            self._auto_metrics
            and self._data.size
            and self._metrics is None
            and not self._metrics_failed
            and self._pending is None
            and not self._service.is_busy
        ):
            data, fs = self.analysis_data, self._fs
            self._request("metrics", lambda: self._service.compute_metrics(data, fs))

    # --------------------------------------------------------------- slots
    @Slot()
    def show_waveform(self) -> None:
        """Announce the current signal for the waveform plot."""
        if self._require_data("waveform"):
            self.plot_ready.emit(
                "waveform", self._waveform_payload(self._data, filtered=self._original is not None)
            )

    @Slot()
    def compute_metrics(self) -> None:
        """(Re)compute ``current_metrics`` in the background."""
        if self._require_data("metrics"):
            self._metrics_failed = False
            data, fs = self.analysis_data, self._fs
            self._request("metrics", lambda: self._service.compute_metrics(data, fs))

    @Slot()
    def show_fft(self) -> None:
        if self._require_data("fft"):
            data, fs, opts = self.analysis_data, self._fs, self.options("fft")
            self._request("fft", lambda: self._service.compute_fft(data, fs, **opts), options=opts)

    @Slot()
    def show_psd(self) -> None:
        """Welch power spectral density of the current signal."""
        if self._require_data("psd"):
            data, fs, opts = self.analysis_data, self._fs, self.options("psd")
            self._request("psd", lambda: self._service.compute_psd(data, fs, **opts))

    @Slot()
    def show_stft(self) -> None:
        if self._require_data("stft"):
            data, fs, opts = self.analysis_data, self._fs, self.options("stft")
            self._request("stft", lambda: self._service.compute_stft(data, fs, **opts))

    @Slot()
    def show_cwt(self) -> None:
        """CWT of the current signal; truncated automatically if it would not fit in memory."""
        if not self._require_data("cwt"):
            return
        data, fs = self.analysis_data, self._fs
        opts = {"fmax_hz": min(_CWT_DEFAULT_FMAX_HZ, fs / 2.0)}
        opts.update(self._options["cwt"])
        try:
            estimate = self._service.estimate_cwt_load(data, fs, **opts)
        except ValueError as exc:
            self.analysis_error.emit("cwt", str(exc))
            return

        max_seconds = opts.get("max_seconds")
        if not estimate["fits_budget"]:
            suggested = estimate.get("suggested_max_seconds")
            if not suggested:
                self.analysis_error.emit(
                    "cwt",
                    f"CWT needs about {estimate['estimated_mb']:.0f} MB "
                    f"(budget {estimate['budget_mb']:.0f} MB) and no shorter duration fits.",
                )
                return
            opts["max_seconds"] = max_seconds = float(suggested)
            self.notice.emit(
                f"CWT limited to the first {suggested:.2f} s to stay within the "
                f"{estimate['budget_mb']:.0f} MB memory budget."
            )
        self._request(
            "cwt",
            lambda: self._service.compute_cwt(data, fs, **opts),
            max_seconds=max_seconds,
        )

    @Slot()
    def show_envelope(self) -> None:
        if self._require_data("envelope"):
            data, fs, opts = self.analysis_data, self._fs, self.options("envelope")
            self._request("envelope", lambda: self._service.compute_envelope(data, fs, **opts))

    @Slot()
    def show_impulses(self) -> None:
        if self._require_data("impulses"):
            data, fs = self.analysis_data, self._fs
            config: ImpulseDetectorConfig | None = self._options["impulses"].get("config")
            self._request("impulses", lambda: self._service.detect_impulses(data, fs, config))

    @Slot()
    def show_autocorrelation(self) -> None:
        if self._require_data("autocorrelation"):
            data, fs, opts = self.analysis_data, self._fs, self.options("autocorrelation")
            self._request(
                "autocorrelation", lambda: self._service.compute_autocorrelation(data, fs, **opts)
            )

    @Slot(object)
    def apply_filter(self, spec: FilterSpec | list[FilterSpec] | tuple[FilterSpec, ...]) -> None:
        """Filter the current signal; the result replaces it (see :meth:`restore_original`)."""
        if self._require_data("filter"):
            data, fs = self._data, self._fs
            self._request("filter", lambda: self._service.apply_filter(data, fs, spec))

    @Slot(float)
    def show_wavelet_slice(self, f_hz: float) -> None:
        """Time cross-section of the CWT at the scale nearest ``f_hz`` (needs a CWT result)."""
        result = self._cwt
        if result is None:
            self.analysis_error.emit("wavelet_slice", "Compute the CWT first.")
            return
        row = int(np.argmin(np.abs(result.frequencies_hz - f_hz)))
        time_rel, values = longitudinal_slice(result, f_hz)
        self.plot_ready.emit(
            "wavelet_slice",
            {
                "time_rel_s": time_rel,
                "values": values,
                "frequency_hz": float(result.frequencies_hz[row]),
            },
        )

    # ----------------------------------------------------- service handlers
    def _on_started(self, kind: str) -> None:
        self._set_busy(True)

    def _on_progress(self, fraction: float, message: str) -> None:
        if self._pending is not None:
            self.analysis_progress.emit(fraction, message)

    def _take_pending(self, kind: str) -> _Pending | None:
        """Claim the pending request if ``kind`` is it and its signal is still current."""
        pending = self._pending
        if pending is None or pending.kind != kind or pending.generation != self._generation:
            return None
        self._pending = None
        return pending

    def _on_error(self, kind: str, message: str) -> None:
        self._set_busy(self._service.is_busy)
        pending = self._take_pending(kind)
        if pending is None:
            return  # cancelled / superseded / not ours
        if kind == "metrics":
            self._metrics_failed = True
        self.analysis_error.emit(kind, message)
        self._maybe_request_metrics()

    def _on_complete(self, kind: str, result: object) -> None:
        self._set_busy(self._service.is_busy)
        pending = self._take_pending(kind)
        if pending is None:
            return
        handler = {
            AnalysisType.METRICS.value: self._done_metrics,
            AnalysisType.FFT.value: self._done_fft,
            AnalysisType.PSD.value: self._done_psd,
            AnalysisType.STFT.value: self._done_stft,
            AnalysisType.CWT.value: self._done_cwt,
            AnalysisType.ENVELOPE.value: self._done_envelope,
            AnalysisType.IMPULSES.value: self._done_impulses,
            AnalysisType.AUTOCORRELATION.value: self._done_autocorrelation,
            AnalysisType.FILTER.value: self._done_filter,
        }.get(kind)
        if handler is not None:
            handler(result, pending)
        self._maybe_request_metrics()

    def _done_metrics(self, result: SignalMetrics, pending: _Pending) -> None:
        self._metrics = result
        self.current_metrics_changed.emit()

    def _done_fft(self, result: tuple[np.ndarray, np.ndarray], pending: _Pending) -> None:
        freqs, mags = result
        fmin = float(pending.context.get("options", {}).get("fmin_hz", 0.0))
        peak_hz, peak_mag = dominant_peak(freqs, mags, fmin_hz=fmin)
        self.plot_ready.emit(
            "fft",
            {
                "freqs_hz": freqs,
                "magnitudes": mags,
                "peak_hz": peak_hz,
                "peak_magnitude": peak_mag,
                "sample_rate": self._fs,
            },
        )

    def _done_psd(self, result: tuple[np.ndarray, np.ndarray], pending: _Pending) -> None:
        freqs, psd = result
        self.plot_ready.emit("psd", {"freqs_hz": freqs, "psd": psd, "sample_rate": self._fs})

    def _done_stft(self, result: StftResult, pending: _Pending) -> None:
        self.plot_ready.emit(
            "stft",
            {
                "time": result.time,
                "frequencies_hz": result.frequencies_hz,
                "magnitude_db": result.magnitude_db,
                "sample_rate": self._fs,
            },
        )

    def _done_cwt(self, result: CwtResult, pending: _Pending) -> None:
        self._cwt = result
        self.cwt_result_changed.emit()
        self.plot_ready.emit(
            "cwt",
            {
                "result": result,
                "sample_rate": result.fs,
                "max_seconds": pending.context.get("max_seconds"),
            },
        )

    def _done_envelope(self, result: np.ndarray, pending: _Pending) -> None:
        self.plot_ready.emit(
            "envelope", {"data": result, "signal": self.analysis_data, "sample_rate": self._fs}
        )

    def _done_impulses(self, result: list[DetectedImpulse], pending: _Pending) -> None:
        self._impulses = list(result)
        self.current_impulses_changed.emit()
        self.plot_ready.emit(
            "impulses", {"impulses": self._impulses, "data": self.analysis_data, "sample_rate": self._fs}
        )

    def _done_autocorrelation(
        self, result: tuple[np.ndarray, np.ndarray], pending: _Pending
    ) -> None:
        lags, corr = result
        self.plot_ready.emit("autocorrelation", {"lags_ms": lags, "correlation": corr})

    def _done_filter(self, result: np.ndarray, pending: _Pending) -> None:
        if self._original is None:
            self._original = self._data
        self._generation += 1  # anything still derived from the unfiltered signal is stale
        filtered = np.asarray(result, dtype=np.float32)
        self._replace_data(filtered, self._fs)
        self.plot_ready.emit("waveform", self._waveform_payload(filtered, filtered=True))


__all__: list[str] = ["AnalysisViewModel"]
