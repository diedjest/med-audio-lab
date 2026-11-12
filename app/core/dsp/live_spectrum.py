"""Cheap per-frame analysis for live monitoring: levels and a windowed FFT spectrum.

Designed to run many times per second on the last ~80 ms of a signal.  The
window, frequency axis and gain depend only on ``(length, sample rate)`` and are
cached, so a steady stream costs one windowed ``rfft`` per frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.dsp.fft import dominant_peak

LIVE_WINDOW_MS: float = 80.0  # window for levels and the live spectrum
LIVE_FMIN_HZ: float = 5000.0  # lower edge of the live spectrum
WAVEFORM_WINDOW_S: float = 0.25  # span of the live waveform
MAX_WAVEFORM_SAMPLES: int = 1 << 16  # cap on the live waveform length
MIN_PEAK_RMS: float = 1e-5  # below this the signal is silence: no peak frequency
DB_FLOOR: float = -120.0
MIN_SPECTRUM_SAMPLES: int = 32
_MIN_FFT_SIZE: int = 256
_MIN_WINDOW_SAMPLES: int = 64


def amplitude_to_dbfs(value: float) -> float:
    """Linear amplitude to dBFS, floored at ``DB_FLOOR``."""
    if value <= 0.0:
        return DB_FLOOR
    return max(DB_FLOOR, float(20.0 * np.log10(value)))


def window_sizes(fs: int) -> tuple[int, int]:
    """Samples needed for the levels / spectrum window and for the live waveform."""
    n_spec = max(_MIN_WINDOW_SAMPLES, int(fs * LIVE_WINDOW_MS / 1000.0))
    n_wave = min(MAX_WAVEFORM_SAMPLES, max(n_spec, int(fs * WAVEFORM_WINDOW_S)))
    return n_spec, n_wave


def level_stats(window: np.ndarray) -> tuple[float, float]:
    """``(peak, rms)`` of a window as linear amplitudes."""
    peak = float(np.max(np.abs(window)))
    rms = float(np.sqrt(np.mean(np.square(window), dtype=np.float64)))
    return peak, rms


@dataclass(frozen=True)
class _SpectrumPlan:
    """Precomputed window / frequency axis for a given (length, sample rate)."""

    key: tuple[int, int]
    n_fft: int
    window: np.ndarray  # float32, length n
    gain: float  # converts |FFT| to single-sided sine amplitude
    freqs: np.ndarray  # float32, read-only, restricted to the live band
    lo: int
    hi: int

    @property
    def has_band(self) -> bool:
        return self.hi > self.lo


def _build_plan(n: int, fs: int, fmin_hz: float) -> _SpectrumPlan:
    n_fft = 1 << int(np.ceil(np.log2(max(_MIN_FFT_SIZE, n))))
    window = np.hanning(n).astype(np.float32)
    all_freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    lo = int(np.searchsorted(all_freqs, fmin_hz, side="left"))
    hi = int(np.searchsorted(all_freqs, fs / 4.0, side="right"))  # Nyquist / 2
    freqs = all_freqs[lo:hi].astype(np.float32)
    freqs.setflags(write=False)
    gain = 2.0 / float(window.sum()) if window.sum() > 0 else 0.0
    return _SpectrumPlan((n, fs), n_fft, window, gain, freqs, lo, hi)


class LiveSpectrum:
    """Single-sided amplitude spectrum of the latest window, limited to
    ``[fmin_hz, Nyquist / 2]``.

    Not thread-safe: give each thread its own instance.  The returned frequency
    array is shared and read-only; the magnitudes are a fresh array each call.
    """

    def __init__(self, fmin_hz: float = LIVE_FMIN_HZ) -> None:
        self._fmin_hz = float(fmin_hz)
        self._plan: _SpectrumPlan | None = None

    def compute(
        self,
        window: np.ndarray,
        fs: int,
        rms: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float | None] | None:
        """Spectrum of ``window`` (DC removed, Hann-windowed).

        Args:
            window: Latest samples (float32, 1D).
            fs: Sample rate in Hz.
            rms: RMS of ``window`` if already known (computed otherwise); used to
                suppress the peak frequency for near-silent input.

        Returns:
            ``(freqs_hz, magnitudes, peak_hz)`` where a sine of amplitude A gives a
            magnitude of about A at its frequency and ``peak_hz`` is None for
            silence; or None if the band ``[fmin_hz, fs / 4]`` is empty or the
            window is too short.
        """
        n = window.size
        if n < MIN_SPECTRUM_SAMPLES or fs <= 0:
            return None

        plan = self._plan
        if plan is None or plan.key != (n, fs):
            plan = self._plan = _build_plan(n, fs, self._fmin_hz)
        if not plan.has_band:
            return None

        if rms is None:
            rms = level_stats(window)[1]
        segment = (window - window.mean(dtype=np.float32)) * plan.window
        spectrum = np.fft.rfft(segment, n=plan.n_fft)
        magnitudes = (np.abs(spectrum[plan.lo : plan.hi]) * plan.gain).astype(np.float32)

        peak_hz: float | None = None
        if rms >= MIN_PEAK_RMS and magnitudes.size:
            peak_hz = dominant_peak(plan.freqs, magnitudes)[0]
        return plan.freqs, magnitudes, peak_hz


__all__: list[str] = [
    "LiveSpectrum",
    "amplitude_to_dbfs",
    "level_stats",
    "window_sizes",
    "LIVE_WINDOW_MS",
    "LIVE_FMIN_HZ",
    "WAVEFORM_WINDOW_S",
    "MAX_WAVEFORM_SAMPLES",
    "MIN_PEAK_RMS",
    "DB_FLOOR",
    "MIN_SPECTRUM_SAMPLES",
]
