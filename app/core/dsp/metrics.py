"""Signal statistics and spectral metrics calculation."""

from __future__ import annotations

import numpy as np

from app.core.dsp.fft import dominant_peak, estimate_thd, fft_spectrum
from app.core.models.analysis_result import SignalMetrics


#: Spectral magnitudes at or below this count as "no signal" (float32 rounding noise level).
_NO_SPECTRAL_ENERGY = 1e-9


def compute_signal_metrics(x: np.ndarray, fs: int) -> SignalMetrics:
    """Compute comprehensive statistical, temporal, and spectral metrics for a signal.

    Args:
        x: Input signal (1D array).
        fs: Sampling rate in Hz.

    Returns:
        SignalMetrics value object populated with signal statistics.
    """
    arr = np.asarray(x, dtype=np.float32).ravel()
    n = arr.size

    if n == 0 or fs <= 0:
        return SignalMetrics(
            duration_s=0.0,
            sample_rate=max(1, int(fs)),
            n_samples=0,
            rms=0.0,
            peak=0.0,
            rms_dbfs=-120.0,
            peak_dbfs=-120.0,
            zero_crossing_rate=0.0,
            dominant_freq_hz=0.0,
            spectral_centroid_hz=0.0,
            spectral_bandwidth_hz=0.0,
            spectral_rolloff_hz=0.0,
            thd=0.0,
        )

    duration_s = n / float(fs)
    peak = float(np.max(np.abs(arr)))
    rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
    rms_dbfs = float(20.0 * np.log10(max(rms, 1e-12)))
    peak_dbfs = float(20.0 * np.log10(max(peak, 1e-12)))

    zero_crossings = (
        float(np.mean(np.abs(np.diff(np.signbit(arr))))) if n > 1 else 0.0
    )

    # Spectral statistics describe the signal's AC content.  A DC offset must not take part:
    # the window leaks it into the lowest bins, where it would drown a small tone.
    centred = (arr.astype(np.float64) - float(np.mean(arr, dtype=np.float64))).astype(np.float32)
    freqs, amp = fft_spectrum(centred, fs, window="hann")

    if amp.size > 1:
        amp_no_dc = amp.copy()
        amp_no_dc[0] = 0.0  # Ignore DC component for frequency estimation

        dom_freq, dom_magnitude = dominant_peak(freqs, amp_no_dc, fmin_hz=float(freqs[1]))
        if dom_magnitude <= _NO_SPECTRAL_ENERGY:  # silence or a constant: there is no dominant frequency
            dom_freq = 0.0

        total_energy = float(np.sum(amp_no_dc)) + 1e-12
        centroid = float(np.sum(freqs * amp_no_dc) / total_energy)
        bandwidth = float(
            np.sqrt(np.sum(((freqs - centroid) ** 2) * amp_no_dc) / total_energy)
        )

        cumulative = np.cumsum(amp_no_dc)
        cutoff_energy = 0.85 * cumulative[-1]
        rolloff_idx = int(np.searchsorted(cumulative, cutoff_energy))
        rolloff = float(freqs[min(rolloff_idx, freqs.size - 1)])
    else:
        dom_freq = centroid = bandwidth = rolloff = 0.0

    thd = estimate_thd(centred, fs, dom_freq) if dom_freq > 0.0 else 0.0

    return SignalMetrics(
        duration_s=duration_s,
        sample_rate=int(fs),
        n_samples=n,
        rms=rms,
        peak=peak,
        rms_dbfs=rms_dbfs,
        peak_dbfs=peak_dbfs,
        zero_crossing_rate=zero_crossings,
        dominant_freq_hz=dom_freq,
        spectral_centroid_hz=centroid,
        spectral_bandwidth_hz=bandwidth,
        spectral_rolloff_hz=rolloff,
        thd=thd,
    )


def format_metrics(metrics: SignalMetrics) -> dict[str, str]:
    """Human-readable ``{label: value}`` strings for a :class:`SignalMetrics`, in display order.

    Used by the analysis view and stored with recordings, so reports show the same text.
    """
    return {
        "Длительность": f"{metrics.duration_s:.3f} с",
        "Частота дискретизации": f"{metrics.sample_rate} Гц",
        "Отсчётов": f"{metrics.n_samples}",
        "СКЗ (RMS)": f"{metrics.rms:.5f}",
        "Пик": f"{metrics.peak:.5f}",
        "Уровень СКЗ": f"{metrics.rms_dbfs:.1f} дБFS",
        "Уровень пика": f"{metrics.peak_dbfs:.1f} дБFS",
        "Частота пересечений нуля": f"{metrics.zero_crossing_rate:.4f}",
        "Доминирующая частота": f"{metrics.dominant_freq_hz:.1f} Гц",
        "Спектральный центроид": f"{metrics.spectral_centroid_hz:.1f} Гц",
        "Ширина полосы": f"{metrics.spectral_bandwidth_hz:.1f} Гц",
        "Срез спектра 85%": f"{metrics.spectral_rolloff_hz:.1f} Гц",
        "КНИ (THD)": f"{metrics.thd * 100.0:.2f} %",
    }


def autocorrelation(
    x: np.ndarray,
    fs: int,
    max_lag_ms: float = 120.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute FFT-based normalized autocorrelation up to max_lag_ms.

    Args:
        x: Input signal array.
        fs: Sampling rate in Hz.
        max_lag_ms: Maximum lag to compute in milliseconds (default: 120.0 ms).

    Returns:
        tuple[lags_ms, correlation_normalized]:
            lags_ms: 1D float32 array of lag times in milliseconds.
            correlation_normalized: 1D float32 array of autocorrelation values in [-1, 1].
    """
    arr = np.asarray(x, dtype=np.float32).ravel()
    if arr.size == 0 or fs <= 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    n = arr.size
    max_lag_samples = min(n, max(1, int(round(fs * float(max_lag_ms) / 1000.0))))

    # Compute autocorrelation via FFT
    n_fft = 1 << int(np.ceil(np.log2(max(1, 2 * n - 1))))
    X = np.fft.rfft(arr, n=n_fft)
    power_spec = X * np.conj(X)
    r = np.fft.irfft(power_spec, n=n_fft)[:max_lag_samples]

    r0 = float(r[0]) if r.size > 0 else 0.0
    if r0 > 1e-12:
        r_norm = (r / r0).astype(np.float32)
    else:
        r_norm = np.zeros(max_lag_samples, dtype=np.float32)

    lags_ms = (np.arange(max_lag_samples, dtype=np.float32) * 1000.0 / fs).astype(
        np.float32
    )

    return lags_ms, r_norm


__all__: list[str] = [
    "compute_signal_metrics",
    "format_metrics",
    "autocorrelation",
]
