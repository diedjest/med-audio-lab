"""Fourier transform analysis routines: single-sided FFT, sub-bin peak estimation, and THD."""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window


def _resolve_window_type(window_name: str) -> str:
    """Normalize common window names for scipy.signal.get_window."""
    cleaned = window_name.strip().lower().replace("-", "").replace("_", "")
    if cleaned in ("hann", "hanning"):
        return "hann"
    if cleaned in ("hamming",):
        return "hamming"
    if cleaned in ("blackmanharris", "blackman_harris"):
        return "blackmanharris"
    if cleaned in ("flattop", "flat_top"):
        return "flattop"
    return window_name.strip().lower()


def fft_spectrum(
    x: np.ndarray,
    fs: int,
    *,
    window: str = "hann",
    fmin_hz: float = 0.0,
    fmax_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute full-resolution single-sided FFT spectrum normalized by sample count.

    Args:
        x: Input signal array (1D or converted to 1D float32).
        fs: Sampling frequency in Hz.
        window: Window function ("hann", "hamming", "blackman-harris", "flat-top", etc.).
        fmin_hz: Lower frequency cutoff in Hz (inclusive).
        fmax_hz: Upper frequency cutoff in Hz (inclusive, defaults to Nyquist).

    Returns:
        tuple[frequencies_hz, magnitudes_normalized]:
            frequencies_hz: 1D float32 array of spectral frequencies.
            magnitudes_normalized: 1D float32 array of single-sided normalized amplitudes.
    """
    arr = np.asarray(x, dtype=np.float32).ravel()
    n = arr.size

    if n == 0:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    if n == 1:
        freqs = np.array([0.0], dtype=np.float32)
        mags = np.abs(arr).astype(np.float32)
        if fmin_hz <= 0.0 <= (fmax_hz if fmax_hz is not None else 0.0):
            return freqs, mags
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    win_name = _resolve_window_type(window)
    win = get_window(win_name, n, fftbins=True).astype(np.float32)
    windowed = arr * win

    # Single-sided FFT via rfft
    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs).astype(np.float32)
    mags = (np.abs(fft_vals) / float(n)).astype(np.float32)

    upper_limit = fs / 2.0 if fmax_hz is None else float(fmax_hz)
    mask = (freqs >= float(fmin_hz)) & (freqs <= upper_limit)

    return freqs[mask], mags[mask]


def dominant_peak(
    freqs: np.ndarray,
    magnitudes: np.ndarray,
    *,
    fmin_hz: float = 0.0,
) -> tuple[float, float]:
    """Find dominant spectral peak frequency using parabolic interpolation for sub-bin precision.

    Args:
        freqs: Array of frequency bin centers in Hz.
        magnitudes: Corresponding magnitude spectrum array.
        fmin_hz: Minimum frequency to consider for peak search (ignores DC / sub-band).

    Returns:
        tuple[peak_frequency_hz, peak_magnitude]:
            Sub-bin interpolated dominant peak frequency and peak magnitude.
    """
    if freqs.size == 0 or magnitudes.size == 0:
        return 0.0, 0.0

    valid_indices = np.where(freqs >= float(fmin_hz))[0]
    if valid_indices.size == 0:
        return 0.0, 0.0

    peak_local_idx = int(np.argmax(magnitudes[valid_indices]))
    k = int(valid_indices[peak_local_idx])
    n = magnitudes.size

    beta = float(magnitudes[k])
    if beta <= 0.0:
        return float(freqs[k]), 0.0

    # Parabolic interpolation requires valid left and right neighbor bins
    if 0 < k < n - 1:
        alpha = float(magnitudes[k - 1])
        gamma = float(magnitudes[k + 1])
        denom = 2.0 * beta - alpha - gamma

        if abs(denom) > 1e-12:
            delta = 0.5 * (gamma - alpha) / denom
            # Clamp offset within [-1, 1] bin for stability
            delta = float(np.clip(delta, -1.0, 1.0))

            df_left = float(freqs[k] - freqs[k - 1])
            df_right = float(freqs[k + 1] - freqs[k])
            df = 0.5 * (df_left + df_right)

            interp_freq = float(freqs[k] + delta * df)
            interp_mag = float(beta - 0.25 * (alpha - gamma) * delta)
            return max(0.0, interp_freq), max(beta, interp_mag)

    return float(freqs[k]), beta


def estimate_thd(
    x: np.ndarray,
    fs: int,
    fundamental_hz: float,
    n_harmonics: int = 6,
) -> float:
    """Estimate Total Harmonic Distortion (THD) as a ratio of harmonic energy to fundamental.

    Calculates THD = sqrt(sum_{k=2..n_harmonics} H_k^2) / H_1.

    Args:
        x: Input audio signal samples.
        fs: Sampling frequency in Hz.
        fundamental_hz: Nominal fundamental frequency in Hz.
        n_harmonics: Maximum harmonic index to accumulate (default: 6th harmonic).

    Returns:
        THD ratio as a float (>= 0.0). Returns 0.0 if fundamental is missing or invalid.
    """
    arr = np.asarray(x, dtype=np.float32).ravel()
    if arr.size < 4 or fs <= 0 or fundamental_hz <= 0.0:
        return 0.0

    nyquist = fs / 2.0
    if fundamental_hz >= nyquist:
        return 0.0

    freqs, mags = fft_spectrum(arr, fs, window="hann")
    if freqs.size < 2:
        return 0.0

    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0

    def _find_peak_in_band(center_hz: float) -> float:
        # Search window width: adaptive based on bin resolution and frequency
        bandwidth = max(3.0 * df, 0.03 * center_hz)
        mask = (freqs >= center_hz - bandwidth) & (freqs <= center_hz + bandwidth)
        if not np.any(mask):
            return 0.0
        return float(np.max(mags[mask]))

    h1 = _find_peak_in_band(fundamental_hz)
    if h1 <= 1e-12:
        return 0.0

    harmonics_sum_sq = 0.0
    for k in range(2, n_harmonics + 1):
        target_harmonic_hz = k * fundamental_hz
        if target_harmonic_hz >= nyquist:
            break
        hk = _find_peak_in_band(target_harmonic_hz)
        harmonics_sum_sq += hk * hk

    return float(np.sqrt(harmonics_sum_sq) / h1)


__all__: list[str] = [
    "fft_spectrum",
    "dominant_peak",
    "estimate_thd",
]
