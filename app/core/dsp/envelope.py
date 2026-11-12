"""Signal conditioning, envelope extraction, noise gating, and trimming operations."""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def remove_dc(x: np.ndarray) -> np.ndarray:
    """Remove DC bias from the signal via mean subtraction.

    Args:
        x: Input signal array.

    Returns:
        DC-centered 1D float32 array.
    """
    arr = np.asarray(x, dtype=np.float32).ravel().copy()
    if arr.size > 0:
        arr -= float(np.mean(arr))
    return arr


def normalize_signal(x: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize signal peak amplitude to target_peak.

    Args:
        x: Input signal array.
        target_peak: Desired maximum absolute peak value (default: 0.95).

    Returns:
        Normalized 1D float32 array.
    """
    arr = np.asarray(x, dtype=np.float32).ravel().copy()
    if arr.size == 0:
        return arr

    peak = float(np.max(np.abs(arr)))
    if peak > 1e-12:
        arr = (arr * (float(target_peak) / peak)).astype(np.float32)
    return arr


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """Compute instantaneous amplitude envelope of the signal via the Hilbert analytic signal.

    Args:
        x: Input signal array.

    Returns:
        1D float32 array of instantaneous envelope amplitudes.
    """
    arr = np.asarray(x, dtype=np.float32).ravel()
    if arr.size == 0:
        return np.empty(0, dtype=np.float32)

    analytic = hilbert(arr)
    return np.abs(analytic).astype(np.float32)


def moving_average_envelope(
    x: np.ndarray,
    fs: int,
    window_ms: float = 12.0,
) -> np.ndarray:
    """Compute smoothed amplitude envelope via moving average of absolute values |x|.

    Args:
        x: Input signal array.
        fs: Sampling frequency in Hz.
        window_ms: Smoothing window length in milliseconds (default: 12.0 ms).

    Returns:
        1D float32 array of smoothed envelope values.
    """
    arr = np.abs(np.asarray(x, dtype=np.float32).ravel())
    if arr.size == 0 or fs <= 0:
        return np.empty(0, dtype=np.float32)

    # Clamp the window to the signal length so short inputs are still smoothed
    win = min(arr.size, max(1, int(round(fs * float(window_ms) / 1000.0))))
    if win <= 1:
        return arr.astype(np.float32)

    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(arr, kernel, mode="same").astype(np.float32)


def noise_gate(
    x: np.ndarray,
    threshold_db: float = -45.0,
) -> np.ndarray:
    """Attenuate audio samples falling below the threshold_db level by 0.15× factor.

    Args:
        x: Input audio signal array.
        threshold_db: Gate threshold in dBFS (default: -45.0 dB).

    Returns:
        Gated 1D float32 array.
    """
    arr = np.asarray(x, dtype=np.float32).ravel().copy()
    if arr.size == 0:
        return arr

    threshold_linear = 10.0 ** (float(threshold_db) / 20.0)
    attenuation_mask = np.abs(arr) < threshold_linear
    arr[attenuation_mask] *= 0.15
    return arr


def trim_silence(
    x: np.ndarray,
    fs: int,
    threshold_db: float = -48.0,
    pad_ms: int = 120,
) -> np.ndarray:
    """Trim leading and trailing silence regions below threshold_db, retaining pad_ms margin.

    Args:
        x: Input audio signal array.
        fs: Sampling rate in Hz.
        threshold_db: Silence threshold in dBFS (default: -48.0 dB).
        pad_ms: Safety margin padding in milliseconds to preserve on either side.

    Returns:
        Trimmed 1D float32 array.
    """
    arr = np.asarray(x, dtype=np.float32).ravel()
    if arr.size == 0 or fs <= 0:
        return np.empty(0, dtype=np.float32)

    threshold_linear = 10.0 ** (float(threshold_db) / 20.0)
    active_indices = np.where(np.abs(arr) >= threshold_linear)[0]

    if active_indices.size == 0:
        return arr.copy()

    pad_samples = max(0, int(round(fs * float(pad_ms) / 1000.0)))
    start_idx = max(0, int(active_indices[0]) - pad_samples)
    end_idx = min(arr.size, int(active_indices[-1]) + pad_samples + 1)

    return arr[start_idx:end_idx].copy()


__all__: list[str] = [
    "remove_dc",
    "normalize_signal",
    "hilbert_envelope",
    "moving_average_envelope",
    "noise_gate",
    "trim_silence",
]
