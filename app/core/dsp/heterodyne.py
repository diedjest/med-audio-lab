"""Ultrasonic heterodyne frequency downshifting and time expansion routines."""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, resample_poly


def heterodyne_shift(x: np.ndarray, fs: int, shift_hz: float) -> np.ndarray:
    """Frequency-domain downshift by shift_hz to render ultrasound signals audible.

    Multiplies the analytic signal by complex exponential exp(-j·2π·shift_hz·t) and
    extracts the real component. Using the analytic signal prevents frequency aliasing
    and mirror frequency artifacts.

    Args:
        x: Input audio signal array (1D).
        fs: Sampling frequency in Hz.
        shift_hz: Frequency shift in Hz (positive value shifts downward).

    Returns:
        Frequency-downshifted 1D float32 signal array.
    """
    arr = np.asarray(x, dtype=np.float32).ravel()
    if arr.size == 0 or fs <= 0 or shift_hz == 0.0:
        return arr.copy()

    t = np.arange(arr.size, dtype=np.float64) / float(fs)
    carrier = np.exp(-1j * 2.0 * np.pi * float(shift_hz) * t)

    analytic = hilbert(arr)
    shifted = np.real(analytic * carrier).astype(np.float32)
    return shifted


def time_expansion(x: np.ndarray, factor: int = 10) -> np.ndarray:
    """Slow down signal by integer factor via polyphase resampling.

    Expanding time by N reduces all frequencies by N, mapping ultrasonic frequencies
    into the human audio range while preserving signal waveshape.

    Args:
        x: Input audio signal array (1D).
        factor: Integer expansion factor >= 1 (default: 10).

    Returns:
        Time-expanded 1D float32 signal array.
    """
    arr = np.asarray(x, dtype=np.float32).ravel()
    if arr.size == 0:
        return arr.copy()

    factor_int = int(factor)
    if factor_int < 1:
        raise ValueError(f"Expansion factor must be >= 1, got {factor}")
    if factor_int == 1:
        return arr.copy()

    expanded = resample_poly(arr, up=factor_int, down=1).astype(np.float32)
    return expanded


__all__: list[str] = [
    "heterodyne_shift",
    "time_expansion",
]
