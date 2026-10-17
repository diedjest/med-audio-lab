"""Short-Time Fourier Transform (STFT) spectrogram and Welch PSD analysis."""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window, welch

from app.core.dsp.fft import _resolve_window_type
from app.core.models.analysis_result import StftResult


def compute_stft(
    x: np.ndarray,
    fs: int,
    *,
    nperseg: int = 4096,
    noverlap: int = 3072,
    window: str = "hann",
    fmin_hz: float = 0.0,
    fmax_hz: float = 100_000.0,
    db: bool = True,
) -> StftResult:
    """Compute Short-Time Fourier Transform (STFT) spectrogram using NumPy stride tricks.

    Constructs windowed frames with zero-copy strided views and evaluates real FFTs.
    Handles signals shorter than nperseg with zero padding and clamps the dB floor to -120 dB.

    Args:
        x: Input 1D audio signal array.
        fs: Sampling rate in Hz.
        nperseg: Length of each FFT segment in samples.
        noverlap: Number of overlapping samples between segments.
        window: Window type identifier (e.g., 'hann', 'hamming', 'blackman-harris').
        fmin_hz: Lower bound of frequency band in Hz.
        fmax_hz: Upper bound of frequency band in Hz.
        db: If True, convert magnitudes to decibels (dB) clamped at -120 dB; otherwise linear.

    Returns:
        StftResult containing time axis, frequency axis, and magnitude/dB matrix.
    """
    arr = np.asarray(x, dtype=np.float32).ravel()

    if arr.size == 0 or fs <= 0:
        return StftResult(
            time=np.empty(0, dtype=np.float32),
            frequencies_hz=np.empty(0, dtype=np.float32),
            magnitude_db=np.empty((0, 0), dtype=np.float32),
        )

    nperseg = int(nperseg)
    noverlap = int(noverlap)
    if nperseg <= 0:
        raise ValueError(f"nperseg must be positive, got {nperseg}")
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError(f"noverlap must be in [0, nperseg), got {noverlap}")

    hop = nperseg - noverlap

    # Zero-pad if signal is shorter than the requested segment length
    if arr.size < nperseg:
        pad_width = nperseg - arr.size
        arr = np.pad(arr, (0, pad_width), mode="constant")

    n_frames = 1 + (arr.size - nperseg) // hop
    total_samples = (n_frames - 1) * hop + nperseg
    arr_slice = np.ascontiguousarray(arr[:total_samples], dtype=np.float32)

    # Fast zero-copy strided frame view
    frame_shape = (n_frames, nperseg)
    frame_strides = (arr_slice.strides[0] * hop, arr_slice.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        arr_slice,
        shape=frame_shape,
        strides=frame_strides,
        writeable=False,
    )

    win_name = _resolve_window_type(window)
    win = get_window(win_name, nperseg, fftbins=True).astype(np.float32)
    windowed = frames * win[None, :]

    # Perform real FFT on all frames along axis 1
    X = np.fft.rfft(windowed, axis=1)  # shape: (n_frames, n_freqs)
    mag = (np.abs(X).T / float(nperseg)).astype(np.float32)  # shape: (n_freqs, n_frames)

    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs).astype(np.float32)
    time_axis = ((np.arange(n_frames) * hop + (nperseg / 2.0)) / fs).astype(np.float32)

    # Filter to requested frequency band [fmin_hz, fmax_hz]
    band_mask = (freqs >= float(fmin_hz)) & (freqs <= float(fmax_hz))
    freqs_band = freqs[band_mask]
    mag_band = mag[band_mask, :]

    if db:
        # 20 * log10(magnitude), clamped to -120 dB floor
        log_vals = 20.0 * np.log10(np.maximum(mag_band, 1e-6))
        result_mag = np.maximum(log_vals, -120.0).astype(np.float32)
    else:
        result_mag = mag_band.astype(np.float32)

    return StftResult(
        time=time_axis,
        frequencies_hz=freqs_band,
        magnitude_db=result_mag,
    )


def welch_psd(
    x: np.ndarray,
    fs: int,
    *,
    nperseg: int = 4096,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Power Spectral Density (PSD) using Welch's averaged periodogram method.

    Wrapper around scipy.signal.welch ensuring robust handling of short buffers and edge cases.

    Args:
        x: Input 1D audio signal array.
        fs: Sampling frequency in Hz.
        nperseg: Length of each segment in samples.
        window: Window specification string.

    Returns:
        tuple[frequencies_hz, psd]:
            frequencies_hz: 1D float32 array of frequencies.
            psd: 1D float32 array of power spectral density values (V²/Hz).
    """
    arr = np.asarray(x, dtype=np.float32).ravel()

    if arr.size == 0 or fs <= 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    if arr.size == 1:
        return (
            np.array([0.0], dtype=np.float32),
            np.array([float(arr[0] ** 2)], dtype=np.float32),
        )

    win_name = _resolve_window_type(window)
    actual_nperseg = min(int(nperseg), arr.size)
    if actual_nperseg < 2:
        actual_nperseg = 2

    freqs, psd_vals = welch(
        arr,
        fs=fs,
        window=win_name,
        nperseg=actual_nperseg,
    )

    return freqs.astype(np.float32), psd_vals.astype(np.float32)


__all__: list[str] = [
    "compute_stft",
    "welch_psd",
]
