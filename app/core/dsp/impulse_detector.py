"""Robust ultrasonic pulse and impulse train detection using adaptive thresholding.

Noise floor estimated via Median Absolute Deviation (MAD) of the signal envelope.

Design rationale:
- MAD-based noise floor is robust to outliers (unlike mean/std)
- Envelope smoothing reduces false triggers from oscillation zero-crossings
- Merge gap prevents a single pulse from splitting into multiple detections
- Per-impulse FFT extracts dominant frequency of each detected burst
"""

from __future__ import annotations

import math

import numpy as np

from app.core.models.impulse import DetectedImpulse, ImpulseDetectorConfig


class ImpulseDetector:
    """Adaptive threshold impulse detector for ultrasonic burst signals.

    Algorithm:
    1. Remove DC offset from signal
    2. Compute smoothed envelope via moving average of |x|
    3. Normalize envelope to [0, 1]
    4. Calculate adaptive threshold = median + noise_factor × 1.4826 × MAD
    5. Threshold is max(fixed_threshold, adaptive_threshold), capped at 0.85
    6. Detect active regions where envelope > threshold
    7. Merge adjacent regions closer than merge_gap_ms
    8. Extract metrics (peak, RMS, dominant frequency) for each impulse

    The 1.4826 factor converts MAD to an estimate of standard deviation
    for normally distributed data.
    """

    def __init__(self, config: ImpulseDetectorConfig | None = None) -> None:
        self._config = config or ImpulseDetectorConfig()

    @property
    def config(self) -> ImpulseDetectorConfig:
        return self._config

    @config.setter
    def config(self, value: ImpulseDetectorConfig) -> None:
        self._config = value

    @staticmethod
    def _remove_dc(x: np.ndarray) -> np.ndarray:
        """Remove DC offset (mean subtraction)."""
        if x.size == 0:
            return x.copy()
        # float64 mean: a float32 mean leaves rounding residue on a constant signal,
        # which the envelope normalisation would then amplify into a fake burst.
        return (x.astype(np.float64) - float(np.mean(x, dtype=np.float64))).astype(np.float32)

    @staticmethod
    def _compute_envelope(
        x: np.ndarray, fs: int, envelope_ms: float
    ) -> np.ndarray:
        """Compute smoothed amplitude envelope."""
        abs_x = np.abs(x).astype(np.float32)
        win = max(1, int(fs * envelope_ms / 1000.0))
        if win > 1:
            kernel = np.ones(win, dtype=np.float32) / win
            return np.convolve(abs_x, kernel, mode="same").astype(np.float32)
        return abs_x

    @staticmethod
    def _estimate_dominant_frequency(
        segment: np.ndarray, fs: int, fmin_hz: float, fmax_hz: float
    ) -> float:
        """FFT-based dominant frequency estimation for a short segment."""
        if segment.size < 32:
            return math.nan

        window = np.hanning(segment.size).astype(np.float32)
        windowed = segment * window
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(segment.size, d=1.0 / fs)

        band = (freqs >= fmin_hz) & (freqs <= fmax_hz)
        if not np.any(band):
            return math.nan

        band_spectrum = spectrum[band]
        band_freqs = freqs[band]

        peak_idx = int(np.argmax(band_spectrum))
        peak_freq = float(band_freqs[peak_idx])

        # Parabolic interpolation for sub-bin precision
        if 0 < peak_idx < band_spectrum.size - 1:
            alpha = float(band_spectrum[peak_idx - 1])
            beta = float(band_spectrum[peak_idx])
            gamma = float(band_spectrum[peak_idx + 1])
            denom = alpha - 2 * beta + gamma
            if abs(denom) > 1e-12:
                delta = 0.5 * (alpha - gamma) / denom
                df = float(band_freqs[1] - band_freqs[0]) if band_freqs.size > 1 else 0.0
                peak_freq += delta * df

        return peak_freq if peak_freq > 0 else math.nan

    def detect(
        self,
        x: np.ndarray,
        fs: int,
        config: ImpulseDetectorConfig | None = None,
    ) -> list[DetectedImpulse]:
        """Detect impulse bursts in the signal.

        Args:
            x: Input signal (1D float).
            fs: Sample rate in Hz.
            config: Override configuration. If None, uses instance config.

        Returns:
            List of DetectedImpulse objects, sorted by time.
        """
        cfg = config or self._config

        # Preprocessing
        arr = np.asarray(x, dtype=np.float32).ravel()
        arr = self._remove_dc(arr)
        fs = max(1, int(fs))

        if arr.size < 8:
            return []

        # Compute envelope
        env = self._compute_envelope(arr, fs, cfg.envelope_ms)
        peak_env = float(np.max(env))
        if peak_env <= 1e-12:
            return []
        env_norm = env / peak_env

        # Adaptive threshold via MAD
        fixed_threshold = 10.0 ** (float(cfg.threshold_db) / 20.0)
        median_val = float(np.median(env_norm))
        mad_val = float(np.median(np.abs(env_norm - median_val)))
        adaptive_threshold = median_val + float(cfg.noise_factor) * 1.4826 * mad_val
        threshold = min(0.85, max(fixed_threshold, adaptive_threshold))

        # Find active regions
        active = env_norm >= threshold
        if not np.any(active):
            return []

        # Detect edges
        padded = np.concatenate(([0], active.astype(np.int8), [0]))
        edges = np.diff(padded)
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)

        # Merge close regions
        merge_gap = max(0, int(fs * float(cfg.merge_gap_ms) / 1000.0))
        merged: list[tuple[int, int]] = []
        for start, end in zip(starts, ends):
            si, ei = int(start), int(end)
            if merged and si - merged[-1][1] <= merge_gap:
                merged[-1] = (merged[-1][0], ei)
            else:
                merged.append((si, ei))

        # Extract impulses
        min_len = max(1, int(fs * float(cfg.min_duration_ms) / 1000.0))
        pre = max(0, int(fs * float(cfg.pre_pad_ms) / 1000.0))
        post = max(0, int(fs * float(cfg.post_pad_ms) / 1000.0))
        fmax = min(float(cfg.fmax_hz) if cfg.fmax_hz else fs / 2.0, fs / 2.0)

        impulses: list[DetectedImpulse] = []
        last_peak_s: float | None = None

        for raw_start, raw_end in merged:
            if raw_end - raw_start < min_len:
                continue

            start_i = max(0, raw_start - pre)
            end_i = min(arr.size, raw_end + post)
            if end_i <= start_i:
                continue

            segment = arr[start_i:end_i]
            peak_rel = int(np.argmax(np.abs(segment)))
            peak_i = start_i + peak_rel
            rms = float(np.sqrt(np.mean(np.square(segment))))

            dom_hz = self._estimate_dominant_frequency(
                segment, fs, cfg.fmin_hz, fmax
            )

            peak_s = peak_i / fs
            interval = math.nan if last_peak_s is None else (peak_s - last_peak_s) * 1000.0
            last_peak_s = peak_s

            impulses.append(
                DetectedImpulse(
                    index=len(impulses) + 1,
                    start_sample=start_i,
                    end_sample=end_i,
                    start_s=start_i / fs,
                    end_s=end_i / fs,
                    duration_ms=(end_i - start_i) / fs * 1000.0,
                    peak_time_s=peak_s,
                    peak_amplitude=float(np.max(np.abs(segment))),
                    rms=rms,
                    dominant_freq_hz=dom_hz,
                    interval_ms=interval,
                    threshold_used=threshold,
                )
            )

        return impulses


__all__: list[str] = [
    "DetectedImpulse",
    "ImpulseDetectorConfig",
    "ImpulseDetector",
]
