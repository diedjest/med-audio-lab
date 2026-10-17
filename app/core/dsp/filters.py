"""Zero-phase digital filtering using second-order sections (SOS).

Supports Butterworth, Chebyshev Type I, and Chebyshev Type II designs.

Design rationale:
- sosfiltfilt provides zero phase distortion (critical for impulse timing)
- SOS representation avoids numerical instability of transfer function form
- Automatic Nyquist guard prevents invalid filter specs
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, cheby1, cheby2, sosfiltfilt, sosfreqz

from app.constants import FilterDesign, FilterType


@dataclass(frozen=True)
class FilterSpec:
    """Immutable filter specification."""

    filter_type: FilterType
    freq_hz: float | tuple[float, float]  # scalar for LP/HP, tuple for BP/BS
    order: int = 4
    design: FilterDesign = FilterDesign.BUTTERWORTH
    ripple_db: float = 0.5  # passband ripple (Cheby I) or stopband atten (Cheby II)


class FilterPipeline:
    """Configurable, zero-phase IIR filter with automatic SOS calculation.

    Usage:
        pipeline = FilterPipeline()
        filtered = pipeline.apply(
            signal, fs=192000,
            spec=FilterSpec(FilterType.BANDPASS, (20000, 80000), order=6)
        )
    """

    @staticmethod
    def _normalize_frequency(
        freq_hz: float | tuple[float, float],
        fs: int,
        filter_type: FilterType,
    ) -> float | tuple[float, float]:
        """Normalize frequency to [0, 1) relative to Nyquist, with safety margin."""
        nyquist = fs / 2.0
        margin = 0.98  # Stay 2% below Nyquist to avoid instability

        if isinstance(freq_hz, tuple):
            lo = min(float(freq_hz[0]) / nyquist, margin)
            hi = min(float(freq_hz[1]) / nyquist, margin)
            if lo <= 0:
                lo = 1e-6
            if hi <= lo:
                raise ValueError(
                    f"Upper cutoff ({freq_hz[1]} Hz) must exceed lower "
                    f"cutoff ({freq_hz[0]} Hz)."
                )
            return (lo, hi)
        else:
            wn = min(float(freq_hz) / nyquist, margin)
            if wn <= 0:
                raise ValueError(f"Cutoff frequency must be > 0 Hz (got {freq_hz}).")
            return wn

    @staticmethod
    def _out_dtype(src: np.ndarray) -> np.dtype:
        """Output dtype: the input's floating dtype, or float64 for non-float input."""
        return src.dtype if np.issubdtype(src.dtype, np.floating) else np.dtype(np.float64)

    @staticmethod
    def design_sos(spec: FilterSpec, fs: int) -> np.ndarray:
        """Compute second-order sections for the given filter specification.

        Returns:
            SOS array of shape (n_sections, 6).
        """
        wn = FilterPipeline._normalize_frequency(
            spec.freq_hz, fs, spec.filter_type
        )
        btype = spec.filter_type.value

        if spec.design == FilterDesign.BUTTERWORTH:
            sos = butter(spec.order, wn, btype=btype, output="sos")
        elif spec.design == FilterDesign.CHEBYSHEV_I:
            sos = cheby1(spec.order, spec.ripple_db, wn, btype=btype, output="sos")
        elif spec.design == FilterDesign.CHEBYSHEV_II:
            sos = cheby2(spec.order, spec.ripple_db, wn, btype=btype, output="sos")
        else:
            raise ValueError(f"Unknown filter design: {spec.design}")

        return sos

    def apply(
        self,
        x: np.ndarray,
        fs: int,
        spec: FilterSpec,
    ) -> np.ndarray:
        """Apply zero-phase filtering to the signal.

        Args:
            x: Input signal (1D float32/float64).
            fs: Sample rate in Hz.
            spec: Filter specification.

        Returns:
            Filtered signal with same length and dtype as input.
        """
        src = np.asarray(x)
        arr = src.astype(np.float64).ravel()
        if arr.size < 3 * spec.order:
            raise ValueError(
                f"Signal too short ({arr.size} samples) for filter order {spec.order}. "
                f"Need at least {3 * spec.order} samples."
            )

        sos = self.design_sos(spec, fs)
        y = sosfiltfilt(sos, arr).astype(self._out_dtype(src), copy=False)
        return y

    def apply_chain(
        self,
        x: np.ndarray,
        fs: int,
        specs: list[FilterSpec],
    ) -> np.ndarray:
        """Apply a chain of filters sequentially.

        Args:
            x: Input signal.
            fs: Sample rate.
            specs: List of filter specs to apply in order.

        Returns:
            Signal filtered through all stages.
        """
        src = np.asarray(x)
        result = src.astype(np.float64).ravel()
        for spec in specs:
            result = self.apply(result, fs, spec)
        return result.astype(self._out_dtype(src), copy=False)

    def frequency_response(
        self,
        spec: FilterSpec,
        fs: int,
        n_points: int = 4096,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the filter's frequency response magnitude.

        Returns:
            (frequencies_hz, magnitude_db) arrays.
        """
        sos = self.design_sos(spec, fs)
        w, h = sosfreqz(sos, worN=n_points, fs=fs)
        mag_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))
        return np.asarray(w, dtype=np.float64), mag_db.astype(np.float64)


__all__: list[str] = [
    "FilterType",
    "FilterDesign",
    "FilterSpec",
    "FilterPipeline",
]
