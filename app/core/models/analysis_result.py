"""Analysis result containers for spectral, wavelet, and statistical DSP routines."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SignalMetrics:
    """Comprehensive statistical and spectral metrics computed for an audio signal."""

    duration_s: float
    sample_rate: int
    n_samples: int
    rms: float
    peak: float
    rms_dbfs: float
    peak_dbfs: float
    zero_crossing_rate: float
    dominant_freq_hz: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    spectral_rolloff_hz: float
    thd: float


@dataclass(frozen=True)
class CwtResult:
    """Immutable result container for CWT computation."""

    fs: int
    time: np.ndarray
    frequencies_hz: np.ndarray
    coefficients: np.ndarray  # complex64, shape (n_scales, n_time)
    scales: np.ndarray

    def __post_init__(self) -> None:
        """Enforce read-only arrays for immutable guarantee."""
        for field_name in ("time", "frequencies_hz", "coefficients", "scales"):
            val = getattr(self, field_name)
            arr = np.asarray(val)
            arr_ro = arr.copy()
            arr_ro.setflags(write=False)
            object.__setattr__(self, field_name, arr_ro)

    @property
    def amplitude(self) -> np.ndarray:
        """Magnitude of the complex coefficients."""
        return np.abs(self.coefficients)

    @property
    def power(self) -> np.ndarray:
        """Squared magnitude of the complex coefficients."""
        return np.abs(self.coefficients) ** 2

    @property
    def phase(self) -> np.ndarray:
        """Phase angle of the complex coefficients in radians."""
        return np.angle(self.coefficients)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CwtResult):
            return False
        return (
            self.fs == other.fs
            and np.array_equal(self.time, other.time)
            and np.array_equal(self.frequencies_hz, other.frequencies_hz)
            and np.array_equal(self.coefficients, other.coefficients)
            and np.array_equal(self.scales, other.scales)
        )

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class StftResult:
    """Immutable result container for Short-Time Fourier Transform (STFT) spectrogram."""

    time: np.ndarray  # shape (n_time,)
    frequencies_hz: np.ndarray  # shape (n_freqs,)
    magnitude_db: np.ndarray  # shape (n_freqs, n_time)

    def __post_init__(self) -> None:
        """Enforce read-only arrays for immutable guarantee."""
        for field_name in ("time", "frequencies_hz", "magnitude_db"):
            val = getattr(self, field_name)
            arr = np.asarray(val)
            arr_ro = arr.copy()
            arr_ro.setflags(write=False)
            object.__setattr__(self, field_name, arr_ro)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StftResult):
            return False
        return (
            np.array_equal(self.time, other.time)
            and np.array_equal(self.frequencies_hz, other.frequencies_hz)
            and np.array_equal(self.magnitude_db, other.magnitude_db)
        )


__all__: list[str] = [
    "SignalMetrics",
    "CwtResult",
    "StftResult",
]
