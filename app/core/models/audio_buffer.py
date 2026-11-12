"""Immutable AudioBuffer value object with DSP convenience metrics and operations."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AudioBuffer:
    """Immutable audio buffer container supporting mono and multi-channel signals."""

    data: np.ndarray  # float32 array, shape (n_samples,) or (n_samples, n_channels)
    sample_rate: int
    channels: int = 1

    def __post_init__(self) -> None:
        """Enforce float32 dtype, array dimensions, and immutability."""
        arr = np.asarray(self.data, dtype=np.float32)
        if arr.ndim == 1:
            detected_channels = 1
        elif arr.ndim == 2:
            detected_channels = arr.shape[1]
        elif arr.ndim == 0:
            arr = arr.reshape((0,))
            detected_channels = 1
        else:
            raise ValueError(f"AudioBuffer data must be 1D or 2D, got shape {arr.shape}")

        arr_copy = arr.copy()
        arr_copy.setflags(write=False)
        object.__setattr__(self, "data", arr_copy)
        if self.channels != detected_channels:
            object.__setattr__(self, "channels", detected_channels)

    @property
    def n_samples(self) -> int:
        """Total number of sample frames in buffer."""
        return int(self.data.shape[0]) if self.data.ndim > 0 else 0

    @property
    def duration_s(self) -> float:
        """Buffer duration in seconds."""
        if self.sample_rate <= 0:
            return 0.0
        return self.n_samples / float(self.sample_rate)

    @property
    def peak(self) -> float:
        """Absolute peak amplitude."""
        if self.n_samples == 0:
            return 0.0
        return float(np.max(np.abs(self.data)))

    @property
    def rms(self) -> float:
        """Root-mean-square amplitude."""
        if self.n_samples == 0:
            return 0.0
        return float(np.sqrt(np.mean(self.data.astype(np.float64) ** 2)))

    @property
    def peak_dbfs(self) -> float:
        """Peak amplitude in decibels relative to full scale (dBFS)."""
        return float(20.0 * np.log10(max(self.peak, 1e-12)))

    @property
    def rms_dbfs(self) -> float:
        """RMS amplitude in decibels relative to full scale (dBFS)."""
        return float(20.0 * np.log10(max(self.rms, 1e-12)))

    def to_mono(self) -> AudioBuffer:
        """Convert multi-channel audio buffer to single-channel mono by averaging.

        Returns:
            New AudioBuffer with 1D float32 array and channels=1.
        """
        if self.channels == 1:
            if self.data.ndim == 2 and self.data.shape[1] == 1:
                return AudioBuffer(
                    data=self.data[:, 0].copy(),
                    sample_rate=self.sample_rate,
                    channels=1,
                )
            return self

        if self.data.ndim == 2:
            mono_data = np.mean(self.data, axis=1, dtype=np.float32)
            return AudioBuffer(
                data=mono_data,
                sample_rate=self.sample_rate,
                channels=1,
            )
        return self

    def normalize(self, target_peak: float = 0.95) -> AudioBuffer:
        """Scale amplitude so peak reaches target_peak.

        Args:
            target_peak: Desired absolute peak amplitude in (0, 1].

        Returns:
            Normalized AudioBuffer instance.
        """
        current_peak = self.peak
        if current_peak == 0.0 or self.n_samples == 0:
            return self

        gain = float(target_peak) / current_peak
        normalized = (self.data * gain).astype(np.float32)
        return AudioBuffer(
            data=normalized,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )

    def trim(self, start_s: float, end_s: float) -> AudioBuffer:
        """Extract a time slice from the audio buffer.

        Args:
            start_s: Start time in seconds.
            end_s: End time in seconds.

        Returns:
            Trimmed AudioBuffer instance.
        """
        start_sample = max(0, int(round(start_s * self.sample_rate)))
        end_sample = max(start_sample, min(self.n_samples, int(round(end_s * self.sample_rate))))
        trimmed_data = self.data[start_sample:end_sample].copy()
        return AudioBuffer(
            data=trimmed_data,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )

    def __eq__(self, other: object) -> bool:
        """Compare two AudioBuffer instances for equality."""
        if not isinstance(other, AudioBuffer):
            return False
        return (
            self.sample_rate == other.sample_rate
            and self.channels == other.channels
            and np.array_equal(self.data, other.data)
        )

    def __repr__(self) -> str:
        """Return human-readable debug representation."""
        return (
            f"AudioBuffer(n_samples={self.n_samples}, "
            f"sample_rate={self.sample_rate}, "
            f"channels={self.channels}, "
            f"duration_s={self.duration_s:.3f})"
        )


__all__: list[str] = ["AudioBuffer"]
