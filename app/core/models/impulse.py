"""Data models for detected ultrasonic pulses and detector configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectedImpulse:
    """Immutable data class for a single detected pulse."""

    index: int
    start_sample: int
    end_sample: int
    start_s: float
    end_s: float
    duration_ms: float
    peak_time_s: float
    peak_amplitude: float
    rms: float
    dominant_freq_hz: float  # NaN if undetermined
    interval_ms: float  # NaN for the first impulse
    threshold_used: float


@dataclass
class ImpulseDetectorConfig:
    """Mutable configuration for impulse detection (defaults from legacy code)."""

    threshold_db: float = -35.0
    min_duration_ms: float = 0.4
    merge_gap_ms: float = 1.5
    pre_pad_ms: float = 0.15
    post_pad_ms: float = 0.35
    envelope_ms: float = 0.35
    fmin_hz: float = 5000.0
    fmax_hz: float | None = None
    noise_factor: float = 6.0


__all__: list[str] = [
    "DetectedImpulse",
    "ImpulseDetectorConfig",
]
