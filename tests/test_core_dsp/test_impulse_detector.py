"""Tests for app.core.dsp.impulse_detector: adaptive-threshold burst detection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.core.dsp.impulse_detector import ImpulseDetector
from app.core.models.impulse import DetectedImpulse, ImpulseDetectorConfig

FS = 192_000
BURST_FREQ = 40_000.0
BURST_CENTRES_S = (0.2, 0.5, 0.8)
BURST_SAMPLES = 240  # ~1.25 ms
NOISE_STD = 0.002


def burst(freq: float = BURST_FREQ, *, samples: int = BURST_SAMPLES, amp: float = 1.0) -> np.ndarray:
    """A tone burst with a Gaussian envelope."""
    n = np.arange(samples)
    envelope = np.exp(-0.5 * ((n - samples / 2) / (samples / 6)) ** 2)
    return amp * envelope * np.sin(2 * np.pi * freq * n / FS)


def pulse_train(
    centres_s: tuple[float, ...] = BURST_CENTRES_S,
    *,
    seconds: float = 1.0,
    freq: float = BURST_FREQ,
    noise_std: float = NOISE_STD,
) -> np.ndarray:
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, noise_std, int(FS * seconds))
    for centre in centres_s:
        start = int(centre * FS) - BURST_SAMPLES // 2
        x[start : start + BURST_SAMPLES] += burst(freq)
    return x.astype(np.float32)


@pytest.fixture(scope="module")
def detected() -> list[DetectedImpulse]:
    return ImpulseDetector().detect(pulse_train(), FS)


# ------------------------------------------------------------------ pulse train
def test_three_bursts_in_noise_give_three_impulses(detected: list[DetectedImpulse]) -> None:
    assert len(detected) == 3
    assert [i.index for i in detected] == [1, 2, 3]


def test_peak_times_are_within_one_millisecond_of_the_burst_centres(detected: list[DetectedImpulse]) -> None:
    for impulse, centre in zip(detected, BURST_CENTRES_S):
        assert impulse.peak_time_s == pytest.approx(centre, abs=1e-3)


def test_dominant_frequencies_are_within_five_percent(detected: list[DetectedImpulse]) -> None:
    for impulse in detected:
        assert impulse.dominant_freq_hz == pytest.approx(BURST_FREQ, rel=0.05)


def test_impulse_geometry_is_consistent(detected: list[DetectedImpulse]) -> None:
    for impulse in detected:
        assert impulse.start_sample < impulse.end_sample
        assert impulse.start_s < impulse.peak_time_s < impulse.end_s
        assert impulse.start_s == pytest.approx(impulse.start_sample / FS)
        assert impulse.end_s == pytest.approx(impulse.end_sample / FS)
        assert impulse.duration_ms == pytest.approx((impulse.end_sample - impulse.start_sample) / FS * 1000.0)
        assert impulse.duration_ms >= 1.0  # at least the burst itself
        assert impulse.peak_amplitude == pytest.approx(1.0, rel=0.1)
        assert 0.0 < impulse.rms <= impulse.peak_amplitude
        assert 0.0 < impulse.threshold_used <= 0.85


def test_intervals_between_bursts(detected: list[DetectedImpulse]) -> None:
    assert math.isnan(detected[0].interval_ms)
    assert detected[1].interval_ms == pytest.approx(300.0, abs=1.0)
    assert detected[2].interval_ms == pytest.approx(300.0, abs=1.0)


def test_impulses_come_back_in_time_order(detected: list[DetectedImpulse]) -> None:
    starts = [i.start_s for i in detected]

    assert starts == sorted(starts)
    assert all(a.end_sample <= b.start_sample for a, b in zip(detected, detected[1:]))


def test_detection_is_deterministic() -> None:
    x = pulse_train()

    first = ImpulseDetector().detect(x, FS)
    second = ImpulseDetector().detect(x.copy(), FS)

    assert [(i.start_sample, i.end_sample) for i in first] == [(i.start_sample, i.end_sample) for i in second]


def test_a_different_burst_frequency_is_measured() -> None:
    impulses = ImpulseDetector().detect(pulse_train(freq=25_000.0), FS)

    assert len(impulses) == 3
    for impulse in impulses:
        assert impulse.dominant_freq_hz == pytest.approx(25_000.0, rel=0.05)


def test_dc_offset_does_not_change_the_result(detected: list[DetectedImpulse]) -> None:
    shifted = ImpulseDetector().detect(pulse_train() + np.float32(0.3), FS)

    assert [i.peak_time_s for i in shifted] == pytest.approx([i.peak_time_s for i in detected], abs=1e-4)


# ------------------------------------------------------------- nothing to find
@pytest.mark.parametrize(
    "signal",
    [
        np.zeros(FS, dtype=np.float32),
        np.full(FS, 0.4, dtype=np.float32),  # constant level: silence once DC is removed
        np.zeros(5, dtype=np.float32),  # too short to analyse
        np.array([], dtype=np.float32),
    ],
    ids=["silence", "dc-only", "tiny", "empty"],
)
def test_signals_without_bursts_give_an_empty_list(signal: np.ndarray) -> None:
    assert ImpulseDetector().detect(signal, FS) == []


def test_continuous_tone_is_not_chopped_into_hundreds_of_impulses() -> None:
    t = np.arange(FS) / FS
    tone = (0.5 * np.sin(2 * np.pi * BURST_FREQ * t)).astype(np.float32)

    impulses = ImpulseDetector().detect(tone, FS)

    assert len(impulses) <= 1


def test_noise_alone_does_not_produce_a_flood_of_detections() -> None:
    noise = np.random.default_rng(42).normal(0.0, 0.1, FS).astype(np.float32)

    assert len(ImpulseDetector().detect(noise, FS)) < 20


# ----------------------------------------------------------------- configuration
def test_config_override_for_one_call_does_not_change_the_detector() -> None:
    detector = ImpulseDetector()
    x = pulse_train()
    strict = ImpulseDetectorConfig(min_duration_ms=50.0)  # bursts last ~1-2 ms

    assert detector.detect(x, FS, strict) == []
    assert len(detector.detect(x, FS)) == 3


def test_config_property_is_used_and_replaceable() -> None:
    detector = ImpulseDetector(ImpulseDetectorConfig(min_duration_ms=50.0))
    x = pulse_train()
    assert detector.detect(x, FS) == []

    detector.config = ImpulseDetectorConfig()

    assert len(detector.detect(x, FS)) == 3


def test_merge_gap_joins_close_bursts() -> None:
    close = pulse_train((0.2, 0.2 + 0.0035))  # 3.5 ms apart: ~2.2 ms between the envelopes
    separate = ImpulseDetector(ImpulseDetectorConfig(merge_gap_ms=0.0)).detect(close, FS)
    merged = ImpulseDetector(ImpulseDetectorConfig(merge_gap_ms=20.0)).detect(close, FS)

    assert len(separate) >= 2
    assert len(merged) == 1


def test_dominant_frequency_is_undetermined_outside_the_search_band() -> None:
    config = ImpulseDetectorConfig(fmin_hz=100_000.0)  # above Nyquist: empty band

    impulses = ImpulseDetector().detect(pulse_train(), FS, config)

    assert len(impulses) == 3
    assert all(math.isnan(i.dominant_freq_hz) for i in impulses)


def test_default_configuration_matches_the_legacy_values() -> None:
    config = ImpulseDetectorConfig()

    assert (config.threshold_db, config.min_duration_ms, config.merge_gap_ms) == (-35.0, 0.4, 1.5)
    assert (config.pre_pad_ms, config.post_pad_ms, config.envelope_ms) == (0.15, 0.35, 0.35)
    assert (config.fmin_hz, config.fmax_hz, config.noise_factor) == (5000.0, None, 6.0)
