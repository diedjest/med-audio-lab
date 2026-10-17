"""Tests for app.core.dsp.metrics: signal statistics, autocorrelation and formatting."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from app.core.dsp.metrics import autocorrelation, compute_signal_metrics, format_metrics
from app.core.models.analysis_result import SignalMetrics

FS = 48_000


def sine(freq: float, *, amp: float = 1.0, seconds: float = 1.0, fs: int = FS) -> np.ndarray:
    t = np.arange(int(fs * seconds)) / fs
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --------------------------------------------------------------------- unit sine
def test_unit_sine_level_statistics() -> None:
    m = compute_signal_metrics(sine(1000.0), FS)

    assert m.rms == pytest.approx(1 / math.sqrt(2), rel=1e-3)  # 0.7071
    assert m.peak == pytest.approx(1.0, rel=1e-4)
    assert m.rms_dbfs == pytest.approx(-3.01, abs=0.02)
    assert m.peak_dbfs == pytest.approx(0.0, abs=0.01)


def test_unit_sine_bookkeeping_fields() -> None:
    m = compute_signal_metrics(sine(1000.0), FS)

    assert m.duration_s == pytest.approx(1.0)
    assert m.sample_rate == FS
    assert m.n_samples == FS


def test_unit_sine_spectral_statistics() -> None:
    m = compute_signal_metrics(sine(1000.0), FS)

    assert m.dominant_freq_hz == pytest.approx(1000.0, abs=1.0)
    assert m.spectral_centroid_hz == pytest.approx(1000.0, abs=20.0)
    assert m.spectral_bandwidth_hz < 50.0  # a pure tone is narrow-band
    assert m.spectral_rolloff_hz == pytest.approx(1000.0, abs=5.0)
    assert m.thd < 1e-3


def test_unit_sine_zero_crossing_rate() -> None:
    m = compute_signal_metrics(sine(1000.0), FS)

    # two crossings per 48-sample period, counted over consecutive sample pairs
    assert m.zero_crossing_rate == pytest.approx(2 / 48, rel=0.03)


@pytest.mark.parametrize("amp", [0.1, 0.5, 2.0])
def test_levels_scale_with_amplitude(amp: float) -> None:
    m = compute_signal_metrics(sine(1000.0, amp=amp), FS)

    assert m.peak == pytest.approx(amp, rel=1e-3)
    assert m.rms == pytest.approx(amp / math.sqrt(2), rel=1e-3)
    assert m.peak_dbfs == pytest.approx(20 * math.log10(amp), abs=0.02)


def test_dominant_frequency_of_a_high_frequency_tone() -> None:
    m = compute_signal_metrics(sine(40_000.0, fs=192_000, seconds=0.5), 192_000)

    assert m.dominant_freq_hz == pytest.approx(40_000.0, abs=5.0)


def test_harmonic_distortion_is_measured() -> None:
    x = sine(1000.0) + sine(2000.0, amp=0.1)

    assert compute_signal_metrics(x, FS).thd == pytest.approx(0.1, rel=0.05)


# ------------------------------------------------------------------------ DC
def test_dc_signal_levels() -> None:
    m = compute_signal_metrics(np.full(FS, 0.5, dtype=np.float32), FS)

    assert m.rms == pytest.approx(0.5)
    assert m.peak == pytest.approx(0.5)
    assert m.rms_dbfs == pytest.approx(-6.02, abs=0.02)
    assert m.zero_crossing_rate == 0.0


def test_negative_dc_signal_has_the_same_levels() -> None:
    m = compute_signal_metrics(np.full(FS, -0.5, dtype=np.float32), FS)

    assert m.rms == pytest.approx(0.5)
    assert m.peak == pytest.approx(0.5)
    assert m.zero_crossing_rate == 0.0


def test_dc_signal_has_no_spectral_content() -> None:
    m = compute_signal_metrics(np.full(FS, 0.5, dtype=np.float32), FS)

    assert m.dominant_freq_hz == 0.0
    assert m.spectral_centroid_hz == 0.0
    assert m.spectral_bandwidth_hz == 0.0
    assert m.thd == 0.0


def test_a_dc_offset_does_not_disturb_the_spectral_statistics() -> None:
    plain = compute_signal_metrics(sine(1000.0, amp=0.2), FS)
    offset = compute_signal_metrics(sine(1000.0, amp=0.2) + np.float32(0.7), FS)

    assert offset.dominant_freq_hz == pytest.approx(plain.dominant_freq_hz, abs=1.0)
    assert offset.spectral_centroid_hz == pytest.approx(plain.spectral_centroid_hz, abs=5.0)
    assert offset.thd == pytest.approx(plain.thd, abs=1e-3)
    assert offset.peak == pytest.approx(0.9, rel=1e-3)  # level statistics do include the offset


# ------------------------------------------------------------------- silence
def test_silence_has_zero_levels_and_a_floor_in_decibels() -> None:
    m = compute_signal_metrics(np.zeros(FS, dtype=np.float32), FS)

    assert m.rms == 0.0
    assert m.peak == 0.0
    assert m.rms_dbfs <= -120.0 and math.isfinite(m.rms_dbfs)
    assert m.peak_dbfs <= -120.0 and math.isfinite(m.peak_dbfs)
    assert m.zero_crossing_rate == 0.0


def test_silence_has_no_dominant_frequency() -> None:
    m = compute_signal_metrics(np.zeros(FS, dtype=np.float32), FS)

    assert m.dominant_freq_hz == 0.0
    assert m.spectral_centroid_hz == 0.0
    assert m.spectral_bandwidth_hz == 0.0
    assert m.spectral_rolloff_hz == 0.0
    assert m.thd == 0.0


def test_empty_signal_and_invalid_sample_rate_give_zeroed_metrics() -> None:
    for m in (compute_signal_metrics(np.array([]), FS), compute_signal_metrics(sine(100.0), 0)):
        assert m.n_samples == 0 and m.duration_s == 0.0
        assert m.rms == 0.0 and m.peak == 0.0
        assert m.rms_dbfs == m.peak_dbfs == -120.0


def test_single_sample_signal_does_not_crash() -> None:
    m = compute_signal_metrics(np.array([0.5], dtype=np.float32), FS)

    assert m.n_samples == 1
    assert m.peak == pytest.approx(0.5)
    assert m.dominant_freq_hz == 0.0


def test_noise_is_broadband() -> None:
    noise = np.random.default_rng(42).normal(0.0, 0.1, FS).astype(np.float32)

    m = compute_signal_metrics(noise, FS)

    assert m.rms == pytest.approx(0.1, rel=0.02)
    assert m.spectral_centroid_hz == pytest.approx(FS / 4, rel=0.05)  # flat spectrum: centre of 0..Nyquist
    assert m.spectral_bandwidth_hz > 5_000.0
    assert m.zero_crossing_rate == pytest.approx(0.5, abs=0.02)


def test_metrics_are_immutable() -> None:
    m = compute_signal_metrics(sine(1000.0), FS)

    with pytest.raises(dataclasses.FrozenInstanceError):
        m.rms = 1.0  # type: ignore[misc]


# ----------------------------------------------------------------- autocorrelation
def test_autocorrelation_starts_at_one_and_peaks_at_the_period() -> None:
    lags_ms, corr = autocorrelation(sine(1000.0, seconds=0.2), FS, max_lag_ms=5.0)

    assert corr[0] == pytest.approx(1.0)
    assert lags_ms[0] == 0.0
    assert lags_ms[np.argmax(corr[10:]) + 10] == pytest.approx(1.0, abs=0.05)  # 1 kHz -> 1 ms period
    assert corr.max() <= 1.0 + 1e-6 and corr.min() >= -1.0 - 1e-6


def test_autocorrelation_limits_the_lag_range() -> None:
    lags_ms, corr = autocorrelation(sine(1000.0, seconds=0.2), FS, max_lag_ms=10.0)

    assert lags_ms.shape == corr.shape
    assert lags_ms[-1] == pytest.approx(10.0, abs=0.05)
    assert lags_ms.dtype == corr.dtype == np.float32


def test_autocorrelation_of_noise_decorrelates_immediately() -> None:
    noise = np.random.default_rng(42).normal(size=20_000).astype(np.float32)

    _, corr = autocorrelation(noise, FS, max_lag_ms=5.0)

    assert corr[0] == pytest.approx(1.0)
    assert np.max(np.abs(corr[5:])) < 0.1


def test_autocorrelation_of_silence_and_empty_input() -> None:
    _, silent = autocorrelation(np.zeros(1000, dtype=np.float32), FS, max_lag_ms=5.0)
    lags, empty = autocorrelation(np.array([]), FS)

    assert np.all(silent == 0.0)
    assert lags.size == 0 and empty.size == 0


# ----------------------------------------------------------------------- formatting
def test_format_metrics_labels_and_units() -> None:
    m = SignalMetrics(
        duration_s=1.5, sample_rate=48000, n_samples=72000, rms=0.25, peak=0.5, rms_dbfs=-12.04,
        peak_dbfs=-6.02, zero_crossing_rate=0.0417, dominant_freq_hz=1000.0, spectral_centroid_hz=1010.5,
        spectral_bandwidth_hz=20.0, spectral_rolloff_hz=1100.0, thd=0.05,
    )

    text = format_metrics(m)

    assert list(text) == [
        "Duration", "Sample rate", "Samples", "RMS", "Peak", "RMS level", "Peak level",
        "Zero crossing rate", "Dominant frequency", "Spectral centroid", "Spectral bandwidth",
        "Roll-off 85%", "THD",
    ]
    assert text["Duration"] == "1.500 s"
    assert text["Sample rate"] == "48000 Hz"
    assert text["RMS level"] == "-12.0 dBFS"
    assert text["Dominant frequency"] == "1000.0 Hz"
    assert text["THD"] == "5.00 %"
