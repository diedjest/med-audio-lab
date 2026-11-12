"""Tests for app.core.dsp.cwt: the Morlet CWT engine, its slices and the ridge."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.dsp.cwt import (
    CwtEngine,
    CwtResult,
    lateral_slice,
    longitudinal_slice,
    ridge_detection,
)

FS = 192_000
FMIN, FMAX = 5_000.0, 96_000.0


def tone(freq: float, *, seconds: float = 0.25, amp: float = 1.0) -> np.ndarray:
    t = np.arange(int(FS * seconds)) / FS
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def chirp(f0: float, f1: float, *, seconds: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Linear chirp and its instantaneous frequency at every sample."""
    t = np.arange(int(FS * seconds)) / FS
    rate = (f1 - f0) / seconds
    return np.sin(2 * np.pi * (f0 * t + 0.5 * rate * t**2)).astype(np.float32), f0 + rate * t


@pytest.fixture(scope="module")
def engine() -> CwtEngine:
    return CwtEngine(chunk_scales=4)


@pytest.fixture(scope="module")
def tone_10k(engine: CwtEngine) -> CwtResult:
    return engine.compute(tone(10_000.0), FS, fmin_hz=FMIN, fmax_hz=FMAX)


def central(values: np.ndarray, lo: float = 0.25, hi: float = 0.75) -> np.ndarray:
    """The middle of a time series, away from the transform's edge effects."""
    n = values.size
    return values[int(lo * n) : int(hi * n)]


# --------------------------------------------------------------------- compute
def test_10khz_sine_ridge_sits_on_10khz(engine: CwtEngine, tone_10k: CwtResult) -> None:
    _, ridge_hz = engine.ridge_detection(tone_10k)

    # the scale grid is ~8 % apart, so the nearest row is at most ~4 % away
    assert float(np.median(central(ridge_hz))) == pytest.approx(10_000.0, rel=0.05)
    assert np.all(central(ridge_hz) > 8_000.0) and np.all(central(ridge_hz) < 12_500.0)


def test_result_shapes_and_frequency_band(tone_10k: CwtResult) -> None:
    n_scales, n_time = tone_10k.coefficients.shape

    assert tone_10k.fs == FS
    assert n_time == tone_10k.time.size == int(FS * 0.25)
    assert tone_10k.frequencies_hz.shape == tone_10k.scales.shape == (n_scales,)
    assert tone_10k.coefficients.dtype == np.complex64
    assert tone_10k.frequencies_hz.min() >= FMIN and tone_10k.frequencies_hz.max() <= FMAX
    assert tone_10k.time[0] == 0.0 and tone_10k.time[-1] == pytest.approx((n_time - 1) / FS)


def test_result_derived_views_and_immutability(tone_10k: CwtResult) -> None:
    assert tone_10k.amplitude.shape == tone_10k.power.shape == tone_10k.phase.shape == tone_10k.coefficients.shape
    np.testing.assert_allclose(tone_10k.power, tone_10k.amplitude**2, rtol=1e-5)
    assert tone_10k.phase.max() <= np.pi and tone_10k.phase.min() >= -np.pi
    for array in (tone_10k.time, tone_10k.frequencies_hz, tone_10k.coefficients, tone_10k.scales):
        assert not array.flags.writeable


def test_progress_callback_reports_every_chunk_up_to_one(engine: CwtEngine) -> None:
    updates: list[tuple[float, str]] = []

    engine.compute(tone(10_000.0, seconds=0.05), FS, fmin_hz=FMIN, fmax_hz=FMAX, progress_callback=lambda f, m: updates.append((f, m)))

    fractions = [f for f, _ in updates]
    assert len(updates) > 3
    assert fractions == sorted(fractions) and fractions[-1] == pytest.approx(1.0)


@pytest.mark.parametrize("size", [0, 1, 7])
def test_too_short_signal_raises_value_error(engine: CwtEngine, size: int) -> None:
    with pytest.raises(ValueError, match="too short"):
        engine.compute(np.zeros(size, dtype=np.float32), FS)


def test_band_without_scales_raises_value_error(engine: CwtEngine) -> None:
    with pytest.raises(ValueError, match="No scales"):
        engine.compute(tone(10_000.0, seconds=0.05), FS, fmin_hz=5e6, fmax_hz=6e6)


def test_memory_budget_is_enforced_before_computing() -> None:
    small = CwtEngine(memory_budget_mb=1.0)

    with pytest.raises(MemoryError, match="budget"):
        small.compute(tone(10_000.0), FS, fmin_hz=FMIN, fmax_hz=FMAX)


def test_oversized_default_request_is_refused() -> None:
    huge = np.zeros(4_000_000, dtype=np.float32)  # ~21 s at 192 kHz: several GB of coefficients

    with pytest.raises(MemoryError):
        CwtEngine().compute(huge, FS, fmin_hz=FMIN, fmax_hz=FMAX)


def test_memory_estimate_grows_with_the_problem_size() -> None:
    engine = CwtEngine()

    base = engine.estimate_memory_mb(50_000, 100)
    assert engine.estimate_memory_mb(100_000, 100) > base
    assert engine.estimate_memory_mb(50_000, 200) > base
    assert CwtEngine(chunk_scales=16).estimate_memory_mb(50_000, 100) > base


def test_target_fs_max_seconds_and_stride_shape_the_result(engine: CwtEngine, tone_10k: CwtResult) -> None:
    result = engine.compute(
        tone(10_000.0), FS, fmin_hz=FMIN, fmax_hz=FMAX, target_fs=96_000, max_seconds=0.1, scale_stride=3
    )

    assert result.fs == 96_000
    assert result.coefficients.shape[1] == int(0.1 * 96_000)
    assert result.coefficients.shape[0] < tone_10k.coefficients.shape[0] / 2


def test_estimate_load_agrees_with_the_budget_decision() -> None:
    engine = CwtEngine(memory_budget_mb=30.0)

    ok = engine.estimate_load(10_000, FS, fmin_hz=FMIN, fmax_hz=FMAX)
    too_big = engine.estimate_load(200_000, FS, fmin_hz=FMIN, fmax_hz=FMAX)

    assert ok["fits_budget"] is True and "suggested_max_seconds" not in ok
    assert too_big["fits_budget"] is False
    assert 0.0 < too_big["suggested_max_seconds"] < 200_000 / FS
    fit = engine.estimate_load(
        200_000, FS, fmin_hz=FMIN, fmax_hz=FMAX, max_seconds=too_big["suggested_max_seconds"]
    )
    assert fit["fits_budget"] is True


# ---------------------------------------------------------------------- slices
def test_lateral_slice_is_a_spectrum_peaking_at_the_tone(engine: CwtEngine, tone_10k: CwtResult) -> None:
    freqs, amplitude = engine.lateral_slice(tone_10k, 0.125)

    assert freqs.shape == amplitude.shape == (tone_10k.coefficients.shape[0],)
    assert amplitude.dtype == np.float64 and np.all(amplitude >= 0)
    assert freqs[np.argmax(amplitude)] == pytest.approx(10_000.0, rel=0.05)
    assert freqs is not tone_10k.frequencies_hz  # a copy, safe to modify


def test_lateral_slice_uses_the_nearest_time_sample(tone_10k: CwtResult) -> None:
    _, at_time = lateral_slice(tone_10k, 0.1)
    index = int(np.argmin(np.abs(tone_10k.time - 0.1)))

    np.testing.assert_allclose(at_time, np.abs(tone_10k.coefficients[:, index]))


def test_longitudinal_slice_is_normalized_and_centred(engine: CwtEngine, tone_10k: CwtResult) -> None:
    time_rel, values = engine.longitudinal_slice(tone_10k, 10_000.0)

    assert time_rel.shape == values.shape == (tone_10k.time.size,)
    assert np.max(np.abs(values)) == pytest.approx(1.0)
    assert time_rel[np.argmax(np.abs(values))] == pytest.approx(0.0)  # centred on the strongest point
    assert np.all(np.diff(time_rel) > 0)


def test_longitudinal_slice_picks_the_row_nearest_the_frequency(tone_10k: CwtResult) -> None:
    _, values = longitudinal_slice(tone_10k, 10_000.0)
    row = int(np.argmin(np.abs(tone_10k.frequencies_hz - 10_000.0)))
    expected = tone_10k.coefficients[row].real.astype(np.float64)
    expected /= np.max(np.abs(expected))

    np.testing.assert_allclose(values, expected)


def test_longitudinal_slice_of_a_zero_row_does_not_divide_by_zero() -> None:
    result = CwtResult(
        fs=FS,
        time=np.arange(8) / FS,
        frequencies_hz=np.array([1000.0, 2000.0]),
        coefficients=np.zeros((2, 8), dtype=np.complex64),
        scales=np.array([2.0, 1.0]),
    )

    _, values = longitudinal_slice(result, 1000.0)

    assert np.all(values == 0.0)


# ----------------------------------------------------------------------- ridge
def test_ridge_follows_a_chirp(engine: CwtEngine) -> None:
    signal, true_hz = chirp(10_000.0, 50_000.0)
    result = engine.compute(signal, FS, fmin_hz=FMIN, fmax_hz=FMAX)

    time, ridge_hz = engine.ridge_detection(result)
    measured, expected = central(ridge_hz, 0.2, 0.8), central(true_hz, 0.2, 0.8)

    assert time.shape == ridge_hz.shape == (signal.size,)
    assert np.corrcoef(measured, expected)[0, 1] > 0.95
    assert float(np.mean(np.abs(measured - expected) / expected)) < 0.12
    # the sweep passes ~20 kHz at 25 % of the signal and ~40 kHz at 75 %
    assert np.mean(central(ridge_hz, 0.7, 0.8)) > 1.7 * np.mean(central(ridge_hz, 0.2, 0.3))


def test_ridge_returns_copies(tone_10k: CwtResult) -> None:
    time, ridge_hz = ridge_detection(tone_10k)

    time[0] = -1.0
    ridge_hz[0] = -1.0
    assert tone_10k.time[0] == 0.0
    assert tone_10k.frequencies_hz.min() > 0
