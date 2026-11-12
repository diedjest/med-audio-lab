"""Tests for app.core.dsp.filters: zero-phase IIR filtering."""

from __future__ import annotations

import numpy as np
import pytest

from app.constants import FilterDesign, FilterType
from app.core.dsp.filters import FilterPipeline, FilterSpec

FS = 192_000


def tone(freq: float, *, seconds: float = 0.2, amp: float = 1.0) -> np.ndarray:
    t = np.arange(int(FS * seconds)) / FS
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def gain_db(filtered: np.ndarray, original: np.ndarray) -> float:
    """RMS change in dB over the middle half (the edges hold the filter's start-up transient)."""
    lo, hi = len(original) // 4, 3 * len(original) // 4
    out = float(np.sqrt(np.mean(filtered[lo:hi].astype(np.float64) ** 2)))
    ref = float(np.sqrt(np.mean(original[lo:hi].astype(np.float64) ** 2)))
    return 20.0 * np.log10(max(out, 1e-12) / ref)


@pytest.fixture(scope="module")
def pipeline() -> FilterPipeline:
    return FilterPipeline()


# --------------------------------------------------------------------- lowpass
def test_lowpass_attenuates_a_20khz_tone_by_more_than_40_db(pipeline: FilterPipeline) -> None:
    x = tone(20_000.0)

    y = pipeline.apply(x, FS, FilterSpec(FilterType.LOWPASS, 10_000.0))

    assert gain_db(y, x) < -40.0


def test_lowpass_keeps_a_tone_well_below_the_cutoff(pipeline: FilterPipeline) -> None:
    x = tone(2_000.0)

    y = pipeline.apply(x, FS, FilterSpec(FilterType.LOWPASS, 10_000.0))

    assert gain_db(y, x) == pytest.approx(0.0, abs=0.2)


def test_filtering_is_zero_phase(pipeline: FilterPipeline) -> None:
    x = tone(2_000.0)

    y = pipeline.apply(x, FS, FilterSpec(FilterType.LOWPASS, 10_000.0))

    middle = slice(len(x) // 4, 3 * len(x) // 4)
    assert np.corrcoef(x[middle], y[middle])[0, 1] > 0.9999  # no delay, no phase shift


@pytest.mark.parametrize(
    ("design", "ripple_db"),
    [
        (FilterDesign.BUTTERWORTH, 0.5),
        (FilterDesign.CHEBYSHEV_I, 0.5),
        (FilterDesign.CHEBYSHEV_II, 40.0),
    ],
)
def test_every_design_attenuates_the_stopband(pipeline: FilterPipeline, design: FilterDesign, ripple_db: float) -> None:
    x = tone(30_000.0)

    y = pipeline.apply(x, FS, FilterSpec(FilterType.LOWPASS, 10_000.0, order=6, design=design, ripple_db=ripple_db))

    assert gain_db(y, x) < -40.0


# -------------------------------------------------------------------- bandpass
def test_bandpass_passes_the_band_and_rejects_both_sides(pipeline: FilterPipeline) -> None:
    spec = FilterSpec(FilterType.BANDPASS, (20_000.0, 40_000.0), order=6)

    passband = pipeline.apply(tone(28_000.0), FS, spec)
    below = pipeline.apply(tone(5_000.0), FS, spec)
    above = pipeline.apply(tone(80_000.0), FS, spec)

    assert gain_db(passband, tone(28_000.0)) == pytest.approx(0.0, abs=1.0)
    assert gain_db(below, tone(5_000.0)) < -40.0
    assert gain_db(above, tone(80_000.0)) < -40.0


def test_highpass_and_bandstop(pipeline: FilterPipeline) -> None:
    high = FilterSpec(FilterType.HIGHPASS, 20_000.0, order=6)
    stop = FilterSpec(FilterType.BANDSTOP, (25_000.0, 35_000.0), order=6)

    assert gain_db(pipeline.apply(tone(2_000.0), FS, high), tone(2_000.0)) < -40.0
    assert gain_db(pipeline.apply(tone(60_000.0), FS, high), tone(60_000.0)) == pytest.approx(0.0, abs=0.5)
    assert gain_db(pipeline.apply(tone(30_000.0), FS, stop), tone(30_000.0)) < -30.0
    assert gain_db(pipeline.apply(tone(5_000.0), FS, stop), tone(5_000.0)) == pytest.approx(0.0, abs=0.5)


def test_apply_chain_equals_applying_the_stages_in_turn(pipeline: FilterPipeline) -> None:
    x = tone(5_000.0) + tone(50_000.0, amp=0.5)
    first = FilterSpec(FilterType.LOWPASS, 20_000.0)
    second = FilterSpec(FilterType.HIGHPASS, 1_000.0)

    chained = pipeline.apply_chain(x, FS, [first, second])
    manual = pipeline.apply(pipeline.apply(x, FS, first), FS, second)

    np.testing.assert_allclose(chained, manual, atol=1e-5)


# -------------------------------------------------------------- input handling
@pytest.mark.parametrize("size", [0, 1, 11])
def test_signal_too_short_for_the_order_raises(pipeline: FilterPipeline, size: int) -> None:
    with pytest.raises(ValueError, match="too short"):
        pipeline.apply(np.zeros(size, dtype=np.float32), FS, FilterSpec(FilterType.LOWPASS, 10_000.0, order=4))


def test_signal_slightly_above_the_minimum_still_raises_value_error(pipeline: FilterPipeline) -> None:
    # 12-15 samples pass our own check but are shorter than the zero-phase padding SciPy needs
    with pytest.raises(ValueError):
        pipeline.apply(np.zeros(13, dtype=np.float32), FS, FilterSpec(FilterType.LOWPASS, 10_000.0, order=4))


@pytest.mark.parametrize("kind", [FilterType.LOWPASS, FilterType.HIGHPASS])
def test_cutoff_at_nyquist_does_not_crash(pipeline: FilterPipeline, kind: FilterType) -> None:
    x = tone(5_000.0)

    y = pipeline.apply(x, FS, FilterSpec(kind, FS / 2.0))

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_lowpass_at_nyquist_leaves_the_signal_alone(pipeline: FilterPipeline) -> None:
    x = tone(5_000.0)

    y = pipeline.apply(x, FS, FilterSpec(FilterType.LOWPASS, FS / 2.0))

    assert gain_db(y, x) == pytest.approx(0.0, abs=0.1)


def test_cutoff_above_nyquist_is_clamped(pipeline: FilterPipeline) -> None:
    y = pipeline.apply(tone(5_000.0), FS, FilterSpec(FilterType.LOWPASS, 150_000.0))

    assert np.all(np.isfinite(y))


def test_band_that_collapses_above_nyquist_is_rejected(pipeline: FilterPipeline) -> None:
    with pytest.raises(ValueError, match="must exceed"):
        pipeline.apply(tone(5_000.0), FS, FilterSpec(FilterType.BANDPASS, (100_000.0, 120_000.0)))


def test_non_positive_cutoff_is_rejected(pipeline: FilterPipeline) -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        pipeline.apply(tone(5_000.0), FS, FilterSpec(FilterType.LOWPASS, 0.0))


def test_dtype_and_length_are_preserved(pipeline: FilterPipeline) -> None:
    x32 = tone(5_000.0)

    assert pipeline.apply(x32, FS, FilterSpec(FilterType.LOWPASS, 10_000.0)).dtype == np.float32
    assert pipeline.apply(x32.astype(np.float64), FS, FilterSpec(FilterType.LOWPASS, 10_000.0)).dtype == np.float64
    assert pipeline.apply(x32, FS, FilterSpec(FilterType.LOWPASS, 10_000.0)).shape == x32.shape


def test_frequency_response_is_minus_3_db_at_the_butterworth_cutoff(pipeline: FilterPipeline) -> None:
    freqs, magnitude_db = pipeline.frequency_response(FilterSpec(FilterType.LOWPASS, 10_000.0), FS)

    at_cutoff = magnitude_db[np.argmin(np.abs(freqs - 10_000.0))]
    assert at_cutoff == pytest.approx(-3.01, abs=0.05)
    assert magnitude_db[0] == pytest.approx(0.0, abs=0.01)
    assert freqs[-1] == pytest.approx(FS / 2.0, rel=0.01)
    assert freqs.dtype == magnitude_db.dtype == np.float64
