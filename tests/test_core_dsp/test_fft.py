"""Tests for app.core.dsp.fft: spectrum, sub-bin peak estimation and THD."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.dsp.fft import dominant_peak, estimate_thd, fft_spectrum

FS = 48_000  # with one second of signal every FFT bin is exactly 1 Hz wide


def sine(freq: float, *, fs: int = FS, seconds: float = 1.0, amp: float = 1.0) -> np.ndarray:
    t = np.arange(int(fs * seconds)) / fs
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------- fft_spectrum
def test_pure_sine_peak_is_at_the_right_frequency_and_height() -> None:
    freqs, mags = fft_spectrum(sine(3000.0, amp=0.8), FS)

    assert freqs[np.argmax(mags)] == pytest.approx(3000.0, abs=1.0)
    # |FFT| / N of a Hann-windowed sine is amplitude * mean(window) / 2 = amplitude / 4
    assert mags.max() == pytest.approx(0.8 / 4.0, rel=1e-2)


@pytest.mark.parametrize("window", ["hann", "hamming", "blackman-harris", "flat-top"])
def test_every_supported_window_finds_the_tone(window: str) -> None:
    freqs, mags = fft_spectrum(sine(5000.0), FS, window=window)

    assert freqs[np.argmax(mags)] == pytest.approx(5000.0, abs=1.0)


def test_spectrum_axes_are_single_sided_float32() -> None:
    x = sine(1000.0)

    freqs, mags = fft_spectrum(x, FS)

    assert freqs.shape == mags.shape == (x.size // 2 + 1,)
    assert freqs.dtype == mags.dtype == np.float32
    assert freqs[0] == 0.0
    assert freqs[-1] == pytest.approx(FS / 2)
    assert np.all(np.diff(freqs) > 0)
    assert np.all(mags >= 0)


def test_frequency_band_limits_are_inclusive() -> None:
    freqs, mags = fft_spectrum(sine(5000.0), FS, fmin_hz=2000.0, fmax_hz=8000.0)

    assert freqs.min() == pytest.approx(2000.0)
    assert freqs.max() == pytest.approx(8000.0)
    assert freqs.shape == mags.shape
    assert freqs[np.argmax(mags)] == pytest.approx(5000.0, abs=1.0)


def test_tone_outside_the_band_is_not_reported() -> None:
    freqs, mags = fft_spectrum(sine(5000.0), FS, fmin_hz=10_000.0)

    assert freqs.min() >= 10_000.0
    assert mags.max() < 1e-3  # only window leakage remains


def test_dc_signal_peaks_at_zero_hertz() -> None:
    freqs, mags = fft_spectrum(np.full(FS, 0.5, dtype=np.float32), FS)

    assert np.argmax(mags) == 0
    assert mags[0] == pytest.approx(0.5 * 0.5)  # level * mean(Hann window)
    assert np.all(mags[3:] < 1e-6)  # a Hann window leaks DC into the two neighbouring bins only


def test_empty_array_returns_empty_arrays() -> None:
    freqs, mags = fft_spectrum(np.array([]), FS)

    assert freqs.size == 0 and mags.size == 0


def test_single_sample_is_a_dc_value() -> None:
    freqs, mags = fft_spectrum(np.array([-0.25]), FS)

    assert freqs.tolist() == [0.0]
    assert mags.tolist() == [0.25]


def test_single_sample_outside_the_band_gives_nothing() -> None:
    freqs, mags = fft_spectrum(np.array([0.5]), FS, fmin_hz=100.0, fmax_hz=200.0)

    assert freqs.size == 0 and mags.size == 0


# --------------------------------------------------------------- dominant_peak
@pytest.mark.parametrize("offset", [0.1, 0.25, 0.4, 0.5])
def test_parabolic_interpolation_beats_the_bin_grid(offset: float) -> None:
    true_freq = 1000.0 + offset  # between two 1 Hz bins
    freqs, mags = fft_spectrum(sine(true_freq, fs=8000), 8000)

    raw_bin = float(freqs[np.argmax(mags)])
    peak_freq, peak_mag = dominant_peak(freqs, mags)

    assert abs(peak_freq - true_freq) < 0.1
    assert abs(peak_freq - true_freq) <= abs(raw_bin - true_freq) + 1e-9
    assert peak_mag >= mags.max()


def test_interpolated_peak_height_is_closer_to_the_true_amplitude() -> None:
    # A Hann window gives amplitude / 4 = 0.25 for a bin-centred tone.  Half a bin off, the
    # bin-grid maximum loses ~1.4 dB (scalloping); the parabola recovers most, not all, of it.
    freqs, mags = fft_spectrum(sine(1000.5, fs=8000, amp=1.0), 8000)

    _, peak_mag = dominant_peak(freqs, mags)

    assert abs(peak_mag - 0.25) < abs(float(mags.max()) - 0.25)
    assert peak_mag == pytest.approx(0.25, rel=0.1)


def test_dominant_peak_fmin_skips_lower_peaks() -> None:
    x = sine(500.0, amp=1.0) + sine(4000.0, amp=0.3)
    freqs, mags = fft_spectrum(x, FS)

    assert dominant_peak(freqs, mags)[0] == pytest.approx(500.0, abs=0.5)
    assert dominant_peak(freqs, mags, fmin_hz=1000.0)[0] == pytest.approx(4000.0, abs=0.5)


def test_dominant_peak_on_the_edge_is_not_interpolated() -> None:
    freqs = np.array([0.0, 10.0, 20.0, 30.0])
    mags = np.array([5.0, 1.0, 0.5, 0.2])

    assert dominant_peak(freqs, mags) == (0.0, 5.0)


@pytest.mark.parametrize(
    ("freqs", "mags", "kwargs"),
    [
        (np.array([]), np.array([]), {}),
        (np.array([0.0, 10.0]), np.array([1.0, 2.0]), {"fmin_hz": 100.0}),
    ],
)
def test_dominant_peak_with_nothing_to_search(freqs: np.ndarray, mags: np.ndarray, kwargs: dict) -> None:
    assert dominant_peak(freqs, mags, **kwargs) == (0.0, 0.0)


def test_dominant_peak_of_a_silent_spectrum_has_zero_height() -> None:
    _, height = dominant_peak(np.arange(5.0), np.zeros(5))

    assert height == 0.0


# ------------------------------------------------------------------ estimate_thd
def test_thd_of_a_known_harmonic_signal() -> None:
    x = sine(1000.0, amp=1.0) + sine(2000.0, amp=0.1) + sine(3000.0, amp=0.05)

    thd = estimate_thd(x, FS, 1000.0)

    assert thd == pytest.approx(np.sqrt(0.1**2 + 0.05**2), rel=0.02)  # ~0.1118


def test_thd_of_a_pure_sine_is_negligible() -> None:
    assert estimate_thd(sine(1000.0), FS, 1000.0) < 1e-3


def test_thd_counts_only_the_requested_number_of_harmonics() -> None:
    x = sine(1000.0) + sine(2000.0, amp=0.1) + sine(3000.0, amp=0.1)

    assert estimate_thd(x, FS, 1000.0, n_harmonics=2) == pytest.approx(0.1, rel=0.02)
    assert estimate_thd(x, FS, 1000.0, n_harmonics=3) == pytest.approx(np.sqrt(0.02), rel=0.02)


def test_thd_ignores_harmonics_above_nyquist() -> None:
    x = sine(10_000.0) + sine(20_000.0, amp=0.2)  # the 3rd harmonic would be at 30 kHz > 24 kHz

    assert estimate_thd(x, FS, 10_000.0, n_harmonics=6) == pytest.approx(0.2, rel=0.02)


@pytest.mark.parametrize(
    ("signal", "fundamental"),
    [
        (np.zeros(FS, dtype=np.float32), 1000.0),  # no fundamental present
        (sine(1000.0), 0.0),
        (sine(1000.0), -5.0),
        (sine(1000.0), FS / 2),  # at Nyquist
        (np.array([0.1, 0.2], dtype=np.float32), 1000.0),  # too short
    ],
)
def test_thd_returns_zero_for_unusable_input(signal: np.ndarray, fundamental: float) -> None:
    assert estimate_thd(signal, FS, fundamental) == 0.0
