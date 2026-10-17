"""Tests for app.infrastructure.storage.file_io: WAV, CSV and TXT input / output."""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.core.models.impulse import DetectedImpulse
from app.infrastructure.storage import file_io

FS = 48_000


def tone(freq: float = 1000.0, *, amp: float = 0.5, seconds: float = 0.25, fs: int = FS) -> np.ndarray:
    t = np.arange(int(fs * seconds)) / fs
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def temp_path() -> Iterator[Callable[[str], Path]]:
    """Hand out paths of real temporary files (created with ``NamedTemporaryFile``) and clean up.

    The files are closed before the test touches them: on Windows an open
    ``NamedTemporaryFile`` cannot be opened a second time.
    """
    created: list[str] = []

    def make(suffix: str) -> Path:
        handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        handle.close()
        created.append(handle.name)
        return Path(handle.name)

    yield make
    for name in created:
        try:
            os.unlink(name)
        except OSError:
            pass


# -------------------------------------------------------------- WAV roundtrip
@pytest.mark.parametrize(
    ("subtype", "tolerance"),
    [("PCM_16", 1 / 32768), ("PCM_24", 1 / 8_388_608), ("FLOAT_32", 0.0)],
)
def test_save_and_reload_roundtrip(temp_path: Callable[[str], Path], subtype: str, tolerance: float) -> None:
    path = temp_path(".wav")
    x = tone()

    file_io.save_wav(x, FS, path, subtype)
    rate, y = file_io.read_wav(path, normalize=True, keep_float_level=True)

    assert rate == FS
    assert y.dtype == np.float32 and y.shape == x.shape
    assert np.max(np.abs(y - x)) <= tolerance + 1e-7


@pytest.mark.parametrize(
    ("subtype", "expected"),
    [("PCM_16", "PCM_16"), ("PCM_24", "PCM_24"), ("FLOAT_32", "FLOAT"), ("FLOAT", "FLOAT")],
)
def test_written_files_have_the_requested_subtype(temp_path: Callable[[str], Path], subtype: str, expected: str) -> None:
    path = temp_path(".wav")

    file_io.save_wav(tone(), FS, path, subtype)

    info = sf.info(str(path))
    assert info.subtype == expected
    assert info.samplerate == FS and info.channels == 1 and info.format == "WAV"


def test_default_subtype_is_16_bit(temp_path: Callable[[str], Path]) -> None:
    path = temp_path(".wav")

    file_io.save_wav(tone(), FS, path)

    assert sf.info(str(path)).subtype == "PCM_16"


def test_save_creates_missing_directories(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c.wav"

    file_io.save_wav(tone(), FS, target)

    assert target.is_file()


# ------------------------------------------------------------ reading subtypes
@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24"])
def test_integer_pcm_is_scaled_to_full_scale(temp_path: Callable[[str], Path], subtype: str) -> None:
    path = temp_path(".wav")
    file_io.save_wav(tone(amp=0.5), FS, path, subtype)

    _, y = file_io.read_wav(path)  # normalize=True

    assert float(np.max(np.abs(y))) == pytest.approx(0.5, abs=1e-3)  # scaled by full scale, not peak-normalized
    assert y.dtype == np.float32


def test_float32_wav_is_peak_normalized_by_default(temp_path: Callable[[str], Path]) -> None:
    path = temp_path(".wav")
    file_io.save_wav(tone(amp=0.25), FS, path, "FLOAT_32")

    _, y = file_io.read_wav(path)

    assert float(np.max(np.abs(y))) == pytest.approx(1.0)


def test_keep_float_level_preserves_float_amplitudes(temp_path: Callable[[str], Path]) -> None:
    path = temp_path(".wav")
    loud = tone(amp=3.0)  # above full scale: only a float file can hold it
    file_io.save_wav(loud, FS, path, "FLOAT_32")

    _, y = file_io.read_wav(path, keep_float_level=True)

    np.testing.assert_allclose(y, loud, atol=1e-6)


def test_normalize_false_returns_the_stored_integer_values(temp_path: Callable[[str], Path]) -> None:
    path = temp_path(".wav")
    file_io.save_wav(tone(amp=0.5), FS, path, "PCM_16")

    _, raw = file_io.read_wav(path, normalize=False)

    assert float(np.max(np.abs(raw))) == pytest.approx(0.5 * 32767, rel=1e-3)
    assert np.all(raw == np.round(raw))  # whole numbers: the stored 16-bit values


def test_normalize_false_leaves_floats_alone(temp_path: Callable[[str], Path]) -> None:
    path = temp_path(".wav")
    file_io.save_wav(tone(amp=0.25), FS, path, "FLOAT_32")

    _, y = file_io.read_wav(path, normalize=False)

    assert float(np.max(np.abs(y))) == pytest.approx(0.25, rel=1e-3)


def test_stereo_files_are_reduced_to_the_first_channel_or_kept(temp_path: Callable[[str], Path]) -> None:
    path = temp_path(".wav")
    left, right = tone(1000.0), tone(3000.0, amp=0.2)
    file_io.save_wav(np.stack([left, right], axis=1), FS, path, "FLOAT_32")

    _, mono = file_io.read_wav(path, mono=True, keep_float_level=True)
    _, both = file_io.read_wav(path, mono=False, keep_float_level=True)

    np.testing.assert_allclose(mono, left, atol=1e-6)
    assert both.shape == (left.size, 2)
    np.testing.assert_allclose(both[:, 1], right, atol=1e-6)


def test_scipy_fallback_reads_the_same_file(temp_path: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch) -> None:
    path = temp_path(".wav")
    file_io.save_wav(tone(amp=0.5), FS, path, "PCM_16")
    _, with_soundfile = file_io.read_wav(path)

    monkeypatch.setattr(file_io, "sf", None)  # as if libsndfile were unavailable
    rate, with_scipy = file_io.read_wav(path)

    assert rate == FS
    np.testing.assert_allclose(with_scipy, with_soundfile, atol=1e-4)


# ----------------------------------------------------------------- saving rules
def test_pcm_output_is_clipped_not_wrapped(temp_path: Callable[[str], Path]) -> None:
    path = temp_path(".wav")

    file_io.save_wav(np.array([0.0, 2.0, -2.0, 0.5], dtype=np.float32), FS, path, "PCM_16")
    _, y = file_io.read_wav(path)

    np.testing.assert_allclose(y, [0.0, 1.0, -1.0, 0.5], atol=1e-3)


def test_float_output_keeps_over_range_values(temp_path: Callable[[str], Path]) -> None:
    path = temp_path(".wav")

    file_io.save_wav(np.array([0.0, 2.0, -2.0], dtype=np.float32), FS, path, "FLOAT_32")
    _, y = file_io.read_wav(path, keep_float_level=True)

    np.testing.assert_allclose(y, [0.0, 2.0, -2.0])


def test_non_finite_samples_are_written_as_zero(temp_path: Callable[[str], Path]) -> None:
    path = temp_path(".wav")

    file_io.save_wav(np.array([0.1, np.nan, np.inf, -np.inf], dtype=np.float32), FS, path, "FLOAT_32")
    _, y = file_io.read_wav(path, keep_float_level=True)

    np.testing.assert_allclose(y, [0.1, 0.0, 0.0, 0.0], atol=1e-6)


@pytest.mark.parametrize(
    ("data", "rate", "subtype", "message"),
    [
        (np.zeros(10, dtype=np.float32), FS, "PCM_8", "Unsupported WAV subtype"),
        (np.zeros(0, dtype=np.float32), FS, "PCM_16", "Nothing to write"),
        (np.zeros(10, dtype=np.float32), 0, "PCM_16", "sample_rate"),
        (np.zeros(10, dtype=np.float32), -5, "PCM_16", "sample_rate"),
    ],
)
def test_invalid_save_requests_are_rejected(
    tmp_path: Path, data: np.ndarray, rate: int, subtype: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        file_io.save_wav(data, rate, tmp_path / "x.wav", subtype)


# ---------------------------------------------------------------- read failures
def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        file_io.read_wav(tmp_path / "absent.wav")


def test_empty_file(temp_path: Callable[[str], Path]) -> None:
    path = temp_path(".wav")  # NamedTemporaryFile creates it empty

    with pytest.raises(ValueError, match="empty"):
        file_io.read_wav(path)


def test_file_that_is_not_audio(temp_path: Callable[[str], Path]) -> None:
    path = temp_path(".wav")
    path.write_bytes(b"this is certainly not a RIFF file" * 4)

    with pytest.raises(ValueError, match="decode"):
        file_io.read_wav(path)


# ------------------------------------------------------------------ CSV export
def read_rows(path: Path, delimiter: str = ";") -> list[list[str]]:
    return list(csv.reader(path.read_text(encoding="utf-8-sig").splitlines(), delimiter=delimiter))


def test_signal_csv_layout(temp_path: Callable[[str], Path]) -> None:
    path = temp_path(".csv")
    x = np.array([0.0, 0.5, -0.25, 1.0], dtype=np.float32)

    file_io.save_signal_csv(x, 4, path)
    rows = read_rows(path)

    assert rows[0] == ["index", "time_s", "value"]
    assert len(rows) == 1 + x.size
    assert [r[0] for r in rows[1:]] == ["0", "1", "2", "3"]
    assert [float(r[1]) for r in rows[1:]] == pytest.approx([0.0, 0.25, 0.5, 0.75])
    assert [float(r[2]) for r in rows[1:]] == pytest.approx(x.tolist())


def test_signal_csv_formatting(temp_path: Callable[[str], Path]) -> None:
    path = temp_path(".csv")

    file_io.save_signal_csv(np.array([0.123456789], dtype=np.float32), 3, path)

    first = read_rows(path)[1]
    assert first[1] == "0.000000000"  # time with nine decimals
    assert first[2] == f"{float(np.float32(0.123456789)):.9g}"  # the float32 value, nine significant digits


def test_signal_csv_has_a_utf8_bom_for_excel(temp_path: Callable[[str], Path]) -> None:
    path = temp_path(".csv")

    file_io.save_signal_csv(tone(seconds=0.001), FS, path)

    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_signal_csv_delimiter_is_configurable(temp_path: Callable[[str], Path]) -> None:
    path = temp_path(".csv")

    file_io.save_signal_csv(np.array([1.0, 2.0], dtype=np.float32), FS, path, delimiter=",")

    assert read_rows(path, ",")[0] == ["index", "time_s", "value"]


def test_signal_csv_of_an_empty_signal_is_just_the_header(temp_path: Callable[[str], Path]) -> None:
    path = temp_path(".csv")

    file_io.save_signal_csv(np.array([], dtype=np.float32), FS, path)

    assert read_rows(path) == [["index", "time_s", "value"]]


def test_signal_csv_rejects_a_bad_sample_rate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        file_io.save_signal_csv(np.zeros(3, dtype=np.float32), 0, tmp_path / "x.csv")


def test_impulse_csv_columns_and_values(temp_path: Callable[[str], Path]) -> None:
    path = temp_path(".csv")
    impulses = [
        DetectedImpulse(1, 100, 340, 0.5, 0.75, 1.25, 0.6, 0.9, 0.2, 40_000.0, float("nan"), 0.3),
        DetectedImpulse(2, 900, 1100, 1.5, 1.75, 1.0, 1.6, 0.8, 0.1, 39_000.0, 300.0, 0.3),
    ]

    file_io.save_impulses_csv(impulses, path)
    rows = read_rows(path)

    assert rows[0] == [
        "index", "start_s", "end_s", "duration_ms", "peak_time_s", "peak_amplitude", "rms",
        "dominant_freq_hz", "interval_ms", "start_sample", "end_sample", "threshold",
    ]
    assert len(rows) == 3
    assert rows[1][0] == "1" and rows[1][9:11] == ["100", "340"] and rows[1][8] == "nan"
    assert float(rows[2][8]) == 300.0 and float(rows[2][7]) == 39_000.0 and float(rows[2][11]) == 0.3


# ------------------------------------------------------------------ TXT tables
def test_single_column_txt_uses_the_default_rate(temp_path: Callable[[str], Path]) -> None:
    path = temp_path(".txt")
    path.write_text("# comment\n1.0\n-2.0\n0.5\n", encoding="utf-8")

    rate, y = file_io.read_txt_as_audio(path)

    assert rate == file_io.DEFAULT_TXT_SAMPLE_RATE
    np.testing.assert_allclose(y, [0.5, -1.0, 0.25])  # peak-normalized


def test_two_column_txt_estimates_the_rate_from_the_time_column(temp_path: Callable[[str], Path]) -> None:
    path = temp_path(".csv")
    path.write_text("t;v\n0.000;1\n0.001;2\n0.002;-4\n", encoding="utf-8")

    rate, y = file_io.read_txt_as_audio(path)

    assert rate == 1000
    np.testing.assert_allclose(y, [0.25, 0.5, -1.0])


def test_txt_without_numbers_is_rejected(temp_path: Callable[[str], Path]) -> None:
    path = temp_path(".txt")
    path.write_text("no numbers here\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No numeric data"):
        file_io.read_txt_as_audio(path)


def test_txt_to_wav_roundtrip(temp_path: Callable[[str], Path], tmp_path: Path) -> None:
    source = temp_path(".txt")
    values = np.sin(np.arange(2000) / 9.0)
    source.write_text("\n".join(f"{v:.6f}" for v in values), encoding="utf-8")
    out = tmp_path / "out.wav"

    rate, count = file_io.txt_to_wav(source, out, sample_rate=8000)
    got_rate, y = file_io.read_wav(out)

    assert (rate, count) == (8000, 2000) and got_rate == 8000
    np.testing.assert_allclose(y, values / np.max(np.abs(values)), atol=2e-4)


def test_txt_to_wav_needs_a_rate_for_a_single_column(temp_path: Callable[[str], Path], tmp_path: Path) -> None:
    source = temp_path(".txt")
    source.write_text("1\n2\n3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sample rate"):
        file_io.txt_to_wav(source, tmp_path / "x.wav")
