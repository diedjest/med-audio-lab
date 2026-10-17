"""Audio and signal file I/O: WAV, numeric TXT/CSV tables, and CSV export."""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from app.constants import BitDepth

try:
    import soundfile as sf
except (ImportError, OSError):  # libsndfile missing
    sf = None  # type: ignore[assignment]

try:
    from scipy.io import wavfile as _wavfile
except ImportError:
    _wavfile = None  # type: ignore[assignment]

#: Sample rate assumed for single-column TXT files that carry no time axis.
DEFAULT_TXT_SAMPLE_RATE: int = 100_000

_NUMBER_RE: re.Pattern[str] = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

_SUBTYPE_ALIASES: dict[str, str] = {"FLOAT": "FLOAT_32"}


# --------------------------------------------------------------------- WAV
def _read_wav_soundfile(path: Path, normalize: bool) -> tuple[int, np.ndarray, bool]:
    """Read with soundfile; the flag is True if integer PCM was already scaled to [-1, 1]."""
    info = sf.info(str(path))
    is_integer_pcm = info.subtype.startswith("PCM")
    if normalize or not is_integer_pcm:
        data, fs = sf.read(str(path), dtype="float32", always_2d=False)
    else:
        # Raw integer values, as stored in the file (no scaling)
        int_dtype = "int16" if info.subtype in ("PCM_16", "PCM_S8", "PCM_U8") else "int32"
        data, fs = sf.read(str(path), dtype=int_dtype, always_2d=False)
    return int(fs), np.asarray(data), bool(is_integer_pcm and normalize)


def _read_wav_scipy(path: Path) -> tuple[int, np.ndarray, bool]:
    fs, data = _wavfile.read(str(path))
    return int(fs), np.asarray(data), False


def read_wav(
    path: str | Path,
    mono: bool = True,
    normalize: bool = True,
    keep_float_level: bool = False,
) -> tuple[int, np.ndarray]:
    """Read a WAV file with soundfile, falling back to ``scipy.io.wavfile``.

    Args:
        path: WAV file path.
        mono: If True and the file has several channels, return the first channel.
        normalize: If True, integer PCM is scaled to [-1, 1] by full scale and
            floating-point files are peak-normalized to 1.0. If False, samples
            are returned as stored (integer PCM keeps its integer values).
        keep_float_level: With ``normalize=True``, do not peak-normalize
            floating-point files, so levels (even above 1.0) are preserved exactly.

    Returns:
        ``(sample_rate_hz, samples_float32)``; 2D ``(n, channels)`` if ``mono`` is False
        and the file has several channels.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: the file is empty or cannot be decoded.
        ImportError: neither soundfile nor scipy is available.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if p.stat().st_size == 0:
        raise ValueError(f"WAV file is empty: {path}")

    result: tuple[int, np.ndarray, bool] | None = None
    failure: Exception | None = None
    if sf is not None:
        try:
            result = _read_wav_soundfile(p, normalize)
        except (RuntimeError, ValueError) as exc:  # sf.LibsndfileError is a RuntimeError
            failure = exc
    if result is None and _wavfile is not None:
        try:
            result = _read_wav_scipy(p)
        except ValueError as exc:
            failure = exc
    if result is None:
        if failure is not None:
            raise ValueError(f"Cannot decode WAV file {path}: {failure}") from failure
        raise ImportError("soundfile or scipy is required to read WAV files.")

    fs, x, pcm_scaled = result
    if mono and x.ndim > 1:
        x = x[:, 0]

    if np.issubdtype(x.dtype, np.integer):
        if normalize:
            info = np.iinfo(x.dtype)
            x = x.astype(np.float32) / float(max(abs(info.min), info.max))
        else:
            x = x.astype(np.float32)
    else:
        x = x.astype(np.float32, copy=False)
        if normalize and not pcm_scaled and not keep_float_level:
            peak = float(np.max(np.abs(x))) if x.size else 0.0
            if peak > 0.0:
                x = x / peak
    return fs, np.ascontiguousarray(x, dtype=np.float32)


def save_wav(
    data: np.ndarray,
    sample_rate: int,
    path: str | Path,
    subtype: str = "PCM_16",
) -> None:
    """Write float samples to a WAV file with soundfile.

    Args:
        data: Samples, shape ``(n,)`` or ``(n, channels)``; non-finite values become 0.
            Integer subtypes clip to [-1, 1].
        sample_rate: Sample rate in Hz.
        path: Destination path; parent directories are created.
        subtype: ``"PCM_16"``, ``"PCM_24"`` or ``"FLOAT_32"`` (``"FLOAT"`` accepted).

    Raises:
        ValueError: unsupported subtype, empty data or bad sample rate.
        ImportError: soundfile is not available.
    """
    if sf is None:
        raise ImportError("soundfile is required to write WAV files.")
    name = _SUBTYPE_ALIASES.get(subtype.upper(), subtype.upper())
    try:
        depth = BitDepth(name)
    except ValueError:
        supported = ", ".join(b.value for b in BitDepth)
        raise ValueError(f"Unsupported WAV subtype {subtype!r}; use one of: {supported}") from None
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    samples = np.nan_to_num(np.asarray(data, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if samples.ndim not in (1, 2) or samples.shape[0] == 0:
        raise ValueError(f"Nothing to write: data has shape {samples.shape}")
    if depth is not BitDepth.FLOAT_32:
        samples = np.clip(samples, -1.0, 1.0)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), samples, int(sample_rate), subtype=depth.soundfile_subtype, format="WAV")


# ---------------------------------------------------------------- TXT / CSV
def _load_numeric_table(path: str | Path) -> np.ndarray:
    """Parse every number on every line into a NaN-padded 2D table.

    Tolerates headers, comments and mixed separators (space, tab, comma,
    semicolon); lines without numbers are skipped.
    """
    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            values = [float(m.group(0)) for m in _NUMBER_RE.finditer(line)]
            if values:
                rows.append(values)
    if not rows:
        raise ValueError(f"No numeric data found in: {path}")

    table = np.full((len(rows), max(len(r) for r in rows)), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        table[i, : len(row)] = row
    return table


def _estimate_sample_rate(time_axis: np.ndarray) -> int | None:
    """Sample rate from the median positive time step, or None if undeterminable."""
    steps = np.diff(time_axis)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    if steps.size == 0:
        return None
    rate = int(round(1.0 / float(np.median(steps))))
    return rate if rate > 0 else None


def read_txt_as_audio(
    path: str | Path,
    default_sample_rate: int = DEFAULT_TXT_SAMPLE_RATE,
) -> tuple[int, np.ndarray]:
    """Load a numeric TXT/CSV table as a peak-normalized audio signal.

    One column is the signal itself (sampled at ``default_sample_rate``). With two
    or more columns the first is the time axis (the rate is its median step) and
    the last column is the signal.

    Returns:
        ``(sample_rate_hz, samples_float32)`` scaled so the peak is 1.0.

    Raises:
        ValueError: the file has no numeric data.
    """
    table = _load_numeric_table(path)
    if table.shape[1] == 1:
        fs = int(default_sample_rate)
    else:
        fs = _estimate_sample_rate(table[:, 0]) or int(default_sample_rate)

    y = np.nan_to_num(table[:, -1], nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(y)))
    if peak > 0.0:
        y = y / peak
    return fs, y.astype(np.float32)


def txt_to_wav(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int | None = None,
    amplitude_col: int = -1,
    normalize: bool = True,
) -> tuple[int, int]:
    """Convert a numeric TXT/CSV table to a 16-bit WAV file.

    Args:
        input_path: Source table (see :func:`read_txt_as_audio` for layouts).
        output_path: Destination WAV path.
        sample_rate: Required for single-column files; for tables with a time
            column it overrides the rate estimated from that column.
        amplitude_col: Column holding the signal (default: last).
        normalize: Scale to peak 1.0; otherwise values are clipped to [-1, 1].

    Returns:
        ``(sample_rate_hz, n_samples)``.

    Raises:
        ValueError: missing/undeterminable sample rate, bad column, or no data.
    """
    table = _load_numeric_table(input_path)
    n_cols = table.shape[1]

    if n_cols == 1:
        if sample_rate is None:
            raise ValueError("A sample rate is required for single-column files.")
        fs = int(sample_rate)
    elif sample_rate is not None:
        fs = int(sample_rate)
    else:
        estimated = _estimate_sample_rate(table[:, 0])
        if estimated is None:
            raise ValueError("Cannot estimate the sample rate from the time column.")
        fs = estimated
    if fs <= 0:
        raise ValueError(f"Invalid sample rate: {fs}")

    try:
        y = table[:, amplitude_col].astype(np.float64)
    except IndexError:
        raise ValueError(
            f"amplitude_col={amplitude_col} is out of range for {n_cols} column(s)"
        ) from None
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    if normalize:
        peak = float(np.max(np.abs(y)))
        if peak > 0.0:
            y = y / peak
    else:
        y = np.clip(y, -1.0, 1.0)

    save_wav(y.astype(np.float32), fs, output_path, "PCM_16")
    return fs, int(y.size)


def read_audio_file(path: str | Path) -> tuple[int, np.ndarray]:
    """Read a WAV, TXT or CSV file as ``(sample_rate, float32 samples)``.

    WAV levels are kept as stored (integer PCM is scaled to [-1, 1]); TXT / CSV tables
    are peak-normalized (see :func:`read_txt_as_audio`).
    """
    if Path(path).suffix.lower() in (".txt", ".csv"):
        return read_txt_as_audio(path)
    return read_wav(path, mono=True, normalize=True, keep_float_level=True)


_IMPULSE_CSV_COLUMNS: tuple[tuple[str, str], ...] = (
    ("index", "index"),
    ("start_s", "start_s"),
    ("end_s", "end_s"),
    ("duration_ms", "duration_ms"),
    ("peak_time_s", "peak_time_s"),
    ("peak_amplitude", "peak_amplitude"),
    ("rms", "rms"),
    ("dominant_freq_hz", "dominant_freq_hz"),
    ("interval_ms", "interval_ms"),
    ("start_sample", "start_sample"),
    ("end_sample", "end_sample"),
    ("threshold", "threshold_used"),
)


def save_impulses_csv(impulses: Sequence[Any], path: str | Path, delimiter: str = ";") -> None:
    """Write detected impulses as CSV (the legacy column set, UTF-8 with BOM).

    Args:
        impulses: ``DetectedImpulse`` objects (any objects with the same attributes).
        path: Destination; parent directories are created.
        delimiter: Field separator.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow([column for column, _ in _IMPULSE_CSV_COLUMNS])
        for impulse in impulses:
            writer.writerow([getattr(impulse, attribute) for _, attribute in _IMPULSE_CSV_COLUMNS])


def save_signal_csv(
    data: np.ndarray,
    sample_rate: int,
    path: str | Path,
    delimiter: str = ";",
) -> None:
    """Write samples as CSV with columns ``index, time_s, value``.

    The file is UTF-8 with BOM and ``;``-separated by default so Excel opens it
    correctly in locales that use a decimal comma.

    Args:
        data: 1D samples (multi-dimensional input is flattened).
        sample_rate: Sample rate in Hz, used for the time column.
        path: Destination path; parent directories are created.
        delimiter: Field separator.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    values = np.asarray(data, dtype=np.float32).ravel()
    rate = float(sample_rate)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(["index", "time_s", "value"])
        writer.writerows(
            (i, f"{i / rate:.9f}", f"{float(v):.9g}") for i, v in enumerate(values)
        )


__all__: list[str] = [
    "DEFAULT_TXT_SAMPLE_RATE",
    "read_wav",
    "save_wav",
    "read_audio_file",
    "read_txt_as_audio",
    "txt_to_wav",
    "save_signal_csv",
    "save_impulses_csv",
]
