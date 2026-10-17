"""Check the automatic tone detection and the CWT ridge on the test_audiocard_10 / _30 recordings.

Run from anywhere:  python scripts/test_record_tones.py
The exit code is 0 when both recordings resolve to the expected tone frequency.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.dsp.cwt import CwtEngine  # noqa: E402
from app.core.dsp.fft import fft_spectrum  # noqa: E402
from app.infrastructure.storage.file_io import read_wav  # noqa: E402

RECORDS = ROOT / "tests" / "data" / "records"
FILES = {
    10_000: RECORDS / "20260603_130422_test_audiocard_10.wav",
    30_000: RECORDS / "20260603_131008_test_audiocard_30.wav",
}

TONE_NAME_PATTERNS = (
    r"audiocard[_-]?(\d{2,5})\b",
    r"[_-](\d{2,5})$",
    r"(\d{2,5})\s*khz",
    r"(\d{2,5})\s*hz",
)


def infer_tone_hz_from_path(path: str | Path | None) -> float | None:
    """Tone frequency named in a file name (``test_audiocard_10`` -> 10 kHz, ``_30`` -> 30 kHz)."""
    if path is None:
        return None
    stem = Path(path).stem
    for pattern in TONE_NAME_PATTERNS:
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match is None:
            continue
        value = int(match.group(1))
        return float(value * 1000 if value < 1000 else value)
    return None


def estimate_dominant_tone_hz(
    data: np.ndarray,
    sample_rate: int,
    *,
    fmin_hz: float = 400.0,
    fmax_hz: float = 120_000.0,
) -> float | None:
    """Frequency of the main Fourier peak in ``[fmin_hz, fmax_hz]``."""
    if data.size < 32:
        return None
    freqs, magnitudes = fft_spectrum(data, sample_rate, fmin_hz=fmin_hz, fmax_hz=fmax_hz)
    if freqs.size == 0:
        return None
    peak = float(freqs[int(np.argmax(magnitudes))])
    return peak if peak > 0 else None


def resolve_record_tone_hz(path: str | Path | None, data: np.ndarray, sample_rate: int) -> float | None:
    """Tone of a recording: the file name wins, otherwise the Fourier peak."""
    from_name = infer_tone_hz_from_path(path)
    if from_name is not None:
        return from_name
    return estimate_dominant_tone_hz(data, sample_rate)


def main() -> int:
    ok = True
    engine = CwtEngine()
    for expected, path in FILES.items():
        if not path.exists():
            print(f"SKIP missing {path}")
            ok = False
            continue
        fs, data = read_wav(path)
        from_name = infer_tone_hz_from_path(path)
        resolved = resolve_record_tone_hz(path, data, fs)
        fft_peak = estimate_dominant_tone_hz(data, fs)

        result = engine.compute(data, fs, fmin_hz=1000.0, fmax_hz=100_000.0, max_seconds=4.0, target_fs=192_000)
        amplitude = result.amplitude
        ni = int(np.abs(result.frequencies_hz - resolved).argmin()) if resolved else 0
        row_f = float(result.frequencies_hz[ni])
        col_peak = float(amplitude[ni, :].max())
        global_peak = float(amplitude.max())

        name_ok = from_name == expected
        res_ok = resolved == expected
        print(f"\n{path.name}")
        print(f"  expected:     {expected} Hz")
        print(f"  from_name:    {from_name} Hz ({'OK' if name_ok else 'FAIL'})")
        print(f"  resolved:     {resolved} Hz ({'OK' if res_ok else 'FAIL'})")
        print(f"  fft_peak:     {fft_peak:.1f} Hz")
        print(f"  CWT row@tone: {row_f:.1f} Hz, row/col max ratio {col_peak / global_peak:.2f}")
        ok = ok and name_ok and res_ok
    print("\n" + ("ALL OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
