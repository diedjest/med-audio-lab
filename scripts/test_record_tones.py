"""Проверка авто-частоты и CWT-гребня для test_audiocard_10 / _30."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from med_audio_lab.backend import analysis
from med_audio_lab.utils.io import read_wav

RECORDS = ROOT / "med_audio_lab" / "records"
FILES = {
    10_000: RECORDS / "20260603_130422_test_audiocard_10.wav",
    30_000: RECORDS / "20260603_131008_test_audiocard_30.wav",
}


def main() -> int:
    ok = True
    for expected, path in FILES.items():
        if not path.exists():
            print(f"SKIP missing {path}")
            ok = False
            continue
        fs, data = read_wav(path)
        from_name = analysis.infer_tone_hz_from_path(path)
        resolved = analysis.resolve_record_tone_hz(path, data, fs)
        fft_peak = analysis.estimate_dominant_tone_hz(data, fs)
        t, freqs, Wx = analysis.cwt_scalogram(data, fs, max_seconds=4.0, target_fs=192000)
        amp = abs(Wx)
        ni = int(abs(freqs - resolved).argmin()) if resolved else 0
        row_f = float(freqs[ni])
        col_peak = float(amp[ni, :].max())
        global_peak = float(amp.max())
        name_ok = from_name == expected
        res_ok = resolved == expected
        print(f"\n{path.name}")
        print(f"  expected:     {expected} Hz")
        print(f"  from_name:    {from_name} Hz ({'OK' if name_ok else 'FAIL'})")
        print(f"  resolved:     {resolved} Hz ({'OK' if res_ok else 'FAIL'})")
        print(f"  fft_peak:     {fft_peak:.1f} Hz")
        print(f"  CWT row@tone: {row_f:.1f} Hz, row/col max ratio {col_peak/global_peak:.2f}")
        ok = ok and name_ok and res_ok
    print("\n" + ("ALL OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
