from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

try:
    from scipy.io import wavfile
except ImportError:
    import soundfile as sf

    class wavfile:
        @staticmethod
        def write(path, rate, data):
            sf.write(path, data, rate, subtype="PCM_16")


def _load_txt(path: str) -> np.ndarray:
    """
    Robust loader for numeric TXT/CSV with optional headers.
    Supports spaces, tabs, commas, semicolons; ignores non-numeric lines.
    """
    rows: list[list[float]] = []
    num_re = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            vals = [float(m.group(0)) for m in num_re.finditer(line)]
            if vals:
                rows.append(vals)

    if not rows:
        raise ValueError(f"Пустой TXT: {path}")

    max_cols = max(len(r) for r in rows)
    data = np.full((len(rows), max_cols), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        data[i, : len(r)] = r
    return data


def _estimate_fs_from_time(t: np.ndarray) -> int:
    if t.size < 2:
        raise ValueError("Недостаточно точек времени для оценки sample rate.")
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if dt.size == 0:
        raise ValueError("Не удалось оценить шаг времени из первой колонки.")
    fs = int(round(1.0 / float(np.median(dt))))
    if fs <= 0:
        raise ValueError("Оцененный sample rate некорректен.")
    return fs


def txt_to_wav(
    input_txt: str | Path,
    output_wav: str | Path,
    *,
    sample_rate: int | None = None,
    amplitude_col: int = -1,
    normalize: bool = True,
) -> tuple[int, int]:
    """
    Конвертирует TXT в WAV.

    Поддерживаемые форматы TXT:
    - 1 колонка: амплитуда (sample_rate обязателен)
    - 2+ колонки: можно передать sample_rate вручную, либо взять fs из первой
      колонки времени; амплитуда берется из amplitude_col (по умолчанию -1).
    """
    arr = _load_txt(str(input_txt))
    n_cols = arr.shape[1]

    if n_cols == 1:
        if sample_rate is None:
            raise ValueError("Для TXT с 1 колонкой укажите --sample-rate.")
        y = arr[:, 0].astype(np.float64)
        fs = int(sample_rate)
    else:
        y = arr[:, amplitude_col].astype(np.float64)
        fs = int(sample_rate) if sample_rate is not None else _estimate_fs_from_time(arr[:, 0].astype(np.float64))

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if y.size == 0:
        raise ValueError("Пустой сигнал после чтения TXT.")

    if normalize:
        peak = float(np.max(np.abs(y)))
        if peak > 0:
            y = y / peak
    else:
        y = np.clip(y, -1.0, 1.0)

    pcm = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(str(output_wav), fs, pcm)
    return fs, pcm.size


def main():
    parser = argparse.ArgumentParser(description="Конвертер TXT -> WAV")
    parser.add_argument("input_txt", help="Путь к входному TXT/CSV")
    parser.add_argument("output_wav", help="Путь к выходному WAV")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Sample rate (Гц). Обязателен для TXT с 1 колонкой.",
    )
    parser.add_argument(
        "--amplitude-col",
        type=int,
        default=-1,
        help="Индекс колонки амплитуды (если колонок >= 2). По умолчанию последняя.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Не нормировать сигнал (только клип в диапазон [-1, 1]).",
    )
    args = parser.parse_args()

    out = Path(args.output_wav)
    out.parent.mkdir(parents=True, exist_ok=True)

    fs, n = txt_to_wav(
        args.input_txt,
        str(out),
        sample_rate=args.sample_rate,
        amplitude_col=args.amplitude_col,
        normalize=not args.no_normalize,
    )
    print(f"Готово: {out} | fs={fs} Гц | samples={n}")


if __name__ == "__main__":
    main()