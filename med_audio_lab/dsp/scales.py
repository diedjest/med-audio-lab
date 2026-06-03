from pathlib import Path
import numpy as np


def _default_scales() -> np.ndarray:
    # Консервативный fallback, если файл отсутствует
    return np.geomspace(2.0, 5986.0, 116).astype(np.float64)


def _densify_scale_gaps(
    scales: np.ndarray,
    *,
    gap_ratio: float = 1.25,
    max_freq_ratio: float = 1.08,
) -> np.ndarray:
    """
    Добавляет промежуточные Morlet-шкалы там, где соседние центральные частоты
    слишком далеко друг от друга (f ∝ 1/scale). Без этого тон, например 30 кГц
    при fs=192 кГц (нужна шкала ≈6.4), не попадает ни в одну строку CWT.
    """
    scales = np.asarray(scales, dtype=np.float64)
    if scales.size < 2:
        return scales
    out: list[float] = [float(scales[0])]
    for s0, s1 in zip(scales[:-1], scales[1:]):
        s0, s1 = float(s0), float(s1)
        ratio = s1 / s0
        if ratio > gap_ratio:
            n_insert = int(np.ceil(np.log(ratio) / np.log(max_freq_ratio))) - 1
            if n_insert > 0:
                inserted = np.geomspace(s0, s1, n_insert + 2, dtype=np.float64)[1:-1]
                out.extend(float(x) for x in inserted)
        out.append(s1)
    return np.unique(np.asarray(out, dtype=np.float64))


def load_scales_from_txt(path: str | Path | None = None) -> np.ndarray:
    """
    Загружает шкалы CWT из текстового файла.
    """
    if path is None:
        # Ищем файл в папке utils
        cfg_path = Path(__file__).resolve().parent.parent / "utils" / "wavelet_scales.txt"
    else:
        cfg_path = Path(path)

    if not cfg_path.exists():
        return _default_scales()

    values: list[float] = []
    with cfg_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            values.append(float(line.replace(",", ".")))

    scales = np.asarray(values, dtype=np.float64)
    if scales.size == 0:
        return _default_scales()
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError(f"Некорректные значения в '{cfg_path}'. Шкалы должны быть > 0.")

    return _densify_scale_gaps(scales)