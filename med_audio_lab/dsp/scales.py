from pathlib import Path
import numpy as np


def _default_scales() -> np.ndarray:
    # Консервативный fallback, если файл отсутствует
    return np.geomspace(2.0, 5986.0, 116).astype(np.float64)


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

    return scales