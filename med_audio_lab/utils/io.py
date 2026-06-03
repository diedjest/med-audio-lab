import numpy as np
from pathlib import Path

try:
    from scipy.io import wavfile as _scipy_wavfile
except ImportError:
    _scipy_wavfile = None

try:
    import soundfile as sf
except ImportError:
    sf = None


def read_wav(file_path: str | Path, *, mono: bool = True, normalize: bool = True) -> tuple[int, np.ndarray]:
    """
    Читает WAV файл и возвращает (sample_rate_hz, samples_float32).
    Использует scipy.io.wavfile если доступно, иначе soundfile.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    if p.stat().st_size == 0:
        raise ValueError(f"WAV файл пустой: {file_path}")

    if _scipy_wavfile is not None:
        fs, data = _scipy_wavfile.read(str(p))
        x = np.asarray(data)
    elif sf is not None:
        x, fs = sf.read(str(p), always_2d=False)
        x = np.asarray(x)
        fs = int(fs)
    else:
        raise ImportError("Нужен пакет scipy или soundfile для чтения WAV.")

    if mono and getattr(x, "ndim", 1) > 1:
        x = x[:, 0]

    if normalize:
        if np.issubdtype(x.dtype, np.integer):
            info = np.iinfo(x.dtype)
            denom = float(max(abs(info.min), info.max))
            x = x.astype(np.float32) / denom
        else:
            x = x.astype(np.float32)
            peak = float(np.max(np.abs(x))) if x.size else 0.0
            if peak > 0:
                x = x / peak
    else:
        x = x.astype(np.float32, copy=False)

    return fs, x