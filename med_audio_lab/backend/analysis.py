from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

# Обновленные импорты из нашего нового математического пакета DSP
from ..dsp.fourier import fft_spectrum_from_signal, stft_from_signal
from ..dsp.wavelet import compute_cwt_array, save_time_frequency_coords, save_time_frequency_coords_csv

EPS = 1e-12
# Ограничение длины CWT (~2.1 с при 192 кГц) — защита от нехватки памяти.
_MAX_CWT_SAMPLES = 400_000


def as_mono(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr[:, 0]
    return np.nan_to_num(arr, copy=False)


def normalize(data: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    x = as_mono(data).copy()
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= EPS:
        return x
    return np.clip(x / peak * target_peak, -1.0, 1.0).astype(np.float32)


def remove_dc(data: np.ndarray) -> np.ndarray:
    x = as_mono(data).copy()
    if x.size:
        x -= float(np.mean(x))
    return x.astype(np.float32)


def noise_gate(data: np.ndarray, threshold_db: float = -45.0) -> np.ndarray:
    x = as_mono(data).copy()
    if not x.size:
        return x
    threshold = 10 ** (threshold_db / 20.0)
    mask = np.abs(x) < threshold
    x[mask] *= 0.15
    return x.astype(np.float32)


def trim_silence(
    data: np.ndarray,
    threshold_db: float = -48.0,
    pad_ms: int = 120,
    samplerate: int = 44100,
) -> np.ndarray:
    x = as_mono(data)
    if not x.size:
        return x
    threshold = 10 ** (threshold_db / 20.0)
    active = np.where(np.abs(x) >= threshold)[0]
    if active.size == 0:
        return x.copy()
    pad = int(samplerate * pad_ms / 1000)
    start = max(0, int(active[0]) - pad)
    end = min(x.size, int(active[-1]) + pad)
    return x[start:end].astype(np.float32)


def fft_filter(
    data: np.ndarray,
    samplerate: int,
    low_hz: float | None = None,
    high_hz: float | None = None,
) -> np.ndarray:
    x = remove_dc(data)
    if x.size < 4:
        return x
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1.0 / samplerate)
    mask = np.ones_like(freqs, dtype=bool)
    if low_hz is not None:
        mask &= freqs >= float(low_hz)
    if high_hz is not None:
        mask &= freqs <= float(high_hz)
    spectrum *= mask
    y = np.fft.irfft(spectrum, n=x.size).astype(np.float32)
    return normalize(y, 0.95)


def envelope(data: np.ndarray, window_ms: int = 12, samplerate: int = 44100) -> np.ndarray:
    x = np.abs(as_mono(data))
    if x.size == 0:
        return x
    window = max(3, int(samplerate * window_ms / 1000))
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def fft_spectrum(data: np.ndarray, samplerate: int):
    return fft_spectrum_from_signal(data, samplerate)


def infer_tone_hz_from_path(path: str | Path | None) -> float | None:
    """
    Частота тона из имени файла (например test_audiocard_10 → 10 кГц, _30 → 30 кГц).
    """
    if path is None:
        return None
    stem = Path(path).stem
    patterns = (
        r"audiocard[_-]?(\d{2,5})\b",
        r"[_-](\d{2,5})$",
        r"(\d{2,5})\s*khz",
        r"(\d{2,5})\s*hz",
    )
    for pat in patterns:
        m = re.search(pat, stem, flags=re.IGNORECASE)
        if not m:
            continue
        v = int(m.group(1))
        if v < 1000:
            return float(v * 1000)
        return float(v)
    return None


def estimate_dominant_tone_hz(
    data: np.ndarray,
    samplerate: int,
    *,
    fmin_hz: float = 400.0,
    fmax_hz: float = 120_000.0,
) -> float | None:
    """Главный пик спектра Фурье (Гц)."""
    x = as_mono(data)
    if x.size < 32:
        return None
    freqs, mag = fft_spectrum_from_signal(x, int(samplerate))
    band = (freqs >= fmin_hz) & (freqs <= fmax_hz)
    if not np.any(band):
        return None
    idx = int(np.argmax(mag[band]))
    peak = float(freqs[band][idx])
    return peak if peak > 0 else None


def resolve_record_tone_hz(
    path: str | Path | None,
    data: np.ndarray,
    samplerate: int,
) -> float | None:
    """Частота записи: сначала из имени файла, иначе по FFT."""
    from_name = infer_tone_hz_from_path(path)
    if from_name is not None:
        return from_name
    return estimate_dominant_tone_hz(data, samplerate)


def is_narrowband_constant_tone(
    data: np.ndarray,
    samplerate: int,
    *,
    energy_ratio_min: float = 0.42,
    bandwidth_frac: float = 0.18,
) -> bool:
    """Узкополосный тон: большая часть энергии в узкой полосе спектра."""
    x = as_mono(data)
    if x.size < 64:
        return False
    freqs, mag = fft_spectrum_from_signal(x, int(samplerate))
    power = np.square(mag, dtype=np.float64)
    total = float(power.sum())
    if total <= EPS:
        return False
    i = int(np.argmax(power))
    peak_f = float(freqs[i])
    if peak_f <= 0:
        return False
    band = (freqs >= peak_f * (1.0 - bandwidth_frac)) & (freqs <= peak_f * (1.0 + bandwidth_frac))
    if not np.any(band):
        return False
    return float(power[band].sum()) / total >= energy_ratio_min


def is_steady_tone(
    data: np.ndarray,
    samplerate: int,
    *,
    segments: int = 6,
    max_freq_jitter: float = 0.06,
) -> bool:
    """Частота тона почти не меняется по длине записи."""
    x = as_mono(data)
    if x.size < 128:
        return True
    n = int(segments)
    chunk = max(64, x.size // n)
    peaks: list[float] = []
    for k in range(n):
        seg = x[k * chunk: (k + 1) * chunk if k < n - 1 else x.size]
        if seg.size < 32:
            continue
        f = estimate_dominant_tone_hz(seg, samplerate)
        if f is not None and f > 0:
            peaks.append(f)
    if len(peaks) < 2:
        return True
    mean_f = float(np.mean(peaks))
    if mean_f <= 0:
        return False
    return float(np.std(peaks)) / mean_f <= max_freq_jitter


def should_use_morlet_ridge_line(data: np.ndarray, samplerate: int) -> bool:
    """Прямая линия на CWT только для ровного узкополосного сигнала."""
    return is_narrowband_constant_tone(data, samplerate) and is_steady_tone(data, samplerate)


def stft_spectrogram(
    data: np.ndarray,
    samplerate: int,
    *,
    nperseg: int = 2048,
    noverlap: int = 1536,
    fmin_hz: float = 0.0,
    fmax_hz: float = 100000.0,
    db: bool = True,
):
    return stft_from_signal(
        data,
        samplerate,
        nperseg=nperseg,
        noverlap=noverlap,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
        db=db,
    )


def estimate_cwt_load(data: np.ndarray, fs: int, *, target_fs: int = 192000, max_seconds: float | None = 4.0) -> dict:
    """Оценка нагрузки CWT для предупреждения пользователя."""
    x = as_mono(data)
    if x.size == 0:
        return {"samples": 0, "too_heavy": False, "message": "Нет данных"}
    eff_fs = int(target_fs) if target_fs > 0 else int(fs)
    ratio = eff_fs / max(1, int(fs))
    n_est = int(min(x.size * ratio, eff_fs * max_seconds)) if max_seconds else int(x.size * ratio)
    n_est = max(1, n_est)
    scales_n = 375
    mem_mb = n_est * scales_n * 8 / (1024 * 1024)
    too_heavy = n_est > _MAX_CWT_SAMPLES or mem_mb > 900
    msg = f"≈{n_est} отсчётов, ~{mem_mb:.0f} МБ для CWT"
    return {"samples": n_est, "too_heavy": too_heavy, "message": msg, "mem_mb": mem_mb}


def cwt_scalogram(
    data: np.ndarray,
    fs: int,
    *,
    fmin_hz: float = 0.1,
    fmax_hz: float = 100000.0,
    max_seconds: float | None = 4.0,
    downsample: int = 1,
    target_fs: int = 192000,
    scale_stride: int = 1,
):
    x = as_mono(data)
    if x.size < 8:
        raise ValueError("Сигнал слишком короткий для вейвлет-преобразования (нужно ≥ 8 отсчётов).")
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    eff_fs = int(target_fs) if target_fs > 0 else int(fs)
    if max_seconds:
        n_after = int(min(x.size * eff_fs / max(1, fs), eff_fs * max_seconds))
    else:
        n_after = int(x.size * eff_fs / max(1, fs))
    stride = max(1, int(scale_stride))
    if n_after > _MAX_CWT_SAMPLES:
        stride = max(stride, 2)
    if n_after > _MAX_CWT_SAMPLES * 2:
        stride = max(stride, 4)

    _fs, t, freqs_hz, Wx = compute_cwt_array(
        x,
        int(fs),
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
        max_seconds=max_seconds,
        downsample=downsample,
        target_fs=target_fs,
        scales_path=None,  # Пакет dsp сам найдет путь к конфигу
        scale_stride=stride,
    )
    if Wx.size == 0 or not np.isfinite(Wx).any():
        raise ValueError("CWT вернул пустой или некорректный результат.")
    return t, freqs_hz, Wx


def autocorrelation(data: np.ndarray, samplerate: int, max_lag_ms: int = 120) -> tuple[np.ndarray, np.ndarray]:
    x = remove_dc(data)
    if x.size == 0:
        return np.zeros(0), np.zeros(0)
    max_lag = min(x.size - 1, int(samplerate * max_lag_ms / 1000))
    nfft = 1 << int(np.ceil(np.log2(max(2, x.size * 2 - 1))))
    spectrum = np.fft.rfft(x, n=nfft)
    corr = np.fft.irfft(spectrum * np.conj(spectrum), n=nfft)[: max_lag + 1]
    peak = float(np.max(np.abs(corr))) if corr.size else 0.0
    if peak > EPS:
        corr = corr / peak
    lags = np.arange(corr.size, dtype=np.float32) / samplerate * 1000.0
    return lags, corr.astype(np.float32)


def metrics(data: np.ndarray, samplerate: int) -> dict[str, str]:
    x = as_mono(data)
    if x.size == 0:
        return {"Состояние": "Нет аудиоданных"}
    duration = x.size / samplerate
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(np.square(x))))
    dbfs = 20 * math.log10(max(rms, EPS))
    zero_crossings = float(np.mean(np.abs(np.diff(np.signbit(x)))))
    freqs, amp = fft_spectrum(x, samplerate)
    if amp.size > 1:
        amp_no_dc = amp.copy()
        amp_no_dc[0] = 0.0
        dom_idx = int(np.argmax(amp_no_dc))
        dom_freq = float(freqs[dom_idx])
        total = float(np.sum(amp_no_dc)) + EPS
        centroid = float(np.sum(freqs * amp_no_dc) / total)
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * amp_no_dc) / total))
        cumulative = np.cumsum(amp_no_dc)
        rolloff_idx = int(np.searchsorted(cumulative, 0.85 * cumulative[-1]))
        rolloff = float(freqs[min(rolloff_idx, freqs.size - 1)])
    else:
        dom_freq = centroid = bandwidth = rolloff = 0.0

    return {
        "Длительность": f"{duration:.3f} с",
        "Частота дискретизации": f"{samplerate} Гц",
        "Количество отсчетов": f"{x.size}",
        "RMS": f"{rms:.5f}",
        "Пик": f"{peak:.5f}",
        "Уровень RMS": f"{dbfs:.1f} dBFS",
        "Переходы через ноль": f"{zero_crossings:.4f}",
        "Доминирующая частота": f"{dom_freq:.1f} Гц",
        "Спектральный центр": f"{centroid:.1f} Гц",
        "Ширина полосы": f"{bandwidth:.1f} Гц",
        "Roll-off 85%": f"{rolloff:.1f} Гц",
    }


def get_cwt_ridge(t: np.ndarray, freqs_hz: np.ndarray, Wx: np.ndarray):
    amp = np.abs(Wx)
    ridge_idx = np.argmax(amp, axis=0)
    return t, freqs_hz[ridge_idx]


def get_wavelet_slice(t: np.ndarray, wx_row: np.ndarray, f_hz: float | None = None):
    y_vals = np.real(np.asarray(wx_row, dtype=np.complex128))
    denom = float(np.max(np.abs(y_vals)))
    if denom > 0:
        y_vals = (y_vals / denom).astype(np.float32)
    else:
        y_vals = y_vals.astype(np.float32)
    t0 = float(t[int(np.argmax(np.abs(y_vals)))])
    return (t - t0).astype(np.float32), y_vals


def save_cwt_ridge_txt(t: np.ndarray, freqs_hz: np.ndarray, Wx: np.ndarray, path: str) -> None:
    if path.endswith(".csv"):
        save_time_frequency_coords_csv(t, freqs_hz, Wx, path=path)
    else:
        save_time_frequency_coords(t, freqs_hz, Wx, path=path)


def load_txt_as_audio(path: str) -> tuple[int, np.ndarray]:
    rows = []
    num_re = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            vals = [float(m.group(0)) for m in num_re.finditer(line)]
            if vals:
                rows.append(vals)
    if not rows:
        raise ValueError(path)
    max_cols = max(len(r) for r in rows)
    data = np.full((len(rows), max_cols), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        data[i, : len(r)] = r
    if data.shape[1] == 1:
        fs = 100000
        y = data[:, 0]
    else:
        t = data[:, 0]
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        fs = int(round(1.0 / float(np.median(dt)))) if dt.size > 0 else 100000
        y = data[:, -1]
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak
    return fs, y.astype(np.float32)