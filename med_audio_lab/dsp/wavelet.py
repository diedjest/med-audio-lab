import numpy as np

try:
    from scipy.signal import resample_poly
except ImportError:
    resample_poly = None

from ..utils.io import read_wav
from .scales import load_scales_from_txt

_DEFAULT_CWT_CHUNK = 8


def _resample_linear(x: np.ndarray, fs: int, target_fs: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if fs == target_fs or x.size == 0:
        return x
    n_out = max(1, int(round(x.size * target_fs / fs)))
    t_in = np.arange(x.size, dtype=np.float64) / fs
    t_out = np.arange(n_out, dtype=np.float64) / target_fs
    return np.interp(t_out, t_in, x).astype(np.float32, copy=False)


def _resample_to_target_fs(x: np.ndarray, fs: int, target_fs: int) -> tuple[np.ndarray, int]:
    if target_fs <= 0 or fs == int(target_fs):
        return x, fs
    if resample_poly is not None:
        y = resample_poly(x, int(target_fs), int(fs)).astype(np.float32, copy=False)
        return y, int(target_fs)
    return _resample_linear(x, fs, target_fs), int(target_fs)


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n - 1).bit_length())


def _cwt_fft_bank(
        X: np.ndarray,
        scales: np.ndarray,
        f_fft: np.ndarray,
        fs: float,
        n: int,
        *,
        w0: float = 2.0 * np.pi,
        chunk_size: int = _DEFAULT_CWT_CHUNK,
) -> np.ndarray:
    scales = np.asarray(scales, dtype=np.float64)
    omega_d = 2.0 * np.pi * f_fft / float(fs)
    neg = f_fft < 0
    Wx = np.empty((scales.size, n), dtype=np.complex64)
    step = max(1, int(chunk_size))
    for c0 in range(0, scales.size, step):
        sc = scales[c0: c0 + step]
        s_col = sc[:, np.newaxis]
        psi_hat = np.exp(-0.5 * ((s_col * omega_d - w0) ** 2), dtype=np.float64)
        np.multiply(psi_hat, np.sqrt(s_col), out=psi_hat)
        psi_hat[:, neg] = 0.0
        conv = np.fft.ifft(X * np.conj(psi_hat), axis=1)
        Wx[c0: c0 + sc.size, :] = conv[:, :n]
    return Wx


def compute_cwt_array(
        x: np.ndarray,
        fs: int,
        *,
        fmin_hz: float = 0.1,
        fmax_hz: float = 100000.0,
        max_seconds: float | None = None,
        downsample: int = 1,
        target_fs: int = 192000,
        scales_path: str | None = None,
        scale_stride: int = 1,
        cwt_chunk_size: int = _DEFAULT_CWT_CHUNK,
):
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0:
        x = (x / peak).astype(np.float32, copy=False)

    if target_fs > 0 and fs != int(target_fs):
        x, fs = _resample_to_target_fs(x, fs, int(target_fs))

    if max_seconds is not None:
        max_n = int(max_seconds * fs)
        if max_n > 0 and len(x) > max_n:
            x = x[:max_n]
    if downsample > 1:
        x = x[::downsample]
        fs = int(fs / downsample)

    w0 = 2.0 * np.pi
    scales = load_scales_from_txt(scales_path)
    stride = max(1, int(scale_stride))
    if stride > 1:
        scales = scales[::stride]
    freqs_hz = w0 * fs / (2.0 * np.pi * scales)
    band = (freqs_hz >= float(fmin_hz)) & (freqs_hz <= float(fmax_hz))
    if not np.any(band):
        raise ValueError(f"Нет шкал в диапазоне {fmin_hz:.1f}..{fmax_hz:.1f} Гц.")

    scales = scales[band]
    freqs_hz = freqs_hz[band]

    n = len(x)
    n_fft = _next_pow2(2 * n - 1)
    X = np.fft.fft(x.astype(np.float32, copy=False), n=n_fft)
    f_fft = np.fft.fftfreq(n_fft, d=1.0 / fs)

    Wx = _cwt_fft_bank(X, scales, f_fft, fs=float(fs), n=n, w0=w0, chunk_size=cwt_chunk_size)
    t = np.arange(len(x)) / fs
    return fs, t, freqs_hz, Wx


def compute_cwt(file_path: str, **kwargs):
    fs, x = read_wav(file_path, mono=True, normalize=True)
    return compute_cwt_array(x, fs, **kwargs)


def save_time_frequency_coords(t: np.ndarray, freqs_hz: np.ndarray, Wx: np.ndarray, *, path: str):
    amp = np.abs(Wx)
    ridge_idx = np.argmax(amp, axis=0)
    ridge_freqs = freqs_hz[ridge_idx]
    data = np.column_stack([t, ridge_freqs])
    np.savetxt(path, data, fmt="%.10g", delimiter="\t", header="t_s\tfreq_hz")


def save_time_frequency_coords_csv(t: np.ndarray, freqs_hz: np.ndarray, Wx: np.ndarray, *, path: str):
    amp = np.abs(Wx)
    ridge_idx = np.argmax(amp, axis=0)
    ridge_freqs = freqs_hz[ridge_idx]
    data = np.column_stack([t, ridge_freqs])
    np.savetxt(path, data, fmt="%.10g", delimiter=",", header="t_s,freq_hz")


def save_longitudinal_slice_txt(
        t: np.ndarray,
        wx_row: np.ndarray,
        *,
        path: str,
        f_hz: float | None = None,
        t_center_s: float | None = None,
):
    t = np.asarray(t, dtype=np.float64)
    z = np.asarray(wx_row, dtype=np.complex128)
    re = np.real(z)

    if t_center_s is None:
        t0 = float(t[int(np.argmax(np.abs(re)))])
    else:
        t0 = float(t_center_s)
    t_rel_us = (t - t0) * 1e6

    lines = [
        f"# longitudinal_slice Morlet CWT",
        f"# f_center_hz\t{f_hz if f_hz is not None else 'nan'}",
        f"# t_center_s\t{t0:.12g}",
        f"# duration_recording_s\t{float(t[-1] - t[0]):.12g}",
        "# columns: t_s\tt_rel_us\tRe_Wx",
    ]
    data = np.column_stack([t, t_rel_us, re])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        np.savetxt(f, data, fmt="%.10g", delimiter="\t")