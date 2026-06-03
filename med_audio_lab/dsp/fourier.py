import numpy as np


def fft_spectrum_from_signal(data: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(data, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0:
        x = x / peak
    window = np.hanning(len(x))
    windowed = x * window
    fft_values = np.fft.fft(windowed)
    freqs = np.fft.fftfreq(len(fft_values), 1 / fs)
    mask = freqs > 0
    n = len(x)
    return freqs[mask], np.abs(fft_values[mask]) / n


def stft_from_signal(
        x: np.ndarray,
        fs: int,
        *,
        nperseg: int = 4096,
        noverlap: int = 3072,
        fmin_hz: float = 0.0,
        fmax_hz: float = 100000.0,
        db: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0:
        x = x / peak

    nperseg = int(nperseg)
    noverlap = int(noverlap)
    if nperseg <= 0:
        raise ValueError("nperseg must be > 0")
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("noverlap must be in [0, nperseg)")

    hop = nperseg - noverlap
    if x.size < nperseg:
        x = np.pad(x, (0, nperseg - x.size))

    n_frames = 1 + (x.size - nperseg) // hop
    window = np.hanning(nperseg).astype(np.float32)

    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, nperseg),
        strides=(x.strides[0] * hop, x.strides[0]),
        writeable=False,
    )
    X = np.fft.rfft(frames * window[None, :], axis=1)
    mag = np.abs(X).T

    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    t = (np.arange(n_frames) * hop + (nperseg / 2.0)) / fs

    band = (freqs >= fmin_hz) & (freqs <= fmax_hz)
    freqs_b = freqs[band]
    mag_b = mag[band, :]

    if db:
        S = 20.0 * np.log10(np.maximum(mag_b, 1e-12))
    else:
        S = mag_b
    return t, freqs_b, S