"""Memory-bounded Continuous Wavelet Transform using frequency-domain FFT convolution
with Complex Morlet wavelet. Designed for ultrasonic signals up to 384 kHz.

Key design decisions:
- Scales are processed in chunks of `chunk_scales` to bound peak memory
- Signal is zero-padded to next power-of-two for FFT efficiency
- Negative frequencies zeroed (analytic wavelet)
- Scale gap densification ensures sub-bin frequency resolution
- Progress callback enables responsive UI during long computations
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np

from app.core.dsp.scales import densify_scale_gaps, load_scales_from_file
from app.core.models.analysis_result import CwtResult

try:
    from scipy.signal import resample_poly
except ImportError:
    resample_poly = None


def _next_pow2(n: int) -> int:
    """Next power of two >= n."""
    return 1 if n <= 1 else 1 << (int(n - 1).bit_length())


def _densify_scale_gaps(
    scales: np.ndarray,
    *,
    gap_ratio: float = 1.25,
    max_freq_ratio: float = 1.08,
) -> np.ndarray:
    """Backward-compatible alias for :func:`app.core.dsp.scales.densify_scale_gaps`."""
    return densify_scale_gaps(scales, gap_ratio, max_freq_ratio)


def load_scales(path: str | Path | None = None) -> np.ndarray:
    """Backward-compatible alias for :func:`app.core.dsp.scales.load_scales_from_file`."""
    return load_scales_from_file(path)


def _resample_to_target(
    x: np.ndarray, fs: int, target_fs: int
) -> tuple[np.ndarray, int]:
    """Resample signal to target sample rate using polyphase filtering."""
    if target_fs <= 0 or fs == target_fs:
        return x, fs
    if resample_poly is not None:
        y = resample_poly(x, int(target_fs), int(fs)).astype(np.float32, copy=False)
        return y, int(target_fs)
    # Fallback: linear interpolation
    n_out = max(1, int(round(x.size * target_fs / fs)))
    t_in = np.arange(x.size, dtype=np.float64) / fs
    t_out = np.arange(n_out, dtype=np.float64) / target_fs
    return np.interp(t_out, t_in, x).astype(np.float32, copy=False), int(target_fs)


def _effective_length(
    n_samples: int,
    fs: int,
    target_fs: int | None,
    max_seconds: float | None,
) -> tuple[int, int]:
    """Signal length and sample rate after the optional resampling and truncation."""
    n, rate = n_samples, fs
    if target_fs is not None and target_fs > 0 and rate != target_fs:
        g = math.gcd(int(target_fs), int(rate))
        n = -(-n * (int(target_fs) // g) // (int(rate) // g))  # resample_poly: ceil
        rate = int(target_fs)
    if max_seconds is not None and max_seconds > 0:
        max_n = int(max_seconds * rate)
        if max_n > 0 and n > max_n:
            n = max_n
    return n, rate


def lateral_slice(result: CwtResult, t0_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Frequency spectrum at the time instant nearest to ``t0_s``.

    Returns:
        ``(frequencies_hz, amplitude)`` arrays.
    """
    t_idx = int(np.argmin(np.abs(result.time - t0_s)))
    amp = np.abs(result.coefficients[:, t_idx])
    return result.frequencies_hz.copy(), amp.astype(np.float64)


def longitudinal_slice(result: CwtResult, f0_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Temporal cross-section Re(W_x) at the scale whose frequency is nearest ``f0_hz``.

    Returns:
        ``(time_relative_s, normalized_real_part)``, centred on the largest excursion.
    """
    f_idx = int(np.argmin(np.abs(result.frequencies_hz - f0_hz)))
    wx_row = result.coefficients[f_idx, :]
    re = np.real(wx_row.astype(np.complex128))
    denom = float(np.max(np.abs(re)))
    if denom > 0:
        re = re / denom
    t0 = float(result.time[int(np.argmax(np.abs(re)))])
    t_rel = result.time - t0
    return t_rel.astype(np.float64), re.astype(np.float64)


def ridge_detection(result: CwtResult) -> tuple[np.ndarray, np.ndarray]:
    """Ridge line: the dominant frequency at every time step.

    Returns:
        ``(time, ridge_frequencies_hz)`` arrays.
    """
    amp = np.abs(result.coefficients)
    ridge_idx = np.argmax(amp, axis=0)
    return result.time.copy(), result.frequencies_hz[ridge_idx].copy()


class CwtEngine:
    """Memory-bounded Morlet CWT with chunked scale processing.

    Architecture:
    1. Signal is peak-normalized and optionally resampled
    2. Zero-padded to N_fft = next_pow2(2*N - 1) for linear convolution via FFT
    3. Morlet wavelets constructed in frequency domain: ψ̂(ω) = √s · exp(-½(sω - ω₀)²)
    4. Scales processed in batches of `chunk_scales` to limit memory to ~O(chunk × N_fft)
    5. Negative frequencies zeroed to enforce analytic wavelet

    Memory budget:
    - Peak allocation ≈ chunk_scales × N_fft × 40 bytes (float64 wavelet +
      two complex128 intermediates) on top of the n_scales × N complex64 output
    - For N=1M samples, chunk=8: ~640 MB working set, so the budget check
      is enforced before any large allocation
    """

    def __init__(
        self,
        w0: float = 2.0 * np.pi,
        memory_budget_mb: float = 800.0,
        chunk_scales: int = 8,
    ) -> None:
        self._w0 = float(w0)
        self._memory_budget_mb = float(memory_budget_mb)
        self._chunk_scales = max(1, int(chunk_scales))

    @property
    def memory_budget_mb(self) -> float:
        """Memory budget in MB enforced by :meth:`compute`."""
        return self._memory_budget_mb

    @property
    def chunk_scales(self) -> int:
        """Number of scales processed per batch."""
        return self._chunk_scales

    def select_scales(
        self,
        fs: int,
        *,
        fmin_hz: float = 100.0,
        fmax_hz: float = 100_000.0,
        scales: np.ndarray | None = None,
        scale_stride: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Choose the scales (and their centre frequencies) used for a computation.

        Args:
            fs: Sample rate in Hz the CWT will run at.
            fmin_hz: Minimum frequency of interest.
            fmax_hz: Maximum frequency of interest.
            scales: Custom scale array; loaded from the config file if None.
            scale_stride: Keep every N-th scale before band filtering.

        Returns:
            ``(scales, frequencies_hz)`` restricted to ``[fmin_hz, fmax_hz]``.

        Raises:
            ValueError: if no scale falls inside the frequency band.
        """
        if scales is None:
            scales = load_scales()
        scales = np.asarray(scales, dtype=np.float64)

        stride = max(1, int(scale_stride))
        if stride > 1:
            scales = scales[::stride]

        # Convert scales to center frequencies: f = w0 * fs / (2π * s)
        freqs_hz = self._w0 * fs / (2.0 * np.pi * scales)

        band = (freqs_hz >= float(fmin_hz)) & (freqs_hz <= float(fmax_hz))
        if not np.any(band):
            raise ValueError(
                f"No scales in frequency range {fmin_hz:.1f}–{fmax_hz:.1f} Hz. "
                f"Available: {freqs_hz.min():.1f}–{freqs_hz.max():.1f} Hz."
            )
        return scales[band], freqs_hz[band]

    def estimate_load(
        self,
        n_samples: int,
        fs: int,
        *,
        fmin_hz: float = 100.0,
        fmax_hz: float = 100_000.0,
        scales: np.ndarray | None = None,
        max_seconds: float | None = None,
        target_fs: int | None = None,
        scale_stride: int = 1,
    ) -> dict[str, float | int | bool | None]:
        """Pre-flight estimate of what :meth:`compute` would need, without computing.

        Takes the same selection arguments as :meth:`compute`.

        Returns:
            A dict with ``n_samples`` / ``sample_rate`` / ``duration_s`` (after
            resampling and truncation), ``n_scales``, ``n_fft``, ``estimated_mb``,
            ``budget_mb``, ``fits_budget``, and - only when the job does not fit -
            ``suggested_max_seconds`` and ``suggested_scale_stride`` (each None if
            no such change would make it fit).

        Raises:
            ValueError: signal too short, or no scales in the frequency band.
        """
        n, eff_fs = _effective_length(int(n_samples), int(fs), target_fs, max_seconds)
        if n < 8:
            raise ValueError("Signal too short for CWT (need >= 8 samples).")

        selection = dict(fmin_hz=fmin_hz, fmax_hz=fmax_hz, scales=scales)
        chosen, _ = self.select_scales(eff_fs, scale_stride=scale_stride, **selection)
        n_scales = int(chosen.size)
        estimated = self.estimate_memory_mb(n, n_scales)
        fits = estimated <= self._memory_budget_mb

        suggested_seconds: float | None = None
        suggested_stride: int | None = None
        if not fits:
            lo, hi = 8, n
            if self.estimate_memory_mb(lo, n_scales) <= self._memory_budget_mb:
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if self.estimate_memory_mb(mid, n_scales) <= self._memory_budget_mb:
                        lo = mid
                    else:
                        hi = mid - 1
                suggested_seconds = lo / eff_fs
            for stride in range(max(2, int(scale_stride) + 1), 65):
                try:
                    coarser, _ = self.select_scales(eff_fs, scale_stride=stride, **selection)
                except ValueError:
                    break  # so few scales remain that none falls in the band: no usable stride
                if self.estimate_memory_mb(n, int(coarser.size)) <= self._memory_budget_mb:
                    suggested_stride = stride
                    break

        result: dict[str, float | int | bool | None] = {
            "n_samples": n,
            "sample_rate": eff_fs,
            "duration_s": n / eff_fs,
            "n_scales": n_scales,
            "n_fft": _next_pow2(2 * n - 1),
            "estimated_mb": estimated,
            "budget_mb": self._memory_budget_mb,
            "fits_budget": fits,
        }
        if not fits:
            result["suggested_max_seconds"] = suggested_seconds
            result["suggested_scale_stride"] = suggested_stride
        return result

    def estimate_memory_mb(
        self, n_samples: int, n_scales: int
    ) -> float:
        """Estimate peak memory usage in MB for given dimensions."""
        n_fft = _next_pow2(2 * n_samples - 1)
        mb = 1024 * 1024
        # Output matrix: n_scales × n_samples × 8 bytes (complex64)
        output_mb = n_scales * n_samples * 8 / mb
        # Signal spectrum X (complex128) plus the frequency/omega axes (float64)
        spectrum_mb = n_fft * (16 + 8 + 8) / mb
        # Working set per chunk: psi_hat (float64) + product and IFFT result
        # (both complex128) = 40 bytes per element
        working_mb = self._chunk_scales * n_fft * 40 / mb
        return output_mb + spectrum_mb + working_mb

    def compute(
        self,
        x: np.ndarray,
        fs: int,
        *,
        fmin_hz: float = 100.0,
        fmax_hz: float = 100_000.0,
        scales: np.ndarray | None = None,
        max_seconds: float | None = None,
        target_fs: int | None = None,
        scale_stride: int = 1,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> CwtResult:
        """Compute the Continuous Wavelet Transform.

        Args:
            x: Input signal (1D float32).
            fs: Sample rate in Hz.
            fmin_hz: Minimum frequency of interest.
            fmax_hz: Maximum frequency of interest.
            scales: Custom scale array. If None, loaded from config file.
            max_seconds: Truncate signal to this duration.
            target_fs: Resample to this rate before CWT.
            scale_stride: Skip every N-th scale (for memory reduction).
            progress_callback: Called with (fraction, message) during computation.

        Returns:
            CwtResult with time axis, frequency axis, and complex coefficients.

        Raises:
            ValueError: If signal is too short or no scales fall in frequency range.
            MemoryError: If estimated memory exceeds budget.
        """
        # --- Preprocessing ---
        x = np.asarray(x, dtype=np.float32).ravel()
        if x.size < 8:
            raise ValueError("Signal too short for CWT (need >= 8 samples).")

        # Peak normalize
        peak = float(np.max(np.abs(x)))
        if peak > 0:
            x = (x / peak).astype(np.float32, copy=False)

        # Resample if requested
        if target_fs is not None and target_fs > 0 and fs != target_fs:
            x, fs = _resample_to_target(x, fs, target_fs)

        # Truncate
        if max_seconds is not None and max_seconds > 0:
            max_n = int(max_seconds * fs)
            if max_n > 0 and x.size > max_n:
                x = x[:max_n]

        # --- Scale selection ---
        scales, freqs_hz = self.select_scales(
            fs,
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
            scales=scales,
            scale_stride=scale_stride,
        )

        # --- Memory budget check ---
        mem_mb = self.estimate_memory_mb(x.size, scales.size)
        if mem_mb > self._memory_budget_mb:
            raise MemoryError(
                f"CWT would require ~{mem_mb:.0f} MB (budget: {self._memory_budget_mb:.0f} MB). "
                f"Reduce signal duration, increase scale_stride, or narrow frequency range."
            )

        # --- FFT of signal ---
        n = x.size
        n_fft = _next_pow2(2 * n - 1)
        X = np.fft.fft(x.astype(np.float32, copy=False), n=n_fft)
        f_fft = np.fft.fftfreq(n_fft, d=1.0 / fs)

        # --- Chunked wavelet convolution ---
        omega_d = 2.0 * np.pi * f_fft / float(fs)
        neg_mask = f_fft < 0
        Wx = np.empty((scales.size, n), dtype=np.complex64)

        total_chunks = (scales.size + self._chunk_scales - 1) // self._chunk_scales

        for chunk_idx, c0 in enumerate(range(0, scales.size, self._chunk_scales)):
            c1 = min(c0 + self._chunk_scales, scales.size)
            sc = scales[c0:c1]
            s_col = sc[:, np.newaxis]  # (chunk, 1)

            # Morlet wavelet in frequency domain:
            # ψ̂(ω) = √s · exp(-½(s·ω - ω₀)²)
            psi_hat = np.exp(
                -0.5 * ((s_col * omega_d - self._w0) ** 2),
                dtype=np.float64,
            )
            np.multiply(psi_hat, np.sqrt(s_col), out=psi_hat)

            # Zero negative frequencies (analytic wavelet)
            psi_hat[:, neg_mask] = 0.0

            # Convolution in frequency domain: W(s,τ) = IFFT[X(ω) · ψ̂(s,ω)]  (ψ̂ is real, so ψ̂* = ψ̂)
            conv = np.fft.ifft(X * psi_hat, axis=1)
            Wx[c0:c1, :] = conv[:, :n]

            # Report progress
            if progress_callback is not None:
                frac = (chunk_idx + 1) / total_chunks
                progress_callback(
                    frac,
                    f"CWT: scales {c0+1}–{c1} of {scales.size} "
                    f"({frac*100:.0f}%)",
                )

        t = np.arange(n, dtype=np.float64) / fs

        return CwtResult(
            fs=fs,
            time=t,
            frequencies_hz=freqs_hz,
            coefficients=Wx,
            scales=scales,
        )

    def lateral_slice(
        self, result: CwtResult, t0_s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Frequency spectrum at a specific time instant (see :func:`lateral_slice`)."""
        return lateral_slice(result, t0_s)

    def longitudinal_slice(
        self, result: CwtResult, f0_hz: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Temporal cross-section at a specific frequency (see :func:`longitudinal_slice`)."""
        return longitudinal_slice(result, f0_hz)

    def ridge_detection(
        self, result: CwtResult
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ridge line, dominant frequency vs time (see :func:`ridge_detection`)."""
        return ridge_detection(result)


__all__: list[str] = [
    "CwtResult",
    "CwtEngine",
    "lateral_slice",
    "longitudinal_slice",
    "ridge_detection",
    "load_scales",
    "_next_pow2",
    "_densify_scale_gaps",
    "_resample_to_target",
]
