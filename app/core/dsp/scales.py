"""Wavelet scales loading and gap densification routines."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from app.resource_path import get_resource_path


def densify_scale_gaps(
    scales: np.ndarray,
    gap_ratio: float = 1.25,
    max_freq_ratio: float = 1.08,
) -> np.ndarray:
    """Insert intermediate Morlet scales where adjacent central frequencies are too far apart.

    Since f ∝ 1/scale, wide scale gaps can cause narrow-band ultrasonic tones to fall between
    analyzed frequency bins. This densifies gaps with geometric progression.

    Args:
        scales: Array of Morlet wavelet scales.
        gap_ratio: Threshold ratio (s1 / s0) above which gap densification triggers.
        max_freq_ratio: Target maximum frequency ratio between adjacent inserted scales.

    Returns:
        1D float64 array of densified scales sorted in ascending order with unique values.
    """
    scales_arr = np.asarray(scales, dtype=np.float64).ravel()
    if scales_arr.size < 2:
        return scales_arr

    scales_sorted = np.sort(scales_arr)
    out: list[float] = [float(scales_sorted[0])]

    for s0, s1 in zip(scales_sorted[:-1], scales_sorted[1:]):
        s0_val, s1_val = float(s0), float(s1)
        if s0_val <= 0:
            continue
        ratio = s1_val / s0_val
        if ratio > gap_ratio:
            n_insert = int(np.ceil(np.log(ratio) / np.log(max_freq_ratio))) - 1
            if n_insert > 0:
                inserted = np.geomspace(s0_val, s1_val, n_insert + 2, dtype=np.float64)[1:-1]
                out.extend(float(x) for x in inserted)
        out.append(s1_val)

    return np.unique(np.asarray(out, dtype=np.float64))


def load_scales_from_file(path: str | Path | None = None) -> np.ndarray:
    """Load CWT scales from a configuration text file with fallback to geometric default.

    Uses app.resource_path.get_resource_path("wavelet_scales.txt") for default path resolution.

    Args:
        path: Optional file path. If None, resolves via get_resource_path.

    Returns:
        Densified 1D float64 array of valid wavelet scale values (> 0).

    Raises:
        ValueError: If scale values in the specified file are invalid or non-positive.
    """
    default = np.geomspace(2.0, 5986.0, 116).astype(np.float64)

    if path is not None:
        cfg = Path(path)
    else:
        try:
            cfg = get_resource_path("wavelet_scales.txt")
        except FileNotFoundError:
            return densify_scale_gaps(default)

    if not cfg.exists():
        return densify_scale_gaps(default)

    values: list[float] = []
    with cfg.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            values.append(float(line.replace(",", ".")))

    scales = np.asarray(values, dtype=np.float64)
    if scales.size == 0:
        return densify_scale_gaps(default)

    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError(f"Invalid scale values in '{cfg}'. All scales must be > 0.")

    return densify_scale_gaps(scales)


__all__: list[str] = [
    "densify_scale_gaps",
    "load_scales_from_file",
]
