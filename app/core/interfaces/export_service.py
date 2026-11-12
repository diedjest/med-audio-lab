"""Abstract interface for exporting audio data, figures, and analysis reports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from app.core.models.recording_session import RecordingSession

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class IExportService(ABC):
    """Abstract service defining export capabilities for audio files, plots, and CSV reports."""

    @abstractmethod
    def save_wav(
        self,
        data: np.ndarray,
        sample_rate: int,
        path: str | Path,
        subtype: str = "PCM_16",
    ) -> None:
        """Export audio sample array to a WAV file using soundfile.

        Args:
            data: Audio sample data array (float32).
            sample_rate: Sampling frequency in Hz.
            path: Target file path.
            subtype: Subtype format (e.g., 'PCM_16', 'PCM_24', 'FLOAT').
        """
        ...

    @abstractmethod
    def save_csv(
        self,
        data: np.ndarray,
        sample_rate: int,
        path: str | Path,
    ) -> None:
        """Export raw time-domain audio samples with timestamp column to CSV.

        Args:
            data: Audio sample array.
            sample_rate: Sampling rate in Hz.
            path: Target CSV destination path.
        """
        ...

    @abstractmethod
    def export_plot_png(
        self,
        figure: Figure | Any,
        path: str | Path,
        dpi: int = 300,
    ) -> None:
        """Export a matplotlib Figure instance to a PNG image file.

        Args:
            figure: Matplotlib Figure instance.
            path: Target PNG file path.
            dpi: Dots-per-inch resolution.
        """
        ...

    @abstractmethod
    def export_plot_pdf(
        self,
        figure: Figure | Any,
        path: str | Path,
    ) -> None:
        """Export a matplotlib Figure instance to a vector PDF file.

        Args:
            figure: Matplotlib Figure instance.
            path: Target PDF destination path.
        """
        ...

    @abstractmethod
    def export_report_csv(
        self,
        session: RecordingSession,
        metrics: dict[str, str],
        path: str | Path,
    ) -> None:
        """Export recording metadata and analysis metrics summary to CSV format.

        Args:
            session: RecordingSession entity.
            metrics: Key-value dictionary of analysis metrics.
            path: Target report CSV path.
        """
        ...


__all__: list[str] = ["IExportService"]
