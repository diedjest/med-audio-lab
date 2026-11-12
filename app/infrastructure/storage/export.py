"""Export service: WAV, CSV, matplotlib figures, and session reports."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from app.core.interfaces.export_service import IExportService
from app.core.models.recording_session import RecordingSession
from app.infrastructure.storage import file_io

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class ExportService(IExportService):
    """Writes audio, plots, and reports to disk."""

    def save_wav(
        self,
        data: np.ndarray,
        sample_rate: int,
        path: str | Path,
        subtype: str = "PCM_16",
    ) -> None:
        """Write a WAV file (``PCM_16``, ``PCM_24`` or ``FLOAT_32``)."""
        file_io.save_wav(data, sample_rate, path, subtype)

    def save_csv(
        self,
        data: np.ndarray,
        sample_rate: int,
        path: str | Path,
    ) -> None:
        """Write samples as ``index, time_s, value`` CSV."""
        file_io.save_signal_csv(data, sample_rate, path)

    def export_plot_png(
        self,
        figure: Figure | Any,
        path: str | Path,
        dpi: int = 150,
    ) -> None:
        """Save a matplotlib figure as PNG at the given resolution."""
        if dpi <= 0:
            raise ValueError(f"dpi must be positive, got {dpi}")
        target = self._prepare(path)
        figure.savefig(
            target,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor=figure.get_facecolor(),
        )

    def export_plot_pdf(
        self,
        figure: Figure | Any,
        path: str | Path,
    ) -> None:
        """Save a matplotlib figure as a vector PDF."""
        target = self._prepare(path)
        figure.savefig(
            target,
            format="pdf",
            bbox_inches="tight",
            facecolor=figure.get_facecolor(),
        )

    def export_report_csv(
        self,
        session: RecordingSession,
        metrics: dict[str, str],
        path: str | Path,
    ) -> None:
        """Write session metadata followed by all metrics as a two-column CSV.

        UTF-8 with BOM and ``;`` separators, so Excel opens it correctly.
        """
        target = self._prepare(path)
        with target.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(["Параметр", "Значение"])
            writer.writerow(["ID", "" if session.id is None else session.id])
            writer.writerow(["Дата", session.created_at])
            writer.writerow(["Название", session.title])
            writer.writerow(["Пациент/образец", session.patient])
            writer.writerow(["Комментарий", session.notes])
            writer.writerow(["Источник", session.source])
            writer.writerow(["Частота дискретизации", session.sample_rate])
            writer.writerow(["Длительность, с", f"{session.duration_s:.3f}"])
            writer.writerow(["Файл", session.file_path])
            writer.writerow([])
            for name, value in metrics.items():
                writer.writerow([name, value])

    @staticmethod
    def _prepare(path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target


__all__: list[str] = ["ExportService"]
