"""Background file export (WAV, CSV, PNG, PDF, report CSV)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from app.core.interfaces.export_service import IExportService
from app.core.models.recording_session import RecordingSession


class ExportWorker(QThread):
    """Runs one export job off the GUI thread.

    Build it with a factory (:meth:`wav`, :meth:`csv`, :meth:`png`, :meth:`pdf`,
    :meth:`report_csv`) or pass any zero-argument ``job`` that writes a file and
    returns its path.  Exactly one signal is emitted per run:

    * ``export_complete(path)`` - the written file.
    * ``export_error(message)`` - the job raised (the worker never crashes).

    Audio, session and metrics arguments are copied when the worker is created, so
    the caller may keep modifying them.  A matplotlib ``Figure`` cannot be copied:
    it must not be drawn or modified while a PNG/PDF export is running (use a
    figure that is not attached to the live GUI canvas).  The worker is single-use.
    """

    export_complete = Signal(str)
    export_error = Signal(str)

    def __init__(
        self,
        job: Callable[[], str | Path],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._job = job

    # ---------------------------------------------------------- factories
    @classmethod
    def wav(
        cls,
        service: IExportService,
        data: np.ndarray,
        sample_rate: int,
        path: str | Path,
        subtype: str = "PCM_16",
        parent: QObject | None = None,
    ) -> ExportWorker:
        """Export audio samples as a WAV file."""
        samples = np.array(data, copy=True)
        target = Path(path)

        def job() -> Path:
            service.save_wav(samples, int(sample_rate), target, subtype)
            return target

        return cls(job, parent)

    @classmethod
    def csv(
        cls,
        service: IExportService,
        data: np.ndarray,
        sample_rate: int,
        path: str | Path,
        parent: QObject | None = None,
    ) -> ExportWorker:
        """Export samples as ``index, time_s, value`` CSV."""
        samples = np.array(data, copy=True)
        target = Path(path)

        def job() -> Path:
            service.save_csv(samples, int(sample_rate), target)
            return target

        return cls(job, parent)

    @classmethod
    def png(
        cls,
        service: IExportService,
        figure: Any,
        path: str | Path,
        dpi: int = 150,
        parent: QObject | None = None,
    ) -> ExportWorker:
        """Export a matplotlib figure as PNG."""
        target = Path(path)

        def job() -> Path:
            service.export_plot_png(figure, target, dpi)
            return target

        return cls(job, parent)

    @classmethod
    def pdf(
        cls,
        service: IExportService,
        figure: Any,
        path: str | Path,
        parent: QObject | None = None,
    ) -> ExportWorker:
        """Export a matplotlib figure as a vector PDF."""
        target = Path(path)

        def job() -> Path:
            service.export_plot_pdf(figure, target)
            return target

        return cls(job, parent)

    @classmethod
    def report_csv(
        cls,
        service: IExportService,
        session: RecordingSession,
        metrics: dict[str, str],
        path: str | Path,
        parent: QObject | None = None,
    ) -> ExportWorker:
        """Export a recording's metadata and metrics as a CSV report."""
        session_copy = dataclasses.replace(session)
        metrics_copy = dict(metrics)
        target = Path(path)

        def job() -> Path:
            service.export_report_csv(session_copy, metrics_copy, target)
            return target

        return cls(job, parent)

    # --------------------------------------------------------------- run
    def run(self) -> None:
        try:
            path = self._job()
        except BaseException as exc:  # nothing may escape the thread
            self.export_error.emit(str(exc) or type(exc).__name__)
        else:
            self.export_complete.emit(str(path))


__all__: list[str] = ["ExportWorker"]
