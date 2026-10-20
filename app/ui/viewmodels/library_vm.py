"""State and commands for the recording library tab."""

from __future__ import annotations

from PySide6.QtCore import Property, QObject, Signal, Slot

from app.core.interfaces.export_service import IExportService
from app.core.models.recording_session import RecordingSession
from app.services.project_manager import ProjectManager, RecordingNotFoundError
from app.ui.workers.analysis_worker import AnalysisWorker


class LibraryViewModel(QObject):
    """Wraps :class:`ProjectManager` for browsing, opening and deleting recordings.

    * ``recordings`` (property, notify ``recordings_changed``): newest first.
    * :meth:`refresh` reloads the list.
    * :meth:`load_recording` reads the audio on a background thread and emits
      ``recording_loaded(session, samples)``; only the most recent load is reported.
    * :meth:`delete_recording` removes the library entry and keeps the WAV file;
      :meth:`delete_recording_with_file` also deletes the file (irreversible).
    * :meth:`export_report` writes a recording's report CSV (needs ``export_service``).
    * ``error_occurred(message)`` reports any failure.
    """

    recordings_changed = Signal()
    is_loading_changed = Signal(bool)
    recording_loaded = Signal(object, object)  # (RecordingSession, np.ndarray)
    recording_deleted = Signal(int)
    report_exported = Signal(str)  # path of the written report
    error_occurred = Signal(str)

    def __init__(
        self,
        project_manager: ProjectManager,
        parent: QObject | None = None,
        *,
        export_service: IExportService | None = None,
    ) -> None:
        super().__init__(parent)
        self._pm = project_manager
        self._export = export_service
        self._recordings: list[RecordingSession] = []
        self._loading = False
        self._active_worker: AnalysisWorker | None = None
        self._workers: set[AnalysisWorker] = set()  # keeps running threads alive

    # ----------------------------------------------------------- properties
    @Property(object, notify=recordings_changed)
    def recordings(self) -> list[RecordingSession]:
        """A copy of the current list of recordings, newest first."""
        return list(self._recordings)

    @Property(bool, notify=is_loading_changed)
    def is_loading(self) -> bool:
        """True while a recording's audio is being read."""
        return self._loading

    def _set_loading(self, loading: bool) -> None:
        if loading != self._loading:
            self._loading = loading
            self.is_loading_changed.emit(loading)

    # ---------------------------------------------------------------- slots
    @Slot()
    def refresh(self) -> None:
        """Reload the list of recordings from the database."""
        try:
            recordings = self._pm.list_recordings()
        except Exception as exc:
            self.error_occurred.emit(f"Не удалось прочитать библиотеку записей: {exc}")
            return
        self._recordings = list(recordings)
        self.recordings_changed.emit()

    @Slot(int)
    def load_recording(self, recording_id: int) -> None:
        """Load a recording's metadata and audio in the background."""
        worker = AnalysisWorker(
            self._pm.load_recording, (int(recording_id),), task_name="load_recording"
        )
        worker.result_ready.connect(self._on_loaded)
        worker.error.connect(self._on_load_error)
        worker.finished.connect(self._on_worker_finished)
        self._workers.add(worker)
        self._active_worker = worker  # earlier loads become stale
        self._set_loading(True)
        worker.start()

    @Slot(int)
    def delete_recording(self, recording_id: int) -> None:
        """Remove a recording from the library; its WAV file is kept on disk."""
        self._delete(recording_id, delete_file=False)

    @Slot(int)
    def delete_recording_with_file(self, recording_id: int) -> None:
        """Remove a recording from the library **and delete its WAV file**."""
        self._delete(recording_id, delete_file=True)

    def _delete(self, recording_id: int, delete_file: bool) -> None:
        try:
            self._pm.delete_recording(int(recording_id), delete_file=delete_file)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            self.refresh()  # the record may be gone even if the file could not be deleted
            return
        self.refresh()
        self.recording_deleted.emit(int(recording_id))

    @Slot(int, str)
    def export_report(self, recording_id: int, path: str) -> None:
        """Write a CSV report (metadata and metrics) for one recording, in the background.

        Missing metrics are computed from the audio first and stored.  Emits
        ``report_exported(path)`` or ``error_occurred(message)``.
        """
        if self._export is None:
            self.error_occurred.emit("Отчёты нельзя экспортировать: сервис экспорта не настроен.")
            return
        worker = AnalysisWorker(
            self._build_report, (int(recording_id), str(path)), task_name="export_report"
        )
        worker.result_ready.connect(self._on_report_done)
        worker.error.connect(self._on_report_error)
        worker.finished.connect(self._on_worker_finished)
        self._workers.add(worker)
        worker.start()

    def _build_report(self, recording_id: int, path: str) -> str:
        session = self._pm.get_recording(recording_id)
        if session is None:
            raise RecordingNotFoundError(f"Запись {recording_id} не найдена.")
        metrics = self._pm.ensure_metrics(recording_id)
        assert self._export is not None
        self._export.export_report_csv(session, metrics, path)
        return path

    @Slot(str, object)
    def _on_report_done(self, task_name: str, result: object) -> None:
        self.report_exported.emit(str(result))

    @Slot(str, str)
    def _on_report_error(self, task_name: str, message: str) -> None:
        self.error_occurred.emit(message)

    def shutdown(self, timeout_ms: int = 5000) -> None:
        """Wait for background loads and exports to finish (call when the window closes)."""
        for worker in list(self._workers):
            worker.wait(timeout_ms)

    # -------------------------------------------------------- worker slots
    @Slot(str, object)
    def _on_loaded(self, task_name: str, result: object) -> None:
        if self.sender() is not self._active_worker:
            return  # a newer load superseded this one
        self._active_worker = None
        self._set_loading(False)
        session, samples = result  # type: ignore[misc]
        self.recording_loaded.emit(session, samples)

    @Slot(str, str)
    def _on_load_error(self, task_name: str, message: str) -> None:
        if self.sender() is not self._active_worker:
            return
        self._active_worker = None
        self._set_loading(False)
        self.error_occurred.emit(message)

    @Slot()
    def _on_worker_finished(self) -> None:
        worker = self.sender()
        if isinstance(worker, AnalysisWorker):
            self._workers.discard(worker)
            worker.deleteLater()


__all__: list[str] = ["LibraryViewModel"]
