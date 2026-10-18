"""Main window and application bootstrap.

``build_context`` is the composition root: it creates the infrastructure, the
services and the view models and hands them to the tabs, so the dependency chain
is::

    infrastructure (engines, SQLite, export) -> services -> view models -> tabs

``MainWindow`` only connects the tabs to each other (a saved recording or a library
entry becomes the Analysis tab's signal), owns the menus and the status bar, and
shuts everything down cleanly.
"""

from __future__ import annotations

import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np
from PySide6.QtCore import QByteArray, QTimer, Qt
from PySide6.QtGui import QAction, QCloseEvent, QGuiApplication, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QTabWidget,
    QWidget,
)

from app import __version__
from app.constants import APP_NAME, MAX_CWT_MEMORY_MB
from app.core.dsp.cwt import CwtEngine
from app.core.dsp.filters import FilterPipeline
from app.core.dsp.impulse_detector import ImpulseDetector
from app.core.interfaces.audio_engine import IAudioCaptureEngine, IAudioPlaybackEngine
from app.core.interfaces.export_service import IExportService
from app.core.interfaces.serial_engine import ISerialEngine
from app.core.models.device_info import AudioDeviceInfo, SerialPortInfo
from app.core.models.recording_session import RecordingSession
from app.infrastructure.audio.capture_engine import CaptureEngine
from app.infrastructure.audio.playback_engine import PlaybackEngine
from app.infrastructure.serial.serial_engine import SerialCaptureEngine
from app.infrastructure.storage.export import ExportService
from app.infrastructure.storage.sqlite_repository import SqliteRepository
from app.resource_path import get_resource_path
from app.services.analysis_service import AnalysisService
from app.services.project_manager import ProjectManager
from app.services.recording_service import RecordingService
from app.services.settings_service import SettingsService
from app.ui.dialogs.settings_dialog import SettingsDialog
from app.ui.tabs.analysis_tab import AnalysisTab
from app.ui.tabs.library_tab import LibraryTab
from app.ui.tabs.record_tab import RecordTab
from app.ui.viewmodels.analysis_vm import AnalysisViewModel
from app.ui.viewmodels.library_vm import LibraryViewModel
from app.ui.viewmodels.recording_vm import RecordingViewModel
from app.ui.workers.analysis_worker import AnalysisWorker
from app.ui.workers.export_worker import ExportWorker

WINDOW_TITLE = "MedAudioLab — Ultrasound Signal Analysis Workstation"
ORGANIZATION_NAME = "MedAudioLab"
DEFAULT_SIZE = (1400, 900)
MINIMUM_SIZE = (1100, 700)
STYLESHEET_RESOURCE = "styles/light.qss"
ICON_RESOURCE = "icons/app_icon.png"

TAB_RECORD, TAB_ANALYSIS, TAB_LIBRARY = 0, 1, 2

_ERROR_MESSAGE_MS = 10_000
_INFO_MESSAGE_MS = 6_000
_SIGINT_POLL_MS = 250
_WORKER_WAIT_MS = 5_000
_FALLBACK_TITLE = "Edited signal"

#: What the user picked when closing during a recording / with an unsaved capture.
CloseChoice = Literal["save", "discard"] | None
ClosePrompt = Callable[[QWidget, str], CloseChoice]


# ===================================================================== bootstrap
@dataclass
class AppContext:
    """Everything the user interface needs, fully wired (see :func:`build_context`)."""

    repository: SqliteRepository
    settings: SettingsService
    project_manager: ProjectManager
    export_service: IExportService
    audio_engine: IAudioCaptureEngine
    serial_engine: ISerialEngine
    playback_engine: IAudioPlaybackEngine
    recording_service: RecordingService
    analysis_service: AnalysisService
    recording_vm: RecordingViewModel
    analysis_vm: AnalysisViewModel
    library_vm: LibraryViewModel
    audio_devices: Callable[[], Sequence[AudioDeviceInfo]] | None = None
    serial_ports: Callable[[], Sequence[SerialPortInfo]] | None = None


def build_context(
    db_path: str | Path | None = None,
    *,
    audio_engine: IAudioCaptureEngine | None = None,
    serial_engine: ISerialEngine | None = None,
    playback_engine: IAudioPlaybackEngine | None = None,
    audio_devices: Callable[[], Sequence[AudioDeviceInfo]] | None = None,
    serial_ports: Callable[[], Sequence[SerialPortInfo]] | None = None,
) -> AppContext:
    """Create the infrastructure, services and view models.

    Every argument is optional: the defaults are the real hardware engines and the
    database in ``~/Documents/MedAudioLab``; tests pass fakes and a temporary path.
    """
    repository = SqliteRepository(db_path)
    repository.ensure_schema()
    settings = SettingsService(repository)
    export_service = ExportService()
    project_manager = ProjectManager(repository, settings)

    audio = audio_engine or CaptureEngine()
    serial = serial_engine or SerialCaptureEngine()
    playback = playback_engine or PlaybackEngine()

    recording_service = RecordingService(audio, serial, project_manager, settings)
    analysis_service = AnalysisService(
        CwtEngine(memory_budget_mb=MAX_CWT_MEMORY_MB), FilterPipeline(), ImpulseDetector()
    )
    return AppContext(
        repository=repository,
        settings=settings,
        project_manager=project_manager,
        export_service=export_service,
        audio_engine=audio,
        serial_engine=serial,
        playback_engine=playback,
        recording_service=recording_service,
        analysis_service=analysis_service,
        recording_vm=RecordingViewModel(recording_service, settings),
        analysis_vm=AnalysisViewModel(analysis_service),
        library_vm=LibraryViewModel(project_manager, export_service=export_service),
        audio_devices=audio_devices,
        serial_ports=serial_ports,
    )


# ======================================================================= resources
def load_stylesheet() -> str:
    """Contents of ``resources/styles/light.qss`` ("" if it cannot be read)."""
    try:
        return get_resource_path(STYLESHEET_RESOURCE).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""


def load_icon() -> QIcon:
    """The application icon from ``resources/icons`` (an empty icon if it is missing)."""
    try:
        return QIcon(str(get_resource_path(ICON_RESOURCE)))
    except FileNotFoundError:
        return QIcon()


def _ask_close_choice(parent: QWidget, kind: str) -> CloseChoice:
    """Default close confirmation: ``kind`` is ``"recording"`` or ``"unsaved"``."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Quit MedAudioLab")
    save = discard = None
    if kind == "recording":
        box.setText("A recording is in progress.")
        box.setInformativeText("Save it before quitting, or discard the audio captured so far?")
        save = box.addButton("Stop and save", QMessageBox.ButtonRole.AcceptRole)
    else:
        box.setText("The last recording could not be saved and is still in memory.")
        box.setInformativeText("Try to save it again, or discard it and quit?")
        save = box.addButton("Try saving again", QMessageBox.ButtonRole.AcceptRole)
    discard = box.addButton("Discard and quit", QMessageBox.ButtonRole.DestructiveRole)
    cancel = box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(save)
    box.setEscapeButton(cancel)
    box.exec()
    clicked = box.clickedButton()
    if clicked is save:
        return "save"
    if clicked is discard:
        return "discard"
    return None


# ====================================================================== main window
class MainWindow(QMainWindow):
    """The application window: Record, Analysis and Library tabs plus menus and a status bar.

    Args:
        context: The wired application; ``build_context()`` is used if omitted.
        close_prompt: ``(parent, "recording" | "unsaved") -> "save" | "discard" | None``;
            the default shows a dialog.  Injectable so quitting can be driven in tests.
        settings_dialog_factory: ``(settings, parent) -> dialog``; ``exec()`` is called on it.
    """

    def __init__(
        self,
        context: AppContext | None = None,
        *,
        close_prompt: ClosePrompt | None = None,
        settings_dialog_factory: Callable[[SettingsService, QWidget], Any] | None = None,
    ) -> None:
        super().__init__()
        self._ctx = context or build_context()
        self._close_prompt = close_prompt or _ask_close_choice
        self._settings_dialog_factory = settings_dialog_factory or SettingsDialog
        self._workers: set[Any] = set()
        self._closing = False

        self.setWindowTitle(WINDOW_TITLE)
        self.setWindowIcon(load_icon())
        self.setMinimumSize(*MINIMUM_SIZE)
        self.resize(*DEFAULT_SIZE)
        stylesheet = load_stylesheet()
        if stylesheet:
            self.setStyleSheet(stylesheet)

        self._build_tabs()
        self._build_menus()
        self._build_status_bar()
        self._connect()
        self._restore_geometry()
        self._ctx.library_vm.refresh()  # the Library tab is hidden at startup; load it now
        self._refresh_state()
        self.statusBar().showMessage("Ready", _INFO_MESSAGE_MS)

    # ============================================================== accessors
    @property
    def context(self) -> AppContext:
        return self._ctx

    @property
    def tabs(self) -> QTabWidget:
        return self._tabs

    @property
    def record_tab(self) -> RecordTab:
        return self._record_tab

    @property
    def analysis_tab(self) -> AnalysisTab:
        return self._analysis_tab

    @property
    def library_tab(self) -> LibraryTab:
        return self._library_tab

    @property
    def state_text(self) -> str:
        """The application state shown in the status bar."""
        return self._state_label.text()

    # ================================================================== build
    def _build_tabs(self) -> None:
        ctx = self._ctx
        self._record_tab = RecordTab(
            ctx.recording_vm,
            ctx.settings,
            ctx.project_manager,
            ctx.playback_engine,
            audio_devices=ctx.audio_devices,
            serial_ports=ctx.serial_ports,
        )
        self._analysis_tab = AnalysisTab(ctx.analysis_vm, ctx.settings)
        self._library_tab = LibraryTab(ctx.library_vm, ctx.settings)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._record_tab, "Record")
        self._tabs.addTab(self._analysis_tab, "Analysis")
        self._tabs.addTab(self._library_tab, "Library")
        self.setCentralWidget(self._tabs)

    def _action(self, menu: QMenu, text: str, slot: Callable[[], Any], shortcut: str | QKeySequence.StandardKey | None = None) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(lambda _checked=False: slot())
        menu.addAction(action)
        return action

    def _build_menus(self) -> None:
        bar = self.menuBar()

        file_menu = bar.addMenu("&File")
        self._open_action = self._action(file_menu, "&Open...", self.open_file, QKeySequence.StandardKey.Open)
        self._save_action = self._action(
            file_menu, "&Save as new recording...", self.save_current_signal, QKeySequence.StandardKey.Save
        )
        export_menu = file_menu.addMenu("&Export")
        self._export_png_action = self._action(export_menu, "Current plot as &PNG...", self.export_plot_png)
        self._export_wav_action = self._action(export_menu, "Signal as &WAV...", self.export_signal_wav)
        self._export_csv_action = self._action(export_menu, "Signal as &CSV...", self.export_signal_csv)
        self._export_segment_action = self._action(
            export_menu, "Selected &segment as WAV...", self.export_segment_wav
        )
        self._export_impulses_action = self._action(
            export_menu, "Detected &impulses as CSV...", self.export_impulses_csv
        )
        file_menu.addSeparator()
        self._quit_action = self._action(file_menu, "&Quit", self.close, "Ctrl+Q")

        edit_menu = bar.addMenu("&Edit")
        self._settings_action = self._action(edit_menu, "&Settings...", self.open_settings, "Ctrl+,")

        help_menu = bar.addMenu("&Help")
        self._about_action = self._action(help_menu, f"&About {APP_NAME}", self.show_about, "F1")

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        self._state_label = QLabel("Ready")
        self._state_label.setObjectName("status")
        self._signal_label = QLabel("No signal")
        self._library_label = QLabel("Library: -")
        for label in (self._state_label, self._signal_label, self._library_label):
            bar.addPermanentWidget(label)

    # ================================================================== wiring
    def _connect(self) -> None:
        ctx = self._ctx
        rec, ana, lib = ctx.recording_vm, ctx.analysis_vm, ctx.library_vm

        # tabs -> tabs
        self._record_tab.audio_loaded.connect(self._on_record_audio)
        self._library_tab.recording_loaded.connect(self.open_recording)
        rec.recording_stopped.connect(lambda _session: lib.refresh())
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # messages
        for tab in (self._record_tab, self._analysis_tab, self._library_tab):
            tab.error_occurred.connect(self._show_error)
        rec.error_occurred.connect(self._show_error)
        rec.status_text_changed.connect(self._show_info)

        # state shown in the status bar and enabled state of the actions
        for signal_ in (rec.is_recording_changed, rec.is_paused_changed, rec.has_unsaved_recording_changed):
            signal_.connect(self._refresh_state)
        ana.is_busy_changed.connect(self._refresh_state)
        lib.is_loading_changed.connect(self._refresh_state)
        for signal_ in (
            ana.current_data_changed,
            ana.selection_changed,
            ana.current_impulses_changed,
        ):
            signal_.connect(self._refresh_state)
        ana.current_sample_rate_changed.connect(lambda _r: self._refresh_state())
        lib.recordings_changed.connect(self._refresh_state)

    def _show_info(self, text: str) -> None:
        self.statusBar().showMessage(text, _INFO_MESSAGE_MS)

    def _show_error(self, text: str) -> None:
        self.statusBar().showMessage(f"Error: {text}" if not text.startswith("Error") else text, _ERROR_MESSAGE_MS)

    def _refresh_state(self, *_: object) -> None:
        ctx = self._ctx
        rec, ana = ctx.recording_vm, ctx.analysis_vm
        if rec.is_recording:
            state = "Paused" if rec.is_paused else "Recording"
        elif rec.has_unsaved_recording:
            state = "Unsaved recording"
        elif ana.is_busy:
            state = "Analyzing..."
        elif ctx.library_vm.is_loading:
            state = "Loading recording..."
        else:
            state = "Ready"
        self._state_label.setText(state)

        if ana.current_data.size:
            duration = ana.current_data.size / max(1, ana.current_sample_rate)
            self._signal_label.setText(f"Signal: {duration:.3f} s at {ana.current_sample_rate} Hz")
        else:
            self._signal_label.setText("No signal")
        self._library_label.setText(f"Library: {len(ctx.library_vm.recordings)} recordings")

        has_signal = ana.current_data.size > 0
        for action in (self._save_action, self._export_png_action, self._export_wav_action, self._export_csv_action):
            action.setEnabled(has_signal)
        self._export_segment_action.setEnabled(ana.selection is not None)
        self._export_impulses_action.setEnabled(bool(ana.current_impulses))
        self._settings_action.setEnabled(not rec.is_recording)

    def _on_tab_changed(self, index: int) -> None:
        if index == TAB_LIBRARY:
            self._library_tab.refresh()

    # ============================================================ tab to tab
    def _on_record_audio(self, samples: np.ndarray, sample_rate: int, label: str) -> None:
        """A recording was saved (or a file loaded) on the Record tab: analyse it."""
        self._analysis_tab.set_signal(samples, sample_rate, label)
        self._show_info(f"'{label}' is ready in the Analysis tab.")

    def open_recording(self, session: RecordingSession, samples: np.ndarray) -> None:
        """Show a library recording in the Analysis tab and switch to it."""
        self._analysis_tab.set_signal(samples, session.sample_rate, session.title)
        self._tabs.setCurrentIndex(TAB_ANALYSIS)

    # ================================================================== menus
    def open_file(self, path: str | Path | None = None) -> None:
        """File > Open: load a WAV / TXT / CSV file into the Analysis tab."""
        self._tabs.setCurrentIndex(TAB_ANALYSIS)
        self._analysis_tab.load_file(path)

    def save_current_signal(self, title: str | None = None) -> None:
        """File > Save: store the Analysis tab's current signal as a new library recording.

        The recording's source is recorded as "microphone", since a processed signal
        has no capture source of its own.
        """
        analysis = self._ctx.analysis_vm
        if analysis.current_data.size == 0:
            self._show_info("There is no signal to save.")
            return
        if title is None:
            title, accepted = QInputDialog.getText(
                self, "Save as new recording", "Title:", text=self._analysis_tab.label or _FALLBACK_TITLE
            )
            if not accepted:
                return
        data = np.array(analysis.current_data, copy=True)
        worker = AnalysisWorker(
            self._ctx.project_manager.save_recording,
            (data, analysis.current_sample_rate, title.strip() or _FALLBACK_TITLE, "",
             "Saved from the Analysis tab", "microphone"),
            task_name="save_recording",
        )
        worker.result_ready.connect(self._on_signal_saved)
        worker.error.connect(lambda _name, message: self._show_error(f"Cannot save the recording: {message}"))
        self._start_worker(worker)

    def _on_signal_saved(self, _name: str, session: RecordingSession) -> None:
        self._ctx.library_vm.refresh()
        self._show_info(f"Saved '{session.title}' to the library.")

    def _start_worker(self, worker: Any) -> None:
        self._workers.add(worker)
        worker.finished.connect(lambda w=worker: self._workers.discard(w))
        worker.start()

    def _export(self, worker: ExportWorker) -> None:
        worker.export_complete.connect(lambda path: self._show_info(f"Exported: {path}"))
        worker.export_error.connect(lambda message: self._show_error(f"Export failed: {message}"))
        self._start_worker(worker)

    def _export_path(self, path: str | Path | None, caption: str, suffix: str, name_filter: str) -> Path | None:
        """The given path, or one chosen in a save dialog (None if cancelled)."""
        if path is not None:
            return Path(path)
        stem = Path(self._analysis_tab.label).stem if self._analysis_tab.label else "signal"
        chosen, _ = QFileDialog.getSaveFileName(
            self, caption, str(self._ctx.settings.save_directory / f"{stem}{suffix}"), name_filter
        )
        return Path(chosen) if chosen else None

    def export_signal_wav(self, path: str | Path | None = None) -> None:
        """File > Export > Signal as WAV (bit depth from Settings; 32-bit float if the signal exceeds full scale)."""
        analysis = self._ctx.analysis_vm
        if analysis.current_data.size == 0:
            self._show_info("There is no signal to export.")
            return
        target = self._export_path(path, "Export signal as WAV", ".wav", "WAV (*.wav)")
        if target is None:
            return
        peak = float(np.max(np.abs(analysis.current_data)))
        subtype = self._ctx.settings.wav_bit_depth if peak <= 1.0001 else "FLOAT_32"
        self._export(
            ExportWorker.wav(self._ctx.export_service, analysis.current_data, analysis.current_sample_rate, target, subtype)
        )

    def export_signal_csv(self, path: str | Path | None = None) -> None:
        """File > Export > Signal as CSV (index, time_s, value)."""
        analysis = self._ctx.analysis_vm
        if analysis.current_data.size == 0:
            self._show_info("There is no signal to export.")
            return
        target = self._export_path(path, "Export signal as CSV", ".csv", "CSV (*.csv)")
        if target is None:
            return
        self._export(
            ExportWorker.csv(self._ctx.export_service, analysis.current_data, analysis.current_sample_rate, target)
        )

    def export_plot_png(self, path: str | Path | None = None) -> None:
        """File > Export > Current plot as PNG."""
        if self._analysis_tab.export_plot_png(path):
            self._show_info("Plot exported.")
        elif path is not None:
            self._show_error("The plot could not be exported.")

    def export_segment_wav(self, path: str | Path | None = None) -> None:
        """File > Export > Selected segment as WAV."""
        self._analysis_tab.export_segment_wav(path)

    def export_impulses_csv(self, path: str | Path | None = None) -> None:
        """File > Export > Detected impulses as CSV."""
        self._analysis_tab.export_impulses_csv(path)

    def open_settings(self) -> None:
        """Edit > Settings: show the settings dialog and apply what was changed."""
        dialog = self._settings_dialog_factory(self._ctx.settings, self)
        if dialog.exec():
            self._record_tab.reload_settings()
            self._show_info("Settings saved.")

    def about_text(self) -> str:
        """The text of the About box."""
        return (
            f"<h3>{APP_NAME} {__version__}</h3>"
            "<p>Ultrasound signal analysis workstation: record from a microphone or a COM-port "
            "sensor, analyse with FFT, STFT and Morlet wavelet transforms, detect impulses and "
            "keep a library of recordings.</p>"
            "<p>Built with PySide6, pyqtgraph, NumPy and SciPy.</p>"
        )

    def show_about(self) -> None:
        """Help > About."""
        QMessageBox.about(self, f"About {APP_NAME}", self.about_text())

    # ================================================================== closing
    def _confirm_close(self) -> bool:
        """Settle a running or unsaved recording; False if quitting should be cancelled."""
        ctx = self._ctx
        rec = ctx.recording_vm
        if rec.is_recording:
            choice = self._close_prompt(self, "recording")
            if choice is None:
                return False
            if choice == "discard":
                rec.on_abort_recording()
                return True
            rec.on_stop()
            return not (rec.is_recording or rec.has_unsaved_recording)  # a failed save keeps the window open
        if rec.has_unsaved_recording:
            choice = self._close_prompt(self, "unsaved")
            if choice is None:
                return False
            if choice == "discard":
                rec.on_discard_unsaved()
                return True
            rec.on_stop()
            return not rec.has_unsaved_recording
        return True

    def _shutdown(self) -> None:
        """Stop every background activity so no thread outlives the window."""
        self._record_tab.shutdown()
        self._analysis_tab.shutdown()
        self._ctx.analysis_service.shutdown()
        self._ctx.library_vm.shutdown()
        for worker in list(self._workers):
            worker.wait(_WORKER_WAIT_MS)

    def _save_geometry(self) -> None:
        try:
            self._ctx.settings.window_geometry = bytes(self.saveGeometry().toHex()).decode("ascii")
        except Exception:  # remembering the window position is never worth failing to quit
            pass

    def _restore_geometry(self) -> None:
        stored = self._ctx.settings.window_geometry
        if stored:
            self.restoreGeometry(QByteArray.fromHex(stored.encode("ascii")))
            self.resize(max(self.width(), MINIMUM_SIZE[0]), max(self.height(), MINIMUM_SIZE[1]))

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._closing:
            event.accept()
            return
        if not self._confirm_close():
            event.ignore()
            return
        self._closing = True
        self._save_geometry()
        self._shutdown()
        event.accept()


# ========================================================================= main
def main(argv: Sequence[str] | None = None) -> int:
    """Create the application and its window and run the event loop.

    Returns:
        The exit code of the Qt event loop.
    """
    if QApplication.instance() is None:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    app = QApplication.instance() or QApplication(list(argv) if argv is not None else sys.argv)
    app.setOrganizationName(ORGANIZATION_NAME)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setWindowIcon(load_icon())

    window = MainWindow()
    window.show()

    # Python only sees Ctrl+C while the interpreter runs: wake it up regularly and
    # close the window (which stops capture and background threads) when it arrives.
    signal.signal(signal.SIGINT, lambda *_: window.close())
    keep_alive = QTimer()
    keep_alive.start(_SIGINT_POLL_MS)
    keep_alive.timeout.connect(lambda: None)

    return int(app.exec())


if __name__ == "__main__":
    sys.exit(main())
