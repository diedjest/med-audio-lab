"""The "Library" tab: browse, search, open, report on, and delete saved recordings."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

from PySide6.QtCore import QUrl, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.models.recording_session import RecordingSession
from app.services.project_manager import slugify
from app.services.settings_service import SettingsService
from app.ui.viewmodels.library_vm import LibraryViewModel
from app.ui.widgets.elided_label import ElidedLabel

#: What the user chose in the delete confirmation: keep the file, delete it too, or cancel.
DeleteChoice = Literal["library", "file"] | None

COLUMNS: tuple[str, ...] = ("ID", "Дата", "Название", "Пациент", "Длительность", "Частота дискретизации", "Путь к файлу")
_COLUMN_WIDTHS = (50, 140, 190, 150, 90, 100)
_NUMERIC_COLUMNS = (0, 4, 5)
_COLOR_MISSING = QBrush(QColor("#b91c1c"))

_STYLE = """
QPushButton#danger { background: #dc2626; color: white; border: none; border-radius: 4px; padding: 5px 12px; }
QPushButton#danger:hover { background: #b91c1c; }
QPushButton#danger:disabled { background: #cbd5e1; color: #64748b; }
QPushButton#primary { background: #2563eb; color: white; border: none; border-radius: 4px; padding: 5px 12px; }
QPushButton#primary:hover { background: #1d4ed8; }
QPushButton#primary:disabled { background: #cbd5e1; color: #64748b; }
QLabel#muted { color: #64748b; }
QLabel#status[error="true"] { color: #b91c1c; }
"""


def format_duration(seconds: float) -> str:
    """``12.5 -> "12.50 с"``, ``75.5 -> "1:15.50"``."""
    if seconds < 60.0:
        return f"{seconds:.2f} с"
    minutes, rest = divmod(seconds, 60.0)
    return f"{int(minutes)}:{rest:05.2f}"


class _RecordingItem(QTreeWidgetItem):
    """A table row holding its :class:`RecordingSession`; sorts by value, not by text."""

    def __init__(self, session: RecordingSession, file_exists: bool) -> None:
        super().__init__(
            [
                "" if session.id is None else str(session.id),
                session.created_at,
                session.title,
                session.patient,
                format_duration(session.duration_s),
                f"{session.sample_rate} Гц",
                session.file_path,
            ]
        )
        self.session = session
        self._keys: tuple[object, ...] = (
            session.id or 0,
            session.created_at,
            session.title.lower(),
            session.patient.lower(),
            session.duration_s,
            session.sample_rate,
            session.file_path.lower(),
        )
        for column in _NUMERIC_COLUMNS:
            self.setTextAlignment(column, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if not file_exists:
            self.setForeground(6, _COLOR_MISSING)
            self.setToolTip(6, "Аудиофайл больше не существует")

    def __lt__(self, other: QTreeWidgetItem) -> bool:
        tree = self.treeWidget()
        column = tree.sortColumn() if tree is not None else 0
        if isinstance(other, _RecordingItem):
            return self._keys[column] < other._keys[column]  # type: ignore[operator]
        return super().__lt__(other)

    def matches(self, tokens: list[str]) -> bool:
        """True if every search token occurs in some field of the recording."""
        s = self.session
        haystack = " ".join(
            (
                str(s.id or ""),
                s.created_at,
                s.title,
                s.patient,
                s.notes,
                s.file_path,
                s.source,
                str(s.sample_rate),
                f"{s.duration_s:.3f}",
            )
        ).lower()
        return all(token in haystack for token in tokens)


def _ask_delete_choice(parent: QWidget, session: RecordingSession) -> DeleteChoice:
    """Default confirmation: three explicit choices, Cancel is the default."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Удаление записи")
    box.setText(f"Удалить «{session.title or 'Без названия'}» из библиотеки?")
    box.setInformativeText(
        f"Файл: {session.file_path}\n\n"
        "«Удалить из библиотеки» оставляет аудиофайл на диске.\n"
        "«Удалить вместе с файлом» также безвозвратно удаляет файл."
    )
    keep = box.addButton("Удалить из библиотеки", QMessageBox.ButtonRole.AcceptRole)
    wipe = box.addButton("Удалить вместе с файлом", QMessageBox.ButtonRole.DestructiveRole)
    cancel = box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(cancel)
    box.setEscapeButton(cancel)
    box.exec()
    clicked = box.clickedButton()
    if clicked is keep:
        return "library"
    if clicked is wipe:
        return "file"
    return None


def _open_in_file_manager(path: Path) -> bool:
    return bool(QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))))


class LibraryTab(QWidget):
    """Recording library bound to a :class:`LibraryViewModel`.

    Args:
        view_model: Source of the recordings and target of load / delete / report commands.
        settings: Provides the default folders for dialogs.
        delete_prompt: ``(parent, session) -> "library" | "file" | None``; the default
            shows a confirmation dialog.  Injectable so the flow can be driven in tests.
        folder_opener: ``(path) -> bool`` that shows a folder in the file manager.

    Signals:
        ``recording_loaded(session, samples)`` - a recording was opened (double-click
        or "Загрузить выбранную"); the owner passes it on to the Analysis tab.
        ``error_occurred(message)`` - anything that went wrong, also shown in the tab.
    """

    recording_loaded = Signal(object, object)
    error_occurred = Signal(str)

    def __init__(
        self,
        view_model: LibraryViewModel,
        settings: SettingsService,
        parent: QWidget | None = None,
        *,
        delete_prompt: Callable[[QWidget, RecordingSession], DeleteChoice] | None = None,
        folder_opener: Callable[[Path], bool] | None = None,
    ) -> None:
        super().__init__(parent)
        self._vm = view_model
        self._settings = settings
        self._delete_prompt = delete_prompt or _ask_delete_choice
        self._open_folder_fn = folder_opener or _open_in_file_manager
        self._shown_once = False

        self.setStyleSheet(_STYLE)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        search_row = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("Поиск по названию, пациенту, дате, заметкам, файлу...")
        self._search.setClearButtonEnabled(True)
        self._count_label = QLabel()
        self._count_label.setObjectName("muted")
        self._refresh_button = QPushButton("Обновить")
        search_row.addWidget(self._search, 1)
        search_row.addWidget(self._count_label)
        search_row.addWidget(self._refresh_button)
        root.addLayout(search_row)

        buttons = QHBoxLayout()
        self._load_button = QPushButton("Загрузить выбранную")
        self._load_button.setObjectName("primary")
        self._report_button = QPushButton("Экспорт отчёта в CSV")
        self._folder_button = QPushButton("Открыть папку")
        self._delete_button = QPushButton("Удалить")
        self._delete_button.setObjectName("danger")
        for button in (self._load_button, self._report_button, self._folder_button, self._delete_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        root.addLayout(buttons)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(len(COLUMNS))
        self._tree.setHeaderLabels(list(COLUMNS))
        self._tree.setRootIsDecorated(False)
        self._tree.setUniformRowHeights(True)
        self._tree.setAlternatingRowColors(True)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._tree.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        for column, width in enumerate(_COLUMN_WIDTHS):
            self._tree.setColumnWidth(column, width)
        self._tree.header().setStretchLastSection(True)
        self._tree.setSortingEnabled(True)
        self._tree.sortByColumn(1, Qt.SortOrder.DescendingOrder)
        root.addWidget(self._tree, 1)

        self._status = ElidedLabel()
        self._status.setObjectName("status")
        root.addWidget(self._status)

        self._connect()
        self._populate()

    # ================================================================== wiring
    def _connect(self) -> None:
        vm = self._vm
        self._search.textChanged.connect(lambda _text: self._apply_filter())
        self._refresh_button.clicked.connect(lambda _c=False: self.refresh())
        self._load_button.clicked.connect(lambda _c=False: self.load_selected())
        self._report_button.clicked.connect(lambda _c=False: self.export_report())
        self._folder_button.clicked.connect(lambda _c=False: self.open_folder())
        self._delete_button.clicked.connect(lambda _c=False: self.delete_selected())
        self._tree.itemSelectionChanged.connect(self._update_controls)
        self._tree.itemDoubleClicked.connect(lambda _item, _column: self.load_selected())

        vm.recordings_changed.connect(self._populate)
        vm.is_loading_changed.connect(self._on_loading_changed)
        vm.recording_loaded.connect(self._on_recording_loaded)
        vm.recording_deleted.connect(lambda rid: self._show_status(f"Запись {rid} удалена."))
        vm.report_exported.connect(lambda path: self._show_status(f"Отчёт сохранён: {path}"))
        vm.error_occurred.connect(self._report)

    # ================================================================== helpers
    def _show_status(self, text: str, *, error: bool = False) -> None:
        self._status.setText(text)
        self._status.setProperty("error", error)
        self._status.style().unpolish(self._status)
        self._status.style().polish(self._status)

    def _report(self, message: str) -> None:
        self._show_status(message, error=True)
        self.error_occurred.emit(message)

    def _items(self) -> list[_RecordingItem]:
        return [
            item
            for i in range(self._tree.topLevelItemCount())
            if isinstance(item := self._tree.topLevelItem(i), _RecordingItem)
        ]

    def selected_session(self) -> RecordingSession | None:
        """The recording whose row is selected, if any."""
        selected = self._tree.selectedItems()
        if selected and isinstance(selected[0], _RecordingItem):
            return selected[0].session
        return None

    @property
    def row_count(self) -> int:
        """Number of recordings in the table (whether or not the filter hides them)."""
        return self._tree.topLevelItemCount()

    @property
    def visible_row_count(self) -> int:
        """Number of rows that match the search text."""
        return sum(1 for item in self._items() if not item.isHidden())

    def select_recording(self, recording_id: int) -> bool:
        """Select the row of a recording; False if it is not in the table."""
        for item in self._items():
            if item.session.id == recording_id:
                self._tree.setCurrentItem(item)
                return True
        return False

    # ================================================================== table
    def refresh(self) -> None:
        """Reload the recordings from the database."""
        self._vm.refresh()

    def _populate(self) -> None:
        previous = self.selected_session()
        previous_id = previous.id if previous is not None else None
        self._tree.setSortingEnabled(False)
        self._tree.clear()
        for session in self._vm.recordings:
            self._tree.addTopLevelItem(_RecordingItem(session, Path(session.file_path).is_file()))
        self._tree.setSortingEnabled(True)
        self._apply_filter()
        if previous_id is not None:
            self.select_recording(previous_id)
        self._update_controls()

    def _apply_filter(self) -> None:
        tokens = self._search.text().lower().split()
        shown = 0
        for item in self._items():
            hidden = not item.matches(tokens)
            item.setHidden(hidden)
            shown += 0 if hidden else 1
        total = self._tree.topLevelItemCount()
        self._count_label.setText(f"{shown} из {total}" if tokens else f"Записей: {total}")
        selected = self._tree.selectedItems()
        if selected and selected[0].isHidden():
            self._tree.clearSelection()
        self._update_controls()

    def _update_controls(self) -> None:
        has_selection = self.selected_session() is not None
        loading = self._vm.is_loading
        self._load_button.setEnabled(has_selection and not loading)
        self._report_button.setEnabled(has_selection)
        self._delete_button.setEnabled(has_selection)
        self._folder_button.setEnabled(True)

    # ================================================================== actions
    def load_selected(self) -> None:
        """Open the selected recording (read in the background)."""
        session = self.selected_session()
        if session is None or session.id is None:
            return
        self._show_status(f"Загрузка «{session.title}»...")
        self._vm.load_recording(session.id)

    def _on_loading_changed(self, loading: bool) -> None:
        self._update_controls()

    def _on_recording_loaded(self, session: RecordingSession, samples: object) -> None:
        self._show_status(f"Загружено: «{session.title}» ({session.duration_s:.2f} с).")
        self.recording_loaded.emit(session, samples)

    def export_report(self, path: str | Path | None = None) -> None:
        """Write a CSV report of the selected recording (a save dialog is shown if ``path`` is None)."""
        session = self.selected_session()
        if session is None or session.id is None:
            return
        if path is None:
            default = self._settings.app_data_dir / "exports" / f"{slugify(session.title)}_report.csv"
            chosen, _ = QFileDialog.getSaveFileName(
                self, "Сохранить отчёт", str(default), "CSV (*.csv);;Все файлы (*)"
            )
            if not chosen:
                return
            path = chosen
        self._show_status(f"Формирование отчёта для «{session.title}»...")
        self._vm.export_report(session.id, str(path))

    def open_folder(self) -> None:
        """Show the selected recording's folder (the save directory if nothing is selected)."""
        session = self.selected_session()
        folder = Path(session.file_path).parent if session is not None else self._settings.save_directory
        if not folder.is_dir():
            if session is None:
                try:
                    folder.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    self._report(f"Не удалось создать {folder}: {exc}")
                    return
            else:
                self._report(f"Папка {folder} больше не существует.")
                return
        if not self._open_folder_fn(folder):
            self._report(f"Не удалось открыть {folder}")

    def delete_selected(self) -> None:
        """Delete the selected recording after asking what to do with its file."""
        session = self.selected_session()
        if session is None or session.id is None:
            return
        choice = self._delete_prompt(self, session)
        if choice == "library":
            self._vm.delete_recording(session.id)
        elif choice == "file":
            self._vm.delete_recording_with_file(session.id)

    # ================================================================== events
    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if not self._shown_once:
            self._shown_once = True
            self.refresh()  # fill the table the first time the tab is shown


__all__: list[str] = ["LibraryTab", "format_duration", "COLUMNS"]
