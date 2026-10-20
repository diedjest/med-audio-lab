"""The "Analysis" tab: one signal, many views of it.

Layout::

    [Load File][TXT->WAV]  Morlet f [..] fmin [..] fmax [..]       [progress][Cancel]
    [Waveform][FFT][PSD][Spectrogram (STFT)][Wavelet (CWT)][Wavelet Slice]
    [Autocorrelation][Envelope][Impulses]
    +----------------------------------------+---------------------+
    | header                    [Back to CWT]| Segment             |
    |                                        | Metrics             |
    | SpectrogramWidget (any analysis)       | Impulse detection   |
    |                                        | Filter              |
    | status                                 | CWT controls        |
    +----------------------------------------+---------------------+

Segment: drag the shaded span on the waveform (or type its bounds in the "Segment"
box) to restrict every analysis to that part of the signal; "Full signal" goes back and
"Save segment" writes it to a WAV file.

Clicking a heat map reads off slices (ported from the legacy ``_on_spectrogram_click``):

* **CWT** - left click: wavelet slice at the clicked frequency (shown in the main
  plot, with a "Back to CWT" button); right click: spectrum slice at the clicked time
  (popup window).
* **STFT** - left click: time slice at the clicked frequency; right click: spectrum
  slice at the clicked time (both in a popup window).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QLocale, Qt, Signal, Slot
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.constants import FilterDesign, FilterType
from app.core.dsp.filters import FilterSpec
from app.core.dsp.metrics import format_metrics
from app.core.models.analysis_result import CwtResult, SignalMetrics
from app.core.models.impulse import DetectedImpulse, ImpulseDetectorConfig
from app.infrastructure.storage import file_io
from app.services.settings_service import SettingsService
from app.ui.viewmodels.analysis_vm import AnalysisViewModel
from app.ui.widgets._plot_base import format_volts
from app.ui.widgets.elided_label import ElidedLabel
from app.ui.widgets.spectrogram_widget import AnalysisPlot, SpectrogramWidget
from app.ui.workers.analysis_worker import AnalysisWorker
from app.ui.workers.export_worker import ExportWorker

SIDEBAR_WIDTH = 400
MAX_IMPULSE_ROWS = 500
_AUDIO_FILE_FILTER = "Аудио и данные (*.wav *.txt *.csv);;WAV (*.wav);;Текст / CSV (*.txt *.csv)"
_TXT_FILE_FILTER = "Текст / CSV (*.txt *.csv);;Все файлы (*)"
_STRONGEST_CHUNK_COLUMNS = 8192
_IMPULSE_FOCUS_PAD_S = 0.010

_STYLE = """
QPushButton#primary { background: #2563eb; color: white; border: none; border-radius: 4px; padding: 5px 12px; }
QPushButton#primary:hover { background: #1d4ed8; }
QPushButton#primary:disabled { background: #cbd5e1; color: #64748b; }
QLabel#muted { color: #64748b; }
QLabel#header { font-weight: 600; }
QLabel#status[error="true"] { color: #b91c1c; }
"""

_ANALYSIS_BUTTONS: tuple[tuple[str, str], ...] = (
    ("Сигнал", "waveform"),
    ("БПФ (FFT)", "fft"),
    ("СПМ (PSD)", "psd"),
    ("Спектрограмма (STFT)", "stft"),
    ("Вейвлет (CWT)", "cwt"),
    ("Срез вейвлета", "slice"),
    ("Автокорреляция", "autocorrelation"),
    ("Огибающая", "envelope"),
    ("Импульсы", "impulses"),
)


def _number_edit(text: str = "", placeholder: str = "", *, integer: bool = False, width: int = 80) -> QLineEdit:
    """Line edit that accepts only positive numbers (dot decimal, independent of the locale)."""
    edit = QLineEdit(text)
    edit.setPlaceholderText(placeholder)
    edit.setMaximumWidth(width)
    if integer:
        edit.setValidator(QIntValidator(1, 1_000_000_000, edit))
    else:
        validator = QDoubleValidator(0.0, 1e9, 6, edit)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        validator.setLocale(QLocale.c())
        edit.setValidator(validator)
    return edit


def _optional_number(edit: QLineEdit, name: str, *, integer: bool = False) -> float | int | None:
    """The positive number in ``edit``; None if it is blank.

    Raises:
        ValueError: not a number, or not positive.
    """
    text = edit.text().strip().replace(",", ".")
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{name}: нужно число, получено {text!r}") from None
    if value <= 0:
        raise ValueError(f"{name}: число должно быть положительным, получено {text}")
    return int(round(value)) if integer else value


def _muted(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setObjectName("muted")
    label.setWordWrap(True)
    return label


def _strongest_frequency(result: CwtResult) -> float:
    """Frequency of the row holding the largest |W| (computed in column chunks)."""
    coefficients = result.coefficients
    row_peak = np.zeros(coefficients.shape[0], dtype=np.float32)
    for start in range(0, coefficients.shape[1], _STRONGEST_CHUNK_COLUMNS):
        block = np.abs(coefficients[:, start : start + _STRONGEST_CHUNK_COLUMNS])
        np.maximum(row_peak, block.max(axis=1), out=row_peak)
    return float(result.frequencies_hz[int(np.argmax(row_peak))])


class _SliceWindow(QWidget):
    """A small separate window showing one slice (the legacy pop-up figures)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Срез")
        self.resize(680, 440)
        self._plot = AnalysisPlot()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._plot)

    @property
    def plot(self) -> AnalysisPlot:
        return self._plot

    def show_slice(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        title: str,
        x_label: str,
        y_label: str,
        log_x: bool = False,
    ) -> None:
        self.setWindowTitle(title)
        self._plot.show_xy(x, y, title=title, x_label=x_label, y_label=y_label, log_x=log_x)
        self.show()
        self.raise_()


class AnalysisTab(QWidget):
    """Analysis screen bound to an :class:`AnalysisViewModel`.

    Args:
        view_model: Source of all analysis state and target of all commands.
        settings: Persisted settings (default folder for file dialogs).

    Signals:
        ``error_occurred(message)`` - any failure shown in the status line.
    """

    error_occurred = Signal(str)

    def __init__(
        self,
        view_model: AnalysisViewModel,
        settings: SettingsService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._vm = view_model
        self._settings = settings
        self._label = ""
        self._last_cwt_payload: dict[str, Any] | None = None
        self._slice_window: _SliceWindow | None = None
        self._active_loader: AnalysisWorker | None = None
        self._workers: set[Any] = set()
        self._loading = False
        self._updating_segment = False
        self._volts_per_fs = 0.0

        self.setStyleSheet(_STYLE)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.addLayout(self._build_toolbar())

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addLayout(self._build_plot_area(), 1)
        body.addWidget(self._build_sidebar())
        root.addLayout(body, 1)

        self._connect_widgets()
        self._connect_view_model()
        self._refresh_all()
        self._display.clear("Загрузите файл или запишите сигнал, затем выберите анализ")

    # ====================================================================== UI
    def _build_toolbar(self) -> QVBoxLayout:
        toolbar = QVBoxLayout()
        toolbar.setSpacing(6)

        files = QHBoxLayout()
        self._load_button = QPushButton("Загрузить файл")
        self._load_button.setObjectName("primary")
        self._txt_button = QPushButton("TXT→WAV")
        files.addWidget(self._load_button)
        files.addWidget(self._txt_button)
        files.addSpacing(16)

        self._target_edit = _number_edit(placeholder="самый сильный гребень")
        self._target_edit.setToolTip("Частота (Гц) среза вейвлета; пусто — самый сильный гребень")
        self._fmin_edit = _number_edit("5000", "по умолчанию")
        self._fmin_edit.setToolTip("Наименьшая частота (Гц) CWT")
        self._fmax_edit = _number_edit(placeholder="Найквист")
        self._fmax_edit.setToolTip("Наибольшая частота (Гц) CWT")
        for caption, edit in (("Морле f (Гц)", self._target_edit), ("f от (Гц)", self._fmin_edit), ("f до (Гц)", self._fmax_edit)):
            label = QLabel(caption)
            label.setObjectName("muted")
            files.addWidget(label)
            files.addWidget(edit)
        files.addStretch(1)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setMaximumWidth(220)
        self._progress.setTextVisible(True)
        self._progress.hide()
        self._cancel_button = QPushButton("Отмена")
        self._cancel_button.hide()
        files.addWidget(self._progress)
        files.addWidget(self._cancel_button)
        toolbar.addLayout(files)

        row = QHBoxLayout()
        self._analysis_buttons: dict[str, QPushButton] = {}
        for text, key in _ANALYSIS_BUTTONS:
            button = QPushButton(text)
            self._analysis_buttons[key] = button
            row.addWidget(button)
        row.addStretch(1)
        toolbar.addLayout(row)
        return toolbar

    def _build_plot_area(self) -> QVBoxLayout:
        area = QVBoxLayout()
        area.setSpacing(4)

        head = QHBoxLayout()
        self._header = ElidedLabel("Сигнал не загружен")
        self._header.setObjectName("header")
        self._back_button = QPushButton("◀ Назад к CWT")
        self._back_button.hide()
        head.addWidget(self._header, 1)
        head.addWidget(self._back_button)
        area.addLayout(head)

        self._display = SpectrogramWidget()
        area.addWidget(self._display, 1)

        self._status = ElidedLabel()
        self._status.setObjectName("status")
        area.addWidget(self._status)
        return area

    def _build_sidebar(self) -> QScrollArea:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(10)
        layout.addWidget(self._build_segment_group())
        layout.addWidget(self._build_metrics_group())
        layout.addWidget(self._build_impulse_group())
        layout.addWidget(self._build_filter_group())
        layout.addWidget(self._build_cwt_group())
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(SIDEBAR_WIDTH)
        self._sidebar = scroll
        return scroll

    def _build_segment_group(self) -> QGroupBox:
        group = QGroupBox("Фрагмент")
        box = QVBoxLayout(group)

        def time_spin() -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setSingleStep(0.01)
            spin.setSuffix(" с")
            spin.setKeyboardTracking(False)
            return spin

        self._segment_start = time_spin()
        self._segment_end = time_spin()
        grid = QGridLayout()
        grid.addWidget(QLabel("Начало"), 0, 0)
        grid.addWidget(self._segment_start, 0, 1)
        grid.addWidget(QLabel("Конец"), 1, 0)
        grid.addWidget(self._segment_end, 1, 1)
        box.addLayout(grid)

        buttons = QHBoxLayout()
        self._select_button = QPushButton("Выделить")
        self._select_button.setObjectName("primary")
        self._select_button.setToolTip("Выделить фрагмент: перетащите границы на осциллограмме или введите их вручную")
        self._full_button = QPushButton("Весь сигнал")
        self._full_button.setToolTip("Снова анализировать весь сигнал")
        self._save_segment_button = QPushButton("Сохранить фрагмент")
        self._save_segment_button.setToolTip("Сохранить выделенный фрагмент в WAV-файл")
        buttons.addWidget(self._select_button, 1)
        buttons.addWidget(self._full_button, 1)
        buttons.addWidget(self._save_segment_button, 1)
        box.addLayout(buttons)

        self._segment_hint = _muted()
        box.addWidget(self._segment_hint)
        return group

    def _build_metrics_group(self) -> QGroupBox:
        group = QGroupBox("Метрики")
        self._metrics_group = group
        box = QVBoxLayout(group)
        self._metrics_tree = QTreeWidget()
        self._metrics_tree.setColumnCount(2)
        self._metrics_tree.setHeaderLabels(["Параметр", "Значение"])
        self._metrics_tree.setRootIsDecorated(False)
        self._metrics_tree.setAlternatingRowColors(True)
        self._metrics_tree.setMinimumHeight(240)
        self._metrics_tree.setColumnWidth(0, 170)
        box.addWidget(self._metrics_tree)
        return group

    def _build_impulse_group(self) -> QGroupBox:
        group = QGroupBox("Поиск импульсов")
        box = QVBoxLayout(group)
        grid = QGridLayout()
        defaults = ImpulseDetectorConfig()
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(-120.0, 0.0)
        self._threshold_spin.setDecimals(1)
        self._threshold_spin.setSuffix(" дБ")
        self._threshold_spin.setValue(defaults.threshold_db)
        self._min_duration_spin = QDoubleSpinBox()
        self._min_duration_spin.setRange(0.01, 1000.0)
        self._min_duration_spin.setDecimals(2)
        self._min_duration_spin.setSingleStep(0.1)
        self._min_duration_spin.setSuffix(" мс")
        self._min_duration_spin.setValue(defaults.min_duration_ms)
        self._impulse_fmin_spin = QDoubleSpinBox()
        self._impulse_fmin_spin.setRange(0.0, 400_000.0)
        self._impulse_fmin_spin.setDecimals(0)
        self._impulse_fmin_spin.setSingleStep(500.0)
        self._impulse_fmin_spin.setSuffix(" Гц")
        self._impulse_fmin_spin.setValue(defaults.fmin_hz)
        for row, (caption, spin) in enumerate(
            (("Порог", self._threshold_spin), ("Мин. длительность", self._min_duration_spin), ("Частота от", self._impulse_fmin_spin))
        ):
            grid.addWidget(QLabel(caption), row, 0)
            grid.addWidget(spin, row, 1)
        box.addLayout(grid)

        buttons = QHBoxLayout()
        self._detect_button = QPushButton("Найти")
        self._detect_button.setObjectName("primary")
        self._export_impulses_button = QPushButton("Экспорт в CSV")
        buttons.addWidget(self._detect_button, 1)
        buttons.addWidget(self._export_impulses_button, 1)
        box.addLayout(buttons)

        self._impulse_tree = QTreeWidget()
        self._impulse_tree.setColumnCount(5)
        self._impulse_tree.setHeaderLabels(["№", "Начало (с)", "Дл. (мс)", "Част. (кГц)", "Пик"])
        self._impulse_tree.setRootIsDecorated(False)
        self._impulse_tree.setMinimumHeight(170)
        for column, width in enumerate((36, 76, 62, 70, 56)):
            self._impulse_tree.setColumnWidth(column, width)
        box.addWidget(self._impulse_tree)
        self._impulse_hint = _muted("Двойной щелчок по импульсу приближает его.")
        box.addWidget(self._impulse_hint)
        return group

    def _build_filter_group(self) -> QGroupBox:
        group = QGroupBox("Фильтр")
        form = QFormLayout(group)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self._filter_form = form

        self._filter_type = QComboBox()
        for text, kind in (
            ("ФНЧ (низких частот)", FilterType.LOWPASS),
            ("ФВЧ (высоких частот)", FilterType.HIGHPASS),
            ("Полосовой", FilterType.BANDPASS),
            ("Режекторный", FilterType.BANDSTOP),
        ):
            self._filter_type.addItem(text, kind)

        def frequency_spin(value: float) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setRange(1.0, 400_000.0)
            spin.setDecimals(1)
            spin.setSingleStep(500.0)
            spin.setSuffix(" Гц")
            spin.setValue(value)
            return spin

        self._filter_low = frequency_spin(10_000.0)
        self._filter_high = frequency_spin(50_000.0)
        self._filter_order = QSpinBox()
        self._filter_order.setRange(1, 12)
        self._filter_order.setValue(4)
        self._filter_design = QComboBox()
        for text, design in (
            ("Баттерворт", FilterDesign.BUTTERWORTH),
            ("Чебышев I", FilterDesign.CHEBYSHEV_I),
            ("Чебышев II", FilterDesign.CHEBYSHEV_II),
        ):
            self._filter_design.addItem(text, design)
        self._filter_ripple = QDoubleSpinBox()
        self._filter_ripple.setRange(0.1, 100.0)
        self._filter_ripple.setDecimals(1)
        self._filter_ripple.setSuffix(" дБ")
        self._filter_ripple.setValue(0.5)
        self._filter_ripple.setToolTip("Пульсации в полосе пропускания (Чебышев I) или затухание в полосе подавления (Чебышев II)")
        self._filter_ripple.setEnabled(False)  # only the Chebyshev designs use it

        form.addRow("Тип", self._filter_type)
        form.addRow("Частота среза", self._filter_low)
        form.addRow("Верхняя частота среза", self._filter_high)
        form.addRow("Порядок", self._filter_order)
        form.addRow("Аппроксимация", self._filter_design)
        form.addRow("Пульсации / затухание", self._filter_ripple)
        buttons = QHBoxLayout()
        self._apply_filter_button = QPushButton("Применить")
        self._apply_filter_button.setObjectName("primary")
        self._restore_button = QPushButton("Вернуть оригинал")
        buttons.addWidget(self._apply_filter_button, 1)
        buttons.addWidget(self._restore_button, 1)
        form.addRow(buttons)
        self._update_filter_inputs()
        return group

    def _build_cwt_group(self) -> QGroupBox:
        group = QGroupBox("Параметры CWT")
        form = QFormLayout(group)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self._target_fs_edit = _number_edit(placeholder="нет (оставить частоту дискретизации)", integer=True, width=400)
        self._target_fs_edit.setToolTip("Передискретизировать до этой частоты (Гц) перед CWT")
        self._max_seconds_edit = _number_edit(placeholder="авто (по лимиту памяти)", width=400)
        self._max_seconds_edit.setToolTip("Использовать только первые N секунд сигнала")
        self._stride_spin = QSpinBox()
        self._stride_spin.setRange(1, 64)
        self._stride_spin.setToolTip("Брать каждый N-й масштаб: меньше строк, меньше памяти")
        form.addRow("Целевая частота дискретизации (Гц)", self._target_fs_edit)
        form.addRow("Макс. длительность (с)", self._max_seconds_edit)
        form.addRow("Шаг по масштабам", self._stride_spin)
        form.addRow(_muted("Диапазон частот задаётся полями «f от» / «f до» на панели сверху."))
        return group

    # ================================================================= wiring
    def _connect_widgets(self) -> None:
        vm = self._vm
        buttons = self._analysis_buttons
        buttons["waveform"].clicked.connect(lambda _c=False: vm.show_waveform())
        buttons["fft"].clicked.connect(lambda _c=False: vm.show_fft())
        buttons["psd"].clicked.connect(lambda _c=False: vm.show_psd())
        buttons["stft"].clicked.connect(lambda _c=False: vm.show_stft())
        buttons["cwt"].clicked.connect(lambda _c=False: self.run_cwt())
        buttons["slice"].clicked.connect(lambda _c=False: self.run_wavelet_slice())
        buttons["autocorrelation"].clicked.connect(lambda _c=False: vm.show_autocorrelation())
        buttons["envelope"].clicked.connect(lambda _c=False: vm.show_envelope())
        buttons["impulses"].clicked.connect(lambda _c=False: self.detect_impulses())

        self._load_button.clicked.connect(lambda _c=False: self.load_file())
        self._txt_button.clicked.connect(lambda _c=False: self.convert_txt_to_wav())
        self._cancel_button.clicked.connect(lambda _c=False: vm.cancel_analysis())
        self._back_button.clicked.connect(lambda _c=False: self._back_to_cwt())

        self._detect_button.clicked.connect(lambda _c=False: self.detect_impulses())
        self._export_impulses_button.clicked.connect(lambda _c=False: self.export_impulses_csv())
        self._impulse_tree.itemDoubleClicked.connect(self._on_impulse_activated)

        self._filter_type.currentIndexChanged.connect(lambda _i: self._update_filter_inputs())
        self._filter_design.currentIndexChanged.connect(lambda _i: self._on_design_changed())
        self._apply_filter_button.clicked.connect(lambda _c=False: self.apply_filter())
        self._restore_button.clicked.connect(lambda _c=False: vm.restore_original())

        self._display.clicked.connect(self._on_display_clicked)

        self._select_button.clicked.connect(lambda _c=False: self.select_segment())
        self._full_button.clicked.connect(lambda _c=False: vm.clear_selection())
        self._save_segment_button.clicked.connect(lambda _c=False: self.export_segment_wav())
        self._segment_start.editingFinished.connect(self._on_segment_edited)
        self._segment_end.editingFinished.connect(self._on_segment_edited)
        self._display.selection_edited.connect(self._on_selection_dragged)

    def _connect_view_model(self) -> None:
        vm = self._vm
        vm.current_data_changed.connect(self._refresh_all)
        vm.current_sample_rate_changed.connect(lambda _r: self._refresh_all())
        vm.current_metrics_changed.connect(self._refresh_metrics)
        vm.current_impulses_changed.connect(self._refresh_impulses)
        vm.cwt_result_changed.connect(self._update_controls)
        vm.selection_changed.connect(self._on_selection_changed)
        vm.is_busy_changed.connect(self._on_busy_changed)
        vm.plot_ready.connect(self._on_plot_ready)
        vm.analysis_progress.connect(self._on_progress)
        vm.analysis_error.connect(self._on_analysis_error)
        vm.notice.connect(lambda text: self._show_status(text))

    # ============================================================== helpers
    def _show_status(self, text: str, *, error: bool = False) -> None:
        self._status.setText(text)
        self._status.setProperty("error", error)
        self._status.style().unpolish(self._status)
        self._status.style().polish(self._status)

    def _report(self, message: str) -> None:
        self._show_status(message, error=True)
        self.error_occurred.emit(message)

    def _refresh_all(self) -> None:
        self._refresh_header()
        self._refresh_segment()
        self._refresh_metrics()
        self._refresh_impulses()
        self._update_controls()

    def _refresh_header(self) -> None:
        vm = self._vm
        if vm.current_data.size == 0:
            self._header.setText("Сигнал не загружен")
            return
        duration = vm.current_data.size / max(1, vm.current_sample_rate)
        name = f"{self._label} - " if self._label else ""
        filtered = " (отфильтрован)" if vm.has_original else ""
        segment = ""
        if vm.selection is not None:
            segment = f" - фрагмент {vm.selection[0]:.3f}-{vm.selection[1]:.3f} с"
        self._header.setText(f"{name}{duration:.3f} с, {vm.current_sample_rate} Гц{filtered}{segment}")

    def _update_controls(self, *_: object) -> None:
        vm = self._vm
        has_data = vm.current_data.size > 0
        for key, button in self._analysis_buttons.items():
            button.setEnabled(has_data and (key != "slice" or vm.cwt_result is not None))
        self._analysis_buttons["slice"].setToolTip(
            "" if vm.cwt_result is not None else "Сначала выполните CWT"
        )
        self._detect_button.setEnabled(has_data)
        self._export_impulses_button.setEnabled(bool(vm.current_impulses))
        self._apply_filter_button.setEnabled(has_data)
        for widget in (self._select_button, self._segment_start, self._segment_end):
            widget.setEnabled(has_data)
        self._full_button.setEnabled(vm.selection is not None)
        self._save_segment_button.setEnabled(vm.selection is not None)
        self._restore_button.setEnabled(vm.has_original)
        self._load_button.setEnabled(not self._loading)
        self._txt_button.setEnabled(not self._loading)

    def _on_busy_changed(self, busy: bool) -> None:
        self._progress.setVisible(busy)
        self._cancel_button.setVisible(busy)
        if busy:
            self._progress.setValue(0)
            self._progress.setFormat("Выполняется...")

    def _on_progress(self, fraction: float, message: str) -> None:
        self._progress.setValue(int(round(min(1.0, max(0.0, fraction)) * 1000)))
        self._progress.setFormat(f"{fraction * 100:.0f}%")
        self._progress.setToolTip(message)

    def _on_analysis_error(self, kind: str, message: str) -> None:
        self._report(f"{kind}: {message}")

    # ================================================================= segment
    def _refresh_segment(self) -> None:
        """Sync the bounds boxes, the hint and the metrics title with the selection."""
        vm = self._vm
        duration = vm.current_data.size / max(1, vm.current_sample_rate)
        start, end = vm.selection if vm.selection is not None else (0.0, duration)
        self._updating_segment = True
        try:
            for spin in (self._segment_start, self._segment_end):
                spin.setRange(0.0, duration)
            self._segment_start.setValue(start)
            self._segment_end.setValue(end)
        finally:
            self._updating_segment = False
        if vm.selection is None:
            self._segment_hint.setText("Анализируется весь сигнал.")
            self._metrics_group.setTitle("Метрики")
        else:
            self._segment_hint.setText(
                f"Анализируется фрагмент {end - start:.6f} с. Время на графиках анализа "
                "отсчитывается от начала фрагмента."
            )
            self._metrics_group.setTitle("Метрики (фрагмент)")

    def select_segment(self, start_s: float | None = None, end_s: float | None = None) -> None:
        """Restrict the analysis to ``[start_s, end_s]`` and show it on the waveform.

        Without bounds the middle half of the visible waveform is selected (the whole
        signal if another plot is shown); the user then adjusts it by dragging its edges.
        """
        vm = self._vm
        if vm.current_data.size == 0:
            self._report("Сначала загрузите или запишите сигнал.")
            return
        if self._display.kind != "waveform":
            vm.show_waveform()
        if start_s is None or end_s is None:
            first, last = self._display.waveform_widget.view_range()
            duration = vm.current_data.size / max(1, vm.current_sample_rate)
            first, last = max(0.0, first), min(duration, last)
            start_s, end_s = first + (last - first) / 4.0, last - (last - first) / 4.0
        self._set_selection(start_s, end_s)

    def _set_selection(self, start_s: float, end_s: float) -> None:
        try:
            self._vm.set_selection(start_s, end_s)
        except ValueError as exc:
            self._report(str(exc))
            self._refresh_segment()  # put the previous bounds back
            if self._vm.selection is not None and self._display.kind == "waveform":
                self._display.set_selection(*self._vm.selection)

    def _on_segment_edited(self) -> None:
        if not self._updating_segment:
            self._set_selection(self._segment_start.value(), self._segment_end.value())

    def _on_selection_dragged(self, start_s: float, end_s: float) -> None:
        self._set_selection(start_s, end_s)

    def _on_selection_changed(self) -> None:
        selection = self._vm.selection
        self._last_cwt_payload = None  # belongs to the previous input
        self._back_button.hide()
        if self._display.kind == "waveform":
            if selection is None:
                self._display.clear_selection()
            else:
                self._display.set_selection(*selection)
            self._show_status("Весь сигнал" if selection is None else "Фрагмент выделен")
        elif self._display.kind != "empty":
            self._show_status("Выделение изменилось — запустите анализ заново.")
        self._refresh_header()
        self._refresh_segment()
        self._update_controls()

    def export_segment_wav(self, path: str | Path | None = None) -> None:
        """Save the selected segment as a WAV file (a save dialog is shown if ``path`` is None).

        The bit depth comes from the settings; a segment above full scale is written as
        32-bit float so that nothing is clipped.
        """
        vm = self._vm
        selection = vm.selection
        if selection is None:
            self._report("Нет выделенного фрагмента для сохранения. Сначала нажмите «Выделить».")
            return
        samples = vm.selected_segment()
        rate = vm.current_sample_rate
        if path is None:
            stem = Path(self._label).stem if self._label else "signal"
            suggested = f"{stem}_{selection[0]:.3f}-{selection[1]:.3f}s.wav"
            chosen, _ = QFileDialog.getSaveFileName(
                self, "Сохранить фрагмент в WAV", str(self._settings.save_directory / suggested), "WAV (*.wav)"
            )
            if not chosen:
                return
            path = chosen
        target = Path(path).with_suffix(".wav")
        peak = float(np.max(np.abs(samples)))
        subtype = self._settings.wav_bit_depth if peak <= 1.0001 else "FLOAT_32"

        def job() -> Path:
            file_io.save_wav(samples, rate, target, subtype)
            return target

        self._run_export(job, "Фрагмент сохранён: {}")

    # ================================================================= metrics
    def set_calibration(self, volts_per_fs: float) -> None:
        """Show amplitudes in volts (``volts_per_fs`` > 0) or in full-scale units (0)."""
        self._volts_per_fs = max(0.0, float(volts_per_fs))
        self._display.set_calibration(self._volts_per_fs)
        self._refresh_metrics()

    def _refresh_metrics(self) -> None:
        self._metrics_tree.clear()
        metrics: SignalMetrics | None = self._vm.current_metrics
        if metrics is None:
            return
        rows = format_metrics(metrics).items()
        for name, value in rows:
            self._metrics_tree.addTopLevelItem(QTreeWidgetItem([name, value]))
        if self._volts_per_fs > 0.0:
            for name, level in (("Пик (В)", metrics.peak), ("СКЗ (В)", metrics.rms)):
                self._metrics_tree.addTopLevelItem(
                    QTreeWidgetItem([name, format_volts(level * self._volts_per_fs)])
                )

    # ================================================================ impulses
    def detect_impulses(self) -> None:
        """Run impulse detection with the parameters in the sidebar."""
        config = ImpulseDetectorConfig(
            threshold_db=self._threshold_spin.value(),
            min_duration_ms=self._min_duration_spin.value(),
            fmin_hz=self._impulse_fmin_spin.value(),
        )
        self._vm.set_options("impulses", config=config)
        self._vm.show_impulses()

    def _refresh_impulses(self) -> None:
        self._impulse_tree.clear()
        impulses: list[DetectedImpulse] = self._vm.current_impulses
        for impulse in impulses[:MAX_IMPULSE_ROWS]:
            freq = "-" if not np.isfinite(impulse.dominant_freq_hz) else f"{impulse.dominant_freq_hz / 1000.0:.2f}"
            item = QTreeWidgetItem(
                [
                    str(impulse.index),
                    f"{impulse.start_s:.5f}",
                    f"{impulse.duration_ms:.2f}",
                    freq,
                    f"{impulse.peak_amplitude:.3f}",
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, impulse)
            for column in range(1, 5):
                item.setTextAlignment(column, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._impulse_tree.addTopLevelItem(item)
        if len(impulses) > MAX_IMPULSE_ROWS:
            self._impulse_hint.setText(f"Показаны первые {MAX_IMPULSE_ROWS} из {len(impulses)} импульсов.")
        else:
            self._impulse_hint.setText("Двойной щелчок по импульсу приближает его.")
        self._update_controls()

    def _on_impulse_activated(self, item: QTreeWidgetItem, _column: int) -> None:
        impulse = item.data(0, Qt.ItemDataRole.UserRole)
        if impulse is not None:
            self.focus_impulse(impulse)

    def focus_impulse(self, impulse: DetectedImpulse) -> None:
        """Zoom the waveform to one impulse (showing the impulse overlay first if needed)."""
        vm = self._vm
        if self._display.kind != "impulses":
            self._display.show_plot(
                "impulses",
                {"impulses": vm.current_impulses, "data": vm.analysis_data, "sample_rate": vm.current_sample_rate},
            )
            self._back_button.hide()
        pad = max(_IMPULSE_FOCUS_PAD_S, 2.0 * (impulse.end_s - impulse.start_s))
        self._display.set_time_range(impulse.start_s - pad, impulse.end_s + pad)

    def export_impulses_csv(self, path: str | Path | None = None) -> None:
        """Save the detected impulses as CSV (a save dialog is shown if ``path`` is None)."""
        impulses = list(self._vm.current_impulses)
        if not impulses:
            self._report("Нет найденных импульсов для экспорта. Сначала нажмите «Найти».")
            return
        if path is None:
            stem = Path(self._label).stem if self._label else "impulses"
            chosen, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить найденные импульсы",
                str(self._settings.save_directory / f"{stem}_impulses.csv"),
                "CSV (*.csv);;Все файлы (*)",
            )
            if not chosen:
                return
            path = chosen
        target = Path(path)

        def job() -> Path:
            file_io.save_impulses_csv(impulses, target)
            return target

        self._run_export(job, "Импульсы сохранены: {}")

    # ================================================================== filter
    def _selected_filter_type(self) -> FilterType:
        # combo item data of a str-based enum comes back as a plain str: restore the enum
        return FilterType(self._filter_type.currentData())

    def _selected_design(self) -> FilterDesign:
        return FilterDesign(self._filter_design.currentData())

    def _update_filter_inputs(self) -> None:
        band = self._selected_filter_type() in (FilterType.BANDPASS, FilterType.BANDSTOP)
        self._filter_form.setRowVisible(self._filter_high, band)
        label = self._filter_form.labelForField(self._filter_low)
        if isinstance(label, QLabel):
            label.setText("Нижняя частота среза" if band else "Частота среза")

    def _on_design_changed(self) -> None:
        design = self._selected_design()
        self._filter_ripple.setEnabled(design is not FilterDesign.BUTTERWORTH)
        if design is FilterDesign.CHEBYSHEV_I:
            self._filter_ripple.setValue(0.5)
        elif design is FilterDesign.CHEBYSHEV_II:
            self._filter_ripple.setValue(40.0)

    def build_filter_spec(self) -> FilterSpec:
        """The filter described by the sidebar controls.

        Raises:
            ValueError: invalid frequencies (not below Nyquist, or upper <= lower).
        """
        nyquist = self._vm.current_sample_rate / 2.0
        low, high = self._filter_low.value(), self._filter_high.value()
        kind = self._selected_filter_type()
        band = kind in (FilterType.BANDPASS, FilterType.BANDSTOP)
        if nyquist > 0 and max(low, high if band else 0.0) >= nyquist:
            raise ValueError(f"Частоты среза должны быть ниже частоты Найквиста ({nyquist:.0f} Гц).")
        if band and high <= low:
            raise ValueError("Верхняя частота среза должна быть выше нижней.")
        return FilterSpec(
            filter_type=kind,
            freq_hz=(low, high) if band else low,
            order=self._filter_order.value(),
            design=self._selected_design(),
            ripple_db=self._filter_ripple.value(),
        )

    def apply_filter(self) -> None:
        """Filter the current signal with the sidebar settings."""
        try:
            spec = self.build_filter_spec()
        except ValueError as exc:
            self._report(str(exc))
            return
        self._vm.apply_filter(spec)

    # ===================================================================== CWT
    def _apply_cwt_options(self) -> bool:
        """Copy the CWT inputs to the view model; False (with a message) if one is invalid."""
        try:
            fmin = _optional_number(self._fmin_edit, "f от")
            fmax = _optional_number(self._fmax_edit, "f до")
            target_fs = _optional_number(self._target_fs_edit, "Целевая частота дискретизации", integer=True)
            max_seconds = _optional_number(self._max_seconds_edit, "Макс. длительность")
        except ValueError as exc:
            self._report(str(exc))
            return False
        if fmin is not None and fmax is not None and fmax <= fmin:
            self._report("«f до» должно быть больше «f от».")
            return False
        stride = self._stride_spin.value()
        self._vm.set_options(
            "cwt",
            fmin_hz=fmin,
            fmax_hz=fmax,
            target_fs=target_fs,
            max_seconds=max_seconds,
            scale_stride=stride if stride > 1 else None,
        )
        return True

    def run_cwt(self) -> None:
        """Compute the CWT with the toolbar / sidebar settings."""
        if self._apply_cwt_options():
            self._vm.show_cwt()

    def run_wavelet_slice(self) -> None:
        """Show the wavelet slice at the 'Morlet f' frequency (the strongest ridge if empty)."""
        result = self._vm.cwt_result
        if result is None:
            self._vm.show_wavelet_slice(0.0)  # the view model reports "Сначала выполните CWT."
            return
        try:
            target = _optional_number(self._target_edit, "Морле f")
        except ValueError as exc:
            self._report(str(exc))
            return
        self._vm.show_wavelet_slice(float(target) if target is not None else _strongest_frequency(result))

    def _back_to_cwt(self) -> None:
        if self._last_cwt_payload is not None:
            self._display.show_plot("cwt", self._last_cwt_payload)
            self._back_button.hide()
            self._show_status("Вейвлет-преобразование Морле (CWT)")

    # ================================================================ plotting
    def _on_plot_ready(self, kind: str, payload: dict[str, Any]) -> None:
        try:
            self._display.show_plot(kind, payload)
        except Exception as exc:
            self._report(f"Не удалось построить график ({kind}): {exc}")
            return
        if kind == "cwt":
            self._last_cwt_payload = payload
        self._back_button.setVisible(kind == "wavelet_slice" and self._last_cwt_payload is not None)
        self._show_status(self._display.title)

    def _on_display_clicked(self, time_s: float, frequency_hz: float, button: int) -> None:
        kind = self._display.kind
        if kind == "cwt":
            if button == 1:
                self._vm.show_wavelet_slice(frequency_hz)
            else:
                self._popup_time_slice(time_s, log_x=True)
        elif kind == "stft":
            if button == 1:
                self._popup_frequency_slice(frequency_hz)
            else:
                self._popup_time_slice(time_s, log_x=False)

    def _slice_window_instance(self) -> _SliceWindow:
        if self._slice_window is None:
            self._slice_window = _SliceWindow(self)
        return self._slice_window

    def _popup_time_slice(self, time_s: float, *, log_x: bool) -> None:
        """Spectrum at one time (the legacy 'lateral' slice)."""
        data = self._display.slice_at_time(time_s)
        if data is None:
            return
        freqs, values, t_actual, y_label = data
        positive = freqs > 0
        if log_x and not positive.any():
            self._report("Нет положительных частот для логарифмической оси.")
            return
        mask = positive if log_x else np.ones_like(positive)
        self._slice_window_instance().show_slice(
            freqs[mask], values[mask],
            title=f"Срез спектра при t = {t_actual:.6f} с ({self._display.title})",
            x_label="Частота (Гц)", y_label=y_label, log_x=log_x,
        )

    def _popup_frequency_slice(self, frequency_hz: float) -> None:
        """Magnitude over time at one frequency (the legacy 'longitudinal' slice)."""
        data = self._display.slice_at_frequency(frequency_hz)
        if data is None:
            return
        time, values, f_actual, y_label = data
        self._slice_window_instance().show_slice(
            time, values,
            title=f"Срез по времени при f = {f_actual:.2f} Гц ({self._display.title})",
            x_label="Время (с)", y_label=y_label,
        )

    # ================================================================== loading
    @property
    def label(self) -> str:
        """Name of the current signal (a file or recording title; "" if unknown)."""
        return self._label

    def export_plot_png(self, path: str | Path | None = None) -> bool:
        """Save the plot that is currently shown as a PNG (file dialog if ``path`` is None)."""
        return self._display.export_png(path)

    def set_signal(self, samples: np.ndarray, sample_rate: int, label: str = "") -> None:
        """Make ``samples`` the current signal and show its waveform.

        Called by the owner when a recording was saved or a file was loaded elsewhere.
        """
        self._label = label
        self._last_cwt_payload = None
        self._back_button.hide()
        self._display.clear("Загрузка...")
        self._vm.load_signal(samples, sample_rate)
        self._refresh_header()
        self._vm.show_waveform()

    def load_file(self, path: str | Path | None = None) -> None:
        """Load a WAV / TXT / CSV file (a file dialog is shown if ``path`` is None)."""
        if path is None:
            chosen, _ = QFileDialog.getOpenFileName(
                self, "Загрузить аудио или файл данных", str(self._settings.save_directory), _AUDIO_FILE_FILTER
            )
            if not chosen:
                return
            path = chosen
        worker = AnalysisWorker(file_io.read_audio_file, (str(path),), task_name=Path(path).name)
        worker.result_ready.connect(self._on_file_loaded)
        worker.error.connect(self._on_file_error)
        worker.finished.connect(self._on_worker_finished)
        self._workers.add(worker)
        self._active_loader = worker
        self._loading = True
        self._update_controls()
        self._show_status(f"Загрузка {Path(path).name}...")
        worker.start()

    @Slot(str, object)
    def _on_file_loaded(self, name: str, result: object) -> None:
        if self.sender() is not self._active_loader:
            return
        self._active_loader = None
        self._loading = False
        rate, samples = result  # type: ignore[misc]
        self.set_signal(samples, rate, name)
        self._update_controls()

    @Slot(str, str)
    def _on_file_error(self, name: str, message: str) -> None:
        if self.sender() is not self._active_loader:
            return
        self._active_loader = None
        self._loading = False
        self._update_controls()
        self._report(f"Не удалось загрузить {name}: {message}")

    @Slot()
    def _on_worker_finished(self) -> None:
        worker = self.sender()
        self._workers.discard(worker)
        if worker is not None:
            worker.deleteLater()

    # =============================================================== TXT -> WAV
    def convert_txt_to_wav(
        self,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
        sample_rate: int | None = None,
    ) -> None:
        """Convert a numeric TXT / CSV table to a WAV file in the background.

        With no arguments the user is asked for the files and (optionally) the sample
        rate; leave the sample rate empty to estimate it from the table's time column.
        """
        if input_path is None:
            chosen, _ = QFileDialog.getOpenFileName(
                self, "Выберите таблицу TXT / CSV", str(self._settings.save_directory), _TXT_FILE_FILTER
            )
            if not chosen:
                return
            input_path = chosen
            if sample_rate is None:
                rate, ok = QInputDialog.getInt(
                    self, "TXT → WAV",
                    "Частота дискретизации (Гц).\nОтмена — оценить её по столбцу времени:",
                    192000, 1, 100_000_000,
                )
                sample_rate = rate if ok else None
        source = Path(input_path)
        if output_path is None:
            chosen, _ = QFileDialog.getSaveFileName(
                self, "Сохранить как WAV", str(source.with_suffix(".wav")), "WAV (*.wav)"
            )
            if not chosen:
                return
            output_path = chosen
        target = Path(output_path)

        def job() -> Path:
            file_io.txt_to_wav(source, target, sample_rate=sample_rate)
            return target

        self._run_export(job, "Преобразовано в WAV: {}")

    # ==================================================================== export
    def _run_export(self, job: Callable[[], Path], success: str) -> None:
        worker = ExportWorker(job)
        worker.export_complete.connect(lambda path: self._show_status(success.format(path)))
        worker.export_error.connect(self._report)
        worker.finished.connect(self._on_worker_finished)
        self._workers.add(worker)
        worker.start()

    # ================================================================ shutdown
    def shutdown(self) -> None:
        """Cancel running analyses and wait for background tasks (call on window close)."""
        self._vm.cancel_analysis()
        for worker in list(self._workers):
            worker.wait(5000)
        if self._slice_window is not None:
            self._slice_window.close()


__all__: list[str] = ["AnalysisTab", "SIDEBAR_WIDTH"]
