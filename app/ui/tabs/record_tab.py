"""The "Record" tab: capture controls on the left, live plots on the right.

Layout (as in the legacy application)::

    +-----------------+--------------------------------------------------+
    | sidebar (350px, | [Start][Pause][Stop][Play][Stop playback][Load]  |
    | scrollable)     |                          status / level meter    |
    |  source         | +----------------------------------------------+ |
    |  microphone     | | live rolling waveform                        | |
    |  COM port       | +----------------------------------------------+ |
    |  save folder    | +----------------------------------------------+ |
    |  record details | | live spectrum                                | |
    |  hints          | +----------------------------------------------+ |
    +-----------------+--------------------------------------------------+

All recording state comes from, and all commands go to, the
:class:`RecordingViewModel`; the tab itself only owns playback and file loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PySide6.QtCore import QTimer, QUrl, Signal, Slot, Qt
from PySide6.QtGui import QDesktopServices, QIntValidator
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.core.interfaces.audio_engine import IAudioPlaybackEngine
from app.core.models.device_info import AudioDeviceInfo, SerialPortInfo
from app.core.models.recording_session import RecordingSession
from app.infrastructure.storage import file_io
from app.services.project_manager import ProjectManager
from app.services.recording_service import SOURCE_MICROPHONE, SOURCE_SERIAL
from app.services.settings_service import SettingsService
from app.ui.viewmodels.recording_vm import RecordingViewModel
from app.ui.widgets.device_selector import DeviceSelector
from app.ui.widgets.elided_label import ElidedLabel
from app.ui.widgets.level_meter import LevelMeter
from app.ui.widgets.spectrum_widget import SpectrumWidget
from app.ui.widgets.waveform_widget import WaveformWidget
from app.ui.workers.analysis_worker import AnalysisWorker

SIDEBAR_WIDTH = 350
SAMPLE_RATE_PRESETS: tuple[int, ...] = (44100, 48000, 96000, 192000)
BAUD_RATES: tuple[int, ...] = (9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600)
_CUSTOM_RATE_RANGE = (1000, 768000)
_MAX_NUMERIC = 100_000_000
_PLAYBACK_POLL_MS = 100
_PLAY_PEAK_LIMIT = 0.95
_AUDIO_FILE_FILTER = "Аудио и данные (*.wav *.txt *.csv);;WAV (*.wav);;Текст / CSV (*.txt *.csv)"

_HINT_MICROPHONE = (
    "1. Выберите микрофон и частоту дискретизации.\n"
    "2. Выберите папку для записей.\n"
    "3. Нажмите «Старт».\n"
    "4. «Стоп и сохранить» запишет WAV-файл и добавит его в библиотеку."
)
_HINT_SERIAL = (
    "1. Выберите COM-порт и скорость обмена.\n"
    "2. Задайте частоту дискретизации для анализа данных.\n"
    "3. Нажмите «Старт». Каждая строка должна содержать число,\n"
    "   либо поток должен состоять из 16-битных слов АЦП.\n"
    "4. «Стоп и сохранить» запишет WAV-файл и добавит его в библиотеку."
)
_SOURCE_HINT = {
    SOURCE_MICROPHONE: "Звук записывается с выбранного микрофона.",
    SOURCE_SERIAL: "Числовые отсчёты читаются из COM-порта (внешний датчик).",
}

_STYLE = """
QPushButton { padding: 6px 12px; }
QPushButton#primary { background: #2563eb; color: white; border: none; border-radius: 4px; }
QPushButton#primary:hover { background: #1d4ed8; }
QPushButton#danger { background: #dc2626; color: white; border: none; border-radius: 4px; }
QPushButton#danger:hover { background: #b91c1c; }
QPushButton#primary:disabled, QPushButton#danger:disabled { background: #cbd5e1; color: #64748b; }
QLabel#muted { color: #64748b; }
QLabel#status { font-weight: 600; }
QLabel#status[error="true"] { color: #b91c1c; }
"""


def _default_audio_devices() -> Sequence[AudioDeviceInfo]:
    from app.infrastructure.audio.device_manager import AudioDeviceManager

    return AudioDeviceManager.list_input_devices()


def _default_serial_ports() -> Sequence[SerialPortInfo]:
    from app.infrastructure.serial.serial_engine import SerialCaptureEngine

    return SerialCaptureEngine().list_ports()


def _muted(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setObjectName("muted")
    label.setWordWrap(True)
    return label


class RecordTab(QWidget):
    """Recording screen bound to a :class:`RecordingViewModel`.

    Args:
        view_model: Source of all recording state and target of all commands.
        settings: Persisted settings (save directory).
        project_manager: Used to read back a recording right after it was saved.
        playback_engine: Used by Play / Stop playback; without one they stay disabled.
        audio_devices: Returns the available input devices (default: PortAudio).
        serial_ports: Returns the available serial ports (default: pyserial).

    Signals:
        ``audio_loaded(samples, sample_rate, label)`` - a recording was saved or a
        file was loaded; this is the signal the analysis tab listens to.
        ``error_occurred(message)`` - a tab-level failure (file loading, playback,
        device listing).  Recording errors come from the view model.
    """

    audio_loaded = Signal(object, int, str)
    error_occurred = Signal(str)

    def __init__(
        self,
        view_model: RecordingViewModel,
        settings: SettingsService,
        project_manager: ProjectManager,
        playback_engine: IAudioPlaybackEngine | None = None,
        parent: QWidget | None = None,
        *,
        audio_devices: Callable[[], Sequence[AudioDeviceInfo]] | None = None,
        serial_ports: Callable[[], Sequence[SerialPortInfo]] | None = None,
    ) -> None:
        super().__init__(parent)
        self._vm = view_model
        self._settings = settings
        self._pm = project_manager
        self._playback = playback_engine

        self._audio: tuple[np.ndarray, int] | None = None
        self._live_rate = 0
        self._is_playing = False
        self._active_loader: AnalysisWorker | None = None
        self._loaders: set[AnalysisWorker] = set()
        self._loader_labels: dict[AnalysisWorker, str] = {}

        self._mic_selector = DeviceSelector(
            audio_devices or _default_audio_devices,
            none_label="Устройство ввода по умолчанию",
            key=lambda d: d.index,
        )
        self._serial_selector = DeviceSelector(
            serial_ports or _default_serial_ports,
            none_label="Выберите COM-порт...",
            key=lambda p: p.device,
        )

        self.setStyleSheet(_STYLE)
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(12)
        root.addWidget(self._build_sidebar())
        root.addLayout(self._build_main_area(), 1)

        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(_PLAYBACK_POLL_MS)
        self._playback_timer.timeout.connect(self._poll_playback)

        self._connect_widgets()
        self._connect_view_model()
        self._load_devices()
        self._sync_from_view_model()
        self._reset_plots()

    # ===================================================================== UI
    def _build_sidebar(self) -> QScrollArea:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(10)

        # --- signal source
        self._source_group = QGroupBox("Источник сигнала")
        box = QVBoxLayout(self._source_group)
        self._mic_radio = QRadioButton("Микрофон")
        self._serial_radio = QRadioButton("COM-порт")
        self._source_buttons = QButtonGroup(self)
        self._source_buttons.addButton(self._mic_radio, 0)
        self._source_buttons.addButton(self._serial_radio, 1)
        self._source_hint = _muted()
        box.addWidget(self._mic_radio)
        box.addWidget(self._serial_radio)
        box.addWidget(self._source_hint)
        layout.addWidget(self._source_group)

        # --- microphone
        self._mic_group = QGroupBox("Микрофон")
        box = QVBoxLayout(self._mic_group)
        box.addWidget(self._mic_selector)
        rate_form = QFormLayout()
        rate_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        rate_row = QHBoxLayout()
        self._rate_combo = QComboBox()
        for rate in SAMPLE_RATE_PRESETS:
            self._rate_combo.addItem(f"{rate} Гц", rate)
        self._rate_combo.addItem("Другая...", None)
        self._rate_spin = QSpinBox()
        self._rate_spin.setRange(*_CUSTOM_RATE_RANGE)
        self._rate_spin.setValue(88200)
        self._rate_spin.setSuffix(" Гц")
        self._rate_spin.hide()
        rate_row.addWidget(self._rate_combo, 1)
        rate_row.addWidget(self._rate_spin, 1)
        rate_form.addRow("Частота дискретизации", rate_row)
        self._channel_combo = QComboBox()
        self._channel_combo.setToolTip("Входной канал, который будет записан (запись моно)")
        rate_form.addRow("Канал", self._channel_combo)
        box.addLayout(rate_form)
        self._mic_info = _muted()
        box.addWidget(self._mic_info)
        layout.addWidget(self._mic_group)

        # --- COM port
        self._serial_group = QGroupBox("COM-порт")
        box = QVBoxLayout(self._serial_group)
        box.addWidget(self._serial_selector)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self._baud_combo = QComboBox()
        self._baud_combo.setEditable(True)
        self._baud_combo.addItems([str(b) for b in BAUD_RATES])
        self._baud_combo.lineEdit().setValidator(QIntValidator(1, _MAX_NUMERIC, self))
        self._analysis_rate_edit = QLineEdit()
        self._analysis_rate_edit.setValidator(QIntValidator(1, _MAX_NUMERIC, self))
        self._analysis_rate_edit.setToolTip("Частота дискретизации (Гц), с которой анализируются данные COM-порта")
        form.addRow("Скорость (бод)", self._baud_combo)
        form.addRow("Частота анализа (Гц)", self._analysis_rate_edit)
        box.addLayout(form)
        self._serial_info = _muted()
        box.addWidget(self._serial_info)
        layout.addWidget(self._serial_group)

        # --- save directory
        self._save_group = QGroupBox("Папка сохранения")
        box = QVBoxLayout(self._save_group)
        self._dir_edit = QLineEdit(str(self._settings.save_directory))
        row = QHBoxLayout()
        self._browse_button = QPushButton("Обзор...")
        self._open_dir_button = QPushButton("Открыть")
        row.addWidget(self._browse_button, 1)
        row.addWidget(self._open_dir_button, 1)
        box.addWidget(self._dir_edit)
        box.addLayout(row)
        layout.addWidget(self._save_group)

        # --- recording details (stay editable while recording)
        self._meta_group = QGroupBox("Сведения о записи")
        form = QFormLayout(self._meta_group)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self._title_edit = QLineEdit()
        self._title_edit.setPlaceholderText("Запись")
        self._patient_edit = QLineEdit()
        self._notes_edit = QPlainTextEdit()
        self._notes_edit.setFixedHeight(84)
        form.addRow("Название", self._title_edit)
        form.addRow("Пациент / образец", self._patient_edit)
        form.addRow("Заметки", self._notes_edit)
        layout.addWidget(self._meta_group)

        # --- hints
        self._hints_group = QGroupBox("Подсказки")
        box = QVBoxLayout(self._hints_group)
        self._hints_label = _muted(_HINT_MICROPHONE)
        box.addWidget(self._hints_label)
        layout.addWidget(self._hints_group)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(SIDEBAR_WIDTH)
        self._sidebar = scroll
        return scroll

    def _build_main_area(self) -> QVBoxLayout:
        main = QVBoxLayout()
        main.setSpacing(8)

        top = QHBoxLayout()
        actions = QGroupBox("Основные действия")
        bar = QHBoxLayout(actions)
        self._start_button = QPushButton("Старт")
        self._start_button.setObjectName("primary")
        self._pause_button = QPushButton("Пауза")
        self._stop_button = QPushButton("Стоп и сохранить")
        self._stop_button.setObjectName("danger")
        self._discard_button = QPushButton("Отбросить")
        self._discard_button.setToolTip("Удалить запись, которую не удалось сохранить")
        self._discard_button.hide()
        self._play_button = QPushButton("Воспроизвести")
        self._stop_playback_button = QPushButton("Остановить воспроизведение")
        self._load_button = QPushButton("Загрузить файл")
        self._load_button.setObjectName("primary")
        for button in (
            self._start_button,
            self._pause_button,
            self._stop_button,
            self._discard_button,
            self._play_button,
            self._stop_playback_button,
            self._load_button,
        ):
            bar.addWidget(button)
        top.addWidget(actions, 0)

        level_box = QVBoxLayout()
        self._status_label = ElidedLabel()
        self._status_label.setObjectName("status")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._level_meter = LevelMeter()
        self._freq_label = ElidedLabel("Частота пика: -")
        self._freq_label.setObjectName("muted")
        self._freq_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        level_box.addWidget(self._status_label)
        level_box.addWidget(self._level_meter)
        level_box.addWidget(self._freq_label)
        top.addLayout(level_box, 1)
        main.addLayout(top)

        self._waveform = WaveformWidget()
        self._spectrum = SpectrumWidget()
        self._waveform.setMinimumHeight(180)
        self._spectrum.setMinimumHeight(180)
        main.addWidget(self._waveform, 3)
        main.addWidget(self._spectrum, 2)
        return main

    # ============================================================== wiring
    def _connect_widgets(self) -> None:
        vm = self._vm
        self._source_buttons.idClicked.connect(self._on_source_clicked)

        self._mic_selector.device_selected.connect(self._on_device_chosen)
        self._serial_selector.device_selected.connect(self._on_device_chosen)
        self._mic_selector.load_failed.connect(lambda m: self._report(f"Не удалось получить список аудиоустройств: {m}"))
        self._serial_selector.load_failed.connect(lambda m: self._report(f"Не удалось получить список COM-портов: {m}"))

        self._rate_combo.activated.connect(self._on_rate_activated)
        self._channel_combo.activated.connect(self._on_channel_activated)
        self._rate_spin.valueChanged.connect(self._on_custom_rate_changed)
        self._baud_combo.currentTextChanged.connect(self._on_baud_text)
        self._analysis_rate_edit.editingFinished.connect(self._on_analysis_rate_edited)

        self._dir_edit.editingFinished.connect(lambda: self._apply_save_directory(self._dir_edit.text()))
        self._browse_button.clicked.connect(lambda _c=False: self._browse_save_directory())
        self._open_dir_button.clicked.connect(lambda _c=False: self._open_save_directory())

        self._title_edit.textChanged.connect(lambda text: setattr(vm, "title", text))
        self._patient_edit.textChanged.connect(lambda text: setattr(vm, "patient", text))
        self._notes_edit.textChanged.connect(lambda: setattr(vm, "notes", self._notes_edit.toPlainText()))

        self._start_button.clicked.connect(lambda _c=False: self._start())
        self._pause_button.clicked.connect(lambda _c=False: self._toggle_pause())
        self._stop_button.clicked.connect(lambda _c=False: vm.on_stop())
        self._discard_button.clicked.connect(lambda _c=False: vm.on_discard_unsaved())
        self._play_button.clicked.connect(lambda _c=False: self.play())
        self._stop_playback_button.clicked.connect(lambda _c=False: self.stop_playback())
        self._load_button.clicked.connect(lambda _c=False: self.load_file())

    def _connect_view_model(self) -> None:
        vm = self._vm
        vm.is_recording_changed.connect(self._update_controls)
        vm.is_paused_changed.connect(self._update_controls)
        vm.has_unsaved_recording_changed.connect(self._update_controls)
        vm.current_source_changed.connect(self._on_source_changed)
        vm.sample_rate_changed.connect(self._sync_rate)
        vm.baudrate_changed.connect(self._sync_baud)
        vm.serial_analysis_rate_changed.connect(self._sync_analysis_rate)
        vm.channel_changed.connect(self._sync_channel)
        vm.title_changed.connect(lambda t: self._set_text(self._title_edit, t))
        vm.patient_changed.connect(lambda t: self._set_text(self._patient_edit, t))
        vm.notes_changed.connect(self._sync_notes)
        vm.status_text_changed.connect(self._show_status)
        vm.peak_dbfs_changed.connect(lambda _v: self._update_meter())
        vm.rms_dbfs_changed.connect(lambda _v: self._update_meter())
        vm.recording_started.connect(self._on_recording_started)
        vm.recording_stopped.connect(self._on_recording_stopped)
        vm.live_waveform.connect(self._on_live_waveform)
        vm.live_spectrum.connect(self._on_live_spectrum)
        vm.error_occurred.connect(self._on_vm_error)

    def _load_devices(self) -> None:
        """Fill the device lists and restore the last used selection."""
        self._mic_selector.refresh()
        self._serial_selector.refresh()
        saved_rate = self._settings.last_sample_rate
        mic = self._mic_selector.select_key(self._settings.last_device_index)
        if mic is not None:
            self._vm.on_device_selected(mic)
            self._vm.sample_rate = saved_rate  # selecting a device adopts its default rate
        self._update_device_info(mic)

    def _sync_from_view_model(self) -> None:
        vm = self._vm
        self._on_source_changed(vm.current_source)
        self._sync_rate(vm.sample_rate)
        self._sync_baud(vm.baudrate)
        self._sync_analysis_rate(vm.serial_analysis_rate)
        for edit, text in ((self._title_edit, vm.title), (self._patient_edit, vm.patient)):
            self._set_text(edit, text)
        self._sync_notes(vm.notes)
        self._show_status(vm.status_text)
        self._update_meter()
        self._update_controls()

    # ======================================================== small helpers
    @staticmethod
    def _set_text(edit: QLineEdit, text: str) -> None:
        if edit.text() != text:
            edit.blockSignals(True)
            edit.setText(text)
            edit.blockSignals(False)

    def _sync_notes(self, text: str) -> None:
        if self._notes_edit.toPlainText() != text:
            self._notes_edit.blockSignals(True)
            self._notes_edit.setPlainText(text)
            self._notes_edit.blockSignals(False)

    def _show_status(self, text: str, *, error: bool | None = None) -> None:
        self._status_label.setText(text)
        is_error = text.startswith("Ошибка") if error is None else error
        self._status_label.setProperty("error", is_error)
        self._status_label.style().unpolish(self._status_label)
        self._status_label.style().polish(self._status_label)

    def _report(self, message: str) -> None:
        """Show a tab-level error and tell the owner."""
        self._show_status(f"Ошибка: {message}", error=True)
        self.error_occurred.emit(message)

    def _reset_plots(self) -> None:
        self._waveform.clear("Нажмите «Старт», чтобы начать запись")
        self._spectrum.clear("Ожидание сигнала")
        self._freq_label.setText("Частота пика: -")

    # ================================================ source / device / rates
    @Slot(int)
    def _on_source_clicked(self, button_id: int) -> None:
        self._vm.on_source_changed(SOURCE_SERIAL if button_id == 1 else SOURCE_MICROPHONE)
        self._on_source_changed(self._vm.current_source)  # undo the click if it was refused

    def _on_source_changed(self, source: str) -> None:
        serial = source == SOURCE_SERIAL
        for radio, checked in ((self._mic_radio, not serial), (self._serial_radio, serial)):
            if radio.isChecked() != checked:
                self._source_buttons.blockSignals(True)
                radio.setChecked(checked)
                self._source_buttons.blockSignals(False)
        self._source_hint.setText(_SOURCE_HINT[source])
        self._hints_label.setText(_HINT_SERIAL if serial else _HINT_MICROPHONE)
        self._update_controls()

    def _on_device_chosen(self, device: Any) -> None:
        self._vm.on_device_selected(device)
        self._update_device_info(device)

    def _input_channel_count(self, device: Any) -> int:
        """Input channels of the chosen device (of the system default one for None)."""
        if isinstance(device, AudioDeviceInfo):
            return max(1, device.input_channels)
        try:
            from app.infrastructure.audio.device_manager import AudioDeviceManager

            default = AudioDeviceManager.default_input_index()
            for info in AudioDeviceManager.list_input_devices():
                if info.index == default:
                    return max(1, info.input_channels)
        except Exception:
            pass
        return 1

    def _fill_channels(self, device: Any) -> None:
        """List the channels of ``device`` and keep the chosen one if it still exists."""
        count = self._input_channel_count(device)
        wanted = min(self._vm.channel, count)
        self._channel_combo.blockSignals(True)
        self._channel_combo.clear()
        for number in range(1, count + 1):
            self._channel_combo.addItem(f"Канал {number}", number)
        self._channel_combo.setCurrentIndex(wanted - 1)
        self._channel_combo.blockSignals(False)
        if wanted != self._vm.channel:
            self._vm.channel = wanted

    def _sync_channel(self, channel: int) -> None:
        index = self._channel_combo.findData(channel)
        if index >= 0 and index != self._channel_combo.currentIndex():
            self._channel_combo.blockSignals(True)
            self._channel_combo.setCurrentIndex(index)
            self._channel_combo.blockSignals(False)

    @Slot(int)
    def _on_channel_activated(self, index: int) -> None:
        number = self._channel_combo.itemData(index)
        if number is not None:
            self._vm.channel = int(number)

    def _update_device_info(self, device: Any) -> None:
        if not isinstance(device, SerialPortInfo):
            self._fill_channels(device)
        if isinstance(device, AudioDeviceInfo):
            self._mic_info.setText(
                f"входных каналов: {device.input_channels}, по умолчанию {device.default_sample_rate} Гц, "
                f"{device.hostapi or 'неизвестный API'}"
            )
        elif isinstance(device, SerialPortInfo):
            self._serial_info.setText(device.description or device.device)
        else:
            self._mic_info.setText("Используется устройство ввода по умолчанию.")
            self._serial_info.setText("Выберите порт, к которому подключён датчик.")

    @Slot(int)
    def _on_rate_activated(self, index: int) -> None:
        rate = self._rate_combo.itemData(index)
        if rate is None:  # "Custom..."
            self._rate_spin.show()
            self._set_rate(self._rate_spin.value())
        else:
            self._rate_spin.hide()
            self._set_rate(int(rate))

    @Slot(int)
    def _on_custom_rate_changed(self, value: int) -> None:
        if self._rate_combo.currentData() is None:
            self._set_rate(value)

    def _set_rate(self, rate: int) -> None:
        try:
            self._vm.sample_rate = rate
        except ValueError as exc:
            self._report(str(exc))

    def _sync_rate(self, rate: int) -> None:
        self._rate_combo.blockSignals(True)
        self._rate_spin.blockSignals(True)
        index = self._rate_combo.findData(rate)
        if index >= 0:
            self._rate_combo.setCurrentIndex(index)
            self._rate_spin.hide()
        else:
            self._rate_combo.setCurrentIndex(self._rate_combo.count() - 1)
            self._rate_spin.setValue(max(_CUSTOM_RATE_RANGE[0], min(_CUSTOM_RATE_RANGE[1], rate)))
            self._rate_spin.show()
        self._rate_combo.blockSignals(False)
        self._rate_spin.blockSignals(False)

    @Slot(str)
    def _on_baud_text(self, text: str) -> None:
        if text.isdigit() and int(text) > 0:
            self._vm.baudrate = int(text)

    def _sync_baud(self, baud: int) -> None:
        if self._baud_combo.currentText() != str(baud):
            self._baud_combo.blockSignals(True)
            self._baud_combo.setCurrentText(str(baud))
            self._baud_combo.blockSignals(False)

    def _on_analysis_rate_edited(self) -> None:
        text = self._analysis_rate_edit.text().strip()
        if text.isdigit() and int(text) > 0:
            self._vm.serial_analysis_rate = int(text)
        else:
            self._sync_analysis_rate(self._vm.serial_analysis_rate)  # revert invalid input

    def _sync_analysis_rate(self, rate: int) -> None:
        if self._analysis_rate_edit.text() != str(rate):
            self._analysis_rate_edit.blockSignals(True)
            self._analysis_rate_edit.setText(str(rate))
            self._analysis_rate_edit.blockSignals(False)

    # ============================================================ save folder
    def set_calibration(self, volts_per_fs: float) -> None:
        """Show the live plots in volts (``volts_per_fs`` > 0) or in full-scale units (0)."""
        self._waveform.set_calibration(volts_per_fs)
        self._spectrum.set_calibration(volts_per_fs)

    def reload_settings(self) -> None:
        """Re-read the persisted settings (after the Settings dialog changed them)."""
        self._dir_edit.setText(str(self._settings.save_directory))
        if self._vm.is_recording:
            return  # the inputs are locked while recording
        self._vm.sample_rate = self._settings.last_sample_rate
        self._vm.channel = self._settings.last_channel
        self._vm.baudrate = self._settings.last_baudrate
        self._vm.serial_analysis_rate = self._settings.last_serial_analysis_rate
        self._vm.on_source_changed(self._settings.last_source)

    def _apply_save_directory(self, text: str) -> None:
        try:
            self._settings.save_directory = text
        except ValueError:
            pass
        self._dir_edit.blockSignals(True)
        self._dir_edit.setText(str(self._settings.save_directory))  # normalised, or reverted if invalid
        self._dir_edit.blockSignals(False)

    def _browse_save_directory(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Выберите папку для записей", str(self._settings.save_directory)
        )
        if chosen:
            self._apply_save_directory(chosen)

    def _open_save_directory(self) -> None:
        directory = self._settings.save_directory
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._report(f"Не удалось создать {directory}: {exc}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    # ============================================================ commands
    def _start(self) -> None:
        self._apply_save_directory(self._dir_edit.text())
        self.stop_playback()
        self._vm.on_start_recording()

    def _toggle_pause(self) -> None:
        if self._vm.is_paused:
            self._vm.on_resume()
        else:
            self._vm.on_pause()

    def _update_controls(self, *_: object) -> None:
        vm = self._vm
        recording, paused, unsaved = vm.is_recording, vm.is_paused, vm.has_unsaved_recording
        busy = self._active_loader is not None
        can_play = self._playback is not None and self._audio is not None
        serial = vm.current_source == SOURCE_SERIAL

        self._start_button.setEnabled(not recording and not unsaved and not busy)
        self._pause_button.setEnabled(recording)
        self._pause_button.setText("Продолжить" if paused else "Пауза")
        self._stop_button.setEnabled(recording or unsaved)
        self._stop_button.setText("Сохранить" if unsaved and not recording else "Стоп и сохранить")
        self._discard_button.setVisible(unsaved and not recording)
        self._play_button.setEnabled(can_play and not recording and not self._is_playing and not busy)
        self._stop_playback_button.setEnabled(self._is_playing)
        self._load_button.setEnabled(not recording and not busy)

        idle = not recording
        self._source_group.setEnabled(idle)
        self._mic_group.setEnabled(idle and not serial)
        self._serial_group.setEnabled(idle and serial)
        self._save_group.setEnabled(idle)

    # ============================================================= live plots
    def _on_recording_started(self, sample_rate: int) -> None:
        self._live_rate = int(sample_rate)
        self.stop_playback()
        self._reset_plots()

    def _on_live_waveform(self, samples: object) -> None:
        if self._live_rate > 0:
            self._waveform.update_live(samples, self._live_rate)  # type: ignore[arg-type]

    def _on_live_spectrum(self, freqs: object, magnitudes: object, peak_hz: object) -> None:
        peak = None if peak_hz is None else float(peak_hz)  # type: ignore[arg-type]
        self._spectrum.update_spectrum(freqs, magnitudes, peak)  # type: ignore[arg-type]
        self._freq_label.setText("Частота пика: -" if peak is None else f"Частота пика: {peak / 1000.0:.2f} кГц")

    def _update_meter(self) -> None:
        self._level_meter.set_levels(self._vm.peak_dbfs, self._vm.rms_dbfs)

    def _on_vm_error(self, message: str) -> None:
        self._update_controls()

    # =============================================== saved recordings & files
    def _on_recording_stopped(self, session: RecordingSession) -> None:
        self._spectrum.clear("Запись остановлена")
        self._freq_label.setText("Частота пика: -")
        if session.id is not None:
            self._start_loader(self._pm.load_recording, (session.id,), "load_saved_recording")

    def load_file(self, path: str | Path | None = None) -> None:
        """Load a WAV / TXT / CSV file (a file dialog is shown if ``path`` is None).

        The audio is read on a background thread, shown as the static waveform, made
        available for playback and announced through ``audio_loaded``.
        """
        if self._vm.is_recording:
            return
        if path is None:
            chosen, _ = QFileDialog.getOpenFileName(
                self, "Загрузить аудио или файл данных", str(self._settings.save_directory), _AUDIO_FILE_FILTER
            )
            if not chosen:
                return
            path = chosen
        self._start_loader(file_io.read_audio_file, (str(path),), "load_file", label=Path(path).name)

    def _start_loader(
        self, fn: Callable[..., Any], args: tuple[Any, ...], name: str, label: str = ""
    ) -> None:
        worker = AnalysisWorker(fn, args, task_name=name)
        worker.result_ready.connect(self._on_loader_result)
        worker.error.connect(self._on_loader_error)
        worker.finished.connect(self._on_loader_finished)
        self._loaders.add(worker)
        self._loader_labels[worker] = label
        self._active_loader = worker  # an older, still running load becomes stale
        self._update_controls()
        worker.start()

    @Slot(str, object)
    def _on_loader_result(self, name: str, result: object) -> None:
        worker = self.sender()
        if worker is not self._active_loader:
            return
        self._active_loader = None
        if name == "load_saved_recording":
            session, samples = result  # type: ignore[misc]
            self.set_audio(samples, session.sample_rate, session.title)
        else:
            rate, samples = result  # type: ignore[misc]
            self.set_audio(samples, rate, self._loader_labels.get(worker, ""))

    @Slot(str, str)
    def _on_loader_error(self, name: str, message: str) -> None:
        if self.sender() is not self._active_loader:
            return
        self._active_loader = None
        self._update_controls()
        self._report(message)

    @Slot()
    def _on_loader_finished(self) -> None:
        worker = self.sender()
        if isinstance(worker, AnalysisWorker):
            self._loaders.discard(worker)
            self._loader_labels.pop(worker, None)
            worker.deleteLater()

    def set_audio(self, samples: np.ndarray, sample_rate: int, label: str = "") -> None:
        """Make ``samples`` the current audio: show it, allow playback, announce it."""
        self.stop_playback()
        self._audio = (np.asarray(samples, dtype=np.float32), int(sample_rate))
        self._waveform.set_data(self._audio[0], self._audio[1])
        self._update_controls()
        self.audio_loaded.emit(self._audio[0], self._audio[1], label)

    @property
    def current_audio(self) -> tuple[np.ndarray, int] | None:
        """``(samples, sample_rate)`` of the loaded / last recorded audio, if any."""
        return self._audio

    # ================================================================ playback
    def play(self) -> None:
        """Play the current audio on the default output device."""
        if self._playback is None or self._audio is None or self._vm.is_recording:
            return
        samples, rate = self._audio
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak > _PLAY_PEAK_LIMIT:  # never send over-range audio to the speakers
            samples = samples * np.float32(_PLAY_PEAK_LIMIT / peak)
        try:
            self._playback.play(samples, rate)
        except Exception as exc:
            self._report(f"Не удалось воспроизвести звук: {exc}")
            return
        self._is_playing = True
        self._playback_timer.start()
        self._update_controls()

    def stop_playback(self) -> None:
        """Stop playback if it is running."""
        if self._playback is not None and (self._is_playing or self._playback.is_playing):
            try:
                self._playback.stop()
            except Exception as exc:
                self._report(f"Не удалось остановить воспроизведение: {exc}")
        self._finish_playback()

    def _finish_playback(self) -> None:
        self._playback_timer.stop()
        if self._is_playing:
            self._is_playing = False
            self._update_controls()

    def _poll_playback(self) -> None:
        if self._playback is None or not self._playback.is_playing:
            self._finish_playback()

    def shutdown(self) -> None:
        """Stop playback and wait for background loads (call when the window closes)."""
        self.stop_playback()
        for worker in list(self._loaders):
            worker.wait(5000)


__all__: list[str] = ["RecordTab", "SIDEBAR_WIDTH"]
