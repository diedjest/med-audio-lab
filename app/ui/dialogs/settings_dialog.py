"""Application settings: where recordings go and the defaults for new recordings."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.services.recording_service import SOURCE_MICROPHONE, SOURCE_SERIAL
from app.services.settings_service import SettingsService

_SOURCES = ((SOURCE_MICROPHONE, "Microphone"), (SOURCE_SERIAL, "COM Port"))
_BIT_DEPTHS = (("PCM_16", "16-bit PCM"), ("PCM_24", "24-bit PCM"), ("FLOAT_32", "32-bit float"))


def _spin(low: int, high: int, value: int, suffix: str = "") -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(low, high)
    spin.setValue(max(low, min(high, value)))
    spin.setSuffix(suffix)
    return spin


class SettingsDialog(QDialog):
    """Edits the persisted settings.  Nothing is written until OK / :meth:`apply`."""

    def __init__(self, settings: SettingsService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self.setWindowTitle("Settings")
        self.setMinimumWidth(480)

        self._dir_edit = QLineEdit(str(settings.save_directory))
        browse = QPushButton("Browse...")
        browse.clicked.connect(lambda _c=False: self._browse())
        dir_row = QHBoxLayout()
        dir_row.addWidget(self._dir_edit, 1)
        dir_row.addWidget(browse)

        self._source = QComboBox()
        for key, text in _SOURCES:
            self._source.addItem(text, key)
        self._source.setCurrentIndex(max(0, self._source.findData(settings.last_source)))
        self._bit_depth = QComboBox()
        for key, text in _BIT_DEPTHS:
            self._bit_depth.addItem(text, key)
        self._bit_depth.setCurrentIndex(max(0, self._bit_depth.findData(settings.wav_bit_depth)))
        self._sample_rate = _spin(1000, 768_000, settings.last_sample_rate, " Hz")
        self._baud = _spin(1, 100_000_000, settings.last_baudrate, " baud")
        self._analysis_rate = _spin(1, 100_000_000, settings.last_serial_analysis_rate, " Hz")

        form = QFormLayout()
        form.addRow("Save recordings in", dir_row)
        form.addRow("WAV bit depth", self._bit_depth)
        form.addRow("Default source", self._source)
        form.addRow("Microphone sample rate", self._sample_rate)
        form.addRow("COM port baud rate", self._baud)
        form.addRow("COM data analysis rate", self._analysis_rate)

        self._error = QLabel()
        self._error.setObjectName("status")
        self._error.setProperty("error", True)
        self._error.setWordWrap(True)
        self._error.hide()

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._error)
        layout.addWidget(self._buttons)

    # ------------------------------------------------------------- fields
    @property
    def save_directory_edit(self) -> QLineEdit:
        return self._dir_edit

    @property
    def source_combo(self) -> QComboBox:
        return self._source

    @property
    def bit_depth_combo(self) -> QComboBox:
        return self._bit_depth

    @property
    def sample_rate_spin(self) -> QSpinBox:
        return self._sample_rate

    @property
    def baud_spin(self) -> QSpinBox:
        return self._baud

    @property
    def analysis_rate_spin(self) -> QSpinBox:
        return self._analysis_rate

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose the folder for recordings", self._dir_edit.text())
        if chosen:
            self._dir_edit.setText(chosen)

    # -------------------------------------------------------------- apply
    def apply(self) -> None:
        """Write the values to the settings.

        Raises:
            ValueError: the save directory is empty.
        """
        text = self._dir_edit.text().strip()
        if not text:
            raise ValueError("The save directory must not be empty.")
        self._settings.save_directory = Path(text)
        self._settings.wav_bit_depth = str(self._bit_depth.currentData())
        self._settings.last_source = str(self._source.currentData())
        self._settings.last_sample_rate = self._sample_rate.value()
        self._settings.last_baudrate = self._baud.value()
        self._settings.last_serial_analysis_rate = self._analysis_rate.value()

    def accept(self) -> None:
        try:
            self.apply()
        except ValueError as exc:
            self._error.setText(str(exc))
            self._error.show()
            return
        super().accept()


__all__: list[str] = ["SettingsDialog"]
