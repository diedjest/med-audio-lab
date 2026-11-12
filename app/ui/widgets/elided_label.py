"""Single-line label that elides its text instead of growing its layout."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPainter, QPaintEvent, QPalette
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget


class ElidedLabel(QLabel):
    """Shows its text on one line, elided (``...``) to the width it is given.

    The full text stays available through ``text()`` and the tooltip, and a long
    message can no longer make its layout grow.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:
        super().setText(text)
        self.setToolTip(text)

    def minimumSizeHint(self) -> QSize:
        return QSize(0, super().minimumSizeHint().height())

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        elided = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, self.width())
        self.style().drawItemText(
            painter,
            self.rect(),
            int(self.alignment()),
            self.palette(),
            self.isEnabled(),
            elided,
            QPalette.ColorRole.WindowText,
        )


__all__: list[str] = ["ElidedLabel"]
