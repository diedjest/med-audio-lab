"""Shared styling and placeholder overlay for the pyqtgraph-based widgets."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QStackedLayout, QWidget

COLOR_BACKGROUND = "#ffffff"
COLOR_WAVEFORM = "#2563eb"  # dark blue
COLOR_SPECTRUM = "#0f766e"  # teal
COLOR_PEAK = "#dc2626"  # red
COLOR_AXIS = "#334155"
COLOR_PLACEHOLDER = "#64748b"
GRID_ALPHA = 0.5


def finite_float32(data: object) -> np.ndarray:
    """1D float32 view/copy of ``data`` (first channel of 2D input), non-finite values -> 0."""
    arr = np.asarray(data)
    if arr.ndim > 1:
        arr = arr[:, 0]
    arr = np.asarray(arr, dtype=np.float32).ravel()
    if arr.size and not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


class PlotWidgetBase(QWidget):
    """A styled white ``pg.PlotWidget`` with a centered placeholder message on top."""

    def __init__(
        self,
        x_label: str,
        y_label: str,
        parent: QWidget | None = None,
        *,
        axis_items: dict[str, pg.AxisItem] | None = None,
    ) -> None:
        super().__init__(parent)
        self._plot = pg.PlotWidget(background=COLOR_BACKGROUND, axisItems=axis_items)
        self._plot.setAntialiasing(True)

        item = self._plot.getPlotItem()
        item.setMenuEnabled(False)  # the widgets provide their own context menus
        item.hideButtons()
        item.showGrid(x=True, y=True, alpha=GRID_ALPHA)
        for name in ("bottom", "left"):
            axis = item.getAxis(name)
            axis.setPen(pg.mkPen(COLOR_AXIS))
            axis.setTextPen(pg.mkPen(COLOR_AXIS))
        item.setLabel("bottom", x_label, color=COLOR_AXIS)
        item.setLabel("left", y_label, color=COLOR_AXIS)
        item.vb.setMouseEnabled(x=True, y=False)  # wheel zooms and drag pans X only

        self._placeholder = QLabel(self)
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._placeholder.setStyleSheet(
            f"background: transparent; color: {COLOR_PLACEHOLDER}; font-size: 14px;"
        )
        self._placeholder.hide()

        layout = QStackedLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setStackingMode(QStackedLayout.StackingMode.StackAll)
        layout.addWidget(self._plot)
        layout.addWidget(self._placeholder)
        self._placeholder.raise_()

    @property
    def plot_widget(self) -> pg.PlotWidget:
        """The underlying ``pyqtgraph.PlotWidget``."""
        return self._plot

    @property
    def placeholder_text(self) -> str:
        """Placeholder message currently shown ("" if hidden)."""
        return self._placeholder.text() if not self._placeholder.isHidden() else ""

    def _show_placeholder(self, message: str) -> None:
        self._placeholder.setText(message)
        self._placeholder.setVisible(bool(message))
        self._placeholder.raise_()

    def _hide_placeholder(self) -> None:
        self._placeholder.hide()


__all__: list[str] = [
    "PlotWidgetBase",
    "finite_float32",
    "COLOR_WAVEFORM",
    "COLOR_SPECTRUM",
    "COLOR_PEAK",
    "COLOR_AXIS",
    "GRID_ALPHA",
]
