"""Shared styling and placeholder overlay for the pyqtgraph-based widgets."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtWidgets import QLabel, QStackedLayout, QWidget

COLOR_BACKGROUND = "#ffffff"
COLOR_WAVEFORM = "#2563eb"  # dark blue
COLOR_SPECTRUM = "#0f766e"  # teal
COLOR_PEAK = "#dc2626"  # red
COLOR_AXIS = "#334155"
COLOR_PLACEHOLDER = "#64748b"
COLOR_HOVER = "#64748b"
GRID_ALPHA = 0.5
_HOVER_MARGIN = 8


def finite_float32(data: object) -> np.ndarray:
    """1D float32 view/copy of ``data`` (first channel of 2D input), non-finite values -> 0."""
    arr = np.asarray(data)
    if arr.ndim > 1:
        arr = arr[:, 0]
    arr = np.asarray(arr, dtype=np.float32).ravel()
    if arr.size and not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def format_volts(volts: float) -> str:
    """``0.0123 -> "12.3 мВ"``: volts with an SI prefix that keeps 1..999 in front."""
    magnitude = abs(volts)
    for factor, unit in ((1.0, "В"), (1e-3, "мВ"), (1e-6, "мкВ")):
        if magnitude >= factor:
            return f"{volts / factor:.4g} {unit}"
    return f"{volts / 1e-6:.4g} мкВ"


def nearest_index(xs: np.ndarray, x: float) -> int:
    """Index of the element of the ascending array ``xs`` closest to ``x``."""
    i = int(np.searchsorted(xs, x))
    if i > 0 and (i == xs.size or x - xs[i - 1] <= xs[i] - x):
        i -= 1
    return i


class PlotWidgetBase(QWidget):
    """A styled white ``pg.PlotWidget`` with a centered placeholder message on top.

    Moving the mouse over the plot shows the value under the cursor in a small box
    in the top-right corner.  Subclasses supply the text by overriding
    :meth:`_hover_text`.
    """

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

        self._hover_label = QLabel(self)
        self._hover_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._hover_label.setStyleSheet(
            f"background: rgba(255, 255, 255, 230); color: {COLOR_AXIS}; font-size: 11px;"
            " border: 1px solid #cbd5e1; padding: 3px 6px;"
        )
        self._hover_label.hide()
        self._hover_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen(COLOR_HOVER, width=1, style=Qt.PenStyle.DashLine)
        )
        self._hover_line.setZValue(50)
        item.addItem(self._hover_line, ignoreBounds=True)
        self._hover_line.hide()
        self._plot.scene().sigMouseMoved.connect(self._on_hover_moved)
        self._volts_per_fs = 0.0

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

    # --------------------------------------------------------- calibration
    @property
    def volts_per_fs(self) -> float:
        """Volts per full-scale amplitude of 1.0; 0.0 while the plot is not calibrated."""
        return self._volts_per_fs

    def set_calibration(self, volts_per_fs: float) -> None:
        """Show amplitudes in volts (``volts_per_fs`` > 0) or in full-scale units (0)."""
        value = float(volts_per_fs)
        self._volts_per_fs = value if value > 0.0 else 0.0
        self._apply_calibration()

    def _apply_calibration(self) -> None:
        """Hook: update the axes after :meth:`set_calibration`."""

    def _set_amplitude_axis(self, name: str) -> None:
        """Label the left axis ``name`` and scale its ticks to volts when calibrated."""
        item = self._plot.getPlotItem()
        axis = item.getAxis("left")
        if self._volts_per_fs > 0.0:
            axis.setScale(self._volts_per_fs)
            item.setLabel("left", name, units="В", color=COLOR_AXIS)
        else:
            axis.setScale(1.0)
            item.setLabel("left", name, units=None, color=COLOR_AXIS)

    def _amplitude_text(self, value: float) -> str:
        """``value`` (full-scale units) as text: volts when calibrated."""
        if self._volts_per_fs > 0.0:
            return format_volts(value * self._volts_per_fs)
        return f"{value:+.5f}"

    # --------------------------------------------------------------- hover
    @property
    def hover_text(self) -> str:
        """Text of the value read-out ("" if hidden)."""
        return self._hover_label.text() if not self._hover_label.isHidden() else ""

    def _hover_text(self, x: float, y: float) -> str | None:
        """Read-out for the view coordinates ``(x, y)``; None to show nothing."""
        return None

    def _hover_crosshair(self) -> bool:
        """Whether a vertical line follows the cursor."""
        return True

    def _on_hover_moved(self, scene_pos: QPointF) -> None:
        vb = self._plot.getPlotItem().vb
        if self._placeholder.isVisible() or not vb.sceneBoundingRect().contains(scene_pos):
            self._clear_hover()
            return
        point = vb.mapSceneToView(scene_pos)
        text = self._hover_text(float(point.x()), float(point.y()))
        if not text:
            self._clear_hover()
            return
        self._hover_label.setText(text)
        self._hover_label.adjustSize()
        self._hover_label.move(self.width() - self._hover_label.width() - _HOVER_MARGIN, _HOVER_MARGIN)
        self._hover_label.show()
        self._hover_label.raise_()
        if self._hover_crosshair():
            self._hover_line.setPos(float(point.x()))
            self._hover_line.show()
        else:
            self._hover_line.hide()

    def _clear_hover(self) -> None:
        self._hover_label.hide()
        self._hover_line.hide()

    def leaveEvent(self, event: QEvent) -> None:
        self._clear_hover()
        super().leaveEvent(event)


__all__: list[str] = [
    "PlotWidgetBase",
    "finite_float32",
    "nearest_index",
    "format_volts",
    "COLOR_WAVEFORM",
    "COLOR_SPECTRUM",
    "COLOR_PEAK",
    "COLOR_AXIS",
    "GRID_ALPHA",
]
