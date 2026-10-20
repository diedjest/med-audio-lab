"""Live frequency spectrum display with a peak marker."""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from app.ui.widgets._plot_base import COLOR_PEAK, COLOR_SPECTRUM, PlotWidgetBase, finite_float32, nearest_index

_TEXT_Y_FRACTION = 0.96  # height of the peak label as a fraction of the y range
_Y_HEADROOM = 1.15


class SpectrumWidget(PlotWidgetBase):
    """pyqtgraph spectrum plot: frequency in kHz, teal line, dashed red peak marker."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Frequency (kHz)", "Magnitude", parent)
        item = self._plot.getPlotItem()
        self._vb = item.vb
        self._curve = item.plot(pen=pg.mkPen(COLOR_SPECTRUM, width=1.5), antialias=True)

        self._peak_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(COLOR_PEAK, width=1.5, style=Qt.PenStyle.DashLine),
        )
        self._peak_label = pg.TextItem(color=COLOR_PEAK, anchor=(0.0, 0.0))
        item.addItem(self._peak_line, ignoreBounds=True)
        item.addItem(self._peak_label, ignoreBounds=True)
        self._peak_line.hide()
        self._peak_label.hide()

        self._x_span: tuple[float, float, int] | None = None
        self._khz: np.ndarray | None = None
        self._mags: np.ndarray | None = None
        self.clear("No signal")

    # ------------------------------------------------------------- state
    @property
    def has_data(self) -> bool:
        """True if a spectrum is currently displayed."""
        return self._x_span is not None

    @property
    def peak_label_text(self) -> str:
        """Text of the visible peak label ("" if no peak is shown)."""
        return self._peak_label.toPlainText() if self._peak_label.isVisible() else ""

    @property
    def peak_line_khz(self) -> float | None:
        """Position of the visible peak marker in kHz, or None."""
        return float(self._peak_line.value()) if self._peak_line.isVisible() else None

    # ------------------------------------------------------------ update
    def update_spectrum(
        self,
        freqs_hz: np.ndarray,
        magnitudes: np.ndarray,
        peak_hz: float | None,
    ) -> None:
        """Show a spectrum and mark its peak.

        Args:
            freqs_hz: Frequency axis in Hz.
            magnitudes: Magnitudes, same length as ``freqs_hz``.
            peak_hz: Peak frequency in Hz, or None for no marker.

        Raises:
            ValueError: ``freqs_hz`` and ``magnitudes`` differ in length.
        """
        freqs = np.asarray(freqs_hz, dtype=np.float64).ravel()
        mags = finite_float32(magnitudes)
        if freqs.size != mags.size:
            raise ValueError(
                f"freqs_hz ({freqs.size}) and magnitudes ({mags.size}) must have the same length"
            )
        if freqs.size == 0:
            self.clear("No signal")
            return

        self._hide_placeholder()
        khz = freqs / 1000.0
        self._curve.setData(khz, mags, skipFiniteCheck=True)
        self._khz, self._mags = khz, mags

        span = (float(khz[0]), float(khz[-1]), khz.size)
        if span != self._x_span:
            self._x_span = span
            if span[1] > span[0]:
                self._vb.setXRange(span[0], span[1], padding=0)

        peak_amp = float(mags.max())
        top = peak_amp * _Y_HEADROOM if peak_amp > 0.0 else 1.0
        self._vb.setYRange(0.0, top, padding=0)

        if peak_hz is not None and math.isfinite(peak_hz):
            peak_khz = float(peak_hz) / 1000.0
            self._peak_line.setValue(peak_khz)
            self._peak_line.show()
            self._peak_label.setText(f"peak: {peak_khz:.2f} kHz")
            # keep the label inside the plot: flip to the left of the line near the right edge
            on_right_half = peak_khz > 0.5 * (span[0] + span[1])
            self._peak_label.setAnchor((1.0, 0.0) if on_right_half else (0.0, 0.0))
            self._peak_label.setPos(peak_khz, top * _TEXT_Y_FRACTION)
            self._peak_label.show()
        else:
            self._peak_line.hide()
            self._peak_label.hide()

    def clear(self, message: str = "No signal") -> None:
        """Remove the spectrum and show ``message`` as a placeholder."""
        self._curve.setData([], [])
        self._peak_line.hide()
        self._peak_label.hide()
        self._x_span = None
        self._khz = self._mags = None
        self._vb.setXRange(0.0, 1.0, padding=0)
        self._vb.setYRange(0.0, 1.0, padding=0)
        self._show_placeholder(message)

    # --------------------------------------------------------------- hover
    def _hover_text(self, x: float, y: float) -> str | None:
        if self._khz is None or self._mags is None:
            return None
        i = nearest_index(self._khz, x)
        return f"Frequency: {self._khz[i]:.3f} kHz\nMagnitude: {self._mags[i]:.4g}"


__all__: list[str] = ["SpectrumWidget"]
