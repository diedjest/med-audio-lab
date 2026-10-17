"""Waveform display: a static view with zoomable min/max decimation and a live view."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QColor, QContextMenuEvent, QResizeEvent
from PySide6.QtWidgets import QFileDialog, QMenu, QWidget

from app.ui.widgets._plot_base import COLOR_PEAK, COLOR_WAVEFORM, PlotWidgetBase, finite_float32

#: Hard cap: more visible samples than this are always reduced to min/max pairs.
RAW_POINT_LIMIT: int = 100_000
#: Below the cap, samples are also reduced once there are more than this many
#: per pixel column (they would not be distinguishable on screen anyway).
RAW_SAMPLES_PER_PIXEL: int = 8
#: Samples summarised per cell of the min/max overview pyramid.
PYRAMID_CELL: int = 64
_MIN_BUCKETS: int = 256
#: Most shaded spans drawn by :meth:`WaveformWidget.set_regions`.
MAX_REGIONS: int = 500


def _reduce_minmax(
    lo: np.ndarray,
    hi: np.ndarray,
    group: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse consecutive groups of ``group`` cells into (min, max, centre) triples.

    Args:
        lo: Per-cell minima (for raw samples, the samples themselves).
        hi: Per-cell maxima (for raw samples, the samples themselves).
        group: Cells per output bucket (>= 1); a shorter last bucket is kept.

    Returns:
        ``(mins, maxs, centres)`` with centres in cell units from the first cell.
    """
    group = max(1, int(group))
    m = lo.size
    full = m // group
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    centres: list[np.ndarray] = []
    if full:
        mins.append(lo[: full * group].reshape(full, group).min(axis=1))
        maxs.append(hi[: full * group].reshape(full, group).max(axis=1))
        centres.append(np.arange(full, dtype=np.float64) * group + group / 2.0)
    tail = m - full * group
    if tail:
        mins.append(lo[full * group :].min(keepdims=True))
        maxs.append(hi[full * group :].max(keepdims=True))
        centres.append(np.array([full * group + tail / 2.0]))
    return np.concatenate(mins), np.concatenate(maxs), np.concatenate(centres)


def _interleave(
    mins: np.ndarray,
    maxs: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn bucket (min, max, time) into a polyline with a min/max pair per bucket."""
    y = np.empty(mins.size * 2, dtype=np.float32)
    y[0::2] = mins
    y[1::2] = maxs
    return np.repeat(times, 2), y


class WaveformWidget(PlotWidgetBase):
    """Anti-aliased waveform plot.

    * :meth:`set_data` shows a recording.  Only the visible part is drawn; when
      more than 100,000 samples (or more than ~8 per pixel column) are visible they
      are reduced to min/max pairs per pixel column, so peaks never disappear and zooming in restores full detail.
    * :meth:`update_live` replaces the content of a rolling window cheaply enough
      for 60 FPS updates.
    * Wheel zooms the time axis, dragging pans it, right-click offers
      "Reset Zoom" and "Export PNG".
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Time (s)", "Amplitude", parent)
        item = self._plot.getPlotItem()
        self._vb = item.vb
        self._curve = item.plot(pen=pg.mkPen(COLOR_WAVEFORM, width=1), antialias=True)

        self._mode = "empty"  # "empty" | "static" | "live"
        self._rendering = False
        # static data
        self._y: np.ndarray | None = None
        self._pyr_min: np.ndarray | None = None
        self._pyr_max: np.ndarray | None = None
        self._sr = 0
        # live data
        self._live_key: tuple[int, int, int] | None = None
        self._live_x: np.ndarray | None = None
        self._live_span: tuple[int, int] | None = None
        self._live_group = 1
        # optional second track (e.g. an envelope) and highlighted time spans (e.g. impulses)
        self._ov_y: np.ndarray | None = None
        self._ov_pmin: np.ndarray | None = None
        self._ov_pmax: np.ndarray | None = None
        self._ov_curve: pg.PlotDataItem | None = None
        self._regions: list[pg.LinearRegionItem] = []

        self._vb.sigXRangeChanged.connect(self._on_x_range_changed)
        self.clear("No data")

    # ------------------------------------------------------------- state
    @property
    def has_data(self) -> bool:
        """True if a waveform is currently displayed."""
        return self._mode != "empty"

    @property
    def displayed_point_count(self) -> int:
        """Number of points currently handed to the plot curve."""
        x, _ = self._curve.getData()
        return 0 if x is None else int(x.size)

    def _target_buckets(self) -> int:
        return max(_MIN_BUCKETS, int(self._plot.viewport().width()))

    def _raw_limit(self) -> int:
        """Most samples drawn one-for-one: few enough to stay smooth at 60 FPS."""
        return min(RAW_POINT_LIMIT, RAW_SAMPLES_PER_PIXEL * self._target_buckets())

    # ------------------------------------------------------------ static
    def set_data(self, data: np.ndarray, sample_rate: int) -> None:
        """Display a static waveform (replaces whatever was shown).

        The data is copied.  The time axis is reset to the full recording.

        Raises:
            ValueError: non-positive sample rate.
        """
        if int(sample_rate) <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        y = finite_float32(data).copy()
        if y.size == 0:
            self.clear("No data")
            return

        self.clear_overlay()
        self.clear_regions()
        self._mode = "static"
        self._y = y
        self._sr = int(sample_rate)
        self._live_key = self._live_x = self._live_span = None
        self._pyr_min = self._pyr_max = None
        if y.size > RAW_POINT_LIMIT:
            self._pyr_min, self._pyr_max, _ = _reduce_minmax(y, y, PYRAMID_CELL)

        self._hide_placeholder()
        duration = y.size / self._sr
        self._vb.setLimits(xMin=0.0, xMax=duration, minXRange=min(duration, 4.0 / self._sr))
        self._vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        self._vb.setXRange(0.0, duration, padding=0)
        self._render_static()

    def _on_x_range_changed(self, *_: object) -> None:
        if self._mode == "static":
            self._render_static()

    def _track_points(
        self,
        y: np.ndarray,
        pyr_min: np.ndarray | None,
        pyr_max: np.ndarray | None,
        x0: float,
        x1: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Points of one track for the visible time range: raw samples when few,
        min/max buckets when many.  None if the range lies outside the track."""
        sr, n = self._sr, y.size
        i0 = max(0, int(math.floor(x0 * sr)) - 1)
        i1 = min(n, int(math.ceil(x1 * sr)) + 2)
        if i1 <= i0:
            return None

        count = i1 - i0
        if count <= self._raw_limit():
            return np.arange(i0, i1, dtype=np.float64) / sr, y[i0:i1]

        group = math.ceil(count / self._target_buckets())
        if pyr_min is not None and pyr_max is not None and group >= 2 * PYRAMID_CELL:
            c0, c1 = i0 // PYRAMID_CELL, -(-i1 // PYRAMID_CELL)
            mins, maxs, centres = _reduce_minmax(pyr_min[c0:c1], pyr_max[c0:c1], group // PYRAMID_CELL)
            times = (c0 + centres) * PYRAMID_CELL / sr
        else:
            mins, maxs, centres = _reduce_minmax(y[i0:i1], y[i0:i1], group)
            times = (i0 + centres) / sr
        return _interleave(mins, maxs, times)

    def _render_static(self) -> None:
        """Draw the visible part of the main track and of the overlay track."""
        y = self._y
        if y is None or self._rendering:
            return
        self._rendering = True
        try:
            x0, x1 = self._vb.viewRange()[0]
            points = self._track_points(y, self._pyr_min, self._pyr_max, x0, x1)
            if points is None:
                self._curve.setData([], [])
            else:
                self._curve.setData(*points, skipFiniteCheck=True)
            if self._ov_y is not None and self._ov_curve is not None:
                overlay = self._track_points(self._ov_y, self._ov_pmin, self._ov_pmax, x0, x1)
                if overlay is None:
                    self._ov_curve.setData([], [])
                else:
                    self._ov_curve.setData(*overlay, skipFiniteCheck=True)
        finally:
            self._rendering = False

    # ------------------------------------------------- overlay and regions
    def set_overlay(self, data: np.ndarray, color: str = COLOR_PEAK) -> None:
        """Draw a second track (for example an envelope) over the static waveform.

        It shares the time axis and the zoom-dependent decimation of the main track.

        Raises:
            RuntimeError: no static waveform is displayed.
        """
        if self._mode != "static" or self._y is None:
            raise RuntimeError("set_overlay() needs a static waveform; call set_data() first")
        y = finite_float32(data).copy()
        self.clear_overlay()
        if y.size == 0:
            return
        self._ov_y = y
        if y.size > RAW_POINT_LIMIT:
            self._ov_pmin, self._ov_pmax, _ = _reduce_minmax(y, y, PYRAMID_CELL)
        self._ov_curve = self._plot.getPlotItem().plot(pen=pg.mkPen(color, width=1.5), antialias=True)
        self._render_static()

    def clear_overlay(self) -> None:
        """Remove the overlay track."""
        if self._ov_curve is not None:
            self._plot.getPlotItem().removeItem(self._ov_curve)
        self._ov_y = self._ov_pmin = self._ov_pmax = None
        self._ov_curve = None

    def set_regions(
        self,
        spans: Sequence[tuple[float, float]],
        color: str = COLOR_WAVEFORM,
        limit: int = MAX_REGIONS,
    ) -> None:
        """Shade time spans ``(start_s, end_s)`` (for example detected impulses).

        At most ``limit`` spans are drawn, to keep the plot responsive.
        """
        self.clear_regions()
        shade = QColor(color)
        shade.setAlpha(60)
        for start, end in list(spans)[:limit]:
            region = pg.LinearRegionItem(
                values=(float(start), float(end)),
                movable=False,
                brush=pg.mkBrush(shade),
                pen=pg.mkPen(color, width=1),
            )
            region.setZValue(-10)
            self._plot.getPlotItem().addItem(region, ignoreBounds=True)
            self._regions.append(region)

    def clear_regions(self) -> None:
        """Remove all shaded spans."""
        for region in self._regions:
            self._plot.getPlotItem().removeItem(region)
        self._regions.clear()

    def set_time_range(self, start_s: float, end_s: float) -> None:
        """Zoom the time axis to ``[start_s, end_s]`` (clamped to the waveform)."""
        if self._mode == "empty" or end_s <= start_s:
            return
        self._vb.setXRange(float(start_s), float(end_s), padding=0)

    def view_range(self) -> tuple[float, float]:
        """The currently visible time range in seconds."""
        x0, x1 = self._vb.viewRange()[0]
        return float(x0), float(x1)

    # -------------------------------------------------------------- live
    def update_live(self, data: np.ndarray, sample_rate: int) -> None:
        """Replace the rolling window content (cheap enough for 60 FPS).

        The first call (or a change of window length / sample rate) fits the time
        axis to the window; later calls only swap the curve data, so a zoom set by
        the user is kept.  ``data`` is read, not copied.
        """
        sr = int(sample_rate)
        if sr <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        y = finite_float32(data)
        n = y.size
        if n == 0:
            return

        if self._mode != "live":
            self.clear_overlay()
            self.clear_regions()
            self._mode = "live"
            self._y = self._pyr_min = self._pyr_max = None
            self._live_key = self._live_span = None
            self._hide_placeholder()
            self._vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

        buckets = self._target_buckets() if n > self._raw_limit() else 0
        key = (n, sr, buckets)
        if key != self._live_key:
            self._live_key = key
            if buckets:
                group = math.ceil(n / buckets)
                _, _, centres = _reduce_minmax(y, y, group)
                self._live_group = group
                self._live_x = np.repeat(centres / sr, 2)
            else:
                self._live_x = np.arange(n, dtype=np.float64) / sr
        if (n, sr) != self._live_span:
            self._live_span = (n, sr)
            duration = n / sr
            self._vb.setLimits(xMin=0.0, xMax=duration, minXRange=min(duration, 4.0 / sr))
            self._vb.setXRange(0.0, duration, padding=0)

        if buckets:
            mins, maxs, _ = _reduce_minmax(y, y, self._live_group)
            ys = np.empty(mins.size * 2, dtype=np.float32)
            ys[0::2], ys[1::2] = mins, maxs
        else:
            ys = y
        self._curve.setData(self._live_x, ys, skipFiniteCheck=True)

    # ------------------------------------------------------------- clear
    def clear(self, message: str = "No data") -> None:
        """Remove the waveform and show ``message`` as a placeholder."""
        self.clear_overlay()
        self.clear_regions()
        self._mode = "empty"
        self._y = self._pyr_min = self._pyr_max = None
        self._live_key = self._live_x = self._live_span = None
        self._curve.setData([], [])
        self._vb.setLimits(xMin=None, xMax=None, minXRange=None)
        self._vb.setXRange(0.0, 1.0, padding=0)
        self._show_placeholder(message)

    # -------------------------------------------------- zoom / export / menu
    def reset_zoom(self) -> None:
        """Fit the time axis to the whole displayed waveform."""
        if self._mode == "static" and self._y is not None:
            duration = self._y.size / self._sr
        elif self._mode == "live" and self._live_span is not None:
            duration = self._live_span[0] / self._live_span[1]
        else:
            return
        self._vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        self._vb.setXRange(0.0, duration, padding=0)
        if self._mode == "static":
            self._render_static()

    def export_png(self, path: str | Path | None = None) -> bool:
        """Save what the plot currently shows as a PNG.

        Args:
            path: Destination; if None a save dialog is shown.

        Returns:
            True if the file was written (False on cancel or failure).
        """
        if path is None:
            chosen, _ = QFileDialog.getSaveFileName(
                self, "Export PNG", "waveform.png", "PNG image (*.png)"
            )
            if not chosen:
                return False
            path = chosen
        target = Path(path)
        if target.suffix.lower() != ".png":
            target = target.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        return bool(self._plot.grab().save(str(target), "PNG"))

    def _build_context_menu(self) -> QMenu:
        menu = QMenu(self)
        reset = menu.addAction("Reset Zoom")
        reset.triggered.connect(self.reset_zoom)
        export = menu.addAction("Export PNG")
        export.triggered.connect(lambda: self.export_png())
        reset.setEnabled(self.has_data)
        export.setEnabled(self.has_data)
        return menu

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        self._build_context_menu().exec(event.globalPos())
        event.accept()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._mode == "static":
            self._render_static()  # bucket count follows the plot width


__all__: list[str] = ["WaveformWidget"]
