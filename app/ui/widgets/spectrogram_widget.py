"""The main analysis display: one widget that can show every analysis result.

* waveform-type plots (waveform, envelope, impulses) use the zoomable, decimating
  :class:`WaveformWidget`;
* line plots (FFT, PSD, autocorrelation, wavelet slice) and heat maps (STFT, CWT)
  use :class:`AnalysisPlot`.

Heat maps accept clicks; see :attr:`SpectrogramWidget.clicked`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtWidgets import QFileDialog, QLabel, QStackedLayout, QWidget

from app.ui.widgets._plot_base import (
    COLOR_AXIS,
    COLOR_PEAK,
    COLOR_SPECTRUM,
    COLOR_WAVEFORM,
    GRID_ALPHA,
    PlotWidgetBase,
    finite_float32,
    nearest_index,
)
from app.ui.widgets.waveform_widget import WaveformWidget

#: Heat maps are max-pooled along time down to this many columns for display.
MAX_IMAGE_COLUMNS = 4096
#: Dynamic range shown for heat maps, in dB below their maximum.
STFT_DYNAMIC_RANGE_DB = 80.0
CWT_DYNAMIC_RANGE_DB = 60.0
_LOG_ROWS_MIN, _LOG_ROWS_MAX = 128, 512
_COLORMAP = "viridis"
_CLICK_BUTTONS = {Qt.MouseButton.LeftButton: 1, Qt.MouseButton.RightButton: 2}


def format_frequency(hz: float) -> str:
    """``12500 -> "12.5k"``, ``440 -> "440"`` (for axis labels)."""
    return f"{hz / 1000.0:g}k" if abs(hz) >= 1000.0 else f"{hz:g}"


def _first_autocorrelation_peak(corr: np.ndarray) -> int | None:
    """Index of the first local maximum after the correlation first drops to zero.

    That peak is the signal's period.  (The maximum right after lag 0 is not: the
    correlation of a slow signal stays near 1 for many samples.)
    """
    values = np.asarray(corr, dtype=np.float64)
    below = np.flatnonzero(values <= 0.0)
    if below.size == 0:
        return None
    for i in range(int(below[0]) + 1, values.size - 1):
        if values[i] > values[i - 1] and values[i] >= values[i + 1]:
            return i
    return None


class _FreqAxis(pg.AxisItem):
    """Vertical axis that can show ``log10(frequency)`` data as Hz with 1-2-...-9 ticks."""

    def __init__(self, orientation: str, **kwargs: Any) -> None:
        super().__init__(orientation, **kwargs)
        self.log_mode = False

    def tickValues(self, minVal: float, maxVal: float, size: float) -> list[tuple[float, list[float]]]:
        if not self.log_mode:
            return super().tickValues(minVal, maxVal, size)
        first, last = math.floor(minVal) - 1, math.ceil(maxVal) + 1
        major = [float(k) for k in range(first, last + 1) if minVal <= k <= maxVal]
        minor = [
            k + math.log10(m)
            for k in range(first, last + 1)
            for m in range(2, 10)
            if minVal <= k + math.log10(m) <= maxVal
        ]
        return [(1.0, major), (0.1, minor)]

    def tickStrings(self, values: Sequence[float], scale: float, spacing: float) -> list[str]:
        if not self.log_mode:
            return super().tickStrings(values, scale, spacing)
        return [format_frequency(10.0 ** v) for v in values]

    def set_log_mode(self, enabled: bool) -> None:
        if enabled != self.log_mode:
            self.log_mode = enabled
            self.picture = None
            self.update()


@dataclass
class _SpectrumData:
    """What a heat map was drawn from, to answer slice queries at full resolution."""

    kind: str  # "stft" | "cwt"
    time: np.ndarray
    freqs_hz: np.ndarray
    column: Callable[[int], np.ndarray]  # values over frequency at one time index
    row: Callable[[int], np.ndarray]  # values over time at one frequency index
    y_label: str


def _reduce_columns(
    count: int,
    max_columns: int,
    block: Callable[[int, int], np.ndarray],
) -> np.ndarray:
    """Max-pool ``count`` columns down to at most ``max_columns``.

    ``block(a, b)`` returns the display values (rows x columns) of columns ``[a, b)``,
    so large inputs are processed in pieces instead of being converted as a whole.
    """
    step = max(1, math.ceil(count / max_columns))
    out_cols = math.ceil(count / step)
    pieces: list[np.ndarray] = []
    chunk_cols = max(1, 512)  # output columns per piece
    for first in range(0, out_cols, chunk_cols):
        a = first * step
        b = min(count, (first + chunk_cols) * step)
        values = block(a, b)
        n = values.shape[1]
        full = n // step
        parts = []
        if full:
            parts.append(values[:, : full * step].reshape(values.shape[0], full, step).max(axis=2))
        if n - full * step:
            parts.append(values[:, full * step :].max(axis=1, keepdims=True))
        pieces.append(np.concatenate(parts, axis=1))
    return np.concatenate(pieces, axis=1)


class AnalysisPlot(PlotWidgetBase):
    """Line plots and heat maps with annotations, a note and click reporting.

    Signals:
        ``clicked(x, frequency_hz, button)`` - a click inside a heat map; ``x`` is
        the time in seconds and ``button`` is 1 (left) or 2 (right).
    """

    clicked = Signal(float, float, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        axis = _FreqAxis("left")
        super().__init__("", "", parent, axis_items={"left": axis})
        item = self._plot.getPlotItem()
        self._item = item
        self._vb = item.vb
        self._freq_axis = axis
        axis.setPen(pg.mkPen(COLOR_AXIS))
        axis.setTextPen(pg.mkPen(COLOR_AXIS))

        self._curve = item.plot(pen=pg.mkPen(COLOR_WAVEFORM, width=1.5), antialias=True)
        self._curve.setDownsampling(auto=True, method="peak")
        self._image = pg.ImageItem()
        self._image.setColorMap(pg.colormap.get(_COLORMAP))
        item.addItem(self._image)
        self._image.hide()
        self._colorbar: pg.ColorBarItem | None = None
        self._extras: list[Any] = []

        self._note = QLabel(self)
        self._note.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._note.setStyleSheet("background: transparent; color: #475569; font-size: 11px; padding: 6px;")
        self._note.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._note.raise_()
        layout = self.layout()
        if isinstance(layout, QStackedLayout):
            layout.addWidget(self._note)

        self._mode = "empty"  # "empty" | "xy" | "image"
        self._log_freq = False
        self._xy_data: tuple[np.ndarray, np.ndarray, bool] | None = None
        self._image_info: tuple[tuple[float, float], tuple[float, float], str] | None = None
        self._x_label = ""
        self._y_label = ""
        self._spectrum: _SpectrumData | None = None
        self._plot.scene().sigMouseClicked.connect(self._on_scene_clicked)
        self.clear("Нет данных")

    # ------------------------------------------------------------- state
    @property
    def mode(self) -> str:
        """``"empty"``, ``"xy"`` (line plot) or ``"image"`` (heat map)."""
        return self._mode

    @property
    def has_spectrum(self) -> bool:
        """True while a heat map with slice data is displayed."""
        return self._mode == "image" and self._spectrum is not None

    @property
    def image_item(self) -> pg.ImageItem:
        return self._image

    @property
    def curve(self) -> pg.PlotDataItem:
        return self._curve

    def _clear_extras(self) -> None:
        for extra in self._extras:
            self._item.removeItem(extra)
        self._extras.clear()

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        is_image = mode == "image"
        self._image.setVisible(is_image)
        self._curve.setVisible(mode == "xy")
        if self._colorbar is not None:
            self._colorbar.setVisible(is_image)
        self._item.showGrid(x=not is_image, y=not is_image, alpha=GRID_ALPHA)
        if not is_image:
            self._freq_axis.set_log_mode(False)
            self._log_freq = False
            self._spectrum = None

    def _decorate(self, title: str, x_label: str, y_label: str, note: str) -> None:
        self._x_label, self._y_label = x_label, y_label
        self._item.setTitle(title, color=COLOR_AXIS)
        self._item.setLabel("bottom", x_label, color=COLOR_AXIS)
        self._item.setLabel("left", y_label, color=COLOR_AXIS)
        self._note.setText(note)
        self._note.setVisible(bool(note))
        self._hide_placeholder()

    # ------------------------------------------------------------ drawing
    def clear(self, message: str = "Нет данных") -> None:
        """Remove everything and show ``message``."""
        self._clear_extras()
        self._curve.setData([], [])
        self._set_mode("empty")
        self._item.setTitle(None)
        self._note.hide()
        self._show_placeholder(message)

    def show_xy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        title: str,
        x_label: str,
        y_label: str,
        color: str = COLOR_WAVEFORM,
        annotations: Sequence[dict[str, Any]] = (),
        note: str = "",
        log_x: bool = False,
    ) -> None:
        """Draw a line plot.

        Args:
            annotations: Dicts with ``x``, ``y``, ``label`` and optional ``color``.
            note: Text shown in the top-left corner of the plot.
            log_x: Logarithmic x axis (only positive x values are drawn).
        """
        xs = np.asarray(x, dtype=np.float64).ravel()
        ys = np.asarray(y, dtype=np.float64).ravel()
        if xs.size != ys.size:
            raise ValueError(f"x ({xs.size}) and y ({ys.size}) must have the same length")
        if xs.size == 0:
            self.clear("Нет данных")
            return

        self._clear_extras()
        self._set_mode("xy")
        self._decorate(title, x_label, y_label, note)
        self._xy_data = (xs, ys, log_x)
        self._item.setLogMode(x=log_x, y=False)
        self._curve.setPen(pg.mkPen(color, width=1.5))
        self._curve.setData(xs, ys)
        # +-inf, not None: pyqtgraph compares the limits numerically in log mode
        self._vb.setLimits(xMin=-np.inf, xMax=np.inf, yMin=-np.inf, yMax=np.inf)
        self._vb.autoRange()

        for annotation in annotations:
            tint = annotation.get("color", COLOR_PEAK)
            marker = pg.ScatterPlotItem([annotation["x"]], [annotation["y"]], size=9, pen=pg.mkPen(tint), brush=pg.mkBrush(tint))
            label = pg.TextItem(str(annotation.get("label", "")), color=tint, anchor=(0.0, 1.0))
            label.setPos(annotation["x"], annotation["y"])
            self._item.addItem(marker, ignoreBounds=True)
            self._item.addItem(label, ignoreBounds=True)
            self._extras += [marker, label]

    def show_image(
        self,
        values: np.ndarray,
        *,
        t_range: tuple[float, float],
        f_range: tuple[float, float],
        log_freq: bool,
        levels: tuple[float, float],
        title: str,
        y_label: str,
        colorbar_label: str,
        note: str = "",
        spectrum: _SpectrumData | None = None,
    ) -> None:
        """Draw a heat map.

        Args:
            values: ``(n_freqs, n_times)`` display values, frequency ascending.  Rows
                must already be spaced uniformly on the chosen (linear or log) axis.
            t_range: Time covered by the columns, seconds.
            f_range: Frequency covered by the rows, Hz.
            log_freq: Put frequency on a logarithmic axis.
            levels: Colour scale limits.
        """
        z = np.ascontiguousarray(values, dtype=np.float32)
        if z.ndim != 2 or z.size == 0:
            self.clear("Нет данных")
            return

        self._clear_extras()
        self._item.setLogMode(x=False, y=False)
        self._set_mode("image")
        self._decorate(title, "Time (s)", y_label, note)
        self._log_freq = log_freq
        self._freq_axis.set_log_mode(log_freq)
        self._spectrum = spectrum

        t0, t1 = t_range
        f0, f1 = f_range
        y0, y1 = (math.log10(f0), math.log10(f1)) if log_freq else (f0, f1)
        self._image_info = ((t0, t1), (y0, y1), colorbar_label)
        self._image.setImage(z.T, autoLevels=False, levels=levels)
        self._image.setRect(QRectF(t0, y0, t1 - t0, y1 - y0))

        if self._colorbar is None:
            self._colorbar = pg.ColorBarItem(
                values=levels, colorMap=pg.colormap.get(_COLORMAP), interactive=False, width=16
            )
            self._colorbar.setImageItem(self._image, insert_in=self._item)
        self._colorbar.setLevels(levels)
        self._colorbar.setVisible(True)
        colorbar_axis = getattr(self._colorbar, "axis", None)
        if colorbar_axis is not None:
            colorbar_axis.setLabel(colorbar_label)
        self._vb.setLimits(xMin=t0, xMax=t1, yMin=y0, yMax=y1)
        self._vb.setRange(xRange=(t0, t1), yRange=(y0, y1), padding=0)

    # --------------------------------------------------------------- hover
    def _hover_crosshair(self) -> bool:
        return self._mode == "xy"

    def _hover_text(self, x: float, y: float) -> str | None:
        if self._mode == "xy" and self._xy_data is not None:
            xs, ys, log_x = self._xy_data
            if log_x:
                valid = xs > 0
                xs, ys = xs[valid], ys[valid]
                if xs.size == 0:
                    return None
                x = 10.0**x
            order = np.argsort(xs)
            xs, ys = xs[order], ys[order]
            i = nearest_index(xs, x)
            return f"{self._x_label}: {xs[i]:.6g}\n{self._y_label}: {ys[i]:.6g}"
        if self._mode == "image" and self._image_info is not None:
            (t0, t1), (y0, y1), value_label = self._image_info
            data = self._image.image  # (n_times, n_rows)
            if data is None or not (t0 <= x <= t1 and y0 <= y <= y1):
                return None
            ti = min(int((x - t0) / (t1 - t0) * data.shape[0]), data.shape[0] - 1)
            fi = min(int((y - y0) / (y1 - y0) * data.shape[1]), data.shape[1] - 1)
            frequency = 10.0**y if self._log_freq else y
            return (
                f"Время: {x:.4f} с\nЧастота: {frequency:.1f} Гц"
                f"\n{value_label}: {data[ti, fi]:.1f}"
            )
        return None

    # ----------------------------------------------------------- clicking
    def _on_scene_clicked(self, event: Any) -> None:
        if self._mode != "image":
            return
        button = _CLICK_BUTTONS.get(event.button(), 0)
        position = event.scenePos()
        if not button or not self._vb.sceneBoundingRect().contains(position):
            return
        point = self._vb.mapSceneToView(position)
        frequency = 10.0 ** point.y() if self._log_freq else point.y()
        self.clicked.emit(float(point.x()), float(frequency), button)
        event.accept()

    # ------------------------------------------------------------- slices
    def slice_at_time(self, t_s: float) -> tuple[np.ndarray, np.ndarray, float, str] | None:
        """Values over frequency at the column nearest ``t_s``.

        Returns:
            ``(freqs_hz ascending, values, actual_time_s, y_label)``, or None if no
            heat map is displayed.
        """
        data = self._spectrum
        if data is None:
            return None
        ti = int(np.argmin(np.abs(data.time - t_s)))
        order = np.argsort(data.freqs_hz)
        return data.freqs_hz[order], np.asarray(data.column(ti))[order], float(data.time[ti]), data.y_label

    def slice_at_frequency(self, f_hz: float) -> tuple[np.ndarray, np.ndarray, float, str] | None:
        """Values over time at the row nearest ``f_hz``.

        Returns:
            ``(time_s, values, actual_frequency_hz, y_label)``, or None.
        """
        data = self._spectrum
        if data is None:
            return None
        fi = int(np.argmin(np.abs(data.freqs_hz - f_hz)))
        return data.time, np.asarray(data.row(fi)), float(data.freqs_hz[fi]), data.y_label


class SpectrogramWidget(QWidget):
    """Displays any analysis result; see the module docstring.

    Use :meth:`show_plot` with the ``(plot_type, payload)`` pair emitted by
    ``AnalysisViewModel.plot_ready``.  ``clicked(time_s, frequency_hz, button)`` is
    emitted for clicks inside a heat map (button 1 = left, 2 = right), and
    ``selection_edited(start_s, end_s)`` when the user drags the segment selection of
    the waveform.
    """

    clicked = Signal(float, float, int)
    selection_edited = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._waveform = WaveformWidget()
        self._waveform.selection_edited.connect(self.selection_edited)
        self._plot = AnalysisPlot()
        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(self._waveform)
        self._stack.addWidget(self._plot)
        self._kind = "empty"
        self._title = ""
        self._plot.clicked.connect(self.clicked)
        self.clear("Загрузите или запишите сигнал, затем выберите анализ")

    # ------------------------------------------------------------- access
    @property
    def kind(self) -> str:
        """The plot type currently shown (``"empty"`` if nothing is)."""
        return self._kind

    @property
    def title(self) -> str:
        """Human-readable title of the current plot."""
        return self._title

    @property
    def waveform_widget(self) -> WaveformWidget:
        return self._waveform

    @property
    def analysis_plot(self) -> AnalysisPlot:
        return self._plot

    @property
    def placeholder_text(self) -> str:
        """The message shown instead of a plot ("" when a plot is shown)."""
        page = self._stack.currentWidget()
        return self._waveform.placeholder_text if page is self._waveform else self._plot.placeholder_text

    def clear(self, message: str = "Нет данных") -> None:
        """Remove the plot and show ``message``."""
        self._kind = "empty"
        self._title = ""
        self._plot.clear(message)
        self._stack.setCurrentWidget(self._plot)

    # ---------------------------------------------------------- dispatch
    def show_plot(self, plot_type: str, payload: dict[str, Any]) -> None:
        """Show an analysis result.

        Raises:
            ValueError: unknown ``plot_type``.
        """
        handlers: dict[str, Callable[[dict[str, Any]], None]] = {
            "waveform": self._show_waveform,
            "envelope": self._show_envelope,
            "impulses": self._show_impulses,
            "fft": self._show_fft,
            "psd": self._show_psd,
            "autocorrelation": self._show_autocorrelation,
            "wavelet_slice": self._show_wavelet_slice,
            "stft": self._show_stft,
            "cwt": self._show_cwt,
        }
        handler = handlers.get(plot_type)
        if handler is None:
            raise ValueError(f"Неизвестный тип графика {plot_type!r}; допустимые: {sorted(handlers)}")
        handler(payload)
        self._kind = plot_type

    def set_selection(self, start_s: float, end_s: float) -> None:
        """Move the segment selection of the plain waveform (no-op for any other plot)."""
        if self._kind == "waveform":
            self._waveform.set_selection(start_s, end_s)

    def clear_selection(self) -> None:
        """Remove the segment selection from the waveform."""
        self._waveform.clear_selection()

    def set_time_range(self, start_s: float, end_s: float) -> None:
        """Zoom a waveform-type plot to a time range."""
        if self._stack.currentWidget() is self._waveform:
            self._waveform.set_time_range(start_s, end_s)

    def slice_at_time(self, t_s: float) -> tuple[np.ndarray, np.ndarray, float, str] | None:
        """See :meth:`AnalysisPlot.slice_at_time`."""
        return self._plot.slice_at_time(t_s)

    def slice_at_frequency(self, f_hz: float) -> tuple[np.ndarray, np.ndarray, float, str] | None:
        """See :meth:`AnalysisPlot.slice_at_frequency`."""
        return self._plot.slice_at_frequency(f_hz)

    def export_png(self, path: str | Path | None = None) -> bool:
        """Save the current plot as a PNG (file dialog if ``path`` is None)."""
        if self._stack.currentWidget() is self._waveform:
            return self._waveform.export_png(path)
        if path is None:
            chosen, _ = QFileDialog.getSaveFileName(self, "Экспорт в PNG", "analysis.png", "Изображение PNG (*.png)")
            if not chosen:
                return False
            path = chosen
        target = Path(path).with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        return bool(self._plot.plot_widget.grab().save(str(target), "PNG"))

    # --------------------------------------------------- waveform-type plots
    def _show_waveform(self, payload: dict[str, Any]) -> None:
        self._waveform.set_data(payload["data"], payload["sample_rate"])
        selection = payload.get("selection")
        if selection is not None:
            self._waveform.set_selection(*selection)
        self._stack.setCurrentWidget(self._waveform)
        self._title = "Сигнал (отфильтрованный)" if payload.get("filtered") else "Сигнал"

    def _show_envelope(self, payload: dict[str, Any]) -> None:
        self._waveform.set_data(payload["signal"], payload["sample_rate"])
        self._waveform.set_overlay(payload["data"], COLOR_PEAK)
        self._stack.setCurrentWidget(self._waveform)
        self._title = "Огибающая"

    def _show_impulses(self, payload: dict[str, Any]) -> None:
        impulses = payload["impulses"]
        self._waveform.set_data(payload["data"], payload["sample_rate"])
        self._waveform.set_regions([(i.start_s, i.end_s) for i in impulses])
        self._stack.setCurrentWidget(self._waveform)
        self._title = f"Импульсы ({len(impulses)})"

    # -------------------------------------------------------- line plots
    def _xy(self, title: str, *args: Any, **kwargs: Any) -> None:
        self._plot.show_xy(*args, title=title, **kwargs)
        self._stack.setCurrentWidget(self._plot)
        self._title = title

    def _show_fft(self, payload: dict[str, Any]) -> None:
        freqs, mags = payload["freqs_hz"], payload["magnitudes"]
        peak_hz = payload.get("peak_hz", 0.0)
        annotations = (
            [{"x": peak_hz, "y": payload["peak_magnitude"], "label": f"{peak_hz:.1f} Гц"}] if peak_hz else []
        )
        self._xy(
            "Спектр (БПФ)", freqs, mags, x_label="Частота (Гц)", y_label="Магнитуда",
            color=COLOR_WAVEFORM, annotations=annotations,
            note=f"Пик: {peak_hz:.1f} Гц" if peak_hz else "",
        )

    def _show_psd(self, payload: dict[str, Any]) -> None:
        freqs = np.asarray(payload["freqs_hz"], dtype=np.float64)
        psd_db = 10.0 * np.log10(np.maximum(np.asarray(payload["psd"], dtype=np.float64), 1e-20))
        annotations = []
        note = ""
        if psd_db.size:
            peak = int(np.argmax(psd_db))
            annotations = [{"x": float(freqs[peak]), "y": float(psd_db[peak]), "label": f"{freqs[peak]:.1f} Гц"}]
            note = f"Пик: {freqs[peak]:.1f} Гц"
        self._xy(
            "Спектральная плотность мощности (Уэлч)", freqs, psd_db, x_label="Частота (Гц)",
            y_label="СПМ (дБ/Гц)", color=COLOR_SPECTRUM, annotations=annotations, note=note,
        )

    def _show_autocorrelation(self, payload: dict[str, Any]) -> None:
        lags, corr = np.asarray(payload["lags_ms"]), np.asarray(payload["correlation"])
        annotations: list[dict[str, Any]] = []
        note = "Пик около 0 мс — ожидаемо"
        idx = _first_autocorrelation_peak(corr)
        if idx is not None:
            lag = float(lags[idx])
            annotations.append({"x": lag, "y": float(corr[idx]), "label": f"период {lag:.2f} мс"})
            if lag > 0:
                note += f"\nОценка частоты: {1000.0 / lag:.1f} Гц"
        self._xy(
            "Автокорреляция", lags, corr, x_label="Задержка (мс)", y_label="Коэффициент",
            color=COLOR_SPECTRUM, annotations=annotations, note=note,
        )

    def _show_wavelet_slice(self, payload: dict[str, Any]) -> None:
        t_rel = np.asarray(payload["time_rel_s"], dtype=np.float64)
        half = max(abs(float(t_rel[0])), abs(float(t_rel[-1]))) if t_rel.size else 0.0
        if half >= 0.1:
            scale, unit = 1.0, "с"
        elif half >= 1e-3:
            scale, unit = 1e3, "мс"
        else:
            scale, unit = 1e6, "мкс"
        frequency = float(payload["frequency_hz"])
        self._xy(
            f"Срез вейвлета: Re(Wx) при {frequency:.1f} Гц", t_rel * scale, payload["values"],
            x_label=f"Время относительно центра ({unit})", y_label="Нормированная амплитуда",
            note=f"f = {frequency:.1f} Гц",
        )

    # ------------------------------------------------------------ heat maps
    def _show_stft(self, payload: dict[str, Any]) -> None:
        time = np.asarray(payload["time"], dtype=np.float64)
        freqs = np.asarray(payload["frequencies_hz"], dtype=np.float64)
        db = np.asarray(payload["magnitude_db"])
        if db.size == 0:
            self._plot.clear("Нет данных")
            self._stack.setCurrentWidget(self._plot)
            return
        shown = _reduce_columns(db.shape[1], MAX_IMAGE_COLUMNS, lambda a, b: db[:, a:b].astype(np.float32))
        vmax = float(np.max(db))
        vmin = max(vmax - STFT_DYNAMIC_RANGE_DB, float(np.min(db)))
        dt = float(np.median(np.diff(time))) if time.size > 1 else 1.0
        spectrum = _SpectrumData("stft", time, freqs, lambda i: db[:, i], lambda i: db[i, :], "Магнитуда (дБ)")
        self._plot.show_image(
            shown,
            t_range=(float(time[0]) - dt / 2, float(time[-1]) + dt / 2),
            f_range=(float(freqs[0]), float(freqs[-1])),
            log_freq=False,
            levels=(vmin, vmax),
            title="Спектрограмма STFT",
            y_label="Частота (Гц)",
            colorbar_label="дБ",
            note="Левый щелчок: срез по времени на этой частоте\nПравый щелчок: срез спектра в этот момент",
            spectrum=spectrum,
        )
        self._stack.setCurrentWidget(self._plot)
        self._title = "Спектрограмма STFT"

    def _show_cwt(self, payload: dict[str, Any]) -> None:
        result = payload["result"]
        coefficients = result.coefficients
        freqs = np.asarray(result.frequencies_hz, dtype=np.float64)
        time = np.asarray(result.time, dtype=np.float64)
        if coefficients.size == 0:
            self._plot.clear("Нет данных")
            self._stack.setCurrentWidget(self._plot)
            return

        order = np.argsort(freqs)  # scales are ordered by decreasing frequency
        amplitude = _reduce_columns(
            coefficients.shape[1],
            MAX_IMAGE_COLUMNS,
            lambda a, b: np.abs(coefficients[order, a:b]).astype(np.float32),
        )
        peak = float(np.max(amplitude))
        db = 20.0 * np.log10(np.maximum(amplitude / peak, 1e-6)) if peak > 0 else np.full_like(amplitude, -120.0)

        f_sorted = freqs[order]
        n_rows = int(min(_LOG_ROWS_MAX, max(_LOG_ROWS_MIN, 2 * f_sorted.size)))
        targets = np.geomspace(f_sorted[0], f_sorted[-1], n_rows) if f_sorted[-1] > f_sorted[0] else f_sorted[:1]
        right = np.clip(np.searchsorted(f_sorted, targets), 1, max(1, f_sorted.size - 1))
        left = right - 1
        nearest = np.where(np.abs(f_sorted[right] - targets) < np.abs(targets - f_sorted[left]), right, left)
        shown = db[nearest] if f_sorted.size > 1 else db

        dt = float(np.median(np.diff(time))) if time.size > 1 else 1.0
        spectrum = _SpectrumData(
            "cwt", time, freqs,
            lambda i: np.abs(coefficients[:, i]).astype(np.float64),
            lambda i: np.abs(coefficients[i, :]).astype(np.float64),
            "Амплитуда |Wx|",
        )
        note = "Левый щелчок: срез вейвлета на этой частоте\nПравый щелчок: срез спектра в этот момент"
        if payload.get("max_seconds"):
            note = f"Ограничено первыми {payload['max_seconds']:.2f} с\n" + note
        self._plot.show_image(
            shown,
            t_range=(float(time[0]) - dt / 2, float(time[-1]) + dt / 2),
            f_range=(float(f_sorted[0]), float(f_sorted[-1] if f_sorted[-1] > f_sorted[0] else f_sorted[0] * 1.01)),
            log_freq=True,
            levels=(-CWT_DYNAMIC_RANGE_DB, 0.0),
            title="CWT Морле",
            y_label="Частота (Гц)",
            colorbar_label="дБ отн. макс.",
            note=note,
            spectrum=spectrum,
        )
        self._stack.setCurrentWidget(self._plot)
        self._title = "CWT Морле"


__all__: list[str] = ["SpectrogramWidget", "AnalysisPlot", "format_frequency"]
