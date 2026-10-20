"""Hover read-out and volt calibration of the plot widgets."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF
from PySide6.QtWidgets import QApplication, QWidget

from app.ui.widgets._plot_base import PlotWidgetBase, format_volts, nearest_index
from app.ui.widgets.spectrogram_widget import AnalysisPlot, SpectrogramWidget
from app.ui.widgets.spectrum_widget import SpectrumWidget
from app.ui.widgets.waveform_widget import WaveformWidget

FS = 48000


def shown(widget: QWidget, qapp: QApplication) -> QWidget:
    widget.resize(600, 400)
    widget.show()
    qapp.processEvents()
    return widget


def hover_at(widget: PlotWidgetBase, x: float, y: float) -> str:
    """Move the 'mouse' to the view coordinates ``(x, y)`` and return the read-out."""
    vb = widget.plot_widget.getPlotItem().vb
    widget._on_hover_moved(vb.mapViewToScene(QPointF(x, y)))
    return widget.hover_text


def hover_outside(widget: PlotWidgetBase) -> str:
    widget._on_hover_moved(QPointF(-50.0, -50.0))
    return widget.hover_text


# ------------------------------------------------------------------- helpers
def test_nearest_index_picks_the_closest_element() -> None:
    xs = np.array([0.0, 1.0, 2.0, 10.0])

    assert nearest_index(xs, -5.0) == 0
    assert nearest_index(xs, 0.4) == 0
    assert nearest_index(xs, 0.6) == 1
    assert nearest_index(xs, 5.0) == 2
    assert nearest_index(xs, 7.0) == 3
    assert nearest_index(xs, 99.0) == 3


@pytest.mark.parametrize(
    ("volts", "text"),
    [(2.5, "2.5 В"), (0.0123, "12.3 мВ"), (-0.004, "-4 мВ"), (5e-5, "50 мкВ"), (0.0, "0 мкВ")],
)
def test_format_volts_uses_si_prefixes(volts: float, text: str) -> None:
    assert format_volts(volts) == text


# ------------------------------------------------------------------ waveform
@pytest.fixture
def waveform(qapp: QApplication) -> WaveformWidget:
    widget = WaveformWidget()
    shown(widget, qapp)
    ramp = np.linspace(0.0, 0.5, FS, dtype=np.float32)  # value = 0.5 * t
    widget.set_data(ramp, FS)
    qapp.processEvents()
    return widget


def test_waveform_hover_reads_time_and_amplitude(waveform: WaveformWidget) -> None:
    text = hover_at(waveform, 0.5, 0.0)

    assert "Время: 0.5000" in text
    assert "Амплитуда: +0.25" in text


def test_waveform_hover_is_hidden_outside_the_widget(waveform: WaveformWidget) -> None:
    hover_at(waveform, 0.5, 0.0)

    assert hover_outside(waveform) == ""


def test_waveform_hover_is_hidden_outside_the_data(waveform: WaveformWidget) -> None:
    assert hover_at(waveform, 50.0, 0.0) == ""


def test_leaving_the_widget_hides_the_hover(waveform: WaveformWidget) -> None:
    hover_at(waveform, 0.5, 0.0)
    waveform.leaveEvent(QEvent(QEvent.Type.Leave))

    assert waveform.hover_text == ""


def test_waveform_hover_shows_the_envelope_when_present(waveform: WaveformWidget) -> None:
    waveform.set_overlay(np.full(FS, 0.4, dtype=np.float32))

    assert "Огибающая: +0.40" in hover_at(waveform, 0.5, 0.0)


def test_waveform_hover_without_data_shows_nothing(qapp: QApplication) -> None:
    widget = WaveformWidget()
    shown(widget, qapp)

    assert hover_at(widget, 0.5, 0.0) == ""


def test_waveform_shows_volts_when_calibrated(waveform: WaveformWidget) -> None:
    waveform.set_calibration(2.0)

    assert "Амплитуда: 500 мВ" in hover_at(waveform, 0.5, 0.0)  # 0.25 * 2 V = 0.5 V


def test_waveform_axis_follows_the_calibration(waveform: WaveformWidget) -> None:
    axis = waveform.plot_widget.getPlotItem().getAxis("left")

    waveform.set_calibration(2.0)
    assert axis.scale == 2.0
    assert axis.labelUnits == "В"

    waveform.set_calibration(0.0)
    assert axis.scale == 1.0
    assert not axis.labelUnits
    assert waveform.volts_per_fs == 0.0


def test_negative_calibration_counts_as_not_calibrated(waveform: WaveformWidget) -> None:
    waveform.set_calibration(-3.0)

    assert waveform.volts_per_fs == 0.0


def test_live_waveform_hover_uses_the_displayed_points(qapp: QApplication) -> None:
    widget = WaveformWidget()
    shown(widget, qapp)
    widget.update_live(np.full(4800, 0.3, dtype=np.float32), FS)
    qapp.processEvents()

    assert "Амплитуда: +0.30" in hover_at(widget, 0.05, 0.3)


# ------------------------------------------------------------------ spectrum
@pytest.fixture
def spectrum(qapp: QApplication) -> SpectrumWidget:
    widget = SpectrumWidget()
    shown(widget, qapp)
    freqs = np.linspace(0.0, 24000.0, 25)  # 0, 1, 2 ... 24 kHz
    mags = np.arange(25, dtype=np.float32) / 100.0
    widget.update_spectrum(freqs, mags, 24000.0)
    return widget


def test_spectrum_hover_reads_the_nearest_bin(spectrum: SpectrumWidget) -> None:
    text = hover_at(spectrum, 10.2, 0.0)

    assert "Частота: 10.000 кГц" in text
    assert "Магнитуда: 0.1" in text


def test_spectrum_hover_in_volts_when_calibrated(spectrum: SpectrumWidget) -> None:
    spectrum.set_calibration(5.0)

    assert "Магнитуда: 500 мВ" in hover_at(spectrum, 10.0, 0.0)  # 0.10 * 5 V


def test_spectrum_without_data_has_no_hover(qapp: QApplication) -> None:
    widget = SpectrumWidget()
    shown(widget, qapp)

    assert hover_at(widget, 0.5, 0.5) == ""


# ------------------------------------------------------------- analysis plot
@pytest.fixture
def analysis(qapp: QApplication) -> AnalysisPlot:
    plot = AnalysisPlot()
    shown(plot, qapp)
    return plot


def show_cells(plot: AnalysisPlot, *, log_freq: bool = False) -> None:
    """A 3 x 4 heat map (values 0..11, rows = frequency, columns = time)."""
    f_range = (100.0, 10000.0) if log_freq else (0.0, 3.0)
    plot.show_image(
        np.arange(12, dtype=np.float32).reshape(3, 4), t_range=(0.0, 4.0), f_range=f_range,
        log_freq=log_freq, levels=(0.0, 11.0), title="h", y_label="Частота (Гц)", colorbar_label="дБ",
    )


def test_line_plot_hover_uses_the_axis_labels(analysis: AnalysisPlot) -> None:
    analysis.show_xy(
        np.array([0.0, 1.0, 2.0]), np.array([5.0, 6.0, 7.0]),
        title="t", x_label="Частота (Гц)", y_label="Магнитуда",
    )

    assert hover_at(analysis, 1.1, 6.0) == "Частота (Гц): 1\nМагнитуда: 6"


def test_log_axis_hover_reports_the_real_x(analysis: AnalysisPlot) -> None:
    analysis.show_xy(
        np.array([1.0, 10.0, 100.0]), np.array([1.0, 2.0, 3.0]),
        title="t", x_label="Частота (Гц)", y_label="Магнитуда", log_x=True,
    )

    assert hover_at(analysis, 1.0, 2.0) == "Частота (Гц): 10\nМагнитуда: 2"  # view x is log10(10)


def test_line_plot_has_a_crosshair_and_heat_map_does_not(analysis: AnalysisPlot) -> None:
    analysis.show_xy(np.array([0.0, 1.0]), np.array([0.0, 1.0]), title="t", x_label="x", y_label="y")
    hover_at(analysis, 0.5, 0.5)
    assert analysis._hover_line.isVisible()

    show_cells(analysis)
    hover_at(analysis, 0.5, 0.5)
    assert not analysis._hover_line.isVisible()


def test_heat_map_hover_reads_the_cell_under_the_cursor(analysis: AnalysisPlot) -> None:
    show_cells(analysis)

    # column 2, row 1 -> value 6
    assert hover_at(analysis, 2.5, 1.5) == "Время: 2.5000 с\nЧастота: 1.5 Гц\nдБ: 6.0"


def test_log_heat_map_hover_reports_hertz_not_log10(analysis: AnalysisPlot) -> None:
    show_cells(analysis, log_freq=True)

    assert "Частота: 1000.0 Гц" in hover_at(analysis, 0.5, 3.0)  # view y is log10(1000)


def test_heat_map_hover_is_hidden_outside_the_image(analysis: AnalysisPlot) -> None:
    show_cells(analysis)

    assert hover_at(analysis, 9.0, 1.0) == ""


def test_cleared_analysis_plot_has_no_hover(analysis: AnalysisPlot) -> None:
    analysis.clear("Нет данных")

    assert hover_at(analysis, 0.5, 0.5) == ""


# ------------------------------------------------------- analysis container
def fft_payload() -> dict[str, object]:
    return {
        "freqs_hz": np.array([0.0, 1000.0, 2000.0]),
        "magnitudes": np.array([0.1, 0.4, 0.2]),
        "peak_hz": 1000.0,
        "peak_magnitude": 0.4,
    }


def test_fft_is_drawn_in_volts_when_calibrated(qapp: QApplication) -> None:
    display = SpectrogramWidget()
    display.set_calibration(2.0)

    display.show_plot("fft", fft_payload())

    plot = display.analysis_plot
    _, ys, _ = plot._xy_data
    np.testing.assert_allclose(ys, [0.2, 0.8, 0.4])
    assert plot._y_label == "Магнитуда (В)"


def test_fft_stays_relative_without_calibration(qapp: QApplication) -> None:
    display = SpectrogramWidget()

    display.show_plot("fft", fft_payload())

    plot = display.analysis_plot
    _, ys, _ = plot._xy_data
    np.testing.assert_allclose(ys, [0.1, 0.4, 0.2])
    assert plot._y_label == "Магнитуда"


def test_container_passes_the_calibration_to_the_waveform(qapp: QApplication) -> None:
    display = SpectrogramWidget()

    display.set_calibration(1.5)

    assert display.waveform_widget.volts_per_fs == 1.5
