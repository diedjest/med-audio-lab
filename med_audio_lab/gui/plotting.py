from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt

BG = "#f6f8fb"
INK = "#1f2937"
GRID = "#d8dee9"
BLUE = "#2563eb"
GREEN = "#0f766e"
RED = "#dc2626"
AMBER = "#d97706"
VIOLET = "#7c3aed"


def _morlet_ridge_track(
    freqs_hz: np.ndarray,
    spec: np.ndarray,
    *,
    tone_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Гребень Morlet: при известной частоте записи — прямая линия на tone_hz.
    """
    spec = np.asarray(spec, dtype=np.float64)
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    n_cols = spec.shape[1]
    cols = np.arange(n_cols, dtype=np.intp)

    if tone_hz is not None and tone_hz > 0:
        ni = int(np.argmin(np.abs(freqs_hz - float(tone_hz))))
        f_line = np.full(n_cols, float(tone_hz), dtype=np.float64)
        return f_line, spec[ni, cols].astype(np.float64)

    peak_idx = np.argmax(spec, axis=0)
    return freqs_hz[peak_idx].astype(np.float64), spec[peak_idx, cols].astype(np.float64)


def _morlet_ridge_line_collection(
    ax,
    t: np.ndarray,
    freqs_hz: np.ndarray,
    spec: np.ndarray,
    *,
    tone_hz: float | None = None,
    linewidth: float = 0.75,
):
    """Тонкая линия гребня Morlet (без «плитки» pcolormesh по полосам)."""
    t = np.asarray(t, dtype=np.float64)
    spec = np.asarray(spec, dtype=np.float64)
    f_line, amp = _morlet_ridge_track(freqs_hz, spec, tone_hz=tone_hz)
    vmax = float(np.max(amp)) if amp.size and float(np.max(amp)) > 0 else 1.0
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("viridis")
    ax.set_facecolor(cmap(0.0))

    if t.size < 2:
        return ax.scatter(
            t,
            f_line,
            c=amp,
            cmap=cmap,
            norm=norm,
            s=2.0,
            linewidths=0,
            zorder=3,
        )

    segments = np.stack(
        [np.column_stack([t[:-1], f_line[:-1]]), np.column_stack([t[1:], f_line[1:]])],
        axis=1,
    )
    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidths=linewidth,
        capstyle="round",
        zorder=3,
    )
    lc.set_array(0.5 * (amp[:-1] + amp[1:]))
    ax.add_collection(lc)
    return lc


def _default_xlim_us_morlet_slice(t: np.ndarray, f_hz: float | None) -> tuple[float, float]:
    """Окно по оси времени в мкс для среза вейвлета."""
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return (-4.0, 4.0)
    by_samples = 40.0 * dt * 1e6
    by_period = (1.5 * 1e6 / float(f_hz)) if (f_hz is not None and f_hz > 0) else 0.0
    half_us = max(4.0, by_samples, by_period)
    return (-half_us, half_us)


def plot_longitudinal_slice_morlet_re(
    t: np.ndarray,
    wx_row: np.ndarray,
    *,
    f_hz: float | None = None,
    ax=None,
    t_center_s: float | None = None,
    xlim_us: tuple[float, float] | None = None,
    show: bool = True,
):
    """Продольный срез одной полосы CWT: нормированная Re(Wx) во времени."""
    y_vals = np.real(np.asarray(wx_row, dtype=np.complex128))
    denom = np.max(np.abs(y_vals))
    if denom > 0:
        y_vals = y_vals / denom

    if t_center_s is None:
        t0 = float(t[int(np.argmax(np.abs(y_vals)))])
    else:
        t0 = float(t_center_s)
    t_rel_s = t - t0

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    if xlim_us is None:
        xlim_us = _default_xlim_us_morlet_slice(t, f_hz)

    half_w_s = max(abs(xlim_us[0]), abs(xlim_us[1])) * 1e-6
    if half_w_s >= 0.1:
        x_plot = t_rel_s
        xlim_plot = (xlim_us[0] * 1e-6, xlim_us[1] * 1e-6)
        x_unit = "с"
    elif half_w_s >= 1e-3:
        x_plot = t_rel_s * 1e3
        xlim_plot = (xlim_us[0] * 1e-3, xlim_us[1] * 1e-3)
        x_unit = "мс"
    else:
        x_plot = t_rel_s * 1e6
        xlim_plot = (xlim_us[0], xlim_us[1])
        x_unit = "мкс"

    t_total = float(t[-1] - t[0])
    ax.plot(x_plot, y_vals, color="#0072BD", linewidth=1.5, label="Morlet Wavelet (Re)")
    title = "Morlet Wavelet"
    if f_hz is not None:
        title = f"{title}\nf ≈ {float(f_hz):.1f} Гц"
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(
        f"Время относительно центра ({x_unit})\n"
        f"фрагмент вокруг вейвлета; длительность всей записи {t_total:.3f} с",
        fontsize=9,
    )
    ax.set_ylabel("Норм. амплитуда")
    ax.set_xlim(xlim_plot)
    ax.set_ylim(-1, 1)
    ax.grid(True, which="both", linestyle="-", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=8)

    if show:
        backend = matplotlib.get_backend().lower()
        if "agg" not in backend:
            plt.show()

    return t_rel_s, y_vals


class CanvasPlot(ttk.Frame):
    def __init__(self, master, title: str = "График") -> None:
        super().__init__(master)
        self.title_var = tk.StringVar(value=title)
        ttk.Label(self, textvariable=self.title_var, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=5, pady=2)

        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.fig.patch.set_facecolor("#ffffff")
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._colorbar = None
        self._last = None
        self.current_spec: dict | None = None
        self._spec_click_cid = None
        self._on_spec_click = None

        self.clear(title, "Нет данных")

    def _detach_colorbar(self) -> None:
        if self._colorbar is None:
            return
        try:
            self._colorbar.remove()
        except Exception:
            pass
        self._colorbar = None

    def _rebuild_axes(self) -> None:
        self._detach_colorbar()
        try:
            self.fig.clear()
        except Exception:
            pass
        self.ax = self.fig.add_subplot(111)

    def _finalize_draw(self, *, with_colorbar: bool = False) -> None:
        if with_colorbar:
            self.fig.subplots_adjust(left=0.10, right=0.86, bottom=0.14, top=0.90)
        else:
            try:
                self.fig.tight_layout()
            except Exception:
                self.fig.subplots_adjust(left=0.10, right=0.96, bottom=0.14, top=0.90)
        self.canvas.draw_idle()

    def clear(self, title: str | None = None, message: str = "Нет данных") -> None:
        if title:
            self.title_var.set(title)
        self._last = ("clear", title, message)
        self._rebuild_axes()
        self.ax.set_axis_off()
        self.ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            color="#64748b",
            fontname="Segoe UI",
            fontsize=14,
            transform=self.ax.transAxes,
        )
        self.current_spec = None
        self._finalize_draw()

    def set_spec_click_handler(self, callback) -> None:
        self._on_spec_click = callback
        if self._spec_click_cid is not None:
            self.canvas.mpl_disconnect(self._spec_click_cid)
        self._spec_click_cid = self.canvas.mpl_connect("button_press_event", self._handle_spec_click)

    def _handle_spec_click(self, event) -> None:
        if self._on_spec_click is not None:
            self._on_spec_click(event, self.current_spec)

    def draw_xy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        title: str,
        x_label: str = "",
        y_label: str = "",
        color: str = BLUE,
        annotations: list[dict] | None = None,
        note: str = "",
        remember: bool = True,
    ) -> None:
        if remember:
            self._last = ("xy", x.copy(), y.copy(), title, x_label, y_label, color, annotations or [], note)
        self.title_var.set(title)
        self._rebuild_axes()
        self.ax.set_axis_on()

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size == 0 or y.size == 0:
            self.clear(title, "Нет данных для графика")
            return
        if x.size != y.size:
            n = min(x.size, y.size)
            x, y = x[:n], y[:n]

        self.ax.plot(x, y, color=color, linewidth=1.5)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.grid(True, linestyle="-", alpha=0.5)

        for item in annotations or []:
            try:
                ax_val = float(item["x"])
                ay_val = float(item["y"])
                lbl = item.get("label", "")
                col = str(item.get("color", RED))
                self.ax.plot(ax_val, ay_val, marker="o", color=col, markersize=5)
                self.ax.axvline(ax_val, color=col, linestyle="--", alpha=0.7)
                if lbl:
                    self.ax.annotate(
                        lbl,
                        xy=(ax_val, ay_val),
                        xytext=(5, 5),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#fff7ed", ec="#fdba74", alpha=0.8),
                        fontsize=9,
                    )
            except (KeyError, TypeError, ValueError):
                continue

        if note:
            self.ax.annotate(
                note,
                xy=(0.98, 0.95),
                xycoords="axes fraction",
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f8fafc", ec="#cbd5e1", alpha=0.9),
                fontsize=9,
            )

        self.current_spec = None
        self._finalize_draw()

    def draw_interactive_spec(
        self,
        t: np.ndarray,
        freqs_hz: np.ndarray,
        spec: np.ndarray,
        *,
        title: str,
        y_label: str = "Частота (Гц)",
        x_label: str = "Время (с)",
        Wx: np.ndarray | None = None,
        spec_kind: str = "STFT",
        log_freq: bool = False,
        target_hz_line: float | None = None,
        freq_ylim: tuple[float, float] | None = None,
        colorbar_label: str | None = None,
        note: str = "",
    ) -> None:
        self.title_var.set(title)
        self._last = ("spec", t, freqs_hz, spec, title, spec_kind)
        self._rebuild_axes()
        self.ax.set_axis_on()

        t = np.asarray(t, dtype=np.float64)
        freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
        spec = np.asarray(spec, dtype=np.float32)
        if spec.size == 0 or spec.ndim != 2:
            self.clear(title, "Нет данных для спектрограммы")
            return

        morlet_display = spec_kind == "Morlet CWT"
        if morlet_display:
            m = _morlet_ridge_line_collection(
                self.ax, t, freqs_hz, spec, tone_hz=target_hz_line, linewidth=0.75
            )
        else:
            m = self.ax.pcolormesh(t, freqs_hz, spec, shading="auto")
        cbar_label = colorbar_label or ("dB" if spec_kind == "STFT" else "Амплитуда")
        self._colorbar = self.fig.colorbar(m, ax=self.ax, label=cbar_label)

        if log_freq:
            pos_freqs = freqs_hz[freqs_hz > 0]
            if pos_freqs.size:
                try:
                    self.ax.set_yscale("log")
                    if freq_ylim is not None:
                        y0, y1 = float(freq_ylim[0]), float(freq_ylim[1])
                        if y0 > 0 and y1 > 0 and y0 < y1:
                            self.ax.set_ylim(y0, y1)
                        else:
                            self.ax.set_ylim(float(np.min(pos_freqs)), float(np.max(pos_freqs)))
                    else:
                        self.ax.set_ylim(float(np.min(pos_freqs)), float(np.max(pos_freqs)))
                    self.ax.yaxis.set_major_formatter(ScalarFormatter())
                    self.ax.yaxis.set_minor_formatter(ScalarFormatter())
                except (ValueError, ZeroDivisionError):
                    pass
            if morlet_display and t.size:
                self.ax.set_xlim(float(t[0]), float(t[-1]))

        if target_hz_line is not None:
            self.ax.axhline(float(target_hz_line), color="white", linewidth=1.0, alpha=0.8)

        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_title(title, fontweight="bold")

        if note:
            self.ax.annotate(
                note,
                xy=(0.98, 0.95),
                xycoords="axes fraction",
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f8fafc", ec="#cbd5e1", alpha=0.9),
                fontsize=9,
            )

        self.current_spec = {
            "t": t,
            "freqs_hz": freqs_hz,
            "spec": spec,
            "title": spec_kind,
            "y_label": y_label,
            "Wx": Wx,
        }
        self._finalize_draw(with_colorbar=True)

    def draw_waveform(self, data: np.ndarray, samplerate: int, title: str = "Форма сигнала") -> None:
        data = np.asarray(data, dtype=np.float32)
        t = np.arange(data.size, dtype=np.float32) / max(1, samplerate)
        self.draw_xy(t, data, title, "время, с", "амплитуда", BLUE)