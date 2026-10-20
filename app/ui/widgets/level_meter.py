"""Horizontal dBFS level meter with RMS bar and a decaying peak-hold marker."""

from __future__ import annotations

import time

from PySide6.QtCore import QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

MIN_DBFS = -60.0
MAX_DBFS = 0.0
_YELLOW_FROM_DBFS = -12.0
_RED_FROM_DBFS = -3.0
_PEAK_HOLD_S = 1.2
_PEAK_FALL_DB_PER_S = 20.0
_TICK_MS = 40

_COLOR_TRACK = QColor("#e2e8f0")
_COLOR_GREEN = QColor("#16a34a")
_COLOR_YELLOW = QColor("#eab308")
_COLOR_RED = QColor("#dc2626")
_COLOR_PEAK = QColor("#0f172a")
_COLOR_TEXT = QColor("#334155")


class LevelMeter(QWidget):
    """Shows the latest RMS level as a bar and the peak with a hold marker.

    Feed it with :meth:`set_levels` (dBFS, as emitted by the recording view model).
    The bar spans -60 .. 0 dBFS: green, yellow above -12 dB, red above -3 dB.  The
    peak marker stays for about a second, then falls.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rms = MIN_DBFS
        self._peak = MIN_DBFS
        self._hold = MIN_DBFS
        self._hold_since = 0.0
        self._last_tick = 0.0

        self.setMinimumSize(120, 22)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName("Индикатор уровня")

        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._decay)

    # ------------------------------------------------------------- values
    @property
    def rms_dbfs(self) -> float:
        """Displayed RMS level in dBFS."""
        return self._rms

    @property
    def peak_dbfs(self) -> float:
        """Latest peak level in dBFS."""
        return self._peak

    @property
    def peak_hold_dbfs(self) -> float:
        """Level of the peak-hold marker in dBFS."""
        return self._hold

    def set_levels(self, peak_dbfs: float, rms_dbfs: float) -> None:
        """Update the meter (values are clamped to -60 .. 0 dBFS)."""
        self._peak = self._clamp(peak_dbfs)
        self._rms = self._clamp(rms_dbfs)
        now = time.perf_counter()
        if self._peak >= self._hold:
            self._hold = self._peak
            self._hold_since = now
        if self._hold > MIN_DBFS and not self._timer.isActive():
            self._last_tick = now
            self._timer.start()
        self.update()

    def reset(self) -> None:
        """Return to silence immediately."""
        self._rms = self._peak = self._hold = MIN_DBFS
        self._timer.stop()
        self.update()

    @staticmethod
    def _clamp(value: float) -> float:
        if value != value:  # NaN
            return MIN_DBFS
        return max(MIN_DBFS, min(MAX_DBFS, float(value)))

    def _decay(self) -> None:
        now = time.perf_counter()
        elapsed, self._last_tick = now - self._last_tick, now
        if now - self._hold_since > _PEAK_HOLD_S:
            self._hold = max(self._peak, self._hold - _PEAK_FALL_DB_PER_S * elapsed, MIN_DBFS)
        if self._hold <= MIN_DBFS and self._peak <= MIN_DBFS:
            self._timer.stop()
        self.update()

    # ------------------------------------------------------------ painting
    def sizeHint(self) -> QSize:
        return QSize(240, 24)

    @staticmethod
    def _fraction(dbfs: float) -> float:
        return (dbfs - MIN_DBFS) / (MAX_DBFS - MIN_DBFS)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = min(6.0, track.height() / 2.0)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_COLOR_TRACK)
        painter.drawRoundedRect(track, radius, radius)

        painter.save()
        clip = track.adjusted(0, 0, 0, 0)
        painter.setClipRect(QRectF(clip.left(), clip.top(), clip.width() * self._fraction(self._rms), clip.height()))
        for start_db, end_db, color in (
            (MIN_DBFS, _YELLOW_FROM_DBFS, _COLOR_GREEN),
            (_YELLOW_FROM_DBFS, _RED_FROM_DBFS, _COLOR_YELLOW),
            (_RED_FROM_DBFS, MAX_DBFS, _COLOR_RED),
        ):
            x0 = track.left() + track.width() * self._fraction(start_db)
            x1 = track.left() + track.width() * self._fraction(end_db)
            painter.setBrush(color)
            painter.drawRect(QRectF(x0, track.top(), x1 - x0, track.height()))
        painter.restore()

        if self._hold > MIN_DBFS:
            x = track.left() + track.width() * self._fraction(self._hold)
            painter.setPen(QPen(_COLOR_PEAK, 2.0))
            painter.drawLine(int(x), int(track.top()) + 2, int(x), int(track.bottom()) - 2)

        font = QFont(self.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.0))
        painter.setFont(font)
        painter.setPen(_COLOR_TEXT)
        text = "-∞ дБ" if self._rms <= MIN_DBFS else f"{self._rms:.1f} дБ"
        painter.drawText(track.adjusted(0, 0, -6, 0), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, text)


__all__: list[str] = ["LevelMeter"]
