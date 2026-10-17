"""Combo box + refresh button for choosing a device or port."""

from __future__ import annotations

from typing import Any, Callable, Hashable, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QSizePolicy, QStyle, QToolButton, QWidget


class DeviceSelector(QWidget):
    """Lists devices from a loader function and reports the user's choice.

    The first entry is always the "no selection" entry (``none_label``), which
    stands for the system default device or for no port.  Selection is tracked by a
    stable *key* (for example a device index or a port name), so it survives
    :meth:`refresh` when the device is still present.

    Signals:

    * ``device_selected(device_or_None)`` - the user picked an entry, or a refresh
      removed the selected device (then with ``None``).  Programmatic
      :meth:`select_key` does not emit.
    * ``load_failed(message)`` - the loader raised; the list is left empty.
    """

    device_selected = Signal(object)
    load_failed = Signal(str)

    def __init__(
        self,
        loader: Callable[[], Sequence[Any]],
        *,
        none_label: str,
        key: Callable[[Any], Hashable],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._loader = loader
        self._key = key
        self._none_label = none_label

        self._combo = QComboBox(self)
        self._combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._combo.setMinimumContentsLength(8)
        self._combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._refresh_button = QToolButton(self)
        self._refresh_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self._refresh_button.setAccessibleName("Refresh devices")
        self._refresh_button.setToolTip("Rescan the available devices")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._combo, 1)
        layout.addWidget(self._refresh_button)

        self._combo.activated.connect(self._on_activated)
        self._refresh_button.clicked.connect(lambda _checked=False: self.refresh())
        self._reset_items([])

    # ------------------------------------------------------------- access
    @property
    def combo_box(self) -> QComboBox:
        return self._combo

    @property
    def refresh_button(self) -> QToolButton:
        return self._refresh_button

    @property
    def selected_device(self) -> Any | None:
        """The selected device object, or None for the "no selection" entry."""
        return self._combo.currentData()

    @property
    def device_count(self) -> int:
        """Number of real devices listed (without the "no selection" entry)."""
        return self._combo.count() - 1

    def _reset_items(self, devices: Sequence[Any]) -> None:
        self._combo.clear()
        self._combo.addItem(self._none_label, None)
        for device in devices:
            self._combo.addItem(str(getattr(device, "label", device)), device)
        self._fit_popup_to_names()

    def _fit_popup_to_names(self) -> None:
        """Widen the drop-down list so long device names are shown in full."""
        metrics = self._combo.fontMetrics()
        widest = 0
        for i in range(self._combo.count()):
            text = self._combo.itemText(i)
            self._combo.setItemData(i, text, Qt.ItemDataRole.ToolTipRole)
            widest = max(widest, metrics.horizontalAdvance(text))
        scrollbar = self.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
        self._combo.view().setMinimumWidth(widest + scrollbar + 32)

    def _index_of_key(self, key: Hashable | None) -> int:
        if key is None:
            return 0
        for i in range(1, self._combo.count()):
            if self._key(self._combo.itemData(i)) == key:
                return i
        return 0

    # ------------------------------------------------------------ actions
    def refresh(self) -> None:
        """Reload the device list, keeping the selection if the device still exists."""
        previous = self.selected_device
        previous_key = None if previous is None else self._key(previous)
        try:
            devices = list(self._loader())
        except Exception as exc:
            self._reset_items([])
            self.load_failed.emit(str(exc) or type(exc).__name__)
            if previous is not None:
                self.device_selected.emit(None)
            return

        self._combo.blockSignals(True)
        self._reset_items(devices)
        self._combo.setCurrentIndex(self._index_of_key(previous_key))
        self._combo.blockSignals(False)
        if previous is not None and self.selected_device is None:
            self.device_selected.emit(None)

    def select_key(self, key: Hashable | None) -> Any | None:
        """Select the entry with this key without emitting ``device_selected``.

        Returns:
            The selected device, or None if the key is unknown (falls back to the
            "no selection" entry).
        """
        self._combo.blockSignals(True)
        self._combo.setCurrentIndex(self._index_of_key(key))
        self._combo.blockSignals(False)
        return self.selected_device

    def _on_activated(self, index: int) -> None:
        self.device_selected.emit(self._combo.itemData(index))


__all__: list[str] = ["DeviceSelector"]
