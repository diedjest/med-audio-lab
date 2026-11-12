"""Abstract interface for serial port telemetry and ADC streaming engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from app.core.models.device_info import SerialPortInfo


class ISerialEngine(ABC):
    """Abstract interface for serial data streaming engines."""

    @abstractmethod
    def list_ports(self) -> list[SerialPortInfo]:
        """Enumerate available serial COM / tty ports."""
        ...

    @abstractmethod
    def start(self, port: str, baudrate: int) -> None:
        """Open serial port connection and begin background sampling."""
        ...

    @abstractmethod
    def pause(self) -> None:
        """Pause reading and buffering from the serial port."""
        ...

    @abstractmethod
    def resume(self) -> None:
        """Resume reading and buffering from the serial port."""
        ...

    @abstractmethod
    def stop(self) -> np.ndarray:
        """Close connection and return all accumulated sensor/audio samples."""
        ...

    @abstractmethod
    def snapshot(self, max_samples: int | None = None) -> np.ndarray:
        """Return the most recent samples without clearing the internal buffer."""
        ...

    @property
    @abstractmethod
    def is_recording(self) -> bool:
        """Return True if the serial engine is actively reading or paused."""
        ...

    @property
    @abstractmethod
    def is_paused(self) -> bool:
        """Return True if serial reading is paused."""
        ...

    @property
    @abstractmethod
    def actual_sample_rate(self) -> int:
        """Return the measured or nominal incoming sample rate in Hz."""
        ...

    @property
    @abstractmethod
    def level(self) -> float:
        """Return the current normalized signal level or amplitude."""
        ...


__all__: list[str] = ["ISerialEngine"]
