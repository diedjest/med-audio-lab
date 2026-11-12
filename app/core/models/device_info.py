"""Device information data models for audio and serial interfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioDeviceInfo:
    """Immutable metadata describing a PortAudio hardware device."""

    index: int
    name: str
    input_channels: int
    output_channels: int
    default_sample_rate: int
    hostapi: str

    @property
    def label(self) -> str:
        """Formatted display string for UI selectors."""
        api_suffix = f" ({self.hostapi})" if self.hostapi else ""
        return f"[{self.index}] {self.name}{api_suffix}"


@dataclass(frozen=True)
class SerialPortInfo:
    """Immutable metadata describing an available serial communication port."""

    device: str
    description: str

    @property
    def label(self) -> str:
        """Formatted display string for UI selectors."""
        if self.description and self.description != self.device:
            return f"{self.device} - {self.description}"
        return self.device


__all__: list[str] = [
    "AudioDeviceInfo",
    "SerialPortInfo",
]
