"""Abstract interfaces for audio capture and playback engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from app.core.models.device_info import AudioDeviceInfo


class IAudioCaptureEngine(ABC):
    """Abstract audio capture engine interface implemented by infrastructure layer."""

    @abstractmethod
    def list_input_devices(self) -> list[AudioDeviceInfo]:
        """List all available input audio capture devices."""
        ...

    @abstractmethod
    def start(
        self,
        device_index: int | None,
        sample_rate: int,
        channels: int = 1,
        dtype: str = "float32",
        block_size: int = 1024,
        channel: int | None = None,
    ) -> None:
        """Start recording from the specified audio input device.

        ``channel`` is the zero-based input channel to record (mono); None records all."""
        ...

    @abstractmethod
    def pause(self) -> None:
        """Temporarily pause stream recording without destroying the stream."""
        ...

    @abstractmethod
    def resume(self) -> None:
        """Resume recording after being paused."""
        ...

    @abstractmethod
    def stop(self) -> np.ndarray:
        """Stop audio stream capture and return all accumulated audio samples."""
        ...

    @abstractmethod
    def snapshot(self, max_samples: int | None = None) -> np.ndarray:
        """Return the most recent audio samples without consuming or modifying the buffer."""
        ...

    @abstractmethod
    def peak_rms(self) -> tuple[float, float]:
        """Compute and return (peak_dbfs, rms_dbfs) of the latest audio frame."""
        ...

    @property
    @abstractmethod
    def is_recording(self) -> bool:
        """Return True if the engine is currently capturing or paused."""
        ...

    @property
    @abstractmethod
    def is_paused(self) -> bool:
        """Return True if capture is currently paused."""
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Return the active capture sample rate in Hz."""
        ...


class IAudioPlaybackEngine(ABC):
    """Abstract audio playback engine interface implemented by infrastructure layer."""

    @abstractmethod
    def list_output_devices(self) -> list[AudioDeviceInfo]:
        """List all available audio playback output devices."""
        ...

    @abstractmethod
    def play(
        self,
        data: np.ndarray,
        sample_rate: int,
        device_index: int | None = None,
    ) -> None:
        """Play audio buffer through the specified or default output device."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop active audio playback immediately."""
        ...

    @property
    @abstractmethod
    def is_playing(self) -> bool:
        """Return True if audio playback is currently in progress."""
        ...


__all__: list[str] = [
    "IAudioCaptureEngine",
    "IAudioPlaybackEngine",
]
