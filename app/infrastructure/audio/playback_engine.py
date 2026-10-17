"""Audio playback engine built on ``sounddevice.play`` (non-blocking)."""

from __future__ import annotations

import threading

import numpy as np

from app.core.interfaces.audio_engine import IAudioPlaybackEngine
from app.core.models.device_info import AudioDeviceInfo
from app.infrastructure.audio.device_manager import (
    AudioDeviceManager,
    require_sounddevice,
    sd,
)


class AudioPlaybackError(RuntimeError):
    """Raised when playback cannot be started."""


class PlaybackEngine(IAudioPlaybackEngine):
    """Plays float32 audio buffers on an output device without blocking the caller."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._playing = False

    def list_output_devices(self) -> list[AudioDeviceInfo]:
        """List all available output devices."""
        return AudioDeviceManager.list_output_devices()

    @property
    def is_playing(self) -> bool:
        """True while audio is actually playing.

        ``sd.play(blocking=False)`` offers no completion callback, so the flag
        is reconciled with the PortAudio stream state on every query.
        """
        with self._lock:
            if self._playing and not self._stream_active():
                self._playing = False
            return self._playing

    @staticmethod
    def _stream_active() -> bool:
        if sd is None:
            return False
        try:
            return bool(sd.get_stream().active)
        except RuntimeError:  # no stream has been created yet
            return False
        except sd.PortAudioError:
            return False

    def play(
        self,
        data: np.ndarray,
        sample_rate: int,
        device_index: int | None = None,
    ) -> None:
        """Start non-blocking playback, replacing any playback in progress.

        Args:
            data: Samples, shape ``(n,)`` or ``(n, channels)``; converted to float32.
            sample_rate: Sample rate in Hz.
            device_index: Output device index, or None for the default device.

        Raises:
            AudioUnavailableError: sounddevice / PortAudio is not available.
            ValueError: empty data or non-positive sample rate.
            AudioPlaybackError: the device could not be opened.
        """
        require_sounddevice()
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        samples = np.ascontiguousarray(data, dtype=np.float32)
        if samples.ndim not in (1, 2):
            raise ValueError(f"data must be 1D or 2D, got shape {samples.shape}")
        if samples.shape[0] == 0:
            raise ValueError("No audio data to play.")

        with self._lock:
            self._playing = False
            try:
                sd.play(samples, int(sample_rate), device=device_index, blocking=False)
            except sd.PortAudioError as exc:
                raise AudioPlaybackError(
                    f"Cannot play on output device {device_index}: {exc}"
                ) from exc
            self._playing = True

    def stop(self) -> None:
        """Stop playback immediately; a no-op when nothing is playing."""
        with self._lock:
            self._playing = False
            if sd is not None:
                try:
                    sd.stop()
                except sd.PortAudioError:
                    pass


__all__: list[str] = ["PlaybackEngine", "AudioPlaybackError"]
