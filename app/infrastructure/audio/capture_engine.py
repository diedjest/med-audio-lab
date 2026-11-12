"""Real-time audio capture engine built on ``sounddevice.InputStream``.

Threading model
---------------
* **PortAudio callback thread** - ``_callback`` only checks two flags and calls
  ``RingBuffer.write`` (``np.copyto`` into pre-allocated storage).  It takes no
  locks and creates no arrays.
* **Accumulator thread** - drains the ring buffer every ``_DRAIN_INTERVAL_S``
  into a list of chunks (the "complete recording"), watches for the stream
  dying (device unplugged) and reports data loss.
* **Caller / UI threads** - ``start``/``pause``/``resume``/``stop`` are
  serialized by a state lock; ``snapshot`` and ``peak_rms`` are non-consuming
  reads of the ring buffer and are safe to call concurrently.
"""

from __future__ import annotations

import threading
from typing import Callable

import numpy as np

from app.constants import DEFAULT_BLOCK_SIZE, RING_BUFFER_CAPACITY
from app.core.interfaces.audio_engine import IAudioCaptureEngine
from app.core.models.device_info import AudioDeviceInfo
from app.infrastructure.audio.device_manager import (
    AudioDeviceManager,
    require_sounddevice,
    sd,
)
from app.infrastructure.audio.ring_buffer import RingBuffer

ErrorCallback = Callable[[Exception], None]

_DRAIN_INTERVAL_S: float = 0.02
_PEAK_RMS_WINDOW: int = 1024
_DB_FLOOR: float = -120.0
_JOIN_TIMEOUT_S: float = 2.0


class AudioCaptureError(RuntimeError):
    """Raised or reported when capture cannot start or the stream is lost."""


def _to_dbfs(value: float) -> float:
    """Convert a linear amplitude to dBFS, clamped at ``_DB_FLOOR``."""
    if value <= 0.0:
        return _DB_FLOOR
    return max(_DB_FLOOR, float(20.0 * np.log10(value)))


class CaptureEngine(IAudioCaptureEngine):
    """Ring-buffer backed audio capture engine."""

    def __init__(self) -> None:
        self._ring = RingBuffer(RING_BUFFER_CAPACITY, np.float32, channels=1)
        self._state_lock = threading.Lock()
        self._chunks_lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._stream: "sd.InputStream | None" = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._error_callback: ErrorCallback | None = None
        self._last_error: Exception | None = None

        # Plain bool/int flags: single-store reads are atomic, so the audio
        # callback can read them without a lock.
        self._recording = False
        self._paused = False
        self._stopping = False
        self._lost = False
        self._sample_rate = 44100
        self._channels = 1
        self._channel: int | None = None  # input channel to record; None = all
        self._overflow_count = 0
        self._reported_dropped = 0

    # ------------------------------------------------------------ errors
    def set_error_callback(self, callback: ErrorCallback | None) -> None:
        """Register ``callback(exception)`` for stream errors and data loss.

        It may be invoked from the accumulator thread or the thread calling
        ``start``/``stop``; it must be quick and must not call ``stop()``.
        """
        self._error_callback = callback

    @property
    def last_error(self) -> Exception | None:
        """Most recent error reported by the engine, if any."""
        return self._last_error

    @property
    def overflow_count(self) -> int:
        """Number of callbacks PortAudio flagged with an input overflow/underflow."""
        return self._overflow_count

    def _emit_error(self, error: Exception) -> None:
        self._last_error = error
        callback = self._error_callback
        if callback is None:
            return
        try:
            callback(error)
        except Exception:  # a faulty UI handler must not kill the engine
            pass

    # ----------------------------------------------------------- devices
    def list_input_devices(self) -> list[AudioDeviceInfo]:
        """List all available input devices."""
        return AudioDeviceManager.list_input_devices()

    # --------------------------------------------------------- properties
    @property
    def is_recording(self) -> bool:
        """True while capturing (paused included) and the stream is alive."""
        return self._recording and not self._lost

    @property
    def is_paused(self) -> bool:
        """True while capture is paused."""
        return self._recording and self._paused

    @property
    def sample_rate(self) -> int:
        """Active (or last used) capture sample rate in Hz."""
        return self._sample_rate

    # ---------------------------------------------------------- callback
    def _callback(self, indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        """PortAudio callback: flag checks and one ``np.copyto`` only."""
        if status:
            self._overflow_count += 1
        if self._paused:
            return
        channel = self._channel
        self._ring.write(indata if channel is None else indata[:, channel])

    # ------------------------------------------------------------- control
    def start(
        self,
        device_index: int | None,
        sample_rate: int,
        channels: int = 1,
        dtype: str = "float32",
        block_size: int = DEFAULT_BLOCK_SIZE,
        channel: int | None = None,
    ) -> None:
        """Open the input stream and begin capturing.

        Args:
            device_index: PortAudio device index, or None for the default input.
            sample_rate: Requested sample rate in Hz.
            channels: Number of input channels.
            dtype: Sample format delivered by PortAudio. The ring buffer stores
                normalized float32, so only ``"float32"`` is accepted.
            block_size: Frames per callback.
            channel: Zero-based index of the one input channel to record (the stream
                is opened with at least ``channel + 1`` channels and the recording is
                mono).  None records all ``channels`` channels.

        Raises:
            AudioUnavailableError: sounddevice / PortAudio is not available.
            AudioCaptureError: already recording, or the device cannot be opened
                (also reported through the error callback).
            ValueError: invalid arguments.
        """
        require_sounddevice()
        if dtype != "float32":
            raise ValueError(f"Only dtype 'float32' is supported, got {dtype!r}")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")
        if block_size < 0:
            raise ValueError(f"block_size must be >= 0, got {block_size}")
        if channel is not None:
            if channel < 0:
                raise ValueError(f"channel must be >= 0, got {channel}")
            channels = max(channels, channel + 1)
        stored_channels = 1 if channel is not None else channels

        with self._state_lock:
            if self._recording:
                raise AudioCaptureError("Запись уже идёт.")

            if self._ring.channels != stored_channels:
                self._ring = RingBuffer(RING_BUFFER_CAPACITY, np.float32, channels=stored_channels)
            else:
                self._ring.reset()
            with self._chunks_lock:
                self._chunks = []
            self._stop_event.clear()
            self._paused = False
            self._stopping = False
            self._lost = False
            self._last_error = None
            self._overflow_count = 0
            self._reported_dropped = 0
            self._channels = int(stored_channels)
            self._channel = None if channel is None else int(channel)

            if device_index is None:
                device_index = AudioDeviceManager.default_input_index()
            extra = None
            if device_index is not None and AudioDeviceManager.is_wasapi(device_index):
                # Let Windows resample when the rate differs from the mixer format.
                extra = sd.WasapiSettings(auto_convert=True)

            stream = None
            try:
                stream = sd.InputStream(
                    device=device_index,
                    samplerate=int(sample_rate),
                    channels=int(channels),
                    dtype="float32",
                    blocksize=int(block_size),
                    callback=self._callback,
                    extra_settings=extra,
                )
                stream.start()
            except sd.PortAudioError as exc:
                if stream is not None:
                    try:
                        stream.close()
                    except sd.PortAudioError:
                        pass
                error = AudioCaptureError(f"Не удалось открыть устройство ввода {device_index}: {exc}")
                error.__cause__ = exc
                self._emit_error(error)
                raise error from exc

            self._stream = stream
            self._sample_rate = int(round(float(stream.samplerate))) or int(sample_rate)
            self._recording = True
            self._thread = threading.Thread(
                target=self._accumulate_loop,
                name="CaptureEngine-accumulator",
                daemon=True,
            )
            self._thread.start()

    def pause(self) -> None:
        """Stop storing audio while keeping the stream open."""
        if self._recording:
            self._paused = True

    def resume(self) -> None:
        """Resume storing audio after ``pause``."""
        if self._recording:
            self._paused = False

    def stop(self) -> np.ndarray:
        """Close the stream and return the complete recording.

        Safe to call when not recording (returns an empty array) and after the
        device was lost (returns whatever was captured before the loss).

        Returns:
            float32 array of shape ``(n,)`` for mono or ``(n, channels)``.
        """
        with self._state_lock:
            self._stopping = True
            stream, self._stream = self._stream, None
            thread, self._thread = self._thread, None

            if stream is not None:
                try:
                    stream.stop()
                except sd.PortAudioError as exc:
                    self._emit_error(AudioCaptureError(f"Ошибка остановки потока: {exc}"))
                try:
                    stream.close()
                except sd.PortAudioError as exc:
                    self._emit_error(AudioCaptureError(f"Ошибка закрытия потока: {exc}"))

            self._stop_event.set()
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=_JOIN_TIMEOUT_S)

            self._collect()
            self._report_dropped()
            with self._chunks_lock:
                chunks, self._chunks = self._chunks, []

            self._recording = False
            self._paused = False

        if not chunks:
            shape = (0,) if self._channels == 1 else (0, self._channels)
            return np.zeros(shape, dtype=np.float32)
        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

    # ------------------------------------------------------------- reading
    def snapshot(self, max_samples: int | None = None) -> np.ndarray:
        """Return the most recent frames without consuming anything.

        Args:
            max_samples: Maximum number of frames; None returns everything the
                ring buffer still holds.
        """
        count = self._ring.capacity if not max_samples or max_samples <= 0 else int(max_samples)
        return self._ring.read_latest(count)

    def peak_rms(self) -> tuple[float, float]:
        """Return ``(peak_dbfs, rms_dbfs)`` of the latest 1024 frames."""
        latest = self._ring.read_latest(_PEAK_RMS_WINDOW)
        if latest.size == 0:
            return _DB_FLOOR, _DB_FLOOR
        samples = latest.astype(np.float64, copy=False)
        peak = float(np.max(np.abs(samples)))
        rms = float(np.sqrt(np.mean(np.square(samples))))
        return _to_dbfs(peak), _to_dbfs(rms)

    # ---------------------------------------------------------- accumulator
    def _collect(self) -> None:
        """Move newly captured frames from the ring buffer into the chunk list."""
        data = self._ring.drain_all()
        if data.shape[0]:
            with self._chunks_lock:
                self._chunks.append(data)

    def _report_dropped(self) -> None:
        dropped = self._ring.dropped_samples
        if dropped > self._reported_dropped:
            self._reported_dropped = dropped
            self._emit_error(
                AudioCaptureError(f"Потеряно отсчётов: {dropped} (накопитель не успевает).")
            )

    def _accumulate_loop(self) -> None:
        """Drain the ring buffer and monitor the stream until stopped."""
        while not self._stop_event.wait(_DRAIN_INTERVAL_S):
            self._collect()
            self._report_dropped()

            stream = self._stream
            if stream is not None and not self._stopping and not stream.active:
                self._lost = True
                self._collect()
                self._emit_error(
                    AudioCaptureError(
                        "Audio stream stopped unexpectedly (device disconnected?). "
                        "Call stop() to retrieve the audio captured so far."
                    )
                )
                return


__all__: list[str] = ["CaptureEngine", "AudioCaptureError"]
