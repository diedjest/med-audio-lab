"""Serial-port capture engine for ADC / telemetry streams (ported from SerialRecorder).

Supported stream formats (auto-detected from the first bytes):

* **binary** - little-endian unsigned 16-bit ADC words, DC-centered on the fly
  with an exponential moving average; USB byte-shift (odd alignment) is
  detected and repaired.
* **text** - one number per line (``\\n``, ``\\r`` or ``\\r\\n`` terminated).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import numpy as np

from app.core.interfaces.serial_engine import ISerialEngine
from app.core.models.device_info import SerialPortInfo
from app.infrastructure.serial.protocol_detector import (
    StreamMode,
    detect_stream_mode,
    parse_numeric_line,
)

_serial_import_error: Exception | None = None
try:
    import serial
    from serial.tools import list_ports
except ImportError as _exc:
    serial = None  # type: ignore[assignment]
    list_ports = None  # type: ignore[assignment]
    _serial_import_error = _exc

ErrorCallback = Callable[[Exception], None]

_READ_TIMEOUT_S: float = 0.15
_READ_CHUNK_BYTES: int = 4096
_IDLE_WAIT_S: float = 0.005
_JOIN_TIMEOUT_S: float = 1.0
_MIN_BYTES_FOR_DETECTION: int = 8
_MAX_TEXT_LINE_BYTES: int = 1 << 16
_DEFAULT_RATE_HZ: int = 1000

_DC_ALPHA: float = 0.95  # weight of the previous DC estimate per chunk
_SHIFT_MIN_SAMPLES: int = 10
_SHIFT_STEP_THRESHOLD: float = 10_000.0  # mean |sample-to-sample step| in ADC counts
_LEVEL_FULL_SCALE_BINARY: float = 2048.0  # centered 12-bit swing
_LEVEL_FULL_SCALE_TEXT: float = 4096.0


class SerialUnavailableError(RuntimeError):
    """Raised when pyserial cannot be imported."""


class SerialCaptureError(RuntimeError):
    """Raised or reported when the port cannot be opened or is lost."""


def _require_serial() -> None:
    if serial is None or list_ports is None:
        raise SerialUnavailableError(f"pyserial is unavailable: {_serial_import_error!r}")


def _mean_abs_step(words: np.ndarray) -> float:
    """Mean absolute difference between consecutive samples (overflow-safe)."""
    return float(np.abs(np.diff(words.astype(np.int32))).mean())


class SerialCaptureEngine(ISerialEngine):
    """Background-thread serial reader that accumulates samples as float32."""

    def __init__(self) -> None:
        self._serial: Any = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()  # start / stop serialization
        self._lock = threading.Lock()  # guards _chunks and _total
        self._chunks: list[np.ndarray] = []
        self._total = 0

        self._recording = False
        self._paused = False
        self._stopping = False
        self._lost = False
        self._level = 0.0
        self._lines = 0
        self._bad_lines = 0
        self._last_value = 0.0
        self._stream_mode: StreamMode | None = None
        self._dc_offset = 0.0
        self._dc_initialized = False

        self._start_time = 0.0
        self._end_time: float | None = None
        self._pause_started = 0.0
        self._paused_total = 0.0

        self._error_callback: ErrorCallback | None = None
        self._last_error: Exception | None = None

    # ------------------------------------------------------------ errors
    def set_error_callback(self, callback: ErrorCallback | None) -> None:
        """Register ``callback(exception)`` for port loss; it must be quick and
        must not call ``stop()``."""
        self._error_callback = callback

    @property
    def last_error(self) -> Exception | None:
        """Most recent error reported by the engine, if any."""
        return self._last_error

    def _emit_error(self, error: Exception) -> None:
        self._last_error = error
        callback = self._error_callback
        if callback is None:
            return
        try:
            callback(error)
        except Exception:  # a faulty UI handler must not kill the reader
            pass

    # -------------------------------------------------------- properties
    @property
    def is_recording(self) -> bool:
        """True while reading (paused included) and the port is alive."""
        return self._recording and not self._lost

    @property
    def is_paused(self) -> bool:
        """True while samples are being discarded."""
        return self._recording and self._paused

    @property
    def level(self) -> float:
        """Normalized level of the most recent sample in [0, 1]."""
        return self._level

    @property
    def stream_mode(self) -> StreamMode | None:
        """Detected stream format, or None until enough bytes have arrived."""
        return self._stream_mode

    @property
    def line_stats(self) -> tuple[int, int]:
        """``(samples_parsed, bad_lines)`` counters."""
        return self._lines, self._bad_lines

    @property
    def actual_sample_rate(self) -> int:
        """Measured rate in Hz: stored samples / un-paused elapsed seconds.

        Returns a nominal 1000 Hz until at least one sample has been stored.
        """
        active = self._active_duration()
        with self._lock:
            count = self._total
        if active <= 0.0 or count <= 0:
            return _DEFAULT_RATE_HZ
        return max(1, int(round(count / active)))

    def _active_duration(self) -> float:
        if self._start_time == 0.0:
            return 0.0
        end = self._end_time
        now = time.perf_counter() if end is None else end
        paused = self._paused_total
        if end is None and self._paused and self._pause_started:
            paused += now - self._pause_started
        return max(0.0, now - self._start_time - paused)

    # ----------------------------------------------------------- devices
    def list_ports(self) -> list[SerialPortInfo]:
        """Enumerate available serial ports."""
        _require_serial()
        return [
            SerialPortInfo(device=port.device, description=port.description or "")
            for port in list_ports.comports()
        ]

    # ----------------------------------------------------------- control
    def start(self, port: str, baudrate: int) -> None:
        """Open ``port`` and start the daemon reader thread.

        Raises:
            SerialUnavailableError: pyserial is not installed.
            SerialCaptureError: already running or the port cannot be opened
                (also reported through the error callback).
            ValueError: non-positive baud rate.
        """
        _require_serial()
        if baudrate <= 0:
            raise ValueError(f"baudrate must be positive, got {baudrate}")

        with self._state_lock:
            if self._recording:
                raise SerialCaptureError("Serial capture is already running.")

            with self._lock:
                self._chunks = []
                self._total = 0
            self._lines = 0
            self._bad_lines = 0
            self._last_value = 0.0
            self._level = 0.0
            self._paused = False
            self._stopping = False
            self._lost = False
            self._last_error = None
            self._stream_mode = None
            self._dc_offset = 0.0
            self._dc_initialized = False
            self._end_time = None
            self._pause_started = 0.0
            self._paused_total = 0.0
            self._stop_event.clear()

            ser = None
            try:
                ser = serial.Serial(port=port, baudrate=int(baudrate), timeout=_READ_TIMEOUT_S)
                ser.reset_input_buffer()
            except (OSError, ValueError) as exc:  # SerialException is an OSError
                if ser is not None:
                    try:
                        ser.close()
                    except OSError:
                        pass
                error = SerialCaptureError(f"Cannot open serial port {port!r}: {exc}")
                self._emit_error(error)
                raise error from exc

            self._serial = ser
            self._recording = True
            self._start_time = time.perf_counter()
            self._thread = threading.Thread(
                target=self._reader_loop, name="SerialCaptureEngine", daemon=True
            )
            self._thread.start()

    def pause(self) -> None:
        """Discard incoming samples while keeping the port open."""
        if self._recording and not self._paused:
            self._pause_started = time.perf_counter()
            self._paused = True

    def resume(self) -> None:
        """Resume storing samples after ``pause``."""
        if self._recording and self._paused:
            self._paused_total += time.perf_counter() - self._pause_started
            self._pause_started = 0.0
            self._paused = False

    def stop(self) -> np.ndarray:
        """Stop the reader, close the port and return the complete recording.

        Safe to call when idle (returns an empty array) or after the port was lost.
        """
        with self._state_lock:
            self._stopping = True
            now = time.perf_counter()
            if self._end_time is None and self._start_time:
                self._end_time = now
            if self._paused and self._pause_started:
                self._paused_total += (self._end_time or now) - self._pause_started
                self._pause_started = 0.0

            self._stop_event.set()
            thread, self._thread = self._thread, None
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=_JOIN_TIMEOUT_S)

            ser, self._serial = self._serial, None
            if ser is not None:
                try:
                    ser.close()
                except OSError as exc:
                    self._emit_error(SerialCaptureError(f"Error closing port: {exc}"))

            self._recording = False
            self._paused = False
            with self._lock:
                chunks, self._chunks = self._chunks, []

        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

    def snapshot(self, max_samples: int | None = None) -> np.ndarray:
        """Return the most recent samples without consuming them.

        Args:
            max_samples: Number of newest samples; None returns everything so far.
        """
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            if not max_samples or max_samples <= 0:
                return np.concatenate(self._chunks).astype(np.float32, copy=False)

            remaining = int(max_samples)
            parts: list[np.ndarray] = []
            for chunk in reversed(self._chunks):
                if remaining <= 0:
                    break
                parts.append(chunk if chunk.size <= remaining else chunk[-remaining:])
                remaining -= chunk.size
            return np.concatenate(parts[::-1]).astype(np.float32, copy=False)

    # ------------------------------------------------------ stream parsing
    def _store(self, samples: np.ndarray) -> None:
        with self._lock:
            self._chunks.append(samples)
            self._total += samples.size

    def _consume_text_buffer(self, buffer: bytearray) -> None:
        """Parse complete lines from ``buffer`` (consumed in place); keep partial tail."""
        values: list[float] = []
        while True:
            nl = buffer.find(b"\n")
            cr = buffer.find(b"\r")
            if nl < 0 and cr < 0:
                break
            sep = min(i for i in (nl, cr) if i >= 0)
            line = bytes(buffer[:sep])
            del buffer[: sep + 1]
            if buffer[:1] in (b"\n", b"\r"):  # second half of \r\n
                del buffer[:1]
            if not line:
                continue

            value = parse_numeric_line(line.decode("utf-8", errors="ignore"))
            if value is None:
                self._bad_lines += 1
                continue
            values.append(value)

        if len(buffer) > _MAX_TEXT_LINE_BYTES:  # no terminator in a huge run: garbage
            buffer.clear()
            self._bad_lines += 1

        if values:
            self._lines += len(values)
            self._last_value = values[-1]
            self._level = min(1.0, abs(self._last_value) / _LEVEL_FULL_SCALE_TEXT)
            if not self._paused:
                self._store(np.asarray(values, dtype=np.float32))

    @staticmethod
    def _looks_byte_shifted(chunk: bytes, words: np.ndarray) -> bool:
        """True if the stream is misaligned by one byte.

        A misaligned stream mixes high and low bytes of different samples, which
        makes consecutive words jump wildly.  The one-byte-shifted reading must be
        markedly smoother than the current one, so a genuinely full-swing signal
        (square wave, clipping) is not mistaken for misalignment.
        """
        if words.size <= _SHIFT_MIN_SAMPLES:
            return False
        here = _mean_abs_step(words)
        if here <= _SHIFT_STEP_THRESHOLD:
            return False
        alt_words = (len(chunk) - 1) // 2
        if alt_words <= _SHIFT_MIN_SAMPLES:
            return False
        alt = np.frombuffer(chunk, dtype="<u2", count=alt_words, offset=1)
        return _mean_abs_step(alt) < 0.5 * here

    def _consume_binary_buffer(self, buffer: bytearray) -> None:
        """Parse little-endian uint16 words from ``buffer`` (consumed in place)."""
        while len(buffer) >= 2:
            num_bytes = (len(buffer) // 2) * 2
            chunk = bytes(buffer[:num_bytes])
            words = np.frombuffer(chunk, dtype="<u2")

            if self._looks_byte_shifted(chunk, words):
                del buffer[:1]  # drop the stray byte and re-parse immediately
                continue
            del buffer[:num_bytes]

            chunk_mean = float(words.mean())
            if not self._dc_initialized:
                self._dc_offset = chunk_mean
                self._dc_initialized = True
            else:
                self._dc_offset = _DC_ALPHA * self._dc_offset + (1.0 - _DC_ALPHA) * chunk_mean

            centered = (words.astype(np.float32) - np.float32(self._dc_offset)).astype(
                np.float32, copy=False
            )
            self._lines += centered.size
            self._last_value = float(centered[-1])
            self._level = min(1.0, abs(self._last_value) / _LEVEL_FULL_SCALE_BINARY)
            if not self._paused:
                self._store(centered)
            return

    # ------------------------------------------------------------- reader
    def _reader_loop(self) -> None:
        """Read raw bytes, auto-detect the format and dispatch to the parsers."""
        buffer = bytearray()
        ser = self._serial
        while not self._stop_event.is_set():
            try:
                raw = ser.read(max(1, ser.in_waiting or _READ_CHUNK_BYTES))
            except OSError as exc:  # port unplugged / closed under us
                if not self._stopping:
                    self._port_lost(exc)
                return
            except Exception:
                self._bad_lines += 1
                self._stop_event.wait(_IDLE_WAIT_S)
                continue

            if not raw:
                self._stop_event.wait(_IDLE_WAIT_S)
                continue

            buffer.extend(raw)
            if self._stream_mode is None and len(buffer) >= _MIN_BYTES_FOR_DETECTION:
                self._stream_mode = detect_stream_mode(buffer)

            if self._stream_mode is None:
                continue  # too few bytes to classify the stream yet; keep buffering

            try:
                if self._stream_mode == "text":
                    self._consume_text_buffer(buffer)
                else:
                    self._consume_binary_buffer(buffer)
            except Exception:
                self._bad_lines += 1
                buffer.clear()

    def _port_lost(self, exc: BaseException) -> None:
        self._end_time = time.perf_counter()
        self._lost = True
        self._emit_error(
            SerialCaptureError(
                f"Serial port lost ({exc}). Call stop() to retrieve the data captured so far."
            )
        )


__all__: list[str] = [
    "SerialCaptureEngine",
    "SerialCaptureError",
    "SerialUnavailableError",
]
