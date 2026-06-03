from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

import numpy as np

try:
    import serial
    from serial.tools import list_ports
except Exception as exc:
    serial = None
    list_ports = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class SerialPortInfo:
    device: str
    description: str

    @property
    def label(self) -> str:
        return f"{self.device} - {self.description}" if self.description else self.device


class SerialUnavailableError(RuntimeError):
    pass


def _require_serial() -> None:
    if serial is None or list_ports is None:
        raise SerialUnavailableError(f"pyserial не загрузился: {_IMPORT_ERROR!r}")


def list_serial_ports() -> list[SerialPortInfo]:
    _require_serial()
    return [SerialPortInfo(port.device, port.description or "") for port in list_ports.comports()]


def parse_numeric_line(text: str) -> float | None:
    match = NUMBER_RE.search(text.strip())
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None


class SerialRecorder:
    def __init__(self) -> None:
        self._serial = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._samples: list[float] = []
        self._paused = False
        self._recording = False
        self._level = 0.0
        self._lines = 0
        self._bad_lines = 0
        self._last_value = 0.0
        self._start_time = 0.0
        self._duration = 0.0
        self._pause_started = 0.0
        self._paused_total = 0.0
        self._last_count = 0
        self._stream_mode: str | None = None  # "binary" | "text"

        # Добавлено для расчета DC Offset "на лету"
        self._dc_offset = 0.0
        self._dc_initialized = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def level(self) -> float:
        return self._level

    @property
    def line_stats(self) -> tuple[int, int]:
        return self._lines, self._bad_lines

    @property
    def actual_samplerate(self) -> int:
        active_duration = self._duration - self._paused_total
        if active_duration <= 0:
            return 1000
        with self._lock:
            count = len(self._samples) or self._last_count
        return max(1, int(round(count / active_duration)))

    def start(self, port: str, baudrate: int) -> None:
        _require_serial()
        if self._recording:
            return
        self._samples = []
        self._lines = 0
        self._bad_lines = 0
        self._last_value = 0.0
        self._level = 0.0
        self._paused = False
        self._duration = 0.0
        self._paused_total = 0.0
        self._pause_started = 0.0
        self._last_count = 0
        self._stream_mode = None
        self._dc_offset = 0.0
        self._dc_initialized = False
        self._stop_event.clear()
        self._serial = serial.Serial(port=port, baudrate=int(baudrate), timeout=0.15)
        self._serial.reset_input_buffer()
        self._recording = True
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._reader_loop, name="SerialRecorder", daemon=True)
        self._thread.start()

    def _detect_stream_mode(self, buffer: bytearray) -> str:
        if len(buffer) < 4:
            return "binary"
        sample = bytes(buffer[: min(128, len(buffer))])
        text_chars = sum(1 for b in sample if b in (9, 10, 13) or 32 <= b <= 126)
        if text_chars / max(1, len(sample)) > 0.65:
            return "text"
        return "binary"

    def _append_sample(self, value: float) -> None:
        self._lines += 1
        self._last_value = float(value)
        # Уровень для UI
        self._level = min(1.0, abs(self._last_value) / 4096.0)
        if self._paused:
            return
        with self._lock:
            self._samples.append(self._last_value)

    def _consume_text_buffer(self, buffer: bytearray) -> None:
        while True:
            nl = buffer.find(b"\n")
            cr = buffer.find(b"\r")
            if nl < 0 and cr < 0:
                break
            candidates = [i for i in (nl, cr) if i >= 0]
            sep = min(candidates)
            line = buffer[:sep]
            del buffer[: sep + 1]
            if buffer[:1] in (b"\n", b"\r"):
                del buffer[:1]
            if not line:
                continue
            try:
                text = line.decode("utf-8", errors="ignore").strip()
            except Exception:
                self._bad_lines += 1
                continue
            val = parse_numeric_line(text)
            if val is None:
                self._bad_lines += 1
                continue
            self._append_sample(val)

    def _consume_binary_buffer(self, buffer: bytearray) -> None:
        if len(buffer) < 2:
            return

        num_bytes = (len(buffer) // 2) * 2
        chunk = buffer[:num_bytes]
        del buffer[:num_bytes]

        # Парсим как 16-битные беззнаковые
        adc_u16 = np.frombuffer(chunk, dtype="<u2")

        # Детект сбитых байтов USB (строго через int32 для исключения переполнения)
        if adc_u16.size > 10 and np.abs(np.diff(adc_u16.astype(np.int32))).mean() > 10000:
            # Смещение байтов: возвращаем всё, кроме 1-го байта, обратно в начало буфера
            buffer[:0] = chunk[1:]
            return

        if adc_u16.size > 0:
            # Центрирование (удаление DC Offset) на лету
            chunk_mean = adc_u16.mean()
            if not self._dc_initialized:
                self._dc_offset = chunk_mean
                self._dc_initialized = True
            else:
                # Плавная подстройка базовой линии (помогает, если АЦП плавает)
                self._dc_offset = 0.95 * self._dc_offset + 0.05 * chunk_mean

            pcm_centered = adc_u16.astype(np.float32) - self._dc_offset

            # Массовое обновление статистики
            self._lines += len(pcm_centered)
            self._last_value = float(pcm_centered[-1])
            self._level = min(1.0, abs(self._last_value) / 2048.0)  # Делим на 2048, т.к. волна центрирована

            # Массовое добавление данных в массив (гораздо быстрее цикла for!)
            if not self._paused:
                with self._lock:
                    self._samples.extend(pcm_centered.tolist())

    def _reader_loop(self) -> None:
        buffer = bytearray()
        while not self._stop_event.is_set():
            try:
                if self._serial is not None:
                    to_read = max(1, self._serial.in_waiting or 4096)
                    raw = self._serial.read(to_read)
                else:
                    raw = b""
            except Exception:
                self._bad_lines += 1
                time.sleep(0.01)
                continue

            if not raw:
                time.sleep(0.005)
                continue

            buffer.extend(raw)
            if self._stream_mode is None and len(buffer) >= 8:
                self._stream_mode = self._detect_stream_mode(buffer)

            mode = self._stream_mode or "binary"
            if mode == "text":
                self._consume_text_buffer(buffer)
            else:
                self._consume_binary_buffer(buffer)

    def pause(self) -> None:
        if self._recording and not self._paused:
            self._paused = True
            self._pause_started = time.perf_counter()

    def resume(self) -> None:
        if self._recording and self._paused:
            self._paused_total += time.perf_counter() - self._pause_started
            self._pause_started = 0.0
            self._paused = False

    def stop(self) -> np.ndarray:
        now = time.perf_counter()
        self._duration = now - self._start_time if self._start_time else 0.0
        if self._paused and self._pause_started:
            self._paused_total += now - self._pause_started
            self._pause_started = 0.0
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._serial is not None:
            try:
                self._serial.close()
            finally:
                self._serial = None
        self._recording = False
        self._paused = False
        with self._lock:
            data = np.asarray(self._samples, dtype=np.float32)
            self._last_count = len(self._samples)
            self._samples = []
        return data

    def snapshot(self, max_samples: int | None = None) -> np.ndarray:
        with self._lock:
            data = np.asarray(self._samples, dtype=np.float32)
        if max_samples and data.size > max_samples:
            return data[-max_samples:]
        return data