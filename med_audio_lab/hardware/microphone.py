from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    import sounddevice as sd
except Exception as exc:  # pragma: no cover - shown in GUI
    sd = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    input_channels: int
    output_channels: int
    default_samplerate: int

    @property
    def label(self) -> str:
        if self.input_channels:
            kind = "микрофон"
        elif self.output_channels:
            kind = "вывод"
        else:
            kind = "устройство"
        return f"{self.index}: {self.name} ({kind}, {self.default_samplerate} Гц)"


class AudioUnavailableError(RuntimeError):
    pass


def _require_sounddevice():
    if sd is None:
        raise AudioUnavailableError(
            "Библиотека sounddevice не загрузилась. "
            f"Причина: {_IMPORT_ERROR!r}"
        )


def list_devices() -> list[AudioDevice]:
    _require_sounddevice()
    devices = []
    for idx, info in enumerate(sd.query_devices()):
        devices.append(
            AudioDevice(
                index=idx,
                name=str(info.get("name", f"Device {idx}")),
                input_channels=int(info.get("max_input_channels", 0)),
                output_channels=int(info.get("max_output_channels", 0)),
                default_samplerate=int(float(info.get("default_samplerate", 44100))),
            )
        )
    return devices


def input_devices() -> list[AudioDevice]:
    return [dev for dev in list_devices() if dev.input_channels > 0]


def output_devices() -> list[AudioDevice]:
    return [dev for dev in list_devices() if dev.output_channels > 0]


class Recorder:
    def __init__(self) -> None:
        self._stream = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._paused = False
        self._recording = False
        self._samplerate = 44100
        self._device_index: int | None = None
        self._level = 0.0

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def samplerate(self) -> int:
        return self._samplerate

    @property
    def level(self) -> float:
        return self._level

    def start(self, device_index: int | None, samplerate: int = 44100) -> None:
        _require_sounddevice()
        if self._recording:
            return
        self._chunks = []
        self._level = 0.0
        self._paused = False
        self._recording = True
        self._samplerate = int(samplerate)
        self._device_index = device_index

        def callback(indata, frames, time_info, status):
            del frames, time_info, status
            mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
            self._level = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
            if self._paused:
                return
            with self._lock:
                self._chunks.append(mono)

        self._stream = sd.InputStream(
            device=device_index,
            channels=1,
            samplerate=self._samplerate,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()

    def pause(self) -> None:
        if self._recording:
            self._paused = True

    def resume(self) -> None:
        if self._recording:
            self._paused = False

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._recording = False
        self._paused = False
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            data = np.concatenate(self._chunks).astype(np.float32, copy=False)
            self._chunks = []
        return data

    def snapshot(self, max_samples: int | None = None) -> np.ndarray:
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            if max_samples and max_samples > 0:
                remaining = int(max_samples)
                parts: list[np.ndarray] = []
                for chunk in reversed(self._chunks):
                    if remaining <= 0:
                        break
                    if chunk.size <= remaining:
                        parts.append(chunk)
                        remaining -= chunk.size
                    else:
                        parts.append(chunk[-remaining:])
                        break
                if not parts:
                    return np.zeros(0, dtype=np.float32)
                return np.concatenate(list(reversed(parts))).astype(np.float32, copy=False)
            return np.concatenate(self._chunks).astype(np.float32, copy=False)


class Player:
    def __init__(self) -> None:
        self._playing = False

    @property
    def is_playing(self) -> bool:
        return self._playing

    def play(self, data: np.ndarray, samplerate: int) -> None:
        _require_sounddevice()
        self.stop()
        if data.size == 0:
            raise AudioUnavailableError("Нет данных для воспроизведения.")
        self._playing = True
        sd.play(np.asarray(data, dtype=np.float32), int(samplerate), blocking=False)

    def stop(self) -> None:
        if sd is not None:
            sd.stop()
        self._playing = False


def first_input_samplerate(devices: Iterable[AudioDevice]) -> int:
    for dev in devices:
        if dev.input_channels > 0:
            return dev.default_samplerate or 44100
    return 44100
