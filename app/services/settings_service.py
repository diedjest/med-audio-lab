"""Typed access to persisted user settings."""

from __future__ import annotations

from pathlib import Path

from app.core.interfaces.storage_repository import IStorageRepository

# "save_dir" is the key used by the legacy application; it is kept so existing
# databases continue to work.
KEY_SAVE_DIR = "save_dir"
KEY_LAST_DEVICE_INDEX = "last_device_index"
KEY_LAST_SAMPLE_RATE = "last_sample_rate"
KEY_LAST_SERIAL_PORT = "last_serial_port"
KEY_LAST_BAUDRATE = "last_baudrate"
KEY_LAST_SERIAL_ANALYSIS_RATE = "last_serial_analysis_rate"
KEY_LAST_SOURCE = "last_source"
KEY_WINDOW_GEOMETRY = "window_geometry"
KEY_WAV_BIT_DEPTH = "wav_bit_depth"

DEFAULT_WAV_BIT_DEPTH = "PCM_24"
VALID_WAV_BIT_DEPTHS: tuple[str, ...] = ("PCM_16", "PCM_24", "FLOAT_32")
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BAUDRATE = 921600
DEFAULT_SERIAL_ANALYSIS_RATE = 200_000
DEFAULT_SOURCE = "microphone"
VALID_SOURCES: tuple[str, ...] = ("microphone", "serial")


class SettingsService:
    """Reads and writes application settings through an ``IStorageRepository``.

    Getters never raise: a missing or corrupt stored value yields the default.
    Setters validate their input and raise ``ValueError`` for invalid values.
    """

    def __init__(self, repository: IStorageRepository) -> None:
        self._repository = repository

    # ------------------------------------------------------------ helpers
    def _get_positive_int(self, key: str, default: int) -> int:
        raw = self._repository.get_setting(key, "")
        try:
            value = int(float(raw))
        except (TypeError, ValueError, OverflowError):
            return default
        return value if value > 0 else default

    def _set_positive_int(self, key: str, value: int, label: str) -> None:
        if int(value) <= 0:
            raise ValueError(f"{label} must be positive, got {value}")
        self._repository.set_setting(key, str(int(value)))

    # --------------------------------------------------------- directories
    @property
    def app_data_dir(self) -> Path:
        """Root directory for application data (database, records, exports)."""
        return Path.home() / "Documents" / "MedAudioLab"

    @property
    def save_directory(self) -> Path:
        """Directory where recordings are saved (default: ``<app_data_dir>/records``)."""
        default = self.app_data_dir / "records"
        raw = self._repository.get_setting(KEY_SAVE_DIR, "").strip()
        return Path(raw).expanduser() if raw else default

    @save_directory.setter
    def save_directory(self, value: Path | str) -> None:
        text = str(value).strip()
        if not text:
            raise ValueError("save_directory must not be empty")
        self._repository.set_setting(KEY_SAVE_DIR, str(Path(text).expanduser()))

    @property
    def wav_bit_depth(self) -> str:
        """WAV sample format: ``"PCM_16"``, ``"PCM_24"`` (default) or ``"FLOAT_32"``."""
        value = self._repository.get_setting(KEY_WAV_BIT_DEPTH, DEFAULT_WAV_BIT_DEPTH)
        return value if value in VALID_WAV_BIT_DEPTHS else DEFAULT_WAV_BIT_DEPTH

    @wav_bit_depth.setter
    def wav_bit_depth(self, value: str) -> None:
        if value not in VALID_WAV_BIT_DEPTHS:
            raise ValueError(f"wav_bit_depth must be one of {VALID_WAV_BIT_DEPTHS}, got {value!r}")
        self._repository.set_setting(KEY_WAV_BIT_DEPTH, value)

    # ------------------------------------------------------------- devices
    @property
    def last_device_index(self) -> int | None:
        """Last used audio input device index, or None if unset."""
        raw = self._repository.get_setting(KEY_LAST_DEVICE_INDEX, "").strip()
        try:
            value = int(raw)
        except ValueError:
            return None
        return value if value >= 0 else None

    @last_device_index.setter
    def last_device_index(self, value: int | None) -> None:
        if value is None:
            self._repository.set_setting(KEY_LAST_DEVICE_INDEX, "")
            return
        if int(value) < 0:
            raise ValueError(f"last_device_index must be >= 0 or None, got {value}")
        self._repository.set_setting(KEY_LAST_DEVICE_INDEX, str(int(value)))

    @property
    def last_sample_rate(self) -> int:
        """Last used capture sample rate in Hz (default 44100)."""
        return self._get_positive_int(KEY_LAST_SAMPLE_RATE, DEFAULT_SAMPLE_RATE)

    @last_sample_rate.setter
    def last_sample_rate(self, value: int) -> None:
        self._set_positive_int(KEY_LAST_SAMPLE_RATE, value, "last_sample_rate")

    # -------------------------------------------------------------- serial
    @property
    def last_serial_port(self) -> str:
        """Last used serial port name (empty if unset)."""
        return self._repository.get_setting(KEY_LAST_SERIAL_PORT, "")

    @last_serial_port.setter
    def last_serial_port(self, value: str) -> None:
        self._repository.set_setting(KEY_LAST_SERIAL_PORT, str(value))

    @property
    def last_baudrate(self) -> int:
        """Last used serial baud rate (default 921600)."""
        return self._get_positive_int(KEY_LAST_BAUDRATE, DEFAULT_BAUDRATE)

    @last_baudrate.setter
    def last_baudrate(self, value: int) -> None:
        self._set_positive_int(KEY_LAST_BAUDRATE, value, "last_baudrate")

    @property
    def last_serial_analysis_rate(self) -> int:
        """Sample rate in Hz assumed when analysing serial data (default 200000)."""
        return self._get_positive_int(
            KEY_LAST_SERIAL_ANALYSIS_RATE, DEFAULT_SERIAL_ANALYSIS_RATE
        )

    @last_serial_analysis_rate.setter
    def last_serial_analysis_rate(self, value: int) -> None:
        self._set_positive_int(KEY_LAST_SERIAL_ANALYSIS_RATE, value, "last_serial_analysis_rate")

    # -------------------------------------------------------------- source
    @property
    def last_source(self) -> str:
        """Last selected signal source: ``"microphone"`` (default) or ``"serial"``."""
        value = self._repository.get_setting(KEY_LAST_SOURCE, DEFAULT_SOURCE)
        return value if value in VALID_SOURCES else DEFAULT_SOURCE

    @last_source.setter
    def last_source(self, value: str) -> None:
        if value not in VALID_SOURCES:
            raise ValueError(f"last_source must be one of {VALID_SOURCES}, got {value!r}")
        self._repository.set_setting(KEY_LAST_SOURCE, value)

    # -------------------------------------------------------------- window
    @property
    def window_geometry(self) -> str:
        """Hex-encoded main-window geometry from the last session ("" if none)."""
        return self._repository.get_setting(KEY_WINDOW_GEOMETRY, "")

    @window_geometry.setter
    def window_geometry(self, value: str) -> None:
        self._repository.set_setting(KEY_WINDOW_GEOMETRY, str(value))


__all__: list[str] = ["SettingsService"]
