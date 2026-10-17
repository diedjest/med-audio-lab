"""PortAudio device enumeration and capability probing."""

from __future__ import annotations

from typing import Any, Literal

from app.constants import SampleRate
from app.core.models.device_info import AudioDeviceInfo

_sd_import_error: Exception | None = None
try:
    import sounddevice as sd
except (ImportError, OSError) as _exc:  # PortAudio library missing / not loadable
    sd = None  # type: ignore[assignment]
    _sd_import_error = _exc


class AudioUnavailableError(RuntimeError):
    """Raised when the sounddevice / PortAudio backend cannot be used."""


def require_sounddevice() -> None:
    """Raise ``AudioUnavailableError`` if sounddevice failed to import."""
    if sd is None:
        raise AudioUnavailableError(
            f"sounddevice / PortAudio is unavailable: {_sd_import_error!r}"
        )


class AudioDeviceManager:
    """Enumerates and validates audio hardware via PortAudio."""

    STANDARD_RATES: tuple[int, ...] = tuple(rate.value for rate in SampleRate)

    @staticmethod
    def _describe(index: int, info: Any, hostapis: Any) -> AudioDeviceInfo:
        """Build an ``AudioDeviceInfo`` from a PortAudio device dictionary."""
        try:
            hostapi = str(hostapis[int(info["hostapi"])]["name"])
        except (KeyError, IndexError, TypeError, ValueError):
            hostapi = ""
        return AudioDeviceInfo(
            index=index,
            name=str(info["name"]),
            input_channels=int(info.get("max_input_channels", 0)),
            output_channels=int(info.get("max_output_channels", 0)),
            default_sample_rate=int(float(info.get("default_samplerate", 44100))),
            hostapi=hostapi,
        )

    @staticmethod
    def _list_devices(channel_key: str) -> list[AudioDeviceInfo]:
        require_sounddevice()
        hostapis = sd.query_hostapis()
        return [
            AudioDeviceManager._describe(idx, info, hostapis)
            for idx, info in enumerate(sd.query_devices())
            if int(info.get(channel_key, 0)) > 0
        ]

    @staticmethod
    def list_input_devices() -> list[AudioDeviceInfo]:
        """Return all devices with at least one input channel, WASAPI ones first.

        MME and DirectSound hand out 16-bit audio, so the WASAPI entries (which
        deliver the full resolution of the Windows sound settings) lead the list.
        """
        devices = AudioDeviceManager._list_devices("max_input_channels")
        return sorted(devices, key=lambda d: "WASAPI" not in d.hostapi.upper())

    @staticmethod
    def is_wasapi(device_index: int) -> bool:
        """True if ``device_index`` belongs to the Windows WASAPI host API."""
        require_sounddevice()
        try:
            info = sd.query_devices(device_index)
            return "WASAPI" in str(sd.query_hostapis(int(info["hostapi"]))["name"]).upper()
        except (sd.PortAudioError, KeyError, ValueError, TypeError):
            return False

    @staticmethod
    def default_input_index() -> int | None:
        """Default input device, preferring WASAPI (full bit depth) over the MME default."""
        require_sounddevice()
        try:
            for api in sd.query_hostapis():
                if "WASAPI" in str(api["name"]).upper() and int(api["default_input_device"]) >= 0:
                    return int(api["default_input_device"])
        except (sd.PortAudioError, KeyError, ValueError, TypeError):
            pass
        return None

    @staticmethod
    def list_output_devices() -> list[AudioDeviceInfo]:
        """Return all devices with at least one output channel."""
        return AudioDeviceManager._list_devices("max_output_channels")

    @staticmethod
    def supported_rates(
        device_index: int,
        kind: Literal["input", "output"] = "input",
    ) -> list[int]:
        """Probe which standard sample rates the device actually supports.

        Args:
            device_index: PortAudio device index.
            kind: Probe the device as an ``"input"`` (default) or ``"output"``.

        Returns:
            Supported rates from ``STANDARD_RATES`` in ascending order.
        """
        require_sounddevice()
        check = sd.check_input_settings if kind == "input" else sd.check_output_settings
        supported: list[int] = []
        for rate in AudioDeviceManager.STANDARD_RATES:
            try:
                check(device=device_index, samplerate=rate)
            except (sd.PortAudioError, ValueError):
                continue
            supported.append(rate)
        return supported


__all__: list[str] = [
    "AudioDeviceManager",
    "AudioUnavailableError",
    "require_sounddevice",
]
