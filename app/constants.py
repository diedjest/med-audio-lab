"""Application-wide constants, enums, and configuration defaults for MedAudioLab v2."""

from __future__ import annotations

from enum import Enum, IntEnum


class SampleRate(IntEnum):
    """Standard audio and ultrasonic sampling rates in Hz."""

    SR_44100 = 44100
    SR_48000 = 48000
    SR_96000 = 96000
    SR_192000 = 192000
    SR_384000 = 384000
    SR_768000 = 768000

    # Convenient aliases
    RATE_44100 = 44100
    RATE_48000 = 48000
    RATE_96000 = 96000
    RATE_192000 = 192000
    RATE_384000 = 384000
    RATE_768000 = 768000

    @classmethod
    def _missing_(cls, value: object) -> SampleRate | None:
        """Handle string-based or alternative lookup formats."""
        if isinstance(value, str):
            clean = value.strip().upper().removeprefix("SR_").removeprefix("RATE_")
            for member in cls:
                if str(member.value) == clean or member.name == value:
                    return member
        return None

    @classmethod
    def as_list(cls) -> list[int]:
        """Return list of supported sampling rate integer values in ascending order."""
        return [44100, 48000, 96000, 192000, 384000, 768000]


class BitDepth(str, Enum):
    """Audio bit depth and sample encoding specification."""

    PCM_16 = "PCM_16"
    PCM_24 = "PCM_24"
    FLOAT_32 = "FLOAT_32"

    @property
    def soundfile_subtype(self) -> str:
        """Map enum value to libsndfile / soundfile subtype string."""
        if self == BitDepth.FLOAT_32:
            return "FLOAT"
        return self.value


class FilterType(str, Enum):
    """Digital filter frequency band topologies."""

    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"


class FilterDesign(str, Enum):
    """IIR filter approximation methods."""

    BUTTERWORTH = "butter"
    CHEBYSHEV_I = "cheby1"
    CHEBYSHEV_II = "cheby2"


# Application constants
APP_NAME: str = "MedAudioLab"
DEFAULT_BLOCK_SIZE: int = 1024
RING_BUFFER_CAPACITY: int = 1 << 20  # 1,048,576 samples
MAX_CWT_MEMORY_MB: float = 800.0
LIVE_UPDATE_INTERVAL_MS: int = 33  # ~30 Hz refresh rate

__all__: list[str] = [
    "SampleRate",
    "BitDepth",
    "FilterType",
    "FilterDesign",
    "APP_NAME",
    "DEFAULT_BLOCK_SIZE",
    "RING_BUFFER_CAPACITY",
    "MAX_CWT_MEMORY_MB",
    "LIVE_UPDATE_INTERVAL_MS",
]
