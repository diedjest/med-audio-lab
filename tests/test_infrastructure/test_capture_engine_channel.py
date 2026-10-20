"""CaptureEngine channel selection, with PortAudio replaced by a fake stream."""

from __future__ import annotations

from collections.abc import Iterator
from unittest import mock

import numpy as np
import pytest

from app.infrastructure.audio import capture_engine as ce

FS = 48000
FRAMES = 1024


class FakeStream:
    """Stands in for ``sounddevice.InputStream``; remembers how it was opened."""

    instances: list[FakeStream] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.samplerate = kwargs["samplerate"]
        self.active = True
        FakeStream.instances.append(self)

    def start(self) -> None: ...

    def stop(self) -> None:
        self.active = False

    def close(self) -> None: ...


@pytest.fixture
def engine() -> Iterator[ce.CaptureEngine]:
    FakeStream.instances.clear()
    with (
        mock.patch.object(ce, "sd") as sd,
        mock.patch.object(ce, "require_sounddevice"),
        mock.patch.object(ce.AudioDeviceManager, "is_wasapi", return_value=False),
    ):
        sd.InputStream = FakeStream
        sd.PortAudioError = RuntimeError
        capture = ce.CaptureEngine()
        yield capture
        capture.stop()


def two_channel_block() -> np.ndarray:
    """Channel 0 is zeros, channel 1 is ones; shape ``(frames, 2)`` like PortAudio delivers."""
    return np.stack([np.zeros(FRAMES, np.float32), np.ones(FRAMES, np.float32)], axis=1)


def test_selected_channel_is_recorded_as_mono(engine: ce.CaptureEngine) -> None:
    engine.start(0, FS, 1, channel=1)
    engine._callback(two_channel_block(), FRAMES, None, None)

    recording = engine.stop()

    assert recording.shape == (FRAMES,)
    assert np.all(recording == 1.0)


def test_first_channel_is_recorded_when_chosen(engine: ce.CaptureEngine) -> None:
    engine.start(0, FS, 1, channel=0)
    engine._callback(two_channel_block(), FRAMES, None, None)

    recording = engine.stop()

    assert recording.shape == (FRAMES,)
    assert np.all(recording == 0.0)


def test_stream_is_opened_with_enough_channels_for_the_choice(engine: ce.CaptureEngine) -> None:
    engine.start(0, FS, 1, channel=2)

    assert FakeStream.instances[-1].kwargs["channels"] == 3


def test_without_a_channel_all_channels_are_recorded(engine: ce.CaptureEngine) -> None:
    engine.start(0, FS, 2)
    engine._callback(two_channel_block(), FRAMES, None, None)

    recording = engine.stop()

    assert recording.shape == (FRAMES, 2)
    assert np.all(recording[:, 1] == 1.0)


def test_negative_channel_is_rejected(engine: ce.CaptureEngine) -> None:
    with pytest.raises(ValueError, match="channel"):
        engine.start(0, FS, 1, channel=-1)


def test_channel_choice_does_not_leak_into_the_next_recording(engine: ce.CaptureEngine) -> None:
    engine.start(0, FS, 1, channel=1)
    engine.stop()

    engine.start(0, FS, 2)
    engine._callback(two_channel_block(), FRAMES, None, None)

    assert engine.stop().shape == (FRAMES, 2)
