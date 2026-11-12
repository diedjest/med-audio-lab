"""Tests for app.infrastructure.audio.ring_buffer.RingBuffer."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from app.constants import RING_BUFFER_CAPACITY
from app.infrastructure.audio.ring_buffer import RingBuffer


def ramp(start: int, stop: int) -> np.ndarray:
    """Consecutive integers as float32: any lost, repeated or reordered sample shows up."""
    return np.arange(start, stop, dtype=np.float32)


# ------------------------------------------------------------- write / read
def test_write_read_roundtrip() -> None:
    buf = RingBuffer(64)

    assert buf.write(ramp(0, 40)) == 40

    np.testing.assert_array_equal(buf.read_latest(40), ramp(0, 40))
    assert buf.total_written == 40
    assert buf.available == 40


def test_read_latest_returns_the_newest_samples_and_does_not_consume() -> None:
    buf = RingBuffer(64)
    buf.write(ramp(0, 40))

    np.testing.assert_array_equal(buf.read_latest(10), ramp(30, 40))
    np.testing.assert_array_equal(buf.read_latest(10), ramp(30, 40))  # peeking twice gives the same data
    assert buf.available == 40


def test_read_latest_asks_for_more_than_is_there() -> None:
    buf = RingBuffer(64)
    buf.write(ramp(0, 5))

    np.testing.assert_array_equal(buf.read_latest(1000), ramp(0, 5))


@pytest.mark.parametrize("count", [0, -3])
def test_read_latest_of_nothing(count: int) -> None:
    buf = RingBuffer(16)
    buf.write(ramp(0, 4))

    assert buf.read_latest(count).size == 0


def test_empty_buffer_reads_empty() -> None:
    buf = RingBuffer(16)

    assert buf.read_latest(8).size == 0
    assert buf.drain_all().size == 0
    assert buf.available == 0 and buf.total_written == 0


def test_writing_nothing_is_a_no_op() -> None:
    buf = RingBuffer(16)

    assert buf.write(np.array([], dtype=np.float32)) == 0
    assert buf.total_written == 0


@pytest.mark.parametrize("chunk", [1, 3, 5, 7, 8])
def test_reads_stay_correct_after_the_buffer_wraps(chunk: int) -> None:
    buf = RingBuffer(8)  # a tiny capacity so it wraps many times
    written = 0
    for _ in range(30):
        buf.write(ramp(written, written + chunk))
        written += chunk
        newest = min(8, written)
        np.testing.assert_array_equal(buf.read_latest(newest), ramp(written - newest, written))


def test_two_dimensional_mono_frames_are_accepted() -> None:
    buf = RingBuffer(16)

    buf.write(ramp(0, 6).reshape(-1, 1))  # PortAudio always delivers (frames, channels)

    np.testing.assert_array_equal(buf.read_latest(6), ramp(0, 6))


def test_multichannel_buffer_keeps_frames_together() -> None:
    buf = RingBuffer(8, channels=2)
    frames = np.stack([ramp(0, 12), -ramp(0, 12)], axis=1)

    buf.write(frames[:7])
    buf.write(frames[7:])  # wraps

    latest = buf.read_latest(8)
    assert latest.shape == (8, 2)
    np.testing.assert_array_equal(latest, frames[4:])
    assert buf.channels == 2


def test_dtype_is_configurable() -> None:
    buf = RingBuffer(16, dtype=np.int16)
    buf.write(np.arange(5, dtype=np.int16))

    out = buf.read_latest(5)

    assert out.dtype == np.int16 and buf.dtype == np.int16
    np.testing.assert_array_equal(out, np.arange(5))


def test_returned_arrays_are_copies() -> None:
    buf = RingBuffer(16)
    buf.write(ramp(0, 8))

    peek = buf.read_latest(8)
    peek[:] = -1.0

    np.testing.assert_array_equal(buf.read_latest(8), ramp(0, 8))


# ------------------------------------------------------------------- overflow
def test_one_oversized_write_keeps_only_the_latest_samples() -> None:
    buf = RingBuffer(16)

    buf.write(ramp(0, 100))

    assert buf.total_written == 100
    np.testing.assert_array_equal(buf.read_latest(1000), ramp(84, 100))
    np.testing.assert_array_equal(buf.drain_all(), ramp(84, 100))
    assert buf.dropped_samples == 100 - 16


def test_many_small_writes_without_a_reader_keep_only_the_latest_capacity() -> None:
    buf = RingBuffer(32)
    for start in range(0, 200, 5):
        buf.write(ramp(start, start + 5))

    np.testing.assert_array_equal(buf.read_latest(32), ramp(168, 200))
    np.testing.assert_array_equal(buf.drain_all(), ramp(168, 200))
    assert buf.dropped_samples == 200 - 32


def test_buffer_filled_exactly_is_not_mistaken_for_empty() -> None:
    buf = RingBuffer(16)

    buf.write(ramp(0, 16))

    assert buf.available == 16
    np.testing.assert_array_equal(buf.read_latest(16), ramp(0, 16))
    np.testing.assert_array_equal(buf.drain_all(), ramp(0, 16))


def test_reading_in_time_loses_nothing() -> None:
    buf = RingBuffer(16)
    drained = []
    for start in range(0, 160, 8):
        buf.write(ramp(start, start + 8))
        drained.append(buf.drain_all())

    np.testing.assert_array_equal(np.concatenate(drained), ramp(0, 160))
    assert buf.dropped_samples == 0


# -------------------------------------------------------------------- drain_all
def test_drain_returns_everything_once() -> None:
    buf = RingBuffer(64)
    buf.write(ramp(0, 20))

    np.testing.assert_array_equal(buf.drain_all(), ramp(0, 20))
    assert buf.available == 0
    assert buf.drain_all().size == 0


def test_drain_returns_only_what_arrived_since_the_last_drain() -> None:
    buf = RingBuffer(64)
    buf.write(ramp(0, 10))
    buf.drain_all()
    buf.write(ramp(10, 25))

    np.testing.assert_array_equal(buf.drain_all(), ramp(10, 25))


def test_peeking_does_not_disturb_draining() -> None:
    buf = RingBuffer(64)
    buf.write(ramp(0, 30))
    buf.read_latest(10)

    np.testing.assert_array_equal(buf.drain_all(), ramp(0, 30))
    np.testing.assert_array_equal(buf.read_latest(5), ramp(25, 30))  # the data is still there to look at


# ------------------------------------------------------------------------ reset
def test_reset_clears_everything() -> None:
    buf = RingBuffer(16)
    buf.write(ramp(0, 100))

    buf.reset()

    assert buf.total_written == 0 and buf.available == 0 and buf.dropped_samples == 0
    assert buf.read_latest(16).size == 0
    assert buf.drain_all().size == 0


def test_reset_buffer_is_reusable_and_old_data_is_gone() -> None:
    buf = RingBuffer(16)
    buf.write(ramp(100, 130))
    buf.reset()

    buf.write(ramp(0, 5))

    np.testing.assert_array_equal(buf.read_latest(16), ramp(0, 5))  # nothing of the earlier data shows through
    np.testing.assert_array_equal(buf.drain_all(), ramp(0, 5))


# --------------------------------------------------------------------- capacity
@pytest.mark.parametrize(
    ("requested", "expected"),
    [(1, 2), (2, 2), (3, 4), (5, 8), (16, 16), (17, 32), (1000, 1024), (1024, 1024), (1025, 2048)],
)
def test_capacity_is_rounded_up_to_a_power_of_two(requested: int, expected: int) -> None:
    buf = RingBuffer(requested)

    assert buf.capacity == expected
    assert buf.capacity & (buf.capacity - 1) == 0


def test_default_capacity_is_the_application_constant() -> None:
    assert RingBuffer().capacity == RING_BUFFER_CAPACITY == 1 << 20


@pytest.mark.parametrize("capacity", [0, -1, -1024])
def test_non_positive_capacity_is_rejected(capacity: int) -> None:
    with pytest.raises(ValueError, match="capacity"):
        RingBuffer(capacity)


def test_zero_channels_are_rejected() -> None:
    with pytest.raises(ValueError, match="channels"):
        RingBuffer(16, channels=0)


# ------------------------------------------------------------------- threading
def test_producer_and_consumer_threads_lose_and_repeat_nothing() -> None:
    total = 400_000
    buf = RingBuffer(1 << 20)  # larger than everything written: the consumer may be arbitrarily slow
    chunks: list[np.ndarray] = []
    peeks_ok = True
    done = threading.Event()

    def producer() -> None:
        sizes = [64, 480, 1024, 333]
        pos, i = 0, 0
        while pos < total:
            n = min(sizes[i % len(sizes)], total - pos)
            buf.write(ramp(pos, pos + n))
            pos += n
            i += 1
        done.set()

    def consumer() -> None:
        nonlocal peeks_ok
        while not done.is_set():
            data = buf.drain_all()
            if data.size:
                chunks.append(data)
            latest = buf.read_latest(1000)
            if latest.size > 1 and not np.all(np.diff(latest) == 1):
                peeks_ok = False  # a torn or reordered view of the newest samples

    threads = [threading.Thread(target=producer), threading.Thread(target=consumer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads)
    chunks.append(buf.drain_all())

    np.testing.assert_array_equal(np.concatenate(chunks), ramp(0, total))
    assert buf.dropped_samples == 0
    assert peeks_ok


def test_slow_consumer_gets_consistent_data_and_an_exact_loss_count() -> None:
    total = 600_000
    buf = RingBuffer(1 << 12)  # small: a consumer that falls behind is overrun
    chunks: list[np.ndarray] = []
    peeks_ok = True
    done = threading.Event()

    def producer() -> None:
        pos = 0
        while pos < total:
            n = min(480, total - pos)
            buf.write(ramp(pos, pos + n))
            pos += n
        done.set()

    def consumer() -> None:
        nonlocal peeks_ok
        while not done.is_set():
            data = buf.drain_all()
            if data.size:
                chunks.append(data)
            latest = buf.read_latest(1000)
            if latest.size > 1 and not np.all(np.diff(latest) == 1):
                peeks_ok = False

    threads = [threading.Thread(target=producer), threading.Thread(target=consumer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads)
    chunks.append(buf.drain_all())

    received = np.concatenate(chunks)
    assert np.all(np.diff(received) > 0)  # strictly in order, no repeats
    assert received.size + buf.dropped_samples == total  # every sample is either delivered or counted as lost
    assert received[-1] == total - 1  # the newest data always arrives
    assert peeks_ok
