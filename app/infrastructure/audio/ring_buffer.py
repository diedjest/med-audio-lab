"""Lock-free single-producer ring buffer for real-time audio capture.

The producer (PortAudio callback thread) never takes a lock and never allocates
array storage: ``write`` only performs ``np.copyto`` into the pre-allocated
buffer and then publishes the new write position with a single integer store.

Consumers (UI / accumulator threads) validate what they copied against the
producer's position afterwards, discarding any prefix the producer may have
overwritten while the copy was in progress (seqlock-style validation).
"""

from __future__ import annotations

import threading

import numpy as np

from app.constants import RING_BUFFER_CAPACITY


class RingBuffer:
    """Circular buffer backed by a pre-allocated NumPy array.

    Single producer (audio callback), any number of readers.  ``read_latest``
    is a non-consuming peek; ``drain_all`` consumes everything written since
    the previous drain (serialized between consumers by an internal lock that
    the producer never touches).

    Positions are absolute, monotonically increasing sample-frame counters;
    the physical index is ``position & mask`` (capacity is a power of two).
    """

    __slots__ = (
        "_buf",
        "_capacity",
        "_mask",
        "_channels",
        "_write_pos",
        "_read_pos",
        "_pending_end",
        "_dropped",
        "_consumer_lock",
    )

    def __init__(
        self,
        capacity: int = RING_BUFFER_CAPACITY,
        dtype: np.dtype | type = np.float32,
        channels: int = 1,
    ) -> None:
        """Create a buffer holding at least ``capacity`` frames.

        Args:
            capacity: Requested capacity in frames; rounded up to a power of two.
            dtype: Sample dtype of the backing array.
            channels: Channels per frame. 1 gives a 1D buffer, otherwise the
                buffer is 2D with shape ``(capacity, channels)``.
        """
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")

        self._capacity: int = 1 << max(1, (int(capacity) - 1).bit_length())
        self._mask: int = self._capacity - 1
        self._channels: int = int(channels)
        shape = (self._capacity,) if channels == 1 else (self._capacity, int(channels))
        self._buf: np.ndarray = np.zeros(shape, dtype=dtype)
        self._write_pos: int = 0
        self._read_pos: int = 0
        self._pending_end: int = 0
        self._dropped: int = 0
        self._consumer_lock = threading.Lock()

    # ------------------------------------------------------------------ props
    @property
    def capacity(self) -> int:
        """Capacity in frames (power of two)."""
        return self._capacity

    @property
    def channels(self) -> int:
        """Channels per frame."""
        return self._channels

    @property
    def dtype(self) -> np.dtype:
        """Sample dtype of the backing array."""
        return self._buf.dtype

    @property
    def total_written(self) -> int:
        """Total frames written since the last reset."""
        return self._write_pos

    @property
    def available(self) -> int:
        """Frames written but not yet consumed by ``drain_all`` (capped at capacity)."""
        return min(self._write_pos - self._read_pos, self._capacity)

    @property
    def dropped_samples(self) -> int:
        """Frames lost because the consumer fell more than one capacity behind."""
        return self._dropped

    # --------------------------------------------------------------- producer
    def write(self, data: np.ndarray) -> int:
        """Copy frames into the buffer. Called from the audio callback thread.

        Lock-free and allocation-free apart from slice views.  Input larger
        than the capacity keeps only its most recent ``capacity`` frames.

        Args:
            data: Array of shape ``(n,)`` or ``(n, channels)`` matching the buffer
                (``(n, 1)`` is accepted for a mono buffer).

        Returns:
            Number of frames stored.
        """
        if data.ndim == 2 and self._channels == 1:
            data = data[:, 0]  # PortAudio always delivers (frames, channels)
        n = data.shape[0]
        if n == 0:
            return 0

        cap = self._capacity
        pos = self._write_pos
        if n > cap:
            pos += n - cap  # skipped frames still advance the timeline
            data = data[-cap:]
            n = cap
        self._pending_end = pos + n  # announce the region about to be overwritten

        buf = self._buf
        start = pos & self._mask
        end = start + n
        if end <= cap:
            np.copyto(buf[start:end], data)
        else:
            first = cap - start
            np.copyto(buf[start:], data[:first])
            np.copyto(buf[: n - first], data[first:])

        self._write_pos = pos + n  # publish only after the copy is complete
        return n

    # ---------------------------------------------------------------- readers
    def _empty(self) -> np.ndarray:
        return np.zeros((0,) + self._buf.shape[1:], dtype=self._buf.dtype)

    def _copy_range(self, start: int, end: int) -> np.ndarray:
        """Copy absolute positions ``[start, end)``; ``end - start <= capacity``."""
        n = end - start
        s = start & self._mask
        e = s + n
        if e <= self._capacity:
            return self._buf[s:e].copy()
        first = self._capacity - s
        out = np.empty((n,) + self._buf.shape[1:], dtype=self._buf.dtype)
        out[:first] = self._buf[s:]
        out[first:] = self._buf[: n - first]
        return out

    def _lowest_stable_position(self) -> int:
        """Lowest absolute position the producer has not started overwriting.

        Must be read *after* copying: any write that could have torn the copy
        announced its end position in ``_pending_end`` before touching the buffer.
        """
        return max(0, self._pending_end - self._capacity)

    def read_latest(self, count: int) -> np.ndarray:
        """Return the most recent ``count`` frames without consuming them.

        Returns fewer frames if less is available (or if the oldest frames were
        overwritten mid-copy); an empty array if nothing was written yet.
        """
        w = self._write_pos
        n = min(int(count), w, self._capacity)
        if n <= 0:
            return self._empty()

        start = w - n
        out = self._copy_range(start, w)
        lowest = self._lowest_stable_position()
        if lowest > start:
            out = out[min(lowest - start, n):]
        return out

    def drain_all(self) -> np.ndarray:
        """Consume and return every frame written since the previous drain.

        If the consumer fell more than one capacity behind, the oldest frames
        are gone and counted in ``dropped_samples``.
        """
        with self._consumer_lock:
            w = self._write_pos
            r = self._read_pos
            if w <= r:
                return self._empty()

            start = max(r, w - self._capacity)
            lost = start - r
            out = self._copy_range(start, w)
            lowest = self._lowest_stable_position()
            if lowest > start:
                cut = min(lowest - start, w - start)
                lost += cut
                out = out[cut:]

            self._dropped += lost
            self._read_pos = w
            return out

    def reset(self) -> None:
        """Clear the buffer. Only call while the producer is stopped."""
        with self._consumer_lock:
            self._write_pos = 0
            self._read_pos = 0
            self._pending_end = 0
            self._dropped = 0
            self._buf[...] = 0


__all__: list[str] = ["RingBuffer"]
