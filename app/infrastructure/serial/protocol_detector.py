"""Serial stream protocol detection and text-line number parsing."""

from __future__ import annotations

import re
from typing import Literal

StreamMode = Literal["binary", "text"]

#: Number of leading bytes inspected by :func:`detect_stream_mode`.
DETECTION_WINDOW_BYTES: int = 128
#: Fraction of printable bytes above which the stream is considered text.
TEXT_RATIO_THRESHOLD: float = 0.65
#: Buffers shorter than this are too small to classify and are treated as binary.
MIN_DETECTION_BYTES: int = 4

_WHITESPACE_CONTROL: frozenset[int] = frozenset((9, 10, 13))  # \t \n \r

NUMBER_RE: re.Pattern[str] = re.compile(
    r"[-+]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][-+]?\d+)?"
)


def detect_stream_mode(buffer: bytes | bytearray) -> StreamMode:
    """Classify a serial byte stream as ``"binary"`` or ``"text"``.

    The first 128 bytes are examined; the stream is text when more than 65 % of
    them are printable ASCII (32-126) or tab / LF / CR.

    Args:
        buffer: Bytes received from the port so far.

    Returns:
        ``"text"`` or ``"binary"``. Buffers shorter than 4 bytes are ``"binary"``.
    """
    if len(buffer) < MIN_DETECTION_BYTES:
        return "binary"
    sample = bytes(buffer[:DETECTION_WINDOW_BYTES])
    printable = sum(1 for b in sample if b in _WHITESPACE_CONTROL or 32 <= b <= 126)
    return "text" if printable / len(sample) > TEXT_RATIO_THRESHOLD else "binary"


def parse_numeric_line(text: str) -> float | None:
    """Extract the first number (comma or dot decimal) from a text line.

    Returns:
        The parsed value, or None if the line contains no number.
    """
    match = NUMBER_RE.search(text.strip())
    if match is None:
        return None
    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None


__all__: list[str] = [
    "StreamMode",
    "NUMBER_RE",
    "detect_stream_mode",
    "parse_numeric_line",
]
