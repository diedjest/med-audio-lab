"""Runtime resource locator supporting both development mode and frozen PyInstaller bundles.

In development: resources are located at project_root/resources/
In frozen mode: resources are extracted to sys._MEIPASS/resources/
"""

from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    """Check if the application is running as a frozen PyInstaller bundle.

    Returns:
        True if running in a frozen bundle (sys.frozen is set and sys._MEIPASS exists),
        False otherwise.
    """
    return bool(getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"))


def _get_base_path() -> Path:
    """Get the base path for resource resolution.

    Returns:
        Base path as Path object.
    """
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parent.parent


def get_resource_dir() -> Path:
    """Get the absolute path to the resources root directory.

    Returns:
        Path pointing to the resources directory.
    """
    return _get_base_path() / "resources"


def get_resource_path(relative: str) -> Path:
    """Resolve a resource path that works in both development and frozen modes.

    Args:
        relative: Path relative to the resources/ directory.
                  Example: "wavelet_scales.txt" or "styles/light.qss"

    Returns:
        Absolute Path to the resource file.

    Raises:
        FileNotFoundError: If the resource doesn't exist.
    """
    base = _get_base_path()
    full_path = base / "resources" / relative
    if not full_path.exists():
        raise FileNotFoundError(
            f"Resource not found: {full_path}\n"
            f"Base path: {base}\n"
            f"Frozen: {is_frozen()}"
        )
    return full_path


__all__: list[str] = [
    "is_frozen",
    "get_resource_dir",
    "get_resource_path",
]
