"""Entry point for the MedAudioLab v2 desktop application."""

from __future__ import annotations

import sys


def main() -> int:
    """Run MedAudioLab v2 main window and handle KeyboardInterrupt gracefully.

    Returns:
        Exit code (0 for normal exit / graceful interrupt, non-zero on failure).
    """
    try:
        from app.ui.main_window import main as app_main

        exit_code = app_main()
        return 0 if exit_code is None else int(exit_code)
    except KeyboardInterrupt:
        print("\nMedAudioLab interrupted by user. Exiting...", file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())