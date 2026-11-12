"""MedAudioLab module execution entry point (`python -m app`).

Delegates execution directly to run.py logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path when invoked via `python -m app`
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from run import main

if __name__ == "__main__":
    sys.exit(main())
