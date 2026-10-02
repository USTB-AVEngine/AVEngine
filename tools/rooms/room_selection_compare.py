#!/usr/bin/env python3
"""Compare shared room measurements against retained calibration evidence."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.rooms.room_selection.compare import main

if __name__ == "__main__":
    main()
