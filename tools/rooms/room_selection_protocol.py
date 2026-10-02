#!/usr/bin/env python3
"""Register and measure the shared HM3D room-selection protocol on CPU."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.rooms.room_selection.run import main

if __name__ == "__main__":
    main()
