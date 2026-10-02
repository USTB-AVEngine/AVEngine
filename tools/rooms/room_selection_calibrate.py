#!/usr/bin/env python3
"""Calibrate, freeze and evaluate the room-selection protocol once."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.rooms.room_selection.workflow import main

if __name__ == "__main__":
    main()
