#!/usr/bin/env python3
"""CPU room-selection stages, human review queues and house-held-out statistics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.rooms.room_selection.run import main

if __name__ == "__main__":
    main()
