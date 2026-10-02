"""CPU room-selection protocol; all external scene inputs are read-only."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
PROTOCOL_VERSION = "room-selection-v1"
