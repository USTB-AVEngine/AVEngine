"""docs/TOOL_INDEX.md must match the tools tree (regenerate on change)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_tool_index_is_current() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "build_tool_index.py"), "--check"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
