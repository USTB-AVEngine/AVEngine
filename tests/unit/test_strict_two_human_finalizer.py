from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/finalize_strict_two_human_canary.py"
SPEC = importlib.util.spec_from_file_location("strict_two_human_finalizer", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def test_pending_scan_is_recursive_and_case_insensitive() -> None:
    assert TOOL._contains_pending({"nested": ["PENDING_NATIVE_CAPTURE"]})
    assert TOOL._contains_pending({"state": "pending_required"})
    assert not TOOL._contains_pending({"status": "pass", "formal_scene_count": 0})


def test_capture_nonzero_exit_fails_before_artifact_trust(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="exit code was non-zero"):
        TOOL._validate_capture(
            capture_root=tmp_path / "missing_capture",
            mixture_path=tmp_path / "missing.wav",
            capture_exit_code=1,
        )
