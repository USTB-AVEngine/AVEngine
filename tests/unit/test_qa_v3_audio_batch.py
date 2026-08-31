"""Contract tests for the QA-v3 sequential audio runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_qa_v3_audio_batch",
    REPOSITORY / "tools/qa/run_qa_v3_audio_batch.py",
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def test_per_point_m1_request_overrides_legacy_batch_fallback(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    point = inputs / "card1F_001"
    point.mkdir(parents=True)
    per_point = point / "m1_capture_request.json"
    per_point.write_text("{}")
    fallback = tmp_path / "fallback.json"
    fallback.write_text("{}")
    assert TOOL.point_m1_request(inputs, "card1F_001", str(fallback)) == per_point


def test_legacy_batch_fallback_remains_available(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    fallback = tmp_path / "fallback.json"
    fallback.write_text("{}")
    assert TOOL.point_m1_request(inputs, "old_point", str(fallback)) == fallback


def test_missing_point_and_fallback_m1_request_fails(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    with pytest.raises(SystemExit, match="M1 request is missing"):
        TOOL.point_m1_request(inputs, "card1F_001", str(tmp_path / "absent.json"))
