"""Pure parsing tests for the packaged runtime sightline probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "probe_qa_v3_runtime_los_batch",
    REPOSITORY / "tools/qa/probe_qa_v3_runtime_los_batch.py",
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def test_required_frames_deduplicates_instant_binding_frame() -> None:
    assert TOOL.required_frames({"anchor_frame": 30, "query_frame": 30}) == [30]
    assert TOOL.required_frames({"anchor_frame": 62, "query_frame": 22}) == [22, 62]


def test_required_frames_rejects_out_of_episode_value() -> None:
    with pytest.raises(RuntimeError, match="fact frames"):
        TOOL.required_frames({"anchor_frame": 75, "query_frame": 0})


def test_trace_parser_accepts_bridge_key_case_and_reports_hit_point() -> None:
    result = TOOL.parse_trace_result({
        "returnValue": True,
        "outHit": {"location": {"x": 1.0, "Y": 2.0, "Z": 3.0}},
    })
    assert result == {"blocked": True, "hit_point_ue_cm": [1.0, 2.0, 3.0]}


def test_trace_parser_reports_clear_without_out_hit() -> None:
    assert TOOL.parse_trace_result({"ReturnValue": False}) == {"blocked": False}


def test_trace_parser_rejects_ambiguous_casefolded_key() -> None:
    with pytest.raises(RuntimeError, match="unique ReturnValue"):
        TOOL.parse_trace_result({"ReturnValue": False, "returnvalue": True})
