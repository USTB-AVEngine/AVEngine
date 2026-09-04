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


def test_collision_presence_separates_an_empty_world_from_a_clear_sightline() -> None:
    """2026-09-03: the Kujiale baked-lit map answers every trace with a miss, so
    a miss cannot be read as "nothing blocks" until control traces have hit."""
    calls = []

    def collisionless(start, end, profile="BlockAll", complex_trace=True):
        calls.append((start, end))
        return {"ReturnValue": False}

    def solid_floor(start, end, profile="BlockAll", complex_trace=True):
        # only the downward control trace hits, which is enough
        hit = end["Z"] < start["Z"]
        return ({"ReturnValue": True,
                 "OutHit": {"Location": {"X": end["X"], "Y": end["Y"], "Z": 27.1}}}
                if hit else {"ReturnValue": False})

    points = [(100.0, 200.0, 174.2), (150.0, 250.0, 174.2)]
    empty = TOOL.collision_presence(None, collisionless, points)
    assert empty["collision_geometry_present"] is False
    assert empty["control_trace_hits"] == 0
    assert empty["control_trace_count"] == len(points) * 3
    assert len(calls) == len(points) * 3
    assert {row["kind"] for row in empty["control_traces"]} == {"down", "up", "sideways"}

    room = TOOL.collision_presence(None, solid_floor, points)
    assert room["collision_geometry_present"] is True
    assert room["control_trace_hits"] == len(points)
