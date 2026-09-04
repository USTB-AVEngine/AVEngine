from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.assets.validate_animated_animal_closure import (
    AnimatedAnimalClosureError,
    blender_frame_components,
    parse_args,
    summarize_action_samples,
    validate_level_manifest,
    validate_retarget_manifest,
    write_report,
)


def _pose(x: float, *, angle: float = 0.0) -> dict[str, dict[str, list[float]]]:
    return {
        "root": {
            "location": [x, 0.0, 0.0],
            "quaternion": [0.0, 0.0, angle, 1.0],
        }
    }


def _level_manifest() -> dict[str, object]:
    return {
        "support_plane": {
            "foot_leaves": ["front_left", "hind_right"],
            "plane_source": "mesh-foot-bottoms",
            "dual_authority": {
                "agreement": {"passed": True},
                "fallback_used": False,
            },
            "maximum_residual_ratio_of_mesh_diagonal": 0.01,
            "maximum_reviewed_residual_ratio_of_mesh_diagonal": 0.02,
            "tilt_deg": 3.0,
            "maximum_tilt_deg": 30.0,
        },
        "output": {"path": "/tmp/leveled.glb"},
    }


def _retarget_manifest() -> dict[str, object]:
    return {
        "target": {"path": "/tmp/leveled.glb"},
        "source_motion": {
            "path": "/tmp/donor.glb",
            "geometry_used": False,
            "weights_used": False,
        },
        "export": {
            "path": "/tmp/animated.glb",
            "action_names": ["Walk", "Idle"],
        },
    }


def test_parser_exposes_custom_actions_and_optional_rig_counts() -> None:
    args = parse_args(
        [
            "animated.glb",
            "level.json",
            "retarget.json",
            "report.json",
            "--required-actions",
            "Walk",
            "Idle",
            "--expected-bone-count",
            "7",
        ]
    )
    assert args.required_actions == ["Walk", "Idle"]
    assert args.expected_bone_count == 7
    assert args.expected_vertex_group_count is None


def test_level_validation_does_not_assume_four_foot_leaves() -> None:
    result = validate_level_manifest(_level_manifest())
    assert result["foot_leaves"] == ["front_left", "hind_right"]
    assert result["agreement_passed"] is True
    assert result["fallback_used"] is False


def test_retarget_validation_uses_requested_action_names() -> None:
    result = validate_retarget_manifest(
        _retarget_manifest(), required_actions=("Walk", "Idle")
    )
    assert result["action_names"] == ["Walk", "Idle"]
    broken = _retarget_manifest()
    broken["source_motion"] = {
        "path": "/tmp/donor.glb",
        "geometry_used": True,
        "weights_used": False,
    }
    with pytest.raises(AnimatedAnimalClosureError, match="source geometry"):
        validate_retarget_manifest(broken, required_actions=("Walk", "Idle"))


def test_action_summary_requires_change_and_reports_closed_cycle() -> None:
    result = summarize_action_samples(
        "Walk",
        frame_range=(0.0, 2.0),
        sample_frames=(0.0, 1.0, 2.0),
        samples=(_pose(0.0), _pose(0.5, angle=0.2), _pose(0.0)),
    )
    assert result["cycle_closed"] is True
    assert result["max_sample_translation_delta"] == pytest.approx(0.5)
    assert result["max_first_last_translation_delta"] == pytest.approx(0.0)


def test_action_summary_rejects_open_cycle_unless_explicitly_allowed() -> None:
    kwargs = {
        "frame_range": (0.0, 2.0),
        "sample_frames": (0.0, 1.0, 2.0),
        "samples": (_pose(0.0), _pose(0.5), _pose(0.2)),
    }
    with pytest.raises(AnimatedAnimalClosureError, match="does not close"):
        summarize_action_samples("Walk", **kwargs)
    result = summarize_action_samples("Walk", require_closed_cycle=False, **kwargs)
    assert result["cycle_closed"] is False


def test_report_writer_is_fresh_and_json_serializable(tmp_path: Path) -> None:
    report_path = write_report(tmp_path / "report.json", {"status": "passed"})
    assert json.loads(report_path.read_text(encoding="utf-8")) == {"status": "passed"}
    with pytest.raises(AnimatedAnimalClosureError, match="refusing to overwrite"):
        write_report(report_path, {"status": "changed"})


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1.5, (1, 0.5)), (2.0, (2, 0.0)), (-0.25, (-1, 0.75))],
)
def test_fractional_blender_frames_use_integer_frame_and_subframe(value, expected):
    integer, subframe = blender_frame_components(value)
    assert integer == expected[0]
    assert subframe == pytest.approx(expected[1])
