"""Pure contract tests for the current-timeline native pixel adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "capture_qa_v3_timeline_pixel",
    REPOSITORY / "tools/qa/capture_qa_v3_timeline_pixel.py",
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def _readback(*, location: float = 0.0, rotation: float = 0.0) -> dict:
    pose = {
        "location_cm": [location, 2.0, 3.0],
        "rotation_deg": [0.0, 0.0, rotation],
    }
    return {
        "frame_index": 40,
        "declared_camera_pose_id": "frame=40;ue_pose=1,2,3,4",
        "camera": dict(pose),
        "actors": {"source1": dict(pose), "source2": dict(pose)},
        "animations": {},
    }


def test_default_selected_frames_cover_start_middle_and_end() -> None:
    assert TOOL._selected_indices(None) == [0, 37, 74]
    assert TOOL._selected_indices([40, 74]) == [40, 74]


@pytest.mark.parametrize("values", [[], [40, 40], [74, 40], [-1], [75]])
def test_invalid_selected_frames_fail_closed(values: list[int]) -> None:
    with pytest.raises(RuntimeError, match="frame-index"):
        TOOL._selected_indices(values)


def test_identical_normal_and_target_replays_have_zero_drift() -> None:
    normal = [_readback()]
    result = TOOL._maximum_pass_drift(
        normal, {"source1": [_readback()], "source2": [_readback()]}
    )
    assert result == {
        "maximum_location_drift_cm": 0.0,
        "maximum_rotation_drift_deg": 0.0,
    }


def test_target_replay_pose_drift_fails_closed() -> None:
    normal = [_readback()]
    with pytest.raises(RuntimeError, match="location drift"):
        TOOL._maximum_pass_drift(
            normal,
            {
                "source1": [_readback(location=0.001)],
                "source2": [_readback()],
            },
        )


def test_target_replay_declared_pose_mismatch_fails_closed() -> None:
    changed = _readback()
    changed["declared_camera_pose_id"] = "different"
    with pytest.raises(RuntimeError, match="different declared camera pose"):
        TOOL._maximum_pass_drift(
            [_readback()], {"source1": [changed], "source2": [_readback()]}
        )


def test_selected_frames_follow_ten_second_timeline() -> None:
    assert TOOL._selected_indices(None, frame_count=150) == [0, 75, 149]
    assert TOOL._selected_indices([80, 149], frame_count=150) == [80, 149]
    with pytest.raises(RuntimeError, match="frame-index"):
        TOOL._selected_indices([150], frame_count=150)
    assert TOOL._selected_indices(None, frame_count=1) == [0]


def test_pixel_camera_uses_the_authored_lens_and_resolution() -> None:
    assert TOOL._camera_settings({
        "resolution_hw": [480, 640], "hfov_degrees": 90,
    }) == {"height": 480, "width": 640, "hfov_deg": 90.0}
    with pytest.raises(RuntimeError, match="camera resolution"):
        TOOL._camera_settings({"resolution_hw": [0, 640], "hfov_degrees": 90})


def test_spear_extension_bootstrap_requires_one_existing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = tmp_path / "host-sdk"
    extension.mkdir()
    original = list(TOOL.sys.path)
    monkeypatch.setattr(TOOL.sys, "path", list(original))
    assert TOOL._preload_spear_extension(
        ["--spear-ext", str(extension)]
    ) == extension.resolve()
    assert TOOL.sys.path[1] == str(extension.resolve())
    with pytest.raises(RuntimeError, match="exactly once"):
        TOOL._preload_spear_extension(
            ["--spear-ext", str(extension), "--spear-ext", str(extension)]
        )
    with pytest.raises(RuntimeError, match="not a directory"):
        TOOL._preload_spear_extension(
            ["--spear-ext", str(tmp_path / "missing")]
        )
