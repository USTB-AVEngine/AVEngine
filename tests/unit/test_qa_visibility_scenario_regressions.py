"""Current-API visibility scenarios adapted from coworker c47e532.

Original design and five canary scenarios: nbh, commit
c47e53293424376ea47ca3fda9c7087e64b13693.
"""

from __future__ import annotations

from collections.abc import Sequence
import importlib.util
from pathlib import Path

import numpy as np

from avengine.qa.pixel_visibility import compile_pixel_visibility_truth


HEIGHT = 64
WIDTH = 64
TARGET_SEMANTIC_ID = 10
TARGET_INSTANCE_ID = "source1"
FURNITURE_OCCLUDER_ID = "native_static_object::furniture_table_01"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OCCLUDER_TOOL_PATH = REPOSITORY_ROOT / "tools/qa/derive_native_occluder_evidence.py"
OCCLUDER_TOOL_SPEC = importlib.util.spec_from_file_location(
    "coworker_c47e532_occluder_tool", OCCLUDER_TOOL_PATH
)
assert OCCLUDER_TOOL_SPEC is not None and OCCLUDER_TOOL_SPEC.loader is not None
OCCLUDER_TOOL = importlib.util.module_from_spec(OCCLUDER_TOOL_SPEC)
OCCLUDER_TOOL_SPEC.loader.exec_module(OCCLUDER_TOOL)


def _context(
    pass_kind: str,
    *,
    frame_count: int,
    camera_pose_ids: Sequence[str],
    target_instance_id: str | None = None,
) -> dict:
    context = {
        "pass_kind": pass_kind,
        "renderer_backend": "spear_ue",
        "rgb_renderer_backend": "spear_ue",
        "camera_contract_id": "coworker_c47e532_regression_camera_v1",
        "semantic_id_namespace": "coworker_c47e532_regression_semantics_v1",
        "resolution_hw": [HEIGHT, WIDTH],
        "frame_indices": list(range(frame_count)),
        "camera_pose_ids": list(camera_pose_ids),
    }
    if target_instance_id is not None:
        context["target_instance_id"] = target_instance_id
    return context


def _footprint(*, left_entry: bool = False) -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    if left_entry:
        frame[16:48, 4:20] = TARGET_SEMANTIC_ID
    else:
        frame[16:48, 16:48] = TARGET_SEMANTIC_ID
    return frame


def _partial_visible(target: np.ndarray) -> np.ndarray:
    frame = np.zeros_like(target)
    target_pixels = target == TARGET_SEMANTIC_ID
    frame[target_pixels & (np.indices(target.shape)[0] >= HEIGHT // 2)] = (
        TARGET_SEMANTIC_ID
    )
    return frame


def _compile(
    normal_frames: Sequence[np.ndarray],
    target_only_frames: Sequence[np.ndarray],
    *,
    camera_pose_ids: Sequence[str] | None = None,
) -> dict:
    frame_count = len(normal_frames)
    poses = (
        list(camera_pose_ids)
        if camera_pose_ids is not None
        else [f"camera_pose_{index:03d}" for index in range(frame_count)]
    )
    return compile_pixel_visibility_truth(
        normal_semantic_masks=normal_frames,
        target_only_semantic_masks_by_instance={TARGET_INSTANCE_ID: target_only_frames},
        semantic_ids_by_instance={TARGET_INSTANCE_ID: TARGET_SEMANTIC_ID},
        normal_context=_context(
            "modal_scene", frame_count=frame_count, camera_pose_ids=poses
        ),
        target_only_contexts_by_instance={
            TARGET_INSTANCE_ID: _context(
                "target_only",
                frame_count=frame_count,
                camera_pose_ids=poses,
                target_instance_id=TARGET_INSTANCE_ID,
            )
        },
    )


def _states(truth: dict) -> list[str]:
    return [
        frame["state"] for frame in truth["per_instance"][TARGET_INSTANCE_ID]["frames"]
    ]


def test_c1_fully_visible_uses_current_pixel_truth_api() -> None:
    target = _footprint()
    truth = _compile([target.copy() for _ in range(5)], [target] * 5)
    assert _states(truth) == ["visible_clear"] * 5


def test_c2_partial_furniture_occlusion_has_one_occluder() -> None:
    target = _footprint()
    truth = _compile([_partial_visible(target) for _ in range(5)], [target] * 5)
    assert _states(truth) == ["visible_occluded"] * 5
    for frame in truth["per_instance"][TARGET_INSTANCE_ID]["frames"]:
        occluded_pixels = frame["target_pixels"] - frame["visible_pixels"]
        assert OCCLUDER_TOOL._admit_unique_occluder(
            occluded_pixels=occluded_pixels,
            grouped={FURNITURE_OCCLUDER_ID: occluded_pixels},
        ) == [FURNITURE_OCCLUDER_ID]


def test_c3_in_view_fully_occluded_keeps_target_footprint() -> None:
    target = _footprint()
    empty = np.zeros_like(target)
    truth = _compile([empty] * 5, [target] * 5)
    assert _states(truth) == ["fully_occluded"] * 5
    for frame in truth["per_instance"][TARGET_INSTANCE_ID]["frames"]:
        assert frame["target_pixels"] > 0
        assert frame["visible_pixels"] == 0


def test_c4_out_of_view_then_enters_from_left() -> None:
    target = _footprint(left_entry=True)
    empty = np.zeros_like(target)
    truth = _compile(
        [empty, empty, empty, target, target],
        [empty, empty, empty, target, target],
    )
    frames = truth["per_instance"][TARGET_INSTANCE_ID]["frames"]
    assert _states(truth) == [
        "out_of_view",
        "out_of_view",
        "out_of_view",
        "visible_clear",
        "visible_clear",
    ]
    assert frames[3]["target_centroid_xy_px"][0] < WIDTH / 2


def test_c5_camera_motion_full_occlusion_then_reappearance() -> None:
    target = _footprint()
    empty = np.zeros_like(target)
    poses = [f"moving_camera_pose_{index:03d}" for index in range(5)]
    truth = _compile(
        [target, empty, empty, empty, target],
        [target] * 5,
        camera_pose_ids=poses,
    )
    assert truth["camera_pose_ids"] == poses
    assert _states(truth) == [
        "visible_clear",
        "fully_occluded",
        "fully_occluded",
        "fully_occluded",
        "visible_clear",
    ]
    frames = truth["per_instance"][TARGET_INSTANCE_ID]["frames"]
    reappearance_frames = [
        frame["frame_index"]
        for previous, frame in zip(frames, frames[1:])
        if previous["state"] == "fully_occluded" and frame["state"] == "visible_clear"
    ]
    assert reappearance_frames == [4]
