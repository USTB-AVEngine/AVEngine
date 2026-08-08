from __future__ import annotations

import numpy as np
import pytest

from avengine.qa.pixel_visibility import (
    PixelVisibilityError,
    bind_pixel_visibility_truth,
    compile_pixel_visibility_truth,
)


HEIGHT = 6
WIDTH = 8
TARGET_ID = 17


def _context(
    pass_kind: str,
    *,
    frame_count: int,
    renderer_backend: str = "spear_ue",
    rgb_renderer_backend: str | None = None,
    camera_pose_ids: list[str] | None = None,
    target_instance_id: str | None = None,
) -> dict:
    value = {
        "pass_kind": pass_kind,
        "renderer_backend": renderer_backend,
        "rgb_renderer_backend": rgb_renderer_backend or renderer_backend,
        "camera_contract_id": "camera_contract_test_v1",
        "semantic_id_namespace": "semantic_test_v1",
        "resolution_hw": [HEIGHT, WIDTH],
        "frame_indices": list(range(frame_count)),
        "camera_pose_ids": camera_pose_ids
        or [f"camera_pose_{index:03d}" for index in range(frame_count)],
    }
    if target_instance_id is not None:
        value["target_instance_id"] = target_instance_id
    return value


def _target_mask() -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    mask[1:5, 2:6] = TARGET_ID
    return mask


def _compile(normal: list[np.ndarray], target: list[np.ndarray]) -> dict:
    frame_count = len(normal)
    return compile_pixel_visibility_truth(
        normal_semantic_masks=normal,
        target_only_semantic_masks_by_instance={"source1": target},
        semantic_ids_by_instance={"source1": TARGET_ID},
        normal_context=_context("modal_scene", frame_count=frame_count),
        target_only_contexts_by_instance={
            "source1": _context(
                "target_only",
                frame_count=frame_count,
                target_instance_id="source1",
            )
        },
    )


def test_clear_partial_full_and_out_of_view_states() -> None:
    target = _target_mask()
    clear = target.copy()
    partial = np.zeros_like(target)
    partial[1:5, 2:4] = TARGET_ID
    absent = np.zeros_like(target)

    truth = _compile(
        [clear, partial, absent, absent],
        [target, target, target, absent],
    )
    frames = truth["per_instance"]["source1"]["frames"]
    assert [frame["state"] for frame in frames] == [
        "visible_clear",
        "visible_occluded",
        "fully_occluded",
        "out_of_view",
    ]
    assert frames[0]["visible_pixels"] == frames[0]["target_pixels"] == 16
    assert frames[0]["visible_fraction"] == pytest.approx(1.0)
    assert frames[0]["occlusion_fraction"] == pytest.approx(0.0)
    assert frames[1]["visible_pixels"] == 8
    assert frames[1]["target_pixels"] == 16
    assert frames[1]["visible_fraction"] == pytest.approx(0.5)
    assert frames[1]["occlusion_fraction"] == pytest.approx(0.5)
    assert frames[2]["visible_fraction"] == pytest.approx(0.0)
    assert frames[2]["occlusion_fraction"] == pytest.approx(1.0)
    assert frames[3]["visible_fraction"] is None
    assert frames[3]["occlusion_fraction"] is None
    assert truth["per_instance"]["source1"]["state_counts"] == {
        "out_of_view": 1,
        "visible_clear": 1,
        "visible_occluded": 1,
        "fully_occluded": 1,
    }


def test_modal_target_pixels_must_be_subset_of_target_only_footprint() -> None:
    target = _target_mask()
    shifted = target.copy()
    shifted[0, 0] = TARGET_ID
    with pytest.raises(PixelVisibilityError, match="not a subset"):
        _compile([shifted], [target])


def test_target_only_pass_rejects_foreign_semantic_ids() -> None:
    target = _target_mask()
    target[0, 0] = 99
    with pytest.raises(PixelVisibilityError, match="foreign semantic ids"):
        _compile([np.zeros_like(target)], [target])


def test_habitat_semantics_cannot_label_nonmatching_ue_rgb() -> None:
    target = _target_mask()
    with pytest.raises(PixelVisibilityError, match="Habitat labels"):
        compile_pixel_visibility_truth(
            normal_semantic_masks=[target],
            target_only_semantic_masks_by_instance={"source1": [target]},
            semantic_ids_by_instance={"source1": TARGET_ID},
            normal_context=_context(
                "modal_scene",
                frame_count=1,
                renderer_backend="habitat_sim",
                rgb_renderer_backend="spear_ue",
            ),
            target_only_contexts_by_instance={
                "source1": _context(
                    "target_only",
                    frame_count=1,
                    renderer_backend="habitat_sim",
                    target_instance_id="source1",
                )
            },
        )


def test_target_only_camera_pose_must_match_modal_pass() -> None:
    target = _target_mask()
    with pytest.raises(PixelVisibilityError, match="camera_pose_ids differs"):
        compile_pixel_visibility_truth(
            normal_semantic_masks=[target],
            target_only_semantic_masks_by_instance={"source1": [target]},
            semantic_ids_by_instance={"source1": TARGET_ID},
            normal_context=_context(
                "modal_scene", frame_count=1, camera_pose_ids=["pose_a"]
            ),
            target_only_contexts_by_instance={
                "source1": _context(
                    "target_only",
                    frame_count=1,
                    camera_pose_ids=["pose_b"],
                    target_instance_id="source1",
                )
            },
        )


def test_fact_binding_rejects_wrong_instance_resolution_and_pose() -> None:
    target = _target_mask()
    truth = _compile([target], [target])
    with pytest.raises(PixelVisibilityError, match="episode source slots"):
        bind_pixel_visibility_truth(
            truth,
            expected_instance_ids=["source2"],
            expected_frame_count=1,
            expected_resolution_hw=[HEIGHT, WIDTH],
            expected_camera_pose_ids=["camera_pose_000"],
        )
    with pytest.raises(PixelVisibilityError, match="resolution"):
        bind_pixel_visibility_truth(
            truth,
            expected_instance_ids=["source1"],
            expected_frame_count=1,
            expected_resolution_hw=[HEIGHT + 1, WIDTH],
            expected_camera_pose_ids=["camera_pose_000"],
        )
    with pytest.raises(PixelVisibilityError, match="SensorRigTrajectory"):
        bind_pixel_visibility_truth(
            truth,
            expected_instance_ids=["source1"],
            expected_frame_count=1,
            expected_resolution_hw=[HEIGHT, WIDTH],
            expected_camera_pose_ids=["different_pose"],
        )


def test_fact_binding_rejects_tampered_fraction_and_state_counts() -> None:
    target = _target_mask()
    truth = _compile([target], [target])
    truth["per_instance"]["source1"]["frames"][0]["visible_fraction"] = 0.5
    with pytest.raises(PixelVisibilityError, match="disagree with counts"):
        bind_pixel_visibility_truth(
            truth,
            expected_instance_ids=["source1"],
            expected_frame_count=1,
            expected_resolution_hw=[HEIGHT, WIDTH],
            expected_camera_pose_ids=["camera_pose_000"],
        )

    truth = _compile([target], [target])
    truth["per_instance"]["source1"]["state_counts"]["visible_clear"] = 0
    with pytest.raises(PixelVisibilityError, match="state_counts"):
        bind_pixel_visibility_truth(
            truth,
            expected_instance_ids=["source1"],
            expected_frame_count=1,
            expected_resolution_hw=[HEIGHT, WIDTH],
            expected_camera_pose_ids=["camera_pose_000"],
        )
