from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from avengine.rooms.qa_evidence import (
    PLACEHOLDER_NONHUMAN_MINIMUM_COLOR_PIXELS,
    annotate_achieved_conditions_visibility,
    annotate_pixel_visibility_semantics,
    bbox_touches_frame_edge,
    build_pixel_appearance_review,
    inspect_registered_appearance,
    summarize_pixel_visibility_semantics,
)


def _coat_image(colors: list[list[int]], size: int = 48) -> np.ndarray:
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    for index, color in enumerate(colors):
        start = index * size // len(colors)
        end = (index + 1) * size // len(colors)
        rgb[:, start:end] = color
    return rgb


def test_placeholder_nonhuman_floor_is_human_order_of_magnitude() -> None:
    assert PLACEHOLDER_NONHUMAN_MINIMUM_COLOR_PIXELS == 512


def test_nearly_invisible_eight_pixel_cat_is_not_observable() -> None:
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[:] = [12, 18, 22]
    rgb[0, :8] = [190, 173, 110]
    mask = np.zeros((64, 64), dtype=bool)
    mask[0, :8] = True
    row = inspect_registered_appearance(rgb, mask, "standard_yellow", entity_kind="animal")
    assert row["visible_pixels"] == 8
    assert row["minimum_color_pixels"] == 512
    assert row["placeholder"] is True
    assert row["calibration"] == "placeholder_coarse_color_only"
    assert row["appearance_thresholds"]["calibration"] == "placeholder_nonhuman_appearance_v1"
    assert row["status"] == "not_observable"
    assert row["observed_value"] is None


def test_eight_white_tan_pixels_are_not_observable() -> None:
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[0, :4] = [235, 235, 235]
    rgb[0, 4:8] = [155, 80, 40]
    mask = np.zeros((32, 32), dtype=bool)
    mask[0, :8] = True
    row = inspect_registered_appearance(rgb, mask, "standard_white_tan", entity_kind="animal")
    assert row["visible_pixels"] == 8
    assert row["status"] == "not_observable"


def test_white_tan_ratio_rule_accepts_large_white_and_tan_mask() -> None:
    rgb = _coat_image([[235, 235, 235], [155, 80, 40]], size=48)
    row = inspect_registered_appearance(
        rgb, np.ones((48, 48), dtype=bool), "standard_white_tan", entity_kind="animal"
    )
    assert row["status"] == "pass"
    assert row["observed_value"] == "standard_white_tan"
    assert row["appearance_thresholds"]["label"] == "placeholder"
    assert row["appearance_thresholds"]["minimum_color_pixels"] == 512


def test_device_with_only_body_color_is_reviewed(tmp_path: Path) -> None:
    height, width = 32, 32
    semantic = {"source1": 11}
    truth = {
        "schema": "avengine_qa_pixel_visibility_truth_v1",
        "status": "computed_modal_target_only_v1",
        "renderer_backend": "test",
        "rgb_renderer_backend": "test",
        "resolution_hw": [height, width],
        "frame_indices": [0],
        "camera_pose_ids": ["camera:0"],
        "semantic_id_namespace": "test",
        "per_instance": {
            "source1": {
                "semantic_id": 11,
                "frames": [
                    {
                        "frame_index": 0,
                        "state": "visible_clear",
                        "target_bbox_xyxy_px": [0, 0, width, height],
                        "target_pixels": height * width,
                        "visible_pixels": height * width,
                        "visible_fraction": 1.0,
                        "occlusion_fraction": 0.0,
                    }
                ],
            }
        },
    }
    (tmp_path / "pixel_visibility_truth.json").write_text(json.dumps(truth), encoding="utf-8")
    modal = np.full((1, height, width), 11, dtype=np.uint32)
    np.savez(
        tmp_path / "native_pixel_masks_depth_authority_v1.npz",
        depth_derived_modal_semantic=modal,
        modal=modal,
        target_only_source1=modal,
    )
    rgb = np.full((1, height, width, 3), 240, dtype=np.uint8)
    np.save(tmp_path / "rgb.npy", rgb)
    plan = {
        "clock": {"frame_count": 1, "frame_rate_hz": 1, "sample_rate_hz": 4, "sample_count": 4},
        "visual_plan": {
            "actors": [
                {
                    "actor_id": "source1",
                    "asset_id": "bathtub_white",
                    "entity_class": "rigid_object",
                    "realized_attributes": {"body_color": "white", "form_factor": "freestanding"},
                }
            ]
        },
    }
    result = build_pixel_appearance_review(tmp_path, plan, frame_stride=1)
    actor = result["actors"]["source1"]
    assert actor["status"] == "reviewed"
    assert actor["value"] == "white"
    assert actor["attribute_field"] == "body_color"
    assert actor["appearance_field_used"] == "body_color"
    assert actor["appearance_source"] == "realized_attributes"
    assert "finish" in actor["searched_fields"]
    assert "body_color" in actor["searched_fields"]
    assert result["appearance_thresholds"]["label"] == "placeholder"
    assert result["appearance_thresholds"]["minimum_color_pixels"] == 512


def test_device_without_finish_or_body_color_stays_null(tmp_path: Path) -> None:
    height, width = 8, 8
    truth = {
        "schema": "avengine_qa_pixel_visibility_truth_v1",
        "status": "computed_modal_target_only_v1",
        "resolution_hw": [height, width],
        "frame_indices": [0],
        "per_instance": {
            "source1": {
                "semantic_id": 11,
                "frames": [
                    {
                        "frame_index": 0,
                        "state": "visible_clear",
                        "target_bbox_xyxy_px": [0, 0, width, height],
                        "target_pixels": 64,
                        "visible_pixels": 64,
                    }
                ],
            }
        },
    }
    (tmp_path / "pixel_visibility_truth.json").write_text(json.dumps(truth), encoding="utf-8")
    modal = np.full((1, height, width), 11, dtype=np.uint32)
    np.savez(
        tmp_path / "native_pixel_masks_depth_authority_v1.npz",
        modal=modal,
        target_only_source1=modal,
    )
    np.save(tmp_path / "rgb.npy", np.zeros((1, height, width, 3), dtype=np.uint8))
    plan = {
        "visual_plan": {
            "actors": [
                {
                    "actor_id": "source1",
                    "asset_id": "television",
                    "entity_class": "rigid_object",
                    "realized_attributes": {"form_factor": "flat_panel"},
                }
            ]
        }
    }
    result = build_pixel_appearance_review(tmp_path, plan, frame_stride=1)
    actor = result["actors"]["source1"]
    assert actor["status"] == "not_observable"
    assert actor["value"] is None
    assert actor["appearance_field_used"] is None
    assert actor["reason"] == "neither finish nor body_color is registered"


def test_finish_is_preferred_over_body_color(tmp_path: Path) -> None:
    height, width = 32, 32
    truth = {
        "schema": "avengine_qa_pixel_visibility_truth_v1",
        "status": "computed_modal_target_only_v1",
        "resolution_hw": [height, width],
        "frame_indices": [0],
        "per_instance": {
            "source1": {
                "semantic_id": 11,
                "frames": [
                    {
                        "frame_index": 0,
                        "state": "visible_clear",
                        "target_bbox_xyxy_px": [0, 0, width, height],
                        "target_pixels": height * width,
                        "visible_pixels": height * width,
                    }
                ],
            }
        },
    }
    (tmp_path / "pixel_visibility_truth.json").write_text(json.dumps(truth), encoding="utf-8")
    modal = np.full((1, height, width), 11, dtype=np.uint32)
    np.savez(
        tmp_path / "native_pixel_masks_depth_authority_v1.npz",
        modal=modal,
        target_only_source1=modal,
    )
    np.save(tmp_path / "rgb.npy", np.full((1, height, width, 3), 20, dtype=np.uint8))
    plan = {
        "visual_plan": {
            "actors": [
                {
                    "actor_id": "source1",
                    "asset_id": "speaker",
                    "entity_class": "rigid_object",
                    "realized_attributes": {"finish": "black_ash", "body_color": "white"},
                }
            ]
        }
    }
    result = build_pixel_appearance_review(tmp_path, plan, frame_stride=1)
    actor = result["actors"]["source1"]
    assert actor["appearance_field_used"] == "finish"
    assert actor["value"] == "black_ash"
    assert actor["status"] == "reviewed"


def test_bbox_touches_frame_edge_counts_truncation() -> None:
    resolution = [20, 20]
    frames = [
        {
            "frame_index": 0,
            "target_pixels": 10,
            "visible_pixels": 10,
            "target_bbox_xyxy_px": [0, 2, 4, 6],
            "state": "visible_clear",
        },
        {
            "frame_index": 1,
            "target_pixels": 16,
            "visible_pixels": 8,
            "target_bbox_xyxy_px": [5, 5, 15, 15],
            "state": "visible_occluded",
        },
        {
            "frame_index": 2,
            "target_pixels": 12,
            "visible_pixels": 0,
            "target_bbox_xyxy_px": [10, 10, 20, 20],
            "state": "fully_occluded",
        },
        {
            "frame_index": 3,
            "target_pixels": 0,
            "visible_pixels": 0,
            "target_bbox_xyxy_px": None,
            "state": "out_of_view",
        },
    ]
    assert bbox_touches_frame_edge([0, 2, 4, 6], resolution) is True
    assert bbox_touches_frame_edge([5, 5, 15, 15], resolution) is False
    assert bbox_touches_frame_edge([10, 10, 20, 20], resolution) is True
    assert bbox_touches_frame_edge(None, resolution) is False
    summary = summarize_pixel_visibility_semantics(frames, resolution_hw=resolution)
    assert summary["in_fov_frame_count"] == 3
    assert summary["visible_pixel_frames"] == 2
    assert summary["bbox_touches_frame_edge_frames"] == 2
    window = summarize_pixel_visibility_semantics(
        frames, resolution_hw=resolution, window_frames=[2, 4]
    )
    assert window["in_fov_frame_count"] == 1
    assert window["visible_pixel_frames"] == 0
    assert window["bbox_touches_frame_edge_frames"] == 1


def test_annotate_pixel_truth_and_achieved_conditions_keep_in_fov_meaning() -> None:
    truth = {
        "schema": "avengine_qa_pixel_visibility_truth_v1",
        "resolution_hw": [20, 20],
        "per_instance": {
            "source1": {
                "semantic_id": 11,
                "frames": [
                    {
                        "frame_index": 0,
                        "target_pixels": 40,
                        "visible_pixels": 0,
                        "target_bbox_xyxy_px": [0, 0, 8, 8],
                        "state": "fully_occluded",
                    },
                    {
                        "frame_index": 1,
                        "target_pixels": 40,
                        "visible_pixels": 20,
                        "target_bbox_xyxy_px": [4, 4, 12, 12],
                        "state": "visible_occluded",
                    },
                    {
                        "frame_index": 2,
                        "target_pixels": 0,
                        "visible_pixels": 0,
                        "target_bbox_xyxy_px": None,
                        "state": "out_of_view",
                    },
                ],
            }
        },
    }
    annotated = annotate_pixel_visibility_semantics(truth)
    source = annotated["per_instance"]["source1"]
    assert source["in_fov_frame_count"] == 2
    assert source["visible_pixel_frames"] == 1
    assert source["bbox_touches_frame_edge_frames"] == 1
    assert source["frames"][0]["in_fov"] is True
    assert source["frames"][0]["state"] == "fully_occluded"
    assert source["frames"][0]["bbox_touches_frame_edge"] is True
    assert "fully_occluded" in annotated["in_fov_definition"]
    achieved = {
        "anchor_event_measurements": [
            {
                "actor_id": "source1",
                "window_frames": [0, 3],
                "in_fov_frame_count": 2,
            }
        ]
    }
    updated = annotate_achieved_conditions_visibility(achieved, annotated)
    row = updated["anchor_event_measurements"][0]
    assert row["in_fov_frame_count"] == 2
    assert row["visible_pixel_frames"] == 1
    assert row["bbox_touches_frame_edge_frames"] == 1
    assert "visible_pixels > 0" in row["in_fov_definition"]
