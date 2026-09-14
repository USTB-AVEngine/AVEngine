from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
import time

import numpy as np

from avengine.rooms.qa_evidence import (
    PLACEHOLDER_NONHUMAN_MINIMUM_COLOR_PIXELS,
    REGISTERED_APPEARANCE_CLASSIFIER_GAP_REASON,
    acquire_shared_visual_evidence,
    annotate_achieved_conditions_visibility,
    annotate_pixel_visibility_semantics,
    appearance_review_frame_selection,
    bbox_touches_frame_edge,
    build_pixel_appearance_review,
    build_shared_visual_evidence,
    capture_visual_input_identities,
    inspect_registered_appearance,
    nonhuman_appearance_placeholder_thresholds,
    summarize_pixel_visibility_semantics,
    verify_shared_visual_evidence,
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


def test_values_the_registry_carries_now_have_a_classifier() -> None:
    """Every value an asset may register is either observed or named as a gap."""
    rgb = np.full((160, 200, 3), 150, dtype=np.uint8)
    rgb[30:150, 40:160] = (208, 206, 202)
    mask = np.zeros((160, 200), dtype=bool)
    mask[30:150, 40:160] = True
    implemented = (
        ("silver", "device"), ("white_satin", "device"), ("warm_gray", "device"),
        ("beige", "device"), ("sandstone", "device"), ("light_gray_fabric", "device"),
        ("standard_seal_point", "animal"),
    )
    for value, kind in implemented:
        row = inspect_registered_appearance(rgb, mask, value, entity_kind=kind)
        assert row.get("reason") != REGISTERED_APPEARANCE_CLASSIFIER_GAP_REASON, value
        assert row.get("gap_category") != "interface_not_implemented", value
    unknown = inspect_registered_appearance(rgb, mask, "lunar_opal", entity_kind="device")
    assert unknown["status"] == "not_observable"
    assert unknown["reason"] == REGISTERED_APPEARANCE_CLASSIFIER_GAP_REASON
    assert unknown["gap_category"] == "interface_not_implemented"


def test_actor_reason_is_classifier_gap_when_value_has_no_classifier(tmp_path: Path) -> None:
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
    np.save(tmp_path / "rgb.npy", np.full((1, height, width, 3), 180, dtype=np.uint8))
    plan = {
        "visual_plan": {
            "actors": [
                {
                    "actor_id": "source1",
                    "asset_id": "blender_silver",
                    "entity_class": "rigid_object",
                    "realized_attributes": {"body_color": "lunar_opal", "form_factor": "jug"},
                }
            ]
        }
    }
    result = build_pixel_appearance_review(tmp_path, plan, frame_stride=1)
    actor = result["actors"]["source1"]
    assert actor["status"] == "not_observable"
    assert actor["value"] == "lunar_opal"
    assert actor["reason"] == "registered_appearance_value_classifier_not_implemented"
    assert actor["gap_category"] == "interface_not_implemented"
    assert actor["checks"]
    assert actor["checks"][0]["reason"] == "registered_appearance_value_classifier_not_implemented"


# --- shared visual evidence for audio members that reuse one capture ------


def _shared_capture(
    root: Path, *, coat: str = "standard_tricolor", occluded_target_columns: int = 0,
) -> dict:
    """A minimal capture whose appearance review has real pixels to observe."""
    root.mkdir(parents=True, exist_ok=True)
    height = width = 32
    frame_indices = [0, 1]
    semantic = {"source1": 11, "source2": 22}
    truth = {
        "schema": "avengine_qa_pixel_visibility_truth_v1",
        "status": "computed_modal_target_only_v1",
        "resolution_hw": [height, width],
        "frame_indices": frame_indices,
        "per_instance": {
            actor: {
                "semantic_id": sid,
                "frames": [
                    {
                        "frame_index": index,
                        "state": "visible_clear",
                        "target_bbox_xyxy_px": [0, 0, width, height],
                        "target_pixels": height * (width // 2),
                        "visible_pixels": height * (width // 2),
                        "visible_fraction": 1.0,
                        "occlusion_fraction": 0.0,
                    }
                    for index in frame_indices
                ],
            }
            for actor, sid in semantic.items()
        },
    }
    (root / "pixel_visibility_truth.json").write_text(json.dumps(truth), encoding="utf-8")
    modal = np.zeros((len(frame_indices), height, width), dtype=np.uint32)
    modal[:, :, :16] = 11
    modal[:, :, 16:] = 22
    target_only = {f"target_only_{actor}": modal == sid for actor, sid in semantic.items()}
    if occluded_target_columns:
        # source1 really extends behind source2: its target-only footprint
        # reaches columns the modal pass awards to source2, which is what an
        # identified actor occlusion looks like.
        extended = np.zeros_like(modal, dtype=bool)
        extended[:, :, : 16 + int(occluded_target_columns)] = True
        target_only["target_only_source1"] = extended
    np.savez(
        root / "native_pixel_masks_depth_authority_v1.npz",
        depth_derived_modal_semantic=modal,
        modal=modal,
        **target_only,
    )
    rgb = np.zeros((len(frame_indices), height, width, 3), dtype=np.uint8)
    rgb[:, :, :4] = [230, 230, 230]
    rgb[:, :, 4:8] = [120, 70, 40]
    rgb[:, :, 8:16] = [25, 25, 25]
    rgb[:, :, 16:] = [28, 18, 10]
    np.save(root / "rgb.npy", rgb)
    plan = {
        "clock": {"frame_count": 2, "frame_rate_hz": 2, "sample_rate_hz": 4, "sample_count": 4},
        "visual_plan": {
            "actors": [
                {
                    "actor_id": "source1",
                    "asset_id": "beagle",
                    "entity_class": "articulated_animal",
                    "realized_attributes": {"coat_profile": {"value": coat}},
                },
                {
                    "actor_id": "source2",
                    "asset_id": "speaker",
                    "entity_class": "rigid_object",
                    "realized_attributes": {"finish": "black_ash"},
                },
            ]
        },
    }
    return {"plan": plan, "truth": truth}


def test_two_audio_members_share_one_visual_pack(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    fixture = _shared_capture(capture)
    shared = tmp_path / "shared"
    first = acquire_shared_visual_evidence(
        capture, fixture["plan"], fixture["truth"], shared_root=shared, frame_stride=1
    )
    second = acquire_shared_visual_evidence(
        capture, fixture["plan"], fixture["truth"], shared_root=shared, frame_stride=1
    )
    assert first["reuse"]["status"] == "built_and_published"
    assert second["reuse"]["status"] == "reused"
    # The reused products are the same evidence, not a re-derivation.
    assert second["appearance_review"] == first["appearance_review"]
    assert second["actor_occluders"] == first["actor_occluders"]
    assert second["annotated_pixel_visibility_truth"] == first["annotated_pixel_visibility_truth"]
    # Reuse re-observed real pixels rather than trusting the stored answer.
    assert second["reuse"]["verification"]["status"] == "pass"
    assert second["reuse"]["verification"]["reobserved_appearance_checks"]
    assert second["build_total_s"] == 0.0
    assert first["stage_timings_s"]["build_pixel_appearance_review_s"] > 0


def test_changed_capture_pixels_are_never_reused(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    fixture = _shared_capture(capture)
    shared = tmp_path / "shared"
    acquire_shared_visual_evidence(
        capture, fixture["plan"], fixture["truth"], shared_root=shared, frame_stride=1
    )
    # Re-render the same capture with a different coat in the actual pixels.
    rgb = np.load(capture / "rgb.npy")
    rgb[:, :, :16] = [40, 90, 200]
    np.save(capture / "rgb.npy", rgb)
    again = acquire_shared_visual_evidence(
        capture, fixture["plan"], fixture["truth"], shared_root=shared, frame_stride=1
    )
    assert again["reuse"]["status"] == "built_and_published"
    assert again["reuse"]["pack_dir"] != ""


def test_changed_asset_binding_is_never_reused(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    fixture = _shared_capture(capture)
    shared = tmp_path / "shared"
    first = acquire_shared_visual_evidence(
        capture, fixture["plan"], fixture["truth"], shared_root=shared, frame_stride=1
    )
    rebound = deepcopy(fixture["plan"])
    rebound["visual_plan"]["actors"][0]["realized_attributes"]["coat_profile"]["value"] = "black_white"
    second = acquire_shared_visual_evidence(
        capture, rebound, fixture["truth"], shared_root=shared, frame_stride=1
    )
    assert second["reuse"]["status"] == "built_and_published"
    assert second["key"]["asset_bindings"] != first["key"]["asset_bindings"]


def test_changed_thresholds_are_never_reused(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    fixture = _shared_capture(capture)
    shared = tmp_path / "shared"
    acquire_shared_visual_evidence(
        capture, fixture["plan"], fixture["truth"], shared_root=shared, frame_stride=1
    )
    second = acquire_shared_visual_evidence(
        capture, fixture["plan"], fixture["truth"], shared_root=shared, frame_stride=1,
        thresholds=nonhuman_appearance_placeholder_thresholds(minimum_color_pixels=7),
    )
    assert second["reuse"]["status"] == "built_and_published"


def test_verification_rejects_a_pack_whose_recorded_observation_is_wrong(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    fixture = _shared_capture(capture)
    pack = build_shared_visual_evidence(capture, fixture["plan"], fixture["truth"], frame_stride=1)
    assert verify_shared_visual_evidence(pack, capture)["status"] == "pass"
    # A registered label must never stand in for the observation it claims.
    tampered = deepcopy(pack)
    check = tampered["appearance_review"]["actors"]["source1"]["checks"][0]
    check["visible_pixels"] = int(check["visible_pixels"]) + 1
    rejected = verify_shared_visual_evidence(tampered, capture)
    assert rejected["status"] == "rejected"
    assert rejected["field"] == "visible_pixels"


def test_verification_rejects_a_pack_whose_occluder_count_is_wrong(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    fixture = _shared_capture(capture, occluded_target_columns=8)
    truth = deepcopy(fixture["truth"])
    for record in truth["per_instance"].values():
        for frame in record["frames"]:
            frame["state"] = "visible_occluded"
    pack = build_shared_visual_evidence(capture, fixture["plan"], truth, frame_stride=1)
    records = pack["actor_occluders"]["frame_records"]
    assert records, "fixture should produce at least one identified occluder record"
    assert verify_shared_visual_evidence(pack, capture)["status"] == "pass"
    tampered = deepcopy(pack)
    tampered["actor_occluders"]["frame_records"][0]["occluded_target_pixels"] += 3
    rejected = verify_shared_visual_evidence(tampered, capture)
    assert rejected["status"] == "rejected"
    assert "occluded pixel count" in rejected["reason"]


def test_no_shared_root_still_builds_the_same_evidence(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    fixture = _shared_capture(capture)
    unshared = acquire_shared_visual_evidence(
        capture, fixture["plan"], fixture["truth"], shared_root=None, frame_stride=1
    )
    direct = build_shared_visual_evidence(capture, fixture["plan"], fixture["truth"], frame_stride=1)
    assert unshared["reuse"]["status"] == "built"
    assert unshared["reuse"]["shared"] is False
    assert unshared["appearance_review"] == direct["appearance_review"]
    assert unshared["actor_occluders"] == direct["actor_occluders"]


def test_frame_selection_is_the_rule_the_review_actually_uses(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    fixture = _shared_capture(capture)
    every = appearance_review_frame_selection(fixture["truth"], fixture["plan"], frame_stride=1)
    assert every == [0, 1]
    strided = appearance_review_frame_selection(fixture["truth"], fixture["plan"], frame_stride=2)
    # The stride walk still keeps the retained last frame.
    assert strided == [0, 1]
    review = build_pixel_appearance_review(capture, fixture["plan"], frame_stride=1)
    reviewed = {
        check["frame_index"]
        for record in review["actors"].values()
        for check in record["checks"]
    }
    assert reviewed == set(every)


def test_capture_input_identity_tracks_size_and_time(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    _shared_capture(capture)
    before = capture_visual_input_identities(capture)
    assert "pixel_visibility_truth.json" in before
    assert "native_pixel_masks_depth_authority_v1.npz" in before
    # A re-render that changes the frame count changes the size outright.
    rgb = np.load(capture / "rgb.npy")
    np.save(capture / "rgb.npy", np.concatenate([rgb, rgb]))
    resized = capture_visual_input_identities(capture)
    assert resized["rgb.npy"]["size_bytes"] != before["rgb.npy"]["size_bytes"]
    # A same-size rewrite is separated by modification time, which this host
    # records at about one millisecond. Two writes inside one tick are not
    # distinguishable by stat alone, which is why reuse also re-observes
    # pixels; see the verification tests below.
    time.sleep(0.01)
    same_size = np.load(capture / "rgb.npy")
    same_size[0, 0, 0] = [1, 2, 3]
    np.save(capture / "rgb.npy", same_size)
    rewritten = capture_visual_input_identities(capture)
    assert rewritten["rgb.npy"]["size_bytes"] == resized["rgb.npy"]["size_bytes"]
    assert rewritten["rgb.npy"]["mtime_ns"] > resized["rgb.npy"]["mtime_ns"]


def test_reobservation_rejects_changed_pixels_behind_an_unchanged_identity(tmp_path: Path) -> None:
    """The pixel re-read, not the stat record, is what makes reuse safe."""
    capture = tmp_path / "capture"
    fixture = _shared_capture(capture)
    pack = build_shared_visual_evidence(capture, fixture["plan"], fixture["truth"], frame_stride=1)
    assert verify_shared_visual_evidence(pack, capture)["status"] == "pass"

    # Repaint source1 in the actual RGB, keeping the byte size identical, then
    # force the stored identity to match the rewritten file so the filesystem
    # record cannot be what catches it.
    rgb = np.load(capture / "rgb.npy")
    rgb[:, :, :16] = [40, 90, 200]
    np.save(capture / "rgb.npy", rgb)
    stale = deepcopy(pack)
    stale["key"]["capture_inputs"] = capture_visual_input_identities(capture)
    assert stale["key"]["capture_inputs"]["rgb.npy"]["size_bytes"] == pack["key"]["capture_inputs"]["rgb.npy"]["size_bytes"]

    rejected = verify_shared_visual_evidence(stale, capture)
    assert rejected["status"] == "rejected"
    assert rejected["reason"] == "re-observed pixels disagree with the recorded appearance check"
    assert rejected["actor_id"] == "source1"


# ---------------------------------------------------------------------------
# Illumination-relative colour families.
#
# Every case below is built twice where it matters: once under a neutral room
# light and once under a warm one. A registered value that survives the neutral
# frame and dies under the warm one would mean the classifier is reading the
# lamp instead of the surface, which is exactly the failure these replace.
# ---------------------------------------------------------------------------

NEUTRAL_ROOM = (1.0, 1.0, 1.0)
WARM_ROOM = (1.0, 0.80, 0.62)
COOL_ROOM = (0.82, 0.92, 1.0)
DIM_WARM_ROOM = (0.55, 0.44, 0.34)


def _room_frame(
    patch: Any,
    *,
    background: tuple[int, int, int] = (150, 150, 150),
    illuminant: tuple[float, float, float] = NEUTRAL_ROOM,
    size: tuple[int, int] = (160, 200),
    box: tuple[int, int, int, int] = (40, 30, 160, 150),
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """A target patch inside a room, then the whole frame lit by one lamp.

    The background is what the scene-neutral estimate has to work from, and the
    illuminant multiplies target and room alike, which is what a coloured room
    light actually does.
    """
    height, width = size
    x0, y0, x1, y1 = box
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:] = background
    region = np.asarray(patch, dtype=np.uint8)
    if region.ndim == 1:
        rgb[y0:y1, x0:x1] = region
    else:
        rgb[y0:y1, x0:x1] = np.resize(region, (y1 - y0, x1 - x0, 3))
    lit = np.clip(rgb.astype(np.float64) * np.asarray(illuminant), 0, 255).astype(np.uint8)
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return lit, mask, [x0, y0, x1, y1]


def _striped(first: tuple[int, int, int], second: tuple[int, int, int],
             height: int, width: int, period: int = 6) -> np.ndarray:
    rows = np.zeros((height, width, 3), dtype=np.uint8)
    rows[:] = first
    for start in range(0, height, period):
        rows[start:start + period // 2] = second
    return rows


def _verdict(patch: Any, value: str, kind: str, *, illuminant=NEUTRAL_ROOM, **frame) -> dict[str, Any]:
    rgb, mask, bbox = _room_frame(patch, illuminant=illuminant, **frame)
    return inspect_registered_appearance(
        rgb, mask, value, entity_kind=kind, target_bbox=bbox,
    )


def test_a_warm_room_light_does_not_rename_a_white_shirt() -> None:
    for room in (NEUTRAL_ROOM, WARM_ROOM, COOL_ROOM, DIM_WARM_ROOM):
        observed = _verdict((236, 234, 230), "white", "human", illuminant=room)
        assert observed["status"] == "pass", (room, observed.get("reason"))
        assert observed["observed_value"] == "white"
        # The same pixels must not also satisfy a warm garment colour.
        for other in ("burgundy", "yellow", "pink"):
            competing = _verdict((236, 234, 230), other, "human", illuminant=room)
            assert competing["status"] == "not_observable", (room, other)


def test_a_green_shirt_is_never_certified_as_yellow() -> None:
    for room in (NEUTRAL_ROOM, WARM_ROOM, DIM_WARM_ROOM):
        assert _verdict((54, 128, 62), "green", "human", illuminant=room)["status"] == "pass", room
        for other in ("yellow", "blue", "white", "burgundy"):
            observed = _verdict((54, 128, 62), other, "human", illuminant=room)
            assert observed["status"] == "not_observable", (room, other)


def test_a_yellow_shirt_survives_a_warm_room_and_stays_yellow() -> None:
    for room in (NEUTRAL_ROOM, WARM_ROOM, DIM_WARM_ROOM):
        assert _verdict((198, 164, 40), "yellow", "human", illuminant=room)["status"] == "pass", room
        assert _verdict((198, 164, 40), "green", "human", illuminant=room)["status"] == "not_observable"
        assert _verdict((198, 164, 40), "white", "human", illuminant=room)["status"] == "not_observable"


def test_a_two_tone_striped_shirt_keeps_its_registered_white() -> None:
    stripes = _striped((238, 236, 232), (150, 122, 96), 120, 120)
    for room in (NEUTRAL_ROOM, WARM_ROOM):
        observed = _verdict(stripes, "white", "human", illuminant=room)
        assert observed["status"] == "pass", (room, observed.get("reason"))
        assert observed["decision"]["support_share"] >= 0.35
        # A registered brown on the same garment loses to the white ground.
        assert _verdict(stripes, "burgundy", "human", illuminant=room)["status"] == "not_observable"


def test_a_mostly_brown_shirt_is_not_certified_as_white() -> None:
    stripes = np.zeros((120, 120, 3), dtype=np.uint8)
    stripes[:] = (150, 108, 74)
    stripes[::10] = (238, 236, 232)
    observed = _verdict(stripes, "white", "human", illuminant=WARM_ROOM)
    assert observed["status"] == "not_observable"


def test_the_grey_family_values_are_implemented() -> None:
    for value in ("light_gray", "light_gray_fabric", "silver", "warm_gray"):
        for room in (NEUTRAL_ROOM, WARM_ROOM):
            observed = _verdict((120, 118, 116), value, "device", illuminant=room)
            assert observed["status"] == "pass", (value, room, observed.get("reason"))
            assert observed.get("reason") != REGISTERED_APPEARANCE_CLASSIFIER_GAP_REASON
        # A mid grey is never certified as black: black has no lighter neighbour.
        assert _verdict((120, 118, 116), "black", "device")["status"] == "not_observable"


def test_the_sand_family_values_are_implemented() -> None:
    for value in ("beige", "sandstone"):
        for room in (NEUTRAL_ROOM, WARM_ROOM):
            observed = _verdict((214, 196, 158), value, "device", illuminant=room)
            assert observed["status"] == "pass", (value, room, observed.get("reason"))
        assert _verdict((40, 70, 150), value, "device")["status"] == "not_observable"


def test_white_satin_is_implemented_and_is_the_white_family() -> None:
    assert _verdict((240, 238, 236), "white_satin", "device")["status"] == "pass"
    assert _verdict((240, 238, 236), "white_satin", "device", illuminant=WARM_ROOM)["status"] == "pass"
    assert _verdict((60, 58, 56), "white_satin", "device")["status"] == "not_observable"


def test_a_sable_coat_is_a_dark_warm_family_not_a_golden_one() -> None:
    for value in ("standard_sable", "dark_sable"):
        observed = _verdict((86, 58, 34), value, "animal", illuminant=WARM_ROOM)
        assert observed["status"] == "pass", (value, observed.get("reason"))
    assert _verdict((86, 58, 34), "standard_yellow", "animal")["status"] == "not_observable"
    assert _verdict((226, 186, 96), "standard_yellow", "animal", illuminant=WARM_ROOM)["status"] == "pass"


def test_seal_point_needs_a_light_body_and_dark_points() -> None:
    pointed = _striped((222, 206, 180), (46, 34, 28), 120, 120, period=10)
    observed = _verdict(pointed, "standard_seal_point", "animal", illuminant=WARM_ROOM)
    assert observed["status"] == "pass", observed.get("reason")
    assert observed.get("reason") != REGISTERED_APPEARANCE_CLASSIFIER_GAP_REASON
    # A cream coat with no dark points is not a pointed coat.
    plain = _verdict((222, 206, 180), "standard_seal_point", "animal")
    assert plain["status"] == "not_observable"
    assert any("missing_component" in reason for reason in plain["decision"]["rejections"])


def test_a_registered_value_with_no_predicate_still_reports_the_gap() -> None:
    observed = _verdict((180, 180, 180), "iridescent_teal_flake", "device")
    assert observed["status"] == "not_observable"
    assert observed["reason"] == REGISTERED_APPEARANCE_CLASSIFIER_GAP_REASON
    assert observed["gap_category"] == "interface_not_implemented"


def test_the_scene_reference_ignores_the_inspected_instance() -> None:
    """A target that fills the view may not become its own lightness scale."""
    from avengine.rooms.appearance_color import estimate_scene_neutral

    rgb, mask, bbox = _room_frame(
        (48, 47, 46), background=(168, 166, 164), box=(6, 6, 194, 154),
    )
    without = estimate_scene_neutral(rgb, exclude_mask=mask)
    with_target = estimate_scene_neutral(rgb)
    assert without["reference_lightness"] > with_target["reference_lightness"] + 10
    # Read against the room, the dark target is dark; read against itself it is not.
    observed = inspect_registered_appearance(
        rgb, mask, "black", entity_kind="device", target_bbox=bbox,
    )
    assert observed["status"] == "pass", observed.get("reason")


def test_a_target_too_small_to_read_stays_not_observable() -> None:
    rgb, mask, bbox = _room_frame((54, 128, 62), box=(40, 30, 52, 44))
    observed = inspect_registered_appearance(
        rgb, mask, "green", entity_kind="human", target_bbox=bbox,
    )
    assert observed["status"] == "not_observable"
    assert "too_few_supporting_pixels" in observed["decision"]["rejections"]


def test_an_empty_target_footprint_is_a_geometric_refusal() -> None:
    rgb, mask, _bbox = _room_frame((54, 128, 62))
    observed = inspect_registered_appearance(
        rgb, mask, "green", entity_kind="human", target_bbox=[10, 10, 10, 10],
    )
    assert observed["status"] == "not_observable"
    assert observed["reason"] == "empty_target_footprint"
    assert observed["gap_category"] == "target_geometry"


def test_the_same_numbers_serve_every_entity_kind() -> None:
    thresholds = nonhuman_appearance_placeholder_thresholds()
    assert thresholds["minimum_color_pixels"] == 512
    assert thresholds["color_model"] == "illumination_relative_colour_families_v1"
    assert thresholds["one_rule_for_every_entity_kind"]["function"] == "inspect_registered_appearance"


def test_the_registrable_vocabulary_is_exactly_what_the_classifier_observes() -> None:
    """A value an asset may register must be a value the pixels can be read for."""
    from avengine.rooms.appearance_color import supported_appearance_values
    from avengine.runtime_profiles import PIXEL_APPEARANCE_VALUE_VOCABULARY

    assert set(PIXEL_APPEARANCE_VALUE_VOCABULARY) == set(supported_appearance_values())


def test_a_shirt_in_shadow_is_judged_among_the_pixels_that_carry_a_hue() -> None:
    """Shadow is the absence of colour evidence, not evidence of another colour."""
    shirt = np.zeros((120, 120, 3), dtype=np.uint8)
    shirt[:] = (18, 16, 15)
    shirt[:, 40:100] = (40, 96, 54)
    observed = _verdict(shirt, "green", "human", illuminant=DIM_WARM_ROOM)
    assert observed["status"] == "pass", observed.get("reason")
    decision = observed["decision"]
    assert decision["support_share_denominator"] == "pixels_with_a_measurable_hue"
    assert decision["denominator_pixels"] < decision["analysed_pixels"]


def test_a_target_almost_entirely_in_shadow_cannot_name_a_hue() -> None:
    shirt = np.zeros((120, 120, 3), dtype=np.uint8)
    shirt[:] = (16, 15, 14)
    shirt[:, 56:64] = (40, 96, 54)
    observed = _verdict(shirt, "green", "human", illuminant=DIM_WARM_ROOM)
    assert observed["status"] == "not_observable"
    assert "target_is_too_dark_or_too_neutral_for_a_hue" in observed["decision"]["rejections"]


def test_a_lightness_band_still_counts_every_pixel_of_the_target() -> None:
    """A white value is weighed against the shadow on it, not excused from it."""
    shirt = np.zeros((120, 120, 3), dtype=np.uint8)
    shirt[:] = (26, 25, 25)
    shirt[:, 50:70] = (238, 236, 232)
    observed = _verdict(shirt, "white", "human")
    assert observed["status"] == "not_observable"
    assert observed["decision"]["support_share_denominator"] == "inspected_pixels"


# ---------------------------------------------------------------------------
# A registered second colour, and what "a different appearance" now means.
# ---------------------------------------------------------------------------


def test_a_declared_second_colour_stops_the_check_from_competing() -> None:
    """A checked garment registers both colours; only then is the second one free."""
    # A check pattern whose second colour is about as large as its first, which
    # is what the measured blue-and-tan plaid actually looks like.
    plaid = np.zeros((120, 120, 3), dtype=np.uint8)
    plaid[:] = (86, 112, 150)
    for start in range(0, 120, 10):
        plaid[start:start + 3] = (176, 152, 96)
        plaid[:, start:start + 3] = (176, 152, 96)
    undeclared = _verdict(plaid, "blue", "human", illuminant=WARM_ROOM)
    assert undeclared["status"] == "not_observable"
    assert any("competes" in reason for reason in undeclared["decision"]["rejections"])

    rgb, mask, bbox = _room_frame(plaid, illuminant=WARM_ROOM)
    declared = inspect_registered_appearance(
        rgb, mask, "blue", entity_kind="human", target_bbox=bbox, secondary_value="beige",
    )
    assert declared["status"] == "pass", declared.get("reason")
    assert declared["registered_secondary_value"] == "beige"
    assert "orange" in declared["decision"]["declared_secondary_families"]


def test_a_declared_second_colour_cannot_rescue_a_minority_first_colour() -> None:
    """The second colour leaves the competition but stays in the denominator."""
    mostly_tan = np.zeros((120, 120, 3), dtype=np.uint8)
    mostly_tan[:] = (176, 152, 96)
    mostly_tan[:, :24] = (86, 112, 150)
    rgb, mask, bbox = _room_frame(mostly_tan, illuminant=WARM_ROOM)
    observed = inspect_registered_appearance(
        rgb, mask, "blue", entity_kind="human", target_bbox=bbox, secondary_value="beige",
    )
    assert observed["status"] == "not_observable"
    assert "registered_families_are_a_minority_of_the_target" in observed["decision"]["rejections"]


def test_a_second_colour_is_read_from_the_registry_by_entity_kind() -> None:
    from avengine.rooms.qa_evidence import _appearance_spec

    human = _appearance_spec(
        {"actor_id": "source1", "asset_id": "shirt", "entity_class": "human"},
        asset_registry={"shirt": {"realized_attributes": {
            "top_color": "blue", "top_secondary_color": "beige"}}},
    )
    assert human["value"] == "blue"
    assert human["secondary_value"] == "beige"
    assert human["secondary_field"] == "top_secondary_color"
    device = _appearance_spec(
        {"actor_id": "source2", "asset_id": "box", "entity_class": "rigid_object"},
        asset_registry={"box": {"realized_attributes": {
            "body_color": "white", "secondary_color": "black"}}},
    )
    assert device["secondary_value"] == "black"
    plain = _appearance_spec(
        {"actor_id": "source3", "asset_id": "plain", "entity_class": "human"},
        asset_registry={"plain": {"realized_attributes": {"top_color": "green"}}},
    )
    assert plain["secondary_value"] is None


def test_a_second_colour_outside_the_vocabulary_is_refused_at_registration() -> None:
    from avengine.runtime_profiles import _validate_appearance_and_resting_pose

    errors = _validate_appearance_and_resting_pose(
        {"realized_attributes": {"top_color": "blue", "top_secondary_color": "lunar_opal"}},
        prefix="assets[0]",
    )
    assert any("outside the native RGB appearance vocabulary" in error for error in errors)
    clean = _validate_appearance_and_resting_pose(
        {"realized_attributes": {"top_color": "blue", "top_secondary_color": "beige"}},
        prefix="assets[0]",
    )
    assert not any("appearance vocabulary" in error for error in clean)


def test_a_tint_and_a_shade_of_one_hue_are_one_colour_family() -> None:
    from avengine.rooms.appearance_color import appearance_distinction_family

    assert appearance_distinction_family("pink") == appearance_distinction_family("burgundy")
    separate = ["blue", "green", "yellow", "white"]
    families = [appearance_distinction_family(value) for value in separate]
    assert len(set(families)) == len(separate)
    assert appearance_distinction_family("iridescent_teal_flake") is None


def test_only_a_declared_pattern_may_name_a_hue_from_a_mostly_neutral_target() -> None:
    """A small warm patch on a person is usually skin, not the garment."""
    muted = np.zeros((120, 120, 3), dtype=np.uint8)
    muted[:] = (118, 120, 122)
    muted[:, :30] = (176, 152, 96)
    undeclared = _verdict(muted, "yellow", "human", illuminant=NEUTRAL_ROOM)
    assert undeclared["status"] == "not_observable"
    assert "target_is_too_dark_or_too_neutral_for_a_hue" in undeclared["decision"]["rejections"]
    assert undeclared["decision"]["minimum_nameable_hue_fraction"] == 0.25

    rgb, mask, bbox = _room_frame(muted)
    declared = inspect_registered_appearance(
        rgb, mask, "yellow", entity_kind="human", target_bbox=bbox, secondary_value="light_gray",
    )
    assert declared["decision"]["minimum_nameable_hue_fraction"] == 0.10
    assert "target_is_too_dark_or_too_neutral_for_a_hue" not in declared["decision"]["rejections"]


def test_a_registration_only_speaks_about_the_colours_it_declares() -> None:
    """Asking whether a blue-and-tan check is yellow gets no pattern discount."""
    plaid = np.zeros((120, 120, 3), dtype=np.uint8)
    plaid[:] = (86, 112, 150)
    for start in range(0, 120, 10):
        plaid[start:start + 3] = (176, 152, 96)
        plaid[:, start:start + 3] = (176, 152, 96)
    rgb, mask, bbox = _room_frame(plaid, illuminant=WARM_ROOM)
    declared = inspect_registered_appearance(
        rgb, mask, "blue", entity_kind="human", target_bbox=bbox,
        secondary_value="beige", declared_primary_value="blue",
    )
    assert declared["status"] == "pass", declared.get("reason")
    probe = inspect_registered_appearance(
        rgb, mask, "yellow", entity_kind="human", target_bbox=bbox,
        secondary_value="beige", declared_primary_value="blue",
    )
    assert probe["decision"]["declared_secondary_families"] == []
    assert probe["status"] == "not_observable"
