from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from avengine.rooms.qa_evidence import (
    PLACEHOLDER_NONHUMAN_MINIMUM_COLOR_PIXELS,
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


def test_unsupported_registered_values_use_classifier_gap_reason() -> None:
    rgb = np.full((48, 48, 3), 180, dtype=np.uint8)
    mask = np.ones((48, 48), dtype=bool)
    unsupported = (
        ("silver", "device"),
        ("white_satin", "device"),
        ("warm_gray", "device"),
        ("beige", "device"),
        ("light_gray", "device"),
        ("sandstone", "device"),
        ("light_gray_fabric", "device"),
        ("dark_sable", "animal"),
        ("standard_sable", "animal"),
        ("standard_seal_point", "animal"),
    )
    for value, kind in unsupported:
        row = inspect_registered_appearance(rgb, mask, value, entity_kind=kind)
        assert row["status"] == "not_observable", value
        assert row["reason"] == "registered_appearance_value_classifier_not_implemented", value
        assert row["gap_category"] == "interface_not_implemented", value
    implemented = inspect_registered_appearance(rgb, mask, "white", entity_kind="device")
    assert implemented.get("reason") != "registered_appearance_value_classifier_not_implemented"


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
                    "realized_attributes": {"body_color": "silver", "form_factor": "jug"},
                }
            ]
        }
    }
    result = build_pixel_appearance_review(tmp_path, plan, frame_stride=1)
    actor = result["actors"]["source1"]
    assert actor["status"] == "not_observable"
    assert actor["value"] == "silver"
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
