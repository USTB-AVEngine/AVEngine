"""Independent geometric and denominator checks for the selection protocol."""

from pathlib import Path
import json
import numpy as np
import pytest
import trimesh
from PIL import Image
from shapely.geometry import box

from tools.rooms.room_selection.geometry import (
    projected_union,
    floor_layers,
    hm3d_annotations,
    short_side,
)
from tools.rooms.room_selection.navigation import ray_clear_batch
from tools.rooms.room_selection.media import polygon_mask
from tools.rooms.room_selection.analysis import agreement, second_sample, manual_iou
from tools.rooms.room_selection.run import load_parameters, qualify_region_evidence
from tools.rooms.room_selection.protocol import decide

pytestmark = pytest.mark.fast_unit
PARAMETERS = load_parameters(
    Path(__file__).parents[2] / "tools/rooms/room_selection/thresholds.yaml"
)[1]


def test_projected_union_does_not_double_count_overlapping_mesh():
    tri = np.array([[0, 0, 0], [2, 0, 0], [0, 0, 2]], float)
    assert projected_union(np.stack([tri, tri]), PARAMETERS).area == pytest.approx(2.0)


def test_floor_split_cannot_chain_across_stair_steps():
    triangles = np.array(
        [[[0, y, 0], [1, y, 0], [0, y, 1]] for y in [0.0, 0.2, 0.4, 3.1]]
    )
    layers = floor_layers(triangles, PARAMETERS)
    assert len(layers) == 3
    assert sum(r["face_count"] for r in layers) == 4
    assert all(r["height_range_m"][1] - r["height_range_m"][0] <= 0.3 for r in layers)


def test_annotation_conflict_keeps_both_regions_but_does_not_guess_identity(tmp_path):
    p = tmp_path / "semantic.txt"
    p.write_text(
        'HM3D Semantic Annotations\n1,ABCDEF,"floor",0\n2,ABCDEF,"bed",1\n3,123456,"wall",-1\n'
    )
    instances, colours, regions = hm3d_annotations(p)
    assert set(regions) == {-1, 0, 1}
    assert set(instances) == {1, 2, 3}
    assert colours[int("ABCDEF", 16)] is None
    assert regions[0]["ambiguous_colours"] == ["ABCDEF"]
    assert regions[1]["ambiguous_colours"] == ["ABCDEF"]


def test_rotated_short_side_is_orientation_invariant():
    from shapely.affinity import rotate

    assert short_side(rotate(box(0, 0, 2.2, 9), 37)) == pytest.approx(2.2)


def test_cpu_rays_detect_a_wall_between_points():
    wall = trimesh.creation.box(extents=[0.1, 4, 4])
    wall.apply_translation([2, 1, 0])
    result = ray_clear_batch(wall, np.array([0, 1, 0]), [[1, 1, 0], [4, 1, 0]], 0.03)
    assert result.tolist() == [True, False]


def test_image_mask_uses_camera_matrices_and_excludes_hole():
    scope = box(-1, -1, 1, 1).difference(box(-0.5, -0.5, 0.5, 0.5))
    view = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, -30], [0, 0, 0, 1]])
    projection = np.diag([0.5, 0.5, -1, 1])
    mask = polygon_mask(
        scope,
        0,
        {"view": view.tolist()},
        {"projection": projection.tolist()},
        (100, 100),
    )
    assert not mask[50, 50]
    assert mask[30, 30]
    assert not mask[0, 0]


def row(human, auto, house="h", label="R0"):
    return dict(
        house=house,
        room_label=label,
        region_id=int(label[1:]),
        human=dict(verdict=human, source="fixture"),
        stage1=dict(status=auto),
    )


def test_unsure_and_missing_evidence_are_not_silently_removed_from_metrics():
    rows = [
        row("use", "pass"),
        row("use", "review"),
        row("skip", "pass"),
        row("skip", "fail"),
        row("unsure", "pass"),
    ]
    m = agreement(rows)
    assert m["human_count"] == 5 and m["binary_denominator"] == 4
    assert m["confusion_matrix"] == dict(TP=1, FP=1, FN=1, TN=1)
    assert m["agreement"] == 0.5 and m["recall"] == 0.5
    assert m["automatic_resolved_fraction"] == 0.75


def test_unknown_scan_quality_does_not_pass_otherwise_eligible_room():
    m = dict(
        floor_area_m2=20,
        short_side_m=3,
        nav_main_area_m2=10,
        black_fraction=None,
        placement={"found": True},
    )
    result = decide(m, "living", PARAMETERS)
    assert result["status"] == "review"
    assert "BLACK_FRACTION_UNAVAILABLE" in result["reason_codes"]


def test_second_reviewer_sampling_is_fixed_and_contains_no_judgments():
    rows = [row("use", "pass", f"h{h}", f"R{r}") for h in range(10) for r in range(20)]
    a = second_sample(rows, PARAMETERS)
    b = second_sample(rows, PARAMETERS)
    assert a == b
    assert 0.10 <= a["actual_fraction"] <= 0.15
    assert len({r["house"] for r in a["samples"]}) == 10
    assert all(r["second_verdict"] is None for r in a["samples"])


def test_pending_manual_boxes_never_become_iou_ground_truth(tmp_path):
    path = tmp_path / "manual.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "source": "manual",
                        "coordinate_status": "pending_original_bbox",
                        "data": {"id": "pending"},
                    }
                ]
            }
        )
    )
    result = manual_iou([], path, PARAMETERS)
    assert result["excluded_count"] == 1
    assert result["compared_manual_count"] == 0
    assert result["mean_matched_iou_all_coordinate_valid"] is None
    assert result["status"] == "qualified_bbox_proxy"


def test_missing_floor_evidence_is_not_a_completed_split_check():
    r = dict(
        floors=[], stage1=dict(status="review"), stage2=dict(status="not_triggered")
    )
    qualify_region_evidence(r)
    assert r["stage2"]["status"] == "not_run"
    assert r["stage2"]["reason_codes"] == ["NO_SEMANTIC_FLOOR_GEOMETRY"]


def test_ambiguous_source_identity_qualifies_derived_children():
    r = dict(
        native_region=dict(ambiguous_colours=["ABCDEF"]),
        stage1=dict(status="review"),
        floors=[
            dict(
                stage1=dict(status="pass", reason_codes=[]),
                split_parts=[dict(stage1=dict(status="pass", reason_codes=[]))],
            )
        ],
    )
    qualify_region_evidence(r)
    assert r["floors"][0]["stage1"]["status"] == "review"
    assert r["floors"][0]["split_parts"][0]["stage1"]["reason_codes"] == [
        "SEMANTIC_COLOUR_CONFLICT"
    ]


def test_explicit_stage1_pilot_never_reports_completed_stage2():
    r = dict(floors=[dict(stage2=dict(status="not_triggered"), split_parts=[])])
    qualify_region_evidence(r, skip_splitting=True)
    assert r["stage2"]["reason_codes"] == ["STAGE2_NOT_REQUESTED"]
    assert r["floors"][0]["stage2"]["status"] == "not_run"


def test_floor_height_is_area_weighted_despite_many_small_artifact_faces():
    # A scan can tessellate tiny rug/noise faces much more densely than floor.
    main = np.array([[0, 0, 0], [10, 0, 0], [0, 0, 10]], float)
    tiny = np.array([[0, 0.25, 0], [0.01, 0.25, 0], [0, 0.25, 0.01]], float)
    layers = floor_layers(np.stack([main] + [tiny] * 100), PARAMETERS)
    assert len(layers) == 1
    assert layers[0]["floor_y_m"] == pytest.approx(0)
    assert layers[0]["geometry"].area == pytest.approx(50)


def test_dominant_layer_scope_does_not_hide_a_real_second_storey():
    from tools.rooms.room_selection.geometry import dominant_layer

    tiny_first = [
        dict(metrics=dict(floor_area_m2=0.01)),
        dict(metrics=dict(floor_area_m2=12)),
    ]
    selected, fraction = dominant_layer(tiny_first, 0.9)
    assert selected == 1 and fraction > 0.99
    assert (
        dominant_layer(
            [
                dict(metrics=dict(floor_area_m2=12)),
                dict(metrics=dict(floor_area_m2=10)),
            ],
            0.9,
        )[0]
        is None
    )


def test_quality_clearance_statistic_is_not_every_devices_radius():
    from tools.rooms.room_selection.navigation import placement

    mesh = trimesh.creation.box(extents=[0.1, 4, 4])
    mesh.apply_translation([100, 1, 0])
    points = np.array([[0, 0, 0], [2, 0, 0], [2, 0, 1.5], [0, 0, 1.5]], float)
    clearance = np.repeat(0.3, 4)
    adj = [[1, 3], [0, 2], [1, 3], [0, 2]]
    old = dict(
        PARAMETERS, placement_camera_clearance_m=0.5, placement_source_clearance_m=0.5
    )
    assert not placement(mesh, points, clearance, adj, old)["found"]
    witness = placement(mesh, points, clearance, adj, PARAMETERS)
    assert witness["found"]
    assert witness["horizontal_angle_deg"] <= 85
    assert all(1 <= d <= 5 for d in witness["pairwise_distances_m"])
    camera = np.array(witness["camera_m"])
    s1, s2 = witness["source_1_m"], witness["source_2_m"]
    assert ray_clear_batch(mesh, camera, [s1, s2], 0.03).all()
    assert ray_clear_batch(mesh, np.array(s1), [s2], 0.03).all()


def test_split_reference_does_not_match_same_outline_on_another_storey(tmp_path):
    from shapely.geometry import mapping

    polygon = mapping(box(0, 0, 4, 4))
    path = tmp_path / "manual.json"
    path.write_text(
        json.dumps(
            dict(
                records=[
                    dict(
                        source="manual",
                        coordinate_status="complete",
                        data=dict(
                            id="part",
                            house="h",
                            source_room_label="R0",
                            bbox_xz_m=[[0, 0], [4, 4]],
                            floor_y_m=0,
                        ),
                    )
                ]
            )
        )
    )
    region = dict(
        house="h",
        room_label="R0",
        stage2=dict(status="proposed", reason_codes=[]),
        floors=[
            dict(metrics=dict(floor_polygon=polygon, floor_y_m=0), split_parts=[]),
            dict(
                metrics=dict(floor_polygon=polygon, floor_y_m=3),
                split_parts=[dict(metrics=dict(floor_polygon=polygon))],
            ),
        ],
    )
    result = manual_iou([region], path, PARAMETERS)
    assert result["compared_manual_count"] == 1
    assert result["mean_matched_iou_all_coordinate_valid"] == 0


def test_completed_holdout_receipt_blocks_another_evaluation(tmp_path, monkeypatch):
    from tools.rooms.room_selection import addendum

    (tmp_path / "addendum1").mkdir()
    (tmp_path / "inputs.json").write_text(json.dumps(dict(houses=[])))
    (tmp_path / "addendum1/holdout_evaluation_once.json").write_text(
        json.dumps(dict(status="complete", evaluation_count=1))
    )
    monkeypatch.setattr(
        addendum, "verify_freeze", lambda path: dict(created_at_sgt="frozen")
    )

    def forbidden_analysis(args):
        pytest.fail("held-out labels must not be evaluated a second time")

    monkeypatch.setattr(addendum, "assemble", forbidden_analysis)
    with pytest.raises(FileExistsError):
        addendum.evaluate(tmp_path)
