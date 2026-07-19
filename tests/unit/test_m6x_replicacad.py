from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from avengine.contracts.json_io import write_json
from avengine.m6x.replicacad import (
    M6XReplicaCADError,
    RetainedReplicaCADReview,
    build_replicacad_runtime_review,
    inspect_replicacad_articulated_room_objects,
    load_replicacad_semantic_categories,
    load_retained_replicacad_review,
)


class _Bounds:
    def __init__(self) -> None:
        self.min = np.asarray((-0.5, -0.5, -0.5), dtype=np.float64)
        self.max = np.asarray((0.5, 0.5, 0.5), dtype=np.float64)


class _Transform:
    def __init__(self, translation: tuple[float, float, float]) -> None:
        self.translation = np.asarray(translation, dtype=np.float64)

    def transform_point(self, point):
        return np.asarray(point, dtype=np.float64) + self.translation


class _Object:
    def __init__(self, object_id: int, translation=(10.0, 1.0, 10.0)) -> None:
        self.object_id = object_id
        self.handle = f"furniture_{object_id:03d}"
        self.semantic_id = 20
        self.collision_shape_aabb = _Bounds()
        self.transformation = _Transform(translation)


class _Manager:
    def __init__(self, count: int) -> None:
        self.objects = {
            str(index): _Object(index, (10.0 + index, 1.0, 10.0))
            for index in range(count)
        }

    def get_objects_by_handle_substring(self):
        return self.objects


class _Mn:
    @staticmethod
    def Vector3(value):
        return np.asarray(value, dtype=np.float64)


class _PathFinder:
    is_loaded = True

    def get_topdown_view(self, meters_per_pixel: float, floor_height_m: float):
        assert meters_per_pixel == pytest.approx(0.5)
        assert floor_height_m == pytest.approx(0.4)
        return np.ones((40, 40), dtype=np.uint8)

    def get_bounds(self):
        return np.asarray(((0.0, 0.0, 0.0), (20.0, 4.0, 20.0)))

    def is_navigable(self, point, maximum_y_delta):
        del maximum_y_delta
        x, _, z = np.asarray(point, dtype=np.float64)
        return 0.0 <= x <= 20.0 and 0.0 <= z <= 20.0

    def snap_point(self, point):
        return np.asarray(point, dtype=np.float64)

    def distance_to_closest_obstacle(self, point, maximum_search_radius):
        del point, maximum_search_radius
        return 1.0


class _Node:
    def __init__(self, position: tuple[float, float, float]) -> None:
        self.absolute_translation = np.asarray(position, dtype=np.float64)
        self.cumulative_bb = _Bounds()


class _ArticulatedObject:
    def __init__(self) -> None:
        self.object_id = 22
        self.handle = "cabinet_:0000"
        self.aabb = _Bounds()
        self.root_scene_node = _Node((4.0, 0.0, 4.0))
        self._links = {0: ("body", _Node((4.0, 0.0, 4.0)))}

    def get_link_ids(self):
        return self._links

    def get_link_name(self, link_id: int):
        return self._links[link_id][0]

    def get_link_scene_node(self, link_id: int):
        return self._links[link_id][1]


class _ArticulatedManager:
    def get_objects_by_handle_substring(self):
        value = _ArticulatedObject()
        return {value.handle: value}


class _FurnitureExcludedPathFinder(_PathFinder):
    def is_navigable(self, point, maximum_y_delta):
        del point, maximum_y_delta
        return False

    def distance_to_closest_obstacle(self, point, maximum_search_radius):
        del point, maximum_search_radius
        return 0.0


def _retained(tmp_path: Path) -> RetainedReplicaCADReview:
    rgb = np.zeros((2, 12, 16, 3), dtype=np.uint8)
    rgb[..., 1] = 80
    semantic = np.zeros((2, 12, 16), dtype=np.uint32)
    semantic[:, 2:5, 2:5] = 10
    semantic[:, 7:10, 10:14] = 11
    paths = {
        "source0": np.asarray(((1.0, 1.6, 1.0), (1.0, 1.6, 2.0))),
        "source1": np.asarray(((3.0, 0.6, 1.0), (3.0, 0.6, 2.0))),
    }
    return RetainedReplicaCADReview(
        frame_count=2,
        frame_rate_hz=15,
        room_id="replicacad_apt_0",
        rgb=rgb,
        semantic=semantic,
        trajectories_m=paths,
        activity_by_frame={
            "source0": np.asarray((False, True), dtype=np.bool_),
            "source1": np.asarray((True, False), dtype=np.bool_),
        },
        events_by_frame={
            "source0": (None, "speech"),
            "source1": ("bark", None),
        },
        bindings={
            "source0": {
                "source_id": "source0",
                "actor_class": "human",
                "semantic_id": 10,
            },
            "source1": {
                "source_id": "source1",
                "actor_class": "dog",
                "semantic_id": 11,
            },
        },
        program_sources={
            "source0": {
                "source_id": "source0",
                "asset_class": "human",
                "voice_taxonomy": {"vocalization_type": "speech"},
            },
            "source1": {
                "source_id": "source1",
                "asset_class": "animal",
                "call_taxonomy": {"call_type": "bark"},
            },
        },
        listener_position_m=(2.0, 1.47, 0.5),
        listener_yaw_deg=0.0,
        camera_hfov_degrees=90.0,
        mixture_wav=tmp_path / "mixture.wav",
    )


def test_build_uses_one_live_snapshot_for_gate_and_all_rigid_footprints(
    tmp_path: Path,
) -> None:
    retained = _retained(tmp_path)
    result = build_replicacad_runtime_review(
        pathfinder=_PathFinder(),
        object_manager=_Manager(3),
        magnum=_Mn,
        retained=retained,
        floor_height_m=0.4,
        meters_per_pixel=0.5,
        expected_rigid_count=3,
        semantic_categories_by_id={20: "chair"},
    )
    assert len(result.obstacle_map.rigid_obstacles) == 3
    assert result.obstacle_map.summary()["rigid_obstacle_role_counts"] == {
        "elevated_object": 0,
        "ground_blocker": 3,
        "unknown": 0,
        "walkable_floor_covering": 0,
    }
    assert result.source_center_gate["status"] == "pass"
    assert result.source_center_gate["full_body_collision_claim"] is False
    assert result.source_center_gate["pathfinder_snapshot_match"] is True
    assert result.topdown_frames.shape == (2, 480, 640, 3)
    assert result.annotated_frames.shape == (2, 480, 1280, 3)


def test_build_rejects_missing_live_furniture(tmp_path: Path) -> None:
    with pytest.raises(M6XReplicaCADError, match="rigid obstacle count"):
        build_replicacad_runtime_review(
            pathfinder=_PathFinder(),
            object_manager=_Manager(2),
            magnum=_Mn,
            retained=_retained(tmp_path),
            floor_height_m=0.4,
            meters_per_pixel=0.5,
            expected_rigid_count=3,
            semantic_categories_by_id={20: "chair"},
        )


def test_loads_semantic_categories_from_declared_replicacad_lexicon(
    tmp_path: Path,
) -> None:
    root = tmp_path / "replica_cad"
    write_json(
        root / "replicaCAD.scene_dataset_config.json",
        {
            "semantic_scene_descriptor_instances": {
                "replicaCAD_ssd_map": "configs/ssd/lexicon.json"
            }
        },
    )
    write_json(
        root / "configs/ssd/lexicon.json",
        {
            "classes": [
                {"id": 20, "name": "Chair"},
                {"id": 98, "name": "Rug"},
            ]
        },
    )
    assert load_replicacad_semantic_categories(root) == {20: "chair", 98: "rug"}


def test_articulated_room_inventory_is_not_promoted_to_collision_obb() -> None:
    report = inspect_replicacad_articulated_room_objects(
        _ArticulatedManager(),
        _FurnitureExcludedPathFinder(),
        floor_height_m=0.4,
    )
    assert report["object_count"] == 1
    assert report["scenario_human_and_dog_included"] is False
    assert report["navmesh_anchor_probe_count"] == 2
    assert report["all_root_and_link_anchor_xz_non_navigable"] is True
    assert report["separate_collision_footprints_added"] is False
    assert report["representation"] == "declared_navmesh_only"
    assert report["objects"][0]["public_collision_shape_aabb_available"] is False


def _write_retained_fixture(root: Path) -> tuple[Path, Path, Path]:
    capture = root / "capture"
    delivery = root / "delivery"
    arrays = capture / "arrays"
    arrays.mkdir(parents=True)
    (delivery / "audio/binaural").mkdir(parents=True)
    write_json(
        capture / "evidence.json",
        {
            "status": "pass",
            "room_id": "replicacad_apt_0",
            "frame_count": 2,
            "frame_rate_hz": 15,
            "anchor_order": [
                "human0.head",
                "human0.mouth_emitter",
                "dog0.mouth_emitter",
            ],
        },
    )
    rgb = np.zeros((2, 4, 6, 3), dtype=np.uint8)
    semantic = np.zeros((2, 4, 6), dtype=np.uint32)
    anchors = np.arange(18, dtype=np.float64).reshape(2, 3, 3)
    np.save(arrays / "rgb.npy", rgb, allow_pickle=False)
    np.save(arrays / "semantic.npy", semantic, allow_pickle=False)
    np.save(arrays / "anchor_positions_m.npy", anchors, allow_pickle=False)
    write_json(
        delivery / "source_actor_bindings.json",
        {
            "room_id": "replicacad_apt_0",
            "bindings": {
                "source0": {
                    "source_id": "source0",
                    "actor_class": "human",
                    "semantic_id": 10,
                    "emitter_anchor_index": 1,
                    "capture_anchor_id": "human0.mouth_emitter",
                },
                "source1": {
                    "source_id": "source1",
                    "actor_class": "dog",
                    "semantic_id": 11,
                    "emitter_anchor_index": 2,
                    "capture_anchor_id": "dog0.mouth_emitter",
                },
            },
        },
    )
    write_json(
        delivery / "source_program_reuse.json",
        {
            "clip_time_and_audio_contract": {"frame_count": 2},
            "sources": [
                {
                    "source_id": "source0",
                    "asset_class": "human",
                    "voice_taxonomy": {"vocalization_type": "speech"},
                    "event_windows": [
                        {
                            "event_id": "speech",
                            "start_frame": 0,
                            "end_frame_exclusive": 2,
                        }
                    ],
                },
                {
                    "source_id": "source1",
                    "asset_class": "animal",
                    "call_taxonomy": {"call_type": "bark"},
                    "event_windows": [
                        {
                            "event_id": "bark",
                            "start_frame": 1,
                            "end_frame_exclusive": 2,
                        }
                    ],
                },
            ],
        },
    )
    (delivery / "audio/binaural/mixture.wav").write_bytes(b"retained")
    request = root / "capture_request.json"
    write_json(
        request,
        {
            "room_id": "replicacad_apt_0",
            "primary_camera_rig": {
                "world_from_rig": {
                    "translation_m": [2.6, 1.47, 3.4],
                    "rotation_xyzw": [0.0, 1.0, 0.0, 0.0],
                },
                "shared_calibration": {"hfov_degrees": 90.0},
            },
        },
    )
    return capture, delivery, request


def test_loads_source_centers_from_binding_indices_not_actor_roots(
    tmp_path: Path,
) -> None:
    capture, delivery, request = _write_retained_fixture(tmp_path)
    result = load_retained_replicacad_review(
        capture_dir=capture,
        delivery_dir=delivery,
        m1_request_path=request,
    )
    anchors = np.load(capture / "arrays/anchor_positions_m.npy", allow_pickle=False)
    assert np.array_equal(result.trajectories_m["source0"], anchors[:, 1, :])
    assert np.array_equal(result.trajectories_m["source1"], anchors[:, 2, :])
    assert result.activity_by_frame["source0"].tolist() == [True, True]
    assert result.activity_by_frame["source1"].tolist() == [False, True]
    assert result.listener_yaw_deg == pytest.approx(180.0)


def test_rejects_binding_to_wrong_capture_anchor(tmp_path: Path) -> None:
    capture, delivery, request = _write_retained_fixture(tmp_path)
    bindings_path = delivery / "source_actor_bindings.json"
    bindings = __import__("json").loads(bindings_path.read_text())
    bindings["bindings"]["source1"]["capture_anchor_id"] = "human0.head"
    write_json(bindings_path, bindings)
    with pytest.raises(M6XReplicaCADError, match="anchor identity differs"):
        load_retained_replicacad_review(
            capture_dir=capture,
            delivery_dir=delivery,
            m1_request_path=request,
        )
