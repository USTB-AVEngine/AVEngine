from types import SimpleNamespace

import numpy as np

from avengine.rooms.apartment import (
    _MarkerTarget,
    _marker_targets_by_anchor,
    _qualify_anchors,
)


class _ObjectManager:
    def __init__(self, *objects: SimpleNamespace) -> None:
        self._objects = {item.object_id: item for item in objects}

    def get_objects_by_handle_substring(self) -> dict[int, SimpleNamespace]:
        return self._objects


class _RayCast:
    def __init__(self, object_id: int, distance_m: float) -> None:
        self.hits = [SimpleNamespace(object_id=object_id, ray_distance=distance_m)]

    def has_hits(self) -> bool:
        return True


class _Simulator:
    def __init__(self, object_id: int, distance_m: float) -> None:
        self._cast = _RayCast(object_id, distance_m)

    def cast_ray(self, _ray: object, *, buffer_distance: float) -> _RayCast:
        assert buffer_distance == 0.0
        return self._cast


def _marker_anchor(anchor_id: str, position_m: list[float]) -> dict:
    return {
        "anchor_id": anchor_id,
        "kind": "entity_spawn",
        "position_m": position_m,
        "los_probe_height_m": 0.0,
        "listener_relative_sector": "front",
        "expected_camera_fov": "in_fov",
        "expected_acoustic_path": "los",
    }


def _ray_modules() -> tuple[SimpleNamespace, SimpleNamespace]:
    habitat_sim = SimpleNamespace(
        geo=SimpleNamespace(Ray=lambda origin, direction: (origin, direction))
    )
    mn = SimpleNamespace(Vector3=lambda value: np.asarray(value, dtype=np.float64))
    return habitat_sim, mn


def test_marker_anchor_binds_to_live_scene_object_by_authored_position() -> None:
    manager = _ObjectManager(
        SimpleNamespace(
            object_id=22,
            handle="source_marker_1_:0000",
            translation=[1.0, 0.7, 2.0],
        ),
        SimpleNamespace(
            object_id=11,
            handle="source_marker_0_:0000",
            translation=[0.0, 0.7, -1.0],
        ),
        SimpleNamespace(
            object_id=99,
            handle="chair_:0000",
            translation=[0.0, 0.7, -1.0],
        ),
    )
    anchors = [
        _marker_anchor("marker_front", [0.0, 0.7, -1.0]),
        _marker_anchor("marker_rear", [1.0, 0.7, 2.0]),
    ]

    targets = _marker_targets_by_anchor(manager, anchors)

    assert targets["marker_front"].object_id == 11
    assert targets["marker_front"].handle == "source_marker_0_:0000"
    assert targets["marker_rear"].object_id == 22
    assert targets["marker_rear"].handle == "source_marker_1_:0000"


def test_marker_los_accepts_only_the_object_bound_to_current_anchor() -> None:
    anchor = _marker_anchor("marker_front", [0.0, 0.0, -1.0])
    targets = {
        "marker_front": _MarkerTarget(11, "source_marker_0_:0000", (0.0, 0.0, -1.0)),
        "marker_rear": _MarkerTarget(22, "source_marker_1_:0000", (0.0, 0.0, -0.98)),
    }
    habitat_sim, mn = _ray_modules()

    wrong = _qualify_anchors(
        _Simulator(22, 0.98),
        habitat_sim,
        mn,
        anchors=[anchor],
        listener_position_m=[0.0, 0.0, 0.0],
        listener_yaw_deg=0.0,
        camera_hfov_degrees=90.0,
        floor_height_m=0.0,
        marker_targets_by_anchor=targets,
    )
    correct = _qualify_anchors(
        _Simulator(11, 0.2),
        habitat_sim,
        mn,
        anchors=[anchor],
        listener_position_m=[0.0, 0.0, 0.0],
        listener_yaw_deg=0.0,
        camera_hfov_degrees=90.0,
        floor_height_m=0.0,
        marker_targets_by_anchor=targets,
    )

    wrong_record = wrong["records"][0]
    assert wrong_record["observed_acoustic_path"] == "nlos"
    assert wrong_record["target_marker_object_hit"] is False
    assert wrong_record["expected_target_marker_object_id"] == 11
    assert wrong["status"] == "fail"

    correct_record = correct["records"][0]
    assert correct_record["observed_acoustic_path"] == "los"
    assert correct_record["target_marker_object_hit"] is True
    assert correct_record["expected_target_marker_object_handle"] == (
        "source_marker_0_:0000"
    )
    assert correct["status"] == "pass"
