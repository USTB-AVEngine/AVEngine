from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from avengine.m5 import mp3d_region_actor_track_materializer as materializer


IDENTITY_QUATERNION = (0.0, 0.0, 0.0, 1.0)
RUNTIME_ORDER = ("joint0",)


@dataclass(frozen=True)
class _Clip:
    loop_duration_ticks: int = 6_400
    sample_ticks: tuple[int, ...] = (0, 3_200)
    rotations_xyzw: tuple[tuple[tuple[float, float, float, float], ...], ...] = (
        (IDENTITY_QUATERNION,),
        (IDENTITY_QUATERNION,),
    )


class _ActionSet:
    def __init__(self, clip: _Clip) -> None:
        self._clip = clip

    def action(self, _action_id: str) -> _Clip:
        return self._clip


def _calibration(tmp_path: Path) -> materializer.MaterializerCalibration:
    asset = {
        "asset_id": "beagle-fixture",
        "revisions": {"skeleton_revision": "skeleton-v1"},
    }
    base_states = [
        {
            "mouth_state": {"open_ratio": 0.0, "vocalizing": False},
            "root_transform": {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_xyzw": list(IDENTITY_QUATERNION),
            },
        }
    ]
    inputs = SimpleNamespace(
        asset=asset, request={"states": base_states, "request_id": "base"}
    )
    clip = _Clip()
    bundle = SimpleNamespace(
        actor_from_skin_root=tuple(tuple(row) for row in np.eye(4)),
        joint_mapping={"runtime_joint_order": list(RUNTIME_ORDER)},
        action_roles_by_id={"idle": "idle", "walk": "walk"},
        action_sets_by_role={"idle": _ActionSet(clip), "walk": _ActionSet(clip)},
    )
    contacts = {
        "paw_front_left": True,
        "paw_front_right": True,
        "paw_hind_left": True,
        "paw_hind_right": True,
    }
    contact_rows = tuple(
        {"contact_id": key, "in_contact": value} for key, value in contacts.items()
    )
    return materializer.MaterializerCalibration(
        asset_path=tmp_path / "asset.json",
        base_request_path=tmp_path / "base.json",
        inputs=inputs,
        bundle=bundle,
        asset_manifest_sha256="a" * 64,
        asset_id="beagle-fixture",
        contact_by_action_tick={
            "idle": {0: contact_rows, 3_200: contact_rows},
            "walk": {0: contact_rows, 3_200: contact_rows},
        },
        base_first_rotation_xyzw=IDENTITY_QUATERNION,
        forward_axis="-Z",
        up_axis="+Y",
        base_compiled_frame_count=75,
    )


def _case(
    *,
    motion_case: str = "both_moving",
    camera_position: float = 1.0,
) -> materializer.RouteCase:
    positions1 = np.column_stack(
        (
            np.linspace(0.0, 1.0, 75),
            np.zeros(75),
            np.zeros(75),
        )
    )
    positions2 = np.column_stack(
        (
            np.zeros(75),
            np.zeros(75),
            np.linspace(2.0, 3.0, 75),
        )
    )
    return materializer.RouteCase(
        region_instance_id="room:region:000",
        route_family_id="room_region_000_route_family_01",
        motion_case=motion_case,
        camera_binding={
            "position_m": [camera_position, 1.0, 2.0],
            "yaw_deg": camera_position * 30.0,
            "selection_order": 1,
        },
        source_positions_m={"source1": positions1, "source2": positions2},
        route_descriptors={
            "identity_continuity": {
                "endpoint_exchange_does_not_swap_source_or_actor_ids": True
            }
        },
    )


def test_tangent_aligns_local_minus_z_to_route_direction() -> None:
    quaternion = materializer._tangent_quaternion_xyzw(np.asarray([1.0, 0.0, 0.0]))
    from avengine.m2.habitat_capture import quaternion_xyzw_to_matrix

    forward = quaternion_xyzw_to_matrix(quaternion) @ np.asarray([0.0, 0.0, -1.0])
    assert forward == pytest.approx([1.0, 0.0, 0.0])


def test_static_heading_is_base_m2_rotation_and_moving_is_walk(
    tmp_path: Path,
) -> None:
    calibration = _calibration(tmp_path)
    static_case = _case(motion_case="static_static")
    moving_case = _case(motion_case="s1_moving_s2_static")
    static_track = materializer._build_actor_track(
        case=static_case,
        calibration=calibration,
        actor_axis=materializer.ACTOR_AXES[0],
        route_positions=static_case.source_positions_m["source1"],
        motion_case="static_static",
    )
    moving_track = materializer._build_actor_track(
        case=moving_case,
        calibration=calibration,
        actor_axis=materializer.ACTOR_AXES[0],
        route_positions=moving_case.source_positions_m["source1"],
        motion_case="s1_moving_s2_static",
    )
    assert {state["action_id"] for state in static_track["states"]} == {"idle"}
    assert {state["action_id"] for state in moving_track["states"]} == {"walk"}
    assert all(
        frame["orientation_source"]["source"] == "base_grounded_m2_first_state_rotation"
        for frame in static_track["frames"]
    )
    assert moving_track["states"][0]["pts_ticks"] == 0
    assert moving_track["states"][-1]["pts_ticks"] == 74 * 3_200
    assert all(
        frame["emitter_actual_position_m"] is None for frame in moving_track["frames"]
    )
    assert moving_track["formal_m2_request"] is False
    assert moving_track["episode_counted"] is False


def test_track_preserves_route_identity_and_camera_cannot_change_transforms(
    tmp_path: Path,
) -> None:
    calibration = _calibration(tmp_path)
    first = _case(camera_position=1.0)
    second = _case(camera_position=99.0)
    left = materializer._build_actor_track(
        case=first,
        calibration=calibration,
        actor_axis=materializer.ACTOR_AXES[0],
        route_positions=first.source_positions_m["source1"],
        motion_case=first.motion_case,
    )
    right = materializer._build_actor_track(
        case=second,
        calibration=calibration,
        actor_axis=materializer.ACTOR_AXES[0],
        route_positions=second.source_positions_m["source1"],
        motion_case=second.motion_case,
    )
    assert left["frames"] == right["frames"]
    assert left["source_id"] == "source1"
    assert left["actor_id"] == "actor0"
    assert left["route_source_center_plan"]["authority"].startswith("r4")
    assert left["native_pending"]["support_contact"] is None


def test_all_motion_cases_have_two_independent_actor_tracks(tmp_path: Path) -> None:
    calibration = _calibration(tmp_path)
    for motion_case in materializer.MOTION_CASES:
        case = _case(motion_case=motion_case)
        tracks = [
            materializer._build_actor_track(
                case=case,
                calibration=calibration,
                actor_axis=axis,
                route_positions=case.source_positions_m[axis["source_id"]],
                motion_case=motion_case,
            )
            for axis in materializer.ACTOR_AXES
        ]
        assert len(tracks) == 2
        assert [track["source_id"] for track in tracks] == ["source1", "source2"]
        assert all(len(track["states"]) == 75 for track in tracks)
        expected_action = {
            "static_static": ("idle", "idle"),
            "s1_moving_s2_static": ("walk", "idle"),
            "s1_static_s2_moving": ("idle", "walk"),
            "both_moving": ("walk", "walk"),
        }[motion_case]
        assert (
            tuple(track["states"][0]["action_id"] for track in tracks)
            == expected_action
        )


def test_different_asset_or_calibration_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(
        materializer.MP3DRegionActorTrackError,
        match="different asset manifests",
    ):
        materializer.validate_shared_asset_calibration(
            [tmp_path / "a.json", tmp_path / "b.json"],
            [tmp_path / "base.json", tmp_path / "base.json"],
        )
    with pytest.raises(
        materializer.MP3DRegionActorTrackError,
        match="different asset manifests",
    ):
        materializer.validate_shared_asset_calibration(
            [tmp_path / "a.json", tmp_path / "a.json"],
            [tmp_path / "base1.json", tmp_path / "base2.json"],
        )
