from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from avengine.m1.habitat_capture import (
    discover_mp3d_root,
    resolve_installed_runtime_prefix,
)
from avengine.m5_1.two_human_capture import (
    _CapturedTwoHumanFrame,
    TwoHumanCaptureError,
    _HumanFrameBinding,
    _TwoHumanCaptureDependencies,
    _action_sample_index,
    _capture_two_human_frame,
    _planned_actor_world_matrix,
    _planned_emitter_world_position,
    _prepare_fresh_output,
    _require_same_runtime_file,
    _safe_regular_path,
    _save_plain_array,
    _semantic_absence_record,
    _validate_formal_capture_arrays,
    _validate_camera_runtime_navigation,
    capture_two_human_mp3d,
    load_two_human_capture_authority,
    validate_two_human_authority_documents,
)
from avengine.runtime_profiles import load_source_asset_runtime_registry


EPISODE = "mp3d_two_human_0004"
ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = ROOT / "tools/m5_1/capture_two_human_mp3d.py"
CLI_SPEC = importlib.util.spec_from_file_location(
    "capture_two_human_mp3d_cli", CLI_PATH
)
assert CLI_SPEC is not None and CLI_SPEC.loader is not None
CLI = importlib.util.module_from_spec(CLI_SPEC)
CLI_SPEC.loader.exec_module(CLI)


MALE_ID = "rocketbox_human_male_adult_01_m5_1_candidate"
MALE_REVISION = "native_runtime_ue_v3"
FEMALE_ID = "lead_b_rocketbox_adults_female_adult_01_original_v1"
FEMALE_REVISION = "native_runtime_ue_v1"
RIG = {
    "translation_m": [0.0, 1.572447, -0.5],
    "rotation_xyzw": [0.0, 0.5, 0.0, 0.8660254037844386],
}
ROOTS = {
    "source1": [-4.6, 0.072447, -2.35],
    "source2": [-3.75, 0.072447, -3.35],
}
OFFSETS = {
    "source1": [0.0, 1.61, 0.0],
    "source2": [0.0, 1.569012451171875, 0.0],
}
ACTOR_ROTATIONS = {
    "source1": [0.0, 0.2, 0.0, 0.9797958971132712],
    "source2": [0.0, -0.1, 0.0, 0.99498743710662],
}

_REAL_V4_AUTHORITY_ENVIRONMENT = (
    "AVENGINE_M5_1_V4_AUTHORITY_ROOT",
    "AVENGINE_HABITAT_RUNTIME_PREFIX",
    "AVENGINE_MP3D_ROOT",
)
_REAL_V4_AUTHORITY_FILES = {
    "atom": Path("atom_request.json"),
    "room": Path("room_manifest.json"),
    "suite": Path("cpu_preflight_v1/suite_execution_plan.json"),
    "sensor_rig": Path("cpu_preflight_v1/sensor_rig_trajectory.json"),
    "trajectory_bank": Path("cpu_preflight_v1/trajectory_bank.json"),
    "rir_plan": Path("cpu_preflight_v1/rir_job_plan.json"),
}


def _external_v4_authority_inputs_or_skip() -> dict[str, Path]:
    # The runtime prefix and MP3D root are standing native_external inputs
    # after the single-repo closure; only the dedicated authority selector
    # opts this loader in. Partial configuration below still fails closed.
    if "AVENGINE_M5_1_V4_AUTHORITY_ROOT" not in os.environ:
        pytest.skip(
            "real v4 authority loader requires all explicit selectors: "
            + ", ".join(_REAL_V4_AUTHORITY_ENVIRONMENT)
        )
    missing = [
        name for name in _REAL_V4_AUTHORITY_ENVIRONMENT if name not in os.environ
    ]
    assert not missing, (
        "configured real v4 authority loader is missing companion selectors: "
        + ", ".join(missing)
    )
    values = {name: os.environ[name] for name in _REAL_V4_AUTHORITY_ENVIRONMENT}
    blank = [name for name, value in values.items() if not value.strip()]
    assert not blank, (
        "configured real v4 authority loader selectors must be non-empty: "
        + ", ".join(blank)
    )
    return {
        name: Path(value).expanduser().resolve()
        for name, value in values.items()
    }


def _configured_v4_authority_fixture(authority_root: Path) -> dict[str, Path]:
    assert authority_root.is_dir(), (
        "configured AVENGINE_M5_1_V4_AUTHORITY_ROOT is not a directory: "
        f"{authority_root}"
    )
    files = {
        name: authority_root / relative_path
        for name, relative_path in _REAL_V4_AUTHORITY_FILES.items()
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    assert not missing, "configured real v4 authority fixture is incomplete: " + ", ".join(
        missing
    )
    return files


def _assert_current_v4_authority_fixture_semantics(
    atom: object,
    room: object,
    *,
    runtime_prefix: Path,
    mp3d_root: Path,
) -> None:
    assert isinstance(atom, dict), "configured v4 atom request must be an object"
    assert atom.get("schema") == "avengine_native_strict_two_human_mp3d_room_atom_request_v2", (
        "configured atom request schema is not current"
    )
    assert atom.get("request_id") == "mp3d_17DRP5sb8fy_strict_two_human_static_rig_v4", (
        "configured atom request is not the v4 authority token"
    )
    assert atom.get("visual_execution_mode") == "habitat_native_production", (
        "configured atom request must select Habitat-native production"
    )
    assert atom.get("qualification_claim") is False and atom.get(
        "formal_dataset_count"
    ) == 0, "configured atom request must remain non-formal"
    atom_room = atom.get("room")
    camera_runtime = atom.get("camera_runtime")
    assert isinstance(atom_room, dict) and isinstance(camera_runtime, dict), (
        "configured atom request lacks room/camera runtime semantics"
    )
    assert camera_runtime.get("loaded_scene_id") == atom_room.get("scene_id") == "17DRP5sb8fy", (
        "configured atom request scene identity drifted"
    )
    for owner, path in {
        "scene": camera_runtime.get("scene_path"),
        "dataset": camera_runtime.get("dataset_config_path"),
        "navmesh": atom_room.get("navmesh_path"),
    }.items():
        assert isinstance(path, str) and "AVENGINE_HABITAT_RUNTIME_ROOT" not in path, (
            f"configured atom {owner} path retains the legacy runtime-root token"
        )
        assert Path(path).resolve().is_relative_to(mp3d_root), (
            f"configured atom {owner} path is outside AVENGINE_MP3D_ROOT"
        )
    assert Path(str(camera_runtime.get("physics_config_path"))).resolve() == (
        runtime_prefix / "config/default.physics_config.json"
    ).resolve(), "configured atom physics path differs from the installed prefix"

    assert isinstance(room, dict), "configured v4 room manifest must be an object"
    assert room.get("room_kind") == "habitat_native" and room.get(
        "geometry_representation"
    ) == "real_surface_mesh", "configured room is not Habitat-native real geometry"
    scene = room.get("scene")
    assets = room.get("assets")
    assert isinstance(scene, dict) and isinstance(assets, list), (
        "configured room manifest lacks scene/assets semantics"
    )
    room_paths = [
        scene.get("scene_id"),
        scene.get("dataset_config_path"),
        scene.get("navmesh_path"),
        *(asset.get("path") for asset in assets if isinstance(asset, dict)),
    ]
    assert all(
        isinstance(path, str) and "${AVENGINE_MP3D_ROOT}" in path
        for path in room_paths
    ), "configured room must declare every MP3D asset through AVENGINE_MP3D_ROOT"


def _documents(tmp_path: Path) -> dict[str, dict[str, object]]:
    male_glb = tmp_path / "male.glb"
    female_glb = tmp_path / "female.glb"
    male_glb.write_bytes(b"male")
    female_glb.write_bytes(b"female")
    actors = [
        {
            "actor_id": "source1_actor",
            "asset_id": MALE_ID,
            "asset_revision": MALE_REVISION,
            "template_id": "rocketbox_human_male_adult_01",
            "body_plan_id": "biped_human",
            "emitter_offset_m": OFFSETS["source1"],
        },
        {
            "actor_id": "source2_actor",
            "asset_id": FEMALE_ID,
            "asset_revision": FEMALE_REVISION,
            "template_id": "rocketbox_adults_female_adult_01",
            "body_plan_id": "biped_human",
            "emitter_offset_m": OFFSETS["source2"],
        },
    ]
    frames = []
    rig_frames = []
    for index in range(75):
        rig_frame = {
            "frame_index": index,
            "pts_ticks": index * 3200,
            "world_from_rig": deepcopy(RIG),
        }
        rig_frames.append(rig_frame)
        frames.append(
            {
                "frame_index": index,
                "pts_ticks": index * 3200,
                "camera_state": deepcopy(rig_frame),
                "actor_states": [
                    {
                        "actor_id": "source1_actor",
                        "asset_id": MALE_ID,
                        "frame_index": index,
                        "action_id": "idle",
                        "action_phase": 0.0,
                        "action_time_ticks": index * 3200,
                        "translation_m": ROOTS["source1"],
                        "rotation_xyzw": ACTOR_ROTATIONS["source1"],
                    },
                    {
                        "actor_id": "source2_actor",
                        "asset_id": FEMALE_ID,
                        "frame_index": index,
                        "action_id": "idle",
                        "action_phase": 0.0,
                        "action_time_ticks": index * 3200,
                        "translation_m": ROOTS["source2"],
                        "rotation_xyzw": ACTOR_ROTATIONS["source2"],
                    },
                ],
            }
        )
    centers = {
        slot: [
            [ROOTS[slot][axis] + OFFSETS[slot][axis] for axis in range(3)]
            for _ in range(75)
        ]
        for slot in ("source1", "source2")
    }
    jobs = [
        {
            "job_id": f"rir_{job_index:06d}",
            "source_position_m": deepcopy(centers[slot][0]),
            "listener_position_m": deepcopy(RIG["translation_m"]),
            "listener_orientation_wxyz": [
                RIG["rotation_xyzw"][3],
                *RIG["rotation_xyzw"][:3],
            ],
            "uses": [
                {
                    "episode_id": EPISODE,
                    "frame_index": index,
                    "source_slot_id": slot,
                }
                for index in range(75)
            ],
        }
        for job_index, slot in enumerate(("source1", "source2"))
    ]
    plan = {
        "schema": "avengine_optional_spear_visual_plan_v1",
        "backend_role": "comparison_visual",
        "actors": actors,
        "camera": {
            "sensor_rig_trajectory_id": f"{EPISODE}__rig",
            "listener_id": "listener0",
        },
        "room": {
            "room_id": "habitat_mp3d_example_17DRP5sb8fy",
            "room_revision": "r1",
            "scene_id": "17DRP5sb8fy",
        },
        "render": {
            "fps_den": 1,
            "fps_num": 15,
            "frame_count": 75,
            "ticks_per_frame": 3200,
        },
        "qualification": {
            "qualification_claim": False,
            "formal_dataset_count": 0,
        },
        "frames": frames,
    }
    return {
        "atom": {
            "schema": "avengine_native_strict_two_human_mp3d_room_atom_request_v2",
            "visual_execution_mode": "habitat_native_production",
            "episode_id": EPISODE,
            "room": {
                "room_id": "habitat_mp3d_example_17DRP5sb8fy",
                "room_revision": "r1",
                "scene_id": "17DRP5sb8fy",
            },
            "actor_framing": {
                "actor_bindings": [
                    {
                        "actor_id": "source1_actor",
                        "asset_id": MALE_ID,
                        "asset_revision": MALE_REVISION,
                        "source_asset_path": str(male_glb),
                        "skin_index": 0,
                        "action_name_by_action_id": {
                            "idle": "Standing_Idle",
                            "walk": "Walking",
                        },
                    },
                    {
                        "actor_id": "source2_actor",
                        "asset_id": FEMALE_ID,
                        "asset_revision": FEMALE_REVISION,
                        "source_asset_path": str(female_glb),
                        "skin_index": 0,
                        "action_name_by_action_id": {
                            "idle": "Standing_Idle",
                            "walk": "Walking",
                        },
                    },
                ]
            },
            "camera_framing": {"calibration": {"near_m": 0.05}},
            "qualification_claim": False,
            "formal_dataset_count": 0,
        },
        "suite": {
            "schema": "avengine_optional_spear_imported_glb_suite_v1",
            "backend_role": "comparison_visual",
            "qualification_claim": False,
            "formal_dataset_count": 0,
            "scenarios": [
                {
                    "schema": "avengine_optional_spear_imported_glb_scenario_v1",
                    "scenario_id": EPISODE,
                    "backend_role": "comparison_visual",
                    "render": {
                        "frame_count": 75,
                        "frame_rate_hz": 15,
                        "height": 720,
                        "width": 1280,
                        "horizontal_fov_deg": 90.0,
                    },
                    "plan": plan,
                }
            ],
        },
        "sensor_rig": {
            "schema": "avengine_sensor_rig_trajectory_v1",
            "trajectory_id": f"{EPISODE}__rig",
            "rig_id": "camera_rig_0",
            "listener_id": "listener0",
            "formal_view_id": "view0",
            "camera_listener_coupling": "rigid_colocated_cooriented",
            "time_base_hz": 48000,
            "ticks_per_frame": 3200,
            "coordinate_frame": "avengine_world_right_handed_y_up_m",
            "pose_model": "yaw_only_about_world_positive_y",
            "program": {
                "kind": "HOLD",
                "position_m": RIG["translation_m"],
                "yaw_deg": 60.0,
            },
            "rig_from_camera": {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "rig_from_listener": {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "frame_count": 75,
            "frame_rate_hz": 15,
            "duration_ticks": 240000,
            "frames": rig_frames,
        },
        "trajectory_bank": {
            "schema": "avengine_room_trajectory_bank_v2",
            "frame_count": 75,
            "frame_rate_hz": 15,
            "seconds_per_episode": 5.0,
            "source_slots": ["source1", "source2"],
            "episodes": [
                {
                    "episode_id": EPISODE,
                    "motion_case": "strict_two_human_static_mp3d",
                    "source_root_paths_m": {
                        slot: [ROOTS[slot] for _ in range(75)]
                        for slot in ("source1", "source2")
                    },
                    "source_center_paths_m": centers,
                }
            ],
        },
        "rir_plan": {
            "schema": "avengine_room_rir_job_plan_v2",
            "status": "planned_not_run",
            "claim_boundary": "planned semantic RIR jobs; RLR has not run",
            "producer_backend": "RLR Audio Propagation",
            "cache_artifact": "room impulse response (RIR)",
            "source_acoustic_profile": "omnidirectional_point_source_v1",
            "listener_pose_mode": "per_episode_frame",
            "dry_audio_independent": True,
            "slot_identity_affects_cache_key": False,
            "cache_key_fields": [
                "source_position_m",
                "listener_position_m",
                "listener_orientation_wxyz",
            ],
            "stride_frames": 1,
            "requested_pair_state_count": 150,
            "unique_rir_job_count": 2,
            "cache_reuse_count": 148,
            "unique_listener_pose_count": 1,
            "jobs": jobs,
        },
        "runtime_profiles": load_source_asset_runtime_registry(
            ROOT / "examples/runtime/source_asset_runtime_profiles.json"
        ),
        "room": {
            "room_id": "habitat_mp3d_example_17DRP5sb8fy",
            "room_kind": "habitat_native",
            "geometry_representation": "real_surface_mesh",
            "coordinate_system": {
                "handedness": "right",
                "up_axis": "+Y",
                "forward_axis": "-Z",
                "linear_unit": "meter",
                "quaternion_order": "xyzw",
            },
            "scene": {
                "scene_id": "/external/17DRP5sb8fy.glb",
                "navmesh_policy": "load_declared",
                "load_semantic_mesh": True,
                "enable_physics": True,
            },
        },
        "m1_request": {
            "room_id": "habitat_mp3d_example_17DRP5sb8fy",
            "primary_camera_rig": {
                "rig_id": "camera_rig_0",
                "view_id": "view0",
                "world_from_rig": deepcopy(RIG),
                "shared_calibration": {
                    "projection": "pinhole",
                    "resolution_hw": [720, 1280],
                    "hfov_degrees": 90.0,
                    "near_m": 0.05,
                    "far_m": 100.0,
                    "rig_from_sensor": {
                        "translation_m": [0.0, 0.0, 0.0],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
                "modalities": [
                    {"modality": "rgb", "sensor_uuid": "rgb"},
                    {"modality": "depth", "sensor_uuid": "depth"},
                    {"modality": "semantic", "sensor_uuid": "semantic"},
                ],
            },
            "listener": {
                "listener_id": "listener0",
                "attached_to": "camera_rig_0",
                "rig_from_listener": {
                    "translation_m": [0.0, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            "sources": [
                {
                    "source_id": slot,
                    "world_from_source": {
                        "translation_m": deepcopy(centers[slot][0]),
                        "rotation_xyzw": ACTOR_ROTATIONS[slot],
                    },
                }
                for slot in ("source1", "source2")
            ],
        },
    }


def _validate(documents: dict[str, dict[str, object]]):
    return validate_two_human_authority_documents(**documents)


def test_authority_join_keeps_ue_comparison_and_frozen_habitat_state(
    tmp_path: Path,
) -> None:
    documents = _documents(tmp_path)
    authority = _validate(documents)
    assert authority.suite_visual_role == "comparison_visual"
    assert authority.episode_id == EPISODE
    assert [actor.package_stem for actor in authority.actors] == [
        "human0",
        "human1",
    ]
    assert [actor.walking_profile_sample_count for actor in authority.actors] == [
        None,
        19,
    ]
    assert [actor.semantic_id for actor in authority.actors] == [62000, 62001]
    assert authority.actor_frames[0][0].rotation_xyzw == (
        0.0,
        0.2,
        0.0,
        0.9797958971132712,
    )
    assert authority.actor_frames[0][-1].pts_ticks == 236800


def test_exact_action_tick_and_frozen_quaternion_helpers(tmp_path: Path) -> None:
    action = SimpleNamespace(loop_duration_ticks=6400, sample_ticks=(0, 3200))
    assert _action_sample_index(action, 0) == 0
    assert _action_sample_index(action, 3200) == 1
    assert _action_sample_index(action, 6400) == 0
    with pytest.raises(TwoHumanCaptureError, match="unauthored sample tick"):
        _action_sample_index(action, 1600)

    authority = _validate(_documents(tmp_path))
    frame = replace(
        authority.actor_frames[0][0],
        translation_m=(1.0, 2.0, 3.0),
        rotation_xyzw=(0.0, 1.0, 0.0, 0.0),
    )
    matrix = _planned_actor_world_matrix(frame)
    assert np.array_equal(matrix[:3, 3], np.asarray([1.0, 2.0, 3.0]))
    assert np.allclose(matrix[:3, :3], np.diag([-1.0, 1.0, -1.0]))

    actor = replace(
        authority.actors[0],
        emitter_offset_m=(1.0, 0.0, 0.0),
        emitter_offset_space="final_scaled_asset_root",
    )
    rotated = replace(
        frame,
        rotation_xyzw=(0.0, 2**-0.5, 0.0, 2**-0.5),
    )
    assert np.allclose(
        _planned_emitter_world_position(actor, rotated),
        np.asarray([1.0, 2.0, 2.0]),
        atol=1.0e-12,
    )
    with pytest.raises(TwoHumanCaptureError, match="offset space"):
        _planned_emitter_world_position(
            replace(actor, emitter_offset_space="mouth_link_local"), rotated
        )


def test_formal_arrays_and_semantic_absence_are_plain_runtime_readbacks(
    tmp_path: Path,
) -> None:
    resolution = (2, 3)
    arrays = _validate_formal_capture_arrays(
        {
            "rgb": np.zeros((*resolution, 4), dtype=np.uint8),
            "depth": np.full(resolution, 1.25, dtype=np.float32),
            "semantic": np.zeros(resolution, dtype=np.int32),
        },
        resolution_hw=resolution,
    )
    assert arrays["rgb"].shape == (2, 3, 3)
    assert arrays["rgb"].dtype == np.uint8
    assert np.all(arrays["depth"] == np.float32(1.25))

    absence = _semantic_absence_record(
        arrays["semantic"],
        resolution_hw=resolution,
        semantic_ids={"human0": 62000, "human1": 62001},
    )
    assert absence["pixel_counts"] == {"human0": 0, "human1": 0}
    collision = arrays["semantic"].copy()
    collision[0, 0] = 62000
    with pytest.raises(TwoHumanCaptureError, match="collide"):
        _semantic_absence_record(
            collision,
            resolution_hw=resolution,
            semantic_ids={"human0": 62000, "human1": 62001},
        )

    output = _prepare_fresh_output(tmp_path / "capture")
    record = _save_plain_array(output, "depth_m", arrays["depth"])
    assert set(record) == {"path", "dtype", "shape", "readback_verified"}
    assert record["readback_verified"] is True
    with pytest.raises(TwoHumanCaptureError, match="refusing to replace"):
        _prepare_fresh_output(output)


@pytest.mark.parametrize(
    "arrays,message",
    [
        (
            {
                "rgb": np.zeros((2, 3, 3), dtype=np.float32),
                "depth": np.ones((2, 3), dtype=np.float32),
                "semantic": np.zeros((2, 3), dtype=np.int32),
            },
            "RGB observation",
        ),
        (
            {
                "rgb": np.zeros((2, 3, 3), dtype=np.uint8),
                "depth": np.full((2, 3), np.inf, dtype=np.float32),
                "semantic": np.zeros((2, 3), dtype=np.int32),
            },
            "metric depth",
        ),
        (
            {
                "rgb": np.zeros((2, 3, 3), dtype=np.uint8),
                "depth": np.ones((2, 3), dtype=np.float32),
                "semantic": np.zeros((2, 3), dtype=np.float32),
            },
            "semantic observation",
        ),
    ],
)
def test_formal_array_mutations_fail_closed(arrays, message: str) -> None:
    with pytest.raises(TwoHumanCaptureError, match=message):
        _validate_formal_capture_arrays(arrays, resolution_hw=(2, 3))


def test_fake_single_simulator_captures_75_two_human_frames_and_detects_mutation(
    tmp_path: Path,
) -> None:
    authority = replace(_validate(_documents(tmp_path)), resolution_hw=(2, 3))
    event_log: list[tuple[str, int | None]] = []

    class FakeAction:
        loop_duration_ticks = 3200
        sample_ticks = (0,)
        translations_m = (((0.0, 0.0, 0.0),),)
        rotations_xyzw = (((0.0, 0.0, 0.0, 1.0),),)

    class FakeActions:
        def action(self, action_id: str):
            assert action_id == "idle"
            return FakeAction()

    class FakeJointBinding:
        def map_pose(self, translations, rotations):
            assert np.asarray(translations).shape == (1, 3)
            assert np.asarray(rotations).shape == (1, 4)
            return np.asarray([0.0, 0.0, 0.0, 1.0])

    class FakeActor:
        def __init__(self, actor_index: int, actor_from_skin: np.ndarray) -> None:
            self.actor_index = actor_index
            self.actor_from_skin = actor_from_skin
            self.skin_world = np.eye(4, dtype=np.float64)
            self.joint_positions = np.asarray([0.0, 0.0, 0.0, 1.0])

    actor_from_skin = np.eye(4, dtype=np.float64)
    actor_from_skin[0, 3] = 0.25
    actors = [FakeActor(index, actor_from_skin.copy()) for index in range(2)]
    packages = [
        SimpleNamespace(
            actions=FakeActions(),
            actor_from_skin_root=tuple(tuple(row) for row in actor_from_skin),
        )
        for _ in range(2)
    ]
    runtimes = tuple(
        _HumanFrameBinding(
            authority=authority.actors[index],
            package=packages[index],
            articulated_object=actors[index],
            joint_binding=FakeJointBinding(),
            link_blocks=(
                SimpleNamespace(joint_position_offset=0, joint_position_count=4),
            ),
            head_link_id="head",
            mouth_link_id="mouth",
        )
        for index in range(2)
    )

    class FakeSimulator:
        def __init__(self, *, mutate_on_render: bool = False) -> None:
            self.world_time = 0.0
            self.render_calls = 0
            self.physics_steps = 0
            self.mutate_on_render = mutate_on_render

        def get_world_time(self) -> float:
            return self.world_time

        def step_physics(self, _seconds: float) -> None:
            self.physics_steps += 1

        def render_sensors(self, _wrappers):
            self.render_calls += 1
            event_log.append(("render", None))
            if self.mutate_on_render:
                actors[0].joint_positions[0] = 1.0
            semantic = np.zeros((2, 3), dtype=np.int32)
            semantic[0, 0] = 62000
            semantic[0, 1] = 62001
            return {
                "rgb_uuid": np.zeros((2, 3, 4), dtype=np.uint8),
                "depth_uuid": np.ones((2, 3), dtype=np.float32),
                "semantic_uuid": semantic,
            }

    def apply_root(actor: FakeActor, matrix: np.ndarray) -> None:
        event_log.append(("apply", actor.actor_index))
        actor.skin_world = np.asarray(matrix, dtype=np.float64).copy()

    def runtime_snapshot(simulator: FakeSimulator, actor: FakeActor):
        return {
            "world_time_seconds": simulator.world_time,
            "world_from_skin_root": actor.skin_world,
            "mixed_joint_positions": actor.joint_positions,
        }

    def joint_errors(actual, expected, _blocks):
        error = float(
            np.max(
                np.abs(
                    np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float)
                )
            )
        )
        return 0.0, error

    def node_position(actor: FakeActor, link_id: str) -> np.ndarray:
        actor_world = actor.skin_world @ np.linalg.inv(actor.actor_from_skin)
        if link_id == "head":
            return (actor_world @ np.asarray([0.0, 1.7, 0.0, 1.0]))[:3]
        planned = (
            actor_world
            @ np.asarray(
                [
                    *authority.actors[actor.actor_index].emitter_offset_m,
                    1.0,
                ]
            )
        )[:3]
        return planned + np.asarray([0.01, 0.0, 0.0])

    modality_to_uuid = {
        "rgb": "rgb_uuid",
        "depth": "depth_uuid",
        "semantic": "semantic_uuid",
    }
    camera_uuids = ("rgb_uuid", "depth_uuid", "semantic_uuid", "listener0")

    def camera_snapshot():
        pose = deepcopy(authority.rig_frames[0]["world_from_rig"])
        return {
            "world_time_seconds": 0.0,
            "agent": deepcopy(pose),
            "sensors": {sensor_uuid: deepcopy(pose) for sensor_uuid in camera_uuids},
        }

    def validate_observation(observation, mapping):
        return {
            modality: observation[sensor_uuid]
            for modality, sensor_uuid in mapping.items()
        }

    def capture(simulator: FakeSimulator, frame_index: int):
        return _capture_two_human_frame(
            authority=authority,
            frame_index=frame_index,
            simulator=simulator,
            runtimes=runtimes,
            modality_to_uuid=modality_to_uuid,
            sensor_wrappers=(object(), object(), object()),
            camera_sensor_uuids=camera_uuids,
            camera_snapshot=camera_snapshot,
            apply_root=apply_root,
            runtime_snapshot=runtime_snapshot,
            joint_readback_errors=joint_errors,
            fk_readback_error=lambda *args, **kwargs: 0.0,
            node_world_position=node_position,
            observation_validator=validate_observation,
        )

    simulator = FakeSimulator()
    captured = [capture(simulator, frame_index) for frame_index in range(75)]
    assert simulator.render_calls == 75
    assert simulator.physics_steps == 0
    assert len(event_log) == 75 * 3
    assert all(
        event_log[frame_index * 3 : frame_index * 3 + 3]
        == [("apply", 0), ("apply", 1), ("render", None)]
        for frame_index in range(75)
    )
    assert all(item.record["physics_steps"] == 0 for item in captured)
    assert all(item.record["formal_render_calls"] == 1 for item in captured)
    assert all(np.all(item.semantic_visibility_pixels > 0) for item in captured)
    assert captured[0].actor_root_world_matrices.shape == (2, 4, 4)
    assert captured[0].skin_root_world_matrices.shape == (2, 4, 4)
    assert captured[0].anchor_positions_m.shape == (2, 2, 3)
    assert captured[0].record["actors"][0]["live_mouth_is_authoritative"] is False
    assert captured[0].record["actors"][0][
        "live_mouth_minus_planned_emitter_diagnostic_m"
    ] == pytest.approx([0.01, 0.0, 0.0])

    actors[0].joint_positions[:] = [0.0, 0.0, 0.0, 1.0]
    actors[1].joint_positions[:] = [0.0, 0.0, 0.0, 1.0]
    with pytest.raises(TwoHumanCaptureError, match="render changed"):
        capture(FakeSimulator(mutate_on_render=True), 0)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["atom"].pop("visual_execution_mode"),
            "explicitly select Habitat-native production",
        ),
        (
            lambda value: value["atom"].update(
                {"visual_execution_mode": "habitat_native_typo"}
            ),
            "explicitly select Habitat-native production",
        ),
        (
            lambda value: value["suite"].update({"backend_role": "production_visual"}),
            "schemas/roles must remain exact",
        ),
        (
            lambda value: value["suite"].update({"schema": "foreign"}),
            "schemas/roles must remain exact",
        ),
        (
            lambda value: value["suite"].update({"formal_dataset_count": 1}),
            "Timeline or non-formal boundary drift",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["render"].update(
                {"frame_count": 74}
            ),
            "Timeline or non-formal boundary drift",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["render"].update(
                {"ticks_per_frame": 3199}
            ),
            "Timeline or non-formal boundary drift",
        ),
        (
            lambda value: value["sensor_rig"]["frames"][9]["world_from_rig"][
                "translation_m"
            ].__setitem__(0, 1.0),
            "camera and sensor rig differ",
        ),
        (
            lambda value: value["trajectory_bank"]["episodes"][0][
                "source_center_paths_m"
            ]["source2"][4].__setitem__(1, 9.0),
            "trajectory centers differ",
        ),
        (
            lambda value: value["trajectory_bank"].update(
                {"source_slots": ["source2", "source1"]}
            ),
            "trajectory bank identity or Timeline drift",
        ),
        (
            lambda value: value["trajectory_bank"]["episodes"][0].update(
                {"motion_case": "moving"}
            ),
            "trajectory motion case drift",
        ),
        (
            lambda value: value["rir_plan"]["jobs"][0].update(
                {"source_position_m": [1.0, 2.0, 3.0]}
            ),
            "RIR state differs",
        ),
        (
            lambda value: value["m1_request"]["primary_camera_rig"][
                "shared_calibration"
            ].update({"resolution_hw": [240, 320]}),
            "M1 camera does not match",
        ),
        (
            lambda value: value["m1_request"]["primary_camera_rig"][
                "shared_calibration"
            ].update({"projection": "orthographic"}),
            "M1 camera does not match",
        ),
        (
            lambda value: value["m1_request"]["primary_camera_rig"][
                "shared_calibration"
            ].update({"near_m": 0.1}),
            "M1 camera does not match",
        ),
        (
            lambda value: value["m1_request"]["primary_camera_rig"][
                "shared_calibration"
            ].update({"far_m": 99.0}),
            "M1 camera does not match",
        ),
        (
            lambda value: value["atom"]["camera_framing"]["calibration"].update(
                {"near_m": 0.04}
            ),
            "M1 camera does not match",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["frames"][7][
                "actor_states"
            ][0].update({"action_time_ticks": 22401}),
            "off the 15 Hz grid",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["frames"][7][
                "actor_states"
            ][0].update({"frame_index": 6}),
            "action_time_ticks is off|frame 7 action_time_ticks",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["frames"][7][
                "actor_states"
            ][0].update({"asset_id": FEMALE_ID}),
            "action_time_ticks is off|frame 7 action_time_ticks",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["frames"][7][
                "actor_states"
            ][0].update({"action_phase": 1.0}),
            "action_time_ticks is off|frame 7 action_time_ticks",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["frames"][7][
                "actor_states"
            ][0].update({"action_id": "walk"}),
            "strict static actor action drift",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["frames"][7][
                "actor_states"
            ][0].update({"action_phase": 0.5}),
            "strict static actor action drift",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["frames"][7][
                "actor_states"
            ][0].update({"action_time_ticks": 25600}),
            "strict static actor action drift",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["frames"][7][
                "actor_states"
            ][0].update({"translation_m": [-4.5, 0.072447, -2.35]}),
            "strict static root/rotation must remain frozen",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["frames"][7][
                "actor_states"
            ][0].update({"rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}),
            "strict static root/rotation must remain frozen",
        ),
        (
            lambda value: value["atom"]["actor_framing"]["actor_bindings"][1].update(
                {"skin_index": 1}
            ),
            "human skin/body/timeline binding drift",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["actors"][1].update(
                {"body_plan_id": "quadruped"}
            ),
            "human skin/body/timeline binding drift",
        ),
        (
            lambda value: value["rir_plan"]["jobs"][1]["uses"].__setitem__(
                74, deepcopy(value["rir_plan"]["jobs"][1]["uses"][73])
            ),
            "maps one use more than once",
        ),
        (
            lambda value: value["rir_plan"]["jobs"].append(
                deepcopy(value["rir_plan"]["jobs"][0])
            ),
            "job identity is invalid",
        ),
        (
            lambda value: value["rir_plan"]["jobs"][0].pop("job_id"),
            "job 0 fields are invalid",
        ),
        (
            lambda value: value["rir_plan"]["jobs"][0].update(
                {"listener_orientation_wxyz": [1.0, 0.0, 0.0, 1.0]}
            ),
            "must be unit normalized",
        ),
        (
            lambda value: value["m1_request"]["listener"].update(
                {"listener_id": "listener9"}
            ),
            "listener or camera rig identity drift",
        ),
        (
            lambda value: value["m1_request"]["sources"][1]["world_from_source"][
                "translation_m"
            ].__setitem__(0, 99.0),
            "source2 position differs",
        ),
        (
            lambda value: value["m1_request"]["sources"][1]["world_from_source"].update(
                {"rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}
            ),
            "source2 rotation differs",
        ),
        (
            lambda value: value["sensor_rig"].update({"listener_id": "listener9"}),
            "sensor rig identity or Timeline drift",
        ),
        (
            lambda value: value["sensor_rig"].update({"time_base_hz": 44100}),
            "sensor rig identity or Timeline drift",
        ),
        (
            lambda value: value["sensor_rig"]["program"].update({"yaw_deg": 59.0}),
            "HOLD program differs",
        ),
        (
            lambda value: value["sensor_rig"]["rig_from_listener"][
                "translation_m"
            ].__setitem__(0, 0.1),
            "offsets must remain identity",
        ),
        (
            lambda value: value["rir_plan"].update({"dry_audio_independent": False}),
            "full planning metadata is invalid",
        ),
        (
            lambda value: value["rir_plan"].update(
                {"cache_key_fields": ["source_position_m"]}
            ),
            "cache key fields are invalid",
        ),
        (
            lambda value: value["atom"]["actor_framing"]["actor_bindings"][1].update(
                {
                    "source_asset_path": value["atom"]["actor_framing"][
                        "actor_bindings"
                    ][0]["source_asset_path"]
                }
            ),
            "distinct semantics, assets, and source paths",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["room"].update(
                {"scene_id": "foreign"}
            ),
            "room/revision/scene differs",
        ),
        (
            lambda value: value["room"]["coordinate_system"].update(
                {"forward_axis": "+Z"}
            ),
            "room coordinate/scene production semantics drift",
        ),
        (
            lambda value: value["room"].update(
                {"geometry_representation": "debug_aabb_proxy"}
            ),
            "room coordinate/scene production semantics drift",
        ),
        (
            lambda value: value["room"]["scene"].update(
                {"navmesh_policy": "recompute"}
            ),
            "room coordinate/scene production semantics drift",
        ),
        (
            lambda value: value["room"]["scene"].update({"load_semantic_mesh": False}),
            "room coordinate/scene production semantics drift",
        ),
        (
            lambda value: value["room"]["scene"].update({"enable_physics": False}),
            "room coordinate/scene production semantics drift",
        ),
    ],
)
def test_authority_join_rejects_cross_document_mutations(
    mutate, message: str, tmp_path: Path
) -> None:
    documents = _documents(tmp_path)
    mutate(documents)
    with pytest.raises(TwoHumanCaptureError, match=message):
        _validate(documents)


def test_authority_join_rejects_symlinked_source_glb(tmp_path: Path) -> None:
    documents = _documents(tmp_path)
    link = tmp_path / "female-link.glb"
    link.symlink_to(tmp_path / "female.glb")
    documents["atom"]["actor_framing"]["actor_bindings"][1]["source_asset_path"] = str(
        link
    )
    with pytest.raises(TwoHumanCaptureError, match="cannot contain symlinks"):
        _validate(documents)


def test_authority_rejects_legacy_runtime_root_manifest_without_data_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = tmp_path / "runtime_profiles.json"
    registry.write_text("{}", encoding="utf-8")
    atom = tmp_path / "atom.json"
    atom.write_text(
        json.dumps(
            {
                "actor_framing": {"runtime_profile_registry": str(registry)},
                "camera_runtime": {},
                "room": {},
            }
        ),
        encoding="utf-8",
    )
    companion = tmp_path / "companion.json"
    companion.write_text("{}", encoding="utf-8")
    prefix = tmp_path / "installed-prefix"
    prefix.mkdir()
    mp3d_root = tmp_path / "mp3d"
    (mp3d_root / "scene_datasets").mkdir(parents=True)
    monkeypatch.setattr(
        "avengine.m5_1.two_human_capture.load_m1_inputs",
        lambda *_args: SimpleNamespace(
            room={
                "scene": {
                    "scene_id": "${AVENGINE_HABITAT_RUNTIME_ROOT}/data/scene.glb",
                    "dataset_config_path": "${AVENGINE_HABITAT_RUNTIME_ROOT}/data/dataset.json",
                    "navmesh_path": "${AVENGINE_HABITAT_RUNTIME_ROOT}/data/scene.navmesh",
                }
            },
            request={},
        ),
    )
    monkeypatch.setattr(
        "avengine.m5_1.two_human_capture.load_source_asset_runtime_registry",
        lambda _path: {},
    )
    monkeypatch.setattr(
        "avengine.m5_1.two_human_capture.validate_two_human_authority_documents",
        lambda **_kwargs: pytest.fail(
            "legacy room must be rejected before authority validation"
        ),
    )

    with pytest.raises(TwoHumanCaptureError, match="do not map the legacy runtime-root"):
        load_two_human_capture_authority(
            atom_request_path=atom,
            suite_plan_path=companion,
            sensor_rig_path=companion,
            trajectory_bank_path=companion,
            rir_plan_path=companion,
            room_manifest_path=companion,
            m1_request_path=companion,
            runtime_prefix=prefix,
            mp3d_root=mp3d_root,
        )


def test_runtime_profile_registry_path_rejects_symlink(tmp_path: Path) -> None:
    registry = tmp_path / "profiles.json"
    registry.write_text("{}", encoding="utf-8")
    link = tmp_path / "profiles-link.json"
    link.symlink_to(registry)
    with pytest.raises(TwoHumanCaptureError, match="cannot contain symlinks"):
        _safe_regular_path(link, owner="runtime profile registry")


def test_runtime_samefile_allows_symlink_alias_and_rejects_other_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "scene.glb"
    target.write_bytes(b"scene")
    alias = tmp_path / "scene-alias.glb"
    alias.symlink_to(target)

    _require_same_runtime_file(alias, target, owner="scene")

    other = tmp_path / "other.glb"
    other.write_bytes(b"other")
    with pytest.raises(TwoHumanCaptureError, match="differs from resolved M1 runtime"):
        _require_same_runtime_file(alias, other, owner="scene")


def test_solver_and_render_agent_radii_are_independent_positive_values() -> None:
    _validate_camera_runtime_navigation(
        {"agent_height_m": 1.5, "agent_radius_m": 0.25},
        {"agent_height_m": 1.5, "agent_radius_m": 0.1},
    )
    for camera, navigation in (
        (
            {"agent_height_m": 1.5, "agent_radius_m": 0.0},
            {"agent_height_m": 1.5, "agent_radius_m": 0.1},
        ),
        (
            {"agent_height_m": 1.5, "agent_radius_m": 0.25},
            {"agent_height_m": 1.5, "agent_radius_m": -0.1},
        ),
    ):
        with pytest.raises(TwoHumanCaptureError, match="navigation values"):
            _validate_camera_runtime_navigation(camera, navigation)


def test_real_v4_authority_loader_skips_only_when_all_selectors_are_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _REAL_V4_AUTHORITY_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(pytest.skip.Exception, match="all explicit selectors"):
        _external_v4_authority_inputs_or_skip()


@pytest.mark.parametrize("missing_name", _REAL_V4_AUTHORITY_ENVIRONMENT)
def test_real_v4_authority_loader_rejects_partial_selector_configuration(
    monkeypatch: pytest.MonkeyPatch, missing_name: str
) -> None:
    for name in _REAL_V4_AUTHORITY_ENVIRONMENT:
        monkeypatch.setenv(name, f"/configured/{name}")
    monkeypatch.delenv(missing_name)

    with pytest.raises(AssertionError, match=missing_name):
        _external_v4_authority_inputs_or_skip()


@pytest.mark.parametrize("blank_name", _REAL_V4_AUTHORITY_ENVIRONMENT)
@pytest.mark.parametrize("blank_value", ("", " \t "))
def test_real_v4_authority_loader_rejects_blank_selector_configuration(
    monkeypatch: pytest.MonkeyPatch, blank_name: str, blank_value: str
) -> None:
    for name in _REAL_V4_AUTHORITY_ENVIRONMENT:
        monkeypatch.setenv(name, f"/configured/{name}")
    monkeypatch.setenv(blank_name, blank_value)

    with pytest.raises(AssertionError, match=f"non-empty: {blank_name}"):
        _external_v4_authority_inputs_or_skip()


def test_configured_v4_fixture_rejects_legacy_visual_mode(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="Habitat-native production"):
        _assert_current_v4_authority_fixture_semantics(
            {
                "schema": "avengine_native_strict_two_human_mp3d_room_atom_request_v2",
                "request_id": "mp3d_17DRP5sb8fy_strict_two_human_static_rig_v4",
                "visual_execution_mode": "comparison_visual",
            },
            {},
            runtime_prefix=tmp_path,
            mp3d_root=tmp_path,
        )


def test_real_v4_authority_loader_with_projected_m1_request(tmp_path: Path) -> None:
    """Load an explicit external v4 bundle; this is not an equivalence claim.

    The authority fixture, installed Habitat prefix, and MP3D data root are
    intentionally supplied by the caller.  Only the absence of all selectors
    means unavailable external fixture data; partial or malformed configuration
    and any configured integrity or semantic defect are ordinary test failures
    rather than skips.
    """

    configured = _external_v4_authority_inputs_or_skip()
    fixture = _configured_v4_authority_fixture(
        configured["AVENGINE_M5_1_V4_AUTHORITY_ROOT"]
    )
    runtime_prefix = resolve_installed_runtime_prefix(
        configured["AVENGINE_HABITAT_RUNTIME_PREFIX"]
    )
    mp3d_root = discover_mp3d_root(configured["AVENGINE_MP3D_ROOT"])
    assert mp3d_root is not None
    atom_payload = json.loads(fixture["atom"].read_text(encoding="utf-8"))
    room_payload = json.loads(fixture["room"].read_text(encoding="utf-8"))
    _assert_current_v4_authority_fixture_semantics(
        atom_payload,
        room_payload,
        runtime_prefix=runtime_prefix,
        mp3d_root=mp3d_root,
    )
    suite = json.loads(fixture["suite"].read_text(encoding="utf-8"))
    rig = json.loads(fixture["sensor_rig"].read_text(encoding="utf-8"))
    bank = json.loads(fixture["trajectory_bank"].read_text(encoding="utf-8"))
    scenario = suite["scenarios"][0]
    first_frame = scenario["plan"]["frames"][0]
    centers = bank["episodes"][0]["source_center_paths_m"]
    projected = {
        "schema": "avengine_m1_capture_request_v1",
        "request_id": "mp3d_two_human_projected_m1_v1",
        "room_id": "habitat_mp3d_example_17DRP5sb8fy",
        "seed": 17,
        "primary_camera_rig": {
            "rig_id": "camera_rig_0",
            "view_id": "view0",
            "world_from_rig": rig["frames"][0]["world_from_rig"],
            "shared_calibration": {
                "projection": "pinhole",
                "resolution_hw": [
                    scenario["render"]["height"],
                    scenario["render"]["width"],
                ],
                "hfov_degrees": scenario["render"]["horizontal_fov_deg"],
                "near_m": 0.05,
                "far_m": 100.0,
                "rig_from_sensor": {
                    "translation_m": [0.0, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            "modalities": [
                {"modality": "rgb", "sensor_uuid": "rig_rgb"},
                {"modality": "depth", "sensor_uuid": "rig_depth"},
                {"modality": "semantic", "sensor_uuid": "rig_semantic"},
            ],
        },
        "listener": {
            "listener_id": "listener0",
            "attached_to": "camera_rig_0",
            "rig_from_listener": {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
        "sources": [
            {
                "source_id": slot,
                "world_from_source": {
                    "translation_m": centers[slot][0],
                    "rotation_xyzw": first_frame["actor_states"][index][
                        "rotation_xyzw"
                    ],
                },
            }
            for index, slot in enumerate(("source1", "source2"))
        ],
        "qa_views": [
            {
                "qa_id": "navmesh_topdown",
                "kind": "topdown",
                "meters_per_pixel": 0.05,
                "height_m": 0.072447,
            }
        ],
    }
    request = tmp_path / "projected_m1.json"
    request.write_text(json.dumps(projected), encoding="utf-8")

    authority = load_two_human_capture_authority(
        atom_request_path=fixture["atom"],
        suite_plan_path=fixture["suite"],
        sensor_rig_path=fixture["sensor_rig"],
        trajectory_bank_path=fixture["trajectory_bank"],
        rir_plan_path=fixture["rir_plan"],
        room_manifest_path=fixture["room"],
        m1_request_path=request,
        runtime_prefix=runtime_prefix,
        mp3d_root=mp3d_root,
    )

    assert authority.episode_id == scenario["scenario_id"]
    assert authority.resolution_hw == (720, 1280)


def _lifecycle_dependencies(
    tmp_path: Path, *, fail_frame: int | None = None
) -> tuple[_TwoHumanCaptureDependencies, dict[str, object]]:
    authority_root = tmp_path / "authority"
    authority_root.mkdir()
    authority = replace(
        _validate(_documents(authority_root)),
        resolution_hw=(2, 3),
        horizontal_fov_deg=90.0,
    )
    runtime_root = tmp_path / "habitat_runtime"
    scene = runtime_root / "scene.glb"
    dataset = runtime_root / "dataset.scene_dataset_config.json"
    navmesh = runtime_root / "scene.navmesh"
    physics = runtime_root / "data/default.physics_config.json"
    for path in (scene, dataset, navmesh, physics):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")
    resolved_scene = {
        "scene_id": scene,
        "dataset_config": dataset,
        "navmesh": navmesh,
        "navmesh_policy": "load_declared",
        "load_semantic_mesh": True,
        "enable_physics": True,
    }
    request = {
        "seed": 17,
        "primary_camera_rig": {
            "rig_id": "camera_rig_0",
            "view_id": "view0",
            "shared_calibration": {
                "resolution_hw": [2, 3],
                "hfov_degrees": 90.0,
            },
        },
        "listener": {"listener_id": "listener0"},
    }
    m1_inputs = SimpleNamespace(request=request, room={})
    counters: dict[str, object] = {
        "factory": 0,
        "prepare": [],
        "instantiate": [],
        "render": 0,
        "physics": 0,
        "frames": [],
        "seed": None,
        "events": [],
        "camera_sensor_sets": [],
    }

    class FakeActions:
        def __init__(self, walking_count: int) -> None:
            self.walking_count = walking_count

        def action(self, action_id: str) -> SimpleNamespace:
            assert action_id == "walk"
            return SimpleNamespace(sample_count=self.walking_count)

    def prepare_runtime(
        source_glb,
        output_dir,
        *,
        package_stem,
        walking_profile_sample_count,
        anatomical_forward_source,
    ):
        output = Path(output_dir)
        output.mkdir(parents=True)
        ao_config = output / f"{package_stem}.ao_config.json"
        ao_config.write_text("{}", encoding="utf-8")
        manifest = output / "human_runtime_manifest.json"
        manifest.write_text(
            json.dumps({"anatomical_frame": {"source": anatomical_forward_source}}),
            encoding="utf-8",
        )
        walking_count = walking_profile_sample_count or 16
        counters["prepare"].append(
            (
                Path(source_glb),
                package_stem,
                walking_profile_sample_count,
                anatomical_forward_source,
            )
        )
        return SimpleNamespace(
            actions=FakeActions(walking_count),
            habitat_ao_config=ao_config,
            package_manifest=manifest,
        )

    sim_cfg = SimpleNamespace(
        scene_id=str(scene),
        scene_dataset_config_file=str(dataset),
        load_semantic_mesh=True,
        enable_physics=True,
        enable_hbao=False,
    )
    configuration = SimpleNamespace(sim_cfg=sim_cfg)
    modality_to_uuid = {
        "rgb": "rig_rgb",
        "depth": "rig_depth",
        "semantic": "rig_semantic",
    }

    class FakeAgentState:
        def __init__(self) -> None:
            self.position = None
            self.rotation = None

    class FakePathfinder:
        def __init__(self) -> None:
            self.is_loaded = False

        def load_nav_mesh(self, path: str) -> bool:
            assert Path(path) == navmesh
            self.is_loaded = True
            return True

    class FakeSimulator:
        def __init__(self, requested_configuration) -> None:
            assert requested_configuration is configuration
            self.config = requested_configuration
            self.pathfinder = FakePathfinder()
            self.sensors = {uuid: object() for uuid in modality_to_uuid.values()}
            self.world_time = 0.0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def seed(self, value: int) -> None:
            counters["seed"] = value

        def initialize_agent(self, index: int, state):
            assert index == 0
            assert state.position is not None and state.rotation is not None
            return SimpleNamespace(simulator=self)

        def get_world_time(self) -> float:
            return self.world_time

        def step_physics(self, _seconds: float) -> None:
            counters["physics"] += 1

        def render_sensors(self, wrappers):
            assert len(wrappers) == 3
            counters["render"] += 1
            counters["events"].append(
                "preflight_render" if not counters["instantiate"] else "formal_render"
            )
            semantic = np.zeros((2, 3), dtype=np.int32)
            if counters["instantiate"]:
                semantic[0, 0] = 62000
                semantic[0, 1] = 62001
            return {
                "rig_rgb": np.zeros((2, 3, 4), dtype=np.uint8),
                "rig_depth": np.ones((2, 3), dtype=np.float32),
                "rig_semantic": semantic,
            }

    simulator_instances: list[FakeSimulator] = []

    def simulator_factory(_habitat_sim, requested_configuration):
        counters["factory"] += 1
        counters["events"].append("factory")
        simulator = FakeSimulator(requested_configuration)
        simulator_instances.append(simulator)
        return simulator

    def instantiate_human(
        simulator,
        *,
        package,
        habitat_sim,
        semantic_id,
        light_setup_key,
        shader_type,
    ):
        del habitat_sim, light_setup_key, shader_type
        counters["instantiate"].append(
            (id(simulator), package.habitat_ao_config.name, semantic_id)
        )
        counters["events"].append(f"instantiate:{semantic_id}")
        actor = SimpleNamespace(simulator=simulator, semantic_id=semantic_id)
        return actor, SimpleNamespace(), (SimpleNamespace(),)

    def state_snapshot(simulator, agent, sensor_uuids, quat_to_coeffs):
        del agent, quat_to_coeffs
        pose = deepcopy(authority.rig_frames[0]["world_from_rig"])
        return {
            "world_time_seconds": simulator.world_time,
            "agent": deepcopy(pose),
            "sensors": {uuid: deepcopy(pose) for uuid in sensor_uuids},
        }

    def frame_capture(**kwargs):
        frame_index = kwargs["frame_index"]
        counters["frames"].append(frame_index)
        if fail_frame == frame_index:
            raise TwoHumanCaptureError("injected frame failure")
        simulator = kwargs["simulator"]
        runtimes = kwargs["runtimes"]
        assert len(runtimes) == 2
        assert all(
            runtime.articulated_object.simulator is simulator for runtime in runtimes
        )
        sensor_uuids = tuple(kwargs["camera_sensor_uuids"])
        assert set(sensor_uuids) == {
            "rig_rgb",
            "rig_depth",
            "rig_semantic",
            "listener0",
        }
        snapshot = kwargs["camera_snapshot"]()
        assert set(snapshot["sensors"]) == set(sensor_uuids)
        counters["camera_sensor_sets"].append(sensor_uuids)
        observation = simulator.render_sensors(kwargs["sensor_wrappers"])
        semantic = observation["rig_semantic"]
        roots = np.stack((np.eye(4), np.eye(4)))
        return _CapturedTwoHumanFrame(
            rgb=observation["rig_rgb"][..., :3].copy(),
            depth_m=observation["rig_depth"].copy(),
            semantic=semantic.copy(),
            actor_root_world_matrices=roots.copy(),
            skin_root_world_matrices=roots.copy(),
            anchor_positions_m=np.zeros((2, 2, 3), dtype=np.float64),
            semantic_visibility_pixels=np.asarray([1, 1], dtype=np.int64),
            record={"frame_index": frame_index, "physics_steps": 0},
        )

    fake_habitat = SimpleNamespace(AgentState=FakeAgentState)

    def fake_quaternion(w, x, y, z):
        return np.asarray([x, y, z, w])

    dependencies = _TwoHumanCaptureDependencies(
        authority_loader=lambda **_kwargs: authority,
        m1_loader=lambda _room, _request: m1_inputs,
        prepare_runtime=prepare_runtime,
        package_manifest_loader=lambda path: json.loads(
            Path(path).read_text(encoding="utf-8")
        ),
        runtime_discover=lambda _root: runtime_root,
        make_configuration=lambda _inputs, _runtime, _scratch: (
            configuration,
            modality_to_uuid,
            "listener0",
            resolved_scene,
        ),
        import_habitat=lambda: (
            SimpleNamespace(quaternion=fake_quaternion),
            fake_habitat,
            SimpleNamespace(),
            lambda value: np.asarray(value),
        ),
        simulator_factory=simulator_factory,
        bind_lighting=lambda *_args, **_kwargs: {"status": "pass"},
        instantiate_human=instantiate_human,
        actor_render_evidence=lambda *_args, **_kwargs: {"status": "pass"},
        link_id_by_name=lambda _actor, name: name,
        state_snapshot=state_snapshot,
        observation_validator=lambda observation, mapping: {
            modality: observation[uuid] for modality, uuid in mapping.items()
        },
        frame_capture=frame_capture,
    )
    context = {
        "authority": authority,
        "runtime_root": runtime_root,
        "counters": counters,
        "simulator_instances": simulator_instances,
    }
    return dependencies, context


def _run_fake_lifecycle(tmp_path: Path, *, fail_frame: int | None = None):
    dependencies, context = _lifecycle_dependencies(tmp_path, fail_frame=fail_frame)
    output = tmp_path / "capture"
    result = capture_two_human_mp3d(
        atom_request_path=tmp_path / "atom.json",
        suite_plan_path=tmp_path / "suite.json",
        sensor_rig_path=tmp_path / "rig.json",
        trajectory_bank_path=tmp_path / "trajectory.json",
        rir_plan_path=tmp_path / "rir.json",
        room_manifest_path=tmp_path / "room.json",
        m1_request_path=tmp_path / "m1.json",
        output_dir=output,
        runtime_root=context["runtime_root"],
        _dependencies=dependencies,
    )
    return result, output, context


def test_capture_lifecycle_uses_one_simulator_and_publishes_plain_evidence(
    tmp_path: Path,
) -> None:
    result, output, context = _run_fake_lifecycle(tmp_path)
    counters = context["counters"]
    authority = context["authority"]
    assert counters["factory"] == 1
    assert counters["render"] == 76
    assert counters["physics"] == 0
    assert counters["seed"] == 17
    assert counters["frames"] == list(range(75))
    assert counters["prepare"] == [
        (
            authority.actors[0].source_glb,
            "human0",
            None,
            authority.actors[0].anatomical_forward_source,
        ),
        (
            authority.actors[1].source_glb,
            "human1",
            19,
            authority.actors[1].anatomical_forward_source,
        ),
    ]
    assert [item[1:] for item in counters["instantiate"]] == [
        ("human0.ao_config.json", 62000),
        ("human1.ao_config.json", 62001),
    ]
    assert len({item[0] for item in counters["instantiate"]}) == 1
    assert counters["events"][:4] == [
        "factory",
        "preflight_render",
        "instantiate:62000",
        "instantiate:62001",
    ]
    assert counters["events"][4:] == ["formal_render"] * 75
    assert len(counters["camera_sensor_sets"]) == 75
    assert result.rgb.shape == (75, 2, 3, 3)
    assert result.depth_m.shape == (75, 2, 3)
    assert result.semantic.shape == (75, 2, 3)
    evidence = result.evidence
    assert evidence["backend_role"] == "production_visual"
    assert evidence["source_suite_role"] == "comparison_visual"
    assert evidence["research_only"] is True
    assert evidence["manual_review_status"] == "pending"
    assert evidence["qualification_claim"] is False
    assert evidence["formal_dataset_count"] == 0
    assert evidence["simulator_instances"] == 1
    assert evidence["preflight_render_calls"] == 1
    assert evidence["formal_render_calls"] == 75
    assert evidence["seed"] == 17
    assert evidence["camera"] == {
        "rig_id": "camera_rig_0",
        "view_id": "view0",
        "listener_id": "listener0",
        "resolution_hw": [2, 3],
        "hfov_degrees": 90.0,
        "modality_to_sensor_uuid": {
            "rgb": "rig_rgb",
            "depth": "rig_depth",
            "semantic": "rig_semantic",
        },
    }
    assert set(evidence["preflight_semantic_absence"]["semantic_ids"]) == {
        "source1_actor",
        "source2_actor",
    }
    assert evidence["semantic_visible_frame_count"] == {
        "source1_actor": 75,
        "source2_actor": 75,
    }
    assert [item["asset_id"] for item in evidence["actors"]] == [
        authority.actors[0].asset_id,
        authority.actors[1].asset_id,
    ]
    assert (output / "evidence.json").is_file()
    assert (output / "frame_readback.json").is_file()
    assert len(evidence["array_artifacts"]) == 7
    assert all(
        (output / record["path"]).is_file() and record["readback_verified"] is True
        for record in evidence["array_artifacts"].values()
    )

    def forbidden_keys(value):
        if isinstance(value, dict):
            return {
                key for key, child in value.items() if key in {"sha256", "byte_size"}
            } | set().union(*(forbidden_keys(child) for child in value.values()))
        if isinstance(value, list):
            return set().union(*(forbidden_keys(child) for child in value))
        return set()

    assert forbidden_keys(evidence) == set()


def test_capture_lifecycle_failure_never_publishes_pass_evidence(
    tmp_path: Path,
) -> None:
    dependencies, context = _lifecycle_dependencies(tmp_path, fail_frame=4)
    output = tmp_path / "capture"
    with pytest.raises(TwoHumanCaptureError, match="injected frame failure"):
        capture_two_human_mp3d(
            atom_request_path=tmp_path / "atom.json",
            suite_plan_path=tmp_path / "suite.json",
            sensor_rig_path=tmp_path / "rig.json",
            trajectory_bank_path=tmp_path / "trajectory.json",
            rir_plan_path=tmp_path / "rir.json",
            room_manifest_path=tmp_path / "room.json",
            m1_request_path=tmp_path / "m1.json",
            output_dir=output,
            runtime_root=context["runtime_root"],
            _dependencies=dependencies,
        )
    assert output.is_dir()
    assert not (output / "evidence.json").exists()
    assert not (output / "frame_readback.json").exists()
    assert not (output / "arrays").exists()


def _two_human_cli_argv(tmp_path: Path, *, include_runtime_root: bool) -> list[str]:
    argv = [
        "--atom-request",
        str(tmp_path / "atom.json"),
        "--suite-plan",
        str(tmp_path / "suite.json"),
        "--sensor-rig",
        str(tmp_path / "rig.json"),
        "--trajectory-bank",
        str(tmp_path / "trajectory.json"),
        "--rir-plan",
        str(tmp_path / "rir.json"),
        "--room-manifest",
        str(tmp_path / "room.json"),
        "--m1-request",
        str(tmp_path / "m1.json"),
        "--output",
        str(tmp_path / "capture"),
    ]
    if include_runtime_root:
        argv.extend(["--runtime-root", str(tmp_path / "runtime")])
    return argv


def test_two_human_cli_maps_all_kernel_arguments_and_reports_plain_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, object]] = []
    output = tmp_path / "capture"
    evidence = {
        "status": "pass",
        "status_scope": "native_capture_execution",
        "backend_role": "production_visual",
        "frame_count": 75,
        "research_only": True,
        "manual_review_status": "pending",
        "formal_dataset_count": 0,
        "semantic_visible_frame_count": {
            "source1_actor": 75,
            "source2_actor": 75,
        },
    }

    def fake_capture(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_dir=output, evidence=evidence)

    monkeypatch.setattr(CLI, "capture_two_human_mp3d", fake_capture)
    assert CLI.main(_two_human_cli_argv(tmp_path, include_runtime_root=True)) == 0
    assert calls == [
        {
            "atom_request_path": tmp_path / "atom.json",
            "suite_plan_path": tmp_path / "suite.json",
            "sensor_rig_path": tmp_path / "rig.json",
            "trajectory_bank_path": tmp_path / "trajectory.json",
            "rir_plan_path": tmp_path / "rir.json",
            "room_manifest_path": tmp_path / "room.json",
            "m1_request_path": tmp_path / "m1.json",
            "output_dir": output,
            "runtime_prefix": None,
            "runtime_root": tmp_path / "runtime",
            "mp3d_root": None,
            "magnum_python_site": None,
        }
    ]
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "status": "pass",
        "status_scope": "native_capture_execution",
        "backend_role": "production_visual",
        "output": str(output),
        "evidence": str(output / "evidence.json"),
        "frame_count": 75,
        "research_only": True,
        "manual_review_status": "pending",
        "formal_dataset_count": 0,
        "semantic_visible_frame_count": {
            "source1_actor": 75,
            "source2_actor": 75,
        },
    }
    assert all("hash" not in key and "sha" not in key for key in payload)


def test_two_human_cli_rejects_prefix_and_root_together(
    tmp_path: Path,
) -> None:
    parser = CLI._parser()
    argv = _two_human_cli_argv(tmp_path, include_runtime_root=False)
    argv.extend(
        [
            "--runtime-prefix",
            str(tmp_path / "prefix"),
            "--runtime-root",
            str(tmp_path / "root"),
        ]
    )
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


def test_two_human_cli_accepts_prefix_and_external_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_capture(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            output_dir=tmp_path / "capture",
            evidence={
                "status": "pass",
                "status_scope": "native_capture_execution",
                "backend_role": "production_visual",
                "frame_count": 75,
                "research_only": True,
                "manual_review_status": "pending",
                "formal_dataset_count": 0,
                "semantic_visible_frame_count": {
                    "source1_actor": 75,
                    "source2_actor": 75,
                },
            },
        )

    monkeypatch.setattr(CLI, "capture_two_human_mp3d", fake_capture)
    argv = _two_human_cli_argv(tmp_path, include_runtime_root=False)
    argv.extend(
        [
            "--runtime-prefix",
            str(tmp_path / "prefix"),
            "--mp3d-root",
            str(tmp_path / "mp3d"),
            "--magnum-python-site",
            str(tmp_path / "magnum"),
        ]
    )
    assert CLI.main(argv) == 0
    assert calls[0]["runtime_prefix"] == tmp_path / "prefix"
    assert calls[0]["runtime_root"] is None
    assert calls[0]["mp3d_root"] == tmp_path / "mp3d"
    assert calls[0]["magnum_python_site"] == tmp_path / "magnum"


@pytest.mark.parametrize("error_type", [TwoHumanCaptureError, RuntimeError])
def test_two_human_cli_reports_capture_error_as_exit_two(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    error_type: type[RuntimeError],
) -> None:
    calls: list[dict[str, object]] = []

    def fail_capture(**kwargs):
        calls.append(kwargs)
        raise error_type("injected CLI capture failure")

    monkeypatch.setattr(CLI, "capture_two_human_mp3d", fail_capture)
    with pytest.raises(SystemExit) as raised:
        CLI.main(_two_human_cli_argv(tmp_path, include_runtime_root=False))
    assert raised.value.code == 2
    assert len(calls) == 1
    assert calls[0]["runtime_prefix"] is None
    assert calls[0]["runtime_root"] is None
    assert calls[0]["mp3d_root"] is None
    assert calls[0]["magnum_python_site"] is None
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "injected CLI capture failure" in captured.err
