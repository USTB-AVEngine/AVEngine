from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from avengine.contracts.json_io import sha256_file, write_json
from avengine.m1.contracts import ValidatedM1Inputs
from avengine.m2.actions import (
    BakedActionClip,
    BakedActionSet,
    TIME_BASE_HZ,
    write_baked_actions_npz,
)
from avengine.m2.contracts import (
    APPLIED_STATE_HASH_ALGORITHM,
    CONTACT_ORDER,
    POSE_HASH_ALGORITHM,
    ValidatedM2Inputs,
    compute_applied_state_hash,
    compute_pose_hash,
)
from avengine.m2.habitat import HabitatJointBinding, HabitatLinkJointBlock
from avengine.m2.habitat_capture import (
    CapturedFrame,
    FrameApplication,
    HabitatCaptureError,
    RuntimeAssetBundle,
    apply_and_capture_fixed_frame,
    compile_frame_applications,
    load_research_review_inputs,
    load_runtime_asset_bundle,
    quaternion_xyzw_to_matrix,
    save_capture_arrays,
    transform_to_matrix,
    validate_capture_context,
    validate_research_review_context,
    verify_saved_capture_arrays,
    write_research_review_media,
)


RUNTIME_ORDER = ("spine", "head")
IDENTITY_QUATERNION = (0.0, 0.0, 0.0, 1.0)
TURN_QUATERNION = (
    0.0,
    0.0,
    math.sin(math.pi / 8.0),
    math.cos(math.pi / 8.0),
)
IDENTITY_MATRIX = tuple(
    tuple(float(value) for value in row) for row in np.eye(4, dtype=np.float64)
)


def _clip(
    action_id: str,
    source_name: str,
    *,
    first: tuple[float, float, float, float] = IDENTITY_QUATERNION,
    second: tuple[float, float, float, float] = TURN_QUATERNION,
) -> BakedActionClip:
    loop_duration_ticks = 6_400
    return BakedActionClip(
        semantic_action_id=action_id,
        source_action_name=source_name,
        clip_start_seconds=0.0,
        clip_end_seconds=loop_duration_ticks / TIME_BASE_HZ,
        loop_duration_ticks=loop_duration_ticks,
        sample_ticks=(0, 3_200),
        source_times_seconds=(0.0, 3_200 / TIME_BASE_HZ),
        rotations_xyzw=(
            (first, first),
            (second, second),
        ),
    )


def _action_set(visual_sha256: str = "a" * 64) -> BakedActionSet:
    return BakedActionSet(
        source_glb_sha256=visual_sha256,
        runtime_joint_order=RUNTIME_ORDER,
        actions=(
            _clip("idle", "Idle"),
            _clip("walk", "Walking"),
        ),
    )


def _asset() -> dict[str, Any]:
    return {
        "asset_id": "dog_canary_v1",
        "admission_state": "canary_qualified",
        "coordinate_system": {
            "handedness": "right",
            "up_axis": "+Y",
            "forward_axis": "-Z",
            "linear_unit": "meter",
            "quaternion_order": "xyzw",
        },
        "revisions": {"skeleton_revision": "dog_skeleton_v1"},
        "skeleton": {
            "root_joint_id": "root",
            "joint_order": ["root", *RUNTIME_ORDER],
            "runtime_joint_order": list(RUNTIME_ORDER),
        },
        "actions": [
            {
                "action_id": "idle",
                "poses_file_role": "idle_poses",
                "sample_count": 2,
            },
            {
                "action_id": "walk",
                "poses_file_role": "walk_poses",
                "sample_count": 2,
            },
        ],
        "files": [],
    }


def _state(
    asset: dict[str, Any],
    *,
    frame_index: int,
    action_id: str,
    action_time_ticks: int,
    pose: tuple[tuple[float, float, float, float], ...],
    asset_manifest_sha256: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "frame_index": frame_index,
        "pts_ticks": frame_index * 3_200,
        "action_id": action_id,
        "action_time_ticks": action_time_ticks,
        "root_transform": {
            "translation_m": [frame_index * 0.01, 0.0, 0.0],
            "rotation_xyzw": list(IDENTITY_QUATERNION),
        },
        "joint_states": [
            {"joint_id": name, "rotation_xyzw": list(rotation)}
            for name, rotation in zip(RUNTIME_ORDER, pose, strict=True)
        ],
        "contact_states": [
            {"contact_id": contact_id, "in_contact": False}
            for contact_id in CONTACT_ORDER
        ],
        "mouth_state": {"open_ratio": 0.0, "vocalizing": False},
    }
    value["pose_hash"] = compute_pose_hash(asset, value)
    value["applied_state_hash"] = compute_applied_state_hash(
        asset,
        value,
        asset_manifest_sha256=asset_manifest_sha256,
    )
    return value


def _compiled_inputs(tmp_path: Path) -> tuple[ValidatedM2Inputs, RuntimeAssetBundle]:
    asset = _asset()
    asset_path = tmp_path / "asset.json"
    asset_path.write_text("fixture manifest bytes\n", encoding="utf-8")
    asset_manifest_sha256 = sha256_file(asset_path)
    action_set = _action_set()
    states: list[dict[str, Any]] = []
    for frame_index in range(75):
        action_id = "idle" if frame_index < 15 else "walk"
        action_time_ticks = (
            frame_index * 3_200 if action_id == "idle" else (frame_index - 15) * 3_200
        )
        clip = action_set.action(action_id)
        sample = clip.rotations_xyzw[
            clip.sample_ticks.index(action_time_ticks % clip.loop_duration_ticks)
        ]
        states.append(
            _state(
                asset,
                frame_index=frame_index,
                action_id=action_id,
                action_time_ticks=action_time_ticks,
                pose=sample,
                asset_manifest_sha256=asset_manifest_sha256,
            )
        )
    request = {
        "request_id": "request_v1",
        "room_id": "room_v1",
        "asset_id": asset["asset_id"],
        "seed": 17,
        "camera_rig_id": "camera_rig_0",
        "listener_id": "listener0",
        "view_ids": ["view0"],
        "modalities": ["rgb", "depth", "semantic"],
        "runtime_joint_order": list(RUNTIME_ORDER),
        "pose_hash_algorithm": POSE_HASH_ALGORITHM,
        "applied_state_hash_algorithm": APPLIED_STATE_HASH_ALGORITHM,
        "states": states,
    }
    inputs = ValidatedM2Inputs(
        asset_path=asset_path,
        request_path=tmp_path / "request.json",
        asset=asset,
        request=request,
    )
    bundle = RuntimeAssetBundle(
        paths_by_role={},
        records_by_role={},
        joint_mapping={},
        actor_from_skin_root=IDENTITY_MATRIX,
        action_sets_by_role={"idle_poses": action_set, "walk_poses": action_set},
        action_roles_by_id={"idle": "idle_poses", "walk": "walk_poses"},
        semantic_id=200,
    )
    return inputs, bundle


def _context_inputs(tmp_path: Path) -> tuple[ValidatedM2Inputs, ValidatedM1Inputs]:
    asset = _asset()
    m2 = ValidatedM2Inputs(
        asset_path=tmp_path / "asset.json",
        request_path=tmp_path / "m2_request.json",
        asset=asset,
        request={
            "room_id": "room_v1",
            "camera_rig_id": "camera_rig_0",
            "listener_id": "listener0",
            "seed": 17,
            "view_ids": ["view0"],
            "modalities": ["rgb", "depth", "semantic"],
        },
    )
    m1 = ValidatedM1Inputs(
        room_path=tmp_path / "room.json",
        request_path=tmp_path / "m1_request.json",
        room={"room_id": "room_v1"},
        request={
            "seed": 17,
            "primary_camera_rig": {
                "rig_id": "camera_rig_0",
                "view_id": "view0",
                "modalities": [
                    {"modality": "rgb", "sensor_uuid": "rgb0"},
                    {"modality": "depth", "sensor_uuid": "depth0"},
                    {"modality": "semantic", "sensor_uuid": "semantic0"},
                ],
            },
            "listener": {"listener_id": "listener0"},
        },
    )
    return m2, m1


def _file_record(path: Path, role: str, package_root: Path) -> dict[str, Any]:
    return {
        "role": role,
        "path": path.relative_to(package_root).as_posix(),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def test_formal_runtime_identity_requires_locked_native_binding() -> None:
    from avengine.m2 import habitat_capture

    identity = {
        "habitat_runtime_worktree_dirty": False,
        "avengine_worktree_dirty": False,
        "runtime_commit_matches_lock": True,
        "runtime_binary_origin_matches": True,
        "native_binding_matches_lock": False,
    }

    assert habitat_capture._formal_runtime_identity_errors(identity) == [
        "native Habitat binding SHA-256 differs from lock"
    ]
    identity["native_binding_matches_lock"] = True
    assert habitat_capture._formal_runtime_identity_errors(identity) == []


def test_locked_native_binding_hash_is_exact(tmp_path: Path) -> None:
    from avengine.m2 import habitat_capture

    expected = "a" * 64
    (tmp_path / "runtime.lock.yaml").write_text(
        f"runtime_test_environment:\n  required_m2_native_binding_sha256: {expected}\n",
        encoding="utf-8",
    )

    assert habitat_capture._locked_native_binding_sha256(tmp_path) == expected

    (tmp_path / "runtime.lock.yaml").write_text(
        "runtime_test_environment:\n  required_m2_native_binding_sha256: not-a-sha\n",
        encoding="utf-8",
    )
    assert habitat_capture._locked_native_binding_sha256(tmp_path) is None


def _runtime_package(tmp_path: Path) -> ValidatedM2Inputs:
    package_root = tmp_path / "package"
    package_root.mkdir()
    visual_path = package_root / "visual.glb"
    visual_path.write_bytes(b"formal visual")
    visual_sha256 = sha256_file(visual_path)
    urdf_path = package_root / "animal.urdf"
    urdf_path.write_text("<robot name='fixture'/>\n", encoding="utf-8")
    config_path = package_root / "animal.ao_config.json"
    write_json(
        config_path,
        {
            "render_asset": "visual.glb",
            "urdf_filepath": "animal.urdf",
            "render_mode": "skin",
            "semantic_id": 200,
            "user_defined": {"avengine_native_gltf_skin_frame": True},
        },
    )
    mapping_path = package_root / "joint_mapping.json"
    asset = _asset()
    write_json(
        mapping_path,
        {
            "schema": "avengine_m2_habitat_joint_mapping_v1",
            "source_glb_sha256": visual_sha256,
            "coordinate_system": asset["coordinate_system"],
            "root_joint_id": "root",
            "joint_order": ["root", *RUNTIME_ORDER],
            "runtime_joint_order": list(RUNTIME_ORDER),
            "actor_from_skin_root": [list(row) for row in IDENTITY_MATRIX],
            "actor_from_skin_root_source": "fixture.rebase",
            "runtime_root_formula": (
                "world_from_skin_root = world_from_actor @ actor_from_skin_root"
            ),
            "habitat_layout": {
                "base_link": "root",
                "runtime_joint_type": "spherical",
                "runtime_joint_position_count": 8,
                "runtime_joint_position_encoding": "xyzw",
                "render_mode": "skin",
            },
        },
    )
    action_set = _action_set(visual_sha256)
    idle_path = package_root / "idle.npz"
    walk_path = package_root / "walk.npz"
    write_baked_actions_npz(action_set, idle_path)
    write_baked_actions_npz(action_set, walk_path)
    role_paths = {
        "visual": visual_path,
        "habitat_urdf": urdf_path,
        "habitat_ao_config": config_path,
        "habitat_joint_mapping": mapping_path,
        "idle_poses": idle_path,
        "walk_poses": walk_path,
    }
    asset["files"] = [
        _file_record(path, role, package_root) for role, path in role_paths.items()
    ]
    asset_path = package_root / "asset.json"
    write_json(asset_path, asset)
    return ValidatedM2Inputs(
        asset_path=asset_path,
        request_path=package_root / "request.json",
        asset=asset,
        request={"runtime_joint_order": list(RUNTIME_ORDER)},
    )


class _FakeRootNode:
    def __init__(self, owner: "_FakeAO") -> None:
        self.owner = owner

    def absolute_transformation(self) -> np.ndarray:
        return self.owner.root.copy()


class _FakeAO:
    def __init__(self, joint_count: int) -> None:
        self.root = np.eye(4, dtype=np.float64)
        self.joint_positions = np.zeros(joint_count, dtype=np.float64)
        self.root_scene_node = _FakeRootNode(self)


class _FakeSimulator:
    def __init__(self, *, advance_time: bool = False) -> None:
        self.time = 0.0
        self.advance_time = advance_time
        self.render_calls = 0

    def get_world_time(self) -> float:
        return self.time

    def render_sensors(self, wrappers: list[Any]) -> dict[str, np.ndarray]:
        assert len(wrappers) == 3
        self.render_calls += 1
        if self.advance_time:
            self.time += 1.0 / 60.0
        return {
            "rgb0": np.arange(24, dtype=np.uint8).reshape(2, 3, 4),
            "depth0": np.arange(6, dtype=np.float32).reshape(2, 3),
            "semantic0": np.arange(6, dtype=np.int32).reshape(2, 3),
        }


def _binding() -> HabitatJointBinding:
    return HabitatJointBinding(
        runtime_joint_order=RUNTIME_ORDER,
        joint_position_count=8,
        blocks=(
            HabitatLinkJointBlock("spine", 0, 4),
            HabitatLinkJointBlock("head", 4, 4),
        ),
    )


def _frame() -> FrameApplication:
    return FrameApplication(
        frame_index=0,
        pts_ticks=0,
        action_id="idle",
        action_time_ticks=0,
        effective_action_tick=0,
        action_sample_index=0,
        world_from_actor=IDENTITY_MATRIX,
        world_from_skin_root=IDENTITY_MATRIX,
        joint_rotations_xyzw=(IDENTITY_QUATERNION, TURN_QUATERNION),
        declared_pose_hash="1" * 64,
        recomputed_pose_hash="1" * 64,
        declared_applied_state_hash="2" * 64,
        recomputed_applied_state_hash="2" * 64,
    )


def _capture(simulator: _FakeSimulator | None = None) -> CapturedFrame:
    sim = simulator or _FakeSimulator()
    ao = _FakeAO(8)

    def apply_root(target: _FakeAO, matrix: np.ndarray) -> None:
        target.root = matrix.copy()

    return apply_and_capture_fixed_frame(
        simulator=sim,
        articulated_object=ao,
        frame=_frame(),
        joint_binding=_binding(),
        modality_to_uuid={
            "rgb": "rgb0",
            "depth": "depth0",
            "semantic": "semantic0",
        },
        sensor_wrappers=[object(), object(), object()],
        apply_root_transform=apply_root,
        required_semantic_id=5,
    )


def test_transform_matrix_uses_xyzw_and_composes_translation() -> None:
    rotation = quaternion_xyzw_to_matrix(
        [0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)]
    )
    matrix = transform_to_matrix(
        {
            "translation_m": [1.0, 2.0, 3.0],
            "rotation_xyzw": [
                0.0,
                0.0,
                math.sin(math.pi / 4.0),
                math.cos(math.pi / 4.0),
            ],
        }
    )

    assert np.allclose(rotation @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(matrix[:3, :3], rotation, atol=1e-12)
    assert matrix[:3, 3].tolist() == [1.0, 2.0, 3.0]


def test_capture_context_requires_joined_m1_view0_and_canary(tmp_path: Path) -> None:
    m2, m1 = _context_inputs(tmp_path)

    assert validate_capture_context(m2, m1) == []

    m1.request["primary_camera_rig"]["view_id"] = "qa_side"
    m2.asset["admission_state"] = "research_candidate"
    errors = validate_capture_context(m2, m1)
    assert "formal Habitat capture accepts only canary_qualified assets" in errors
    assert "M1 runtime input must provide only camera_rig_0/view0" in errors
    assert validate_capture_context({}, m1) == [
        "inputs must be ValidatedM2Inputs from M2 load_and_validate_inputs"
    ]


def test_research_context_is_separate_and_never_accepted_as_formal(
    tmp_path: Path,
) -> None:
    m2, m1 = _context_inputs(tmp_path)
    m2.asset["admission_state"] = "research_candidate"

    assert "formal Habitat capture accepts only canary_qualified assets" in (
        validate_capture_context(m2, m1)
    )
    assert validate_research_review_context(m2, m1) == []

    m2.asset["admission_state"] = "canary_qualified"
    assert validate_research_review_context(m2, m1) == [
        "research review accepts only research_candidate assets"
    ]


def test_review_loader_exempts_only_the_formal_admission_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from avengine.m2 import habitat_capture

    asset_path = tmp_path / "asset.json"
    request_path = tmp_path / "request.json"
    write_json(asset_path, {"admission_state": "research_candidate"})
    write_json(request_path, {"states": []})
    monkeypatch.setattr(
        habitat_capture, "validate_animal_asset_package", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        habitat_capture,
        "validate_capture_request",
        lambda *_args, **_kwargs: [
            "M2 capture accepts only a canary_qualified animal package"
        ],
    )

    inputs = load_research_review_inputs(asset_path, request_path)

    assert inputs.asset["admission_state"] == "research_candidate"
    monkeypatch.setattr(
        habitat_capture,
        "validate_capture_request",
        lambda *_args, **_kwargs: [
            "M2 capture accepts only a canary_qualified animal package",
            "states must contain exactly 75 items",
        ],
    )
    with pytest.raises(HabitatCaptureError, match="exactly 75"):
        load_research_review_inputs(asset_path, request_path)


def test_runtime_bundle_loads_role_bound_complete_action_sets(tmp_path: Path) -> None:
    inputs = _runtime_package(tmp_path)

    bundle = load_runtime_asset_bundle(inputs)

    assert bundle.semantic_id == 200
    assert bundle.actor_from_skin_root == IDENTITY_MATRIX
    assert bundle.action_sets_by_role["idle_poses"].action("idle").sample_count == 2
    assert bundle.action_sets_by_role["walk_poses"].action("walk").sample_count == 2


def test_runtime_bundle_rejects_ao_config_role_divergence(tmp_path: Path) -> None:
    inputs = _runtime_package(tmp_path)
    config_record = next(
        record
        for record in inputs.asset["files"]
        if record["role"] == "habitat_ao_config"
    )
    config_path = inputs.asset_path.parent / config_record["path"]
    config = copy.deepcopy(load_json_for_test(config_path))
    config["render_asset"] = "other.glb"
    write_json(config_path, config)
    config_record["byte_size"] = config_path.stat().st_size
    config_record["sha256"] = sha256_file(config_path)

    with pytest.raises(HabitatCaptureError, match="render_asset differs"):
        load_runtime_asset_bundle(inputs)


@pytest.mark.parametrize(
    "user_defined",
    [None, {}, {"avengine_native_gltf_skin_frame": False}],
)
def test_runtime_bundle_requires_native_gltf_skin_frame_opt_in(
    tmp_path: Path, user_defined: object
) -> None:
    inputs = _runtime_package(tmp_path)
    config_record = next(
        record
        for record in inputs.asset["files"]
        if record["role"] == "habitat_ao_config"
    )
    config_path = inputs.asset_path.parent / config_record["path"]
    config = copy.deepcopy(load_json_for_test(config_path))
    if user_defined is None:
        config.pop("user_defined")
    else:
        config["user_defined"] = user_defined
    write_json(config_path, config)
    config_record["byte_size"] = config_path.stat().st_size
    config_record["sha256"] = sha256_file(config_path)

    with pytest.raises(HabitatCaptureError, match="explicitly opt in"):
        load_runtime_asset_bundle(inputs)


def load_json_for_test(path: Path) -> dict[str, Any]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_compile_frame_applications_resolves_endpoint_exclusive_loop(
    tmp_path: Path,
) -> None:
    inputs, bundle = _compiled_inputs(tmp_path)

    frames = compile_frame_applications(inputs, bundle)

    assert len(frames) == 75
    assert frames[17].action_id == "walk"
    assert frames[17].action_time_ticks == 6_400
    assert frames[17].effective_action_tick == 0
    assert frames[17].action_sample_index == 0
    assert frames[17].declared_pose_hash == frames[17].recomputed_pose_hash
    assert (
        frames[17].declared_applied_state_hash
        == frames[17].recomputed_applied_state_hash
    )
    assert np.allclose(
        np.asarray(frames[17].world_from_skin_root),
        np.asarray(frames[17].world_from_actor),
    )


def test_compile_frame_applications_rejects_request_npz_pose_divergence(
    tmp_path: Path,
) -> None:
    inputs, bundle = _compiled_inputs(tmp_path)
    inputs.request["states"][3]["joint_states"][0]["rotation_xyzw"] = list(
        IDENTITY_QUATERNION
    )

    with pytest.raises(HabitatCaptureError, match="declared pose differs"):
        compile_frame_applications(inputs, bundle)


def test_fixed_frame_applies_readbacks_and_renders_modalities_once() -> None:
    simulator = _FakeSimulator()

    captured = _capture(simulator)

    assert simulator.render_calls == 1
    assert captured.record["runtime_application"]["world_time_advance_seconds"] == 0.0
    assert (
        captured.record["runtime_application"]["before"]["sha256"]
        == (captured.record["runtime_application"]["after"]["sha256"])
    )
    assert list(captured.record["modalities"]) == ["rgb", "depth", "semantic"]
    assert captured.record["modalities"]["rgb"]["shape"] == [2, 3, 4]
    assert captured.record["animal_semantic_visibility"] == {
        "semantic_id": 5,
        "pixel_count": 1,
        "visible": True,
    }
    assert all(
        len(captured.record["modalities"][modality]["payload_sha256"]) == 64
        for modality in ("rgb", "depth", "semantic")
    )


def test_fixed_frame_rejects_any_world_time_advance() -> None:
    with pytest.raises(HabitatCaptureError, match="advanced Habitat world time"):
        _capture(_FakeSimulator(advance_time=True))


def test_saved_array_stacks_round_trip_and_detect_tampering(tmp_path: Path) -> None:
    captures = [_capture(), _capture()]

    artifacts = save_capture_arrays(captures, tmp_path)
    evidence = {"array_artifacts": artifacts}

    assert verify_saved_capture_arrays(evidence, tmp_path) == []
    assert artifacts["rgb"]["shape"] == [2, 2, 3, 4]
    rgb_path = tmp_path / artifacts["rgb"]["artifact"]["path"]
    rgb_path.write_bytes(b"x" * rgb_path.stat().st_size)
    assert verify_saved_capture_arrays(evidence, tmp_path) == [
        "rgb artifact bytes changed"
    ]


def test_research_media_encodes_only_same_view_review_videos(tmp_path: Path) -> None:
    captures = [_capture(), _capture()]
    encoded: list[tuple[str, int]] = []

    def fake_encoder(frame_dir: Path, destination: Path) -> None:
        frames = sorted(frame_dir.glob("frame_*.png"))
        encoded.append((frame_dir.name, len(frames)))
        destination.write_bytes((frame_dir.name + " review-only").encode())

    media = write_research_review_media(captures, tmp_path, encode_video=fake_encoder)

    assert media["review_only"] is True
    assert media["qualification_claim"] is False
    assert media["formal_view_ids"] == []
    assert media["view_ids"] == ["view0"]
    assert encoded == [("rgb", 2), ("depth", 2), ("semantic", 2)]
    assert set(media["videos"]) == {"rgb", "depth", "semantic"}
    assert all(
        (tmp_path / media["videos"][modality]["artifact"]["path"]).is_file()
        for modality in ("rgb", "depth", "semantic")
    )
    assert not (tmp_path / "review_media/frames").exists()
