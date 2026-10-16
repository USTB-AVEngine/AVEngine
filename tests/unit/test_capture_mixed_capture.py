from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.rooms.habitat_capture import resolve_installed_runtime_prefix
from avengine.assets.glb import extract_actions, extract_skins, parse_glb
from avengine.assets.glb_write import build_glb
from avengine.capture.human_runtime import (
    ARMATURE_NODE_NAME,
    HEAD_LINK_NAME,
    MOUTH_LINK_NAME,
    SYNTHETIC_ROOT_NAME,
    promote_rocketbox_skin_ancestors,
)
from avengine.capture.mixed_capture import (
    MIXED_CAPTURE_INSTALLED_SCHEMA_V2,
    MixedCaptureError,
    _actor_heading_evidence,
    _beagle_anatomical_forward_binding,
    _continuous_beagle_walk_states,
    _human_anatomical_forward_binding,
    _select_mixed_capture_runtime,
    _validate_used_action_render_evidence,
    capture_human_beagle_paths,
    capture_legacy_route,
    locomotion_schedule_from_root_trajectory,
    trajectory_world_matrices,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _append(binary: bytearray, values: np.ndarray) -> tuple[int, int]:
    while len(binary) % 4:
        binary.append(0)
    offset = len(binary)
    payload = np.ascontiguousarray(values).tobytes(order="C")
    binary.extend(payload)
    return offset, len(payload)


def _rocketbox_shape_fixture() -> bytes:
    binary = bytearray()
    inverse_offset, inverse_length = _append(
        binary,
        np.tile(np.eye(4, dtype=np.dtype("<f4")).T.reshape(1, 16), (80, 1)),
    )
    times_offset, times_length = _append(
        binary, np.asarray([0.0, 1.0], dtype=np.dtype("<f4"))
    )
    rotations_offset, rotations_length = _append(
        binary,
        np.asarray([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=np.dtype("<f4")),
    )

    names = ["Bip01 Pelvis", HEAD_LINK_NAME, MOUTH_LINK_NAME]
    names.extend(f"Bip01 Fixture {index:02d}" for index in range(77))
    nodes: list[dict[str, object]] = []
    for index, name in enumerate(names):
        node: dict[str, object] = {"name": name}
        if index + 1 < len(names):
            node["children"] = [index + 1]
        nodes.append(node)
    nodes.append({"name": ARMATURE_NODE_NAME, "children": [0]})
    document: dict[str, object] = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [80]}],
        "nodes": nodes,
        "skins": [
            {
                "name": ARMATURE_NODE_NAME,
                "joints": list(range(80)),
                "inverseBindMatrices": 0,
            }
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": inverse_offset, "byteLength": inverse_length},
            {"buffer": 0, "byteOffset": times_offset, "byteLength": times_length},
            {
                "buffer": 0,
                "byteOffset": rotations_offset,
                "byteLength": rotations_length,
            },
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 80, "type": "MAT4"},
            {"bufferView": 1, "componentType": 5126, "count": 2, "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5126, "count": 2, "type": "VEC4"},
        ],
        "animations": [
            {
                "name": name,
                "samplers": [
                    {"input": 1, "output": 2, "interpolation": "LINEAR"}
                ],
                "channels": [
                    {"sampler": 0, "target": {"node": 80, "path": "rotation"}}
                ],
            }
            for name in ("Walking", "Standing_Idle")
        ],
    }
    return build_glb(document, binary)


def test_promote_rocketbox_appends_zero_weight_ancestors_without_reindexing() -> None:
    source = parse_glb(_rocketbox_shape_fixture())
    before = extract_skins(source)[0]
    promoted = parse_glb(promote_rocketbox_skin_ancestors(source))
    after = extract_skins(promoted)[0]

    assert tuple(joint.node_index for joint in after.joints[:80]) == tuple(
        joint.node_index for joint in before.joints
    )
    assert [joint.name for joint in after.joints[-2:]] == [
        ARMATURE_NODE_NAME,
        SYNTHETIC_ROOT_NAME,
    ]
    assert after.skeleton_node_index == after.joints[-1].node_index
    assert len(after.inverse_bind_matrices or ()) == 82
    assert [action.name for action in extract_actions(promoted)] == ["Walking", "Idle"]
    assert all(
        any(
            channel.target_node_name == SYNTHETIC_ROOT_NAME
            and channel.target_path == "rotation"
            for channel in action.channels
        )
        for action in extract_actions(promoted)
    )


@pytest.mark.parametrize(
    "local_forward_axis",
    [
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
    ],
)
def test_trajectory_world_matrices_preserve_points_and_align_asset_forward(
    local_forward_axis: tuple[float, float, float],
) -> None:
    points = np.column_stack(
        (
            np.linspace(-1.0, 2.0, 270),
            np.full(270, 0.271),
            np.linspace(3.0, -4.0, 270),
        )
    )
    matrices = trajectory_world_matrices(
        points, local_forward_axis=local_forward_axis
    )
    expected_forward = points[-1] - points[0]
    expected_forward[1] = 0.0
    expected_forward /= np.linalg.norm(expected_forward)

    assert matrices.shape == (270, 4, 4)
    assert np.array_equal(matrices[:, :3, 3], points)
    assert np.allclose(
        matrices[:, :3, :3] @ np.asarray(local_forward_axis), expected_forward
    )
    assert np.allclose(np.linalg.det(matrices[:, :3, :3]), 1.0)


def test_trajectory_world_matrices_reject_vertical_or_implicit_forward_axis() -> None:
    points = np.column_stack(
        (
            np.linspace(0.0, 1.0, 270),
            np.zeros(270),
            np.zeros(270),
        )
    )
    with pytest.raises(TypeError, match="local_forward_axis"):
        trajectory_world_matrices(points)  # type: ignore[call-arg]
    with pytest.raises(MixedCaptureError, match="must be horizontal"):
        trajectory_world_matrices(
            points, local_forward_axis=(0.0, 1.0, 0.0)
        )


def test_stationary_trajectory_uses_authored_fallback_without_changing_moving_route() -> None:
    stationary = np.repeat(
        np.asarray([[1.25, 0.271, -0.75]], dtype=np.float64), 270, axis=0
    )
    fallback_xz = np.asarray((0.6, -0.8), dtype=np.float64)
    matrices = trajectory_world_matrices(
        stationary,
        local_forward_axis=(0.0, 0.0, 1.0),
        fallback_forward_xz=fallback_xz,
    )
    world_forward = matrices[:, :3, :3] @ np.asarray((0.0, 0.0, 1.0))

    assert np.array_equal(matrices[:, :3, 3], stationary)
    assert np.allclose(world_forward, (0.6, 0.0, -0.8))
    evidence = _actor_heading_evidence(
        actor_id="stationary0",
        points_m=stationary,
        actor_world_matrices=matrices,
        local_forward_axis=(0.0, 0.0, 1.0),
        binding_source={"kind": "unit_test"},
        fallback_forward_xz=fallback_xz,
    )
    assert evidence["gate"]["all_frames_passed"] is True
    assert evidence["heading_authority"] == "authored_first_anchor_yaw_fallback"
    assert evidence["stationary_fallback_forward_xz"] == [0.6, -0.8]

    with pytest.raises(MixedCaptureError, match="authored fallback_forward_xz"):
        trajectory_world_matrices(
            stationary, local_forward_axis=(0.0, 0.0, 1.0)
        )

    moving = stationary.copy()
    moving[:, 0] += np.linspace(0.0, 1.0, 270)
    without_fallback = trajectory_world_matrices(
        moving, local_forward_axis=(0.0, 0.0, 1.0)
    )
    with_fallback = trajectory_world_matrices(
        moving,
        local_forward_axis=(0.0, 0.0, 1.0),
        fallback_forward_xz=(-1.0, 0.0),
    )
    assert np.array_equal(with_fallback, without_fallback)


def test_locomotion_schedule_uses_root_speed_and_resets_each_action_clock() -> None:
    points = np.zeros((270, 3), dtype=np.float64)
    points[76:195, 0] = np.linspace(0.01, 1.19, 119)
    points[195:, 0] = points[194, 0]

    schedule = locomotion_schedule_from_root_trajectory(
        points,
        action_sample_counts={"idle": 175, "walk": 16},
    )

    assert [state.action_id for state in schedule[:75]] == ["idle"] * 75
    assert [state.action_id for state in schedule[75:195]] == ["walk"] * 120
    assert [state.action_id for state in schedule[195:]] == ["idle"] * 75
    assert schedule[0].action_frame_index == schedule[0].action_sample_index == 0
    assert schedule[74].action_frame_index == 74
    assert schedule[75].state_transition is True
    assert schedule[75].action_frame_index == schedule[75].action_sample_index == 0
    assert schedule[76].action_phase == 1 / 16
    assert schedule[194].action_sample_index == 7
    assert schedule[195].state_transition is True
    assert schedule[195].action_frame_index == 0
    assert schedule[195].action_phase == 0.0


def test_locomotion_schedule_hysteresis_ignores_subthreshold_root_jitter() -> None:
    points = np.zeros((270, 3), dtype=np.float64)
    # 0.02 m/s lies between the 0.015 idle-enter and 0.03 walk-enter gates.
    points[:, 0] = np.arange(270) * (0.02 / 15.0)
    schedule = locomotion_schedule_from_root_trajectory(
        points,
        action_sample_counts={"idle": 25, "walk": 25},
    )
    assert {state.action_id for state in schedule} == {"idle"}

    moving = points.copy()
    moving[1:, 0] += np.arange(1, 270) * (0.04 / 15.0)
    moving_schedule = locomotion_schedule_from_root_trajectory(
        moving,
        action_sample_counts={"idle": 25, "walk": 25},
    )
    assert {state.action_id for state in moving_schedule} == {"walk"}


def test_path_helpers_support_one_five_second_episode() -> None:
    points = np.zeros((75, 3), dtype=np.float64)
    points[:, 0] = np.linspace(0.0, 1.0, 75)
    schedule = locomotion_schedule_from_root_trajectory(
        points,
        action_sample_counts={"idle": 25, "walk": 16},
    )
    matrices = trajectory_world_matrices(
        points,
        local_forward_axis=(0.0, 0.0, 1.0),
    )
    assert len(schedule) == 75
    assert {state.action_id for state in schedule} == {"walk"}
    assert matrices.shape == (75, 4, 4)


def test_render_evidence_requires_only_actions_selected_by_the_route() -> None:
    stationary = np.zeros((270, 3), dtype=np.float64)
    idle_schedule = locomotion_schedule_from_root_trajectory(
        stationary,
        action_sample_counts={"idle": 25, "walk": 25},
    )
    _validate_used_action_render_evidence(
        actor_id="dog0",
        schedule=idle_schedule,
        pose_hashes_by_action={"idle": {"idle-hash"}, "walk": set()},
    )
    with pytest.raises(MixedCaptureError, match="used actions.*idle"):
        _validate_used_action_render_evidence(
            actor_id="dog0",
            schedule=idle_schedule,
            pose_hashes_by_action={"idle": set(), "walk": {"walk-hash"}},
        )

    moving = stationary.copy()
    moving[:, 0] = np.linspace(0.0, 1.0, 270)
    walk_schedule = locomotion_schedule_from_root_trajectory(
        moving,
        action_sample_counts={"idle": 25, "walk": 25},
    )
    _validate_used_action_render_evidence(
        actor_id="dog0",
        schedule=walk_schedule,
        pose_hashes_by_action={"idle": set(), "walk": {"walk-hash"}},
    )


def test_actor_heading_evidence_retains_every_frame_and_fails_closed() -> None:
    points = np.column_stack(
        (
            np.linspace(0.0, 2.0, 270),
            np.full(270, 0.271),
            np.zeros(270),
        )
    )
    matrices = trajectory_world_matrices(
        points, local_forward_axis=(0.0, 0.0, 1.0)
    )
    evidence = _actor_heading_evidence(
        actor_id="human0",
        points_m=points,
        actor_world_matrices=matrices,
        local_forward_axis=(0.0, 0.0, 1.0),
        binding_source={"kind": "unit_test", "declared_axis": "+Z"},
    )

    assert evidence["status"] == "pass"
    assert evidence["gate"]["all_frames_passed"] is True
    assert len(evidence["frames"]) == 270
    assert all(frame["passed"] for frame in evidence["frames"])
    assert evidence["frames"][137]["path_tangent_world"] == [1.0, 0.0, 0.0]

    wrong = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 270, axis=0)
    wrong[:, :3, 3] = points
    with pytest.raises(MixedCaptureError, match="heading gate failed"):
        _actor_heading_evidence(
            actor_id="human0",
            points_m=points,
            actor_world_matrices=wrong,
            local_forward_axis=(0.0, 0.0, 1.0),
            binding_source={"kind": "unit_test", "declared_axis": "+Z"},
        )


def test_human_forward_binding_reads_hash_closed_runtime_metadata(
    tmp_path: Path,
) -> None:
    manifest = {
        "schema": "avengine_m5_1_rocketbox_human_runtime_v1",
        "status": "pass",
        "anatomical_frame": {
            "actor_up_axis": "+Y",
            "actor_forward_axis": "+Z",
            "source": "fixture_rest_pose_head_to_mjaw_axis",
        },
    }
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    path = tmp_path / "human_runtime_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    axis, source = _human_anatomical_forward_binding(path)

    assert axis == (0.0, 0.0, 1.0)
    assert source["declared_axis"] == "+Z"
    assert source["sha256"] == sha256_file(path)


def test_beagle_forward_binding_reads_manifest_bound_animation_qa(
    tmp_path: Path,
) -> None:
    qa = {
        "schema": "avengine_m2_animation_qa_v1",
        "status": "pass",
        "semantic_terminal_motion": {
            "actor_up_axis": "+Y",
            "source_facing_axis_in_actor_frame": "+X",
        },
    }
    qa_path = tmp_path / "animation.json"
    qa_path.write_text(json.dumps(qa), encoding="utf-8")
    asset = {
        "files": [
            {
                "role": "animation_qa",
                "path": qa_path.name,
                "byte_size": qa_path.stat().st_size,
                "sha256": sha256_file(qa_path),
            }
        ]
    }

    axis, source = _beagle_anatomical_forward_binding(
        asset=asset, asset_manifest_path=tmp_path / "asset_manifest.json"
    )

    assert axis == (1.0, 0.0, 0.0)
    assert source["declared_axis"] == "+X"
    assert source["sha256"] == sha256_file(qa_path)

    qa_path.write_text(json.dumps({**qa, "status": "fail"}), encoding="utf-8")
    with pytest.raises(MixedCaptureError, match="bytes differ"):
        _beagle_anatomical_forward_binding(
            asset=asset, asset_manifest_path=tmp_path / "asset_manifest.json"
        )


@dataclass(frozen=True)
class _State:
    action_id: str


def test_beagle_selection_accepts_only_one_continuous_45_state_walk_block() -> None:
    states = tuple([_State("idle")] * 15 + [_State("walk")] * 45 + [_State("idle")] * 15)
    indices, selected = _continuous_beagle_walk_states(states)
    assert indices == tuple(range(15, 60))
    assert len(selected) == 45
    assert all(state.action_id == "walk" for state in selected)

    split = tuple([_State("walk")] * 22 + [_State("idle")] + [_State("walk")] * 23)
    with pytest.raises(MixedCaptureError, match="continuous 45-state walk block"):
        _continuous_beagle_walk_states(split)


def test_legacy_wrapper_consumes_manifest_habitat_paths_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_path = REPOSITORY_ROOT / "examples/capture/legacy_apartment/route_manifest.json"
    route = json.loads(route_path.read_text(encoding="utf-8"))
    sentinel = object()
    retained: dict[str, object] = {}

    def fake_capture(**kwargs: object) -> object:
        retained.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "avengine.capture.mixed_capture.capture_human_beagle_paths", fake_capture
    )
    result = capture_legacy_route(
        route_manifest_path=route_path,
        room_manifest_path="room.json",
        m1_request_path="request.json",
        human_runtime_glb_path="human.glb",
        beagle_animal_manifest_path="beagle.json",
        beagle_m2_request_path="beagle_request.json",
        output_dir="output",
    )

    assert result is sentinel
    assert retained["human_root_path_m"] == route["routes"]["human_path"][
        "habitat_trajectory_m"
    ]
    assert retained["beagle_root_path_m"] == route["routes"]["dog_path"][
        "habitat_trajectory_m"
    ]
    assert retained["legacy_runtime_root"] is None
    assert "runtime_root" not in retained
    assert retained["require_legacy_camera"] is True
    assert retained["route_provenance"]["path_consumption"] == (
        "verbatim_manifest_routes_habitat_trajectory_m"
    )


def _migrated_mp3d_room() -> dict[str, object]:
    return {
        "scene": {
            "scene_id": "${AVENGINE_MP3D_ROOT}/scene_datasets/mp3d_example/a.glb",
            "dataset_config_path": "${AVENGINE_MP3D_ROOT}/scene_datasets/mp3d_example/mp3d.scene_dataset_config.json",
            "navmesh_path": "${AVENGINE_MP3D_ROOT}/scene_datasets/mp3d_example/a.navmesh",
        },
        "assets": [
            {"path": "${AVENGINE_MP3D_ROOT}/scene_datasets/mp3d_example/a.glb"}
        ],
    }


def _legacy_runtime_room() -> dict[str, object]:
    return {
        "scene": {
            "scene_id": "${AVENGINE_HABITAT_RUNTIME_ROOT}/data/scene.glb",
            "dataset_config_path": "${AVENGINE_HABITAT_RUNTIME_ROOT}/data/config.json",
            "navmesh_path": "${AVENGINE_HABITAT_RUNTIME_ROOT}/data/scene.navmesh",
        },
        "assets": [
            {"path": "${AVENGINE_HABITAT_RUNTIME_ROOT}/data/scene.glb"}
        ],
    }


def _retained_replicacad_room() -> dict[str, object]:
    return {
        "scene": {
            "scene_id": "apt_0",
            "dataset_config_path": "${AVENGINE_REPLICACAD_ROOT}/replicaCAD.scene_dataset_config.json",
            "navmesh_path": "${AVENGINE_REPLICACAD_ROOT}/navmeshes/apt_0.navmesh",
        },
        "assets": [
            {
                "path": "${AVENGINE_REPLICACAD_ROOT}/configs/scenes/apt_0.scene_instance.json"
            }
        ],
    }


def _clear_installed_runtime_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in (
        "AVENGINE_HABITAT_RUNTIME_PREFIX",
        "AVENGINE_MP3D_ROOT",
        "AVENGINE_HABITAT_MAGNUM_PYTHON_SITE",
    ):
        monkeypatch.delenv(variable, raising=False)


def test_mixed_capture_runtime_root_is_installed_alias_for_migrated_room(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_installed_runtime_environment(monkeypatch)
    expected = object()
    calls: list[dict[str, object]] = []

    def fake_prepare(**kwargs: object) -> object:
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(
        "avengine.capture.mixed_capture.prepare_installed_habitat_runtime",
        fake_prepare,
    )

    result = _select_mixed_capture_runtime(
        room=_migrated_mp3d_room(),
        runtime_prefix=None,
        runtime_root=tmp_path / "installed-prefix",
        legacy_runtime_root=None,
        mp3d_root=None,
        magnum_python_site=None,
        pbr_asset_root=tmp_path / "pbr-assets",
        installed_runtime=None,
    )

    assert result is expected
    assert calls == [
        {
            "runtime_prefix": None,
            "runtime_root": tmp_path / "installed-prefix",
            "mp3d_root": None,
            "pbr_asset_root": tmp_path / "pbr-assets",
            "magnum_python_site": None,
        }
    ]


def test_mixed_capture_prefix_environment_preempts_legacy_room(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = object()
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("AVENGINE_HABITAT_RUNTIME_PREFIX", str(tmp_path / "prefix"))
    monkeypatch.setenv("AVENGINE_MP3D_ROOT", str(tmp_path / "mp3d"))
    monkeypatch.setenv(
        "AVENGINE_HABITAT_MAGNUM_PYTHON_SITE", str(tmp_path / "magnum")
    )

    def fake_prepare(**kwargs: object) -> object:
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(
        "avengine.capture.mixed_capture.prepare_installed_habitat_runtime",
        fake_prepare,
    )

    result = _select_mixed_capture_runtime(
        room=_legacy_runtime_room(),
        runtime_prefix=None,
        runtime_root=None,
        legacy_runtime_root=None,
        mp3d_root=None,
        magnum_python_site=None,
        pbr_asset_root=tmp_path / "pbr-assets",
        installed_runtime=None,
    )

    assert result is expected
    assert calls == [
        {
            "runtime_prefix": None,
            "runtime_root": None,
            "mp3d_root": None,
            "pbr_asset_root": tmp_path / "pbr-assets",
            "magnum_python_site": None,
        }
    ]


def test_mixed_capture_prepared_runtime_accepts_its_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = object()
    monkeypatch.setenv("AVENGINE_HABITAT_RUNTIME_PREFIX", str(tmp_path / "prefix"))
    monkeypatch.setenv("AVENGINE_MP3D_ROOT", str(tmp_path / "mp3d"))
    monkeypatch.setenv(
        "AVENGINE_HABITAT_MAGNUM_PYTHON_SITE", str(tmp_path / "magnum")
    )

    def unexpected_prepare(**kwargs: object) -> object:
        raise AssertionError(f"prepared runtime was not reused: {kwargs}")

    monkeypatch.setattr(
        "avengine.capture.mixed_capture.prepare_installed_habitat_runtime",
        unexpected_prepare,
    )

    assert (
        _select_mixed_capture_runtime(
            room=_migrated_mp3d_room(),
            runtime_prefix=None,
            runtime_root=None,
            legacy_runtime_root=None,
            mp3d_root=None,
            magnum_python_site=None,
            installed_runtime=expected,  # type: ignore[arg-type]
        )
        is expected
    )


def test_mixed_capture_runtime_root_alias_rejects_git_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_installed_runtime_environment(monkeypatch)
    checkout = tmp_path / "old-habitat"
    checkout.mkdir()
    (checkout / ".git").mkdir()

    def resolve_prefix_alias(**kwargs: object) -> object:
        return resolve_installed_runtime_prefix(
            kwargs["runtime_prefix"], runtime_root=kwargs["runtime_root"]
        )

    monkeypatch.setattr(
        "avengine.capture.mixed_capture.prepare_installed_habitat_runtime",
        resolve_prefix_alias,
    )

    with pytest.raises(ValueError, match="must not be inside a Git checkout"):
        _select_mixed_capture_runtime(
            room=_migrated_mp3d_room(),
            runtime_prefix=None,
            runtime_root=checkout,
            legacy_runtime_root=None,
            mp3d_root=None,
            magnum_python_site=None,
            pbr_asset_root=tmp_path / "pbr-assets",
            installed_runtime=None,
        )


def test_mixed_capture_legacy_room_keeps_checkout_branch_without_new_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_installed_runtime_environment(monkeypatch)

    def unexpected_prepare(**kwargs: object) -> object:
        raise AssertionError(f"legacy caller unexpectedly selected installed runtime: {kwargs}")

    monkeypatch.setattr(
        "avengine.capture.mixed_capture.prepare_installed_habitat_runtime",
        unexpected_prepare,
    )

    assert (
        _select_mixed_capture_runtime(
            room=_legacy_runtime_room(),
            runtime_prefix=None,
            runtime_root=tmp_path / "legacy-runtime",
            legacy_runtime_root=None,
            mp3d_root=None,
            magnum_python_site=None,
            installed_runtime=None,
        )
        is None
    )


def test_mixed_capture_replicacad_room_keeps_legacy_branch_without_new_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_installed_runtime_environment(monkeypatch)

    def unexpected_prepare(**kwargs: object) -> object:
        raise AssertionError(
            f"ReplicaCAD caller unexpectedly selected installed runtime: {kwargs}"
        )

    monkeypatch.setattr(
        "avengine.capture.mixed_capture.prepare_installed_habitat_runtime",
        unexpected_prepare,
    )

    assert (
        _select_mixed_capture_runtime(
            room=_retained_replicacad_room(),
            runtime_prefix=None,
            runtime_root=tmp_path / "legacy-runtime",
            legacy_runtime_root=None,
            mp3d_root=None,
            magnum_python_site=None,
            installed_runtime=None,
        )
        is None
    )


def _runtime_selection_capture_arguments(tmp_path: Path) -> dict[str, object]:
    paths = np.zeros((75, 3), dtype=np.float64)
    return {
        "room_manifest_path": tmp_path / "room.json",
        "m1_request_path": tmp_path / "request.json",
        "human_runtime_glb_path": tmp_path / "human.glb",
        "beagle_animal_manifest_path": tmp_path / "beagle.json",
        "beagle_m2_request_path": tmp_path / "beagle_request.json",
        "human_root_path_m": paths,
        "beagle_root_path_m": paths,
        "output_dir": tmp_path / "capture",
    }


def test_mixed_capture_direct_runtime_root_alias_uses_installed_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_installed_runtime_environment(monkeypatch)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "avengine.capture.mixed_capture.prepare_rocketbox_habitat_runtime",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "avengine.capture.mixed_capture.load_m1_inputs",
        lambda *args, **kwargs: SimpleNamespace(room=_migrated_mp3d_room()),
    )

    def selected_installed_runtime(**kwargs: object) -> object:
        calls.append(kwargs)
        raise ValueError("installed runtime selected")

    monkeypatch.setattr(
        "avengine.capture.mixed_capture.prepare_installed_habitat_runtime",
        selected_installed_runtime,
    )

    with pytest.raises(MixedCaptureError, match="installed runtime selected"):
        capture_human_beagle_paths(
            **_runtime_selection_capture_arguments(tmp_path),
            runtime_root=tmp_path / "installed-prefix",
            pbr_asset_root=tmp_path / "pbr-assets",
        )

    assert calls == [
        {
            "runtime_prefix": None,
            "runtime_root": tmp_path / "installed-prefix",
            "mp3d_root": None,
            "pbr_asset_root": tmp_path / "pbr-assets",
            "magnum_python_site": None,
        }
    ]


def test_mixed_capture_direct_prefix_environment_uses_installed_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AVENGINE_HABITAT_RUNTIME_PREFIX", str(tmp_path / "prefix"))
    monkeypatch.setenv("AVENGINE_MP3D_ROOT", str(tmp_path / "mp3d"))
    monkeypatch.setenv(
        "AVENGINE_HABITAT_MAGNUM_PYTHON_SITE", str(tmp_path / "magnum")
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "avengine.capture.mixed_capture.prepare_rocketbox_habitat_runtime",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "avengine.capture.mixed_capture.load_m1_inputs",
        lambda *args, **kwargs: SimpleNamespace(room=_legacy_runtime_room()),
    )

    def selected_installed_runtime(**kwargs: object) -> object:
        calls.append(kwargs)
        raise ValueError("installed environment selected")

    monkeypatch.setattr(
        "avengine.capture.mixed_capture.prepare_installed_habitat_runtime",
        selected_installed_runtime,
    )

    with pytest.raises(MixedCaptureError, match="installed environment selected"):
        capture_human_beagle_paths(
            **_runtime_selection_capture_arguments(tmp_path),
            pbr_asset_root=tmp_path / "pbr-assets",
        )

    assert calls == [
        {
            "runtime_prefix": None,
            "runtime_root": None,
            "mp3d_root": None,
            "pbr_asset_root": tmp_path / "pbr-assets",
            "magnum_python_site": None,
        }
    ]


def test_mixed_capture_installed_missing_pbr_root_fails_without_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AVENGINE_HABITAT_RUNTIME_PREFIX", str(tmp_path / "prefix"))
    monkeypatch.setenv("AVENGINE_MP3D_ROOT", str(tmp_path / "mp3d"))
    monkeypatch.setenv(
        "AVENGINE_HABITAT_MAGNUM_PYTHON_SITE", str(tmp_path / "magnum")
    )

    def unexpected_before_output(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(
            "missing PBR root must fail before runtime, assets, or output"
        )

    monkeypatch.setattr(
        "avengine.capture.mixed_capture.load_m1_inputs",
        unexpected_before_output,
    )
    monkeypatch.setattr(
        "avengine.capture.mixed_capture.prepare_installed_habitat_runtime",
        unexpected_before_output,
    )
    monkeypatch.setattr(
        "avengine.capture.mixed_capture.prepare_rocketbox_habitat_runtime",
        unexpected_before_output,
    )
    output = tmp_path / "capture"
    with pytest.raises(MixedCaptureError, match="explicit pbr_asset_root"):
        capture_human_beagle_paths(
            **{
                **_runtime_selection_capture_arguments(tmp_path),
                "output_dir": output,
            }
        )
    assert not output.exists()


def test_installed_mixed_capture_preserves_visual_sensors_without_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    room_inputs = SimpleNamespace(room=_migrated_mp3d_room())
    human_action = SimpleNamespace(sample_count=16)
    human_package = SimpleNamespace(
        package_manifest={},
        actions=SimpleNamespace(action=lambda _action_id: human_action),
    )
    beagle_action = SimpleNamespace(sample_count=25)
    beagle_bundle = SimpleNamespace(
        action_roles_by_id={"idle": "locomotion", "walk": "locomotion"},
        action_sets_by_role={
            "locomotion": SimpleNamespace(
                action=lambda _action_id: beagle_action
            )
        },
    )
    m2_inputs = SimpleNamespace(asset={}, asset_path=tmp_path / "beagle.json")
    visual_modalities = {
        "rgb": "rig_rgb",
        "depth": "rig_depth",
        "semantic": "rig_semantic",
    }
    configuration_calls: list[dict[str, object]] = []

    def fake_make_configuration(
        _inputs: object,
        runtime_root: object,
        output_dir: object,
        **kwargs: object,
    ) -> tuple[object, dict[str, str], str, dict[str, object]]:
        configuration_calls.append(
            {
                "runtime_root": runtime_root,
                "output_dir": output_dir,
                **kwargs,
            }
        )
        configuration = SimpleNamespace(
            sim_cfg=SimpleNamespace(enable_hbao=False, enable_physics=False),
            sensor_uuids=list(visual_modalities.values()),
        )
        return configuration, visual_modalities, "listener0", {
            "enable_physics": False
        }

    class FakeHabitat:
        @staticmethod
        def Simulator(configuration: object) -> object:
            assert configuration.sensor_uuids == [
                "rig_rgb",
                "rig_depth",
                "rig_semantic",
            ]
            raise RuntimeError("installed visual configuration reached Simulator")

    monkeypatch.setattr(
        "avengine.capture.mixed_capture.load_m1_inputs",
        lambda *_args: room_inputs,
    )
    monkeypatch.setattr(
        "avengine.capture.mixed_capture.prepare_rocketbox_habitat_runtime",
        lambda *_args: human_package,
    )
    monkeypatch.setattr(
        "avengine.capture.mixed_capture.load_m2_inputs",
        lambda *_args: m2_inputs,
    )
    monkeypatch.setattr(
        "avengine.capture.mixed_capture.load_runtime_asset_bundle",
        lambda _inputs: beagle_bundle,
    )
    monkeypatch.setattr(
        "avengine.capture.mixed_capture._human_anatomical_forward_binding",
        lambda _manifest: ((0.0, 0.0, -1.0), "fixture"),
    )
    monkeypatch.setattr(
        "avengine.capture.mixed_capture._beagle_anatomical_forward_binding",
        lambda **_kwargs: ((0.0, 0.0, -1.0), "fixture"),
    )
    monkeypatch.setattr(
        "avengine.capture.mixed_capture.trajectory_world_matrices",
        lambda *_args, **_kwargs: np.repeat(np.eye(4)[None, ...], 75, axis=0),
    )
    monkeypatch.setattr(
        "avengine.capture.mixed_capture._actor_heading_evidence",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        "avengine.capture.mixed_capture.compile_frame_applications",
        lambda *_args: [object()] * 75,
    )
    monkeypatch.setattr(
        "avengine.capture.mixed_capture.locomotion_schedule_from_root_trajectory",
        lambda *_args, **_kwargs: [object()] * 75,
    )
    monkeypatch.setattr(
        "avengine.capture.mixed_capture._resolved_assets",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "avengine.capture.mixed_capture._make_configuration",
        fake_make_configuration,
    )
    pbr_preparation_calls: list[tuple[object, object]] = []

    def fake_prepare_pbr(
        _configuration: object,
        *,
        installed_runtime: object,
        habitat_sim: object,
    ) -> dict[str, str]:
        pbr_preparation_calls.append((installed_runtime, habitat_sim))
        return {"status": "pass"}

    monkeypatch.setattr(
        "avengine.capture.mixed_capture._prepare_m5_1_installed_pbr_ibl",
        fake_prepare_pbr,
    )
    installed_runtime = SimpleNamespace(
        mp3d_root=tmp_path / "mp3d",
        pbr_asset_root=tmp_path / "pbr-assets",
        quaternion=object(),
        habitat_sim=FakeHabitat,
        magnum=object(),
        physics_config_path=tmp_path / "default.physics_config.json",
    )

    with pytest.raises(RuntimeError, match="installed visual configuration"):
        capture_human_beagle_paths(
            **_runtime_selection_capture_arguments(tmp_path),
            installed_runtime=installed_runtime,
            research_capture_schema=MIXED_CAPTURE_INSTALLED_SCHEMA_V2,
        )

    assert configuration_calls == [
        {
            "runtime_root": None,
            "output_dir": (tmp_path / "capture/scene_scratch"),
            "mp3d_root": tmp_path / "mp3d",
            "include_audio_sensor": False,
            "physics_config_path": tmp_path / "default.physics_config.json",
        }
    ]
    assert pbr_preparation_calls == [(installed_runtime, FakeHabitat)]
