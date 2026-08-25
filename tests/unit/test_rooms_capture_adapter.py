import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import avengine.rooms.capture_adapter as capture_adapter_module
from avengine.contracts.json_io import load_json
from avengine.capture.mixed_capture import (
    locomotion_schedule_from_root_trajectory,
    trajectory_world_matrices,
)
from avengine.capture.orientation import habitat_basis_from_yaw_degrees
from avengine.rooms.capture_adapter import (
    CaptureAdapterError,
    CaptureData,
    HUMAN_BEAGLE_CAPTURE_ADAPTER,
)


ROOT = Path(__file__).resolve().parents[2]


def test_adapter_bindings_cover_fixed_suite_endpoints_and_registry() -> None:
    endpoint_registry = load_json(
        ROOT / "examples/m6/registries/source_endpoints_v1.json"
    )
    suite = load_json(ROOT / "examples/m6x/fixed_apartment/scenario_suite.json")
    scenario_endpoints = {
        item["source_endpoint_id"]
        for scenario in suite["scenarios"]
        for item in scenario["source_bindings"]
    }

    assert (
        HUMAN_BEAGLE_CAPTURE_ADAPTER.validate_registry_bindings(endpoint_registry) == []
    )
    assert scenario_endpoints == set(HUMAN_BEAGLE_CAPTURE_ADAPTER.sources_by_id)
    assert (
        HUMAN_BEAGLE_CAPTURE_ADAPTER.source_binding(
            "m6x_human0_mouth"
        ).capture_anchor_id
        == "human0.mouth_emitter"
    )
    assert (
        HUMAN_BEAGLE_CAPTURE_ADAPTER.source_binding("m6x_dog0_muzzle").capture_anchor_id
        == "dog0.mouth_emitter"
    )


def test_registry_actor_mismatch_fails_at_adapter_boundary() -> None:
    registry = load_json(ROOT / "examples/m6/registries/source_endpoints_v1.json")
    endpoint = next(
        item
        for item in registry["source_endpoints"]
        if item["source_endpoint_id"] == "m6x_dog0_muzzle"
    )
    endpoint["binding"]["entity_instance_id"] = "some_other_animal"

    errors = HUMAN_BEAGLE_CAPTURE_ADAPTER.validate_registry_bindings(registry)

    assert any("m6x_dog0_muzzle" in item and "actor must be" in item for item in errors)


def test_timeline_state_uses_actor_binding_instead_of_actor_id_branch() -> None:
    human = HUMAN_BEAGLE_CAPTURE_ADAPTER.actor_binding("human0")
    swapped_human = replace(
        human,
        capture_matrix_index=1,
        pose_hash_record_path=("provider", "pose"),
        action_id="adapter_action",
        action_phase_period_frames=10,
        action_id_record_path=("provider", "action_id"),
        action_time_ticks_record_path=("provider", "action_time_ticks"),
        action_phase_record_path=("provider", "action_phase"),
    )
    adapter = replace(
        HUMAN_BEAGLE_CAPTURE_ADAPTER,
        actor_bindings=tuple(
            swapped_human if item.actor_id == "human0" else item
            for item in HUMAN_BEAGLE_CAPTURE_ADAPTER.actor_bindings
        ),
    )
    matrices = np.repeat(np.eye(4)[None, None, :, :], 270 * 2, axis=0).reshape(
        270, 2, 4, 4
    )
    matrices[17, 1, :3, 3] = (1.0, 2.0, 3.0)
    capture = CaptureData(
        root=ROOT,
        rgb=np.empty(0),
        semantic=np.empty(0),
        actor_world_matrices=matrices,
        anchor_positions_m=np.empty(0),
        records=tuple(
            {
                "provider": {
                    "pose": "a" * 64,
                    "action_id": "idle" if index < 20 else "walk",
                    "action_time_ticks": (index if index < 20 else index - 20) * 3_200,
                    "action_phase": index / 20 if index < 20 else (index - 20) / 250,
                }
            }
            for index in range(270)
        ),
        evidence={},
    )

    state = adapter.timeline_actor_state(
        capture,
        actor_id="human0",
        master_frame=17,
        local_frame=0,
        source_endpoint_id=None,
        trajectories={},
    )

    assert state.translation_m == (1.0, 2.0, 3.0)
    assert state.action_id == "idle"
    assert state.action_time_ticks == 17 * 3_200
    assert state.action_phase == 17 / 20
    assert state.pose_hash == "a" * 64


def test_adapter_materializes_first_anchor_yaw_as_fallback_forward() -> None:
    trajectories = load_json(
        ROOT / "examples/m6x/fixed_apartment/trajectory_templates.json"
    )
    anchors = load_json(ROOT / "examples/m6x/fixed_apartment/anchor_library.json")

    forwards = HUMAN_BEAGLE_CAPTURE_ADAPTER.materialize_actor_fallback_forwards_xz(
        trajectories, anchors
    )

    assert set(forwards) == {"dog0", "human0"}
    assert np.allclose(
        forwards["human0"], habitat_basis_from_yaw_degrees(180.0).forward_xz
    )
    assert np.allclose(
        forwards["dog0"],
        habitat_basis_from_yaw_degrees(26.565051).forward_xz,
    )
    roots = HUMAN_BEAGLE_CAPTURE_ADAPTER.materialize_actor_root_paths(
        trajectories, anchors
    )
    for actor_id in ("human0", "dog0"):
        previous_master = trajectory_world_matrices(
            roots[actor_id], local_forward_axis=(0.0, 0.0, 1.0)
        )
        fallback_enabled = trajectory_world_matrices(
            roots[actor_id],
            local_forward_axis=(0.0, 0.0, 1.0),
            fallback_forward_xz=forwards[actor_id],
        )
        assert np.array_equal(fallback_enabled, previous_master)


def _locomotion_closure_capture() -> CaptureData:
    trajectories = load_json(
        ROOT / "examples/m6x/fixed_apartment/trajectory_templates.json"
    )
    anchors = load_json(ROOT / "examples/m6x/fixed_apartment/anchor_library.json")
    roots = HUMAN_BEAGLE_CAPTURE_ADAPTER.materialize_actor_root_paths(
        trajectories, anchors
    )
    forwards = HUMAN_BEAGLE_CAPTURE_ADAPTER.materialize_actor_fallback_forwards_xz(
        trajectories, anchors
    )
    sample_counts = {
        "human0": {"idle": 175, "walk": 16},
        "dog0": {"idle": 25, "walk": 25},
    }
    schedules = {
        actor_id: locomotion_schedule_from_root_trajectory(
            roots[actor_id], action_sample_counts=sample_counts[actor_id]
        )
        for actor_id in ("human0", "dog0")
    }
    matrices = np.empty((270, 2, 4, 4), dtype=np.float64)
    matrices[:, 0] = trajectory_world_matrices(
        roots["human0"],
        local_forward_axis=(0.0, 0.0, 1.0),
        fallback_forward_xz=forwards["human0"],
    )
    matrices[:, 1] = trajectory_world_matrices(
        roots["dog0"],
        local_forward_axis=(1.0, 0.0, 0.0),
        fallback_forward_xz=forwards["dog0"],
    )
    records = []
    for frame_index in range(270):
        record: dict[str, object] = {}
        for actor_id, key in (("human0", "human"), ("dog0", "beagle")):
            state = schedules[actor_id][frame_index]
            record[key] = {
                "action_id": state.action_id,
                "action_time_ticks": state.action_frame_index * 3_200,
                "action_sample_index": state.action_sample_index,
                "action_phase": state.action_phase,
                "actor_root_position_m": roots[actor_id][frame_index].tolist(),
            }
        records.append(record)
    return CaptureData(
        root=ROOT,
        rgb=np.empty(0),
        semantic=np.empty(0),
        actor_world_matrices=matrices,
        anchor_positions_m=np.empty(0),
        records=tuple(records),
        evidence={
            "runtime": {
                "human_action_sample_counts": sample_counts["human0"],
                "beagle_action_sample_counts": sample_counts["dog0"],
            }
        },
    )


def test_retained_locomotion_closes_against_actor_world_matrix_roots() -> None:
    HUMAN_BEAGLE_CAPTURE_ADAPTER.validate_capture_locomotion(
        _locomotion_closure_capture()
    )


def _authoritative_heading_inputs() -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray]
]:
    trajectories = load_json(
        ROOT / "examples/m6x/fixed_apartment/trajectory_templates.json"
    )
    anchors = load_json(ROOT / "examples/m6x/fixed_apartment/anchor_library.json")
    return (
        HUMAN_BEAGLE_CAPTURE_ADAPTER.materialize_actor_root_paths(
            trajectories, anchors
        ),
        HUMAN_BEAGLE_CAPTURE_ADAPTER.materialize_actor_fallback_forwards_xz(
            trajectories, anchors
        ),
    )


def test_retained_orientation_closes_against_current_route_and_anchor_yaw() -> None:
    roots, forwards = _authoritative_heading_inputs()

    HUMAN_BEAGLE_CAPTURE_ADAPTER.validate_capture_orientation(
        _locomotion_closure_capture(),
        actor_root_paths=roots,
        actor_fallback_forwards_xz=forwards,
    )


def test_retained_orientation_rejects_nonrigid_rotation() -> None:
    roots, forwards = _authoritative_heading_inputs()
    original = _locomotion_closure_capture()
    matrices = original.actor_world_matrices.copy()
    matrices[100, 0, 0, 0] *= 1.25

    with pytest.raises(CaptureAdapterError, match="not rigid/orthonormal"):
        HUMAN_BEAGLE_CAPTURE_ADAPTER.validate_capture_orientation(
            replace(original, actor_world_matrices=matrices),
            actor_root_paths=roots,
            actor_fallback_forwards_xz=forwards,
        )


def test_retained_orientation_rejects_right_handed_wrong_yaw() -> None:
    roots, forwards = _authoritative_heading_inputs()
    original = _locomotion_closure_capture()
    matrices = original.actor_world_matrices.copy()
    yaw_180 = np.diag((-1.0, 1.0, -1.0))
    matrices[100, 0, :3, :3] = yaw_180 @ matrices[100, 0, :3, :3]

    with pytest.raises(CaptureAdapterError, match="world forward differs"):
        HUMAN_BEAGLE_CAPTURE_ADAPTER.validate_capture_orientation(
            replace(original, actor_world_matrices=matrices),
            actor_root_paths=roots,
            actor_fallback_forwards_xz=forwards,
        )


def test_retained_orientation_rejects_roll_even_when_world_forward_matches() -> None:
    roots, forwards = _authoritative_heading_inputs()
    original = _locomotion_closure_capture()
    matrices = original.actor_world_matrices.copy()
    roll_90_about_human_forward = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    matrices[100, 0, :3, :3] = (
        matrices[100, 0, :3, :3] @ roll_90_about_human_forward
    )

    with pytest.raises(CaptureAdapterError, match="canonical current route"):
        HUMAN_BEAGLE_CAPTURE_ADAPTER.validate_capture_orientation(
            replace(original, actor_world_matrices=matrices),
            actor_root_paths=roots,
            actor_fallback_forwards_xz=forwards,
        )


def test_retained_orientation_rejects_static_route_with_wrong_fallback_yaw() -> None:
    roots, forwards = _authoritative_heading_inputs()
    original = _locomotion_closure_capture()
    matrices = original.actor_world_matrices.copy()
    static_human = np.repeat(roots["human0"][:1], 270, axis=0)
    matrices[:, 0] = trajectory_world_matrices(
        static_human,
        local_forward_axis=(0.0, 0.0, 1.0),
        fallback_forward_xz=(0.0, 1.0),
    )
    current_roots = dict(roots)
    current_roots["human0"] = static_human
    current_forwards = dict(forwards)
    current_forwards["human0"] = np.asarray((1.0, 0.0), dtype=np.float64)

    with pytest.raises(CaptureAdapterError, match="first-anchor yaw fallback"):
        HUMAN_BEAGLE_CAPTURE_ADAPTER.validate_capture_orientation(
            replace(original, actor_world_matrices=matrices),
            actor_root_paths=current_roots,
            actor_fallback_forwards_xz=current_forwards,
        )


@pytest.mark.parametrize(
    ("field", "expected_error"),
    [
        ("action_id", "action_id differs"),
        ("action_time_ticks", "action_time_ticks differs"),
        ("action_sample_index", "action_sample_index differs"),
        ("action_phase", "action_phase differs"),
        ("actor_root_position_m", "root position differs"),
    ],
)
def test_retained_locomotion_rejects_each_mismatched_record_field(
    field: str, expected_error: str
) -> None:
    original = _locomotion_closure_capture()
    records = copy.deepcopy(original.records)
    human = records[100]["human"]
    assert isinstance(human, dict)
    if field == "action_id":
        human[field] = "idle"
    elif field in {"action_time_ticks", "action_sample_index"}:
        human[field] = int(human[field]) + 1
    elif field == "action_phase":
        human[field] = float(human[field]) + 0.01
    else:
        position = list(human[field])
        position[0] += 0.1
        human[field] = position
    corrupted = replace(original, records=tuple(records))

    with pytest.raises(CaptureAdapterError, match=expected_error):
        HUMAN_BEAGLE_CAPTURE_ADAPTER.validate_capture_locomotion(corrupted)


def test_capture_filters_mixed_capture_hook_arguments(monkeypatch, tmp_path: Path) -> None:
    """The provider hook must not forward mixed-capture-only keywords."""

    class _StopAfterHook(RuntimeError):
        pass

    profile = object()
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        capture_adapter_module, "load_review_visual_profile", lambda _path: profile
    )
    monkeypatch.setattr(
        capture_adapter_module, "validate_profile_capture_request", lambda *_args: None
    )
    monkeypatch.setattr(capture_adapter_module, "load_json", lambda _path: {})
    monkeypatch.setattr(
        capture_adapter_module,
        "configure_runtime_review_profile",
        lambda **_kwargs: {"status": "pass"},
    )

    def fake_apply(
        *,
        simulator,
        profile,
        exterior_proxy_glb_path,
        camera_listener_position_m,
        habitat_sim,
        mn,
    ):
        observed.update(
            {
                "simulator": simulator,
                "profile": profile,
                "exterior": exterior_proxy_glb_path,
                "camera": camera_listener_position_m,
                "habitat_sim": habitat_sim,
                "mn": mn,
            }
        )
        return {"status": "pass"}

    monkeypatch.setattr(
        capture_adapter_module, "apply_runtime_review_profile", fake_apply
    )

    def fake_capture(**kwargs):
        result = kwargs["review_scene_hook"](
            simulator="simulator",
            configuration="mixed-capture-only-value",
            camera_listener_position_m=(1.0, 2.0, 3.0),
            habitat_sim="habitat_sim",
            mn="magnum",
        )
        assert result == {"status": "pass"}
        raise _StopAfterHook

    monkeypatch.setattr(
        capture_adapter_module, "capture_human_beagle_paths", fake_capture
    )
    paths = {
        "human_runtime_glb_path": tmp_path / "human.glb",
        "animal_manifest_path": tmp_path / "animal.json",
        "animal_request_path": tmp_path / "request.json",
        "review_visual_profile_path": tmp_path / "profile.json",
        "exterior_proxy_glb_path": tmp_path / "exterior.glb",
    }
    roots = {
        actor_id: np.zeros((270, 3), dtype=np.float64)
        for actor_id in ("human0", "dog0")
    }
    forwards = {
        actor_id: np.asarray((0.0, -1.0), dtype=np.float64)
        for actor_id in roots
    }

    with pytest.raises(_StopAfterHook):
        HUMAN_BEAGLE_CAPTURE_ADAPTER.capture(
            room_manifest_path=tmp_path / "room.json",
            m1_request_path=tmp_path / "m1.json",
            provider_assets=paths,
            actor_root_paths=roots,
            actor_fallback_forwards_xz=forwards,
            output_dir=tmp_path / "capture",
            runtime_root=None,
            route_provenance={},
        )
    assert observed == {
        "simulator": "simulator",
        "profile": profile,
        "exterior": paths["exterior_proxy_glb_path"],
        "camera": (1.0, 2.0, 3.0),
        "habitat_sim": "habitat_sim",
        "mn": "magnum",
    }
