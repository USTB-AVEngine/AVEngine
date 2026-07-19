from dataclasses import replace
from pathlib import Path

import numpy as np

from avengine.contracts.json_io import load_json
from avengine.m6x.capture_adapter import (
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
        records=tuple({"provider": {"pose": "a" * 64}} for _ in range(270)),
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
    assert state.action_id == "adapter_action"
    assert state.action_phase == 0.7
    assert state.pose_hash == "a" * 64
