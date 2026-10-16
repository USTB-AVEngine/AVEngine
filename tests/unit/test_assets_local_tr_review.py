from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from avengine.contracts.json_io import canonical_json_sha256
from avengine.assets.local_tr_review import (
    EVIDENCE_SCHEMA,
    REBASE_REPORT_SCHEMA,
    LocalTRReviewError,
    actor_from_skin_root_from_rebase_report,
    compile_review_frames,
    mixed_joint_readback_errors,
    validate_review_schedule,
    verify_local_tr_review_evidence,
)


@dataclass(frozen=True)
class _Block:
    link_name: str
    joint_position_offset: int
    joint_position_count: int


_LOCAL_TR_ORDER = tuple(f"source_joint_{index}" for index in range(38))


def _schedule() -> dict[str, object]:
    states = []
    for frame_index in range(75):
        states.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * 3200,
                "action_id": "idle" if frame_index < 30 else "walk",
                "action_time_ticks": frame_index * 3200,
                "root_transform": {
                    "translation_m": [float(frame_index) / 100.0, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                # These formal-v1 fields are intentionally irrelevant to local-TR.
                "joint_states": [{"invalid_for_local_tr": True}],
                "pose_hash": "0" * 64,
                "applied_state_hash": "1" * 64,
            }
        )
    return {
        "schema": "avengine_m2_articulated_capture_request_v1",
        "request_id": "schedule_only",
        "room_id": "room",
        "camera_rig_id": "camera_rig_0",
        "listener_id": "listener0",
        "seed": 17,
        "modalities": ["rgb", "depth", "semantic"],
        # Historical rotation-only order: deliberately different from the 38
        # local-TR source joints used by this unit fixture.
        "runtime_joint_order": [f"old_projected_joint_{index}" for index in range(27)],
        "capture_policy": {
            "state_evaluation": "explicit_fixed_state",
            "advance_clock_between_modalities": False,
            "free_running_animation": False,
        },
        "states": states,
    }


def _actions() -> SimpleNamespace:
    first_translation = [(0.0, 0.0, 0.0)] * len(_LOCAL_TR_ORDER)
    second_translation = list(first_translation)
    first_translation[1] = (0.0, -0.5, 0.0)
    second_translation[1] = (0.1, -0.4, 0.0)
    translations = (tuple(first_translation), tuple(second_translation))
    rotations = tuple(
        tuple((0.0, 0.0, 0.0, 1.0) for _name in _LOCAL_TR_ORDER) for _sample in range(2)
    )
    clips = {
        action_id: SimpleNamespace(
            source_action_name=source_name,
            loop_duration_ticks=6400,
            sample_ticks=(0, 3200),
            translations_m=translations,
            rotations_xyzw=rotations,
        )
        for action_id, source_name in (("idle", "Idle"), ("walk", "Walking"))
    }
    return SimpleNamespace(
        runtime_joint_order=_LOCAL_TR_ORDER,
        action=lambda action_id: clips[action_id],
    )


def test_compile_review_frames_uses_local_tr_and_maps_root() -> None:
    actor_from_skin_root = np.eye(4, dtype=np.float64)
    actor_from_skin_root[2, 3] = 0.25
    mapping = SimpleNamespace(
        runtime_joint_order=_LOCAL_TR_ORDER,
        actor_from_skin_root=actor_from_skin_root,
    )

    frames = compile_review_frames(_schedule(), _actions(), mapping)

    assert len(frames) == 75
    assert frames[0].translations_m[1] == (0.0, -0.5, 0.0)
    assert frames[1].translations_m[1] == (0.1, -0.4, 0.0)
    assert frames[1].world_from_actor[0][3] == 0.01
    assert frames[1].world_from_skin_root[0][3] == 0.01
    assert frames[1].world_from_skin_root[2][3] == 0.25
    assert frames[1].action_sample_index == 1


def test_rebase_report_hash_mismatch_fails_before_runtime(tmp_path: Path) -> None:
    report = {
        "schema": REBASE_REPORT_SCHEMA,
        "status": "pass",
        "qualification_claim": False,
        "output": {"sha256": "1" * 64},
        "skin": {"actor_from_canonical_root": np.eye(4).tolist()},
        "runtime_contract": {
            "schema": "avengine_m2_local_tr_runtime_v2",
            "per_bone_dynamic_translation": True,
        },
    }

    with pytest.raises(LocalTRReviewError, match="output hash differs"):
        actor_from_skin_root_from_rebase_report(
            report,
            visual_sha256="2" * 64,
            visual_path=tmp_path / "visual.glb",
            visual_byte_size=123,
            report_path=tmp_path / "rebase.json",
            report_sha256="3" * 64,
        )


def test_validate_schedule_joins_the_existing_m1_view() -> None:
    room_inputs = SimpleNamespace(
        room={"room_id": "room"},
        request={
            "seed": 17,
            "primary_camera_rig": {"rig_id": "camera_rig_0", "view_id": "view0"},
            "listener": {"listener_id": "listener0"},
        },
    )
    assert not validate_review_schedule(
        _schedule(), runtime_joint_order=_LOCAL_TR_ORDER, room_inputs=room_inputs
    )

    changed = _schedule()
    changed["camera_rig_id"] = "view1"
    errors = validate_review_schedule(
        changed, runtime_joint_order=_LOCAL_TR_ORDER, room_inputs=room_inputs
    )
    assert "schedule camera_rig_id differs from M1 camera rig" in errors


def test_mixed_readback_separates_prismatic_and_spherical_errors() -> None:
    expected = np.asarray([0.2, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0])
    actual = expected.copy()
    actual[1] += 3e-7
    # A quaternion sign flip is the same spherical state.
    actual[3:7] *= -1.0
    blocks = (
        _Block("foot__tx", 0, 1),
        _Block("foot__ty", 1, 1),
        _Block("foot__tz", 2, 1),
        _Block("foot", 3, 4),
    )

    prismatic_error, spherical_error = mixed_joint_readback_errors(
        actual, expected, blocks
    )

    assert np.isclose(prismatic_error, 3e-7)
    assert spherical_error == 0.0


def test_evidence_verifier_fails_closed_on_formal_claim(tmp_path: Path) -> None:
    evidence: dict[str, object] = {
        "schema": EVIDENCE_SCHEMA,
        "status": "review_only",
        "review_only": True,
        "formal_capture": False,
        "qualification_claim": True,
        "formal_view_ids": ["view0"],
        "formal_modalities": ["rgb"],
        "review_view_ids": ["view0"],
        "sensor_view_created": False,
        "inputs": {},
        "runtime_assets": {},
        "runtime_application": {},
        "frames": [],
        "array_artifacts": {},
        "review_media": {},
    }
    evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)

    errors = verify_local_tr_review_evidence(evidence, tmp_path)

    assert "local-TR evidence must declare qualification_claim=false" in errors
    assert "local-TR evidence must not claim formal views/modalities" in errors
