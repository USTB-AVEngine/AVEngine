from __future__ import annotations

from pathlib import Path

import pytest

from avengine.contracts.json_io import canonical_json_sha256
from avengine.assets.actions import BakedActionClip, BakedActionSet
from avengine.assets.contracts import (
    CONTACT_ORDER,
    compute_applied_state_hash,
    compute_pose_hash,
    validate_capture_request,
)
from avengine.assets.timeline import (
    FRAME_COUNT,
    IDLE_LEAD_FRAME_COUNT,
    IDLE_TAIL_FRAME_COUNT,
    M2CanaryTrajectory,
    TimelineBuildError,
    WALK_FRAME_COUNT,
    build_m2_capture_request,
    build_m2_research_review_request,
    build_m2_state_sequence,
)


SHA256 = "a" * 64
JOINT_ORDER = ("spine", "paw")


def _clip(action_id: str, source_name: str, offset: float) -> BakedActionClip:
    frames = []
    for index in range(20):
        # These are exact unit quaternions and remain in the canonical hemisphere.
        if (index + int(offset)) % 2:
            first = (0.0, 0.0, 0.0, 1.0)
        else:
            first = (0.0, 0.0, 0.6, 0.8)
        frames.append((first, (0.0, 0.0, 0.0, 1.0)))
    return BakedActionClip(
        semantic_action_id=action_id,
        source_action_name=source_name,
        clip_start_seconds=0.0,
        clip_end_seconds=4.0 / 3.0,
        loop_duration_ticks=64_000,
        sample_ticks=tuple(range(0, 64_000, 3_200)),
        source_times_seconds=tuple(index / 15.0 for index in range(20)),
        rotations_xyzw=tuple(frames),
    )


def _actions() -> BakedActionSet:
    return BakedActionSet(
        source_glb_sha256="b" * 64,
        runtime_joint_order=JOINT_ORDER,
        actions=(
            _clip("idle", "Idle", 0.0),
            _clip("walk", "Walking", 1.0),
        ),
    )


def _asset(*, state: str = "canary_qualified") -> dict:
    return {
        "schema": "avengine_animal_asset_package_v1",
        "asset_id": "dog_canary",
        "admission_state": state,
        "revisions": {"skeleton_revision": "skeleton-v1"},
        "skeleton": {
            "root_joint_id": "root",
            "joint_order": ["root", *JOINT_ORDER],
            "runtime_joint_order": list(JOINT_ORDER),
        },
        "contacts": {"contact_order": list(CONTACT_ORDER)},
    }


def _contacts() -> dict[str, list[list[bool]]]:
    idle = [[True, True, True, True] for _ in range(20)]
    walk = [[index % 2 == 0, index % 2 == 1, True, True] for index in range(20)]
    return {"idle": idle, "walk": walk}


def _states(**overrides):
    arguments = {
        "asset": _asset(),
        "asset_manifest_sha256": SHA256,
        "actions": _actions(),
        "contact_phases": _contacts(),
    }
    arguments.update(overrides)
    return build_m2_state_sequence(**arguments)


def test_exact_schedule_ticks_trajectory_and_hashes() -> None:
    asset = _asset()
    states = _states(asset=asset)

    assert len(states) == FRAME_COUNT == 75
    assert [state["action_id"] for state in states] == (
        ["idle"] * IDLE_LEAD_FRAME_COUNT
        + ["walk"] * WALK_FRAME_COUNT
        + ["idle"] * IDLE_TAIL_FRAME_COUNT
    )
    assert [state["frame_index"] for state in states] == list(range(75))
    assert [state["pts_ticks"] for state in states] == [
        index * 3_200 for index in range(75)
    ]
    assert states[0]["action_time_ticks"] == 0
    assert states[14]["action_time_ticks"] == 44_800
    assert states[15]["action_time_ticks"] == 0
    assert states[34]["action_time_ticks"] == 60_800
    assert states[35]["action_time_ticks"] == 0
    assert states[59]["action_time_ticks"] == 12_800
    assert states[60]["action_time_ticks"] == 0
    assert states[74]["action_time_ticks"] == 44_800

    assert states[0]["root_transform"]["translation_m"] == [-0.15, 0.02, 0.8]
    assert states[14]["root_transform"] == states[0]["root_transform"]
    assert states[15]["root_transform"]["translation_m"] == [-0.15, 0.02, 0.8]
    assert states[59]["root_transform"]["translation_m"] == [-0.15, 0.02, -0.8]
    assert states[60]["root_transform"] == states[59]["root_transform"]
    assert states[74]["root_transform"] == states[59]["root_transform"]
    assert states[37]["root_transform"]["translation_m"] == [-0.15, 0.02, 0.0]
    assert states[0]["root_transform"]["rotation_xyzw"] == [
        0.0,
        0.7071067811865475,
        0.0,
        0.7071067811865476,
    ]

    for state in states:
        assert [joint["joint_id"] for joint in state["joint_states"]] == list(
            JOINT_ORDER
        )
        assert [contact["contact_id"] for contact in state["contact_states"]] == list(
            CONTACT_ORDER
        )
        assert state["mouth_state"] == {"open_ratio": 0.0, "vocalizing": False}
        assert state["pose_hash"] == compute_pose_hash(asset, state)
        assert state["applied_state_hash"] == compute_applied_state_hash(
            asset, state, asset_manifest_sha256=SHA256
        )
        assert state["pose_hash"] != state["applied_state_hash"]


def test_state_builder_is_deterministic_and_returns_detached_data() -> None:
    first = _states()
    second = _states()
    assert first == second
    assert canonical_json_sha256({"states": first}) == canonical_json_sha256(
        {"states": second}
    )
    first[0]["joint_states"][0]["rotation_xyzw"][0] = 0.25
    assert second[0]["joint_states"][0]["rotation_xyzw"][0] == 0.0


def test_custom_trajectory_is_linear_only_during_walk() -> None:
    trajectory = M2CanaryTrajectory(
        start_translation_m=(1.0, 0.0, 2.0),
        end_translation_m=(3.0, 0.0, -2.0),
        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    states = _states(trajectory=trajectory)
    assert states[0]["root_transform"]["translation_m"] == [1.0, 0.0, 2.0]
    assert states[37]["root_transform"]["translation_m"] == [2.0, 0.0, 0.0]
    assert states[-1]["root_transform"]["translation_m"] == [3.0, 0.0, -2.0]


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (
            {"idle": [[True] * 4] * 20},
            "exactly the baked idle and walk",
        ),
        (
            {"idle": [[True] * 4] * 19, "walk": [[True] * 4] * 20},
            "must have 20 frames",
        ),
        (
            {"idle": [[True] * 4] * 20, "walk": [[True] * 3] * 20},
            "must contain four booleans",
        ),
        (
            {
                "idle": [[False, True, True, True]] + [[True] * 4] * 19,
                "walk": [[True] * 4] * 20,
            },
            "idle contact frame",
        ),
    ],
)
def test_contact_phase_input_is_fail_closed(value, match: str) -> None:
    with pytest.raises(TimelineBuildError, match=match):
        _states(contact_phases=value)


def test_asset_and_action_joint_order_must_match() -> None:
    asset = _asset()
    asset["skeleton"]["runtime_joint_order"] = ["paw", "spine"]
    with pytest.raises(TimelineBuildError, match="joint order"):
        _states(asset=asset)


@pytest.mark.parametrize(
    "value",
    ["A" * 64, "a" * 63, "z" * 64, None],
)
def test_manifest_hash_must_be_canonical(value) -> None:
    with pytest.raises(TimelineBuildError, match="lowercase SHA-256"):
        _states(asset_manifest_sha256=value)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"start_translation_m": (0.0, float("nan"), 0.0)}, "three finite"),
        ({"rotation_xyzw": (0.0, 0.0, 0.0, -1.0)}, "canonical hemisphere"),
        ({"rotation_xyzw": (0.0, 0.0, 0.0, 2.0)}, "unit normalized"),
    ],
)
def test_trajectory_rejects_noncanonical_values(kwargs, match: str) -> None:
    with pytest.raises(TimelineBuildError, match=match):
        M2CanaryTrajectory(**kwargs)


def test_formal_request_requires_qualified_asset_before_building() -> None:
    with pytest.raises(TimelineBuildError, match="canary_qualified"):
        build_m2_capture_request(
            asset=_asset(state="research_candidate"),
            asset_manifest_sha256=SHA256,
            actions=_actions(),
            contact_phases=_contacts(),
            request_id="review-only",
            room_id="room",
            seed=17,
        )


def test_research_review_request_keeps_formal_admission_as_only_blocker() -> None:
    asset = _asset(state="research_candidate")
    request = build_m2_research_review_request(
        asset=asset,
        asset_manifest_sha256=SHA256,
        actions=_actions(),
        contact_phases=_contacts(),
        request_id="m2_custom_room_research_review_v1",
        room_id="room",
        seed=17,
    )

    assert len(request["states"]) == 75
    assert request["view_ids"] == ["view0"]
    assert request["modalities"] == ["rgb", "depth", "semantic"]
    assert validate_capture_request(
        request, asset=asset, asset_manifest_sha256=SHA256
    ) == ["M2 capture accepts only a canary_qualified animal package"]


def test_research_review_request_rejects_non_candidate_asset() -> None:
    with pytest.raises(TimelineBuildError, match="research_candidate"):
        build_m2_research_review_request(
            asset=_asset(state="canary_qualified"),
            asset_manifest_sha256=SHA256,
            actions=_actions(),
            contact_phases=_contacts(),
            request_id="review",
            room_id="room",
            seed=17,
        )


def test_formal_request_is_independently_semantically_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The package validator is tested separately.  This focused test supplies
    # the extra package fields the capture validator consumes, while replacing
    # only JSON-Schema I/O so it can assert the full semantic construction.
    asset = _asset()
    monkeypatch.setattr(
        "avengine.assets.contracts._json_schema_errors", lambda value, schema: []
    )
    request = build_m2_capture_request(
        asset=asset,
        asset_manifest_sha256=SHA256,
        actions=_actions(),
        contact_phases=_contacts(),
        request_id="m2_custom_room_canary_v1",
        room_id="blender_custom_two_zone_v1",
        seed=17,
    )

    assert request["view_ids"] == ["view0"]
    assert request["modalities"] == ["rgb", "depth", "semantic"]
    assert request["capture_policy"] == {
        "state_evaluation": "explicit_fixed_state",
        "advance_clock_between_modalities": False,
        "free_running_animation": False,
    }
    assert not validate_capture_request(
        request, asset=asset, asset_manifest_sha256=SHA256
    )


def test_invalid_request_identity_inputs_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "avengine.assets.contracts._json_schema_errors", lambda value, schema: []
    )
    common = {
        "asset": _asset(),
        "asset_manifest_sha256": SHA256,
        "actions": _actions(),
        "contact_phases": _contacts(),
        "request_id": "request",
        "room_id": "room",
        "seed": 17,
    }
    for key, value in (("request_id", ""), ("room_id", ""), ("seed", True)):
        arguments = dict(common)
        arguments[key] = value
        with pytest.raises(TimelineBuildError):
            build_m2_capture_request(**arguments)


def test_timeline_module_does_not_write_files(tmp_path: Path) -> None:
    before = list(tmp_path.iterdir())
    _states()
    assert list(tmp_path.iterdir()) == before
