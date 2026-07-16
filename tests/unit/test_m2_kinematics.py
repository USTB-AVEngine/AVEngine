from __future__ import annotations

from dataclasses import replace
import json
import math

import pytest

from avengine.m2.actions import (
    SAMPLE_RATE_HZ,
    TICKS_PER_SAMPLE,
    TIME_BASE_HZ,
    BakedActionClip,
    BakedActionSet,
    baked_actions_content_sha256,
)
from avengine.m2.contracts import CONTACT_ORDER as CONTRACT_CONTACT_ORDER
from avengine.m2.habitat import HabitatAssetMapping, HabitatJointRest
from avengine.m2.kinematics import (
    CONTACT_ORDER,
    AnchorDefinition,
    ContactInferenceThresholds,
    KinematicsError,
    RigidTransform,
    derive_contact_phases,
    forward_kinematics,
    resolve_actor_anchors,
)


SOURCE_SHA256 = "12" * 32
IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _joint(
    ordinal: int,
    node_index: int,
    joint_id: str,
    parent_joint_id: str | None,
    translation: tuple[float, float, float],
    rotation: tuple[float, float, float, float] = IDENTITY,
) -> HabitatJointRest:
    return HabitatJointRest(
        joint_ordinal=ordinal,
        node_index=node_index,
        joint_id=joint_id,
        parent_joint_id=parent_joint_id,
        local_translation_m=translation,
        rest_rotation_xyzw=rotation,
        local_scale=(1.0, 1.0, 1.0),
    )


def _fk_mapping() -> HabitatAssetMapping:
    # Deliberately not topological: the child precedes both its parent and root.
    joints = (
        _joint(0, 2, "paw", "hip", (1.0, 0.0, 0.0)),
        _joint(1, 0, "root", None, (0.0, 0.0, 0.0)),
        _joint(2, 1, "hip", "root", (1.0, 0.0, 0.0)),
    )
    return HabitatAssetMapping(
        source_glb_sha256=SOURCE_SHA256,
        root_joint_id="root",
        joint_order=tuple(joint.joint_id for joint in joints),
        runtime_joint_order=("paw", "hip"),
        joints=joints,
        actor_from_skin_root=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        actor_from_skin_root_source="test.rebase_report",
    )


def _z_rotation(angle: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(0.5 * angle), math.cos(0.5 * angle))


def test_forward_kinematics_uses_absolute_child_local_rotation() -> None:
    mapping = _fk_mapping()
    pose = (IDENTITY, _z_rotation(math.pi / 2.0))

    frame = forward_kinematics(mapping, pose)

    assert frame.joint_order == mapping.joint_order
    assert frame.joint_transform("root") == RigidTransform.identity()
    hip = frame.joint_transform("hip")
    assert hip.translation_m == pytest.approx((1.0, 0.0, 0.0), abs=1.0e-12)
    assert hip.rotation_xyzw == pytest.approx(_z_rotation(math.pi / 2.0))
    paw = frame.joint_transform("paw")
    assert paw.translation_m == pytest.approx((1.0, 1.0, 0.0), abs=1.0e-12)
    assert paw.rotation_xyzw == pytest.approx(_z_rotation(math.pi / 2.0))
    assert frame.to_json_data()["joints"][0]["joint_id"] == "paw"


def test_forward_kinematics_applies_explicit_actor_from_skin_root() -> None:
    mapping = replace(
        _fk_mapping(),
        actor_from_skin_root=(
            (0.0, -1.0, 0.0, 10.0),
            (1.0, 0.0, 0.0, 20.0),
            (0.0, 0.0, 1.0, 30.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )

    frame = forward_kinematics(mapping, (IDENTITY, IDENTITY))

    assert frame.joint_transform("root").translation_m == (10.0, 20.0, 30.0)
    assert frame.joint_transform("hip").translation_m == pytest.approx(
        (10.0, 21.0, 30.0), abs=1.0e-12
    )
    assert frame.joint_transform("paw").translation_m == pytest.approx(
        (10.0, 22.0, 30.0), abs=1.0e-12
    )


def test_anchor_transform_direction_is_actor_from_joint_times_joint_from_anchor() -> (
    None
):
    mapping = _fk_mapping()
    anchors = (
        AnchorDefinition(
            anchor_id="toe_tip",
            joint_id="paw",
            joint_from_anchor=RigidTransform(
                translation_m=(0.25, 0.0, 0.0),
                rotation_xyzw=IDENTITY,
            ),
        ),
    )

    frame = resolve_actor_anchors(
        mapping,
        (IDENTITY, _z_rotation(math.pi / 2.0)),
        anchors,
    )

    anchor = frame.anchor_transform("toe_tip")
    assert anchor.translation_m == pytest.approx((1.0, 1.25, 0.0), abs=1.0e-12)
    assert anchor.rotation_xyzw == pytest.approx(_z_rotation(math.pi / 2.0))
    assert frame.to_json_data()["anchors"] == [
        {
            "anchor_id": "toe_tip",
            "joint_id": "paw",
            "actor_from_anchor": {
                "translation_m": [1.0, 1.25, 0.0],
                "rotation_xyzw": list(_z_rotation(math.pi / 2.0)),
            },
        }
    ]


@pytest.mark.parametrize(
    ("pose", "message"),
    [
        ((IDENTITY,), "shape"),
        ((IDENTITY, (0.0, 0.0, float("nan"), 1.0)), "finite"),
        ((IDENTITY, (0.0, 0.0, 0.0, 2.0)), "unit normalized"),
        ((IDENTITY, (0.0, 0.0, 0.0, -1.0)), "canonical quaternion hemisphere"),
        ((IDENTITY, (-0.0, 0.0, 0.0, 1.0)), "signed zero"),
        ((IDENTITY, (False, 0.0, 0.0, 1.0)), "numeric"),
        ((IDENTITY, ("0", 0.0, 0.0, 1.0)), "numeric"),
    ],
)
def test_forward_kinematics_rejects_noncanonical_pose(
    pose: object, message: str
) -> None:
    with pytest.raises(KinematicsError, match=message):
        forward_kinematics(_fk_mapping(), pose)


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        (
            replace(_fk_mapping(), runtime_joint_order=("hip", "paw")),
            "runtime_joint_order",
        ),
        (
            replace(
                _fk_mapping(),
                actor_from_skin_root=(
                    (2.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
            ),
            "proper rigid",
        ),
        (
            replace(
                _fk_mapping(),
                joints=(
                    replace(_fk_mapping().joints[0], parent_joint_id="missing"),
                    *_fk_mapping().joints[1:],
                ),
            ),
            "unknown parent",
        ),
        (
            replace(
                _fk_mapping(),
                joints=(
                    replace(_fk_mapping().joints[0], parent_joint_id="hip"),
                    _fk_mapping().joints[1],
                    replace(_fk_mapping().joints[2], parent_joint_id="paw"),
                ),
            ),
            "cycle",
        ),
    ],
)
def test_forward_kinematics_rejects_invalid_mapping(
    mapping: HabitatAssetMapping, message: str
) -> None:
    with pytest.raises(KinematicsError, match=message):
        forward_kinematics(mapping, (IDENTITY, IDENTITY))


def test_anchor_definitions_must_be_unique_ordered_and_reference_known_joint() -> None:
    valid = AnchorDefinition(
        "tip",
        "paw",
        RigidTransform((0.0, 0.0, 0.0), IDENTITY),
    )
    duplicate = replace(valid, joint_id="hip")
    unknown = replace(valid, anchor_id="unknown", joint_id="missing")

    with pytest.raises(KinematicsError, match="immutable tuple"):
        resolve_actor_anchors(_fk_mapping(), (IDENTITY, IDENTITY), [valid])
    with pytest.raises(KinematicsError, match="duplicated"):
        resolve_actor_anchors(_fk_mapping(), (IDENTITY, IDENTITY), (valid, duplicate))
    with pytest.raises(KinematicsError, match="unknown joint"):
        resolve_actor_anchors(_fk_mapping(), (IDENTITY, IDENTITY), (unknown,))


@pytest.mark.parametrize(
    ("translation", "rotation", "message"),
    [
        ((float("inf"), 0.0, 0.0), IDENTITY, "finite"),
        ((-0.0, 0.0, 0.0), IDENTITY, "signed zero"),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 2.0), "unit normalized"),
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, -1.0),
            "canonical quaternion hemisphere",
        ),
    ],
)
def test_rigid_transform_is_strictly_canonical(
    translation: tuple[float, float, float],
    rotation: tuple[float, float, float, float],
    message: str,
) -> None:
    with pytest.raises(KinematicsError, match=message):
        RigidTransform(translation, rotation)


def _contact_mapping() -> HabitatAssetMapping:
    joints = (
        _joint(0, 0, "root", None, (0.0, 0.0, 0.0)),
        _joint(1, 1, "front_left_joint", "root", (-0.3, 0.0, -0.2)),
        _joint(2, 2, "front_right_joint", "root", (0.3, 0.0, -0.2)),
        _joint(3, 3, "hind_left_joint", "root", (-0.3, 0.0, 0.2)),
        _joint(4, 4, "hind_right_joint", "root", (0.3, 0.0, 0.2)),
    )
    return HabitatAssetMapping(
        source_glb_sha256=SOURCE_SHA256,
        root_joint_id="root",
        joint_order=tuple(joint.joint_id for joint in joints),
        runtime_joint_order=tuple(joint.joint_id for joint in joints[1:]),
        joints=joints,
        actor_from_skin_root=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        actor_from_skin_root_source="test.rebase_report",
    )


def _contact_anchors() -> tuple[AnchorDefinition, ...]:
    return tuple(
        AnchorDefinition(
            anchor_id=contact_id,
            joint_id=joint_id,
            joint_from_anchor=RigidTransform((0.1, 0.0, 0.0), IDENTITY),
        )
        for contact_id, joint_id in zip(
            CONTACT_ORDER,
            (
                "front_left_joint",
                "front_right_joint",
                "hind_left_joint",
                "hind_right_joint",
            ),
            strict=True,
        )
    )


def _clip(
    semantic_action_id: str,
    source_action_name: str,
    frames: tuple[tuple[tuple[float, float, float, float], ...], ...],
) -> BakedActionClip:
    loop_duration_ticks = len(frames) * TICKS_PER_SAMPLE
    return BakedActionClip(
        semantic_action_id=semantic_action_id,
        source_action_name=source_action_name,
        clip_start_seconds=0.0,
        clip_end_seconds=loop_duration_ticks / TIME_BASE_HZ,
        loop_duration_ticks=loop_duration_ticks,
        sample_ticks=tuple(range(0, loop_duration_ticks, TICKS_PER_SAMPLE)),
        source_times_seconds=tuple(
            tick / TIME_BASE_HZ
            for tick in range(0, loop_duration_ticks, TICKS_PER_SAMPLE)
        ),
        rotations_xyzw=frames,
    )


def _actions(
    *,
    sample_count: int = 20,
    front_amplitude: float = 0.20,
    hind_amplitude: float = 0.002,
    idle_amplitude: float = 0.0,
) -> BakedActionSet:
    idle_frames: list[tuple[tuple[float, float, float, float], ...]] = []
    walk_frames: list[tuple[tuple[float, float, float, float], ...]] = []
    for index in range(sample_count):
        phase = 2.0 * math.pi * index / sample_count
        idle_angle = idle_amplitude * (1.0 + math.sin(phase))
        idle_frames.append(tuple(_z_rotation(idle_angle) for _ in CONTACT_ORDER))
        walk_frames.append(
            (
                _z_rotation(front_amplitude * (1.0 + math.sin(phase))),
                _z_rotation(front_amplitude * (1.0 - math.sin(phase))),
                _z_rotation(hind_amplitude * (1.0 + math.cos(phase))),
                _z_rotation(hind_amplitude * (1.0 - math.cos(phase))),
            )
        )
    return BakedActionSet(
        source_glb_sha256=SOURCE_SHA256,
        runtime_joint_order=_contact_mapping().runtime_joint_order,
        actions=(
            _clip("idle", "Idle", tuple(idle_frames)),
            _clip("walk", "Walking", tuple(walk_frames)),
        ),
    )


def test_contact_order_is_exactly_the_contract_order() -> None:
    assert CONTACT_ORDER == tuple(CONTRACT_CONTACT_ORDER)


def test_contact_inference_uses_all_twenty_frames_and_keeps_idle_planted() -> None:
    report = derive_contact_phases(
        _contact_mapping(),
        _actions(),
        _contact_anchors(),
    )

    assert report.contact_order == CONTACT_ORDER
    assert report.sample_rate_hz == SAMPLE_RATE_HZ
    assert tuple(action.semantic_action_id for action in report.actions) == (
        "idle",
        "walk",
    )
    idle = report.action("idle")
    walk = report.action("walk")
    assert len(idle.frames) == len(walk.frames) == 20
    assert all(frame.in_contact == (True, True, True, True) for frame in idle.frames)
    for front_contact in CONTACT_ORDER[:2]:
        metric = walk.metric(front_contact)
        assert metric.inference_mode == "height_dynamic"
        assert metric.contact_frame_count > 0
        assert metric.swing_frame_count > 0
        assert metric.vertical_range_m > 0.01
    for hind_contact in CONTACT_ORDER[2:]:
        metric = walk.metric(hind_contact)
        assert metric.inference_mode == "low_excursion_kept_contact"
        assert metric.contact_frame_count == 20
        assert metric.swing_frame_count == 0
        assert metric.confidence == "low"
    assert {(warning.code, warning.contact_id) for warning in report.warnings} == {
        ("low_vertical_excursion_kept_contact", "paw_hind_left"),
        ("low_vertical_excursion_kept_contact", "paw_hind_right"),
    }


def test_contact_inference_follows_the_baked_action_sample_count() -> None:
    report = derive_contact_phases(
        _contact_mapping(),
        _actions(sample_count=25),
        _contact_anchors(),
    )

    assert len(report.action("idle").frames) == 25
    assert len(report.action("walk").frames) == 25
    assert report.action("walk").frames[-1].sample_index == 24


def test_contact_report_has_stable_canonical_json_and_quantitative_metrics() -> None:
    actions = _actions()
    report = derive_contact_phases(
        _contact_mapping(),
        actions,
        _contact_anchors(),
    )

    first = report.to_canonical_json()
    second = report.to_canonical_json()
    decoded = json.loads(first)

    assert first == second
    assert first.endswith("\n")
    assert decoded["schema"] == "avengine_m2_contact_phases_v1"
    assert decoded["source_glb_sha256"] == SOURCE_SHA256
    assert decoded["baked_actions_sha256"] == baked_actions_content_sha256(actions)
    assert decoded["runtime_joint_order"] == list(actions.runtime_joint_order)
    assert decoded["qualification_state"] == "research_candidate"
    assert decoded["qualification_claim"] is False
    assert decoded["contact_order"] == list(CONTACT_ORDER)
    assert decoded["notes"] == [
        "Contact phases are inferred from declared actor-space paw-anchor trajectories.",
        "Actor-space contact warnings are diagnostic; world-space foot-lock "
        "certification also requires a hash-bound root trajectory.",
    ]
    assert (
        decoded["actions"][1]["metrics"][0]["maximum_height_m"]
        > decoded["actions"][1]["metrics"][0]["minimum_height_m"]
    )
    assert len(report.content_sha256()) == 64
    assert report.content_sha256() == report.content_sha256()
    assert "-0.0" not in first


def test_idle_motion_is_warned_but_idle_contact_states_remain_true() -> None:
    report = derive_contact_phases(
        _contact_mapping(),
        _actions(idle_amplitude=0.10),
        _contact_anchors(),
    )

    idle = report.action("idle")
    assert all(frame.in_contact == (True, True, True, True) for frame in idle.frames)
    assert any(warning.code == "idle_anchor_motion" for warning in report.warnings)
    assert all(metric.confidence == "low" for metric in idle.metrics)


def test_contact_inference_rejects_front_leg_without_supported_swing() -> None:
    with pytest.raises(KinematicsError, match="front paw.*vertical excursion"):
        derive_contact_phases(
            _contact_mapping(),
            _actions(front_amplitude=0.002),
            _contact_anchors(),
        )


@pytest.mark.parametrize(
    ("mapping", "actions", "anchors", "message"),
    [
        (
            _contact_mapping(),
            _actions(sample_count=2),
            _contact_anchors(),
            "at least three",
        ),
        (
            replace(_contact_mapping(), source_glb_sha256="34" * 32),
            _actions(),
            _contact_anchors(),
            "source_glb_sha256",
        ),
        (
            _contact_mapping(),
            replace(
                _actions(),
                runtime_joint_order=tuple(
                    reversed(_contact_mapping().runtime_joint_order)
                ),
            ),
            _contact_anchors(),
            "runtime_joint_order",
        ),
        (
            _contact_mapping(),
            _actions(),
            tuple(reversed(_contact_anchors())),
            "CONTACT_ORDER",
        ),
    ],
)
def test_contact_inference_rejects_mismatched_inputs(
    mapping: HabitatAssetMapping,
    actions: BakedActionSet,
    anchors: tuple[AnchorDefinition, ...],
    message: str,
) -> None:
    with pytest.raises(KinematicsError, match=message):
        derive_contact_phases(mapping, actions, anchors)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"minimum_dynamic_vertical_range_m": 0.0}, "positive"),
        ({"minimum_dynamic_vertical_range_m": -0.0}, "signed zero"),
        ({"contact_height_fraction": 1.0}, "between 0 and 1"),
        ({"contact_height_fraction": -0.0}, "signed zero"),
        ({"maximum_idle_step_displacement_m": float("nan")}, "finite"),
    ],
)
def test_contact_thresholds_are_strict(changes: dict[str, float], message: str) -> None:
    with pytest.raises(KinematicsError, match=message):
        ContactInferenceThresholds(**changes)
