from __future__ import annotations

from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from avengine.assets.glb import GlbDocument
from avengine.assets.habitat import HabitatLinkJointBlock, build_habitat_asset_mapping
from avengine.assets.local_tr_actions import LocalTRActionClip, LocalTRActionSet
from avengine.assets.local_tr_habitat import (
    LocalTRHabitatMappingError,
    bind_local_tr_habitat_layout,
    build_local_tr_habitat_mapping,
)


GLB_SHA256 = "34" * 32
IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
STATIC_REST = (-0.25, 0.75, 0.125)
DRIVEN_REST = (0.5, -1.25, 1.75)
DRIVEN_DELTAS = ((-0.2, 0.3, -0.4), (0.1, -0.15, 0.2))
HALF_TURN_Z = (0.0, 0.0, 1.0, 0.0)


def _document(*, static_name: str = "static fin") -> GlbDocument:
    # Skin order is deliberately static, root, driven. The driven joint's skin
    # ordinal is therefore 2 and is stable regardless of hierarchy traversal.
    return GlbDocument(
        json={
            "asset": {"version": "2.0"},
            "nodes": [
                {
                    "name": "root",
                    "children": [1, 2],
                    "translation": [0.0, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "scale": [1.0, 1.0, 1.0],
                },
                {
                    "name": 'driven leg <R> "A"',
                    "translation": list(DRIVEN_REST),
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "scale": [1.0, 1.0, 1.0],
                },
                {
                    "name": static_name,
                    "translation": list(STATIC_REST),
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "scale": [1.0, 1.0, 1.0],
                },
            ],
            "skins": [{"skeleton": 0, "joints": [2, 0, 1]}],
        },
        binary=b"",
        sha256=GLB_SHA256,
        byte_length=0,
    )


def _add(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _actions(*, static_name: str = "static fin") -> SimpleNamespace:
    identity = (0.0, 0.0, 0.0, 1.0)
    translations = tuple(
        (STATIC_REST, _add(DRIVEN_REST, delta)) for delta in DRIVEN_DELTAS
    )
    rotations = (
        (identity, identity),
        (identity, HALF_TURN_Z),
    )
    clip = SimpleNamespace(
        semantic_action_id="Walking",
        translations_m=translations,
        rotations_xyzw=rotations,
    )
    return SimpleNamespace(
        source_glb_sha256=GLB_SHA256,
        runtime_joint_order=(static_name, 'driven leg <R> "A"'),
        rest_translations_m=(STATIC_REST, DRIVEN_REST),
        translation_driven_joint_ids=('driven leg <R> "A"',),
        actions=(clip,),
    )


def _mapping():
    return build_local_tr_habitat_mapping(
        _document(),
        _actions(),
        actor_from_skin_root=IDENTITY,
        actor_from_skin_root_source="research_manifest.actor_from_skin_root",
    )


def _scrambled_blocks() -> tuple[HabitatLinkJointBlock, ...]:
    # Dense coverage is deliberate, but neither link traversal nor offsets use
    # the asset/runtime joint order.
    return (
        HabitatLinkJointBlock('driven leg <R> "A"', 0, 4),
        HabitatLinkJointBlock("__avengine_local_tr__000002__z", 4, 1),
        HabitatLinkJointBlock("static fin", 5, 4),
        HabitatLinkJointBlock("__avengine_local_tr__000002__x", 9, 1),
        HabitatLinkJointBlock("__avengine_local_tr__000002__y", 10, 1),
    )


def test_mapping_expands_only_translation_driven_joint() -> None:
    mapping = _mapping()

    assert mapping.runtime_joint_order == (
        "static fin",
        'driven leg <R> "A"',
    )
    assert mapping.translation_driven_joint_ids == ('driven leg <R> "A"',)
    assert mapping.runtime_joint_position_count == 11  # 4*N + 3*D
    assert [
        (spec.link_name, spec.joint_position_count)
        for spec in mapping.runtime_link_specs
    ] == [
        ("static fin", 4),
        ("__avengine_local_tr__000002__x", 1),
        ("__avengine_local_tr__000002__y", 1),
        ("__avengine_local_tr__000002__z", 1),
        ('driven leg <R> "A"', 4),
    ]
    data = mapping.joint_mapping_data()
    assert data["schema"] == "avengine_m2_habitat_joint_mapping_local_tr_v2"
    assert data["research_only"] is True
    assert data["qualification_claim"] is False
    assert data["prismatic_state_semantics"] == (
        "absolute_child_local_translation_m - rest_local_translation_m"
    )
    assert data["habitat_layout"]["runtime_joint_position_count"] == 11
    assert data["habitat_layout"]["runtime_joint_position_count_formula"] == (
        "4*N + 3*D"
    )


def test_build_integrates_with_public_local_tr_action_set() -> None:
    identity = (0.0, 0.0, 0.0, 1.0)

    def clip(
        semantic_action_id: str,
        source_action_name: str,
        driven_delta: tuple[float, float, float],
    ) -> LocalTRActionClip:
        return LocalTRActionClip(
            semantic_action_id=semantic_action_id,
            source_action_name=source_action_name,
            clip_start_seconds=0.0,
            clip_end_seconds=1.0 / 15.0,
            loop_duration_ticks=3200,
            sample_ticks=(0,),
            source_times_seconds=(0.0,),
            translations_m=((STATIC_REST, _add(DRIVEN_REST, driven_delta)),),
            rotations_xyzw=((identity, identity),),
        )

    actions = LocalTRActionSet(
        source_glb_sha256=GLB_SHA256,
        runtime_joint_order=("static fin", 'driven leg <R> "A"'),
        rest_translations_m=(STATIC_REST, DRIVEN_REST),
        translation_driven_joint_ids=('driven leg <R> "A"',),
        actions=(
            clip("idle", "Idle", (-0.1, 0.0, 0.2)),
            clip("walk", "Walking", (0.1, -0.2, -0.1)),
        ),
    )

    mapping = build_local_tr_habitat_mapping(
        _document(),
        actions,
        actor_from_skin_root=IDENTITY,
        actor_from_skin_root_source="test.public_local_tr_action_set",
    )

    assert mapping.translation_driven_joint_ids == ('driven leg <R> "A"',)
    assert mapping.driven_joints[0].delta_min_m == pytest.approx((-0.1, -0.2, -0.1))
    assert mapping.driven_joints[0].delta_max_m == pytest.approx((0.1, 0.0, 0.2))


def test_urdf_uses_nonzero_rest_once_and_derives_bounded_limits() -> None:
    mapping = _mapping()
    first = mapping.render_urdf(robot_name='horse <local TR> "review"')
    assert first == mapping.render_urdf(robot_name='horse <local TR> "review"')
    assert first.endswith("\n")

    root = ET.fromstring(first)
    assert root.attrib["name"] == 'horse <local TR> "review"'
    by_child = {
        joint.find("child").attrib["link"]: joint for joint in root.findall("joint")
    }
    static = by_child["static fin"]
    assert static.attrib["type"] == "spherical"
    assert static.find("origin").attrib["xyz"] == "-0.25 0.75 0.125"

    x_joint = by_child["__avengine_local_tr__000002__x"]
    y_joint = by_child["__avengine_local_tr__000002__y"]
    z_joint = by_child["__avengine_local_tr__000002__z"]
    rotation_joint = by_child['driven leg <R> "A"']
    assert [joint.attrib["type"] for joint in (x_joint, y_joint, z_joint)] == [
        "prismatic",
        "prismatic",
        "prismatic",
    ]
    assert x_joint.find("origin").attrib["xyz"] == "0.5 -1.25 1.75"
    assert y_joint.find("origin").attrib["xyz"] == "0 0 0"
    assert z_joint.find("origin").attrib["xyz"] == "0 0 0"
    assert [
        joint.find("axis").attrib["xyz"] for joint in (x_joint, y_joint, z_joint)
    ] == ["1 0 0", "0 1 0", "0 0 1"]
    assert rotation_joint.attrib["type"] == "spherical"
    assert rotation_joint.find("parent").attrib["link"] == (
        "__avengine_local_tr__000002__z"
    )
    assert rotation_joint.find("origin").attrib["xyz"] == "0 0 0"

    expected_extrema = tuple(zip(*DRIVEN_DELTAS, strict=True))
    for axis_index, joint in enumerate((x_joint, y_joint, z_joint)):
        limit = joint.find("limit")
        lower = float(limit.attrib["lower"])
        upper = float(limit.attrib["upper"])
        assert lower == pytest.approx(min(0.0, *expected_extrema[axis_index]) - 1.0e-4)
        assert upper == pytest.approx(max(0.0, *expected_extrema[axis_index]) + 1.0e-4)
        assert lower < 0.0 < upper


def test_binding_maps_absolute_nonzero_rest_to_delta_and_rotation_by_name() -> None:
    mapping = _mapping()
    binding = bind_local_tr_habitat_layout(
        mapping, _scrambled_blocks(), joint_position_count=11
    )
    translations = (STATIC_REST, _add(DRIVEN_REST, DRIVEN_DELTAS[0]))
    rotations = ((0.0, 0.0, 0.0, 1.0), HALF_TURN_Z)

    positions = binding.map_pose(translations, rotations)

    assert positions == pytest.approx(
        (
            *HALF_TURN_Z,
            DRIVEN_DELTAS[0][2],
            0.0,
            0.0,
            0.0,
            1.0,
            DRIVEN_DELTAS[0][0],
            DRIVEN_DELTAS[0][1],
        )
    )
    # If absolute T had been written after origin=T_rest, these values would be
    # 0.3, -0.95, 1.35. The actual state is the exact rest-relative delta.
    assert positions[9] == pytest.approx(-0.2)
    assert positions[10] == pytest.approx(0.3)
    assert positions[4] == pytest.approx(-0.4)
    assert binding.to_json_data()["links"] == [
        {
            "link_name": "static fin",
            "joint_id": "static fin",
            "component": "rotation_xyzw",
            "joint_position_offset": 5,
            "joint_position_count": 4,
        },
        {
            "link_name": "__avengine_local_tr__000002__x",
            "joint_id": 'driven leg <R> "A"',
            "component": "translation_delta_x_m",
            "joint_position_offset": 9,
            "joint_position_count": 1,
        },
        {
            "link_name": "__avengine_local_tr__000002__y",
            "joint_id": 'driven leg <R> "A"',
            "component": "translation_delta_y_m",
            "joint_position_offset": 10,
            "joint_position_count": 1,
        },
        {
            "link_name": "__avengine_local_tr__000002__z",
            "joint_id": 'driven leg <R> "A"',
            "component": "translation_delta_z_m",
            "joint_position_offset": 4,
            "joint_position_count": 1,
        },
        {
            "link_name": 'driven leg <R> "A"',
            "joint_id": 'driven leg <R> "A"',
            "component": "rotation_xyzw",
            "joint_position_offset": 0,
            "joint_position_count": 4,
        },
    ]


@pytest.mark.parametrize(
    ("blocks", "count", "message"),
    [
        (
            (*_scrambled_blocks()[:-1], HabitatLinkJointBlock("extra", 10, 1)),
            11,
            "exactly match",
        ),
        (
            (
                *_scrambled_blocks()[:-1],
                HabitatLinkJointBlock("__avengine_local_tr__000002__y", 9, 1),
            ),
            11,
            "must not overlap",
        ),
        (
            (
                HabitatLinkJointBlock('driven leg <R> "A"', 0, 3),
                *_scrambled_blocks()[1:],
            ),
            11,
            "exactly 4",
        ),
        (
            (
                *_scrambled_blocks()[:-1],
                HabitatLinkJointBlock("__avengine_local_tr__000002__y", 11, 1),
            ),
            12,
            "4.N . 3.D",
        ),
    ],
)
def test_binding_rejects_extra_wrong_sized_overlapping_or_sparse_links(
    blocks: tuple[HabitatLinkJointBlock, ...], count: int, message: str
) -> None:
    with pytest.raises(LocalTRHabitatMappingError, match=message):
        bind_local_tr_habitat_layout(_mapping(), blocks, joint_position_count=count)


def test_build_rejects_dummy_link_name_collision() -> None:
    collision = "__avengine_local_tr__000002__x"
    with pytest.raises(LocalTRHabitatMappingError, match="collide"):
        build_local_tr_habitat_mapping(
            _document(static_name=collision),
            _actions(static_name=collision),
            actor_from_skin_root=IDENTITY,
            actor_from_skin_root_source="test",
        )


def test_build_rejects_non_driven_translation_samples_away_from_rest() -> None:
    actions = _actions()
    clips = list(actions.actions)
    translations = np.asarray(clips[0].translations_m, dtype=np.float64)
    translations[0, 0, 0] += 1.0e-3
    clips[0] = SimpleNamespace(
        semantic_action_id="Walking",
        translations_m=translations,
        rotations_xyzw=clips[0].rotations_xyzw,
    )
    actions.actions = tuple(clips)

    with pytest.raises(LocalTRHabitatMappingError, match="non-driven joint"):
        build_local_tr_habitat_mapping(
            _document(),
            actions,
            actor_from_skin_root=IDENTITY,
            actor_from_skin_root_source="test",
        )


def test_build_accepts_draft_driven_order_alias_but_rejects_wrong_order() -> None:
    actions = _actions()
    actions.translation_driven_joint_order = actions.translation_driven_joint_ids
    del actions.translation_driven_joint_ids
    assert _mapping().translation_driven_joint_ids == ('driven leg <R> "A"',)
    mapping = build_local_tr_habitat_mapping(
        _document(),
        actions,
        actor_from_skin_root=IDENTITY,
        actor_from_skin_root_source="test",
    )
    assert mapping.translation_driven_joint_ids == ('driven leg <R> "A"',)

    actions.translation_driven_joint_order = (
        'driven leg <R> "A"',
        "static fin",
    )
    with pytest.raises(LocalTRHabitatMappingError, match="preserve"):
        build_local_tr_habitat_mapping(
            _document(),
            actions,
            actor_from_skin_root=IDENTITY,
            actor_from_skin_root_source="test",
        )


@pytest.mark.parametrize(
    ("translations", "rotations", "message"),
    [
        ((STATIC_REST,), ((0.0, 0.0, 0.0, 1.0),), "shape"),
        (
            (STATIC_REST, (100.0, 100.0, 100.0)),
            ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
            "outside",
        ),
        (
            ((0.0, 0.0, 0.0), DRIVEN_REST),
            ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
            "differs from rest",
        ),
        (
            (STATIC_REST, DRIVEN_REST),
            ((0.0, 0.0, 0.0, 2.0), (0.0, 0.0, 0.0, 1.0)),
            "unit normalized",
        ),
        (
            (STATIC_REST, DRIVEN_REST),
            ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, -1.0)),
            "canonical quaternion hemisphere",
        ),
    ],
)
def test_binding_rejects_unrepresentable_or_invalid_pose(
    translations: object, rotations: object, message: str
) -> None:
    binding = bind_local_tr_habitat_layout(
        _mapping(), _scrambled_blocks(), joint_position_count=11
    )
    with pytest.raises(LocalTRHabitatMappingError, match=message):
        binding.map_pose(translations, rotations)


def test_build_binds_hash_rest_and_explicit_root_source() -> None:
    actions = _actions()
    actions.source_glb_sha256 = "ff" * 32
    with pytest.raises(LocalTRHabitatMappingError, match="source_glb_sha256"):
        build_local_tr_habitat_mapping(
            _document(),
            actions,
            actor_from_skin_root=IDENTITY,
            actor_from_skin_root_source="test",
        )

    actions = _actions()
    rest = np.asarray(actions.rest_translations_m, dtype=np.float64)
    rest[1, 2] += 0.01
    actions.rest_translations_m = rest
    with pytest.raises(LocalTRHabitatMappingError, match="rest pose"):
        build_local_tr_habitat_mapping(
            _document(),
            actions,
            actor_from_skin_root=IDENTITY,
            actor_from_skin_root_source="test",
        )

    with pytest.raises(TypeError, match="actor_from_skin_root"):
        build_local_tr_habitat_mapping(_document(), _actions())  # type: ignore[call-arg]


def test_formal_spherical_v1_mapping_is_unchanged_by_research_mapping() -> None:
    document = _document()
    before = build_habitat_asset_mapping(
        document,
        actor_from_skin_root=IDENTITY,
        actor_from_skin_root_source="formal_test",
    ).render_urdf()
    _mapping()
    after = build_habitat_asset_mapping(
        document,
        actor_from_skin_root=IDENTITY,
        actor_from_skin_root_source="formal_test",
    ).render_urdf()

    assert before == after
    assert {
        joint.attrib["type"] for joint in ET.fromstring(after).findall("joint")
    } == {"spherical"}


def test_mapping_json_is_detached() -> None:
    mapping = _mapping()
    data = mapping.joint_mapping_data()
    data["runtime_joint_order"].clear()
    data["translation_driven_joints"][0]["dummy_link_names"].clear()

    assert mapping.runtime_joint_order == (
        "static fin",
        'driven leg <R> "A"',
    )
    assert mapping.driven_joints[0].dummy_link_names == (
        "__avengine_local_tr__000002__x",
        "__avengine_local_tr__000002__y",
        "__avengine_local_tr__000002__z",
    )
