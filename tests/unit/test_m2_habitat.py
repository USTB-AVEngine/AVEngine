from __future__ import annotations

from copy import deepcopy
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from avengine.m2.glb import GlbDocument
from avengine.m2.habitat import (
    AVENGINE_NATIVE_GLTF_SKIN_FRAME_KEY,
    HabitatLinkJointBlock,
    HabitatMappingError,
    bind_habitat_link_layout,
    build_habitat_ao_config_data,
    build_habitat_asset_mapping,
    build_habitat_asset_mapping_from_rebase_report,
    map_runtime_pose_to_habitat_joint_positions,
)


GLB_SHA256 = "ab" * 32
ACTOR_FROM_SKIN_ROOT = (
    (0.0, 0.0, -1.0, 1.25),
    (0.0, 1.0, 0.0, 0.5),
    (1.0, 0.0, 0.0, -2.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _document() -> GlbDocument:
    # The skin order intentionally differs from hierarchy order and does not
    # put the root first. Runtime mapping must therefore bind by joint name.
    return GlbDocument(
        json={
            "asset": {"version": "2.0"},
            "nodes": [
                {
                    "name": "pelvis",
                    "children": [1, 2],
                    "translation": [0.0, -0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "scale": [1.0, 1.0, 1.0],
                },
                {
                    "name": 'left paw <review> "A"',
                    "translation": [0.25, -0.5, 0.0],
                    "rotation": [0.0, 0.0, 0.0, -1.0],
                    "scale": [1.0, 1.0, 1.0],
                },
                {
                    "name": "right paw",
                    "translation": [-0.25, -0.5, 0.0],
                    "rotation": [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
                    "scale": [1.0, 1.0, 1.0],
                },
            ],
            "skins": [{"skeleton": 0, "joints": [2, 0, 1]}],
        },
        binary=b"",
        sha256=GLB_SHA256,
        byte_length=0,
    )


def _report() -> dict[str, object]:
    return {
        "schema": "avengine_m2_skin_root_rebase_v1",
        "status": "pass",
        "output": {"sha256": GLB_SHA256},
        "skin": {
            "root_joint": "pelvis",
            "actor_from_canonical_root": [list(row) for row in ACTOR_FROM_SKIN_ROOT],
        },
    }


def _mapping():
    return build_habitat_asset_mapping_from_rebase_report(_document(), _report())


def test_build_mapping_preserves_skin_order_and_removes_only_root() -> None:
    mapping = _mapping()

    assert mapping.source_glb_sha256 == GLB_SHA256
    assert mapping.root_joint_id == "pelvis"
    assert mapping.joint_order == ("right paw", "pelvis", 'left paw <review> "A"')
    assert mapping.runtime_joint_order == (
        "right paw",
        'left paw <review> "A"',
    )
    assert mapping.actor_from_skin_root == ACTOR_FROM_SKIN_ROOT
    assert mapping.joints[2].rest_rotation_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert mapping.joints[2].local_translation_m == (0.25, -0.5, 0.0)
    assert all(joint.local_scale == (1.0, 1.0, 1.0) for joint in mapping.joints)

    data = mapping.joint_mapping_data()
    assert data["coordinate_system"]["up_axis"] == "+Y"
    assert data["coordinate_system"]["forward_axis"] == "-Z"
    assert data["runtime_root_formula"] == (
        "world_from_skin_root = world_from_actor @ actor_from_skin_root"
    )
    assert data["habitat_layout"] == {
        "base_link": "pelvis",
        "runtime_joint_type": "spherical",
        "runtime_joint_position_count": 8,
        "runtime_joint_position_encoding": "xyzw",
        "render_mode": "skin",
    }
    data["runtime_joint_order"].clear()
    assert mapping.runtime_joint_order == (
        "right paw",
        'left paw <review> "A"',
    )


def test_urdf_is_deterministic_escaped_and_uses_root_as_base() -> None:
    mapping = _mapping()

    first = mapping.render_urdf(robot_name='dog <canary> "review"')
    second = mapping.render_urdf(robot_name='dog <canary> "review"')

    assert first == second
    assert first.endswith("\n")
    root = ET.fromstring(first)
    assert root.attrib["name"] == 'dog <canary> "review"'
    assert [link.attrib["name"] for link in root.findall("link")] == list(
        mapping.joint_order
    )
    joints = root.findall("joint")
    assert len(joints) == 2
    assert all(joint.attrib["type"] == "spherical" for joint in joints)
    assert {joint.find("child").attrib["link"] for joint in joints} == set(
        mapping.runtime_joint_order
    )
    assert {joint.find("parent").attrib["link"] for joint in joints} == {"pelvis"}
    left = next(
        joint
        for joint in joints
        if joint.find("child").attrib["link"] == 'left paw <review> "A"'
    )
    assert left.find("origin").attrib == {"xyz": "0.25 -0.5 0", "rpy": "0 0 0"}


def test_ao_config_is_skin_rendering_and_detached() -> None:
    config = build_habitat_ao_config_data(
        render_asset="visual.glb", urdf_filepath="animal.urdf", semantic_id=200
    )

    assert config == {
        "urdf_filepath": "animal.urdf",
        "render_asset": "visual.glb",
        "uniform_scale": 1.0,
        "mass_scale": 1.0,
        "semantic_id": 200,
        "base_type": "free",
        "inertia_source": "computed",
        "link_order": "tree_traversal",
        "render_mode": "skin",
        "shader_type": "phong",
        "user_defined": {AVENGINE_NATIVE_GLTF_SKIN_FRAME_KEY: True},
    }
    config["render_mode"] = "link_visuals"
    assert (
        build_habitat_ao_config_data(
            render_asset="visual.glb", urdf_filepath="animal.urdf", semantic_id=200
        )["render_mode"]
        == "skin"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("render_asset", "", "render_asset"),
        ("urdf_filepath", "", "urdf_filepath"),
        ("semantic_id", True, "semantic_id"),
        ("semantic_id", -1, "semantic_id"),
    ],
)
def test_ao_config_rejects_invalid_fields(
    field: str, value: object, message: str
) -> None:
    arguments: dict[str, object] = {
        "render_asset": "visual.glb",
        "urdf_filepath": "animal.urdf",
        "semantic_id": 200,
    }
    arguments[field] = value
    with pytest.raises(HabitatMappingError, match=message):
        build_habitat_ao_config_data(**arguments)  # type: ignore[arg-type]


def test_rebase_report_binds_hash_root_and_transform() -> None:
    report = _report()
    mapping = build_habitat_asset_mapping_from_rebase_report(_document(), report)
    assert mapping.actor_from_skin_root_source == (
        "avengine_m2_skin_root_rebase_v1.skin.actor_from_canonical_root"
    )

    report = deepcopy(report)
    report["output"]["sha256"] = "cd" * 32  # type: ignore[index]
    with pytest.raises(HabitatMappingError, match="output sha256"):
        build_habitat_asset_mapping_from_rebase_report(_document(), report)

    report = _report()
    report["skin"]["root_joint"] = "wrong"  # type: ignore[index]
    with pytest.raises(HabitatMappingError, match="root_joint"):
        build_habitat_asset_mapping_from_rebase_report(_document(), report)


def test_explicit_identity_transform_is_allowed_but_never_implicit() -> None:
    identity = np.eye(4).tolist()
    mapping = build_habitat_asset_mapping(
        _document(),
        actor_from_skin_root=identity,
        actor_from_skin_root_source="manifest.habitat.actor_from_skin_root",
    )
    assert mapping.actor_from_skin_root == tuple(tuple(row) for row in identity)

    with pytest.raises(TypeError, match="actor_from_skin_root"):
        build_habitat_asset_mapping(_document())  # type: ignore[call-arg]
    with pytest.raises(HabitatMappingError, match="source"):
        build_habitat_asset_mapping(
            _document(), actor_from_skin_root=identity, actor_from_skin_root_source=""
        )


@pytest.mark.parametrize(
    "transform",
    [
        np.diag([2.0, 1.0, 1.0, 1.0]).tolist(),
        np.diag([-1.0, 1.0, 1.0, 1.0]).tolist(),
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0, 0.0, 0.0]] * 4,
    ],
)
def test_mapping_rejects_nonproper_actor_transform(transform: object) -> None:
    with pytest.raises(HabitatMappingError, match="actor_from_skin_root"):
        build_habitat_asset_mapping(
            _document(),
            actor_from_skin_root=transform,  # type: ignore[arg-type]
            actor_from_skin_root_source="test",
        )


def test_mapping_rejects_multiple_skins() -> None:
    document = _document()
    value = document.json
    value["skins"].append(deepcopy(value["skins"][0]))
    multiple = GlbDocument(
        json=value,
        binary=b"",
        sha256=GLB_SHA256,
        byte_length=0,
    )

    with pytest.raises(HabitatMappingError, match="exactly one skin"):
        build_habitat_asset_mapping(
            multiple,
            actor_from_skin_root=ACTOR_FROM_SKIN_ROOT,
            actor_from_skin_root_source="test",
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["skins"][0].__setitem__("joints", [1, 2]),
            "exactly one joint root",
        ),
        (
            lambda value: value["nodes"][1].__setitem__("name", "right paw"),
            "names must be unique",
        ),
        (
            lambda value: value["nodes"][2].__setitem__("scale", [1.0, 1.01, 1.0]),
            "local scale must be exactly unit",
        ),
        (
            lambda value: value["nodes"][0].__setitem__("translation", [0.0, 1.0, 0.0]),
            "identity local translation/rotation",
        ),
    ],
)
def test_mapping_rejects_non_rebased_skin(mutate, message: str) -> None:
    value = _document().json
    mutate(value)
    document = GlbDocument(
        json=value,
        binary=b"",
        sha256=GLB_SHA256,
        byte_length=0,
    )
    with pytest.raises(HabitatMappingError, match=message):
        build_habitat_asset_mapping(
            document,
            actor_from_skin_root=ACTOR_FROM_SKIN_ROOT,
            actor_from_skin_root_source="test",
        )


def _blocks() -> tuple[HabitatLinkJointBlock, ...]:
    # Deliberately opposite to runtime_joint_order.
    return (
        HabitatLinkJointBlock(
            link_name='left paw <review> "A"',
            joint_position_offset=0,
            joint_position_count=4,
        ),
        HabitatLinkJointBlock(
            link_name="right paw",
            joint_position_offset=4,
            joint_position_count=4,
        ),
    )


def test_binding_maps_pose_by_name_instead_of_link_traversal_order() -> None:
    mapping = _mapping()
    binding = bind_habitat_link_layout(
        mapping.runtime_joint_order, _blocks(), joint_position_count=8
    )
    pose = np.asarray(
        [
            [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    positions = binding.map_pose(pose)

    assert positions == (
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        math.sqrt(0.5),
        math.sqrt(0.5),
    )
    assert binding.to_json_data()["links"] == [
        {
            "link_name": "right paw",
            "joint_position_offset": 4,
            "joint_position_count": 4,
        },
        {
            "link_name": 'left paw <review> "A"',
            "joint_position_offset": 0,
            "joint_position_count": 4,
        },
    ]
    assert (
        map_runtime_pose_to_habitat_joint_positions(
            mapping.runtime_joint_order,
            pose,
            _blocks(),
            joint_position_count=8,
        )
        == positions
    )


@pytest.mark.parametrize(
    ("blocks", "count", "message"),
    [
        (
            (
                HabitatLinkJointBlock("right paw", 0, 4),
                HabitatLinkJointBlock("unexpected", 4, 4),
            ),
            8,
            "exactly match",
        ),
        (
            (
                HabitatLinkJointBlock("right paw", 0, 4),
                HabitatLinkJointBlock('left paw <review> "A"', 2, 4),
            ),
            8,
            "must not overlap",
        ),
        (
            (
                HabitatLinkJointBlock("right paw", 0, 3),
                HabitatLinkJointBlock('left paw <review> "A"', 4, 4),
            ),
            8,
            "exactly 4",
        ),
        (
            (
                HabitatLinkJointBlock("right paw", 0, 4),
                HabitatLinkJointBlock('left paw <review> "A"', 8, 4),
            ),
            12,
            "densely cover",
        ),
    ],
)
def test_binding_rejects_invalid_runtime_layout(
    blocks: tuple[HabitatLinkJointBlock, ...], count: int, message: str
) -> None:
    with pytest.raises(HabitatMappingError, match=message):
        bind_habitat_link_layout(
            _mapping().runtime_joint_order,
            blocks,
            joint_position_count=count,
        )


@pytest.mark.parametrize(
    ("pose", "message"),
    [
        ([[0.0, 0.0, 0.0, 1.0]], "shape"),
        (
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, float("nan"), 1.0]],
            "finite",
        ),
        (
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 2.0]],
            "unit normalized",
        ),
        (
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]],
            "canonical quaternion hemisphere",
        ),
        (
            [[0.0, 0.0, 0.0, 1.0], [-1.0, 0.0, 0.0, 0.0]],
            "canonical quaternion hemisphere",
        ),
    ],
)
def test_binding_rejects_invalid_pose(pose: object, message: str) -> None:
    binding = bind_habitat_link_layout(
        _mapping().runtime_joint_order, _blocks(), joint_position_count=8
    )
    with pytest.raises(HabitatMappingError, match=message):
        binding.map_pose(pose)


def test_binding_canonicalizes_signed_zero_in_flat_output() -> None:
    binding = bind_habitat_link_layout(
        _mapping().runtime_joint_order, _blocks(), joint_position_count=8
    )
    output = binding.map_pose([[-0.0, 0.0, -0.0, 1.0], [0.0, -0.0, 0.0, 1.0]])
    assert all(math.copysign(1.0, value) == 1.0 for value in output if value == 0.0)
