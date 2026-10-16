from __future__ import annotations

import math
import struct

import pytest

from avengine.assets.actions import (
    BakedActionClip,
    BakedActionSet,
    TICKS_PER_SAMPLE,
    TIME_BASE_HZ,
    baked_actions_content_sha256,
)
from avengine.assets.glb import AnimationAction, AnimationChannel, GlbDocument
from avengine.assets.habitat import HabitatAssetMapping, HabitatJointRest
from avengine.assets.qa import M2QaError, audit_m2_candidate
import avengine.assets.qa as qa_module


SOURCE_SHA256 = "ab" * 32
IDENTITY = (0.0, 0.0, 0.0, 1.0)
CLIP_END_SECONDS = TICKS_PER_SAMPLE / TIME_BASE_HZ


def _append_accessor(
    document: dict,
    binary: bytearray,
    element_type: str,
    component_type: int,
    values: list[tuple[float | int, ...]],
) -> int:
    components = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT4": 16,
    }[element_type]
    formats = {5126: "f", 5123: "H"}
    offset = len(binary)
    packer = struct.Struct("<" + formats[component_type] * components)
    for value in values:
        binary.extend(packer.pack(*value))
    view_index = len(document.setdefault("bufferViews", []))
    document["bufferViews"].append(
        {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(binary) - offset,
        }
    )
    accessor_index = len(document.setdefault("accessors", []))
    document["accessors"].append(
        {
            "bufferView": view_index,
            "componentType": component_type,
            "count": len(values),
            "type": element_type,
        }
    )
    return accessor_index


def _document() -> GlbDocument:
    document: dict = {
        "asset": {"version": "2.0"},
        "nodes": [
            {
                "name": "root",
                "children": [1],
                "translation": [0.0, 0.0, 0.0],
                "rotation": list(IDENTITY),
                "scale": [1.0, 1.0, 1.0],
            },
            {
                "name": "paw",
                "translation": [0.0, 0.0, 0.0],
                "rotation": list(IDENTITY),
                "scale": [1.0, 1.0, 1.0],
            },
        ],
    }
    binary = bytearray()
    identity_matrix = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    inverse_bind = _append_accessor(
        document, binary, "MAT4", 5126, [identity_matrix, identity_matrix]
    )
    positions = _append_accessor(
        document,
        binary,
        "VEC3",
        5126,
        [(-0.1, -0.1, 0.0), (0.1, -0.1, 0.0), (0.0, 0.1, 0.0)],
    )
    normals = _append_accessor(
        document,
        binary,
        "VEC3",
        5126,
        [(0.0, 0.0, 1.0)] * 3,
    )
    texcoords = _append_accessor(
        document,
        binary,
        "VEC2",
        5126,
        [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)],
    )
    joints = _append_accessor(
        document,
        binary,
        "VEC4",
        5123,
        [(1, 0, 0, 0)] * 3,
    )
    weights = _append_accessor(
        document,
        binary,
        "VEC4",
        5126,
        [(1.0, 0.0, 0.0, 0.0)] * 3,
    )
    indices = _append_accessor(
        document,
        binary,
        "SCALAR",
        5123,
        [(0,), (1,), (2,)],
    )
    document["skins"] = [
        {
            "skeleton": 0,
            "joints": [0, 1],
            "inverseBindMatrices": inverse_bind,
        }
    ]
    document["meshes"] = [
        {
            "primitives": [
                {
                    "attributes": {
                        "POSITION": positions,
                        "NORMAL": normals,
                        "TEXCOORD_0": texcoords,
                        "JOINTS_0": joints,
                        "WEIGHTS_0": weights,
                    },
                    "indices": indices,
                }
            ]
        }
    ]
    document["buffers"] = [{"byteLength": len(binary)}]
    return GlbDocument(
        json=document,
        binary=bytes(binary),
        sha256=SOURCE_SHA256,
        byte_length=len(binary),
    )


def _mapping() -> HabitatAssetMapping:
    joints = (
        HabitatJointRest(
            joint_ordinal=0,
            node_index=0,
            joint_id="root",
            parent_joint_id=None,
            local_translation_m=(0.0, 0.0, 0.0),
            rest_rotation_xyzw=IDENTITY,
            local_scale=(1.0, 1.0, 1.0),
        ),
        HabitatJointRest(
            joint_ordinal=1,
            node_index=1,
            joint_id="paw",
            parent_joint_id="root",
            local_translation_m=(0.0, 0.0, 0.0),
            rest_rotation_xyzw=IDENTITY,
            local_scale=(1.0, 1.0, 1.0),
        ),
    )
    return HabitatAssetMapping(
        source_glb_sha256=SOURCE_SHA256,
        root_joint_id="root",
        joint_order=("root", "paw"),
        runtime_joint_order=("paw",),
        joints=joints,
        actor_from_skin_root=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        actor_from_skin_root_source="test",
    )


def _baked_actions() -> BakedActionSet:
    clips = tuple(
        BakedActionClip(
            semantic_action_id=semantic_id,
            source_action_name=source_name,
            clip_start_seconds=0.0,
            clip_end_seconds=CLIP_END_SECONDS,
            loop_duration_ticks=TICKS_PER_SAMPLE,
            sample_ticks=(0,),
            source_times_seconds=(0.0,),
            rotations_xyzw=((IDENTITY,),),
        )
        for semantic_id, source_name in (("idle", "Idle"), ("walk", "Walking"))
    )
    return BakedActionSet(
        source_glb_sha256=SOURCE_SHA256,
        runtime_joint_order=("paw",),
        actions=clips,
    )


def _channel(
    channel_index: int,
    node_index: int,
    node_name: str,
    path: str,
    values: tuple[tuple[float, ...], ...],
) -> AnimationChannel:
    return AnimationChannel(
        channel_index=channel_index,
        sampler_index=channel_index,
        target_node_index=node_index,
        target_node_name=node_name,
        target_path=path,
        interpolation="LINEAR",
        input_accessor_index=0,
        output_accessor_index=0,
        timestamps_seconds=(0.0, CLIP_END_SECONDS),
        values=values,
    )


def _source_actions(
    *,
    walking_end_rotation: tuple[float, float, float, float] = IDENTITY,
    walking_end_translation: tuple[float, float, float] | None = None,
) -> tuple[AnimationAction, ...]:
    result = []
    for action_index, action_name in enumerate(("Idle", "Walking")):
        paw_end = walking_end_rotation if action_name == "Walking" else IDENTITY
        channels = [
            _channel(0, 0, "root", "rotation", (IDENTITY, IDENTITY)),
            _channel(1, 1, "paw", "rotation", (IDENTITY, paw_end)),
        ]
        if action_name == "Walking" and walking_end_translation is not None:
            channels.append(
                _channel(
                    2,
                    1,
                    "paw",
                    "translation",
                    ((0.0, 0.0, 0.0), walking_end_translation),
                )
            )
        result.append(
            AnimationAction(
                animation_index=action_index,
                name=action_name,
                duration_seconds=CLIP_END_SECONDS,
                channels=tuple(channels),
            )
        )
    return tuple(result)


SEMANTIC_JOINT_MAP = {
    "body": "root",
    "head": "paw",
    "muzzle": "paw",
    "paw_front_left": "paw",
    "paw_front_right": "paw",
    "paw_hind_left": "paw",
    "paw_hind_right": "paw",
}


def _audit(monkeypatch: pytest.MonkeyPatch, source_actions):
    monkeypatch.setattr(qa_module, "extract_actions", lambda document: source_actions)
    return audit_m2_candidate(
        _document(),
        _baked_actions(),
        _mapping(),
        semantic_joint_map=SEMANTIC_JOINT_MAP,
    )


def test_real_source_endpoints_are_measured_and_candidate_stays_unqualified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # q and -q are the same endpoint orientation and must close without a fake
    # comparison against the first baked frame.
    result = _audit(
        monkeypatch,
        _source_actions(walking_end_rotation=(0.0, 0.0, 0.0, -1.0)),
    )

    walk = next(
        item
        for item in result.deformation["actions"]
        if item["semantic_action_id"] == "walk"
    )
    assert walk["source_loop_endpoint_vertex_error_m"] == pytest.approx(0.0)
    assert walk["source_loop_endpoint_maximum_joint_rotation_error"] == pytest.approx(
        0.0
    )
    assert "declared_loop_boundary_vertex_error_m" not in walk
    assert result.static_geometry["qualification_state"] == "research_candidate"
    assert result.deformation["qualification_state"] == "research_candidate"
    assert result.animation["qualification_state"] == "research_candidate"
    expected_actions_sha256 = baked_actions_content_sha256(_baked_actions())
    assert result.deformation["baked_actions_sha256"] == expected_actions_sha256
    assert result.animation["baked_actions_sha256"] == expected_actions_sha256
    assert "actions_content_sha256" not in result.deformation
    assert result.deformation["qualification_claim"] is False
    assert result.animation["human_visual_review_required"] is True
    summary = result.animation["semantic_terminal_motion"]["walking_summary"]
    assert summary["legacy_hind_gait_metric_triggered"] is False
    assert result.animation["known_limitations"] == []
    assert not any(
        "Known gait limitations remain visible" in note
        for note in result.animation["notes"]
    )


def test_measured_legacy_hind_gait_keeps_the_existing_limitations() -> None:
    limitations = qa_module._legacy_hind_gait_limitations(True)
    assert len(limitations) == 2
    assert "Known legacy gait limitation carried forward" in limitations[0]
    assert "much less hind-paw forward excursion" in limitations[1]


@pytest.mark.parametrize(
    "source_actions",
    [
        _source_actions(walking_end_rotation=(0.0, 0.0, math.sin(0.1), math.cos(0.1))),
        _source_actions(walking_end_translation=(0.01, 0.0, 0.0)),
    ],
)
def test_open_source_glb_endpoint_fails_instead_of_self_comparison(
    monkeypatch: pytest.MonkeyPatch,
    source_actions: tuple[AnimationAction, ...],
) -> None:
    with pytest.raises(
        M2QaError,
        match=r"source GLB action 'Walking' does not close at its true loop endpoint",
    ):
        _audit(monkeypatch, source_actions)
