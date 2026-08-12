from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import pytest

from avengine.m2.glb import load_glb
from avengine.m2.glb_write import build_glb
from avengine.m2.skinning import (
    SkinningCompileCache,
    SkinningError,
    action_time_bounds,
    compile_skinning,
    sample_action_global_matrices,
    sample_action_vertices,
)


IDENTITY_MATRIX = (
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


def _append_accessor(
    document: dict,
    binary: bytearray,
    element_type: str,
    component_type: int,
    values: list[tuple[float | int, ...]],
) -> int:
    component_count = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[element_type]
    component_format = {5121: "B", 5123: "H", 5125: "I", 5126: "f"}[component_type]
    packer = struct.Struct("<" + component_format * component_count)
    offset = len(binary)
    for value in values:
        binary.extend(packer.pack(*value))
    view_index = len(document.setdefault("bufferViews", []))
    document["bufferViews"].append(
        {"buffer": 0, "byteOffset": offset, "byteLength": len(binary) - offset}
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


def _fixture_payload() -> tuple[dict, bytes, bytes]:
    document: dict = {
        "asset": {"version": "2.0", "generator": "actor-envelope-test"},
        "nodes": [
            {"name": "animated_parent", "children": [1, 3]},
            {"name": "skin_root", "children": [2]},
            {"name": "joint", "rotation": [0.0, 0.0, 0.0, 1.0]},
            {"name": "mesh", "mesh": 0, "skin": 0},
        ],
    }
    binary = bytearray()
    inverse_bind = _append_accessor(
        document,
        binary,
        "MAT4",
        5126,
        [IDENTITY_MATRIX, IDENTITY_MATRIX],
    )
    positions = _append_accessor(
        document,
        binary,
        "VEC3",
        5126,
        [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
    )
    joints = _append_accessor(document, binary, "VEC4", 5123, [(1, 0, 0, 0)] * 3)
    weights = _append_accessor(
        document, binary, "VEC4", 5126, [(1.0, 0.0, 0.0, 0.0)] * 3
    )
    timestamps = _append_accessor(document, binary, "SCALAR", 5126, [(0.0,), (1.0,)])
    translations = _append_accessor(
        document,
        binary,
        "VEC3",
        5126,
        [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
    )
    rotations = _append_accessor(
        document,
        binary,
        "VEC4",
        5126,
        [(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 0.0)],
    )
    scales = _append_accessor(
        document,
        binary,
        "VEC3",
        5126,
        [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
    )
    document["skins"] = [
        {
            "name": "actor_skin",
            "skeleton": 1,
            "joints": [1, 2],
            "inverseBindMatrices": inverse_bind,
        }
    ]
    document["meshes"] = [
        {
            "primitives": [
                {
                    "attributes": {
                        "POSITION": positions,
                        "JOINTS_0": joints,
                        "WEIGHTS_0": weights,
                    }
                }
            ]
        }
    ]
    document["animations"] = [
        {
            "name": "Walking",
            "samplers": [
                {
                    "input": timestamps,
                    "output": translations,
                    "interpolation": "LINEAR",
                },
                {
                    "input": timestamps,
                    "output": rotations,
                    "interpolation": "LINEAR",
                },
                {
                    "input": timestamps,
                    "output": scales,
                    "interpolation": "STEP",
                },
            ],
            "channels": [
                {"sampler": 0, "target": {"node": 0, "path": "translation"}},
                {"sampler": 1, "target": {"node": 2, "path": "rotation"}},
                {"sampler": 2, "target": {"node": 1, "path": "scale"}},
            ],
        }
    ]
    document["buffers"] = [{"byteLength": len(binary)}]
    payload = build_glb(document, bytes(binary))
    return document, bytes(binary), payload


def make_document(*, source_path: Path):
    return load_glb(source_path)


def test_full_graph_sampling_includes_animated_non_skin_ancestor_and_slerp(
    tmp_path: Path,
) -> None:
    _, _, payload = _fixture_payload()
    source = tmp_path / "actor.glb"
    source.write_bytes(payload)
    compiled = compile_skinning(make_document(source_path=source))

    assert compiled.joint_node_indices == (1, 2)
    assert len(compiled.nodes) == 4
    assert action_time_bounds(compiled, "Walking") == (0.0, 1.0)
    assert compiled.qualification_state == "planning_only"
    assert compiled.qualification_claim is False
    assert compiled.formal_eligible is False
    assert compiled.primitives[0].positions.flags.writeable is False

    globals_ = sample_action_global_matrices(compiled, "Walking", 0.5)
    np.testing.assert_allclose(globals_[2][:3, 3], [1.0, 0.0, 0.0], atol=1e-12)
    vertices = sample_action_vertices(compiled, "Walking", 0.5)
    np.testing.assert_allclose(
        vertices,
        [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
        atol=1e-7,
    )


def test_step_transform_uses_left_key_until_exact_right_endpoint(
    tmp_path: Path,
) -> None:
    _, _, payload = _fixture_payload()
    source = tmp_path / "actor.glb"
    source.write_bytes(payload)
    compiled = compile_skinning(make_document(source_path=source))

    halfway = sample_action_vertices(compiled, "Walking", 0.5)
    endpoint = sample_action_vertices(compiled, "Walking", 1.0)

    assert np.max(np.abs(halfway[:, 2])) == pytest.approx(1.0)
    assert np.max(np.abs(endpoint[:, 2])) == pytest.approx(2.0)
    with pytest.raises(SkinningError, match="outside action bounds"):
        sample_action_vertices(compiled, "Walking", 1.01)


def test_compile_cache_reuses_one_resolved_source_path(tmp_path: Path) -> None:
    _, _, payload = _fixture_payload()
    source = tmp_path / "actor.glb"
    source.write_bytes(payload)
    cache = SkinningCompileCache()

    first = cache.load(source)
    second = cache.load(source)

    assert first is second
    assert len(cache) == 1
    assert first.source_path == source.resolve()
    assert first.cache_key.source_path == str(source.resolve())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda document: document["meshes"][0]["primitives"][0].update(targets=[]),
            "morph targets",
        ),
        (
            lambda document: document["meshes"][0]["primitives"][0][
                "attributes"
            ].update(
                JOINTS_1=document["meshes"][0]["primitives"][0]["attributes"][
                    "JOINTS_0"
                ]
            ),
            "JOINTS_1",
        ),
        (
            lambda document: document["animations"][0]["samplers"][0].update(
                interpolation="CUBICSPLINE"
            ),
            "CUBICSPLINE",
        ),
    ],
)
def test_compiler_fails_closed_on_unimplemented_deformation_features(
    tmp_path: Path, mutation, message: str
) -> None:
    document, binary, _ = _fixture_payload()
    mutation(document)
    candidate_path = tmp_path / f"candidate_{message.replace(' ', '_')}.glb"
    candidate_path.write_bytes(build_glb(document, binary))
    candidate = load_glb(candidate_path)

    with pytest.raises(SkinningError, match=message):
        compile_skinning(candidate)
