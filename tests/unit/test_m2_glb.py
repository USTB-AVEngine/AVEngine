from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct

import pytest

from avengine.m2.glb import (
    GlbError,
    decode_accessor,
    extract_actions,
    extract_node_hierarchy,
    extract_skins,
    load_glb,
    parse_glb,
)


JSON_CHUNK_TYPE = 0x4E4F534A
BIN_CHUNK_TYPE = 0x004E4942


def _pad(payload: bytes, byte: bytes) -> bytes:
    return payload + byte * ((-len(payload)) % 4)


def _build_glb(
    document: dict,
    binary: bytes = b"",
    *,
    raw_json: bytes | None = None,
    second_chunk_type: int = BIN_CHUNK_TYPE,
) -> bytes:
    json_payload = _pad(
        raw_json
        if raw_json is not None
        else json.dumps(document, separators=(",", ":")).encode("utf-8"),
        b" ",
    )
    chunks = struct.pack("<II", len(json_payload), JSON_CHUNK_TYPE) + json_payload
    if binary:
        binary_payload = _pad(binary, b"\x00")
        chunks += struct.pack("<II", len(binary_payload), second_chunk_type)
        chunks += binary_payload
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


def _append_float_accessor(
    document: dict,
    binary: bytearray,
    element_type: str,
    values: list[tuple[float, ...]],
) -> int:
    component_counts = {
        "SCALAR": 1,
        "VEC3": 3,
        "VEC4": 4,
        "MAT4": 16,
    }
    component_count = component_counts[element_type]
    offset = len(binary)
    for value in values:
        assert len(value) == component_count
        binary.extend(struct.pack("<" + "f" * component_count, *value))
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
            "componentType": 5126,
            "count": len(values),
            "type": element_type,
        }
    )
    return accessor_index


def _animated_skin_fixture() -> tuple[dict, bytes]:
    document: dict = {
        "asset": {"version": "2.0", "generator": "unit-test"},
        "nodes": [
            {
                "name": "root",
                "children": [1],
                "translation": [1.0, 2.0, 3.0],
            },
            {
                "name": "paw",
                "translation": [0.0, -0.5, 0.25],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
        ],
    }
    binary = bytearray()
    identity = (
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
    inverse_bind = _append_float_accessor(
        document, binary, "MAT4", [identity, identity]
    )
    timestamps = _append_float_accessor(
        document, binary, "SCALAR", [(0.0,), (0.5,), (1.0,)]
    )
    translations = _append_float_accessor(
        document,
        binary,
        "VEC3",
        [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)],
    )
    rotations = _append_float_accessor(
        document,
        binary,
        "VEC4",
        [
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.70710677, 0.70710677),
            (0.0, 0.0, 0.0, 1.0),
        ],
    )
    document["skins"] = [
        {
            "name": "dog_skin",
            "skeleton": 0,
            "joints": [0, 1],
            "inverseBindMatrices": inverse_bind,
        }
    ]
    document["animations"] = [
        {
            "name": "Walking",
            "samplers": [
                {"input": timestamps, "output": translations},
                {
                    "input": timestamps,
                    "output": rotations,
                    "interpolation": "STEP",
                },
            ],
            "channels": [
                {"sampler": 0, "target": {"node": 0, "path": "translation"}},
                {"sampler": 1, "target": {"node": 1, "path": "rotation"}},
            ],
        }
    ]
    document["buffers"] = [{"byteLength": len(binary)}]
    return document, bytes(binary)


def test_load_glb_extracts_sha_skin_hierarchy_trs_and_action(tmp_path: Path) -> None:
    source_json, binary = _animated_skin_fixture()
    payload = _build_glb(source_json, binary)
    path = tmp_path / "dog.glb"
    path.write_bytes(payload)

    document = load_glb(path)

    assert document.source_path == path.resolve()
    assert document.byte_length == len(payload)
    assert document.sha256 == hashlib.sha256(payload).hexdigest()
    assert document.binary[: len(binary)] == binary

    nodes = extract_node_hierarchy(document)
    assert [node.name for node in nodes] == ["root", "paw"]
    assert nodes[0].parent_node_index is None
    assert nodes[0].children_node_indices == (1,)
    assert nodes[0].local_trs.translation == (1.0, 2.0, 3.0)
    assert nodes[0].local_trs.rotation_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert nodes[0].local_trs.scale == (1.0, 1.0, 1.0)

    skins = extract_skins(document)
    assert len(skins) == 1
    assert skins[0].name == "dog_skin"
    assert skins[0].skeleton_node_index == 0
    assert len(skins[0].inverse_bind_matrices or ()) == 2
    assert [joint.node_index for joint in skins[0].joints] == [0, 1]
    assert skins[0].joints[1].parent_joint_node_index == 0
    assert skins[0].joints[0].child_joint_node_indices == (1,)

    actions = extract_actions(document)
    assert len(actions) == 1
    assert actions[0].name == "Walking"
    assert actions[0].duration_seconds == 1.0
    assert [channel.target_path for channel in actions[0].channels] == [
        "translation",
        "rotation",
    ]
    assert actions[0].channels[0].timestamps_seconds == (0.0, 0.5, 1.0)
    assert actions[0].channels[0].values[-1] == (1.0, 0.0, 0.0)
    assert actions[0].channels[1].interpolation == "STEP"


def test_decode_accessor_honors_float_interleaved_stride() -> None:
    binary = struct.pack("<ffffffff", 1.0, 2.0, 3.0, 99.0, 4.0, 5.0, 6.0, 88.0)
    source_json = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteLength": len(binary), "byteStride": 16}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 2, "type": "VEC3"}
        ],
    }

    decoded = decode_accessor(parse_glb(_build_glb(source_json, binary)), 0)

    assert decoded.element_type == "VEC3"
    assert decoded.values == ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))


def test_decode_accessor_cannot_read_glb_padding_as_buffer_data() -> None:
    binary = b"\x00" * 16
    source_json = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": 13}],
        "bufferViews": [{"buffer": 0, "byteOffset": 12, "byteLength": 4}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 1, "type": "SCALAR"}
        ],
    }
    document = parse_glb(_build_glb(source_json, binary))

    with pytest.raises(GlbError, match=r"beyond buffers\[0\].byteLength"):
        decode_accessor(document, 0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda accessor: accessor.update(sparse={"count": 0}), "sparse"),
        (lambda accessor: accessor.update(componentType=5123), "expected FLOAT"),
        (lambda accessor: accessor.update(type="NOT_A_TYPE"), "type is unsupported"),
        (lambda accessor: accessor.update(byteOffset=10_000), "bufferView"),
    ],
)
def test_decode_accessor_rejects_unsupported_or_out_of_bounds_storage(
    mutation, message: str
) -> None:
    source_json, binary = _animated_skin_fixture()
    mutation(source_json["accessors"][1])
    document = parse_glb(_build_glb(source_json, binary))

    with pytest.raises(GlbError, match=message):
        decode_accessor(document, 1)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.__setitem__(0, ord("x")), "magic"),
        (lambda data: struct.pack_into("<I", data, 4, 1), "container version"),
        (
            lambda data: struct.pack_into("<I", data, 8, len(data) - 4),
            "declared length",
        ),
    ],
)
def test_parse_glb_rejects_invalid_container_header(mutate, message: str) -> None:
    payload = bytearray(_build_glb({"asset": {"version": "2.0"}}))
    mutate(payload)

    with pytest.raises(GlbError, match=message):
        parse_glb(payload)


def test_parse_glb_rejects_unknown_second_chunk() -> None:
    payload = _build_glb(
        {"asset": {"version": "2.0"}, "buffers": [{"byteLength": 4}]},
        b"data",
        second_chunk_type=0x12345678,
    )

    with pytest.raises(GlbError, match="second GLB chunk must be BIN"):
        parse_glb(payload)


def test_parse_glb_rejects_duplicate_json_keys() -> None:
    payload = _build_glb(
        {}, raw_json=b'{"asset":{"version":"2.0"},"asset":{"version":"2.0"}}'
    )

    with pytest.raises(GlbError, match="duplicate JSON object key"):
        parse_glb(payload)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (
            {
                "asset": {"version": "2.0"},
                "extensionsRequired": ["KHR_example"],
            },
            "required glTF extensions are unsupported",
        ),
        (
            {
                "asset": {"version": "2.0"},
                "buffers": [{"byteLength": 4, "uri": "external.bin"}],
            },
            "external or data-URI buffers",
        ),
    ],
)
def test_parse_glb_fails_closed_on_required_extensions_and_external_buffers(
    document: dict, message: str
) -> None:
    binary = b"data" if document.get("buffers") else b""

    with pytest.raises(GlbError, match=message):
        parse_glb(_build_glb(document, binary))


@pytest.mark.parametrize(
    ("nodes", "message"),
    [
        ([{"children": [2]}, {"children": [2]}, {}], "multiple parents"),
        ([{"children": [1]}, {"children": [0]}], "cycle"),
        ([{"matrix": [1.0] * 16}], "explicit TRS required"),
        (
            [{"matrix": [1.0] * 16, "translation": [0.0, 0.0, 0.0]}],
            "both matrix and TRS",
        ),
    ],
)
def test_node_hierarchy_fails_closed_on_ambiguous_or_unsupported_nodes(
    nodes: list[dict], message: str
) -> None:
    document = parse_glb(_build_glb({"asset": {"version": "2.0"}, "nodes": nodes}))

    with pytest.raises(GlbError, match=message):
        extract_node_hierarchy(document)


def test_skin_rejects_duplicate_joint_and_invalid_skeleton_root() -> None:
    duplicate = {
        "asset": {"version": "2.0"},
        "nodes": [{"children": [1]}, {}],
        "skins": [{"joints": [0, 0]}],
    }
    with pytest.raises(GlbError, match="duplicate node"):
        extract_skins(parse_glb(_build_glb(duplicate)))

    wrong_root = {
        "asset": {"version": "2.0"},
        "nodes": [{}, {}],
        "skins": [{"skeleton": 0, "joints": [1]}],
    }
    with pytest.raises(GlbError, match="not an ancestor"):
        extract_skins(parse_glb(_build_glb(wrong_root)))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda animation: animation["channels"][0]["target"].update(path="weights"),
            "target.path is unsupported",
        ),
        (
            lambda animation: animation["channels"].append(
                copy.deepcopy(animation["channels"][0])
            ),
            "duplicate channels",
        ),
        (
            lambda animation: animation["samplers"][0].update(interpolation="BEZIER"),
            "interpolation is unsupported",
        ),
    ],
)
def test_action_extraction_rejects_unsupported_or_ambiguous_channels(
    mutate, message: str
) -> None:
    source_json, binary = _animated_skin_fixture()
    mutate(source_json["animations"][0])

    with pytest.raises(GlbError, match=message):
        extract_actions(parse_glb(_build_glb(source_json, binary)))


def test_action_extraction_rejects_duplicate_names_and_non_monotonic_time() -> None:
    source_json, binary = _animated_skin_fixture()
    source_json["animations"].append(copy.deepcopy(source_json["animations"][0]))
    with pytest.raises(GlbError, match="duplicated"):
        extract_actions(parse_glb(_build_glb(source_json, binary)))

    source_json, binary = _animated_skin_fixture()
    timestamp_view = source_json["bufferViews"][1]
    offset = timestamp_view["byteOffset"]
    edited_binary = bytearray(binary)
    struct.pack_into("<fff", edited_binary, offset, 0.0, 0.5, 0.5)
    with pytest.raises(GlbError, match="strictly increasing"):
        extract_actions(parse_glb(_build_glb(source_json, bytes(edited_binary))))


def test_cubic_spline_requires_three_output_elements_per_timestamp() -> None:
    source_json, binary = _animated_skin_fixture()
    source_json["animations"][0]["samplers"][0]["interpolation"] = "CUBICSPLINE"

    with pytest.raises(GlbError, match="does not match CUBICSPLINE input count"):
        extract_actions(parse_glb(_build_glb(source_json, binary)))


def test_document_json_mutation_cannot_change_hash_bound_extracts() -> None:
    source_json, binary = _animated_skin_fixture()
    document = parse_glb(_build_glb(source_json, binary))
    original_hash = document.sha256
    original_actions = extract_actions(document)

    detached = document.json
    detached["animations"] = []
    detached["nodes"][0]["name"] = "tampered"

    assert document.sha256 == original_hash
    assert extract_actions(document) == original_actions
    assert document.json["nodes"][0].get("name") != "tampered"
