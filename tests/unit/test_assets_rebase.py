from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any, Sequence

import numpy as np
import pytest

from avengine.assets.glb import extract_actions, extract_node_hierarchy, load_glb
from avengine.assets.rebase import (
    RebaseError,
    rebase_skin_root,
    rebase_skin_root_preserving_local_tr,
)
from tools.assets import rebase_skin_root as rebase_cli


_JSON_CHUNK_TYPE = 0x4E4F534A
_BIN_CHUNK_TYPE = 0x004E4942


def _pad(payload: bytes, fill: bytes) -> bytes:
    return payload + fill * ((-len(payload)) % 4)


def _build_glb(document: dict[str, Any], binary: bytes) -> bytes:
    json_payload = _pad(
        json.dumps(document, separators=(",", ":")).encode("utf-8"), b" "
    )
    binary_payload = _pad(binary, b"\0")
    chunks = b"".join(
        [
            struct.pack("<II", len(json_payload), _JSON_CHUNK_TYPE),
            json_payload,
            struct.pack("<II", len(binary_payload), _BIN_CHUNK_TYPE),
            binary_payload,
        ]
    )
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


def _align(binary: bytearray) -> None:
    binary.extend(b"\0" * ((-len(binary)) % 4))


def _append_float_accessor(
    document: dict[str, Any],
    binary: bytearray,
    element_type: str,
    values: Sequence[Sequence[float]],
) -> int:
    component_counts = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}
    component_count = component_counts[element_type]
    _align(binary)
    offset = len(binary)
    packer = struct.Struct("<" + "f" * component_count)
    for value in values:
        assert len(value) == component_count
        binary.extend(packer.pack(*value))
    view_index = len(document.setdefault("bufferViews", []))
    document["bufferViews"].append(
        {"buffer": 0, "byteOffset": offset, "byteLength": len(binary) - offset}
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


def _append_joint_accessor(
    document: dict[str, Any],
    binary: bytearray,
    values: Sequence[Sequence[int]],
) -> int:
    _align(binary)
    offset = len(binary)
    packer = struct.Struct("<4B")
    for value in values:
        assert len(value) == 4
        binary.extend(packer.pack(*value))
    view_index = len(document.setdefault("bufferViews", []))
    document["bufferViews"].append(
        {"buffer": 0, "byteOffset": offset, "byteLength": len(binary) - offset}
    )
    accessor_index = len(document.setdefault("accessors", []))
    document["accessors"].append(
        {
            "bufferView": view_index,
            "componentType": 5121,
            "count": len(values),
            "type": "VEC4",
        }
    )
    return accessor_index


def _identity_matrix() -> tuple[float, ...]:
    return (
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


def _child_inverse_bind_matrix() -> tuple[float, ...]:
    # glTF MAT4 accessor components are column-major.  The child bind pose is
    # translated +1 on Y, so its inverse bind matrix translates -1 on Y.
    return (
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
        -1.0,
        0.0,
        1.0,
    )


def _synthetic_skin_fixture(
    *, interpolation: str = "LINEAR"
) -> tuple[dict[str, Any], bytes]:
    document: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "m2-rebase-unit-test"},
        "nodes": [
            {
                "name": "root",
                "children": [1, 2],
                "translation": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
            {
                "name": "paw",
                "translation": [0.0, 1.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
            {
                "name": "dog_mesh",
                "mesh": 0,
                "skin": 0,
                "translation": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
        ],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    binary = bytearray()
    positions = _append_float_accessor(
        document,
        binary,
        "VEC3",
        [(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 1.0, 0.0)],
    )
    joints = _append_joint_accessor(
        document,
        binary,
        [(0, 0, 0, 0), (0, 0, 0, 0), (1, 0, 0, 0)],
    )
    weights = _append_float_accessor(
        document,
        binary,
        "VEC4",
        [(1.0, 0.0, 0.0, 0.0)] * 3,
    )
    inverse_bind_matrices = _append_float_accessor(
        document,
        binary,
        "MAT4",
        [_identity_matrix(), _child_inverse_bind_matrix()],
    )
    timestamps = _append_float_accessor(document, binary, "SCALAR", [(0.0,), (1.0,)])
    translation_values = [(0.0, 1.0, 0.0)] * (
        6 if interpolation == "CUBICSPLINE" else 2
    )
    translations = _append_float_accessor(document, binary, "VEC3", translation_values)
    document["meshes"] = [
        {
            "name": "dog",
            "primitives": [
                {
                    "attributes": {
                        "POSITION": positions,
                        "JOINTS_0": joints,
                        "WEIGHTS_0": weights,
                    }
                }
            ],
        }
    ]
    document["skins"] = [
        {
            "name": "dog_skin",
            "skeleton": 0,
            "joints": [0, 1],
            "inverseBindMatrices": inverse_bind_matrices,
        }
    ]
    document["animations"] = [
        {
            "name": "Idle",
            "samplers": [
                {
                    "input": timestamps,
                    "output": translations,
                    "interpolation": interpolation,
                }
            ],
            "channels": [{"sampler": 0, "target": {"node": 1, "path": "translation"}}],
        }
    ]
    document["buffers"] = [{"byteLength": len(binary)}]
    return document, bytes(binary)


def _accessor_byte_offset(document: dict[str, Any], accessor_index: int) -> int:
    accessor = document["accessors"][accessor_index]
    view = document["bufferViews"][accessor["bufferView"]]
    return view.get("byteOffset", 0) + accessor.get("byteOffset", 0)


def _write_source(
    tmp_path: Path, document: dict[str, Any], binary: bytes
) -> tuple[Path, Path]:
    source = tmp_path / "source.glb"
    output = tmp_path / "rebased.glb"
    source.write_bytes(_build_glb(document, binary))
    return source, output


def test_rebase_minimal_valid_synthetic_skin(tmp_path: Path) -> None:
    document, binary = _synthetic_skin_fixture()
    source, output = _write_source(tmp_path, document, binary)

    report = rebase_skin_root(source, output)

    assert output.is_file()
    assert report["schema"] == "avengine_m2_skin_root_rebase_v1"
    assert report["status"] == "pass"
    assert report["qualification_state"] == "research_candidate"
    assert report["qualification_claim"] is False
    assert report["skin"]["root_joint"] == "root"
    assert report["skin"]["joint_count"] == 2
    assert report["skin"]["maximum_source_bind_frame_consistency_error"] == 0.0
    assert report["skin"]["maximum_output_bind_closure_error"] == 0.0
    assert report["runtime_contract"]["spherical_joint_count"] == 1
    assert report["runtime_contract"]["joint_position_count"] == 4

    rebased = load_glb(output)
    nodes = extract_node_hierarchy(rebased)
    assert nodes[0].local_trs.translation == (0.0, 0.0, 0.0)
    assert nodes[2].local_trs.translation == (0.0, 0.0, 0.0)
    action = extract_actions(rebased)[0]
    assert action.channels[0].values == ((0.0, 1.0, 0.0),) * 2
    assert report["output"]["sha256"] == rebased.sha256


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_rebase_refuses_existing_output_without_replacing_it(
    tmp_path: Path, kind: str
) -> None:
    document, binary = _synthetic_skin_fixture()
    source, output = _write_source(tmp_path, document, binary)
    if kind == "file":
        output.write_bytes(b"sentinel")
    else:
        output.symlink_to(tmp_path / "dangling-rebased.glb")

    with pytest.raises(RebaseError, match="refusing to replace output"):
        rebase_skin_root(source, output)

    if kind == "file":
        assert output.read_bytes() == b"sentinel"
    else:
        assert output.is_symlink()


def _rebase_cli_arguments(
    source: Path,
    output: Path,
    report: Path,
    *,
    preserve_local_tr: bool = False,
) -> list[str]:
    arguments = [
        "--input",
        str(source),
        "--output",
        str(output),
        "--report",
        str(report),
    ]
    if preserve_local_tr:
        arguments.append("--preserve-local-tr")
    return arguments


@pytest.mark.parametrize("occupied", ["output", "report"])
@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_rebase_cli_preflights_both_outputs(
    tmp_path: Path, occupied: str, kind: str
) -> None:
    document, binary = _synthetic_skin_fixture()
    source, output = _write_source(tmp_path, document, binary)
    report = tmp_path / "report.json"
    path = output if occupied == "output" else report
    if kind == "file":
        path.write_bytes(b"sentinel")
    else:
        path.symlink_to(tmp_path / f"dangling-{occupied}")

    with pytest.raises(SystemExit):
        rebase_cli.main(_rebase_cli_arguments(source, output, report))

    counterpart = report if occupied == "output" else output
    assert not counterpart.exists()
    assert not counterpart.is_symlink()
    assert path.is_symlink() if kind == "symlink" else path.read_bytes() == b"sentinel"


def test_rebase_cli_rolls_back_output_when_report_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document, binary = _synthetic_skin_fixture()
    source, output = _write_source(tmp_path, document, binary)
    report = tmp_path / "report.json"

    def fail_report(_path: Path, _payload: bytes) -> None:
        raise OSError("injected report failure")

    monkeypatch.setattr(rebase_cli, "_write_exclusive", fail_report)
    with pytest.raises(SystemExit):
        rebase_cli.main(_rebase_cli_arguments(source, output, report))

    assert not output.exists()
    assert not report.exists()


def test_rebase_rejects_weights_that_do_not_sum_to_one(tmp_path: Path) -> None:
    document, raw_binary = _synthetic_skin_fixture()
    binary = bytearray(raw_binary)
    weights = document["meshes"][0]["primitives"][0]["attributes"]["WEIGHTS_0"]
    struct.pack_into(
        "<4f", binary, _accessor_byte_offset(document, weights), 0.5, 0.0, 0.0, 0.0
    )
    source, output = _write_source(tmp_path, document, bytes(binary))

    with pytest.raises(RebaseError, match="non-negative and sum to one"):
        rebase_skin_root(source, output)


def test_rebase_rejects_joint_ordinal_outside_skin(tmp_path: Path) -> None:
    document, raw_binary = _synthetic_skin_fixture()
    binary = bytearray(raw_binary)
    joints = document["meshes"][0]["primitives"][0]["attributes"]["JOINTS_0"]
    binary[_accessor_byte_offset(document, joints)] = 2
    source, output = _write_source(tmp_path, document, bytes(binary))

    with pytest.raises(RebaseError, match="joint outside the skin"):
        rebase_skin_root(source, output)


def test_rebase_rejects_duplicate_joint_node_names(tmp_path: Path) -> None:
    document, binary = _synthetic_skin_fixture()
    document["nodes"][1]["name"] = "root"
    source, output = _write_source(tmp_path, document, binary)

    with pytest.raises(RebaseError, match="one named joint tree"):
        rebase_skin_root(source, output)


def test_rebase_rejects_skin_not_reachable_from_default_scene(tmp_path: Path) -> None:
    document, binary = _synthetic_skin_fixture()
    document["scenes"][0]["nodes"] = [1]
    source, output = _write_source(tmp_path, document, binary)

    with pytest.raises(RebaseError, match="must be in the default scene"):
        rebase_skin_root(source, output)


def test_rebase_rejects_inconsistent_source_bind_frames(tmp_path: Path) -> None:
    document, raw_binary = _synthetic_skin_fixture()
    binary = bytearray(raw_binary)
    accessor = document["skins"][0]["inverseBindMatrices"]
    second_matrix_y_translation = (
        _accessor_byte_offset(document, accessor) + 16 * 4 + 13 * 4
    )
    struct.pack_into("<f", binary, second_matrix_y_translation, 0.0)
    source, output = _write_source(tmp_path, document, bytes(binary))

    with pytest.raises(RebaseError, match="disagree on the skin bind frame"):
        rebase_skin_root(source, output)


def test_rebase_rejects_dynamic_per_bone_translation(tmp_path: Path) -> None:
    document, raw_binary = _synthetic_skin_fixture()
    binary = bytearray(raw_binary)
    accessor = document["animations"][0]["samplers"][0]["output"]
    second_translation = _accessor_byte_offset(document, accessor) + 3 * 4
    struct.pack_into("<3f", binary, second_translation, 0.25, 1.0, 0.0)
    source, output = _write_source(tmp_path, document, bytes(binary))

    with pytest.raises(RebaseError, match="dynamic/ambiguous translation"):
        rebase_skin_root(source, output)


@pytest.mark.parametrize("interpolation", ["LINEAR", "STEP"])
def test_local_tr_rebase_preserves_dynamic_nonroot_translation_and_unitizes_scale(
    tmp_path: Path, interpolation: str
) -> None:
    document, raw_binary = _synthetic_skin_fixture(interpolation=interpolation)
    document["nodes"][1]["scale"] = [1.00000036, 0.99999976, 1.00000012]
    binary = bytearray(raw_binary)
    accessor = document["animations"][0]["samplers"][0]["output"]
    second_translation = _accessor_byte_offset(document, accessor) + 3 * 4
    struct.pack_into("<3f", binary, second_translation, 0.25, 1.125, -0.5)
    source, output = _write_source(tmp_path, document, bytes(binary))

    report = rebase_skin_root_preserving_local_tr(source, output)

    assert report["schema"] == "avengine_m2_skin_root_rebase_local_tr_v2"
    assert report["skin"]["maximum_joint_scale_normalization_error"] == pytest.approx(
        3.6e-7, abs=3.0e-8
    )
    runtime = report["runtime_contract"]
    assert runtime == {
        "schema": "avengine_m2_local_tr_runtime_v2",
        "base_link": "root",
        "coordinate_layout": "xyz_prismatic_then_xyzw_spherical",
        "runtime_joint_order": ["paw"],
        "translation_driven_joint_ids": ["paw"],
        "translation_semantics": "absolute_child_local_meters",
        "rotation_semantics": "absolute_child_local_xyzw",
        "spherical_joint_count": 1,
        "prismatic_joint_count": 3,
        "joint_position_count": 7,
        "actor_root_transform_source": "actor_from_canonical_root",
        "per_bone_dynamic_translation": True,
        "per_bone_dynamic_scale": False,
    }
    rebased = load_glb(output)
    action = extract_actions(rebased)[0]
    assert action.channels[0].interpolation == interpolation
    assert np.asarray(action.channels[0].values) == pytest.approx(
        np.asarray(((0.0, 1.0, 0.0), (0.25, 1.125, -0.5)))
    )
    assert all(
        node.local_trs.scale == (1.0, 1.0, 1.0)
        for node in extract_node_hierarchy(rebased)[:2]
    )
    assert report["output"]["sha256"] == rebased.sha256


def test_local_tr_rebase_cli_requires_explicit_flag_and_writes_v2_report(
    tmp_path: Path,
) -> None:
    document, raw_binary = _synthetic_skin_fixture()
    binary = bytearray(raw_binary)
    accessor = document["animations"][0]["samplers"][0]["output"]
    second_translation = _accessor_byte_offset(document, accessor) + 3 * 4
    struct.pack_into("<3f", binary, second_translation, 0.25, 1.0, 0.0)
    source, output = _write_source(tmp_path, document, bytes(binary))
    report = tmp_path / "local-tr-report.json"

    assert (
        rebase_cli.main(
            _rebase_cli_arguments(
                source,
                output,
                report,
                preserve_local_tr=True,
            )
        )
        == 0
    )
    assert json.loads(report.read_text())["schema"] == (
        "avengine_m2_skin_root_rebase_local_tr_v2"
    )


@pytest.mark.parametrize(
    ("target_path", "first", "second", "message"),
    [
        ("translation", (0.0, 0.0, 0.0), (0.25, 0.0, 0.0), "skin-root translation"),
        ("scale", (1.0, 1.0, 1.0), (1.0, 1.1, 1.0), "scale is not exporter noise"),
    ],
)
def test_local_tr_rebase_rejects_dynamic_root_and_scale(
    tmp_path: Path,
    target_path: str,
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    message: str,
) -> None:
    document, raw_binary = _synthetic_skin_fixture()
    binary = bytearray(raw_binary)
    channel = document["animations"][0]["channels"][0]
    channel["target"]["path"] = target_path
    if target_path == "translation":
        channel["target"]["node"] = 0
    accessor = document["animations"][0]["samplers"][0]["output"]
    offset = _accessor_byte_offset(document, accessor)
    struct.pack_into("<3f", binary, offset, *first)
    struct.pack_into("<3f", binary, offset + 3 * 4, *second)
    source, output = _write_source(tmp_path, document, bytes(binary))

    with pytest.raises(RebaseError, match=message):
        rebase_skin_root_preserving_local_tr(source, output)


def test_local_tr_rebase_rejects_dynamic_root_rotation(tmp_path: Path) -> None:
    document, raw_binary = _synthetic_skin_fixture()
    binary = bytearray(raw_binary)
    animation = document["animations"][0]
    timestamps = animation["samplers"][0]["input"]
    rotations = _append_float_accessor(
        document,
        binary,
        "VEC4",
        [
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.2588190451, 0.0, 0.9659258263),
        ],
    )
    sampler = len(animation["samplers"])
    animation["samplers"].append(
        {"input": timestamps, "output": rotations, "interpolation": "LINEAR"}
    )
    animation["channels"].append(
        {"sampler": sampler, "target": {"node": 0, "path": "rotation"}}
    )
    document["buffers"][0]["byteLength"] = len(binary)
    source, output = _write_source(tmp_path, document, bytes(binary))

    with pytest.raises(RebaseError, match="skin-root animation is not constant"):
        rebase_skin_root_preserving_local_tr(source, output)


def test_rebase_rejects_cubic_spline_channel(tmp_path: Path) -> None:
    document, binary = _synthetic_skin_fixture(interpolation="CUBICSPLINE")
    source, output = _write_source(tmp_path, document, binary)

    with pytest.raises(RebaseError, match="CUBICSPLINE channels"):
        rebase_skin_root(source, output)

    with pytest.raises(RebaseError, match="CUBICSPLINE channels"):
        rebase_skin_root_preserving_local_tr(source, output)
