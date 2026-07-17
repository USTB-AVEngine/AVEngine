from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any, Sequence

import pytest

from avengine.m2.glb import extract_actions, extract_skins, load_glb
from avengine.m2.preprocess import GlbPreprocessError, preprocess_glb
from avengine.m2.rebase import rebase_skin_root
from tools.m2 import preprocess_glb as preprocess_glb_cli


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


def _append_accessor(
    document: dict[str, Any],
    binary: bytearray,
    *,
    component_type: int,
    element_type: str,
    values: Sequence[Sequence[float | int]],
) -> int:
    formats = {5121: "B", 5123: "H", 5126: "f"}
    components = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}
    _align(binary)
    offset = len(binary)
    packer = struct.Struct("<" + formats[component_type] * components[element_type])
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


def _identity() -> tuple[float, ...]:
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


def _synthetic_multi_root(*, weighted_controller: bool = False) -> bytes:
    document: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "m2-preprocess-unit-test"},
        "nodes": [
            {"name": "Armature", "children": [1, 3, 5]},
            {"name": "deform_root", "children": [2]},
            {"name": "paw"},
            {"name": "IK_controller", "children": [4]},
            {"name": "IK_controller_end"},
            {"name": "animal_mesh", "mesh": 0, "skin": 0},
        ],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    binary = bytearray()
    positions = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="VEC3",
        values=[(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 1.0, 0.0)],
    )
    joints = _append_accessor(
        document,
        binary,
        component_type=5121,
        element_type="VEC4",
        values=[(1, 2, 0, 0), (0, 2, 0, 0), (1, 3, 0, 0)],
    )
    first_weights = (
        (0.75, 0.25, 0.0, 0.0) if weighted_controller else (1.0, 0.0, 0.0, 0.0)
    )
    weights = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="VEC4",
        values=[first_weights, (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
    )
    inverse_bind = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="MAT4",
        values=[_identity()] * 4,
    )
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
    document["skins"] = [
        {
            "name": "animal_skin",
            "joints": [1, 2, 3, 4],
            "inverseBindMatrices": inverse_bind,
        }
    ]

    timestamps = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="SCALAR",
        values=[(0.0,), (1.0,)],
    )

    def action(name: str, *, dynamic_paw_translation: bool = False) -> dict[str, Any]:
        armature_values = _append_accessor(
            document,
            binary,
            component_type=5126,
            element_type="VEC3",
            values=[(0.0, 0.0, 0.0)] * 2,
        )
        root_values = _append_accessor(
            document,
            binary,
            component_type=5126,
            element_type="VEC3",
            values=[(0.0, 0.0, 0.0)] * 2,
        )
        paw_values = _append_accessor(
            document,
            binary,
            component_type=5126,
            element_type="VEC3" if dynamic_paw_translation else "VEC4",
            values=(
                [(0.0, 0.0, 0.0), (0.0, 0.25, 0.0)]
                if dynamic_paw_translation
                else [(0.0, 0.0, 0.0, 1.0)] * 2
            ),
        )
        controller_values = _append_accessor(
            document,
            binary,
            component_type=5126,
            element_type="VEC3",
            values=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        )
        paths = [
            "translation",
            "translation",
            "translation" if dynamic_paw_translation else "rotation",
            "translation",
        ]
        targets = [0, 1, 2, 3]
        outputs = [armature_values, root_values, paw_values, controller_values]
        return {
            "name": name,
            "samplers": [
                {"input": timestamps, "output": output, "interpolation": "LINEAR"}
                for output in outputs
            ],
            "channels": [
                {"sampler": index, "target": {"node": node, "path": path}}
                for index, (node, path) in enumerate(zip(targets, paths, strict=True))
            ],
        }

    document["animations"] = [
        action("Death", dynamic_paw_translation=True),
        action("Idle"),
        action("Walk"),
    ]
    document["buffers"] = [{"byteLength": len(binary)}]
    return _build_glb(document, bytes(binary))


def _write_fixture(tmp_path: Path, *, weighted_controller: bool = False) -> Path:
    source = tmp_path / "source.glb"
    source.write_bytes(_synthetic_multi_root(weighted_controller=weighted_controller))
    return source


def _cli_arguments(source: Path, output: Path, report: Path) -> list[str]:
    return [
        "--input",
        str(source),
        "--output",
        str(output),
        "--report",
        str(report),
        "--action",
        "Idle=Idle",
        "--action",
        "Walk=Walking",
    ]


def test_preprocess_prunes_only_zero_weight_root_and_filters_actions(
    tmp_path: Path,
) -> None:
    source = _write_fixture(tmp_path)
    output = tmp_path / "prepared.glb"

    report = preprocess_glb(
        source,
        output,
        action_map=[("Idle", "Idle"), ("Walk", "Walking")],
    )

    prepared = load_glb(output)
    skin = extract_skins(prepared)[0]
    assert [joint.name for joint in skin.joints] == ["deform_root", "paw"]
    assert skin.inverse_bind_matrices is not None
    assert len(skin.inverse_bind_matrices) == 2
    assert [action.name for action in extract_actions(prepared)] == ["Idle", "Walking"]
    assert all(
        channel.target_node_name in {"deform_root", "paw"}
        for action in extract_actions(prepared)
        for channel in action.channels
    )
    assert report["actions"]["dropped_source_actions"] == ["Death"]
    assert report["skin"]["input_root_count"] == 2
    assert report["skin"]["output_root_count"] == 1
    assert report["skin"]["output_joint_count"] == 2
    assert report["skin"]["removed_joints"] == [
        {
            "old_ordinal": 2,
            "node_index": 3,
            "name": "IK_controller",
            "maximum_weight": 0.0,
            "nonzero_weight_slots": 0,
            "zero_weight_reference_slots": 2,
        },
        {
            "old_ordinal": 3,
            "node_index": 4,
            "name": "IK_controller_end",
            "maximum_weight": 0.0,
            "nonzero_weight_slots": 0,
            "zero_weight_reference_slots": 1,
        },
    ]
    assert report["primitives"][0]["zero_weight_removed_ordinal_substitutions"] == 3

    rebased = tmp_path / "rebased.glb"
    rebase_report = rebase_skin_root(output, rebased)
    assert rebase_report["status"] == "pass"


def test_preprocess_fails_closed_when_second_root_has_nonzero_weight(
    tmp_path: Path,
) -> None:
    source = _write_fixture(tmp_path, weighted_controller=True)

    with pytest.raises(GlbPreprocessError, match="deleting a weighted branch"):
        preprocess_glb(
            source,
            tmp_path / "prepared.glb",
            action_map=[("Idle", "Idle"), ("Walk", "Walking")],
        )


@pytest.mark.parametrize(
    ("action_map", "message"),
    [
        ([("Missing", "Idle")], "not found"),
        ([("Idle", "Action"), ("Walk", "Action")], "target is ambiguous"),
        ([("Idle", "Idle"), ("Idle", "Walking")], "source is mapped more than once"),
    ],
)
def test_preprocess_requires_unambiguous_exact_action_map(
    tmp_path: Path,
    action_map: list[tuple[str, str]],
    message: str,
) -> None:
    source = _write_fixture(tmp_path)
    with pytest.raises(GlbPreprocessError, match=message):
        preprocess_glb(source, tmp_path / "prepared.glb", action_map=action_map)


def test_preprocess_real_quaternius_cat_strips_four_controller_roots(
    tmp_path: Path,
) -> None:
    source = Path("assets/mesh_library/quaternius_animalpack/Cat.glb")
    report = preprocess_glb(
        source,
        tmp_path / "cat.glb",
        action_map=[("Idle", "Idle"), ("Walking", "Walking")],
    )

    assert report["skin"]["input_root_count"] == 5
    assert report["skin"]["output_root_count"] == 1
    assert report["skin"]["input_joint_count"] == 34
    assert report["skin"]["output_joint_count"] == 26
    assert all(
        joint["maximum_weight"] == 0.0 for joint in report["skin"]["removed_joints"]
    )


def test_preprocess_real_quaternius_horse_keeps_only_idle_and_walk(
    tmp_path: Path,
) -> None:
    source = Path("assets/mesh_library/quaternius_farm/Horse.glb")
    report = preprocess_glb(
        source,
        tmp_path / "horse.glb",
        action_map=[("Idle", "Idle"), ("Walk", "Walking")],
    )

    assert report["actions"]["dropped_source_actions"] == [
        "Death",
        "Jump",
        "Run",
        "WalkSlow",
    ]
    assert report["skin"]["input_root_count"] == 1
    assert report["skin"]["removed_joints"] == []
    assert [
        action.name for action in extract_actions(load_glb(tmp_path / "horse.glb"))
    ] == ["Idle", "Walking"]


def test_preprocess_cli_writes_hash_bound_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write_fixture(tmp_path)
    output = tmp_path / "prepared.glb"
    report_path = tmp_path / "report.json"

    result = preprocess_glb_cli.main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--report",
            str(report_path),
            "--action",
            "Idle=Idle",
            "--action",
            "Walk=Walking",
        ]
    )

    assert result == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = json.loads(capsys.readouterr().out)
    assert report["output"]["path"] == str(output.resolve())
    assert summary["output"]["sha256"] == report["output"]["sha256"]


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_preprocess_library_refuses_existing_or_symlink_output(
    tmp_path: Path, kind: str
) -> None:
    source = _write_fixture(tmp_path)
    output = tmp_path / "prepared.glb"
    if kind == "file":
        output.write_bytes(b"sentinel")
    else:
        output.symlink_to(tmp_path / "dangling-target.glb")

    with pytest.raises(GlbPreprocessError, match="refusing to replace output"):
        preprocess_glb(
            source,
            output,
            action_map=[("Idle", "Idle"), ("Walk", "Walking")],
        )

    if kind == "file":
        assert output.read_bytes() == b"sentinel"
    else:
        assert output.is_symlink()
        assert not (tmp_path / "dangling-target.glb").exists()


@pytest.mark.parametrize("occupied", ["output", "report"])
@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_preprocess_cli_preflights_both_paired_outputs(
    tmp_path: Path, occupied: str, kind: str
) -> None:
    source = _write_fixture(tmp_path)
    output = tmp_path / "prepared.glb"
    report = tmp_path / "report.json"
    path = output if occupied == "output" else report
    if kind == "file":
        path.write_bytes(b"sentinel")
    else:
        path.symlink_to(tmp_path / f"dangling-{occupied}")

    with pytest.raises(SystemExit):
        preprocess_glb_cli.main(_cli_arguments(source, output, report))

    counterpart = report if occupied == "output" else output
    assert not counterpart.exists()
    assert not counterpart.is_symlink()
    if kind == "file":
        assert path.read_bytes() == b"sentinel"
    else:
        assert path.is_symlink()


def test_preprocess_cli_cleans_output_when_report_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_fixture(tmp_path)
    output = tmp_path / "prepared.glb"
    report = tmp_path / "report.json"

    def fail_report(_path: Path, _payload: bytes) -> None:
        raise OSError("injected report failure")

    monkeypatch.setattr(preprocess_glb_cli, "_write_exclusive", fail_report)
    with pytest.raises(SystemExit):
        preprocess_glb_cli.main(_cli_arguments(source, output, report))

    assert not output.exists()
    assert not report.exists()
