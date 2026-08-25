from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Sequence

import numpy as np
import pytest

from avengine.assets.action_rebind import (
    ActionRebindError,
    ActionRebindResult,
    rebind_compatible_action_set,
    verify_appearance_glb_compatibility,
    write_action_rebind,
)
from avengine.assets.actions import (
    TICKS_PER_SAMPLE,
    TIME_BASE_HZ,
    BakedActionClip,
    BakedActionSet,
    read_baked_actions_npz,
    write_baked_actions_npz,
)
from avengine.assets.glb_write import build_glb


IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _append_accessor(
    document: dict[str, Any],
    binary: bytearray,
    *,
    component_type: int,
    element_type: str,
    values: Sequence[Sequence[int | float]],
    fmt: str,
) -> int:
    binary.extend(b"\0" * ((-len(binary)) % 4))
    offset = len(binary)
    packer = struct.Struct("<" + fmt)
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


def _identity_matrix(*, translation_y: float = 0.0) -> tuple[float, ...]:
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
        translation_y,
        0.0,
        1.0,
    )


def _appearance_glb(
    *, size_scale: float, output: bool, mutation: str | None = None
) -> bytes:
    document: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "appearance-compat-test"},
        "nodes": [
            {
                "name": "Root",
                "children": [1, 2],
                "translation": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
            {
                "name": "Paw",
                "translation": [
                    0.0,
                    (size_scale if output else 1.0)
                    + (0.1 if output and mutation == "rest" else 0.0),
                    0.0,
                ],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
            {
                "name": "Mesh",
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
    position = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="VEC3",
        values=[(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 1.0, 0.0)],
        fmt="fff",
    )
    texcoord_values = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
    if output and mutation == "uv":
        texcoord_values[0] = (0.25, 0.0)
    texcoord = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="VEC2",
        values=texcoord_values,
        fmt="ff",
    )
    joint_values = [(0, 1, 0, 0)] * 3
    if output and mutation == "joints":
        joint_values[0] = (1, 0, 0, 0)
    joints = _append_accessor(
        document,
        binary,
        component_type=5121,
        element_type="VEC4",
        values=joint_values,
        fmt="BBBB",
    )
    weight_values: list[tuple[float, float, float, float]] = [
        (0.75, 0.25, 0.0, 0.0)
    ] * 3
    if output and mutation == "weights":
        weight_values[0] = (
            float(np.nextafter(np.float32(0.75), np.float32(1.0))),
            0.25,
            0.0,
            0.0,
        )
    weights = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="VEC4",
        values=weight_values,
        fmt="ffff",
    )
    indices_values = [(0,), (1,), (2,)]
    if output and mutation == "topology":
        indices_values = [(0,), (2,), (1,)]
    indices = _append_accessor(
        document,
        binary,
        component_type=5123,
        element_type="SCALAR",
        values=indices_values,
        fmt="H",
    )
    inverse_binds = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="MAT4",
        values=[
            _identity_matrix(),
            _identity_matrix(translation_y=-(size_scale if output else 1.0)),
        ],
        fmt="f" * 16,
    )
    timestamps = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="SCALAR",
        values=[(0.0,), (1.0,)],
        fmt="f",
    )
    translations = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="VEC3",
        values=[
            (0.0, size_scale if output else 1.0, 0.0),
            (0.0, 2.0 * size_scale if output else 2.0, 0.0),
        ],
        fmt="fff",
    )
    rotations = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="VEC4",
        values=[IDENTITY, IDENTITY],
        fmt="ffff",
    )
    scales = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="VEC3",
        values=[(1.0, 1.0, 1.0)] * 2,
        fmt="fff",
    )
    document["meshes"] = [
        {
            "primitives": [
                {
                    "attributes": {
                        "POSITION": position,
                        "TEXCOORD_0": texcoord,
                        "JOINTS_0": joints,
                        "WEIGHTS_0": weights,
                    },
                    "indices": indices,
                    "mode": 4,
                }
            ]
        }
    ]
    document["skins"] = [
        {
            "skeleton": 0,
            "joints": [0, 1],
            "inverseBindMatrices": inverse_binds,
        }
    ]
    document["animations"] = []
    for name in ("Idle", "Walking"):
        target = (
            0 if output and mutation == "action_target" and name == "Walking" else 1
        )
        document["animations"].append(
            {
                "name": name,
                "samplers": [
                    {"input": timestamps, "output": translations},
                    {"input": timestamps, "output": rotations},
                    {"input": timestamps, "output": scales},
                ],
                "channels": [
                    {"sampler": 0, "target": {"node": target, "path": "translation"}},
                    {"sampler": 1, "target": {"node": 1, "path": "rotation"}},
                    {"sampler": 2, "target": {"node": 1, "path": "scale"}},
                ],
            }
        )
    document["buffers"] = [{"byteLength": len(binary)}]
    return build_glb(document, binary)


def _appearance_pair(
    tmp_path: Path, *, mutation: str | None = None
) -> tuple[Path, Path]:
    source = tmp_path / "source.glb"
    output = tmp_path / "output.glb"
    source.write_bytes(_appearance_glb(size_scale=1.5, output=False))
    output.write_bytes(_appearance_glb(size_scale=1.5, output=True, mutation=mutation))
    return source, output


def _clip(semantic: str, source_name: str) -> BakedActionClip:
    ticks = (0, TICKS_PER_SAMPLE)
    return BakedActionClip(
        semantic_action_id=semantic,
        source_action_name=source_name,
        clip_start_seconds=0.0,
        clip_end_seconds=2 * TICKS_PER_SAMPLE / TIME_BASE_HZ,
        loop_duration_ticks=2 * TICKS_PER_SAMPLE,
        sample_ticks=ticks,
        source_times_seconds=tuple(tick / TIME_BASE_HZ for tick in ticks),
        rotations_xyzw=((IDENTITY,), (IDENTITY,)),
    )


def _actions(source_sha256: str = "1" * 64) -> BakedActionSet:
    return BakedActionSet(
        source_glb_sha256=source_sha256,
        runtime_joint_order=("spine",),
        actions=(
            _clip("idle", "Idle"),
            _clip("walk", "Walking"),
        ),
    )


def test_rebind_changes_only_visual_identity() -> None:
    source = _actions()

    target = rebind_compatible_action_set(
        source,
        target_glb_sha256="2" * 64,
        target_runtime_joint_order=("spine",),
    )

    assert target.source_glb_sha256 == "2" * 64
    assert target.runtime_joint_order == source.runtime_joint_order
    assert target.actions == source.actions
    for original, rebound in zip(source.actions, target.actions, strict=True):
        np.testing.assert_array_equal(
            np.asarray(original.rotations_xyzw),
            np.asarray(rebound.rotations_xyzw),
        )


def test_rebind_rejects_joint_order_drift() -> None:
    with pytest.raises(ActionRebindError, match="joint order"):
        rebind_compatible_action_set(
            _actions(),
            target_glb_sha256="2" * 64,
            target_runtime_joint_order=("other",),
        )


def test_appearance_compatibility_is_measured_from_glb_bytes(tmp_path: Path) -> None:
    source, output = _appearance_pair(tmp_path)

    audit = verify_appearance_glb_compatibility(
        source,
        output,
        requested_size_scale=1.5,
    )

    assert audit["requested_size_scale"] == 1.5
    assert audit["mesh"]["maximum_texcoord_0_error"] == 0.0
    assert audit["mesh"]["maximum_weights_0_error"] == 0.0
    assert audit["skin"]["maximum_scaled_rest_translation_error_m"] == 0.0
    assert max(audit["actions"]["maximum_errors"].values()) == 0.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("topology", "topology/indices"),
        ("uv", "TEXCOORD_0"),
        ("joints", "JOINTS_0"),
        ("weights", "WEIGHTS_0"),
        ("rest", "skin rest/IBM"),
        ("action_target", "channel target/path"),
    ],
)
def test_appearance_compatibility_rejects_tampered_glb_even_without_report_gates(
    tmp_path: Path, mutation: str, message: str
) -> None:
    source, output = _appearance_pair(tmp_path, mutation=mutation)

    with pytest.raises(ActionRebindError, match=message):
        verify_appearance_glb_compatibility(
            source,
            output,
            requested_size_scale=1.5,
        )


def test_appearance_compatibility_cross_checks_requested_size(tmp_path: Path) -> None:
    source, output = _appearance_pair(tmp_path)

    with pytest.raises(ActionRebindError, match="skin rest/IBM"):
        verify_appearance_glb_compatibility(
            source,
            output,
            requested_size_scale=1.0,
        )


def _result(tmp_path: Path) -> ActionRebindResult:
    actions = _actions("2" * 64)
    temporary = tmp_path / "temporary.npz"
    artifact_sha256 = write_baked_actions_npz(actions, temporary)
    payload = temporary.read_bytes()
    assert artifact_sha256 == hashlib.sha256(payload).hexdigest()
    return ActionRebindResult(
        actions=actions,
        artifact_bytes=payload,
        report={
            "schema": "avengine_m2_action_bake_report_v1",
            "status": "pass",
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "artifact": {
                "path": None,
                "sha256": artifact_sha256,
                "byte_size": len(payload),
                "canonical_content_sha256": artifact_sha256,
                "readback_equal": True,
            },
        },
    )


def test_write_action_rebind_emits_hash_bound_pair(tmp_path: Path) -> None:
    result = _result(tmp_path)
    artifact = tmp_path / "derived" / "actions.npz"
    report = tmp_path / "derived" / "action_bake_report.json"

    emitted_artifact, emitted_report = write_action_rebind(
        result,
        output_npz=artifact,
        report_output=report,
    )

    assert read_baked_actions_npz(emitted_artifact) == result.actions
    value = json.loads(emitted_report.read_text(encoding="utf-8"))
    assert value["artifact"]["path"] == str(artifact.resolve())
    assert (
        value["artifact"]["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    )


def test_write_action_rebind_preflights_both_outputs(tmp_path: Path) -> None:
    result = _result(tmp_path)
    artifact = tmp_path / "derived" / "actions.npz"
    report = tmp_path / "derived" / "action_bake_report.json"
    report.parent.mkdir(parents=True)
    report.write_text("occupied\n", encoding="utf-8")

    with pytest.raises(ActionRebindError, match="refusing to replace"):
        write_action_rebind(
            result,
            output_npz=artifact,
            report_output=report,
        )

    assert not artifact.exists()
    assert report.read_text(encoding="utf-8") == "occupied\n"
