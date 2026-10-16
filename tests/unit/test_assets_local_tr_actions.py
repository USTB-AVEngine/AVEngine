from __future__ import annotations

import copy
from io import BytesIO
import json
import math
from pathlib import Path
import struct
from typing import Any, Sequence
import zipfile

import numpy as np
import pytest

from avengine.assets.glb import load_glb, parse_glb
from avengine.assets.glb_write import build_glb
from avengine.assets.local_tr_actions import (
    LOCAL_TR_ACTIONS_NPZ_SCHEMA,
    SAMPLE_RATE_HZ,
    TICKS_PER_SAMPLE,
    TIME_BASE_HZ,
    LocalTRActionBakeError,
    bake_local_tr_actions,
    local_tr_actions_content_sha256,
    parse_local_tr_actions_npz,
    read_local_tr_actions_npz,
    serialize_local_tr_actions_npz,
    write_local_tr_actions_npz,
)
from avengine.assets.preprocess import preprocess_glb
from avengine.assets.similarity import bake_uniform_skin_ancestor_scale
from avengine.assets.timing import retime_glb_actions
from tools.assets import wrap_uniform_scene_scale


def _append_accessor(
    document: dict[str, Any],
    binary: bytearray,
    *,
    element_type: str,
    values: Sequence[Sequence[float]],
) -> int:
    component_count = {
        "SCALAR": 1,
        "VEC3": 3,
        "VEC4": 4,
        "MAT4": 16,
    }[element_type]
    binary.extend(b"\0" * ((-len(binary)) % 4))
    offset = len(binary)
    packer = struct.Struct("<" + "f" * component_count)
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
            "componentType": 5126,
            "count": len(values),
            "type": element_type,
        }
    )
    return accessor_index


def _cubic_values(values: Sequence[Sequence[float]]) -> list[tuple[float, ...]]:
    zero = tuple(0.0 for _ in values[0])
    expanded: list[tuple[float, ...]] = []
    for value in values:
        expanded.extend((zero, tuple(value), zero))
    return expanded


def _fixture(
    *,
    dynamic_root_translation: bool = False,
    dynamic_root_rotation: bool = False,
    dynamic_scale: bool = False,
    cubic_path: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    document: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "local-tr-v2-unit-test"},
        "nodes": [
            {
                "name": "Root",
                "children": [1, 2, 3],
                "translation": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
            {
                "name": "Body",
                "translation": [0.0, 2.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
            {
                "name": "Foot",
                "translation": [0.2, -1.0, 0.3],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
            {
                "name": "Tail",
                "translation": [-0.4, 0.5, -1.25],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
        ],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
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
        document,
        binary,
        element_type="MAT4",
        values=[identity_matrix] * 4,
    )
    # Root removal must preserve this deliberately non-node-order skin order.
    document["skins"] = [
        {
            "name": "SyntheticLocalTRSkin",
            "skeleton": 0,
            "joints": [0, 2, 1, 3],
            "inverseBindMatrices": inverse_bind,
        }
    ]
    start = 0.25
    end = start + 4.0 / 15.0
    middle = start + 2.0 / 15.0
    endpoint_times = _append_accessor(
        document,
        binary,
        element_type="SCALAR",
        values=[(start,), (end,)],
    )
    step_times = _append_accessor(
        document,
        binary,
        element_type="SCALAR",
        values=[(start,), (middle,), (end,)],
    )

    rest_translations = {
        0: (0.0, 0.0, 0.0),
        1: (0.0, 2.0, 0.0),
        2: (0.2, -1.0, 0.3),
        3: (-0.4, 0.5, -1.25),
    }
    animations: list[dict[str, Any]] = []
    for action_name in ("Idle", "Walking"):
        animation: dict[str, Any] = {
            "name": action_name,
            "samplers": [],
            "channels": [],
        }

        def add_channel(
            node: int,
            path: str,
            values: Sequence[Sequence[float]],
            *,
            interpolation: str = "LINEAR",
            times: int = endpoint_times,
        ) -> None:
            encoded_values = (
                _cubic_values(values) if interpolation == "CUBICSPLINE" else values
            )
            output = _append_accessor(
                document,
                binary,
                element_type="VEC4" if path == "rotation" else "VEC3",
                values=encoded_values,
            )
            sampler_index = len(animation["samplers"])
            animation["samplers"].append(
                {
                    "input": times,
                    "output": output,
                    "interpolation": interpolation,
                }
            )
            animation["channels"].append(
                {"sampler": sampler_index, "target": {"node": node, "path": path}}
            )

        for node in range(4):
            rotation_end = (0.0, 0.0, 0.0, 1.0)
            if action_name == "Walking" and node == 0 and dynamic_root_rotation:
                rotation_end = (
                    0.0,
                    math.sin(math.pi / 12.0),
                    0.0,
                    math.cos(math.pi / 12.0),
                )
            rotation_interpolation = (
                "CUBICSPLINE"
                if action_name == "Walking" and node == 1 and cubic_path == "rotation"
                else "LINEAR"
            )
            add_channel(
                node,
                "rotation",
                [(0.0, 0.0, 0.0, 1.0), rotation_end],
                interpolation=rotation_interpolation,
            )

            scale_end = (1.0, 1.0, 1.0)
            if action_name == "Walking" and node == 1 and dynamic_scale:
                scale_end = (1.0, 1.1, 1.0)
            scale_interpolation = (
                "CUBICSPLINE"
                if action_name == "Walking" and node == 1 and cubic_path == "scale"
                else "STEP"
            )
            add_channel(
                node,
                "scale",
                [(1.0, 1.0, 1.0), scale_end],
                interpolation=scale_interpolation,
            )

        root_end = rest_translations[0]
        if action_name == "Walking" and dynamic_root_translation:
            root_end = (0.25, 0.0, 0.0)
        add_channel(
            0,
            "translation",
            [rest_translations[0], root_end],
            interpolation="LINEAR",
        )
        if action_name == "Walking":
            body_interpolation = (
                "CUBICSPLINE" if cubic_path == "translation" else "LINEAR"
            )
            add_channel(
                1,
                "translation",
                [rest_translations[1], (0.0, 2.4, 0.0)],
                interpolation=body_interpolation,
            )
            add_channel(
                2,
                "translation",
                [
                    rest_translations[2],
                    (0.2, -0.6, 0.8),
                    rest_translations[2],
                ],
                interpolation="STEP",
                times=step_times,
            )
        else:
            add_channel(
                1,
                "translation",
                [rest_translations[1], rest_translations[1]],
                interpolation="LINEAR",
            )
            add_channel(
                2,
                "translation",
                [rest_translations[2], rest_translations[2]],
                interpolation="STEP",
            )
        # Tail intentionally has no translation channel: rest fallback is part
        # of the local-TR contract, not an implicit zero.
        animations.append(animation)
    document["animations"] = animations
    document["buffers"] = [{"byteLength": len(binary)}]
    return document, bytes(binary)


def _bake_fixture(**kwargs: Any):
    document, binary = _fixture(**kwargs)
    return bake_local_tr_actions(parse_glb(build_glb(document, binary)))


def test_bake_samples_absolute_child_local_translation_and_rotation() -> None:
    baked = _bake_fixture()

    assert baked.sample_rate_hz == SAMPLE_RATE_HZ
    assert baked.time_base_hz == TIME_BASE_HZ
    assert TICKS_PER_SAMPLE == 3_200
    assert baked.runtime_joint_order == ("Foot", "Body", "Tail")
    assert np.asarray(baked.rest_translations_m) == pytest.approx(
        np.asarray(((0.2, -1.0, 0.3), (0.0, 2.0, 0.0), (-0.4, 0.5, -1.25)))
    )
    assert baked.translation_driven_joint_ids == ("Foot", "Body")

    walk = baked.action("walk")
    assert walk.sample_ticks == (0, 3_200, 6_400, 9_600)
    assert walk.loop_duration_ticks == 12_800
    # LINEAR Body bob is sampled at exact quarter-cycle points.
    assert [frame[1][1] for frame in walk.translations_m] == pytest.approx(
        (2.0, 2.1, 2.2, 2.3), abs=1.0e-7
    )
    # STEP is right-continuous at the exact middle key.
    assert walk.translations_m[1][0] == pytest.approx((0.2, -1.0, 0.3))
    assert walk.translations_m[2][0] == pytest.approx((0.2, -0.6, 0.8))
    # A missing translation channel means the authored non-zero rest value.
    assert all(
        frame[2] == pytest.approx((-0.4, 0.5, -1.25)) for frame in walk.translations_m
    )
    rotations = np.asarray(walk.rotations_xyzw, dtype=np.float64)
    assert rotations.shape == (4, 3, 4)
    assert np.linalg.norm(rotations, axis=2) == pytest.approx(1.0, abs=1.0e-12)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dynamic_root_translation": True}, "Root translation.*static"),
        ({"dynamic_root_rotation": True}, "root.*rotation.*static"),
        ({"dynamic_scale": True}, "scale must be static"),
        ({"cubic_path": "translation"}, "CUBICSPLINE"),
        ({"cubic_path": "rotation"}, "CUBICSPLINE"),
        ({"cubic_path": "scale"}, "CUBICSPLINE"),
    ],
)
def test_bake_rejects_root_motion_scale_animation_and_cubic(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(LocalTRActionBakeError, match=message):
        _bake_fixture(**kwargs)


def test_npz_is_deterministic_roundtrips_and_binds_content(
    tmp_path: Path,
) -> None:
    baked = _bake_fixture()
    first = serialize_local_tr_actions_npz(baked)
    second = serialize_local_tr_actions_npz(baked)

    assert first == second
    assert parse_local_tr_actions_npz(first) == baked
    digest = local_tr_actions_content_sha256(baked)
    assert digest == __import__("hashlib").sha256(first).hexdigest()

    path_a = tmp_path / "a.npz"
    path_b = tmp_path / "b.npz"
    assert write_local_tr_actions_npz(baked, path_a) == digest
    assert write_local_tr_actions_npz(baked, path_b) == digest
    assert path_a.read_bytes() == path_b.read_bytes() == first
    assert read_local_tr_actions_npz(path_a) == baked
    with zipfile.ZipFile(path_a) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        assert metadata["schema"] == LOCAL_TR_ACTIONS_NPZ_SCHEMA
        assert metadata["translation_driven_joint_ids"] == ["Foot", "Body"]
        assert archive.namelist()[:2] == [
            "metadata.json",
            "rest_translations_m.npy",
        ]


def _canonical_repack(source_payload: bytes, replacements: dict[str, bytes]) -> bytes:
    output = BytesIO()
    with (
        zipfile.ZipFile(BytesIO(source_payload)) as source,
        zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as target,
    ):
        for source_info in source.infolist():
            info = zipfile.ZipInfo(source_info.filename, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            target.writestr(
                info,
                replacements.get(
                    source_info.filename, source.read(source_info.filename)
                ),
            )
    return output.getvalue()


def test_npz_parser_rejects_translation_contract_tamper() -> None:
    canonical = serialize_local_tr_actions_npz(_bake_fixture())
    with zipfile.ZipFile(BytesIO(canonical)) as archive:
        metadata = json.loads(archive.read("metadata.json"))
    metadata["translation_driven_joint_ids"] = []
    changed_metadata = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    tampered = _canonical_repack(canonical, {"metadata.json": changed_metadata})

    with pytest.raises(LocalTRActionBakeError, match="non-driven joint"):
        parse_local_tr_actions_npz(tampered)


def test_npz_parser_rejects_noncanonical_zip_metadata() -> None:
    canonical = serialize_local_tr_actions_npz(_bake_fixture())
    output = BytesIO()
    with (
        zipfile.ZipFile(BytesIO(canonical)) as source,
        zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as target,
    ):
        for source_info in source.infolist():
            changed = copy.copy(source_info)
            changed.date_time = (2026, 7, 17, 0, 0, 0)
            target.writestr(changed, source.read(source_info.filename))

    with pytest.raises(LocalTRActionBakeError, match="not canonically encoded"):
        parse_local_tr_actions_npz(output.getvalue())


def test_real_horse_timing_normalized_and_scaled_sources_keep_body_and_feet(
    tmp_path: Path,
) -> None:
    source = Path("assets/mesh_library/quaternius_farm/Horse.glb")
    prepared = tmp_path / "horse-prepared.glb"
    timing_normalized = tmp_path / "horse-timing-normalized.glb"
    wrapped = tmp_path / "horse-scale-wrapped.glb"
    scaled = tmp_path / "horse-scaled.glb"
    preprocess_glb(
        source,
        prepared,
        action_map=[("Idle", "Idle"), ("Walk", "Walking")],
    )
    retime_glb_actions(
        prepared,
        timing_normalized,
        durations_seconds={"Idle": 6.2, "Walking": 8.0 / 3.0},
    )
    wrap_uniform_scene_scale.wrap(timing_normalized, wrapped, 0.2)
    bake_uniform_skin_ancestor_scale(wrapped, scaled)

    normalized_actions = bake_local_tr_actions(load_glb(timing_normalized))
    scaled_actions = bake_local_tr_actions(load_glb(scaled))
    for actions in (normalized_actions, scaled_actions):
        driven = set(actions.translation_driven_joint_ids)
        assert {"Body", "FrontFoot.R", "BackFoot.R", "FrontFoot.L", "BackFoot.L"} <= (
            driven
        )
        body_index = actions.runtime_joint_order.index("Body")
        foot_index = actions.runtime_joint_order.index("FrontFoot.R")
        walk = actions.action("walk")
        body = np.asarray(walk.translations_m)[:, body_index]
        foot = np.asarray(walk.translations_m)[:, foot_index]
        assert float(np.ptp(body, axis=0).max()) > 1.0e-3
        assert float(np.ptp(foot, axis=0).max()) > 1.0e-2

    normalized = np.asarray(normalized_actions.action("walk").translations_m)
    scaled_values = np.asarray(scaled_actions.action("walk").translations_m)
    nonzero = np.abs(normalized) > 1.0e-6
    assert scaled_values[nonzero] == pytest.approx(
        0.2 * normalized[nonzero], abs=2.0e-6
    )
