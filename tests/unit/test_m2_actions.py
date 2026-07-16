from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import struct
import zipfile

import numpy as np
import pytest

from avengine.m2.actions import (
    ActionBakeError,
    BakedActionClip,
    SAMPLE_RATE_HZ,
    TICKS_PER_SAMPLE,
    TIME_BASE_HZ,
    bake_required_actions,
    baked_actions_content_sha256,
    parse_baked_actions_npz,
    read_baked_actions_npz,
    serialize_baked_actions_npz,
    write_baked_actions_npz,
)
from avengine.m2.glb import parse_glb


JSON_CHUNK_TYPE = 0x4E4F534A
BIN_CHUNK_TYPE = 0x004E4942


def _pad(payload: bytes, byte: bytes) -> bytes:
    return payload + byte * ((-len(payload)) % 4)


def _build_glb(document: dict, binary: bytes) -> bytes:
    document = copy.deepcopy(document)
    document["buffers"] = [{"byteLength": len(binary)}]
    json_payload = _pad(
        json.dumps(document, separators=(",", ":")).encode("utf-8"), b" "
    )
    binary_payload = _pad(binary, b"\0")
    chunks = b"".join(
        [
            struct.pack("<II", len(json_payload), JSON_CHUNK_TYPE),
            json_payload,
            struct.pack("<II", len(binary_payload), BIN_CHUNK_TYPE),
            binary_payload,
        ]
    )
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


def _append_accessor(
    document: dict,
    binary: bytearray,
    element_type: str,
    values: list[tuple[float, ...]],
) -> int:
    components = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[element_type]
    offset = len(binary)
    for value in values:
        assert len(value) == components
        binary.extend(struct.pack("<" + "f" * components, *value))
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


def _fixture(
    *,
    duration_seconds: float = 4.0 / 15.0,
    start_seconds: float = 0.25,
    cubic_rotation: bool = False,
    non_unit_rotation: bool = False,
    dynamic_translation: bool = False,
    dynamic_scale: bool = False,
    dynamic_root_rotation: bool = False,
    small_angle_rotation: bool = False,
) -> tuple[dict, bytes]:
    document: dict = {
        "asset": {"version": "2.0", "generator": "m2-action-unit-test"},
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
                "name": "tail",
                "translation": [0.0, 0.0, 1.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
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
    inverse_bind = _append_accessor(document, binary, "MAT4", [identity_matrix] * 3)
    # Deliberately order tail before paw. Runtime order must preserve the skin
    # list while removing root, rather than sorting names or traversing nodes.
    document["skins"] = [
        {
            "name": "test_skin",
            "skeleton": 0,
            "joints": [0, 2, 1],
            "inverseBindMatrices": inverse_bind,
        }
    ]

    end = start_seconds + duration_seconds
    middle = start_seconds + duration_seconds / 2.0
    endpoint_times = _append_accessor(
        document, binary, "SCALAR", [(start_seconds,), (end,)]
    )
    step_times = _append_accessor(
        document,
        binary,
        "SCALAR",
        [(start_seconds,), (middle,), (end,)],
    )

    defaults = {
        0: {
            "translation": (0.0, 0.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
        },
        1: {
            "translation": (0.0, 1.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
        },
        2: {
            "translation": (0.0, 0.0, 1.0),
            "scale": (1.0, 1.0, 1.0),
        },
    }
    animations: list[dict] = []
    for action_name in ("Idle", "Walking"):
        animation = {"name": action_name, "samplers": [], "channels": []}

        def add_channel(
            node: int,
            path: str,
            values: list[tuple[float, ...]],
            *,
            interpolation: str,
            input_accessor: int = endpoint_times,
        ) -> None:
            output = _append_accessor(
                document,
                binary,
                "VEC4" if path == "rotation" else "VEC3",
                values,
            )
            sampler = len(animation["samplers"])
            animation["samplers"].append(
                {
                    "input": input_accessor,
                    "output": output,
                    "interpolation": interpolation,
                }
            )
            animation["channels"].append(
                {"sampler": sampler, "target": {"node": node, "path": path}}
            )

        root_end = (
            (0.0, 0.0, math.sin(math.pi / 8.0), math.cos(math.pi / 8.0))
            if dynamic_root_rotation and action_name == "Walking"
            else (0.0, 0.0, 0.0, 1.0)
        )
        add_channel(
            0,
            "rotation",
            [(0.0, 0.0, 0.0, 1.0), root_end],
            interpolation="LINEAR",
        )
        if non_unit_rotation and action_name == "Walking":
            paw_end = (0.0, 0.0, 0.0, 2.0)
        elif small_angle_rotation and action_name == "Walking":
            # Negative representative of a small positive rotation.  Its dot
            # with identity exceeds the historical 0.9995 nlerp shortcut.
            paw_end = (
                0.0,
                0.0,
                -math.sin(0.01),
                -math.cos(0.01),
            )
        else:
            paw_end = (
                0.0,
                0.0,
                -math.sin(math.pi / 4.0),
                -math.cos(math.pi / 4.0),
            )
        paw_values = [(0.0, 0.0, 0.0, 1.0), paw_end]
        if cubic_rotation and action_name == "Walking":
            zero = (0.0, 0.0, 0.0, 0.0)
            paw_values = [
                zero,
                (0.0, 0.0, 0.0, 1.0),
                zero,
                zero,
                paw_end,
                zero,
            ]
        add_channel(
            1,
            "rotation",
            paw_values,
            interpolation=(
                "CUBICSPLINE"
                if cubic_rotation and action_name == "Walking"
                else "LINEAR"
            ),
        )
        add_channel(
            2,
            "rotation",
            [
                (0.0, 0.0, 0.0, 1.0),
                (1.0, 0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
            ],
            interpolation="STEP",
            input_accessor=step_times,
        )
        for node in range(3):
            translation = defaults[node]["translation"]
            second_translation = translation
            if dynamic_translation and action_name == "Walking" and node == 1:
                second_translation = (0.25, 1.0, 0.0)
            add_channel(
                node,
                "translation",
                [translation, second_translation],
                interpolation="STEP",
            )
            scale = defaults[node]["scale"]
            second_scale = scale
            if dynamic_scale and action_name == "Walking" and node == 2:
                second_scale = (1.0, 1.1, 1.0)
            add_channel(
                node,
                "scale",
                [scale, second_scale],
                interpolation="STEP",
            )
        animations.append(animation)
    document["animations"] = animations
    return document, bytes(binary)


def _bake_fixture(**kwargs):
    document, binary = _fixture(**kwargs)
    return bake_required_actions(parse_glb(_build_glb(document, binary)))


def test_bake_preserves_rootless_skin_order_and_samples_step_and_shortest_slerp() -> (
    None
):
    baked = _bake_fixture()

    assert baked.sample_rate_hz == SAMPLE_RATE_HZ
    assert baked.time_base_hz == TIME_BASE_HZ
    assert baked.runtime_joint_order == ("tail", "paw")
    assert [
        (action.semantic_action_id, action.source_action_name)
        for action in baked.actions
    ] == [
        ("idle", "Idle"),
        ("walk", "Walking"),
    ]
    walk = baked.action("walk")
    assert walk.clip_start_seconds == pytest.approx(0.25)
    assert walk.clip_end_seconds == pytest.approx(0.25 + 4.0 / 15.0)
    assert walk.loop_duration_ticks == 12_800
    assert walk.sample_ticks == (0, 3_200, 6_400, 9_600)
    assert walk.source_times_seconds == pytest.approx(
        [0.25, 0.25 + 1.0 / 15.0, 0.25 + 2.0 / 15.0, 0.25 + 3.0 / 15.0]
    )
    assert all(
        source_time < walk.clip_end_seconds for source_time in walk.source_times_seconds
    )

    # STEP is right-continuous at its exact middle key.
    assert walk.rotations_xyzw[1][0] == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert walk.rotations_xyzw[2][0] == pytest.approx((1.0, 0.0, 0.0, 0.0))
    # Paw's endpoint is authored as the negative representative of +90deg.
    # The half-cycle sample must follow the short path and remain unit length.
    assert walk.rotations_xyzw[2][1] == pytest.approx(
        (0.0, 0.0, math.sin(math.pi / 8.0), math.cos(math.pi / 8.0)),
        # Source accessors are float32; the bake output is float64, but it
        # cannot recover precision that was not present in the GLB.
        abs=3.0e-8,
    )
    rotations = np.asarray(walk.rotations_xyzw, dtype=np.float64)
    assert rotations.dtype == np.float64
    assert np.linalg.norm(rotations, axis=2) == pytest.approx(1.0, abs=1.0e-12)


def test_four_thirds_second_cycle_has_20_samples_and_no_loop_endpoint() -> None:
    baked = _bake_fixture(duration_seconds=4.0 / 3.0, start_seconds=0.0)

    for action in baked.actions:
        assert action.loop_duration_ticks == 64_000
        assert action.sample_count == 20
        assert action.sample_ticks == tuple(range(0, 64_000, TICKS_PER_SAMPLE))
        assert action.sample_ticks[-1] == 60_800
        assert action.source_times_seconds[-1] == pytest.approx(19.0 / 15.0)
        assert action.source_times_seconds[-1] < action.clip_end_seconds


def test_linear_small_angle_uses_exact_shortest_slerp_not_nlerp() -> None:
    walk = _bake_fixture(small_angle_rotation=True).action("walk")
    fraction = (walk.source_times_seconds[1] - walk.clip_start_seconds) / (
        walk.clip_end_seconds - walk.clip_start_seconds
    )
    # Reconstruct the float32-authored endpoint, then independently evaluate
    # the identity-to-endpoint great-circle rotation at the sampled fraction.
    endpoint_z = float(np.float32(math.sin(0.01)))
    endpoint_w = float(np.float32(math.cos(0.01)))
    endpoint_norm = math.hypot(endpoint_z, endpoint_w)
    endpoint_angle = math.atan2(endpoint_z / endpoint_norm, endpoint_w / endpoint_norm)
    assert walk.rotations_xyzw[1][1] == pytest.approx(
        (
            0.0,
            0.0,
            math.sin(fraction * endpoint_angle),
            math.cos(fraction * endpoint_angle),
        ),
        abs=2.0e-15,
    )


def test_baked_records_are_deeply_immutable_tuples() -> None:
    baked = _bake_fixture()

    with pytest.raises(FrozenInstanceError):
        baked.source_glb_sha256 = "0" * 64  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        baked.action("idle").sample_ticks = ()  # type: ignore[misc]
    assert isinstance(baked.runtime_joint_order, tuple)
    assert isinstance(baked.action("idle").rotations_xyzw[0][0], tuple)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cubic_rotation": True}, "CUBICSPLINE"),
        ({"non_unit_rotation": True}, "unit quaternion"),
        ({"dynamic_translation": True}, "translation must be constant"),
        ({"dynamic_scale": True}, "scale must be constant"),
        ({"dynamic_root_rotation": True}, "root rotation"),
    ],
)
def test_bake_rejects_unsupported_channel_values(kwargs: dict, message: str) -> None:
    with pytest.raises(ActionBakeError, match=message):
        _bake_fixture(**kwargs)


def test_bake_rejects_missing_duplicate_and_non_skin_targets() -> None:
    document, binary = _fixture()
    walking = document["animations"][1]
    walking["channels"] = [
        channel
        for channel in walking["channels"]
        if channel["target"] != {"node": 1, "path": "rotation"}
    ]
    with pytest.raises(ActionBakeError, match="missing rotation targets.*paw"):
        bake_required_actions(parse_glb(_build_glb(document, binary)))

    document, binary = _fixture()
    walking = document["animations"][1]
    walking["channels"].append(copy.deepcopy(walking["channels"][0]))
    with pytest.raises(ActionBakeError, match="duplicate channels"):
        bake_required_actions(parse_glb(_build_glb(document, binary)))

    document, binary = _fixture()
    document["nodes"].append({"name": "camera_helper"})
    walking = document["animations"][1]
    walking["channels"].append(
        {"sampler": 0, "target": {"node": 3, "path": "rotation"}}
    )
    with pytest.raises(ActionBakeError, match="non-skin node"):
        bake_required_actions(parse_glb(_build_glb(document, binary)))


def test_bake_requires_exact_semantic_source_mapping() -> None:
    document, binary = _fixture()
    document["animations"][1]["name"] = "Walk"
    with pytest.raises(ActionBakeError, match="idle->Idle, walk->Walking"):
        bake_required_actions(parse_glb(_build_glb(document, binary)))

    document, binary = _fixture()
    extra = copy.deepcopy(document["animations"][1])
    extra["name"] = "Running"
    document["animations"].append(extra)
    with pytest.raises(ActionBakeError, match="extra=.*Running"):
        bake_required_actions(parse_glb(_build_glb(document, binary)))


def test_npz_is_deterministic_roundtrips_and_has_known_content_hash(
    tmp_path: Path,
) -> None:
    baked = _bake_fixture()
    first = serialize_baked_actions_npz(baked)
    second = serialize_baked_actions_npz(baked)

    assert first == second
    assert parse_baked_actions_npz(first) == baked
    assert baked_actions_content_sha256(baked) == hashlib.sha256(first).hexdigest()
    # Locks metadata canonicalization, npy dtype/layout, member ordering, and
    # fixed ZIP metadata. Update only with an explicit artifact format bump.
    assert baked_actions_content_sha256(baked) == (
        "a92ef156f6d164cda03294cce71027191b8a1771127c3d0373d6631c4268e8ef"
    )

    path_a = tmp_path / "a.npz"
    path_b = tmp_path / "b.npz"
    hash_a = write_baked_actions_npz(baked, path_a)
    hash_b = write_baked_actions_npz(baked, path_b)
    assert hash_a == hash_b == hashlib.sha256(first).hexdigest()
    assert path_a.read_bytes() == path_b.read_bytes() == first
    assert read_baked_actions_npz(path_a) == baked
    with zipfile.ZipFile(path_a) as archive:
        assert all(
            info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist()
        )
        assert all(
            info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()
        )


def test_npz_parser_rejects_noncanonical_member_metadata() -> None:
    baked = _bake_fixture()
    canonical = serialize_baked_actions_npz(baked)
    # Repack exactly the same member payloads with a non-fixed timestamp.
    source = zipfile.ZipFile(BytesIO(canonical))
    output = BytesIO()
    with source, zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as target:
        for info in source.infolist():
            changed = zipfile.ZipInfo(info.filename, date_time=(2026, 7, 16, 0, 0, 0))
            target.writestr(changed, source.read(info.filename))

    with pytest.raises(ActionBakeError, match="not canonically encoded"):
        parse_baked_actions_npz(output.getvalue())


def test_npz_parser_wraps_an_empty_array_member_as_action_bake_error() -> None:
    canonical = serialize_baked_actions_npz(_bake_fixture())
    source = zipfile.ZipFile(BytesIO(canonical))
    output = BytesIO()
    with source, zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as target:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == "idle.sample_ticks.npy":
                payload = b""
            target.writestr(info, payload)

    with pytest.raises(ActionBakeError, match="unable to decode.*sample_ticks"):
        parse_baked_actions_npz(output.getvalue())


def test_action_set_validation_rejects_noncanonical_mutable_or_inconsistent_data() -> (
    None
):
    baked = _bake_fixture()
    original = baked.action("idle")
    endpoint = BakedActionClip(
        semantic_action_id=original.semantic_action_id,
        source_action_name=original.source_action_name,
        clip_start_seconds=original.clip_start_seconds,
        clip_end_seconds=original.clip_end_seconds,
        loop_duration_ticks=original.loop_duration_ticks,
        sample_ticks=original.sample_ticks + (original.loop_duration_ticks,),
        source_times_seconds=original.source_times_seconds
        + (original.clip_end_seconds,),
        rotations_xyzw=original.rotations_xyzw + (original.rotations_xyzw[0],),
    )
    with pytest.raises(ActionBakeError, match="endpoint-exclusive"):
        type(baked)(
            source_glb_sha256=baked.source_glb_sha256,
            runtime_joint_order=baked.runtime_joint_order,
            actions=(endpoint, baked.action("walk")),
        )

    nonunit_frames = list(original.rotations_xyzw)
    first_frame = list(nonunit_frames[0])
    first_frame[0] = (0.0, 0.0, 0.0, 2.0)
    nonunit_frames[0] = tuple(first_frame)
    nonunit = BakedActionClip(
        semantic_action_id=original.semantic_action_id,
        source_action_name=original.source_action_name,
        clip_start_seconds=original.clip_start_seconds,
        clip_end_seconds=original.clip_end_seconds,
        loop_duration_ticks=original.loop_duration_ticks,
        sample_ticks=original.sample_ticks,
        source_times_seconds=original.source_times_seconds,
        rotations_xyzw=tuple(nonunit_frames),
    )
    with pytest.raises(ActionBakeError, match="float64 unit normalized"):
        type(baked)(
            source_glb_sha256=baked.source_glb_sha256,
            runtime_joint_order=baked.runtime_joint_order,
            actions=(nonunit, baked.action("walk")),
        )

    inconsistent_bounds = BakedActionClip(
        semantic_action_id=original.semantic_action_id,
        source_action_name=original.source_action_name,
        clip_start_seconds=original.clip_start_seconds,
        clip_end_seconds=original.clip_end_seconds + 1.0 / TIME_BASE_HZ,
        loop_duration_ticks=original.loop_duration_ticks,
        sample_ticks=original.sample_ticks,
        source_times_seconds=original.source_times_seconds,
        rotations_xyzw=original.rotations_xyzw,
    )
    with pytest.raises(ActionBakeError, match="does not match.*clip bounds"):
        type(baked)(
            source_glb_sha256=baked.source_glb_sha256,
            runtime_joint_order=baked.runtime_joint_order,
            actions=(inconsistent_bounds, baked.action("walk")),
        )

    negative_zero_frames = list(original.rotations_xyzw)
    negative_zero_first_frame = list(negative_zero_frames[0])
    negative_zero_first_frame[0] = (-0.0, 0.0, 0.0, 1.0)
    negative_zero_frames[0] = tuple(negative_zero_first_frame)
    negative_zero = BakedActionClip(
        semantic_action_id=original.semantic_action_id,
        source_action_name=original.source_action_name,
        clip_start_seconds=original.clip_start_seconds,
        clip_end_seconds=original.clip_end_seconds,
        loop_duration_ticks=original.loop_duration_ticks,
        sample_ticks=original.sample_ticks,
        source_times_seconds=original.source_times_seconds,
        rotations_xyzw=tuple(negative_zero_frames),
    )
    with pytest.raises(ActionBakeError, match="canonical positive zero"):
        type(baked)(
            source_glb_sha256=baked.source_glb_sha256,
            runtime_joint_order=baked.runtime_joint_order,
            actions=(negative_zero, baked.action("walk")),
        )

    with pytest.raises(ActionBakeError, match="immutable tuple"):
        type(baked)(
            source_glb_sha256=baked.source_glb_sha256,
            runtime_joint_order=baked.runtime_joint_order,
            actions=[original, baked.action("walk")],  # type: ignore[arg-type]
        )
