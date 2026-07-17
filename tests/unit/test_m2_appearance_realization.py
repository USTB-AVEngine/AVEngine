from __future__ import annotations

import io
from pathlib import Path
import struct
from typing import Any, Sequence

import numpy as np
from PIL import Image
import pytest

from avengine.m2.appearance_realization import (
    AppearanceRealizationError,
    verify_appearance_realization,
)
from avengine.m2.glb_write import build_glb


IDENTITY_QUATERNION = (0.0, 0.0, 0.0, 1.0)
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
GRID_SIZE = 16
TORSO_GIRTH_SCALE = 1.2
HEAD_SCALE = 1.1


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


def _append_blob(document: dict[str, Any], binary: bytearray, payload: bytes) -> int:
    binary.extend(b"\0" * ((-len(binary)) % 4))
    offset = len(binary)
    binary.extend(payload)
    view_index = len(document.setdefault("bufferViews", []))
    document["bufferViews"].append(
        {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
    )
    return view_index


def _png_payload(pixels: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8), mode="RGBA").save(output, format="PNG")
    return output.getvalue()


def _base_color_payload(*, tampered: bool = False) -> bytes:
    pixels = np.empty((64, 64, 4), dtype=np.uint8)
    pixels[:32, :32] = (210, 210, 210, 255)
    pixels[:32, 32:] = (20, 20, 20, 255)
    pixels[32:, :32] = (190, 100, 40, 255)
    pixels[32:, 32:] = (40, 80, 140, 255)
    if tampered:
        pixels[0, 0, :3] = (10, 240, 30)
    return _png_payload(pixels)


def _auxiliary_texture_payload(value: tuple[int, int, int, int]) -> bytes:
    pixels = np.empty((4, 4, 4), dtype=np.uint8)
    pixels[:] = value
    return _png_payload(pixels)


def _source_positions() -> np.ndarray:
    positions = []
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            x = -1.0 + 2.0 * column / (GRID_SIZE - 1)
            y = -0.5 + row / (GRID_SIZE - 1)
            z = 0.08 * np.sin(column * np.pi / (GRID_SIZE - 1))
            positions.append((x, y, z))
    return np.asarray(positions, dtype=np.float32)


def _realized_positions(source: np.ndarray) -> np.ndarray:
    # Reproduce the registered Blender realizer's documented frame and
    # weighted torso/head operations.  Every vertex carries 0.5 weight for
    # each semantic group in this compact fixture.
    source = source.astype(np.float64)
    transformed = np.column_stack((source[:, 0], -source[:, 2], source[:, 1]))
    semantic_weight = np.full(len(source), 0.5, dtype=np.float64)
    center = np.sum(transformed * semantic_weight[:, None], axis=0) / float(
        semantic_weight.sum()
    )
    transformed[:, 1] = center[1] + (transformed[:, 1] - center[1]) * (
        1.0 + (TORSO_GIRTH_SCALE - 1.0) * semantic_weight
    )
    transformed[:, 2] = center[2] + (transformed[:, 2] - center[2]) * (
        1.0 + (TORSO_GIRTH_SCALE - 1.0) * 0.55 * semantic_weight
    )
    center = np.sum(transformed * semantic_weight[:, None], axis=0) / float(
        semantic_weight.sum()
    )
    factor = 1.0 + (HEAD_SCALE - 1.0) * semantic_weight
    transformed = center + (transformed - center) * factor[:, None]
    return np.column_stack(
        (transformed[:, 0], transformed[:, 2], -transformed[:, 1])
    ).astype(np.float32)


def _triangle_indices() -> list[tuple[int]]:
    indices: list[tuple[int]] = []
    for row in range(GRID_SIZE - 1):
        for column in range(GRID_SIZE - 1):
            lower_left = row * GRID_SIZE + column
            lower_right = lower_left + 1
            upper_left = lower_left + GRID_SIZE
            upper_right = upper_left + 1
            indices.extend(
                [
                    (lower_left,),
                    (lower_right,),
                    (upper_right,),
                    (lower_left,),
                    (upper_right,),
                    (upper_left,),
                ]
            )
    return indices


def _appearance_glb(
    *,
    output: bool,
    base_color: bytes,
    geometry_tampered: bool = False,
) -> bytes:
    document: dict[str, Any] = {
        "asset": {
            "version": "2.0",
            "generator": "byte-realization-output"
            if output
            else "byte-realization-source",
        },
        "extensionsUsed": ["KHR_materials_specular"],
        "nodes": [
            {
                "name": "Root",
                "children": [1, 3],
                "translation": [0.0, 0.0, 0.0],
                "rotation": list(IDENTITY_QUATERNION),
                "scale": [1.0, 1.0, 1.0],
            },
            {
                "name": "Torso",
                "children": [2],
                "translation": [0.0, 0.0, 0.0],
                "rotation": list(IDENTITY_QUATERNION),
                "scale": [1.0, 1.0, 1.0],
            },
            {
                "name": "Head",
                "translation": [0.0, 0.0, 0.0],
                "rotation": list(IDENTITY_QUATERNION),
                "scale": [1.0, 1.0, 1.0],
            },
            {
                "name": "AnimalMesh",
                "mesh": 0,
                "skin": 0,
                "translation": [0.0, 0.0, 0.0],
                "rotation": list(IDENTITY_QUATERNION),
                "scale": [1.0, 1.0, 1.0],
            },
        ],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    binary = bytearray()
    source_positions = _source_positions()
    positions = _realized_positions(source_positions) if output else source_positions
    if geometry_tampered:
        positions = positions.copy()
        positions[0, 0] += 0.02
    position = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="VEC3",
        values=positions.tolist(),
        fmt="fff",
    )
    normal = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="VEC3",
        values=[(0.0, 1.0, 0.0)] * len(positions),
        fmt="fff",
    )
    texcoord = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="VEC2",
        values=[
            (column / (GRID_SIZE - 1), row / (GRID_SIZE - 1))
            for row in range(GRID_SIZE)
            for column in range(GRID_SIZE)
        ],
        fmt="ff",
    )
    joints = _append_accessor(
        document,
        binary,
        component_type=5121,
        element_type="VEC4",
        values=[(1, 2, 0, 0)] * len(positions),
        fmt="BBBB",
    )
    weights = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="VEC4",
        values=[(0.5, 0.5, 0.0, 0.0)] * len(positions),
        fmt="ffff",
    )
    indices = _append_accessor(
        document,
        binary,
        component_type=5123,
        element_type="SCALAR",
        values=_triangle_indices(),
        fmt="H",
    )
    inverse_binds = _append_accessor(
        document,
        binary,
        component_type=5126,
        element_type="MAT4",
        values=[IDENTITY_MATRIX] * 3,
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
        values=[(0.0, 0.0, 0.0), (0.05, 0.0, 0.0)],
        fmt="fff",
    )
    base_view = _append_blob(document, binary, base_color)
    normal_payload = _auxiliary_texture_payload((128, 128, 255, 255))
    normal_view = _append_blob(document, binary, normal_payload)
    specular_payload = _auxiliary_texture_payload((64, 64, 64, 255))
    specular_view = _append_blob(document, binary, specular_payload)
    document.update(
        {
            "meshes": [
                {
                    "primitives": [
                        {
                            "attributes": {
                                "POSITION": position,
                                "NORMAL": normal,
                                "TEXCOORD_0": texcoord,
                                "JOINTS_0": joints,
                                "WEIGHTS_0": weights,
                            },
                            "indices": indices,
                            "material": 0,
                            "mode": 4,
                        }
                    ]
                }
            ],
            "skins": [
                {
                    "skeleton": 0,
                    "joints": [0, 1, 2],
                    "inverseBindMatrices": inverse_binds,
                }
            ],
            "animations": [
                {
                    "name": name,
                    "samplers": [
                        {"input": timestamps, "output": translations},
                    ],
                    "channels": [
                        {
                            "sampler": 0,
                            "target": {"node": 1, "path": "translation"},
                        }
                    ],
                }
                for name in ("Idle", "Walking")
            ],
            "images": [
                {"bufferView": base_view, "mimeType": "image/png"},
                {"bufferView": normal_view, "mimeType": "image/png"},
                {"bufferView": specular_view, "mimeType": "image/png"},
            ],
            "textures": [{"source": 0}, {"source": 1}, {"source": 2}],
            "materials": [
                {
                    "pbrMetallicRoughness": {
                        "baseColorTexture": {"index": 0},
                        "metallicFactor": 0.0,
                        "roughnessFactor": 0.72,
                    },
                    "normalTexture": {"index": 1},
                    "extensions": {
                        "KHR_materials_specular": {
                            "specularFactor": 0.25,
                            "specularTexture": {"index": 2},
                        }
                    },
                }
            ],
            "buffers": [{"byteLength": len(binary)}],
        }
    )
    return build_glb(document, binary)


def _request() -> dict[str, Any]:
    return {
        "realization_operations": [
            {
                "attribute": "size",
                "parameters": {"scale_ratio": 1.0},
            },
            {
                "attribute": "body_build",
                "parameters": {
                    "semantic_joint_names": ["Torso"],
                    "torso_girth_scale": TORSO_GIRTH_SCALE,
                },
            },
            {
                "attribute": "coat_profile",
                "parameters": {
                    "luminance_gain": 1.0,
                    "preserve_pattern": "tricolor",
                },
            },
            {
                "attribute": "life_stage",
                "parameters": {
                    "semantic_joint_names": ["Head"],
                    "head_scale": HEAD_SCALE,
                    "coat_desaturation": 0.0,
                    "muzzle_gray_mix": 0.0,
                    "muzzle_gray_target": 0.62,
                },
            },
        ]
    }


def _realization_pair(
    tmp_path: Path,
    *,
    geometry_tampered: bool = False,
    texture_tampered: bool = False,
) -> tuple[Path, Path, bytes, bytes, bytes]:
    source_base = _base_color_payload()
    output_base = _base_color_payload(tampered=texture_tampered)
    source_payload = _appearance_glb(output=False, base_color=source_base)
    output_payload = _appearance_glb(
        output=True,
        base_color=output_base,
        geometry_tampered=geometry_tampered,
    )
    source_path = tmp_path / "source.glb"
    output_path = tmp_path / "output.glb"
    source_path.write_bytes(source_payload)
    output_path.write_bytes(output_payload)
    return source_path, output_path, source_payload, output_payload, output_base


def test_byte_level_realization_verifier_accepts_reconstructed_bytes(
    tmp_path: Path,
) -> None:
    source, output, source_payload, output_payload, texture = _realization_pair(
        tmp_path
    )

    audit = verify_appearance_realization(
        source_path=source,
        output_path=output,
        source_payload=source_payload,
        output_payload=output_payload,
        standalone_texture_payload=texture,
        request=_request(),
    )

    assert audit["shape"]["torso_selected_vertices"] == GRID_SIZE**2
    assert audit["shape"]["head_selected_vertices"] == GRID_SIZE**2
    assert audit["shape"]["maximum_expected_position_error_m"] < 1.0e-6
    assert audit["texture"]["maximum_expected_pixel_channel_error"] == 0.0
    assert audit["texture"]["muzzle_mask_nonzero_pixels"] > 32


def test_byte_level_realization_verifier_uses_snapshotted_payloads(
    tmp_path: Path,
) -> None:
    source, output, source_payload, output_payload, texture = _realization_pair(
        tmp_path
    )
    source.write_bytes(b"raced source path")
    output.write_bytes(b"raced output path")

    audit = verify_appearance_realization(
        source_path=source,
        output_path=output,
        source_payload=source_payload,
        output_payload=output_payload,
        standalone_texture_payload=texture,
        request=_request(),
    )

    assert audit["shape"]["maximum_expected_position_error_m"] < 1.0e-6


def test_byte_level_realization_verifier_rejects_geometry_byte_tamper(
    tmp_path: Path,
) -> None:
    source, output, source_payload, output_payload, texture = _realization_pair(
        tmp_path,
        geometry_tampered=True,
    )

    with pytest.raises(AppearanceRealizationError, match="POSITION"):
        verify_appearance_realization(
            source_path=source,
            output_path=output,
            source_payload=source_payload,
            output_payload=output_payload,
            standalone_texture_payload=texture,
            request=_request(),
        )


def test_byte_level_realization_verifier_rejects_texture_byte_tamper(
    tmp_path: Path,
) -> None:
    source, output, source_payload, output_payload, texture = _realization_pair(
        tmp_path,
        texture_tampered=True,
    )

    with pytest.raises(AppearanceRealizationError, match="base-color pixels"):
        verify_appearance_realization(
            source_path=source,
            output_path=output,
            source_payload=source_payload,
            output_payload=output_payload,
            standalone_texture_payload=texture,
            request=_request(),
        )


def test_byte_level_realization_verifier_rejects_detached_standalone_texture(
    tmp_path: Path,
) -> None:
    source, output, source_payload, output_payload, _texture = _realization_pair(
        tmp_path
    )

    with pytest.raises(
        AppearanceRealizationError,
        match="embedded base-color image bytes differ",
    ):
        verify_appearance_realization(
            source_path=source,
            output_path=output,
            source_payload=source_payload,
            output_payload=output_payload,
            standalone_texture_payload=b"detached",
            request=_request(),
        )
