#!/usr/bin/env python3
"""Add deterministic zero UV0 accessors when a GLB safely omits them.

This is a research-only compatibility compiler.  It preserves every existing
binary byte and every existing accessor, then appends one all-zero VEC2
accessor per primitive that lacks ``TEXCOORD_0``.  The operation does not make
an asset admissible and refuses any material texture that would sample the
synthesized coordinate set.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.m2.glb import decode_accessor, load_glb, parse_glb
from avengine.m2.glb_write import build_glb


SCHEMA = "avengine_m2_missing_uv0_augmentation_v1"

_CORE_TEXTURE_INFO_FIELDS = (
    ("pbrMetallicRoughness", "baseColorTexture"),
    ("pbrMetallicRoughness", "metallicRoughnessTexture"),
    (None, "normalTexture"),
    (None, "occlusionTexture"),
    (None, "emissiveTexture"),
)

_MATERIAL_EXTENSION_TEXTURE_FIELDS: dict[str, tuple[str, ...]] = {
    "KHR_materials_anisotropy": ("anisotropyTexture",),
    "KHR_materials_clearcoat": (
        "clearcoatTexture",
        "clearcoatRoughnessTexture",
        "clearcoatNormalTexture",
    ),
    "KHR_materials_diffuse_transmission": (
        "diffuseTransmissionTexture",
        "diffuseTransmissionColorTexture",
    ),
    "KHR_materials_dispersion": (),
    "KHR_materials_emissive_strength": (),
    "KHR_materials_ior": (),
    "KHR_materials_iridescence": (
        "iridescenceTexture",
        "iridescenceThicknessTexture",
    ),
    "KHR_materials_pbrSpecularGlossiness": (
        "diffuseTexture",
        "specularGlossinessTexture",
    ),
    "KHR_materials_sheen": ("sheenColorTexture", "sheenRoughnessTexture"),
    "KHR_materials_specular": ("specularTexture", "specularColorTexture"),
    "KHR_materials_transmission": ("transmissionTexture",),
    "KHR_materials_unlit": (),
    "KHR_materials_volume": ("thicknessTexture",),
}


def _objects(value: Any, *, owner: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{owner} must be an array of objects")
    return value


def _unaltered_json(document: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in document.items()
        if key not in {"buffers", "bufferViews", "accessors", "meshes"}
    }


def _index(value: Any, *, owner: str, upper_bound: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{owner} must be a non-negative integer")
    if upper_bound is not None and value >= upper_bound:
        raise ValueError(f"{owner} is out of range")
    return value


def _texture_info_texcoord(value: Any, *, owner: str, texture_count: int) -> int:
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be a textureInfo object")
    _index(value.get("index"), owner=f"{owner}.index", upper_bound=texture_count)
    texcoord = _index(value.get("texCoord", 0), owner=f"{owner}.texCoord")
    extensions = value.get("extensions", {})
    if not isinstance(extensions, dict):
        raise ValueError(f"{owner}.extensions must be an object")
    unknown = sorted(set(extensions) - {"KHR_texture_transform"})
    if unknown:
        raise ValueError(
            f"{owner} contains unknown textureInfo extension(s): {unknown}"
        )
    transform = extensions.get("KHR_texture_transform")
    if transform is not None:
        if not isinstance(transform, dict):
            raise ValueError(
                f"{owner}.extensions.KHR_texture_transform must be an object"
            )
        texcoord = _index(
            transform.get("texCoord", texcoord),
            owner=f"{owner}.extensions.KHR_texture_transform.texCoord",
        )
    return texcoord


def _audit_materials(document: dict[str, Any]) -> None:
    """Reject every material route that could consume synthesized UV0."""

    materials = _objects(document.get("materials", []), owner="materials")
    textures = _objects(document.get("textures", []), owner="textures")
    for material_index, material in enumerate(materials):
        owner = f"materials[{material_index}]"
        pbr = material.get("pbrMetallicRoughness", {})
        if not isinstance(pbr, dict):
            raise ValueError(f"{owner}.pbrMetallicRoughness must be an object")

        texture_infos: list[tuple[str, Any]] = []
        for container_name, field in _CORE_TEXTURE_INFO_FIELDS:
            container = material if container_name is None else pbr
            if field in container:
                prefix = (
                    owner if container_name is None else f"{owner}.{container_name}"
                )
                texture_infos.append((f"{prefix}.{field}", container[field]))

        extensions = material.get("extensions", {})
        if not isinstance(extensions, dict):
            raise ValueError(f"{owner}.extensions must be an object")
        unknown_extensions = sorted(
            set(extensions) - set(_MATERIAL_EXTENSION_TEXTURE_FIELDS)
        )
        if unknown_extensions:
            raise ValueError(
                f"{owner} contains unknown material extension(s): {unknown_extensions}"
            )
        for extension_name, extension in extensions.items():
            extension_owner = f"{owner}.extensions.{extension_name}"
            if not isinstance(extension, dict):
                raise ValueError(f"{extension_owner} must be an object")
            known_texture_fields = _MATERIAL_EXTENSION_TEXTURE_FIELDS[extension_name]
            unexpected_texture_fields = sorted(
                key
                for key in extension
                if key.endswith("Texture") and key not in known_texture_fields
            )
            if unexpected_texture_fields:
                raise ValueError(
                    f"{extension_owner} contains unhandled texture field(s): "
                    f"{unexpected_texture_fields}"
                )
            for field in known_texture_fields:
                if field in extension:
                    texture_infos.append(
                        (f"{extension_owner}.{field}", extension[field])
                    )

        for texture_owner, texture_info in texture_infos:
            texcoord = _texture_info_texcoord(
                texture_info, owner=texture_owner, texture_count=len(textures)
            )
            if texcoord == 0:
                raise ValueError(
                    "refusing to synthesize TEXCOORD_0 because a material texture "
                    f"would consume it: {texture_owner}"
                )


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise


def augment(source_path: Path, output_path: Path) -> dict[str, Any]:
    source = load_glb(source_path)
    document = copy.deepcopy(source.json)
    buffers = _objects(document.get("buffers"), owner="buffers")
    views = _objects(document.get("bufferViews"), owner="bufferViews")
    accessors = _objects(document.get("accessors"), owner="accessors")
    meshes = _objects(document.get("meshes"), owner="meshes")
    if len(buffers) != 1 or buffers[0].get("uri") is not None:
        raise ValueError("input must use one embedded GLB buffer")
    _audit_materials(document)
    materials = _objects(document.get("materials", []), owner="materials")

    binary = bytearray(source.binary)
    additions: list[dict[str, Any]] = []
    for mesh_index, mesh in enumerate(meshes):
        primitives = _objects(
            mesh.get("primitives"), owner=f"meshes[{mesh_index}].primitives"
        )
        for primitive_index, primitive in enumerate(primitives):
            attributes = primitive.get("attributes")
            if not isinstance(attributes, dict):
                raise ValueError("primitive attributes must be an object")
            if "TEXCOORD_0" in attributes:
                continue
            if "material" in primitive:
                _index(
                    primitive["material"],
                    owner=f"meshes[{mesh_index}].primitives[{primitive_index}].material",
                    upper_bound=len(materials),
                )
            position_index = attributes.get("POSITION")
            if not isinstance(position_index, int):
                raise ValueError("primitive lacks a POSITION accessor")
            position = decode_accessor(source, position_index)
            count = position.count
            offset = len(binary)
            binary.extend(b"\0" * (count * 2 * 4))
            view_index = len(views)
            views.append(
                {
                    "buffer": 0,
                    "byteOffset": offset,
                    "byteLength": count * 2 * 4,
                }
            )
            accessor_index = len(accessors)
            accessors.append(
                {
                    "bufferView": view_index,
                    "componentType": 5126,
                    "count": count,
                    "type": "VEC2",
                    "min": [0.0, 0.0],
                    "max": [0.0, 0.0],
                }
            )
            attributes["TEXCOORD_0"] = accessor_index
            additions.append(
                {
                    "mesh_index": mesh_index,
                    "primitive_index": primitive_index,
                    "vertex_count": count,
                    "buffer_view_index": view_index,
                    "accessor_index": accessor_index,
                }
            )
    if not additions:
        raise ValueError("input has no missing TEXCOORD_0 attributes")
    buffers[0]["byteLength"] = len(binary)

    payload = build_glb(document, binary)
    readback = parse_glb(payload)
    if readback.binary[: len(source.binary)] != source.binary:
        raise ValueError("existing binary prefix changed")
    if _unaltered_json(readback.json) != _unaltered_json(source.json):
        raise ValueError("unrelated glTF JSON changed")
    for addition in additions:
        primitive = readback.json["meshes"][addition["mesh_index"]]["primitives"][
            addition["primitive_index"]
        ]
        accessor_index = primitive["attributes"]["TEXCOORD_0"]
        uv = np.asarray(
            decode_accessor(readback, accessor_index).values, dtype=np.float64
        )
        if uv.shape != (addition["vertex_count"], 2) or np.any(uv != 0.0):
            raise ValueError("synthesized UV accessor failed exact readback")
    _write_exclusive(output_path, payload)

    return {
        "schema": SCHEMA,
        "status": "pass",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "source": {
            "path": str(source_path),
            "sha256": source.sha256,
            "byte_size": source.byte_length,
        },
        "output": {
            "path": str(output_path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_size": len(payload),
        },
        "policy": {
            "added_semantic": "TEXCOORD_0",
            "value": [0.0, 0.0],
            "texture_using_synthesized_texcoord_0_allowed": False,
            "existing_binary_prefix_preserved": True,
            "existing_accessors_preserved": True,
        },
        "additions": additions,
        "source_binary_sha256": hashlib.sha256(source.binary).hexdigest(),
        "output_existing_binary_prefix_sha256": hashlib.sha256(
            readback.binary[: len(source.binary)]
        ).hexdigest(),
        "unaltered_json_sha256": canonical_json_sha256(_unaltered_json(source.json)),
        "notes": [
            "No audited core or known-extension material textureInfo consumes TEXCOORD_0.",
            "Zero UV0 is compatibility metadata and does not alter positions, weights, skinning, or animation.",
            "This augmentation does not resolve motion-projection equivalence or qualify the asset.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    for label, path in (("output", args.output), ("report", args.report)):
        if path.exists() or path.is_symlink():
            parser.error(f"refusing to replace {label}: {path}")
    source = args.input.resolve()
    output = args.output.resolve()
    report = args.report.resolve()
    if len({source, output, report}) != 3:
        parser.error("input, output, and report must differ")
    output_created = False
    try:
        value = augment(source, output)
        output_created = True
        value["report_content_sha256"] = canonical_json_sha256(value)
        _write_exclusive(
            report,
            (
                json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            ).encode(),
        )
    except (OSError, ValueError) as exc:
        message = str(exc)
        if output_created:
            try:
                output.unlink()
            except OSError as cleanup_exc:
                message += f"; failed to clean newly created output: {cleanup_exc}"
        parser.error(message)
    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(output),
                "output_sha256": sha256_file(output),
                "report": str(report),
                "report_sha256": sha256_file(report),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
