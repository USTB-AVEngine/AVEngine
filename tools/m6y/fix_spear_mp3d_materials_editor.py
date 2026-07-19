"""Repair and verify MP3D glTF color semantics inside an isolated UE project.

Run this file through UnrealEditor ``-run=pythonscript``.  The original MP3D
GLB uses each of its 23 JPEG textures twice: as an sRGB base-color image and as
a linear red-channel occlusion map.  UE Interchange imported one linear
Texture2D for both slots, which makes the base color pale and desaturated.

Repair mode preserves the imported linear texture for occlusion, duplicates it
as ``*_basecolor_srgb``, enables sRGB on that duplicate, and rebinds the two
material parameters to their semantically correct texture views.  Verify mode
is read-only and is intended for a second fresh editor process.

Environment variables:

``AVENGINE_MP3D_SOURCE_GLB``
    Immutable raw MP3D GLB used to establish the source material contract.
``AVENGINE_MP3D_MATERIAL_RESULT``
    External JSON result path.
``AVENGINE_MP3D_MATERIAL_MODE``
    ``repair`` or ``verify``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import Any, Mapping

import unreal


SCHEMA = "avengine_optional_spear_mp3d_material_color_v1"
CONTENT_ROOT = "/Game/MyAssets/Audioset/Scenes/mp3d_17DRP5sb8fy"
EXPECTED_COUNT = 23
BASE_COLOR_PARAMETER = "BaseColorTexture"
OCCLUSION_PARAMETER = "OcclusionTexture"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_contract(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if len(payload) < 20:
        raise RuntimeError(f"MP3D GLB is incomplete: {path}")
    magic, version, declared_length = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(payload):
        raise RuntimeError("MP3D source must be a complete GLB 2.0 container")
    json_length, json_type = struct.unpack_from("<II", payload, 12)
    if json_type != 0x4E4F534A or 20 + json_length > len(payload):
        raise RuntimeError("MP3D GLB lacks its JSON chunk")
    document = json.loads(
        payload[20 : 20 + json_length].rstrip(b" \t\r\n\x00").decode("utf-8")
    )
    materials = document.get("materials")
    textures = document.get("textures")
    images = document.get("images")
    if not all(isinstance(value, list) for value in (materials, textures, images)):
        raise RuntimeError("MP3D GLB material arrays are incomplete")
    if not (len(materials) == len(textures) == len(images) == EXPECTED_COUNT):
        raise RuntimeError("MP3D GLB must contain exactly 23 materials/textures/images")

    records = []
    for index, material in enumerate(materials):
        if not isinstance(material, Mapping):
            raise RuntimeError(f"MP3D material {index} is invalid")
        pbr = material.get("pbrMetallicRoughness")
        base = pbr.get("baseColorTexture") if isinstance(pbr, Mapping) else None
        occlusion = material.get("occlusionTexture")
        if (
            not isinstance(base, Mapping)
            or not isinstance(occlusion, Mapping)
            or base.get("index") != index
            or occlusion.get("index") != index
            or pbr.get("metallicRoughnessTexture") is not None
            or material.get("normalTexture") is not None
            or material.get("emissiveTexture") is not None
        ):
            raise RuntimeError(
                f"MP3D material {index} does not share one texture between "
                "base-color and occlusion as expected"
            )
        texture = textures[index]
        if not isinstance(texture, Mapping) or texture.get("source") != index:
            raise RuntimeError(f"MP3D texture {index} source mapping differs")
        image = images[index]
        image_name = image.get("name") if isinstance(image, Mapping) else None
        material_name = material.get("name")
        if not isinstance(image_name, str) or not isinstance(material_name, str):
            raise RuntimeError(f"MP3D material/image {index} lacks a stable name")
        records.append(
            {
                "source_material_index": index,
                "source_material_name": material_name,
                "source_texture_index": index,
                "source_image_index": index,
                "source_image_name": image_name,
                "base_color_texture_index": index,
                "occlusion_texture_index": index,
                "shared_source_texture": True,
            }
        )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "material_count": EXPECTED_COUNT,
        "texture_count": EXPECTED_COUNT,
        "image_count": EXPECTED_COUNT,
        "base_color_reference_count": EXPECTED_COUNT,
        "occlusion_reference_count": EXPECTED_COUNT,
        "shared_base_color_and_occlusion_texture_count": EXPECTED_COUNT,
        "other_texture_reference_count": 0,
        "records": records,
    }


def _package_path(value: Any) -> str | None:
    if value is None:
        return None
    text = value.get_path_name() if hasattr(value, "get_path_name") else str(value)
    return text.split(".", 1)[0]


def _asset_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _load_asset(path: str, expected_type: type) -> Any:
    value = unreal.EditorAssetLibrary.load_asset(path)
    if value is None or not isinstance(value, expected_type):
        raise RuntimeError(f"cannot load expected {expected_type.__name__}: {path}")
    return value


def _explicit_texture_parameters(material: Any) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for record in material.get_editor_property("texture_parameter_values"):
        info = record.get_editor_property("parameter_info")
        name = str(info.get_editor_property("name"))
        result[name] = _package_path(record.get_editor_property("parameter_value"))
    return result


def _save_asset(asset: Any) -> None:
    if not unreal.EditorAssetLibrary.save_loaded_asset(asset, only_if_is_dirty=False):
        raise RuntimeError(f"could not save UE asset: {_package_path(asset)}")


def _repair(source: Mapping[str, Any]) -> dict[str, int]:
    created = 0
    rebound = 0
    for record in source["records"]:
        imported_texture_path = (
            f"{CONTENT_ROOT}/{_asset_name(record['source_image_name'])}"
        )
        base_color_path = f"{imported_texture_path}_basecolor_srgb"
        material_path = f"{CONTENT_ROOT}/{_asset_name(record['source_material_name'])}"
        occlusion_texture = _load_asset(imported_texture_path, unreal.Texture2D)
        if unreal.EditorAssetLibrary.does_asset_exist(base_color_path):
            base_color_texture = _load_asset(base_color_path, unreal.Texture2D)
        else:
            base_color_texture = unreal.EditorAssetLibrary.duplicate_asset(
                imported_texture_path, base_color_path
            )
            if base_color_texture is None or not isinstance(
                base_color_texture, unreal.Texture2D
            ):
                raise RuntimeError(
                    f"could not duplicate base-color texture: {base_color_path}"
                )
            created += 1

        occlusion_texture.set_editor_property("srgb", False)
        base_color_texture.set_editor_property("srgb", True)
        _save_asset(occlusion_texture)
        _save_asset(base_color_texture)

        material = _load_asset(material_path, unreal.MaterialInstanceConstant)
        unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(
            material, BASE_COLOR_PARAMETER, base_color_texture
        )
        unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(
            material, OCCLUSION_PARAMETER, occlusion_texture
        )
        unreal.MaterialEditingLibrary.update_material_instance(material)
        _save_asset(material)
        rebound += 1
    unreal.AssetRegistryHelpers.get_asset_registry().wait_for_completion()
    return {
        "created_base_color_texture_count": created,
        "rebound_material_count": rebound,
    }


def _readback(source: Mapping[str, Any]) -> dict[str, Any]:
    base_color_textures = []
    occlusion_textures = []
    bindings = []
    for record in source["records"]:
        imported_texture_path = (
            f"{CONTENT_ROOT}/{_asset_name(record['source_image_name'])}"
        )
        base_color_path = f"{imported_texture_path}_basecolor_srgb"
        material_path = f"{CONTENT_ROOT}/{_asset_name(record['source_material_name'])}"
        occlusion_texture = _load_asset(imported_texture_path, unreal.Texture2D)
        base_color_texture = _load_asset(base_color_path, unreal.Texture2D)
        material = _load_asset(material_path, unreal.MaterialInstanceConstant)
        parameters = _explicit_texture_parameters(material)
        non_null = {name: path for name, path in parameters.items() if path is not None}
        expected = {
            BASE_COLOR_PARAMETER: base_color_path,
            OCCLUSION_PARAMETER: imported_texture_path,
        }
        if non_null != expected:
            raise RuntimeError(
                f"MP3D material texture bindings differ for {material_path}: {non_null}"
            )
        base_srgb = bool(base_color_texture.get_editor_property("srgb"))
        occlusion_srgb = bool(occlusion_texture.get_editor_property("srgb"))
        if not base_srgb or occlusion_srgb:
            raise RuntimeError(
                f"MP3D texture color-space split differs for {material_path}: "
                f"base_srgb={base_srgb} occlusion_srgb={occlusion_srgb}"
            )
        base_color_textures.append(
            {
                "source_texture_index": record["source_texture_index"],
                "texture_path": base_color_path,
                "srgb": base_srgb,
                "semantic": "base_color_srgb",
            }
        )
        occlusion_textures.append(
            {
                "source_texture_index": record["source_texture_index"],
                "texture_path": imported_texture_path,
                "srgb": occlusion_srgb,
                "semantic": "occlusion_linear_red_channel",
            }
        )
        bindings.append(
            {
                "source_material_index": record["source_material_index"],
                "source_material_name": record["source_material_name"],
                "material_path": material_path,
                "base_color_parameter_name": BASE_COLOR_PARAMETER,
                "base_color_texture_path": non_null[BASE_COLOR_PARAMETER],
                "occlusion_parameter_name": OCCLUSION_PARAMETER,
                "occlusion_texture_path": non_null[OCCLUSION_PARAMETER],
                "unexpected_bound_texture_parameters": [],
            }
        )

    return {
        "counts": {
            "source_texture_count": EXPECTED_COUNT,
            "material_count": len(bindings),
            "base_color_texture_count": len(base_color_textures),
            "base_color_binding_count": len(bindings),
            "base_color_srgb_true_count": sum(
                item["srgb"] for item in base_color_textures
            ),
            "occlusion_texture_count": len(occlusion_textures),
            "occlusion_binding_count": len(bindings),
            "occlusion_srgb_false_count": sum(
                not item["srgb"] for item in occlusion_textures
            ),
            "unexpected_texture_binding_count": sum(
                len(item["unexpected_bound_texture_parameters"]) for item in bindings
            ),
        },
        "base_color_textures": base_color_textures,
        "occlusion_textures": occlusion_textures,
        "material_bindings": bindings,
    }


def main() -> None:
    source_path = Path(os.environ["AVENGINE_MP3D_SOURCE_GLB"]).resolve()
    result_path = Path(os.environ["AVENGINE_MP3D_MATERIAL_RESULT"]).resolve()
    mode = os.environ.get("AVENGINE_MP3D_MATERIAL_MODE", "verify").strip().casefold()
    if mode not in {"repair", "verify"}:
        raise RuntimeError("AVENGINE_MP3D_MATERIAL_MODE must be repair or verify")
    source = _source_contract(source_path)
    changes = (
        _repair(source)
        if mode == "repair"
        else {
            "created_base_color_texture_count": 0,
            "rebound_material_count": 0,
        }
    )
    readback = _readback(source)
    result = {
        "schema": SCHEMA,
        "status": "pass",
        "operation": "repair" if mode == "repair" else "verify_only",
        "fresh_editor_reload": mode == "verify",
        "content_root": CONTENT_ROOT,
        "project_file": str(Path(unreal.Paths.get_project_file_path()).resolve()),
        "source_gltf_contract": source,
        "changes": changes,
        **readback,
        "claim_boundary": (
            "UE editor asset readback for the MP3D base-color/occlusion texture "
            "views only; it is not render, lighting, navigation or admission evidence"
        ),
    }
    _write_json(result_path, result)
    print(
        "SPEAR_MP3D_MATERIAL_COLOR_OK "
        f"mode={mode} base={readback['counts']['base_color_srgb_true_count']} "
        f"occlusion={readback['counts']['occlusion_srgb_false_count']} "
        f"result={result_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
