#!/usr/bin/env python3
"""Bind one realized appearance request into package spec and source lineage."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, BinaryIO, Mapping, Sequence

from avengine.appearance.contracts import validate_l9_batch
from avengine.contracts.json_io import (
    canonical_json_sha256,
)
from avengine.assets.glb import GlbError, parse_glb
from avengine.assets.appearance_realization import (
    AppearanceRealizationError,
    verify_appearance_realization,
)
from avengine.assets.kinematics import AnchorDefinition, KinematicsError, RigidTransform
from avengine.assets.package import AnimalPackageIdentity, PackageCompileError
from avengine.assets.variant_package import (
    VariantPackageError,
    load_variant_package_spec,
)


SPEC_SCHEMA = "avengine_m2_variant_package_spec_v1"
LINEAGE_SCHEMA = "avengine_m2_appearance_variant_lineage_v1"
_REALIZER = Path(__file__).resolve().parents[1] / "blender/realize_animal_appearance.py"
_MATERIAL_NORMALIZER = (
    Path(__file__).resolve().parents[2] / "src/avengine/assets/materials.py"
)
_OUTPUT_FLOAT_TOLERANCE = 5.0e-5
_MAXIMUM_SPECULAR_FACTOR = 0.25
_MAXIMUM_SPECULAR_COLOR_FACTOR = 1.0
_ZERO_EMISSIVE_FACTOR = [0.0, 0.0, 0.0]
_EXPECTED_EXPORT_PROFILE = {
    "animation_mode": "ACTIONS",
    "force_sampling": True,
    "format": "GLB",
    "image_format": "AUTO",
    "normals": True,
    "skins": True,
    "texcoords": True,
}
_IDENTITY_FIELDS = {
    "asset_id",
    "template_id",
    "body_plan_id",
    "morphotype_id",
    "skeleton_revision",
    "weights_revision",
    "collision_revision",
    "action_revision",
    "source",
    "source_revision",
    "license",
    "allowed_use",
    "redistribution",
    "semantic_id",
}
_REQUIRED_ANCHORS = {
    "body",
    "head",
    "muzzle",
    "paw_front_left",
    "paw_front_right",
    "paw_hind_left",
    "paw_hind_right",
}
# Common-name appearance requests need an explicit bridge to package taxonomy.
# Unknown species fail closed until a reviewed mapping is added here.
_SPECIES_COMPATIBILITY = {
    "dog": ("canis_lupus_familiaris", "quadruped_mammal_canid"),
    "cat": ("felis_catus", "quadruped_mammal_felid"),
}


class AppearanceVariantInputError(RuntimeError):
    """Appearance bytes cannot be bound to a package input pair."""


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    payload: bytes
    device: int
    inode: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    @property
    def byte_size(self) -> int:
        return len(self.payload)


@dataclass(frozen=True)
class _JsonSnapshot:
    file: _FileSnapshot
    value: Mapping[str, Any]


@dataclass(frozen=True)
class _OutputReservation:
    path: Path
    name: str
    stream: BinaryIO
    parent_descriptor: int
    device: int
    inode: int


def _absolute_without_symlinks(path: str | Path, *, owner: str) -> Path:
    absolute = Path(os.path.abspath(Path(path)))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise AppearanceVariantInputError(
                f"{owner} path must not contain a symbolic link: {cursor}"
            )
    return absolute


def _open_directory_chain(directory: Path, *, owner: str, create: bool = False) -> int:
    """Open an absolute directory without ever following an ancestor symlink."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(directory.anchor, flags)
    except OSError as exc:
        raise AppearanceVariantInputError(
            f"unable to open {owner} root directory: {exc}"
        ) from exc
    try:
        for part in directory.parts[1:]:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except OSError as exc:
        os.close(descriptor)
        raise AppearanceVariantInputError(
            f"unable to open {owner} without symbolic links: {exc}"
        ) from exc
    return descriptor


def _open_readonly_no_symlinks(path: Path, *, owner: str) -> int:
    parent = _open_directory_chain(path.parent, owner=f"{owner} parent")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(path.name, flags, dir_fd=parent)
    except OSError as exc:
        raise AppearanceVariantInputError(f"unable to open {owner}: {exc}") from exc
    finally:
        os.close(parent)


def _snapshot_file(path: str | Path, owner: str) -> _FileSnapshot:
    absolute = _absolute_without_symlinks(path, owner=owner)
    descriptor = _open_readonly_no_symlinks(absolute, owner=owner)
    try:
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
                raise AppearanceVariantInputError(
                    f"{owner} must be a non-empty regular file: {absolute}"
                )
            payload = stream.read()
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise AppearanceVariantInputError(f"unable to read {owner}: {exc}") from exc
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(payload) != after.st_size
    ):
        raise AppearanceVariantInputError(f"{owner} changed while being read")
    current_descriptor = _open_readonly_no_symlinks(
        absolute, owner=f"rechecked {owner}"
    )
    try:
        current = os.fstat(current_descriptor)
    finally:
        os.close(current_descriptor)
    if (
        current.st_dev != after.st_dev
        or current.st_ino != after.st_ino
        or current.st_size != after.st_size
        or current.st_mtime_ns != after.st_mtime_ns
    ):
        raise AppearanceVariantInputError(f"{owner} path changed while being read")
    return _FileSnapshot(absolute, payload, after.st_dev, after.st_ino)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise AppearanceVariantInputError(f"JSON contains duplicate key {key!r}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise AppearanceVariantInputError(f"JSON contains non-finite number {value}")


def _snapshot_json(path: str | Path, owner: str) -> _JsonSnapshot:
    snapshot = _snapshot_file(path, owner)
    return _JsonSnapshot(snapshot, _decode_json_object(snapshot.payload, owner=owner))


def _decode_json_object(payload: bytes, *, owner: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except UnicodeDecodeError as exc:
        raise AppearanceVariantInputError(f"{owner} must be UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise AppearanceVariantInputError(f"{owner} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AppearanceVariantInputError(f"{owner} must contain one JSON object")
    return value


def _record(
    snapshot: _FileSnapshot | _JsonSnapshot, *, include_snapshot: bool = False
) -> dict[str, Any]:
    file = snapshot.file if isinstance(snapshot, _JsonSnapshot) else snapshot
    value: dict[str, Any] = {
        "path": str(file.path),
        "byte_size": file.byte_size,
        "sha256": file.sha256,
    }
    if include_snapshot:
        if not isinstance(snapshot, _JsonSnapshot):
            raise AppearanceVariantInputError(
                "only a strict JSON input can include a lineage snapshot"
            )
        value["canonical_content_sha256"] = canonical_json_sha256(snapshot.value)
        value["snapshot"] = deepcopy(dict(snapshot.value))
    return value


def _json_payload(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AppearanceVariantInputError(
            f"unable to encode output JSON: {exc}"
        ) from exc


def _request(batch: Mapping[str, Any], ordinal: int) -> Mapping[str, Any]:
    requests = batch.get("requests")
    if not isinstance(requests, list):
        raise AppearanceVariantInputError("appearance batch requests must be an array")
    matches = [
        item
        for item in requests
        if isinstance(item, Mapping) and item.get("ordinal") == ordinal
    ]
    if len(matches) != 1:
        raise AppearanceVariantInputError(
            f"batch has no unique appearance request ordinal {ordinal}"
        )
    return matches[0]


def _mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AppearanceVariantInputError(f"{owner} must be an object")
    return value


def _sequence(value: Any, *, owner: str) -> list[Any]:
    if not isinstance(value, list):
        raise AppearanceVariantInputError(f"{owner} must be an array")
    return value


def _finite(value: Any, *, owner: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise AppearanceVariantInputError(f"{owner} must be a finite number")
    return float(value)


def _nonnegative_below_tolerance(value: Any, *, owner: str) -> float:
    return _nonnegative_at_most(value, owner=owner, maximum=_OUTPUT_FLOAT_TOLERANCE)


def _nonnegative_at_most(value: Any, *, owner: str, maximum: float) -> float:
    number = _finite(value, owner=owner)
    if number < 0.0 or number > maximum:
        raise AppearanceVariantInputError(f"{owner} must be in [0, {maximum}]")
    return number


def _positive_integer(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AppearanceVariantInputError(f"{owner} must be a positive integer")
    return value


def _bounded_material_vector(
    value: Any, *, owner: str, length: int, maximum: float = 1.0
) -> list[float]:
    raw = _sequence(value, owner=owner)
    if len(raw) != length:
        raise AppearanceVariantInputError(
            f"{owner} must contain exactly {length} factors"
        )
    result = [
        _finite(item, owner=f"{owner}[{index}]") for index, item in enumerate(raw)
    ]
    if any(item < 0.0 or item > maximum for item in result):
        raise AppearanceVariantInputError(f"{owner} factors must be in [0, {maximum}]")
    return result


def _strict_material_values(value: Any) -> dict[str, Any]:
    """Derive strict effective matte values from the GLB material itself."""

    material = _mapping(value, owner="realized GLB material")
    pbr = _mapping(
        material.get("pbrMetallicRoughness"),
        owner="realized GLB material.pbrMetallicRoughness",
    )
    alpha_mode = material.get("alphaMode", "OPAQUE")
    base_color_factor = _bounded_material_vector(
        pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0]),
        owner="realized GLB material baseColorFactor",
        length=4,
    )
    emissive_factor = _bounded_material_vector(
        material.get("emissiveFactor", list(_ZERO_EMISSIVE_FACTOR)),
        owner="realized GLB material emissiveFactor",
        length=3,
    )
    extensions = _mapping(
        material.get("extensions"), owner="realized GLB material.extensions"
    )
    specular = _mapping(
        extensions.get("KHR_materials_specular"),
        owner="realized GLB KHR_materials_specular",
    )
    specular_factor = _finite(
        specular.get("specularFactor", 1.0),
        owner="realized GLB KHR_materials_specular.specularFactor",
    )
    specular_color_factor = _bounded_material_vector(
        specular.get("specularColorFactor", [1.0, 1.0, 1.0]),
        owner="realized GLB KHR_materials_specular.specularColorFactor",
        length=3,
        maximum=_MAXIMUM_SPECULAR_COLOR_FACTOR,
    )
    effective_specular_peak = specular_factor * max(specular_color_factor)
    if (
        alpha_mode != "OPAQUE"
        or base_color_factor[3] != 1.0
        or emissive_factor != _ZERO_EMISSIVE_FACTOR
        or "emissiveTexture" in material
        or pbr.get("metallicFactor") != 0.0
        or not 0.72
        <= _finite(
            pbr.get("roughnessFactor"),
            owner="realized GLB material roughnessFactor",
        )
        <= 1.0
        or "metallicRoughnessTexture" in pbr
        or set(extensions) != {"KHR_materials_specular"}
        or not 0.0 <= specular_factor <= _MAXIMUM_SPECULAR_FACTOR
        or effective_specular_peak > _MAXIMUM_SPECULAR_FACTOR
    ):
        raise AppearanceVariantInputError(
            "realized GLB material bytes violate the strict opaque matte policy"
        )
    return {
        "alpha_mode": alpha_mode,
        "base_color_factor": base_color_factor,
        "emissive_factor": emissive_factor,
        "emissive_texture_present": False,
        "effective_khr_materials_specular_factor": specular_factor,
        "effective_khr_materials_specular_color_factor": specular_color_factor,
        "maximum_effective_khr_materials_specular_channel": (effective_specular_peak),
    }


def _lower_sha256(value: Any, *, owner: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise AppearanceVariantInputError(f"{owner} must be lowercase SHA-256")
    return value


def _validate_record(
    value: Any, snapshot: _FileSnapshot, *, owner: str, require_path: bool = True
) -> None:
    record = _mapping(value, owner=owner)
    if (
        record.get("sha256") != snapshot.sha256
        or record.get("byte_size") != snapshot.byte_size
    ):
        raise AppearanceVariantInputError(f"{owner} does not bind exact file bytes")
    if require_path:
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise AppearanceVariantInputError(f"{owner}.path must be non-empty")
        declared = _absolute_without_symlinks(raw_path, owner=f"{owner}.path")
        if declared != snapshot.path:
            raise AppearanceVariantInputError(f"{owner}.path differs from its input")


def _validate_tool_identity(value: Any) -> Mapping[str, Any]:
    tool = _mapping(value, owner="appearance report tool_identity")
    expected = _snapshot_file(_REALIZER, "approved appearance realizer")
    raw_path = tool.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise AppearanceVariantInputError("tool_identity.path must be non-empty")
    declared = _absolute_without_symlinks(raw_path, owner="tool_identity.path")
    if declared != expected.path or tool.get("sha256") != expected.sha256:
        raise AppearanceVariantInputError(
            "appearance report tool_identity does not bind the approved realizer"
        )
    material_tool = _mapping(
        tool.get("material_normalizer"), owner="tool_identity.material_normalizer"
    )
    material_expected = _snapshot_file(
        _MATERIAL_NORMALIZER, "approved material normalizer"
    )
    material_path = material_tool.get("path")
    if not isinstance(material_path, str) or not material_path:
        raise AppearanceVariantInputError(
            "tool_identity.material_normalizer.path must be non-empty"
        )
    if (
        _absolute_without_symlinks(
            material_path, owner="tool_identity.material_normalizer.path"
        )
        != material_expected.path
        or material_tool.get("sha256") != material_expected.sha256
    ):
        raise AppearanceVariantInputError(
            "appearance report does not bind the approved material normalizer"
        )
    if (
        not isinstance(tool.get("blender_version"), str)
        or not tool["blender_version"].strip()
    ):
        raise AppearanceVariantInputError("tool_identity.blender_version is invalid")
    if tool.get("export_profile") != _EXPECTED_EXPORT_PROFILE:
        raise AppearanceVariantInputError(
            "tool_identity.export_profile differs from the strict GLB profile"
        )
    tolerance = _finite(
        tool.get("output_readback_float_tolerance"),
        owner="tool_identity.output_readback_float_tolerance",
    )
    if tolerance != _OUTPUT_FLOAT_TOLERANCE:
        raise AppearanceVariantInputError(
            "tool_identity output tolerance differs from the approved realizer"
        )
    return tool


def _validate_mesh_readback(value: Any) -> None:
    mesh = _mapping(value, owner="readback_audit.mesh_invariants")
    if mesh.get("indices_exact") is not True or mesh.get("joints_0_exact") is not True:
        raise AppearanceVariantInputError("mesh identity arrays are not exact")
    _positive_integer(mesh.get("vertex_count"), owner="mesh_invariants.vertex_count")
    _positive_integer(mesh.get("index_count"), owner="mesh_invariants.index_count")
    _nonnegative_at_most(
        mesh.get("maximum_expected_position_error_m"),
        owner="mesh_invariants.maximum_expected_position_error_m",
        maximum=_OUTPUT_FLOAT_TOLERANCE,
    )
    _nonnegative_at_most(
        mesh.get("maximum_output_normal_norm_error"),
        owner="mesh_invariants.maximum_output_normal_norm_error",
        maximum=5.0e-4,
    )
    for field in ("maximum_texcoord_0_error", "maximum_weights_0_error"):
        _nonnegative_at_most(
            mesh.get(field), owner=f"mesh_invariants.{field}", maximum=1.0e-6
        )
    for field in ("head_weight_sum", "torso_weight_sum"):
        if _finite(mesh.get(field), owner=f"mesh_invariants.{field}") <= 0.0:
            raise AppearanceVariantInputError(
                f"mesh_invariants.{field} must be positive"
            )
    if mesh.get("geometry_frame") != {
        "basis_formula": "blender=(gltf.x,-gltf.z,gltf.y)",
        "blender_import": {
            "forward": "positive_x",
            "lateral": "positive_y",
            "up": "positive_z",
        },
        "source": "gltf_positive_y_up",
    }:
        raise AppearanceVariantInputError("mesh geometry frame is not canonical")


def _validate_skin_readback(value: Any, *, expected_joint_count: int) -> None:
    skin = _mapping(value, owner="readback_audit.skin_invariants")
    if (
        skin.get("joint_order_unchanged") is not True
        or skin.get("joint_count") != expected_joint_count
        or _finite(skin.get("tolerance"), owner="skin_invariants.tolerance")
        != _OUTPUT_FLOAT_TOLERANCE
    ):
        raise AppearanceVariantInputError("skin readback identity is invalid")
    for field in (
        "maximum_rest_rotation_error",
        "maximum_rest_scale_error",
        "maximum_scaled_inverse_bind_matrix_error",
        "maximum_scaled_rest_translation_error_m",
    ):
        _nonnegative_below_tolerance(skin.get(field), owner=f"skin_invariants.{field}")


def _validate_action_readback(value: Any) -> None:
    actions = _mapping(value, owner="readback_audit.action_invariants")
    if (
        actions.get("channel_targets_unchanged") is not True
        or actions.get("translations_scaled_by_size") is not True
        or _finite(actions.get("tolerance"), owner="action_invariants.tolerance")
        != _OUTPUT_FLOAT_TOLERANCE
    ):
        raise AppearanceVariantInputError("action readback identity is invalid")
    maximum = _nonnegative_below_tolerance(
        actions.get("maximum_error"), owner="action_invariants.maximum_error"
    )
    records = _sequence(actions.get("actions"), owner="action_invariants.actions")
    if len(records) != 2:
        raise AppearanceVariantInputError(
            "action readback must contain Idle and Walking"
        )
    seen: set[str] = set()
    observed_maximum = 0.0
    for index, item in enumerate(records):
        record = _mapping(item, owner=f"action_invariants.actions[{index}]")
        name = record.get("action")
        if name not in {"Idle", "Walking"} or name in seen:
            raise AppearanceVariantInputError("action readback names are not canonical")
        seen.add(name)
        _positive_integer(
            record.get("channel_count"),
            owner=f"action_invariants.actions[{index}].channel_count",
        )
        errors = _mapping(
            record.get("maximum_errors"),
            owner=f"action_invariants.actions[{index}].maximum_errors",
        )
        if set(errors) != {"rotation", "scale", "timestamps", "translation"}:
            raise AppearanceVariantInputError(
                "action maximum-error fields are incomplete"
            )
        for field, error in errors.items():
            observed_maximum = max(
                observed_maximum,
                _nonnegative_below_tolerance(
                    error,
                    owner=(
                        f"action_invariants.actions[{index}].maximum_errors.{field}"
                    ),
                ),
            )
    if seen != {"Idle", "Walking"} or maximum < observed_maximum:
        raise AppearanceVariantInputError("action global maximum error is inconsistent")


def _validate_texture_readback(
    value: Any,
    *,
    base_color_texture: _FileSnapshot,
    expected_material: Mapping[str, Any],
) -> None:
    material = _mapping(value, owner="readback_audit.material_invariants")
    _positive_integer(
        material.get("material_count"), owner="material_invariants.material_count"
    )
    metallic = _finite(
        material.get("metallic_factor"), owner="material_invariants.metallic_factor"
    )
    roughness = _finite(
        material.get("roughness_factor"), owner="material_invariants.roughness_factor"
    )
    if metallic != 0.0 or not 0.72 <= roughness <= 1.0:
        raise AppearanceVariantInputError(
            "material PBR readback is not non-metallic/rough"
        )
    if material.get("metallic_roughness_texture_present") is not False:
        raise AppearanceVariantInputError(
            "material readback retains a metallic-roughness multiplier texture"
        )
    if material.get("alpha_mode") != expected_material["alpha_mode"]:
        raise AppearanceVariantInputError("material readback alpha mode differs")
    base_color_factor = _bounded_material_vector(
        material.get("base_color_factor"),
        owner="material_invariants.base_color_factor",
        length=4,
    )
    if (
        base_color_factor != expected_material["base_color_factor"]
        or base_color_factor[3] != 1.0
    ):
        raise AppearanceVariantInputError(
            "material readback base-color alpha is not opaque"
        )
    emissive_factor = _bounded_material_vector(
        material.get("emissive_factor"),
        owner="material_invariants.emissive_factor",
        length=3,
    )
    if (
        emissive_factor != _ZERO_EMISSIVE_FACTOR
        or emissive_factor != expected_material["emissive_factor"]
        or material.get("emissive_texture_present") is not False
        or material.get("emissive_texture_present")
        != expected_material["emissive_texture_present"]
    ):
        raise AppearanceVariantInputError(
            "material readback retains an emissive contribution"
        )
    specular_factor = _finite(
        material.get("effective_khr_materials_specular_factor"),
        owner="material_invariants.effective_khr_materials_specular_factor",
    )
    if (
        not 0.0 <= specular_factor <= _MAXIMUM_SPECULAR_FACTOR
        or specular_factor
        != expected_material["effective_khr_materials_specular_factor"]
    ):
        raise AppearanceVariantInputError(
            "material readback effective specular factor exceeds 0.25"
        )
    specular_color_factor = _bounded_material_vector(
        material.get("effective_khr_materials_specular_color_factor"),
        owner="material_invariants.effective_khr_materials_specular_color_factor",
        length=3,
        maximum=_MAXIMUM_SPECULAR_COLOR_FACTOR,
    )
    effective_specular_peak = _finite(
        material.get("maximum_effective_khr_materials_specular_channel"),
        owner=("material_invariants.maximum_effective_khr_materials_specular_channel"),
    )
    if (
        specular_color_factor
        != expected_material["effective_khr_materials_specular_color_factor"]
        or effective_specular_peak
        != expected_material["maximum_effective_khr_materials_specular_channel"]
        or not math.isclose(
            effective_specular_peak,
            specular_factor * max(specular_color_factor),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not 0.0 <= effective_specular_peak <= _MAXIMUM_SPECULAR_FACTOR
    ):
        raise AppearanceVariantInputError(
            "material readback specular color hides an excessive highlight"
        )
    if material.get("allowed_material_extensions") != ["KHR_materials_specular"]:
        raise AppearanceVariantInputError(
            "material readback extension allowlist differs"
        )
    textures = _mapping(
        material.get("texture_images"), owner="material_invariants.texture_images"
    )
    if set(textures) != {"base_color", "normal", "specular"}:
        raise AppearanceVariantInputError(
            "material readback must contain base_color, normal and specular"
        )
    for texture_name, raw_record in textures.items():
        record = _mapping(
            raw_record, owner=f"material_invariants.texture_images.{texture_name}"
        )
        source_sha = _lower_sha256(
            record.get("source_sha256"), owner=f"texture {texture_name}.source_sha256"
        )
        output_sha = _lower_sha256(
            record.get("output_sha256"), owner=f"texture {texture_name}.output_sha256"
        )
        if record.get("mime_type") != "image/png" or not isinstance(
            record.get("unchanged"), bool
        ):
            raise AppearanceVariantInputError(
                f"texture {texture_name} readback contract is invalid"
            )
        if record["unchanged"] and source_sha != output_sha:
            raise AppearanceVariantInputError(
                f"unchanged texture {texture_name} differs from its source"
            )
    base_color = _mapping(textures["base_color"], owner="base_color readback")
    if (
        base_color.get("embedded_matches_standalone") is not True
        or base_color.get("unchanged") is not False
        or base_color.get("output_sha256") != base_color_texture.sha256
        or base_color.get("standalone_sha256") != base_color_texture.sha256
    ):
        raise AppearanceVariantInputError(
            "base_color embedded/standalone/output bytes do not match"
        )
    for role in ("normal", "specular"):
        record = _mapping(textures[role], owner=f"{role} texture readback")
        if record.get("unchanged") is not True or record.get(
            "source_sha256"
        ) != record.get("output_sha256"):
            raise AppearanceVariantInputError(f"{role} texture did not stay unchanged")


def _operation_parameters(
    request: Mapping[str, Any], attribute: str
) -> Mapping[str, Any]:
    operations = _sequence(
        request.get("realization_operations"), owner="instance realization_operations"
    )
    matches = [
        item
        for item in operations
        if isinstance(item, Mapping) and item.get("attribute") == attribute
    ]
    if len(matches) != 1:
        raise AppearanceVariantInputError(
            f"instance request lacks one {attribute!r} realization operation"
        )
    return _mapping(matches[0].get("parameters"), owner=f"{attribute} parameters")


def _validate_weighted_shape_ratio(
    shape: Mapping[str, Any], *, prefix: str, requested_scale: float
) -> None:
    before = _finite(shape.get(f"{prefix}_before"), owner=f"shape.{prefix}_before")
    after = _finite(shape.get(f"{prefix}_after"), owner=f"shape.{prefix}_after")
    ratio = _finite(shape.get(f"{prefix}_ratio"), owner=f"shape.{prefix}_ratio")
    if before <= 0.0 or after <= 0.0 or ratio <= 0.0:
        raise AppearanceVariantInputError(f"shape.{prefix} metrics must be positive")
    if not math.isclose(ratio, after / before, rel_tol=1.0e-9, abs_tol=1.0e-9):
        raise AppearanceVariantInputError(
            f"shape.{prefix}_ratio is inconsistent with before/after metrics"
        )
    lower = min(1.0, requested_scale) - 1.0e-6
    upper = max(1.0, requested_scale) + 1.0e-6
    if not lower <= ratio <= upper:
        raise AppearanceVariantInputError(
            f"shape.{prefix}_ratio is inconsistent with its requested scale"
        )
    if requested_scale < 1.0 and ratio >= 1.0 - 1.0e-6:
        raise AppearanceVariantInputError(
            f"shape.{prefix}_ratio does not show the requested contraction"
        )
    if requested_scale > 1.0 and ratio <= 1.0 + 1.0e-6:
        raise AppearanceVariantInputError(
            f"shape.{prefix}_ratio does not show the requested expansion"
        )


def _validate_realization_operations(
    value: Any,
    *,
    request: Mapping[str, Any],
    base_color_texture: _FileSnapshot,
) -> None:
    realization = _mapping(value, owner="appearance report realization")
    topology_before = _lower_sha256(
        realization.get("topology_uv_skin_sha256_before"),
        owner="realization.topology_uv_skin_sha256_before",
    )
    topology_after = _lower_sha256(
        realization.get("topology_uv_skin_sha256_after"),
        owner="realization.topology_uv_skin_sha256_after",
    )
    action_before = _lower_sha256(
        realization.get("action_curve_sha256_before"),
        owner="realization.action_curve_sha256_before",
    )
    action_after = _lower_sha256(
        realization.get("action_curve_sha256_after"),
        owner="realization.action_curve_sha256_after",
    )
    if (
        topology_before != topology_after
        or realization.get("topology_uv_skin_unchanged") is not True
        or action_before != action_after
        or realization.get("in_memory_authored_action_curves_unchanged") is not True
    ):
        raise AppearanceVariantInputError(
            "realization changed topology/UV/skin or authored action curves"
        )

    size_parameters = _operation_parameters(request, "size")
    uniform_size = _mapping(realization.get("uniform_size"), owner="uniform_size")
    if (
        uniform_size.get("strategy") != "armature_data_and_mesh_data_matrix_bake_v1"
        or uniform_size.get("ancestor_scale_node_created") is not False
        or _finite(uniform_size.get("scale_ratio"), owner="uniform_size.scale_ratio")
        != _finite(size_parameters.get("scale_ratio"), owner="requested scale_ratio")
        or not 0.0
        <= _finite(
            uniform_size.get("maximum_mesh_scale_error"),
            owner="uniform_size.maximum_mesh_scale_error",
        )
        <= 1.0e-6
    ):
        raise AppearanceVariantInputError("uniform-size realization is not exact")

    body_parameters = _operation_parameters(request, "body_build")
    life_parameters = _operation_parameters(request, "life_stage")
    shape = _mapping(realization.get("shape"), owner="realization.shape")
    requested_torso_scale = _finite(
        shape.get("requested_torso_girth_scale"),
        owner="shape.requested_torso_girth_scale",
    )
    requested_head_scale = _finite(
        shape.get("requested_head_scale"), owner="shape.requested_head_scale"
    )
    if requested_torso_scale != _finite(
        body_parameters.get("torso_girth_scale"),
        owner="requested torso_girth_scale",
    ) or requested_head_scale != _finite(
        life_parameters.get("head_scale"), owner="requested head_scale"
    ):
        raise AppearanceVariantInputError("shape report differs from requested scales")
    for field in ("head_selected_vertices", "torso_selected_vertices"):
        _positive_integer(shape.get(field), owner=f"shape.{field}")
    _validate_weighted_shape_ratio(
        shape,
        prefix="head_weighted_radius_rms",
        requested_scale=requested_head_scale,
    )
    _validate_weighted_shape_ratio(
        shape,
        prefix="torso_weighted_yz_rms",
        requested_scale=requested_torso_scale,
    )
    for field in ("head_group_names", "torso_group_names"):
        names = _sequence(shape.get(field), owner=f"shape.{field}")
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise AppearanceVariantInputError(f"shape.{field} is invalid")

    coat_parameters = _operation_parameters(request, "coat_profile")
    texture = _mapping(realization.get("texture"), owner="realization.texture")
    requested_texture = {
        "luminance_gain": coat_parameters.get("luminance_gain"),
        "coat_desaturation": life_parameters.get("coat_desaturation"),
        "muzzle_gray_mix": life_parameters.get("muzzle_gray_mix"),
        "muzzle_gray_target": life_parameters.get("muzzle_gray_target"),
    }
    for field, requested in requested_texture.items():
        if _finite(texture.get(field), owner=f"texture.{field}") != _finite(
            requested, owner=f"requested {field}"
        ):
            raise AppearanceVariantInputError(
                f"texture.{field} differs from the requested operation"
            )
    raw_output_texture = texture.get("output_texture")
    if not isinstance(raw_output_texture, str) or (
        _absolute_without_symlinks(
            raw_output_texture, owner="realization.texture.output_texture"
        )
        != base_color_texture.path
    ):
        raise AppearanceVariantInputError(
            "realization texture path differs from standalone output"
        )
    resolution = _sequence(texture.get("resolution"), owner="texture.resolution")
    if len(resolution) != 2 or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in resolution
    ):
        raise AppearanceVariantInputError("texture resolution is invalid")
    for field in ("pigmented_pixel_count", "muzzle_mask_nonzero_pixels"):
        _positive_integer(texture.get(field), owner=f"texture.{field}")
    for field in (
        "mean_pigmented_luminance_before",
        "mean_pigmented_luminance_after",
        "muzzle_mask_max",
        "muzzle_forward_quantile",
        "uv_minimum",
        "uv_maximum",
    ):
        number = _finite(texture.get(field), owner=f"texture.{field}")
        if not 0.0 <= number <= 1.0:
            raise AppearanceVariantInputError(f"texture.{field} must be in [0, 1]")
    if (
        texture.get("uv_addressing_assumption") != "non_tiled_clamp_0_1"
        or not isinstance(texture.get("source_image"), str)
        or not texture["source_image"]
    ):
        raise AppearanceVariantInputError("texture addressing/source is invalid")

    materials = _sequence(realization.get("materials"), owner="realization.materials")
    if not materials:
        raise AppearanceVariantInputError("realization materials cannot be empty")
    for index, raw_material in enumerate(materials):
        material = _mapping(raw_material, owner=f"realization.materials[{index}]")
        before = _mapping(material.get("before"), owner="material.before")
        after = _mapping(material.get("after"), owner="material.after")
        if (
            not isinstance(material.get("material"), str)
            or not material["material"]
            or _finite(before.get("metallic"), owner="material.before.metallic") != 0.0
            or _finite(after.get("metallic"), owner="material.after.metallic") != 0.0
            or _finite(before.get("roughness"), owner="material.before.roughness")
            < 0.72
            or _finite(after.get("roughness"), owner="material.after.roughness") < 0.72
            or after.get("roughness_texture_driven") is not False
        ):
            raise AppearanceVariantInputError("realization material is not canonical")


def _require_measured_number(
    claimed: Any,
    measured: Any,
    *,
    owner: str,
    tolerance: float,
) -> None:
    claimed_number = _finite(claimed, owner=owner)
    measured_number = _finite(measured, owner=f"independent {owner}")
    if not math.isclose(
        claimed_number,
        measured_number,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise AppearanceVariantInputError(
            f"{owner} differs from independent source+request recomputation: "
            f"claimed={claimed_number:.12g}, measured={measured_number:.12g}"
        )


def _validate_independent_realization_metrics(
    report: Mapping[str, Any],
    independent: Mapping[str, Any],
    *,
    require_pattern_record: bool,
) -> None:
    """Cross-check report metrics; independent bytes remain authoritative."""

    realization = _mapping(report.get("realization"), owner="appearance realization")
    claimed_shape = _mapping(realization.get("shape"), owner="realization.shape")
    measured_shape = _mapping(independent.get("shape"), owner="independent shape")
    for field in (
        "head_group_names",
        "torso_group_names",
        "head_selected_vertices",
        "torso_selected_vertices",
    ):
        if claimed_shape.get(field) != measured_shape.get(field):
            raise AppearanceVariantInputError(
                f"shape.{field} differs from independent GLB recomputation"
            )
    for field in (
        "requested_head_scale",
        "requested_torso_girth_scale",
        "head_weighted_radius_rms_before",
        "head_weighted_radius_rms_after",
        "head_weighted_radius_rms_ratio",
        "torso_weighted_yz_rms_before",
        "torso_weighted_yz_rms_after",
        "torso_weighted_yz_rms_ratio",
    ):
        _require_measured_number(
            claimed_shape.get(field),
            measured_shape.get(field),
            owner=f"shape.{field}",
            tolerance=2.0e-7,
        )

    claimed_texture = _mapping(
        realization.get("texture"), owner="realization.texture"
    )
    measured_texture = _mapping(
        independent.get("texture"), owner="independent texture"
    )
    if claimed_texture.get("resolution") != measured_texture.get("resolution"):
        raise AppearanceVariantInputError(
            "texture.resolution differs from independently decoded PNG bytes"
        )
    for field, tolerance in (
        ("mean_pigmented_luminance_before", 1.0e-5),
        ("mean_pigmented_luminance_after", 1.0e-5),
        ("muzzle_mask_max", 1.0e-5),
        ("muzzle_forward_quantile", 1.0e-12),
        ("uv_minimum", 1.0e-8),
        ("uv_maximum", 1.0e-8),
    ):
        _require_measured_number(
            claimed_texture.get(field),
            measured_texture.get(field),
            owner=f"texture.{field}",
            tolerance=tolerance,
        )
    if abs(
        _positive_integer(
            claimed_texture.get("pigmented_pixel_count"),
            owner="texture.pigmented_pixel_count",
        )
        - _positive_integer(
            measured_texture.get("pigmented_pixel_count"),
            owner="independent texture.pigmented_pixel_count",
        )
    ) > 128:
        raise AppearanceVariantInputError(
            "texture.pigmented_pixel_count differs from decoded source pixels"
        )
    if abs(
        _positive_integer(
            claimed_texture.get("muzzle_mask_nonzero_pixels"),
            owner="texture.muzzle_mask_nonzero_pixels",
        )
        - _positive_integer(
            measured_texture.get("muzzle_mask_nonzero_pixels"),
            owner="independent texture.muzzle_mask_nonzero_pixels",
        )
    ) > 1024:
        raise AppearanceVariantInputError(
            "texture.muzzle_mask_nonzero_pixels differs from source UV rasterization"
        )
    pattern = claimed_texture.get("preserve_pattern")
    if require_pattern_record and pattern is None:
        raise AppearanceVariantInputError(
            "current appearance producer report lacks preserve_pattern audit"
        )
    if pattern is not None and pattern != measured_texture.get("preserve_pattern"):
        raise AppearanceVariantInputError(
            "texture.preserve_pattern differs from the registered source+request audit"
        )
    if require_pattern_record:
        pattern_audit = _mapping(
            claimed_texture.get("pattern_audit"), owner="texture.pattern_audit"
        )
        if (
            pattern_audit.get("status") != "pass"
            or pattern_audit.get("registered_pattern")
            != measured_texture.get("preserve_pattern")
            or pattern_audit.get(
                "coat_gain_and_desaturation_preserve_region_membership"
            )
            is not True
        ):
            raise AppearanceVariantInputError(
                "texture.pattern_audit does not record registered pattern preservation"
            )
        # Blender classifies decoded float texels while the independent audit
        # starts from the exported 8-bit PNG.  The ratio-based warm boundary
        # is slightly more sensitive to that round-trip than the scalar
        # white/dark/pigmented thresholds.  This tolerance authenticates only
        # the producer's summary count; the output pixels themselves are
        # still checked independently, channel by channel, above.
        pattern_count_tolerances = {
            "white_pixel_count": 128,
            "pigmented_pixel_count": 128,
            "dark_pixel_count": 128,
            "warm_pixel_count": 1024,
        }
        for field, tolerance in pattern_count_tolerances.items():
            claimed = _positive_integer(
                pattern_audit.get(field), owner=f"texture.pattern_audit.{field}"
            )
            measured = _positive_integer(
                measured_texture.get(field), owner=f"independent texture.{field}"
            )
            if abs(claimed - measured) > tolerance:
                raise AppearanceVariantInputError(
                    f"texture.pattern_audit.{field} differs from source PNG pixels"
                )

    output = _mapping(report.get("output"), owner="appearance output")
    readback = _mapping(output.get("readback_audit"), owner="output.readback_audit")
    mesh = _mapping(readback.get("mesh_invariants"), owner="mesh_invariants")
    compatibility = _mapping(
        independent.get("compatibility"), owner="independent compatibility"
    )
    compatible_mesh = _mapping(
        compatibility.get("mesh"), owner="independent compatibility.mesh"
    )
    for field in ("vertex_count", "index_count"):
        if mesh.get(field) != compatible_mesh.get(field):
            raise AppearanceVariantInputError(
                f"mesh_invariants.{field} differs from actual GLB accessors"
            )
    for claimed_field, measured_owner, measured_field, tolerance in (
        (
            "maximum_texcoord_0_error",
            compatible_mesh,
            "maximum_texcoord_0_error",
            1.0e-9,
        ),
        (
            "maximum_weights_0_error",
            measured_shape,
            "maximum_raw_weights_0_error",
            1.0e-9,
        ),
        (
            "maximum_expected_position_error_m",
            measured_shape,
            "maximum_expected_position_error_m",
            1.0e-9,
        ),
        (
            "maximum_output_normal_norm_error",
            measured_shape,
            "maximum_output_normal_norm_error",
            1.0e-9,
        ),
        ("torso_weight_sum", measured_shape, "torso_weight_sum", 1.0e-8),
        ("head_weight_sum", measured_shape, "head_weight_sum", 1.0e-8),
    ):
        _require_measured_number(
            mesh.get(claimed_field),
            measured_owner.get(measured_field),
            owner=f"mesh_invariants.{claimed_field}",
            tolerance=tolerance,
        )

    skin = _mapping(readback.get("skin_invariants"), owner="skin_invariants")
    compatible_skin = _mapping(
        compatibility.get("skin"), owner="independent compatibility.skin"
    )
    if skin.get("joint_count") != compatible_skin.get("joint_count"):
        raise AppearanceVariantInputError(
            "skin_invariants.joint_count differs from actual GLB skin"
        )
    for field in (
        "maximum_scaled_rest_translation_error_m",
        "maximum_rest_rotation_error",
        "maximum_rest_scale_error",
        "maximum_scaled_inverse_bind_matrix_error",
    ):
        _require_measured_number(
            skin.get(field),
            compatible_skin.get(field),
            owner=f"skin_invariants.{field}",
            tolerance=1.0e-9,
        )
    actions = _mapping(readback.get("action_invariants"), owner="action_invariants")
    compatible_actions = _mapping(
        compatibility.get("actions"), owner="independent compatibility.actions"
    )
    action_errors = _mapping(
        compatible_actions.get("maximum_errors"),
        owner="independent compatibility.actions.maximum_errors",
    )
    measured_action_maximum = max(
        _finite(value, owner=f"independent action {name}")
        for name, value in action_errors.items()
    )
    _require_measured_number(
        actions.get("maximum_error"),
        measured_action_maximum,
        owner="action_invariants.maximum_error",
        tolerance=1.0e-9,
    )


def _validate_readback_audit(
    value: Any,
    *,
    visual: _FileSnapshot,
    base_color_texture: _FileSnapshot,
) -> None:
    audit = _mapping(value, owner="appearance report output.readback_audit")
    try:
        document = parse_glb(visual.payload)
    except GlbError as exc:
        raise AppearanceVariantInputError(
            f"realized appearance GLB is invalid: {exc}"
        ) from exc
    root = document.json
    if root.get("extensionsUsed") != ["KHR_materials_specular"]:
        raise AppearanceVariantInputError(
            "realized GLB root extensionsUsed must be exactly "
            "['KHR_materials_specular'] to match the strict material extension"
        )
    meshes = _sequence(root.get("meshes"), owner="realized GLB meshes")
    skins = _sequence(root.get("skins"), owner="realized GLB skins")
    materials = _sequence(root.get("materials"), owner="realized GLB materials")
    images = _sequence(root.get("images"), owner="realized GLB images")
    animations = _sequence(root.get("animations"), owner="realized GLB animations")
    if len(skins) != 1:
        raise AppearanceVariantInputError("realized GLB must contain exactly one skin")
    skin = _mapping(skins[0], owner="realized GLB skin")
    joints = _sequence(skin.get("joints"), owner="realized GLB skin.joints")
    joint_count = len(joints)
    if joint_count <= 0:
        raise AppearanceVariantInputError("realized GLB skin has no joints")
    nodes = _sequence(root.get("nodes"), owner="realized GLB nodes")
    root_index = skin.get("skeleton", joints[0])
    if (
        isinstance(root_index, bool)
        or not isinstance(root_index, int)
        or not 0 <= root_index < len(nodes)
    ):
        raise AppearanceVariantInputError("realized GLB skin root is invalid")
    root_name = _mapping(nodes[root_index], owner="realized GLB root node").get("name")
    animation_names = [
        _mapping(item, owner="realized GLB animation").get("name")
        for item in animations
    ]
    if (
        audit.get("skin_count") != 1
        or audit.get("skin_joint_count") != joint_count
        or audit.get("skin_root_name") != root_name
        or audit.get("mesh_count") != len(meshes)
        or audit.get("material_count") != len(materials)
        or audit.get("image_count") != len(images)
        or len(animation_names) != 2
        or set(animation_names) != {"Idle", "Walking"}
        or audit.get("animation_names") != animation_names
    ):
        raise AppearanceVariantInputError(
            "readback counts/names differ from the realized GLB"
        )
    if len(meshes) != 1 or len(materials) != 1 or len(images) != 3:
        raise AppearanceVariantInputError(
            "strict appearance profile requires one mesh/material and three images"
        )
    strict_material = _strict_material_values(materials[0])
    if (
        not 0.0
        <= _finite(
            audit.get("maximum_skin_ancestor_scale_error"),
            owner="readback_audit.maximum_skin_ancestor_scale_error",
        )
        <= 1.0e-6
    ):
        raise AppearanceVariantInputError("skin ancestor scale error exceeds its gate")
    ancestors = _sequence(audit.get("skin_ancestors"), owner="readback skin_ancestors")
    for index, raw_ancestor in enumerate(ancestors):
        ancestor = _mapping(raw_ancestor, owner=f"skin_ancestors[{index}]")
        scale = _sequence(ancestor.get("scale"), owner=f"skin_ancestors[{index}].scale")
        if (
            not isinstance(ancestor.get("name"), str)
            or not ancestor["name"]
            or isinstance(ancestor.get("node"), bool)
            or not isinstance(ancestor.get("node"), int)
            or ancestor["node"] < 0
            or len(scale) != 3
            or any(
                abs(_finite(item, owner="ancestor scale") - 1.0) > 1.0e-6
                for item in scale
            )
        ):
            raise AppearanceVariantInputError("skin ancestor record is invalid")
    _validate_mesh_readback(audit.get("mesh_invariants"))
    _validate_skin_readback(
        audit.get("skin_invariants"), expected_joint_count=joint_count
    )
    _validate_action_readback(audit.get("action_invariants"))
    _validate_texture_readback(
        audit.get("material_invariants"),
        base_color_texture=base_color_texture,
        expected_material=strict_material,
    )


def _validate_realization(
    report: Mapping[str, Any],
    *,
    batch_snapshot: _JsonSnapshot,
    batch: Mapping[str, Any],
    request: Mapping[str, Any],
) -> _FileSnapshot:
    declared_report_digest = report.get("report_content_sha256")
    if declared_report_digest != canonical_json_sha256(
        {
            key: deepcopy(value)
            for key, value in report.items()
            if key != "report_content_sha256"
        }
    ):
        raise AppearanceVariantInputError(
            "appearance report_content_sha256 does not authenticate the report"
        )
    if (
        report.get("schema") != "avengine_animal_appearance_realization_v1"
        or report.get("status") != "pass"
        or report.get("state_classification") != "research_candidate"
        or report.get("qualification_claim") is not False
        or report.get("formal_dataset_registration_authorized") is not False
    ):
        raise AppearanceVariantInputError(
            "appearance report must be a non-qualifying research pass"
        )
    report_batch = report.get("batch")
    report_request = report.get("instance_request")
    if not isinstance(report_batch, Mapping) or not isinstance(report_request, Mapping):
        raise AppearanceVariantInputError("appearance report bindings are invalid")
    report_batch_path = report_batch.get("path")
    if not isinstance(report_batch_path, str) or not report_batch_path:
        raise AppearanceVariantInputError(
            "appearance report batch.path must be a non-empty string"
        )
    if (
        report_batch.get("sha256") != batch_snapshot.file.sha256
        or report_batch.get("batch_id") != batch["batch_id"]
        or report_batch.get("batch_content_sha256") != batch["batch_content_sha256"]
        or _absolute_without_symlinks(
            report_batch_path, owner="appearance report batch.path"
        )
        != batch_snapshot.file.path
        or report_request.get("ordinal") != request["ordinal"]
        or report_request.get("instance_request_id") != request["instance_request_id"]
        or report_request.get("request_sha256") != request["request_sha256"]
        or report_request.get("taxonomy") != request["taxonomy"]
        or report_request.get("attributes") != request["attributes"]
        or report_request.get("realization_operations")
        != request["realization_operations"]
    ):
        raise AppearanceVariantInputError(
            "appearance report does not bind the exact batch request"
        )

    batch_source = _mapping(batch.get("source_asset"), owner="batch.source_asset")
    report_source = _mapping(report.get("source"), owner="appearance report source")
    raw_source_path = report_source.get("path")
    if not isinstance(raw_source_path, str) or not raw_source_path:
        raise AppearanceVariantInputError("appearance report source.path is invalid")
    source = _snapshot_file(raw_source_path, "appearance report source GLB")
    _validate_record(report_source, source, owner="appearance report source")
    batch_source_path = batch_source.get("path")
    if (
        not isinstance(batch_source_path, str)
        or _absolute_without_symlinks(
            batch_source_path, owner="batch.source_asset.path"
        )
        != source.path
        or batch_source.get("sha256") != source.sha256
        or batch_source.get("byte_size") != source.byte_size
    ):
        raise AppearanceVariantInputError(
            "appearance report source differs from the authenticated batch source"
        )
    _validate_tool_identity(report.get("tool_identity"))

    output = report.get("output")
    output_mapping = _mapping(output, owner="appearance report output")
    glb_record = _mapping(output_mapping.get("glb"), owner="appearance output GLB")
    raw_glb_path = glb_record.get("path")
    if not isinstance(raw_glb_path, str) or not raw_glb_path:
        raise AppearanceVariantInputError(
            "appearance report output GLB path is invalid"
        )
    visual = _snapshot_file(raw_glb_path, "realized appearance GLB")
    _validate_record(glb_record, visual, owner="appearance output GLB")
    if visual.sha256 == source.sha256:
        raise AppearanceVariantInputError(
            "realized appearance GLB is byte-identical to its source; "
            "the declared appearance operations were not realized"
        )

    texture_record = _mapping(
        output_mapping.get("base_color_texture"),
        owner="appearance output base_color_texture",
    )
    raw_texture_path = texture_record.get("path")
    if not isinstance(raw_texture_path, str) or not raw_texture_path:
        raise AppearanceVariantInputError("base_color_texture.path is invalid")
    texture = _snapshot_file(raw_texture_path, "realized base-color texture")
    _validate_record(
        texture_record, texture, owner="appearance output base_color_texture"
    )
    _validate_realization_operations(
        report.get("realization"), request=request, base_color_texture=texture
    )
    _validate_readback_audit(
        output_mapping.get("readback_audit"),
        visual=visual,
        base_color_texture=texture,
    )
    try:
        independent = verify_appearance_realization(
            source_path=source.path,
            output_path=visual.path,
            source_payload=source.payload,
            output_payload=visual.payload,
            standalone_texture_payload=texture.payload,
            request=request,
        )
    except AppearanceRealizationError as exc:
        raise AppearanceVariantInputError(
            f"independent appearance realization audit failed: {exc}"
        ) from exc
    _validate_independent_realization_metrics(
        report,
        independent,
        require_pattern_record=True,
    )
    return visual


def _validate_taxonomy(
    *, template_value: Mapping[str, Any], request: Mapping[str, Any]
) -> None:
    template_taxonomy = _mapping(
        template_value.get("taxonomy"), owner="template taxonomy"
    )
    request_taxonomy = _mapping(request.get("taxonomy"), owner="request taxonomy")
    species = request_taxonomy.get("species")
    breed = request_taxonomy.get("breed")
    if (
        not isinstance(species, str)
        or not species
        or not isinstance(breed, str)
        or not breed
    ):
        raise AppearanceVariantInputError("request species/breed must be non-empty")
    try:
        expected_species_id, body_plan_prefix = _SPECIES_COMPATIBILITY[species]
    except KeyError as exc:
        raise AppearanceVariantInputError(
            f"appearance species {species!r} lacks an approved package mapping"
        ) from exc
    identity = _mapping(template_value.get("identity"), owner="template identity")
    body_plan_id = identity.get("body_plan_id")
    if (
        template_taxonomy.get("species_id") != expected_species_id
        or template_taxonomy.get("breed_id") != breed
        or not isinstance(body_plan_id, str)
        or not (
            body_plan_id == body_plan_prefix
            or body_plan_id.startswith(body_plan_prefix + "_")
        )
    ):
        raise AppearanceVariantInputError(
            "template species/breed/body_plan is incompatible with the appearance request"
        )


def _validate_upstream_source(
    value: Mapping[str, Any], *, template_value: Mapping[str, Any]
) -> None:
    schema = value.get("schema")
    if (
        not isinstance(schema, str)
        or re.fullmatch(
            r"avengine_m2_[a-z0-9]+(?:_[a-z0-9]+)*_source_snapshot_v1", schema
        )
        is None
    ):
        raise AppearanceVariantInputError(
            "upstream source manifest schema must be an M2 source snapshot v1"
        )
    if (
        value.get("qualification_state") != "research_candidate"
        or value.get("qualification_claim") is not False
        or value.get("formal_dataset_registration_authorized") is not False
    ):
        raise AppearanceVariantInputError(
            "upstream source manifest must be a non-qualifying research candidate"
        )
    repository = _mapping(
        value.get("source_repository"), owner="upstream source_repository"
    )
    template_identity = _mapping(
        template_value.get("identity"), owner="template identity"
    )
    if repository.get("revision") != template_identity.get("source_revision"):
        raise AppearanceVariantInputError(
            "upstream source revision differs from template identity.source_revision"
        )
    if not isinstance(repository.get("url"), str) or not repository["url"].strip():
        raise AppearanceVariantInputError("upstream source_repository.url is invalid")
    artifacts = _sequence(
        value.get("source_artifacts"), owner="upstream source_artifacts"
    )
    if not artifacts:
        raise AppearanceVariantInputError("upstream source_artifacts cannot be empty")
    for index, raw_artifact in enumerate(artifacts):
        artifact = _mapping(raw_artifact, owner=f"source_artifacts[{index}]")
        _lower_sha256(artifact.get("sha256"), owner=f"source_artifacts[{index}].sha256")
        _positive_integer(
            artifact.get("byte_size"), owner=f"source_artifacts[{index}].byte_size"
        )
        for field in ("path", "root_id"):
            if not isinstance(artifact.get(field), str) or not artifact[field].strip():
                raise AppearanceVariantInputError(
                    f"source_artifacts[{index}].{field} is invalid"
                )


def _validate_generated_spec(
    value: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    known_joint_ids: set[str],
) -> AnimalPackageIdentity:
    if set(value) != {
        "schema",
        "identity",
        "taxonomy",
        "appearance",
        "rendering",
        "anchors",
    }:
        raise AppearanceVariantInputError(
            "derived variant spec root fields are invalid"
        )
    if value.get("schema") != SPEC_SCHEMA:
        raise AppearanceVariantInputError("derived variant spec schema is invalid")
    _validate_taxonomy(template_value=value, request=request)
    identity_value = _mapping(value.get("identity"), owner="derived identity")
    if set(identity_value) != _IDENTITY_FIELDS:
        raise AppearanceVariantInputError("derived identity fields are invalid")
    try:
        identity = AnimalPackageIdentity(**dict(identity_value))
    except (TypeError, PackageCompileError) as exc:
        raise AppearanceVariantInputError(
            f"derived identity is invalid: {exc}"
        ) from exc
    attributes = _mapping(request.get("attributes"), owner="request attributes")
    if value.get("appearance") != {
        "size": attributes.get("size"),
        "body_build": attributes.get("body_build"),
        "coat": attributes.get("coat_profile"),
        "life_stage": attributes.get("life_stage"),
    }:
        raise AppearanceVariantInputError(
            "derived spec appearance differs from the authenticated request"
        )
    rendering = _mapping(value.get("rendering"), owner="derived rendering")
    shader_type = rendering.get("shader_type")
    if (
        set(rendering) != {"shader_type"}
        or not isinstance(shader_type, str)
        or shader_type not in {"phong", "pbr"}
    ):
        raise AppearanceVariantInputError(
            "derived rendering must bind exactly one supported shader_type"
        )
    anchors = _sequence(value.get("anchors"), owner="derived anchors")
    anchor_ids: list[str] = []
    for index, raw_anchor in enumerate(anchors):
        anchor = _mapping(raw_anchor, owner=f"derived anchors[{index}]")
        if set(anchor) != {"anchor_id", "joint_id", "joint_from_anchor"}:
            raise AppearanceVariantInputError(
                f"derived anchors[{index}] fields are invalid"
            )
        transform = _mapping(
            anchor.get("joint_from_anchor"),
            owner=f"derived anchors[{index}].joint_from_anchor",
        )
        if set(transform) != {"translation_m", "rotation_xyzw"}:
            raise AppearanceVariantInputError(
                f"derived anchors[{index}] transform fields are invalid"
            )
        translation = transform.get("translation_m")
        rotation = transform.get("rotation_xyzw")
        if not isinstance(translation, list) or not isinstance(rotation, list):
            raise AppearanceVariantInputError(
                f"derived anchors[{index}] transform values must be arrays"
            )
        try:
            definition = AnchorDefinition(
                anchor_id=anchor.get("anchor_id"),
                joint_id=anchor.get("joint_id"),
                joint_from_anchor=RigidTransform(tuple(translation), tuple(rotation)),
            )
        except (TypeError, KinematicsError) as exc:
            raise AppearanceVariantInputError(
                f"derived anchors[{index}] is invalid: {exc}"
            ) from exc
        if definition.joint_id not in known_joint_ids:
            raise AppearanceVariantInputError(
                f"derived anchor {definition.anchor_id!r} references unknown visual "
                f"joint {definition.joint_id!r}"
            )
        anchor_ids.append(definition.anchor_id)
    if len(anchor_ids) != len(set(anchor_ids)) or not _REQUIRED_ANCHORS.issubset(
        anchor_ids
    ):
        raise AppearanceVariantInputError(
            "derived anchors are duplicated or missing required semantic IDs"
        )
    return identity


def _visual_joint_ids(visual: _FileSnapshot) -> set[str]:
    try:
        document = parse_glb(visual.payload)
    except GlbError as exc:
        raise AppearanceVariantInputError(
            f"realized appearance GLB is invalid: {exc}"
        ) from exc
    root = document.json
    skins = _sequence(root.get("skins"), owner="realized GLB skins")
    nodes = _sequence(root.get("nodes"), owner="realized GLB nodes")
    if len(skins) != 1:
        raise AppearanceVariantInputError("realized GLB must contain exactly one skin")
    joint_indices = _sequence(
        _mapping(skins[0], owner="realized GLB skin").get("joints"),
        owner="realized GLB skin.joints",
    )
    names: list[str] = []
    for index in joint_indices:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(nodes)
        ):
            raise AppearanceVariantInputError(
                "realized GLB skin joint index is invalid"
            )
        name = _mapping(nodes[index], owner=f"realized GLB nodes[{index}]").get("name")
        if not isinstance(name, str) or not name:
            raise AppearanceVariantInputError(
                "every realized GLB skin joint must have a name"
            )
        names.append(name)
    if len(names) != len(set(names)):
        raise AppearanceVariantInputError(
            "realized GLB skin joint names are duplicated"
        )
    return set(names)


def _lineage_json_input(
    lineage: Mapping[str, Any], *, key: str, owner: str
) -> _JsonSnapshot:
    inputs = _mapping(lineage.get("inputs"), owner="lineage.inputs")
    binding = _mapping(inputs.get(key), owner=f"lineage.inputs.{key}")
    if set(binding) != {
        "path",
        "byte_size",
        "sha256",
        "canonical_content_sha256",
        "snapshot",
    }:
        raise AppearanceVariantInputError(f"{owner} lineage fields are invalid")
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise AppearanceVariantInputError(f"{owner} lineage path is invalid")
    current = _snapshot_json(raw_path, f"lineage {owner}")
    _validate_record(binding, current.file, owner=f"lineage {owner}")
    declared_snapshot = _mapping(
        binding.get("snapshot"), owner=f"{owner} lineage snapshot"
    )
    if (
        binding.get("canonical_content_sha256") != canonical_json_sha256(current.value)
        or declared_snapshot != current.value
    ):
        raise AppearanceVariantInputError(
            f"{owner} lineage does not bind its exact snapshot"
        )
    return current


def _template_snapshot_binding(lineage: Mapping[str, Any]) -> dict[str, Any]:
    current = _lineage_json_input(
        lineage,
        key="template_variant_spec",
        owner="template variant spec",
    )
    try:
        validated = load_variant_package_spec(current.file.path)
    except VariantPackageError as exc:
        raise AppearanceVariantInputError(
            f"lineage template variant spec is invalid: {exc}"
        ) from exc
    if validated.sha256 != current.file.sha256 or validated.value != current.value:
        raise AppearanceVariantInputError(
            "template variant spec parser identity differs from lineage bytes"
        )
    return deepcopy(dict(current.value))


def _expected_spec_from_lineage(
    template: dict[str, Any], instance: Mapping[str, Any]
) -> dict[str, Any]:
    if set(instance) != {
        "instance_request_id",
        "request_sha256",
        "ordinal",
        "taxonomy",
        "attributes",
    }:
        raise AppearanceVariantInputError("lineage instance request fields are invalid")
    if (
        not isinstance(instance.get("instance_request_id"), str)
        or not instance["instance_request_id"]
    ):
        raise AppearanceVariantInputError(
            "lineage instance_request_id must be non-empty"
        )
    request_sha256 = _lower_sha256(
        instance.get("request_sha256"), owner="lineage instance request_sha256"
    )
    ordinal = instance.get("ordinal")
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or not 1 <= ordinal <= 9
    ):
        raise AppearanceVariantInputError(
            "lineage instance ordinal must be an integer in [1, 9]"
        )
    taxonomy = _mapping(instance.get("taxonomy"), owner="lineage taxonomy")
    attributes = _mapping(instance.get("attributes"), owner="lineage attributes")
    if set(attributes) != {"size", "body_build", "coat_profile", "life_stage"} or any(
        not isinstance(value, str) or not value for value in attributes.values()
    ):
        raise AppearanceVariantInputError("lineage appearance attributes are invalid")
    request = {"taxonomy": taxonomy, "attributes": attributes}
    _validate_taxonomy(template_value=template, request=request)

    identity = _mapping(template.get("identity"), owner="lineage template identity")
    if set(identity) != _IDENTITY_FIELDS:
        raise AppearanceVariantInputError(
            "lineage template identity fields are invalid"
        )
    template_id = identity.get("template_id")
    if not isinstance(template_id, str) or not template_id:
        raise AppearanceVariantInputError("lineage template_id must be non-empty")
    derivative = request_sha256[:12]
    identity["asset_id"] = f"{template_id}_appearance_l9_{ordinal:02d}_{derivative}"
    for field in ("skeleton_revision", "collision_revision", "action_revision"):
        revision = identity.get(field)
        if not isinstance(revision, str) or not revision:
            raise AppearanceVariantInputError(
                f"lineage template {field} must be non-empty"
            )
        identity[field] = f"{revision}-appearance-{derivative}"
    template["appearance"] = {
        "size": attributes["size"],
        "body_build": attributes["body_build"],
        "coat": attributes["coat_profile"],
        "life_stage": attributes["life_stage"],
    }
    template["schema"] = SPEC_SCHEMA
    return template


def validate_spec_lineage_binding(
    spec: Mapping[str, Any],
    lineage: Mapping[str, Any],
    *,
    serialized_spec: bytes | None = None,
) -> None:
    """Validate one exact derived spec/lineage pair without writing files."""

    payload = _json_payload(spec) if serialized_spec is None else bytes(serialized_spec)
    if _decode_json_object(payload, owner="serialized derived spec") != spec:
        raise AppearanceVariantInputError(
            "serialized derived spec bytes differ from the supplied spec"
        )
    if spec.get("schema") != SPEC_SCHEMA:
        raise AppearanceVariantInputError("derived spec schema is invalid")
    if lineage.get("schema") != LINEAGE_SCHEMA:
        raise AppearanceVariantInputError("appearance lineage schema is invalid")
    inputs = _mapping(lineage.get("inputs"), owner="lineage.inputs")
    derivative = _mapping(lineage.get("derivative"), owner="lineage.derivative")
    if set(lineage) != {
        "schema",
        "status",
        "qualification_state",
        "qualification_claim",
        "formal_dataset_registration_authorized",
        "instance_request",
        "inputs",
        "derivative",
        "decision_reason",
        "lineage_content_sha256",
    }:
        raise AppearanceVariantInputError("appearance lineage root fields are invalid")
    if set(inputs) != {
        "appearance_batch",
        "appearance_realization_report",
        "template_variant_spec",
        "upstream_source_manifest",
    }:
        raise AppearanceVariantInputError("appearance lineage input fields are invalid")
    if set(derivative) != {
        "pre_rebase_visual_glb",
        "tool_identity",
        "derived_variant_spec",
    }:
        raise AppearanceVariantInputError(
            "appearance lineage derivative fields are invalid"
        )
    if (
        lineage.get("status") != "pass"
        or lineage.get("qualification_state") != "research_candidate"
        or lineage.get("qualification_claim") is not False
        or lineage.get("formal_dataset_registration_authorized") is not False
    ):
        raise AppearanceVariantInputError(
            "appearance lineage cannot claim qualification"
        )
    if (
        not isinstance(lineage.get("decision_reason"), str)
        or not lineage["decision_reason"].strip()
    ):
        raise AppearanceVariantInputError(
            "appearance lineage decision_reason is invalid"
        )
    expected_lineage_sha = canonical_json_sha256(
        {
            key: value
            for key, value in lineage.items()
            if key != "lineage_content_sha256"
        }
    )
    if lineage.get("lineage_content_sha256") != expected_lineage_sha:
        raise AppearanceVariantInputError("lineage_content_sha256 is invalid")
    instance = _mapping(
        lineage.get("instance_request"), owner="lineage.instance_request"
    )
    batch_snapshot = _lineage_json_input(
        lineage, key="appearance_batch", owner="appearance batch"
    )
    report_snapshot = _lineage_json_input(
        lineage,
        key="appearance_realization_report",
        owner="appearance realization report",
    )
    upstream_snapshot = _lineage_json_input(
        lineage,
        key="upstream_source_manifest",
        owner="upstream source manifest",
    )
    try:
        validate_l9_batch(batch_snapshot.value)
    except ValueError as exc:
        raise AppearanceVariantInputError(
            f"lineage appearance batch is invalid: {exc}"
        ) from exc
    ordinal = instance.get("ordinal")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        raise AppearanceVariantInputError("lineage instance ordinal is invalid")
    request = _request(batch_snapshot.value, ordinal)
    expected_instance = {
        "instance_request_id": request.get("instance_request_id"),
        "request_sha256": request.get("request_sha256"),
        "ordinal": ordinal,
        "taxonomy": request.get("taxonomy"),
        "attributes": request.get("attributes"),
    }
    if instance != expected_instance:
        raise AppearanceVariantInputError(
            "lineage instance request differs from its authenticated batch"
        )
    visual = _validate_realization(
        report_snapshot.value,
        batch_snapshot=batch_snapshot,
        batch=batch_snapshot.value,
        request=request,
    )
    template = _template_snapshot_binding(lineage)
    _validate_upstream_source(upstream_snapshot.value, template_value=template)
    visual_binding = _mapping(
        derivative.get("pre_rebase_visual_glb"),
        owner="lineage.derivative.pre_rebase_visual_glb",
    )
    if set(visual_binding) != {"path", "byte_size", "sha256"}:
        raise AppearanceVariantInputError(
            "lineage pre-rebase visual fields are invalid"
        )
    _validate_record(visual_binding, visual, owner="lineage pre-rebase visual")
    if derivative.get("tool_identity") != report_snapshot.value.get("tool_identity"):
        raise AppearanceVariantInputError(
            "lineage derivative tool identity differs from its report"
        )
    expected_spec = _expected_spec_from_lineage(template, instance)
    if spec != expected_spec:
        raise AppearanceVariantInputError(
            "spec is not the exact derived template/request result"
        )
    identity = _mapping(spec.get("identity"), owner="derived spec identity")
    binding = _mapping(
        derivative.get("derived_variant_spec"),
        owner="lineage.derivative.derived_variant_spec",
    )
    if binding != {
        "schema": SPEC_SCHEMA,
        "asset_id": identity.get("asset_id"),
        "byte_size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "canonical_content_sha256": canonical_json_sha256(spec),
    }:
        raise AppearanceVariantInputError(
            "lineage does not bind the exact derived variant spec"
        )
    appearance = _mapping(spec.get("appearance"), owner="derived spec appearance")
    attributes = _mapping(instance.get("attributes"), owner="lineage attributes")
    if appearance != {
        "size": attributes.get("size"),
        "body_build": attributes.get("body_build"),
        "coat": attributes.get("coat_profile"),
        "life_stage": attributes.get("life_stage"),
    }:
        raise AppearanceVariantInputError(
            "spec appearance and lineage request attributes are mismatched"
        )


def _output_path(path: str | Path, *, owner: str) -> Path:
    output = _absolute_without_symlinks(path, owner=owner)
    if output.suffix.lower() != ".json":
        raise AppearanceVariantInputError(f"{owner} must use a .json suffix")
    if output.exists() or output.is_symlink():
        raise AppearanceVariantInputError(f"refusing to replace {owner}: {output}")
    return output


def _reserve_output(path: Path, *, owner: str) -> _OutputReservation:
    parent = _open_directory_chain(path.parent, owner=f"{owner} parent", create=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path.name, flags, 0o666, dir_fd=parent)
    except OSError as exc:
        os.close(parent)
        raise AppearanceVariantInputError(f"unable to reserve {owner}: {exc}") from exc
    try:
        stream = os.fdopen(descriptor, "wb")
    except Exception:
        os.close(descriptor)
        try:
            os.unlink(path.name, dir_fd=parent)
        finally:
            os.close(parent)
        raise
    try:
        identity = os.fstat(stream.fileno())
    except Exception:
        stream.close()
        try:
            os.unlink(path.name, dir_fd=parent)
        except FileNotFoundError:
            pass
        finally:
            os.close(parent)
        raise
    return _OutputReservation(
        path=path,
        name=path.name,
        stream=stream,
        parent_descriptor=parent,
        device=identity.st_dev,
        inode=identity.st_ino,
    )


def _cleanup_created(created: Sequence[_OutputReservation]) -> list[str]:
    errors: list[str] = []
    for reservation in created:
        try:
            current = os.stat(
                reservation.name,
                dir_fd=reservation.parent_descriptor,
                follow_symlinks=False,
            )
            if (
                current.st_dev == reservation.device
                and current.st_ino == reservation.inode
            ):
                os.unlink(reservation.name, dir_fd=reservation.parent_descriptor)
        except FileNotFoundError:
            pass
        except OSError as exc:
            errors.append(f"{reservation.path}: {exc}")
        finally:
            os.close(reservation.parent_descriptor)
    return errors


def _write_output_pair(
    *,
    spec_output: str | Path,
    lineage_output: str | Path,
    spec: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> tuple[Path, Path]:
    spec_payload = _json_payload(spec)
    lineage_payload = _json_payload(lineage)
    validate_spec_lineage_binding(spec, lineage, serialized_spec=spec_payload)
    spec_path = _output_path(spec_output, owner="variant spec output")
    lineage_path = _output_path(lineage_output, owner="appearance lineage output")
    if spec_path == lineage_path:
        raise AppearanceVariantInputError("spec and lineage outputs must differ")
    try:
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        lineage_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AppearanceVariantInputError(
            f"unable to prepare output directories: {exc}"
        ) from exc
    for path, owner in (
        (spec_path, "variant spec output"),
        (lineage_path, "appearance lineage output"),
    ):
        if _absolute_without_symlinks(path, owner=owner) != path:
            raise AppearanceVariantInputError(f"{owner} path changed before emission")

    outputs = (
        (spec_path, spec_payload),
        (lineage_path, lineage_payload),
    )
    reservations: list[_OutputReservation] = []
    try:
        for path, _ in outputs:
            reservations.append(_reserve_output(path, owner=f"output {path.name}"))
        for reservation, (_, payload) in zip(reservations, outputs, strict=True):
            reservation.stream.write(payload)
            reservation.stream.flush()
            os.fsync(reservation.stream.fileno())
        for reservation in reservations:
            reservation.stream.close()

        spec_readback = _snapshot_json(spec_path, "emitted variant spec")
        lineage_readback = _snapshot_json(lineage_path, "emitted appearance lineage")
        if (
            spec_readback.file.payload != spec_payload
            or lineage_readback.file.payload != lineage_payload
            or spec_readback.value != spec
            or lineage_readback.value != lineage
        ):
            raise AppearanceVariantInputError(
                "output pair readback differs from inputs"
            )
        validated = load_variant_package_spec(spec_path)
        if validated.sha256 != spec_readback.file.sha256 or validated.value != spec:
            raise AppearanceVariantInputError(
                "generic package parser disagrees with emitted variant spec"
            )
        validate_spec_lineage_binding(
            spec_readback.value,
            lineage_readback.value,
            serialized_spec=spec_readback.file.payload,
        )
    except Exception as exc:
        for reservation in reservations:
            if not reservation.stream.closed:
                reservation.stream.close()
        cleanup_errors = _cleanup_created(reservations)
        suffix = f"; cleanup failed: {cleanup_errors}" if cleanup_errors else ""
        if isinstance(exc, AppearanceVariantInputError):
            raise AppearanceVariantInputError(f"{exc}{suffix}") from exc
        raise AppearanceVariantInputError(
            f"unable to emit output pair: {exc}{suffix}"
        ) from exc
    for reservation in reservations:
        os.close(reservation.parent_descriptor)
    return spec_path, lineage_path


def build_inputs(
    *,
    batch_path: str | Path,
    ordinal: int,
    appearance_report_path: str | Path,
    template_spec_path: str | Path,
    upstream_source_manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one variant spec and a snapshot-complete non-claiming lineage."""

    if (
        not isinstance(ordinal, int)
        or isinstance(ordinal, bool)
        or not 1 <= ordinal <= 9
    ):
        raise AppearanceVariantInputError("ordinal must be an integer in [1, 9]")
    batch_snapshot = _snapshot_json(batch_path, "appearance batch")
    report_snapshot = _snapshot_json(appearance_report_path, "appearance report")
    source_snapshot = _snapshot_json(
        upstream_source_manifest_path, "upstream source manifest"
    )
    template_snapshot = _snapshot_json(template_spec_path, "template variant spec")
    batch = batch_snapshot.value
    validate_l9_batch(batch)
    request = _request(batch, ordinal)
    report = report_snapshot.value
    visual = _validate_realization(
        report,
        batch_snapshot=batch_snapshot,
        batch=batch,
        request=request,
    )
    try:
        template = load_variant_package_spec(template_snapshot.file.path)
    except VariantPackageError as exc:
        raise AppearanceVariantInputError(
            f"template variant spec is invalid: {exc}"
        ) from exc
    if (
        template.sha256 != template_snapshot.file.sha256
        or template.value != template_snapshot.value
    ):
        raise AppearanceVariantInputError(
            "template parser identity differs from strict template bytes"
        )
    template_value = deepcopy(dict(template.value))
    _validate_taxonomy(template_value=template_value, request=request)
    _validate_upstream_source(source_snapshot.value, template_value=template_value)

    derivative = request["request_sha256"][:12]
    identity = template_value["identity"]
    identity["asset_id"] = (
        f"{identity['template_id']}_appearance_l9_{ordinal:02d}_{derivative}"
    )
    for field in ("skeleton_revision", "collision_revision", "action_revision"):
        identity[field] = f"{identity[field]}-appearance-{derivative}"
    attributes = request["attributes"]
    template_value["appearance"] = {
        "size": attributes["size"],
        "body_build": attributes["body_build"],
        "coat": attributes["coat_profile"],
        "life_stage": attributes["life_stage"],
    }
    template_value["schema"] = SPEC_SCHEMA
    validated_identity = _validate_generated_spec(
        template_value,
        request=request,
        known_joint_ids=_visual_joint_ids(visual),
    )
    spec_payload = _json_payload(template_value)

    lineage_core: dict[str, Any] = {
        "schema": LINEAGE_SCHEMA,
        "status": "pass",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "formal_dataset_registration_authorized": False,
        "instance_request": {
            "instance_request_id": request["instance_request_id"],
            "request_sha256": request["request_sha256"],
            "ordinal": ordinal,
            "taxonomy": deepcopy(request["taxonomy"]),
            "attributes": deepcopy(request["attributes"]),
        },
        "inputs": {
            "appearance_batch": _record(batch_snapshot, include_snapshot=True),
            "appearance_realization_report": _record(
                report_snapshot, include_snapshot=True
            ),
            "template_variant_spec": _record(template_snapshot, include_snapshot=True),
            "upstream_source_manifest": _record(source_snapshot, include_snapshot=True),
        },
        "derivative": {
            "pre_rebase_visual_glb": _record(visual),
            "tool_identity": deepcopy(report.get("tool_identity")),
            "derived_variant_spec": {
                "schema": SPEC_SCHEMA,
                "asset_id": validated_identity.asset_id,
                "byte_size": len(spec_payload),
                "sha256": hashlib.sha256(spec_payload).hexdigest(),
                "canonical_content_sha256": canonical_json_sha256(template_value),
            },
        },
        "decision_reason": (
            "The derived research asset is bound to one authenticated L9 request, "
            "strict Blender output readback, and the upstream source snapshot."
        ),
    }
    lineage_core["lineage_content_sha256"] = canonical_json_sha256(lineage_core)
    validate_spec_lineage_binding(
        template_value, lineage_core, serialized_spec=spec_payload
    )
    return template_value, lineage_core


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--ordinal", type=int, required=True)
    parser.add_argument("--appearance-report", type=Path, required=True)
    parser.add_argument("--template-spec", type=Path, required=True)
    parser.add_argument("--upstream-source-manifest", type=Path, required=True)
    parser.add_argument("--spec-output", type=Path, required=True)
    parser.add_argument("--lineage-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        spec, lineage = build_inputs(
            batch_path=args.batch,
            ordinal=args.ordinal,
            appearance_report_path=args.appearance_report,
            template_spec_path=args.template_spec,
            upstream_source_manifest_path=args.upstream_source_manifest,
        )
        spec_path, lineage_path = _write_output_pair(
            spec_output=args.spec_output,
            lineage_output=args.lineage_output,
            spec=spec,
            lineage=lineage,
        )
        spec_readback = _snapshot_file(spec_path, "emitted variant spec")
        lineage_readback = _snapshot_file(lineage_path, "emitted appearance lineage")
    except (
        OSError,
        ValueError,
        VariantPackageError,
        AppearanceVariantInputError,
    ) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": "pass",
                "qualification_state": "research_candidate",
                "qualification_claim": False,
                "asset_id": spec["identity"]["asset_id"],
                "spec": str(spec_path),
                "spec_sha256": spec_readback.sha256,
                "lineage": str(lineage_path),
                "lineage_sha256": lineage_readback.sha256,
                "lineage_content_sha256": lineage["lineage_content_sha256"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
