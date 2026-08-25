#!/usr/bin/env python3
"""Realize one hash-bound animal appearance request without changing its rig.

Run with Blender, for example::

    blender --background --python tools/blender/realize_animal_appearance.py -- \
      --batch tmp/m2/beagle_appearance_l9_v1.json --ordinal 1 \
      --output-glb tmp/m2/beagle_l9/01/appearance.glb \
      --report tmp/m2/beagle_l9/01/appearance_report.json

The operation is deliberately research-only.  It changes rest-mesh shape and
the base-colour raster according to the authenticated request, and bakes the
requested uniform size into both mesh and armature data.  It never creates an
ancestor scale node, re-rigs the animal, edits skin weights, or silently picks
an animation fallback.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import types
from typing import Any, Iterable

import bpy
from mathutils import Matrix
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def _namespace_package(name: str, path: Path) -> None:
    """Register a source-only namespace without importing package __init__."""

    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


def _load_repository_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load repository module {name}: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Blender's embedded Python intentionally has no project dependency install.
# Load the four pure-Python compiler modules by exact repository path while
# avoiding ``avengine.assets.__init__`` (which pulls the JSON-schema dependency).
_namespace_package("avengine", SOURCE_ROOT / "avengine")
_namespace_package("avengine.contracts", SOURCE_ROOT / "avengine" / "contracts")
_namespace_package("avengine.assets", SOURCE_ROOT / "avengine" / "assets")
_load_repository_module(
    "avengine.contracts.json_io",
    SOURCE_ROOT / "avengine" / "contracts" / "json_io.py",
)
_GLB_MODULE_PATH = SOURCE_ROOT / "avengine" / "assets" / "glb.py"
_GLB_MODULE = _load_repository_module("avengine.assets.glb", _GLB_MODULE_PATH)
_load_repository_module(
    "avengine.assets.glb_write", SOURCE_ROOT / "avengine" / "assets" / "glb_write.py"
)
_MATERIALS_MODULE = _load_repository_module(
    "avengine.assets.materials", SOURCE_ROOT / "avengine" / "assets" / "materials.py"
)
GlbDocument = _GLB_MODULE.GlbDocument
extract_actions = _GLB_MODULE.extract_actions
extract_skins = _GLB_MODULE.extract_skins
load_glb = _GLB_MODULE.load_glb
MAXIMUM_SPECULAR_FACTOR = _MATERIALS_MODULE.MAXIMUM_SPECULAR_FACTOR
MAXIMUM_SPECULAR_COLOR_FACTOR = _MATERIALS_MODULE.MAXIMUM_SPECULAR_COLOR_FACTOR
MINIMUM_ROUGHNESS_FACTOR = _MATERIALS_MODULE.MINIMUM_ROUGHNESS_FACTOR
ZERO_EMISSIVE_FACTOR = _MATERIALS_MODULE.ZERO_EMISSIVE_FACTOR
normalize_glb_materials = _MATERIALS_MODULE.normalize_glb_materials


SCHEMA = "avengine_animal_appearance_realization_v1"
AXES = ("size", "body_build", "coat_profile", "life_stage")
OPERATIONS = {
    "size": "uniform_actor_scale_v1",
    "body_build": "semantic_torso_girth_scale_v1",
    "coat_profile": "breed_scoped_coat_luminance_v1",
    "life_stage": "semantic_life_stage_cues_v1",
}
REGISTERED_COAT_PATTERNS = {
    ("dog", "beagle", "dog_beagle_tricolor_v1"): "tricolor",
}


def _arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--ordinal", type=int, choices=range(1, 10), required=True)
    parser.add_argument("--output-glb", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-texture", type=Path)
    return parser.parse_args(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _regular(path: Path, owner: str) -> Path:
    if path.is_symlink():
        raise RuntimeError(f"{owner} must not be a symbolic link: {path}")
    resolved = path.resolve(strict=True)
    cursor = Path(resolved.anchor)
    for part in resolved.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise RuntimeError(f"{owner} path contains a symbolic link: {cursor}")
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise RuntimeError(f"{owner} is not a safe non-empty regular file: {resolved}")
    return resolved


def _new_output(path: Path, owner: str) -> Path:
    resolved = path.resolve()
    if path.exists() or path.is_symlink() or resolved.exists():
        raise RuntimeError(f"refusing to replace {owner}: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _load_object(path: Path, owner: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load {owner}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{owner} must be a JSON object")
    return value


def _validate_batch(
    batch_path: Path, ordinal: int
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    batch = _load_object(batch_path, "appearance batch")
    if batch.get("schema") != "avengine_animal_appearance_batch_v1":
        raise RuntimeError("appearance batch schema differs")
    if batch.get("state_classification") != "research_candidate":
        raise RuntimeError("appearance batch must remain a research candidate")
    core = {
        key: deepcopy(value)
        for key, value in batch.items()
        if key not in {"batch_id", "batch_content_sha256"}
    }
    digest = _canonical_sha256(core)
    if batch.get("batch_content_sha256") != digest:
        raise RuntimeError("batch_content_sha256 does not authenticate the batch")
    if not str(batch.get("batch_id", "")).endswith(f"_{digest[:12]}"):
        raise RuntimeError("batch_id does not match batch_content_sha256")

    source_record = batch.get("source_asset")
    if not isinstance(source_record, dict):
        raise RuntimeError("batch source_asset is invalid")
    raw_source = source_record.get("path")
    if not isinstance(raw_source, str) or not raw_source:
        raise RuntimeError("batch source_asset.path is invalid")
    source = _regular(Path(raw_source), "batch source GLB")
    if source.stat().st_size != source_record.get("byte_size") or _sha256(
        source
    ) != source_record.get("sha256"):
        raise RuntimeError("batch source GLB bytes changed")

    requests = batch.get("requests")
    if not isinstance(requests, list) or len(requests) != 9:
        raise RuntimeError("appearance batch must contain exactly nine requests")
    matches = [item for item in requests if item.get("ordinal") == ordinal]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise RuntimeError(f"batch does not contain unique ordinal {ordinal}")
    request = matches[0]
    request_core = {
        key: deepcopy(value)
        for key, value in request.items()
        if key not in {"instance_request_id", "request_sha256"}
    }
    request_digest = _canonical_sha256(request_core)
    if request.get("request_sha256") != request_digest:
        raise RuntimeError("request_sha256 does not authenticate the instance request")
    expected_suffix = f"_{ordinal:02d}_{request_digest[:12]}"
    if not str(request.get("instance_request_id", "")).endswith(expected_suffix):
        raise RuntimeError("instance_request_id does not match request content")
    if request.get("source_asset_sha256") != source_record.get("sha256"):
        raise RuntimeError("instance request is not bound to the batch source")

    operations = request.get("realization_operations")
    if not isinstance(operations, list) or [
        item.get("attribute") for item in operations if isinstance(item, dict)
    ] != list(AXES):
        raise RuntimeError("instance request does not contain the four ordered axes")
    for operation in operations:
        axis = operation["attribute"]
        if operation.get("operation_id") != OPERATIONS[axis]:
            raise RuntimeError(f"unexpected realization operation for {axis}")
        if operation.get("selected_value") != request.get("attributes", {}).get(axis):
            raise RuntimeError(f"selected value differs from attributes.{axis}")
        if not isinstance(operation.get("parameters"), dict):
            raise RuntimeError(f"realization parameters for {axis} are invalid")
    _registered_preserve_pattern(request)
    return batch, request, source


def _registered_preserve_pattern(request: dict[str, Any]) -> str:
    taxonomy = request.get("taxonomy")
    key = (
        taxonomy.get("species") if isinstance(taxonomy, dict) else None,
        taxonomy.get("breed") if isinstance(taxonomy, dict) else None,
        request.get("coat_profile_id"),
    )
    expected = REGISTERED_COAT_PATTERNS.get(key)
    if expected is None:
        raise RuntimeError(f"appearance coat profile is not registered: {key!r}")
    operations = request.get("realization_operations")
    coats = [
        item
        for item in operations
        if isinstance(item, dict) and item.get("attribute") == "coat_profile"
    ]
    parameters = coats[0].get("parameters") if len(coats) == 1 else None
    if not isinstance(parameters, dict) or parameters.get("preserve_pattern") != expected:
        raise RuntimeError(
            "coat preserve_pattern differs from the exact registered profile value"
        )
    return expected


def _clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _import_source(path: Path) -> tuple[Any, Any]:
    result = bpy.ops.import_scene.gltf(filepath=str(path))
    if "FINISHED" not in result:
        raise RuntimeError(f"GLB import failed: {result}")
    armatures = [item for item in bpy.context.scene.objects if item.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(
            f"source must import as one armature, found {[item.name for item in armatures]}"
        )
    armature = armatures[0]
    skinned = [
        item
        for item in bpy.context.scene.objects
        if item.type == "MESH"
        and any(
            modifier.type == "ARMATURE" and modifier.object == armature
            for modifier in item.modifiers
        )
    ]
    if len(skinned) != 1:
        raise RuntimeError(
            "appearance realization currently requires exactly one skinned mesh; "
            f"found {[item.name for item in skinned]}"
        )
    mesh = skinned[0]
    for item in list(bpy.context.scene.objects):
        if item not in {armature, mesh}:
            bpy.data.objects.remove(item, do_unlink=True)
    if len(mesh.data.vertices) < 100 or not mesh.data.polygons:
        raise RuntimeError("primary skinned mesh is unexpectedly small")
    return armature, mesh


def _canonical_actions(armature: Any) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for action in list(bpy.data.actions):
        lower = action.name.casefold()
        semantic = (
            "walk"
            if lower == "walking" or lower.startswith("walking_")
            else "idle"
            if lower == "idle" or lower.startswith("idle_")
            else None
        )
        if semantic is None:
            continue
        if semantic in selected:
            raise RuntimeError(f"ambiguous {semantic} actions in source")
        selected[semantic] = action
    if set(selected) != {"idle", "walk"}:
        raise RuntimeError(
            f"source must expose unique Idle/Walking actions: {[a.name for a in bpy.data.actions]}"
        )
    for action in list(bpy.data.actions):
        if action not in selected.values():
            bpy.data.actions.remove(action)
    selected["idle"].name = "__AVENGINE_IDLE__"
    selected["walk"].name = "__AVENGINE_WALK__"
    selected["idle"].name = "Idle"
    selected["walk"].name = "Walking"
    armature.animation_data_create()
    armature.animation_data.action = None
    while armature.animation_data.nla_tracks:
        armature.animation_data.nla_tracks.remove(armature.animation_data.nla_tracks[0])
    return selected


def _topology_uv_skin_sha256(mesh: Any) -> str:
    digest = hashlib.sha256()
    for polygon in mesh.data.polygons:
        digest.update(struct.pack("<I", len(polygon.vertices)))
        digest.update(struct.pack(f"<{len(polygon.vertices)}I", *polygon.vertices))
    for layer in mesh.data.uv_layers:
        digest.update(layer.name.encode("utf-8"))
        for item in layer.data:
            digest.update(struct.pack("<2d", *map(float, item.uv)))
    for group in mesh.vertex_groups:
        digest.update(group.name.encode("utf-8"))
    for vertex in mesh.data.vertices:
        for membership in sorted(vertex.groups, key=lambda item: item.group):
            digest.update(
                struct.pack("<Id", int(membership.group), float(membership.weight))
            )
    return digest.hexdigest()


def _action_curve_sha256(actions: Iterable[Any]) -> str:
    """Hash authored curve values while deliberately excluding action names."""

    digest = hashlib.sha256()
    for action in sorted(actions, key=lambda item: item.name):
        for curve in sorted(
            action.fcurves, key=lambda item: (item.data_path, item.array_index)
        ):
            digest.update(curve.data_path.encode("utf-8"))
            digest.update(struct.pack("<I", int(curve.array_index)))
            for point in curve.keyframe_points:
                digest.update(struct.pack("<2d", *map(float, point.co)))
    return digest.hexdigest()


def _resolve_group_names(mesh: Any, semantic_names: Any, owner: str) -> tuple[str, ...]:
    if (
        not isinstance(semantic_names, list)
        or not semantic_names
        or any(not isinstance(item, str) or not item for item in semantic_names)
    ):
        raise RuntimeError(f"{owner} must be a non-empty string list")
    available = [group.name for group in mesh.vertex_groups]
    resolved: list[str] = []
    for semantic in semantic_names:
        folded = semantic.casefold()
        matches = [
            name
            for name in available
            if name.casefold() == folded or name.casefold().endswith(" " + folded)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"{owner} semantic group {semantic!r} resolved to {matches}; "
                f"available={available}"
            )
        if matches[0] in resolved:
            raise RuntimeError(f"{owner} resolves duplicate group {matches[0]!r}")
        resolved.append(matches[0])
    return tuple(resolved)


def _weights(mesh: Any, names: tuple[str, ...]) -> np.ndarray:
    indices = {group.index for group in mesh.vertex_groups if group.name in names}
    result = np.zeros(len(mesh.data.vertices), dtype=np.float64)
    for vertex in mesh.data.vertices:
        result[vertex.index] = min(
            1.0,
            sum(
                float(membership.weight)
                for membership in vertex.groups
                if membership.group in indices
            ),
        )
    return result


def _weighted_rms(
    coordinates: np.ndarray, weights: np.ndarray, axes: tuple[int, ...]
) -> float:
    total = float(weights.sum())
    if total <= 1.0e-12:
        raise RuntimeError("semantic weights sum to zero")
    center = np.sum(coordinates * weights[:, None], axis=0) / total
    offsets = coordinates[:, list(axes)] - center[list(axes)]
    return float(np.sqrt(np.sum(weights * np.sum(np.square(offsets), axis=1)) / total))


def _apply_shape(
    mesh: Any,
    *,
    torso_scale: float,
    head_scale: float,
    torso_names: tuple[str, ...],
    head_names: tuple[str, ...],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if not math.isfinite(torso_scale) or not 0.75 <= torso_scale <= 1.25:
        raise RuntimeError("torso_girth_scale is outside the admitted [0.75, 1.25]")
    if not math.isfinite(head_scale) or not 0.85 <= head_scale <= 1.20:
        raise RuntimeError("head_scale is outside the admitted [0.85, 1.20]")
    coordinates = np.empty(len(mesh.data.vertices) * 3, dtype=np.float64)
    mesh.data.vertices.foreach_get("co", coordinates)
    coordinates = coordinates.reshape((-1, 3))
    torso = _weights(mesh, torso_names)
    head = _weights(mesh, head_names)
    torso_count = int(np.count_nonzero(torso > 0.05))
    head_count = int(np.count_nonzero(head > 0.05))
    if torso_count < 50 or head_count < 20:
        raise RuntimeError(
            f"semantic skin coverage is too small: torso={torso_count}, head={head_count}"
        )
    torso_before = _weighted_rms(coordinates, torso, (1, 2))
    head_before = _weighted_rms(coordinates, head, (0, 1, 2))

    torso_center = np.sum(coordinates * torso[:, None], axis=0) / float(torso.sum())
    lateral_factor = 1.0 + (torso_scale - 1.0) * torso
    vertical_factor = 1.0 + (torso_scale - 1.0) * 0.55 * torso
    coordinates[:, 1] = (
        torso_center[1] + (coordinates[:, 1] - torso_center[1]) * lateral_factor
    )
    coordinates[:, 2] = (
        torso_center[2] + (coordinates[:, 2] - torso_center[2]) * vertical_factor
    )

    head_center = np.sum(coordinates * head[:, None], axis=0) / float(head.sum())
    factor = 1.0 + (head_scale - 1.0) * head
    coordinates = head_center + (coordinates - head_center) * factor[:, None]
    torso_after = _weighted_rms(coordinates, torso, (1, 2))
    head_after = _weighted_rms(coordinates, head, (0, 1, 2))
    mesh.data.vertices.foreach_set("co", coordinates.reshape(-1))
    mesh.data.update()
    return (
        {
            "torso_group_names": list(torso_names),
            "head_group_names": list(head_names),
            "torso_selected_vertices": torso_count,
            "head_selected_vertices": head_count,
            "requested_torso_girth_scale": torso_scale,
            "requested_head_scale": head_scale,
            "torso_weighted_yz_rms_before": torso_before,
            "torso_weighted_yz_rms_after": torso_after,
            "torso_weighted_yz_rms_ratio": torso_after / torso_before,
            "head_weighted_radius_rms_before": head_before,
            "head_weighted_radius_rms_after": head_after,
            "head_weighted_radius_rms_ratio": head_after / head_before,
        },
        head,
        coordinates,
    )


def _base_color_node(mesh: Any) -> Any:
    materials = [material for material in mesh.data.materials if material is not None]
    if len(materials) != 1:
        raise RuntimeError(
            "appearance realization v1 requires exactly one material so coat and "
            "muzzle edits cannot silently omit a mesh part"
        )
    linked: list[Any] = []
    fallback: list[Any] = []
    for material in mesh.data.materials:
        if material is None or not material.use_nodes:
            continue
        principled = next(
            (
                node
                for node in material.node_tree.nodes
                if node.type == "BSDF_PRINCIPLED"
            ),
            None,
        )
        if principled is not None:
            base = principled.inputs.get("Base Color")
            if base is not None and base.is_linked:
                for link in base.links:
                    if (
                        link.from_node.type == "TEX_IMAGE"
                        and link.from_node.image is not None
                    ):
                        linked.append(link.from_node)
        for node in material.node_tree.nodes:
            if (
                node.type == "TEX_IMAGE"
                and node.image is not None
                and node.image.colorspace_settings.name == "sRGB"
            ):
                fallback.append(node)
    linked_unique = {id(node): node for node in linked}
    if len(linked_unique) == 1:
        return next(iter(linked_unique.values()))
    if len(linked_unique) > 1:
        raise RuntimeError(
            "skinned mesh has multiple distinct images directly connected to "
            "Principled Base Color"
        )
    fallback_unique = {id(node): node for node in fallback}
    if len(fallback_unique) != 1:
        raise RuntimeError("skinned mesh has no identifiable sRGB base-colour texture")
    return next(iter(fallback_unique.values()))


def _muzzle_mask(
    mesh: Any,
    head_weights: np.ndarray,
    coordinates: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    indices = np.flatnonzero(head_weights > 0.05)
    head_x = coordinates[indices, 0]
    threshold = float(np.quantile(head_x, 0.58))
    maximum = float(head_x.max())
    span = max(maximum - threshold, 1.0e-9)
    vertex_mask = np.clip(
        head_weights * np.clip((coordinates[:, 0] - threshold) / span, 0.0, 1.0),
        0.0,
        1.0,
    )
    if mesh.data.uv_layers.active is None:
        raise RuntimeError("skinned mesh has no active UV map")
    mask = np.zeros((height, width), dtype=np.float32)
    uv_data = mesh.data.uv_layers.active.data
    all_uv = np.asarray([item.uv[:] for item in uv_data], dtype=np.float64)
    uv_minimum = float(np.min(all_uv))
    uv_maximum = float(np.max(all_uv))
    if uv_minimum < -1.0e-6 or uv_maximum > 1.0 + 1.0e-6:
        raise RuntimeError(
            "muzzle rasterizer only admits non-tiled UVs in [0, 1]; "
            f"measured [{uv_minimum:.9g}, {uv_maximum:.9g}]"
        )
    for polygon in mesh.data.polygons:
        if len(polygon.loop_indices) != 3:
            raise RuntimeError("muzzle rasterizer requires a triangulated mesh")
        vertices = list(polygon.vertices)
        weights = vertex_mask[vertices]
        if float(weights.max()) <= 0.01:
            continue
        uv = np.asarray(
            [uv_data[index].uv[:] for index in polygon.loop_indices], dtype=np.float64
        )
        uv = np.clip(uv, 0.0, 1.0)
        points = np.column_stack((uv[:, 0] * (width - 1), uv[:, 1] * (height - 1)))
        x0 = max(0, int(math.floor(points[:, 0].min())))
        x1 = min(width - 1, int(math.ceil(points[:, 0].max())))
        y0 = max(0, int(math.floor(points[:, 1].min())))
        y1 = min(height - 1, int(math.ceil(points[:, 1].max())))
        denominator = (points[1, 1] - points[2, 1]) * (points[0, 0] - points[2, 0]) + (
            points[2, 0] - points[1, 0]
        ) * (points[0, 1] - points[2, 1])
        if abs(denominator) < 1.0e-8:
            continue
        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        b0 = (
            (points[1, 1] - points[2, 1]) * (xx - points[2, 0])
            + (points[2, 0] - points[1, 0]) * (yy - points[2, 1])
        ) / denominator
        b1 = (
            (points[2, 1] - points[0, 1]) * (xx - points[2, 0])
            + (points[0, 0] - points[2, 0]) * (yy - points[2, 1])
        ) / denominator
        b2 = 1.0 - b0 - b1
        inside = (b0 >= -1.0e-5) & (b1 >= -1.0e-5) & (b2 >= -1.0e-5)
        interpolated = b0 * weights[0] + b1 * weights[1] + b2 * weights[2]
        region = mask[y0 : y1 + 1, x0 : x1 + 1]
        np.maximum(region, np.where(inside, interpolated, 0.0), out=region)
    if int(np.count_nonzero(mask > 0.01)) < 32:
        raise RuntimeError("semantic muzzle mask is unexpectedly empty")
    return mask, {
        "muzzle_forward_quantile": 0.58,
        "muzzle_mask_nonzero_pixels": int(np.count_nonzero(mask > 0.01)),
        "muzzle_mask_max": float(mask.max()),
        "uv_minimum": uv_minimum,
        "uv_maximum": uv_maximum,
        "uv_addressing_assumption": "non_tiled_clamp_0_1",
    }


def _realize_texture(
    mesh: Any,
    texture_path: Path,
    *,
    luminance_gain: float,
    coat_desaturation: float,
    muzzle_gray_mix: float,
    muzzle_gray_target: float,
    preserve_pattern: str,
    head_weights: np.ndarray,
    coordinates: np.ndarray,
) -> dict[str, Any]:
    for name, value in (
        ("luminance_gain", luminance_gain),
        ("coat_desaturation", coat_desaturation),
        ("muzzle_gray_mix", muzzle_gray_mix),
        ("muzzle_gray_target", muzzle_gray_target),
    ):
        if not math.isfinite(value):
            raise RuntimeError(f"{name} is not finite")
    if not 0.65 <= luminance_gain <= 1.35:
        raise RuntimeError("luminance_gain is outside the admitted [0.65, 1.35]")
    if not 0.0 <= coat_desaturation <= 1.0:
        raise RuntimeError("coat_desaturation must be in [0, 1]")
    if not 0.0 <= muzzle_gray_mix <= 1.0 or not 0.0 <= muzzle_gray_target <= 1.0:
        raise RuntimeError("muzzle gray controls must be in [0, 1]")
    if preserve_pattern != "tricolor":
        raise RuntimeError(
            "Beagle coat realizer requires preserve_pattern='tricolor'"
        )
    node = _base_color_node(mesh)
    source = node.image
    width, height = map(int, source.size)
    if width <= 0 or height <= 0:
        raise RuntimeError("base-colour texture is not loaded")
    pixels = np.empty(width * height * 4, dtype=np.float32)
    source.pixels.foreach_get(pixels)
    pixels = pixels.reshape((height, width, 4))
    rgb = pixels[:, :, :3]
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = (maximum - minimum) / np.maximum(maximum, 1.0e-6)
    luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    white = (luminance > 0.52) & (saturation < 0.20)
    pigmented = (~white) & (maximum > 0.01)
    if int(np.count_nonzero(pigmented)) < 100:
        raise RuntimeError("base-colour texture has no measurable pigmented coat")
    dark = pigmented & (luminance < 0.12)
    warm = (
        pigmented
        & (rgb[:, :, 0] > rgb[:, :, 2] * 1.15)
        & (rgb[:, :, 0] > rgb[:, :, 1] * 1.03)
    )
    pattern_counts = {
        "white_pixel_count": int(np.count_nonzero(white)),
        "pigmented_pixel_count": int(np.count_nonzero(pigmented)),
        "dark_pixel_count": int(np.count_nonzero(dark)),
        "warm_pixel_count": int(np.count_nonzero(warm)),
    }
    if min(pattern_counts.values()) < 100:
        raise RuntimeError(
            "source texture lacks measurable white/dark/warm tricolor regions"
        )
    pattern_mask_sha256 = hashlib.sha256(
        np.packbits(white.reshape(-1)).tobytes()
        + np.packbits(dark.reshape(-1)).tobytes()
        + np.packbits(warm.reshape(-1)).tobytes()
    ).hexdigest()
    before = float(np.mean(luminance[pigmented]))
    rgb[:] = np.clip(
        rgb * (1.0 + (luminance_gain - 1.0) * pigmented[:, :, None]),
        0.0,
        1.0,
    )
    if coat_desaturation > 0.0:
        adjusted = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        gray = np.repeat(adjusted[:, :, None], 3, axis=2)
        amount = pigmented[:, :, None].astype(np.float32) * coat_desaturation
        rgb[:] = np.clip(rgb * (1.0 - amount) + gray * amount, 0.0, 1.0)

    mask, mask_record = _muzzle_mask(mesh, head_weights, coordinates, width, height)
    if muzzle_gray_mix > 0.0:
        amount = np.clip(mask * muzzle_gray_mix, 0.0, 1.0)[:, :, None]
        target = np.full(rgb.shape, muzzle_gray_target, dtype=np.float32)
        rgb[:] = np.clip(rgb * (1.0 - amount) + target * amount, 0.0, 1.0)
    final_luminance = (
        0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    )
    after = float(np.mean(final_luminance[pigmented]))

    realized = source.copy()
    realized.name = "AVEngineAppearanceBaseColor"
    realized.colorspace_settings.name = source.colorspace_settings.name
    realized.pixels.foreach_set(pixels.reshape(-1))
    realized.update()
    realized.filepath_raw = str(texture_path)
    realized.file_format = "PNG"
    realized.save()
    realized.pack()
    node.image = realized
    return {
        "source_image": source.name,
        "output_texture": str(texture_path),
        "resolution": [width, height],
        "luminance_gain": luminance_gain,
        "coat_desaturation": coat_desaturation,
        "muzzle_gray_mix": muzzle_gray_mix,
        "muzzle_gray_target": muzzle_gray_target,
        "preserve_pattern": preserve_pattern,
        "pattern_audit": {
            "status": "pass",
            "registered_pattern": preserve_pattern,
            "spatial_region_mask_sha256": pattern_mask_sha256,
            "coat_gain_and_desaturation_preserve_region_membership": True,
            **pattern_counts,
        },
        "pigmented_pixel_count": int(np.count_nonzero(pigmented)),
        "mean_pigmented_luminance_before": before,
        "mean_pigmented_luminance_after": after,
        **mask_record,
    }


def _normalise_materials(mesh: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for material in mesh.data.materials:
        if material is None or not material.use_nodes:
            raise RuntimeError("appearance mesh contains a missing/non-node material")
        principled = next(
            (
                node
                for node in material.node_tree.nodes
                if node.type == "BSDF_PRINCIPLED"
            ),
            None,
        )
        if principled is None:
            raise RuntimeError(f"material {material.name!r} lacks Principled BSDF")
        metallic = principled.inputs.get("Metallic")
        roughness = principled.inputs.get("Roughness")
        if metallic is None or roughness is None:
            raise RuntimeError(f"material {material.name!r} lacks PBR inputs")
        before = {
            "metallic": float(metallic.default_value),
            "roughness": float(roughness.default_value),
        }
        if metallic.is_linked:
            raise RuntimeError(
                f"material {material.name!r} has texture-driven metallic; "
                "use the GLB material normalizer instead"
            )
        metallic.default_value = 0.0
        if not roughness.is_linked:
            roughness.default_value = max(
                MINIMUM_ROUGHNESS_FACTOR, float(roughness.default_value)
            )
        coat = principled.inputs.get("Coat Weight")
        if coat is not None and not coat.is_linked:
            coat.default_value = 0.0
        records.append(
            {
                "material": material.name,
                "before": before,
                "after": {
                    "metallic": float(metallic.default_value),
                    "roughness": float(roughness.default_value),
                    "roughness_texture_driven": bool(roughness.is_linked),
                },
            }
        )
    return records


def _bake_uniform_scale(armature: Any, mesh: Any, scale_ratio: float) -> dict[str, Any]:
    if not math.isfinite(scale_ratio) or not 0.70 <= scale_ratio <= 1.30:
        raise RuntimeError("scale_ratio is outside the admitted [0.70, 1.30]")
    for owner, obj in (("armature", armature), ("mesh", mesh)):
        if not np.allclose(np.asarray(obj.scale), np.ones(3), atol=1.0e-6):
            raise RuntimeError(
                f"{owner} object scale must be identity before size baking"
            )
    before_mesh = np.asarray([vertex.co[:] for vertex in mesh.data.vertices])
    armature.data.transform(Matrix.Scale(scale_ratio, 4))
    mesh.data.transform(Matrix.Scale(scale_ratio, 4))
    mesh.data.update()
    bpy.context.view_layer.update()
    after_mesh = np.asarray([vertex.co[:] for vertex in mesh.data.vertices])
    measured = float(np.max(np.abs(after_mesh - before_mesh * scale_ratio)))
    if measured > 1.0e-6:
        raise RuntimeError(f"uniform size bake has vertex error {measured:.9g}")
    if not np.allclose(np.asarray(armature.scale), np.ones(3), atol=1.0e-7):
        raise RuntimeError("size bake created a non-identity armature object scale")
    return {
        "scale_ratio": scale_ratio,
        "strategy": "armature_data_and_mesh_data_matrix_bake_v1",
        "ancestor_scale_node_created": False,
        "maximum_mesh_scale_error": measured,
    }


def _export(path: Path, armature: Any, mesh: Any) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = armature
    result = bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_animations=True,
        export_animation_mode="ACTIONS",
        export_force_sampling=True,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        export_image_format="AUTO",
    )
    if "FINISHED" not in result or not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"GLB export failed: {result}")


def _read_glb_json(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if len(payload) < 20 or payload[:4] != b"glTF":
        raise RuntimeError("appearance output is not a GLB")
    version, declared = struct.unpack_from("<II", payload, 4)
    if version != 2 or declared != len(payload):
        raise RuntimeError("appearance output GLB header differs")
    offset = 12
    documents: list[dict[str, Any]] = []
    while offset < len(payload):
        length, kind = struct.unpack_from("<II", payload, offset)
        offset += 8
        end = offset + length
        if end > len(payload):
            raise RuntimeError("appearance output GLB is truncated")
        if kind == 0x4E4F534A:
            documents.append(json.loads(payload[offset:end].decode("utf-8")))
        offset = end
    if offset != len(payload) or len(documents) != 1:
        raise RuntimeError("appearance output must have one GLB JSON chunk")
    return documents[0]


_COMPONENT_DTYPES = {
    5120: np.dtype("i1"),
    5121: np.dtype("u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_ACCESSOR_WIDTHS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
_OUTPUT_FLOAT_TOLERANCE = 5.0e-5


def _accessor_array(document: GlbDocument, index: int) -> np.ndarray:
    value = document.json
    accessors = value.get("accessors", [])
    views = value.get("bufferViews", [])
    if not isinstance(index, int) or not 0 <= index < len(accessors):
        raise RuntimeError(f"invalid accessor index {index}")
    accessor = accessors[index]
    if "sparse" in accessor or "bufferView" not in accessor:
        raise RuntimeError("strict appearance readback does not admit sparse accessors")
    view_index = accessor["bufferView"]
    if not isinstance(view_index, int) or not 0 <= view_index < len(views):
        raise RuntimeError("accessor bufferView is invalid")
    view = views[view_index]
    if view.get("buffer", 0) != 0:
        raise RuntimeError("strict appearance readback accepts only GLB buffer 0")
    component_type = accessor.get("componentType")
    dtype = _COMPONENT_DTYPES.get(component_type)
    width = _ACCESSOR_WIDTHS.get(accessor.get("type"))
    count = accessor.get("count")
    if dtype is None or width is None or not isinstance(count, int) or count < 0:
        raise RuntimeError(f"unsupported accessor {index}")
    element_bytes = dtype.itemsize * width
    stride = view.get("byteStride", element_bytes)
    if not isinstance(stride, int) or stride < element_bytes:
        raise RuntimeError(f"invalid accessor stride for {index}")
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    result = np.empty((count, width), dtype=dtype)
    for row in range(count):
        offset = start + row * stride
        end = offset + element_bytes
        if offset < 0 or end > len(document.binary):
            raise RuntimeError(f"accessor {index} escapes the GLB BIN chunk")
        result[row] = np.frombuffer(
            document.binary, dtype=dtype, count=width, offset=offset
        )
    if np.issubdtype(dtype, np.floating) and not np.all(np.isfinite(result)):
        raise RuntimeError(f"accessor {index} contains non-finite values")
    return result


def _skinned_primitive(document: GlbDocument) -> tuple[dict[str, Any], dict[str, Any]]:
    value = document.json
    nodes = value.get("nodes", [])
    meshes = value.get("meshes", [])
    candidates = [
        node
        for node in nodes
        if isinstance(node, dict) and "skin" in node and "mesh" in node
    ]
    if len(candidates) != 1 or candidates[0].get("skin") != 0:
        raise RuntimeError("GLB must have exactly one mesh node bound to skin 0")
    mesh_index = candidates[0].get("mesh")
    if not isinstance(mesh_index, int) or not 0 <= mesh_index < len(meshes):
        raise RuntimeError("skinned mesh index is invalid")
    mesh = meshes[mesh_index]
    primitives = mesh.get("primitives") if isinstance(mesh, dict) else None
    if not isinstance(primitives, list) or len(primitives) != 1:
        raise RuntimeError("appearance readback requires one skinned primitive")
    primitive = primitives[0]
    if not isinstance(primitive, dict) or "targets" in primitive:
        raise RuntimeError("appearance readback does not admit morph targets")
    attributes = primitive.get("attributes")
    required = {"POSITION", "NORMAL", "TEXCOORD_0", "JOINTS_0", "WEIGHTS_0"}
    if not isinstance(attributes, dict) or not required.issubset(attributes):
        raise RuntimeError("skinned primitive lacks required PBR/skin attributes")
    if {"JOINTS_1", "WEIGHTS_1"} & set(attributes):
        raise RuntimeError("appearance readback has no silent secondary skin set")
    if "indices" not in primitive:
        raise RuntimeError("skinned primitive must use an index accessor")
    return primitive, attributes


def _maximum_abs(left: np.ndarray, right: np.ndarray, owner: str) -> float:
    if left.shape != right.shape:
        raise RuntimeError(f"{owner} shape changed: {left.shape} != {right.shape}")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _quaternion_error(left: np.ndarray, right: np.ndarray, owner: str) -> float:
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 4:
        raise RuntimeError(f"{owner} quaternion shape changed")
    direct = np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)), axis=1)
    negated = np.max(np.abs(left.astype(np.float64) + right.astype(np.float64)), axis=1)
    return float(np.max(np.minimum(direct, negated))) if len(left) else 0.0


def _joint_names(document: GlbDocument) -> tuple[str, ...]:
    skins = extract_skins(document)
    if len(skins) != 1:
        raise RuntimeError("appearance output must contain exactly one skin")
    names = tuple(joint.name for joint in skins[0].joints)
    if any(name is None for name in names) or len(set(names)) != len(names):
        raise RuntimeError("skin joint names must be unique and non-empty")
    return names  # type: ignore[return-value]


def _expected_positions(
    source: GlbDocument,
    *,
    torso_names: tuple[str, ...],
    head_names: tuple[str, ...],
    torso_scale: float,
    head_scale: float,
    size_scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    primitive, attributes = _skinned_primitive(source)
    del primitive
    positions = _accessor_array(source, attributes["POSITION"]).astype(np.float64)
    joint_ordinals = _accessor_array(source, attributes["JOINTS_0"]).astype(np.int64)
    weights = _accessor_array(source, attributes["WEIGHTS_0"]).astype(np.float64)
    names = _joint_names(source)
    if joint_ordinals.shape != weights.shape or joint_ordinals.shape[1] != 4:
        raise RuntimeError("JOINTS_0/WEIGHTS_0 layout differs")
    if np.any(joint_ordinals < 0) or np.any(joint_ordinals >= len(names)):
        raise RuntimeError("JOINTS_0 contains an invalid joint ordinal")

    def semantic_weights(selected: tuple[str, ...]) -> np.ndarray:
        ordinals = {index for index, name in enumerate(names) if name in selected}
        if len(ordinals) != len(selected):
            raise RuntimeError("semantic group names differ from source skin joints")
        mask = np.isin(joint_ordinals, list(ordinals))
        return np.clip(np.sum(np.where(mask, weights, 0.0), axis=1), 0.0, 1.0)

    torso = semantic_weights(torso_names)
    head = semantic_weights(head_names)
    # glTF +Y-up imports into Blender +Z-up as (x, -z, y).  This is the
    # explicit geometry frame used by the admitted v1 operations.
    blender = np.column_stack((positions[:, 0], -positions[:, 2], positions[:, 1]))
    torso_center = np.sum(blender * torso[:, None], axis=0) / float(torso.sum())
    blender[:, 1] = torso_center[1] + (blender[:, 1] - torso_center[1]) * (
        1.0 + (torso_scale - 1.0) * torso
    )
    blender[:, 2] = torso_center[2] + (blender[:, 2] - torso_center[2]) * (
        1.0 + (torso_scale - 1.0) * 0.55 * torso
    )
    head_center = np.sum(blender * head[:, None], axis=0) / float(head.sum())
    head_factor = 1.0 + (head_scale - 1.0) * head
    blender = head_center + (blender - head_center) * head_factor[:, None]
    blender *= size_scale
    expected = np.column_stack((blender[:, 0], blender[:, 2], -blender[:, 1]))
    return expected, {
        "geometry_frame": {
            "source": "gltf_positive_y_up",
            "blender_import": {
                "forward": "positive_x",
                "lateral": "positive_y",
                "up": "positive_z",
            },
            "basis_formula": "blender=(gltf.x,-gltf.z,gltf.y)",
        },
        "torso_weight_sum": float(torso.sum()),
        "head_weight_sum": float(head.sum()),
    }


def _mesh_readback(
    source: GlbDocument,
    output: GlbDocument,
    *,
    torso_names: tuple[str, ...],
    head_names: tuple[str, ...],
    torso_scale: float,
    head_scale: float,
    size_scale: float,
) -> dict[str, Any]:
    source_primitive, source_attributes = _skinned_primitive(source)
    output_primitive, output_attributes = _skinned_primitive(output)
    indices_source = _accessor_array(source, source_primitive["indices"])
    indices_output = _accessor_array(output, output_primitive["indices"])
    if not np.array_equal(indices_source, indices_output):
        raise RuntimeError("triangle indices changed across appearance export")
    joints_source = _accessor_array(source, source_attributes["JOINTS_0"])
    joints_output = _accessor_array(output, output_attributes["JOINTS_0"])
    if not np.array_equal(joints_source, joints_output):
        raise RuntimeError("JOINTS_0 changed across appearance export")
    uv_error = _maximum_abs(
        _accessor_array(source, source_attributes["TEXCOORD_0"]),
        _accessor_array(output, output_attributes["TEXCOORD_0"]),
        "TEXCOORD_0",
    )
    weight_error = _maximum_abs(
        _accessor_array(source, source_attributes["WEIGHTS_0"]),
        _accessor_array(output, output_attributes["WEIGHTS_0"]),
        "WEIGHTS_0",
    )
    expected, geometry = _expected_positions(
        source,
        torso_names=torso_names,
        head_names=head_names,
        torso_scale=torso_scale,
        head_scale=head_scale,
        size_scale=size_scale,
    )
    actual = _accessor_array(output, output_attributes["POSITION"]).astype(np.float64)
    position_error = _maximum_abs(expected, actual, "POSITION")
    normals = _accessor_array(output, output_attributes["NORMAL"]).astype(np.float64)
    normal_norm_error = float(np.max(np.abs(np.linalg.norm(normals, axis=1) - 1.0)))
    if (
        uv_error > 1.0e-6
        or weight_error > 1.0e-6
        or position_error > _OUTPUT_FLOAT_TOLERANCE
        or normal_norm_error > 5.0e-4
    ):
        raise RuntimeError(
            "appearance mesh output readback failed: "
            f"uv={uv_error:.9g}, weights={weight_error:.9g}, "
            f"position={position_error:.9g}, normal_norm={normal_norm_error:.9g}"
        )
    return {
        "vertex_count": len(actual),
        "index_count": int(indices_output.size),
        "indices_exact": True,
        "joints_0_exact": True,
        "maximum_texcoord_0_error": uv_error,
        "maximum_weights_0_error": weight_error,
        "maximum_expected_position_error_m": position_error,
        "maximum_output_normal_norm_error": normal_norm_error,
        **geometry,
    }


def _skin_readback(
    source: GlbDocument, output: GlbDocument, *, size_scale: float
) -> dict[str, Any]:
    source_skin = extract_skins(source)[0]
    output_skin = extract_skins(output)[0]
    source_names = tuple(joint.name for joint in source_skin.joints)
    output_names = tuple(joint.name for joint in output_skin.joints)
    if source_names != output_names:
        raise RuntimeError("skin joint order/names changed across appearance export")
    maximum_translation = 0.0
    maximum_rotation = 0.0
    maximum_scale = 0.0
    for left, right in zip(source_skin.joints, output_skin.joints, strict=True):
        maximum_translation = max(
            maximum_translation,
            _maximum_abs(
                np.asarray(left.local_trs.translation)[None, :] * size_scale,
                np.asarray(right.local_trs.translation)[None, :],
                f"joint {left.name} translation",
            ),
        )
        maximum_rotation = max(
            maximum_rotation,
            _quaternion_error(
                np.asarray(left.local_trs.rotation_xyzw)[None, :],
                np.asarray(right.local_trs.rotation_xyzw)[None, :],
                f"joint {left.name} rotation",
            ),
        )
        maximum_scale = max(
            maximum_scale,
            _maximum_abs(
                np.asarray(left.local_trs.scale)[None, :],
                np.asarray(right.local_trs.scale)[None, :],
                f"joint {left.name} scale",
            ),
        )
    if (
        source_skin.inverse_bind_matrices_accessor_index is None
        or output_skin.inverse_bind_matrices_accessor_index is None
    ):
        raise RuntimeError("both skins must expose inverse bind matrices")
    left_ibm = _accessor_array(
        source, source_skin.inverse_bind_matrices_accessor_index
    ).astype(np.float64)
    right_ibm = _accessor_array(
        output, output_skin.inverse_bind_matrices_accessor_index
    ).astype(np.float64)
    expected_ibm = left_ibm.copy()
    expected_ibm[:, 12:15] *= size_scale
    ibm_error = _maximum_abs(expected_ibm, right_ibm, "inverse bind matrices")
    maximum = max(maximum_translation, maximum_rotation, maximum_scale, ibm_error)
    if maximum > _OUTPUT_FLOAT_TOLERANCE:
        raise RuntimeError(
            "appearance skin output readback failed: "
            f"translation={maximum_translation:.9g}, rotation={maximum_rotation:.9g}, "
            f"scale={maximum_scale:.9g}, ibm={ibm_error:.9g}"
        )
    return {
        "joint_count": len(output_names),
        "joint_order_unchanged": True,
        "maximum_scaled_rest_translation_error_m": maximum_translation,
        "maximum_rest_rotation_error": maximum_rotation,
        "maximum_rest_scale_error": maximum_scale,
        "maximum_scaled_inverse_bind_matrix_error": ibm_error,
        "tolerance": _OUTPUT_FLOAT_TOLERANCE,
    }


def _action_readback(
    source: GlbDocument, output: GlbDocument, *, size_scale: float
) -> dict[str, Any]:
    source_actions = {action.name: action for action in extract_actions(source)}
    output_actions = {action.name: action for action in extract_actions(output)}
    if set(source_actions) != {"Idle", "Walking"} or set(output_actions) != {
        "Idle",
        "Walking",
    }:
        raise RuntimeError("source/output must contain exactly Idle and Walking")
    records: list[dict[str, Any]] = []
    global_maximum = 0.0
    for name in ("Idle", "Walking"):
        left = source_actions[name]
        right = output_actions[name]

        def channels(action: Any) -> dict[tuple[str, str], Any]:
            result: dict[tuple[str, str], Any] = {}
            for channel in action.channels:
                if channel.target_node_name is None:
                    raise RuntimeError(f"{name} contains an unnamed animation target")
                key = (channel.target_node_name, channel.target_path)
                if key in result:
                    raise RuntimeError(f"{name} contains duplicate channel {key}")
                result[key] = channel
            return result

        left_channels = channels(left)
        right_channels = channels(right)
        if set(left_channels) != set(right_channels):
            raise RuntimeError(f"{name} channel target/path set changed")
        maxima = {"timestamps": 0.0, "translation": 0.0, "rotation": 0.0, "scale": 0.0}
        for key in sorted(left_channels):
            left_channel = left_channels[key]
            right_channel = right_channels[key]
            if left_channel.interpolation != right_channel.interpolation:
                raise RuntimeError(f"{name} {key} interpolation changed")
            timestamp_error = _maximum_abs(
                np.asarray(left_channel.timestamps_seconds)[:, None],
                np.asarray(right_channel.timestamps_seconds)[:, None],
                f"{name} {key} timestamps",
            )
            left_values = np.asarray(left_channel.values, dtype=np.float64)
            right_values = np.asarray(right_channel.values, dtype=np.float64)
            if key[1] == "translation":
                value_error = _maximum_abs(
                    left_values * size_scale,
                    right_values,
                    f"{name} {key} translation",
                )
            elif key[1] == "rotation":
                value_error = _quaternion_error(
                    left_values, right_values, f"{name} {key} rotation"
                )
            elif key[1] == "scale":
                value_error = _maximum_abs(
                    left_values, right_values, f"{name} {key} scale"
                )
            else:
                raise RuntimeError(f"{name} contains unsupported path {key[1]!r}")
            maxima["timestamps"] = max(maxima["timestamps"], timestamp_error)
            maxima[key[1]] = max(maxima[key[1]], value_error)
        action_maximum = max(maxima.values())
        global_maximum = max(global_maximum, action_maximum)
        records.append(
            {
                "action": name,
                "channel_count": len(right_channels),
                "maximum_errors": maxima,
            }
        )
    if global_maximum > _OUTPUT_FLOAT_TOLERANCE:
        raise RuntimeError(
            f"appearance action output readback error {global_maximum:.9g} "
            f"> {_OUTPUT_FLOAT_TOLERANCE:.9g}"
        )
    return {
        "actions": records,
        "channel_targets_unchanged": True,
        "translations_scaled_by_size": True,
        "maximum_error": global_maximum,
        "tolerance": _OUTPUT_FLOAT_TOLERANCE,
    }


def _image_payload(document: GlbDocument, image_index: int) -> tuple[bytes, str | None]:
    value = document.json
    images = value.get("images", [])
    views = value.get("bufferViews", [])
    if not isinstance(image_index, int) or not 0 <= image_index < len(images):
        raise RuntimeError("material image index is invalid")
    image = images[image_index]
    if not isinstance(image, dict) or "bufferView" not in image or "uri" in image:
        raise RuntimeError("appearance output images must be embedded GLB bufferViews")
    view_index = image["bufferView"]
    if not isinstance(view_index, int) or not 0 <= view_index < len(views):
        raise RuntimeError("image bufferView is invalid")
    view = views[view_index]
    start = int(view.get("byteOffset", 0))
    end = start + int(view.get("byteLength", 0))
    if start < 0 or end > len(document.binary):
        raise RuntimeError("embedded image escapes GLB BIN")
    return document.binary[start:end], image.get("mimeType")


def _texture_image_index(value: dict[str, Any], texture_info: Any, owner: str) -> int:
    if not isinstance(texture_info, dict) or not isinstance(
        texture_info.get("index"), int
    ):
        raise RuntimeError(f"{owner} texture info is invalid")
    textures = value.get("textures", [])
    texture_index = texture_info["index"]
    if not 0 <= texture_index < len(textures):
        raise RuntimeError(f"{owner} texture index is invalid")
    image_index = textures[texture_index].get("source")
    if not isinstance(image_index, int):
        raise RuntimeError(f"{owner} texture has no image source")
    return image_index


def _material_readback(
    source: GlbDocument, output: GlbDocument, *, texture_path: Path
) -> dict[str, Any]:
    source_value = source.json
    output_value = output.json
    source_materials = source_value.get("materials", [])
    output_materials = output_value.get("materials", [])
    if len(source_materials) != 1 or len(output_materials) != 1:
        raise RuntimeError("appearance v1 currently requires exactly one material")
    source_material = source_materials[0]
    output_material = output_materials[0]
    source_pbr = source_material.get("pbrMetallicRoughness", {})
    output_pbr = output_material.get("pbrMetallicRoughness", {})
    if output_material.get("alphaMode") != "OPAQUE":
        raise RuntimeError("appearance output alphaMode is not explicitly OPAQUE")
    base_color_factor = output_pbr.get("baseColorFactor")
    if (
        not isinstance(base_color_factor, list)
        or len(base_color_factor) != 4
        or any(
            isinstance(component, bool)
            or not isinstance(component, (int, float))
            or not math.isfinite(float(component))
            or not 0.0 <= float(component) <= 1.0
            for component in base_color_factor
        )
        or float(base_color_factor[3]) != 1.0
    ):
        raise RuntimeError("appearance output baseColorFactor must have opaque alpha")
    emissive_factor = output_material.get("emissiveFactor")
    if emissive_factor != ZERO_EMISSIVE_FACTOR:
        raise RuntimeError("appearance output emissiveFactor must be [0, 0, 0]")
    if "emissiveTexture" in output_material:
        raise RuntimeError("appearance output retains an emissiveTexture")
    if (
        output_pbr.get("metallicFactor", 1.0) != 0
        or float(output_pbr.get("roughnessFactor", 1.0)) < MINIMUM_ROUGHNESS_FACTOR
    ):
        raise RuntimeError("appearance output PBR factors are not normalized")
    if "metallicRoughnessTexture" in output_pbr:
        raise RuntimeError(
            "appearance output retains a multiplier texture that defeats the "
            "roughness lower bound"
        )
    output_extensions = output_material.get("extensions", {})
    if not isinstance(output_extensions, dict):
        raise RuntimeError("appearance output material extensions must be an object")
    unsupported_extensions = set(output_extensions) - {"KHR_materials_specular"}
    if unsupported_extensions:
        raise RuntimeError(
            "appearance output retains unsupported material extensions: "
            f"{sorted(unsupported_extensions)}"
        )
    output_specular = output_extensions.get("KHR_materials_specular")
    if output_specular is not None and not isinstance(output_specular, dict):
        raise RuntimeError("appearance output specular extension is invalid")
    effective_specular_factor = (
        float(output_specular.get("specularFactor", 1.0))
        if output_specular is not None
        else None
    )
    if (
        effective_specular_factor is not None
        and not 0.0 <= effective_specular_factor <= MAXIMUM_SPECULAR_FACTOR
    ):
        raise RuntimeError(
            "appearance output effective KHR_materials_specular factor exceeds "
            "the matte-material gate"
        )
    specular_color_factor = (
        output_specular.get("specularColorFactor")
        if output_specular is not None
        else None
    )
    if (
        not isinstance(specular_color_factor, list)
        or len(specular_color_factor) != 3
        or any(
            isinstance(component, bool)
            or not isinstance(component, (int, float))
            or not math.isfinite(float(component))
            or not 0.0 <= float(component) <= MAXIMUM_SPECULAR_COLOR_FACTOR
            for component in specular_color_factor
        )
    ):
        raise RuntimeError(
            "appearance output KHR_materials_specular.specularColorFactor is "
            "outside the bounded range"
        )
    effective_specular_peak = effective_specular_factor * max(
        float(component) for component in specular_color_factor
    )
    if effective_specular_peak > MAXIMUM_SPECULAR_FACTOR:
        raise RuntimeError("appearance output hides an excessive specular channel")
    roles = {
        "base_color": (
            source_pbr.get("baseColorTexture"),
            output_pbr.get("baseColorTexture"),
        ),
        "normal": (
            source_material.get("normalTexture"),
            output_material.get("normalTexture"),
        ),
        "specular": (
            source_material.get("extensions", {})
            .get("KHR_materials_specular", {})
            .get("specularTexture"),
            output_material.get("extensions", {})
            .get("KHR_materials_specular", {})
            .get("specularTexture"),
        ),
    }
    records: dict[str, Any] = {}
    for role, (source_info, output_info) in roles.items():
        source_index = _texture_image_index(source_value, source_info, f"source {role}")
        output_index = _texture_image_index(output_value, output_info, f"output {role}")
        source_bytes, source_mime = _image_payload(source, source_index)
        output_bytes, output_mime = _image_payload(output, output_index)
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        output_sha = hashlib.sha256(output_bytes).hexdigest()
        if role != "base_color" and (
            source_sha != output_sha or source_mime != output_mime
        ):
            raise RuntimeError(f"unchanged {role} image bytes changed across export")
        records[role] = {
            "source_sha256": source_sha,
            "output_sha256": output_sha,
            "mime_type": output_mime,
            "unchanged": role != "base_color",
        }
    texture_sha = _sha256(texture_path)
    if records["base_color"]["output_sha256"] != texture_sha:
        raise RuntimeError("embedded base-color bytes differ from standalone texture")
    records["base_color"]["standalone_sha256"] = texture_sha
    records["base_color"]["embedded_matches_standalone"] = True
    return {
        "material_count": 1,
        "metallic_factor": output_pbr.get("metallicFactor", 1.0),
        "roughness_factor": output_pbr.get("roughnessFactor", 1.0),
        "metallic_roughness_texture_present": False,
        "alpha_mode": output_material["alphaMode"],
        "base_color_factor": base_color_factor,
        "emissive_factor": emissive_factor,
        "emissive_texture_present": False,
        "effective_khr_materials_specular_factor": effective_specular_factor,
        "effective_khr_materials_specular_color_factor": specular_color_factor,
        "maximum_effective_khr_materials_specular_channel": (effective_specular_peak),
        "allowed_material_extensions": ["KHR_materials_specular"],
        "texture_images": records,
    }


def _output_audit(
    source_path: Path,
    output_path: Path,
    *,
    texture_path: Path,
    torso_names: tuple[str, ...],
    head_names: tuple[str, ...],
    torso_scale: float,
    head_scale: float,
    size_scale: float,
) -> dict[str, Any]:
    source = load_glb(source_path)
    parsed = load_glb(output_path)
    document = parsed.json
    animations = document.get("animations")
    names = (
        [item.get("name") for item in animations]
        if isinstance(animations, list)
        else []
    )
    if len(names) != 2 or set(names) != {"Idle", "Walking"}:
        raise RuntimeError(f"appearance output actions differ: {names}")
    nodes = document.get("nodes", [])
    if len(document.get("skins", [])) != 1:
        raise RuntimeError("appearance output must contain exactly one skin")
    maximum_ancestor_scale_error = 0.0
    joints = set(document["skins"][0].get("joints", []))
    parents: dict[int, int] = {}
    for parent, node in enumerate(nodes):
        for child in node.get("children", []):
            parents[child] = parent
    roots = [joint for joint in joints if parents.get(joint) not in joints]
    if len(roots) != 1:
        raise RuntimeError(f"appearance output skin root differs: {roots}")
    cursor = parents.get(roots[0])
    ancestors: list[dict[str, Any]] = []
    while cursor is not None:
        scale = np.asarray(nodes[cursor].get("scale", [1.0, 1.0, 1.0]))
        maximum_ancestor_scale_error = max(
            maximum_ancestor_scale_error,
            float(np.max(np.abs(scale - 1.0))),
        )
        ancestors.append(
            {"node": cursor, "name": nodes[cursor].get("name"), "scale": scale.tolist()}
        )
        cursor = parents.get(cursor)
    if maximum_ancestor_scale_error > 1.0e-6:
        raise RuntimeError("appearance output reintroduced a skin ancestor scale")
    return {
        "animation_names": names,
        "skin_count": 1,
        "skin_joint_count": len(joints),
        "skin_root_name": nodes[roots[0]].get("name"),
        "skin_ancestors": ancestors,
        "maximum_skin_ancestor_scale_error": maximum_ancestor_scale_error,
        "mesh_count": len(document.get("meshes", [])),
        "material_count": len(document.get("materials", [])),
        "image_count": len(document.get("images", [])),
        "mesh_invariants": _mesh_readback(
            source,
            parsed,
            torso_names=torso_names,
            head_names=head_names,
            torso_scale=torso_scale,
            head_scale=head_scale,
            size_scale=size_scale,
        ),
        "skin_invariants": _skin_readback(source, parsed, size_scale=size_scale),
        "action_invariants": _action_readback(source, parsed, size_scale=size_scale),
        "material_invariants": _material_readback(
            source, parsed, texture_path=texture_path
        ),
    }


def _write_json_exclusive(path: Path, value: Any) -> None:
    created = False
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            created = True
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise


def _remember_created(path: Path, created: dict[Path, tuple[int, int]]) -> None:
    current = path.lstat()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"appearance output is not a regular file: {path}")
    created[path] = (current.st_dev, current.st_ino)


def _rollback_created(created: dict[Path, tuple[int, int]]) -> None:
    for path, identity in reversed(list(created.items())):
        try:
            current = path.lstat()
        except FileNotFoundError:
            continue
        if (
            not path.is_symlink()
            and path.is_file()
            and (current.st_dev, current.st_ino) == identity
        ):
            try:
                path.unlink()
            except OSError:
                pass


def _run(args: argparse.Namespace, created: dict[Path, tuple[int, int]]) -> int:
    batch_path = _regular(args.batch, "appearance batch")
    output = _new_output(args.output_glb, "appearance GLB")
    report_path = _new_output(args.report, "appearance report")
    texture_path = _new_output(
        args.output_texture
        if args.output_texture is not None
        else output.with_name(f"{output.stem}.base_color.png"),
        "appearance base-colour texture",
    )
    if len({output, report_path, texture_path}) != 3:
        raise RuntimeError("appearance outputs must be distinct")
    batch, request, source = _validate_batch(batch_path, args.ordinal)
    operations = {
        operation["attribute"]: operation
        for operation in request["realization_operations"]
    }
    size = operations["size"]["parameters"]
    body = operations["body_build"]["parameters"]
    coat = operations["coat_profile"]["parameters"]
    age = operations["life_stage"]["parameters"]

    _clear_scene()
    bpy.context.scene.render.fps = 30
    armature, mesh = _import_source(source)
    actions = _canonical_actions(armature)
    contract_before = _topology_uv_skin_sha256(mesh)
    action_before = _action_curve_sha256(actions.values())
    torso_names = _resolve_group_names(
        mesh, body.get("semantic_joint_names"), "body_build.semantic_joint_names"
    )
    head_names = _resolve_group_names(
        mesh, age.get("semantic_joint_names"), "life_stage.semantic_joint_names"
    )
    shape, head_weights, coordinates = _apply_shape(
        mesh,
        torso_scale=float(body["torso_girth_scale"]),
        head_scale=float(age["head_scale"]),
        torso_names=torso_names,
        head_names=head_names,
    )
    texture = _realize_texture(
        mesh,
        texture_path,
        luminance_gain=float(coat["luminance_gain"]),
        coat_desaturation=float(age["coat_desaturation"]),
        muzzle_gray_mix=float(age["muzzle_gray_mix"]),
        muzzle_gray_target=float(age["muzzle_gray_target"]),
        preserve_pattern=_registered_preserve_pattern(request),
        head_weights=head_weights,
        coordinates=coordinates,
    )
    _remember_created(texture_path, created)
    materials = _normalise_materials(mesh)
    scale = _bake_uniform_scale(armature, mesh, float(size["scale_ratio"]))
    contract_after = _topology_uv_skin_sha256(mesh)
    action_after = _action_curve_sha256(actions.values())
    if contract_after != contract_before:
        raise RuntimeError(
            "appearance realization changed topology, UVs, or skin weights"
        )
    if action_after != action_before:
        raise RuntimeError("appearance realization changed authored action keyframes")
    raw_export = _new_output(
        output.with_name(f".{output.stem}.blender_raw.glb"),
        "temporary Blender GLB",
    )
    try:
        _export(raw_export, armature, mesh)
        material_normalization_report = normalize_glb_materials(
            raw_export, output, force_opaque=True
        )
        _remember_created(output, created)
    except Exception:
        # ``output`` did not exist at entry and can only have been created by
        # this invocation.  Never leave a half-published normalized GLB when
        # normalization/readback fails.
        if output.is_file() and not output.is_symlink():
            output.unlink()
        raise
    finally:
        if raw_export.is_file() and not raw_export.is_symlink():
            raw_export.unlink()
    output_audit = _output_audit(
        source,
        output,
        texture_path=texture_path,
        torso_names=torso_names,
        head_names=head_names,
        torso_scale=float(body["torso_girth_scale"]),
        head_scale=float(age["head_scale"]),
        size_scale=float(size["scale_ratio"]),
    )
    report = {
        "schema": SCHEMA,
        "status": "pass",
        "state_classification": "research_candidate",
        "qualification_claim": False,
        "formal_dataset_registration_authorized": False,
        "tool_identity": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
            "material_normalizer": {
                "path": str((SOURCE_ROOT / "avengine/m2/materials.py").resolve()),
                "sha256": _sha256(SOURCE_ROOT / "avengine/m2/materials.py"),
            },
            "blender_version": bpy.app.version_string,
            "export_profile": {
                "format": "GLB",
                "animation_mode": "ACTIONS",
                "force_sampling": True,
                "skins": True,
                "texcoords": True,
                "normals": True,
                "image_format": "AUTO",
            },
            "output_readback_float_tolerance": _OUTPUT_FLOAT_TOLERANCE,
        },
        "batch": {
            "path": str(batch_path),
            "sha256": _sha256(batch_path),
            "batch_id": batch["batch_id"],
            "batch_content_sha256": batch["batch_content_sha256"],
        },
        "instance_request": {
            "ordinal": args.ordinal,
            "instance_request_id": request["instance_request_id"],
            "request_sha256": request["request_sha256"],
            "taxonomy": request["taxonomy"],
            "attributes": request["attributes"],
            "realization_operations": request["realization_operations"],
        },
        "source": {
            "path": str(source),
            "sha256": _sha256(source),
            "byte_size": source.stat().st_size,
        },
        "realization": {
            "shape": shape,
            "texture": texture,
            "materials": materials,
            "glb_material_normalization": {
                "schema": material_normalization_report["schema"],
                "status": material_normalization_report["status"],
                "policy": material_normalization_report["policy"],
                "material_count": material_normalization_report["material_count"],
                "invariants": material_normalization_report["invariants"],
                "source_glb_sha256": material_normalization_report["source"]["sha256"],
                "output_glb_sha256": material_normalization_report["output"]["sha256"],
                "normalization_report_content_sha256": (
                    material_normalization_report["report_content_sha256"]
                ),
            },
            "uniform_size": scale,
            "topology_uv_skin_sha256_before": contract_before,
            "topology_uv_skin_sha256_after": contract_after,
            "topology_uv_skin_unchanged": True,
            "action_curve_sha256_before": action_before,
            "action_curve_sha256_after": action_after,
            "in_memory_authored_action_curves_unchanged": True,
        },
        "output": {
            "glb": {
                "path": str(output),
                "sha256": _sha256(output),
                "byte_size": output.stat().st_size,
            },
            "base_color_texture": {
                "path": str(texture_path),
                "sha256": _sha256(texture_path),
                "byte_size": texture_path.stat().st_size,
            },
            "readback_audit": output_audit,
        },
        "limitations": [
            "This pass realizes appearance only and does not qualify the asset.",
            "Rebase, action bake, automatic QA, Habitat playback, contacts, and human visual review remain separate gates.",
            "The L9 batch is a balanced combination design; separate OFAT evidence remains required before formal promotion.",
        ],
    }
    report["report_content_sha256"] = _canonical_sha256(report)
    _write_json_exclusive(report_path, report)
    _remember_created(report_path, created)
    print(
        json.dumps(
            {
                "status": "pass",
                "instance_request_id": request["instance_request_id"],
                "output": str(output),
                "output_sha256": report["output"]["glb"]["sha256"],
                "report": str(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    args = _arguments()
    created: dict[Path, tuple[int, int]] = {}
    try:
        return _run(args, created)
    except Exception:
        # Only unlink exact device/inode pairs published by this invocation.
        # A late readback or report failure therefore cannot leave a texture,
        # GLB, or report half-set or delete a concurrent replacement.
        _rollback_created(created)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
