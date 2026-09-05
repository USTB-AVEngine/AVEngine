"""Load the class-level Pixal3D / SkinTokens transform profile.

The profile is recovered from Pixal3D mesh export, historical heading
normalization, and the TokenRig human static-audit yaw. It is keyed by body
class, never by asset id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "avengine_pixal3d_transform_profile_v1"
DEFAULT_PROFILE = (
    Path(__file__).resolve().parents[2]
    / "examples/assets/pixal3d_transform_profile_v1.json"
)
BODY_CLASSES = ("quadruped", "biped")
FORBIDDEN_SELECTOR_KEYS = ("asset_id", "asset-id", "assetId")


class Pixal3DTransformProfileError(ValueError):
    """The transform profile cannot be used as written."""


def _as_mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Pixal3DTransformProfileError(f"{owner} must be an object")
    return value


def _as_matrix4(value: Any, *, owner: str) -> list[list[float]]:
    if not isinstance(value, Sequence) or len(value) != 4:
        raise Pixal3DTransformProfileError(f"{owner} must be a 4x4 matrix")
    rows: list[list[float]] = []
    for index, row in enumerate(value):
        if not isinstance(row, Sequence) or len(row) != 4:
            raise Pixal3DTransformProfileError(f"{owner} row {index} must have 4 numbers")
        parsed: list[float] = []
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, (int, float)):
                raise Pixal3DTransformProfileError(f"{owner} must contain finite numbers")
            number = float(cell)
            if number != number or number in (float("inf"), float("-inf")):
                raise Pixal3DTransformProfileError(f"{owner} must contain finite numbers")
            parsed.append(number)
        rows.append(parsed)
    return rows


def _reject_asset_id_selectors(document: Mapping[str, Any], *, owner: str) -> None:
    for key, value in document.items():
        if key in FORBIDDEN_SELECTOR_KEYS:
            raise Pixal3DTransformProfileError(
                f"{owner} selects transforms by {key!r}; class-level profiles "
                "must not be keyed by asset id"
            )
        if isinstance(value, Mapping):
            _reject_asset_id_selectors(value, owner=f"{owner}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, Mapping):
                    _reject_asset_id_selectors(item, owner=f"{owner}[{index}]")


def load_profile(path: Path | None = None) -> dict[str, Any]:
    profile_path = Path(path) if path is not None else DEFAULT_PROFILE
    profile_path = profile_path.expanduser()
    if profile_path.is_symlink() or not profile_path.is_file():
        raise Pixal3DTransformProfileError(
            f"transform profile is missing or unsafe: {profile_path}"
        )
    try:
        document = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Pixal3DTransformProfileError(
            f"could not read transform profile {profile_path}: {error}"
        ) from error
    document = dict(_as_mapping(document, owner="transform profile"))
    if document.get("schema") != SCHEMA:
        raise Pixal3DTransformProfileError(
            f"transform profile schema must be {SCHEMA}"
        )
    _reject_asset_id_selectors(document, owner="transform profile")
    mesh_export = _as_mapping(document.get("mesh_export"), owner="mesh_export")
    _as_matrix4(
        mesh_export.get("apply_transform_4x4_row_major"),
        owner="mesh_export.apply_transform_4x4_row_major",
    )
    blender_import = _as_mapping(
        document.get("blender_import"), owner="blender_import"
    )
    if blender_import.get("operator") != "bpy.ops.import_scene.gltf":
        raise Pixal3DTransformProfileError(
            "blender_import.operator must be bpy.ops.import_scene.gltf"
        )
    body_classes = _as_mapping(document.get("body_classes"), owner="body_classes")
    missing = [name for name in BODY_CLASSES if name not in body_classes]
    if missing:
        raise Pixal3DTransformProfileError(
            f"body_classes missing {missing}"
        )
    for name in BODY_CLASSES:
        entry = _as_mapping(body_classes[name], owner=f"body_classes.{name}")
        _as_matrix4(
            entry.get("extra_root_transform_4x4_row_major"),
            owner=f"body_classes.{name}.extra_root_transform_4x4_row_major",
        )
        if not isinstance(entry.get("target_front_axis"), str) or not entry["target_front_axis"]:
            raise Pixal3DTransformProfileError(
                f"body_classes.{name}.target_front_axis is required"
            )
    document["_path"] = str(profile_path.resolve())
    return document


def mesh_export_matrix(profile: Mapping[str, Any]) -> list[list[float]]:
    mesh_export = _as_mapping(profile.get("mesh_export"), owner="mesh_export")
    return _as_matrix4(
        mesh_export.get("apply_transform_4x4_row_major"),
        owner="mesh_export.apply_transform_4x4_row_major",
    )


def body_class_root_matrix(
    profile: Mapping[str, Any], body_class: str
) -> list[list[float]]:
    if body_class not in BODY_CLASSES:
        raise Pixal3DTransformProfileError(
            f"body_class must be one of {BODY_CLASSES}, got {body_class!r}"
        )
    body_classes = _as_mapping(profile.get("body_classes"), owner="body_classes")
    entry = _as_mapping(
        body_classes.get(body_class), owner=f"body_classes.{body_class}"
    )
    return _as_matrix4(
        entry.get("extra_root_transform_4x4_row_major"),
        owner=f"body_classes.{body_class}.extra_root_transform_4x4_row_major",
    )


def body_class_target_front_axis(
    profile: Mapping[str, Any], body_class: str
) -> str:
    if body_class not in BODY_CLASSES:
        raise Pixal3DTransformProfileError(
            f"body_class must be one of {BODY_CLASSES}, got {body_class!r}"
        )
    body_classes = _as_mapping(profile.get("body_classes"), owner="body_classes")
    entry = _as_mapping(
        body_classes.get(body_class), owner=f"body_classes.{body_class}"
    )
    axis = entry.get("target_front_axis")
    if not isinstance(axis, str) or not axis:
        raise Pixal3DTransformProfileError(
            f"body_classes.{body_class}.target_front_axis is required"
        )
    return axis


def blender_import_kwargs(profile: Mapping[str, Any]) -> dict[str, Any]:
    blender_import = _as_mapping(
        profile.get("blender_import"), owner="blender_import"
    )
    pack = blender_import.get("import_pack_images", False)
    if not isinstance(pack, bool):
        raise Pixal3DTransformProfileError(
            "blender_import.import_pack_images must be a boolean"
        )
    return {"import_pack_images": pack}
