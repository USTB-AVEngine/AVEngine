"""Data helpers for the seated-human research asset preparation route."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


class SeatedHumanSpecError(ValueError):
    """A seated-human preparation record is malformed or unsafe."""


@dataclass(frozen=True)
class SeatedHumanSpec:
    asset_id: str
    display_label: str
    color_name: str
    source_glb: Path
    emitter_offset_blender_m: tuple[float, float, float]
    seat_anchor_id: str
    shirt_color_rgb: tuple[float, float, float] | None
    seat_top_m: float
    chair_center_blender_m: tuple[float, float, float]
    floor_correction_m: float
    reference_actor_yaw_degrees: float | None
    reference_chair_yaw_degrees: float
    static_candidate_glb: Path | None


def _text(value: Any, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SeatedHumanSpecError(f"{owner} must be a non-empty string")
    return value.strip()


def _vec3(value: Any, owner: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise SeatedHumanSpecError(f"{owner} must contain three numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise SeatedHumanSpecError(f"{owner} must contain finite numbers")
    return result


def _rgb(value: Any, owner: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    result = _vec3(value, owner)
    if any(item < 0.0 or item > 1.0 for item in result):
        raise SeatedHumanSpecError(f"{owner} must be in [0,1]")
    return result


def _path(value: Any, owner: str, *, must_exist: bool) -> Path:
    path = Path(_text(value, owner)).expanduser()
    if not path.is_absolute():
        raise SeatedHumanSpecError(f"{owner} must be absolute")
    path = path.resolve()
    if must_exist and (not path.is_file() or path.is_symlink()):
        raise SeatedHumanSpecError(f"{owner} must be a regular file: {path}")
    return path


def load_seated_human_batch(path: str | Path, *, require_sources: bool = True) -> tuple[SeatedHumanSpec, ...]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SeatedHumanSpecError(f"could not read seated-human batch: {source}") from error
    if not isinstance(value, Mapping) or value.get("kind") != "avengine_seated_human_batch_v1":
        raise SeatedHumanSpecError("seated-human batch kind is unsupported")
    raw_assets = value.get("assets")
    if not isinstance(raw_assets, list) or len(raw_assets) != 4:
        raise SeatedHumanSpecError("the current seated-human batch must contain four assets")
    result: list[SeatedHumanSpec] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_assets):
        if not isinstance(raw, Mapping):
            raise SeatedHumanSpecError(f"assets[{index}] must be an object")
        asset_id = _text(raw.get("asset_id"), f"assets[{index}].asset_id")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", asset_id):
            raise SeatedHumanSpecError(f"assets[{index}].asset_id is not a stable ID")
        if asset_id in seen:
            raise SeatedHumanSpecError(f"duplicate seated asset ID: {asset_id}")
        seen.add(asset_id)
        source_glb = _path(raw.get("source_glb"), f"assets[{index}].source_glb", must_exist=require_sources)
        static_candidate = raw.get("static_candidate_glb")
        result.append(
            SeatedHumanSpec(
                asset_id=asset_id,
                display_label=_text(raw.get("display_label"), f"assets[{index}].display_label"),
                color_name=_text(raw.get("color_name"), f"assets[{index}].color_name"),
                source_glb=source_glb,
                emitter_offset_blender_m=_vec3(raw.get("emitter_offset_blender_m"), f"assets[{index}].emitter_offset_blender_m"),
                seat_anchor_id=_text(raw.get("seat_anchor_id"), f"assets[{index}].seat_anchor_id"),
                shirt_color_rgb=_rgb(raw.get("shirt_color_rgb"), f"assets[{index}].shirt_color_rgb"),
                seat_top_m=float(raw.get("seat_top_m", 0.53)),
                chair_center_blender_m=_vec3(raw.get("chair_center_blender_m", [0.0, 0.18, 0.0]), f"assets[{index}].chair_center_blender_m"),
                floor_correction_m=float(raw.get("floor_correction_m", -0.01)),
                reference_actor_yaw_degrees=(float(raw["reference_actor_yaw_degrees"]) if raw.get("reference_actor_yaw_degrees") is not None else None),
                reference_chair_yaw_degrees=float(raw.get("reference_chair_yaw_degrees", 0.0)),
                static_candidate_glb=_path(static_candidate, f"assets[{index}].static_candidate_glb", must_exist=True) if static_candidate is not None else None,
            )
        )
    for item in result:
        if not math.isfinite(item.seat_top_m) or not math.isfinite(item.floor_correction_m):
            raise SeatedHumanSpecError(f"{item.asset_id} seat values must be finite")
    return tuple(result)


def avengine_emitter_offset_m(spec: SeatedHumanSpec) -> tuple[float, float, float]:
    x, y, z = spec.emitter_offset_blender_m
    return x, z, -y


def seat_root_offset_blender_m(spec: SeatedHumanSpec) -> tuple[float, float, float]:
    theta = math.radians(spec.reference_chair_yaw_degrees)
    x, y, z = spec.chair_center_blender_m
    rotated_x = math.cos(theta) * x - math.sin(theta) * y
    rotated_y = math.sin(theta) * x + math.cos(theta) * y
    return (-rotated_x, -rotated_y, z + spec.floor_correction_m)

def build_ue_import_request(
    specs: Sequence[SeatedHumanSpec],
    *,
    output_root: str | Path,
    content_root: str = "/Game/AVEngine/SeatedHumans",
) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    if not specs or len(specs) != 4:
        raise SeatedHumanSpecError("UE import request requires exactly four seated assets")
    if not re.fullmatch(r"/Game/(?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+", content_root):
        raise SeatedHumanSpecError("content_root is not a safe /Game path")
    assets = []
    for spec in specs:
        source = root / spec.asset_id / "asset" / f"{spec.asset_id}.glb"
        if not source.is_file():
            raise SeatedHumanSpecError(f"generated skeletal GLB is missing: {source}")
        assets.append(
            {
                "asset_id": spec.asset_id,
                "display_label": spec.display_label,
                "source_glb": str(source),
                "destination": f"{content_root}/{spec.asset_id}",
                "animation_name": "Seated_Idle",
                "anatomical_forward_axis_avengine": [0.0, 0.0, 1.0],
                "ue_anatomical_forward_yaw_deg": 90.0,
                "emitter_offset_avengine_m": list(avengine_emitter_offset_m(spec)),
                "seat_reference": {
                    "seat_anchor_id": spec.seat_anchor_id,
                    "seat_top_m": spec.seat_top_m,
                    "chair_center_blender_m": list(spec.chair_center_blender_m),
                    "floor_correction_m": spec.floor_correction_m,
                    "reference_chair_yaw_degrees": spec.reference_chair_yaw_degrees,
                    "reference_actor_yaw_degrees": spec.reference_actor_yaw_degrees,
                    "root_offset_from_seat_anchor_blender_m": list(seat_root_offset_blender_m(spec)),
                    "root_transform_policy": "derive_from_room_seat_anchor",
                    "reference_only": True,
                },
            }
        )
    return {
        "kind": "avengine_seated_human_ue_import_request_v1",
        "status": "research_only",
        "content_root": content_root,
        "assets": assets,
        "claim_boundary": "skeletal seated idle only; no sit transition or lip animation",
    }
