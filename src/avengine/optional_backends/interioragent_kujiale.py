"""Pure planning helpers for external InteriorAgent/Kujiale UE previews.

InteriorAgent assets are optional, user-downloaded research data.  This module
does not import USD, Unreal or SPEAR and does not make those dependencies part
of the default AVEngine install.  Habitat-native AVEngine remains authoritative
for episode state, navigation, audio, Topdown and metadata; this adapter only
plans a comparison-visual render.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import re
from typing import Any


BACKEND_ROLE = "comparison_visual"
DATASET_ID = "spatialverse/InteriorAgent"
PROFILE_SCHEMA = "avengine_optional_interioragent_kujiale_profile_v1"
PLAN_SCHEMA = "avengine_optional_interioragent_kujiale_plan_v1"
LICENSE_URL = (
    "https://kloudsim-usa-cos.kujiale.com/InteriorAgent/"
    "InteriorAgent_Terms_of_Use.pdf"
)
DATASET_URL = "https://huggingface.co/datasets/spatialverse/InteriorAgent"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SCOPE = re.compile(r"^[A-Za-z0-9_]+$")


class InteriorAgentPlanError(ValueError):
    """Raised when an optional InteriorAgent visual profile is invalid."""


def _finite(value: Any, *, owner: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise InteriorAgentPlanError(f"{owner} must be numeric") from exc
    if not math.isfinite(result):
        raise InteriorAgentPlanError(f"{owner} must be finite")
    return result


def _vector3(value: Any, *, owner: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise InteriorAgentPlanError(f"{owner} must contain three numbers")
    return tuple(
        _finite(item, owner=f"{owner}[{index}]") for index, item in enumerate(value)
    )


def _identifier(value: Any, *, owner: str) -> str:
    result = str(value)
    if not _IDENTIFIER.fullmatch(result):
        raise InteriorAgentPlanError(f"{owner} is not a stable identifier: {result!r}")
    return result


def usd_meters_to_unreal_cm(value: Sequence[float]) -> tuple[float, float, float]:
    """Convert InteriorAgent's X-forward/Y-right/Z-up meters to UE centimeters."""

    vector = _vector3(value, owner="USD position")
    return tuple(item * 100.0 for item in vector)


def _rgb(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if (
        value is None
        or not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) < 3
    ):
        return default
    return tuple(
        _finite(value[index], owner=f"color[{index}]") for index in range(3)
    )


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, _finite(value, owner="material scalar")))


def _average_color(value: Any, default: float) -> float:
    return sum(_rgb(value, (default, default, default))) / 3.0


def preview_material_parameters(
    values: Mapping[str, Any], *, mdl_source: str = ""
) -> dict[str, Any]:
    """Translate useful InteriorAgent MDL inputs to USD PreviewSurface values.

    Texture file resolution remains the responsibility of the USD adapter.
    Keeping this numeric translation pure makes it testable without installing
    Pixar USD or NVIDIA MDL.
    """

    base_color = _rgb(
        values.get("BaseColor_Color", values.get("diffuse_color_constant")),
        (0.65, 0.65, 0.65),
    )
    gloss = _average_color(values.get("Gloss_Color"), 0.45)
    metallic = _clamp(_average_color(values.get("Metallic_Color"), 0.0))
    opacity = _clamp(values.get("Opacity", 1.0))
    is_glass = "OmniGlass" in mdl_source
    if is_glass:
        base_color = (0.92, 0.97, 1.0)
        gloss = 0.96
        metallic = 0.0
        opacity = 0.12

    ior = 1.5 if is_glass else _finite(values.get("FresnelB", 1.5), owner="FresnelB")
    emissive_intensity = _finite(
        values.get("EmissiveIntensity", 0.0) or 0.0,
        owner="EmissiveIntensity",
    )
    emissive_color = None
    if emissive_intensity > 0.0:
        raw_emissive = _rgb(values.get("Emissive_Color"), (1.0, 1.0, 1.0))
        emissive_color = tuple(
            _clamp(item * emissive_intensity) for item in raw_emissive
        )

    return {
        "base_color": base_color,
        "roughness": _clamp(1.0 - gloss, 0.03, 0.97),
        "metallic": metallic,
        "opacity": opacity,
        "ior": ior,
        "is_glass": is_glass,
        "use_base_texture": _finite(
            values.get("IsBaseColorTex", 0.0) or 0.0,
            owner="IsBaseColorTex",
        )
        > 0.5,
        "use_normal_texture": _finite(
            values.get("IsNormalTex", 0.0) or 0.0,
            owner="IsNormalTex",
        )
        > 0.5,
        "base_uv": values.get("BaseColor_UVA"),
        "normal_uv": values.get("Normal_UVA"),
        "emissive_color": emissive_color,
    }


def load_room_metadata(path: Path | str) -> list[dict[str, Any]]:
    """Load the simple room polygons shipped with an InteriorAgent scene."""

    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InteriorAgentPlanError(f"cannot load room metadata: {source}: {exc}") from exc
    if not isinstance(value, list) or not value:
        raise InteriorAgentPlanError("room metadata must be a non-empty list")

    result: list[dict[str, Any]] = []
    for index, room in enumerate(value):
        if not isinstance(room, Mapping):
            raise InteriorAgentPlanError(f"rooms[{index}] must be an object")
        room_type = str(room.get("room_type", "")).strip()
        polygon = room.get("polygon")
        if not room_type or not isinstance(polygon, Sequence) or len(polygon) < 3:
            raise InteriorAgentPlanError(f"rooms[{index}] is incomplete")
        points = []
        for point_index, point in enumerate(polygon):
            if (
                not isinstance(point, Sequence)
                or isinstance(point, (str, bytes))
                or len(point) != 2
            ):
                raise InteriorAgentPlanError(
                    f"rooms[{index}].polygon[{point_index}] must be XY"
                )
            points.append(
                [
                    _finite(point[0], owner="room polygon x"),
                    _finite(point[1], owner="room polygon y"),
                ]
            )
        result.append({"room_type": room_type, "polygon_xy_m": points})
    return result


def load_profile(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InteriorAgentPlanError(f"cannot load profile: {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise InteriorAgentPlanError("profile root must be an object")
    return value


def build_kujiale_review_plan(
    profile: Mapping[str, Any],
    *,
    source_stage: Path | str | None = None,
    rooms: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate a compact profile and compile UE-ready camera/light values."""

    if profile.get("schema_version") != PROFILE_SCHEMA:
        raise InteriorAgentPlanError(f"profile schema must be {PROFILE_SCHEMA!r}")
    if profile.get("dataset_id") != DATASET_ID:
        raise InteriorAgentPlanError(f"dataset_id must be {DATASET_ID!r}")
    scene_id = _identifier(profile.get("scene_id"), owner="scene_id")
    room_type = str(profile.get("room_type", "")).strip()
    if not room_type:
        raise InteriorAgentPlanError("room_type must be non-empty")

    map_path = str(profile.get("map_path", ""))
    if not map_path.startswith("/Game/") or not _IDENTIFIER.fullmatch(
        map_path.rsplit("/", 1)[-1]
    ):
        raise InteriorAgentPlanError("map_path must be a stable /Game/... asset path")

    raw_scopes = profile.get("selected_scopes")
    if (
        not isinstance(raw_scopes, Sequence)
        or isinstance(raw_scopes, (str, bytes))
        or not raw_scopes
    ):
        raise InteriorAgentPlanError("selected_scopes must be non-empty")
    scopes = tuple(str(item) for item in raw_scopes)
    if len(set(scopes)) != len(scopes) or any(
        not _SCOPE.fullmatch(item) for item in scopes
    ):
        raise InteriorAgentPlanError(
            "selected_scopes must be unique USD child names without paths"
        )
    required_structure = {"wall", "ceiling", "floor"}
    if not required_structure.issubset(scopes):
        raise InteriorAgentPlanError(
            "selected_scopes must retain wall, ceiling and floor"
        )

    camera_views = []
    camera_ids: set[str] = set()
    raw_views = profile.get("camera_views")
    if (
        not isinstance(raw_views, Sequence)
        or isinstance(raw_views, (str, bytes))
        or len(raw_views) != 4
    ):
        raise InteriorAgentPlanError("camera_views must contain exactly four views")
    for index, raw_view in enumerate(raw_views):
        if not isinstance(raw_view, Mapping):
            raise InteriorAgentPlanError(f"camera_views[{index}] must be an object")
        view_id = _identifier(raw_view.get("view_id"), owner=f"camera_views[{index}]")
        if view_id in camera_ids:
            raise InteriorAgentPlanError(f"duplicate camera view: {view_id}")
        camera_ids.add(view_id)
        position_m = _vector3(
            raw_view.get("position_m"), owner=f"camera_views[{index}].position_m"
        )
        if position_m[2] <= 0.0:
            raise InteriorAgentPlanError("camera height must be positive")
        camera_views.append(
            {
                "view_id": view_id,
                "position_m": list(position_m),
                "position_ue_cm": list(usd_meters_to_unreal_cm(position_m)),
                "yaw_deg": _finite(
                    raw_view.get("yaw_deg"), owner=f"camera_views[{index}].yaw_deg"
                ),
            }
        )

    review_lights = []
    light_ids: set[str] = set()
    raw_lights = profile.get("review_lights")
    if (
        not isinstance(raw_lights, Sequence)
        or isinstance(raw_lights, (str, bytes))
    ):
        raise InteriorAgentPlanError("review_lights must be a list")
    for index, raw_light in enumerate(raw_lights):
        if not isinstance(raw_light, Mapping):
            raise InteriorAgentPlanError(f"review_lights[{index}] must be an object")
        light_id = _identifier(
            raw_light.get("light_id"), owner=f"review_lights[{index}]"
        )
        if light_id in light_ids:
            raise InteriorAgentPlanError(f"duplicate review light: {light_id}")
        light_ids.add(light_id)
        position_m = _vector3(
            raw_light.get("position_m"), owner=f"review_lights[{index}].position_m"
        )
        intensity = _finite(
            raw_light.get("intensity_lumens"),
            owner=f"review_lights[{index}].intensity_lumens",
        )
        radius_m = _finite(
            raw_light.get("attenuation_radius_m"),
            owner=f"review_lights[{index}].attenuation_radius_m",
        )
        temperature = _finite(
            raw_light.get("temperature_kelvin"),
            owner=f"review_lights[{index}].temperature_kelvin",
        )
        source_radius_m = _finite(
            raw_light.get("source_radius_m", 0.0),
            owner=f"review_lights[{index}].source_radius_m",
        )
        soft_radius_m = _finite(
            raw_light.get("soft_source_radius_m", 0.0),
            owner=f"review_lights[{index}].soft_source_radius_m",
        )
        if (
            intensity <= 0.0
            or radius_m <= 0.0
            or not 1000.0 <= temperature <= 20000.0
            or source_radius_m < 0.0
            or soft_radius_m < source_radius_m
        ):
            raise InteriorAgentPlanError(
                f"review_lights[{index}] has invalid photometric values"
            )
        review_lights.append(
            {
                "light_id": light_id,
                "source_prim": str(raw_light.get("source_prim", "")),
                "position_m": list(position_m),
                "position_ue_cm": list(usd_meters_to_unreal_cm(position_m)),
                "intensity_lumens": intensity,
                "attenuation_radius_cm": radius_m * 100.0,
                "temperature_kelvin": temperature,
                "source_radius_cm": source_radius_m * 100.0,
                "soft_source_radius_cm": soft_radius_m * 100.0,
                "generated_review_light": True,
            }
        )

    source = None
    if source_stage is not None:
        source_path = Path(source_stage).expanduser().resolve()
        if not source_path.is_file():
            raise InteriorAgentPlanError(f"source stage does not exist: {source_path}")
        source = str(source_path)

    room_polygons = []
    if rooms is not None:
        for room in rooms:
            if str(room.get("room_type", "")).casefold() == room_type.casefold():
                room_polygons.append(room.get("polygon_xy_m"))
        if not room_polygons:
            raise InteriorAgentPlanError(
                f"room_type {room_type!r} is absent from rooms metadata"
            )

    return {
        "schema_version": PLAN_SCHEMA,
        "backend_role": BACKEND_ROLE,
        "dataset_id": DATASET_ID,
        "scene_id": scene_id,
        "room_type": room_type,
        "source_stage": source,
        "map_path": map_path,
        "selected_scopes": list(scopes),
        "include_rendering_scope": bool(profile.get("include_rendering_scope", True)),
        "room_polygons_xy_m": room_polygons,
        "camera_views": camera_views,
        "review_lights": review_lights,
        "external_asset_policy": {
            "license_url": LICENSE_URL,
            "dataset_url": DATASET_URL,
            "allowed_scope": "noncommercial_research_and_education",
            "redistribute_downloaded_data": False,
            "repository_contains_downloaded_data": False,
            "adapter_mode": "external_usd_references",
        },
        "authority": {
            "visual_pixels": "spear_ue_comparison_only",
            "timeline_navigation_audio_topdown_flags_metadata": (
                "habitat_native_avengine"
            ),
            "review_lights_are_acoustic_truth": False,
        },
        "claim_boundary": (
            "Optional four-view visual canary. It does not qualify the room, "
            "reconstruct physical lighting, or execute an AVEngine episode."
        ),
    }


__all__ = [
    "BACKEND_ROLE",
    "DATASET_ID",
    "DATASET_URL",
    "InteriorAgentPlanError",
    "LICENSE_URL",
    "PLAN_SCHEMA",
    "PROFILE_SCHEMA",
    "build_kujiale_review_plan",
    "load_profile",
    "load_room_metadata",
    "preview_material_parameters",
    "usd_meters_to_unreal_cm",
]
