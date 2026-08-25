"""Native SPEAR Apartment production-visual execution contracts.

The Habitat-native bundle remains authoritative for Timeline actor state,
source logic, source-center qualification, flags, binaural audio, and the QA
Topdown.  This module only binds those records to the already-cooked native
SPEAR Apartment map and its UE visual assets.

Everything in this module is pure Python.  Importing it never imports SPEAR,
starts Unreal, or mutates either repository.  The actual optional runtime is
the small script in ``tools/rooms/run_spear_apartment_canary.py``.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from avengine.contracts.json_io import canonical_json_sha256
from avengine.optional_backends.spear_visual import (
    FRAME_COUNT,
    PLAN_SCHEMA,
    PRODUCTION_VISUAL_ROLE,
    build_spear_visual_plan_from_files,
    camera_ue_yaw_degrees,
    habitat_point_to_apartment_ue_cm,
)
from avengine.capture.orientation import habitat_yaw_degrees_from_xyzw
from avengine.runtime_profiles import (
    load_default_room_runtime_profile_registry,
    load_default_source_asset_runtime_registry,
    resolve_room_runtime_profile,
    resolve_source_asset_alias,
    spear_actor_bindings,
)


SUITE_SCHEMA = "avengine_optional_spear_apartment_suite_v1"
SCENARIO_SCHEMA = "avengine_optional_spear_apartment_scenario_v1"
BACKEND_ROLE = PRODUCTION_VISUAL_ROLE

DEFAULT_SOURCE_ASSET_RUNTIME_REGISTRY = (
    load_default_source_asset_runtime_registry()
)
DEFAULT_ROOM_RUNTIME_PROFILE_REGISTRY = (
    load_default_room_runtime_profile_registry()
)
DEFAULT_ROOM_RUNTIME_PROFILE: Mapping[str, Any] = resolve_room_runtime_profile(
    DEFAULT_ROOM_RUNTIME_PROFILE_REGISTRY
)

# Backward-compatible names are views of editable JSON, not implementation
# authority. New rooms and sources are selected by passing their profile data.
NATIVE_APARTMENT_MAP = str(DEFAULT_ROOM_RUNTIME_PROFILE["scene"]["map_path"])
NATIVE_APARTMENT_SCENE_ID = str(
    DEFAULT_ROOM_RUNTIME_PROFILE["scene"]["scene_id"]
)
WIDTH = int(DEFAULT_ROOM_RUNTIME_PROFILE["render"]["width"])
HEIGHT = int(DEFAULT_ROOM_RUNTIME_PROFILE["render"]["height"])
FPS = int(DEFAULT_ROOM_RUNTIME_PROFILE["render"]["frame_rate_hz"])
STREAMING_WARMUP_FRAMES = int(
    DEFAULT_ROOM_RUNTIME_PROFILE["render"]["streaming_warmup_frames"]
)
CAMERA_WARMUP_FRAMES = int(
    DEFAULT_ROOM_RUNTIME_PROFILE["render"]["camera_warmup_frames"]
)
DURATION_SECONDS = FRAME_COUNT / FPS
POSITION_TOLERANCE_CM = 0.02
ROTATION_TOLERANCE_DEGREES = 0.02
ANIMATION_TOLERANCE_SECONDS = 1.0e-4
COMPONENT_TRANSFORM_TOLERANCE = 1.0e-3
# Skeletal bounds include animated paws/fur and do not share the Timeline
# source-center origin. Keep this as a coarse visual sanity gate only; route
# legality is intentionally decided by the source center. Five centimetres
# still catches a gross import-frame error without moving a visually grounded
# asset merely to make its bounding box equal the source root.
QUADRUPED_FLOOR_TOLERANCE_CM = 5.0
ANATOMICAL_FORWARD_TOLERANCE_DEGREES = 25.0
COMPONENT_FRAME_DELTA_SCHEMA = "avengine_spear_component_frame_delta_v1"
LIGHTING_PROFILE_SCHEMA = "avengine_optional_spear_apartment_lighting_profiles_v1"
ASSET_BOUND_EPISODE_BINDING_SCHEMA = "avengine_m7_asset_bound_apartment_episode_binding_v1"
EXACT_ASSET_BOUND_RUNTIME_BINDING_SCHEMA = "avengine_exact_asset_bound_runtime_binding_v1"
ASSET_BOUND_BUNDLE_SCHEMA = "avengine_m7_asset_bound_apartment_ue_input_bundle_v1"
ACOUSTIC_SELECTION_BINDING_SCHEMA = (
    "avengine_rir_cache_acoustic_selection_binding_v1"
)
ACOUSTIC_VISUAL_IDENTITY_SCHEMA = (
    "avengine_spear_acoustic_visual_runtime_identity_v1"
)
_ASSET_BOUND_SOURCE_SLOTS = ("source1", "source2")
_ROOM_REF_FIELDS = frozenset({"registry_id", "room_id", "revision"})
_ACOUSTIC_SELECTION_FIELDS = frozenset(
    {
        "schema",
        "selection_mode",
        "registry_selection_applied",
        "room_ref",
        "profile_ref",
        "binding_id",
        "registry_selection_content_sha256",
        "effective_selection_content_sha256",
        "acoustic_package_manifest_sha256",
        "simulation_request_sha256",
        "input_receipt_sha256",
        "binding_content_sha256",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UNSPECIFIED_ACOUSTIC_BINDING_SHA256 = object()


NATIVE_LIGHTING_PROFILE: Mapping[str, Any] = {
    "profile_id": "native",
    "label": "Unmodified native Apartment lighting",
    "generated_lights": [],
    "claim_boundary": "native SPEAR map lighting and post-process only",
}


SCENARIO_DIRECTORIES: Mapping[str, tuple[str, str]] = {
    "S0": ("S0_routing_sanity", "A"),
    "S3": ("S3_moving_source", "A"),
    "S4": ("S4_overlapping_sources", "A"),
}


MOTION_PILOT_DIRECTORIES: Mapping[str, str] = {
    "P0": "00_static_static",
    "P1": "01_human_moving_dog_static",
    "P2": "02_both_moving",
    "P3": "03_human_static_dog_moving",
}


BEAGLE_ASSET_ID = str(
    resolve_source_asset_alias(
        DEFAULT_SOURCE_ASSET_RUNTIME_REGISTRY, "legacy_beagle"
    )["asset_id"]
)
HUMAN_ASSET_ID = str(
    resolve_source_asset_alias(
        DEFAULT_SOURCE_ASSET_RUNTIME_REGISTRY, "legacy_human"
    )["asset_id"]
)
BORDER_COLLIE_ASSET_ID = str(
    resolve_source_asset_alias(
        DEFAULT_SOURCE_ASSET_RUNTIME_REGISTRY, "current_generated_dog"
    )["asset_id"]
)
CAT_ASSET_ID = str(
    resolve_source_asset_alias(
        DEFAULT_SOURCE_ASSET_RUNTIME_REGISTRY, "current_generated_cat"
    )["asset_id"]
)
DEFAULT_ACTOR_BINDINGS: Mapping[str, Mapping[str, Any]] = spear_actor_bindings(
    DEFAULT_SOURCE_ASSET_RUNTIME_REGISTRY
)


class SpearApartmentError(ValueError):
    """The Apartment production visual cannot preserve its authority boundary."""


def _validated_room_ref(value: Any, *, owner: str) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _ROOM_REF_FIELDS
        or any(
            not isinstance(value.get(key), str) or not value[key].strip()
            for key in _ROOM_REF_FIELDS
        )
    ):
        raise SpearApartmentError(
            f"{owner} must be an exact registry_id/room_id/revision reference"
        )
    return {key: str(value[key]) for key in ("registry_id", "room_id", "revision")}


def _validated_room_runtime_profile(
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the subset owned by the SPEAR Apartment adapter.

    Room identity and scene selection are data-driven. The adapter still owns
    one fixed five-second camera/render transport, so a room using a different
    transport must register a different adapter instead of silently changing
    this one.
    """

    if not isinstance(profile, Mapping):
        raise SpearApartmentError("room runtime profile must be a mapping")
    if (
        profile.get("backend_id") != "spear_unreal"
        or profile.get("adapter_id") != "spear_apartment_v1"
    ):
        raise SpearApartmentError(
            "room runtime profile is not compatible with spear_apartment_v1"
        )
    room_ref = profile.get("room_ref")
    scene = profile.get("scene")
    render = profile.get("render")
    if not all(isinstance(value, Mapping) for value in (room_ref, scene, render)):
        raise SpearApartmentError("room runtime profile is incomplete")
    expected_render = {
        "width": WIDTH,
        "height": HEIGHT,
        "frame_count": FRAME_COUNT,
        "frame_rate_hz": FPS,
        "horizontal_fov_deg": 105.0,
        "streaming_warmup_frames": STREAMING_WARMUP_FRAMES,
        "camera_warmup_frames": CAMERA_WARMUP_FRAMES,
    }
    if dict(render) != expected_render:
        raise SpearApartmentError(
            "spear_apartment_v1 room profile changed its fixed render transport"
        )
    validated_room_ref = _validated_room_ref(
        room_ref, owner="room runtime profile room_ref"
    )
    for owner, value in (
        ("profile_id", profile.get("profile_id")),
        ("scene_id", scene.get("scene_id")),
        ("map_path", scene.get("map_path")),
    ):
        if not isinstance(value, str) or not value.strip():
            raise SpearApartmentError(f"room runtime profile {owner} is invalid")
    if not str(scene["map_path"]).startswith("/Game/"):
        raise SpearApartmentError("room runtime map path must start with /Game/")
    result = deepcopy(dict(profile))
    result["room_ref"] = validated_room_ref
    return result


def _finite_scalar(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpearApartmentError(f"{owner} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise SpearApartmentError(f"{owner} must be finite")
    return result


def resolve_apartment_lighting_profile(
    document: Mapping[str, Any], profile_id: str | None = None
) -> dict[str, Any]:
    """Resolve one small visual-only runtime-light preset from JSON data."""

    if document.get("schema") != LIGHTING_PROFILE_SCHEMA:
        raise SpearApartmentError(
            f"lighting profile schema must be {LIGHTING_PROFILE_SCHEMA}"
        )
    selected_id = profile_id or document.get("default_profile_id")
    if not isinstance(selected_id, str) or not selected_id.strip():
        raise SpearApartmentError("lighting profile id must be non-empty")
    raw_profiles = document.get("profiles")
    if not isinstance(raw_profiles, list):
        raise SpearApartmentError("lighting profiles must be a list")
    matches = [
        value
        for value in raw_profiles
        if isinstance(value, Mapping) and value.get("profile_id") == selected_id
    ]
    if len(matches) != 1:
        raise SpearApartmentError(
            f"lighting profile {selected_id!r} must resolve exactly once"
        )
    raw = matches[0]
    label = raw.get("label")
    claim = raw.get("claim_boundary")
    if not isinstance(label, str) or not label.strip():
        raise SpearApartmentError("lighting profile label must be non-empty")
    if not isinstance(claim, str) or not claim.strip():
        raise SpearApartmentError("lighting profile claim_boundary must be non-empty")
    raw_lights = raw.get("generated_lights")
    if not isinstance(raw_lights, list):
        raise SpearApartmentError("generated_lights must be a list")
    lights = []
    light_ids: set[str] = set()
    for index, light in enumerate(raw_lights):
        if not isinstance(light, Mapping):
            raise SpearApartmentError(f"generated_lights[{index}] must be an object")
        light_id = light.get("light_id")
        if (
            not isinstance(light_id, str)
            or not light_id.strip()
            or light_id in light_ids
        ):
            raise SpearApartmentError(
                "generated light ids must be non-empty and unique"
            )
        light_ids.add(light_id)
        position = _finite_triplet(
            light.get("position_ue_cm"), owner=f"generated light {light_id} position"
        )
        intensity = _finite_scalar(
            light.get("intensity_lumens"), owner=f"generated light {light_id} intensity"
        )
        attenuation = _finite_scalar(
            light.get("attenuation_radius_cm"),
            owner=f"generated light {light_id} attenuation",
        )
        temperature = _finite_scalar(
            light.get("temperature_kelvin"),
            owner=f"generated light {light_id} temperature",
        )
        source_radius = _finite_scalar(
            light.get("source_radius_cm", 0.0),
            owner=f"generated light {light_id} source radius",
        )
        soft_radius = _finite_scalar(
            light.get("soft_source_radius_cm", 0.0),
            owner=f"generated light {light_id} soft source radius",
        )
        cast_shadows = light.get("cast_shadows", True)
        if not isinstance(cast_shadows, bool):
            raise SpearApartmentError(
                f"generated light {light_id} cast_shadows must be boolean"
            )
        if (
            intensity <= 0.0
            or attenuation <= 0.0
            or not 1000.0 <= temperature <= 20_000.0
            or source_radius < 0.0
            or soft_radius < source_radius
        ):
            raise SpearApartmentError(f"generated light {light_id} is not physical")
        lights.append(
            {
                "light_id": light_id,
                "light_type": "point",
                "position_ue_cm": position,
                "intensity_lumens": intensity,
                "attenuation_radius_cm": attenuation,
                "temperature_kelvin": temperature,
                "source_radius_cm": source_radius,
                "soft_source_radius_cm": soft_radius,
                "cast_shadows": cast_shadows,
            }
        )
    return {
        "profile_id": selected_id,
        "label": label.strip(),
        "generated_lights": lights,
        "claim_boundary": claim.strip(),
    }


def load_apartment_lighting_profile(
    path: str | Path, profile_id: str | None = None
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise SpearApartmentError(f"lighting profile file is missing: {source}")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpearApartmentError(
            f"could not read lighting profiles: {source}"
        ) from exc
    if not isinstance(document, Mapping):
        raise SpearApartmentError("lighting profile document must be an object")
    return resolve_apartment_lighting_profile(document, profile_id)


def _direct_file(path: Path, *, owner: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise SpearApartmentError(f"{owner} is missing: {resolved}")
    return resolved


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise SpearApartmentError(f"input escapes bundle root: {path}") from exc


def _finite_triplet(value: Any, *, owner: str) -> list[float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise SpearApartmentError(f"{owner} must contain exactly three finite numbers")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise SpearApartmentError(f"{owner}[{index}] must be finite")
        number = float(item)
        if not math.isfinite(number):
            raise SpearApartmentError(f"{owner}[{index}] must be finite")
        result.append(number)
    return result


def _component_frame_delta(
    value: Mapping[str, Any], *, asset_id: str
) -> dict[str, Any]:
    raw = value.get("ue_component_frame_delta")
    if not isinstance(raw, Mapping):
        raise SpearApartmentError(
            f"actor binding {asset_id!r} lacks ue_component_frame_delta"
        )
    if raw.get("schema") != COMPONENT_FRAME_DELTA_SCHEMA:
        raise SpearApartmentError(
            f"actor binding {asset_id!r} has an unsupported component delta schema"
        )
    if raw.get("composition") != "add_relative_preserving_blueprint_transform":
        raise SpearApartmentError(
            f"actor binding {asset_id!r} may not replace the Blueprint transform"
        )
    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise SpearApartmentError(
            f"actor binding {asset_id!r} component delta lacks a reason"
        )
    return {
        "schema": COMPONENT_FRAME_DELTA_SCHEMA,
        "rotation_deg": _finite_triplet(
            raw.get("rotation_deg"),
            owner=f"actor binding {asset_id!r} component rotation delta",
        ),
        "translation_cm": _finite_triplet(
            raw.get("translation_cm"),
            owner=f"actor binding {asset_id!r} component translation delta",
        ),
        "composition": "add_relative_preserving_blueprint_transform",
        "reason": reason,
    }


def component_frame_delta_for_asset(
    asset_id: str,
    *,
    actor_bindings: Mapping[str, Mapping[str, Any]] = DEFAULT_ACTOR_BINDINGS,
) -> dict[str, Any]:
    """Return one validated asset-local UE frame correction.

    The declaration is asset-specific rather than room-specific, so optional
    Apartment, MP3D and future room renderers can share the same imported
    animal/human frame without duplicating calibration numbers.
    """

    binding = actor_bindings.get(asset_id)
    if not isinstance(binding, Mapping):
        raise SpearApartmentError(
            f"actor binding {asset_id!r} must expose its component frame delta"
        )
    return _component_frame_delta(binding, asset_id=asset_id)


def anatomical_basis_bones_for_asset(
    asset_id: str,
    *,
    actor_bindings: Mapping[str, Mapping[str, Any]] = DEFAULT_ACTOR_BINDINGS,
) -> dict[str, str] | None:
    """Return an optional, asset-specific rendered-skeleton role mapping.

    Named Rocketbox and Quaternius rigs remain auto-detected at runtime. A
    generated rig with anonymous bone names must instead publish the exact
    five roles inferred while that instance was bound; the indices are never
    generalized to another asset merely because its bones are also numbered.
    """

    binding = actor_bindings.get(asset_id)
    if not isinstance(binding, Mapping):
        raise SpearApartmentError(f"actor binding {asset_id!r} is missing")
    raw = binding.get("ue_anatomical_basis_bones")
    if raw is None:
        return None
    required = {"rear", "front", "body", "left_foot", "right_foot"}
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise SpearApartmentError(
            f"actor binding {asset_id!r} anatomical basis must define exactly "
            f"{sorted(required)}"
        )
    result: dict[str, str] = {}
    for role in sorted(required):
        bone_name = raw[role]
        if not isinstance(bone_name, str) or not bone_name.strip():
            raise SpearApartmentError(
                f"actor binding {asset_id!r} anatomical role {role!r} is invalid"
            )
        result[role] = bone_name
    return result


def _unreal_struct_triplet(
    value: Any, names: Sequence[str], *, owner: str
) -> list[float]:
    expected = [name.casefold() for name in names]
    current = value
    for _ in range(3):
        if not isinstance(current, Mapping):
            break
        lowered = {str(key).casefold(): item for key, item in current.items()}
        if all(name in lowered for name in expected):
            return _finite_triplet([lowered[name] for name in expected], owner=owner)
        if "returnvalue" in lowered and isinstance(lowered["returnvalue"], Mapping):
            current = lowered["returnvalue"]
            continue
        if len(current) == 1:
            candidate = next(iter(current.values()))
            if isinstance(candidate, Mapping):
                current = candidate
                continue
        break
    raise SpearApartmentError(f"could not read {owner}: {value}")


def read_ue_component_relative_transform(component: Any) -> dict[str, list[float]]:
    """Read a SPEAR-wrapped SceneComponent transform without importing SPEAR."""

    return {
        "translation_cm": _unreal_struct_triplet(
            component.get_property_value(property_name="RelativeLocation"),
            ("x", "y", "z"),
            owner="UE component RelativeLocation",
        ),
        "rotation_deg": _unreal_struct_triplet(
            component.get_property_value(property_name="RelativeRotation"),
            ("roll", "pitch", "yaw"),
            owner="UE component RelativeRotation",
        ),
    }


def _rotator_quaternion_xyzw(rotation_deg: Sequence[float]) -> tuple[float, ...]:
    """Convert UE-style Roll/Pitch/Yaw degrees to an equivalent quaternion.

    Rotators at +/-90 degree pitch have multiple valid Euler representations;
    comparing their three reported components independently can therefore
    report a false 180 degree error for the same physical orientation.
    """

    roll, pitch, yaw = (math.radians(float(value)) / 2.0 for value in rotation_deg)
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _rotator_equivalence_error_degrees(
    expected_deg: Sequence[float], observed_deg: Sequence[float]
) -> float:
    expected = _rotator_quaternion_xyzw(expected_deg)
    observed = _rotator_quaternion_xyzw(observed_deg)
    dot = abs(sum(left * right for left, right in zip(expected, observed, strict=True)))
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))


def apply_ue_component_frame_delta(
    component: Any, declaration: Mapping[str, Any]
) -> dict[str, Any]:
    """Compose an asset-local delta on an attached visual root component.

    The operation deliberately calls Unreal's relative-*add* functions after
    reading the component's authored Blueprint transform.  At runtime this
    component belongs to the visual child actor, not the hidden actor that
    owns the authoritative Timeline/source-center transform.  The function
    never replaces the component transform with a hard-coded absolute value.
    """

    delta = declaration.get("ue_component_frame_delta")
    if not isinstance(delta, Mapping):
        raise SpearApartmentError("actor declaration lacks component frame delta")
    before = read_ue_component_relative_transform(component)
    rotation = _finite_triplet(
        delta.get("rotation_deg"), owner="UE component rotation delta"
    )
    translation = _finite_triplet(
        delta.get("translation_cm"), owner="UE component translation delta"
    )
    if any(abs(value) > 0.0 for value in rotation):
        component.K2_AddRelativeRotation(
            DeltaRotation={
                "Roll": rotation[0],
                "Pitch": rotation[1],
                "Yaw": rotation[2],
            },
            bSweep=False,
            bTeleport=True,
        )
    if any(abs(value) > 0.0 for value in translation):
        component.K2_AddRelativeLocation(
            DeltaLocation={
                "X": translation[0],
                "Y": translation[1],
                "Z": translation[2],
            },
            bSweep=False,
            bTeleport=True,
        )
    after = read_ue_component_relative_transform(component)
    translation_error = max(
        abs(
            (after["translation_cm"][axis] - before["translation_cm"][axis])
            - translation[axis]
        )
        for axis in range(3)
    )
    component_rotation_error = max(
        abs(
            (
                after["rotation_deg"][axis]
                - before["rotation_deg"][axis]
                - rotation[axis]
                + 180.0
            )
            % 360.0
            - 180.0
        )
        for axis in range(3)
    )
    expected_rotation = [
        before["rotation_deg"][axis] + rotation[axis] for axis in range(3)
    ]
    quaternion_rotation_error = _rotator_equivalence_error_degrees(
        expected_rotation, after["rotation_deg"]
    )
    rotation_error = min(component_rotation_error, quaternion_rotation_error)
    if (
        translation_error > COMPONENT_TRANSFORM_TOLERANCE
        or rotation_error > COMPONENT_TRANSFORM_TOLERANCE
    ):
        raise SpearApartmentError(
            f"{declaration.get('actor_id', '<unknown>')} component frame delta "
            f"readback failed: translation_error={translation_error}, "
            f"rotation_error={rotation_error}"
        )
    return {
        "status": "pass",
        "asset_id": declaration.get("asset_id"),
        "blueprint_relative_before": before,
        "applied_delta": deepcopy(dict(delta)),
        "blueprint_relative_after": after,
        "maximum_translation_delta_error_cm": translation_error,
        "maximum_rotation_delta_error_deg": rotation_error,
        "euler_component_rotation_delta_error_deg": component_rotation_error,
        "quaternion_equivalence_rotation_error_deg": quaternion_rotation_error,
        "timeline_anchor_mutated": False,
        "target": "attached_visual_actor_root_component",
    }


def scenario_input_paths(bundle_root: str | Path, scenario_id: str) -> dict[str, Path]:
    """Resolve one S0/S3/S4 input closure inside an existing M6.x bundle."""

    root = Path(bundle_root).resolve()
    if not root.is_dir():
        raise SpearApartmentError(f"bundle root is missing: {root}")
    try:
        scenario_directory, variant_id = SCENARIO_DIRECTORIES[scenario_id]
    except KeyError as exc:
        raise SpearApartmentError(
            f"unsupported native Apartment scenario: {scenario_id!r}"
        ) from exc
    variant = root / "scenarios" / scenario_directory / "variants" / variant_id
    metadata = variant / "metadata"
    videos = variant / "videos"
    return {
        "timeline": _direct_file(metadata / "timeline.json", owner="Timeline"),
        "source_manifest": _direct_file(
            metadata / "source_manifest.json", owner="source manifest"
        ),
        "flags": _direct_file(metadata / "flags.json", owner="flag report"),
        "room_capsule": _direct_file(
            root / "inputs/fixed_apartment_config/room_capsule.json",
            owner="RoomCapsule",
        ),
        "qualification": _direct_file(
            root / "room/qualification.json", owner="room qualification"
        ),
        "authoritative_clean_binaural": _direct_file(
            videos / "clean_binaural.mp4", owner="clean binaural review"
        ),
        "authoritative_diagnostic_topdown": _direct_file(
            videos / "diagnostic_topdown_binaural.mp4",
            owner="diagnostic Topdown review",
        ),
    }


def motion_pilot_input_paths(
    bundle_root: str | Path, scenario_id: str
) -> dict[str, Path]:
    """Resolve one P0--P3 input closure from a four-motion pilot bundle."""

    root = Path(bundle_root).resolve()
    if not root.is_dir():
        raise SpearApartmentError(f"bundle root is missing: {root}")
    try:
        episode_directory = MOTION_PILOT_DIRECTORIES[scenario_id]
    except KeyError as exc:
        raise SpearApartmentError(
            f"unsupported Apartment motion-pilot scenario: {scenario_id!r}"
        ) from exc
    episode = root / "episodes" / episode_directory
    metadata = episode / "metadata"
    videos = episode / "videos"
    return {
        "timeline": _direct_file(metadata / "timeline.json", owner="Timeline"),
        "source_manifest": _direct_file(
            metadata / "source_manifest.json", owner="source manifest"
        ),
        "flags": _direct_file(metadata / "flags.json", owner="flag report"),
        "room_capsule": _direct_file(
            root / "room/room_capsule.json", owner="RoomCapsule"
        ),
        "qualification": _direct_file(
            root / "room/qualification.json", owner="room qualification"
        ),
        "authoritative_clean_binaural": _direct_file(
            videos / "clean_binaural.mp4", owner="clean binaural review"
        ),
        "authoritative_diagnostic_topdown": _direct_file(
            videos / "diagnostic_topdown_binaural.mp4",
            owner="diagnostic Topdown review",
        ),
    }


def asset_bound_episode_input_paths(
    bundle_root: str | Path, episode_id: str
) -> dict[str, Path]:
    """Resolve one generic source1/source2 episode from an M7 UE input bundle."""

    root = Path(bundle_root).resolve()
    if not root.is_dir():
        raise SpearApartmentError(f"bundle root is missing: {root}")
    if (
        not isinstance(episode_id, str)
        or not episode_id
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in episode_id)
    ):
        raise SpearApartmentError(f"invalid asset-bound episode ID: {episode_id!r}")
    episode = root / "episodes" / episode_id
    metadata = episode / "metadata"
    videos = episode / "videos"
    paths = {
        "timeline": _direct_file(metadata / "timeline.json", owner="Timeline"),
        "source_manifest": _direct_file(
            metadata / "source_manifest.json", owner="source manifest"
        ),
        "flags": _direct_file(metadata / "flags.json", owner="flag report"),
        "batch_binding": _direct_file(
            metadata / "batch_binding.json", owner="asset-bound batch binding"
        ),
        "room_capsule": _direct_file(
            root / "room/room_capsule.json", owner="RoomCapsule"
        ),
        "qualification": _direct_file(
            root / "room/qualification.json", owner="room qualification"
        ),
        "authoritative_clean_binaural": _direct_file(
            videos / "clean_binaural.mp4", owner="clean binaural review"
        ),
        "authoritative_diagnostic_topdown": _direct_file(
            videos / "diagnostic_topdown_binaural.mp4",
            owner="diagnostic Topdown review",
        ),
    }
    sensor_rig_trajectory = metadata / "sensor_rig_trajectory.json"
    if sensor_rig_trajectory.exists():
        paths["sensor_rig_trajectory"] = _direct_file(
            sensor_rig_trajectory, owner="SensorRigTrajectory"
        )
    return paths


def _load_asset_bound_bundle_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpearApartmentError(
            f"could not read asset-bound bundle manifest: {manifest_path}"
        ) from exc
    if not isinstance(value, dict):
        raise SpearApartmentError(
            "asset-bound bundle manifest must be a JSON object"
        )
    return value


def asset_bound_bundle_acoustic_visual_identity(
    bundle_root: str | Path,
    *,
    room_runtime_profile: Mapping[str, Any] = DEFAULT_ROOM_RUNTIME_PROFILE,
) -> dict[str, Any]:
    """Close the bundle's acoustic/visual identity against the UE room.

    Registry-selected acoustics are verified only when the canonical binding,
    the visual RoomCapsule identity, and the selected SPEAR runtime room are
    exactly equal. Explicit legacy modes remain usable for compatibility, but
    are deliberately recorded as ``not_verified`` and receive no inferred
    acoustic room identity.
    """

    root = Path(bundle_root).resolve()
    if not root.is_dir():
        raise SpearApartmentError(f"bundle root is missing: {root}")
    manifest = _load_asset_bound_bundle_manifest(root)
    if (
        manifest.get("schema") != ASSET_BOUND_BUNDLE_SCHEMA
        or manifest.get("status") != "pass"
    ):
        raise SpearApartmentError(
            "asset-bound bundle manifest identity/status is invalid"
        )

    room_profile = _validated_room_runtime_profile(room_runtime_profile)
    runtime_room_ref = _validated_room_ref(
        room_profile["room_ref"], owner="selected runtime room_ref"
    )
    visual_room_ref = _validated_room_ref(
        manifest.get("visual_room_ref"), owner="bundle visual_room_ref"
    )
    if visual_room_ref != runtime_room_ref:
        raise SpearApartmentError(
            "bundle visual_room_ref differs from the selected runtime room_ref"
        )

    raw_binding = manifest.get("acoustic_selection_binding")
    if (
        not isinstance(raw_binding, Mapping)
        or set(raw_binding) != _ACOUSTIC_SELECTION_FIELDS
        or raw_binding.get("schema") != ACOUSTIC_SELECTION_BINDING_SCHEMA
    ):
        raise SpearApartmentError(
            "bundle acoustic_selection_binding is invalid"
        )
    binding = deepcopy(dict(raw_binding))
    mode = binding.get("selection_mode")
    binding_sha256 = binding.get("binding_content_sha256")
    acoustic_room_ref: dict[str, str] | None
    verification_status: str
    status: str
    compatibility: str | None

    if mode == "explicit_legacy_unbound":
        if (
            binding_sha256 is not None
            or binding.get("registry_selection_applied") is not False
            or binding.get("room_ref") is not None
            or binding.get("profile_ref") is not None
            or binding.get("binding_id") is not None
        ):
            raise SpearApartmentError(
                "legacy unbound acoustic selection fabricated an identity"
            )
        acoustic_room_ref = None
        verification_status = "not_verified"
        status = "not_verified"
        compatibility = "legacy_acoustic_selection_without_room_ref"
    else:
        if (
            mode
            not in {
                "explicit_legacy",
                "registry",
                "registry_with_verified_equivalent_overrides",
            }
            or not isinstance(binding_sha256, str)
            or _SHA256_RE.fullmatch(binding_sha256) is None
            or canonical_json_sha256(
                {
                    key: item
                    for key, item in binding.items()
                    if key != "binding_content_sha256"
                }
            )
            != binding_sha256
        ):
            raise SpearApartmentError(
                "bundle acoustic selection binding hash is invalid"
            )
        if mode == "explicit_legacy":
            if (
                binding.get("registry_selection_applied") is not False
                or binding.get("room_ref") is not None
                or binding.get("profile_ref") is not None
                or binding.get("binding_id") is not None
            ):
                raise SpearApartmentError(
                    "explicit legacy acoustic selection contains a registry identity"
                )
            acoustic_room_ref = None
            verification_status = "not_verified"
            status = "not_verified"
            compatibility = "legacy_acoustic_selection_without_room_ref"
        else:
            profile_ref = binding.get("profile_ref")
            if (
                binding.get("registry_selection_applied") is not True
                or not isinstance(profile_ref, Mapping)
                or set(profile_ref) != {"profile_id", "revision"}
                or any(
                    not isinstance(profile_ref.get(key), str)
                    or not profile_ref[key].strip()
                    for key in ("profile_id", "revision")
                )
                or not isinstance(binding.get("binding_id"), str)
                or not binding["binding_id"].strip()
            ):
                raise SpearApartmentError(
                    "registry acoustic selection lacks its exact profile binding"
                )
            acoustic_room_ref = _validated_room_ref(
                binding.get("room_ref"),
                owner="acoustic selection room_ref",
            )
            if acoustic_room_ref != visual_room_ref:
                raise SpearApartmentError(
                    "acoustic selection room_ref differs from visual/runtime room_ref"
                )
            verification_status = "verified"
            status = "pass"
            compatibility = None

    alignment = manifest.get("acoustic_visual_room_alignment")
    if not isinstance(alignment, Mapping):
        raise SpearApartmentError(
            "bundle lacks acoustic/visual room alignment evidence"
        )
    alignment_visual_room_ref = _validated_room_ref(
        alignment.get("visual_room_ref"),
        owner="acoustic/visual alignment visual_room_ref",
    )
    if alignment_visual_room_ref != visual_room_ref:
        raise SpearApartmentError(
            "bundle acoustic/visual alignment changed visual_room_ref"
        )
    if acoustic_room_ref is None:
        if (
            alignment.get("status") != "not_verified"
            or alignment.get("acoustic_room_ref") is not None
            or alignment.get("compatibility")
            != "legacy_acoustic_selection_without_room_ref"
        ):
            raise SpearApartmentError(
                "legacy acoustic/visual alignment fabricated a pass"
            )
    else:
        alignment_acoustic_room_ref = _validated_room_ref(
            alignment.get("acoustic_room_ref"),
            owner="acoustic/visual alignment acoustic_room_ref",
        )
        if (
            alignment.get("status") != "pass"
            or alignment.get("compatibility") is not None
            or alignment_acoustic_room_ref != acoustic_room_ref
        ):
            raise SpearApartmentError(
                "bundle acoustic/visual alignment differs from the binding"
            )

    return {
        "schema": ACOUSTIC_VISUAL_IDENTITY_SCHEMA,
        "status": status,
        "verification_status": verification_status,
        "selection_mode": mode,
        "compatibility": compatibility,
        "acoustic_selection_binding_sha256": binding_sha256,
        "binding_id": binding.get("binding_id"),
        "profile_ref": deepcopy(binding.get("profile_ref")),
        "visual_room_ref": visual_room_ref,
        "acoustic_room_ref": acoustic_room_ref,
        "runtime_room_ref": runtime_room_ref,
        "runtime_profile_id": room_profile["profile_id"],
        "runtime_map_id": room_profile["scene"]["scene_id"],
        "runtime_map_path": room_profile["scene"]["map_path"],
    }


def _attach_exact_asset_bound_runtime_bindings(
    *,
    plan: dict[str, Any],
    scenario_id: str,
    batch_binding_path: Path,
    actor_bindings: Mapping[str, Mapping[str, Any]],
    expected_acoustic_selection_binding_sha256: Any = (
        _UNSPECIFIED_ACOUSTIC_BINDING_SHA256
    ),
) -> None:
    """Attach exact snapshots when present; keep legacy batch bindings valid."""

    try:
        batch = json.loads(batch_binding_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpearApartmentError("could not read asset-bound batch binding") from exc
    if (
        not isinstance(batch, Mapping)
        or batch.get("schema") != ASSET_BOUND_EPISODE_BINDING_SCHEMA
        or batch.get("status") != "pass"
        or batch.get("episode_id") != scenario_id
    ):
        raise SpearApartmentError(
            "asset-bound batch binding identity/status is invalid"
        )
    if (
        expected_acoustic_selection_binding_sha256
        is not _UNSPECIFIED_ACOUSTIC_BINDING_SHA256
        and (
            "acoustic_selection_binding_sha256" not in batch
            or batch.get("acoustic_selection_binding_sha256")
            != expected_acoustic_selection_binding_sha256
        )
    ):
        raise SpearApartmentError(
            "episode acoustic selection binding SHA differs from its bundle"
        )
    selected_assets = batch.get("asset_ids_by_source_slot")
    if not isinstance(selected_assets, Mapping) or set(selected_assets) != set(
        _ASSET_BOUND_SOURCE_SLOTS
    ):
        raise SpearApartmentError(
            "asset-bound batch binding must select source1/source2 assets"
        )
    exact_by_slot = batch.get("runtime_bindings_by_source_slot")
    if exact_by_slot is None:
        return
    if not isinstance(exact_by_slot, Mapping) or not set(exact_by_slot) <= set(
        _ASSET_BOUND_SOURCE_SLOTS
    ):
        raise SpearApartmentError(
            "runtime_bindings_by_source_slot contains an invalid source slot"
        )

    actors = {
        actor.get("actor_id"): actor
        for actor in plan.get("actors", ())
        if isinstance(actor, dict)
    }
    for slot, exact in exact_by_slot.items():
        if not isinstance(exact, Mapping):
            raise SpearApartmentError(f"{slot} exact runtime snapshot is incomplete")
        actor_id = f"{slot}_actor"
        asset_id = selected_assets.get(slot)
        actor = actors.get(actor_id)
        raw_binding = actor_bindings.get(str(asset_id))
        timeline = exact.get("timeline")
        spear = exact.get("spear_unreal")
        values = (
            actor,
            raw_binding,
            timeline,
            spear,
            exact.get("emitter"),
            exact.get("asset_bound_lineage"),
        )
        if not all(isinstance(value, Mapping) for value in values):
            raise SpearApartmentError(f"{slot} exact runtime snapshot is incomplete")
        assert isinstance(actor, dict)
        assert isinstance(raw_binding, Mapping)
        assert isinstance(timeline, Mapping)
        assert isinstance(spear, Mapping)
        revision = exact.get("asset_revision")
        if (
            not isinstance(revision, str)
            or (
                exact.get("schema"),
                exact.get("source_slot_id"),
                exact.get("asset_id"),
            )
            != (EXACT_ASSET_BOUND_RUNTIME_BINDING_SCHEMA, slot, asset_id)
            or (actor.get("asset_id"), raw_binding.get("asset_revision"))
            != (asset_id, revision)
            or (
                timeline.get("template_id"),
                timeline.get("body_plan_id"),
                timeline.get("local_anatomical_forward_axis"),
            )
            != (
                actor.get("template_id"),
                actor.get("body_plan_id"),
                actor.get("habitat_local_anatomical_forward_axis"),
            )
        ):
            raise SpearApartmentError(
                f"{slot} exact runtime snapshot differs from compiled inputs"
            )

        runtime_binding = {
            key: value
            for key, value in raw_binding.items()
            if key != "asset_revision"
        }
        if dict(spear) != runtime_binding:
            raise SpearApartmentError(
                f"{slot} exact SPEAR/UE binding differs from actor bindings"
            )
        actor_scale = exact.get("actor_scale")
        action_paths = timeline.get("animation_paths_by_action_id")
        if (
            isinstance(actor_scale, bool)
            or not isinstance(actor_scale, (int, float))
            or not math.isfinite(float(actor_scale))
            or float(actor_scale) <= 0.0
            or spear.get("actor_scale") != actor_scale
            or not isinstance(action_paths, Mapping)
            or dict(action_paths) != spear.get("animation_paths_by_action_id")
        ):
            raise SpearApartmentError(
                f"{slot} exact scale/action binding is invalid"
            )

        actor["actor_scale"] = float(actor_scale)
        actor["animation_paths_by_action_id"] = deepcopy(dict(action_paths))
        actor["exact_runtime_binding"] = deepcopy(dict(exact))


def materialize_camera_states(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Return one validated expected UE camera state per formal frame.

    Current plans carry ``frames[].camera_state``.  Historical retained plans
    are accepted only when every frame lacks that field, in which case the old
    top-level fixed pose is repeated explicitly.  A partially upgraded plan is
    rejected rather than mixing moving metadata with a fixed renderer.
    """

    # Lazy for the same reason as spear_visual._compiled_camera_states: the
    # formal SensorRig contract must remain independently importable.
    from avengine.sensor_rig_trajectory import (
        SensorRigTrajectoryError,
        compute_sensor_rig_pose_hash,
    )

    camera = plan.get("camera")
    frames = plan.get("frames")
    if not isinstance(camera, Mapping):
        raise SpearApartmentError("compiled plan has no camera")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise SpearApartmentError("compiled plan frames must be a sequence")
    if len(frames) != FRAME_COUNT or not all(
        isinstance(frame, Mapping) for frame in frames
    ):
        raise SpearApartmentError("camera state materialization requires 75 frames")

    per_frame = [frame.get("camera_state") for frame in frames]
    present = [state is not None for state in per_frame]
    if any(present) and not all(present):
        raise SpearApartmentError(
            "compiled plan only partially provides per-frame camera_state"
        )

    if not any(present):
        position = _finite_triplet(
            camera.get("ue_position_cm"), owner="legacy camera.ue_position_cm"
        )
        yaw = camera.get("ue_yaw_deg")
        if (
            isinstance(yaw, bool)
            or not isinstance(yaw, (int, float))
            or not math.isfinite(float(yaw))
        ):
            raise SpearApartmentError("legacy camera.ue_yaw_deg must be finite")
        return tuple(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * 3_200,
                "ue_position_cm": list(position),
                "ue_yaw_deg": float(yaw),
                "state_source": "legacy_top_level_fixed_camera_compatibility",
            }
            for frame_index in range(FRAME_COUNT)
        )

    result: list[dict[str, Any]] = []
    for frame_index, (frame, state) in enumerate(zip(frames, per_frame)):
        assert isinstance(frame, Mapping)
        if not isinstance(state, Mapping):
            raise SpearApartmentError(
                f"frame {frame_index} camera_state must be a mapping"
            )
        if (
            frame.get("frame_index") != frame_index
            or frame.get("pts_ticks") != frame_index * 3_200
            or state.get("frame_index") != frame_index
            or state.get("pts_ticks") != frame_index * 3_200
        ):
            raise SpearApartmentError(
                f"frame {frame_index} camera_state is off the Timeline clock"
            )
        world_from_rig = state.get("world_from_rig")
        if not isinstance(world_from_rig, Mapping):
            raise SpearApartmentError(
                f"frame {frame_index} camera_state lacks world_from_rig"
            )
        try:
            expected_hash = compute_sensor_rig_pose_hash(world_from_rig)
        except SensorRigTrajectoryError as exc:
            raise SpearApartmentError(
                f"frame {frame_index} camera_state world_from_rig is invalid: {exc}"
            ) from exc
        if state.get("pose_hash") != expected_hash:
            raise SpearApartmentError(
                f"frame {frame_index} camera_state pose_hash does not bind "
                "world_from_rig"
            )
        habitat_position = _finite_triplet(
            state.get("habitat_position_m"),
            owner=f"frame {frame_index} camera_state.habitat_position_m",
        )
        world_position = _finite_triplet(
            world_from_rig.get("translation_m"),
            owner=(
                f"frame {frame_index} camera_state."
                "world_from_rig.translation_m"
            ),
        )
        if max(
            abs(habitat_position[axis] - world_position[axis])
            for axis in range(3)
        ) > 1.0e-9:
            raise SpearApartmentError(
                f"frame {frame_index} camera_state Habitat position is not "
                "derived from world_from_rig"
            )
        try:
            expected_habitat_yaw = habitat_yaw_degrees_from_xyzw(
                world_from_rig.get("rotation_xyzw")
            )
        except (TypeError, ValueError) as exc:
            raise SpearApartmentError(
                f"frame {frame_index} camera_state rotation is invalid"
            ) from exc
        habitat_yaw = state.get("habitat_yaw_deg")
        ue_yaw = state.get("ue_yaw_deg")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (habitat_yaw, ue_yaw)
        ):
            raise SpearApartmentError(
                f"frame {frame_index} camera_state yaw is not finite"
            )
        if abs(
            wrap_angle_difference_degrees(
                float(habitat_yaw), expected_habitat_yaw
            )
        ) > 1.0e-9:
            raise SpearApartmentError(
                f"frame {frame_index} camera_state Habitat yaw is not derived "
                "from world_from_rig"
            )
        ue_position = _finite_triplet(
            state.get("ue_position_cm"),
            owner=f"frame {frame_index} camera_state.ue_position_cm",
        )
        expected_ue_position = habitat_point_to_apartment_ue_cm(world_position)
        if max(
            abs(ue_position[axis] - expected_ue_position[axis])
            for axis in range(3)
        ) > 1.0e-7:
            raise SpearApartmentError(
                f"frame {frame_index} camera_state UE position is not derived "
                "from world_from_rig"
            )
        expected_ue_yaw = camera_ue_yaw_degrees(expected_habitat_yaw)
        if abs(
            wrap_angle_difference_degrees(float(ue_yaw), expected_ue_yaw)
        ) > 1.0e-9:
            raise SpearApartmentError(
                f"frame {frame_index} camera_state UE yaw is not derived from "
                "world_from_rig"
            )
        result.append(deepcopy(dict(state)))

    first = result[0]
    if (
        _finite_triplet(
            camera.get("ue_position_cm"), owner="camera default ue_position_cm"
        )
        != first["ue_position_cm"]
        or abs(
            wrap_angle_difference_degrees(
                camera.get("ue_yaw_deg"), first["ue_yaw_deg"]
            )
        )
        > 1.0e-9
    ):
        raise SpearApartmentError(
            "top-level camera default differs from frame-zero camera_state"
        )
    return tuple(result)


def _validate_native_plan(
    plan: Mapping[str, Any],
    *,
    scenario_id: str,
    room_runtime_profile: Mapping[str, Any] = DEFAULT_ROOM_RUNTIME_PROFILE,
) -> None:
    if plan.get("schema") != PLAN_SCHEMA or plan.get("backend_role") != BACKEND_ROLE:
        raise SpearApartmentError("input did not compile to a production-visual plan")
    authority = plan.get("authority")
    if (
        not isinstance(authority, Mapping)
        or authority.get("backend_may_replan") is not False
    ):
        raise SpearApartmentError("SPEAR backend must not replan authoritative state")
    if plan.get("source_logic", {}).get("scenario_id") != scenario_id:
        raise SpearApartmentError(
            f"scenario directory {scenario_id!r} disagrees with source manifest"
        )
    room_profile = _validated_room_runtime_profile(room_runtime_profile)
    provenance = plan.get("room", {}).get("source_scene_provenance")
    if not isinstance(provenance, Mapping) or (
        provenance.get("provider") != "SPEAR_Unreal"
        or provenance.get("scene_id") != room_profile["scene"]["scene_id"]
    ):
        raise SpearApartmentError(
            "RoomCapsule is not the native scene selected by the SPEAR room profile"
        )
    plan_room_id = plan.get("room", {}).get("room_id")
    if (
        plan_room_id is not None
        and plan_room_id != room_profile["room_ref"]["room_id"]
    ):
        raise SpearApartmentError(
            "RoomCapsule room_id does not match the selected room profile"
        )
    render = plan.get("render")
    if render != {
        "frame_count": FRAME_COUNT,
        "fps_num": FPS,
        "fps_den": 1,
        "ticks_per_frame": 3_200,
    }:
        raise SpearApartmentError("Timeline render clock changed")
    camera = plan.get("camera")
    if not isinstance(camera, Mapping):
        raise SpearApartmentError("compiled plan has no camera")
    if camera.get("horizontal_fov_deg") != 105.0:
        raise SpearApartmentError(
            "native Apartment production visual requires the frozen 105 degree HFOV"
        )
    if len(plan.get("frames", ())) != FRAME_COUNT:
        raise SpearApartmentError(
            "native Apartment production visual requires exactly 75 frames"
        )
    materialize_camera_states(plan)
    actor_ids = [actor.get("actor_id") for actor in plan.get("actors", ())]
    if actor_ids not in (
        ["dog0", "human0"],
        ["source1_actor", "source2_actor"],
    ):
        raise SpearApartmentError(
            "native Apartment actor closure must be the legacy pair or ordered source slots"
        )


def _build_native_apartment_scenario_from_paths(
    *,
    root: Path,
    scenario_id: str,
    scenario_directory: str,
    variant_id: str,
    paths: Mapping[str, Path],
    actor_bindings: Mapping[str, Mapping[str, Any]],
    lighting_profile: Mapping[str, Any],
    room_runtime_profile: Mapping[str, Any],
    acoustic_visual_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a resolved Apartment input closure into one UE execution record."""

    plan = build_spear_visual_plan_from_files(
        timeline_path=paths["timeline"],
        source_manifest_path=paths["source_manifest"],
        flags_path=paths["flags"],
        room_capsule_path=paths["room_capsule"],
        qualification_path=paths["qualification"],
        actor_bindings=actor_bindings,
        backend_role=BACKEND_ROLE,
        sensor_rig_trajectory_path=paths.get("sensor_rig_trajectory"),
    )
    room_profile = _validated_room_runtime_profile(room_runtime_profile)
    if acoustic_visual_identity is not None and (
        acoustic_visual_identity.get("runtime_room_ref")
        != room_profile["room_ref"]
        or acoustic_visual_identity.get("visual_room_ref")
        != room_profile["room_ref"]
        or acoustic_visual_identity.get("runtime_profile_id")
        != room_profile["profile_id"]
        or acoustic_visual_identity.get("runtime_map_id")
        != room_profile["scene"]["scene_id"]
        or acoustic_visual_identity.get("runtime_map_path")
        != room_profile["scene"]["map_path"]
    ):
        raise SpearApartmentError(
            "asset-bound acoustic/visual identity differs from the native scene"
        )
    _validate_native_plan(
        plan,
        scenario_id=scenario_id,
        room_runtime_profile=room_profile,
    )
    for actor in plan["actors"]:
        asset_id = actor["asset_id"]
        raw_binding = actor_bindings.get(asset_id)
        if not isinstance(raw_binding, Mapping):
            raise SpearApartmentError(f"actor binding {asset_id!r} is missing")
        actor["ue_component_frame_delta"] = component_frame_delta_for_asset(
            asset_id, actor_bindings=actor_bindings
        )
        actor["floor_contact_gate"] = bool(
            raw_binding.get("floor_contact_gate", False)
        )
        actor["skeletal_mesh_binding"] = raw_binding["skeletal_mesh_binding"]
        actor["skeletal_mesh_path"] = raw_binding.get("skeletal_mesh_path")
        actor["asset_revision"] = raw_binding.get("asset_revision")
        basis_bones = anatomical_basis_bones_for_asset(
            asset_id, actor_bindings=actor_bindings
        )
        if basis_bones is not None:
            actor["ue_anatomical_basis_bones"] = basis_bones
    if "batch_binding" in paths:
        attach_arguments: dict[str, Any] = {
            "plan": plan,
            "scenario_id": scenario_id,
            "batch_binding_path": paths["batch_binding"],
            "actor_bindings": actor_bindings,
        }
        if acoustic_visual_identity is not None:
            attach_arguments["expected_acoustic_selection_binding_sha256"] = (
                acoustic_visual_identity.get(
                    "acoustic_selection_binding_sha256"
                )
            )
        _attach_exact_asset_bound_runtime_bindings(
            **attach_arguments,
        )
    lighting = deepcopy(dict(lighting_profile))
    generated_lights = lighting.get("generated_lights")
    if not isinstance(generated_lights, list):
        raise SpearApartmentError("resolved lighting profile lacks generated_lights")
    scenario = {
        "schema": SCENARIO_SCHEMA,
        "scenario_id": scenario_id,
        "scenario_directory": scenario_directory,
        "variant_id": variant_id,
        "backend_role": BACKEND_ROLE,
        "native_scene": {
            "room_runtime_profile_id": room_profile["profile_id"],
            "room_ref": deepcopy(dict(room_profile["room_ref"])),
            "map": room_profile["scene"]["map_path"],
            "layout": room_profile["scene"]["layout_policy"],
            "lighting": (
                "native_map_unchanged_no_added_lights"
                if not generated_lights
                else "native_map_plus_generated_runtime_lights"
            ),
            "lighting_profile": lighting,
            "outdoor_view": room_profile["scene"]["outdoor_view_policy"],
        },
        "render": {
            "width": WIDTH,
            "height": HEIGHT,
            "frame_count": FRAME_COUNT,
            "frame_rate_hz": FPS,
            "horizontal_fov_deg": 105.0,
            "streaming_warmup_frames": STREAMING_WARMUP_FRAMES,
            "camera_warmup_frames": CAMERA_WARMUP_FRAMES,
        },
        "authoritative_inputs": {
            key: _relative(value, root) for key, value in paths.items()
        },
        "reuse_contract": {
            "actor_transforms_and_actions": "Timeline_v2",
            "source_logic_and_flags": "Habitat-native metadata unchanged",
            "source_center_gate": "Habitat-native qualification unchanged",
            "clean_audio": "copy authoritative binaural stream",
            "diagnostic_right_panel": "reuse authoritative Habitat Topdown",
            "audio_camera_fov_cutoff": False,
            "timeline_anchor_after_asset_frame_correction": "Timeline_v2 unchanged",
            "runtime_actor_hierarchy": (
                "hidden Timeline anchor with attached visual Blueprint child"
            ),
            "component_frame_correction": (
                "versioned relative delta on attached visual root"
            ),
        },
        "plan": plan,
    }
    if acoustic_visual_identity is not None:
        scenario["acoustic_visual_identity"] = deepcopy(
            dict(acoustic_visual_identity)
        )
    return scenario


def build_native_apartment_scenario(
    bundle_root: str | Path,
    scenario_id: str,
    *,
    actor_bindings: Mapping[str, Mapping[str, Any]] = DEFAULT_ACTOR_BINDINGS,
    lighting_profile: Mapping[str, Any] = NATIVE_LIGHTING_PROFILE,
    room_runtime_profile: Mapping[str, Any] = DEFAULT_ROOM_RUNTIME_PROFILE,
) -> dict[str, Any]:
    """Compile one existing M6.x scenario into a native UE execution record."""

    root = Path(bundle_root).resolve()
    paths = scenario_input_paths(root, scenario_id)
    scenario_directory, variant_id = SCENARIO_DIRECTORIES[scenario_id]
    return _build_native_apartment_scenario_from_paths(
        root=root,
        scenario_id=scenario_id,
        scenario_directory=scenario_directory,
        variant_id=variant_id,
        paths=paths,
        actor_bindings=actor_bindings,
        lighting_profile=lighting_profile,
        room_runtime_profile=room_runtime_profile,
    )


def build_native_apartment_motion_pilot_scenario(
    bundle_root: str | Path,
    scenario_id: str,
    *,
    actor_bindings: Mapping[str, Mapping[str, Any]] = DEFAULT_ACTOR_BINDINGS,
    lighting_profile: Mapping[str, Any] = NATIVE_LIGHTING_PROFILE,
    room_runtime_profile: Mapping[str, Any] = DEFAULT_ROOM_RUNTIME_PROFILE,
) -> dict[str, Any]:
    """Compile one existing four-motion pilot episode for native UE pixels."""

    root = Path(bundle_root).resolve()
    paths = motion_pilot_input_paths(root, scenario_id)
    episode_directory = MOTION_PILOT_DIRECTORIES[scenario_id]
    return _build_native_apartment_scenario_from_paths(
        root=root,
        scenario_id=scenario_id,
        scenario_directory=episode_directory,
        variant_id="A",
        paths=paths,
        actor_bindings=actor_bindings,
        lighting_profile=lighting_profile,
        room_runtime_profile=room_runtime_profile,
    )


def build_native_apartment_asset_bound_scenario(
    bundle_root: str | Path,
    episode_id: str,
    *,
    actor_bindings: Mapping[str, Mapping[str, Any]] = DEFAULT_ACTOR_BINDINGS,
    lighting_profile: Mapping[str, Any] = NATIVE_LIGHTING_PROFILE,
    room_runtime_profile: Mapping[str, Any] = DEFAULT_ROOM_RUNTIME_PROFILE,
) -> dict[str, Any]:
    """Compile one generic M7 source1/source2 episode for native UE pixels."""

    root = Path(bundle_root).resolve()
    acoustic_visual_identity = asset_bound_bundle_acoustic_visual_identity(
        root,
        room_runtime_profile=room_runtime_profile,
    )
    paths = asset_bound_episode_input_paths(root, episode_id)
    return _build_native_apartment_scenario_from_paths(
        root=root,
        scenario_id=episode_id,
        scenario_directory=episode_id,
        variant_id="A",
        paths=paths,
        actor_bindings=actor_bindings,
        lighting_profile=lighting_profile,
        room_runtime_profile=room_runtime_profile,
        acoustic_visual_identity=acoustic_visual_identity,
    )


def build_native_apartment_suite(
    bundle_root: str | Path,
    *,
    scenario_ids: Sequence[str] = ("S0", "S3", "S4"),
    actor_bindings: Mapping[str, Mapping[str, Any]] = DEFAULT_ACTOR_BINDINGS,
    lighting_profile: Mapping[str, Any] = NATIVE_LIGHTING_PROFILE,
    room_runtime_profile: Mapping[str, Any] = DEFAULT_ROOM_RUNTIME_PROFILE,
) -> dict[str, Any]:
    """Compile the requested native Apartment production-visual scenarios."""

    selected = tuple(scenario_ids)
    if not selected or len(selected) != len(set(selected)):
        raise SpearApartmentError("scenario selection must be nonempty and unique")
    scenarios = [
        build_native_apartment_scenario(
            bundle_root,
            scenario_id,
            actor_bindings=actor_bindings,
            lighting_profile=lighting_profile,
            room_runtime_profile=room_runtime_profile,
        )
        for scenario_id in selected
    ]
    return {
        "schema": SUITE_SCHEMA,
        "backend_role": BACKEND_ROLE,
        "room_runtime_profile": deepcopy(dict(room_runtime_profile)),
        "native_map": room_runtime_profile["scene"]["map_path"],
        "lighting_profile": deepcopy(dict(lighting_profile)),
        "authority": {
            "habitat_native": [
                "Timeline_v2",
                "source logic",
                "source-center qualification",
                "binaural audio",
                "Topdown",
                "flags and metadata",
            ],
            "spear_unreal": ["production RGB pixels"],
        },
        "scenarios": scenarios,
    }


def build_native_apartment_motion_pilot_suite(
    bundle_root: str | Path,
    *,
    scenario_ids: Sequence[str] = ("P0", "P1", "P2", "P3"),
    actor_bindings: Mapping[str, Mapping[str, Any]] = DEFAULT_ACTOR_BINDINGS,
    lighting_profile: Mapping[str, Any] = NATIVE_LIGHTING_PROFILE,
    room_runtime_profile: Mapping[str, Any] = DEFAULT_ROOM_RUNTIME_PROFILE,
) -> dict[str, Any]:
    """Compile the requested four-motion pilot episodes for one UE launch."""

    selected = tuple(scenario_ids)
    if not selected or len(selected) != len(set(selected)):
        raise SpearApartmentError("scenario selection must be nonempty and unique")
    scenarios = [
        build_native_apartment_motion_pilot_scenario(
            bundle_root,
            scenario_id,
            actor_bindings=actor_bindings,
            lighting_profile=lighting_profile,
            room_runtime_profile=room_runtime_profile,
        )
        for scenario_id in selected
    ]
    return {
        "schema": SUITE_SCHEMA,
        "backend_role": BACKEND_ROLE,
        "room_runtime_profile": deepcopy(dict(room_runtime_profile)),
        "native_map": room_runtime_profile["scene"]["map_path"],
        "lighting_profile": deepcopy(dict(lighting_profile)),
        "authority": {
            "habitat_native": [
                "Timeline_v2",
                "source logic",
                "source-center qualification",
                "binaural audio",
                "Topdown",
                "flags and metadata",
            ],
            "spear_unreal": ["production RGB pixels"],
        },
        "scenarios": scenarios,
    }


def build_native_apartment_asset_bound_suite(
    bundle_root: str | Path,
    *,
    scenario_ids: Sequence[str] | None = None,
    actor_bindings: Mapping[str, Mapping[str, Any]] = DEFAULT_ACTOR_BINDINGS,
    lighting_profile: Mapping[str, Any] = NATIVE_LIGHTING_PROFILE,
    room_runtime_profile: Mapping[str, Any] = DEFAULT_ROOM_RUNTIME_PROFILE,
) -> dict[str, Any]:
    """Compile selected M7 episodes, or the bundle manifest's full order."""

    root = Path(bundle_root).resolve()
    if scenario_ids is None:
        selected = asset_bound_bundle_episode_ids(root)
    else:
        selected = tuple(scenario_ids)
    if not selected or len(selected) != len(set(selected)):
        raise SpearApartmentError("scenario selection must be nonempty and unique")
    acoustic_visual_identity = asset_bound_bundle_acoustic_visual_identity(
        root,
        room_runtime_profile=room_runtime_profile,
    )
    scenarios = []
    for episode_id in selected:
        paths = asset_bound_episode_input_paths(root, episode_id)
        scenarios.append(
            _build_native_apartment_scenario_from_paths(
                root=root,
                scenario_id=episode_id,
                scenario_directory=episode_id,
                variant_id="A",
                paths=paths,
                actor_bindings=actor_bindings,
                lighting_profile=lighting_profile,
                room_runtime_profile=room_runtime_profile,
                acoustic_visual_identity=acoustic_visual_identity,
            )
        )
    return {
        "schema": SUITE_SCHEMA,
        "backend_role": BACKEND_ROLE,
        "room_runtime_profile": deepcopy(dict(room_runtime_profile)),
        "native_map": room_runtime_profile["scene"]["map_path"],
        "lighting_profile": deepcopy(dict(lighting_profile)),
        "acoustic_visual_identity": deepcopy(acoustic_visual_identity),
        "authority": {
            "habitat_native": [
                "Timeline_v2",
                "source1/source2 asset binding",
                "source-center qualification",
                "binaural audio",
                "Topdown",
                "flags and metadata",
            ],
            "spear_unreal": ["production RGB pixels"],
        },
        "scenarios": scenarios,
    }


def asset_bound_bundle_episode_ids(bundle_root: str | Path) -> tuple[str, ...]:
    """Read the complete, ordered episode declaration from an M7 visual bundle."""

    root = Path(bundle_root).resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpearApartmentError(
            f"could not read asset-bound bundle manifest: {manifest_path}"
        ) from exc
    raw_ids = manifest.get("episode_ids")
    if (
        manifest.get("status") != "pass"
        or not isinstance(raw_ids, list)
        or not raw_ids
        or any(not isinstance(value, str) or not value for value in raw_ids)
        or len(raw_ids) != len(set(raw_ids))
        or manifest.get("episode_count") != len(raw_ids)
    ):
        raise SpearApartmentError(
            "asset-bound bundle manifest has an invalid episode declaration"
        )
    return tuple(raw_ids)


def contiguous_episode_shard(
    episode_ids: Sequence[str],
    *,
    shard_count: int,
    shard_index: int,
) -> tuple[str, ...]:
    """Return one balanced, contiguous and non-overlapping episode partition."""

    values = tuple(episode_ids)
    if not values or len(values) != len(set(values)):
        raise SpearApartmentError("episode IDs must be nonempty and unique")
    if (
        isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or not 1 <= shard_count <= len(values)
    ):
        raise SpearApartmentError(
            "shard_count must be between one and the episode count"
        )
    if (
        isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < shard_count
    ):
        raise SpearApartmentError("shard_index must be in [0, shard_count)")
    base_size, remainder = divmod(len(values), shard_count)
    start = shard_index * base_size + min(shard_index, remainder)
    size = base_size + int(shard_index < remainder)
    return values[start : start + size]


def animation_position_seconds(
    action_phase: float, animation_length_seconds: float
) -> float:
    """Convert Timeline's normalized phase to a stopped UE animation position."""

    phase = float(action_phase)
    length = float(animation_length_seconds)
    if not math.isfinite(phase) or not 0.0 <= phase < 1.0:
        raise SpearApartmentError("Timeline action phase must be in [0,1)")
    if not math.isfinite(length) or length <= 0.0:
        raise SpearApartmentError("UE animation length must be positive and finite")
    return phase * length


def wrap_angle_difference_degrees(observed: float, expected: float) -> float:
    return (float(observed) - float(expected) + 180.0) % 360.0 - 180.0


def summarize_root_readbacks(
    *,
    expected_frames: Sequence[Mapping[str, Any]],
    actor_readbacks: Mapping[str, Sequence[Mapping[str, Any]]],
    camera_readbacks: Sequence[Mapping[str, Any]],
    camera_position_cm: Sequence[float] | None = None,
    camera_yaw_deg: float | None = None,
) -> dict[str, Any]:
    """Fail closed on UE actor/camera root drift without inventing new QA."""

    if len(expected_frames) != FRAME_COUNT or len(camera_readbacks) != FRAME_COUNT:
        raise SpearApartmentError("root readback requires exactly 75 frames")
    expected_actor_ids = [
        state["actor_id"] for state in expected_frames[0]["actor_states"]
    ]
    if set(actor_readbacks) != set(expected_actor_ids):
        raise SpearApartmentError("actor readback closure differs from Timeline")

    summaries: dict[str, Any] = {}
    for actor_id in expected_actor_ids:
        records = actor_readbacks[actor_id]
        if len(records) != FRAME_COUNT:
            raise SpearApartmentError(f"{actor_id} root readback lacks 75 frames")
        position_errors = []
        yaw_errors = []
        for frame_index, (frame, record) in enumerate(zip(expected_frames, records)):
            expected = next(
                item for item in frame["actor_states"] if item["actor_id"] == actor_id
            )
            location = record["location_cm"]
            rotation = record["rotation_deg"]
            position_errors.append(
                max(
                    abs(float(location[axis]) - expected["translation_ue_cm"][axis])
                    for axis in range(3)
                )
            )
            yaw_errors.append(
                abs(
                    wrap_angle_difference_degrees(
                        rotation[2], expected["actor_yaw_ue_deg"]
                    )
                )
            )
            if record.get("frame_index") != frame_index:
                raise SpearApartmentError(f"{actor_id} readback frame order changed")
        maximum_position = max(position_errors)
        maximum_yaw = max(yaw_errors)
        if (
            maximum_position > POSITION_TOLERANCE_CM
            or maximum_yaw > ROTATION_TOLERANCE_DEGREES
        ):
            raise SpearApartmentError(f"{actor_id} UE root readback drifted")
        summaries[actor_id] = {
            "status": "pass",
            "maximum_position_error_cm": maximum_position,
            "maximum_yaw_error_deg": maximum_yaw,
        }

    declared_camera_states = [
        frame.get("camera_state") if isinstance(frame, Mapping) else None
        for frame in expected_frames
    ]
    if any(state is not None for state in declared_camera_states) and not all(
        isinstance(state, Mapping) for state in declared_camera_states
    ):
        raise SpearApartmentError(
            "expected frames only partially provide camera_state"
        )
    if all(isinstance(state, Mapping) for state in declared_camera_states):
        expected_camera_states = declared_camera_states
        per_frame_camera_state = True
    else:
        if camera_position_cm is None or camera_yaw_deg is None:
            raise SpearApartmentError(
                "legacy root readback requires the fixed camera pose"
            )
        fixed_position = _finite_triplet(
            camera_position_cm, owner="legacy expected camera position"
        )
        if (
            isinstance(camera_yaw_deg, bool)
            or not isinstance(camera_yaw_deg, (int, float))
            or not math.isfinite(float(camera_yaw_deg))
        ):
            raise SpearApartmentError("legacy expected camera yaw must be finite")
        expected_camera_states = [
            {
                "frame_index": frame_index,
                "ue_position_cm": fixed_position,
                "ue_yaw_deg": float(camera_yaw_deg),
            }
            for frame_index in range(FRAME_COUNT)
        ]
        per_frame_camera_state = False

    camera_position_errors = []
    camera_yaw_errors = []
    checked_pose_hash_count = 0
    for frame_index, (record, expected_camera) in enumerate(
        zip(camera_readbacks, expected_camera_states)
    ):
        assert isinstance(expected_camera, Mapping)
        if record.get("frame_index") != frame_index:
            raise SpearApartmentError("camera readback frame order changed")
        if expected_camera.get("frame_index") != frame_index:
            raise SpearApartmentError("expected camera state frame order changed")
        expected_pose_hash = expected_camera.get("pose_hash")
        if expected_pose_hash is not None:
            if record.get("expected_pose_hash") != expected_pose_hash:
                raise SpearApartmentError(
                    "UE camera readback pose hash differs from "
                    f"frame {frame_index} camera_state"
                )
            checked_pose_hash_count += 1
        expected_position = _finite_triplet(
            expected_camera.get("ue_position_cm"),
            owner=f"expected camera state {frame_index} position",
        )
        expected_yaw = expected_camera.get("ue_yaw_deg")
        if (
            isinstance(expected_yaw, bool)
            or not isinstance(expected_yaw, (int, float))
            or not math.isfinite(float(expected_yaw))
        ):
            raise SpearApartmentError(
                f"expected camera state {frame_index} yaw must be finite"
            )
        camera_position_errors.append(
            max(
                abs(
                    float(record["location_cm"][axis])
                    - float(expected_position[axis])
                )
                for axis in range(3)
            )
        )
        camera_yaw_errors.append(
            abs(
                wrap_angle_difference_degrees(
                    record["rotation_deg"][2], float(expected_yaw)
                )
            )
        )
    maximum_camera_position = max(camera_position_errors)
    maximum_camera_yaw = max(camera_yaw_errors)
    if (
        maximum_camera_position > POSITION_TOLERANCE_CM
        or maximum_camera_yaw > ROTATION_TOLERANCE_DEGREES
    ):
        raise SpearApartmentError("UE camera root readback drifted")
    summaries["camera"] = {
        "status": "pass",
        "maximum_position_error_cm": maximum_camera_position,
        "maximum_yaw_error_deg": maximum_camera_yaw,
        "per_frame_camera_state": per_frame_camera_state,
        "checked_pose_hash_count": checked_pose_hash_count,
        "unique_expected_pose_hash_count": len(
            {
                state.get("pose_hash")
                for state in expected_camera_states
                if isinstance(state, Mapping) and state.get("pose_hash") is not None
            }
        ),
    }
    return summaries


def summarize_actor_bounds(
    *,
    expected_frames: Sequence[Mapping[str, Any]],
    actor_declarations: Sequence[Mapping[str, Any]],
    actor_bounds: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Summarize visual bounds and fail closed on calibrated quadruped frames.

    Bounds are measured in UE world centimetres after animation evaluation.
    A calibrated quadruped actor root is the authoritative floor anchor.  This
    gate proves that its asset-local correction does not move that root while
    the rendered mesh remains in floor contact. Bounds aspect ratio is retained
    only as a descriptive observation: it is not rotation-invariant and a
    compact cat may legitimately be taller than its horizontal AABB span.
    """

    if not expected_frames:
        raise SpearApartmentError("visual bounds require at least one frame")
    declarations = {item["actor_id"]: item for item in actor_declarations}
    if set(actor_bounds) != set(declarations):
        raise SpearApartmentError("visual bounds actor closure differs from plan")
    summaries: dict[str, Any] = {}
    for actor_id, declaration in declarations.items():
        records = actor_bounds[actor_id]
        if len(records) != len(expected_frames):
            raise SpearApartmentError(
                f"{actor_id} visual bounds count differs from expected frames"
            )
        clearances: list[float] = []
        horizontal_vertical_ratios: list[float] = []
        spans_by_axis: list[list[float]] = []
        for frame, record in zip(expected_frames, records):
            frame_index = frame.get("frame_index")
            if record.get("frame_index") != frame_index:
                raise SpearApartmentError(f"{actor_id} bounds frame order changed")
            state = next(
                value
                for value in frame["actor_states"]
                if value["actor_id"] == actor_id
            )
            minimum = _finite_triplet(
                record.get("minimum_cm"), owner=f"{actor_id} bounds minimum"
            )
            maximum = _finite_triplet(
                record.get("maximum_cm"), owner=f"{actor_id} bounds maximum"
            )
            spans = [maximum[axis] - minimum[axis] for axis in range(3)]
            if any(value <= 0.0 for value in spans):
                raise SpearApartmentError(f"{actor_id} has degenerate visual bounds")
            clearance = minimum[2] - float(state["translation_ue_cm"][2])
            clearances.append(clearance)
            spans_by_axis.append(spans)
            horizontal_vertical_ratios.append(max(spans[:2]) / spans[2])

        summary = {
            "status": "observed",
            "minimum_floor_clearance_from_actor_root_cm": min(clearances),
            "maximum_floor_clearance_from_actor_root_cm": max(clearances),
            "minimum_horizontal_to_vertical_span_ratio": min(
                horizontal_vertical_ratios
            ),
            "span_cm": {
                "minimum_by_axis": [
                    min(value[axis] for value in spans_by_axis) for axis in range(3)
                ],
                "maximum_by_axis": [
                    max(value[axis] for value in spans_by_axis) for axis in range(3)
                ],
            },
        }
        floor_contact_gate = declaration.get("floor_contact_gate")
        if floor_contact_gate is None:
            default_binding = DEFAULT_ACTOR_BINDINGS.get(
                declaration.get("asset_id")
            )
            floor_contact_gate = (
                default_binding.get("floor_contact_gate")
                if isinstance(default_binding, Mapping)
                else False
            )
        if floor_contact_gate is True:
            maximum_floor_error = max(abs(value) for value in clearances)
            if maximum_floor_error > QUADRUPED_FLOOR_TOLERANCE_CM:
                raise SpearApartmentError(
                    f"{actor_id} corrected mesh no longer meets its actor-root floor"
                )
            summary.update(
                {
                    "status": "pass",
                    "maximum_floor_error_cm": maximum_floor_error,
                    "floor_tolerance_cm": QUADRUPED_FLOOR_TOLERANCE_CM,
                    "horizontal_to_vertical_span_ratio_role": (
                        "descriptive_only_not_an_orientation_gate"
                    ),
                }
            )
        summaries[actor_id] = summary
    return summaries


def summarize_anatomical_forward_readbacks(
    *,
    expected_frames: Sequence[Mapping[str, Any]],
    visual_forward_readbacks: Mapping[str, Sequence[Mapping[str, Any]]],
    tolerance_degrees: float = ANATOMICAL_FORWARD_TOLERANCE_DEGREES,
) -> dict[str, Any]:
    """Fail when a rendered skeleton faces away from its declared forward.

    Root-yaw readback alone can only prove that UE accepted the requested
    actor transform.  It cannot prove that an imported mesh's head is on the
    declared side of that local frame.  The optional runtime therefore samples
    a semantic skeletal basis and compares its world-space forward vector with
    the authoritative per-frame anatomical forward from Timeline v2.
    """

    if not expected_frames:
        raise SpearApartmentError("anatomical-forward readback needs frames")
    if (
        isinstance(tolerance_degrees, bool)
        or not isinstance(tolerance_degrees, (int, float))
        or not math.isfinite(float(tolerance_degrees))
        or not 0.0 < float(tolerance_degrees) < 90.0
    ):
        raise SpearApartmentError(
            "anatomical-forward tolerance must be finite and between 0 and 90"
        )
    expected_actor_ids = {
        state["actor_id"] for state in expected_frames[0]["actor_states"]
    }
    if set(visual_forward_readbacks) != expected_actor_ids:
        raise SpearApartmentError(
            "anatomical-forward actor closure differs from Timeline"
        )

    summaries: dict[str, Any] = {}
    for actor_id in sorted(expected_actor_ids):
        records = visual_forward_readbacks[actor_id]
        if not records:
            raise SpearApartmentError(
                f"{actor_id} has no anatomical-forward readback"
            )
        angular_errors = []
        vertical_fractions = []
        sampled_frames = []
        basis_kinds = set()
        bone_name_sets = []
        for record in records:
            frame_index = record.get("frame_index")
            if (
                isinstance(frame_index, bool)
                or not isinstance(frame_index, int)
                or not 0 <= frame_index < len(expected_frames)
                or frame_index in sampled_frames
            ):
                raise SpearApartmentError(
                    f"{actor_id} anatomical-forward frame indices are invalid"
                )
            state = next(
                value
                for value in expected_frames[frame_index]["actor_states"]
                if value["actor_id"] == actor_id
            )
            expected = _finite_triplet(
                state.get("anatomical_forward_ue_world"),
                owner=f"{actor_id} expected anatomical forward",
            )
            observed = _finite_triplet(
                record.get("forward_vector_ue"),
                owner=f"{actor_id} observed anatomical forward",
            )
            observed_norm = math.sqrt(sum(value * value for value in observed))
            if observed_norm < 1.0e-6:
                raise SpearApartmentError(
                    f"{actor_id} anatomical forward has zero length"
                )
            expected_yaw = math.degrees(math.atan2(expected[1], expected[0]))
            observed_yaw = math.degrees(math.atan2(observed[1], observed[0]))
            if math.hypot(expected[0], expected[1]) < 1.0e-6 or math.hypot(
                observed[0], observed[1]
            ) < 1.0e-6:
                raise SpearApartmentError(
                    f"{actor_id} anatomical forward lacks a horizontal direction"
                )
            vertical_fraction = abs(observed[2]) / observed_norm
            maximum_vertical_fraction = math.sin(
                math.radians(float(tolerance_degrees))
            )
            if vertical_fraction > maximum_vertical_fraction:
                raise SpearApartmentError(
                    f"{actor_id} anatomical forward is not horizontal: "
                    f"vertical fraction {vertical_fraction:.6f}"
                )
            vertical_fractions.append(vertical_fraction)
            angular_errors.append(
                abs(wrap_angle_difference_degrees(observed_yaw, expected_yaw))
            )
            sampled_frames.append(frame_index)
            basis_kind = record.get("basis_kind")
            if not isinstance(basis_kind, str) or not basis_kind:
                raise SpearApartmentError(
                    f"{actor_id} anatomical-forward basis kind is missing"
                )
            basis_kinds.add(basis_kind)
            bone_names = record.get("bone_names")
            if not isinstance(bone_names, Mapping) or not bone_names:
                raise SpearApartmentError(
                    f"{actor_id} anatomical-forward bone names are missing"
                )
            bone_name_sets.append(dict(bone_names))

        maximum_error = max(angular_errors)
        if maximum_error > float(tolerance_degrees):
            raise SpearApartmentError(
                f"{actor_id} rendered skeleton faces away from Timeline motion: "
                f"maximum error {maximum_error:.3f} degrees"
            )
        summaries[actor_id] = {
            "status": "pass",
            "sampled_frame_indices": sampled_frames,
            "maximum_angular_error_deg": maximum_error,
            "maximum_abs_vertical_fraction": max(vertical_fractions),
            "maximum_abs_vertical_fraction_allowed": math.sin(
                math.radians(float(tolerance_degrees))
            ),
            "tolerance_deg": float(tolerance_degrees),
            "basis_kinds": sorted(basis_kinds),
            "bone_names": bone_name_sets,
        }
    return summaries


def _h264_encoder_arguments(
    video_encoder: str, *, encoder_gpu: int | None
) -> list[str]:
    if video_encoder == "libx264":
        return ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    if video_encoder == "h264_nvenc":
        if encoder_gpu is not None and (
            isinstance(encoder_gpu, bool)
            or not isinstance(encoder_gpu, int)
            or encoder_gpu < 0
        ):
            raise SpearApartmentError("NVENC GPU index must be non-negative")
        arguments = [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p5",
            "-cq",
            "18",
            "-b:v",
            "0",
        ]
        if encoder_gpu is not None:
            arguments.extend(("-gpu", str(encoder_gpu)))
        return arguments
    raise SpearApartmentError(f"unsupported Apartment H.264 encoder: {video_encoder!r}")


def build_png_encode_command(
    *,
    frames_pattern: str | Path,
    output_path: str | Path,
    video_encoder: str = "libx264",
    encoder_gpu: int | None = None,
) -> list[str]:
    return [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(FPS),
        "-i",
        str(frames_pattern),
        "-frames:v",
        str(FRAME_COUNT),
        "-an",
        *_h264_encoder_arguments(video_encoder, encoder_gpu=encoder_gpu),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def build_rawvideo_encode_command(
    *,
    output_path: str | Path,
    video_encoder: str = "libx264",
    encoder_gpu: int | None = None,
    width: int = WIDTH,
    height: int = HEIGHT,
    frame_rate_hz: int = FPS,
    frame_count: int = FRAME_COUNT,
    pixel_format: str = "bgr24",
) -> list[str]:
    """Encode captured UE frames directly from stdin without a PNG round trip."""

    for name, value in (
        ("width", width),
        ("height", height),
        ("frame_rate_hz", frame_rate_hz),
        ("frame_count", frame_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise SpearApartmentError(f"rawvideo {name} must be a positive integer")
    if pixel_format not in {"bgr24", "rgb24"}:
        raise SpearApartmentError(
            "rawvideo pixel_format must be bgr24 or rgb24"
        )
    return [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        pixel_format,
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(frame_rate_hz),
        "-i",
        "pipe:0",
        "-frames:v",
        str(frame_count),
        "-an",
        *_h264_encoder_arguments(video_encoder, encoder_gpu=encoder_gpu),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def build_clean_binaural_mux_command(
    *,
    ue_video_path: str | Path,
    authoritative_clean_path: str | Path,
    output_path: str | Path,
) -> list[str]:
    """Mux the unchanged Habitat-native binaural packets with UE pixels."""

    return [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(ue_video_path),
        "-i",
        str(authoritative_clean_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def build_topdown_visual_command(
    *,
    ue_video_path: str | Path,
    authoritative_diagnostic_path: str | Path,
    output_path: str | Path,
    video_encoder: str = "libx264",
    encoder_gpu: int | None = None,
) -> list[str]:
    """Pair UE main pixels with only the authoritative Topdown visual panel."""

    filter_graph = (
        "[0:v]scale=640:360:flags=lanczos,"
        "pad=640:480:0:60:color=black[ue];"
        "[1:v]crop=640:480:iw-640:0[topdown];"
        "[ue][topdown]hstack=inputs=2[video]"
    )
    return [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(ue_video_path),
        "-i",
        str(authoritative_diagnostic_path),
        "-filter_complex",
        filter_graph,
        "-map",
        "[video]",
        "-frames:v",
        str(FRAME_COUNT),
        *_h264_encoder_arguments(video_encoder, encoder_gpu=encoder_gpu),
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def detached_suite_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy for tools that append runtime evidence."""

    return deepcopy(dict(value))


__all__ = [
    "ACOUSTIC_VISUAL_IDENTITY_SCHEMA",
    "ANATOMICAL_FORWARD_TOLERANCE_DEGREES",
    "ANIMATION_TOLERANCE_SECONDS",
    "BACKEND_ROLE",
    "CAMERA_WARMUP_FRAMES",
    "BORDER_COLLIE_ASSET_ID",
    "QUADRUPED_FLOOR_TOLERANCE_CM",
    "COMPONENT_FRAME_DELTA_SCHEMA",
    "COMPONENT_TRANSFORM_TOLERANCE",
    "DEFAULT_ACTOR_BINDINGS",
    "DURATION_SECONDS",
    "FPS",
    "FRAME_COUNT",
    "HEIGHT",
    "LIGHTING_PROFILE_SCHEMA",
    "NATIVE_APARTMENT_MAP",
    "NATIVE_LIGHTING_PROFILE",
    "MOTION_PILOT_DIRECTORIES",
    "SCENARIO_DIRECTORIES",
    "SCENARIO_SCHEMA",
    "STREAMING_WARMUP_FRAMES",
    "SUITE_SCHEMA",
    "SpearApartmentError",
    "WIDTH",
    "anatomical_basis_bones_for_asset",
    "animation_position_seconds",
    "apply_ue_component_frame_delta",
    "asset_bound_bundle_acoustic_visual_identity",
    "asset_bound_bundle_episode_ids",
    "build_clean_binaural_mux_command",
    "build_native_apartment_scenario",
    "build_native_apartment_suite",
    "build_native_apartment_motion_pilot_scenario",
    "build_native_apartment_motion_pilot_suite",
    "build_native_apartment_asset_bound_scenario",
    "build_native_apartment_asset_bound_suite",
    "build_png_encode_command",
    "build_rawvideo_encode_command",
    "build_topdown_visual_command",
    "component_frame_delta_for_asset",
    "detached_suite_copy",
    "load_apartment_lighting_profile",
    "materialize_camera_states",
    "resolve_apartment_lighting_profile",
    "scenario_input_paths",
    "motion_pilot_input_paths",
    "read_ue_component_relative_transform",
    "summarize_root_readbacks",
    "summarize_actor_bounds",
    "summarize_anatomical_forward_readbacks",
    "wrap_angle_difference_degrees",
]
