"""Native SPEAR Apartment comparison-visual execution contracts.

The Habitat-native bundle remains authoritative for Timeline actor state,
source logic, source-center qualification, flags, binaural audio, and the QA
Topdown.  This module only binds those records to the already-cooked native
SPEAR Apartment map and its UE visual assets.

Everything in this module is pure Python.  Importing it never imports SPEAR,
starts Unreal, or mutates either repository.  The actual optional runtime is
the small script in ``tools/m6y/run_spear_apartment_canary.py``.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from avengine.optional_backends.spear_visual import (
    BACKEND_ROLE,
    FRAME_COUNT,
    PLAN_SCHEMA,
    build_spear_visual_plan_from_files,
)


SUITE_SCHEMA = "avengine_optional_spear_apartment_suite_v1"
SCENARIO_SCHEMA = "avengine_optional_spear_apartment_scenario_v1"
NATIVE_APARTMENT_MAP = "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000"
NATIVE_APARTMENT_SCENE_ID = "apartment_0000"
WIDTH = 1280
HEIGHT = 720
FPS = 15
STREAMING_WARMUP_FRAMES = 120
CAMERA_WARMUP_FRAMES = 40
DURATION_SECONDS = FRAME_COUNT / FPS
POSITION_TOLERANCE_CM = 0.02
ROTATION_TOLERANCE_DEGREES = 0.02
ANIMATION_TOLERANCE_SECONDS = 1.0e-4
COMPONENT_TRANSFORM_TOLERANCE = 1.0e-3
BEAGLE_FLOOR_TOLERANCE_CM = 1.25
BEAGLE_HORIZONTAL_TO_VERTICAL_MIN_RATIO = 1.05
ANATOMICAL_FORWARD_TOLERANCE_DEGREES = 25.0
COMPONENT_FRAME_DELTA_SCHEMA = "avengine_spear_component_frame_delta_v1"
LIGHTING_PROFILE_SCHEMA = "avengine_optional_spear_apartment_lighting_profiles_v1"


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


BEAGLE_ASSET_ID = "rocketbox_dog_beagle_01_m2_v7_world_contact_candidate"
HUMAN_ASSET_ID = "rocketbox_human_male_adult_01_m5_1_candidate"

BEAGLE_TAG = "m2_beagle_v7_world_contact_r5"
HUMAN_TAG = "rocketbox_male_adult_01_original_ue_v3"

DEFAULT_ACTOR_BINDINGS: Mapping[str, Mapping[str, Any]] = {
    BEAGLE_ASSET_ID: {
        "blueprint_class_path": (
            "/Game/MyAssets/Audioset/Blueprints/"
            f"gate_{BEAGLE_TAG}/BP_gate_{BEAGLE_TAG}.BP_gate_{BEAGLE_TAG}_C"
        ),
        "idle_animation": (
            f"/Game/MyAssets/Audioset/Meshes/gate_{BEAGLE_TAG}/Idle.Idle"
        ),
        "walking_animation": (
            f"/Game/MyAssets/Audioset/Meshes/gate_{BEAGLE_TAG}/Walking.Walking"
        ),
        # Runtime bone-basis readback shows that this imported Blueprint's
        # nose points along UE actor-local -X after its asset-frame correction.
        # This is independent of the source GLB's Habitat-local +X forward.
        "ue_anatomical_forward_yaw_deg": 180.0,
        # The exact M2 GLB imports with Z-up interpreted as a standing local
        # mesh frame in UE.  This is an asset-local visual correction only:
        # it is added to (never replaces) the Blueprint component transform,
        # while the authoritative Timeline actor root remains untouched.
        "ue_component_frame_delta": {
            "schema": COMPONENT_FRAME_DELTA_SCHEMA,
            "rotation_deg": [0.0, 90.0, 0.0],
            "translation_cm": [0.0, 0.0, 33.64],
            "composition": "add_relative_preserving_blueprint_transform",
            "reason": "exact_M2_GLTF_to_UE_asset_local_axis_and_floor_calibration",
        },
    },
    HUMAN_ASSET_ID: {
        "blueprint_class_path": (
            "/Game/MyAssets/Audioset/Blueprints/"
            f"gate_{HUMAN_TAG}/BP_gate_{HUMAN_TAG}.BP_gate_{HUMAN_TAG}_C"
        ),
        "idle_animation": (
            "/Game/MyAssets/Audioset/Meshes/"
            f"gate_{HUMAN_TAG}/Standing_Idle.Standing_Idle"
        ),
        "walking_animation": (
            f"/Game/MyAssets/Audioset/Meshes/gate_{HUMAN_TAG}/Walking.Walking"
        ),
        # Rocketbox walking/idle clips face UE actor-local +Y.
        "ue_anatomical_forward_yaw_deg": 90.0,
        "ue_component_frame_delta": {
            "schema": COMPONENT_FRAME_DELTA_SCHEMA,
            "rotation_deg": [0.0, 0.0, 0.0],
            "translation_cm": [0.0, 0.0, 0.0],
            "composition": "add_relative_preserving_blueprint_transform",
            "reason": "identity_delta_native_Rocketbox_UE_asset_frame",
        },
    },
}


class SpearApartmentError(ValueError):
    """The native Apartment comparison cannot preserve its authority boundary."""


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


def _validate_native_plan(plan: Mapping[str, Any], *, scenario_id: str) -> None:
    if plan.get("schema") != PLAN_SCHEMA or plan.get("backend_role") != BACKEND_ROLE:
        raise SpearApartmentError("input did not compile to a comparison-visual plan")
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
    provenance = plan.get("room", {}).get("source_scene_provenance")
    if not isinstance(provenance, Mapping) or (
        provenance.get("provider") != "SPEAR_Unreal"
        or provenance.get("scene_id") != NATIVE_APARTMENT_SCENE_ID
    ):
        raise SpearApartmentError("RoomCapsule is not the native SPEAR Apartment")
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
            "native comparison requires the frozen 105 degree HFOV"
        )
    if len(plan.get("frames", ())) != FRAME_COUNT:
        raise SpearApartmentError("native comparison requires exactly 75 frames")
    actor_ids = [actor.get("actor_id") for actor in plan.get("actors", ())]
    if actor_ids != ["dog0", "human0"]:
        raise SpearApartmentError("native Apartment human/Beagle actor closure changed")


def _build_native_apartment_scenario_from_paths(
    *,
    root: Path,
    scenario_id: str,
    scenario_directory: str,
    variant_id: str,
    paths: Mapping[str, Path],
    actor_bindings: Mapping[str, Mapping[str, Any]],
    lighting_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile a resolved Apartment input closure into one UE execution record."""

    plan = build_spear_visual_plan_from_files(
        timeline_path=paths["timeline"],
        source_manifest_path=paths["source_manifest"],
        flags_path=paths["flags"],
        room_capsule_path=paths["room_capsule"],
        qualification_path=paths["qualification"],
        actor_bindings=actor_bindings,
    )
    _validate_native_plan(plan, scenario_id=scenario_id)
    for actor in plan["actors"]:
        asset_id = actor["asset_id"]
        actor["ue_component_frame_delta"] = component_frame_delta_for_asset(
            asset_id, actor_bindings=actor_bindings
        )
    lighting = deepcopy(dict(lighting_profile))
    generated_lights = lighting.get("generated_lights")
    if not isinstance(generated_lights, list):
        raise SpearApartmentError("resolved lighting profile lacks generated_lights")
    return {
        "schema": SCENARIO_SCHEMA,
        "scenario_id": scenario_id,
        "scenario_directory": scenario_directory,
        "variant_id": variant_id,
        "backend_role": BACKEND_ROLE,
        "native_scene": {
            "map": NATIVE_APARTMENT_MAP,
            "layout": "native_map_unchanged",
            "lighting": (
                "native_map_unchanged_no_added_lights"
                if not generated_lights
                else "native_map_plus_generated_runtime_lights"
            ),
            "lighting_profile": lighting,
            "outdoor_view": "native_map_assets_and_postprocess",
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


def build_native_apartment_scenario(
    bundle_root: str | Path,
    scenario_id: str,
    *,
    actor_bindings: Mapping[str, Mapping[str, Any]] = DEFAULT_ACTOR_BINDINGS,
    lighting_profile: Mapping[str, Any] = NATIVE_LIGHTING_PROFILE,
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
    )


def build_native_apartment_motion_pilot_scenario(
    bundle_root: str | Path,
    scenario_id: str,
    *,
    actor_bindings: Mapping[str, Mapping[str, Any]] = DEFAULT_ACTOR_BINDINGS,
    lighting_profile: Mapping[str, Any] = NATIVE_LIGHTING_PROFILE,
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
    )


def build_native_apartment_suite(
    bundle_root: str | Path,
    *,
    scenario_ids: Sequence[str] = ("S0", "S3", "S4"),
    actor_bindings: Mapping[str, Mapping[str, Any]] = DEFAULT_ACTOR_BINDINGS,
    lighting_profile: Mapping[str, Any] = NATIVE_LIGHTING_PROFILE,
) -> dict[str, Any]:
    """Compile the requested native Apartment comparison scenarios."""

    selected = tuple(scenario_ids)
    if not selected or len(selected) != len(set(selected)):
        raise SpearApartmentError("scenario selection must be nonempty and unique")
    scenarios = [
        build_native_apartment_scenario(
            bundle_root,
            scenario_id,
            actor_bindings=actor_bindings,
            lighting_profile=lighting_profile,
        )
        for scenario_id in selected
    ]
    return {
        "schema": SUITE_SCHEMA,
        "backend_role": BACKEND_ROLE,
        "native_map": NATIVE_APARTMENT_MAP,
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
            "spear_unreal": ["comparison visual pixels only"],
        },
        "scenarios": scenarios,
    }


def build_native_apartment_motion_pilot_suite(
    bundle_root: str | Path,
    *,
    scenario_ids: Sequence[str] = ("P0", "P1", "P2", "P3"),
    actor_bindings: Mapping[str, Mapping[str, Any]] = DEFAULT_ACTOR_BINDINGS,
    lighting_profile: Mapping[str, Any] = NATIVE_LIGHTING_PROFILE,
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
        )
        for scenario_id in selected
    ]
    return {
        "schema": SUITE_SCHEMA,
        "backend_role": BACKEND_ROLE,
        "native_map": NATIVE_APARTMENT_MAP,
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
            "spear_unreal": ["final RGB pixels"],
        },
        "scenarios": scenarios,
    }


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
    camera_position_cm: Sequence[float],
    camera_yaw_deg: float,
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

    camera_position_errors = []
    camera_yaw_errors = []
    for frame_index, record in enumerate(camera_readbacks):
        if record.get("frame_index") != frame_index:
            raise SpearApartmentError("camera readback frame order changed")
        camera_position_errors.append(
            max(
                abs(
                    float(record["location_cm"][axis]) - float(camera_position_cm[axis])
                )
                for axis in range(3)
            )
        )
        camera_yaw_errors.append(
            abs(
                wrap_angle_difference_degrees(record["rotation_deg"][2], camera_yaw_deg)
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
    }
    return summaries


def summarize_actor_bounds(
    *,
    expected_frames: Sequence[Mapping[str, Any]],
    actor_declarations: Sequence[Mapping[str, Any]],
    actor_bounds: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Summarize visual bounds and fail closed on the corrected Beagle frame.

    Bounds are measured in UE world centimetres after animation evaluation.
    The Beagle's actor root is intentionally the authoritative floor anchor;
    this gate proves that the asset-local correction does not move that root
    while the rendered mesh remains horizontal and in floor contact.
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
        if declaration.get("asset_id") == BEAGLE_ASSET_ID:
            maximum_floor_error = max(abs(value) for value in clearances)
            minimum_ratio = min(horizontal_vertical_ratios)
            if maximum_floor_error > BEAGLE_FLOOR_TOLERANCE_CM:
                raise SpearApartmentError(
                    f"{actor_id} corrected mesh no longer meets its actor-root floor"
                )
            if minimum_ratio < BEAGLE_HORIZONTAL_TO_VERTICAL_MIN_RATIO:
                raise SpearApartmentError(
                    f"{actor_id} corrected quadruped frame is not horizontal"
                )
            summary.update(
                {
                    "status": "pass",
                    "maximum_floor_error_cm": maximum_floor_error,
                    "floor_tolerance_cm": BEAGLE_FLOOR_TOLERANCE_CM,
                    "horizontal_to_vertical_minimum_required": (
                        BEAGLE_HORIZONTAL_TO_VERTICAL_MIN_RATIO
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
            expected_yaw = math.degrees(math.atan2(expected[1], expected[0]))
            observed_yaw = math.degrees(math.atan2(observed[1], observed[0]))
            if math.hypot(expected[0], expected[1]) < 1.0e-6 or math.hypot(
                observed[0], observed[1]
            ) < 1.0e-6:
                raise SpearApartmentError(
                    f"{actor_id} anatomical forward lacks a horizontal direction"
                )
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
        "[1:v]crop=640:480:640:0[topdown];"
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
    "ANATOMICAL_FORWARD_TOLERANCE_DEGREES",
    "ANIMATION_TOLERANCE_SECONDS",
    "BACKEND_ROLE",
    "CAMERA_WARMUP_FRAMES",
    "BEAGLE_FLOOR_TOLERANCE_CM",
    "BEAGLE_HORIZONTAL_TO_VERTICAL_MIN_RATIO",
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
    "animation_position_seconds",
    "apply_ue_component_frame_delta",
    "build_clean_binaural_mux_command",
    "build_native_apartment_scenario",
    "build_native_apartment_suite",
    "build_native_apartment_motion_pilot_scenario",
    "build_native_apartment_motion_pilot_suite",
    "build_png_encode_command",
    "build_topdown_visual_command",
    "component_frame_delta_for_asset",
    "detached_suite_copy",
    "load_apartment_lighting_profile",
    "resolve_apartment_lighting_profile",
    "scenario_input_paths",
    "motion_pilot_input_paths",
    "read_ue_component_relative_transform",
    "summarize_root_readbacks",
    "summarize_actor_bounds",
    "summarize_anatomical_forward_readbacks",
    "wrap_angle_difference_degrees",
]
