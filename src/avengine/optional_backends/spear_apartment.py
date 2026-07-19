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
COMPONENT_FRAME_DELTA_SCHEMA = "avengine_spear_component_frame_delta_v1"


SCENARIO_DIRECTORIES: Mapping[str, tuple[str, str]] = {
    "S0": ("S0_routing_sanity", "A"),
    "S3": ("S3_moving_source", "A"),
    "S4": ("S4_overlapping_sources", "A"),
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
            "/Game/MyAssets/Audioset/Meshes/"
            f"gate_{BEAGLE_TAG}/Idle.Idle"
        ),
        "walking_animation": (
            "/Game/MyAssets/Audioset/Meshes/"
            f"gate_{BEAGLE_TAG}/Walking.Walking"
        ),
        # The imported Beagle's anatomical forward is UE actor-local +X.
        "ue_anatomical_forward_yaw_deg": 0.0,
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
            "/Game/MyAssets/Audioset/Meshes/"
            f"gate_{HUMAN_TAG}/Walking.Walking"
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


def _unreal_struct_triplet(value: Any, names: Sequence[str], *, owner: str) -> list[float]:
    expected = [name.casefold() for name in names]
    current = value
    for _ in range(3):
        if not isinstance(current, Mapping):
            break
        lowered = {str(key).casefold(): item for key, item in current.items()}
        if all(name in lowered for name in expected):
            return _finite_triplet(
                [lowered[name] for name in expected], owner=owner
            )
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
    rotation_error = max(
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
        "timeline_anchor_mutated": False,
        "target": "attached_visual_actor_root_component",
    }


def scenario_input_paths(
    bundle_root: str | Path, scenario_id: str
) -> dict[str, Path]:
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


def _validate_native_plan(plan: Mapping[str, Any], *, scenario_id: str) -> None:
    if plan.get("schema") != PLAN_SCHEMA or plan.get("backend_role") != BACKEND_ROLE:
        raise SpearApartmentError("input did not compile to a comparison-visual plan")
    authority = plan.get("authority")
    if not isinstance(authority, Mapping) or authority.get("backend_may_replan") is not False:
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
        raise SpearApartmentError("native comparison requires the frozen 105 degree HFOV")
    if len(plan.get("frames", ())) != FRAME_COUNT:
        raise SpearApartmentError("native comparison requires exactly 75 frames")
    actor_ids = [actor.get("actor_id") for actor in plan.get("actors", ())]
    if actor_ids != ["dog0", "human0"]:
        raise SpearApartmentError("native Apartment S0/S3/S4 actor closure changed")


def build_native_apartment_scenario(
    bundle_root: str | Path,
    scenario_id: str,
    *,
    actor_bindings: Mapping[str, Mapping[str, Any]] = DEFAULT_ACTOR_BINDINGS,
) -> dict[str, Any]:
    """Compile one existing M6.x scenario into a native UE execution record."""

    root = Path(bundle_root).resolve()
    paths = scenario_input_paths(root, scenario_id)
    plan = build_spear_visual_plan_from_files(
        timeline_path=paths["timeline"],
        source_manifest_path=paths["source_manifest"],
        flags_path=paths["flags"],
        room_capsule_path=paths["room_capsule"],
        qualification_path=paths["qualification"],
        actor_bindings=actor_bindings,
    )
    _validate_native_plan(plan, scenario_id=scenario_id)
    # The generic visual compiler owns actor-root/yaw semantics.  Apartment's
    # imported-asset frame deltas are deliberately attached afterwards so
    # they cannot leak into or modify those authoritative root transforms.
    for actor in plan["actors"]:
        asset_id = actor["asset_id"]
        actor["ue_component_frame_delta"] = component_frame_delta_for_asset(
            asset_id, actor_bindings=actor_bindings
        )
    scenario_directory, variant_id = SCENARIO_DIRECTORIES[scenario_id]
    return {
        "schema": SCENARIO_SCHEMA,
        "scenario_id": scenario_id,
        "scenario_directory": scenario_directory,
        "variant_id": variant_id,
        "backend_role": BACKEND_ROLE,
        "native_scene": {
            "map": NATIVE_APARTMENT_MAP,
            "layout": "native_map_unchanged",
            "lighting": "native_map_unchanged_no_added_lights",
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


def build_native_apartment_suite(
    bundle_root: str | Path,
    *,
    scenario_ids: Sequence[str] = ("S0", "S3", "S4"),
    actor_bindings: Mapping[str, Mapping[str, Any]] = DEFAULT_ACTOR_BINDINGS,
) -> dict[str, Any]:
    """Compile the requested native Apartment comparison scenarios."""

    selected = tuple(scenario_ids)
    if not selected or len(selected) != len(set(selected)):
        raise SpearApartmentError("scenario selection must be nonempty and unique")
    scenarios = [
        build_native_apartment_scenario(
            bundle_root, scenario_id, actor_bindings=actor_bindings
        )
        for scenario_id in selected
    ]
    return {
        "schema": SUITE_SCHEMA,
        "backend_role": BACKEND_ROLE,
        "native_map": NATIVE_APARTMENT_MAP,
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


def animation_position_seconds(action_phase: float, animation_length_seconds: float) -> float:
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
                abs(float(record["location_cm"][axis]) - float(camera_position_cm[axis]))
                for axis in range(3)
            )
        )
        camera_yaw_errors.append(
            abs(
                wrap_angle_difference_degrees(
                    record["rotation_deg"][2], camera_yaw_deg
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


def build_png_encode_command(
    *, frames_pattern: str | Path, output_path: str | Path
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
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
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


def build_topdown_binaural_command(
    *,
    ue_video_path: str | Path,
    authoritative_diagnostic_path: str | Path,
    output_path: str | Path,
) -> list[str]:
    """Pair UE main pixels with the unchanged authoritative Topdown panel."""

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
        "-map",
        "1:a:0",
        "-frames:v",
        str(FRAME_COUNT),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def detached_suite_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy for tools that append runtime evidence."""

    return deepcopy(dict(value))


__all__ = [
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
    "NATIVE_APARTMENT_MAP",
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
    "build_png_encode_command",
    "build_topdown_binaural_command",
    "component_frame_delta_for_asset",
    "detached_suite_copy",
    "scenario_input_paths",
    "read_ue_component_relative_transform",
    "summarize_root_readbacks",
    "summarize_actor_bounds",
    "wrap_angle_difference_degrees",
]
