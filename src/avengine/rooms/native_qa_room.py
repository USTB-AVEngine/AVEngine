"""CPU adapter for a native SPEAR Apartment room in the common QA planner.

The Apartment map already owns its rendered room, object placement and
navigation.  This module turns the retained real-surface audit and native UE
route bank into the normalized room/layout/navigation records consumed by the
question-driven planner.  It does not start SPEAR/UE and never edits the
upstream room package.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.camera_pose import yaw_rotation_xyzw
from avengine.rooms.qa_episode import (
    QAPlanningError,
    build_room_navigation,
    match_question_conditions,
    room_capabilities,
    schedule_audio,
    select_question_camera,
    source_declaration,
)
from avengine.rooms.furniture_layout import habitat_to_ue_cm


SCHEMA = "avengine_native_spear_apartment_qa_room_v1"
ROOM_CATALOG_SCHEMA = "avengine_qa_room_catalog_v1"
DEFAULT_FRAME_COUNT = 240
DEFAULT_FRAME_RATE_HZ = 15
DEFAULT_SAMPLE_RATE_HZ = 16_000
DEFAULT_SEED = 20260906
DEFAULT_SOURCE_ASSET_IDS = (
    "rocketbox_human_male_adult_01_top_blue_research_v1",
    "rocketbox_human_male_adult_01_top_green_research_v1",
)


class NativeQAResourceError(QAPlanningError):
    """Native Apartment resources cannot be adapted into a QA room."""


@dataclass(frozen=True)
class NativeApartmentResources:
    """Immutable references to retained upstream Apartment resources."""

    source_root: Path
    room_manifest: Path
    mesh_audit: Path
    surface_glb: Path
    ue_export_manifest: Path
    navmesh: Path
    route_bank: Path
    acoustic_package: Path | None
    scene_id: str
    room_id: str
    map_path: str
    room_profile_path: Path | None = None


def _read_json(path: Path, *, owner: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NativeQAResourceError(f"cannot read {owner}: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise NativeQAResourceError(f"{owner} must be a JSON object: {path}")
    return value


def _required_file(path: Path, *, owner: str) -> Path:
    value = path.expanduser().resolve()
    if not value.is_file():
        raise NativeQAResourceError(f"{owner} is missing: {value}")
    return value


def _profile(repository: Path, profile_path: Path | None) -> tuple[str, str, str, Path | None]:
    path = profile_path or (repository / "examples/runtime/room_runtime_profiles.json")
    if not path.is_file():
        return "apartment_0000", "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000", "spear_apartment_0000", None
    document = _read_json(path, owner="room runtime profile registry")
    profiles = document.get("profiles")
    if not isinstance(profiles, Sequence):
        raise NativeQAResourceError("room runtime profile registry has no profiles")
    for profile in profiles:
        if not isinstance(profile, Mapping) or profile.get("profile_id") != "spear_apartment_0000":
            continue
        scene = profile.get("scene")
        if not isinstance(scene, Mapping):
            break
        scene_id = scene.get("scene_id")
        map_path = scene.get("map_path")
        if isinstance(scene_id, str) and isinstance(map_path, str):
            return scene_id, map_path, str(profile.get("profile_id")), path.resolve()
    raise NativeQAResourceError("spear_apartment_0000 runtime profile is absent")


def discover_native_apartment_resources(
    *,
    repository: str | Path | None = None,
    source_root: str | Path | None = None,
    route_bank: str | Path | None = None,
    room_profile_path: str | Path | None = None,
) -> NativeApartmentResources:
    """Resolve and verify the retained Apartment inputs without mutating them."""

    repo = Path(repository).expanduser().resolve() if repository is not None else Path(__file__).resolve().parents[3]
    source_value = source_root or os.environ.get("AVENGINE_NATIVE_APARTMENT_SOURCE_ROOT")
    route_value = route_bank or os.environ.get("AVENGINE_NATIVE_APARTMENT_ROUTE_BANK")
    if not source_value:
        raise NativeQAResourceError(
            "native Apartment source_root is required (pass source_root or "
            "AVENGINE_NATIVE_APARTMENT_SOURCE_ROOT)"
        )
    if not route_value:
        raise NativeQAResourceError(
            "native Apartment route_bank is required (pass route_bank or "
            "AVENGINE_NATIVE_APARTMENT_ROUTE_BANK)"
        )
    root = Path(source_value).expanduser().resolve()
    scene_id, map_path, _profile_id, profile_path = _profile(
        repo, Path(room_profile_path).expanduser().resolve() if room_profile_path else None
    )
    room_manifest = _required_file(
        root / "m1/legacy_apartment_package/room_manifest.json",
        owner="native Apartment room manifest",
    )
    mesh_audit = _required_file(
        root / "m1/legacy_apartment_export/mesh_audit.json",
        owner="native Apartment mesh audit",
    )
    surface_glb = _required_file(
        root / "m1/legacy_apartment_export/scene.glb",
        owner="native Apartment real-surface GLB",
    )
    ue_export_manifest = _required_file(
        root / "m1/legacy_apartment_export/ue_export_manifest.json",
        owner="native Apartment UE export manifest",
    )
    navmesh = _required_file(
        root / "m1/legacy_apartment_package/visual/navmeshes/legacy_apartment_0000.navmesh",
        owner="native Apartment navmesh",
    )
    route_path = _required_file(Path(route_value), owner="native Apartment route bank")
    acoustic = root / "m3/root_ue_package_current_20260718_02/manifest.json"
    return NativeApartmentResources(
        source_root=root,
        room_manifest=room_manifest,
        mesh_audit=mesh_audit,
        surface_glb=surface_glb,
        ue_export_manifest=ue_export_manifest,
        navmesh=navmesh,
        route_bank=route_path,
        acoustic_package=acoustic.resolve() if acoustic.is_file() else None,
        scene_id=scene_id,
        room_id="legacy_ue_apartment_0000_v1",
        map_path=map_path,
        room_profile_path=profile_path,
    )


def _canonical_to_authoring_bounds(raw: Any, *, owner: str) -> list[list[float]]:
    if not isinstance(raw, Sequence) or len(raw) != 2:
        raise NativeQAResourceError(f"{owner} must be [min,max]")
    try:
        low = [float(x) for x in raw[0]]
        high = [float(x) for x in raw[1]]
    except (TypeError, ValueError) as exc:
        raise NativeQAResourceError(f"{owner} must contain numeric vectors") from exc
    if len(low) != 3 or len(high) != 3 or not all(math.isfinite(x) for x in low + high):
        raise NativeQAResourceError(f"{owner} must contain finite 3-vectors")
    if any(high[i] <= low[i] for i in range(3)):
        raise NativeQAResourceError(f"{owner} has nonpositive extent")
    # Source audit/GLB is canonical (X,Y,Z) with +Y up.  The common planner
    # uses the authoring (+Z up) frame: authoring=(X,-Z,Y).
    return [[low[0], -high[2], low[1]], [high[0], -low[2], high[1]]]


def _semantic_class(node: str, mesh: str) -> str:
    value = f"{node} {mesh}".casefold().replace("_", " ")
    ordered = (
        ("floor", "floor"),
        ("ceiling", "ceiling"),
        ("wall", "wall"),
        ("door frame", "door_frame"),
        ("door", "door"),
        ("casement", "window"),
        ("curtain", "curtain"),
        ("sofa", "sofa"),
        ("couch", "sofa"),
        ("chair", "chair"),
        ("table", "table"),
        ("cabinet", "cabinet"),
        ("shelf", "shelf"),
        ("fireplace", "fireplace"),
        ("carpet", "rug"),
        ("mirror", "mirror"),
        ("picture", "picture"),
        ("lamp", "lamp"),
        ("vase", "vase"),
        ("cushion", "cushion"),
        ("drawer", "drawer"),
        ("oven", "oven"),
        ("sink", "sink"),
        ("extractor", "extractor"),
        ("vinyl", "picture"),
        ("prop", "prop"),
    )
    for token, category in ordered:
        if token in value:
            return category
    return "scene_object"


def build_native_apartment_layout(
    resources: NativeApartmentResources,
) -> dict[str, Any]:
    """Build common layout metadata from the real-surface audit."""

    audit = _read_json(resources.mesh_audit, owner="native Apartment mesh audit")
    bounds = _canonical_to_authoring_bounds(
        [audit.get("bounds", {}).get("min"), audit.get("bounds", {}).get("max")],
        owner="mesh audit bounds",
    )
    details = audit.get("details")
    breakdown = details.get("mesh_breakdown") if isinstance(details, Mapping) else None
    if not isinstance(breakdown, Sequence) or not breakdown:
        raise NativeQAResourceError("mesh audit has no mesh_breakdown object semantics")
    objects: list[dict[str, Any]] = []
    for index, raw in enumerate(breakdown):
        if not isinstance(raw, Mapping):
            continue
        node = str(raw.get("node") or f"mesh_node_{index:03d}")
        mesh = str(raw.get("mesh_datablock") or "")
        item_bounds = _canonical_to_authoring_bounds(
            [raw.get("bounds", {}).get("min"), raw.get("bounds", {}).get("max")],
            owner=f"mesh audit object {node}",
        )
        category = _semantic_class(node, mesh)
        role = (
            "walkable_surface"
            if category == "floor"
            else "walkable_floor_covering"
            if category == "rug"
            else "ground_blocker"
        )
        objects.append(
            {
                "object_id": f"native_mesh::{node}",
                "semantic_class": category,
                "navigation_role": role,
                "static": True,
                "bounds_xyz_m": item_bounds,
                "source": "native_mesh_audit",
                "geometry_ref": {
                    "node": node,
                    "mesh_datablock": mesh,
                    "triangle_count": raw.get("triangles"),
                },
            }
        )
    floors = [item for item in objects if item["semantic_class"] == "floor"]
    if not floors:
        raise NativeQAResourceError("native mesh audit has no floor object")
    floor_height = max(float(item["bounds_xyz_m"][1][2]) for item in floors)
    return {
        "schema": SCHEMA,
        "room_id": resources.room_id,
        "scene_id": resources.scene_id,
        "room_family_id": "spear_native_apartment",
        "status": "research_candidate",
        "backend_route": "spear_unreal",
        "map_path": resources.map_path,
        "manifest_path": str(resources.room_manifest),
        "coordinate_contract": {
            "authoring": "native GLB inverse transform (X,-Z,Y), metres",
            "habitat": "[authoring_x, authoring_z, -authoring_y]",
        },
        "geometry": {
            "bounds_xy_m": [bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]],
            "bounds_source": "native_mesh_audit",
            "authoring_geometry_status": "native_surface_audit",
            "native_validation_status": "not_run",
            "claim_boundary": "native exported surface audit; UE readback is a later stage",
        },
        "objects": objects,
        "furniture_assemblies": [],
        "seats": [],
        "resources": {
            "visual_geometry": {
                "declared": str(resources.surface_glb),
                "resolved": str(resources.surface_glb),
                "status": "available",
                "authority": "native_real_surface_export",
            },
            "collision_geometry": {
                "declared": str(resources.surface_glb),
                "resolved": str(resources.surface_glb),
                "status": "available",
                "authority": "native_real_surface_export",
            },
            "navmesh": {
                "declared": str(resources.navmesh),
                "resolved": str(resources.navmesh),
                "status": "available",
                "authority": "native_spear_navmesh",
            },
            "route_bank": {
                "declared": str(resources.route_bank),
                "resolved": str(resources.route_bank),
                "status": "available",
                "authority": "native_spear_route_bank",
            },
            "mesh_audit": {
                "declared": str(resources.mesh_audit),
                "resolved": str(resources.mesh_audit),
                "status": "available",
                "authority": "native_mesh_audit",
            },
        },
        "visual_lighting": {
            "profile_id": "native_spear_apartment_map",
            "status": "declared",
            "native_validation_status": "not_run",
            "claim_boundary": "native map lighting and post-process",
        },
        "native_floor_height_m": floor_height,
        "native_source": {
            "room_manifest": str(resources.room_manifest),
            "ue_export_manifest": str(resources.ue_export_manifest),
            "surface_glb": str(resources.surface_glb),
            "mesh_audit": str(resources.mesh_audit),
        },
        "camera_source": "native_surface_geometry_grid",
        "review_cameras_used": False,
    }


def _route_points(raw: Mapping[str, Any]) -> np.ndarray:
    samples = raw.get("samples_ue_cm")
    waypoints = raw.get("waypoints_ue_cm")
    if not isinstance(samples, Sequence) or not isinstance(waypoints, Sequence) or not waypoints:
        raise NativeQAResourceError("route lacks native UE samples or waypoints")
    heights = [float(item[2]) for item in waypoints if isinstance(item, Sequence) and len(item) >= 3]
    if not heights or not all(math.isfinite(item) for item in heights):
        raise NativeQAResourceError("route has no finite UE floor height")
    floor_cm = float(np.median(heights))
    points: list[list[float]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Sequence) or len(sample) < 2:
            raise NativeQAResourceError(f"route sample {index} is invalid")
        x_cm, y_cm = float(sample[0]), float(sample[1])
        if not all(math.isfinite(item) for item in (x_cm, y_cm)):
            raise NativeQAResourceError("route sample contains a nonfinite coordinate")
        points.append([x_cm / 100.0, floor_cm / 100.0, y_cm / 100.0])
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) < 2:
        raise NativeQAResourceError("native route needs at least two samples")
    return result


def _hold_endpoint_path(
    path: np.ndarray,
    frame_count: int,
    *,
    start_hold_frames: int = 0,
) -> np.ndarray:
    """Use one native path, then hold its endpoint for the remaining frames."""

    if (
        isinstance(start_hold_frames, bool)
        or not isinstance(start_hold_frames, int)
        or start_hold_frames < 0
    ):
        raise NativeQAResourceError("start_hold_frames must be a nonnegative integer")
    prefix = np.repeat(path[:1], min(start_hold_frames, frame_count), axis=0)
    remaining = frame_count - len(prefix)
    motion = path[: min(len(path), remaining)]
    remaining -= len(motion)
    suffix = np.repeat(path[-1:], max(0, remaining), axis=0)
    result = np.concatenate((prefix, motion, suffix), axis=0)
    if len(result) != frame_count:
        raise NativeQAResourceError("native endpoint-hold route has the wrong frame count")
    return np.ascontiguousarray(result)


def select_native_walking_routes(
    resources: NativeApartmentResources,
    *,
    frame_count: int = DEFAULT_FRAME_COUNT,
    frame_rate_hz: int | float = DEFAULT_FRAME_RATE_HZ,
    seed: int = DEFAULT_SEED,
    minimum_separation_m: float = 0.95,
    start_hold_frames: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Select two moving paths from the native UE Recast route bank."""

    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 2:
        raise NativeQAResourceError("frame_count must be an integer >= 2")
    if (
        isinstance(frame_rate_hz, bool)
        or not isinstance(frame_rate_hz, (int, float))
        or not math.isfinite(float(frame_rate_hz))
        or float(frame_rate_hz) <= 0.0
        or not float(frame_rate_hz).is_integer()
        or 48_000 % int(frame_rate_hz) != 0
    ):
        raise NativeQAResourceError(
            "requested frame_rate_hz must be a positive integer divisor of 48000"
        )
    requested_rate_hz = int(frame_rate_hz)
    bank = _read_json(resources.route_bank, owner="native Apartment route bank")
    raw_routes = bank.get("routes")
    if not isinstance(raw_routes, Sequence):
        raise NativeQAResourceError("native route bank has no routes")
    bank_frame_count = int(bank.get("frame_count") or 0)
    bank_seconds = float(bank.get("clip_seconds") or 0.0)
    bank_rate = bank.get("frame_rate_hz", bank.get("frame_rate"))
    bank_rate_source = "declared_frame_rate_hz"
    if bank_rate is None:
        if bank_frame_count < 2 or bank_seconds <= 0.0:
            raise NativeQAResourceError(
                "native route bank must declare frame_rate_hz or a valid frame_count/clip_seconds clock"
            )
        derived_rate = bank_frame_count / bank_seconds
        if not math.isfinite(derived_rate) or not derived_rate.is_integer():
            raise NativeQAResourceError(
                "native route bank frame_count/clip_seconds does not yield an integral frame rate"
            )
        bank_rate = int(derived_rate)
        bank_rate_source = "derived_from_frame_count_and_clip_seconds"
    if (
        isinstance(bank_rate, bool)
        or not isinstance(bank_rate, (int, float))
        or not math.isfinite(float(bank_rate))
        or float(bank_rate) <= 0.0
        or not float(bank_rate).is_integer()
        or 48_000 % int(bank_rate) != 0
    ):
        raise NativeQAResourceError(
            "native route bank clock must be a positive integer frame rate divisor of 48000"
        )
    declared_rate_hz = int(bank_rate)
    if declared_rate_hz != requested_rate_hz:
        raise NativeQAResourceError(
            "requested frame_rate_hz does not match native route bank clock: "
            f"requested {requested_rate_hz}, bank {declared_rate_hz}"
        )
    if bank_frame_count < 2 or bank_seconds <= 0.0:
        raise NativeQAResourceError("native route bank clock is invalid")
    expected_seconds = bank_frame_count / declared_rate_hz
    if not math.isclose(bank_seconds, expected_seconds, rel_tol=0.0, abs_tol=1.0e-9):
        raise NativeQAResourceError(
            "native route bank clip_seconds disagrees with its declared clock: "
            f"{bank_seconds} != {expected_seconds}"
        )
    candidates: list[tuple[str, np.ndarray, float, float]] = []
    for raw in raw_routes:
        if not isinstance(raw, Mapping):
            continue
        try:
            path = _route_points(raw)
        except NativeQAResourceError:
            continue
        if len(path) != bank_frame_count:
            continue
        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        speed = length / bank_seconds
        if length < 2.0 or not 0.6 <= speed <= 1.5:
            continue
        candidates.append((str(raw.get("route_id") or f"route_{len(candidates):05d}"), path, length, speed))
    if len(candidates) < 2:
        raise NativeQAResourceError("native route bank has fewer than two walking candidates")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))[: min(512, len(candidates))]
    expanded_paths = {
        index: _hold_endpoint_path(
            candidate[1], frame_count, start_hold_frames=start_hold_frames
        )
        for index, candidate in enumerate(candidates)
    }
    best: tuple[float, int, int, float] | None = None
    for left, first_index in enumerate(order):
        for second_index in order[left + 1 :]:
            first = expanded_paths[int(first_index)]
            second = expanded_paths[int(second_index)]
            distances = np.linalg.norm(first - second, axis=1)
            separation = float(distances.min())
            endpoint_separation = max(float(distances[0]), float(distances[-1]))
            if separation < minimum_separation_m:
                continue
            # Holding at the endpoints is deliberate, so pair selection also
            # keeps the two people in one camera group after the walk.  This
            # is still a route-bank constraint, not a room-coordinate special
            # case.
            if endpoint_separation > 3.5:
                continue
            score = abs(separation - 1.5) + 0.25 * endpoint_separation
            value = (score, int(first_index), int(second_index), separation)
            if best is None or value < best:
                best = value
    if best is None:
        raise NativeQAResourceError("native route bank has no separated two-person walking pair")
    _, first_index, second_index, bank_separation = best
    first = candidates[first_index]
    second = candidates[second_index]
    first_path = expanded_paths[first_index]
    second_path = expanded_paths[second_index]
    separation = float(np.linalg.norm(first_path - second_path, axis=1).min())
    if separation < minimum_separation_m:
        raise NativeQAResourceError("native endpoint-hold routes violate the actor separation gate")
    return {"source1": first_path, "source2": second_path}, {
        "authority": "native_spear_ue_recast_route_bank",
        "route_bank": str(resources.route_bank),
        "bank_frame_count": bank_frame_count,
        "bank_frame_rate_hz": declared_rate_hz,
        "bank_frame_rate_source": bank_rate_source,
        "bank_clip_seconds": bank_seconds,
        "selected_route_ids": {"source1": first[0], "source2": second[0]},
        "selected_route_speeds_mps": {"source1": first[3], "source2": second[3]},
        "selected_route_lengths_m": {"source1": first[2], "source2": second[2]},
        "bank_minimum_pair_separation_m": bank_separation,
        "endpoint_pair_separation_m": max(
            float(np.linalg.norm(first_path[0] - second_path[0])),
            float(np.linalg.norm(first_path[-1] - second_path[-1])),
        ),
        "expanded_frame_count": frame_count,
        "expanded_route_policy": "native_samples_once_then_endpoint_hold",
        "start_hold_frames": start_hold_frames,
        "minimum_actor_separation_m": separation,
        "native_navmesh": str(resources.navmesh),
        "native_navmesh_validation": "route_bank_source_readback",
        "claim_boundary": "source-center paths come from native UE navigation; body collision and native capture remain later gates",
    }


def load_sound_pool(path: str | Path) -> list[dict[str, Any]]:
    """Read an existing checked sound-pool manifest without rewriting it."""

    value = _read_json(Path(path).expanduser().resolve(), owner="native QA sound pool")
    sounds = value.get("sounds", value.get("pool", value))
    if not isinstance(sounds, Sequence) or isinstance(sounds, (str, bytes)):
        raise NativeQAResourceError("sound pool must contain a sounds list")
    result: list[dict[str, Any]] = []
    for index, sound in enumerate(sounds):
        if not isinstance(sound, Mapping):
            raise NativeQAResourceError(f"sound pool item {index} must be an object")
        asset_id = sound.get("sound_asset_id")
        sound_path = sound.get("path")
        if not isinstance(asset_id, str) or not asset_id.strip():
            raise NativeQAResourceError(f"sound pool item {index} lacks sound_asset_id")
        if not isinstance(sound_path, str) or not sound_path.strip():
            raise NativeQAResourceError(f"sound pool item {asset_id!r} lacks path")
        resolved = Path(sound_path).expanduser().resolve()
        if not resolved.is_file():
            raise NativeQAResourceError(f"sound pool item {asset_id!r} path is missing: {resolved}")
        result.append({**deepcopy(dict(sound)), "path": str(resolved)})
    if len(result) < 2:
        raise NativeQAResourceError("native QA walking plan requires at least two sounds")
    return result


def native_apartment_room_entry(
    resources: NativeApartmentResources,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a room-pool record with no authored actor coordinates."""

    return {
        "room_id": resources.room_id,
        "manifest": str(Path(manifest_path).expanduser().resolve()) if manifest_path else str(resources.room_manifest),
        "backend": "spear_unreal",
        "map_path": resources.map_path,
        "navmesh_path": str(resources.navmesh),
        "route_bank": str(resources.route_bank),
        "acoustic_package": str(resources.acoustic_package) if resources.acoustic_package else None,
        "native_room_adapter": SCHEMA,
        "native_scene_id": resources.scene_id,
        "native_input_root": str(resources.source_root),
        "native_room_profile": (
            str(resources.room_profile_path)
            if resources.room_profile_path is not None
            else None
        ),
        "source_surface": str(resources.surface_glb),
        "source_mesh_audit": str(resources.mesh_audit),
        "placement_policy": "native_room_metadata_and_native_route_bank",
        "coordinate_policy": "derive_from_native_readbacks_and_audit",
    }


def _build_actor_states(
    actors: Sequence[Mapping[str, Any]],
    routes: Mapping[str, np.ndarray],
    *,
    frame_count: int,
    ticks_per_frame: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    headings: dict[str, float] = {}
    phases = {str(actor["actor_id"]): float(rng.random()) for actor in actors}
    for actor in actors:
        aid = str(actor["actor_id"])
        path = routes[aid]
        delta = path[1] - path[0]
        headings[aid] = math.degrees(math.atan2(float(delta[2]), float(delta[0])))
    frames: list[dict[str, Any]] = []
    for frame_index in range(frame_count):
        states = []
        for actor in actors:
            aid = str(actor["actor_id"])
            path = routes[aid]
            delta = path[min(frame_index + 1, frame_count - 1)] - path[frame_index]
            moving = float(np.linalg.norm(delta[[0, 2]])) > 1.0e-5
            if moving:
                headings[aid] = math.degrees(math.atan2(float(delta[2]), float(delta[0])))
            anatomical = float(actor["ue_anatomical_forward_yaw_deg"])
            yaw = (headings[aid] - anatomical + 180.0) % 360.0 - 180.0
            action = actor["walking_action_id"] if moving else actor["idle_action_id"]
            if moving:
                phases[aid] = (phases[aid] + 1.0 / float(actor["walk_phase_period_frames"])) % 1.0
            point = path[frame_index].tolist()
            rotation = yaw_rotation_xyzw(-yaw)
            states.append(
                {
                    "actor_id": aid,
                    "translation_m": point,
                    "translation_ue_cm": habitat_to_ue_cm(point),
                    "rotation_xyzw": list(rotation),
                    "actor_yaw_ue_deg": yaw,
                    "root_transform": {
                        "translation_m": point,
                        "rotation_xyzw": list(rotation),
                        "scale": [1.0, 1.0, 1.0],
                    },
                    "action_id": action,
                    "action_phase": phases[aid],
                    "action_time_ticks": frame_index * ticks_per_frame,
                    "ue_animation": actor["animation_paths_by_action_id"][action],
                    "moving": moving,
                    "frame_index": frame_index,
                }
            )
        frames.append({"frame_index": frame_index, "pts_ticks": frame_index * ticks_per_frame, "actor_states": states})
    return frames


def build_native_apartment_qa_plan(
    *,
    resources: NativeApartmentResources,
    source_registry: Mapping[str, Any],
    sounds: Sequence[Mapping[str, Any]],
    episode_id: str,
    source_asset_ids: Sequence[str] = DEFAULT_SOURCE_ASSET_IDS,
    qa_ids: Sequence[str] | None = None,
    seed: int = DEFAULT_SEED,
    frame_count: int = DEFAULT_FRAME_COUNT,
    frame_rate_hz: int = DEFAULT_FRAME_RATE_HZ,
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    camera_motion: str = "follow_group",
    audio_mode: str = "sequential",
    start_hold_frames: int = 0,
) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, np.ndarray]]:
    """Build one common-plan Episode using native routes and room geometry."""

    if not isinstance(episode_id, str) or not episode_id.strip():
        raise NativeQAResourceError("episode_id must be a nonempty string")
    if frame_rate_hz <= 0 or sample_rate_hz <= 0:
        raise NativeQAResourceError("episode clock rates must be positive")
    selected = list(source_asset_ids)
    if len(selected) != 2 or len(set(selected)) != 2:
        raise NativeQAResourceError("exactly two distinct human source assets are required")
    layout = build_native_apartment_layout(resources)
    routes, route_record = select_native_walking_routes(
        resources,
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
        seed=seed,
        start_hold_frames=start_hold_frames,
    )
    # This raster is only the common camera candidate adapter.  Actor routes
    # and their legality stay owned by the native UE route bank above.
    pf, raster_nav = build_room_navigation(
        layout,
        resolution_m=0.08,
        clearance_m=0.38,
        body_height_m=1.9,
        floor_height_m=float(layout["native_floor_height_m"]),
    )
    native_nav = {
        **raster_nav,
        **route_record,
        "camera_candidate_adapter": raster_nav["authority"],
    }
    actors = [source_declaration(source_registry, asset_id, f"source{index + 1}") for index, asset_id in enumerate(selected)]
    ids = list(qa_ids or [f"QA-{index:02d}" for index in range(1, 25)])
    capability = room_capabilities(layout, native_nav, actors, sounds)
    matching = match_question_conditions(ids, capability)
    if matching["status"] == "unsupported":
        raise NativeQAResourceError(f"native Apartment room has no requested potential: {matching}")
    camera, cameras, camera_record = select_question_camera(
        layout,
        pf,
        routes,
        actors,
        rng=np.random.default_rng(seed),
        camera_motion=camera_motion,
        qa_ids=ids,
        camera_fov_deg=105.0,
    )
    ticks_per_frame = 48_000 // int(frame_rate_hz)
    clock = {
        "frame_count": int(frame_count),
        "frame_rate_hz": int(frame_rate_hz),
        "sample_rate_hz": int(sample_rate_hz),
        "clip_seconds": float(frame_count / frame_rate_hz),
        "sample_count": int(round(frame_count / frame_rate_hz * sample_rate_hz)),
        "time_base_hz": 48_000,
        "ticks_per_frame": ticks_per_frame,
        "duration_seconds": float(frame_count / frame_rate_hz),
        "compatibility": "native_apartment_route_bank_endpoint_hold",
    }
    frames = _build_actor_states(
        actors,
        routes,
        frame_count=frame_count,
        ticks_per_frame=ticks_per_frame,
        seed=seed,
    )
    for frame, camera_state in zip(frames, cameras, strict=True):
        frame["camera_state"] = camera_state
    events, bindings = schedule_audio(
        actors,
        sounds,
        clock=clock,
        rng=np.random.default_rng(seed + 1),
        mode=audio_mode,
    )
    room_entry = native_apartment_room_entry(resources)
    plan = {
        "kind": "avengine_question_driven_episode",
        "status": "research_candidate",
        "episode_id": episode_id,
        "seed": int(seed),
        "native_input_root": str(resources.source_root),
        "native_room_profile": (
            str(resources.room_profile_path)
            if resources.room_profile_path is not None
            else None
        ),
        "clock": clock,
        "scene": {
            "scene_id": resources.scene_id,
            "room_id": resources.room_id,
            "map_path": resources.map_path,
            "backend": "spear_unreal",
        },
        "request": {
            "episode_id": episode_id,
            "source_asset_ids": selected,
            "qa_ids": ids,
            "activity": "walking",
            "camera_motion": camera_motion,
            "audio_mode": audio_mode,
            "frame_count": frame_count,
            "frame_rate_hz": frame_rate_hz,
            "sample_rate_hz": sample_rate_hz,
            "native_room_adapter": SCHEMA,
        },
        "room_capabilities": capability,
        "question_condition_match": matching,
        "activity_plan": route_record,
        "camera_condition_sampling": camera_record,
        "audio_events": events,
        "voice_bindings": bindings,
        "visual_lighting": deepcopy(layout["visual_lighting"]),
        "visual_plan": {
            "backend_role": "production_visual",
            "camera": camera,
            "actors": actors,
            "frames": frames,
            "render": {
                "frame_count": frame_count,
                "fps_num": frame_rate_hz,
                "fps_den": 1,
                "ticks_per_frame": ticks_per_frame,
            },
            "authority": {
                "actor_state": "native_spear_route_bank_adapter",
                "camera_listener": "common_question_condition_camera_adapter",
                "backend_may_replan": False,
            },
        },
        "resources": {
            **room_entry,
            "native_input_root": str(resources.source_root),
            "native_room_profile": (
                str(resources.room_profile_path)
                if resources.room_profile_path is not None
                else None
            ),
            "manifest": str(resources.room_manifest),
            "surface_glb": str(resources.surface_glb),
            "mesh_audit": str(resources.mesh_audit),
            "native_navmesh": str(resources.navmesh),
            "native_route_bank": str(resources.route_bank),
            "acoustic_package": str(resources.acoustic_package) if resources.acoustic_package else None,
        },
        "evidence_status": {
            "native_visual": "not_run",
            "native_audio": "not_run",
            "qa_validity": "not_run",
            "model_evaluation": "not_run",
        },
        "qualification_claim": False,
        "formal_dataset_registration_authorized": False,
    }
    return plan, layout, pf, routes


def write_json(path: str | Path, value: Any) -> Path:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return target


__all__ = [
    "DEFAULT_FRAME_COUNT",
    "DEFAULT_FRAME_RATE_HZ",
    "DEFAULT_SAMPLE_RATE_HZ",
    "DEFAULT_SEED",
    "DEFAULT_SOURCE_ASSET_IDS",
    "NativeApartmentResources",
    "NativeQAResourceError",
    "build_native_apartment_layout",
    "build_native_apartment_qa_plan",
    "discover_native_apartment_resources",
    "load_sound_pool",
    "native_apartment_room_entry",
    "select_native_walking_routes",
    "write_json",
]
