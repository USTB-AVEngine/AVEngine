"""Code-owned qualification matrix for registered source assets.

The matrix separates registry identity from evidence that was actually
observed.  Every dimension here is derived from an input artifact -- the
runtime registry, a measured asset-geometry catalog, a support-surface
catalog, a real placement plan, a retained native readback, or a media probe
of the stem on disk.  No caller may hand this module a verdict: an evidence
input that carries a ``status``/``*_status`` verdict key is rejected, so a
provider can only widen what is measured, never assert a pass.

A row is eligible only when every required dimension has an observed ``pass``.
Anything not measured stays ``not_run`` and names what is missing.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import json
import math

SCHEMA = "avengine_source_asset_qualification_matrix_v1"
STATUSES = frozenset({"pass", "fail", "not_run"})
REQUIRED_DIMENSIONS = (
    "registry",
    "geometry_scale",
    "support",
    "placement",
    "clearance",
    "emitter",
    "visibility",
    "sound_pcm",
)
#: Evidence inputs are observations, never verdicts.  These key names would let
#: a caller write its own qualification result, so they are refused outright.
FORBIDDEN_EVIDENCE_KEYS = frozenset(
    {"status", "qualification_status", "overall_status", "verdict", "eligible"}
) | frozenset(f"{name}_status" for name in REQUIRED_DIMENSIONS)
#: A planner row states its own planning stage in ``status``.  That is an
#: observation about the plan, not a qualification verdict, so ``status`` is
#: allowed inside a declared placement row and refused everywhere else.  The
#: verdict names above stay refused at every depth.
PLANNING_STAGE_CONTAINERS = frozenset({"placement_rows", "instances"})
#: What a planner row itself may say about its own stage.
PLANNING_STAGE_STATUS_VALUES = frozenset(
    {"planned", "rejected", "partial", "planning", "candidate"}
)
#: What a sub-check inside that row may say about itself: a clearance block, an
#: AABB check, a room query. These are the planner's own readings, not a
#: qualification verdict for the asset.
PLANNING_SUBCHECK_STATUS_VALUES = PLANNING_STAGE_STATUS_VALUES | frozenset(
    {"pass", "fail", "not_run"}
)
#: A row may disclaim a qualification claim; it may never assert one.
PLANNING_DISCLAIMER_KEYS = frozenset({"qualification_status", "overall_status"})
PLANNING_DISCLAIMER_VALUES = frozenset({"not_run"})

DEFAULT_HEIGHT_TOLERANCE_M = 0.02
DEFAULT_FOOTPRINT_TOLERANCE_M = 0.02
DEFAULT_SUPPORT_PLANE_TOLERANCE_M = 0.05


class QualificationEvidenceError(ValueError):
    """An evidence input carried a verdict instead of an observation."""


def _load(value: Any) -> Any:
    if isinstance(value, (str, Path)):
        return json.loads(Path(value).expanduser().read_text(encoding="utf-8"))
    return value


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _vector(value: Any, length: int = 3) -> list[float] | None:
    items = _list(value)
    if len(items) != length:
        return None
    out = [_finite(item) for item in items]
    return None if any(item is None for item in out) else out  # type: ignore[return-value]


def _source_ref(value: Any) -> str:
    if isinstance(value, (str, Path)):
        return str(value)
    return "mapping"


def _reject_verdict_keys(value: Any, owner: str, *, stage: str | None = None) -> None:
    """Refuse an evidence input that tries to assert its own qualification.

    A declared placement row legitimately carries ``status: "planned"``: that is
    the planner saying which stage the row reached, which is exactly the kind of
    observation this channel is for.  Inside such a row ``status`` is allowed
    when its value is one of the planner's own stages, and a qualification
    verdict smuggled in as ``status: "pass"`` is still refused.  Everywhere else
    ``status`` stays refused, and the verdict names are refused at any depth.
    """
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            if name not in FORBIDDEN_EVIDENCE_KEYS:
                continue
            if stage and name in PLANNING_DISCLAIMER_KEYS:
                if str(child) in PLANNING_DISCLAIMER_VALUES:
                    continue
                raise QualificationEvidenceError(
                    f"{owner}.{name} is {child!r}; a placement row may disclaim a "
                    "qualification claim, never assert one")
            if stage and name == "status":
                allowed = (PLANNING_STAGE_STATUS_VALUES if stage == "row"
                           else PLANNING_SUBCHECK_STATUS_VALUES)
                if str(child) in allowed:
                    continue
                raise QualificationEvidenceError(
                    f"{owner}.{name} is {child!r}; here a placement row may state "
                    f"{sorted(allowed)}, not a qualification verdict")
            raise QualificationEvidenceError(
                f"{owner} carries the verdict key {name!r}; qualification "
                "evidence must be observations, not statuses"
            )
        for key, child in value.items():
            child_stage = "row" if str(key) in PLANNING_STAGE_CONTAINERS else (
                "sub" if stage else None)
            _reject_verdict_keys(child, f"{owner}.{key}", stage=child_stage)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _reject_verdict_keys(child, f"{owner}[{index}]", stage=stage)


def _asset_type(record: Mapping[str, Any]) -> str:
    identity = _mapping(record.get("identity"))
    entity_class = str(record.get("entity_class") or "")
    if entity_class == "articulated_human" or identity.get("species_id") == "human":
        return "human"
    if entity_class == "articulated_animal" or identity.get("breed_id"):
        return f"animal/{identity.get('species_id')}/{identity.get('breed_id')}"
    object_type = identity.get("object_type") or record.get("object_type")
    return f"device/{object_type}" if object_type else "device/unknown"


def _status(status: Any) -> str:
    value = str(status or "not_run")
    return value if value in STATUSES else "not_run"


def _dimension(
    status: str,
    reason: str,
    *,
    refs: Iterable[str] = (),
    facts: Mapping[str, Any] | None = None,
    missing: Iterable[str] = (),
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": _status(status),
        "reason": str(reason),
        "refs": sorted({str(ref) for ref in refs if ref}),
    }
    if facts:
        result["facts"] = deepcopy(dict(facts))
    missing_list = sorted({str(item) for item in missing if item})
    if missing_list:
        result["missing_observations"] = missing_list
    return result


# --------------------------------------------------------------------------
# input readers
# --------------------------------------------------------------------------

def _walk_instance_context(value: Any, *, path: tuple[str, ...] = ()) -> Iterable[dict[str, Any]]:
    if isinstance(value, Mapping):
        for key in ("instances", "entity_instances"):
            rows = value.get(key)
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
                episode_id = value.get("episode_id") or value.get("request_id")
                group_id = value.get("group_id")
                for row in rows:
                    if isinstance(row, Mapping) and row.get("asset_id"):
                        yield {
                            "asset_id": str(row["asset_id"]),
                            "episode_id": episode_id,
                            "group_id": group_id,
                            "instance_id": row.get("instance_id") or row.get("entity_instance_id"),
                            "role": row.get("role"),
                            "speaking": row.get("speaking"),
                            "path": "/".join(path + (str(key),)),
                        }
        for key, child in value.items():
            yield from _walk_instance_context(child, path=path + (str(key),))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            yield from _walk_instance_context(child, path=path + (str(index),))


def _contexts(config: Any) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for row in _walk_instance_context(_load(config)):
        key = (row["asset_id"], str(row.get("episode_id")), str(row.get("instance_id")))
        if key not in seen:
            result[row["asset_id"]].append(row)
            seen.add(key)
    return dict(result)


def _retained_rows(retained: Any) -> dict[str, dict[str, Any]]:
    payload = _load(retained) or {}
    result: dict[str, dict[str, Any]] = {}
    rows = _list(payload.get("rows")) if isinstance(payload, Mapping) else []
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("asset_id"):
            continue
        existing = result.setdefault(
            str(row["asset_id"]), {"asset_id": str(row["asset_id"]), "observations": []}
        )
        existing.update(
            {key: deepcopy(value) for key, value in row.items() if key != "observations"}
        )
        existing["observations"].extend(deepcopy(_list(row.get("observations"))))
    return result


def _geometry_rows(sources: Any) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Measured asset geometry keyed by asset id, from support/geometry catalogs."""
    rows: dict[str, dict[str, Any]] = {}
    refs: list[str] = []
    for source in _list(sources) if not isinstance(sources, (Mapping, str, Path)) else [sources]:
        if source is None:
            continue
        refs.append(_source_ref(source))
        payload = _mapping(_load(source))
        measured = payload.get("asset_visual_geometry_measurements")
        if not isinstance(measured, Mapping):
            measured = payload.get("measurements") if isinstance(payload.get("measurements"), Mapping) else payload
        for asset_id, value in _mapping(measured).items():
            if isinstance(value, Mapping) and value.get("bounds_min_m") is not None:
                rows[str(asset_id)] = dict(value)
    return rows, refs


def _support_surfaces(sources: Any) -> tuple[list[dict[str, Any]], list[str]]:
    surfaces: list[dict[str, Any]] = []
    refs: list[str] = []
    for source in _list(sources) if not isinstance(sources, (Mapping, str, Path)) else [sources]:
        if source is None:
            continue
        refs.append(_source_ref(source))
        payload = _mapping(_load(source))
        layout = payload.get("layout") if isinstance(payload.get("layout"), Mapping) else payload
        room_id = _mapping(payload.get("room")).get("room_id")
        for surface in _list(_mapping(layout).get("support_surfaces")):
            if isinstance(surface, Mapping) and surface.get("surface_id"):
                row = dict(surface)
                row.setdefault("room_id", room_id)
                surfaces.append(row)
    return surfaces, refs


def _placement_rows(sources: Any) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    refs: list[str] = []
    for source in _list(sources) if not isinstance(sources, (Mapping, str, Path)) else [sources]:
        if source is None:
            continue
        ref = _source_ref(source)
        refs.append(ref)
        payload = _load(source)
        for plan in _list(payload) if isinstance(payload, list) else [payload]:
            plan_map = _mapping(plan)
            episode_id = plan_map.get("episode_id")
            planned = [row for row in _list(plan_map.get("instances"))
                       if isinstance(row, Mapping) and str(row.get("status")) == "planned"]
            # The batch planner checks each new instance against the ones already
            # placed, so the first instance lists no peer even though every later
            # one was checked against it. Record who checked this row, so a pair
            # that was checked once is not read as unchecked from one side.
            checked_by: dict[str, list[str]] = {}
            for row in planned:
                peers = _list(_mapping(_mapping(row.get("clearance")).get(
                    "inter_instance_aabb")).get("checked_against"))
                for peer in peers:
                    checked_by.setdefault(str(peer), []).append(str(row.get("instance_id")))
            planned_ids = [str(row.get("instance_id")) for row in planned]
            for row in _list(plan_map.get("instances")):
                if isinstance(row, Mapping) and row.get("asset_id"):
                    entry = dict(row)
                    entry["_plan_ref"] = ref
                    entry["_episode_id"] = episode_id
                    entry["_plan_planned_instance_ids"] = planned_ids
                    entry["_plan_checked_by"] = sorted(
                        checked_by.get(str(row.get("instance_id")), []))
                    rows[str(row["asset_id"])].append(entry)
    return dict(rows), refs


def _worklist_rows(worklist: Any) -> dict[str, dict[str, Any]]:
    payload = _load(worklist) or {}
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    return {
        str(row["type"]): dict(row)
        for row in _list(rows)
        if isinstance(row, Mapping) and row.get("type")
    }


def _inventory_types(inventory: Any) -> dict[str, list[dict[str, Any]]]:
    payload = _load(inventory) or {}
    raw = payload.get("types") if isinstance(payload, Mapping) else None
    return {
        str(key): [dict(item) for item in _list(value) if isinstance(item, Mapping)]
        for key, value in _mapping(raw).items()
    }


def _obs(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [item for item in _list(row.get("observations")) if isinstance(item, Mapping)]


def _retained_refs(row: Mapping[str, Any]) -> list[str]:
    refs: list[str] = []
    for observation in _obs(row):
        for key in ("facts_path", "world_id"):
            value = observation.get(key)
            if isinstance(value, str):
                refs.append(value)
        for value in _mapping(observation.get("native_evidence_paths")).values():
            if isinstance(value, str):
                refs.append(value)
    return refs


# --------------------------------------------------------------------------
# per-dimension extraction
# --------------------------------------------------------------------------

GROUND_ATTACHMENT_SURFACES = frozenset({"floor", "ground"})
SUPPORT_SURFACE_ATTACHMENTS = frozenset({"tabletop", "wall", "ceiling", "shelf"})
DEFAULT_FLOOR_CONTACT_TOLERANCE_M = 0.03


def _registered_pose(asset: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(_mapping(_mapping(asset.get("runtime_backends")).get("habitat")).get("resting_pose"))


def support_protocol(
    asset: Mapping[str, Any],
    measured: Mapping[str, Any] | None = None,
    intent: Mapping[str, Any] | None = None,
) -> str:
    """Which support protocol this asset is qualified under.

    Decided from the asset's own registration and its measured geometry, never
    from a room or an asset id. A measured support kind is the authority, because
    a device whose resting pose was taken under the floor assumption still turns
    out to be a tabletop or wall unit once its mesh is measured. Failing that, an
    explicitly registered non-floor attachment decides. Otherwise an articulated
    source is qualified by native floor contact, and a rigid one still needs a
    support surface even when none has been measured yet.
    """
    kind = str(_mapping(measured).get("support_kind") or "").lower()
    if kind in SUPPORT_SURFACE_ATTACHMENTS:
        return "support_surface"
    pose = _registered_pose(asset)
    surface = str(pose.get("attachment_surface") or "").lower()
    assumed = bool(pose.get("attachment_surface_assumed"))
    if surface in SUPPORT_SURFACE_ATTACHMENTS and not assumed:
        return "support_surface"
    intended = str(_mapping(intent).get("support_kind") or "").lower()
    if intended in SUPPORT_SURFACE_ATTACHMENTS:
        return "support_surface"
    if intended in GROUND_ATTACHMENT_SURFACES:
        return "ground_contact"
    if str(asset.get("entity_class") or "").startswith("articulated"):
        return "ground_contact"
    return "support_surface"


def _world_room(world_id: Any, world_room_map: Mapping[str, Any]) -> str | None:
    value = world_room_map.get(str(world_id))
    return None if value is None else str(value)


def _floor_reference(room_id: Any, floor_references: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(floor_references.get(str(room_id)))


def _foot_support(measured: Mapping[str, Any] | None) -> dict[str, Any]:
    """The asset's own foot support plane, relative to its actor root.

    An actor root is not a sole. Some rigs put the root at the ground and some
    put it in the body: across the measured animals the mesh bottom sits between
    0.004 m and 0.44 m below the root, so a root height says nothing about where
    the feet are until that offset is read.
    """
    support = _mapping(_mapping(measured).get("articulated_support"))
    offset = _finite(support.get("support_plane_y_actor_m"))
    state = str(support.get("measurement") or ("measured" if offset is not None else "not_run"))
    return {
        "measurement": state,
        "support_plane_y_actor_m": offset,
        "support_plane_definition": support.get("support_plane_definition"),
        "contact_footprint_extent_m": support.get("contact_footprint_extent_m"),
        "audited_frame_count": support.get("frame_count"),
        "method": support.get("method"),
        "source_ref": support.get("source_ref"),
        "measurement_boundary": support.get("measurement_boundary"),
        "absent_reason": support.get("reason") if offset is None else None,
        "missing_inputs": _list(support.get("missing_inputs")),
    }


def _world_floor_plane(world_id: Any, room_id: Any, world_floor_planes: Mapping[str, Any]) -> dict[str, Any]:
    """The visual floor plane of the world an observation ran in.

    Feet rest on the floor that is drawn, not on the navigation mesh. In this
    room the two differ by 0.0615 m, which is more than any contact tolerance
    here, so a navmesh snap height cannot stand in for the visual floor.
    """
    for key in (world_id, room_id):
        row = _mapping(world_floor_planes.get(str(key)))
        if row:
            return {**row, "matched_on": "world_id" if str(key) == str(world_id) else "room_id"}
    return {}


def _ground_contact_observations(
    retained: Mapping[str, Any],
    floor_references: Mapping[str, Any],
    world_room_map: Mapping[str, Any],
    tolerance_m: float,
    foot: Mapping[str, Any] | None = None,
    world_floor_planes: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Where each native observation put this actor's feet, against the drawn floor.

    The chain is: the retained root height for that world and frame, plus the
    asset's own measured foot support offset, compared against the visual floor
    plane of that world. Any missing link leaves the row unmeasured; a navmesh
    height alone never decides it.
    """
    foot_row = dict(_mapping(foot))
    floor_planes = _mapping(world_floor_planes)
    rows: list[dict[str, Any]] = []
    for observation in _obs(retained):
        world_id = observation.get("world_id")
        room_id = _world_room(world_id, world_room_map)
        # the world the capture ran in is the authority, then the room. A room with
        # two storeys has two measured floors, so a single room-level height would
        # compare a capture on one level against the floor of the other.
        reference = _floor_reference(world_id, floor_references)
        floor_source = "floor_references[world_id]"
        if not reference:
            reference = _floor_reference(room_id, floor_references)
            floor_source = "floor_references[room_id]"
        floor = _finite(reference.get("floor_height_m"))
        observed_floor = _finite(observation.get("sampled_floor_height_m"))
        if observed_floor is not None:
            floor = observed_floor
            floor_source = "observation.sampled_floor_height_m"
        root_y = _finite(observation.get("first_observed_root_y_m"))
        frames = int(_finite(observation.get("native_frames")) or 0)
        row = {
            "world_id": world_id,
            "room_id": room_id,
            "native_frames": frames,
            "first_observed_root_y_m": root_y,
            "floor_height_m": floor,
            "floor_reference_source": reference.get("source") or reference.get("path"),
            "floor_reference_basis": floor_source if floor is not None else None,
            "root_max_abs_delta_m": _finite(observation.get("facts_native_root_max_abs_delta_m")),
            "facts_path": observation.get("facts_path"),
        }
        row["navmesh_reference_m"] = floor
        row["navmesh_reference_basis"] = floor_source if floor is not None else None
        if floor is not None and root_y is not None:
            row["root_above_navmesh_reference_m"] = round(root_y - floor, 9)
        plane = _world_floor_plane(world_id, room_id, floor_planes)
        visual_floor = _finite(plane.get("visual_floor_plane_y_m"))
        row["visual_floor_plane_y_m"] = visual_floor
        row["visual_floor_source"] = plane.get("source")
        row["visual_floor_matched_on"] = plane.get("matched_on")
        offset = _finite(foot_row.get("support_plane_y_actor_m"))
        row["foot_support_plane_y_actor_m"] = offset
        row["foot_support_measurement"] = foot_row.get("measurement")
        row["foot_support_method"] = foot_row.get("method")
        # A native foot-contact readback measures the sole and the floor under it
        # in the same world and the same frame. That is the strongest form of this
        # evidence, so when an observation carries it, it is used as measured and
        # nothing is composed.
        direct_sole = _finite(observation.get("sole_world_y_m"))
        direct_floor = _finite(observation.get("visual_floor_median_y_m"))
        if direct_sole is not None and direct_floor is not None:
            row.update({
                "contact_basis": "native foot contact readback: the sole and the visual floor "
                                 "under it, measured in this world and this frame",
                "sole_world_y_m": direct_sole,
                "visual_floor_plane_y_m": direct_floor,
                "visual_floor_source": observation.get("visual_floor_source") or row.get("visual_floor_source"),
                "foot_above_visual_floor_m": round(direct_sole - direct_floor, 9),
                "contact": ("on_floor" if abs(direct_sole - direct_floor) <= tolerance_m
                            else "off_floor"),
                "measured_frame_count": observation.get("measured_frame_count"),
                "foot_contact_source": observation.get("foot_contact_source"),
            })
            rows.append(row)
            continue
        missing: list[str] = []
        if root_y is None:
            missing.append("retained first_observed_root_y_m for this world")
        if offset is None:
            missing.append(
                "a measured foot support plane for this asset"
                + (f" ({foot_row.get('absent_reason')})" if foot_row.get("absent_reason") else ""))
        if visual_floor is None:
            missing.append(
                "the visual floor plane of this world; a navmesh snap height is a "
                "navigation layer and does not locate the drawn floor")
        if missing:
            row["contact"] = "not_measured"
            row["missing"] = missing
        else:
            foot_y = root_y + offset
            row["foot_world_y_m"] = round(foot_y, 9)
            row["foot_above_visual_floor_m"] = round(foot_y - visual_floor, 9)
            row["contact"] = ("on_floor" if abs(foot_y - visual_floor) <= tolerance_m
                              else "off_floor")
            row["contact_basis"] = ("retained root for this world and frame, plus the asset's "
                                    "measured foot support offset, against the world's visual floor")
        rows.append(row)
    return rows


def _ground_support_dimension(
    asset: Mapping[str, Any], contacts: Sequence[Mapping[str, Any]],
    refs: Sequence[str], tolerance_m: float,
) -> dict[str, Any]:
    pose = _registered_pose(asset)
    facts = {
        "support_protocol": "ground_contact",
        "evidence_chain": ("retained root for the world and frame + measured foot support "
                           "offset for the asset, against that world's visual floor plane"),
        "navmesh_is_not_foot_contact": True,
        "registered_attachment_surface": pose.get("attachment_surface"),
        "registered_attachment_surface_assumed": pose.get("attachment_surface_assumed"),
        "measured_from": pose.get("measured_from"),
        "floor_contact_tolerance_m": tolerance_m,
        "observations": [dict(row) for row in contacts],
        "world_ids": sorted({str(row.get("world_id")) for row in contacts if row.get("world_id")}),
    }
    surface = str(pose.get("attachment_surface") or "").lower()
    if surface not in GROUND_ATTACHMENT_SURFACES:
        return _dimension(
            "fail",
            "this asset is qualified by ground contact but its registered resting pose "
            f"names attachment surface {surface!r}",
            refs=refs, facts=facts)
    on_floor = [row for row in contacts if row.get("contact") == "on_floor"]
    off_floor = [row for row in contacts if row.get("contact") == "off_floor"]
    if off_floor and not on_floor:
        worlds = sorted({str(row.get("world_id")) for row in off_floor if row.get("world_id")})
        return _dimension(
            "fail",
            "every world with a measured foot reading puts this asset's sole off that world's "
            f"visual floor ({', '.join(worlds) or 'unnamed world'}); the media from those worlds "
            "is kept, but they are not evidence of a legal physical placement",
            refs=refs, facts=facts)
    if not on_floor:
        return _dimension(
            "not_run",
            "no native observation could be resolved into a foot-on-floor reading; a root "
            "height on the navigation mesh is not a foot contact",
            refs=refs, facts=facts,
            missing=sorted({item for row in contacts for item in _list(row.get("missing"))})
                    or ["a retained native observation for this asset"])
    return _dimension(
        "pass",
        "a native observation put this asset's measured foot support plane on the visual "
        "floor of its own world",
        refs=refs, facts=facts)


def _ground_placement_dimension(
    asset: Mapping[str, Any], contacts: Sequence[Mapping[str, Any]], refs: Sequence[str],
) -> dict[str, Any]:
    usable = [row for row in contacts
              if row.get("contact") == "on_floor" and int(row.get("native_frames") or 0) > 0]
    facts = {
        "support_protocol": "ground_contact",
        "foot_contact_required": True,
        "native_frame_counts": [int(row.get("native_frames") or 0) for row in contacts],
        "root_max_abs_delta_m": [row.get("root_max_abs_delta_m") for row in contacts],
        "world_ids": sorted({str(row.get("world_id")) for row in usable if row.get("world_id")}),
        "claim_boundary": ("the native trajectory is the placement evidence; a tabletop "
                           "resting-pose transform is not applicable to a walking source"),
    }
    if not usable:
        return _dimension(
            "not_run",
            "no native trajectory whose feet were resolved onto the visual floor of its world",
            refs=refs, facts=facts,
            missing=sorted({item for row in contacts for item in _list(row.get("missing"))})
                    or ["a retained native trajectory for this asset"])
    return _dimension(
        "pass",
        "a retained native trajectory placed this asset in its world for every frame",
        refs=refs, facts=facts)


def _ground_clearance_dimension(refs: Sequence[str]) -> dict[str, Any]:
    return _dimension(
        "not_run",
        "a ground-contact source has no planned world bounds here, so no room collision "
        "query could be run on it",
        refs=refs,
        facts={"support_protocol": "ground_contact"},
        missing=["measured mesh bounds for this asset", "room collision query at its native positions"])



def _registry_dimension(asset: Mapping[str, Any]) -> dict[str, Any]:
    asset_id = str(asset["asset_id"])
    habitat = _mapping(_mapping(asset.get("runtime_backends")).get("habitat"))
    anchors = [item for item in _list(asset.get("emitter_anchors")) if isinstance(item, Mapping)]
    default_anchor = asset.get("default_emitter_anchor_id")
    matched = [item for item in anchors if item.get("anchor_id") == default_anchor]
    missing = []
    if not _mapping(asset.get("identity")):
        missing.append("identity")
    if not _mapping(asset.get("geometry")).get("source_mesh_uri"):
        missing.append("geometry.source_mesh_uri")
    if not habitat:
        missing.append("runtime_backends.habitat")
    if not _mapping(habitat.get("resting_pose")):
        missing.append("runtime_backends.habitat.resting_pose")
    if len(matched) != 1:
        missing.append("default_emitter_anchor_id resolving to exactly one emitter anchor")
    facts = {
        "entity_class": asset.get("entity_class"),
        "revision": asset.get("revision"),
        "emitter_anchor_count": len(anchors),
        "declared_backends": sorted(_mapping(asset.get("runtime_backends"))),
        "source_mesh_uri": _mapping(asset.get("geometry")).get("source_mesh_uri"),
    }
    if missing:
        return _dimension(
            "fail",
            "registered declaration is incomplete",
            refs=[f"registry:{asset_id}"],
            facts=facts,
            missing=missing,
        )
    return _dimension(
        "pass",
        "registered identity, mesh URI, habitat backend, resting pose and one resolvable emitter anchor are all present",
        refs=[f"registry:{asset_id}"],
        facts=facts,
    )


def _geometry_dimension(
    asset: Mapping[str, Any],
    measured: Mapping[str, Any] | None,
    refs: Sequence[str],
    *,
    height_tolerance_m: float,
    footprint_tolerance_m: float,
) -> dict[str, Any]:
    registry_pose = _mapping(_mapping(_mapping(asset.get("runtime_backends")).get("habitat")).get("resting_pose"))
    if not measured:
        return _dimension(
            "not_run",
            "no measured mesh bounds for this asset; the registered mesh URI alone is not a scale observation",
            refs=[str(_mapping(asset.get("geometry")).get("source_mesh_uri") or "")],
            missing=["asset_visual_geometry_measurements[asset_id].bounds_min_m/bounds_max_m"],
        )
    lo = _vector(measured.get("bounds_min_m"))
    hi = _vector(measured.get("bounds_max_m"))
    normal = _vector(measured.get("plane_normal_m"))
    footprint = _vector(measured.get("footprint_extent_m"), 2)
    missing = []
    if lo is None or hi is None:
        missing.append("finite bounds_min_m/bounds_max_m")
    if normal is None:
        missing.append("finite plane_normal_m")
    if footprint is None:
        missing.append("finite footprint_extent_m")
    if missing:
        return _dimension("not_run", "measured geometry row is incomplete", refs=refs, missing=missing)
    assert lo is not None and hi is not None and footprint is not None
    if any(a >= b for a, b in zip(lo, hi)):
        return _dimension("fail", "measured bounds are not strictly ordered", refs=refs,
                          facts={"bounds_min_m": lo, "bounds_max_m": hi})
    measured_height = hi[1] - lo[1]
    registry_height = _finite(registry_pose.get("height_m"))
    registry_footprint = _vector(registry_pose.get("footprint_extent_m"), 2)
    facts = {
        "bounds_min_m": lo,
        "bounds_max_m": hi,
        "measured_extent_m": [round(b - a, 6) for a, b in zip(lo, hi)],
        "measured_height_m": round(measured_height, 6),
        "registry_height_m": registry_height,
        "measured_footprint_extent_m": footprint,
        "registry_footprint_extent_m": registry_footprint,
        "measured_from": measured.get("measured_from"),
        "source_ref": measured.get("source_ref"),
        "support_kind": measured.get("support_kind"),
    }
    disagreements = []
    if registry_height is not None and abs(registry_height - measured_height) > height_tolerance_m:
        disagreements.append(
            f"height differs by {abs(registry_height - measured_height):.4f} m (tolerance {height_tolerance_m} m)"
        )
    if registry_footprint is not None and any(
        abs(a - b) > footprint_tolerance_m for a, b in zip(sorted(registry_footprint), sorted(footprint))
    ):
        disagreements.append(f"footprint differs by more than {footprint_tolerance_m} m")
    if disagreements:
        facts["disagreements"] = disagreements
        return _dimension("fail", "measured mesh disagrees with the registered resting pose", refs=refs, facts=facts)
    return _dimension(
        "pass",
        "mesh bounds, footprint and plane normal were measured from the asset file and agree with the registered resting pose",
        refs=refs,
        facts=facts,
    )


def _support_dimension(
    asset: Mapping[str, Any],
    measured: Mapping[str, Any] | None,
    surfaces: Sequence[Mapping[str, Any]],
    refs: Sequence[str],
    intent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Match this asset against a measured support surface.

    Three things are kept apart. The *measured* support kind is an observation of
    the asset's own geometry. A *registry* attachment surface is a declaration,
    and where the registry marks it assumed it is not even that. A *declared
    placement intent* is an episode saying which kind of surface it means to use;
    it selects candidates and is verified against them, and it is never written
    back as a measurement.
    """
    pose = _registered_pose(asset)
    registry_surface = str(pose.get("attachment_surface") or "").lower()
    registry_assumed = bool(pose.get("attachment_surface_assumed"))
    intent_map = _mapping(intent)
    intended = str(intent_map.get("support_kind") or "").lower()
    measured_kind = str((measured or {}).get("support_kind") or "").lower()
    basis = str((measured or {}).get("support_kind_basis") or "")
    # A registry declaration is binding when it names a mounting surface outright.
    # A bare floor value with no assumed flag is the ambiguous default, and an
    # episode may declare which kind of surface it means there.
    registry_binding = (registry_surface in SUPPORT_SURFACE_ATTACHMENTS
                        and not registry_assumed)
    if registry_binding and intended and intended != registry_surface:
        return _dimension(
            "fail",
            f"the episode declares a {intended!r} placement intent while the registry declares "
            f"this asset mounts on {registry_surface!r}; a registered attachment is not overridden",
            refs=refs,
            facts={"declared_placement_intent": intended,
                   "registry_attachment_surface": registry_surface,
                   "registry_attachment_surface_assumed": registry_assumed})
    kind = measured_kind or (registry_surface if registry_binding else "") or intended
    kind_source = ("measured_support_kind" if measured_kind
                   else "registry_declared_attachment" if registry_binding
                   else "declared_placement_intent" if intended else None)
    if not kind:
        return _dimension(
            "not_run",
            "neither a measured support kind nor a declared placement intent names the kind of "
            "surface this asset belongs on",
            refs=refs,
            facts={"registry_attachment_surface": registry_surface,
                   "registry_attachment_surface_assumed": registry_assumed,
                   "measured_support_kind": measured_kind or None,
                   "measured_support_kind_basis": basis or None,
                   "declared_placement_intent": intended or None},
            missing=["asset_visual_geometry_measurements[asset_id].support_kind",
                     "or an episode declared_placement_intent.support_kind"],
        )
    matching = [row for row in surfaces if str(row.get("surface_kind")) == kind]
    facts = {
        "support_kind_used": kind,
        "support_kind_source": kind_source,
        "measured_support_kind": measured_kind or None,
        "measured_support_kind_basis": basis or None,
        "declared_placement_intent": intended or None,
        "registry_attachment_surface": registry_surface or None,
        "registry_attachment_surface_assumed": registry_assumed,
        "matching_surface_ids": [str(row.get("surface_id")) for row in matching],
        "matching_surface_count": len(matching),
        "available_surface_kinds": sorted({str(row.get("surface_kind")) for row in surfaces}),
        "rooms": sorted({str(row.get("room_id")) for row in matching if row.get("room_id")}),
    }
    if not surfaces:
        return _dimension("not_run", "no support surface catalog was supplied", refs=refs,
                          facts=facts, missing=["support catalog layout.support_surfaces"])
    if not matching:
        # Absent is not false. No measured surface of this kind means the surface
        # was never measured, which is missing evidence; only an observed wrong
        # support or placement is a failure.
        return _dimension(
            "not_run",
            f"no measured support surface of kind {kind!r} exists in the supplied catalog, "
            "so this asset's support has not been observed; the asset is not moved onto a "
            "surface of another kind to fill the gap",
            refs=refs,
            facts=facts,
            missing=[f"a measured {kind} support surface in the target room"],
        )
    if kind_source == "declared_placement_intent":
        return _dimension(
            "pass",
            f"a measured {kind} surface exists for the episode's declared {kind} placement "
            "intent; the intent selected the candidate and is not recorded as a measurement",
            refs=refs,
            facts=facts,
        )
    return _dimension(
        "pass",
        f"a measured {kind} surface exists in the supplied catalog for this asset",
        refs=refs,
        facts=facts,
    )


def _placement_dimension(
    asset: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    measured: Mapping[str, Any] | None,
    floor_reference_m: float | None,
    refs: Sequence[str],
    intent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not rows:
        return _dimension(
            "not_run",
            "no planned placement row for this asset",
            refs=refs,
            missing=["source placement plan row with status=planned"],
        )
    planned = [row for row in rows if str(row.get("status")) == "planned"]
    rejected = [row for row in rows if str(row.get("status")) == "rejected"]
    if not planned:
        return _dimension(
            "fail",
            "every placement attempt for this asset was rejected",
            refs=refs + tuple(str(row.get("_plan_ref")) for row in rejected),
            facts={"rejections": [
                {"instance_id": row.get("instance_id"),
                 "episode_id": row.get("_episode_id"),
                 "reason": _mapping(row.get("reason")).get("code"),
                 "detail": _mapping(row.get("reason")).get("reason")}
                for row in rejected]},
        )
    row = planned[0]
    support = _mapping(row.get("support_identity"))
    transform = _mapping(row.get("root_transform"))
    matrix = _list(transform.get("matrix_row_major"))
    translation = _vector(transform.get("translation_m"))
    quaternion = _vector(transform.get("rotation_xyzw"), 4)
    bounds_min = _vector(_mapping(row.get("asset_bounds")).get("world_aabb_min_m"))
    problems = []
    if len(matrix) != 16 or any(_finite(v) is None for v in matrix):
        problems.append("root_transform.matrix_row_major is not 16 finite numbers")
    if translation is None:
        problems.append("root_transform.translation_m is not a finite 3-vector")
    if quaternion is None or abs(math.sqrt(sum(v * v for v in quaternion)) - 1.0) > 1e-6:
        problems.append("root_transform.rotation_xyzw is not a unit quaternion")
    measured_kind = str((measured or {}).get("support_kind") or "")
    placed_kind = str(support.get("surface_kind") or "")
    intended_kind = str(_mapping(intent).get("support_kind") or "").lower()
    if measured_kind and placed_kind != measured_kind:
        problems.append(
            f"placed on a {placed_kind!r} surface while the asset measures as {measured_kind!r}"
        )
    elif not measured_kind and intended_kind and placed_kind != intended_kind:
        problems.append(
            f"placed on a {placed_kind!r} surface while the episode declares a "
            f"{intended_kind!r} placement intent"
        )
    facts = {
        "instance_id": row.get("instance_id"),
        "episode_id": row.get("_episode_id"),
        "plan_ref": row.get("_plan_ref"),
        "support_surface_id": support.get("surface_id"),
        "support_surface_kind": support.get("surface_kind"),
        "measured_support_kind": measured_kind or None,
        "declared_placement_intent": intended_kind or None,
        "verified_against": ("measured_support_kind" if measured_kind
                             else "declared_placement_intent" if intended_kind else None),
        "candidate": {key: _mapping(row.get("candidate")).get(key)
                      for key in ("index", "count", "selection_mode")},
        "root_translation_m": translation,
        "root_rotation_xyzw": quaternion,
        "asset_bounds_min_m": bounds_min,
        "planned_rows": len(planned),
        "rejected_rows": len(rejected),
    }
    if floor_reference_m is not None and translation is not None:
        facts["floor_reference_m"] = floor_reference_m
        facts["root_height_above_floor_m"] = round(translation[1] - floor_reference_m, 6)
        if bounds_min is not None:
            facts["bounds_min_height_above_floor_m"] = round(bounds_min[1] - floor_reference_m, 6)
            if bounds_min[1] < floor_reference_m - DEFAULT_SUPPORT_PLANE_TOLERANCE_M:
                problems.append("planned world bounds reach below the measured floor")
    if problems:
        facts["problems"] = problems
        return _dimension("fail", "planned placement is not legal", refs=refs, facts=facts)
    return _dimension(
        "pass",
        "a real planner produced a finite rigid root transform on a support surface of the asset's own measured kind",
        refs=refs,
        facts=facts,
    )


#: A box query and a mesh query do not license the same conclusion.  A world AABB
#: that contains the asset mesh and is disjoint from the room proves the mesh is
#: disjoint too.  A box that overlaps proves nothing about the mesh, because the
#: box is larger than the mesh it encloses.
BOX_COLLISION_METHODS = frozenset({"world_aabb_shrunk", "box_bounds", None, ""})
MESH_COLLISION_METHODS = frozenset({"asset_mesh_narrowphase", "asset_mesh_triangles"})
MESH_BOUNDS_SOURCES = frozenset({"request.asset_geometry", "measured_asset_mesh"})


def _collision_evidence(row: Mapping[str, Any], bounds: Mapping[str, Any]) -> dict[str, Any]:
    """Classify what a room-collision observation can and cannot establish."""
    method = _mapping(row.get("room_collision")).get("method")
    method_name = None if method is None else str(method)
    bounds_source = str(bounds.get("source") or "")
    contains_mesh = bounds_source in MESH_BOUNDS_SOURCES
    return {
        "method": method_name,
        "is_mesh_narrowphase": method_name in MESH_COLLISION_METHODS,
        "bounds_source": bounds_source or None,
        "world_bounds_contain_asset_mesh": contains_mesh,
        "containment_basis": (
            "the world AABB was built from the measured asset mesh under the planned rigid "
            "transform, so the mesh lies inside the box"
            if contains_mesh else
            "the world AABB was derived conservatively, so it is not known to be a tight "
            "container of the asset mesh"),
    }


def _clearance_dimension(rows: Sequence[Mapping[str, Any]], refs: Sequence[str]) -> dict[str, Any]:
    planned = [row for row in rows if str(row.get("status")) == "planned"]
    if not planned:
        return _dimension("not_run", "no planned placement row to check clearance on", refs=refs,
                          missing=["source placement plan row with status=planned"])
    row = planned[0]
    clearance = _mapping(row.get("clearance"))
    inter = _mapping(clearance.get("inter_instance_aabb"))
    room = _mapping(clearance.get("room_collision"))
    evidence = _collision_evidence(clearance, _mapping(row.get("asset_bounds")))
    checked = _list(inter.get("checked_against"))
    checked_by = _list(row.get("_plan_checked_by"))
    peers = [value for value in _list(row.get("_plan_planned_instance_ids"))
             if str(value) != str(row.get("instance_id"))]
    mutually = sorted({str(value) for value in checked} | {str(value) for value in checked_by})
    conflicts = _list(inter.get("overlap_conflicts"))
    facts = {
        "inter_instance_status": inter.get("status"),
        "checked_against": checked,
        "checked_by": checked_by,
        "mutually_checked_instance_ids": mutually,
        "planned_peer_instance_ids": peers,
        "overlap_conflict_count": len(conflicts),
        "room_collision_status": room.get("status"),
        "room_collision_reason": room.get("reason"),
        "room_collision_evidence": evidence,
        "room_collision_tolerance_m": _mapping(room.get("geometry")).get("penetration_tolerance_m"),
        "clearance_status": clearance.get("status"),
    }
    missing = []
    if conflicts:
        facts["overlap_conflicts"] = deepcopy(conflicts)
        return _dimension("fail", "the planned placement overlaps another planned instance", refs=refs, facts=facts)
    if _status(room.get("status")) == "fail":
        facts["room_collision_geometry"] = deepcopy(_mapping(room.get("geometry")))
        if evidence["is_mesh_narrowphase"]:
            return _dimension(
                "fail",
                "the asset-mesh collision query found room geometry inside this placement",
                refs=refs, facts=facts)
        # A world AABB overlapping the room does not establish that the asset mesh
        # does. The box is a conservative container, so the overlap may lie in the
        # empty corners of the box. This stays an open question until a mesh
        # narrowphase runs; it is neither a pass nor a proven failure.
        facts["pending_precise_detection"] = True
        return _dimension(
            "not_run",
            "a world AABB overlap was observed, which bounds the question but cannot "
            "establish that the asset mesh interpenetrates; an asset-mesh narrowphase "
            "has to decide it",
            refs=refs, facts=facts,
            missing=["asset-mesh narrowphase room collision for this placement"])
    if peers and sorted(str(value) for value in peers) != mutually:
        missing.append("inter-instance AABB check against every planned peer instance")
    elif not peers and not mutually:
        missing.append("inter-instance AABB check against at least one peer instance")
    if _status(room.get("status")) != "pass":
        missing.append("room-wide collision query")
    elif not (evidence["is_mesh_narrowphase"] or evidence["world_bounds_contain_asset_mesh"]):
        # A disjoint box only clears the mesh when the box is known to contain it.
        facts["pending_precise_detection"] = True
        missing.append(
            "either an asset-mesh narrowphase, or world bounds built from the measured "
            "asset mesh so that a disjoint box also clears the mesh")
    if missing:
        return _dimension(
            "not_run",
            "clearance is only partly observed; the remaining checks were never run",
            refs=refs,
            facts=facts,
            missing=missing,
        )
    reason = (
        "no inter-instance overlap, and an asset-mesh collision query found the placement clear"
        if evidence["is_mesh_narrowphase"] else
        "no inter-instance overlap, and a world AABB built from the measured asset mesh is "
        "disjoint from the room, which clears the mesh it contains")
    return _dimension("pass", reason, refs=refs, facts=facts)


def _emitter_dimension(
    asset: Mapping[str, Any],
    retained: Mapping[str, Any],
    placement_rows: Sequence[Mapping[str, Any]],
    refs: Sequence[str],
) -> dict[str, Any]:
    anchor_id = asset.get("default_emitter_anchor_id")
    anchors = [item for item in _list(asset.get("emitter_anchors"))
               if isinstance(item, Mapping) and item.get("anchor_id") == anchor_id]
    observations = _obs(retained)
    emitter_frames = max((int(_finite(item.get("native_emitter_frames")) or 0) for item in observations), default=0)
    planned = [row for row in placement_rows if str(row.get("status")) == "planned"]
    planned_emitter = None
    if planned:
        planned_emitter = _vector(_mapping(planned[0].get("emitter_transform")).get("position_m"))
    facts = {
        "default_emitter_anchor_id": anchor_id,
        "world_ids": sorted({str(item.get("world_id")) for item in observations if item.get("world_id")}),
        "registry_anchor_resolved": len(anchors) == 1,
        "registry_anchor_offset_m": _vector(_mapping(anchors[0]).get("offset_m")) if anchors else None,
        "native_emitter_frames": emitter_frames,
        "planned_emitter_position_m": planned_emitter,
    }
    missing = []
    if len(anchors) != 1:
        return _dimension("fail", "the registered default emitter anchor does not resolve", refs=refs, facts=facts)
    if emitter_frames <= 0:
        missing.append("native emitter frames in a retained readback")
    if missing:
        return _dimension(
            "not_run",
            "the emitter anchor is registered and, where planned, placed, but no native emitter frame was observed",
            refs=refs,
            facts=facts,
            missing=missing,
        )
    return _dimension(
        "pass",
        "the registered emitter anchor resolves and native emitter frames were observed",
        refs=refs,
        facts=facts,
    )


def _visibility_dimension(retained: Mapping[str, Any], refs: Sequence[str]) -> dict[str, Any]:
    observations = _obs(retained)
    clear = sum(int(_finite(_mapping(item.get("visibility_state_counts")).get("visible_clear")) or 0)
                for item in observations)
    occluded = sum(int(_finite(_mapping(item.get("visibility_state_counts")).get("visible_occluded")) or 0)
                   for item in observations)
    out_of_view = sum(int(_finite(_mapping(item.get("visibility_state_counts")).get("out_of_view")) or 0)
                      for item in observations)
    fully_occluded = sum(int(_finite(_mapping(item.get("visibility_state_counts")).get("fully_occluded")) or 0)
                         for item in observations)
    pixels = max((int(_finite(item.get("max_visible_pixels")) or 0) for item in observations), default=0)
    facts = {
        "observation_count": len(observations),
        "world_ids": sorted({str(item.get("world_id")) for item in observations if item.get("world_id")}),
        "max_visible_pixels": pixels,
        "visible_clear_frames": clear,
        "visible_occluded_frames": occluded,
        "out_of_view_frames": out_of_view,
        "fully_occluded_frames": fully_occluded,
        "claim_boundary": "nonzero target pixels only; this is not a claim that any frame was unoccluded",
        "requirement": ("at least one retained observation of this asset shows target pixels; a "
                        "representative needs one visible appearance somewhere, not an appearance "
                        "in every world, and an actor that is deliberately off screen in some "
                        "world does not count against it"),
    }
    if pixels <= 0 or (clear + occluded) <= 0:
        return _dimension(
            "not_run",
            "no native capture reported target pixels for this asset",
            refs=refs,
            facts=facts,
            missing=["native pixel visibility readback with max_visible_pixels > 0"],
        )
    return _dimension(
        "pass",
        "a native capture reported nonzero target pixels for this asset",
        refs=refs,
        facts=facts,
    )


def _sound_dimension(
    retained: Mapping[str, Any],
    refs: Sequence[str],
    media_probe: Callable[[str], Mapping[str, Any]] | None,
) -> dict[str, Any]:
    stems: list[dict[str, Any]] = []
    for observation in _obs(retained):
        for event in _list(observation.get("audio_events")):
            stem = _mapping(_mapping(event).get("stem"))
            if not stem.get("path"):
                continue
            row = {
                "path": str(stem["path"]),
                "declared_frames": _finite(stem.get("frames")),
                "declared_channels": _finite(stem.get("channels")),
                "declared_sample_rate_hz": _finite(stem.get("sample_rate_hz")),
                "declared_finite": stem.get("finite"),
                "declared_peak_abs": _finite(stem.get("peak_abs")),
                "sound_asset_id": _mapping(event).get("sound_asset_id"),
            }
            if media_probe is not None:
                try:
                    row["probe"] = dict(media_probe(row["path"]))
                except Exception as exc:  # a probe failure is evidence, not a crash
                    row["probe"] = {"error": f"{type(exc).__name__}: {exc}"}
            stems.append(row)
    facts = {"stem_count": len(stems), "stems": stems[:8], "media_probe_run": media_probe is not None,
             "world_ids": sorted({str(item.get("world_id")) for item in _obs(retained) if item.get("world_id")})}
    if not stems:
        return _dimension("not_run", "no source PCM stem was recorded for this asset", refs=refs,
                          facts=facts, missing=["retained audio_events[].stem"])
    def usable(row: Mapping[str, Any]) -> bool:
        probe = _mapping(row.get("probe"))
        if media_probe is not None:
            if not probe.get("exists"):
                return False
            frames = _finite(probe.get("frames")) or 0
            channels = _finite(probe.get("channels")) or 0
            peak = _finite(probe.get("peak_abs"))
            return bool(probe.get("finite", True)) and frames > 0 and channels >= 1 and (peak is None or peak > 0)
        frames = row.get("declared_frames") or 0
        channels = row.get("declared_channels") or 0
        peak = row.get("declared_peak_abs")
        return bool(row.get("declared_finite")) and frames > 0 and channels >= 1 and (peak is None or peak > 0)
    good = [row for row in stems if usable(row)]
    facts["usable_stem_count"] = len(good)
    if not good:
        return _dimension(
            "fail" if media_probe is not None else "not_run",
            "no recorded stem is a finite nonzero multi-sample PCM file",
            refs=refs,
            facts=facts,
            missing=[] if media_probe is not None else ["media probe of the recorded stem path"],
        )
    return _dimension(
        "pass",
        "a finite nonzero PCM stem was observed for this asset",
        refs=refs,
        facts=facts,
    )


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def qualification_evidence_interface() -> dict[str, Any]:
    """Describe the inputs an override provider may supply.

    Every entry is an observation source.  There is deliberately no way to pass
    a status: :func:`build_qualification_matrix` refuses any evidence input that
    carries one of :data:`FORBIDDEN_EVIDENCE_KEYS`.
    """
    return {
        "schema": "avengine_source_asset_qualification_evidence_interface_v1",
        "builder": "avengine.dataset.source_asset_qualification.build_qualification_matrix",
        "required_dimensions": list(REQUIRED_DIMENSIONS),
        "support_protocols": {
            "support_surface": "measured support surface plus a planner transform; used for assets "
                               "whose resting pose was measured against a tabletop, wall or ceiling",
            "ground_contact": "registered floor attachment plus a retained native trajectory whose "
                              "root sits on that world's measured floor; used for articulated sources",
        },
        "world_compatibility": "an asset passes only when the dimensions that name a world share one",
        "remaining_work": ("every asset row carries remaining_work, and the matrix carries "
                           "work_buckets; both are derived from the current dimension states, so a "
                           "consumer reads the schedule from the matrix and never from a copied "
                           "coordination marker"),
        "statuses": sorted(STATUSES),
        "inputs": {
            "registry": "runtime source asset registry (path or mapping) with assets[]",
            "source_type_inventory": "fine-type inventory with types{} and the expected counts",
            "asset_worklist": "acceptance worklist rows keyed by type",
            "config": "production spec whose episodes bind assets to instances",
            "geometry_measurements": "one or more catalogs carrying asset_visual_geometry_measurements",
            "support_catalogs": "one or more support surface catalogs carrying layout.support_surfaces",
            "placement_plans": "one or more avengine_source_placement_plan_v1 outputs",
            "retained_readback": "retained native readback with rows[].observations[]",
            "floor_reference_m": "measured room floor height for the placement check",
            "floor_references": "{room_id: {floor_height_m, source}} measured floors, for ground-contact assets",
            "world_room_map": "{world_id: room_id} so a retained observation can find its room's floor",
            "world_floor_planes": ("{world_id or room_id: {visual_floor_plane_y_m, source}} the drawn "
                                   "floor a foot contact is judged against; a navmesh snap height is "
                                   "not accepted in its place"),
            "placement_intents": ("{asset_id: {support_kind, candidate_surface_ids, source}} an "
                                  "episode's declared placement intent; it selects candidates and is "
                                  "verified, and is never recorded as a measurement"),
            "floor_contact_tolerance_m": "how far a native root may sit from the measured floor and still count as contact",
            "media_probe": "callable(path) -> {exists, frames, channels, sample_rate_hz, peak_abs, finite}",
            "provider_evidence": ("{asset_id: {geometry_measurement, placement_rows[], "
                                  "retained_observations[]}} -- the override provider hook; "
                                  "observations only, verdict keys are refused"),
        },
        "forbidden_evidence_keys": sorted(FORBIDDEN_EVIDENCE_KEYS),
        "planning_stage_status_allowed_in": sorted(PLANNING_STAGE_CONTAINERS),
        "planning_stage_status_values": sorted(PLANNING_STAGE_STATUS_VALUES),
        "planning_subcheck_status_values": sorted(PLANNING_SUBCHECK_STATUS_VALUES),
        "planning_disclaimer_keys": sorted(PLANNING_DISCLAIMER_KEYS),
        "claim_boundary": (
            "a provider may only widen what is observed; the module derives every "
            "dimension status itself and never accepts a caller-written verdict"
        ),
    }


def build_qualification_matrix(
    *,
    registry: Any,
    source_type_inventory: Any,
    asset_worklist: Any,
    config: Any,
    geometry_measurements: Any | None = None,
    support_catalogs: Any | None = None,
    placement_plans: Any | None = None,
    retained_readback: Any | None = None,
    provider_evidence: Mapping[str, Any] | None = None,
    floor_reference_m: float | None = None,
    floor_references: Mapping[str, Any] | None = None,
    world_room_map: Mapping[str, Any] | None = None,
    world_floor_planes: Mapping[str, Any] | None = None,
    placement_intents: Mapping[str, Any] | None = None,
    floor_contact_tolerance_m: float = DEFAULT_FLOOR_CONTACT_TOLERANCE_M,
    media_probe: Callable[[str], Mapping[str, Any]] | None = None,
    height_tolerance_m: float = DEFAULT_HEIGHT_TOLERANCE_M,
    footprint_tolerance_m: float = DEFAULT_FOOTPRINT_TOLERANCE_M,
) -> dict[str, Any]:
    """Build the complete fine-type/asset matrix for ordinary program use."""
    _reject_verdict_keys(provider_evidence, "provider_evidence")

    registry_payload = _load(registry) or {}
    records = [
        dict(item)
        for item in _list(_mapping(registry_payload).get("assets"))
        if isinstance(item, Mapping) and item.get("asset_id")
    ]
    types = _inventory_types(source_type_inventory)
    work = _worklist_rows(asset_worklist)
    contexts = _contexts(config)
    retained_by_asset = _retained_rows(retained_readback)
    geometry_by_asset, geometry_refs = _geometry_rows(geometry_measurements)
    surfaces, support_refs = _support_surfaces(support_catalogs)
    placements_by_asset, placement_refs = _placement_rows(placement_plans)
    provider_refs = _merge_provider_evidence(
        provider_evidence, geometry_by_asset, placements_by_asset, retained_by_asset)
    floor_reference_rows = {str(key): _mapping(value)
                            for key, value in _mapping(_load(floor_references)).items()}
    world_rooms = {str(key): value for key, value in _mapping(_load(world_room_map)).items()}
    floor_planes = {str(key): _mapping(value)
                    for key, value in _mapping(_load(world_floor_planes)).items()}
    intents = {str(key): _mapping(value)
               for key, value in _mapping(_load(placement_intents)).items()}

    asset_rows: list[dict[str, Any]] = []
    asset_detail_refs: dict[str, str] = {}
    for index, asset in enumerate(records, start=1):
        asset_id = str(asset["asset_id"])
        type_name = _asset_type(asset)
        measured = geometry_by_asset.get(asset_id)
        rows = placements_by_asset.get(asset_id, [])
        retained = retained_by_asset.get(asset_id, {})
        retained_ref = _retained_refs(retained)
        row_refs = tuple(str(row.get("_plan_ref")) for row in rows) or tuple(placement_refs)
        intent = intents.get(asset_id) or {}
        protocol = support_protocol(asset, measured, intent)
        if protocol == "ground_contact":
            contacts = _ground_contact_observations(
                retained, floor_reference_rows, world_rooms, floor_contact_tolerance_m,
                foot=_foot_support(measured), world_floor_planes=floor_planes)
            support_dim = _ground_support_dimension(
                asset, contacts, retained_ref, floor_contact_tolerance_m)
            placement_dim = _ground_placement_dimension(asset, contacts, retained_ref)
            clearance_dim = _ground_clearance_dimension(retained_ref)
        else:
            support_dim = _support_dimension(asset, measured, surfaces, support_refs, intent)
            placement_dim = _placement_dimension(
                asset, rows, measured, floor_reference_m, row_refs, intent)
            clearance_dim = _clearance_dimension(rows, row_refs)
        dims = {
            "registry": _registry_dimension(asset),
            "geometry_scale": _geometry_dimension(
                asset, measured, geometry_refs,
                height_tolerance_m=height_tolerance_m,
                footprint_tolerance_m=footprint_tolerance_m),
            "support": support_dim,
            "placement": placement_dim,
            "clearance": clearance_dim,
            "emitter": _emitter_dimension(asset, retained, rows, retained_ref),
            "visibility": _visibility_dimension(retained, retained_ref),
            "sound_pcm": _sound_dimension(retained, retained_ref, media_probe),
        }
        worlds = world_evidence(dims)
        work = derive_remaining_work({"dimensions": dims})
        status, missing = _overall(dims)
        if status == "pass" and not worlds["compatible"]:
            status = "not_run"
            missing = list(worlds["conflicting_dimensions"])
        asset_detail_id = f"ASSET-{index:03d}"
        asset_detail_refs[asset_id] = asset_detail_id
        asset_rows.append({
            "asset_detail_id": asset_detail_id,
            "asset_id": asset_id,
            "type": type_name,
            "entity_class": asset.get("entity_class"),
            "revision": asset.get("revision"),
            "worklist_row_present": type_name in work,
            "contexts": contexts.get(asset_id, []),
            "dimensions": dims,
            "support_protocol": protocol,
            "world_evidence": worlds,
            "remaining_work": work,
            "status": status,
            "missing_dimensions": missing,
            "failed_dimensions": [key for key in REQUIRED_DIMENSIONS
                                  if _status(dims[key]["status"]) == "fail"],
            "claim_boundary": "asset acceptance requires every required dimension to pass",
        })

    asset_by_id = {row["asset_id"]: row for row in asset_rows}
    type_rows: list[dict[str, Any]] = []
    for type_name in sorted(types):
        candidates = [str(item.get("asset_id")) for item in types[type_name] if item.get("asset_id")]
        selected = [asset_id for asset_id in candidates if contexts.get(asset_id)]
        if not selected:
            selected = candidates[:1]
        selected_rows = [asset_by_id[asset_id] for asset_id in selected if asset_id in asset_by_id]
        candidate_rows = [asset_by_id[asset_id] for asset_id in candidates if asset_id in asset_by_id]
        passing = [row["asset_id"] for row in candidate_rows if row["status"] == "pass"]
        statuses = [row["status"] for row in selected_rows]
        if passing:
            type_status = "pass"
        elif "fail" in statuses and all(value == "fail" for value in statuses):
            type_status = "fail"
        else:
            type_status = "not_run"
        type_rows.append({
            "type": type_name,
            "candidate_asset_ids": candidates,
            "candidate_asset_count": len(candidates),
            "selected_representative_asset_ids": selected,
            "asset_detail_refs": [asset_detail_refs[a] for a in selected if a in asset_detail_refs],
            "qualified_asset_ids": passing,
            "status": type_status,
            "missing_assets": [row["asset_id"] for row in selected_rows if row["status"] != "pass"],
            "blocking_dimensions": sorted({
                dimension
                for row in candidate_rows
                for dimension in row["missing_dimensions"]
            }) if not passing else [],
            "claim_boundary": "registry/worklist presence is not acceptance; one asset must pass every dimension",
        })

    inventory_payload = _mapping(_load(source_type_inventory))
    expected_types = int(inventory_payload.get("fine_type_count") or len(types))
    expected_assets = int(inventory_payload.get("asset_count") or len(asset_rows))
    overall = (
        "pass"
        if (len(type_rows) == expected_types
            and len(asset_rows) == expected_assets
            and type_rows
            and all(row["status"] == "pass" for row in type_rows))
        else "not_run"
    )
    matrix = {
        "schema": SCHEMA,
        "status": overall,
        "counts": {
            "type_count": len(type_rows),
            "asset_count": len(asset_rows),
            "expected_type_count": expected_types,
            "expected_asset_count": expected_assets,
            "types_pass": sum(1 for row in type_rows if row["status"] == "pass"),
            "types_fail": sum(1 for row in type_rows if row["status"] == "fail"),
            "assets_pass": sum(1 for row in asset_rows if row["status"] == "pass"),
            "assets_fail": sum(1 for row in asset_rows if row["status"] == "fail"),
        },
        "dimension_totals": {
            dimension: {
                state: sum(1 for row in asset_rows if _status(row["dimensions"][dimension]["status"]) == state)
                for state in sorted(STATUSES)
            }
            for dimension in REQUIRED_DIMENSIONS
        },
        "work_buckets": _work_buckets(asset_rows, type_rows),
        "type_status_rows": type_rows,
        "asset_status_rows": asset_rows,
        "qualification_dimensions": list(REQUIRED_DIMENSIONS),
        "sources": {
            "registry": _source_ref(registry),
            "source_type_inventory": _source_ref(source_type_inventory),
            "asset_worklist": _source_ref(asset_worklist),
            "config": _source_ref(config),
            "geometry_measurements": geometry_refs,
            "support_catalogs": support_refs,
            "placement_plans": placement_refs,
            "retained_readback": _source_ref(retained_readback) if retained_readback is not None else None,
            "provider_evidence_assets": provider_refs,
            "floor_reference_m": floor_reference_m,
            "floor_references": sorted(floor_reference_rows),
            "world_room_map": dict(world_rooms),
            "world_floor_planes": sorted(floor_planes),
            "placement_intents": sorted(intents),
            "floor_contact_tolerance_m": floor_contact_tolerance_m,
            "media_probe_run": media_probe is not None,
        },
        "claim_boundary": (
            "This matrix classifies observed evidence; it does not render, run native/RLR, "
            "or authorize dataset admission, and it never accepts a caller-written status."
        ),
        "api": qualification_evidence_interface(),
    }
    validate_qualification_matrix(matrix)
    return matrix


def _merge_provider_evidence(
    provider_evidence: Mapping[str, Any] | None,
    geometry_by_asset: dict[str, dict[str, Any]],
    placements_by_asset: dict[str, list[dict[str, Any]]],
    retained_by_asset: dict[str, dict[str, Any]],
) -> list[str]:
    """Fold an override provider's extra observations into the evidence pools.

    The provider supplies observations only.  Its payload has already been
    checked for verdict keys, so it can add a measurement, a placement row or a
    native observation, but it cannot declare a dimension passed.
    """
    touched: list[str] = []
    for asset_id, payload in _mapping(provider_evidence).items():
        entry = _mapping(payload)
        if not entry:
            continue
        touched.append(str(asset_id))
        measurement = _mapping(entry.get("geometry_measurement"))
        if measurement:
            geometry_by_asset[str(asset_id)] = dict(measurement)
        rows = [dict(row) for row in _list(entry.get("placement_rows")) if isinstance(row, Mapping)]
        if rows:
            for row in rows:
                row.setdefault("_plan_ref", "provider_evidence")
            placements_by_asset.setdefault(str(asset_id), []).extend(rows)
        observations = [dict(row) for row in _list(entry.get("retained_observations"))
                        if isinstance(row, Mapping)]
        if observations:
            target = retained_by_asset.setdefault(
                str(asset_id), {"asset_id": str(asset_id), "observations": []})
            target.setdefault("observations", []).extend(observations)
    return sorted(set(touched))


#: Dimensions that can only be closed by running a world: a capture, a render.
NATIVE_EVIDENCE_DIMENSIONS = ("emitter", "visibility", "sound_pcm")
#: Dimensions that describe where the asset stands. If any of these is unsettled
#: the pose may still move, and native media recorded at the old pose is not
#: exempt from being redone.
POSE_DIMENSIONS = ("support", "placement", "clearance")


def derive_remaining_work(row: Mapping[str, Any]) -> dict[str, Any]:
    """What this asset still needs, derived from the dimensions as they stand.

    This is recomputed from the current evidence every time the matrix is built,
    so it is never a stale note. The one rule worth stating: native media stays
    reusable only while the pose it was recorded at is settled. A placement that
    has to move, or whose interpenetration is still an open question, takes its
    media with it -- otherwise a re-planned asset would inherit a pass for pixels
    and audio recorded somewhere it no longer stands.
    """
    dims = _mapping(row.get("dimensions"))

    def state(name: str) -> str:
        return _status(_mapping(dims.get(name)).get("status"))

    def pending(name: str) -> bool:
        return bool(_mapping(_mapping(dims.get(name)).get("facts")).get(
            "pending_precise_detection"))

    pose_failed = [name for name in POSE_DIMENSIONS if state(name) == "fail"]
    pose_pending = [name for name in POSE_DIMENSIONS if pending(name)]
    pose_open = [name for name in POSE_DIMENSIONS if state(name) != "pass"]
    native_held = [name for name in NATIVE_EVIDENCE_DIMENSIONS if state(name) == "pass"]
    native_missing = [name for name in NATIVE_EVIDENCE_DIMENSIONS if state(name) != "pass"]
    measurement_missing = [name for name in ("geometry_scale",) + POSE_DIMENSIONS
                           if state(name) != "pass"]
    pose_settled = not pose_open
    media_reusable = bool(native_held) and pose_settled
    if pose_failed:
        bucket = "replan_then_recapture"
    elif pose_pending:
        bucket = "precise_detection_then_decide"
    elif not measurement_missing and not native_missing:
        bucket = "already_qualified"
    elif not measurement_missing:
        bucket = "needs_native_capture_and_audio_only"
    elif not native_missing and pose_settled:
        bucket = "measurement_only_no_new_native"
    elif not native_missing:
        bucket = "settle_pose_then_recheck_media"
    else:
        bucket = "needs_measurement_and_native"
    return {
        "bucket": bucket,
        "pose_settled": pose_settled,
        "pose_failed_dimensions": pose_failed,
        "pose_pending_dimensions": pose_pending,
        "pose_open_dimensions": pose_open,
        "native_evidence_held": native_held,
        "native_evidence_missing": native_missing,
        "measurement_missing": measurement_missing,
        "native_media_reusable": media_reusable,
        "native_media_reuse_reason": (
            "the pose these were recorded at is settled"
            if media_reusable else
            "no native media is held yet" if not native_held else
            "the pose is not settled, so media recorded at it is not exempt from being redone"),
        "requires_new_world": bucket in {
            "replan_then_recapture", "needs_native_capture_and_audio_only",
            "settle_pose_then_recheck_media", "needs_measurement_and_native"},
        "claim_boundary": ("derived from the current dimension states; it schedules work and "
                           "grants no pass"),
    }


def _work_buckets(
    asset_rows: Sequence[Mapping[str, Any]], type_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Group the fine types by what their best candidate still needs."""
    by_asset = {str(row.get("asset_id")): row for row in asset_rows}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for type_row in type_rows:
        best = None
        for asset_id in _list(type_row.get("candidate_asset_ids")):
            candidate = by_asset.get(str(asset_id))
            if candidate is None:
                continue
            if best is None or len(candidate["missing_dimensions"]) < len(best["missing_dimensions"]):
                best = candidate
        if best is None:
            continue
        work = _mapping(best.get("remaining_work"))
        buckets[str(work.get("bucket"))].append({
            "type": type_row.get("type"),
            "representative_asset_id": best.get("asset_id"),
            "support_protocol": best.get("support_protocol"),
            "native_media_reusable": work.get("native_media_reusable"),
            "native_evidence_held": work.get("native_evidence_held"),
            "missing_dimensions": best.get("missing_dimensions"),
            "failed_dimensions": best.get("failed_dimensions"),
            "reusable_world_ids": _mapping(best.get("world_evidence")).get("shared_world_ids"),
            "episodes": sorted({str(_mapping(item).get("episode_id"))
                                for item in _list(best.get("contexts"))
                                if _mapping(item).get("episode_id")}),
        })
    return {
        "derived_from": "the current dimension states of each type's best candidate",
        "counts": {key: len(value) for key, value in sorted(buckets.items())},
        "types_by_bucket": {key: sorted(item["type"] for item in value)
                            for key, value in sorted(buckets.items())},
        "rows_by_bucket": {key: value for key, value in sorted(buckets.items())},
        "claim_boundary": "a work schedule, not an admission or a pass",
    }


def world_evidence(dimensions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Which worlds each dimension was observed in, and whether they can be one asset's case.

    Evidence from unrelated worlds does not add up: a native capture in one room
    and a placement in another describe two different situations. Dimensions that
    name no world (a registry or mesh reading, which is world-independent) never
    block the intersection.
    """
    by_dimension: dict[str, list[str]] = {}
    for name, value in dimensions.items():
        worlds = _list(_mapping(_mapping(value).get("facts")).get("world_ids"))
        if worlds:
            by_dimension[name] = sorted({str(item) for item in worlds})
    shared: set[str] | None = None
    for worlds in by_dimension.values():
        shared = set(worlds) if shared is None else (shared & set(worlds))
    compatible = shared is None or bool(shared)
    conflicting = [] if compatible else sorted(by_dimension)
    return {
        "world_ids_by_dimension": by_dimension,
        "shared_world_ids": sorted(shared) if shared else [],
        "compatible": compatible,
        "conflicting_dimensions": conflicting,
        "claim_boundary": ("dimensions that name no world are world-independent readings and "
                           "do not constrain the intersection"),
    }


def _overall(dimensions: Mapping[str, Mapping[str, Any]]) -> tuple[str, list[str]]:
    statuses = {key: _status(value.get("status")) for key, value in dimensions.items()}
    failures = [key for key in REQUIRED_DIMENSIONS if statuses.get(key) == "fail"]
    missing = [key for key in REQUIRED_DIMENSIONS if statuses.get(key) != "pass"]
    if failures:
        return "fail", failures
    if missing:
        return "not_run", missing
    return "pass", []


def validate_qualification_matrix(matrix: Mapping[str, Any]) -> None:
    if matrix.get("schema") != SCHEMA:
        raise ValueError("qualification matrix schema mismatch")
    counts = _mapping(matrix.get("counts"))
    if len(_list(matrix.get("type_status_rows"))) != int(counts.get("expected_type_count", -1)):
        raise ValueError("qualification matrix type count mismatch")
    if len(_list(matrix.get("asset_status_rows"))) != int(counts.get("expected_asset_count", -1)):
        raise ValueError("qualification matrix asset count mismatch")
    ids = [row.get("asset_id") for row in _list(matrix.get("asset_status_rows"))]
    if len(ids) != len(set(ids)):
        raise ValueError("qualification matrix asset IDs must be unique")
    for row in _list(matrix.get("asset_status_rows")):
        dims = _mapping(row.get("dimensions"))
        if row.get("status") == "pass" and any(
            _status(_mapping(dims.get(key, {})).get("status")) != "pass" for key in REQUIRED_DIMENSIONS
        ):
            raise ValueError(f"asset marked pass with missing dimension: {row.get('asset_id')}")
    for row in _list(matrix.get("type_status_rows")):
        if row.get("status") == "pass" and not _list(row.get("qualified_asset_ids")):
            raise ValueError(f"type marked pass with no fully qualified asset: {row.get('type')}")


def eligible_assets(matrix: Mapping[str, Any], type_name: str | None = None) -> list[str]:
    """Return only assets whose complete acceptance dimensions actually pass."""
    validate_qualification_matrix(matrix)
    return sorted(
        row["asset_id"]
        for row in _list(matrix.get("asset_status_rows"))
        if row.get("status") == "pass" and (type_name is None or row.get("type") == type_name)
    )


def missing_requirements(matrix: Mapping[str, Any], asset_id: str) -> list[str]:
    validate_qualification_matrix(matrix)
    for row in _list(matrix.get("asset_status_rows")):
        if row.get("asset_id") == asset_id:
            return list(row.get("missing_dimensions") or [])
    raise KeyError(asset_id)


def qualification_matrix_csv_rows(matrix: Mapping[str, Any]) -> list[list[str]]:
    """Flatten the asset detail rows for review; every candidate row is kept."""
    validate_qualification_matrix(matrix)
    header = [
        "asset_detail_id", "type", "asset_id", "entity_class", "revision", "overall_status",
        "support_protocol", "shared_world_ids", "missing_dimensions", "failed_dimensions", "episodes",
    ] + [f"{name}_status" for name in REQUIRED_DIMENSIONS] + [
        f"{name}_reason" for name in REQUIRED_DIMENSIONS
    ]
    rows = [header]
    for row in _list(matrix.get("asset_status_rows")):
        dims = _mapping(row.get("dimensions"))
        episodes = sorted({str(_mapping(item).get("episode_id")) for item in _list(row.get("contexts"))
                           if _mapping(item).get("episode_id")})
        rows.append([
            str(row.get("asset_detail_id") or ""),
            str(row.get("type") or ""),
            str(row.get("asset_id") or ""),
            str(row.get("entity_class") or ""),
            str(row.get("revision") or ""),
            str(row.get("status") or ""),
            str(row.get("support_protocol") or ""),
            ";".join(_mapping(row.get("world_evidence")).get("shared_world_ids") or []),
            ";".join(row.get("missing_dimensions") or []),
            ";".join(row.get("failed_dimensions") or []),
            ";".join(episodes),
        ] + [
            _status(_mapping(dims.get(name)).get("status")) for name in REQUIRED_DIMENSIONS
        ] + [
            str(_mapping(dims.get(name)).get("reason") or "") for name in REQUIRED_DIMENSIONS
        ])
    return rows


__all__ = [
    "SCHEMA",
    "STATUSES",
    "REQUIRED_DIMENSIONS",
    "FORBIDDEN_EVIDENCE_KEYS",
    "PLANNING_STAGE_STATUS_VALUES",
    "QualificationEvidenceError",
    "qualification_evidence_interface",
    "build_qualification_matrix",
    "validate_qualification_matrix",
    "eligible_assets",
    "missing_requirements",
    "qualification_matrix_csv_rows",
    "support_protocol",
    "world_evidence",
    "derive_remaining_work",
    "NATIVE_EVIDENCE_DIMENSIONS",
    "POSE_DIMENSIONS",
    "GROUND_ATTACHMENT_SURFACES",
    "MESH_COLLISION_METHODS",
    "MESH_BOUNDS_SOURCES",
    "SUPPORT_SURFACE_ATTACHMENTS",
]
