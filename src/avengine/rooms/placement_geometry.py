"""Measure a planned static source against the room's own surface mesh.

A support surface fitted from one depth readback is a plane, not the slab it
was fitted on.  The plane can sit centimetres away from the triangles a ray
actually hits, and the fitted normal can point out of the room instead of into
it.  A placement built on that plane alone inherits both errors, which is how a
doorbell ends up buried in a wall and a ceiling alarm ends up in the void above
the ceiling, where no camera can see it and no listener shares its room.

Everything in this module is read from the room mesh the rest of the chain
already uses for line of sight, plus the asset's own registered bounds.  There
is no room name, asset name, map name or literal coordinate here: which side is
the room, how far the real surface is from the fitted plane, and whether the
placement penetrates anything are all measured per placement.

The three readings a caller asks for are:

``measure_contact_offset``
    How far along the mounting direction the real surface lies from the fitted
    plane, bounded by the tolerance the catalog itself declares for its fit.

``evaluate_placement``
    Whether the support is solid behind the asset, the asset's own volume is
    free, its corners do not reach into geometry, the emitter keeps a stated
    gap from every triangle, and the placement is inside the room.

``interior_reference_points``
    Standing-height points a room's own navigation says are walkable and whose
    own vertical probe says are indoors; ``evaluate_placement`` asks for a clear
    line to one of them before it calls a placement interior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

PLACEMENT_CHECK_SCHEMA = "avengine_static_placement_geometry_checks_v1"

_EPSILON_M = 1.0e-6


@dataclass(frozen=True)
class PlacementCheckConfig:
    """Geometric tolerances for the mesh checks; every one is a length in metres.

    These are instrument tolerances, not room properties: they say how thin a
    contact counts as touching and how much free air a placement must keep, and
    they read the same way in every room.  ``max_contact_search_m`` is the only
    one with no default, because the bound on how far a fitted plane may be
    from the real surface is the fit tolerance the support catalog declares.
    """

    # A support counts as solid when a triangle is within this distance behind
    # the seated asset.  It is the thickness of the contact, not of the slab.
    support_probe_m: float = 0.02
    # The asset is seated this far off the measured surface so a ray leaving
    # the contact point does not start on the triangle it just found.
    contact_gap_m: float = 0.003
    # Free air the asset's own depth must have in front of the support.
    body_clearance_m: float = 0.01
    # How far a corner probe reaches before it stops caring what is out there.
    corner_probe_m: float = 0.02
    # The emitter is a point source; it must not sit on a wall.
    min_emitter_clearance_m: float = 0.03
    # How far a device may be stood off its surface so that its own emitter
    # keeps the acoustic minimum. Some assets carry their emitter anchor a
    # couple of centimetres inside the face they mount by, and a millimetre of
    # extra air is the difference between a legal source and a refused one. An
    # asset that would need more than this is refused instead: a device sitting
    # further than this off its wall is not mounted on it.
    max_standoff_m: float = 0.01
    # How far above the mesh a vertical probe is allowed to look for a ceiling.
    ceiling_probe_margin_m: float = 0.5
    # Standing eye height used when a room's walkable points are consulted.
    interior_eye_height_m: float = 1.5
    # How many of the nearest walkable points to try before giving up.
    interior_reference_limit: int = 16
    max_contact_search_m: float | None = None

    @classmethod
    def from_mapping(cls, value: Any, *, fallback_search_m: float | None = None):
        """Read the declared subset of tolerances; omitted keys keep defaults."""
        config = cls()
        if isinstance(value, Mapping):
            fields = {
                "support_probe_m", "contact_gap_m", "body_clearance_m",
                "corner_probe_m", "min_emitter_clearance_m", "max_standoff_m",
                "ceiling_probe_margin_m", "interior_eye_height_m",
                "max_contact_search_m",
            }
            updates: dict[str, Any] = {}
            for key in fields:
                if value.get(key) is not None:
                    number = float(value[key])
                    if not math.isfinite(number) or number < 0.0:
                        raise ValueError(
                            f"placement check {key} must be a finite nonnegative length"
                        )
                    updates[key] = number
            if value.get("interior_reference_limit") is not None:
                limit = int(value["interior_reference_limit"])
                if limit < 1:
                    raise ValueError(
                        "placement check interior_reference_limit must be positive"
                    )
                updates["interior_reference_limit"] = limit
            config = replace(config, **updates)
        if config.max_contact_search_m is None and fallback_search_m is not None:
            config = replace(config, max_contact_search_m=float(fallback_search_m))
        return config


def _mesh_arrays(mesh: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(getattr(mesh, "vertices"), dtype=float)
    triangles = np.asarray(getattr(mesh, "triangles"), dtype=np.int64)
    minimum = getattr(mesh, "minimum", None)
    maximum = getattr(mesh, "maximum", None)
    if minimum is None or maximum is None:
        corners = vertices[triangles]
        minimum = corners.min(axis=1)
        maximum = corners.max(axis=1)
    return vertices, triangles, np.asarray(minimum, float), np.asarray(maximum, float)


@dataclass(frozen=True)
class LocalMesh:
    """The room triangles inside one box, with the same reading surface as the room.

    Every probe a seated placement needs lives inside a box a few tens of
    centimetres across, while the room carries a million triangles or more.
    Selecting that box once turns each probe's broad phase from a scan of the
    room into a scan of its neighbourhood; the triangles and therefore the
    readings are the room's own, unchanged.
    """

    vertices: np.ndarray
    triangles: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    source: Any = None


def local_mesh(mesh: Any, centre, *, radius: float) -> LocalMesh:
    """Room triangles whose own box reaches within ``radius`` of ``centre``."""
    vertices, triangles, low, high = _mesh_arrays(mesh)
    point = np.asarray(centre, dtype=float)
    near = (
        np.all(high >= point - radius, axis=1)
        & np.all(low <= point + radius, axis=1)
    )
    selected = triangles[near]
    return LocalMesh(
        vertices=vertices,
        triangles=selected,
        minimum=low[near],
        maximum=high[near],
        source={"origin_m": [float(value) for value in point],
                "radius_m": float(radius),
                "triangle_count": int(len(selected)),
                "room_triangle_count": int(len(triangles))},
    )


def segment_hits(mesh: Any, origin, direction, *, t_min: float = _EPSILON_M,
                 t_max: float) -> np.ndarray:
    """Sorted distances at which a ray meets the mesh inside ``[t_min, t_max]``.

    Both faces count.  A room export is not watertight and its triangles carry
    no reliable winding, so a one-sided test would miss half the walls.
    """
    vertices, triangles, low, high = _mesh_arrays(mesh)
    start = np.asarray(origin, dtype=float)
    ray = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(ray))
    if not np.all(np.isfinite(start)) or not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("a mesh probe needs a finite origin and a non-zero direction")
    ray = ray / norm
    if not math.isfinite(t_max) or t_max <= t_min:
        return np.empty(0, dtype=float)
    near = start + ray * min(t_min, 0.0)
    far = start + ray * t_max
    box_low = np.minimum(near, far) - 1.0e-3
    box_high = np.maximum(near, far) + 1.0e-3
    relevant = np.all(high >= box_low, axis=1) & np.all(low <= box_high, axis=1)
    selected = triangles[relevant]
    if not len(selected):
        return np.empty(0, dtype=float)
    a = vertices[selected[:, 0]]
    edge1 = vertices[selected[:, 1]] - a
    edge2 = vertices[selected[:, 2]] - a
    pvec = np.cross(ray, edge2)
    determinant = np.einsum("ij,ij->i", edge1, pvec)
    usable = np.abs(determinant) > 1.0e-14
    inverse = np.zeros_like(determinant)
    inverse[usable] = 1.0 / determinant[usable]
    tvec = start - a
    u = inverse * np.einsum("ij,ij->i", tvec, pvec)
    qvec = np.cross(tvec, edge1)
    v = inverse * np.einsum("j,ij->i", ray, qvec)
    t = inverse * np.einsum("ij,ij->i", edge2, qvec)
    hit = (
        usable
        & (u >= -1.0e-9) & (u <= 1.0 + 1.0e-9)
        & (v >= -1.0e-9) & (u + v <= 1.0 + 1.0e-9)
        & (t >= t_min) & (t <= t_max)
    )
    return np.sort(t[hit])


def nearest_surface_distance(mesh: Any, point, *, radius: float) -> float | None:
    """Exact point-to-triangle distance inside ``radius``; ``None`` when farther.

    The broad phase keeps triangles whose own box reaches within ``radius``, so
    the answer is exact whenever it is not ``None``, and ``None`` is itself the
    statement that nothing is nearer than ``radius``.
    """
    vertices, triangles, low, high = _mesh_arrays(mesh)
    probe = np.asarray(point, dtype=float)
    if probe.shape != (3,) or not np.all(np.isfinite(probe)):
        raise ValueError("a clearance probe needs one finite point")
    near = np.all(high >= probe - radius, axis=1) & np.all(low <= probe + radius, axis=1)
    selected = triangles[near]
    if not len(selected):
        return None
    a = vertices[selected[:, 0]]
    b = vertices[selected[:, 1]]
    c = vertices[selected[:, 2]]
    ab = b - a
    ac = c - a
    ap = probe - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = probe - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = probe - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    closest = np.array(a, dtype=float, copy=True)
    resolved = (d1 <= 0.0) & (d2 <= 0.0)
    mask = ~resolved & (d3 >= 0.0) & (d4 <= d3)
    closest[mask] = b[mask]
    resolved |= mask
    mask = ~resolved & (d6 >= 0.0) & (d5 <= d6)
    closest[mask] = c[mask]
    resolved |= mask
    vc = d1 * d4 - d3 * d2
    mask = ~resolved & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    denominator = np.where(np.abs(d1 - d3) > 1.0e-18, d1 - d3, 1.0)
    ratio = np.clip(d1 / denominator, 0.0, 1.0)
    closest[mask] = a[mask] + ratio[mask, None] * ab[mask]
    resolved |= mask
    vb = d5 * d2 - d1 * d6
    mask = ~resolved & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    denominator = np.where(np.abs(d2 - d6) > 1.0e-18, d2 - d6, 1.0)
    ratio = np.clip(d2 / denominator, 0.0, 1.0)
    closest[mask] = a[mask] + ratio[mask, None] * ac[mask]
    resolved |= mask
    va = d3 * d6 - d5 * d4
    mask = ~resolved & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    denominator = np.where(
        np.abs((d4 - d3) + (d5 - d6)) > 1.0e-18, (d4 - d3) + (d5 - d6), 1.0
    )
    ratio = np.clip((d4 - d3) / denominator, 0.0, 1.0)
    closest[mask] = b[mask] + ratio[mask, None] * (c[mask] - b[mask])
    resolved |= mask
    interior = ~resolved
    if np.any(interior):
        total = va + vb + vc
        total = np.where(np.abs(total) > 1.0e-18, total, 1.0)
        w1 = (vb / total)[interior, None]
        w2 = (vc / total)[interior, None]
        closest[interior] = a[interior] + w1 * ab[interior] + w2 * ac[interior]
    distances = np.linalg.norm(closest - probe, axis=1)
    if not len(distances):
        return None
    smallest = float(distances.min())
    return smallest if smallest <= radius else None


def vertical_enclosure(mesh: Any, point, *, margin_m: float) -> dict[str, Any]:
    """Is something above this point, bounded by the room mesh's own top?

    A point under a ceiling has geometry above it; a point standing on the roof
    or floating outside the building has open sky, and the probe stops at the
    highest triangle in the room plus ``margin_m`` so it can never run forever.
    A hollow wall also has a ceiling above it, so this reading refutes an
    outdoor placement but does not on its own prove an indoor one.
    """
    vertices, _triangles, _low, _high = _mesh_arrays(mesh)
    probe = np.asarray(point, dtype=float)
    ceiling_limit = float(vertices[:, 1].max()) + float(margin_m)
    reach = ceiling_limit - float(probe[1])
    if reach <= 0.0:
        return {
            "status": "fail",
            "reason": "the point is at or above the highest triangle in the room",
            "mesh_top_m": float(vertices[:, 1].max()),
            "probe_length_m": 0.0,
            "first_hit_above_m": None,
        }
    hits = segment_hits(mesh, probe, (0.0, 1.0, 0.0), t_max=reach)
    return {
        "status": "pass" if len(hits) else "fail",
        "reason": (
            "a triangle covers this point"
            if len(hits)
            else "nothing covers this point below the room's highest triangle"
        ),
        "mesh_top_m": float(vertices[:, 1].max()),
        "probe_length_m": float(reach),
        "first_hit_above_m": float(hits[0]) if len(hits) else None,
    }


def _standing_grid_on_every_floor(space, floor_heights, *, height_above_floor_m,
                                  step_m, region, tolerance_m):
    """Standing points on each of a room's floors, not only the one it starts on.

    A navigation grid is probed from the floor height the space announces, and a
    snap to the mesh from there lands on that storey. A house with an upstairs
    therefore offers no standing point upstairs, and a device on an upstairs
    wall looks like it is outside the building because nobody can see it. Each
    declared floor is probed in turn and the results are pooled.
    """
    from avengine.rooms.walkable_space import camera_grid

    bounds = space.bounds() if region is None else np.asarray(region, dtype=float)
    metadata = getattr(space, "metadata", None)
    saved = metadata.get("floor_height_m") if isinstance(metadata, dict) else None
    had = isinstance(metadata, dict) and "floor_height_m" in metadata
    rows = []
    try:
        for floor in floor_heights:
            window = np.asarray(bounds, dtype=float).copy()
            window[0, 1] = float(floor) - tolerance_m
            window[1, 1] = float(floor) + tolerance_m
            if isinstance(metadata, dict):
                metadata["floor_height_m"] = float(floor)
            for point in camera_grid(space, step_m=step_m,
                                     height_above_floor_m=height_above_floor_m,
                                     region=window):
                if abs(float(point[1]) - height_above_floor_m - float(floor)) <= tolerance_m:
                    rows.append([float(value) for value in point])
    finally:
        if isinstance(metadata, dict):
            if had:
                metadata["floor_height_m"] = saved
            else:
                metadata.pop("floor_height_m", None)
    return rows


def interior_reference_points(mesh: Any, space: Any, *, config: PlacementCheckConfig,
                              step_m: float = 0.55, region=None,
                              floor_heights: Sequence[float] = (),
                              floor_tolerance_m: float = 0.3) -> dict[str, Any]:
    """Standing-height walkable points whose own vertical probe says indoors.

    The room's navigation decides what is walkable; the vertical probe then
    drops the walkable points that are outdoors, which is what a navmesh baked
    over a roof produces.  Without that filter a roof point could certify a
    placement that is outside the room it is supposed to be in.
    """
    from avengine.rooms.walkable_space import camera_grid

    if space is None or mesh is None:
        return {
            "status": "not_run",
            "reason": "no walkable space or room mesh was supplied",
            "points_m": np.empty((0, 3), dtype=float),
            "walkable_count": 0,
            "indoor_count": 0,
        }
    try:
        if floor_heights:
            grid = _standing_grid_on_every_floor(
                space, floor_heights, height_above_floor_m=config.interior_eye_height_m,
                step_m=step_m, region=region, tolerance_m=floor_tolerance_m,
            )
        else:
            grid = camera_grid(
                space, step_m=step_m,
                height_above_floor_m=config.interior_eye_height_m, region=region,
            )
    except (ValueError, KeyError, TypeError) as error:
        return {
            "status": "not_run",
            "reason": f"the room's navigation produced no standing grid: {error}",
            "points_m": np.empty((0, 3), dtype=float),
            "walkable_count": 0,
            "indoor_count": 0,
        }
    candidates = np.asarray(grid, dtype=float).reshape(-1, 3)
    kept = [
        row for row in candidates
        if vertical_enclosure(mesh, row, margin_m=config.ceiling_probe_margin_m)["status"] == "pass"
    ]
    points = np.asarray(kept, dtype=float).reshape(-1, 3)
    return {
        "status": "measured" if len(points) else "empty",
        "reason": (
            "walkable standing points that are covered by the room"
            if len(points)
            else "no walkable standing point in this room is covered by its own mesh"
        ),
        "points_m": points,
        "walkable_count": int(len(candidates)),
        "indoor_count": int(len(points)),
        "eye_height_m": float(config.interior_eye_height_m),
        "grid_step_m": float(step_m),
    }


def body_span_along_normal(local_min, local_max, plane_normal_local,
                           base_plane_offset_m: float) -> dict[str, Any]:
    """Where the asset's own bounds sit relative to its registered base plane.

    A wall box grows away from its back plane and a ceiling disc hangs below its
    top plane.  Both are ordinary registered poses, so the mounting direction is
    read from the measured bounds rather than assumed from the surface kind.
    """
    low = np.asarray(local_min, dtype=float)
    high = np.asarray(local_max, dtype=float)
    normal = np.asarray(plane_normal_local, dtype=float)
    normal = normal / float(np.linalg.norm(normal))
    contact = normal * float(base_plane_offset_m)
    corners = np.array([[x, y, z] for x in (low[0], high[0])
                        for y in (low[1], high[1])
                        for z in (low[2], high[2])], dtype=float)
    projected = (corners - contact) @ normal
    ahead = float(projected.max())
    behind = float(-projected.min())
    if ahead >= behind:
        return {"sign": 1.0, "depth_m": ahead, "overhang_m": max(behind, 0.0),
                "span_m": [float(projected.min()), float(projected.max())]}
    return {"sign": -1.0, "depth_m": behind, "overhang_m": max(ahead, 0.0),
            "span_m": [float(projected.min()), float(projected.max())]}


def measure_contact_offset(mesh: Any, *, plane_point, out_direction,
                           body_depth_m: float,
                           config: PlacementCheckConfig,
                           footprint_points: Sequence[Any] = ()) -> dict[str, Any]:
    """Distance from the fitted plane to the real surface the asset rests on.

    The reading is taken over the asset's whole footprint, not just under its
    centre.  A plane fitted to a depth readback leans a degree or two against
    the surface it was fitted on, so a disc seated by its centre has one edge
    a few millimetres inside the ceiling; seating it on the most protruding
    part of the surface under its footprint puts the whole contact face clear.
    That is what the existing ``surface_normal_offset_m`` field already means,
    measured here instead of stated by the request.

    A crossing only counts when the asset's depth plus its clearance fits in
    front of it, which rules out seating against the near face of a slab the
    asset would then have to grow through.  Among the offsets that qualify the
    one nearest the fitted plane wins, so the asset moves as little as the
    geometry allows and never hops through a wall into the next room when the
    near face was usable.
    """
    search = config.max_contact_search_m
    if search is None or search <= 0.0:
        return {"status": "not_run", "reason": "no contact search bound was declared",
                "offset_m": 0.0}
    origin = np.asarray(plane_point, dtype=float)
    direction = np.asarray(out_direction, dtype=float)
    direction = direction / float(np.linalg.norm(direction))
    probes = [origin]
    for extra in footprint_points or ():
        point = np.asarray(extra, dtype=float)
        # Keep only the in-plane part: every probe starts on the fitted plane.
        probes.append(point - float(np.dot(point - origin, direction)) * direction)
    needed = float(body_depth_m) + config.body_clearance_m

    def crossings_at(point):
        forward = segment_hits(mesh, point, direction, t_min=-search, t_max=search)
        backward = -segment_hits(mesh, point, -direction, t_min=_EPSILON_M, t_max=search)
        return np.unique(np.round(np.concatenate([forward, backward[::-1]]), 6))

    def free_after(point, value):
        return not len(segment_hits(
            mesh, point + direction * value, direction,
            t_min=config.contact_gap_m, t_max=needed + config.contact_gap_m,
        ))

    per_probe = []
    for point in probes:
        values = crossings_at(point)
        usable = [float(value) for value in values if free_after(point, value)]
        per_probe.append({"crossings_m": [float(value) for value in values],
                          "usable_m": usable})
    # The centre is what says the asset is resting on something. A footprint
    # corner with nothing under it is the fitted rectangle reaching past the
    # real surface, not an unsupported asset: it cannot raise the seat, so it
    # is recorded and passed over. Whether the asset is really seated is
    # decided afterwards by probing behind it, and whether its own volume is
    # free by probing in front.
    unsupported = [index for index, row in enumerate(per_probe) if not row["usable_m"]]
    if not per_probe[0]["usable_m"]:
        return {
            "status": "fail",
            "reason": ("no room triangle crosses the mounting axis under the centre "
                       "of the footprint within the catalog's declared plane tolerance"
                       if not per_probe[0]["crossings_m"] else
                       "every surface under the centre of the footprint has less free "
                       "depth in front of it than the asset needs"),
            "offset_m": 0.0,
            "search_bound_m": float(search),
            "required_free_depth_m": needed,
            "probe_count": len(probes),
            "probes_without_seat": unsupported,
            "probe_crossings_m": [row["crossings_m"] for row in per_probe],
        }
    nearest = [min(row["usable_m"], key=abs) for row in per_probe if row["usable_m"]]
    chosen = max(nearest)
    obstructed = [index for index, point in enumerate(probes)
                  if not free_after(point, chosen)]
    if obstructed:
        return {
            "status": "fail",
            "reason": ("the seat that clears the whole footprint leaves part of it "
                       "without the free depth the asset needs"),
            "offset_m": 0.0,
            "search_bound_m": float(search),
            "required_free_depth_m": needed,
            "probe_count": len(probes),
            "obstructed_probes": obstructed,
            "surface_offset_m": float(chosen),
            "probe_seat_offsets_m": [float(value) for value in nearest],
        }
    return {
        "status": "measured",
        "reason": ("the most protruding surface on the mounting axis under the "
                   "footprint that leaves room for the asset"),
        "offset_m": float(chosen + config.contact_gap_m),
        "surface_offset_m": float(chosen),
        "centre_offset_m": float(nearest[0]),
        "contact_gap_m": float(config.contact_gap_m),
        "search_bound_m": float(search),
        "required_free_depth_m": needed,
        "probe_count": len(probes),
        "probes_without_seat": unsupported,
        "probe_seat_offsets_m": [float(value) for value in nearest],
        "footprint_unevenness_m": float(max(nearest) - min(nearest)),
    }


def evaluate_placement(mesh: Any, *, contact_point, out_direction, plane_u, plane_v,
                       body_depth_m: float, corners_m, emitter_point,
                       interior_points=None, near_mesh: Any = None,
                       config: PlacementCheckConfig | None = None) -> dict[str, Any]:
    """Read a seated placement against the room mesh and say what it is.

    ``corners_m`` are the asset's own eight oriented box corners, not its world
    axis-aligned box: the axis-aligned box of a thin panel on a slanted wall
    reaches past the wall by construction, and probing from there would refuse
    a placement that is in fact flush.
    """
    config = config or PlacementCheckConfig()
    contact = np.asarray(contact_point, dtype=float)
    forward = np.asarray(out_direction, dtype=float)
    forward = forward / float(np.linalg.norm(forward))
    axis_u = np.asarray(plane_u, dtype=float)
    axis_u = axis_u / float(np.linalg.norm(axis_u))
    axis_v = np.asarray(plane_v, dtype=float)
    axis_v = axis_v / float(np.linalg.norm(axis_v))
    emitter = np.asarray(emitter_point, dtype=float)
    corners = np.asarray(corners_m, dtype=float).reshape(-1, 3)
    # Every reading below this line but the line of sight stays inside one
    # small box, so it is taken against that box's triangles when the caller
    # already selected them.
    near = mesh if near_mesh is None else near_mesh

    behind = segment_hits(
        near, contact, -forward,
        t_min=_EPSILON_M, t_max=config.contact_gap_m + config.support_probe_m,
    )
    support = {
        "status": "pass" if len(behind) else "fail",
        "probe_length_m": float(config.contact_gap_m + config.support_probe_m),
        "first_hit_m": float(behind[0]) if len(behind) else None,
        "reason": (
            "a room triangle sits directly behind the seated asset"
            if len(behind)
            else "nothing solid is behind the asset, so it is not resting on this surface"
        ),
    }

    ahead = segment_hits(
        near, contact, forward,
        t_min=config.contact_gap_m,
        t_max=float(body_depth_m) + config.body_clearance_m,
    )
    body = {
        "status": "pass" if not len(ahead) else "fail",
        "probe_length_m": float(body_depth_m) + config.body_clearance_m,
        "first_hit_m": float(ahead[0]) if len(ahead) else None,
        "reason": (
            "the volume the asset occupies is free"
            if not len(ahead)
            else "room geometry reaches into the volume the asset occupies"
        ),
    }

    centre = corners.mean(axis=0)
    corner_failures = []
    corner_probes = 0
    depths = (corners - contact) @ forward
    for index, corner in enumerate(corners):
        offset = corner - centre
        directions = [forward]
        # Sideways probes are taken only from the corners that stand clear of
        # the support.  A corner seated on the support is by definition touching
        # it, and on a fitted plane that leans a degree or two against the real
        # surface a sideways probe from there grazes the surface it is resting
        # on, which is a contact rather than a penetration.
        if float(depths[index]) > 0.5 * float(body_depth_m):
            for axis in (axis_u, axis_v):
                component = float(np.dot(offset, axis))
                if abs(component) > 1.0e-9:
                    directions.append(axis * math.copysign(1.0, component))
        for direction in directions:
            corner_probes += 1
            if len(segment_hits(near, corner, direction,
                                t_min=_EPSILON_M, t_max=config.corner_probe_m)):
                corner_failures.append({
                    "corner_index": int(index),
                    "corner_m": [float(value) for value in corner],
                    "direction": [float(value) for value in direction],
                })
    penetration = {
        "status": "pass" if not corner_failures else "fail",
        "probe_length_m": float(config.corner_probe_m),
        "probe_count": int(corner_probes),
        "blocked_probes": corner_failures[:8],
        "blocked_probe_count": len(corner_failures),
        "reason": (
            "no probe leaving the asset's own box reaches into geometry"
            if not corner_failures
            else "the asset's box reaches into room geometry"
        ),
    }

    distance = nearest_surface_distance(
        near, emitter, radius=config.min_emitter_clearance_m
    )
    # "at least this far" includes being exactly that far: a placement seated
    # to buy precisely the stated clearance must not then fail the check that
    # asked for it.
    too_close = distance is not None and distance < config.min_emitter_clearance_m - 1.0e-9
    emitter_clearance = {
        "status": "fail" if too_close else "pass",
        "minimum_clearance_m": float(config.min_emitter_clearance_m),
        "measured_distance_m": distance,
        "reason": (
            f"the emitter is {distance:.4f} m from a room triangle"
            if too_close
            else "no room triangle is within the required emitter clearance"
        ),
    }

    earlier = [
        name for name, row in (
            ("support_contact", support), ("body_volume", body),
            ("box_penetration", penetration), ("emitter_clearance", emitter_clearance),
        ) if row["status"] == "fail"
    ]
    if earlier:
        # The room already refused this seating on a reading that costs one
        # short probe. The long line-of-sight sweep would not change that, and
        # running it for every refused candidate is what makes the search slow.
        covered = {
            "status": "not_run",
            "reason": "an earlier check already refused this seating",
        }
        reference = dict(covered)
    else:
        covered = vertical_enclosure(mesh, emitter, margin_m=config.ceiling_probe_margin_m)
        reference = {
            "status": "not_run",
            "reason": "no indoor walkable reference points were supplied",
            "checked_count": 0,
            "clear_point_m": None,
        }
    if interior_points is not None and not earlier:
        from avengine.qa.answerability import line_of_sight

        points = np.asarray(interior_points, dtype=float).reshape(-1, 3)
        if not len(points):
            reference = {
                "status": "fail",
                "reason": "this room offered no indoor walkable reference point",
                "checked_count": 0,
                "clear_point_m": None,
            }
        else:
            order = np.argsort(np.linalg.norm(points - emitter, axis=1))
            order = order[: config.interior_reference_limit]
            selected = points[order]
            # The lines to test all live inside one box.  Selecting it once and
            # asking the shared line-of-sight the same question against those
            # triangles keeps the answer identical and stops every candidate
            # placement from re-scanning the whole room a dozen times.
            box_low = np.minimum(selected.min(axis=0), emitter)
            box_high = np.maximum(selected.max(axis=0), emitter)
            corridor = local_mesh(
                mesh, (box_low + box_high) / 2.0,
                radius=float(np.max(box_high - box_low)) / 2.0 + 0.05,
            )
            found = None
            examined = 0
            for candidate in selected:
                examined += 1
                if line_of_sight(corridor, emitter, candidate) == "clear":
                    found = candidate
                    break
            reference = {
                "status": "pass" if found is not None else "fail",
                "checked_count": examined,
                "available_count": int(len(points)),
                "clear_point_m": None if found is None else [float(v) for v in found],
                "reason": (
                    "the emitter has a clear line to a walkable point in this room"
                    if found is not None
                    else "no walkable point in this room can see the emitter"
                ),
            }
    interior_status = "fail"
    if earlier:
        interior_status = "not_run"
    elif covered["status"] == "pass" and reference["status"] == "pass":
        interior_status = "pass"
    elif covered["status"] == "pass" and reference["status"] == "not_run":
        interior_status = "partial"
    interior = {
        "status": interior_status,
        "vertical_cover": covered,
        "walkable_reference": reference,
        "reason": (
            "covered by the room and visible from a walkable point in it"
            if interior_status == "pass"
            else "covered by the room, with no walkable reference supplied"
            if interior_status == "partial"
            else "not read; an earlier check already refused this seating"
            if interior_status == "not_run"
            else "this placement is not inside the room"
        ),
    }

    checks = {
        "support_contact": support,
        "body_volume": body,
        "box_penetration": penetration,
        "emitter_clearance": emitter_clearance,
        "interior": interior,
    }
    failed = sorted(name for name, row in checks.items() if row["status"] == "fail")
    return {
        "schema": PLACEMENT_CHECK_SCHEMA,
        "status": "fail" if failed else ("partial" if interior_status == "partial" else "pass"),
        "failed_checks": failed,
        "mounting_direction_m": [float(value) for value in forward],
        "body_depth_m": float(body_depth_m),
        "contact_point_m": [float(value) for value in contact],
        "emitter_point_m": [float(value) for value in emitter],
        "tolerances_m": {
            "support_probe_m": config.support_probe_m,
            "contact_gap_m": config.contact_gap_m,
            "body_clearance_m": config.body_clearance_m,
            "corner_probe_m": config.corner_probe_m,
            "min_emitter_clearance_m": config.min_emitter_clearance_m,
            "ceiling_probe_margin_m": config.ceiling_probe_margin_m,
        },
        **checks,
        "claim_boundary": (
            "room surface mesh readings only; they do not claim renderer pixels, "
            "acoustic audibility or dataset admission"
        ),
    }


def visible_walkable_count(mesh: Any, point, interior_points, *, limit: int = 12) -> dict[str, Any]:
    """How many of the nearest indoor walkable points can see this point.

    A surface can be perfectly good to mount on and still be somewhere nobody
    ever looks. Counting the standing points that can see a seated emitter is a
    cheap, camera-independent way to prefer the wall or ceiling a room actually
    presents over the one that merely has the most area.
    """
    from avengine.qa.answerability import line_of_sight

    if mesh is None or interior_points is None:
        return {"status": "not_run", "visible": 0, "checked": 0}
    points = np.asarray(interior_points, dtype=float).reshape(-1, 3)
    probe = np.asarray(point, dtype=float)
    if not len(points):
        return {"status": "not_run", "visible": 0, "checked": 0}
    order = np.argsort(np.linalg.norm(points - probe, axis=1))[:limit]
    selected = points[order]
    box_low = np.minimum(selected.min(axis=0), probe)
    box_high = np.maximum(selected.max(axis=0), probe)
    corridor = local_mesh(mesh, (box_low + box_high) / 2.0,
                          radius=float(np.max(box_high - box_low)) / 2.0 + 0.05)
    visible = sum(1 for candidate in selected
                  if line_of_sight(corridor, probe, candidate) == "clear")
    return {"status": "measured", "visible": int(visible), "checked": int(len(selected))}


def check_acoustic_point(mesh: Any, point, *, role: str, identifier: str,
                         config: PlacementCheckConfig | None = None) -> dict[str, Any]:
    """The pre-audio reading for one source or listener pose.

    RLR is given a point in a room.  A point in the void above a ceiling, or a
    point sitting on a wall, produces an invalid native payload several minutes
    into the render; the same two readings taken here name the reason before
    the simulation starts.
    """
    config = config or PlacementCheckConfig()
    probe = np.asarray(point, dtype=float)
    covered = vertical_enclosure(mesh, probe, margin_m=config.ceiling_probe_margin_m)
    distance = nearest_surface_distance(
        mesh, probe, radius=config.min_emitter_clearance_m
    )
    clearance = {
        "status": "fail"
        if distance is not None and distance < config.min_emitter_clearance_m - 1.0e-9
        else "pass",
        "minimum_clearance_m": float(config.min_emitter_clearance_m),
        "measured_distance_m": distance,
    }
    failed = [name for name, row in (("interior", covered), ("clearance", clearance))
              if row["status"] == "fail"]
    return {
        "role": str(role),
        "id": str(identifier),
        "position_m": [float(value) for value in probe],
        "status": "fail" if failed else "pass",
        "failed_checks": failed,
        "interior": covered,
        "clearance": clearance,
    }


def check_acoustic_scene_points(mesh: Any, rows: Sequence[Mapping[str, Any]], *,
                                config: PlacementCheckConfig | None = None) -> dict[str, Any]:
    """Read every declared source and listener pose before any RIR job is built.

    The listener is the room's own witness that the vertical criterion applies
    here.  A camera standing in the room is indoors by construction, so if the
    room mesh has nothing above the listener either the mesh carries no ceiling
    and the criterion cannot separate inside from outside in this room.  In
    that case the reading is recorded as not applicable rather than used to
    refuse poses it cannot judge, and the clearance reading still stands.
    """
    config = config or PlacementCheckConfig()
    if mesh is None:
        return {
            "schema": "avengine_acoustic_pose_enclosure_check_v1",
            "status": "not_run",
            "reason": "the acoustic package carries no readable surface mesh",
            "rows": [],
        }
    results = [
        check_acoustic_point(
            mesh, row["position_m"], role=row.get("role", "source"),
            identifier=row.get("id", ""), config=config,
        )
        for row in rows
    ]
    listeners = [row for row in results if row["role"] == "listener"]
    interior_applies = bool(listeners) and any(
        row["interior"]["status"] == "pass" for row in listeners
    )
    for row in results:
        if not interior_applies:
            row["interior"]["status"] = "not_applicable"
            row["interior"]["reason"] = (
                "no listener in this room is covered by its own mesh, so the "
                "vertical criterion cannot separate inside from outside here"
            )
        row["failed_checks"] = [
            name for name, entry in (("interior", row["interior"]),
                                     ("clearance", row["clearance"]))
            if entry["status"] == "fail"
        ]
        row["status"] = "fail" if row["failed_checks"] else "pass"
    failed = [row for row in results if row["status"] == "fail"]
    return {
        "schema": "avengine_acoustic_pose_enclosure_check_v1",
        "status": "fail" if failed else ("pass" if results else "not_run"),
        "interior_criterion": "applied" if interior_applies else "not_applicable",
        "checked_count": len(results),
        "failed_count": len(failed),
        "minimum_clearance_m": float(config.min_emitter_clearance_m),
        "rows": results,
        "claim_boundary": (
            "each pose is read against the acoustic package's own surface mesh; "
            "it is not an audibility or propagation claim"
        ),
    }


def acoustic_pose_refusal(report: Mapping[str, Any]) -> str | None:
    """One readable sentence per refused pose, or ``None`` when none was refused."""
    if not isinstance(report, Mapping) or report.get("status") != "fail":
        return None
    lines = []
    for row in report.get("rows") or ():
        if row.get("status") != "fail":
            continue
        details = []
        if "interior" in (row.get("failed_checks") or ()):
            details.append(row["interior"].get("reason", "outside the room"))
        if "clearance" in (row.get("failed_checks") or ()):
            measured = row["clearance"].get("measured_distance_m")
            details.append(
                f"it is {measured:.4f} m from a room triangle, closer than the "
                f"{row['clearance']['minimum_clearance_m']:.3f} m a source or "
                "listener needs"
                if measured is not None else "it is too close to a room triangle"
            )
        lines.append(
            f"{row.get('role')} {row.get('id')} at "
            f"{[round(value, 3) for value in row.get('position_m') or ()]}: "
            + "; ".join(details)
        )
    if not lines:
        return None
    return (
        "the acoustic package's own surface mesh refuses "
        f"{report.get('failed_count')} of {report.get('checked_count')} poses "
        "before any RIR is simulated: " + " | ".join(lines)
    )


__all__ = [
    "PLACEMENT_CHECK_SCHEMA",
    "PlacementCheckConfig",
    "body_span_along_normal",
    "check_acoustic_point",
    "acoustic_pose_refusal",
    "check_acoustic_scene_points",
    "evaluate_placement",
    "interior_reference_points",
    "local_mesh",
    "LocalMesh",
    "measure_contact_offset",
    "nearest_surface_distance",
    "segment_hits",
    "vertical_enclosure",
    "visible_walkable_count",
]
