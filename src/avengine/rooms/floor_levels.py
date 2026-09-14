"""Decide which navigable levels of a room are interior floors to plan on.

A navigation mesh built from a room's render surface does not distinguish the
rooms inside a building from its roof, its balconies or the ground outside it.
All of those are horizontal surfaces an agent fits on, so Recast returns all of
them, and a planner that draws a floor by navigable area can put the camera and
the actors on a roof that no light reaches.

Planning does have to distinguish them, and the test here asks nothing about
which room, dataset or map a level belongs to: stand on the level, look
straight up, and see whether the room's own static triangles are overhead. A
point inside a building is covered by its ceiling or by the storey above it; a
point on a roof, a balcony or outdoor ground is open to the sky. A real
two-storey house therefore keeps both of its floors, while the roof of a
single-storey house is dropped, and neither outcome needs a rule about that
particular house.

Scanned and exported meshes have holes, so a level is decided by a majority of
its probes rather than by every one of them, and a level that no probe can
reach stays undecided instead of being called either way.

Being indoors is not enough on its own. Recast also walks onto a bed, a desk
and the run of a staircase, and each of those is a level with a ceiling over
it. What separates them from a room is how much floor they offer, so a level
also has to hold a minimum number of square metres, measured in metres and
not as a share of the building: the same share is a whole storey in a large
house and a cupboard in a small flat.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.qa.answerability import MeshHandle, line_of_sight

# A level is a floor when a majority of the points on it have room geometry
# within this distance overhead. Six metres clears a tall storey (the measured
# apartment ceiling sits 3.9 m above its floor) while staying far below the
# height at which an unrelated overhang could be mistaken for a ceiling.
INTERIOR_CEILING_PROBE_M = 6.0
# The upward ray starts above the navigable height so that floor triangles,
# which a Recast surface sits a few centimetres above, cannot block it.
INTERIOR_PROBE_START_OFFSET_M = 0.5
INTERIOR_COVERED_SHARE = 0.7
INTERIOR_PROBE_SAMPLES = 32
INTERIOR_PROBE_MINIMUM = 3

# A level smaller than this is a ledge, a stair run or a piece of furniture
# Recast walked onto, not a room an episode can happen in. One episode is a
# fixed camera, ten seconds, an actor that has to walk a route of a metre and a
# half, and a camera that has to stand more than a metre from that actor: about
# 2 m by 3 m of navigable surface holds all of it, and a three to four square
# metre bathroom does not. This is a starting value and is meant to be tuned -
# the measured area of every level of every registered room is in the 2026-09-14
# floors report. Measured that day: kujiale_0020's 0.46 m bed top is 3.3 m2 and
# hm3d_val_00800's 0.76 m stair run is 4.0 m2, while the four real floors of
# those two rooms are 35.3, 45.4, 29.2 and 59.1 m2.
MIN_FLOOR_NAVIGABLE_AREA_M2 = 6.0

# The navigable pool describes the room, not the request, so it is drawn from a
# fixed generator and never from the request seed.
NAVIGABLE_POOL_SAMPLES = 400
NAVIGABLE_POOL_SEED = 20260914
NAVIGABLE_POOL_CAP = 2000

DEFAULT_SAME_FLOOR_TOLERANCE_M = 0.3

_CACHE_ATTRIBUTE = '_avengine_floor_level_decisions'
_CACHE_LIMIT = 8


def navigable_sample_pool(space, *, samples=NAVIGABLE_POOL_SAMPLES,
                          seed=NAVIGABLE_POOL_SEED, cap=NAVIGABLE_POOL_CAP):
    """Read this space's navigable points, or draw a fixed-seed pool of them.

    A raster space can list every navigable cell, and a native navmesh can only
    be asked for random points, so both are supported and the source is named
    in the returned record.
    """
    reader = getattr(space, 'points', None)
    if callable(reader):
        raw = None
        try:
            raw = np.asarray(reader(None), dtype=float)
        except (ValueError, TypeError, IndexError, RuntimeError, KeyError):
            raw = None
        if raw is not None and raw.ndim == 2 and raw.shape[1] == 3 and len(raw):
            if len(raw) > cap:
                raw = raw[np.linspace(0, len(raw) - 1, cap).astype(int)]
            return np.asarray(raw, dtype=float), 'space_navigable_points'
    rng = np.random.default_rng(int(seed))
    drawn = []
    for _ in range(int(samples)):
        try:
            point = np.asarray(space.sample_navigable(rng, None), dtype=float)
        except (ValueError, RuntimeError):
            continue
        if point.shape == (3,) and bool(np.all(np.isfinite(point))):
            drawn.append(point)
    return np.asarray(drawn, dtype=float).reshape(-1, 3), 'fixed_seed_sample_navigable'


def navigable_level_clusters(points, *, tolerance_m=DEFAULT_SAME_FLOOR_TOLERANCE_M):
    """Group navigable heights into levels, densest band first.

    Single-linkage grouping would chain a staircase into one level that spans
    the whole building, so each level is taken as the tolerance-wide band that
    holds the most remaining samples, and those samples are then removed.
    """
    heights = np.asarray(points, dtype=float).reshape(-1, 3)[:, 1]
    heights = np.sort(heights[np.isfinite(heights)])
    tolerance = float(tolerance_m)
    levels = []
    while len(heights):
        left = np.searchsorted(heights, heights - tolerance, side='left')
        right = np.searchsorted(heights, heights + tolerance, side='right')
        best = int(np.argmax(right - left))
        centre = float(np.median(heights[left[best]:right[best]]))
        member = np.abs(heights - centre) <= tolerance
        levels.append((centre, int(member.sum())))
        heights = heights[~member]
    return sorted(levels)


def navigable_level_areas(space, levels: Sequence[float], *,
                          tolerance_m=DEFAULT_SAME_FLOOR_TOLERANCE_M, pool=None):
    """Navigable square metres on each level, measured the way the space allows.

    A raster space lists its cells and declares how big one is, so each level is
    counted exactly. A native navmesh cannot be enumerated but knows its own
    total navigable area, so each level takes the share of the fixed-seed pool
    that falls on it. When a space offers neither, every level reads ``None``
    and no area rule may be applied to it.

    Returns the areas and the name of the measurement that produced them.
    """
    levels = [float(value) for value in levels]
    tolerance = float(tolerance_m)
    metadata = getattr(space, 'metadata', None)
    resolution = metadata.get('resolution_m') if isinstance(metadata, Mapping) else None
    reader = getattr(space, 'points', None)
    if callable(reader) and resolution:
        cells, cell_area = None, 0.
        try:
            cells = np.asarray(reader(None), dtype=float)
            cell_area = float(resolution) ** 2
        except (ValueError, TypeError, IndexError, RuntimeError, KeyError):
            cells = None
        if (cells is not None and cells.ndim == 2 and cells.shape[1] == 3
                and len(cells) and math.isfinite(cell_area) and cell_area > 0):
            return ([float(np.count_nonzero(np.abs(cells[:, 1] - value) <= tolerance)
                           * cell_area) for value in levels],
                    'navigable_cells_times_declared_cell_area')
    total = getattr(getattr(space, 'pathfinder', None), 'navigable_area', None)
    try:
        total = float(total) if total is not None else None
    except (TypeError, ValueError):
        total = None
    if total is not None and math.isfinite(total) and total > 0:
        shares, pool_size = navigable_level_shares(
            space, levels, tolerance_m=tolerance, pool=pool)
        if pool_size:
            return ([float(total * share) for share in shares],
                    'navmesh_navigable_area_times_pool_share')
    return [None] * len(levels), 'unavailable'


def _band_mesh(mesh, low, high):
    """Keep only triangles whose height range meets the probe band."""
    keep = (mesh.maximum[:, 1] >= low) & (mesh.minimum[:, 1] <= high)
    if bool(keep.all()):
        return mesh, int(len(mesh.triangles))
    return MeshHandle(mesh.vertices, mesh.triangles[keep], mesh.source), int(keep.sum())


def _merge_candidate_heights(declared, clustered, tolerance_m):
    """Union of declared and measured levels, declared values kept verbatim."""
    heights = []
    sources = {}
    for value in sorted(float(v) for v in declared):
        if any(abs(value - kept) <= tolerance_m for kept in heights):
            continue
        heights.append(value)
        sources[value] = ['room_declaration']
    for value in sorted(float(v) for v in clustered):
        near = [kept for kept in heights if abs(value - kept) <= tolerance_m]
        if near:
            sources[near[0]].append('navigable_cluster')
            continue
        heights.append(value)
        sources[value] = ['navigable_cluster']
    return sorted(heights), sources


def interior_floor_levels(space, mesh, *, candidates=None,
                          tolerance_m=DEFAULT_SAME_FLOOR_TOLERANCE_M, pool=None,
                          pool_source=None, probe_samples=INTERIOR_PROBE_SAMPLES,
                          probe_minimum=INTERIOR_PROBE_MINIMUM,
                          start_offset_m=INTERIOR_PROBE_START_OFFSET_M,
                          ceiling_probe_m=INTERIOR_CEILING_PROBE_M,
                          covered_share=INTERIOR_COVERED_SHARE,
                          minimum_area_m2=MIN_FLOOR_NAVIGABLE_AREA_M2):
    """Judge every candidate level of ``space`` against the room's own mesh.

    ``candidates`` are the levels the room and its navigation already declare;
    the levels measured from the navigable points are added to them, so a room
    that declares nothing still gets floors from geometry rather than from a
    single random navigable point. The returned record names, for each level,
    the share of the navigable samples it holds, the share of its upward probes
    that met room geometry, and the reason it was kept or dropped.

    A level that is indoors but holds less than ``minimum_area_m2`` of navigable
    surface is dropped as well, because a bed, a desk and a stair run are all
    indoors and none of them is a room. The area is measured in metres rather
    than as a share of the building, so the rule means the same thing in a
    large house and a small flat.

    Nothing is dropped when the judgement cannot be made: without a mesh,
    without navigable samples, without a way to measure area, or when no level
    at all survives, the candidate levels are returned unchanged and ``status``
    says why.
    """
    tolerance = float(tolerance_m)
    if pool is None:
        pool, pool_source = navigable_sample_pool(space)
    pool = np.asarray(pool, dtype=float).reshape(-1, 3)
    clustered = navigable_level_clusters(pool, tolerance_m=tolerance) if len(pool) else []
    heights, sources = _merge_candidate_heights(
        candidates or (), [height for height, _ in clustered], tolerance)
    criterion = {
        'kind': 'upward_ray_to_room_geometry_v1',
        'ceiling_probe_m': float(ceiling_probe_m),
        'probe_start_offset_m': float(start_offset_m),
        'covered_share_required': float(covered_share),
        'probe_samples': int(probe_samples),
        'same_floor_tolerance_m': tolerance,
        'minimum_navigable_area_m2': float(minimum_area_m2),
    }
    record = {
        'schema': 'avengine_interior_floor_levels_v1',
        'criterion': criterion,
        'navigable_pool_size': int(len(pool)),
        'navigable_pool_source': pool_source,
        'candidate_heights_m': [float(value) for value in heights],
        'levels': [],
        'legal_heights_m': [float(value) for value in heights],
        'claim_boundary': 'a planning guard measured from room geometry; it '
                          'certifies no rendered frame',
    }
    if mesh is None or not len(heights) or not len(pool):
        record['status'] = 'unmeasured'
        record['reason'] = (
            'no static mesh, so no interior judgement was made' if mesh is None
            else 'navigation offered no sampled points, so no interior judgement was made'
            if not len(pool) else 'no candidate level to judge')
        record['navigable_area_source'] = 'not_measured'
        record['levels'] = [
            {'height_m': float(value), 'sources': sources.get(value, []),
             'verdict': 'unmeasured', 'reason': record['reason'],
             'navigable_share': float(np.count_nonzero(
                 np.abs(pool[:, 1] - value) <= tolerance) / len(pool)) if len(pool) else None,
             'probe_samples': 0, 'covered_share': None, 'area_m2': None}
            for value in heights]
        return record

    areas, area_source = navigable_level_areas(
        space, heights, tolerance_m=tolerance, pool=pool)
    record['navigable_area_source'] = area_source
    for index, value in enumerate(heights):
        member = pool[np.abs(pool[:, 1] - value) <= tolerance]
        share = float(len(member) / len(pool))
        area = areas[index]
        row = {'height_m': float(value), 'sources': sources.get(value, []),
               'navigable_share': share, 'navigable_samples': int(len(member)),
               'area_m2': None if area is None else float(area)}
        if len(member) < probe_minimum:
            row.update(verdict='undetermined', covered_share=None, probe_samples=int(len(member)),
                       reason='fewer navigable samples than the probe minimum')
            record['levels'].append(row)
            continue
        probes = member
        if len(probes) > probe_samples:
            probes = probes[np.linspace(0, len(probes) - 1, probe_samples).astype(int)]
        low = float(value) + float(start_offset_m)
        high = float(value) + float(ceiling_probe_m)
        band, band_triangles = _band_mesh(mesh, low, high)
        covered = 0
        for point in probes:
            start = [float(point[0]), float(point[1]) + float(start_offset_m), float(point[2])]
            end = [float(point[0]), float(point[1]) + float(ceiling_probe_m), float(point[2])]
            if line_of_sight(band, start, end) == 'blocked':
                covered += 1
        measured = float(covered / len(probes))
        row.update(probe_samples=int(len(probes)), covered_share=measured,
                   band_triangles=int(band_triangles))
        if measured < float(covered_share):
            row.update(verdict='open_to_sky',
                       reason='most upward probes left the room without meeting geometry')
        elif area is not None and float(area) < float(minimum_area_m2):
            row.update(verdict='too_small',
                       reason='indoors, but it holds less navigable surface than an '
                              'episode needs, so it is a ledge, a stair run or a '
                              'piece of furniture rather than a room')
        else:
            row.update(verdict='interior',
                       reason=('most upward probes met room geometry within the probe height'
                               if area is not None else
                               'most upward probes met room geometry within the probe '
                               'height; this space cannot measure level area, so no '
                               'minimum-area rule was applied'))
        record['levels'].append(row)

    interior = [row['height_m'] for row in record['levels'] if row['verdict'] == 'interior']
    if interior:
        record['status'] = 'measured'
        record['reason'] = ('levels open to the sky, and indoor levels too small to '
                            'hold an episode, were dropped')
        record['legal_heights_m'] = interior
        return record
    small = [row['height_m'] for row in record['levels'] if row['verdict'] == 'too_small']
    if small:
        record['status'] = 'no_level_above_minimum_area'
        record['reason'] = (
            'every indoor level of this room holds less than the minimum navigable '
            'area, so they were all kept rather than leaving the room with no floor')
        record['legal_heights_m'] = small
        return record
    record['status'] = 'no_interior_level'
    record['reason'] = ('no candidate level had room geometry overhead, so every '
                        'candidate was kept and none was judged interior')
    return record


def floor_level_decision(space, mesh, *, candidates=None,
                         tolerance_m=DEFAULT_SAME_FLOOR_TOLERANCE_M, **options):
    """:func:`interior_floor_levels` once per space, mesh and candidate set.

    The planner asks for a floor many times while it redraws candidates, and
    the answer is a property of the room, so it is measured once and reused.
    """
    key = (tuple(round(float(value), 6) for value in sorted(candidates or ())),
           round(float(tolerance_m), 6),
           tuple(sorted((str(name), repr(value)) for name, value in options.items())))
    store = getattr(space, _CACHE_ATTRIBUTE, None)
    if isinstance(store, dict):
        cached = store.get(key)
        if cached is not None and cached[0] is mesh:
            return cached[1]
    record = interior_floor_levels(
        space, mesh, candidates=candidates, tolerance_m=tolerance_m, **options)
    if not isinstance(store, dict):
        try:
            setattr(space, _CACHE_ATTRIBUTE, {})
        except (AttributeError, TypeError):
            return record
        store = getattr(space, _CACHE_ATTRIBUTE, None)
    if isinstance(store, dict):
        if len(store) >= _CACHE_LIMIT:
            store.clear()
        store[key] = (mesh, record)
    return record


def decision_summary(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The part of a decision a plan keeps: the verdicts and why."""
    if not isinstance(record, Mapping):
        return None
    return {
        'schema': record.get('schema'),
        'status': record.get('status'),
        'reason': record.get('reason'),
        'criterion': record.get('criterion'),
        'navigable_pool_size': record.get('navigable_pool_size'),
        'navigable_pool_source': record.get('navigable_pool_source'),
        'navigable_area_source': record.get('navigable_area_source'),
        'legal_heights_m': [float(value) for value in record.get('legal_heights_m') or ()],
        'levels': [
            {key: row.get(key) for key in (
                'height_m', 'sources', 'verdict', 'reason', 'navigable_share',
                'navigable_samples', 'area_m2', 'probe_samples', 'covered_share')}
            for row in record.get('levels') or ()
        ],
        'claim_boundary': record.get('claim_boundary'),
    }


def navigable_level_shares(space, levels: Sequence[float], *,
                           tolerance_m=DEFAULT_SAME_FLOOR_TOLERANCE_M, pool=None):
    """Share of the room's navigable samples each level holds.

    The denominator is every navigable sample the room offers, not only the
    samples on the levels still under consideration, so dropping a level does
    not promote a sliver of navigation - a step, or a bed Recast walked onto -
    into a floor worth planning on.
    """
    if pool is None:
        pool, _ = navigable_sample_pool(space)
    pool = np.asarray(pool, dtype=float).reshape(-1, 3)
    if not len(pool):
        return np.zeros(len(levels), dtype=float), 0
    counts = np.asarray([
        np.count_nonzero(np.abs(pool[:, 1] - float(value)) <= float(tolerance_m))
        for value in levels], dtype=float)
    return counts / float(len(pool)), int(len(pool))
