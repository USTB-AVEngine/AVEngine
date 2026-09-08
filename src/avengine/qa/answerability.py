"""Shared geometric/statistical facts; thresholds do not certify answerability."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def listener_azimuth_deg(listener_pose: Mapping[str, Any], target_position: Sequence[float]) -> float | None:
    """Horizontal listener azimuth, exactly the catalog convention (+right)."""
    p = np.asarray(listener_pose['position_m'], dtype=float)
    d = np.asarray(target_position, dtype=float) - p
    basis = listener_pose['basis']
    # The existing catalog deliberately projects onto the horizontal plane.
    f, r = np.asarray(basis['forward']), np.asarray(basis['right'])
    fd, rd = float(d[0] * f[0] + d[2] * f[2]), float(d[0] * r[0] + d[2] * r[2])
    if math.isclose(fd, 0., abs_tol=1e-12) and math.isclose(rd, 0., abs_tol=1e-12):
        return None
    return float((math.degrees(math.atan2(rd, fd)) + 180.) % 360. - 180.)


def separation_stats(az_target: Sequence[float], az_others: Mapping[str, Sequence[float]],
                     window: Sequence[int], *, frame_rate_hz: float = 1.,
                     thresholds_deg: Sequence[float] = ()) -> dict[str, Any]:
    """Nearest competitor over a half-open frame window; keep offscreen sources."""
    start, end = map(int, window)
    target = np.asarray(az_target, dtype=float)
    if not 0 <= start < end <= len(target) or frame_rate_hz <= 0:
        raise ValueError('invalid half-open separation window or frame rate')
    ids = sorted(az_others)
    if not ids:
        return {'status': 'unmeasured', 'reason': 'no_competitors'}
    other = np.asarray([az_others[i] for i in ids], dtype=float)
    if other.shape != (len(ids), len(target)) or not np.all(np.isfinite(other)) or not np.all(np.isfinite(target)):
        raise ValueError('azimuth arrays must be finite and share their frame clock')
    diff = np.abs((other[:, start:end] - target[None, start:end] + 180.) % 360. - 180.)
    nearest = diff.min(axis=0)
    nearest_ids = [[ids[i] for i in np.flatnonzero(np.isclose(diff[:, j], nearest[j], atol=1e-10, rtol=0))]
                   for j in range(end-start)]
    sustained = {}
    for theta in thresholds_deg:
        longest = run = 0
        for good in nearest >= float(theta):
            run = run + 1 if good else 0
            longest = max(longest, run)
        sustained[str(float(theta))] = longest / frame_rate_hz
    return {'status': 'measured', 'min': float(nearest.min()), 'p10': float(np.quantile(nearest, .1)),
            'p50': float(np.median(nearest)), 'max': float(nearest.max()),
            'nearest_competitor_ids': nearest_ids, 'nearest_competitor_changed': any(x != nearest_ids[0] for x in nearest_ids),
            'sustained_s_above': sustained, 'window_frames': [start, end], 'frame_rate_hz': float(frame_rate_hz)}


def max_concurrent_entities(intervals: Sequence[tuple[int, int, str]]) -> int:
    """Sweep half-open intervals; multiple overlapping events of one entity count once."""
    edges = defaultdict(list)
    for start, end, entity in intervals:
        if start < 0 or end <= start:
            raise ValueError('activity intervals must be positive half-open intervals')
        edges[start].append((entity, 1)); edges[end].append((entity, -1))
    active = Counter(); maximum = 0
    for time in sorted(edges):
        for entity, delta in edges[time]:
            active[entity] += delta
            if not active[entity]:
                del active[entity]
        maximum = max(maximum, len(active))
    return maximum


def structural_baselines(candidate_values: Mapping[str, Any], gold_actor: str,
                         *, answer_domain_size: int | None = None) -> dict[str, Any]:
    """Value strategies, with undefined majority/minority falling back to random.

    A majority strategy chooses a value, not one of the entities sharing that
    value. The caller supplies the actual MCQ K. Without it, random means a
    uniform candidate-entity draw; that is not a calibrated Open chance level.
    """
    values = list(candidate_values.values())
    counts = {str(value): sum(other == value for other in values if other is not None)
              for value in values if value is not None}
    missing = [actor for actor, value in candidate_values.items() if value is None]
    n = len(candidate_values)
    k = int(answer_domain_size) if answer_domain_size is not None else n
    if k < 0:
        raise ValueError('answer domain size cannot be negative')
    result = {'candidate_value_multiplicity': counts, 'candidate_count': n, 'k': k,
              'random_policy': 'uniform_answer_domain' if answer_domain_size is not None else 'uniform_candidate_entity',
              'gold_is_majority': None, 'gold_is_unique_minority': None,
              'majority_hits': None, 'unique_minority_hits': None, 'random_hits': None,
              'majority_available': False, 'unique_minority_available': False}
    if gold_actor not in candidate_values or missing or not n:
        return {**result, 'status': 'unmeasured', 'reason': 'missing_candidate_value', 'missing_actor_ids': missing}
    gold = candidate_values[gold_actor]
    maximum = max(counts.values())
    largest = [value for value, count in counts.items() if count == maximum]
    singles = [value for value, count in counts.items() if count == 1]
    majority = largest[0] if len(largest) == 1 and maximum > 1 else None
    minority = singles[0] if len(singles) == 1 and len(counts) > 1 else None
    random_hit = (1. / k if k else None) if answer_domain_size is not None else counts[str(gold)] / n
    return {**result, 'status': 'measured',
            'majority_available': majority is not None, 'unique_minority_available': minority is not None,
            'majority_hits': float(str(gold) == majority) if majority is not None else random_hit,
            'unique_minority_hits': float(str(gold) == minority) if minority is not None else random_hit,
            'random_hits': random_hit, 'gold_is_majority': str(gold) == majority,
            'gold_is_unique_minority': str(gold) == minority}


@dataclass
class MeshHandle:
    """Loaded static triangles, all in the shared meter/+Y/right-handed frame."""
    vertices: np.ndarray
    triangles: np.ndarray
    source: Any = None

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=float)
        self.triangles = np.asarray(self.triangles, dtype=np.int64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or not np.all(np.isfinite(self.vertices)):
            raise ValueError('mesh vertices must be finite [vertex,3]')
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3 or (self.triangles.size and (self.triangles.min() < 0 or self.triangles.max() >= len(self.vertices))):
            raise ValueError('mesh triangles must index [triangle,3]')
        self.minimum = self.vertices[self.triangles].min(axis=1)
        self.maximum = self.vertices[self.triangles].max(axis=1)

    @classmethod
    def from_paths(cls, vertices_path: str | Path, triangles_path: str | Path):
        return cls(np.load(vertices_path, allow_pickle=False), np.load(triangles_path, allow_pickle=False),
                   {'vertices': str(vertices_path), 'triangles': str(triangles_path)})


def line_of_sight(mesh_handle: MeshHandle | Mapping[str, Any] | None,
                  from_xyz: Sequence[float], to_xyz: Sequence[float]) -> str:
    """Exact retained triangle test with conservative bounding-box rejection."""
    from avengine.rooms.furniture_layout import _mesh_ray_occluded
    if mesh_handle is None:
        return 'unmeasured'
    if isinstance(mesh_handle, Mapping):
        vertices, triangles = mesh_handle.get('vertices'), mesh_handle.get('triangles')
        if isinstance(vertices, (str, Path)) and isinstance(triangles, (str, Path)):
            if not Path(vertices).is_file() or not Path(triangles).is_file():
                return 'unmeasured'
            mesh_handle = MeshHandle.from_paths(vertices, triangles)
        elif vertices is not None and triangles is not None:
            mesh_handle = MeshHandle(vertices, triangles)
        else:
            return 'unmeasured'
    origin, target = np.asarray(from_xyz, dtype=float), np.asarray(to_xyz, dtype=float)
    if origin.shape != (3,) or target.shape != (3,) or not np.all(np.isfinite([origin, target])):
        raise ValueError('LOS endpoints must be finite 3-vectors')
    lo, hi = np.minimum(origin, target)-1e-4, np.maximum(origin, target)+1e-4
    relevant = np.all(mesh_handle.maximum >= lo, axis=1) & np.all(mesh_handle.minimum <= hi, axis=1)
    blocked = _mesh_ray_occluded(origin, target, mesh_handle.vertices, mesh_handle.triangles[relevant])
    return 'blocked' if blocked else 'clear'
