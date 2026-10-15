"""Reusable conservative static room masks; native full-pose gates stay authoritative.

Arrays use [z, x]. origin_xz_m is the lower corner of cell [0, 0].
No native simulator or mesh is imported when reading a map.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import json
import math
import os
import numpy as np

VERSION = 'walkable-map-v1'
RESOLUTION_M = .05
SPECIES_RADII_M = {'human': .35, 'dog': .25, 'cat': .20, 'slide': .20}
LAYER_KEYS = {species: 'walkable_' + species for species in SPECIES_RADII_M}


def source_identity(paths):
    """Ordinary file identity invalidates stale geometry/navmesh caches."""
    result = []
    for path in paths:
        path = Path(path).resolve()
        stat = path.stat()
        result.append({'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    return result


def room_map_path(root, room_id):
    if not room_id or Path(str(room_id)).name != str(room_id) or str(room_id) in {'.', '..'}:
        raise ValueError('walkable map needs a single room ID, without path components')
    return Path(root) / str(room_id) / 'walkable_map_v1.npz'


class WalkableMap:
    def __init__(self, origin, resolution, floor_reference, layers, metadata, floor_y=None):
        self.origin = np.asarray(origin, float)
        self.resolution = float(resolution)
        self.floor_reference = float(floor_reference)
        self.layers = {k: np.asarray(v) for k, v in layers.items()}
        self.metadata = metadata
        self.floor_y = floor_y
        shapes = {a.shape for a in self.layers.values()}
        if (self.origin.shape != (2,) or not np.isfinite(self.origin).all()
                or not math.isfinite(self.floor_reference)
                or abs(self.resolution - RESOLUTION_M) > 1e-10
                or set(self.layers) != set(SPECIES_RADII_M)
                or len(shapes) != 1 or any(a.ndim != 2 or a.dtype != np.bool_ for a in self.layers.values())):
            raise ValueError('invalid walkable-map-v1 geometry or boolean layers')
        self.shape = next(iter(shapes))
        if not all(self.shape):
            raise ValueError('walkable map must have cells')
        if metadata.get('version') != VERSION:
            raise ValueError('unsupported walkable map version')
        if any(metadata.get('species_radii_m', {}).get(s) != r for s, r in SPECIES_RADII_M.items()):
            raise ValueError('walkable map body radius policy mismatch')
        if np.any(self.layers['human'] & ~self.layers['dog']) or np.any(self.layers['dog'] & ~self.layers['cat']) or not np.array_equal(self.layers['cat'], self.layers['slide']):
            raise ValueError('walkable map species layers are not nested')

    @classmethod
    def load(cls, path, *, validate_sources=True):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata_json'].item()))
            if str(data['version'].item()) != VERSION:
                raise ValueError('unsupported walkable map version')
            result = cls(data['origin_xz_m'], data['resolution_m'].item(),
                         data['floor_reference_y_m'].item(),
                         {s: data[k].copy() for s, k in LAYER_KEYS.items()}, metadata,
                         data['floor_y_m'].copy())
        if validate_sources:
            expected = metadata['source_identity']
            if source_identity([r['path'] for r in expected]) != expected:
                raise ValueError('walkable map static geometry or NavMesh changed; rebuild required')
        return result

    def contains_many(self, points, species='human'):
        if species not in self.layers:
            raise ValueError('unsupported walkable map species: ' + str(species))
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('walkable map queries require N x 2 XZ metres')
        finite = np.isfinite(points).all(axis=1)
        cells = np.floor((np.where(np.isfinite(points), points, self.origin) - self.origin) / self.resolution).astype(np.int64)
        x, z = cells.T
        valid = finite & (x >= 0) & (z >= 0) & (x < self.shape[1]) & (z < self.shape[0])
        result = np.zeros(len(points), bool)
        result[valid] = self.layers[species][z[valid], x[valid]]
        return result

    def contains(self, point, species='human'):
        return bool(self.contains_many(np.asarray(point, float).reshape(1, 2), species)[0])

    def segment_free(self, start, end, species='human'):
        """Supercover: include every crossed cell and both sides of grid corners."""
        a = (np.asarray(start, float) - self.origin) / self.resolution
        b = (np.asarray(end, float) - self.origin) / self.resolution
        if not np.isfinite([a, b]).all():
            return False
        cuts = [0., 1.]
        delta = b-a
        for axis in (0, 1):
            if abs(delta[axis]) > 1e-14:
                lo, hi = sorted((a[axis], b[axis]))
                boundaries = np.arange(math.ceil(lo), math.floor(hi)+1)
                cuts.extend(((boundaries-a[axis])/delta[axis]).tolist())
        cuts = np.unique(np.clip(cuts, 0., 1.))
        t = np.r_[cuts, (cuts[:-1]+cuts[1:])/2]
        p = a + t[:, None]*delta
        # At an exact edge/corner both adjacent cells must be free.
        eps = 1e-9
        samples = np.concatenate([p+[dx, dz] for dx in (-eps, eps) for dz in (-eps, eps)])
        return bool(self.contains_many(self.origin+samples*self.resolution, species).all())


def _load_cached(path, size, mtime_ns):
    return WalkableMap.load(path)
_load_cached = lru_cache(maxsize=32)(_load_cached)


def load_room_map(room_id, *, path=None, floor_reference=None):
    """Configured production/acceptance runs fail closed on missing/stale maps.

    Unconfigured older callers retain their proposal-only NavMesh behavior.
    Set AVENGINE_WALKABLE_MAP_ROOT for both planning and native execution.
    The route also carries its exact map path into native fitting.
    """
    root = os.environ.get('AVENGINE_WALKABLE_MAP_ROOT')
    if path is None and root:
        path = room_map_path(root, room_id)
    if path is None:
        if os.environ.get('AVENGINE_REQUIRE_WALKABLE_MAP') == '1':
            raise ValueError('walkable map is required but no cache root/path was configured')
        return None
    path = Path(path).resolve()
    stat = path.stat()
    result = _load_cached(str(path), stat.st_size, stat.st_mtime_ns)
    # Source files can change while a long-running planner retains a loaded map.
    expected = result.metadata['source_identity']
    if source_identity([r['path'] for r in expected]) != expected:
        raise ValueError('walkable map source changed; rebuild required')
    if room_id and result.metadata['room_id'] != room_id:
        raise ValueError('walkable map belongs to another room')
    if floor_reference is not None and abs(result.floor_reference-float(floor_reference)) > .0001:
        raise ValueError('walkable map measured floor differs from room reference')
    result.path = str(path)
    return result
