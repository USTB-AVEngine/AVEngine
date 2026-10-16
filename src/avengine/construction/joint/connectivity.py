"""Cached four-neighbour components of the unchanged species walkable mask."""
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class WalkableComponents:
    labels_zx: np.ndarray
    sizes: np.ndarray
    origin_xz_m: tuple
    resolution_m: float
    species: str
    mask_sha256: str

    @property
    def component_count(self):
        return len(self.sizes)-1

    def labels_for_points(self, points_m):
        p = np.asarray(points_m, float)
        if p.ndim == 1:
            p = p[None, :]
        if p.ndim != 2 or p.shape[1] not in (2, 3):
            raise ValueError("expected XZ or XYZ points")
        xz = p[:, [0, 2]] if p.shape[1] == 3 else p
        result = np.zeros(len(p), np.int32)
        finite = np.isfinite(xz).all(axis=1)
        cells = np.zeros((len(p), 2), np.int64)
        cells[finite] = np.floor((xz[finite]-self.origin_xz_m)/self.resolution_m).astype(np.int64)
        good = finite & (cells[:, 0] >= 0) & (cells[:, 0] < self.labels_zx.shape[1])
        good &= (cells[:, 1] >= 0) & (cells[:, 1] < self.labels_zx.shape[0])
        result[good] = self.labels_zx[cells[good, 1], cells[good, 0]]
        return result

    def connected(self, start_m, end_m):
        labels = self.labels_for_points([start_m, end_m])
        return bool(labels[0] != 0 and labels[0] == labels[1])


def walkable_components(context, species="human"):
    """Return cached labels; zero means outside the selected walkable domain.

    Eight-neighbour A* in route_between admits a diagonal only if BOTH adjoining
    cardinal cells are free. Every such edge has a cardinal two-edge path, so
    four-neighbour component membership has exactly the same reachability.
    It is not native NavMesh/whole-body/scene certification.
    """
    from scipy.ndimage import label
    m = context.walkable_map
    mask = np.ascontiguousarray(m.layers[species], dtype=bool)
    digest = hashlib.sha256(mask.tobytes()).hexdigest()
    key = (digest, mask.shape, tuple(map(float, m.origin)), float(m.resolution))
    cache = getattr(context, "_joint_components_cache", None)
    if cache is None:
        cache = {}
        context._joint_components_cache = cache
    if species not in cache or cache[species][0] != key:
        labels, _ = label(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))
        labels = labels.astype(np.int32, copy=False)
        sizes = np.bincount(labels.ravel())
        labels.setflags(write=False)
        sizes.setflags(write=False)
        result = WalkableComponents(labels, sizes, key[2], key[3], species, digest)
        cache[species] = key, result
    return cache[species][1]


def reachable_point_mask(context, start_m, *, species="human", points_m=None):
    """Filter endpoint domains before route search, preserving context grid order."""
    components = walkable_components(context, species)
    start = components.labels_for_points(start_m)[0]
    points = context.points_m if points_m is None else points_m
    labels = components.labels_for_points(points)
    return (labels == start) & (start != 0)
