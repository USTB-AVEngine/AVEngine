"""Fixed-camera projection and real-visual-triangle depth prediction.

A body envelope is a planning proxy. Bounds are conditional on that envelope;
only unchanged native masks decide pixel truth. Unknown identity stays unknown.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from typing import Any
import math
import numpy as np
from .types import BodySpec, CameraPose


@dataclass
class VisibilityPrediction:
    state: str
    visible_pixels_estimate: int = 0
    visible_pixels_lower: int = 0
    visible_pixels_upper: int = 0
    occluded_fraction: float = 0.
    edge_margin_px: float = 0.
    depth_margin_m: float = 0.
    occluder_ids: tuple[str, ...] = ()
    confidence: str = "unknown"
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class DepthBuffer:
    camera: CameraPose
    depth_m: np.ndarray  # positive camera-forward depth, NOT radial distance
    object_ids: np.ndarray | None = None
    source: str = "visual_mesh"
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.depth_m = np.asarray(self.depth_m, float)
        if self.depth_m.shape != (self.camera.height, self.camera.width):
            raise ValueError("depth dimensions differ from the camera")
        if np.any(self.depth_m < 0):
            raise ValueError("depth must be nonnegative forward metres")
        if self.object_ids is not None and np.shape(self.object_ids) != self.depth_m.shape:
            raise ValueError("depth identity dimensions differ")


class MeshRaycaster:
    """Original world-space visual triangles; optional per-face registered IDs.

    Source identity must describe any transform and the original input path.
    Acoustic proxies are accepted for diagnostics, never high confidence.
    """
    def __init__(self, mesh, *, face_ids=None, source="visual_mesh", evidence=None):
        self.mesh = mesh
        self.source = source
        self.evidence = dict(evidence or {})
        self.face_ids = None if face_ids is None else np.asarray(["" if x is None else str(x) for x in face_ids], dtype=str)
        if self.face_ids is not None and len(self.face_ids) != len(mesh.faces):
            raise ValueError("one identity per original face is required")
        try:
            from trimesh.ray.ray_pyembree import RayMeshIntersector
        except ImportError:
            from trimesh.ray.ray_triangle import RayMeshIntersector
        self.intersector = RayMeshIntersector(mesh)

    def first_hit(self, origins, directions):
        origins, directions = np.asarray(origins, float), np.asarray(directions, float)
        directions = directions.reshape(-1, 3)
        origins = np.broadcast_to(origins, directions.shape)
        norms = np.linalg.norm(directions, axis=1)
        if not np.isfinite(origins).all() or not np.isfinite(directions).all() or np.any(norms <= 0):
            raise ValueError("rays must be finite and nonzero")
        ranges = np.full(len(directions), np.inf)
        ids = np.full(len(directions), "", dtype=self.face_ids.dtype if self.face_ids is not None else "U1")
        if not len(self.mesh.faces):
            return ranges, ids
        for begin in range(0, len(directions), 16384):
            end = min(begin + 16384, len(directions))
            loc, ri, ti = self.intersector.intersects_location(
                origins[begin:end], directions[begin:end] / norms[begin:end, None], multiple_hits=False)
            ranges[begin + ri] = np.linalg.norm(loc - origins[begin + ri], axis=1)
            if self.face_ids is not None:
                ids[begin + ri] = self.face_ids[ti]
        return ranges, ids


class CompositeRaycaster:
    """Combine a scan and exact registered occluder meshes without modifying either."""
    def __init__(self, *raycasters):
        if not raycasters:
            raise ValueError("at least one raycaster required")
        self.raycasters = raycasters
        self.source = "visual_mesh" if all(r.source == "visual_mesh" for r in raycasters) else "unknown"
        self.evidence = {"components": [getattr(r, "evidence", {}) for r in raycasters]}

    def first_hit(self, origins, directions):
        ranges, ids = self.raycasters[0].first_hit(origins, directions)
        ids = ids.astype(object)
        for ray in self.raycasters[1:]:
            r, i = ray.first_hit(origins, directions)
            use = r < ranges
            ranges[use], ids[use] = r[use], i[use]
        return ranges, ids


def camera_coordinates(camera, points):
    delta = np.asarray(points, float) - np.asarray(camera.position_m)
    return np.stack((delta @ camera.right, delta[..., 1], delta @ camera.forward), axis=-1)


def project_points(camera, points):
    xyz = camera_coordinates(camera, points)
    z = xyz[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack((camera.width / 2 + camera.focal_px * xyz[..., 0] / z,
                       camera.height / 2 - camera.focal_px * xyz[..., 1] / z), axis=-1)
    return uv, z


def body_box(root_m, body, *, margin=True):
    root = np.asarray(root_m, float)
    extra = body.margin_m if margin else 0.
    r = body.radius_m + extra
    # Expanded in height as well: cover head/foot/root-fit uncertainty.
    return root + np.array(list(product((-r, r), (-extra, body.height_m + extra), (-r, r))))


def framing(camera, root_m, body=BodySpec()):
    """Whole expanded box projection: inside / outside / edge, signed pixel margin."""
    points = body_box(root_m, body)
    xyz = camera_coordinates(camera, points)
    x, y, z = xyz.T
    tan_h, tan_v = camera.width / (2 * camera.focal_px), camera.height / (2 * camera.focal_px)
    planes = np.column_stack((z - .01, z * tan_h - x, z * tan_h + x, z * tan_v - y, z * tan_v + y))
    if np.any(np.all(planes < 0, axis=0)):
        return "outside", -math.inf
    if np.any(z <= .01):
        return "edge", -math.inf
    uv, _ = project_points(camera, points)
    margin = float(min(uv[:, 0].min(), camera.width - uv[:, 0].max(),
                       uv[:, 1].min(), camera.height - uv[:, 1].max()))
    return ("inside" if margin >= 0 else "edge"), margin


def pixel_rays(camera, u, v):
    return (camera.forward + np.asarray(u)[..., None] * camera.right / camera.focal_px
            - np.asarray(v)[..., None] * np.array([0., 1., 0.]) / camera.focal_px)


def build_depth_buffer(camera, raycaster):
    yy, xx = np.mgrid[:camera.height, :camera.width]
    directions = pixel_rays(camera, xx.ravel() + .5 - camera.width / 2,
                            yy.ravel() + .5 - camera.height / 2)
    ranges, ids = raycaster.first_hit(np.asarray(camera.position_m), directions)
    depth = ranges / np.linalg.norm(directions, axis=1)
    identity = ids.astype(str).reshape(yy.shape) if np.any(ids != "") else None
    return DepthBuffer(camera, depth.reshape(yy.shape), identity, raycaster.source,
                       {"method": "original_visual_triangles_pixel_center_rays", **getattr(raycaster, "evidence", {})})


def _cylinder_depth(camera, directions, root, radius, height):
    """First positive intersection with a closed vertical cylinder, forward metres."""
    q = np.asarray(camera.position_m) - np.asarray(root)
    d = np.asarray(directions, float)
    a = d[..., 0] ** 2 + d[..., 2] ** 2
    b = 2 * (q[0] * d[..., 0] + q[2] * d[..., 2])
    c = q[0] ** 2 + q[2] ** 2 - radius ** 2
    disc = b * b - 4 * a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
        hi = (-b + np.sqrt(np.maximum(disc, 0))) / (2 * a)
        y0, y1 = -q[1] / d[..., 1], (height - q[1]) / d[..., 1]
    parallel = np.abs(d[..., 1]) < 1e-14
    yl, yh = np.minimum(y0, y1), np.maximum(y0, y1)
    yl = np.where(parallel, -np.inf if 0 <= q[1] <= height else np.inf, yl)
    yh = np.where(parallel, np.inf if 0 <= q[1] <= height else -np.inf, yh)
    low, high = np.maximum(lo, yl), np.minimum(hi, yh)
    return np.where((disc >= 0) & (a > 1e-14) & (high >= np.maximum(low, .01)), np.maximum(low, .01), np.inf)


def predict_visibility(camera, root_m, body=BodySpec(), *, depth=None, raycaster=None,
                       occluders=(), depth_margin_m=.04, edge_margin_px=2.):
    root = np.asarray(root_m, float)
    if root.shape != (3,) or not np.isfinite(root).all():
        raise ValueError("finite world root required")
    frame, edge = framing(camera, root, body)
    evidence = {"body_envelope": {"radius_m": body.radius_m, "height_m": body.height_m, "margin_m": body.margin_m},
                "native_pixel_truth": False, "bounds_conditional_on_body_envelope": True,
                "required_depth_margin_m": depth_margin_m}
    if frame == "outside":
        return VisibilityPrediction("out_of_view", edge_margin_px=edge,
                                    confidence="envelope_outside", evidence=evidence)
    if depth is None and raycaster is not None:
        depth = build_depth_buffer(camera, raycaster)
    if depth is None:
        return VisibilityPrediction("unknown", edge_margin_px=edge,
                                    evidence={**evidence, "reason": "missing_visual_depth"})
    if depth.camera != camera:
        raise ValueError("depth belongs to a different fixed camera")
    uv, z = project_points(camera, body_box(root, body))
    if np.any(z <= .01):
        return VisibilityPrediction("unknown", edge_margin_px=edge, evidence={**evidence, "reason": "near_plane_crossing"})
    x0, y0 = np.floor(np.min(uv, axis=0) - edge_margin_px).astype(int)
    x1, y1 = np.ceil(np.max(uv, axis=0) + edge_margin_px).astype(int)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(camera.width, x1), min(camera.height, y1)
    if x1 <= x0 or y1 <= y0:
        return VisibilityPrediction("unknown", evidence={**evidence, "reason": "subpixel_envelope"})
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rays = pixel_rays(camera, xx + .5 - camera.width / 2, yy + .5 - camera.height / 2)
    scene = depth.depth_m[y0:y1, x0:x1].copy()
    ids = np.full(scene.shape, "", dtype=object) if depth.object_ids is None else depth.object_ids[y0:y1, x0:x1].astype(object).copy()
    for entity_id, oc_root, oc_body in occluders:
        od = _cylinder_depth(camera, rays, oc_root, oc_body.radius_m, oc_body.height_m)
        front = od < scene
        scene[front], ids[front] = od[front], str(entity_id)
    nominal = _cylinder_depth(camera, rays, root, body.radius_m, body.height_m)
    outer = _cylinder_depth(camera, rays, root - [0., body.margin_m, 0.],
                            body.radius_m + body.margin_m, body.height_m + 2 * body.margin_m)
    inner = _cylinder_depth(camera, rays, root + [0., body.margin_m, 0.],
                            max(.01, body.radius_m - body.margin_m), max(.01, body.height_m - 2 * body.margin_m))
    occupied = np.isfinite(nominal)
    known = ~np.isnan(scene)
    visible = occupied & known & (nominal < scene)
    blocked = occupied & known & (nominal > scene)
    n = int(occupied.sum())
    fraction = float(blocked.sum() / n) if n else 0.
    upper = int((np.isfinite(outer) & (~known | (outer <= scene + depth_margin_m))).sum())
    # This is a conservative INNER-PROXY count, not a mathematical lower bound
    # on a real rig's silhouette. Qualification is explicit in confidence/evidence.
    lower = int((np.isfinite(inner) & known & (inner + depth_margin_m < scene)).sum())
    blockers = tuple(sorted(set(ids[blocked].tolist()) - {""}))
    evidence.update({"target_proxy_pixels": n, "unknown_depth_pixels": int((occupied & ~known).sum()),
                     "unknown_occluder_pixels": int((blocked & (ids == "")).sum()),
                     "proxy_occluders": bool(occluders), "depth_source": depth.source,
                     "visible_pixels_lower_authority": "no_native_silhouette_lower_bound",
                     "inner_proxy_visible_pixels": lower,
                     "centroid_xy_px": [float(xx[visible].mean()), float(yy[visible].mean())] if visible.any() else None,
                     "target_centroid_xy_px": [float(xx[occupied].mean()), float(yy[occupied].mean())] if occupied.any() else None})
    confidence = "uncalibrated_proxy"
    if not n or np.any(occupied & ~known) or depth.source != "visual_mesh":
        state = "unknown"
    elif not visible.any():
        # Sufficient full-occlusion test: every pixel in the expanded projected
        # BOX has a real scan hit strictly before the nearest expanded corner.
        # Sparse sample rays alone are never allowed to certify this.
        box_clearance = float(np.min(z) - np.max(scene))
        if (np.isfinite(scene).all() and box_clearance >= depth_margin_m and not occluders):
            state, confidence = "fully_occluded", "outer_box_depth_margin"
        else:
            state = "unknown"
    else:
        state = "visible_occluded" if blocked.any() else "visible_clear"
        if frame == "inside" and edge >= edge_margin_px and lower >= 100:
            confidence = "inner_proxy_margin"
    clearance = np.abs(nominal[occupied & known] - scene[occupied & known])
    finite = clearance[np.isfinite(clearance)]
    dmargin = float(finite.min()) if len(finite) else (math.inf if known.all() else 0.)
    return VisibilityPrediction(state, int(visible.sum()), 0, upper, fraction, edge,
                                dmargin, blockers, confidence, evidence)


def framing_vertices(world_vertices, world_to_camera, projection, width, height):
    """Whole actual mesh against native clip planes: (inside/outside/edge, margin_px).

    Occlusion does not affect this projection check. Different vertices outside
    different planes do NOT prove an entirely out-of-view triangle/mesh.
    """
    vertices = np.asarray(world_vertices, float)
    view, project = np.asarray(world_to_camera, float), np.asarray(projection, float)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or not np.isfinite(vertices).all() or view.shape != (4, 4) or project.shape != (4, 4)
            or not np.isfinite(view).all() or not np.isfinite(project).all()
            or width <= 0 or height <= 0):
        raise ValueError("finite native vertices, matrices and image dimensions required")
    clip = np.c_[vertices, np.ones(len(vertices))]@view.T@project.T
    w = clip[:, 3]
    planes = np.column_stack((w+clip[:, 0], w-clip[:, 0], w+clip[:, 1], w-clip[:, 1],
                              w+clip[:, 2], w-clip[:, 2]))
    if np.any(np.all(planes < 0, axis=0)):
        return "outside", -math.inf
    if np.any(w <= 1e-12):
        return "edge", -math.inf
    ndc = clip[:, :2]/w[:, None]
    margin = float(min((1-np.abs(ndc[:, 0])).min()*width/2,
                       (1-np.abs(ndc[:, 1])).min()*height/2))
    return ("inside" if margin >= -1e-6 and np.all(planes[:, 4:] >= -1e-8) else "edge"), margin
