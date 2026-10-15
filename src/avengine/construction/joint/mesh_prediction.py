"""CPU exact-triangle silhouette refinement for a prepared pose.

This predicts native masks from actual visual geometry, never consumes a target
mask. Material transparency, alpha cutout and renderer rasterization differences
remain calibration limits. Dynamic intervals require every relevant pose.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .geometry import (DepthBuffer, MeshRaycaster, VisibilityPrediction, pixel_rays,
                       project_points)


@dataclass
class MeshInstance:
    entity_id: str
    raycaster: MeshRaycaster
    transform_m: np.ndarray = field(default_factory=lambda: np.eye(4))

    def __post_init__(self):
        self.transform_m = np.asarray(self.transform_m, float)
        if (self.transform_m.shape != (4, 4) or not np.isfinite(self.transform_m).all()
                or not np.allclose(self.transform_m[3], [0, 0, 0, 1])
                or abs(np.linalg.det(self.transform_m[:3, :3])) < 1e-12):
            raise ValueError("mesh instance needs an invertible affine transform")


def raster_mesh_depth(camera, instance):
    vertices = np.asarray(instance.raycaster.mesh.vertices)
    result = np.full((camera.height, camera.width), np.inf, dtype=np.float32)
    if not len(vertices):
        return result
    transform = instance.transform_m
    world = vertices@transform[:3, :3].T+transform[:3, 3]
    uv, z = project_points(camera, world)
    if np.all(z <= .01):
        return result
    if np.any(z <= .01):
        x0, y0, x1, y1 = 0, 0, camera.width, camera.height
    else:
        x0, y0 = np.maximum(0, np.floor(uv.min(axis=0))-1).astype(int)
        x1, y1 = np.minimum([camera.width, camera.height], np.ceil(uv.max(axis=0))+1).astype(int)
    if x1 <= x0 or y1 <= y0:
        return result
    yy, xx = np.mgrid[y0:y1, x0:x1]
    directions = pixel_rays(camera, xx.ravel()+.5-camera.width/2, yy.ravel()+.5-camera.height/2)
    inverse = np.linalg.inv(transform)
    local_origin = inverse[:3, :3]@np.asarray(camera.position_m)+inverse[:3, 3]
    local_direction = directions@inverse[:3, :3].T
    ranges, _ = instance.raycaster.first_hit(local_origin, local_direction)
    forward_depth = ranges/np.linalg.norm(local_direction, axis=1)
    forward_depth[forward_depth < .01] = np.inf
    result[y0:y1, x0:x1] = forward_depth.reshape(yy.shape)
    return result


def predict_mesh_visibility(camera, target, *, scene_depth, occluders=(), depth_margin_m=.04):
    if scene_depth.camera != camera:
        raise ValueError("scene depth belongs to another camera")
    target_depth = raster_mesh_depth(camera, target)
    silhouette = np.isfinite(target_depth)
    total = int(silhouette.sum())
    evidence = {"method": "exact_visual_triangles_cpu_pixel_center_rays", "native_pixel_truth": False,
                "target_entity_id": target.entity_id, "material_alpha_not_modeled": True,
                "target_only_pixels": total, "geometry": target.raycaster.evidence}
    if not total:
        return VisibilityPrediction("out_of_view", confidence="exact_mesh_prediction", evidence=evidence)
    scene = scene_depth.depth_m.copy()
    ids = np.full(scene.shape, "", dtype=object)
    if scene_depth.object_ids is not None:
        ids[:] = scene_depth.object_ids
    for instance in occluders:
        if instance.entity_id == target.entity_id:
            raise ValueError("target must not be included among its own occluders")
        d = raster_mesh_depth(camera, instance)
        use = d < scene
        scene[use], ids[use] = d[use], instance.entity_id
    known = ~np.isnan(scene)
    visible = silhouette & known & (target_depth <= scene+1e-5)
    blocked = silhouette & known & ~visible
    count = int(visible.sum())
    fraction = float(blocked.sum()/total)
    blocker_ids = tuple(sorted(set(ids[blocked].tolist())-{ "" }))
    yy, xx = np.nonzero(silhouette)
    edge = float(min(xx.min(), camera.width-1-xx.max(), yy.min(), camera.height-1-yy.max()))
    gaps = scene[silhouette & known]-target_depth[silhouette & known]
    finite = gaps[np.isfinite(gaps)]
    margin = float(np.abs(finite).min()) if len(finite) else float("inf")
    source_ok = (scene_depth.source == target.raycaster.source == "visual_mesh"
                 and all(o.raycaster.source == "visual_mesh" for o in occluders))
    if np.any(silhouette & ~known) or not source_ok:
        state = "unknown"
    elif count == 0:
        state = "fully_occluded"
    elif count == total:
        state = "visible_clear"
    else:
        state = "visible_occluded"
    robust_visible = int((silhouette & known & (target_depth+depth_margin_m < scene)).sum())
    upper = int((silhouette & (~known | (target_depth <= scene+depth_margin_m))).sum())
    evidence.update({"unknown_occluder_pixels": int((blocked & (ids == "")).sum()),
                     "unknown_depth_pixels": int((silhouette & ~known).sum()),
                     "target_centroid_xy_px": [float(xx.mean()), float(yy.mean())],
                     "robust_visible_mesh_pixels": robust_visible,
                     "visible_pixels_lower_authority": "exact_mesh_assuming_opaque_materials_and_final_pose",
                     "required_depth_margin_m": depth_margin_m})
    return VisibilityPrediction(state, count, robust_visible, upper, fraction, edge,
                                margin, blocker_ids, "exact_mesh_prediction", evidence)
