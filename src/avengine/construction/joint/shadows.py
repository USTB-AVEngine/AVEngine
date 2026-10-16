"""Continuous foreground positions from silhouette bounds, then exact mesh rays."""
from dataclasses import dataclass
import math
import time
import numpy as np
from .types import BodySpec
from .geometry import camera_coordinates, framing
from .mesh_prediction import MeshInstance, predict_mesh_visibility
from .routes import validate_track, ConstructionError
from .layouts import _separated


@dataclass
class MeshShadowPlacement:
    foreground: MeshInstance
    prediction: object
    evidence: dict


def _clip_polygon(vertices, normal, bound):
    """Clip a convex polygon by normal @ p <= bound (numerical tolerance only)."""
    result = []
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        da, db = float(normal@a-bound), float(normal@b-bound)
        ina, inb = da <= 1e-10, db <= 1e-10
        if ina:
            result.append(a)
        if ina != inb:
            result.append(a+(b-a)*(da/(da-db)))
    return np.asarray(result, float).reshape(-1, 2)


def mesh_shadow_foreground_candidates(context, camera, target, foreground, *,
        target_body=BodySpec(), foreground_body=BodySpec(), max_candidates=64,
        budget_s=60., diagnostics=None):
    """Yield exact-mesh full shadows at continuous positions in v2 free cells.

    target and foreground are MeshInstance objects with original local visual
    meshes. Target pose is fixed. Foreground rotation/scale is preserved; only
    its root position is solved. Its lowest transformed visual vertex is placed
    on the map's measured floor. No fixed depth ratios or lateral offset list.

    Every free foreground-species cell is intersected with necessary linear
    perspective-bounds constraints. Interior and boundary witnesses of these
    continuous polygons are tested with the shared exact triangle predictor.
    This finite witness search is NOT a continuous-space impossibility proof,
    an all-pose motion certificate, or a substitute for native masks/physics.
    """
    if not isinstance(target, MeshInstance) or not isinstance(foreground, MeshInstance):
        raise TypeError("original target and foreground MeshInstance required")
    if target.entity_id == foreground.entity_id:
        raise ValueError("target and registered foreground identities must differ")
    if max_candidates < 0 or not math.isfinite(budget_s) or budget_s < 0:
        raise ValueError("invalid computation budget")
    diag = diagnostics if diagnostics is not None else {}
    diag.update(status="searching", free_cells=0, bounding_feasible_cells=0,
                tested_positions=0, exact_mesh_tests=0, full_shadow_candidates=0,
                native_pixel_truth=False, continuous_infeasibility_proved=False,
                foreground_orientation_solved=False)
    started = time.monotonic()
    if max_candidates == 0:
        diag["status"] = "candidate_limit"
        return
    tv = np.asarray(target.raycaster.mesh.vertices) @ target.transform_m[:3, :3].T + target.transform_m[:3, 3]
    fv = np.asarray(foreground.raycaster.mesh.vertices) @ foreground.transform_m[:3, :3].T
    if not len(tv) or not len(fv):
        diag["status"] = "empty_visual_mesh"
        return
    xyz = camera_coordinates(camera, tv)
    if np.any(xyz[:, 2] <= .01):
        diag["status"] = "target_crosses_camera_near_plane"
        return
    projected = xyz[:, :2]/xyz[:, 2, None]
    u0, v0 = projected.min(axis=0)
    u1, v1 = projected.max(axis=0)
    local = camera_coordinates(camera, fv+np.asarray(camera.position_m))
    x, y, z = local.T
    # Perspective silhouette bounding containment, a necessary condition only:
    # u1*d-max(x-u1*z) <= lateral <= u0*d-min(x-u0*z).
    normals = np.array([[u1, -1.], [-u0, 1.], [v1, 0.], [-v0, 0.], [-1., 0.]])
    constants = np.array([np.max(x-u1*z), -np.min(x-u0*z),
                          np.max(y-v1*z), -np.min(y-v0*z), np.min(z)-.01])
    m = context.walkable_map
    zz, xx = np.nonzero(m.layers[foreground_body.species])
    diag["free_cells"] = len(xx)
    centres = m.origin + (np.column_stack((xx, zz))+.5)*m.resolution
    floor = np.full(len(xx), m.floor_reference)
    if m.floor_y is not None:
        measured = m.floor_y[zz, xx]
        floor = np.where(np.isfinite(measured), measured, floor)
    root_y = floor - float(fv[:, 1].min())
    relative_y = root_y-camera.position_m[1]
    bounds = np.tile(constants, (len(xx), 1))
    bounds[:, 2] += relative_y
    bounds[:, 3] -= relative_y
    basis = np.column_stack((camera.forward[[0, 2]], camera.right[[0, 2]]))
    dl = (centres-np.asarray(camera.position_m)[[0, 2]]) @ basis
    # Conservatively retain a cell if each half-plane can meet some point in it.
    world_normals = normals @ basis.T
    allowance = m.resolution/2*np.abs(world_normals).sum(axis=1)
    possible = np.all(dl @ normals.T-allowance <= bounds+1e-10, axis=1)
    ids = np.flatnonzero(possible)
    ids = ids[np.argsort(dl[ids, 0], kind="stable")]
    depth = context.depth(camera)
    if depth is None:
        diag["status"] = "missing_original_visual_depth"
        return
    seen = set()
    target_root = target.transform_m[:3, 3]
    bodies = {target.entity_id: target_body, foreground.entity_id: foreground_body}
    try:
        for index in ids:
            if time.monotonic()-started >= budget_s:
                diag["status"] = "budget_exhausted"
                return
            corner = m.origin+np.array([xx[index], zz[index]])*m.resolution
            world = corner+m.resolution*np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
            polygon = (world-np.asarray(camera.position_m)[[0, 2]]) @ basis
            for normal, bound in zip(normals, bounds[index]):
                polygon = _clip_polygon(polygon, normal, bound)
                if not len(polygon):
                    break
            if not len(polygon):
                continue
            diag["bounding_feasible_cells"] += 1
            witnesses = np.vstack((polygon.mean(axis=0), polygon))
            for value in witnesses:
                if time.monotonic()-started >= budget_s:
                    diag["status"] = "budget_exhausted"
                    return
                xz = np.asarray(camera.position_m)[[0, 2]] + basis @ value
                token = tuple(np.round(xz, 8))
                if token in seen:
                    continue
                seen.add(token)
                root = np.array([xz[0], root_y[index], xz[1]])
                ground = np.array([xz[0], floor[index], xz[1]])
                diag["tested_positions"] += 1
                if not _separated(root, foreground_body, {target.entity_id: target_root}, bodies):
                    continue
                # The original fixed-pose mesh, not an over-wide capsule box,
                # decides whether this static foreground fits completely.
                forward = z+value[0]
                lateral = x+value[1]
                vertical = y+relative_y[index]
                if (np.any(forward <= .01)
                        or np.any(np.abs(lateral) > forward*camera.width/(2*camera.focal_px))
                        or np.any(np.abs(vertical) > forward*camera.height/(2*camera.focal_px))):
                    continue
                try:
                    _, track = validate_track(context, [ground], foreground_body.species, camera=camera, snap=False)
                except ConstructionError:
                    continue
                transform = foreground.transform_m.copy()
                transform[:3, 3] = root
                candidate = MeshInstance(foreground.entity_id, foreground.raycaster, transform)
                prediction = predict_mesh_visibility(camera, target, scene_depth=depth, occluders=(candidate,))
                diag["exact_mesh_tests"] += 1
                if (prediction.state != "fully_occluded"
                        or set(prediction.occluder_ids) != {foreground.entity_id}
                        or prediction.evidence.get("unknown_occluder_pixels", 0)):
                    continue
                diag["full_shadow_candidates"] += 1
                yield MeshShadowPlacement(candidate, prediction, {
                    "authority": "planning_exact_opaque_triangles_only", "native_pixel_truth": False,
                    "root_position_m": root.tolist(), "ground_position_m": ground.tolist(),
                    "cell_zx": [int(zz[index]), int(xx[index])],
                    "continuous_feasible_polygon_depth_lateral_m": polygon.tolist(),
                    "foreground_rotation_preserved": True,
                    "original_native_pose_physics_and_masks_required": True,
                    "track_validation": track})
                if diag["full_shadow_candidates"] >= max_candidates:
                    diag["status"] = "candidate_limit"
                    return
        diag["status"] = "selected_cell_polygon_witnesses_exhausted"
    finally:
        if diag["status"] == "searching":
            diag["status"] = "consumer_stopped"
        diag["elapsed_s"] = time.monotonic()-started
