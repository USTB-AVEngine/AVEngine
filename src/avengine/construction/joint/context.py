"""Reusable room geometry, legal camera candidates, and lazily persisted grids."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import os
import tempfile
import numpy as np
from .types import BodySpec, CameraPose
from .geometry import (DepthBuffer, MeshRaycaster, build_depth_buffer, camera_coordinates,
                       framing, predict_visibility)


@dataclass
class CandidateGrid:
    points_m: np.ndarray
    distances_m: np.ndarray
    azimuth_deg: np.ndarray
    frustum_state: np.ndarray
    visible_fraction: np.ndarray
    shadow_mask: np.ndarray
    blocker_ids: np.ndarray
    evidence: dict


def camera_legal(room, camera, *, wall_clearance_m=.5):
    from shapely.geometry import Point, shape
    polygon = shape(room["floor_polygon"])
    p = np.asarray(camera.position_m)[[0, 2]]
    point = Point(p)
    clearance = float(polygon.boundary.distance(point))
    inward = p + camera.forward[[0, 2]] * .75
    return (polygon.covers(point) and clearance + 1e-7 >= wall_clearance_m
            and polygon.covers(Point(inward))), clearance


def candidate_cameras(room, walkable_map, pathfinder, *, limit=24, heights_m=None, width=640, height=360):
    """Deterministic perimeter cameras facing legal interior points; no random seed."""
    from shapely.geometry import Point, shape
    polygon = shape(room["floor_polygon"])
    inner = polygon.buffer(-.5001)
    if inner.is_empty:
        return ()
    floor = float(room["measured_floor_y_m"])
    old = room.get("placement", {}).get("camera_m")
    heights = tuple(heights_m) if heights_m is not None else (float(old[1]) - floor if old else 1.25,)
    if any(not math.isfinite(h) or h <= .05 for h in heights):
        raise ValueError("camera heights must be positive above the room floor")
    parts = list(inner.geoms) if inner.geom_type == "MultiPolygon" else [inner]
    points = [np.asarray(old)[[0, 2]]] if old else []
    parts.sort(key=lambda p: -p.area)
    for part in parts:
        ring = part.exterior
        points.extend(np.asarray(ring.interpolate(float(d)).coords[0]) for d in np.arange(0, ring.length, .75))
    # A legal camera can lie well inside the room while every sampled inset
    # boundary point is furniture-blocked. Include the v2 interior free grid.
    if walkable_map is not None:
        stride = max(1, round(.25 / walkable_map.resolution))
        zz, xx = np.nonzero(walkable_map.layers["human"][::stride, ::stride])
        interior = walkable_map.origin + (np.column_stack((xx, zz))*stride+.5)*walkable_map.resolution
        points.extend(interior)
    seen, legal = set(), []
    for p in points:
        key = tuple(np.round(p, 3))
        if key in seen or not polygon.covers(Point(p)) or polygon.boundary.distance(Point(p)) < .5 - 1e-7:
            continue
        seen.add(key)
        if walkable_map is not None and not walkable_map.contains(p, "human"):
            continue
        world = np.array([p[0], floor, p[1]])
        if pathfinder is not None:
            snap = np.asarray(pathfinder.snap_point(world), float)
            if snap.shape != (3,) or not np.isfinite(snap).all() or np.linalg.norm((snap-world)[[0, 2]]) > .02:
                continue
        legal.append(p)
    # Spatially spread candidates instead of exhausting one wall first.
    selected = []
    while legal and len(selected) < max(1, math.ceil(limit / (3 * len(heights)))):
        index = 0 if not selected else int(np.argmax([min(np.linalg.norm(p-q) for q in selected) for p in legal]))
        selected.append(legal.pop(index))
    target = np.asarray(inner.representative_point().coords[0])
    look_targets = np.asarray(selected+legal)
    result = []
    for hi in heights:
        for p in selected:
            delta = target-p
            nominal = math.degrees(math.atan2(delta[0], -delta[1]))
            yaws = [nominal, nominal-30., nominal+30.]
            if len(look_targets):
                rays = look_targets-p
                order = np.argsort(-np.linalg.norm(rays, axis=1), kind="stable")
                yaws.extend(math.degrees(math.atan2(rays[i,0],-rays[i,1])) for i in order[:12])
            ranked = []
            for yaw in dict.fromkeys(round(v, 5) for v in yaws):
                pose = CameraPose((float(p[0]), floor+hi, float(p[1])), yaw,
                                  "candidate", width=width, height=height)
                if not camera_legal(room, pose)[0]:
                    continue
                rays = look_targets-p
                forward = rays@pose.forward[[0,2]]
                lateral = rays@pose.right[[0,2]]
                visible = (forward > max(hi,1.8-hi)*pose.focal_px/(height/2)+.35)
                visible &= np.abs(lateral)+.35 < forward*math.tan(math.radians(pose.hfov_deg/2))
                ranked.append((-int(visible.sum()), abs((yaw-nominal+180)%360-180), yaw))
            chosen_yaws = []
            for _, _, yaw in sorted(ranked):
                if any(abs((yaw-y+180)%360-180) < 12. for y in chosen_yaws):
                    continue
                chosen_yaws.append(yaw)
                result.append(CameraPose((float(p[0]),floor+hi,float(p[1])),yaw,
                                         "cam_%03d"%len(result),width=width,height=height))
                if len(result)>=limit:
                    return tuple(result)
                if len(chosen_yaws)>=3:
                    break
    return tuple(result)



class RoomContext:
    def __init__(self, room, *, walkable_map, pathfinder=None, raycaster=None, cache_dir=None,
                 camera_limit=24, grid_step_m=.25, camera_heights_m=None, width=640, height=360,
                 owned_video=None, allow_native_map_fallback=False):
        self.room, self.walkable_map = room, walkable_map
        self.pathfinder, self.raycaster = pathfinder, raycaster
        self.floor_raycaster = raycaster
        self.allow_native_map_fallback = bool(allow_native_map_fallback)
        self._owned_video = owned_video
        if walkable_map is None:
            raise ValueError("joint planning requires a room walkable map")
        if not math.isfinite(grid_step_m) or grid_step_m < walkable_map.resolution:
            raise ValueError("grid step must be at least the walkable map resolution")
        old_camera = room.get("placement", {}).get("camera_m")
        self.camera_heights_m = (tuple(camera_heights_m) if camera_heights_m is not None else
                                (float(old_camera[1])-float(room["measured_floor_y_m"]) if old_camera else 1.25,))
        self.camera_width, self.camera_height = width, height
        self.cameras = candidate_cameras(room, walkable_map, pathfinder, limit=camera_limit,
                                        heights_m=self.camera_heights_m, width=width, height=height)
        stride = max(1, round(grid_step_m / walkable_map.resolution))
        union = np.logical_or.reduce(list(walkable_map.layers.values()))
        sampled = np.zeros_like(union)
        sampled[::stride, ::stride] = union[::stride, ::stride]
        z, x = np.nonzero(sampled)
        self.cells_zx = np.column_stack((z, x))
        xz = walkable_map.origin + (np.column_stack((x, z)) + .5) * walkable_map.resolution
        y = np.full(len(x), walkable_map.floor_reference)
        if walkable_map.floor_y is not None:
            measured = walkable_map.floor_y[z, x]
            y = np.where(np.isfinite(measured), measured, y)
        self.points_m = np.column_stack((xz[:, 0], y, xz[:, 1]))
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._grids, self._depth = {}, {}
        self._key_cache = {}
        self.identity = {"revision": "joint-room-grid-3-memoized-pose-keys", "room": room,
                         "map_metadata": walkable_map.metadata, "grid_stride": stride,
                         "geometry": getattr(raycaster, "evidence", {}),
                         "geometry_source": getattr(raycaster, "source", "missing")}
        self.cache_hits = self.cache_misses = 0

    def solve_cameras(self, roles, **kwargs):
        """Opt-in demand-driven positions/heights/continuous yaw; legacy cameras unchanged."""
        from .camera_solve import solve_cameras
        return solve_cameras(self, roles, **kwargs)

    def role_points(self, camera, role):
        """Return grid points satisfying a camera-relative PointRole template."""
        from .camera_solve import role_points
        return role_points(self, camera, role)

    def close(self):
        if self._owned_video is not None:
            self._owned_video.sim.close()
            self._owned_video = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _key(self, camera, body=None):
        # A room context is immutable geometry. Hash its large floor polygon only
        # once per camera/body query, not once per candidate target position.
        # This remains a memoization key, not a truth hash or admission gate.
        pair=(camera,body)
        if pair not in self._key_cache:
            value={"input":self.identity,"camera":asdict(camera),"body":asdict(body)if body else None}
            self._key_cache[pair]=hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()
        return self._key_cache[pair]

    def depth(self, camera):
        if self.raycaster is None:
            return None
        key = self._key(camera)
        if key not in self._depth:
            path = self.cache_dir / ("depth_" + key + ".npz") if self.cache_dir is not None else None
            if path is not None and path.exists():
                with np.load(path, allow_pickle=False) as data:
                    ids = data["object_ids"] if "object_ids" in data else None
                    self._depth[key] = DepthBuffer(camera, data["depth_m"], ids, str(data["source"].item()),
                                                  json.loads(str(data["evidence"].item())))
                self.cache_hits += 1
            else:
                depth = build_depth_buffer(camera, self.raycaster)
                self._depth[key] = depth
                self.cache_misses += 1
                if path is not None:
                    arrays = {"depth_m": depth.depth_m.astype(np.float32), "source": np.array(depth.source),
                              "evidence": np.array(json.dumps(depth.evidence, default=str))}
                    if depth.object_ids is not None:
                        arrays["object_ids"] = np.asarray(depth.object_ids, dtype=str)
                    self._save_cache(path, arrays)
        return self._depth[key]

    def predict(self, camera, root_m, body=BodySpec(), *, occluders=()):
        return predict_visibility(camera, root_m, body, depth=self.depth(camera), occluders=occluders)

    def grid(self, camera, body=BodySpec()):
        key = self._key(camera, body)
        if key in self._grids:
            self.cache_hits += 1
            return self._grids[key]
        path = self.cache_dir / ("grid_" + key + ".npz") if self.cache_dir is not None else None
        if path is not None and path.exists():
            with np.load(path, allow_pickle=False) as d:
                grid = CandidateGrid(self.points_m, d["distances_m"], d["azimuth_deg"], d["frustum_state"],
                                     d["visible_fraction"], d["shadow_mask"], d["blocker_ids"],
                                     json.loads(str(d["evidence"].item())))
            self._grids[key] = grid
            self.cache_hits += 1
            return grid
        self.cache_misses += 1
        xyz = camera_coordinates(camera, self.points_m)
        distance = np.linalg.norm(xyz[:, [0, 2]], axis=1)
        azimuth = np.degrees(np.arctan2(xyz[:, 0], xyz[:, 2]))
        frames = np.asarray([framing(camera, p, body)[0] for p in self.points_m], dtype="U8")
        fraction = np.full(len(self.points_m), np.nan)
        shadow = np.zeros(len(self.points_m), bool)
        blockers = np.full(len(self.points_m), "", dtype=object)
        if self.raycaster is not None and len(self.points_m):
            radius = body.radius_m + body.margin_m
            angles = np.arange(12) * (2 * math.pi / 12)
            offsets = np.array([[radius*math.cos(a), h, radius*math.sin(a)]
                                for h in (.05, body.height_m / 2, body.height_m + body.margin_m) for a in angles])
            samples = self.points_m[:, None, :] + offsets
            delta = samples.reshape(-1, 3) - np.asarray(camera.position_m)
            target_ranges = np.linalg.norm(delta, axis=1)
            hit, ids = self.raycaster.first_hit(np.asarray(camera.position_m), delta)
            blocked = (hit < target_ranges - .04).reshape(len(samples), -1)
            fraction = 1 - blocked.mean(axis=1)
            shadow = blocked.all(axis=1) & (frames == "inside")
            ids = ids.reshape(len(samples), -1)
            for i in np.flatnonzero(shadow):
                known = set(ids[i].tolist()) - {""}
                if len(known) == 1 and np.all(ids[i] != ""):
                    blockers[i] = next(iter(known))
        evidence = {"authority": "sampled_outer_envelope_broad_phase_only", "source": getattr(self.raycaster, "source", "missing"),
                    "native_pixel_truth": False, "points": len(self.points_m), "grid_step_m": self.identity["grid_stride"] * self.walkable_map.resolution,
                    "body": asdict(body)}
        grid = CandidateGrid(self.points_m, distance, azimuth, frames, fraction, shadow, blockers, evidence)
        self._grids[key] = grid
        if path is not None:
            self._save_cache(path, dict(distances_m=distance, azimuth_deg=azimuth, frustum_state=frames,
                                        visible_fraction=fraction, shadow_mask=shadow, blocker_ids=blockers.astype(str),
                                        evidence=np.array(json.dumps(evidence))))
        return grid


    def _save_cache(self, path, arrays):
        # Only a new derived file in the caller-owned cache, never source geometry.
        fd, tmp = tempfile.mkstemp(prefix=".joint-cache-", suffix=".npz", dir=self.cache_dir)
        os.close(fd)
        try:
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def with_occluders(self, *instances):
        """A shared-room view with exact registered meshes; never mutate the base."""
        from copy import copy
        from .geometry import CompositeRaycaster
        from .mesh_prediction import InstanceRaycaster
        if self.raycaster is None:
            raise ValueError("occlusion queries require the original visual background")
        other = copy(self)
        other._owned_video = None
        other.raycaster = CompositeRaycaster(self.raycaster, *(InstanceRaycaster(i) for i in instances))
        other.identity = {**self.identity, "geometry": other.raycaster.evidence,
                          "geometry_source": other.raycaster.source}
        other._depth, other._grids = {}, {}
        other._key_cache = {}
        other.cache_hits = other.cache_misses = 0
        return other


def build_room_context(room, *, house=None, runtime=None, cache_dir=None, walkable_map=None,
                       pathfinder=None, raycaster=None, camera_limit=24, grid_step_m=.25,
                       camera_heights_m=None, width=640, height=360, allow_native_map_fallback=False):
    from avengine.rooms.walkable_map import load_room_map, source_identity
    video = None
    if house is not None and house.get("split") != "train":
        raise ValueError("joint construction only accepts explicitly train houses")
    if walkable_map is None:
        walkable_map = load_room_map(room.get("room_label", room.get("room_id")),
                                     path=room.get("walkable_map_path"), floor_reference=room["measured_floor_y_m"])
    try:
        if house is not None and runtime is not None and (pathfinder is None or raycaster is None):
            from avengine.rooms.walkable_map_builder import native_static_video, stage_path
            from avengine.assets.v7_phase2_floor import original_native_stage_mesh
            video = native_static_video(house, runtime)
            if pathfinder is None:
                pathfinder = video.pathfinder
            if raycaster is None:
                path = stage_path(house)
                mesh, evidence = original_native_stage_mesh(video, path)
                # Source stat, original native up/front transform; no proxy geometry.
                evidence = {**evidence, "source_identity": source_identity([path])}
                raycaster = MeshRaycaster(mesh, evidence=evidence)
        return RoomContext(room, walkable_map=walkable_map, pathfinder=pathfinder, raycaster=raycaster,
                           cache_dir=cache_dir, camera_limit=camera_limit, grid_step_m=grid_step_m,
                           camera_heights_m=camera_heights_m, width=width, height=height, owned_video=video,
                           allow_native_map_fallback=allow_native_map_fallback)
    except Exception:
        if video is not None:
            video.sim.close()
        raise
