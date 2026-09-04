#!/usr/bin/env python3
"""Per-scene camera clearance table: one actor-free depth cube ring per camera point.

Why
---
On 2026-09-02 sixteen of 37 fresh card1 candidates had their camera parked
behind a sofa back, a kitchen island, a lamp shade or a door leaf.  The
solver only knows geometry (routes, azimuths, bands) and nothing about the
furniture, so it happily picks such poses and the quota is eaten by
candidates that pixel truth later rejects.  The depth preflight caught them,
but only after generation.  This tool moves that knowledge in front of the
solver: it renders, once per scene, an actor-free depth view in every
direction from every camera point the solver may choose, and stores both the
raw (downsampled) depth ring and per-yaw clearance summaries.  Rooms are
rendered once; question types and assets never trigger a re-render, they
only change where the consumer samples the stored depth.

What is rendered
----------------
For each camera point (the solver's navigable points at the scene camera
height, optionally more heights) four 90-degree depth faces are rendered at
world yaw 0/90/180/270.  Together they cover the whole horizontal ring; the
portrait face aspect (default 512x768, a 112.6-degree vertical field, both
verified against the engine on 2026-09-02) covers elevations that the
production camera or a sight line to a nearby dog can reach.  Any production
view (105 degrees at 16:9, any yaw) is re-projected from the ring, so no yaw
discretisation error enters the camera-clearance verdict; the 2-degree yaw
grid of the stored summaries is a lookup convenience, and the exact-yaw value
can always be recomputed from the stored faces.

Phases
------
1. engine: render faces for every point/height into shards, and render the
   production camera directly at random (point, yaw) pairs for validation;
2. cpu (multiprocess): per-yaw clearance summaries from the stored faces;
3. cpu: compare the direct validation renders with the table.

Conventions
-----------
UE world frame: X forward, Y right, Z up, yaw positive from +X towards +Y
(the same reference as scene_sampler.bearing_deg).  Depth buffers are radial
distance in metres (see camera_clearance.py for the measurement behind this).
No-hit pixels (non-finite, non-positive or beyond 1 km) are stored as 65504.

Boundary
--------
This is a placement input for candidate generation, not pixel truth and not
question admission.  Every threshold here is a research placeholder.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "tools" / "qa"))

from camera_clearance import (  # noqa: E402
    FACE_HFOV_DEG,
    FACE_YAWS_DEG,
    NO_HIT_M,
    SCHEMA,
    VirtualCamera,
    band_column_medians,
    blocked_column_fraction,
    clean_depth,
    min_pool,
    point_key,
    yaw_bin_index,
)
from preflight_camera_clearance_depth import (  # noqa: E402
    clearance_statistics,
    target_band_rows,
)
from scene_sampler import load_scene, read_scene_config, scene_hfov_deg  # noqa: E402

DEPTH_COMPONENT = "DefaultSceneRoot.sp_depth_meters_"
DEFAULT_NEAR_M = (1.0, 1.5, 2.5)
DEFAULT_TARGET_HEIGHTS_M = (0.5, 1.0, 1.7)
DEFAULT_TARGET_DISTANCE_RANGE_M = (2.5, 10.0)
DEFAULT_YAW_STEP_DEG = 2.0
SUMMARY_GRID_HW = (180, 320)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _git_worktree_state(repo: Path = REPOSITORY) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, text=True, capture_output=True).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--short"],
        check=True, text=True, capture_output=True).stdout.splitlines()
    return {"revision": revision, "dirty": bool(status), "status": status}


# ---------------------------------------------------------------------------
# summaries (geometry lives in camera_clearance.py, shared with the solver)
# ---------------------------------------------------------------------------

def yaw_summaries(faces_radial: np.ndarray, *, camera: VirtualCamera,
                  camera_height_m: float, yaws_deg: Sequence[float],
                  nears_m: Sequence[float], target_heights_m: Sequence[float],
                  target_distance_range_m: Sequence[float]) -> dict[str, np.ndarray]:
    """Clearance summaries for every yaw on the grid.

    target_band[h, n, y]: blocked-column fraction in the band where a
    floor-standing target of height h at 2.5-10 m projects, for near n and
    yaw y.  eye_band[n, y]: same for the middle third of rows.  Column
    medians are computed once per band and reused across the near distances."""
    bands = [target_band_rows(camera.height, hfov_deg=camera.hfov_deg,
                              aspect=camera.aspect, camera_height_m=camera_height_m,
                              target_height_m=h,
                              distance_range_m=target_distance_range_m)
             for h in target_heights_m]
    eye = (camera.height // 3, 2 * camera.height // 3)
    nears = np.asarray(nears_m, dtype=np.float64)
    target = np.zeros((len(target_heights_m), len(nears), len(yaws_deg)), np.float32)
    eye_band = np.zeros((len(nears), len(yaws_deg)), np.float32)
    for yi, yaw in enumerate(yaws_deg):
        depth = camera.reproject_depth(faces_radial, float(yaw))
        for hi, rows in enumerate(bands):
            median = band_column_medians(depth, rows)
            finite = np.isfinite(median)
            for ni, near in enumerate(nears):
                target[hi, ni, yi] = float((finite & (median < near)).mean())
        median = band_column_medians(depth, eye)
        finite = np.isfinite(median)
        for ni, near in enumerate(nears):
            eye_band[ni, yi] = float((finite & (median < near)).mean())
    return {"target_band_blocked_column_fraction": target,
            "eye_band_blocked_column_fraction": eye_band,
            "target_band_rows": np.asarray(bands, dtype=np.int32)}


def _summaries_for_shard(task: dict[str, Any]) -> dict[str, Any]:
    """Worker: summaries for every (point, height) entry stored in one shard."""
    camera = VirtualCamera(task["hfov_deg"], SUMMARY_GRID_HW[1], SUMMARY_GRID_HW[0])
    heights = task["heights"]
    with np.load(task["path"]) as data:
        faces = data["radial_m"]
        point_index = data["point_index"].tolist()
        height_index = data["height_index"].tolist()
        target, eye = [], []
        for i in range(faces.shape[0]):
            summary = yaw_summaries(
                faces[i].astype(np.float32), camera=camera,
                camera_height_m=float(heights[height_index[i]]),
                yaws_deg=task["yaws"], nears_m=task["nears"],
                target_heights_m=task["target_heights"],
                target_distance_range_m=task["target_distance_range"])
            target.append(summary["target_band_blocked_column_fraction"])
            eye.append(summary["eye_band_blocked_column_fraction"])
    return {"point_index": point_index, "height_index": height_index,
            "target": np.stack(target), "eye": np.stack(eye)}


# ---------------------------------------------------------------------------
# engine side
# ---------------------------------------------------------------------------

def spawn_depth_camera(game: Any, *, hfov_deg: float, width: int, height: int,
                       camera_blueprint: str) -> tuple[Any, Any]:
    """Depth-only capture camera.  UE binds FOVAngle to the horizontal axis
    for portrait targets too (verified 2026-09-02: a 512x768 face at 90
    degrees reproduces the central columns of a 768x768 face exactly)."""
    camera_class = game.unreal_service.load_class(uclass="AActor", name=camera_blueprint)
    camera = game.unreal_service.spawn_actor(uclass=camera_class)
    depth = game.unreal_service.get_component_by_name(
        actor=camera, component_name=DEPTH_COMPONENT,
        uclass="USpSceneCaptureComponent2D")
    viewport = game.rendering_service.get_current_viewport_desc()
    game.rendering_service.align_camera_with_viewport(
        camera_sensor=camera, camera_components=[depth], viewport_desc=viewport,
        widths=int(width), heights=int(height))
    depth.Initialize()
    depth.initialize_sp_funcs()
    depth.set_property_value(property_name="FOVAngle", property_value=float(hfov_deg))
    observed = float(depth.get_property_value(property_name="FOVAngle"))
    _require(abs(observed - float(hfov_deg)) <= 1.0e-4, "depth camera FOV readback drift")
    depth.PrimitiveRenderMode = "PRM_RenderScenePrimitives"
    depth.ShowOnlyActors = []
    return camera, depth


def set_pose(camera: Any, xyz_cm: Sequence[float], yaw_deg: float) -> None:
    camera.K2_SetActorLocationAndRotation(
        NewLocation={"X": float(xyz_cm[0]), "Y": float(xyz_cm[1]), "Z": float(xyz_cm[2])},
        NewRotation={"Roll": 0.0, "Pitch": 0.0, "Yaw": float(yaw_deg)},
        bSweep=False, bTeleport=True)


def read_depth(component: Any) -> np.ndarray:
    return clean_depth(component.read_pixels()["arrays"]["data"][:, :, 0])


def close_cameras(*, instance, game, cameras: Sequence[tuple[Any, Any]]) -> None:
    if not cameras:
        return
    with instance.begin_frame():
        pass
    with instance.end_frame():
        for camera, component in cameras:
            try:
                component.terminate_sp_funcs()
            finally:
                component.Terminate()
            game.unreal_service.destroy_actor(actor=camera)


def scene_render_facts(config: dict, scene) -> dict[str, Any]:
    render = config.get("render") or {}
    missing = [k for k in ("native_map", "ground_z_ue_cm") if render.get(k) is None]
    _require(not missing, f"{scene.scene_id}: scene render config lacks {missing}")
    native_map = str(render["native_map"])
    _require(native_map.startswith("/Game/"), "native_map must be a /Game package path")
    ground_z = float(render["ground_z_ue_cm"])
    _require(math.isfinite(ground_z), "ground_z_ue_cm must be finite")
    return {"native_map": native_map, "ground_z_ue_cm": ground_z,
            "room_profile_id": render.get("room_profile_id")}


class ShardWriter:
    def __init__(self, faces_dir: Path, shard_size: int):
        self.faces_dir = faces_dir
        self.shard_size = shard_size
        self.records: list[dict[str, Any]] = []
        self._faces: list[np.ndarray] = []
        self._index: list[tuple[int, int]] = []

    def add(self, faces: np.ndarray, point_index: int, height_index: int) -> None:
        self._faces.append(faces.astype(np.float16))
        self._index.append((point_index, height_index))
        if len(self._faces) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._faces:
            return
        number = len(self.records)
        path = self.faces_dir / f"shard_{number:04d}.npz"
        np.savez_compressed(
            path, radial_m=np.stack(self._faces),
            point_index=np.asarray([i for i, _ in self._index], np.int32),
            height_index=np.asarray([h for _, h in self._index], np.int32))
        self.records.append({"path": f"faces/{path.name}", "count": len(self._faces),
                             "first": list(self._index[0]), "last": list(self._index[-1])})
        self._faces.clear()
        self._index.clear()


def render_phase(*, args, facts, heights, points, contract_hfov, output) -> dict[str, Any]:
    from avengine.backends.spear_ue.research_runtime import launch_external_game_instance
    from avengine.timeline import current_apartment_visual as VISUAL

    faces_dir = output / "faces"
    faces_dir.mkdir()
    executable = Path(args.spear_executable)
    _require(executable.is_file(), f"missing SpearSim executable: {executable}")
    _require(Path(args.stage_root).is_dir(), f"missing stage root: {args.stage_root}")
    started = time.time()
    instance = launch_external_game_instance(
        spear_executable=executable, native_map=facts["native_map"],
        frame_rate_hz=VISUAL.FRAME_RATE_HZ, rpc_port=args.rpc_port,
        graphics_adapter=args.graphics_adapter)
    launch_seconds = time.time() - started
    game = instance.get_game()
    cameras: list[tuple[Any, Any]] = []
    n_points, n_heights = len(points), len(heights)
    point_seconds = np.zeros((n_points, n_heights), np.float32)
    writer = ShardWriter(faces_dir, args.shard_size)
    validation_samples: list[dict[str, Any]] = []
    try:
        with instance.begin_frame():
            for _ in FACE_YAWS_DEG:
                cameras.append(spawn_depth_camera(
                    game, hfov_deg=FACE_HFOV_DEG, width=args.face_width,
                    height=args.face_height, camera_blueprint=VISUAL.CAMERA_BLUEPRINT))
            first_xyz = [points[0][0], points[0][1],
                         facts["ground_z_ue_cm"] + heights[0] * 100.0]
            for (cam, _), face_yaw in zip(cameras, FACE_YAWS_DEG):
                set_pose(cam, first_xyz, face_yaw)
            game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(bPaused=False)
        with instance.end_frame():
            pass
        instance.step(num_frames=args.warmup_frames)

        for hi, height_m in enumerate(heights):
            for pi, xy in enumerate(points):
                tick = time.time()
                xyz = [xy[0], xy[1], facts["ground_z_ue_cm"] + height_m * 100.0]
                with instance.begin_frame():
                    for (cam, _), face_yaw in zip(cameras, FACE_YAWS_DEG):
                        set_pose(cam, xyz, face_yaw)
                with instance.end_frame():
                    pass
                instance.step(num_frames=args.settle_frames)
                with instance.begin_frame():
                    for (cam, _), face_yaw in zip(cameras, FACE_YAWS_DEG):
                        set_pose(cam, xyz, face_yaw)
                with instance.end_frame():
                    faces = np.stack([read_depth(component) for _, component in cameras])
                writer.add(np.stack([min_pool(f, args.store_downsample) for f in faces]),
                           pi, hi)
                point_seconds[pi, hi] = time.time() - tick
                if (pi + 1) % 100 == 0 or pi + 1 == n_points:
                    print(f"height {height_m:.3f} m: {pi + 1}/{n_points} points, "
                          f"{point_seconds[:pi + 1, hi].sum():.1f} s", flush=True)
        writer.flush()

        if args.validate_yaws > 0:
            validation_samples = render_validation_samples(
                args=args, instance=instance, game=game, points=points, heights=heights,
                ground_z=facts["ground_z_ue_cm"], contract_hfov=contract_hfov,
                output=output, blueprint=VISUAL.CAMERA_BLUEPRINT)
    finally:
        try:
            close_cameras(instance=instance, game=game, cameras=cameras)
        finally:
            instance.close(force=True)
    return {"launch_seconds": launch_seconds, "point_seconds": point_seconds,
            "shards": writer.records, "validation_samples": validation_samples,
            "engine_wall_clock_seconds": time.time() - started}


def render_validation_samples(*, args, instance, game, points, heights, ground_z,
                              contract_hfov, output, blueprint) -> list[dict[str, Any]]:
    """Render the production camera directly at random (point, yaw) pairs.

    The verdict metric is computed on the full-resolution frame here; the
    frame itself is kept downsampled for the per-pixel depth comparison."""
    rng = np.random.default_rng(int(args.seed) + 1)
    direct_dir = output / "validation_direct"
    direct_dir.mkdir()
    with instance.begin_frame():
        camera, component = spawn_depth_camera(
            game, hfov_deg=contract_hfov, width=args.contract_width,
            height=args.contract_height, camera_blueprint=blueprint)
    with instance.end_frame():
        pass
    samples: list[dict[str, Any]] = []
    try:
        for k in range(args.validate_yaws):
            pi = int(rng.integers(len(points)))
            hi = int(rng.integers(len(heights)))
            yaw = float(rng.uniform(0.0, 360.0))
            xyz = [points[pi][0], points[pi][1], ground_z + heights[hi] * 100.0]
            with instance.begin_frame():
                set_pose(camera, xyz, yaw)
            with instance.end_frame():
                pass
            instance.step(num_frames=args.settle_frames)
            with instance.begin_frame():
                set_pose(camera, xyz, yaw)
            with instance.end_frame():
                direct = read_depth(component)
            stats = clearance_statistics(
                direct, args.near_m, hfov_deg=contract_hfov,
                camera_height_m=heights[hi],
                target_height_m=args.verdict_target_height_m,
                target_distance_range_m=args.target_distance_range_m)
            stride = args.contract_height // SUMMARY_GRID_HW[0]
            np.savez_compressed(direct_dir / f"direct_{k:03d}.npz",
                                depth_m=direct[::stride, ::stride].astype(np.float16),
                                stride=np.int32(stride))
            samples.append({
                "sample": k, "point_index": pi, "height_index": hi,
                "point_key": point_key(points[pi]), "camera_height_m": heights[hi],
                "yaw_deg": yaw,
                "direct_fraction": float(stats["target_band_blocked_column_fraction"][
                    f"{float(args.verdict_near_m):g}"]),
                "direct_path": f"validation_direct/direct_{k:03d}.npz"})
    finally:
        close_cameras(instance=instance, game=game, cameras=[(camera, component)])
    return samples


# ---------------------------------------------------------------------------
# cpu phases
# ---------------------------------------------------------------------------

def summaries_phase(*, output, shards, heights, points, contract_hfov, args
                    ) -> tuple[np.ndarray, np.ndarray, list[float]]:
    yaws = [i * args.yaw_step_deg for i in range(int(round(360.0 / args.yaw_step_deg)))]
    n_points, n_heights = len(points), len(heights)
    target = np.full((n_points, n_heights, len(args.target_heights_m), len(args.near_m),
                      len(yaws)), np.nan, np.float32)
    eye = np.full((n_points, n_heights, len(args.near_m), len(yaws)), np.nan, np.float32)
    tasks = [{"path": str(output / shard["path"]), "hfov_deg": contract_hfov,
              "heights": list(heights), "yaws": yaws, "nears": list(args.near_m),
              "target_heights": list(args.target_heights_m),
              "target_distance_range": list(args.target_distance_range_m)}
             for shard in shards]
    workers = max(1, min(args.workers, len(tasks)))
    if workers == 1:
        results = [_summaries_for_shard(task) for task in tasks]
    else:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            results = pool.map(_summaries_for_shard, tasks)
    for result in results:
        for i, (pi, hi) in enumerate(zip(result["point_index"], result["height_index"])):
            target[pi, hi] = result["target"][i]
            eye[pi, hi] = result["eye"][i]
    _require(np.isfinite(target).all(), "summaries left undefined entries")
    return target, eye, yaws


def load_faces(output: Path, shards) -> dict[tuple[int, int], np.ndarray]:
    faces: dict[tuple[int, int], np.ndarray] = {}
    for shard in shards:
        with np.load(output / shard["path"]) as data:
            radial = data["radial_m"]
            for i, (pi, hi) in enumerate(zip(data["point_index"], data["height_index"])):
                faces[(int(pi), int(hi))] = radial[i].astype(np.float32)
    return faces


def validation_phase(*, output, samples, shards, heights, target, yaws, contract_hfov,
                     args) -> list[dict[str, Any]]:
    """Compare direct renders with the table: verdict metric at the exact yaw
    (re-projected from stored faces on the summary grid) and at the yaw bin,
    plus per-pixel depth agreement on the summary grid."""
    if not samples:
        return []
    faces = load_faces(output, shards)
    camera = VirtualCamera(contract_hfov, SUMMARY_GRID_HW[1], SUMMARY_GRID_HW[0])
    h_index = args.target_heights_m.index(args.verdict_target_height_m)
    n_index = args.near_m.index(args.verdict_near_m)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        pi, hi, yaw = sample["point_index"], sample["height_index"], sample["yaw_deg"]
        with np.load(output / sample["direct_path"]) as data:
            direct = data["depth_m"].astype(np.float32)
        reprojected = camera.reproject_depth(faces[(pi, hi)], yaw)
        band = target_band_rows(
            camera.height, hfov_deg=contract_hfov, aspect=camera.aspect,
            camera_height_m=heights[hi], target_height_m=args.verdict_target_height_m,
            distance_range_m=args.target_distance_range_m)
        exact_fraction = blocked_column_fraction(reprojected, band, args.verdict_near_m)
        bin_fraction = float(target[pi, hi, h_index, n_index,
                                    yaw_bin_index(yaw, args.yaw_step_deg)])
        valid = (np.isfinite(reprojected) & (reprojected < NO_HIT_M)
                 & (direct < NO_HIT_M) & (direct > 0.0))
        rel = np.abs(reprojected[valid] - direct[valid]) / np.maximum(direct[valid], 1e-3)
        rows.append({
            **sample,
            "table_exact_yaw_fraction": exact_fraction,
            "table_bin_fraction": bin_fraction,
            "verdict_direct": bool(sample["direct_fraction"] <= args.blocked_fraction_max),
            "verdict_table_exact": bool(exact_fraction <= args.blocked_fraction_max),
            "verdict_table_bin": bool(bin_fraction <= args.blocked_fraction_max),
            "depth_valid_pixel_fraction": float(valid.mean()),
            "depth_rel_error_median": float(np.median(rel)) if rel.size else None,
            "depth_rel_error_p90": float(np.percentile(rel, 90)) if rel.size else None,
            "depth_within_5pct_fraction": float((rel <= 0.05).mean()) if rel.size else None,
        })
    (output / "validation_direct.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows


def validation_summary(rows: list[dict[str, Any]], args) -> dict[str, Any]:
    exact_agree = sum(r["verdict_direct"] == r["verdict_table_exact"] for r in rows)
    bin_agree = sum(r["verdict_direct"] == r["verdict_table_bin"] for r in rows)
    diffs_exact = [abs(r["direct_fraction"] - r["table_exact_yaw_fraction"]) for r in rows]
    diffs_bin = [abs(r["direct_fraction"] - r["table_bin_fraction"]) for r in rows]
    medians = [r["depth_rel_error_median"] for r in rows if r["depth_rel_error_median"] is not None]
    within = [r["depth_within_5pct_fraction"] for r in rows if r["depth_within_5pct_fraction"] is not None]
    return {
        "samples": len(rows),
        "verdict_agreement_exact_yaw": f"{exact_agree}/{len(rows)}",
        "verdict_agreement_yaw_bin": f"{bin_agree}/{len(rows)}",
        "fraction_abs_diff_exact_yaw": {"median": float(np.median(diffs_exact)),
                                        "max": float(np.max(diffs_exact))},
        "fraction_abs_diff_yaw_bin": {"median": float(np.median(diffs_bin)),
                                      "max": float(np.max(diffs_bin))},
        "depth_rel_error_median_over_samples": {"median": float(np.median(medians)),
                                                "max": float(np.max(medians))} if medians else None,
        "depth_within_5pct_fraction_mean": float(np.mean(within)) if within else None,
        "rows_path": "validation_direct.json",
        "rule": {"target_height_m": args.verdict_target_height_m,
                 "near_m": args.verdict_near_m,
                 "blocked_fraction_max": args.blocked_fraction_max},
        "note": ("direct_fraction is computed on the full-resolution direct render; "
                 "table values come from stored min-pooled faces on the summary grid"),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> Path:
    from avengine.timeline import current_apartment_visual as VISUAL

    config = read_scene_config(args.scene_config)
    scene = load_scene(config)
    facts = scene_render_facts(config, scene)
    contract_hfov = scene_hfov_deg(config)
    heights = [float(scene.camera_height_m)] + [
        float(h) for h in (args.extra_camera_height_m or [])
        if not math.isclose(float(h), float(scene.camera_height_m))]
    points = list(scene.camera_points)
    if args.point_limit is not None:
        rng = np.random.default_rng(int(args.seed))
        chosen = sorted(rng.choice(len(points), size=min(args.point_limit, len(points)),
                                   replace=False).tolist())
        points = [points[i] for i in chosen]

    output = VISUAL._new_external_output_directory(args.output, owner="clearance table output")
    started = time.time()
    rendered = render_phase(args=args, facts=facts, heights=heights, points=points,
                            contract_hfov=contract_hfov, output=output)
    cpu_started = time.time()
    target, eye, yaws = summaries_phase(
        output=output, shards=rendered["shards"], heights=heights, points=points,
        contract_hfov=contract_hfov, args=args)
    summaries_seconds = time.time() - cpu_started
    rows = validation_phase(
        output=output, samples=rendered["validation_samples"], shards=rendered["shards"],
        heights=heights, target=target, yaws=yaws, contract_hfov=contract_hfov, args=args)

    default_h = args.target_heights_m.index(args.verdict_target_height_m)
    default_n = args.near_m.index(args.verdict_near_m)
    clear = target[:, :, default_h, default_n, :] <= args.blocked_fraction_max
    point_seconds = rendered["point_seconds"]
    np.savez_compressed(
        output / "summaries.npz",
        target_band_blocked_column_fraction=target.astype(np.float16),
        eye_band_blocked_column_fraction=eye.astype(np.float16),
        clear_default_rule=clear,
        points_xy_cm=np.asarray(points, np.float32),
        camera_heights_m=np.asarray(heights, np.float32),
        yaws_deg=np.asarray(yaws, np.float32),
        nears_m=np.asarray(args.near_m, np.float32),
        target_heights_m=np.asarray(args.target_heights_m, np.float32),
        point_seconds=point_seconds)
    n_points, n_heights = len(points), len(heights)
    index = {
        "schema": SCHEMA,
        "status": "research_candidate",
        "qualification_claim": False,
        "claim_boundary": (
            "actor-free depth ring per solver camera point; a placement input "
            "for candidate generation, not pixel truth and not question admission"),
        "scene_id": scene.scene_id,
        "scene_config": str(Path(args.scene_config).resolve()),
        "scene_config_content": config,
        "stage": {"stage_root": str(Path(args.stage_root).resolve()),
                  "spear_executable": str(Path(args.spear_executable).resolve()),
                  "native_map": facts["native_map"],
                  "room_profile_id": facts["room_profile_id"]},
        "code": _git_worktree_state(),
        "camera_contract": {"hfov_deg": contract_hfov,
                            "resolution_hw": [args.contract_height, args.contract_width],
                            "camera_heights_m": heights,
                            "ground_z_ue_cm": facts["ground_z_ue_cm"]},
        "faces": {"count_per_point": len(FACE_YAWS_DEG), "yaws_deg": list(FACE_YAWS_DEG),
                  "hfov_deg": FACE_HFOV_DEG, "render_hw": [args.face_height, args.face_width],
                  "stored_hw": [args.face_height // args.store_downsample,
                                args.face_width // args.store_downsample],
                  "store_downsample": args.store_downsample,
                  "store_pooling": "min", "depth_convention": "radial_metres",
                  "no_hit_sentinel_m": NO_HIT_M, "dtype": "float16",
                  "shards": rendered["shards"]},
        "summaries": {"path": "summaries.npz", "summary_grid_hw": list(SUMMARY_GRID_HW),
                      "yaw_step_deg": args.yaw_step_deg, "yaw_count": len(yaws),
                      "nears_m": list(args.near_m),
                      "target_heights_m": list(args.target_heights_m),
                      "target_distance_range_m": list(args.target_distance_range_m),
                      "array_axes": {
                          "target_band_blocked_column_fraction":
                              ["point", "camera_height", "target_height", "near", "yaw"],
                          "eye_band_blocked_column_fraction":
                              ["point", "camera_height", "near", "yaw"],
                          "clear_default_rule": ["point", "camera_height", "yaw"]},
                      "column_rule": "median of finite band pixels per column < near"},
        "default_rule": {"metric": "target_band_blocked_column_fraction",
                         "target_height_m": args.verdict_target_height_m,
                         "near_m": args.verdict_near_m,
                         "blocked_fraction_max": args.blocked_fraction_max,
                         "status": "placeholder_research_not_human_calibrated"},
        "points": {"count": n_points, "keys": [point_key(xy) for xy in points],
                   "source": "scene_sampler.load_scene camera_points",
                   "point_limit": args.point_limit,
                   "navigable_points_in_scene": len(scene.camera_points)},
        "coverage": {
            "points_with_any_clear_yaw": [int(clear[:, hi, :].any(axis=1).sum())
                                          for hi in range(n_heights)],
            "clear_yaw_fraction_mean": [float(clear[:, hi, :].mean())
                                        for hi in range(n_heights)]},
        "timing": {"launch_seconds": rendered["launch_seconds"],
                   "render_seconds_total": float(point_seconds.sum()),
                   "seconds_per_point_mean": float(point_seconds.mean()),
                   "engine_wall_clock_seconds": rendered["engine_wall_clock_seconds"],
                   "summaries_seconds": summaries_seconds,
                   "summaries_workers": args.workers,
                   "wall_clock_total_seconds": time.time() - started,
                   "warmup_frames": args.warmup_frames,
                   "settle_frames": args.settle_frames},
        "validation": validation_summary(rows, args) if rows else None,
    }
    (output / "camera_clearance_table.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"QA_V3_CLEARANCE_TABLE_OK output={output} points={n_points} heights={n_heights} "
          f"seconds_per_point={point_seconds.mean():.3f} summaries_s={summaries_seconds:.1f}",
          flush=True)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-config", required=True, type=Path)
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--spear-executable", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extra-camera-height-m", type=float, action="append",
                        help="additional camera heights to render (the scene height is always rendered)")
    parser.add_argument("--face-width", type=int, default=512)
    parser.add_argument("--face-height", type=int, default=768)
    parser.add_argument("--store-downsample", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--contract-width", type=int, default=1280)
    parser.add_argument("--contract-height", type=int, default=720)
    parser.add_argument("--near-m", type=float, action="append")
    parser.add_argument("--target-heights-m", type=float, nargs="+",
                        default=list(DEFAULT_TARGET_HEIGHTS_M))
    parser.add_argument("--target-distance-range-m", type=float, nargs=2,
                        default=list(DEFAULT_TARGET_DISTANCE_RANGE_M))
    parser.add_argument("--yaw-step-deg", type=float, default=DEFAULT_YAW_STEP_DEG)
    parser.add_argument("--verdict-target-height-m", type=float, default=0.5)
    parser.add_argument("--verdict-near-m", type=float, default=1.5)
    parser.add_argument("--blocked-fraction-max", type=float, default=0.2)
    parser.add_argument("--point-limit", type=int,
                        help="render a random subset of camera points (smoke runs)")
    parser.add_argument("--validate-yaws", type=int, default=0,
                        help="number of random (point, yaw) direct renders to compare")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="processes for the summary phase")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--rpc-port", type=int, default=39561)
    parser.add_argument("--graphics-adapter", type=int, default=1)
    parser.add_argument("--warmup-frames", type=int, default=40)
    parser.add_argument("--settle-frames", type=int, default=2)
    args = parser.parse_args(argv)
    args.near_m = list(args.near_m) if args.near_m else list(DEFAULT_NEAR_M)
    if args.face_width <= 0 or args.face_height <= 0:
        parser.error("face size must be positive")
    if args.store_downsample < 1 or args.face_width % args.store_downsample or \
            args.face_height % args.store_downsample:
        parser.error("--store-downsample must divide the face size")
    if args.contract_height % SUMMARY_GRID_HW[0] or args.contract_width % SUMMARY_GRID_HW[1]:
        parser.error("contract resolution must be a multiple of the summary grid")
    if not 0.0 < args.yaw_step_deg <= 90.0 or abs(360.0 / args.yaw_step_deg
                                                  - round(360.0 / args.yaw_step_deg)) > 1e-9:
        parser.error("--yaw-step-deg must divide 360")
    if args.verdict_near_m not in args.near_m:
        parser.error("--verdict-near-m must be one of --near-m")
    if args.verdict_target_height_m not in args.target_heights_m:
        parser.error("--verdict-target-height-m must be one of --target-heights-m")
    if not 0.0 <= args.blocked_fraction_max <= 1.0:
        parser.error("--blocked-fraction-max must be within [0,1]")
    if not 1024 <= args.rpc_port <= 65535:
        parser.error("--rpc-port must be in [1024,65535]")
    if args.point_limit is not None and args.point_limit <= 0:
        parser.error("--point-limit must be positive")
    if args.validate_yaws < 0:
        parser.error("--validate-yaws must be >= 0")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
