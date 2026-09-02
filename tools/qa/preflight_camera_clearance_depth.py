#!/usr/bin/env python3
"""Camera-only depth preflight: is the view from a candidate camera pose clear?

Thirteen of 37 fresh card1 candidates on 2026-09-02 had their camera parked
behind a floor-lamp shade, a sofa back, an open door leaf or a kitchen island,
so the dogs were invisible before any question was asked.  A 2-D navigable
raster could not separate those cameras from clear ones (see the calibration
log referenced in the readiness report): it cannot see eye-height occluders.
This tool renders one actor-free depth frame per candidate pose at the real
camera height and reports how much of the frame is filled by geometry closer
than explicit near distances.  It is a placement prefilter for candidate
generation, not pixel truth and not question admission.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "tools" / "qa"))

from avengine.backends.spear_ue.research_runtime import (  # noqa: E402
    launch_external_game_instance,
)
from avengine.timeline import current_apartment_visual as VISUAL  # noqa: E402
from scene_sampler import scene_hfov_deg  # noqa: E402

SPIKE_PATH = REPOSITORY / "tools/qa/spike_spear_native_pixel_visibility.py"
SCHEMA = "qa_v3_camera_clearance_depth_preflight_v1"
DEFAULT_NEAR_M = (1.0, 1.5, 2.5)
VERDICT_METRICS = ("near_fraction", "eye_band_blocked_column_fraction",
                   "target_band_blocked_column_fraction")
DEPTH_DOWNSAMPLE = 4


def _load_spike():
    spec = importlib.util.spec_from_file_location("qa_v3_camera_preflight_helpers",
                                                  SPIKE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SPIKE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def poses_from_facts(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """One pose per fact.  Point ids repeat across batches (core batch and
    smoke both have card1F_002), so a repeated id gets a running suffix; the
    resolved fact path stays the traceable identity."""
    poses = []
    seen: dict[str, int] = {}
    for path in paths:
        fact = json.loads(Path(path).read_text(encoding="utf-8"))
        camera = fact["camera"]
        base = f"{fact.get('scene_id')}::{fact['point_id']}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        poses.append({
            "pose_id": base if count == 0 else f"{base}#{count + 1}",
            "translation_ue_cm": [float(v) for v in camera["ue_cm"]],
            "yaw_ue_deg": float(camera["ue_yaw_deg"]),
            "source_fact": str(Path(path).resolve()),
        })
    return poses


def load_poses(args) -> list[dict[str, Any]]:
    poses = []
    if args.poses:
        value = json.loads(Path(args.poses).read_text(encoding="utf-8"))
        rows = value.get("poses") if isinstance(value, dict) else value
        for row in rows:
            poses.append({
                "pose_id": str(row["pose_id"]),
                "translation_ue_cm": [float(v) for v in row["translation_ue_cm"]],
                "yaw_ue_deg": float(row["yaw_ue_deg"]),
                "source_fact": row.get("source_fact"),
            })
    if args.fact:
        poses.extend(poses_from_facts(args.fact))
    _require(bool(poses), "no camera poses supplied (--poses or --fact)")
    ids = [pose["pose_id"] for pose in poses]
    _require(len(ids) == len(set(ids)), "pose ids must be unique")
    for pose in poses:
        _require(all(math.isfinite(v) for v in pose["translation_ue_cm"])
                 and math.isfinite(pose["yaw_ue_deg"]),
                 f"non-finite pose {pose['pose_id']}")
    return poses


def target_band_rows(height: int, *, hfov_deg: float, aspect: float,
                     camera_height_m: float, target_height_m: float,
                     distance_range_m: Sequence[float]) -> tuple[int, int]:
    """Rows where a floor-standing target of the given height appears.

    A 0.5 m dog standing 2.5 to 10 m from a 1.47 m camera projects between
    roughly 5 and 30 degrees below the optical axis, i.e. the lower half of a
    16:9 frame with a 105 degree horizontal FOV.  Sofa backs and kitchen
    islands sit below eye height and only block this band, which is why an
    eye-height band misses them."""
    half_v = math.atan(math.tan(math.radians(hfov_deg / 2.0)) / aspect)
    near_d, far_d = sorted(float(v) for v in distance_range_m)
    top_depression = math.atan((camera_height_m - target_height_m) / far_d)
    bottom_depression = math.atan(camera_height_m / near_d)
    def row(depression):
        ndc = math.tan(depression) / math.tan(half_v)      # +1 = bottom edge
        return int(round((min(max(ndc, -1.0), 1.0) + 1.0) / 2.0 * height))
    top, bottom = row(top_depression), row(bottom_depression)
    return max(0, min(top, height - 1)), max(top + 1, min(bottom, height))


def _blocked_columns(depth, rows, near):
    band = depth[rows[0]: rows[1], :]
    band_finite = np.isfinite(band) & (band > 0.0)
    column_median = np.nanmedian(np.where(band_finite, band, np.nan), axis=0)
    return np.isfinite(column_median) & (column_median < near)


def _spans(blocked_columns, width, hfov_deg):
    spans, start = [], None
    for index, blocked in enumerate(list(blocked_columns) + [False]):
        if blocked and start is None:
            start = index
        elif not blocked and start is not None:
            spans.append([_column_to_azimuth(start, width, hfov_deg),
                          _column_to_azimuth(index - 1, width, hfov_deg)])
            start = None
    return spans


def clearance_statistics(depth_m: np.ndarray, near_m: Sequence[float],
                         hfov_deg: float, *, camera_height_m: float = 1.47,
                         target_height_m: float = 0.5,
                         target_distance_range_m: Sequence[float] = (2.5, 10.0)
                         ) -> dict[str, Any]:
    """Frame-level clearance numbers from one actor-free depth image."""
    depth = np.asarray(depth_m, dtype=np.float64)
    finite = np.isfinite(depth) & (depth > 0.0)
    total = int(finite.sum())
    _require(total > 0, "depth frame carries no finite pixels")
    height, width = depth.shape
    eye_rows = (height // 3, 2 * height // 3)
    target_rows = target_band_rows(
        height, hfov_deg=hfov_deg, aspect=width / height,
        camera_height_m=camera_height_m, target_height_m=target_height_m,
        distance_range_m=target_distance_range_m)
    stats = {
        "finite_pixel_fraction": total / depth.size,
        "min_depth_m": float(depth[finite].min()),
        "median_depth_m": float(np.median(depth[finite])),
        "eye_band_rows": list(eye_rows),
        "target_band_rows": list(target_rows),
        "target_band_definition": {
            "camera_height_m": camera_height_m,
            "target_height_m": target_height_m,
            "target_distance_range_m": [float(v) for v in target_distance_range_m],
            "status": "placeholder_research_geometry",
        },
        "near_fraction": {},
        "eye_band_blocked_column_fraction": {},
        "target_band_blocked_column_fraction": {},
        "blocked_azimuth_spans_deg": {},
        "target_band_blocked_azimuth_spans_deg": {},
    }
    for near in near_m:
        key = f"{float(near):g}"
        stats["near_fraction"][key] = float((depth[finite] < near).mean())
        eye_blocked = _blocked_columns(depth, eye_rows, near)
        target_blocked = _blocked_columns(depth, target_rows, near)
        stats["eye_band_blocked_column_fraction"][key] = float(eye_blocked.mean())
        stats["target_band_blocked_column_fraction"][key] = float(target_blocked.mean())
        stats["blocked_azimuth_spans_deg"][key] = _spans(eye_blocked, width, hfov_deg)
        stats["target_band_blocked_azimuth_spans_deg"][key] = _spans(
            target_blocked, width, hfov_deg)
    return stats


def _column_to_azimuth(column: int, width: int, hfov_deg: float) -> float:
    half = math.tan(math.radians(hfov_deg / 2.0))
    ndc = (column + 0.5) / width * 2.0 - 1.0
    return round(math.degrees(math.atan(ndc * half)), 2)


def verdict(stats: dict[str, Any], *, near_m: float,
            blocked_fraction_max: float,
            metric: str = "target_band_blocked_column_fraction") -> dict[str, Any]:
    _require(metric in VERDICT_METRICS, f"unknown verdict metric {metric!r}")
    key = f"{float(near_m):g}"
    fraction = stats[metric][key]
    return {
        "rule": (f"{metric} at {near_m:g} m must be <= {blocked_fraction_max:g}"),
        "metric": metric,
        "near_m": float(near_m),
        "blocked_fraction_max": float(blocked_fraction_max),
        "blocked_fraction": fraction,
        "camera_view_clear": bool(fraction <= blocked_fraction_max),
        "status": "placeholder_research_not_human_calibrated",
    }


def _close_camera(*, instance, game, camera, components) -> None:
    if camera is None and not components:
        return
    with instance.begin_frame():
        pass
    with instance.end_frame():
        for component in components.values():
            try:
                component.terminate_sp_funcs()
            finally:
                component.Terminate()
        if camera is not None:
            game.unreal_service.destroy_actor(actor=camera)


def resolve_camera_hfov(args: argparse.Namespace) -> dict[str, Any]:
    """The camera HFOV comes from the scene contract or an explicit flag.

    Nothing here hard-codes 105 degrees: the value used for the depth camera
    and for the band geometry is the one recorded in the output, together
    with where it came from."""
    if getattr(args, "hfov_deg", None) is not None:
        value = float(args.hfov_deg)
        source = "cli:--hfov-deg"
    elif getattr(args, "scene_config", None) is not None:
        config = json.loads(Path(args.scene_config).read_text(encoding="utf-8"))
        value = scene_hfov_deg(config)
        source = f"scene_config:{Path(args.scene_config).resolve()}"
    else:
        raise RuntimeError(
            "camera HFOV must come from --scene-config or --hfov-deg")
    _require(0.0 < value < 180.0, f"implausible camera HFOV {value}")
    return {"hfov_deg": value, "source": source}


def run(args: argparse.Namespace) -> Path:
    from PIL import Image

    spike = _load_spike()
    poses = load_poses(args)
    camera_contract = resolve_camera_hfov(args)
    hfov_deg = camera_contract["hfov_deg"]
    output = VISUAL._new_external_output_directory(args.output, owner="preflight output")
    thumbs = output / "thumbnails"
    if args.save_png:
        thumbs.mkdir()
    depth_dir = output / "depth"
    depth_dir.mkdir()
    executable = Path(args.spear_executable)
    _require(executable.is_file(), f"missing SpearSim executable: {executable}")
    _require(Path(args.stage_root).is_dir(), f"missing stage root: {args.stage_root}")
    instance = launch_external_game_instance(
        spear_executable=executable, native_map=args.native_map,
        frame_rate_hz=VISUAL.FRAME_RATE_HZ, rpc_port=args.rpc_port,
        graphics_adapter=args.graphics_adapter)
    game = instance.get_game()
    camera = None
    components: dict[str, Any] = {}
    records = []
    try:
        with instance.begin_frame():
            camera, components = spike._spawn_multimodal_camera(
                game, hfov_deg=hfov_deg)
            components["depth"].PrimitiveRenderMode = "PRM_RenderScenePrimitives"
            components["depth"].ShowOnlyActors = []
            _set_pose(camera, poses[0])
            game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(bPaused=False)
        with instance.end_frame():
            pass
        instance.step(num_frames=args.warmup_frames)
        for pose in poses:
            with instance.begin_frame():
                _set_pose(camera, pose)
            with instance.end_frame():
                pass
            instance.step(num_frames=args.settle_frames)
            with instance.begin_frame():
                _set_pose(camera, pose)
            with instance.end_frame():
                capture = spike._capture_buffers(
                    game=game, components=components, include_rgb_depth=True)
            depth = np.asarray(capture["depth_m"], dtype=np.float32)
            camera_height_m = float(pose["translation_ue_cm"][2]) / 100.0
            stats = clearance_statistics(
                depth, args.near_m, hfov_deg=hfov_deg,
                camera_height_m=camera_height_m,
                target_height_m=args.target_height_m,
                target_distance_range_m=args.target_distance_range_m)
            safe = "".join(ch if ch.isalnum() or ch in "-_." else "_"
                           for ch in pose["pose_id"])
            # keep a downsampled raw depth so band definitions can be
            # re-analysed offline without relaunching the engine
            small = depth[::DEPTH_DOWNSAMPLE, ::DEPTH_DOWNSAMPLE]
            np.savez_compressed(depth_dir / f"{safe}.npz", depth_m=small,
                                downsample=np.int32(DEPTH_DOWNSAMPLE))
            record = {
                **pose,
                "resolution_hw": list(depth.shape),
                "depth_npz": f"depth/{safe}.npz",
                "statistics": stats,
                "verdict": verdict(stats, near_m=args.verdict_near_m,
                                   blocked_fraction_max=args.blocked_fraction_max,
                                   metric=args.verdict_metric),
            }
            if args.save_png:
                Image.fromarray(np.ascontiguousarray(
                    capture["rgb"][:, :, ::-1]), mode="RGB").save(thumbs / f"{safe}_rgb.png")
                shown = np.clip(depth / 5.0, 0.0, 1.0)
                Image.fromarray((255 * (1.0 - shown)).astype(np.uint8), mode="L").save(
                    thumbs / f"{safe}_depth_0to5m.png")
                record["thumbnails"] = {"rgb": f"thumbnails/{safe}_rgb.png",
                                        "depth": f"thumbnails/{safe}_depth_0to5m.png"}
            records.append(record)
    finally:
        try:
            _close_camera(instance=instance, game=game, camera=camera,
                          components=components)
        finally:
            instance.close(force=True)
    summary = {
        "schema": SCHEMA,
        "status": "research_candidate",
        "qualification_claim": False,
        "claim_boundary": (
            "actor-free camera clearance preflight from one native depth frame "
            "per pose; a placement prefilter, not pixel truth and not question "
            "admission"),
        "native_map": args.native_map,
        "inputs": {"stage_root": str(Path(args.stage_root).resolve()),
                   "spear_executable": str(executable.resolve()),
                   "poses": str(Path(args.poses).resolve()) if args.poses else None,
                   "facts": [str(Path(p).resolve()) for p in (args.fact or [])]},
        "camera_contract": {"hfov_deg": hfov_deg,
                            "hfov_source": camera_contract["source"],
                            "eye_band": "middle third of rows",
                            "target_band": "rows where a floor-standing target "
                                           "of --target-height-m at "
                                           "--target-distance-range-m projects"},
        "near_m": [float(v) for v in args.near_m],
        "verdict_rule": {"metric": args.verdict_metric,
                         "near_m": float(args.verdict_near_m),
                         "blocked_fraction_max": float(args.blocked_fraction_max),
                         "status": "placeholder_research_not_human_calibrated"},
        "depth_downsample": DEPTH_DOWNSAMPLE,
        "pose_count": len(records),
        "clear_count": sum(record["verdict"]["camera_view_clear"] for record in records),
        "poses": records,
    }
    (output / "camera_clearance_preflight.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8")
    print(f"QA_V3_CAMERA_PREFLIGHT_OK output={output} poses={len(records)} "
          f"clear={summary['clear_count']}", flush=True)
    return output


def _set_pose(camera, pose) -> None:
    x, y, z = pose["translation_ue_cm"]
    camera.K2_SetActorLocationAndRotation(
        NewLocation={"X": x, "Y": y, "Z": z},
        NewRotation={"Roll": 0.0, "Pitch": 0.0, "Yaw": pose["yaw_ue_deg"]},
        bSweep=False, bTeleport=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--spear-executable", required=True)
    parser.add_argument("--native-map", required=True)
    parser.add_argument("--poses", help="JSON list of {pose_id, translation_ue_cm, yaw_ue_deg}")
    parser.add_argument("--fact", action="append", type=Path,
                        help="fact_record.json whose camera pose is probed")
    parser.add_argument("--output", required=True)
    parser.add_argument("--near-m", type=float, action="append")
    parser.add_argument("--verdict-near-m", type=float, default=1.5)
    parser.add_argument("--blocked-fraction-max", type=float, default=0.2)
    # Default chosen on 2026-09-02 from 37 candidate cameras with native pixel
    # truth: the target-band column fraction at 1.5 m with cap 0.2 flagged all
    # 16 cameras whose pixel join showed camera-side blockage and none of the
    # other 21.  Still a research placeholder.
    parser.add_argument("--verdict-metric", choices=VERDICT_METRICS,
                        default="target_band_blocked_column_fraction")
    parser.add_argument("--target-height-m", type=float, default=0.5)
    parser.add_argument("--target-distance-range-m", type=float, nargs=2,
                        default=(2.5, 10.0))
    parser.add_argument("--scene-config", type=Path,
                        help="scene config whose camera contract supplies the "
                             "horizontal FOV (hfov_deg or camera_base_request)")
    parser.add_argument("--hfov-deg", type=float,
                        help="explicit horizontal FOV in degrees; overrides the "
                             "scene config and is recorded as such")
    parser.add_argument("--save-png", action="store_true")
    parser.add_argument("--rpc-port", type=int, default=39561)
    parser.add_argument("--graphics-adapter", type=int, default=1)
    parser.add_argument("--warmup-frames", type=int, default=40)
    parser.add_argument("--settle-frames", type=int, default=2)
    args = parser.parse_args(argv)
    args.near_m = tuple(args.near_m) if args.near_m else DEFAULT_NEAR_M
    if not 1024 <= args.rpc_port <= 65535:
        parser.error("--rpc-port must be in [1024,65535]")
    if args.verdict_near_m not in args.near_m:
        parser.error("--verdict-near-m must be one of --near-m")
    if not 0.0 <= args.blocked_fraction_max <= 1.0:
        parser.error("--blocked-fraction-max must be within [0,1]")
    if args.scene_config is None and args.hfov_deg is None:
        parser.error("camera HFOV must come from --scene-config or --hfov-deg")
    if args.hfov_deg is not None and not 0.0 < args.hfov_deg < 180.0:
        parser.error("--hfov-deg must lie in (0,180)")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
