#!/usr/bin/env python3
"""Measure a room's floor height in the engine and write its floor reference.

Why
---
See ``floor_reference.py``: the Apartment scene config carried a hand-written
``ground_z_ue_cm: 0.0`` while the cooked floor sits about 27 cm higher, so
every Apartment render put the dogs 27 cm into the floor and the camera 1.20 m
above it instead of 1.47 m.  From now on the floor offset of every room is
measured before anything else is designed or rendered for it.

How
---
The packaged game is launched on the room's own map with no actors, and the
floor is measured under the solver's own camera points (the route bank's
navigable points, the same set the clearance table covers) and under random
walkable cells of the room's walkable grid, with two independent methods:

* ``line_trace``: one vertical line trace per point (profile BlockAll,
  complex collision, the call the strict two-human ground-contact tool
  uses).  Fast and exact, but it needs collision on the floor mesh: the
  Kujiale baked-lit floor has none and every trace misses (2026-09-03).
* ``depth_capture``: a depth camera looks straight down from a known height;
  the central pixels' depth is the distance to the floor.  Works on any
  rendered floor because it is exactly what the production camera sees.

The floor height is the median of the primary method's hits (line trace when
it hits, depth otherwise); when both methods hit, their medians must agree
within ``--agreement-cm``.  Spread (p05/p95), the fraction of hits within
2 cm of the median and the hit component paths are stored so a split-level
room or a trace that landed on furniture is visible instead of averaged away.

Boundary
--------
Thresholds are research placeholders.  The tool never edits a scene config;
it writes a fresh room product and the operator points the config at it.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "qa"))

from floor_reference import (  # noqa: E402
    STATUS_MEASURED,
    summarize_floor_hits,
    write_floor_reference,
)

METHOD_LINE_TRACE = "ue_line_trace_down_blockall_complex_v1"
METHOD_DEPTH = "ue_depth_capture_straight_down_v1"
DEFAULT_THRESHOLDS = {
    # 占位:至少 200 次命中、命中率 98%、95% 的命中在中位数 ±2 cm 内才算"量到了一层地板"。
    "min_hits": 200,
    "min_hit_fraction": 0.98,
    "min_within_fraction": 0.95,
}
DEPTH_CENTER_HALF_WIDTH_PX = 2       # 中心 5×5 像素取中位数
DEPTH_NO_HIT_M = 65504.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def git_worktree_state(repo: Path = REPO) -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, text=True,
                              capture_output=True).stdout
    status = run("status", "--short").splitlines()
    return {"revision": run("rev-parse", "HEAD").strip(), "dirty": bool(status),
            "status": status}


def _lookup(mapping: Mapping[str, Any], key: str) -> Any:
    """Case-insensitive key lookup: UE JSON marshalling is not consistent about case."""
    if key in mapping:
        return mapping[key]
    wanted = key.lower()
    for candidate, value in mapping.items():
        if str(candidate).lower() == wanted:
            return value
    return None


def _xyz(value: Any) -> list[float] | None:
    if isinstance(value, Mapping):
        parts = [_lookup(value, axis) for axis in ("X", "Y", "Z")]
        if all(p is not None for p in parts):
            return [float(p) for p in parts]
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return [float(p) for p in value]
    return None


def parse_trace(raw: Any) -> dict[str, Any]:
    _require(isinstance(raw, Mapping), "line trace result is not a mapping")
    hit = bool(_lookup(raw, "ReturnValue"))
    record: dict[str, Any] = {"hit": hit}
    if not hit:
        return record
    out_hit = _lookup(raw, "OutHit")
    _require(isinstance(out_hit, Mapping), "trace OutHit is invalid")
    location = _xyz(_lookup(out_hit, "Location"))
    _require(location is not None, "trace hit Location is invalid")
    record["hit_point_ue_cm"] = location
    normal = _xyz(_lookup(out_hit, "Normal"))
    if normal is not None:
        record["hit_normal_ue"] = normal
    for name in ("Component", "Actor", "BoneName", "PhysMaterial"):
        value = _lookup(out_hit, name)
        if isinstance(value, str) and value:
            record[name.lower()] = value
    return record


def sample_points(config: dict, *, walkable_samples: int, margin_cm: float,
                  seed: int) -> tuple[list[tuple[float, float, str]], dict[str, Any]]:
    """The solver's camera points plus random walkable cells, tagged by origin."""
    import scene_sampler as SS  # noqa: WPS433 (tool-local import path)
    from walkable_grid import grid_from_config

    # 这个工具就是来产出 render 事实的,所以载入场景时不带 render 段
    # (带了就会被要求先有地板参照,而地板参照正是这里要量的)。
    stripped = {k: v for k, v in config.items()
                if k not in ("render", "camera_clearance_table", "floor_reference")}
    scene = SS.load_scene(stripped)
    points = [(float(x), float(y), "camera_point") for x, y in scene.camera_points]
    facts: dict[str, Any] = {"camera_points": len(points), "walkable_samples": 0,
                             "walkable_margin_cm": float(margin_cm)}
    grid_config = config.get("walkable_grid")
    if grid_config is not None and walkable_samples > 0:
        grid = grid_from_config(grid_config)
        _require(grid.scene_id == str(config["scene_id"]),
                 f"walkable grid belongs to {grid.scene_id!r}")
        rng = np.random.default_rng(seed)
        for _ in range(walkable_samples):
            xy = grid.sample_xy(rng, margin_cm)
            points.append((float(xy[0]), float(xy[1]), "walkable_cell"))
        facts["walkable_samples"] = int(walkable_samples)
        facts["walkable_grid"] = grid.identity
    return points, facts


def _launch(args, native_map: str):
    from avengine.backends.spear_ue.research_runtime import launch_external_game_instance
    from avengine.timeline import current_apartment_visual as VISUAL

    executable = Path(args.spear_executable)
    _require(executable.is_file(), f"missing SpearSim executable: {executable}")
    _require(Path(args.stage_root).is_dir(), f"missing stage root: {args.stage_root}")
    return launch_external_game_instance(
        spear_executable=executable, native_map=native_map,
        frame_rate_hz=VISUAL.FRAME_RATE_HZ, rpc_port=args.rpc_port,
        graphics_adapter=args.graphics_adapter)


def trace_points(instance, game, points: Sequence[tuple[float, float, str]],
                 start_z_cm: float, end_z_cm: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.time()
    rows: list[dict[str, Any]] = []
    raw_examples: list[Any] = []
    with instance.begin_frame():
        kismet = game.get_unreal_object(uclass="UKismetSystemLibrary")
        for index, (x, y, origin) in enumerate(points):
            start = {"X": x, "Y": y, "Z": float(start_z_cm)}
            end = {"X": x, "Y": y, "Z": float(end_z_cm)}
            raw = kismet.LineTraceSingleByProfile(
                Start=start, End=end, ProfileName="BlockAll", bTraceComplex=True,
                ActorsToIgnore=[], DrawDebugType="None", bIgnoreSelf=True,
                TraceColor={"R": 1.0, "G": 0.0, "B": 0.0, "A": 1.0},
                TraceHitColor={"R": 0.0, "G": 1.0, "B": 0.0, "A": 1.0},
                DrawTime=0.0, as_dict=True)
            if len(raw_examples) < 2 and isinstance(raw, Mapping):
                raw_examples.append(json.loads(json.dumps(raw, default=str)))
            record = parse_trace(raw)
            record.update({"method": "line_trace", "index": index, "origin": origin,
                           "xy_ue_cm": [x, y]})
            if record["hit"]:
                point = record["hit_point_ue_cm"]
                record["floor_z_ue_cm"] = point[2]
                record["horizontal_error_cm"] = max(abs(point[0] - x), abs(point[1] - y))
            rows.append(record)
            if (index + 1) % 1000 == 0 or index + 1 == len(points):
                print(f"line traces: {index + 1}/{len(points)} ({time.time() - started:.1f} s)",
                      flush=True)
    with instance.end_frame():
        pass
    facts = {"kind": METHOD_LINE_TRACE, "profile_name": "BlockAll", "trace_complex": True,
             "start_z_ue_cm": float(start_z_cm), "end_z_ue_cm": float(end_z_cm),
             "points": len(points), "seconds": round(time.time() - started, 1),
             "raw_examples": raw_examples}
    return rows, facts


def _look_down(camera, xyz_cm: Sequence[float]) -> None:
    camera.K2_SetActorLocationAndRotation(
        NewLocation={"X": float(xyz_cm[0]), "Y": float(xyz_cm[1]), "Z": float(xyz_cm[2])},
        NewRotation={"Roll": 0.0, "Pitch": -90.0, "Yaw": 0.0},
        bSweep=False, bTeleport=True)


def depth_points(args, instance, game, points: Sequence[tuple[float, float, str]],
                 camera_z_cm: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from avengine.timeline import current_apartment_visual as VISUAL
    from build_qa_v3_camera_clearance_table import (
        close_cameras,
        read_depth,
        spawn_depth_camera,
    )

    started = time.time()
    rows: list[dict[str, Any]] = []
    cameras: list[tuple[Any, Any]] = []
    half = DEPTH_CENTER_HALF_WIDTH_PX
    size = int(args.depth_size_px)
    try:
        with instance.begin_frame():
            cameras.append(spawn_depth_camera(
                game, hfov_deg=float(args.depth_hfov_deg), width=size, height=size,
                camera_blueprint=VISUAL.CAMERA_BLUEPRINT))
            _look_down(cameras[0][0], [points[0][0], points[0][1], camera_z_cm])
            game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(bPaused=False)
        with instance.end_frame():
            pass
        instance.step(num_frames=int(args.warmup_frames))
        camera, component = cameras[0]
        for index, (x, y, origin) in enumerate(points):
            xyz = [x, y, camera_z_cm]
            with instance.begin_frame():
                _look_down(camera, xyz)
            with instance.end_frame():
                pass
            instance.step(num_frames=int(args.settle_frames))
            with instance.begin_frame():
                _look_down(camera, xyz)
            with instance.end_frame():
                depth = read_depth(component)
            h, w = depth.shape
            center = depth[h // 2 - half:h // 2 + half + 1, w // 2 - half:w // 2 + half + 1]
            valid = center[np.isfinite(center) & (center < DEPTH_NO_HIT_M)]
            record: dict[str, Any] = {"method": "depth_capture", "index": index,
                                      "origin": origin, "xy_ue_cm": [x, y],
                                      "camera_z_ue_cm": float(camera_z_cm),
                                      "center_valid_pixels": int(valid.size),
                                      "hit": bool(valid.size >= (2 * half + 1) ** 2 // 2)}
            if record["hit"]:
                distance_m = float(np.median(valid))
                record["center_depth_m"] = distance_m
                record["center_depth_spread_m"] = float(valid.max() - valid.min())
                record["floor_z_ue_cm"] = float(camera_z_cm) - distance_m * 100.0
            rows.append(record)
            if (index + 1) % 100 == 0 or index + 1 == len(points):
                print(f"depth captures: {index + 1}/{len(points)} ({time.time() - started:.1f} s)",
                      flush=True)
    finally:
        close_cameras(instance=instance, game=game, cameras=cameras)
    facts = {"kind": METHOD_DEPTH, "camera_z_ue_cm": float(camera_z_cm),
             "pitch_deg": -90.0, "hfov_deg": float(args.depth_hfov_deg),
             "render_hw": [size, size], "center_window_px": 2 * half + 1,
             "depth_convention": "radial_metres (vertical at the image centre)",
             "warmup_frames": int(args.warmup_frames), "settle_frames": int(args.settle_frames),
             "points": len(points), "seconds": round(time.time() - started, 1)}
    return rows, facts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--spear-executable", required=True)
    parser.add_argument("--method", choices=("line_trace", "depth_capture", "both"),
                        default="both")
    parser.add_argument("--walkable-samples", type=int, default=1000)
    parser.add_argument("--walkable-margin-m", type=float, default=0.2)
    parser.add_argument("--start-offset-cm", type=float, default=150.0,
                        help="trace start / depth camera height above the configured ground_z_ue_cm")
    parser.add_argument("--trace-length-cm", type=float, default=450.0)
    parser.add_argument("--depth-point-limit", type=int, default=600,
                        help="depth captures are slower: at most this many camera points (random subset)")
    parser.add_argument("--depth-walkable-samples", type=int, default=300)
    parser.add_argument("--depth-size-px", type=int, default=64)
    parser.add_argument("--depth-hfov-deg", type=float, default=30.0)
    parser.add_argument("--warmup-frames", type=int, default=40)
    parser.add_argument("--settle-frames", type=int, default=2)
    parser.add_argument("--agreement-cm", type=float, default=2.0,
                        help="both methods hitting: their medians must agree within this")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--min-hits", type=int, default=DEFAULT_THRESHOLDS["min_hits"])
    parser.add_argument("--min-hit-fraction", type=float,
                        default=DEFAULT_THRESHOLDS["min_hit_fraction"])
    parser.add_argument("--min-within-fraction", type=float,
                        default=DEFAULT_THRESHOLDS["min_within_fraction"])
    parser.add_argument("--rpc-port", type=int, default=39581)
    parser.add_argument("--graphics-adapter", type=int, default=1)
    return parser.parse_args(argv)


def _summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hits = [row["floor_z_ue_cm"] for row in rows if row.get("hit")]
    summary = summarize_floor_hits(hits, total_traces=len(rows)) if rows else {}
    by_origin = {}
    for origin in sorted({row["origin"] for row in rows}):
        sub = [row["floor_z_ue_cm"] for row in rows if row["origin"] == origin and row.get("hit")]
        count = sum(1 for row in rows if row["origin"] == origin)
        by_origin[origin] = summarize_floor_hits(sub, total_traces=count)
    return {"summary": summary, "by_origin": by_origin}


def run(args: argparse.Namespace) -> int:
    _require(not args.output.exists(), f"refusing to overwrite: {args.output}")
    config = json.loads(args.scene_config.read_text(encoding="utf-8"))
    render = config.get("render") or {}
    native_map = str(render.get("native_map") or "")
    _require(native_map.startswith("/Game/"), "scene config render.native_map must be a /Game path")
    configured_ground = float(render.get("ground_z_ue_cm", 0.0))
    start_z = configured_ground + float(args.start_offset_cm)
    end_z = start_z - float(args.trace_length_cm)
    points, point_facts = sample_points(config, walkable_samples=args.walkable_samples,
                                        margin_cm=args.walkable_margin_m * 100.0, seed=args.seed)
    _require(len(points) >= 1, "no points to trace")
    rng = np.random.default_rng(args.seed + 1)
    camera_points = [p for p in points if p[2] == "camera_point"]
    walkable_points = [p for p in points if p[2] == "walkable_cell"]
    if len(camera_points) > int(args.depth_point_limit):
        chosen = rng.choice(len(camera_points), size=int(args.depth_point_limit), replace=False)
        depth_camera_points = [camera_points[i] for i in sorted(chosen)]
    else:
        depth_camera_points = camera_points
    depth_sample_points = depth_camera_points + walkable_points[:int(args.depth_walkable_samples)]

    print(f"floor measurement on {native_map}: {len(points)} trace points, "
          f"{len(depth_sample_points)} depth points, start z={start_z:.1f}", flush=True)
    started = time.time()
    instance = _launch(args, native_map)
    launch_seconds = time.time() - started
    measurements: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    try:
        game = instance.get_game()
        if args.method in ("line_trace", "both"):
            rows, facts = trace_points(instance, game, points, start_z, end_z)
            all_rows.extend(rows)
            measurements["line_trace"] = dict(facts, **_summaries(rows))
        if args.method in ("depth_capture", "both"):
            rows, facts = depth_points(args, instance, game, depth_sample_points, start_z)
            all_rows.extend(rows)
            measurements["depth_capture"] = dict(facts, **_summaries(rows))
    finally:
        try:
            instance.close(force=True)
        except Exception:
            pass
    engine_seconds = time.time() - started

    # 主量法:线追踪有命中就用线追踪(精确到面),否则用深度;两者都命中时必须一致。
    min_hits = int(args.min_hits)
    primary = None
    for name in ("line_trace", "depth_capture"):
        if name in measurements and measurements[name]["summary"].get("hit_count", 0) >= min_hits:
            primary = name
            break
    agreement: dict[str, Any] = {"checked": False}
    consistent_methods = True
    medians = {name: m["summary"].get("median_cm") for name, m in measurements.items()
               if m["summary"].get("hit_count", 0) >= min_hits}
    if len(medians) == 2:
        diff = abs(medians["line_trace"] - medians["depth_capture"])
        consistent_methods = diff <= float(args.agreement_cm)
        agreement = {"checked": True, "median_difference_cm": diff,
                     "tolerance_cm": float(args.agreement_cm), "agree": consistent_methods}
    if primary is None:
        summary = summarize_floor_hits([], total_traces=max(1, len(all_rows)))
        primary_facts = {"kind": "none_of_" + "_".join(measurements)}
    else:
        summary = dict(measurements[primary]["summary"])
        primary_facts = {k: v for k, v in measurements[primary].items()
                         if k not in ("summary", "by_origin", "raw_examples")}
    if not consistent_methods:
        # 两种量法不一致:不能宣布量到了地板;把 within_fraction 置零让状态成为 inconsistent。
        summary = dict(summary, within_fraction=0.0, methods_disagree=True)

    components = collections.Counter(str(row.get("component", "?")) for row in all_rows
                                    if row.get("hit") and row.get("method") == "line_trace")
    median = summary.get("median_cm")
    outliers = []
    if median is not None:
        outliers = [{"method": row["method"], "index": row["index"], "origin": row["origin"],
                     "xy_ue_cm": row["xy_ue_cm"], "floor_z_ue_cm": row["floor_z_ue_cm"],
                     "component": row.get("component")}
                    for row in all_rows if row.get("hit") and abs(row["floor_z_ue_cm"] - median) > 5.0]
    # 家具顶面让命中点偏高,不会偏低;比中位数低 5 cm 以上的命中才可能是另一层地面。
    above = sum(1 for o in outliers if o["floor_z_ue_cm"] > (median or 0.0))
    below = len(outliers) - above
    misses = collections.Counter(row["method"] for row in all_rows if not row.get("hit"))
    thresholds = {"min_hits": min_hits, "min_hit_fraction": float(args.min_hit_fraction),
                  "min_within_fraction": float(args.min_within_fraction),
                  "method_agreement_cm": float(args.agreement_cm),
                  "status": "placeholder_research_not_calibrated"}
    method = {"kind": (primary_facts.get("kind") if primary else primary_facts["kind"]),
              "primary": primary, "requested": args.method,
              "configured_ground_z_ue_cm_before_measurement": configured_ground,
              "points": point_facts, "stage_root": str(args.stage_root),
              "spear_executable": str(args.spear_executable), "seed": int(args.seed),
              "launch_seconds": round(launch_seconds, 1),
              "engine_wall_clock_seconds": round(engine_seconds, 1),
              "primary_details": primary_facts}
    extra = {"measurements": measurements, "method_agreement": agreement,
             "hit_components": components.most_common(12),
             "outliers_over_5cm": outliers[:200], "outlier_count": len(outliers),
             "outliers_above_median": above, "outliers_below_median": below,
             "miss_count_by_method": dict(misses),
             "offset_from_configured_cm": ((median - configured_ground)
                                           if median is not None else None),
             "scene_config": str(args.scene_config),
             "scene_config_content": config}
    root = write_floor_reference(args.output, scene_id=str(config["scene_id"]),
                                 native_map=native_map, method=method, summary=summary,
                                 rows=all_rows, thresholds=thresholds,
                                 code=git_worktree_state(), extra=extra)
    index = json.loads((root / "floor_reference.json").read_text(encoding="utf-8"))
    report = {"status": index["status"], "ground_z_ue_cm": index["ground_z_ue_cm"],
              "primary": primary, "agreement": agreement,
              "per_method": {name: {"summary": m["summary"], "by_origin": m["by_origin"]}
                             for name, m in measurements.items()},
              "hit_components": components.most_common(5),
              "outlier_count": len(outliers), "outliers_above_median": above,
              "outliers_below_median": below, "misses": dict(misses)}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if index["status"] != STATUS_MEASURED:
        print("QA_V3_FLOOR_REFERENCE_INCONSISTENT", flush=True)
        return 3
    print("QA_V3_FLOOR_REFERENCE_OK", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
