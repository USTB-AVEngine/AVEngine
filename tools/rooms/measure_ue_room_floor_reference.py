#!/usr/bin/env python3
"""Measure an UE room floor from the currently loaded map.

The primary method is the same Kismet ``LineTraceSingleByProfile`` query used
by the existing QA floor tool. When a map floor has no collision (the current
Kujiale map is one such case), the tool samples the same map through its native
metric depth SceneCapture, looking down from a known UE height. Every result is
written to a fresh room product and keeps per-point readbacks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]

LINE_TRACE_KIND = "ue_line_trace_down_blockall_complex_v1"
DEPTH_FALLBACK_KIND = "depth_readback_fallback"
DEPTH_FALLBACK_PRECISION_M = 0.00025
PLAUSIBLE_ABS_FLOOR_HEIGHT_M = 10.0


def classify_ue_floor_measurement(
    *,
    line_hit_count: int,
    selected_method: str,
    floor_height_m: float,
) -> dict[str, Any]:
    """Label a floor result so a depth fallback cannot be called a line trace."""
    selected = str(selected_method)
    depth_selected = selected in {
        "depth_capture",
        "ue_depth_capture_straight_down_v1",
        "ue_line_trace_then_depth_fallback_v1",
    }
    if int(line_hit_count) <= 0 or depth_selected:
        kind = DEPTH_FALLBACK_KIND
        precision = DEPTH_FALLBACK_PRECISION_M
    else:
        kind = LINE_TRACE_KIND
        precision = None
    status = "measured"
    invalid_reason = None
    if not math.isfinite(float(floor_height_m)) or abs(float(floor_height_m)) > PLAUSIBLE_ABS_FLOOR_HEIGHT_M:
        status = "invalid"
        invalid_reason = (
            "floor_height_m is non-finite or outside the plausible residential range of +/-10 m"
        )
        if int(line_hit_count) <= 0 or depth_selected:
            kind = DEPTH_FALLBACK_KIND
            precision = DEPTH_FALLBACK_PRECISION_M
    return {
        "status": status,
        "measurement_kind": kind,
        "precision_m": precision,
        "invalid_reason": invalid_reason,
    }



def _lookup(value: Any, key: str) -> Any:
    if not isinstance(value, Mapping):
        return None
    if key in value:
        return value[key]
    wanted = key.casefold()
    lowered = {str(name).casefold(): item for name, item in value.items()}
    return lowered.get(wanted)


def _vector(value: Any, *, owner: str) -> list[float]:
    if isinstance(value, Mapping):
        value = [_lookup(value, axis) for axis in ("x", "y", "z")]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise RuntimeError(f"{owner} is not a three-vector: {value!r}")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise RuntimeError(f"{owner} contains non-finite values: {value!r}")
    return result


def _parse_trace(raw: Any, *, point_cm: Sequence[float], index: int) -> dict[str, Any]:
    hit = bool(_lookup(raw, "ReturnValue"))
    row: dict[str, Any] = {
        "index": int(index),
        "method": "ue_line_trace_down_blockall_complex_v1",
        "origin": "grid_or_bounds_point",
        "xy_ue_cm": [float(point_cm[0]), float(point_cm[1])],
        "hit": hit,
    }
    if not hit:
        return row
    out_hit = _lookup(raw, "OutHit")
    location = _vector(_lookup(out_hit, "Location"), owner="trace Location")
    row["hit_point_ue_cm"] = location
    row["floor_z_ue_cm"] = float(location[2])
    normal = _lookup(out_hit, "Normal")
    if normal is not None:
        row["hit_normal_ue"] = _vector(normal, owner="trace Normal")
    component = _lookup(out_hit, "Component")
    if isinstance(component, str) and component:
        row["component"] = component
    phys = _lookup(out_hit, "PhysMaterial")
    if isinstance(phys, str) and phys:
        row["physmaterial"] = phys
    row["horizontal_error_cm"] = max(
        abs(location[0] - float(point_cm[0])), abs(location[1] - float(point_cm[1]))
    )
    return row


def _bounds_points(args: argparse.Namespace) -> list[list[float]]:
    x0, y0, x1, y1 = [float(item) for item in args.bounds_xy_cm]
    if not x0 < x1 or not y0 < y1:
        raise ValueError("bounds_xy_cm must be increasing")
    xs = np.linspace(x0, x1, int(args.grid_x) + 2, dtype=np.float64)[1:-1]
    ys = np.linspace(y0, y1, int(args.grid_y) + 2, dtype=np.float64)[1:-1]
    return [[float(x), float(y), 0.0] for y in ys for x in xs]


def _grid_points(path: Path, *, seed: int, count: int, margin_cm: float) -> list[list[float]]:
    metadata_path = path / "walkable_grid.json" if path.is_dir() else path
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "qa_v3_walkable_grid_v1":
        raise ValueError(f"unsupported walkable grid schema: {metadata.get('schema')!r}")
    arrays_path = metadata_path.parent / str(metadata["arrays"]["path"])
    with np.load(arrays_path, allow_pickle=False) as arrays:
        walkable = np.asarray(arrays["walkable"], dtype=bool)
        clearance = np.asarray(arrays["clearance_cm"], dtype=np.float64)
    eligible = np.argwhere(walkable & (clearance >= float(margin_cm)))
    if len(eligible) == 0:
        raise ValueError("walkable grid has no eligible cells")
    rng = np.random.default_rng(int(seed))
    selected = eligible[rng.choice(len(eligible), size=min(int(count), len(eligible)), replace=False)]
    origin = np.asarray(metadata["origin_xy_cm"], dtype=np.float64)
    cell = float(metadata["cell_cm"])
    # The retained grid declares rows=UE Y and columns=UE X.
    return [[float(origin[0] + col * cell), float(origin[1] + row * cell), 0.0] for row, col in selected]


def _points(args: argparse.Namespace) -> tuple[list[list[float]], dict[str, Any]]:
    if args.grid_json:
        points = _grid_points(args.grid_json, seed=args.seed, count=args.point_count, margin_cm=args.grid_margin_cm)
        return points, {"source": str(args.grid_json), "kind": "walkable_grid", "count": len(points), "margin_cm": float(args.grid_margin_cm)}
    points = _bounds_points(args)
    return points, {"source": "explicit_bounds", "kind": "bounds_grid", "count": len(points)}


def _summary(values: Sequence[float], total: int) -> dict[str, Any]:
    finite = np.asarray([float(item) for item in values if math.isfinite(float(item))], dtype=np.float64)
    if finite.size == 0:
        return {"trace_count": int(total), "hit_count": 0, "hit_fraction": 0.0}
    median = float(np.median(finite))
    deviations = np.abs(finite - median)
    return {
        "trace_count": int(total),
        "hit_count": int(finite.size),
        "hit_fraction": float(finite.size / max(total, 1)),
        "median_cm": median,
        "mean_cm": float(np.mean(finite)),
        "mad_cm": float(np.median(deviations)),
        "min_cm": float(np.min(finite)),
        "p05_cm": float(np.quantile(finite, 0.05)),
        "p95_cm": float(np.quantile(finite, 0.95)),
        "max_cm": float(np.max(finite)),
        "within_2cm_fraction": float(np.mean(deviations <= 2.0)),
    }


def _configure(args: argparse.Namespace, map_path: str) -> Any:
    extension = args.spear_ext_dir.expanduser().resolve()
    if not extension.is_dir():
        raise FileNotFoundError(f"missing SPEAR extension directory: {extension}")
    sys.path.insert(0, str(extension))
    sys.path.insert(0, str(REPOSITORY / "tools/rooms"))
    from run_spear_kujiale_canary import _configure_spear

    config_args = argparse.Namespace(
        rpc_port=int(args.rpc_port),
        graphics_adapter=int(args.graphics_adapter),
        unreal_editor=args.unreal_editor,
        uproject=args.uproject,
        source_stage=args.uproject,
    )
    return _configure_spear(config_args, {"map_path": map_path})


def _depth_measure(instance: Any, game: Any, points: Sequence[Sequence[float]], *, width: int, height: int, camera_z_cm: float) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    from run_spear_residential_episode import _spawn_multimodal_camera

    camera = None
    components = None
    rows: list[dict[str, Any]] = []
    try:
        with instance.begin_frame():
            camera, components = _spawn_multimodal_camera(game, horizontal_fov_deg=30.0, width=width, height=height)
            components["depth"].PrimitiveRenderMode = "PRM_RenderScenePrimitives"
            components["depth"].ShowOnlyActors = []
        with instance.end_frame():
            pass
        instance.step(num_frames=2)
        for index, point in enumerate(points):
            with instance.begin_frame():
                camera.K2_SetActorLocationAndRotation(
                    NewLocation={"X": float(point[0]), "Y": float(point[1]), "Z": float(camera_z_cm)},
                    NewRotation={"Roll": 0.0, "Pitch": -90.0, "Yaw": 0.0},
                    bSweep=False,
                    bTeleport=True,
                )
            with instance.end_frame():
                pass
            instance.step(num_frames=2)
            with instance.begin_frame():
                pass
            with instance.end_frame():
                depth = np.asarray(components["depth"].read_pixels()["arrays"]["data"][:, :, 0], dtype=np.float64)
            half = 2
            cy, cx = depth.shape[0] // 2, depth.shape[1] // 2
            window = depth[max(0, cy-half):cy+half+1, max(0, cx-half):cx+half+1]
            valid = window[np.isfinite(window) & (window > 0.0) & (window < 65504.0)]
            row = {"index": int(index), "method": "ue_depth_capture_straight_down_v1", "origin": "grid_or_bounds_point", "xy_ue_cm": [float(point[0]), float(point[1])], "camera_z_ue_cm": float(camera_z_cm), "hit": bool(valid.size > 0), "valid_center_pixels": int(valid.size)}
            if valid.size:
                depth_m = float(np.median(valid))
                row["center_depth_m"] = depth_m
                row["floor_z_ue_cm"] = float(camera_z_cm - depth_m * 100.0)
            rows.append(row)
        values = [row["floor_z_ue_cm"] for row in rows if row.get("hit")]
        return rows, (camera, components), {"kind": "ue_depth_capture_straight_down_v1", "camera_z_ue_cm": float(camera_z_cm), "render_hw": [height, width], "center_window_px": 5, "points": len(points), "hit_count": len(values)}
    except Exception:
        raise


def measure(args: argparse.Namespace) -> dict[str, Any]:
    points, point_source = _points(args)
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace floor measurement: {output}")
    output.mkdir(parents=True)
    instance = None
    camera = None
    components = None
    game = None
    trace_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    try:
        instance = _configure(args, args.map_path)
        game = instance.get_game()
        instance.step(num_frames=int(args.warmup_frames))
        with instance.begin_frame():
            level = str(game.get_unreal_object(uclass="UGameplayStatics").GetCurrentLevelName(bRemovePrefixString=True))
            kismet = game.get_unreal_object(uclass="UKismetSystemLibrary")
        with instance.end_frame():
            pass
        expected_level = str(args.map_path).rsplit("/", 1)[-1]
        if level != expected_level:
            raise RuntimeError(f"loaded map level differs: {level!r} != {expected_level!r}")
        with instance.begin_frame():
            for index, point in enumerate(points):
                raw = kismet.LineTraceSingleByProfile(
                    Start={"X": float(point[0]), "Y": float(point[1]), "Z": float(args.trace_start_z_cm)},
                    End={"X": float(point[0]), "Y": float(point[1]), "Z": float(args.trace_end_z_cm)},
                    ProfileName="BlockAll",
                    bTraceComplex=True,
                    ActorsToIgnore=[],
                    DrawDebugType="None",
                    bIgnoreSelf=True,
                    TraceColor={"R": 1.0, "G": 0.0, "B": 0.0, "A": 1.0},
                    TraceHitColor={"R": 0.0, "G": 1.0, "B": 0.0, "A": 1.0},
                    DrawTime=0.0,
                    as_dict=True,
                )
                trace_rows.append(_parse_trace(raw, point_cm=point, index=index))
        with instance.end_frame():
            pass
        line_values = [row["floor_z_ue_cm"] for row in trace_rows if row.get("hit") and "floor_z_ue_cm" in row]
        method: dict[str, Any] = {"kind": "ue_line_trace_down_blockall_complex_v1", "profile_name": "BlockAll", "trace_complex": True, "start_z_ue_cm": float(args.trace_start_z_cm), "end_z_ue_cm": float(args.trace_end_z_cm), "points": len(points), "hit_count": len(line_values)}
        line_summary = _summary(line_values, len(points))
        selected_values = line_values
        selected_rows = trace_rows
        # A native Apartment trace can hit the floor even when a few points
        # hit a raised prop.  Keep the line trace as authority whenever it has
        # enough samples; its median and spread remain visible in the receipt.
        if len(line_values) < int(args.min_line_hits):
            depth_rows, (camera, components), depth_method = _depth_measure(instance, game, points, width=int(args.depth_width), height=int(args.depth_height), camera_z_cm=float(args.depth_camera_z_cm))
            depth_values = [row["floor_z_ue_cm"] for row in depth_rows if row.get("hit") and "floor_z_ue_cm" in row]
            if not depth_values:
                raise RuntimeError("line traces did not provide a floor and depth fallback had no valid pixels")
            depth_summary = _summary(depth_values, len(points))
            method = {"kind": "ue_line_trace_then_depth_fallback_v1", "line_trace": method, "line_summary": line_summary, "depth_capture": depth_method, "depth_summary": depth_summary}
            selected_values = depth_values
            selected_rows = depth_rows
        summary = _summary(selected_values, len(selected_rows))
        if summary.get("hit_count", 0) == 0:
            raise RuntimeError("no valid floor measurements")
        floor_cm = float(summary["median_cm"])
        selected_method = "line_trace" if selected_rows is trace_rows else "depth_capture"
        labeled = classify_ue_floor_measurement(
            line_hit_count=len(line_values),
            selected_method=selected_method,
            floor_height_m=floor_cm / 100.0,
        )
        result = {
            "schema": "avengine_qa_ue_floor_reference_v1",
            "status": labeled["status"],
            "measurement_kind": labeled["measurement_kind"],
            "precision_m": labeled["precision_m"],
            "room_id": str(args.room_id),
            "native_map": str(args.map_path),
            "level_readback": level,
            "floor_height_m": floor_cm / 100.0,
            "floor_z_ue_cm": floor_cm,
            "method": method,
            "point_source": point_source,
            "summary": summary,
            "rows": {"path": "floor_trace_rows.json", "count": len(selected_rows)},
            "raw_measurements": {"line_trace_rows": len(trace_rows), "line_trace_hits": len(line_values), "depth_rows": len(depth_rows), "selected_method": ("line_trace" if selected_rows is trace_rows else "depth_capture")},
            "claim_boundary": "Measured floor readback from the currently loaded UE map; outliers remain in rows and this is not a question-admission claim",
        }
        if labeled["invalid_reason"]:
            result["invalid_reason"] = labeled["invalid_reason"]
            result["claim_boundary"] = labeled["invalid_reason"]
        _write_fresh(output / "floor_trace_rows.json", {"schema": result["schema"], "room_id": result["room_id"], "native_map": result["native_map"], "selected_method": result["raw_measurements"]["selected_method"], "line_trace_rows": trace_rows, "depth_rows": depth_rows, "rows": selected_rows})
        _write_fresh(output / "floor_reference.json", result)
        return result
    finally:
        if instance is not None:
            try:
                if components is not None:
                    from avengine.backends.spear_ue.research_runtime import close_scene_capture
                    for component in components.values():
                        close_scene_capture(instance=instance, game=game, camera=None, capture=component)
                    close_scene_capture(instance=instance, game=game, camera=camera, capture=None)
            finally:
                instance.close(force=True)


def _write_fresh(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace {path}")
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + chr(10), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-id", required=True)
    parser.add_argument("--map-path", required=True)
    parser.add_argument("--uproject", type=Path, required=True)
    parser.add_argument("--unreal-editor", type=Path, required=True)
    parser.add_argument("--spear-ext-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rpc-port", type=int, default=39424)
    parser.add_argument("--graphics-adapter", type=int, default=2)
    parser.add_argument("--warmup-frames", type=int, default=4)
    parser.add_argument("--trace-start-z-cm", type=float, default=300.0)
    parser.add_argument("--trace-end-z-cm", type=float, default=-100.0)
    parser.add_argument("--min-line-hits", type=int, default=8)
    parser.add_argument("--min-within-fraction", type=float, default=0.8)
    parser.add_argument("--grid-json", type=Path)
    parser.add_argument("--point-count", type=int, default=48)
    parser.add_argument("--grid-margin-cm", type=float, default=20.0)
    parser.add_argument("--bounds-xy-cm", type=float, nargs=4, default=[-500.0, -500.0, 500.0, 500.0])
    parser.add_argument("--grid-x", type=int, default=8)
    parser.add_argument("--grid-y", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--depth-width", type=int, default=64)
    parser.add_argument("--depth-height", type=int, default=64)
    parser.add_argument("--depth-camera-z-cm", type=float, default=30.0)
    args = parser.parse_args()
    result = measure(args)
    print(json.dumps({"status": result["status"], "room_id": result["room_id"], "native_map": result["native_map"], "floor_height_m": result["floor_height_m"], "measurement_kind": result.get("measurement_kind"), "precision_m": result.get("precision_m"), "method": result["method"]["kind"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
