#!/usr/bin/env python3
"""Render a 360-degree room tour of one HM3D room as a short mp4 clip.

The camera is placed at the room centre, 1.5 m above the floor, and rotates a
full circle in place (one frame every 10 degrees by default), always looking
outward at a fixed downward pitch. The result is the room seen from inside:
walls, furniture and openings pass through the frame once per circle.

Coordinates come from the caller (the batch driver reads rooms.json and the
studio API), never parsed from the 3D file: the raw file and the engine use
different conventions, so deriving them yourself is wrong. The camera aim is a
direction vector (target minus eye, normalised) - the repo has two historical
definitions of "angle" that differ by 60 degrees, a vector has no ambiguity.

Refuses a <id>.basis.glb: that is the compressed-texture variant and loading it
segfaults the program (no BasisImporter in this runtime).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# 工具脚本直接运行（不 pip 安装），把仓库 src 目录放进导入路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from PIL import Image


def look_at(direction, np_quaternion):
    """Build a rotation that points +Z along `direction`.

    Copied verbatim from tools/visual/render_moving_source_video.py; the
    direction vector convention is the one the repo settled on.
    """
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    yaw = math.atan2(-d[0], -d[2])
    pitch = math.asin(float(np.clip(d[1], -1.0, 1.0)))
    qy = np_quaternion(math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0)
    qx = np_quaternion(math.cos(pitch / 2), math.sin(pitch / 2), 0.0, 0.0)
    return qy * qx


def parse_center(value: str) -> np.ndarray:
    parts = [float(v) for v in value.split(",")]
    if len(parts) != 2:
        raise SystemExit(f"--center-xz expects X,Z, got {value!r}")
    return np.asarray([parts[0], 0.0, parts[1]], dtype=float)


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(v) for v in value.split(",")]
    if len(parts) != 4:
        raise SystemExit(f"--bbox-xz expects MIN_X,MIN_Z,MAX_X,MAX_Z, got {value!r}")
    min_x, min_z, max_x, max_z = parts
    if min_x > max_x or min_z > max_z:
        raise SystemExit(f"invalid --bbox-xz bounds: {value!r}")
    return min_x, min_z, max_x, max_z


def choose_camera_floor_point(
    pathfinder,
    centre: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    probe_height: float = 0.5,
    grid_step: float = 0.25,
    max_vertical_error: float = 0.75,
    max_snap_distance: float = 0.40,
) -> tuple[np.ndarray | None, str, float | None]:
    """Choose the nearest navigable point inside the room's XZ bounds.

    The geometric centre is authoritative when navigable. Otherwise try the
    pathfinder's nearest-point projection, then a deterministic grid ordered by
    distance from the centre. A projected point is accepted only when it stays
    inside the room bbox, near the room floor, and near the sampled candidate;
    this prevents snapping through a wall into an adjacent room.
    """
    min_x, min_z, max_x, max_z = bbox
    eps = 1e-5

    def inside(point: np.ndarray) -> bool:
        return (
            min_x - eps <= float(point[0]) <= max_x + eps
            and min_z - eps <= float(point[2]) <= max_z + eps
            and abs(float(point[1]) - float(centre[1])) <= max_vertical_error
        )

    probe = centre.copy()
    probe[1] += probe_height
    if pathfinder.is_navigable(np.asarray(probe, dtype=float)):
        return centre.copy(), "geometric_centre", 0.0

    # Fast path: the engine's nearest navigable projection of the centre.
    snapped = np.asarray(pathfinder.snap_point(np.asarray(probe, dtype=float)), dtype=float)
    if np.all(np.isfinite(snapped)) and inside(snapped):
        floor_point = snapped.copy()
        offset = float(np.linalg.norm(floor_point[[0, 2]] - centre[[0, 2]]))
        return floor_point, "snap_from_centre", offset

    # Deterministic fallback: sample the bbox and consider points nearest to
    # centre first. snap_point bridges small navmesh quantisation gaps only.
    xs = np.arange(min_x, max_x + grid_step * 0.5, grid_step)
    zs = np.arange(min_z, max_z + grid_step * 0.5, grid_step)
    candidates = [(float(x), float(z)) for x in xs for z in zs]
    candidates.sort(key=lambda p: ((p[0] - centre[0]) ** 2 + (p[1] - centre[2]) ** 2, p))
    for x, z in candidates:
        candidate = np.asarray([x, centre[1] + probe_height, z], dtype=float)
        if pathfinder.is_navigable(candidate):
            floor_point = candidate.copy()
            floor_point[1] = centre[1]
            offset = float(np.linalg.norm(floor_point[[0, 2]] - centre[[0, 2]]))
            return floor_point, "bbox_grid", offset
        snapped = np.asarray(pathfinder.snap_point(candidate), dtype=float)
        if not np.all(np.isfinite(snapped)) or not inside(snapped):
            continue
        if float(np.linalg.norm(snapped - candidate)) > max_snap_distance:
            continue
        offset = float(np.linalg.norm(snapped[[0, 2]] - centre[[0, 2]]))
        return snapped, "bbox_grid_snap", offset

    return None, "not_found", None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", required=True, help="absolute path to <id>.glb")
    parser.add_argument(
        "--center-xz",
        required=True,
        help="room centre as X,Z in engine metres (bbox_xz_m midpoint)",
    )
    parser.add_argument("--floor-y", required=True, type=float)
    parser.add_argument(
        "--bbox-xz",
        required=True,
        help="room bounds as MIN_X,MIN_Z,MAX_X,MAX_Z from rooms.json",
    )
    parser.add_argument("--label", required=True, help="room label, e.g. R3")
    parser.add_argument("--house", required=True, help="house id, e.g. hm3d_val_...")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/avengine_external/studio/room_curation_media"),
        help="media root; clip lands at <output-dir>/<house>/<label>.mp4",
    )
    parser.add_argument("--frames", type=int, default=36, help="frames per circle")
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument(
        "--cam-height",
        type=float,
        default=1.5,
        help="camera height above the room floor, in metres",
    )
    parser.add_argument("--pitch-deg", type=float, default=-8.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--frame-rate", type=int, default=12)
    parser.add_argument("--runtime-prefix", required=True)
    parser.add_argument("--magnum-site", required=True)
    parser.add_argument("--rlr-sdk-root", required=True)
    args = parser.parse_args()

    scene = str(args.glb)
    if Path(scene).name.endswith(".basis.glb"):
        raise SystemExit(
            f"refusing {Path(scene).name}: basis glb has no BasisImporter here "
            "and segfaults"
        )

    # 和 render_moving_source_video.py 一样的运行时加载方式。
    dataset_root = None
    for parent in Path(scene).resolve().parents:
        if (parent / "scene_datasets").is_dir():
            dataset_root = parent
            break
    from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime

    runtime = prepare_installed_habitat_runtime(
        runtime_prefix=args.runtime_prefix,
        magnum_python_site=args.magnum_site,
        rlr_sdk_root=args.rlr_sdk_root,
        mp3d_root=str(dataset_root),
        allow_mp3d_environment=False,
    )
    hs = runtime.habitat_sim
    np_quaternion = runtime.quaternion.quaternion

    backend = hs.SimulatorConfiguration()
    backend.scene_id = scene
    backend.load_semantic_mesh = False
    backend.enable_physics = False  # 纯视觉巡房不需要物理。
    colour = hs.CameraSensorSpec()
    colour.uuid = "colour"
    colour.sensor_type = hs.SensorType.COLOR
    colour.resolution = [args.height, args.width]  # [H, W]，与模板一致
    colour.hfov = args.hfov
    colour.position = [0.0, 0.0, 0.0]
    agent_cfg = hs.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [colour]
    sim = hs.Simulator(hs.Configuration(backend, [agent_cfg]))

    # HM3D 数据里导航网格只有 <id>.basis.navmesh（压缩变体），habitat 不会
    # 自动加载它（自动找 <id>.navmesh），导致 pathfinder 不可用甚至崩溃。
    # 手动加载兜底：先试普通名，再试 .basis 变体。
    if not sim.pathfinder.is_loaded:
        for navmesh in (
            Path(scene).with_suffix(".navmesh"),
            Path(scene).with_suffix(".basis.navmesh"),
        ):
            if navmesh.is_file():
                loaded = sim.pathfinder.load_nav_mesh(str(navmesh))
                print(f"navmesh {navmesh.name} loaded={loaded}")
                break

    centre = parse_center(args.center_xz)
    centre[1] = float(args.floor_y)

    bbox = parse_bbox(args.bbox_xz)
    camera_floor, camera_strategy, centre_offset = choose_camera_floor_point(
        sim.pathfinder, centre, bbox
    )
    if camera_floor is None:
        # 约定退出码 3 = 房间边界内找不到可走点（批处理记为 skip）。
        print(
            f"no navigable camera point inside bbox {bbox} near centre "
            f"{centre[:3].round(2).tolist()}; refusing to render",
            file=sys.stderr,
        )
        raise SystemExit(3)

    agent = sim.get_agent(0)
    eye = camera_floor.copy()
    eye[1] += float(args.cam_height)
    state = agent.get_state()
    state.position = np.asarray(eye, dtype=np.float32)

    output = args.output_dir / args.house
    output.mkdir(parents=True, exist_ok=True)
    pitch = math.radians(args.pitch_deg)
    for index in range(args.frames):
        theta = 2.0 * math.pi * index / args.frames
        # 视线方向：θ 控制绕 Y 轴整圈，pitch 固定向下；方向向量无歧义（坑 #2）。
        direction = np.array(
            [
                -math.sin(theta) * math.cos(pitch),
                math.sin(pitch),
                -math.cos(theta) * math.cos(pitch),
            ]
        )
        state.rotation = look_at(direction, np_quaternion)
        state.sensor_states = {}  # 清残留，否则旧朝向会被记住
        agent.set_state(state, True)
        rgb = np.asarray(sim.get_sensor_observations()["colour"])[..., :3]
        Image.fromarray(rgb).save(output / f"frame_{index:04d}.png")

    clip = output / f"{args.label}.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-framerate", str(args.frame_rate),
            "-i", str(output / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(clip),
        ],
        check=True,
    )
    for frame in output.glob("frame_*.png"):
        frame.unlink()
    metadata = {
        "schema": "avengine_room_tour_camera_v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "house": args.house,
        "room_label": args.label,
        "source_scene": str(Path(scene).resolve()),
        "room_bbox_xz_m": [[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
        "floor_y_m": float(args.floor_y),
        "requested_geometric_centre_xyz_m": [float(v) for v in centre],
        "camera_floor_point_xyz_m": [float(v) for v in camera_floor],
        "camera_eye_xyz_m": [float(v) for v in eye],
        "centre_offset_xz_m": float(centre_offset),
        "placement_strategy": camera_strategy,
        "placement_policy": {
            "centre_first": True,
            "bbox_grid_step_m": 0.25,
            "max_vertical_error_m": 0.75,
            "max_snap_distance_m": 0.40,
        },
        "frames": int(args.frames),
        "frame_rate_hz": int(args.frame_rate),
    }
    (output / f"{args.label}.camera.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"camera strategy={camera_strategy} floor_point="
        f"{camera_floor.round(3).tolist()} centre_offset_xz={centre_offset:.3f}m"
    )
    print(f"wrote {clip} ({args.frames} frames, {args.frame_rate} fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
