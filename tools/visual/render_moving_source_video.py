"""Render the route the acoustic pass rendered, as frames that can carry that audio.

The point is not a pretty clip. A moving source is only demonstrated when the
picture and the sound come from one description of one point: the same bank
file, the same episode, the same listener, and the same emitter. So this takes
the listener from the acoustic report rather than choosing its own, places a
published loudspeaker so that its own emitter anchor lands exactly where the
acoustic pass put the source, and steps it along the identical frames.

Two things make that exact rather than approximate:

  * the asset's emitter is an offset inside the asset, not its origin - a
    driver array 0.65 m up a 1 m tower. The acoustic pass is told that height
    with --emitter-height-m and the mesh is placed on the floor, so both halves
    describe the same point in the room without anything floating.
  * the speaker is turned to face the listener each frame, which is what makes
    the front baffle the radiating surface, matching a point source at the
    emitter.

The camera stays put at the listener and does not track. A tracking camera
would keep the source centred and hide the very thing worth seeing, which is
the source crossing the frame while the sound crosses the stereo field, and
walking behind something while the direction error jumps.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.episode_clock import (  # noqa: E402
    EpisodeClock,
    EpisodeClockError,
    LEGACY_FRAME_RATE_HZ,
    LEGACY_SAMPLE_RATE_HZ,
)


def _clock_compatibility(base: EpisodeClock, args: argparse.Namespace) -> str:
    values = (
        args.frame_count,
        args.frame_rate_hz,
        args.sample_rate,
        args.clip_seconds,
    )
    return (
        base.compatibility
        if not any(value is not None for value in values)
        else "configured"
    )

def look_at(direction, np_quaternion):
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    yaw = math.atan2(-d[0], -d[2])
    pitch = math.asin(float(np.clip(d[1], -1.0, 1.0)))
    qy = np_quaternion(math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0)
    qx = np_quaternion(math.cos(pitch / 2), math.sin(pitch / 2), 0.0, 0.0)
    return qy * qx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-prefix", required=True)
    parser.add_argument("--magnum-site", required=True)
    parser.add_argument("--rlr-sdk-root", required=True)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--acoustic-report", required=True, type=Path)
    parser.add_argument(
        "--listener-pose",
        type=Path,
        help=(
            "pose file from tools/scene/choose_listener_pose.py. Uses the entry "
            "the acoustic pass accepted, so the picture and the sound share one "
            "decision instead of this tool choosing an aim of its own"
        ),
    )
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--frame-count", type=int)
    parser.add_argument("--frame-rate-hz", type=float)
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--clip-seconds", type=float)
    parser.add_argument(
        "--hfov-deg",
        type=float,
        help="fixed field of view; omit to fit it to the route's angular span",
    )
    parser.add_argument("--margin-deg", type=float, default=12.0)
    parser.add_argument(
        "--aim-open",
        action="store_true",
        help=(
            "point the camera at the most open direction rather than at the "
            "source. Aiming at a source sample is what the first pass did, and "
            "on an orbit that sample is often inside a wall, so the whole clip "
            "faced plaster from 30 cm away. Chosen by depth: candidate yaws are "
            "rendered and the one with the greatest median depth wins"
        ),
    )
    parser.add_argument(
        "--aim-frame",
        type=int,
        help=(
            "aim the fixed camera at the source's position in this frame. For a "
            "full orbit the mean direction is degenerate - a circle averages to "
            "its centre - so the aim has to be named rather than fitted"
        ),
    )
    parser.add_argument(
        "--overhead-m",
        type=float,
        help=(
            "look down from this height above the listener instead of out from "
            "it. A fixed forward camera cannot show an orbit: the source spans "
            "the full 360 and only a third of that fits in any field of view, "
            "so two thirds of the clip has nothing on screen. Hearing it behind "
            "you is correct; seeing nothing is not informative"
        ),
    )
    parser.add_argument(
        "--place-at-emitter",
        action="store_true",
        help=(
            "put the mesh where its own emitter anchor lands on the acoustic "
            "source point, instead of standing it on the floor. Needed whenever "
            "the source is not at floor level - an orbit at ear height, for "
            "instance - because otherwise the mesh and the sound are 1.5 m "
            "apart and the picture stops meaning anything"
        ),
    )
    args = parser.parse_args()

    report = json.loads(args.acoustic_report.read_text(encoding="utf-8"))
    bank = json.loads(args.bank.read_text(encoding="utf-8"))
    episode = next(
        e for e in bank["episodes"] if e["episode_id"] == report["episode_id"]
    )
    route_frame_count = len(episode["source_center_paths_m"][report["slot"]])
    raw_clock = report.get("clock") or bank.get("clock")
    try:
        if raw_clock is not None:
            base_clock = EpisodeClock.from_mapping(raw_clock)
        else:
            base_clock = EpisodeClock.from_values(
                frame_count=report.get("frame_count", route_frame_count),
                frame_rate_hz=report.get(
                    "frame_rate_hz",
                    bank.get("frame_rate_hz", LEGACY_FRAME_RATE_HZ),
                ),
                sample_rate_hz=(
                    report.get("sample_rate_hz", LEGACY_SAMPLE_RATE_HZ)
                ),
                compatibility="legacy_inferred",
            )
        clock = EpisodeClock.from_values(
            frame_count=(
                args.frame_count
                if args.frame_count is not None
                else base_clock.frame_count
            ),
            frame_rate_hz=(
                args.frame_rate_hz
                if args.frame_rate_hz is not None
                else base_clock.frame_rate_hz
            ),
            sample_rate_hz=(
                args.sample_rate
                if args.sample_rate is not None
                else base_clock.sample_rate_hz
            ),
            clip_seconds=args.clip_seconds,
            compatibility=_clock_compatibility(base_clock, args),
        )
    except EpisodeClockError as error:
        raise SystemExit(f"invalid episode clock: {error}") from error
    if clock.frame_count != route_frame_count:
        raise SystemExit(
            "video clock frame_count differs from the bank route: "
            f"clock={clock.frame_count}, route={route_frame_count}"
        )
    asset = json.loads((args.asset_dir / "asset.json").read_text(encoding="utf-8"))
    offset = np.asarray(asset["emitter"]["offset_m"], dtype=float)

    scene = bank["scene"]
    if Path(scene).name.endswith(".basis.glb"):
        raise SystemExit("refusing a basis glb: no BasisImporter here, it segfaults")

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
    hs, mn = runtime.habitat_sim, runtime.magnum
    np_quaternion = runtime.quaternion.quaternion

    floor_path = np.asarray(
        episode["source_center_paths_m"][report["slot"]], dtype=float
    )
    listener = np.asarray(report["listener_m"], dtype=float)
    frames = [f["frame"] for f in report["per_frame"]]
    emitters = floor_path.copy()
    emitters[:, 1] += float(report["source_center_height_m"])
    directions = emitters[frames] - listener
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    if args.aim_frame is not None:
        aim = emitters[args.aim_frame] - listener
    else:
        aim = directions.sum(axis=0)
    norm = float(np.linalg.norm(aim))
    if norm < 1.0e-6:
        # A full orbit sums to nothing. Fall back to the first sample rather
        # than dividing by zero and aiming at nowhere.
        aim = emitters[frames[0]] - listener
        norm = float(np.linalg.norm(aim))
    aim = aim / norm
    spread = float(
        np.degrees(np.arccos(np.clip(directions @ aim, -1.0, 1.0))).max()
    )
    if args.overhead_m:
        # Cover the orbit radius plus margin from the chosen height.
        reach = float(np.linalg.norm(emitters[frames][:, (0, 2)] - listener[[0, 2]],
                                    axis=1).max())
        hfov = (
            float(args.hfov_deg)
            if args.hfov_deg
            else float(
                np.clip(
                    2.0 * math.degrees(math.atan2(reach * 1.25, args.overhead_m)),
                    50.0,
                    130.0,
                )
            )
        )
        # A 16:9 frame covers less vertically than horizontally, so a circular
        # orbit leaves the top and bottom of the picture while still inside the
        # horizontal field. Overhead framing wants a square.
        if args.width != args.height:
            print(
                f"overhead view: squaring the frame to {args.height}x{args.height} "
                "so the orbit does not leave the top and bottom"
            )
            args.width = args.height
        print(f"overhead {args.overhead_m:.2f} m, reach {reach:.2f} m, hfov {hfov:.1f}")
    else:
        hfov = (
            float(args.hfov_deg)
            if args.hfov_deg
            else float(np.clip(2.0 * (spread + args.margin_deg), 50.0, 120.0))
        )
    print(f"route spans {2 * spread:.1f} deg from the listener; hfov {hfov:.1f} deg")

    backend = hs.SimulatorConfiguration()
    backend.scene_id = scene
    backend.load_semantic_mesh = False
    backend.enable_physics = True
    if runtime.physics_config_path:
        backend.physics_config_file = str(runtime.physics_config_path)
    colour = hs.CameraSensorSpec()
    colour.uuid = "colour"
    colour.sensor_type = hs.SensorType.COLOR
    colour.resolution = [args.height, args.width]
    colour.hfov = hfov
    colour.position = [0.0, 0.0, 0.0]
    depth = hs.CameraSensorSpec()
    depth.uuid = "depth"
    depth.sensor_type = hs.SensorType.DEPTH
    depth.resolution = [args.height, args.width]
    depth.hfov = hfov
    depth.position = [0.0, 0.0, 0.0]
    agent_cfg = hs.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [colour, depth]
    sim = hs.Simulator(hs.Configuration(backend, [agent_cfg]))

    templates = sim.get_object_template_manager()
    objects = sim.get_rigid_object_manager()
    glb = args.asset_dir / "finalized.glb"
    template = templates.create_new_template(args.asset_dir.name)
    template.render_asset_handle = str(glb)
    template.collision_asset_handle = str(glb)
    template.is_collidable = False
    obj = objects.add_object_by_template_id(
        templates.register_template(template, args.asset_dir.name)
    )
    obj.motion_type = hs.physics.MotionType.KINEMATIC

    errors = {f["frame"]: f["error_deg"] for f in report["per_frame"]}
    ranges = {f["frame"]: f["range_m"] for f in report["per_frame"]}

    # Aim and widen so the whole route is in shot. Pointing at the midpoint and
    # taking a default field of view is what a first attempt does, and it left
    # the source outside the frame for the first two thirds of the clip while a
    # third of the picture was blank wall. The route's own angular span from the
    # listener is the thing that should set both.
    agent = sim.get_agent(0)
    def rotation_matrix(quat):
        w, x, y, z = quat.w, quat.x, quat.y, quat.z
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    def median_depth(direction):
        probe = agent.get_state()
        probe.position = listener.astype(np.float32)
        probe.rotation = look_at(direction, np_quaternion)
        probe.sensor_states = {}
        agent.set_state(probe, True)
        d = np.asarray(sim.get_sensor_observations()["depth"], dtype=float)
        d = d[np.isfinite(d) & (d > 0)]
        return float(np.median(d)) if d.size else 0.0

    pose = (
        json.loads(args.listener_pose.read_text(encoding="utf-8"))
        if args.listener_pose
        else None
    )
    if pose is not None and not args.overhead_m:
        index = pose.get("accepted_index")
        if index is None:
            raise SystemExit(
                "the pose file has no accepted_index; run the acoustic pass "
                "first so the picture follows the listener the sound accepted"
            )
        aim = np.asarray(pose["candidates"][index]["aim_world"], dtype=float)
        aim = aim / np.linalg.norm(aim)
        print(f"aim from pose candidate {index}: {np.round(aim, 3).tolist()}")
    elif args.aim_open and not args.overhead_m:
        best = None
        for degrees in range(0, 360, 30):
            radians = math.radians(degrees)
            candidate = np.array([math.sin(radians), 0.0, -math.cos(radians)])
            score = median_depth(candidate)
            if best is None or score > best[0]:
                best = (score, candidate, degrees)
        aim = best[1]
        print(
            f"aiming at the most open direction: yaw {best[2]} deg, "
            f"median depth {best[0]:.2f} m"
        )

    state = agent.get_state()
    if args.overhead_m:
        eye = listener.copy()
        eye[1] += float(args.overhead_m)
        state.position = eye.astype(np.float32)
        state.rotation = look_at(np.array([1e-6, -1.0, 0.0]), np_quaternion)
    else:
        state.position = listener.astype(np.float32)
        state.rotation = look_at(aim, np_quaternion)
    state.sensor_states = {}
    agent.set_state(state, True)

    height = float(report["source_center_height_m"])
    basis = rotation_matrix(state.rotation)
    half_h = hfov / 2.0
    half_v = math.degrees(
        math.atan(math.tan(math.radians(half_h)) * args.height / args.width)
    )

    def framing(emitter):
        """Where the source sits relative to the frame, in degrees.

        Reported rather than left implicit: a clip where the source is off
        screen is a case we want, and without saying so it reads as a bug.
        """

        local = basis.T @ (emitter - listener)
        forward = -local[2]
        if forward <= 1.0e-6:
            return "behind the camera", 180.0
        azimuth = math.degrees(math.atan2(local[0], forward))
        elevation = math.degrees(math.atan2(local[1], math.hypot(local[0], forward)))
        if abs(azimuth) <= half_h and abs(elevation) <= half_v:
            return "in frame", azimuth
        side = "right" if azimuth > 0 else "left"
        if abs(elevation) > half_v and abs(azimuth) <= half_h:
            return ("above" if elevation > 0 else "below") + " the frame", azimuth
        return f"{abs(azimuth) - half_h:.0f} deg off {side}", azimuth

    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(
            f"output directory already exists (fresh/no-clobber): {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True)
    from PIL import Image, ImageDraw

    for index, frame in enumerate(frames):
        ground = floor_path[frame]
        # Face the listener, so the front baffle is what radiates.
        towards = listener - ground
        yaw = math.degrees(math.atan2(-towards[2], towards[0]))
        if args.place_at_emitter:
            emitter = ground.copy()
            emitter[1] += height
            angle = math.radians(yaw)
            rotated = np.array(
                [
                    offset[0] * math.cos(angle) + offset[2] * math.sin(angle),
                    offset[1],
                    -offset[0] * math.sin(angle) + offset[2] * math.cos(angle),
                ]
            )
            placed = emitter - rotated
        else:
            placed = ground
        obj.translation = mn.Vector3(*[float(v) for v in placed])
        obj.rotation = mn.Quaternion.rotation(mn.Deg(yaw), mn.Vector3.y_axis())

        rgb = np.asarray(sim.get_sensor_observations()["colour"])[..., :3]
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        if args.overhead_m:
            # The listener is directly under the camera, so it is the centre of
            # the frame. Marking it is what makes an overhead orbit readable.
            cx, cy = args.width / 2.0, args.height / 2.0
            draw.line([cx - 14, cy, cx + 14, cy], fill=(255, 225, 90), width=2)
            draw.line([cx, cy - 14, cx, cy + 14], fill=(255, 225, 90), width=2)
            draw.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], outline=(255, 225, 90),
                         width=2)
            draw.text((cx + 18, cy - 7), "listener", fill=(255, 225, 90))
        error = errors.get(frame)
        verdict = "occluded" if (error or 0) > 5 else "verified"
        colour_rgb = (255, 90, 90) if verdict == "occluded" else (150, 255, 150)
        emitter_world = floor_path[frame].copy()
        emitter_world[1] += height
        where, azimuth = framing(emitter_world)
        draw.rectangle([0, 0, 396, 70], fill=(0, 0, 0))
        draw.text((8, 4), f"frame {frame:>3}   range {ranges.get(frame, 0):.2f} m",
                  fill=(235, 235, 235))
        draw.text((8, 20),
                  f"DoA error {('n/a' if error is None else f'{error:5.1f}')} deg  "
                  f"{verdict}", fill=colour_rgb)
        draw.text((8, 36),
                  f"source {where}   azimuth {azimuth:+.0f} deg",
                  fill=(150, 220, 255) if where == "in frame" else (255, 200, 120))
        draw.text((8, 52),
                  f"T20 {report['reverberation'].get('t20_ms_median')} ms  "
                  f"materials {report.get('acoustic_materials')}",
                  fill=(190, 190, 190))
        image.save(args.output_dir / f"frame_{index:04d}.png")

    manifest = {
        "schema": "avengine_moving_source_video_v1",
        "scene": scene,
        "episode_id": report["episode_id"],
        "listener_m": report["listener_m"],
        "asset": str(args.asset_dir),
        "emitter_offset_m": [round(float(v), 4) for v in offset],
        "emitter_height_used_m": report["source_center_height_m"],
        "frames": len(frames),
        "frame_count": clock.frame_count,
        "rendered_frame_count": len(frames),
        "frame_rate_hz": clock.frame_rate_float / max(report["frame_stride"], 1),
        "sample_rate_hz": clock.sample_rate_hz,
        "clip_seconds": clock.clip_seconds_float,
        "sample_count": clock.sample_count,
        "clock": clock.to_dict(),
        "clock_compatibility": clock.compatibility,
        "camera_aim_world": [round(float(x), 6) for x in aim],
        "camera_hfov_deg": round(hfov, 3),
        "camera_aim_note": (
            "pass this vector, not an angle, to any binaural render of the same "
            "clip. Yaw means different things in different places here and the "
            "vector does not"
        ),
        "binding": (
            "the loudspeaker's own emitter anchor sits at the point the "
            "acoustic pass rendered the source from; the camera does not track"
        ),
    }
    (args.output_dir / "video_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(frames)} frames to {args.output_dir}")
    print(f"frame rate for encoding: {manifest['frame_rate_hz']:.4f} fps")
    sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
