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

import numpy as np


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
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument(
        "--hfov-deg",
        type=float,
        help="fixed field of view; omit to fit it to the route's angular span",
    )
    parser.add_argument("--margin-deg", type=float, default=12.0)
    args = parser.parse_args()

    report = json.loads(args.acoustic_report.read_text(encoding="utf-8"))
    bank = json.loads(args.bank.read_text(encoding="utf-8"))
    episode = next(
        e for e in bank["episodes"] if e["episode_id"] == report["episode_id"]
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
    aim = directions.sum(axis=0)
    aim = aim / np.linalg.norm(aim)
    spread = float(
        np.degrees(np.arccos(np.clip(directions @ aim, -1.0, 1.0))).max()
    )
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
    agent_cfg = hs.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [colour]
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
    state = agent.get_state()
    state.position = listener.astype(np.float32)
    state.rotation = look_at(aim, np_quaternion)
    state.sensor_states = {}
    agent.set_state(state, True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageDraw

    for index, frame in enumerate(frames):
        ground = floor_path[frame]
        # Face the listener, so the front baffle is what radiates.
        towards = listener - ground
        yaw = math.degrees(math.atan2(-towards[2], towards[0]))
        obj.translation = mn.Vector3(*[float(v) for v in ground])
        obj.rotation = mn.Quaternion.rotation(mn.Deg(yaw), mn.Vector3.y_axis())

        rgb = np.asarray(sim.get_sensor_observations()["colour"])[..., :3]
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        error = errors.get(frame)
        verdict = "occluded" if (error or 0) > 5 else "verified"
        colour_rgb = (255, 90, 90) if verdict == "occluded" else (150, 255, 150)
        draw.rectangle([0, 0, 330, 54], fill=(0, 0, 0))
        draw.text((8, 4), f"frame {frame:>3}   range {ranges.get(frame, 0):.2f} m",
                  fill=(235, 235, 235))
        draw.text((8, 20),
                  f"DoA error {('n/a' if error is None else f'{error:5.1f}')} deg  "
                  f"{verdict}", fill=colour_rgb)
        draw.text((8, 36),
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
        "frame_rate_hz": bank["frame_rate_hz"] / max(report["frame_stride"], 1),
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
