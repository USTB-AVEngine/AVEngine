"""Render the same speaker placement the acoustic chain used, in AVEngine's own runtime.

The acoustic half already answers "which direction did the sound come from".
This answers "and is that where the speaker actually is", from the renderer,
using the one placement file both halves read. The two claims are only worth
something together: a placement can satisfy a direction-of-arrival check while
the mesh sits somewhere else entirely, because the audio source is a point that
was derived from the placement rather than read back from the scene.

So nothing here re-derives geometry. Every number comes back out of the
renderer:

  * the semantic buffer gives each speaker's pixels, so its bearing is measured
    from the image rather than recomputed from the numbers that put it there;
  * habitat reports the inserted object's own bounding box, so "where the
    engine thinks this object is" is compared against "where its pixels are";
  * the emitter point the audio chain used is projected into the same image and
    has to land on that speaker.

That last one is the audio-visual binding claim in its weakest, and therefore
checkable, form. Sub-degree agreement is not the standard - the emitter is a
deliberate offset from the object centre, a woofer cone rather than a
centroid - the standard is that the sound leaves the object you can see.

Two things about this runtime cost a day to find and are worth stating:

  * habitat_sim is not importable from any environment here. It is activated
    through avengine.rooms.habitat_capture.prepare_installed_habitat_runtime
    against an installed prefix under /data/avengine_external/runtime-prefixes.
    Every installed prefix is adapter-linked with no RUNPATH, so a visual-only
    caller still has to pass an explicit rlr_sdk_root or the binding will not
    load.
  * do not point --scene at an HM3D *.basis.glb. The Magnum site here has no
    BasisImporter, so every texture silently fails to load and the first
    get_sensor_observations segfaults - exit 139, no traceback, stdout never
    flushed. Each HM3D scene ships an uncompressed sibling; use it.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def look_at_rotation(direction, np_quaternion):
    """Agent rotation whose camera -Z axis points along `direction`.

    Derived rather than searched: yaw about +Y then pitch about the rotated +X
    reproduces the direction exactly, which the caller re-checks by projecting
    the target back into the image.
    """

    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    yaw = math.atan2(-d[0], -d[2])
    pitch = math.asin(float(np.clip(d[1], -1.0, 1.0)))
    qy = np_quaternion(math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0)
    qx = np_quaternion(math.cos(pitch / 2), math.sin(pitch / 2), 0.0, 0.0)
    return qy * qx


def rotation_matrix(quat):
    w, x, y, z = quat.w, quat.x, quat.y, quat.z
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def pixel_to_world(u, v, width, height, hfov_deg, rot):
    """Direction of the ray through a pixel, in world coordinates."""

    focal = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    cam = np.array(
        [(u - width / 2.0 + 0.5) / focal, -(v - height / 2.0 + 0.5) / focal, -1.0]
    )
    world = rot @ cam
    return world / np.linalg.norm(world)


def world_to_pixel(point, eye, width, height, hfov_deg, rot):
    """Where a world point lands in the image, or None if it is behind the camera."""

    cam = rot.T @ (np.asarray(point, dtype=float) - np.asarray(eye, dtype=float))
    if cam[2] >= -1e-6:
        return None
    focal = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return (
        focal * cam[0] / -cam[2] + width / 2.0 - 0.5,
        -focal * cam[1] / -cam[2] + height / 2.0 - 0.5,
    )


def angle_between(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(float(np.clip(np.dot(a, b), -1.0, 1.0)))))


def emitter_world_position(asset_dir, position, yaw_deg):
    """Identical to the acoustic tool: the offset is in the asset frame, yaw only.

    Kept as its own copy on purpose. If the two chains ever disagree about
    where the emitter is, that has to show up as a failed check here rather
    than be hidden by a shared helper.
    """

    record = json.loads((asset_dir / "asset.json").read_text(encoding="utf-8"))
    offset = np.array(record["emitter"]["offset_m"], dtype=float)
    angle = math.radians(yaw_deg)
    rotated = np.array(
        [
            offset[0] * math.cos(angle) + offset[2] * math.sin(angle),
            offset[1],
            -offset[0] * math.sin(angle) + offset[2] * math.cos(angle),
        ]
    )
    return np.array(position, dtype=float) + rotated, record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-prefix", required=True)
    parser.add_argument("--magnum-site", required=True)
    parser.add_argument("--rlr-sdk-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--placement", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--acoustic-report", type=Path, default=None)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="the directory containing scene_datasets; found from --scene if omitted",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--hfov-deg", type=float, default=70.0)
    parser.add_argument(
        "--centroid-tolerance-deg",
        type=float,
        default=2.0,
        help="engine bbox centre against measured pixel centroid",
    )
    args = parser.parse_args()

    if Path(args.scene).name.endswith(".basis.glb"):
        raise SystemExit(
            "refusing a *.basis.glb: this Magnum site has no BasisImporter, so "
            "textures fail to load and rendering segfaults. Use the "
            "uncompressed sibling glb."
        )

    dataset_root = args.dataset_root
    if dataset_root is None:
        # The loader wants the directory that holds scene_datasets, which sits a
        # different number of levels above the mesh in every dataset. Walk up
        # rather than count.
        for parent in Path(args.scene).resolve().parents:
            if (parent / "scene_datasets").is_dir():
                dataset_root = parent
                break
    if dataset_root is None:
        raise SystemExit(
            "no scene_datasets directory above --scene; pass --dataset-root"
        )

    from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime

    rt = prepare_installed_habitat_runtime(
        runtime_prefix=args.runtime_prefix,
        magnum_python_site=args.magnum_site,
        rlr_sdk_root=args.rlr_sdk_root,
        mp3d_root=str(dataset_root),
        allow_mp3d_environment=False,
    )
    hs = rt.habitat_sim
    mn = rt.magnum
    np_quaternion = rt.quaternion.quaternion

    plan = json.loads(args.placement.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    backend = hs.SimulatorConfiguration()
    backend.scene_id = args.scene
    backend.load_semantic_mesh = False
    backend.enable_physics = True
    if rt.physics_config_path:
        backend.physics_config_file = str(rt.physics_config_path)

    colour = hs.CameraSensorSpec()
    colour.uuid = "colour"
    colour.sensor_type = hs.SensorType.COLOR
    colour.resolution = [args.height, args.width]
    colour.hfov = args.hfov_deg
    colour.position = [0.0, 0.0, 0.0]

    labels = hs.CameraSensorSpec()
    labels.uuid = "labels"
    labels.sensor_type = hs.SensorType.SEMANTIC
    labels.resolution = [args.height, args.width]
    labels.hfov = args.hfov_deg
    labels.position = [0.0, 0.0, 0.0]

    agent_cfg = hs.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [colour, labels]
    sim = hs.Simulator(hs.Configuration(backend, [agent_cfg]))

    templates = sim.get_object_template_manager()
    objects = sim.get_rigid_object_manager()

    placed = []
    for index, item in enumerate(plan["speakers"], start=1):
        asset_dir = args.asset_root / item["asset"]
        glb = asset_dir / "finalized.glb"
        if not glb.is_file():
            raise SystemExit(f"missing asset mesh: {glb}")
        emitter, record = emitter_world_position(
            asset_dir, item["position"], item.get("yaw_deg", 0.0)
        )
        handle = item["asset"].replace("/", "__")
        template = templates.create_new_template(handle)
        template.render_asset_handle = str(glb)
        template.collision_asset_handle = str(glb)
        template.is_collidable = False
        template_id = templates.register_template(template, handle)
        obj = objects.add_object_by_template_id(template_id)
        if obj is None:
            raise SystemExit(f"habitat refused to instance {handle}")
        obj.translation = mn.Vector3(*[float(v) for v in item["position"]])
        obj.rotation = mn.Quaternion.rotation(
            mn.Deg(float(item.get("yaw_deg", 0.0))), mn.Vector3.y_axis()
        )
        obj.motion_type = hs.physics.MotionType.KINEMATIC
        obj.semantic_id = index

        # Ask the engine where it thinks the object is, rather than assuming the
        # translation landed. The bounding box comes back in the object frame.
        box = obj.root_scene_node.cumulative_bb
        centre = box.center()
        local_centre = np.array([centre[0], centre[1], centre[2]], dtype=float)
        size = box.size()
        yaw = math.radians(float(item.get("yaw_deg", 0.0)))
        engine_centre = np.array(item["position"], dtype=float) + np.array(
            [
                local_centre[0] * math.cos(yaw) + local_centre[2] * math.sin(yaw),
                local_centre[1],
                -local_centre[0] * math.sin(yaw) + local_centre[2] * math.cos(yaw),
            ]
        )
        placed.append(
            {
                "semantic_id": index,
                "asset": item["asset"],
                "object_type": record["identity"]["object_type"],
                "realized_attributes": record["realized_attributes"],
                "position": [float(v) for v in item["position"]],
                "yaw_deg": float(item.get("yaw_deg", 0.0)),
                "emitter_world": [round(float(v), 4) for v in emitter],
                "engine_bbox_centre_world": [round(float(v), 4) for v in engine_centre],
                "engine_bbox_extent_m": [round(float(size[i]), 4) for i in range(3)],
            }
        )

    acoustic = {}
    if args.acoustic_report and args.acoustic_report.is_file():
        for source in json.loads(args.acoustic_report.read_text(encoding="utf-8"))[
            "sources"
        ]:
            acoustic[source["asset"]] = source

    receiver = np.array(plan["receiver"], dtype=float)
    agent = sim.get_agent(0)

    try:
        from PIL import Image
    except ImportError:
        Image = None

    failures = []
    for entry in placed:
        emitter = np.array(entry["emitter_world"], dtype=float)
        geometric = emitter - receiver
        entry["range_m"] = round(float(np.linalg.norm(geometric)), 3)
        geometric = geometric / np.linalg.norm(geometric)

        state = agent.get_state()
        state.position = receiver.astype(np.float32)
        state.rotation = look_at_rotation(geometric, np_quaternion)
        state.sensor_states = {}
        agent.set_state(state, True)
        rot = rotation_matrix(state.rotation)

        observation = sim.get_sensor_observations()
        rgb = np.asarray(observation["colour"])[..., :3]
        mask = np.asarray(observation["labels"]).reshape(args.height, args.width)
        pixels = np.argwhere(mask == entry["semantic_id"])
        entry["visible_pixels"] = int(pixels.shape[0])

        if pixels.shape[0] == 0:
            entry["verdict"] = "not_visible"
            failures.append((entry["asset"], "not visible from the receiver"))
        else:
            centroid_v, centroid_u = pixels.mean(axis=0)
            measured = pixel_to_world(
                centroid_u, centroid_v, args.width, args.height, args.hfov_deg, rot
            )
            entry["measured_pixel_centroid"] = [
                round(float(centroid_u), 2),
                round(float(centroid_v), 2),
            ]
            entry["measured_direction"] = [round(float(v), 4) for v in measured]

            engine_direction = np.array(entry["engine_bbox_centre_world"]) - receiver
            entry["engine_vs_measured_deg"] = round(
                angle_between(engine_direction, measured), 3
            )
            if entry["engine_vs_measured_deg"] > args.centroid_tolerance_deg:
                entry["verdict"] = "render_disagrees_with_engine"
                failures.append(
                    (
                        entry["asset"],
                        f"engine bbox centre is {entry['engine_vs_measured_deg']} deg "
                        f"from where its pixels are",
                    )
                )
            else:
                entry["verdict"] = "ok"

            # Does the emitter the audio chain used land on this speaker?
            spot = world_to_pixel(
                emitter, receiver, args.width, args.height, args.hfov_deg, rot
            )
            entry["emitter_pixel"] = (
                [round(float(spot[0]), 2), round(float(spot[1]), 2)] if spot else None
            )
            if spot is not None:
                su, sv = int(round(spot[0])), int(round(spot[1]))
                on_speaker = (
                    0 <= sv < args.height
                    and 0 <= su < args.width
                    and mask[sv, su] == entry["semantic_id"]
                )
                entry["emitter_lands_on_speaker"] = bool(on_speaker)
                if not on_speaker:
                    failures.append(
                        (
                            entry["asset"],
                            "the emitter the audio chain used does not project "
                            "onto this speaker",
                        )
                    )

            heard = acoustic.get(entry["asset"], {})
            if "measured_direction" in heard:
                entry["acoustic_vs_visual_deg"] = round(
                    angle_between(heard["measured_direction"], measured), 3
                )
            if "direction_error_deg" in heard:
                entry["acoustic_direction_error_deg"] = heard["direction_error_deg"]

        stem = entry["asset"].replace("/", "__")
        if Image is not None:
            Image.fromarray(rgb).save(args.output_dir / f"{stem}.colour.png")
            palette = (mask[..., None] * np.array([73, 149, 211])) % 256
            Image.fromarray(palette.astype(np.uint8)).save(
                args.output_dir / f"{stem}.labels.png"
            )

        finish = entry["realized_attributes"].get("finish", "")
        print(
            f"{entry['object_type']:<22} {finish:<18} "
            f"pixels {entry['visible_pixels']:>7}  "
            f"engine-vs-render {entry.get('engine_vs_measured_deg', float('nan')):6.2f} deg  "
            f"emitter-on-speaker {entry.get('emitter_lands_on_speaker')}"
        )

    report = {
        "scene": args.scene,
        "receiver": [float(v) for v in receiver],
        "renderer": "AVEngine installed habitat prefix",
        "runtime_prefix": str(rt.prefix),
        "magnum_python_site": str(rt.magnum_python_site),
        "camera": {
            "width": args.width,
            "height": args.height,
            "hfov_deg": args.hfov_deg,
        },
        "centroid_tolerance_deg": args.centroid_tolerance_deg,
        "sources": placed,
        "failures": [{"asset": a, "reason": r} for a, r in failures],
    }
    (args.output_dir / "visual_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {args.output_dir}  sources={len(placed)} failures={len(failures)}")
    for asset, reason in failures:
        print(f"  {asset}: {reason}")
    sim.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
