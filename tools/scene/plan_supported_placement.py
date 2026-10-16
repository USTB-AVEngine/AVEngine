"""Place speakers on surfaces that exist, and check the sightline against geometry.

This replaces tools/audio/plan_from_navmesh.py, whose two placement rules were
both wrong in a furnished scene and both looked right in an empty one:

  * it raised every speaker a fixed 0.75 m above the navmesh, reasoning that a
    raised speaker is "on something rather than inside it". Nothing holds it up.
    Rendered into HM3D 00800-TEEsavR23oF, all three visible speakers hang in
    mid-air - one of them out over a stairwell, one embedded in a door.
  * it approximated the sightline with the ratio of geodesic to straight-line
    distance on the navmesh, because cast_ray appeared to hit nothing. That
    proxy passes a pair whose straight line clips a wall corner near a doorway,
    which is how a speaker ended up acoustically 36 degrees wrong and visually
    behind a wall.

Both are fixed by the same discovery: cast_ray does hit the static stage, but
only when the simulator is built with physics enabled. Every earlier probe ran
with physics off and measured zero hits, which read as "the scene is not
ray-castable" rather than "collision geometry was never loaded".

So placement here is: sample navigable floor, drop a ray to find whatever
surface is actually at that spot - floor, table, shelf - reject it unless the
surface faces up and there is headroom for the object, seat the object's own
bounding box on it, then require an unobstructed ray from the receiver to the
emitter rather than a navmesh proxy for one.

The output is the same placement file the acoustic and visual tools already
read, so neither has to change.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

UP = np.array([0.0, 1.0, 0.0])


def emitter_offset_world(record, yaw_deg):
    offset = np.array(record["emitter"]["offset_m"], dtype=float)
    angle = math.radians(yaw_deg)
    return np.array(
        [
            offset[0] * math.cos(angle) + offset[2] * math.sin(angle),
            offset[1],
            -offset[0] * math.sin(angle) + offset[2] * math.cos(angle),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-prefix", required=True)
    parser.add_argument("--magnum-site", required=True)
    parser.add_argument("--rlr-sdk-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--navmesh", required=True)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--assets", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--receiver-height", type=float, default=1.5)
    parser.add_argument(
        "--probe-height",
        type=float,
        default=1.4,
        help="how far above the floor to start looking for a support surface",
    )
    parser.add_argument("--minimum-range-m", type=float, default=1.0)
    parser.add_argument("--maximum-range-m", type=float, default=6.0)
    parser.add_argument("--minimum-separation-m", type=float, default=0.8)
    parser.add_argument(
        "--level-normal-cosine",
        type=float,
        default=0.9,
        help="how close to horizontal a surface has to be to hold a speaker",
    )
    parser.add_argument("--attempts", type=int, default=4000)
    parser.add_argument(
        "--face-receiver",
        action="store_true",
        default=True,
        help="turn each speaker so its declared front faces the listener",
    )
    parser.add_argument("--no-face-receiver", dest="face_receiver", action="store_false")
    args = parser.parse_args()

    if Path(args.scene).name.endswith(".basis.glb"):
        raise SystemExit(
            "refusing a *.basis.glb: no BasisImporter in this Magnum site, so "
            "rendering the result would segfault. Use the uncompressed sibling."
        )

    dataset_root = None
    for parent in Path(args.scene).resolve().parents:
        if (parent / "scene_datasets").is_dir():
            dataset_root = parent
            break
    if dataset_root is None:
        raise SystemExit("no scene_datasets directory above --scene")

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

    backend = hs.SimulatorConfiguration()
    backend.scene_id = args.scene
    backend.load_semantic_mesh = False
    # Not optional. cast_ray returns no hits against the static stage unless
    # physics is on, because that is when its collision mesh is built.
    backend.enable_physics = True
    if rt.physics_config_path:
        backend.physics_config_file = str(rt.physics_config_path)
    sim = hs.Simulator(hs.Configuration(backend, [hs.agent.AgentConfiguration()]))

    pathfinder = sim.pathfinder
    if not pathfinder.is_loaded:
        pathfinder.load_nav_mesh(args.navmesh)
    if not pathfinder.is_loaded:
        raise SystemExit(f"navmesh did not load: {args.navmesh}")
    pathfinder.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    def drop(point, distance):
        ray = hs.geo.Ray(mn.Vector3(*[float(v) for v in point]), mn.Vector3(0.0, -1.0, 0.0))
        result = sim.cast_ray(ray, max_distance=distance)
        if not result.has_hits():
            return None
        return result.hits[0]

    def obstructed(start, end):
        """True when scene geometry sits between the two points."""

        delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        span = float(np.linalg.norm(delta))
        ray = hs.geo.Ray(
            mn.Vector3(*[float(v) for v in start]),
            mn.Vector3(*[float(v) for v in (delta / span)]),
        )
        result = sim.cast_ray(ray, max_distance=span)
        if not result.has_hits():
            return False
        # A hit essentially at the far end is the surface the speaker rests
        # against, not an obstruction in front of it.
        return float(result.hits[0].ray_distance) < span - 0.05

    # Each asset's own bounding box decides how far above the support surface
    # its origin has to sit, and how much headroom the spot needs.
    metrics = {}
    templates = sim.get_object_template_manager()
    objects = sim.get_rigid_object_manager()
    for asset in args.assets:
        asset_dir = args.asset_root / asset
        glb = asset_dir / "finalized.glb"
        if not glb.is_file():
            raise SystemExit(f"missing asset mesh: {glb}")
        handle = asset.replace("/", "__")
        template = templates.create_new_template(handle)
        template.render_asset_handle = str(glb)
        template.collision_asset_handle = str(glb)
        template.is_collidable = False
        obj = objects.add_object_by_template_id(templates.register_template(template, handle))
        box = obj.root_scene_node.cumulative_bb
        metrics[asset] = {
            "base_below_origin_m": float(-box.min[1]),
            "height_m": float(box.size()[1]),
            "footprint_radius_m": float(
                max(abs(box.min[0]), abs(box.max[0]), abs(box.min[2]), abs(box.max[2]))
            ),
            "record": json.loads((asset_dir / "asset.json").read_text(encoding="utf-8")),
        }
        objects.remove_object_by_id(obj.object_id)

    receiver = np.array(pathfinder.get_random_navigable_point(), dtype=float)
    receiver[1] += args.receiver_height

    speakers = []
    rejected = {
        "no_support_surface": 0,
        "surface_not_level": 0,
        "no_headroom": 0,
        "out_of_range": 0,
        "too_close_to_another": 0,
        "sightline_obstructed": 0,
    }
    remaining = list(args.assets)
    attempts = 0
    while remaining and attempts < args.attempts:
        attempts += 1
        asset = remaining[0]
        spec = metrics[asset]

        floor = np.array(pathfinder.get_random_navigable_point(), dtype=float)
        hit = drop(floor + UP * args.probe_height, args.probe_height + 0.5)
        if hit is None:
            rejected["no_support_surface"] += 1
            continue
        normal = np.array([hit.normal[0], hit.normal[1], hit.normal[2]], dtype=float)
        norm = float(np.linalg.norm(normal))
        if norm <= 0 or float(np.dot(normal / norm, UP)) < args.level_normal_cosine:
            rejected["surface_not_level"] += 1
            continue
        support = np.array(
            [hit.point[0], hit.point[1], hit.point[2]], dtype=float
        )

        # Headroom: nothing may hang directly over the spot within the object's
        # own height, or the speaker would be placed inside it.
        ceiling_ray = hs.geo.Ray(
            mn.Vector3(*[float(v) for v in support + UP * 0.02]), mn.Vector3(0.0, 1.0, 0.0)
        )
        overhead = sim.cast_ray(ceiling_ray, max_distance=spec["height_m"] + 0.05)
        if overhead.has_hits():
            rejected["no_headroom"] += 1
            continue

        position = support + UP * spec["base_below_origin_m"]
        offset = position - receiver
        distance = float(np.linalg.norm(offset))
        if not (args.minimum_range_m <= distance <= args.maximum_range_m):
            rejected["out_of_range"] += 1
            continue
        if any(
            np.linalg.norm(position - np.array(other["position"])) < args.minimum_separation_m
            for other in speakers
        ):
            rejected["too_close_to_another"] += 1
            continue

        if args.face_receiver:
            towards = receiver - position
            yaw_deg = math.degrees(math.atan2(-towards[2], towards[0]))
        else:
            yaw_deg = float(rng.uniform(-180.0, 180.0))

        emitter = position + emitter_offset_world(spec["record"], yaw_deg)
        if obstructed(receiver, emitter):
            rejected["sightline_obstructed"] += 1
            continue

        speakers.append(
            {
                "asset": asset,
                "position": [round(float(v), 4) for v in position],
                "yaw_deg": round(float(yaw_deg), 2),
                "range_m": round(float(np.linalg.norm(emitter - receiver)), 3),
                "support_surface_y": round(float(support[1]), 4),
                "support_above_floor_m": round(float(support[1] - floor[1]), 4),
            }
        )
        remaining.pop(0)

    if remaining:
        raise SystemExit(
            f"only placed {len(speakers)} of {len(args.assets)} in {attempts} attempts; "
            f"rejections {rejected}"
        )

    plan = {
        "scene": args.scene,
        "navmesh": args.navmesh,
        "seed": args.seed,
        "planner": "plan_supported_placement.py",
        "receiver": [round(float(v), 4) for v in receiver],
        "receiver_rotation": [1, 0, 0, 0],
        "speakers": speakers,
        "attempts": attempts,
        "rejections": rejected,
    }
    args.output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"receiver {plan['receiver']}")
    for entry in speakers:
        print(
            f"  {entry['asset']:<58} {str(entry['position']):<34} "
            f"yaw {entry['yaw_deg']:>7.1f}  range {entry['range_m']:.3f}  "
            f"support +{entry['support_above_floor_m']:.3f} m over navmesh"
        )
    print(f"attempts {attempts}  rejections {rejected}")
    sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
