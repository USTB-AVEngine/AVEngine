"""Build a speaker placement from the scene's navmesh.

Positions cannot be guessed. A source dropped inside a wall has no direct path,
so the first arrival the intensity vector finds is a reflection and the measured
direction is wrong by a hundred degrees - which is exactly what hand-picked
coordinates produced on apartment_1.

Navigable points are floor the agent can stand on, so a receiver placed there is
in the room rather than in the geometry, and a speaker raised above one is on
something rather than inside it. Every pair is then checked for a clear line of
sight before it is written out.

Superseded by tools/scene/plan_supported_placement.py. Both placement rules
here are wrong in a furnished scene and were only ever exercised in an empty
one; see that file for what replaced them and why.
"""

import argparse
import json

import numpy as np
import quaternion  # noqa: F401
import habitat_sim


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--navmesh", required=True)
    parser.add_argument("--assets", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--receiver-height", type=float, default=1.5)
    # 0.75 m above the navmesh holds nothing up. Rendered into a furnished
    # HM3D scene every speaker placed this way hangs in mid-air - over a
    # stairwell, inside a door. Use tools/scene/plan_supported_placement.py,
    # which drops a ray onto whatever surface is actually there.
    parser.add_argument("--speaker-height", type=float, default=0.75)
    parser.add_argument("--minimum-range-m", type=float, default=1.6)
    parser.add_argument("--maximum-range-m", type=float, default=5.0)
    parser.add_argument(
        "--detour-limit",
        type=float,
        default=1.08,
        help="largest geodesic-over-straight ratio still counted as one open space",
    )
    args = parser.parse_args()

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = args.scene
    backend.load_semantic_mesh = False
    backend.enable_physics = False
    sim = habitat_sim.Simulator(
        habitat_sim.Configuration(backend, [habitat_sim.agent.AgentConfiguration()])
    )
    pathfinder = sim.pathfinder
    if not pathfinder.is_loaded:
        pathfinder.load_nav_mesh(args.navmesh)
    if not pathfinder.is_loaded:
        raise SystemExit(f"navmesh did not load: {args.navmesh}")
    pathfinder.seed(args.seed)

    receiver = np.array(pathfinder.get_random_navigable_point(), dtype=float)
    receiver[1] += args.receiver_height

    def clear_line(a, b) -> bool:
        """Same open space, so the first arrival is the direct one.

        Superseded. cast_ray does hit the static stage, but only when the
        simulator is built with enable_physics=True, which this tool leaves
        off; every probe that concluded otherwise ran with physics disabled.
        tools/scene/plan_supported_placement.py does the real cast and rejects
        pairs this ratio lets through. Kept below as written for the record:

        Not a ray cast: cast_ray in this build only hits rigid objects, never
        the static stage, so it reports a clear line through solid walls and
        the check silently passes everything. The navmesh does know about
        walls, and a geodesic path that is barely longer than the straight
        line means nothing is between the two points.
        """

        start = pathfinder.snap_point(np.array(a, dtype=np.float32))
        end = pathfinder.snap_point(np.array(b, dtype=np.float32))
        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = end
        if not pathfinder.find_path(path):
            return False
        straight = float(np.linalg.norm(np.array(end) - np.array(start)))
        if straight < 1e-6:
            return False
        return path.geodesic_distance / straight <= args.detour_limit

    speakers = []
    attempts = 0
    while len(speakers) < len(args.assets) and attempts < 4000:
        attempts += 1
        point = np.array(pathfinder.get_random_navigable_point(), dtype=float)
        point[1] += args.speaker_height
        distance = float(np.linalg.norm(point - receiver))
        if not args.minimum_range_m <= distance <= args.maximum_range_m:
            continue
        if any(np.linalg.norm(point - np.array(s["position"])) < 0.8 for s in speakers):
            continue
        if not clear_line(receiver, point):
            continue
        speakers.append(
            {
                "asset": args.assets[len(speakers)],
                "position": [round(float(v), 4) for v in point],
                "yaw_deg": 0.0,
                "range_m": round(distance, 3),
            }
        )

    if len(speakers) < len(args.assets):
        raise SystemExit(
            f"only placed {len(speakers)} of {len(args.assets)} after {attempts} tries"
        )

    plan = {
        "scene": args.scene,
        "navmesh": args.navmesh,
        "seed": args.seed,
        "receiver": [round(float(v), 4) for v in receiver],
        "receiver_rotation": [1, 0, 0, 0],
        "speakers": speakers,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=1)
    print(f"receiver {plan['receiver']}")
    for item in speakers:
        print(f"  {item['asset']:<52} {item['position']}  range {item['range_m']} m")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
