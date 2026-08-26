"""Choose listener poses for a route, so nothing downstream has to invent one.

A listener pose is a property of the scene and the route, not of the camera and
not of the audio. Before this existed the two halves each chose part of it - the
acoustic render picked the position, the video picked the orientation - which
split one decision across two tools, forced an order between them, and left the
caller passing a bare direction vector that only made sense for one room and one
camera. Whoever ran the second tool had to know what the first had decided.

So the decision is made once here and written down. Both halves read the file
and neither invents anything.

The two things a pose needs cannot be measured in one place, which is why this
emits a ranked list rather than a single answer:

  * the orientation wants depth. A camera aimed at a wall thirty centimetres
    away is legal, points at the route, and shows nothing for the whole clip.
    Candidate yaws are rendered and scored by median depth, which needs the
    visual runtime.
  * whether a position can actually hear the route wants a render of the sound.
    A navmesh knows the floor is walkable and knows nothing about what stands at
    ear height, so a position that looks fine can be screened from the route by
    a wall. That check belongs to the acoustic pass, which runs in a different
    environment against a different build.

So this ranks candidates by openness and by how squarely they face the route,
the acoustic pass auditions them in order and records which one it accepted, and
the video pass uses that same entry.
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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--episode-id")
    parser.add_argument("--motion-case", default="source1_moving_source2_static")
    parser.add_argument("--slot", default="source1")
    parser.add_argument("--emitter-height-m", type=float)
    parser.add_argument("--listener-height-m", type=float, default=1.5)
    parser.add_argument("--minimum-range-m", type=float, default=2.0)
    parser.add_argument("--maximum-range-m", type=float, default=6.0)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--yaw-step-deg", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--hfov-deg", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    bank = json.loads(args.bank.read_text(encoding="utf-8"))
    if args.episode_id:
        episode = next(
            e for e in bank["episodes"] if e["episode_id"] == args.episode_id
        )
    else:
        matching = [
            e for e in bank["episodes"] if e["motion_case"] == args.motion_case
        ]
        if not matching:
            raise SystemExit(f"no {args.motion_case} episodes in {args.bank}")
        episode = matching[args.episode_index]

    scene = bank["scene"]
    if Path(scene).name.endswith(".basis.glb"):
        raise SystemExit("refusing a *.basis.glb: no BasisImporter here, it segfaults")
    height = (
        float(args.emitter_height_m)
        if args.emitter_height_m is not None
        else float(bank["source_center_heights_m"][args.slot])
    )
    path = np.asarray(episode["source_center_paths_m"][args.slot], dtype=float)
    emitters = path.copy()
    emitters[:, 1] += height
    floor = float(bank["floor_height_m"])

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
    backend.enable_physics = False
    depth = hs.CameraSensorSpec()
    depth.uuid = "depth"
    depth.sensor_type = hs.SensorType.DEPTH
    depth.resolution = [args.height, args.width]
    depth.hfov = args.hfov_deg
    depth.position = [0.0, 0.0, 0.0]
    agent_cfg = hs.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [depth]
    sim = hs.Simulator(hs.Configuration(backend, [agent_cfg]))
    pathfinder = sim.pathfinder
    navmesh = bank.get("navmesh")
    if navmesh and not pathfinder.is_loaded:
        pathfinder.load_nav_mesh(navmesh)
    if not pathfinder.is_loaded:
        raise SystemExit("no navmesh; querying an unloaded PathFinder segfaults")
    agent = sim.get_agent(0)

    def median_depth(position, direction):
        state = agent.get_state()
        state.position = np.asarray(position, dtype=np.float32)
        state.rotation = look_at(direction, np_quaternion)
        state.sensor_states = {}
        agent.set_state(state, True)
        d = np.asarray(sim.get_sensor_observations()["depth"], dtype=float)
        d = d[np.isfinite(d) & (d > 0)]
        return float(np.median(d)) if d.size else 0.0

    midpoint = emitters[len(emitters) // 2]

    def route_share(position, direction):
        """Fraction of the route this orientation holds inside the frame."""

        directions = emitters - position
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        return float(
            np.mean(
                np.degrees(np.arccos(np.clip(directions @ direction, -1.0, 1.0)))
                <= args.hfov_deg / 2.0
            )
        )

    def entry(position, direction, degrees, depth, clearance, span):
        return {
            "position_m": [round(float(v), 4) for v in position],
            "aim_world": [round(float(v), 6) for v in direction],
            "aim_yaw_deg": round(float(degrees), 1),
            "open_depth_m": round(float(depth), 3),
            "clearance_m": round(float(clearance), 3),
            "range_to_route_midpoint_m": round(float(span), 3),
            "route_share_in_frame": round(route_share(position, direction), 3),
        }

    yaws = [
        (float(d), np.array([math.sin(math.radians(float(d))), 0.0,
                             -math.cos(math.radians(float(d)))]))
        for d in np.arange(0.0, 360.0, args.yaw_step_deg)
    ]

    # A bank may already be built around a listener - an orbit is defined as a
    # circle about one - and then the position is not this tool's to choose.
    # Choosing freely broke exactly that: an orbit bank came back with a listener
    # 4.4 m from the circle's centre, so the source stayed on one side of the
    # head for the whole clip and the interaural difference never changed sign.
    declared = bank.get("listener_m")
    candidates = []
    if declared is not None:
        position = np.asarray(declared, dtype=float)
        clearance = float(pathfinder.distance_to_closest_obstacle(position, 6.0))
        span = float(np.linalg.norm(position - midpoint))
        print(
            f"the bank declares its listener at {np.round(position, 3).tolist()}; "
            "choosing orientation only"
        )
        for degrees, direction in yaws:
            depth = median_depth(position, direction)
            candidates.append(
                entry(position, direction, degrees, depth, clearance, span)
            )
            print(f"  yaw {degrees:5.1f} deg  depth {depth:5.2f} m  "
                  f"route in frame "
                  f"{100 * candidates[-1]['route_share_in_frame']:5.1f}%")
    else:
        positions = []
        for _ in range(args.samples):
            candidate = np.asarray(
                pathfinder.get_random_navigable_point(), dtype=float
            )
            if abs(candidate[1] - floor) > 0.5:
                continue
            candidate[1] = floor + args.listener_height_m
            span = float(np.linalg.norm(candidate - midpoint))
            if not (args.minimum_range_m <= span <= args.maximum_range_m):
                continue
            clearance = float(
                pathfinder.distance_to_closest_obstacle(candidate, 6.0)
            )
            positions.append((clearance, span, candidate))
        if not positions:
            raise SystemExit(
                "no navigable position on this floor inside the range band; "
                "widen --minimum-range-m / --maximum-range-m"
            )
        positions.sort(key=lambda item: -item[0])
        for clearance, span, position in positions[: args.candidates]:
            best = max(
                ((median_depth(position, d), deg, d) for deg, d in yaws),
                key=lambda item: item[0],
            )
            candidates.append(
                entry(position, best[2], best[1], best[0], clearance, span)
            )
            print(
                f"  candidate {len(candidates):>2}  clearance {clearance:5.2f} m  "
                f"depth {best[0]:5.2f} m  range {span:5.2f} m  "
                f"route in frame "
                f"{100 * candidates[-1]['route_share_in_frame']:5.1f}%"
            )

    if declared is not None:
        # An orbit puts the same share of itself in front of every orientation,
        # so depth is what separates them and openness decides.
        candidates.sort(key=lambda c: (-c["open_depth_m"], -c["route_share_in_frame"]))
    else:
        # A deep view of an empty corner is worse than a shallower one the source
        # crosses, so how much of the route is held comes first.
        candidates.sort(key=lambda c: (-c["route_share_in_frame"], -c["open_depth_m"]))
    candidates = candidates[: args.candidates]
    pose = {
        "schema": "avengine_listener_pose_v1",
        "decides": (
            "position and orientation for one route, so the acoustic and visual "
            "passes read one decision instead of each making half of it"
        ),
        "scene": scene,
        "navmesh": navmesh,
        "bank": str(args.bank),
        "episode_id": episode["episode_id"],
        "slot": args.slot,
        "emitter_height_m": height,
        "listener_height_m": args.listener_height_m,
        "hfov_deg": args.hfov_deg,
        "position_from": (
            "the bank's declared listener, orientation only"
            if bank.get("listener_m") is not None
            else "sampled navigable positions inside the range band"
        ),
        "ranked_by": (
            "open_depth_m, then route_share_in_frame"
            if bank.get("listener_m") is not None
            else "route_share_in_frame, then open_depth_m"
        ),
        "acoustic_audition": (
            "not done here: whether a position can hear the route needs a render "
            "of the sound, in the acoustic environment. That pass tries these in "
            "order and writes accepted_index back into this file"
        ),
        "accepted_index": None,
        "candidates": candidates,
    }
    args.output.write_text(
        json.dumps(pose, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {args.output}  {len(candidates)} candidates, best first")
    sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
