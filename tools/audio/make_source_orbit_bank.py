"""Write a bank whose source circles the listener, for hearing the surround field.

The navmesh banks produce routes that pass by. Passing by is the honest case for
a walking source and it is a poor way to hear a first-order ambisonic field,
because the direction only sweeps through part of a turn. An orbit sweeps the
whole 360 and makes front, side, back and the crossings between them audible in
one clip.

An orbit is deliberately allowed to leave the navmesh. The navmesh constraint
exists because a walking source has to walk; a sound does not - it can come from
behind a wall, from a shelf, from the next room. Each sample is annotated with
whether it was navigable and how far it sits from walkable floor, so the part of
the circle that passes through geometry is labelled rather than hidden. Those
are the samples that will read as occluded when the direction is measured, which
is the point of including them.

Output is bank-shaped, so the same acoustic and video tools consume it without
knowing an orbit is not a route.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-prefix", required=True)
    parser.add_argument("--magnum-site", required=True)
    parser.add_argument("--rlr-sdk-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--navmesh", required=True)
    parser.add_argument("--listener", nargs=3, type=float, required=True)
    parser.add_argument("--floor-height-m", type=float, required=True)
    parser.add_argument(
        "--orbit",
        action="append",
        required=True,
        metavar="RADIUS:TURNS:START_DEG",
        help="one episode per flag, e.g. 2.0:1:0",
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--frame-rate-hz", type=int, default=15)
    parser.add_argument("--emitter-height-m", type=float, default=1.5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    dataset_root = None
    for parent in Path(args.scene).resolve().parents:
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
    backend = hs.SimulatorConfiguration()
    backend.scene_id = args.scene
    backend.load_semantic_mesh = False
    backend.enable_physics = False
    sim = hs.Simulator(
        hs.Configuration(backend, [hs.agent.AgentConfiguration()])
    )
    pathfinder = sim.pathfinder
    if not pathfinder.is_loaded:
        pathfinder.load_nav_mesh(args.navmesh)
    if not pathfinder.is_loaded:
        raise SystemExit("no navmesh; querying an unloaded PathFinder segfaults")

    listener = np.asarray(args.listener, dtype=float)
    floor = float(args.floor_height_m)
    episodes = []
    for spec in args.orbit:
        parts = spec.split(":")
        radius = float(parts[0])
        turns = float(parts[1]) if len(parts) > 1 else 1.0
        start = float(parts[2]) if len(parts) > 2 else 0.0
        angles = np.radians(
            start + np.linspace(0.0, 360.0 * turns, args.frames, endpoint=False)
        )
        path = np.stack(
            (
                listener[0] + radius * np.cos(angles),
                np.full(args.frames, floor),
                listener[2] + radius * np.sin(angles),
            ),
            axis=1,
        )
        navigable = []
        snap = []
        for point in path:
            navigable.append(bool(pathfinder.is_navigable(point, 0.05)))
            snapped = np.asarray(pathfinder.snap_point(point), dtype=float)
            snap.append(
                float(np.linalg.norm(snapped[[0, 2]] - point[[0, 2]]))
                if np.all(np.isfinite(snapped))
                else float("inf")
            )
        share = float(np.mean(navigable))
        episodes.append(
            {
                "episode_id": f"orbit_r{radius:g}_t{turns:g}_s{start:g}",
                "motion_case": "source1_moving_source2_static",
                "orbit": {
                    "radius_m": radius,
                    "turns": turns,
                    "start_deg": start,
                    "navigable_share": round(share, 3),
                    "median_snap_to_navmesh_m": round(float(np.median(snap)), 3),
                },
                "statistics": {},
                "source_center_paths_m": {
                    "source1": path.tolist(),
                    "source2": np.repeat(
                        listener.reshape(1, 3), args.frames, axis=0
                    ).tolist(),
                },
                "source_root_paths_m": {
                    "source1": path.tolist(),
                    "source2": np.repeat(
                        listener.reshape(1, 3), args.frames, axis=0
                    ).tolist(),
                },
            }
        )
        print(
            f"{episodes[-1]['episode_id']:<24} radius {radius:.2f} m  "
            f"navigable {100 * share:5.1f}%  "
            f"median snap {np.median(snap):.2f} m"
        )

    bank = {
        "schema": "avengine_source_orbit_bank_v1",
        "semantics": (
            "an orbit, not a walkable route; samples are annotated with "
            "navigability rather than filtered by it"
        ),
        "scene": args.scene,
        "navmesh": args.navmesh,
        "floor_height_m": floor,
        "frame_count": args.frames,
        "frame_rate_hz": args.frame_rate_hz,
        "seconds_per_episode": args.frames / args.frame_rate_hz,
        "source_slots": ["source1", "source2"],
        "source_center_heights_m": {
            "source1": args.emitter_height_m,
            "source2": args.emitter_height_m,
        },
        "listener_m": listener.tolist(),
        "episode_count": len(episodes),
        "episodes": episodes,
    }
    args.output.write_text(
        json.dumps(bank, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}  episodes {len(episodes)}  "
          f"{args.frames} frames at {args.frame_rate_hz} Hz "
          f"({args.frames / args.frame_rate_hz:.1f} s)")
    sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
