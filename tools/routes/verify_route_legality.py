"""Is the path itself legal? Occlusion is not the question here.

Whether a listener can hear a route is a property of the listener and the
geometry between them, and an occluded stretch is a case worth having rather
than a defect. Whether the route could be walked at all is a different question
and is not optional: a source that passes through a table did not move, it
teleported, and no amount of correct audio makes that sample usable.

So this checks the path and says nothing about audibility. Three things, in the
order they bite:

  * every sample has to sit on the navmesh at a tight vertical tolerance. The
    0.5 m default habitat uses will happily match a polygon on the storey above,
    so it reports success on a route that is airborne.
  * how close the path runs to real geometry, reported rather than gated. A
    body brushing a corner by a few centimetres is acceptable and a route is not
    thrown away for it; what matters is knowing. Only a gross breach counts as a
    failure - a sample off the navmesh, or clearance under --hard-floor-m -
    because that is a path through a wall rather than a path along one.

    The usual few centimetres are not corner cutting, which was the first guess
    and was wrong. Recast erodes by whole voxels, so a 0.20 m radius request
    realises a minimum inset of 0.102 m at habitat's default 0.05 m cell, 0.158
    at 0.02 and 0.182 at 0.01. Adding a horizontal snap-back for corner-cut
    samples changed not one measured number, because the samples were never off
    the navmesh in the first place.
  * the emitter height has to be within the headroom the navmesh was built for.
    A navmesh generated for a 1.5 m agent guarantees 1.5 m of clear column above
    each walkable cell and guarantees nothing above that, so an emitter placed
    higher is unverified rather than legal.

The reference for clearance is a navmesh rebuilt at a near-zero radius, which
traces the real geometry rather than an already-inset boundary. Measuring
against the inset navmesh the route was planned on answers a different question
- how close the route runs to its own boundary - and always returns roughly
zero, because a shortest path hugs that boundary by construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-prefix", required=True)
    parser.add_argument("--magnum-site", required=True)
    parser.add_argument("--rlr-sdk-root", required=True)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--slot", default="source1")
    parser.add_argument("--body-radius-m", type=float, default=0.20)
    parser.add_argument("--body-height-m", type=float, default=1.5)
    parser.add_argument("--vertical-tolerance-m", type=float, default=0.05)
    parser.add_argument(
        "--hard-floor-m",
        type=float,
        default=0.05,
        help=(
            "below this clearance a sample counts as a real breach rather than "
            "a body brushing a corner. Loose on purpose"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero unless every sample clears the full body radius",
    )
    parser.add_argument(
        "--navmesh-cell-size-m",
        type=float,
        default=0.01,
        help=(
            "voxel size for both navmeshes built here, and they must "
            "match: each is quantised to its own cell, so different "
            "cells compare two quantisations rather than the inset. "
            "Recast erodes by whole cells, which is why the default "
            "0.05 realises as little as 0.10 m of inset from a 0.20 m "
            "request while 0.01 realises 0.18"
        ),
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--episodes", type=int, default=0, help="0 for all")
    args = parser.parse_args()

    bank = json.loads(args.bank.read_text(encoding="utf-8"))
    scene = bank["scene"]
    if Path(scene).name.endswith(".basis.glb"):
        raise SystemExit("refusing a *.basis.glb: no BasisImporter here, it segfaults")
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
    backend = hs.SimulatorConfiguration()
    backend.scene_id = scene
    backend.load_semantic_mesh = False
    backend.enable_physics = False
    sim = hs.Simulator(
        hs.Configuration(backend, [hs.agent.AgentConfiguration()])
    )

    # The route's own navmesh, inset by the body radius: what "on the path"
    # means.
    planned = hs.NavMeshSettings()
    planned.set_defaults()
    planned.agent_radius = args.body_radius_m
    planned.agent_height = args.body_height_m
    planned.cell_size = args.navmesh_cell_size_m
    if not sim.recompute_navmesh(sim.pathfinder, planned):
        raise SystemExit("could not rebuild the planning navmesh")
    inset = sim.pathfinder

    # A separate navmesh at a near-zero radius, which follows the real geometry.
    # Clearance measured against the inset one is always about zero: a shortest
    # path hugs its own boundary, so that number says nothing about the body.
    traced = hs.NavMeshSettings()
    traced.set_defaults()
    traced.agent_radius = 0.01
    traced.agent_height = args.body_height_m
    traced.cell_size = args.navmesh_cell_size_m
    reference = hs.nav.PathFinder()
    sim.recompute_navmesh(reference, traced)
    if not reference.is_loaded:
        raise SystemExit("could not build the geometry-tracing reference navmesh")

    emitter_height = max(
        float(v) for v in bank.get("source_center_heights_m", {"x": 0.0}).values()
    )
    headroom_ok = emitter_height <= args.body_height_m
    # The tracing reference is itself inset by 0.01, so a legal sample should
    # show the body radius less that, not the body radius.
    expected_clearance = max(args.body_radius_m - 0.01, 0.0)
    print(f"a legal sample should clear {expected_clearance:.3f} m at cell size "
          f"{args.navmesh_cell_size_m:.3f} m")
    print(f"emitter height {emitter_height:.2f} m against navmesh headroom "
          f"{args.body_height_m:.2f} m: {'within' if headroom_ok else 'ABOVE, unverified'}")

    # Select the episodes where this slot actually moves before applying the
    # limit. Slicing first and skipping static slots afterwards checked nothing
    # at all on a bank whose first episodes happen to be static_static.
    def moves(episode):
        path = np.asarray(
            episode["source_center_paths_m"][args.slot], dtype=float
        )
        return float(np.linalg.norm(path[-1][[0, 2]] - path[0][[0, 2]])) >= 1.0e-6

    episodes = [e for e in bank["episodes"] if moves(e)]
    if not episodes:
        raise SystemExit(
            f"no episode in this bank moves {args.slot!r}; nothing to verify"
        )
    if args.episodes:
        episodes = episodes[: args.episodes]
    print(f"{len(episodes)} episodes move {args.slot!r}")
    results = []
    for episode in episodes:
        path = np.asarray(
            episode["source_center_paths_m"][args.slot], dtype=float
        )
        off_navmesh = 0
        pierces = 0
        breaches = 0
        worst = float("inf")
        clearances = []
        for sample in path:
            if not inset.is_navigable(sample, args.vertical_tolerance_m):
                off_navmesh += 1
            clearance = float(
                reference.distance_to_closest_obstacle(sample, 10.0)
            )
            if np.isfinite(clearance):
                clearances.append(clearance)
                worst = min(worst, clearance)
                if clearance < expected_clearance:
                    pierces += 1
                if clearance < args.hard_floor_m:
                    breaches += 1
        # Loose by default: off the navmesh or under the hard floor is a path
        # through a wall. Brushing a corner is not.
        usable = off_navmesh == 0 and breaches == 0 and headroom_ok
        strict = usable and pierces == 0
        results.append(
            {
                "episode_id": episode["episode_id"],
                "samples": len(path),
                "off_navmesh": off_navmesh,
                "samples_inside_body_radius": pierces,
                "worst_clearance_m": None if not np.isfinite(worst) else round(worst, 4),
                "median_clearance_m": round(float(np.median(clearances)), 4)
                if clearances else None,
                "emitter_within_headroom": headroom_ok,
                "samples_under_hard_floor": breaches,
                "usable": usable,
                "clears_full_body_radius": strict,
            }
        )
        print(
            f"  {episode['episode_id']:<28} off-navmesh {off_navmesh:>4}/{len(path)}  "
            f"brushing {pierces:>4}  breaching {breaches:>4}  worst clearance "
            f"{results[-1]['worst_clearance_m']}  "
            f"{'usable' if usable else 'NOT USABLE'}"
        )

    usable_count = sum(1 for r in results if r["usable"])
    strict_count = sum(1 for r in results if r["clears_full_body_radius"])
    worst = [r["worst_clearance_m"] for r in results
             if r["worst_clearance_m"] is not None]
    print(f"\n{usable_count}/{len(results)} paths usable "
          f"(on the navmesh, nothing under {args.hard_floor_m} m)")
    print(f"{strict_count}/{len(results)} also clear the full "
          f"{expected_clearance:.3f} m")
    if worst:
        print(f"worst clearance across paths: min {min(worst):.3f}  "
              f"median {float(np.median(worst)):.3f}  max {max(worst):.3f} m")
    if args.report:
        args.report.write_text(
            json.dumps(
                {
                    "schema": "avengine_route_legality_v1",
                    "asks": (
                        "whether the path could be walked, not whether it can be "
                        "heard. Occlusion is not a legality question"
                    ),
                    "bank": str(args.bank),
                    "body_radius_m": args.body_radius_m,
                    "navmesh_cell_size_m": args.navmesh_cell_size_m,
                    "expected_clearance_m": round(expected_clearance, 4),
                    "body_height_m": args.body_height_m,
                    "vertical_tolerance_m": args.vertical_tolerance_m,
                    "clearance_reference": (
                        "a navmesh rebuilt at 0.01 m radius, which traces real "
                        "geometry; the planning navmesh is already inset and a "
                        "shortest path hugs its boundary, so clearance measured "
                        "against it is always about zero"
                    ),
                    "emitter_height_m": emitter_height,
                    "usable": usable_count,
                    "clears_full_body_radius": strict_count,
                    "hard_floor_m": args.hard_floor_m,
                    "policy": (
                        "brushing a corner is acceptable and reported; a sample "
                        "off the navmesh or under the hard floor is not"
                    ),
                    "checked": len(results),
                    "episodes": results,
                },
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.report}")
    sim.close()
    if args.strict:
        return 0 if strict_count == len(results) else 1
    return 0 if usable_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
