#!/usr/bin/env python3
"""Precompute an apartment route bank from UE's own navigation system.

Question design needs routes that satisfy a quota, not a renderer: pick a
target speed, ask for routes whose arc length is speed x clip_seconds, then
check the question gates on the sampled frames.  Asking UE for a path costs a
SpearSim launch, so this tool asks once, in bulk, and writes a bank that later
runs read as plain JSON.

The bank stays camera-agnostic on purpose.  Side-of-view and visibility depend
on which camera a point uses, and the camera is a separate axis; joining the
two belongs in batch design, not here.  Everything stored is a property of the
route alone.

    python tools/m6x/build_apartment_route_bank.py \
        --spear-executable .../SpearSim.sh --samples 4000 --output bank.json

SPEAR call conventions this tool encodes (each one cost a wedged run to find):

* services live on ``instance.get_game()``, not on the instance;
* reads belong inside ``run_frame_transaction``; entering ``begin_frame()``
  and ``end_frame()`` in one ``with`` statement deadlocks the game thread;
* never hand ``get_static_class`` a guessed name -- a miss raises a modal
  engine assert and the process hangs until it is killed;
* ``navigation_data`` must be the ``RecastNavMesh`` actor; the
  ``AbstractNavData`` fallback answers every query with zeros;
* ``navigation_system`` must be the raw uint64 pointer held by the world's
  ``NavigationSystem`` property -- a class handle or the PropertyValue
  wrapper trips SP_ASSERT inside the service.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from avengine.backends.spear_ue.research_runtime import (
    launch_external_game_instance,
    run_frame_transaction,
)
from avengine.route_sampling import (
    arc_length_cm,
    max_turn_degrees,
    resample_route,
)

NATIVE_APARTMENT_MAP = "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000"
FRAME_COUNT = 75
CLIP_SECONDS = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spear-executable", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=4000,
                        help="navigable points to draw; pairs make the routes")
    parser.add_argument("--min-arc-cm", type=float, default=80.0,
                        help="drop degenerate routes shorter than this")
    parser.add_argument("--native-map", default=NATIVE_APARTMENT_MAP)
    parser.add_argument("--rpc-port", type=int, default=31700)
    parser.add_argument("--graphics-adapter", type=int, default=0)
    return parser.parse_args()


@contextlib.contextmanager
def _scratch_working_directory():
    """Run the engine from a scratch directory.

    SpearSim writes ``tmp/spear_instance_<port>/`` into the process working
    directory.  Doing that inside a checkout makes the retained-workspace
    tests stop skipping and fail, so keep the litter out of the tree.
    """
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="avengine-route-bank-") as scratch:
        os.chdir(scratch)
        try:
            yield Path(scratch)
        finally:
            os.chdir(previous)


def _navigation_handles(instance, game) -> tuple[object, int, str]:
    """Resolve the Recast navmesh actor and the world's navigation system."""
    unreal = game.unreal_service

    actors = run_frame_transaction(
        instance, apply=lambda: None,
        readback=lambda: unreal.find_actors_as_dict(),
    )
    navigation_actors = {
        name: handle for name, handle in actors.items() if "nav" in str(name).lower()
    }
    if not navigation_actors:
        raise RuntimeError(
            "the cooked map has no ANavigationData actor: it carries no built "
            "navigation data, so UE cannot answer path queries"
        )
    data_name = sorted(
        navigation_actors,
        key=lambda name: (0 if "recast" in name.lower() else 1, name),
    )[0]
    if "recast" not in data_name.lower():
        raise RuntimeError(
            f"only {data_name!r} is available; the AbstractNavData fallback "
            "returns zeroed points, so refusing to build a bank from it"
        )

    world = game.get_world()

    def _read_system():
        descriptor = unreal.resolve_property_for_object(
            uobject=world, property_name="NavigationSystem"
        )
        return unreal.get_property_value(property_desc=descriptor)

    value = run_frame_transaction(instance, apply=lambda: None, readback=_read_system)
    raw = getattr(value, "value", None)
    if raw is None:
        raise RuntimeError("world.NavigationSystem did not expose a pointer value")
    system = int(raw, 16) if isinstance(raw, str) else int(raw)
    return navigation_actors[data_name], system, data_name


def _query_routes(args, executable: Path):
    """Launch SpearSim once and pull raw paths out of UE's navigation system."""
    with _scratch_working_directory():
        instance = launch_external_game_instance(
            spear_executable=executable,
            native_map=args.native_map,
            frame_rate_hz=15,
            rpc_port=args.rpc_port,
            graphics_adapter=args.graphics_adapter,
        )
        try:
            game = instance.get_game()
            navigation = game.navigation_service
            navigation_data, navigation_system, data_name = _navigation_handles(
                instance, game
            )

            points = run_frame_transaction(
                instance,
                apply=lambda: None,
                readback=lambda: navigation.get_random_points(
                    navigation_data=navigation_data, num_points=args.samples
                ),
            )
            sampled = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            sampled = sampled[(sampled != 0).any(axis=1)]
            if sampled.shape[0] < 4:
                raise RuntimeError("navigation returned no usable points")

            half = sampled.shape[0] // 2
            starts, ends = sampled[:half], sampled[half : half * 2]
            raw_routes = run_frame_transaction(
                instance,
                apply=lambda: None,
                readback=lambda: navigation.find_paths(
                    navigation_system=navigation_system,
                    navigation_data=navigation_data,
                    num_paths=int(starts.shape[0]),
                    start_points=starts,
                    end_points=ends,
                ),
            )
        finally:
            try:
                instance.close(force=True)
            except Exception:
                pass
    return raw_routes, {
        "navigation_data_actor": data_name,
        "sampled_points": int(sampled.shape[0]),
        "requested_pairs": int(starts.shape[0]),
    }


def main() -> int:
    args = parse_args()
    executable = args.spear_executable.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    if args.samples < 4 or args.samples % 2:
        raise SystemExit("--samples must be an even number of at least 4")

    raw_routes, source = _query_routes(args, executable)

    routes = []
    for index, raw in enumerate(raw_routes):
        waypoints = np.asarray(raw, dtype=np.float64)
        if waypoints.ndim != 2 or waypoints.shape[0] < 2:
            continue
        points_list = [[round(float(v), 2) for v in point] for point in waypoints]
        length = arc_length_cm(points_list)
        if length < args.min_arc_cm:
            continue
        frames = resample_route(points_list, FRAME_COUNT)
        routes.append({
            "route_id": f"r{index:05d}",
            "waypoints_ue_cm": points_list,
            "waypoint_count": len(points_list),
            "arc_length_cm": round(length, 2),
            "implied_speed_mps": round(length / 100.0 / CLIP_SECONDS, 4),
            "max_turn_deg": round(max_turn_degrees(points_list), 2),
            "samples_ue_cm": [[round(p[0], 2), round(p[1], 2)] for p in frames],
            "bbox_ue_cm": [
                [round(float(waypoints[:, 0].min()), 2),
                 round(float(waypoints[:, 1].min()), 2)],
                [round(float(waypoints[:, 0].max()), 2),
                 round(float(waypoints[:, 1].max()), 2)],
            ],
        })

    speeds = np.array([route["implied_speed_mps"] for route in routes])
    bank = {
        "schema": "avengine_apartment_route_bank_v1",
        "claim_boundary": (
            "routes queried from UE's own navigation system on the production "
            "stage; camera-relative properties (side of view, visibility) are "
            "deliberately absent because the camera is a separate design axis"
        ),
        "research_only": True,
        "episode_counted": False,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "native_map": args.native_map,
            "spear_executable": str(executable),
            "navigation_system": "world.NavigationSystem (UNavigationSystemV1)",
            **source,
        },
        "frame_count": FRAME_COUNT,
        "clip_seconds": CLIP_SECONDS,
        "speed_identity": "implied_speed_mps = arc_length_cm / 100 / clip_seconds",
        "counts": {
            "routes": len(routes),
            "speed_min_mps": round(float(speeds.min()), 3) if routes else None,
            "speed_median_mps": round(float(np.median(speeds)), 3) if routes else None,
            "speed_max_mps": round(float(speeds.max()), 3) if routes else None,
            "with_turns": int(sum(1 for r in routes if r["waypoint_count"] > 2)),
        },
        "routes": routes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bank, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    print(json.dumps({"output": str(output), **bank["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
