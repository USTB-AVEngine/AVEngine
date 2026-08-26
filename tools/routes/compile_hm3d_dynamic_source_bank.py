"""Ask whether a moving sound source can find legal routes in an HM3D scene.

The feasibility and trajectory machinery in avengine.routes was built for the
fixed-room M6.x canary, where one floor plane and an explicit set of rigid
collision boxes describe the room. HM3D breaks the first of those assumptions
and removes the second, so this asks the same question against it without
loosening any of the gates:

  * HM3D scenes are houses, and a house has storeys. get_topdown_view slices
    the navmesh at one height, so a single call silently answers for whichever
    floor the caller guessed. Floors are found here by sampling navigable
    points and clustering their heights, then every floor is compiled and
    reported separately. A scene where only the ground floor yields routes is a
    real answer; a scene where the caller happened to slice an empty height and
    concluded "no routes" is not.
  * HM3D ships a navmesh computed over the full scene, furniture included, so
    there is no separate rigid-obstacle list to intersect. That makes the
    navmesh the single authority here, which is worth stating rather than
    leaving implied: a source centre is legal because the navmesh says the
    floor under it is walkable and clear, not because anything checked the
    furniture at source height.

Where the clearance is enforced matters, and getting it wrong looks like
success. The gates as written ask a geodesic route to hold a minimum distance
from the nearest navmesh obstacle at every frame. A shortest path cannot: going
around a corner optimally means touching that corner, so its clearance is zero
by construction. Measured on this scene, every route that bent failed with a
worst clearance of 0.000 to 0.005 m while the only survivors were the straight
ones, which sat in open floor at 0.14 and 0.28. The bank filled up, reported
success, and contained nothing but straight lines - no source ever went through
a doorway.

So the clearance lives in the navmesh instead. The navmesh is recomputed with
an agent radius equal to the source's body radius, which insets it from the
real geometry before any route is searched, and the post-hoc distance gate is
then released to zero because it would only re-reject the same corners. The
clearance is still verified, but against the shipped navmesh held aside as a
reference: routing at radius 0.20 yields a measured 0.10 m against it, radius
0.30 yields 0.20, because HM3D ships its navmesh already inset by 0.10. Bent
routes survive that - 56 percent of in-band routes at radius 0.20.

Nothing about audibility is claimed. A legal route can still be occluded from a
given receiver, which is a per-placement question the acoustic gate answers
separately.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def find_floors(pathfinder, samples: int, bin_m: float, minimum_share: float):
    """Find storeys as dense modes in the navigable height distribution.

    Splitting sorted heights on vertical gaps looks reasonable and is wrong in
    exactly the houses this is for: a staircase is navigable, so it lays down a
    thin continuum of heights between two floors and there is no gap to split
    on. Both storeys then merge into one cluster whose median lands on
    whichever floor had more area, and the other storey is dropped without a
    word - a silent loss of half the scene, reported as one floor.

    A floor is dense and a staircase is sparse, so the modes are what to look
    for. On the scene this was built against the two floors hold 61 and 30
    percent of samples while every staircase bin holds under 5.
    """

    heights = []
    for _ in range(samples):
        point = pathfinder.get_random_navigable_point()
        if np.all(np.isfinite(point)):
            heights.append(float(point[1]))
    if not heights:
        return []
    heights = np.asarray(heights, dtype=float)
    edges = np.arange(heights.min(), heights.max() + 2.0 * bin_m, bin_m)
    counts, edges = np.histogram(heights, bins=edges)
    dense = counts >= minimum_share * len(heights)

    floors = []
    index = 0
    while index < len(dense):
        if not dense[index]:
            index += 1
            continue
        start = index
        while index < len(dense) and dense[index]:
            index += 1
        # Adjacent dense bins are one floor whose surface straddles a boundary.
        inside = (heights >= edges[start]) & (heights < edges[index])
        if not inside.any():
            continue
        floors.append(
            {
                "height_m": round(float(np.median(heights[inside])), 4),
                "navigable_share": round(float(inside.mean()), 4),
                "samples": int(inside.sum()),
            }
        )
    return floors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-prefix", required=True)
    parser.add_argument("--magnum-site", required=True)
    parser.add_argument("--rlr-sdk-root", required=True)
    parser.add_argument("--scene", required=True, action="append")
    parser.add_argument("--navmesh", action="append", default=[])
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--topdown-dir",
        type=Path,
        help="write the repository's own feasibility/trajectory diagnostic per floor",
    )
    parser.add_argument("--source1-height-m", type=float, default=1.2)
    parser.add_argument("--source2-height-m", type=float, default=0.35)
    parser.add_argument("--episodes-per-motion-case", type=int, default=8)
    parser.add_argument("--minimum-route-distance-m", type=float, default=3.5)
    parser.add_argument("--maximum-route-distance-m", type=float, default=5.5)
    parser.add_argument("--meters-per-pixel", type=float, default=0.05)
    parser.add_argument(
        "--source-body-radius-m",
        type=float,
        default=0.20,
        help=(
            "half-width of the moving source. The navmesh is recomputed with "
            "this as the agent radius, which is where clearance is enforced; a "
            "post-hoc distance gate cannot do it because a shortest path "
            "touches the corners it turns"
        ),
    )
    parser.add_argument("--source-body-height-m", type=float, default=1.5)
    parser.add_argument(
        "--no-reground",
        action="store_true",
        help=(
            "keep every path sample pinned to one floor height, as the library "
            "does for a flat authored room. Left on for HM3D it puts most "
            "samples off the navmesh vertically; the flag exists to measure that"
        ),
    )
    parser.add_argument(
        "--shipped-navmesh-inset-m",
        type=float,
        default=0.10,
        help="HM3D builds its navmesh at this agent radius; the expected "
        "clearance against it is the source radius minus this",
    )
    parser.add_argument("--floor-samples", type=int, default=600)
    parser.add_argument(
        "--floor-bin-m",
        type=float,
        default=0.25,
        help="height bin width used to find floors as dense modes",
    )
    parser.add_argument("--floor-minimum-share", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    for scene in args.scene:
        if Path(scene).name.endswith(".basis.glb"):
            raise SystemExit(
                f"{scene}: refusing a *.basis.glb. This Magnum site has no "
                "BasisImporter, so textures fail to load and the first render "
                "segfaults. Use the uncompressed sibling glb."
            )

    from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime
    from avengine.routes.geometry import build_runtime_obstacle_map
    from avengine.routes.room_feasibility import (
        RoomFeasibilityCompiler,
        RoomFeasibilityError,
        TrajectoryBankBuilder,
    )

    dataset_root = None
    for parent in Path(args.scene[0]).resolve().parents:
        if (parent / "scene_datasets").is_dir():
            dataset_root = parent
            break
    if dataset_root is None:
        raise SystemExit("no scene_datasets directory above --scene")

    runtime = prepare_installed_habitat_runtime(
        runtime_prefix=args.runtime_prefix,
        magnum_python_site=args.magnum_site,
        rlr_sdk_root=args.rlr_sdk_root,
        mp3d_root=str(dataset_root),
        allow_mp3d_environment=False,
    )
    hs = runtime.habitat_sim
    mn = runtime.magnum
    shortest_path_class = getattr(hs, "ShortestPath", None) or hs.nav.ShortestPath

    navmeshes = dict(zip(args.scene, args.navmesh))
    report = {
        "schema": "avengine_hm3d_dynamic_source_bank_v1",
        "runtime_prefix": str(runtime.prefix),
        "authority": (
            "HM3D ships a navmesh built over the full scene including "
            "furniture, so the navmesh is the only obstacle authority here and "
            "no rigid collision boxes are intersected"
        ),
        "legality": (
            "floor snap within tolerance, minimum navmesh clearance at every "
            "frame, and a geodesic route inside the distance band between two "
            "points of one connected feasible component. Audibility from a "
            "receiver is not claimed"
        ),
        "source_center_heights_m": {
            "source1": args.source1_height_m,
            "source2": args.source2_height_m,
        },
        "route_distance_band_m": [
            args.minimum_route_distance_m,
            args.maximum_route_distance_m,
        ],
        "regrounded_per_sample": not args.no_reground,
        "scenes": [],
    }

    total_floors = 0
    floors_with_routes = 0
    for scene in args.scene:
        name = Path(scene).parent.name or Path(scene).stem
        backend = hs.SimulatorConfiguration()
        backend.scene_id = scene
        backend.load_semantic_mesh = False
        backend.enable_physics = True
        if runtime.physics_config_path:
            backend.physics_config_file = str(runtime.physics_config_path)
        simulator = hs.Simulator(
            hs.Configuration(backend, [hs.agent.AgentConfiguration()])
        )

        pathfinder = simulator.pathfinder
        navmesh_path = navmeshes.get(scene)
        # A bare glb carries no scene dataset config, so habitat has nothing to
        # point the PathFinder at. Querying an unloaded PathFinder segfaults
        # instead of raising, so this is checked rather than assumed.
        if not pathfinder.is_loaded and navmesh_path:
            pathfinder.load_nav_mesh(navmesh_path)

        # The shipped navmesh, held aside untouched, is the clearance reference.
        reference = None
        if navmesh_path:
            reference = hs.nav.PathFinder()
            reference.load_nav_mesh(navmesh_path)
            if not reference.is_loaded:
                reference = None

        settings = hs.NavMeshSettings()
        settings.set_defaults()
        settings.agent_radius = args.source_body_radius_m
        settings.agent_height = args.source_body_height_m
        inset_ok = bool(simulator.recompute_navmesh(pathfinder, settings))

        scene_record = {
            "scene": scene,
            "navmesh_loaded": bool(pathfinder.is_loaded),
            "navmesh_recomputed_at_radius_m": (
                args.source_body_radius_m if inset_ok else None
            ),
            "clearance_reference_available": reference is not None,
            "expected_clearance_m": round(
                max(args.source_body_radius_m - args.shipped_navmesh_inset_m, 0.0), 4
            ),
            "floors": [],
        }
        if not inset_ok:
            scene_record["warning"] = (
                "recompute_navmesh failed, so routes were searched on the "
                "shipped navmesh and clearance is not enforced"
            )
            print("   warning: recompute_navmesh failed")
        if not pathfinder.is_loaded:
            scene_record["verdict"] = "no_navmesh"
            report["scenes"].append(scene_record)
            print(f"{name:<24} no navmesh")
            simulator.close()
            continue

        floors = find_floors(
            pathfinder, args.floor_samples, args.floor_bin_m, args.floor_minimum_share
        )
        scene_record["floors_found"] = len(floors)
        print(f"{name:<24} navmesh ok, {len(floors)} floor(s)")

        for floor in floors:
            total_floors += 1
            entry = dict(floor)
            try:
                obstacle_map = build_runtime_obstacle_map(
                    pathfinder,
                    simulator.get_rigid_object_manager(),
                    mn,
                    floor_height_m=floor["height_m"],
                    meters_per_pixel=args.meters_per_pixel,
                )
            except Exception as error:
                entry["verdict"] = "obstacle_map_failed"
                entry["reason"] = f"{type(error).__name__}: {error}"
                scene_record["floors"].append(entry)
                print(f"   y={floor['height_m']:+7.3f}  {entry['reason'][:70]}")
                continue

            compiler = RoomFeasibilityCompiler(obstacle_map)
            try:
                # Released to zero deliberately. The inset navmesh already
                # holds the source's body clear of real geometry; leaving this
                # at its default would re-reject exactly the corners a route
                # has to turn.
                regions = {
                    slot: compiler.compile(
                        source_center_height_m=height,
                        minimum_navmesh_clearance_m=0.0,
                    )
                    for slot, height in (
                        ("source1", args.source1_height_m),
                        ("source2", args.source2_height_m),
                    )
                }
            except RoomFeasibilityError as error:
                entry["verdict"] = "no_feasible_region"
                entry["reason"] = str(error)
                scene_record["floors"].append(entry)
                print(f"   y={floor['height_m']:+7.3f}  no feasible region: {error}")
                continue

            entry["feasible_components"] = {
                slot: len(index.components) for slot, index in regions.items()
            }
            entry["feasible_area_m2"] = {
                slot: round(
                    float(np.count_nonzero(index.feasible_mask))
                    * index.pixel_size_x_m
                    * index.pixel_size_z_m,
                    3,
                )
                for slot, index in regions.items()
            }
            entry["feasible_samples"] = {
                slot: int(len(index.sample_pixels_rc))
                for slot, index in regions.items()
            }

            def reground(roots, _pathfinder=pathfinder):
                """Put every sample on the floor that is actually under it.

                The library pins a whole path to one declared floor height,
                which is right for an authored room with a flat floor and wrong
                for a scanned storey. On these scenes a single storey's navmesh
                spans about half a metre, so pinning leaves the source hanging
                in the air or sunk into the floor - measured against a 0.05 m
                vertical tolerance, up to 93 percent of samples on some floors
                were off the navmesh, while the default 0.5 m tolerance hid all
                of it behind a clean zero.

                Only the height is taken from the snap. The horizontal position
                has already been through the feasible-region sampling and must
                not be moved by a grounding step.
                """

                grounded = {}
                for slot, path in roots.items():
                    samples = np.array(path, dtype=np.float64)
                    for index, sample in enumerate(samples):
                        snapped = np.asarray(
                            _pathfinder.snap_point(sample), dtype=np.float64
                        )
                        if np.all(np.isfinite(snapped)):
                            samples[index, 1] = snapped[1]
                    grounded[slot] = samples
                return grounded

            builder = TrajectoryBankBuilder(
                pathfinder=pathfinder,
                obstacle_map=obstacle_map,
                region_by_source=regions,
                shortest_path_factory=shortest_path_class,
                source_path_materializer=None if args.no_reground else reground,
            )
            try:
                bank = builder.build(
                    episodes_per_motion_case=args.episodes_per_motion_case,
                    seed=args.seed,
                    minimum_route_distance_m=args.minimum_route_distance_m,
                    maximum_route_distance_m=args.maximum_route_distance_m,
                )
            except RoomFeasibilityError as error:
                entry["verdict"] = "no_routes"
                entry["reason"] = str(error)
                scene_record["floors"].append(entry)
                print(f"   y={floor['height_m']:+7.3f}  no routes: {error}")
                continue

            cases = Counter(episode.motion_case for episode in bank.episodes)
            # Static slots have to be excluded from the length statistics. Half
            # the paths in a bank hold still by design, and mixing them in puts
            # the median below the requested minimum, which reads as the
            # distance band having been ignored.
            moving_lengths = []
            detours = []
            offmesh = 0
            worst_clearance = float("inf")
            clearances = []
            checked = 0
            off_raster = 0
            off_raster_runs = []
            mask = regions["source1"].feasible_mask
            map_bounds = np.asarray(obstacle_map.bounds_m, dtype=float)
            cell_x = (map_bounds[1][0] - map_bounds[0][0]) / mask.shape[1]
            cell_z = (map_bounds[1][2] - map_bounds[0][2]) / mask.shape[0]
            for episode in bank.episodes:
                for slot, path in episode.source_center_paths_m.items():
                    points = np.asarray(path)
                    planar = points[:, (0, 2)]
                    length = float(
                        np.linalg.norm(np.diff(planar, axis=0), axis=1).sum()
                    )
                    chord = float(np.linalg.norm(planar[-1] - planar[0]))
                    if chord < 1.0e-6:
                        continue  # a static slot
                    moving_lengths.append(length)
                    detours.append(length / chord)
                    # Independent re-check. The builder already gates on this,
                    # so agreement is the point: a route is being called legal
                    # by something other than the code that produced it.
                    # source_center_paths_m, not source_root_paths_m. The
                    # roots keep whatever the sampler produced; the centres are
                    # what the materializer returned and what the gate ran on,
                    # so the roots are the wrong thing to verify. With no
                    # materializer the two are equal, which is exactly why
                    # reading the wrong one looked harmless.
                    root = np.asarray(episode.source_center_paths_m[slot])
                    run = 0
                    for sample in root:
                        checked += 1
                        # Against the raster the feasibility map is drawn
                        # from. This is a proximity measure, not a gate: the
                        # raster tests cell centres, so a legal point beside a
                        # wall falls in an infeasible cell. See off_raster_note.
                        row = int((sample[2] - map_bounds[0][2]) / cell_z)
                        col = int((sample[0] - map_bounds[0][0]) / cell_x)
                        inside_raster = (
                            0 <= row < mask.shape[0]
                            and 0 <= col < mask.shape[1]
                            and bool(mask[row, col])
                        )
                        if inside_raster:
                            if run:
                                off_raster_runs.append(run)
                                run = 0
                        else:
                            off_raster += 1
                            run += 1
                        # A 0.5 m vertical slack is the default and it is too
                        # loose here: in a house it can match a polygon on the
                        # storey above or below, so every sample passes and the
                        # check means nothing.
                        if not pathfinder.is_navigable(sample, 0.05):
                            offmesh += 1
                            continue
                        if reference is None:
                            continue
                        # Measured on the shipped navmesh, not the inset one it
                        # was routed on, so the number means distance from real
                        # geometry rather than distance from its own boundary.
                        clearance = float(
                            reference.distance_to_closest_obstacle(sample, 10.0)
                        )
                        clearances.append(clearance)
                        worst_clearance = min(worst_clearance, clearance)
            entry["episodes"] = len(bank.episodes)
            entry["motion_cases"] = dict(sorted(cases.items()))
            entry["moving_paths"] = len(moving_lengths)
            entry["recheck"] = {
                "samples_checked": checked,
                "samples_off_navmesh": offmesh,
                "worst_navmesh_clearance_m": (
                    None if worst_clearance == float("inf")
                    else round(worst_clearance, 4)
                ),
                "median_navmesh_clearance_m": (
                    round(float(np.median(clearances)), 4) if clearances else None
                ),
                "samples_off_feasible_raster": off_raster,
                "off_raster_longest_run": (
                    max(off_raster_runs) if off_raster_runs else 0
                ),
                "off_raster_note": (
                    "not an illegality. get_topdown_view samples cell centres, "
                    "so a point can be navigable while the centre of its cell "
                    "is not, and a shortest path hugs the navmesh boundary for "
                    "long stretches - which is why the runs are long. Every "
                    "such point lands within one cell of feasible space (max "
                    "0.075 m against a 0.071 m cell diagonal), all of them "
                    "answer yes to is_navigable at the declared floor height, "
                    "and refining the raster from 0.05 to 0.01 m drops the "
                    "share from 5.67 to 1.38 percent. Read this as how closely "
                    "the routes run to walls, not as a gate"
                ),
                "navigable_vertical_slack_m": 0.05,
                "clearance_note": (
                    "measured against the shipped navmesh. The worst value sits "
                    "a little under the inset because resampling a geodesic by "
                    "arc length cuts its corners slightly; the median is what "
                    "the inset buys"
                ),
            }
            if moving_lengths:
                bent = [value for value in detours if value > 1.02]
                # The number that says whether these are routes or just chords.
                entry["bent_route_share"] = round(len(bent) / len(detours), 3)
                entry["route_length_m"] = {
                    "minimum": round(float(min(moving_lengths)), 3),
                    "median": round(float(np.median(moving_lengths)), 3),
                    "maximum": round(float(max(moving_lengths)), 3),
                }
                entry["geodesic_detour_ratio"] = {
                    "median": round(float(np.median(detours)), 3),
                    "maximum": round(float(max(detours)), 3),
                }
            entry["verdict"] = "routes_found" if bank.episodes else "no_routes"
            if bank.episodes:
                floors_with_routes += 1
            scene_record["floors"].append(entry)
            if args.topdown_dir and bank.episodes:
                from avengine.routes.feasibility_topdown import (
                    render_feasibility_topdown,
                )
                from PIL import Image

                args.topdown_dir.mkdir(parents=True, exist_ok=True)
                listener = np.asarray(
                    regions["source1"].pixel_to_world(
                        regions["source1"].sample_pixels_rc[
                            len(regions["source1"].sample_pixels_rc) // 2
                        ],
                        height_m=floor["height_m"],
                    ),
                    dtype=np.float64,
                )
                panel = render_feasibility_topdown(
                    regions,
                    bank,
                    listener_position_m=listener,
                    listener_yaw_deg=0.0,
                    camera_hfov_degrees=70.0,
                    room_label=f"{name} floor y={floor['height_m']:+.3f}",
                    navigation_authority_label=(
                        "HM3D shipped navmesh, recomputed at agent radius "
                        f"{args.source_body_radius_m:.2f} m"
                    ),
                )
                out = (
                    args.topdown_dir
                    / f"{name}_y{floor['height_m']:+.3f}.topdown.png"
                )
                Image.fromarray(np.asarray(panel, dtype=np.uint8)).save(out)
                entry["topdown"] = str(out)

            band = entry.get("route_length_m", {})
            print(
                f"   y={floor['height_m']:+7.3f}  "
                f"share {floor['navigable_share']:5.2f}  "
                f"area {entry['feasible_area_m2']['source1']:7.1f} m2  "
                f"components {entry['feasible_components']['source1']:>3}  "
                f"episodes {entry['episodes']:>3}  "
                f"moving {entry['moving_paths']:>3}  "
                f"route {band.get('minimum', 0):.2f}-{band.get('maximum', 0):.2f} m  "
                f"bent {100 * entry.get('bent_route_share', 0):3.0f}%  "
                f"detour x{entry.get('geodesic_detour_ratio', {}).get('median', 0):.2f}  "
                f"clear {entry['recheck']['median_navmesh_clearance_m']}"
                f"/{entry['recheck']['worst_navmesh_clearance_m']}  "
                f"off-navmesh {entry['recheck']['samples_off_navmesh']}"
                f"/{entry['recheck']['samples_checked']}  "
                f"off-raster {entry['recheck']['samples_off_feasible_raster']}"
                f"(run {entry['recheck']['off_raster_longest_run']})"
            )

        scene_record["verdict"] = (
            "routes_found"
            if any(f.get("verdict") == "routes_found" for f in scene_record["floors"])
            else "no_routes"
        )
        report["scenes"].append(scene_record)
        simulator.close()

    report["floors_examined"] = total_floors
    report["floors_with_routes"] = floors_with_routes
    scenes_ok = sum(s.get("verdict") == "routes_found" for s in report["scenes"])
    report["scenes_with_routes"] = scenes_ok
    print(
        f"\n{scenes_ok}/{len(report['scenes'])} scenes yielded routes; "
        f"{floors_with_routes}/{total_floors} floors did"
    )
    if args.report:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.report}")
    return 0 if scenes_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
