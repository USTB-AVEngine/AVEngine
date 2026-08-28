#!/usr/bin/env python3
"""Emit an AVEngine room manifest for an HM3D scene directory.

HM3D ships 1000 scenes across five splits and 216 of them are annotated, so the
manifest cannot be a hand-written example per room. Everything a manifest needs
is derivable from the directory: the scene id is the directory name after the
numeric prefix, and the five files are named after it.

Two choices are worth stating because they are easy to get wrong later.

The render mesh declared here is the uncompressed ``<id>.glb`` and not
``<id>.basis.glb``. Both exist in every scene directory and the compressed one
is the default in HM3D's own dataset configs, but the Magnum build AVEngine
installs carries no BasisImporter: every texture fails to load, leaves an empty
handle, and the process takes SIGSEGV inside get_sensor_observations with no
traceback and an unflushed stdout. That failure looks exactly like a broken
renderer.

The semantic mesh is declared without a source-to-canonical rotation because it
needs none. HM3D's ``<id>.semantic.glb`` is the render mesh repainted - measured
on 00800-TEEsavR23oF the two agree to the last digit on all six bounding-box
coordinates and carry the same 395018 triangles - so acoustics and vision share
one geometry rather than two that have to be kept in step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LICENSE = "Habitat-Matterport 3D Dataset (HM3D) Terms of Use"
REDISTRIBUTION = "external_test_asset_not_committed"


def scene_files(scene_dir: Path) -> tuple[str, dict[str, Path]]:
    name = scene_dir.name
    if "-" not in name:
        raise SystemExit(f"{name} is not an HM3D scene directory (expected NNNNN-ID)")
    prefix, scene_id = name.split("-", 1)
    if not prefix.isdigit():
        raise SystemExit(f"{name} does not start with a numeric HM3D index")
    files = {
        "render": scene_dir / f"{scene_id}.glb",
        "semantic": scene_dir / f"{scene_id}.semantic.glb",
        "annotations": scene_dir / f"{scene_id}.semantic.txt",
        "navmesh": scene_dir / f"{scene_id}.basis.navmesh",
    }
    missing = [role for role, path in files.items() if not path.is_file()]
    if missing:
        raise SystemExit(
            f"{name} is missing {', '.join(sorted(missing))}; an unannotated "
            "scene carries no semantic mesh and cannot be compiled acoustically"
        )
    return scene_id, files


def declared(path: Path, root: Path, variable: str) -> str:
    return "${" + variable + "}/" + str(path.resolve().relative_to(root.resolve()))


def derive_connectivity_pair(
    scene_glb: Path,
    navmesh: Path,
    *,
    runtime_prefix: str,
    magnum_site: str,
    rlr_sdk_root: str,
    samples: int = 64,
    seed: int = 20260826,
    rooms: list | None = None,
) -> dict:
    """Measure one reachable pair across the scene with the shipped navmesh.

    A connectivity pair is a claim that two points are actually reachable from
    one another, so it is measured rather than guessed at from geometry. The
    pair chosen is the most distant reachable one found, because a pair of
    adjacent points would satisfy the contract while claiming almost nothing.

    An unloaded PathFinder segfaults on query instead of raising, so its loaded
    state is checked rather than assumed.
    """

    from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime

    dataset_root = None
    for parent in scene_glb.resolve().parents:
        if (parent / "scene_datasets").is_dir():
            dataset_root = parent
            break
    runtime = prepare_installed_habitat_runtime(
        runtime_prefix=runtime_prefix,
        magnum_python_site=magnum_site,
        rlr_sdk_root=rlr_sdk_root,
        mp3d_root=str(dataset_root) if dataset_root else None,
        allow_mp3d_environment=False,
    )
    hs = runtime.habitat_sim
    shortest_path_class = getattr(hs, "ShortestPath", None) or hs.nav.ShortestPath

    backend = hs.SimulatorConfiguration()
    backend.scene_id = str(scene_glb)
    backend.load_semantic_mesh = False
    backend.enable_physics = False
    simulator = hs.Simulator(
        hs.Configuration(backend, [hs.agent.AgentConfiguration()])
    )
    try:
        pathfinder = simulator.pathfinder
        if not pathfinder.is_loaded:
            pathfinder.load_nav_mesh(str(navmesh))
        if not pathfinder.is_loaded:
            raise SystemExit(f"{navmesh} did not load; no reachability to claim")
        pathfinder.seed(seed)
        points = [pathfinder.get_random_navigable_point() for _ in range(samples)]

        def farthest_from(start):
            found = None
            for end in points:
                path = shortest_path_class()
                path.requested_start = start
                path.requested_end = end
                if not pathfinder.find_path(path):
                    continue
                distance = float(path.geodesic_distance)
                if distance != distance or distance == float("inf"):
                    continue
                if found is None or distance > found[0]:
                    found = (distance, end)
            return found

        # Double sweep instead of all pairs. All pairs is samples-squared
        # find_path calls, which on a large scanned building measured out at
        # twenty-five CPU-minutes for one registration; two sweeps cost two
        # times samples calls and land on a pair at least half the true
        # diameter, which is far more than a connectivity claim needs.
        best = None
        for start in points[:4]:
            first = farthest_from(start)
            if first is None:
                continue
            second = farthest_from(first[1])
            candidate = (
                (second[0], first[1], second[1]) if second else (first[0], start, first[1])
            )
            if best is None or candidate[0] > best[0]:
                best = candidate
            break
        if best is None:
            raise SystemExit(
                f"{scene_glb.name}: no reachable pair among {samples} navigable "
                "samples, so this scene cannot make an M1 connectivity claim"
            )
        distance, start, end = best
        path = shortest_path_class()
        path.requested_start = start
        path.requested_end = end
        pathfinder.find_path(path)
        waypoints = [list(map(float, point)) for point in path.points]
        legality = _waypoint_legality(pathfinder, waypoints)
        topdowns = _connectivity_topdowns(pathfinder, waypoints, legality, rooms=rooms)
        return {
            "topdowns": topdowns,
            "pair": {
                "pair_id": "navmesh_widest_reachable_pair",
                "start_m": [float(value) for value in start],
                "end_m": [float(value) for value in end],
            },
            "evidence": {
                **legality,
                "measured_geodesic_distance_m": round(distance, 6),
                "navmesh": navmesh.name,
                "navigable_samples": samples,
                "sample_seed": seed,
                "selection": (
                    "double-sweep diameter approximation over the samples "
                    "(farthest point from a start, then farthest from that)"
                ),
            },
        }
    finally:
        simulator.close()


def _waypoint_legality(pathfinder, waypoints):
    """The machine's own reasons, stated as numbers a reader can dispute.

    A claim of connectivity is only as good as the route backing it, so every
    waypoint is asked two questions of the navmesh itself: are you navigable,
    and how far would snapping move you. These land in the evidence sidecar
    and on the drawing, because a picture that asks to be trusted should carry
    the argument for itself.
    """

    import numpy as np

    if not waypoints:
        return {
            "waypoint_count": 0,
            "waypoints_on_navmesh": 0,
            "worst_snap_distance_m": None,
            "waypoint_height_range_m": None,
        }
    on_mesh = sum(bool(pathfinder.is_navigable(point)) for point in waypoints)
    snaps = [
        float(
            np.linalg.norm(
                np.asarray(pathfinder.snap_point(point), dtype=float)
                - np.asarray(point, dtype=float)
            )
        )
        for point in waypoints
    ]
    heights = [point[1] for point in waypoints]
    return {
        "waypoint_count": len(waypoints),
        "waypoints_on_navmesh": on_mesh,
        "worst_snap_distance_m": round(max(snaps), 4),
        "waypoint_height_range_m": [round(min(heights), 2), round(max(heights), 2)],
    }


def inventory_rooms(scene_dir: Path, scene_id: str) -> dict:
    """List the scene's rooms from the annotation's own region column.

    HM3D tags every annotated instance with a region index - the fourth
    column this tool previously threw away - and a region is a room. The
    footprint here is the axis-aligned XZ box over the region's member faces
    and the floor height is the median of its floor-category faces, which is
    exactly enough to scope route banks and listeners to one room. Rooms are
    the benchmark's working unit; the acoustic package deliberately stays
    whole-house, because reverberation and sound leaking through doorways
    are properties of the building, not of the room.
    """

    import numpy as np
    from avengine.acoustics.gltf import (
        extract_triangle_scene_document,
        load_glb_bytes,
        triangle_vertex_colours,
    )
    from avengine.acoustics.semantic import _linear_to_srgb_bytes

    text_path = scene_dir / f"{scene_id}.semantic.txt"
    glb_path = scene_dir / f"{scene_id}.semantic.glb"
    colour_to_instance: dict[tuple[int, int, int], tuple[int, str, int]] = {}
    for line in text_path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 4 or not parts[0].strip().isdigit():
            continue
        colour = parts[1].strip().upper()
        if len(colour) != 6 or any(c not in "0123456789ABCDEF" for c in colour):
            continue
        key = (int(colour[0:2], 16), int(colour[2:4], 16), int(colour[4:6], 16))
        region_text = parts[3].strip()
        region = int(region_text) if region_text.lstrip("-").isdigit() else -1
        colour_to_instance[key] = (
            int(parts[0]),
            parts[2].strip().strip('"').lower(),
            region,
        )

    document = load_glb_bytes(glb_path.read_bytes(), source_path=str(glb_path))
    scene = extract_triangle_scene_document(document)
    linear, _mixed = triangle_vertex_colours(document, scene)
    face_colours = _linear_to_srgb_bytes(linear)
    corners = scene.vertices[scene.triangles.astype(np.int64)].astype(np.float64)
    # The raw semantic GLB is Z-up; everything downstream of this tool -
    # navmesh, route banks, listener poses - lives in Habitat's +Y-up frame.
    # Same rotation the acoustic compiler applies: x, y, z -> x, z, -y.
    # Getting this wrong produced a "floor" 0.45 m thick and 7.8 m tall.
    corners = np.stack(
        (corners[..., 0], corners[..., 2], -corners[..., 1]), axis=-1
    )
    areas = (
        np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            axis=1,
        )
        / 2.0
    )

    rooms: dict[int, dict] = {}
    for index, colour in enumerate(map(tuple, face_colours)):
        entry = colour_to_instance.get(colour)
        if entry is None:
            continue
        _instance, category, region = entry
        room = rooms.setdefault(
            region,
            {
                "min_x": np.inf, "max_x": -np.inf,
                "min_z": np.inf, "max_z": -np.inf,
                "floor_ys": [], "floor_area": 0.0,
                "categories": {}, "instances": set(),
            },
        )
        face = corners[index]
        room["min_x"] = min(room["min_x"], float(face[:, 0].min()))
        room["max_x"] = max(room["max_x"], float(face[:, 0].max()))
        room["min_z"] = min(room["min_z"], float(face[:, 2].min()))
        room["max_z"] = max(room["max_z"], float(face[:, 2].max()))
        room["instances"].add(entry[0])
        room["categories"][category] = room["categories"].get(category, 0) + 1
        if category in ("floor", "carpet", "rug", "flooring"):
            room["floor_ys"].append(float(face[:, 1].mean()))
            room["floor_area"] += float(areas[index])

    records = []
    for region, room in sorted(rooms.items()):
        if region < 0 or not np.isfinite(room["min_x"]):
            continue
        floor_y = float(np.median(room["floor_ys"])) if room["floor_ys"] else None
        top = sorted(room["categories"].items(), key=lambda kv: -kv[1])[:6]
        records.append(
            {
                "region_id": region,
                "instance_count": len(room["instances"]),
                "bbox_xz_m": [
                    [round(room["min_x"], 3), round(room["min_z"], 3)],
                    [round(room["max_x"], 3), round(room["max_z"], 3)],
                ],
                "extent_m": [
                    round(room["max_x"] - room["min_x"], 2),
                    round(room["max_z"] - room["min_z"], 2),
                ],
                "floor_y_m": None if floor_y is None else round(floor_y, 3),
                "floor_area_m2": round(room["floor_area"], 2),
                "top_categories": [name for name, _count in top],
            }
        )
    return {
        "schema": "avengine_hm3d_room_inventory_v1",
        "scene_id": scene_id,
        "authority": (
            "region column of the HM3D semantic annotations; footprints are "
            "axis-aligned boxes over member faces, floor heights the median "
            "of floor-category faces"
        ),
        "room_count": len(records),
        "rooms": records,
    }


def _connectivity_topdowns(pathfinder, path_points, legality, rooms=None):
    """Slice the route by floor, and never draw a segment where it is not.

    The first version projected the whole polyline onto the endpoint floors,
    and on a route that crosses twenty metres of stairwell that painted most
    of the line outside the very region it was supposed to be judged against -
    a reader correctly called it broken. Now each slice draws solid only the
    segments whose both ends live on that floor; the rest appears as a thin
    trace so the eye can follow where the route leaves for another storey,
    and the caption on the image says which is which and what the navmesh
    itself measured.
    """

    from PIL import Image, ImageDraw

    if not path_points:
        return []
    meters_per_pixel = 0.05
    floor_band = 0.75
    lower, _upper = pathfinder.get_bounds()

    def to_pixel(point):
        return (
            (point[0] - float(lower[0])) / meters_per_pixel,
            (point[2] - float(lower[2])) / meters_per_pixel,
        )

    # Endpoint floors first, then greedily the floor covering the most
    # not-yet-covered waypoints, until at least nine in ten waypoints appear
    # solid on some slice. A fixed share threshold left the stairwell middle
    # of a 28-level route invisible on every image - drawn, but never
    # judgeable, which a reviewer rightly refused to accept.
    heights = []

    def add_height(value):
        if all(abs(value - seen) > floor_band for seen in heights):
            heights.append(value)

    add_height(path_points[0][1])
    add_height(path_points[-1][1])

    def covered(point):
        return any(abs(point[1] - height) <= floor_band for height in heights)

    while len(heights) < 6:
        uncovered = [point for point in path_points if not covered(point)]
        if len(uncovered) <= 0.1 * len(path_points):
            break
        candidates = sorted({round(point[1] * 4) / 4 for point in uncovered})
        best = max(
            candidates,
            key=lambda h: sum(1 for point in uncovered if abs(point[1] - h) <= floor_band),
        )
        add_height(best)

    caption = (
        f"waypoints on navmesh {legality['waypoints_on_navmesh']}"
        f"/{legality['waypoint_count']} · worst snap "
        f"{legality['worst_snap_distance_m']} m · heights "
        f"{legality['waypoint_height_range_m'][0]} to "
        f"{legality['waypoint_height_range_m'][1]} m"
    )

    images = []
    for height in heights:
        import numpy as np

        navigable = np.asarray(
            pathfinder.get_topdown_view(meters_per_pixel, float(height))
        )
        canvas = np.full((*navigable.shape, 3), 245, dtype=np.uint8)
        canvas[navigable] = (203, 213, 225)
        pad = 34
        image = Image.new(
            "RGB", (canvas.shape[1], canvas.shape[0] + pad), (245, 245, 245)
        )
        image.paste(Image.fromarray(canvas), (0, pad))
        draw = ImageDraw.Draw(image)
        draw.text(
            (6, 4),
            f"floor y={height:+.2f}  solid=this floor, thin=other floors",
            fill=(60, 60, 60),
        )
        draw.text((6, 17), caption, fill=(120, 120, 120))

        def pixel(point):
            x, z = to_pixel(point)
            return (x, z + pad)

        for first, second in zip(path_points, path_points[1:]):
            on_floor = (
                abs(first[1] - height) <= floor_band
                and abs(second[1] - height) <= floor_band
            )
            draw.line(
                [pixel(first), pixel(second)],
                fill=(83, 74, 183) if on_floor else (190, 188, 214),
                width=4 if on_floor else 1,
            )
        for room in rooms or []:
            if room.get("floor_y_m") is None or abs(room["floor_y_m"] - height) > 1.0:
                continue
            (x0, z0), (x1, z1) = room["bbox_xz_m"]
            left, top_edge = pixel((x0, 0, z0))
            right, bottom_edge = pixel((x1, 0, z1))
            draw.rectangle(
                (left, top_edge, right, bottom_edge), outline=(180, 120, 40), width=2
            )
            draw.text(
                (left + 3, top_edge + 2), f"R{room['region_id']}", fill=(150, 95, 20)
            )
        radius = 6
        for point, colour in (
            (path_points[0], (31, 107, 74)),
            (path_points[-1], (163, 51, 51)),
        ):
            x, z = pixel(point)
            on_this_floor = abs(point[1] - height) <= floor_band
            draw.ellipse(
                (x - radius, z - radius, x + radius, z + radius),
                fill=colour if on_this_floor else None,
                outline=colour if on_this_floor else (150, 150, 150),
                width=2,
            )
        images.append((float(height), image))
    return images


def build_manifest(
    scene_dir: Path,
    *,
    hm3d_root: Path,
    split: str,
    variable: str,
    connectivity_pairs: list[dict],
) -> dict:
    scene_id, files = scene_files(scene_dir)
    index = scene_dir.name.split("-", 1)[0]
    room_id = f"hm3d_{split}_{index}_{scene_id}"
    dataset_config = hm3d_root / f"hm3d_annotated_{split}_basis.scene_dataset_config.json"
    assets = [
        ("render_surface_mesh", files["render"]),
        ("semantic_surface_mesh", files["semantic"]),
        ("semantic_descriptor", files["annotations"]),
        ("navmesh", files["navmesh"]),
    ]
    if dataset_config.is_file():
        assets.append(("scene_dataset_config", dataset_config))
    return {
        "schema": "avengine_room_package_v1",
        "room_id": room_id,
        "room_kind": "habitat_native",
        "geometry_representation": "real_surface_mesh",
        "coordinate_system": {
            "handedness": "right",
            "up_axis": "+Y",
            "forward_axis": "-Z",
            "linear_unit": "meter",
            "quaternion_order": "xyzw",
        },
        "scene": {
            "scene_id_kind": "path",
            "scene_id": declared(files["render"], hm3d_root, variable),
            "dataset_config_path": (
                declared(dataset_config, hm3d_root, variable)
                if dataset_config.is_file()
                else declared(files["render"], hm3d_root, variable)
            ),
            "navmesh_path": declared(files["navmesh"], hm3d_root, variable),
            "navmesh_policy": "load_declared",
            "load_semantic_mesh": True,
            "enable_physics": False,
        },
        "assets": [
            {
                "role": role,
                "path": declared(path, hm3d_root, variable),
                "license": LICENSE,
                "redistribution": REDISTRIBUTION,
            }
            for role, path in assets
        ],
        "semantics": {
            "interpretation": (
                "HM3D semantic instance identity is painted into the semantic "
                "GLB's COLOR_0 attribute and keyed by the paired .semantic.txt "
                "listing, whose colours are sRGB while COLOR_0 is linear"
            )
        },
        "navigation": {
            "agent_height_m": 1.5,
            "agent_radius_m": 0.1,
            "include_static_objects": False,
        },
        "openings": [],
        "connectivity_pairs": connectivity_pairs,
        "ray_checks": [],
        "acoustics": {
            "status": "deferred_to_m3",
            "reason": "M1 validates visual geometry and coordinate contracts only",
        },
        "provenance": {
            "source": f"HM3D 1.0 {split} split, scene {scene_dir.name}",
            "source_revision": "hm3d-1.0 glb + semantic-annots-v0.2",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", required=True, type=Path, nargs="+")
    parser.add_argument("--hm3d-root", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--path-variable",
        default="AVENGINE_HM3D_ROOT",
        help="environment variable the declared paths are written against",
    )
    parser.add_argument("--runtime-prefix", required=True)
    parser.add_argument("--magnum-site", required=True)
    parser.add_argument("--rlr-sdk-root", required=True)
    parser.add_argument("--connectivity-samples", type=int, default=64)
    parser.add_argument("--connectivity-seed", type=int, default=20260826)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for scene_dir in args.scene_dir:
        scene_id, files = scene_files(scene_dir)
        inventory = inventory_rooms(scene_dir, scene_id)
        measured = derive_connectivity_pair(
            files["render"],
            files["navmesh"],
            rooms=inventory["rooms"],
            runtime_prefix=args.runtime_prefix,
            magnum_site=args.magnum_site,
            rlr_sdk_root=args.rlr_sdk_root,
            samples=args.connectivity_samples,
            seed=args.connectivity_seed,
        )
        manifest = build_manifest(
            scene_dir,
            hm3d_root=args.hm3d_root,
            split=args.split,
            variable=args.path_variable,
            connectivity_pairs=[measured["pair"]],
        )
        destination = args.output_dir / manifest["room_id"] / "room_manifest.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for height, image in measured.pop("topdowns", []):
            sign = "+" if height >= 0 else "-"
            image.save(
                destination.parent
                / f"connectivity_topdown_y{sign}{abs(height):.2f}.png"
            )
        (destination.parent / "rooms.json").write_text(
            json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (destination.parent / "connectivity_measurement.json").write_text(
            json.dumps(
                {
                    "room_id": manifest["room_id"],
                    "pair_id": measured["pair"]["pair_id"],
                    **measured["evidence"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        written.append(str(destination))
    print(json.dumps({"written": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
