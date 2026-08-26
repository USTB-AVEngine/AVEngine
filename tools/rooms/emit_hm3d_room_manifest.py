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
        best = None
        for index, start in enumerate(points):
            for end in points[index + 1 :]:
                path = shortest_path_class()
                path.requested_start = start
                path.requested_end = end
                if not pathfinder.find_path(path):
                    continue
                distance = float(path.geodesic_distance)
                if distance != distance or distance == float("inf"):
                    continue
                if best is None or distance > best[0]:
                    best = (distance, start, end)
        if best is None:
            raise SystemExit(
                f"{scene_glb.name}: no reachable pair among {samples} navigable "
                "samples, so this scene cannot make an M1 connectivity claim"
            )
        distance, start, end = best
        return {
            "pair": {
                "pair_id": "navmesh_widest_reachable_pair",
                "start_m": [float(value) for value in start],
                "end_m": [float(value) for value in end],
            },
            "evidence": {
                "measured_geodesic_distance_m": round(distance, 6),
                "navmesh": navmesh.name,
                "navigable_samples": samples,
                "sample_seed": seed,
                "selection": "largest finite geodesic distance among sampled pairs",
            },
        }
    finally:
        simulator.close()


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
        measured = derive_connectivity_pair(
            files["render"],
            files["navmesh"],
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
