"""Prepare the real-surface UE apartment export as an M1 Habitat room package.

This is a packaging step, not a geometry conversion step.  It references the
audited UE GLB as both the render and collision surface, copies the two small
tracked semantic source markers, and writes the Habitat dataset configuration
plus AVEngine room/capture contracts needed by ``avengine m1``.

The script deliberately refuses the old 252-triangle actor-bounds export.  It
also cross-checks the UE export manifest, Blender mesh audit, and GLB bytes so
that a stale audit cannot accidentally bless a different scene.

Example::

    python tools/rooms/prepare_legacy_apartment.py \
      --scene-glb tmp/m1/legacy_apartment_export/scene.glb \
      --ue-manifest tmp/m1/legacy_apartment_export/ue_export_manifest.json \
      --mesh-audit tmp/m1/legacy_apartment_export/mesh_audit.json \
      --spear-root /path/to/clean/SPEAR \
      --output-dir tmp/m1/legacy_apartment_package
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import subprocess
from typing import Any


ROOM_ID = "legacy_ue_apartment_0000_v1"
SCENE_HANDLE = "legacy_apartment_0000"
KNOWN_AABB_TRIANGLE_COUNT = 252
SPEAR_MAP_PACKAGE = Path(
    "cpp/unreal_projects/SpearSim/Content/SPEAR/Scenes/"
    "apartment_0000/Maps/apartment_0000.umap"
)

# UE (X, Y, Z) cm -> glTF/Habitat (X, Z, Y) m.  These values preserve the
# approved legacy camera/listener and two source placements in the UE map.
CAMERA_LISTENER_WORLD_M = [-0.7, 1.471, 0.65]
RIG_NAVIGATION_FLOOR_M = [-0.7, 0.271, 0.65]
SHARED_LOCAL_SENSOR_M = [0.0, 0.0, 0.0]
FACE_NEGATIVE_X_XYZW = [0.0, 0.7071067811865475, 0.0, 0.7071067811865476]
SOURCE_POSITIONS_M = [
    [-2.88, 0.721, -0.75],
    [1.48, 0.721, 4.36],
]
SOURCE_FLOOR_POSITIONS_M = [
    [-2.88, 0.271, -0.75],
    [1.48, 0.271, 4.36],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrap an audited UE apartment GLB as an M1 Habitat package."
    )
    parser.add_argument("--scene-glb", required=True, help="Audited UE GLB export")
    parser.add_argument(
        "--ue-manifest", required=True, help="Manifest written by the UE exporter"
    )
    parser.add_argument(
        "--mesh-audit", required=True, help="Passing real-surface Blender audit"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Generated package root (normally tmp/)"
    )
    parser.add_argument(
        "--spear-root",
        required=True,
        help="SPEAR Git checkout containing the source apartment package",
    )
    parser.add_argument(
        "--marker-dir",
        help=(
            "Optional directory containing source_marker_0.glb and "
            "source_marker_1.glb; defaults to the tracked Blender M1 example"
        ),
    )
    return parser.parse_args()


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def spear_source_snapshot(
    root: Path, *, capture_phase: str = "before_ue_gltf_export"
) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"SPEAR checkout does not exist: {root}")

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"SPEAR git {' '.join(arguments)} failed: {result.stderr.strip()}"
            )
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    tracked_status = git("status", "--porcelain", "--untracked-files=no")
    require(not tracked_status, "SPEAR tracked files are dirty")
    map_package = (root / SPEAR_MAP_PACKAGE).resolve()
    require(
        map_package.is_file(), f"SPEAR apartment map package is missing: {map_package}"
    )
    expected_project_dir = (root / "cpp" / "unreal_projects" / "SpearSim").resolve()
    return {
        "schema": "avengine_spear_source_snapshot_v1",
        "capture_phase": capture_phase,
        "repository_root": str(root),
        "actual_project_dir": str(expected_project_dir),
        "commit": commit,
        "tracked_worktree_dirty": False,
        "map_asset": "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000",
        "map_package_path": str(map_package),
        "map_package_sha256": sha256_file(map_package),
    }


def validate_selected_project_packages(
    ue_manifest: dict[str, Any], spear_root: Path
) -> None:
    records = ue_manifest.get("selected_project_asset_packages")
    require(
        isinstance(records, list) and bool(records),
        "UE export is missing its selected project asset-package closure",
    )
    require(
        ue_manifest.get("selected_project_asset_package_count") == len(records),
        "UE selected project asset-package count is inconsistent",
    )
    result = subprocess.run(
        ["git", "-C", str(spear_root), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    require(result.returncode == 0, "Unable to enumerate tracked SPEAR files")
    tracked = {value.decode("utf-8") for value in result.stdout.split(b"\0") if value}

    declared_object_paths: set[str] = set()
    for actor in ue_manifest.get("actors", []):
        for component in actor.get("static_mesh_components", []):
            for value in [
                component.get("static_mesh_asset"),
                *component.get("material_assets", []),
            ]:
                if isinstance(value, str) and value.startswith("/Game/"):
                    declared_object_paths.add(value)

    recorded_object_paths: set[str] = set()
    package_names: set[str] = set()
    for record in records:
        require(isinstance(record, dict), "UE package closure entry must be an object")
        package_name = record.get("package_name")
        relative_raw = record.get("repository_relative_path")
        require(
            isinstance(package_name, str) and package_name.startswith("/Game/"),
            "UE package closure contains a non-project package",
        )
        require(package_name not in package_names, "UE package closure has duplicates")
        package_names.add(package_name)
        expected_relative = (
            Path("cpp/unreal_projects/SpearSim/Content")
            / f"{package_name.removeprefix('/Game/')}.uasset"
        ).as_posix()
        require(
            relative_raw == expected_relative,
            "UE package closure path does not match its package name",
        )
        require(
            relative_raw in tracked, "UE package closure contains an untracked file"
        )
        expected_path = (spear_root / relative_raw).resolve()
        try:
            expected_path.relative_to(spear_root)
        except ValueError as error:
            raise ValueError("UE package closure escapes the SPEAR checkout") from error
        require(expected_path.is_file(), f"UE package is missing: {expected_path}")
        require(
            record.get("resolved_path") == str(expected_path),
            "UE package closure resolved path changed",
        )
        require(record.get("git_tracked") is True, "UE package was not tracked")
        require(
            record.get("byte_size") == expected_path.stat().st_size
            and record.get("sha256") == sha256_file(expected_path),
            "UE selected project package bytes changed",
        )
        object_paths = record.get("asset_object_paths")
        require(
            isinstance(object_paths, list) and bool(object_paths),
            "UE package closure entry has no selected object paths",
        )
        require(
            all(
                isinstance(value, str) and value.split(".", 1)[0] == package_name
                for value in object_paths
            ),
            "UE package closure object path does not match its package",
        )
        recorded_object_paths.update(object_paths)
    require(
        recorded_object_paths == declared_object_paths,
        "UE selected mesh/material object paths are not exactly closed by packages",
    )


def validate_real_surface_inputs(
    scene_glb: Path,
    ue_manifest_path: Path,
    ue_manifest: dict[str, Any],
    mesh_audit_path: Path,
    mesh_audit: dict[str, Any],
    current_spear_snapshot: dict[str, Any],
) -> str:
    require(scene_glb.is_file(), f"Scene GLB does not exist: {scene_glb}")
    require(scene_glb.suffix.lower() == ".glb", "--scene-glb must be a .glb file")
    scene_sha256 = sha256_file(scene_glb)

    require(
        ue_manifest.get("schema") == "avengine_legacy_ue_apartment_export_v1",
        f"Unexpected UE manifest schema in {ue_manifest_path}",
    )
    require(ue_manifest.get("status") == "pass", "UE export manifest did not pass")
    export_spear_snapshot = ue_manifest.get("source_snapshot")
    require(
        isinstance(export_spear_snapshot, dict),
        "UE export manifest is missing its pre-export SPEAR source snapshot",
    )
    require(
        export_spear_snapshot == current_spear_snapshot,
        "UE export SPEAR snapshot does not match the current clean source checkout",
    )
    expected_after_snapshot = dict(current_spear_snapshot)
    expected_after_snapshot["capture_phase"] = "after_ue_gltf_export"
    require(
        ue_manifest.get("source_snapshot_after_export") == expected_after_snapshot,
        "UE export post-snapshot does not match its pre-export source checkout",
    )
    require(
        ue_manifest.get("actual_project_dir")
        == current_spear_snapshot["actual_project_dir"],
        "UE export ran from a different Unreal project checkout",
    )
    validate_selected_project_packages(
        ue_manifest, Path(current_spear_snapshot["repository_root"]).resolve()
    )
    dirty_packages = ue_manifest.get("dirty_packages")
    expected_clean = {"content": [], "maps": []}
    require(
        isinstance(dirty_packages, dict)
        and dirty_packages.get("before_reload") == expected_clean
        and dirty_packages.get("after_reload") == expected_clean
        and dirty_packages.get("after_export") == expected_clean,
        "UE export observed unsaved map or content packages",
    )
    require(
        str(ue_manifest.get("loaded_editor_world", "")).startswith(
            "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000."
        ),
        "UE exporter ran against the wrong loaded world",
    )
    require(
        isinstance(ue_manifest.get("engine_version"), str)
        and ue_manifest["engine_version"].startswith("5.5."),
        "UE export did not use the required Unreal Engine 5.5.x runtime",
    )
    require(
        ue_manifest.get("gltf_exporter_plugin", {}).get("version_name") == "1.3.1",
        "Unexpected or missing GLTFExporter plugin version",
    )
    require(
        ue_manifest.get("geometry_representation") == "real_surface_mesh",
        "UE export is not declared as real-surface geometry",
    )
    require(
        ue_manifest.get("uses_actor_bounds_as_geometry") is False,
        "UE export used actor bounds as replacement geometry",
    )
    require(
        "render" in str(ue_manifest.get("geometry_source", "")).lower(),
        "UE geometry_source does not identify render-surface data",
    )
    require(
        ue_manifest.get("option_warnings") == [],
        "UE exporter could not apply all requested options",
    )
    require(
        ue_manifest.get("export_messages", {}).get("errors") == [],
        "UE exporter reported errors",
    )
    export_output = ue_manifest.get("output")
    require(isinstance(export_output, dict), "UE manifest output record is missing")
    require(
        export_output.get("sha256") == scene_sha256,
        "UE manifest GLB hash does not match --scene-glb",
    )

    require(
        mesh_audit.get("schema") == "avengine_real_surface_mesh_audit_v1",
        f"Unexpected mesh audit schema in {mesh_audit_path}",
    )
    gate = mesh_audit.get("real_surface_gate")
    require(isinstance(gate, dict), "Mesh audit real_surface_gate is missing")
    require(gate.get("status") == "pass", "Real-surface mesh gate did not pass")
    triangles = mesh_audit.get("triangles")
    require(
        isinstance(triangles, int) and not isinstance(triangles, bool),
        "Mesh audit triangle count must be an integer",
    )
    require(
        triangles > KNOWN_AABB_TRIANGLE_COUNT,
        "Mesh is too small to pass the real-surface gate",
    )
    require(
        triangles != KNOWN_AABB_TRIANGLE_COUNT,
        "Rejected known 252-triangle legacy actor-bounds export",
    )
    indicators = mesh_audit.get("aabb_proxy_indicators")
    require(isinstance(indicators, dict), "Mesh audit AABB indicators are missing")
    require(
        indicators.get("known_legacy_triangle_signature") is False,
        "Mesh audit matched the known legacy AABB signature",
    )
    require(
        indicators.get("all_mesh_nodes_are_simple_boxes") is False,
        "All audited mesh nodes are simple boxes",
    )
    require(
        mesh_audit.get("sha256") == scene_sha256,
        "Mesh audit GLB hash does not match --scene-glb",
    )
    return scene_sha256


def tracked_marker_directory(repository_root: Path) -> Path:
    return (
        repository_root
        / "examples"
        / "rooms"
        / "blender_custom"
        / "visual"
        / "objects"
    )


def copy_markers(marker_dir: Path, object_dir: Path) -> list[Path]:
    copied: list[Path] = []
    object_dir.mkdir(parents=True, exist_ok=True)
    for index in range(2):
        source = (marker_dir / f"source_marker_{index}.glb").resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Tracked source marker does not exist: {source}")
        destination = object_dir / source.name
        if source != destination.resolve():
            shutil.copyfile(source, destination)
        copied.append(destination)
    return copied


def make_stage_config(scene_glb: Path) -> dict[str, Any]:
    # An absolute asset reference avoids copying a 65 MB audited artifact and
    # makes it impossible for Habitat to silently load another same-named GLB.
    return {
        "render_asset": str(scene_glb),
        "collision_asset": str(scene_glb),
        "up": [0, 1, 0],
        "front": [0, 0, -1],
        "units_to_meters": 1.0,
        "requires_lighting": True,
        "force_flat_shading": False,
        "margin": 0.01,
        "friction_coefficient": 0.8,
        "restitution_coefficient": 0.0,
    }


def make_object_config(index: int) -> dict[str, Any]:
    return {
        "render_asset": f"source_marker_{index}.glb",
        "collision_asset": f"source_marker_{index}.glb",
        "semantic_id": 101 + index,
        "mass": 1.0,
        "join_collision_meshes": True,
    }


def make_lighting_config() -> dict[str, Any]:
    # The audited apartment spans roughly X [-5.9, 6.0], Y [0, 4.2], and
    # Z [-7.0, 7.1].  Three neutral point lights cover the occupied floor
    # without changing geometry or semantic IDs.
    return {
        "lights": {
            "0": {
                "type": "point",
                "position": [-2.5, 3.0, -2.5],
                "intensity": 6.0,
                "color": [1.0, 0.94, 0.86],
            },
            "1": {
                "type": "point",
                "position": [2.5, 3.0, 0.5],
                "intensity": 6.0,
                "color": [0.9, 0.94, 1.0],
            },
            "2": {
                "type": "point",
                "position": [0.0, 3.0, 4.5],
                "intensity": 5.0,
                "color": [1.0, 0.96, 0.9],
            },
        }
    }


def make_scene_instance() -> dict[str, Any]:
    return {
        "stage_instance": {"template_name": SCENE_HANDLE},
        "default_lighting": SCENE_HANDLE,
        "object_instances": [
            {
                "template_name": f"source_marker_{index}",
                "motion_type": "STATIC",
                "translation": position,
                # Habitat scene-instance files use WXYZ ordering.
                "rotation": [1.0, 0.0, 0.0, 0.0],
            }
            for index, position in enumerate(SOURCE_POSITIONS_M)
        ],
    }


def make_dataset_config() -> dict[str, Any]:
    return {
        "stages": {"paths": {".json": ["stages"]}},
        "objects": {"paths": {".json": ["objects"]}},
        "light_setups": {"paths": {".json": ["lighting"]}},
        "scene_instances": {"paths": {".json": ["scenes"]}},
        "navmesh_instances": {SCENE_HANDLE: f"navmeshes/{SCENE_HANDLE}.navmesh"},
    }


def make_room_manifest(
    *,
    scene_glb: Path,
    ue_manifest_path: Path,
    mesh_audit_path: Path,
    ue_manifest: dict[str, Any],
    mesh_audit: dict[str, Any],
    scene_sha256: str,
    spear_snapshot: dict[str, Any],
) -> dict[str, Any]:
    gate = mesh_audit["real_surface_gate"]
    indicators = mesh_audit["aabb_proxy_indicators"]
    return {
        "schema": "avengine_room_package_v1",
        "room_id": ROOM_ID,
        "room_kind": "legacy_ue_real_surface_export",
        "geometry_representation": "real_surface_mesh",
        "coordinate_system": {
            "handedness": "right",
            "up_axis": "+Y",
            "forward_axis": "-Z",
            "linear_unit": "meter",
            "quaternion_order": "xyzw",
        },
        "scene": {
            "scene_id_kind": "handle",
            "scene_id": SCENE_HANDLE,
            "dataset_config_path": f"visual/{SCENE_HANDLE}.scene_dataset_config.json",
            "navmesh_path": f"visual/navmeshes/{SCENE_HANDLE}.navmesh",
            "navmesh_policy": "load_declared",
            "load_semantic_mesh": False,
            "enable_physics": True,
        },
        "assets": [
            {"role": "render_surface_mesh", "path": str(scene_glb)},
            {"role": "ue_export_manifest", "path": str(ue_manifest_path)},
            {"role": "real_surface_mesh_audit", "path": str(mesh_audit_path)},
            {
                "role": "legacy_source_map_package",
                "path": spear_snapshot["map_package_path"],
            },
            {
                "role": "scene_dataset_config",
                "path": f"visual/{SCENE_HANDLE}.scene_dataset_config.json",
            },
            {
                "role": "stage_config",
                "path": f"visual/stages/{SCENE_HANDLE}.stage_config.json",
            },
            {
                "role": "semantic_object_source0",
                "path": "visual/objects/source_marker_0.glb",
            },
            {
                "role": "semantic_object_source1",
                "path": "visual/objects/source_marker_1.glb",
            },
            {
                "role": "object_config_source0",
                "path": "visual/objects/source_marker_0.object_config.json",
            },
            {
                "role": "object_config_source1",
                "path": "visual/objects/source_marker_1.object_config.json",
            },
            {
                "role": "scene_instance",
                "path": f"visual/scenes/{SCENE_HANDLE}.scene_instance.json",
            },
            {
                "role": "lighting_config",
                "path": f"visual/lighting/{SCENE_HANDLE}.lighting_config.json",
            },
            {
                "role": "navmesh",
                "path": f"visual/navmeshes/{SCENE_HANDLE}.navmesh",
            },
        ],
        "semantics": {
            "interpretation": (
                "Raw Habitat semantic IDs: UE stage/background is 0 and the "
                "two explicit source-marker objects are 101 and 102"
            ),
            "id_to_label": {
                "0": "legacy_ue_apartment_stage/background",
                "101": "source_marker_0",
                "102": "source_marker_1",
            },
        },
        "navigation": {
            "agent_height_m": 1.5,
            "agent_radius_m": 0.2,
            "include_static_objects": False,
        },
        "openings": [],
        "connectivity_pairs": [
            {
                "pair_id": "legacy_rig_to_source0",
                "start_m": RIG_NAVIGATION_FLOOR_M,
                "end_m": SOURCE_FLOOR_POSITIONS_M[0],
            },
            {
                "pair_id": "legacy_rig_to_source1",
                "start_m": RIG_NAVIGATION_FLOOR_M,
                "end_m": SOURCE_FLOOR_POSITIONS_M[1],
            },
        ],
        # No opening ray is asserted until a named UE opening is measured in
        # Habitat coordinates.  Connectivity still exercises the real mesh.
        "ray_checks": [],
        "acoustics": {
            "status": "deferred_to_m3",
            "reason": (
                "M1 proves real visual/collision surfaces and transforms; no "
                "acoustic-material or RLR propagation claim is made"
            ),
        },
        "provenance": {
            "source": ue_manifest.get("source_map_asset"),
            "source_revision": spear_snapshot["commit"],
            "source_repository_root": spear_snapshot["repository_root"],
            "source_repository_tracked_dirty": spear_snapshot["tracked_worktree_dirty"],
            "source_map_package_path": spear_snapshot["map_package_path"],
            "source_map_package_sha256": spear_snapshot["map_package_sha256"],
            "exported_scene_sha256": scene_sha256,
            "loaded_editor_world": ue_manifest.get("loaded_editor_world"),
            "coordinate_conversion": ue_manifest.get("coordinate_conversion"),
            "geometry_source": ue_manifest.get("geometry_source"),
            "ue_static_mesh_component_count": ue_manifest.get(
                "static_mesh_component_count"
            ),
            "ue_unique_static_mesh_asset_count": ue_manifest.get(
                "unique_static_mesh_asset_count"
            ),
        },
        "surface_audit": {
            "aabb_proxy": False,
            "method": (
                "Blender evaluated mesh-node audit of UE StaticMesh render-data "
                "LOD0; actor bounds are measurements only and never geometry"
            ),
            "triangle_count": mesh_audit["triangles"],
            "vertex_count": mesh_audit.get("vertices"),
            "mesh_count": mesh_audit.get("meshes"),
            "material_count": mesh_audit.get("materials"),
            "mesh_sha256": scene_sha256,
            "real_surface_gate_status": gate["status"],
            "minimum_triangles": gate.get("minimum_triangles"),
            "rejects_known_252_triangle_aabb_proxy": gate.get(
                "rejects_known_252_triangle_aabb_proxy"
            ),
            "all_mesh_nodes_are_simple_boxes": indicators.get(
                "all_mesh_nodes_are_simple_boxes"
            ),
            "bounds": mesh_audit.get("bounds"),
        },
    }


def make_capture_request() -> dict[str, Any]:
    shared_pose = {
        "translation_m": SHARED_LOCAL_SENSOR_M,
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    return {
        "schema": "avengine_m1_capture_request_v1",
        "request_id": "m1_legacy_ue_apartment_single_view_v1",
        "room_id": ROOM_ID,
        "seed": 29,
        "primary_camera_rig": {
            "rig_id": "camera_rig_0",
            "view_id": "view0",
            "world_from_rig": {
                "translation_m": CAMERA_LISTENER_WORLD_M,
                "rotation_xyzw": FACE_NEGATIVE_X_XYZW,
            },
            "shared_calibration": {
                "projection": "pinhole",
                "resolution_hw": [240, 320],
                "hfov_degrees": 90.0,
                "near_m": 0.05,
                "far_m": 50.0,
                "rig_from_sensor": shared_pose,
            },
            "modalities": [
                {"modality": "rgb", "sensor_uuid": "rig_rgb"},
                {"modality": "depth", "sensor_uuid": "rig_depth"},
                {"modality": "semantic", "sensor_uuid": "rig_semantic"},
            ],
        },
        "listener": {
            "listener_id": "listener0",
            "attached_to": "camera_rig_0",
            "rig_from_listener": shared_pose,
        },
        "sources": [
            {
                "source_id": f"source{index}",
                "world_from_source": {
                    "translation_m": position,
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
            for index, position in enumerate(SOURCE_POSITIONS_M)
        ],
        "qa_views": [
            {
                "qa_id": "navmesh_topdown",
                "kind": "topdown",
                "meters_per_pixel": 0.04,
                "height_m": RIG_NAVIGATION_FLOOR_M[1],
            }
        ],
    }


def generate_package(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[2]
    scene_glb = Path(args.scene_glb).resolve()
    ue_manifest_path = Path(args.ue_manifest).resolve()
    mesh_audit_path = Path(args.mesh_audit).resolve()
    output_dir = Path(args.output_dir).resolve()
    spear_root = Path(args.spear_root).resolve()
    marker_dir = (
        Path(args.marker_dir).resolve()
        if args.marker_dir
        else tracked_marker_directory(repository_root).resolve()
    )

    ue_manifest = load_json_object(ue_manifest_path, "UE export manifest")
    mesh_audit = load_json_object(mesh_audit_path, "real-surface mesh audit")
    spear_snapshot = spear_source_snapshot(spear_root)
    scene_sha256 = validate_real_surface_inputs(
        scene_glb,
        ue_manifest_path,
        ue_manifest,
        mesh_audit_path,
        mesh_audit,
        spear_snapshot,
    )

    visual_dir = output_dir / "visual"
    stage_path = visual_dir / "stages" / f"{SCENE_HANDLE}.stage_config.json"
    object_dir = visual_dir / "objects"
    lighting_path = visual_dir / "lighting" / f"{SCENE_HANDLE}.lighting_config.json"
    scene_path = visual_dir / "scenes" / f"{SCENE_HANDLE}.scene_instance.json"
    dataset_path = visual_dir / f"{SCENE_HANDLE}.scene_dataset_config.json"
    room_path = output_dir / "room_manifest.json"
    request_path = output_dir / "capture_request.json"

    markers = copy_markers(marker_dir, object_dir)
    write_json(stage_path, make_stage_config(scene_glb))
    for index in range(2):
        write_json(
            object_dir / f"source_marker_{index}.object_config.json",
            make_object_config(index),
        )
    write_json(lighting_path, make_lighting_config())
    write_json(scene_path, make_scene_instance())
    write_json(dataset_path, make_dataset_config())
    (visual_dir / "navmeshes").mkdir(parents=True, exist_ok=True)
    write_json(
        room_path,
        make_room_manifest(
            scene_glb=scene_glb,
            ue_manifest_path=ue_manifest_path,
            mesh_audit_path=mesh_audit_path,
            ue_manifest=ue_manifest,
            mesh_audit=mesh_audit,
            scene_sha256=scene_sha256,
            spear_snapshot=spear_snapshot,
        ),
    )
    write_json(request_path, make_capture_request())

    files = [
        room_path,
        request_path,
        dataset_path,
        stage_path,
        lighting_path,
        scene_path,
        *markers,
        *(
            object_dir / f"source_marker_{index}.object_config.json"
            for index in range(2)
        ),
    ]
    return {
        "status": "pass",
        "output_dir": str(output_dir),
        "room_manifest": str(room_path),
        "capture_request": str(request_path),
        "scene_sha256": scene_sha256,
        "triangle_count": mesh_audit["triangles"],
        "spear_source_snapshot": spear_snapshot,
        "generated_files": [
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(files)
        ],
    }


def main() -> None:
    result = generate_package(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
