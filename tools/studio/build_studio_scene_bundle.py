#!/usr/bin/env python3
"""Build a Studio scene bundle: preview mesh plus draft obstacle snapshot.

The bundle is the "pre-rendered" input for the Studio 3D editor: the M3
acoustic package's triangle mesh (clay preview, colored per material
category) plus, when a navmesh is supplied, a serialized draft obstacle
snapshot (binary navmesh grid + rigid OBBs) for millisecond client checks.
The native gates inside the render chain remain the placement authority;
the bundle never replaces them.

MP3D example:
    python tools/studio/build_studio_scene_bundle.py \
        --room-id habitat_mp3d_example_17DRP5sb8fy \
        --acoustic-package-manifest .../m3_current_mp3d_semantic_.../manifest.json \
        --navmesh .../17DRP5sb8fy.navmesh \
        --runtime-prefix ... --magnum-python-site ... \
        --known-walkable-points .../research_m1_request.json \
        --output <scenes_root>/habitat_mp3d_example_17DRP5sb8fy

Apartment (mesh-only bundle; authoring uses explicit UE-cm points):
    python tools/studio/build_studio_scene_bundle.py \
        --room-id legacy_ue_apartment_0000 \
        --acoustic-package-manifest .../m3/root_ue_package_current_20260718_02/manifest.json \
        --output <scenes_root>/legacy_ue_apartment_0000
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path

import numpy as np

BUNDLE_SCHEMA = "avengine_studio_scene_bundle_v1"

# Fixed clay palette; category name hashes pick a stable entry.
_PALETTE = (
    "#8d99ae", "#bc9b7a", "#a3b18a", "#7f9fb8", "#b58f9f", "#9a8c98",
    "#c9ada7", "#84a59d", "#b5838d", "#6d8ea0", "#a68a64", "#8e9aaf",
    "#b7b7a4", "#7d8c78", "#a5a58d", "#94789f",
)


def _category_color(name: str) -> str:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return _PALETTE[digest[0] % len(_PALETTE)]


def _load_array(package_root: Path, spec: dict) -> np.ndarray:
    array = np.load(package_root / spec["path"])
    if list(array.shape) != list(spec["shape"]):
        raise SystemExit(f"array shape mismatch for {spec['path']}: {array.shape}")
    return array


def _material_names(package_root: Path, manifest: dict) -> dict[int, str]:
    categories_path = package_root / manifest["materials"]["categories"]["path"]
    payload = json.loads(categories_path.read_text(encoding="utf-8"))
    names: dict[int, str] = {}
    # accepted shapes: {"categories": {id: name}} / {id: name} / list of
    # {material_id, category}
    body = payload.get("categories", payload) if isinstance(payload, dict) else payload
    if isinstance(body, dict):
        for key, value in body.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            names[index] = str(value if not isinstance(value, dict) else value.get("category", key))
    elif isinstance(body, list):
        for record in body:
            if isinstance(record, dict) and "material_id" in record:
                names[int(record["material_id"])] = str(
                    record.get("source_material_name")
                    or record.get("category_name")
                    or record.get("category")
                    or record.get("name")
                    or record["material_id"]
                )
    return names


def _write_mesh(package_manifest: Path, bundle_dir: Path) -> dict:
    manifest = json.loads(package_manifest.read_text(encoding="utf-8"))
    package_root = package_manifest.parent
    arrays = manifest["arrays"]
    vertices = _load_array(package_root, arrays["vertices"]).astype(np.float32)
    triangles = _load_array(package_root, arrays["triangles"]).astype(np.uint32)
    material_ids = _load_array(
        package_root, arrays["triangle_material_ids"]
    ).astype(np.uint32)
    scale = float(manifest.get("unit_scale_to_m", 1.0))
    if scale != 1.0:
        vertices = vertices * np.float32(scale)

    (bundle_dir / "mesh_positions.bin").write_bytes(vertices.tobytes(order="C"))
    (bundle_dir / "mesh_indices.bin").write_bytes(triangles.tobytes(order="C"))
    (bundle_dir / "mesh_material_ids.bin").write_bytes(material_ids.tobytes(order="C"))

    names = _material_names(package_root, manifest)
    materials = [
        {"id": int(index), "category": name, "color": _category_color(name)}
        for index, name in sorted(names.items())
    ]
    bounds = [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()]
    return {
        "positions": {"file": "mesh_positions.bin", "dtype": "float32", "count": int(vertices.shape[0])},
        "indices": {"file": "mesh_indices.bin", "dtype": "uint32", "count": int(triangles.size)},
        "triangle_material_ids": {
            "file": "mesh_material_ids.bin",
            "dtype": "uint32",
            "count": int(material_ids.shape[0]),
        },
        "materials": materials,
        "bounds_m": bounds,
        "coordinate_system": manifest.get("coordinate_system"),
        "source_package_id": manifest.get("package_id"),
        "source_manifest_path": str(package_manifest.resolve()),
        "source_package_sha256": manifest.get("package_content_sha256"),
    }


def _load_known_walkable_points(path: Path) -> list[list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    points: list[list[float]] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("position_m", "start_position_m", "end_position_m") and (
                    isinstance(value, list) and len(value) == 3
                ):
                    points.append([float(item) for item in value])
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return points


def _build_obstacle_snapshot(
    navmesh_path: Path,
    *,
    floor_height_m: float | None,
    meters_per_pixel: float,
    known_walkable_points: list[list[float]],
) -> dict:
    import habitat_sim  # requires the installed runtime environment

    pathfinder = habitat_sim.nav.PathFinder()
    if not pathfinder.load_nav_mesh(str(navmesh_path)):
        raise SystemExit(f"could not load navmesh: {navmesh_path}")
    bounds = np.asarray(pathfinder.get_bounds(), dtype=np.float64)
    floor = float(bounds[0][1]) if floor_height_m is None else float(floor_height_m)
    grid = np.asarray(
        pathfinder.get_topdown_view(meters_per_pixel, floor), dtype=np.uint8
    )
    if grid.ndim != 2 or not np.any(grid):
        raise SystemExit("navmesh produced an empty topdown grid")

    # Self-check the world→cell convention (row↔Z, col↔X from the lower
    # bound) against pathfinder truth before the convention can ship.
    rows, cols = grid.shape
    checked = 0
    for point in known_walkable_points:
        query = np.asarray([point[0], floor, point[2]], dtype=np.float64)
        if not pathfinder.is_navigable(query, 0.5):
            continue
        row = int((point[2] - bounds[0][2]) / meters_per_pixel)
        col = int((point[0] - bounds[0][0]) / meters_per_pixel)
        if not (0 <= row < rows and 0 <= col < cols):
            raise SystemExit(f"walkable point {point} maps outside the grid")
        window = grid[
            max(0, row - 1) : min(rows, row + 2), max(0, col - 1) : min(cols, col + 2)
        ]
        if not np.any(window):
            raise SystemExit(
                f"grid convention self-check failed at {point} (row={row}, col={col})"
            )
        checked += 1
    if known_walkable_points and checked == 0:
        raise SystemExit("no known walkable point was navigable; wrong navmesh?")

    packed = np.packbits(grid.reshape(-1))
    return {
        "authority": "habitat PathFinder topdown of the declared navmesh",
        "claim_boundary": (
            "draft Studio preview check only; the native placement gates in "
            "the render chain remain the authority"
        ),
        "navmesh_path": str(navmesh_path.resolve()),
        "floor_height_m": floor,
        "meters_per_pixel": float(meters_per_pixel),
        "bounds_m": bounds.tolist(),
        "grid_shape": [int(rows), int(cols)],
        "grid_order": "row_is_z_col_is_x_from_lower_bounds",
        "navmesh_grid_packbits_b64": base64.b64encode(packed.tobytes()).decode("ascii"),
        "rigid_obstacles": [],
        "self_check_points": int(checked),
    }


def _build_mesh_occupancy_snapshot(
    bundle_dir: Path,
    mesh_meta: dict,
    *,
    meters_per_pixel: float,
    floor_height_m: float | None = None,
    walk_band_low_m: float = 0.15,
    walk_band_high_m: float = 1.5,
    known_walkable_points: list[list[float]] | None = None,
) -> dict:
    """Draft walkability from the acoustic mesh itself (no navmesh needed).

    A cell is walkable when floor geometry exists below the walk band and no
    triangle occupies the band above it. Conservative bbox painting; heuristic
    by construction and labeled as such.
    """

    vertices = np.frombuffer(
        (bundle_dir / mesh_meta["positions"]["file"]).read_bytes(), dtype=np.float32
    ).reshape(-1, 3)
    triangles = np.frombuffer(
        (bundle_dir / mesh_meta["indices"]["file"]).read_bytes(), dtype=np.uint32
    ).reshape(-1, 3)
    bounds = np.asarray(mesh_meta["bounds_m"], dtype=np.float64)
    if floor_height_m is not None:
        floor = float(floor_height_m)
    else:
        # The operating floor is the lowest strong horizontal surface, not the
        # bounds minimum (stray vertices sit below the real slab). Take flat
        # triangles, weight by XZ area, and pick the lowest well-supported
        # 5 cm y-bin in the bottom half of the height range.
        all_tris = vertices[triangles].astype(np.float64)
        y_extent = all_tris[:, :, 1].max(axis=1) - all_tris[:, :, 1].min(axis=1)
        flat = y_extent < 0.02
        flat_tris = all_tris[flat]
        edge_a = flat_tris[:, 1, (0, 2)] - flat_tris[:, 0, (0, 2)]
        edge_b = flat_tris[:, 2, (0, 2)] - flat_tris[:, 0, (0, 2)]
        area = 0.5 * np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])
        y_mid = flat_tris[:, :, 1].mean(axis=1)
        half = bounds[0][1] + (bounds[1][1] - bounds[0][1]) * 0.5
        bins = np.arange(bounds[0][1], half + 0.05, 0.05)
        weights, _ = np.histogram(y_mid, bins=bins, weights=area)
        strong = np.nonzero(weights >= max(weights.max() * 0.2, 1.0))[0]
        if strong.size == 0:
            raise SystemExit("could not detect a floor surface in the mesh")
        floor = float(bins[strong[0]] + 0.025)
    cols = int(np.ceil((bounds[1][0] - bounds[0][0]) / meters_per_pixel)) + 1
    rows = int(np.ceil((bounds[1][2] - bounds[0][2]) / meters_per_pixel)) + 1
    floor_grid = np.zeros((rows, cols), dtype=bool)
    block_grid = np.zeros((rows, cols), dtype=bool)

    tri_vertices = vertices[triangles]  # [M, 3, 3]
    tri_y_min = tri_vertices[:, :, 1].min(axis=1)
    tri_y_max = tri_vertices[:, :, 1].max(axis=1)
    band_low = floor + walk_band_low_m
    band_high = floor + walk_band_high_m
    is_floor = tri_y_max <= band_low
    is_blocker = (tri_y_min < band_high) & (tri_y_max > band_low)

    def paint(mask: np.ndarray, grid: np.ndarray) -> None:
        subset = tri_vertices[mask].astype(np.float64)
        if subset.size == 0:
            return
        min_xz = subset[:, :, (0, 2)].min(axis=1)
        max_xz = subset[:, :, (0, 2)].max(axis=1)
        span = max_xz - min_xz
        # Small triangles: a bounding-box fill is cheap and at most one cell
        # over. Large ones get an exact XZ rasterization — a bbox fill would
        # flood the room and sparse sampling would leave holes.
        small = (span[:, 0] <= 2 * meters_per_pixel) & (span[:, 1] <= 2 * meters_per_pixel)

        col_lo = np.clip(((min_xz[small, 0] - bounds[0][0]) / meters_per_pixel).astype(int), 0, cols - 1)
        col_hi = np.clip(((max_xz[small, 0] - bounds[0][0]) / meters_per_pixel).astype(int), 0, cols - 1)
        row_lo = np.clip(((min_xz[small, 1] - bounds[0][2]) / meters_per_pixel).astype(int), 0, rows - 1)
        row_hi = np.clip(((max_xz[small, 1] - bounds[0][2]) / meters_per_pixel).astype(int), 0, rows - 1)
        for r0, r1, c0, c1 in zip(row_lo, row_hi, col_lo, col_hi):
            grid[r0 : r1 + 1, c0 : c1 + 1] = True

        for triangle in subset[~small]:
            a, b, c = triangle[:, (0, 2)]
            c0 = max(0, int((min(a[0], b[0], c[0]) - bounds[0][0]) / meters_per_pixel))
            c1 = min(cols - 1, int((max(a[0], b[0], c[0]) - bounds[0][0]) / meters_per_pixel))
            r0 = max(0, int((min(a[1], b[1], c[1]) - bounds[0][2]) / meters_per_pixel))
            r1 = min(rows - 1, int((max(a[1], b[1], c[1]) - bounds[0][2]) / meters_per_pixel))
            if c1 < c0 or r1 < r0:
                continue
            xs = bounds[0][0] + (np.arange(c0, c1 + 1) + 0.5) * meters_per_pixel
            zs = bounds[0][2] + (np.arange(r0, r1 + 1) + 0.5) * meters_per_pixel
            gx, gz = np.meshgrid(xs, zs)
            d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(d) < 1e-12:
                continue
            w0 = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gz - c[1])) / d
            w1 = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gz - c[1])) / d
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -0.02) & (w1 >= -0.02) & (w2 >= -0.02)
            grid[r0 : r1 + 1, c0 : c1 + 1] |= inside

    paint(is_floor, floor_grid)
    paint(is_blocker, block_grid)
    walkable = (floor_grid & ~block_grid).astype(np.uint8)
    if not np.any(walkable):
        raise SystemExit("mesh occupancy produced no walkable cells")

    for point in known_walkable_points or []:
        row = int((point[2] - bounds[0][2]) / meters_per_pixel)
        col = int((point[0] - bounds[0][0]) / meters_per_pixel)
        window = walkable[
            max(0, row - 2) : min(rows, row + 3), max(0, col - 2) : min(cols, col + 3)
        ]
        if not np.any(window):
            raise SystemExit(
                f"occupancy self-check failed: known walkable point {point} is blocked"
            )

    packed = np.packbits(walkable.reshape(-1))
    return {
        "authority": "acoustic-mesh occupancy heuristic (draft)",
        "claim_boundary": (
            "draft Studio preview check only; the authoring verbs and native "
            "gates remain the authority"
        ),
        "floor_height_m": floor,
        "meters_per_pixel": float(meters_per_pixel),
        "bounds_m": [
            [float(bounds[0][0]), floor, float(bounds[0][2])],
            [float(bounds[1][0]), float(bounds[1][1]), float(bounds[1][2])],
        ],
        "grid_shape": [int(rows), int(cols)],
        "grid_order": "row_is_z_col_is_x_from_lower_bounds",
        "navmesh_grid_packbits_b64": base64.b64encode(packed.tobytes()).decode("ascii"),
        "rigid_obstacles": [],
        "walk_band_m": [walk_band_low_m, walk_band_high_m],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-id", required=True)
    parser.add_argument("--display-name")
    parser.add_argument("--acoustic-package-manifest", required=True, type=Path)
    parser.add_argument("--navmesh", type=Path)
    parser.add_argument(
        "--mesh-occupancy-grid",
        action="store_true",
        help="derive a draft walkability grid from the acoustic mesh itself "
        "(for rooms without a navmesh file)",
    )
    parser.add_argument("--floor-height-m", type=float)
    parser.add_argument("--meters-per-pixel", type=float, default=0.02)
    parser.add_argument(
        "--known-walkable-points",
        type=Path,
        help="JSON with *_position_m fields used to self-check the grid convention",
    )
    parser.add_argument("--runtime-prefix", type=Path)
    parser.add_argument("--magnum-python-site", type=Path)
    parser.add_argument("--rlr-sdk-root", type=Path)
    parser.add_argument("--authoring-json", type=Path, help="per-room authoring defaults")
    parser.add_argument(
        "--textured-glb",
        type=Path,
        help="external textured glTF-binary of the room (dataset file); "
        "recorded by absolute path and served read-only for the editor's "
        "textured view",
    )
    parser.add_argument(
        "--reference-frame-npy",
        type=Path,
        help="rgb.npy of an approved capture; frame 0 is exported as the "
        "editor's real-render reference image",
    )
    parser.add_argument(
        "--reference-frame-order",
        choices=("rgb", "bgr"),
        default="rgb",
        help="channel order stored in the npy (UE captures store bgr)",
    )
    parser.add_argument("--reference-frame-index", type=int, default=0)
    parser.add_argument(
        "--textured-alignment",
        choices=("auto", "identity"),
        default="auto",
        help="identity: the glb is already in the engine world frame (e.g. "
        "the UE glTF export matches H=(U.x,U.z,U.y)/100); auto: fit against "
        "the clay bounds",
    )
    parser.add_argument(
        "--actor-model",
        action="append",
        default=[],
        metavar="NAME=GLB_PATH",
        help="external actor glb served for the editor's 3D actor preview; "
        "repeatable (e.g. human=/path/runtime.glb)",
    )
    parser.add_argument(
        "--scene-instance",
        type=Path,
        help="habitat scene_instance.json; its object placements are resolved "
        "into composition.json for the editor's full textured composition",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="dataset root the scene-instance object glbs live under "
        "(served read-only)",
    )
    parser.add_argument(
        "--default-listener-m1-request",
        type=Path,
        help="M1 request whose primary_camera_rig world_from_rig becomes the "
        "editor's pre-authored render-camera pose",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    bundle_dir = args.output.resolve()
    if bundle_dir.exists():
        raise SystemExit(f"output already exists (fresh/no-clobber): {bundle_dir}")
    bundle_dir.mkdir(parents=True)

    from datetime import datetime, timezone

    bundle: dict = {
        "schema": BUNDLE_SCHEMA,
        "room_id": args.room_id,
        "display_name": args.display_name or args.room_id,
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "mesh": _write_mesh(args.acoustic_package_manifest.resolve(), bundle_dir),
    }

    if args.navmesh is not None:
        if args.runtime_prefix is not None:
            os.environ["AVENGINE_HABITAT_RUNTIME_PREFIX"] = str(
                args.runtime_prefix.resolve()
            )
        if args.magnum_python_site is not None:
            os.environ["AVENGINE_HABITAT_MAGNUM_PYTHON_SITE"] = str(
                args.magnum_python_site.resolve()
            )
        from avengine.m1.habitat_capture import prepare_installed_habitat_runtime

        prepare_installed_habitat_runtime(
            rlr_sdk_root=(
                args.rlr_sdk_root.resolve() if args.rlr_sdk_root is not None else None
            ),
        )
        known_points = (
            _load_known_walkable_points(args.known_walkable_points)
            if args.known_walkable_points
            else []
        )
        bundle["obstacle_map"] = _build_obstacle_snapshot(
            args.navmesh.resolve(),
            floor_height_m=args.floor_height_m,
            meters_per_pixel=args.meters_per_pixel,
            known_walkable_points=known_points,
        )

    if args.authoring_json is not None:
        bundle["authoring"] = json.loads(
            args.authoring_json.read_text(encoding="utf-8")
        )

    if args.mesh_occupancy_grid:
        if "obstacle_map" in bundle:
            raise SystemExit("--mesh-occupancy-grid conflicts with --navmesh")
        known: list[list[float]] = []
        authoring = bundle.get("authoring", {})
        for key, value in authoring.get("defaults_ue_cm", {}).items():
            if key != "camera":
                # UE cm → world m via the m7 apartment transform
                known.append([value[0] / 100.0, value[2] / 100.0, value[1] / 100.0])
        bundle["obstacle_map"] = _build_mesh_occupancy_snapshot(
            bundle_dir,
            bundle["mesh"],
            meters_per_pixel=args.meters_per_pixel,
            floor_height_m=args.floor_height_m,
            known_walkable_points=known,
        )

    if args.textured_glb is not None:
        glb_path = args.textured_glb.resolve()
        if not glb_path.is_file():
            raise SystemExit(f"textured glb not found: {glb_path}")
        bundle["textured_mesh"] = {
            "source_path": str(glb_path),
            "byte_size": glb_path.stat().st_size,
            "alignment": args.textured_alignment,
            "note": "external dataset file served read-only; never copied into Git",
        }

    if args.reference_frame_npy is not None:
        import imageio.v2 as imageio

        frames = np.load(args.reference_frame_npy.resolve(), mmap_mode="r")
        frame = np.asarray(frames[args.reference_frame_index])
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise SystemExit(f"unexpected rgb.npy shape: {frames.shape}")
        frame = frame[:, :, :3]
        if args.reference_frame_order == "bgr":
            frame = frame[:, :, ::-1]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        imageio.imwrite(bundle_dir / "reference_frame.png", frame)
        bundle["reference_frame"] = {
            "file": "reference_frame.png",
            "source_npy": str(args.reference_frame_npy.resolve()),
            "frame_index": args.reference_frame_index,
            "note": "frame from an approved engine capture; the editor shows "
            "it as the real-render reference",
        }

    if args.actor_model:
        actor_models: dict[str, dict] = {}
        for spec in args.actor_model:
            name, _, glb_value = spec.partition("=")
            if not name or not glb_value:
                raise SystemExit(f"--actor-model must be NAME=GLB_PATH, got {spec!r}")
            glb_path = Path(glb_value).resolve()
            if not glb_path.is_file():
                raise SystemExit(f"actor model not found: {glb_path}")
            actor_models[name] = {
                "source_path": str(glb_path),
                "byte_size": glb_path.stat().st_size,
            }
        bundle["actor_models"] = actor_models

    if args.scene_instance is not None:
        if args.dataset_root is None:
            raise SystemExit("--scene-instance requires --dataset-root")
        dataset_root = args.dataset_root.resolve()
        instance = json.loads(args.scene_instance.resolve().read_text(encoding="utf-8"))
        objects = []
        skipped = 0
        for record in instance.get("object_instances", []):
            template = str(record.get("template_name", ""))
            glb_rel = template + ".glb"
            if not (dataset_root / glb_rel).is_file():
                skipped += 1
                continue
            objects.append(
                {
                    "glb": glb_rel,
                    "translation": record.get("translation", [0, 0, 0]),
                    "rotation_wxyz": record.get("rotation", [1, 0, 0, 0]),
                    "scale": record.get("non_uniform_scale", [1, 1, 1]),
                }
            )
        articulated = len(instance.get("articulated_object_instances", []))
        composition = {
            "schema": "avengine_studio_scene_composition_v1",
            "source_scene_instance": str(args.scene_instance.resolve()),
            "objects": objects,
            "skipped_missing_glb": skipped,
            "articulated_objects_not_composed": articulated,
        }
        (bundle_dir / "composition.json").write_text(
            json.dumps(composition, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        bundle["composition"] = {
            "file": "composition.json",
            "dataset_root": str(dataset_root),
            "object_count": len(objects),
            "articulated_objects_not_composed": articulated,
        }

    if args.default_listener_m1_request is not None:
        request = json.loads(
            args.default_listener_m1_request.resolve().read_text(encoding="utf-8")
        )
        rig = request["primary_camera_rig"]["world_from_rig"]
        position = [float(value) for value in rig["translation_m"]]
        x, y, z, w = (float(value) for value in rig["rotation_xyzw"])
        # forward = q * (0, 0, -1); yaw about +Y from -Z
        forward_x = -(2.0 * (x * z + w * y))
        forward_z = -(1.0 - 2.0 * (x * x + y * y))
        yaw_deg = float(np.degrees(np.arctan2(-forward_x, -forward_z)))
        bundle.setdefault("authoring", {})["default_listener"] = {
            "position_m": position,
            "yaw_deg": yaw_deg,
            "source_m1_request": str(args.default_listener_m1_request.resolve()),
        }

    (bundle_dir / "bundle.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "bundle": str(bundle_dir / "bundle.json"),
                "room_id": args.room_id,
                "triangles": bundle["mesh"]["triangle_material_ids"]["count"],
                "obstacle_map": "obstacle_map" in bundle,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
