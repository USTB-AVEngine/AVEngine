"""Check complete bird meshes against original Habitat scene triangles and native floor rays."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from v7_human_roster_materials import read_glb
from v7_human_roster_pose import accessor, pose
from v7_roster_pet_parrot_geometry import downward_hit
from v7_roster_scene_collision import model


def mesh(path):
    document, blob = read_glb(path)
    parts, _ = pose(document, blob, canonical_flip=False, ground=False)
    vertices, faces, offset = [], [], 0
    for part in parts:
        primitive = document["meshes"][document["nodes"][part["node"]]["mesh"]]["primitives"][part["primitive"]]
        if primitive.get("mode", 4) != 4:
            raise ValueError("Only actual triangle primitives are supported")
        positions = part["positions"]
        indices = accessor(document, blob, primitive["indices"]).reshape(-1, 3).astype(np.int32)
        vertices.append(positions)
        faces.append(indices + offset)
        offset += len(positions)
    return np.concatenate(vertices), np.concatenate(faces)


def stage_rotation(attributes):
    up = np.asarray(attributes.get("up", [0, 1, 0]), dtype=float)
    up /= np.linalg.norm(up)
    front = np.asarray(attributes.get("front", [0, 0, -1]), dtype=float)
    front -= up * np.dot(up, front)
    front /= np.linalg.norm(front)
    return np.stack([np.cross(front, up), up, -front])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    args = parser.parse_args()
    task = args.task_root
    sys.path.insert(0, str(args.reference_root / "round3/pydeps"))
    import fcl

    native_rows = json.loads((task / "evidence/habitat_results.json").read_text())
    records = []
    for view in json.loads(args.views.read_text()):
        house = view["house"]
        declaration = json.loads(Path(view["room_manifest"]).read_text())["scene"]
        stage = Path(declaration["scene_id"])
        dataset = json.loads(Path(declaration["dataset_config_path"]).read_text())
        attributes = dataset["stages"]["default_attributes"]
        vertices, indices = mesh(stage)
        rotation = stage_rotation(attributes)
        vertices = (vertices - attributes.get("origin", [0, 0, 0])) @ rotation.T
        triangles = vertices[indices]
        low, high = triangles.min(1), triangles.max(1)
        for kind in ("canary", "budgerigar"):
            native = next(row for row in native_rows if row["house"] == house and row["kind"] == kind)
            meta = json.loads((task / "assets" / kind / "asset.json").read_text())
            root = np.asarray(native["root_m"])
            bird_vertices, bird_faces = mesh(task / "assets" / kind / "runtime.glb")
            world_vertices = bird_vertices + root
            body_low, body_high = world_vertices.min(0), world_vertices.max(0)
            window_low = root - [.8, .1, .8]
            window_high = root + [.8, meta["dimensions_m"][1] + .1, .8]
            keep = np.all(high >= window_low, axis=1) & np.all(low <= window_high, axis=1)
            near = triangles[keep]
            normals = np.cross(near[:, 1] - near[:, 0], near[:, 2] - near[:, 0])
            vertical_fraction = np.abs(normals[:, 1]) / np.maximum(np.linalg.norm(normals, axis=1), 1e-12)
            floor = (vertical_fraction > .98) & (near[:, :, 1].max(1) < root[1] + .02) & (near[:, :, 1].min(1) > root[1] - .02)
            obstacles = near[~floor]
            radius = meta["perch"]["base_radius_m"] * .7
            support = [downward_hit(root + [x, 0, z], near) for x, z in ((0, 0), (-radius, 0), (radius, 0), (0, -radius), (0, radius))]
            if any(value is None for value in support):
                raise RuntimeError("Original scene floor missing; coordinate check failed")
            native_error = max(abs(value - native_value) for value, native_value in zip(support, native["base_support_deltas_m"]))
            if native_error > .001:
                raise RuntimeError("Original triangle/native ray floor mismatch")
            body = model(world_vertices, bird_faces)
            collision, distance = False, None
            if len(obstacles):
                environment = model(obstacles.reshape(-1, 3), np.arange(obstacles.size // 3).reshape(-1, 3))
                collision = bool(fcl.collide(body, environment, fcl.CollisionRequest(num_max_contacts=1), fcl.CollisionResult()))
                distance = float(fcl.distance(body, environment, fcl.DistanceRequest(), fcl.DistanceResult()))
            separation = np.maximum(np.maximum(low[~keep] - body_high, body_low - high[~keep]), 0)
            excluded_clearance = float(np.linalg.norm(separation, axis=1).min()) if len(separation) else None
            clearances = [value for value in (distance, excluded_clearance) if value is not None]
            clearance = min(clearances) if clearances else None
            ground = near[floor]
            if not len(ground):
                raise RuntimeError("No identified ground triangles for positive control")
            ground_model = model(ground.reshape(-1, 3), np.arange(ground.size // 3).reshape(-1, 3))
            buried = model(world_vertices - [0, .03, 0], bird_faces)
            control = bool(fcl.collide(buried, ground_model, fcl.CollisionRequest(num_max_contacts=1), fcl.CollisionResult()))
            passed = not collision and control and clearance is not None and clearance > .02 and max(abs(value) for value in support) < .005
            row = {
                "kind": kind, "asset_id": meta["asset_id"], "house": house,
                "status": "pass" if passed else "fail",
                "source_scene_glb": str(stage),
                "dataset_config": declaration["dataset_config_path"],
                "source_scene_triangles": len(triangles),
                "stage_rotation_to_y_up": rotation.tolist(),
                "root_m": root.tolist(),
                "source_collision": collision,
                "queried_nonfloor_distance_m": distance,
                "excluded_triangle_aabb_distance_lower_bound_m": excluded_clearance,
                "certified_nonfloor_clearance_lower_bound_m": clearance,
                "obstacle_triangles": len(obstacles),
                "expected_ground_triangles_excluded": int(floor.sum()),
                "floor_support_deltas_m": support,
                "max_triangle_native_floor_ray_error_m": native_error,
                "buried_3cm_ground_collision_positive_control": control,
                "method": "whole exported mesh vs source GLB triangle BVHs; stage up/front frame, five native-ray comparisons; intended near-horizontal floor +/-2cm excluded; outside-window AABBs bound clearance",
                "scope": "static geometry; no dynamics or all-room physics admission",
            }
            records.append(row)
            (task / "evidence/habitat_source_geometry.json").write_text(json.dumps(records, indent=2) + "\n")
            print("HABITAT_SOURCE_GEOMETRY", kind, house, row["status"], clearance, native_error, control, flush=True)
    return 0 if len(records) == 4 and all(row["status"] == "pass" for row in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
