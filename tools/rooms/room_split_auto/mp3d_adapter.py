"""Read-only MP3D inputs for the shared HM3D room construction pipeline."""
from __future__ import annotations
import importlib.util
import math
from pathlib import Path
import numpy as np
import shapely
import trimesh
from shapely.geometry import Point, mapping
from avengine.acoustics.semantic import _parse_semantic_ply_bytes
from tools.rooms.room_selection.geometry import canonical
from tools.rooms.room_split_auto.cpu_rays import CPUScanRayIntersector
from tools.rooms.room_split_auto.seam_connectivity import exterior
from tools.rooms.room_selection.navigation import placement, sample_navigation, ray_clear_batch


def selection_adapter(root):
    """Load the authorised, unchanged MP3D adapter without replacing HM3D modules."""
    path = Path(root) / "tools/rooms/room_screening/mp3d_geometry.py"
    spec = importlib.util.spec_from_file_location("_room_split_mp3d_input_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_native():
    """PathFinder and llvmpipe use the installed runtime's explicit loader."""
    import os
    from avengine.acoustics.runtime import RUNTIME_MODE_CURRENT_INSTALLED, load_habitat_runtime
    return load_habitat_runtime(
        runtime_prefix=os.environ["AVENGINE_HABITAT_RUNTIME_PREFIX"],
        magnum_python_site=os.environ["AVENGINE_HABITAT_MAGNUM_PYTHON_SITE"],
        rlr_sdk_root=os.environ.get("AVENGINE_RLR_SDK_ROOT"),
        runtime_mode=RUNTIME_MODE_CURRENT_INSTALLED,
    )[0]


def raw_collision(scene_directory):
    path = Path(scene_directory)
    raw = path / (path.name + ".glb")
    scene = trimesh.load(raw, force="scene", process=False, skip_materials=True)
    mesh = scene.to_geometry()
    mesh.vertices = canonical(mesh.vertices)
    mesh.ray = CPUScanRayIntersector(mesh)
    return mesh, dict(mesh.ray.receipt, raw_glb=str(raw),
                      coordinate_transform="raw Z-up (x,y,z) -> Habitat (x,z,-y); scene transforms applied")


def structural_instances(scene_directory, adapter):
    """Doors, stairs and wall axes use actual object triangles and .house identity."""
    path = Path(scene_directory)
    annotation = path / (path.name + ".house")
    semantic = path / (path.name + "_semantic.ply")
    labels, _, _ = adapter.read_house(annotation)
    wanted = {i for i, a in labels.items()
              if any(t in a["category"] for t in ("door", "stair", "wall"))
              or a["category"] in ("step", "steps")}
    vertices, faces, ids = _parse_semantic_ply_bytes(semantic.read_bytes())
    chosen = np.flatnonzero(np.isin(ids, list(wanted)))
    values, inverse, counts = np.unique(ids[chosen], return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)]
    xyz = canonical(vertices)
    markers = []
    for k, iid in enumerate(values):
        a = labels[int(iid)]
        tri = xyz[faces[chosen[order[starts[k]:starts[k + 1]]]]].astype(np.float64)
        flat = tri.reshape(-1, 3)
        xz = flat[:, [0, 2]]
        _, vecs = np.linalg.eigh(np.cov(xz.T)) if len(xz) > 2 else (None, np.eye(2))
        v = vecs[:, -1]
        markers.append(dict(a, triangles=tri, centre_xz_m=((xz.min(0) + xz.max(0)) / 2).tolist(),
                            extent_xz_m=np.ptp(xz, axis=0).tolist(),
                            axis_angle_deg=math.degrees(math.atan2(v[1], v[0])),
                            height_range_m=[float(flat[:, 1].min()), float(flat[:, 1].max())],
                            bounds_xyz_m=[flat.min(0).tolist(), flat.max(0).tolist()],
                            semantic_source=str(annotation), geometry_source=str(semantic)))
    return markers, dict(door_annotation_count=sum("door" in a["category"] for a in labels.values()),
                         door_mesh_instance_count=sum("door" in a["category"] for a in markers),
                         invalid_ground_zero_area_faces_by_region={}, ambiguous_palette_regions=[],
                         input_family="mp3d", semantic_identity="house O/C/R", semantic_source=str(semantic))


def wall_axis(markers, geometry, floor_y, fallback):
    angles, weights = [], []
    x0, z0, x1, z1 = geometry.buffer(.3).bounds
    for item in markers:
        if "wall" not in item["category"]:
            continue
        tri = item["triangles"]
        chosen = ((tri[:, :, 1].min(1) <= floor_y + 2.4) &
                  (tri[:, :, 1].max(1) >= floor_y + .2) &
                  (tri[:, :, 0].max(1) >= x0) & (tri[:, :, 0].min(1) <= x1) &
                  (tri[:, :, 2].max(1) >= z0) & (tri[:, :, 2].min(1) <= z1))
        tri = tri[chosen]
        if not len(tri):
            continue
        edge = np.roll(tri[:, :, [0, 2]], -1, axis=1) - tri[:, :, [0, 2]]
        length = np.linalg.norm(edge, axis=2)
        longest = np.argmax(length, axis=1)
        v = edge[np.arange(len(tri)), longest]
        weight = length[np.arange(len(tri)), longest] * np.ptp(tri[:, :, 1], axis=1)
        keep = weight > 1e-6
        angles.extend((np.degrees(np.arctan2(v[keep, 1], v[keep, 0])) % 90).tolist())
        weights.extend(weight[keep].tolist())
    if not angles:
        return dict(primary_deg=float(fallback), perpendicular_deg=float(fallback + 90),
                    source="no local semantic wall mesh; shared ground minimum rotated rectangle fallback",
                    verified_wall_mesh=False)
    a, weight = np.asarray(angles), np.asarray(weights)
    hist = np.bincount(np.floor(a).astype(int), weights=weight, minlength=90)
    smooth = sum(np.roll(hist, k) for k in range(-2, 3))
    peak = float(np.argmax(smooth)) + .5
    delta = (a - peak + 45) % 90 - 45
    selected = np.abs(delta) <= 3
    axis = float((peak + np.average(delta[selected], weights=weight[selected])) % 90)
    return dict(primary_deg=axis, perpendicular_deg=axis + 90,
                source="local .house wall objects; dominant horizontal wall-triangle edges weighted by vertical area",
                verified_wall_mesh=True, supporting_triangles=int(selected.sum()))


def validate_witness(g, witness, mesh, pf, p, margin=.25):
    """Validate actual pose footprints, native feet and all three scan rays."""
    if not witness.get("found"):
        return dict(passed=False, reason="witness_not_found")
    checks = []
    names = ("camera_m", "source_1_m", "source_2_m")
    for name in names:
        pose = np.asarray(witness[name], float)
        point = Point(pose[0], pose[2])
        outline = exterior(g)  # No nav bridge or neighbour floor is a placement shape.
        height = p["camera_height_m"] if name == "camera_m" else p["source_height_m"]
        foot = pose - [0, height, 0]
        snap = np.asarray(pf.snap_point(foot.astype(np.float32)), float)
        nav_ok = bool(np.isfinite(snap).all() and np.linalg.norm(snap - foot) <= 1e-3
                      and pf.is_navigable(foot.astype(np.float32)))
        checks.append(dict(pose=name, inside_real_filled_outline=bool(outline.covers(point)),
                           boundary_distance_m=float(outline.boundary.distance(point)),
                           margin_ok=bool(outline.covers(point) and outline.boundary.distance(point) >= margin - 1e-7),
                           foot_m=foot.tolist(), navmesh_foot_valid=nav_ok))
    poses = [np.asarray(witness[k], float) for k in names]
    rays = [bool(ray_clear_batch(mesh, poses[a], [poses[b]], p["ray_endpoint_tolerance_m"])[0])
            for a, b in ((0, 1), (0, 2), (1, 2))]
    return dict(passed=all(q["margin_ok"] and q["navmesh_foot_valid"] for q in checks) and all(rays),
                boundary_margin_m=margin, shape_basis="each raw polygon exterior with holes filled; no bridges",
                poses=checks, three_scan_rays_clear=rays, raw_scan_ray_backend=mesh.ray.receipt)


def placement_in_outline(mesh, pf, hs, g, floor_y, p, historical=None):
    """Reuse a valid witness; otherwise search frozen candidates with the production margin."""
    if historical:
        audit = validate_witness(g, historical, mesh, pf, p)
        if audit["passed"]:
            return dict(historical, validation=audit, reused_frozen_witness=True)
    scope = exterior(g)
    _, points, clearance, adj, _ = sample_navigation(pf, hs, scope, floor_y, p)
    if len(points):
        distance = shapely.distance(shapely.points(points[:, 0], points[:, 2]), scope.boundary)
        allowed = np.flatnonzero(distance >= .25 - 1e-7).tolist()
    else:
        allowed = []
    witness = placement(mesh, points, clearance, adj, p, allowed)
    witness.update(reused_frozen_witness=False, boundary_margin_m=.25,
                   placement_scope="real exterior with holes filled; diagnostic nav bridges excluded")
    if witness.get("found"):
        audit = validate_witness(g, witness, mesh, pf, p)
        witness["validation"] = audit
        if not audit["passed"]:
            witness.update(found=False, validation_failed=True)
    return witness


def source_region(room):
    floor = dict(floor_id=room["selected_floor_id"], floor_y_m=room["floor_y_m"],
                 height_range_m=room["height_range_m"], face_count=room["ground_face_count"],
                 floor_area_m2=room["floor_area_m2"], short_side_m=room["short_side_m"],
                 floor_polygon=room["floor_polygon_xz_m"])
    row = dict(house=room["house"], room_label=room["room_label"], region_id=room["region_id"],
               floors=[floor], floor_area_sum_m2=room["floor_area_m2"], origins=[room["source_list_name"]],
               source_selection=room["source_list_name"], ground_measurement=room["floor_area_method"],
               semantic_source=room["semantic_source"], scene_directory=room["scene_directory"],
               annotation_source=room["annotation_source"], navmesh_source=room["navmesh_source"])
    return dict(house=room["house"], source_region=room["room_label"], source_region_id=room["region_id"],
                source_floor_area_m2=room["floor_area_m2"], source_geometry=row, requires_split=True,
                blocks=[], cut_lines=[], source_native_room_id=room["room_id"])


def foreign_floor_geometry(objects, region_id, floor_y, scope, p):
    """Exclude other semantic rooms' actual ground from local navigation bridges."""
    if not hasattr(objects, "_mp3d_ground_index"):
        faces = [(rid, f) for rid, rows in objects.ground.items() for f in rows]
        geometries = [f["polygon"] for _, f in faces]
        objects._mp3d_ground_index = (shapely.STRtree(geometries) if geometries else None,
                                      faces, geometries)
    tree, faces, geometries = objects._mp3d_ground_index
    if tree is None:
        return shapely.GeometryCollection()
    ids = [int(i) for i in tree.query(scope) if faces[int(i)][0] != region_id and
           abs(faces[int(i)][1]["ys"] - floor_y) <= p["floor_height_separation_m"]]
    if not ids:
        return shapely.GeometryCollection()
    from tools.rooms.room_screening.geometry import union_projected_polygons
    return union_projected_polygons([geometries[i] for i in ids]).intersection(scope)
