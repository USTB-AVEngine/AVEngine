"""CPU scan views and raw-mesh boundary locators for acoustic review.

These software images inspect raw GLB geometry, textures and source projection.
They are diagnostics, not Habitat production captures or counted Episodes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import trimesh
from trimesh.ray.ray_triangle import RayMeshIntersector


def camera_rays(forward, width=160, height=90, hfov_deg=85.):
    f = np.asarray(forward, dtype=float).copy()
    f[1] = 0.; f /= np.linalg.norm(f)
    up = np.array([0., 1., 0.]); right = np.cross(f, up)
    focal = width / (2 * math.tan(math.radians(hfov_deg / 2)))
    xx, yy = np.meshgrid(np.arange(width) + .5, np.arange(height) + .5)
    d = f + ((xx.ravel() - width / 2) / focal)[:, None] * right
    d += ((height / 2 - yy.ravel()) / focal)[:, None] * up
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d, focal, f, right, up


def project_point(value, listener, forward, right, up, focal, width=160, height=90):
    v = np.asarray(value, float) - np.asarray(listener, float)
    z = float(v @ forward)
    if z <= 0:
        return None
    return [float(width / 2 + focal * (v @ right) / z),
            float(height / 2 - focal * (v @ up) / z)]


def load_raw_scan(path):
    """Read external assets without exporting, repairing or modifying them."""
    scene = trimesh.load_scene(path, process=False)
    meshes = list(scene.dump())
    vertices, faces, colors, modes = [], [], [], []
    offset = 0
    for m in meshes:
        if not isinstance(m, trimesh.Trimesh):
            continue
        v = np.asarray(m.vertices, float)
        vertices.append(v[:, [0, 2, 1]] * [1., 1., -1.])
        faces.append(np.asarray(m.faces) + offset); offset += len(v)
        try:
            if m.visual.kind == "texture" and m.visual.uv is not None:
                uv = np.asarray(m.visual.uv)[m.faces].mean(axis=1)
                material = m.visual.material
                image = getattr(material, "image", None)
                if image is None and hasattr(material, "to_simple"):
                    image = material.to_simple().image
                if image is not None:
                    rgb = trimesh.visual.color.uv_to_color(uv, image)[:, :3]
                    modes.append("texture sampled at triangle centroid")
                else:
                    rgb = np.full((len(m.faces), 3), 175, np.uint8)
                    modes.append("texture unavailable; neutral shading")
            else:
                rgb = np.asarray(m.visual.face_colors)[:, :3]
                modes.append("face color")
        except Exception as exc:
            rgb = np.full((len(m.faces), 3), 175, np.uint8)
            modes.append("texture fallback: " + type(exc).__name__)
        colors.append(rgb)
    if not vertices:
        raise ValueError("Raw GLB contains no triangle geometry")
    mesh = trimesh.Trimesh(vertices=np.concatenate(vertices), faces=np.concatenate(faces),
                           process=False)
    return mesh, np.concatenate(colors), sorted(set(modes))


def boundary_locators(mesh, leakage, output):
    """Identify raw boundary edges after welding category/primitive seams.

    An edge is a potential hole only: scan borders and intended openings need
    human interpretation. An escaped ray alone cannot recover a hole polygon.
    """
    welded = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(),
                             process=False)
    welded.merge_vertices(digits_vertex=6)
    unique = trimesh.grouping.group_rows(welded.edges_sorted, require_count=1)
    edges = welded.edges_sorted[unique]
    points = welded.vertices[edges]
    np.savez_compressed(output / "raw_boundary_edges.npz", segments_m=points)
    mid = points.mean(axis=1)
    examples = []
    for item in leakage:
        for ray in item["escaped_rays"]:
            if not len(mid):
                continue
            d = np.asarray(ray["direction"], float)
            vectors = mid - np.asarray(item["origin_m"], float)
            along = vectors @ d
            lateral = np.linalg.norm(vectors - along[:, None] * d, axis=1)
            mask = (along > .1) & (lateral / np.maximum(along, .1) < math.tan(math.radians(3)))
            idx = np.flatnonzero(mask)
            if len(idx):
                i = idx[np.argmin(lateral[idx])]
                examples.append({"room_label": item["room_label"], "role": item["role"],
                                 "direction_index": ray["direction_index"],
                                 "candidate_boundary_midpoint_m": mid[i].tolist(),
                                 "perpendicular_distance_m": float(lateral[i]),
                                 "distance_along_ray_m": float(along[i])})
    result = {"boundary_edge_count": len(edges), "weld_precision_m": 1e-6,
              "escaped_ray_boundary_candidates": examples,
              "boundary_segments_path": str(output / "raw_boundary_edges.npz"),
              "claim": "potential raw surface holes/open scan edges; candidates, not verified closed hole polygons"}
    with (output / "hole_locators.json").open("x") as f:
        json.dump(result, f, indent=2)
    return result


def render_probe(mesh, colors, intersector, direction, output, index):
    width, height = 160, 90
    listener = np.asarray(direction["listener_m"], float)
    source = np.asarray(direction["source_m"], float)
    rays, focal, forward, right, up = camera_rays(direction["camera_forward"], width, height)
    background_distance = np.full(len(rays), np.inf)
    pixels = np.full((len(rays), 3), [15, 18, 22], dtype=np.uint8)
    for start in range(0, len(rays), 256):
        end = min(len(rays), start + 256)
        hit, ri, tri = intersector.intersects_location(
            np.repeat(listener[None], end - start, axis=0), rays[start:end],
            multiple_hits=False)
        if len(ri):
            ids = start + ri
            background_distance[ids] = np.linalg.norm(hit - listener, axis=1)
            normals = mesh.face_normals[tri]
            shade = .35 + .65 * np.abs(np.einsum("ij,ij->i", normals, -rays[ids]))
            pixels[ids] = np.clip(colors[tri] * shade[:, None], 0, 255).astype(np.uint8)
    v = source - listener
    along = rays @ v
    discriminant = along ** 2 - (v @ v - .15 ** 2)
    sphere_distance = along - np.sqrt(np.maximum(discriminant, 0))
    visible = (discriminant >= 0) & (sphere_distance > 0) & (sphere_distance < background_distance)
    pixels[visible] = [15, 235, 240]
    expected = project_point(source, listener, forward, right, up, focal, width, height)
    ids = np.flatnonzero(visible)
    measured = ([float(np.mean(ids % width + .5)), float(np.mean(ids // width + .5))]
                if len(ids) else None)
    error = float(np.linalg.norm(np.asarray(expected) - measured)) if expected and measured else None
    from PIL import Image, ImageDraw
    image = Image.fromarray(pixels.reshape(height, width, 3)).resize((640, 360))
    draw = ImageDraw.Draw(image)
    check = direction["check"]
    # FOA intensity direction is independent of the raw visual-mesh readback.
    world = check.get("foa_world_direction")
    projected_foa = (project_point(listener + np.asarray(world), listener, forward, right, up,
                                   focal, width, height) if world is not None else None)
    if projected_foa is not None:
        x, y = [q * 4 for q in projected_foa]
        draw.line((x - 14, y, x + 14, y), fill="orange", width=3)
        draw.line((x, y - 14, x, y + 14), fill="orange", width=3)
    draw.text((5, 5), "CPU raw GLB / cyan: source marker / orange: v6 FOA", fill="white",
              stroke_width=1, stroke_fill="black")
    image_path = output / ("cpu_view_" + str(index) + ".png")
    image.save(image_path)
    return {"probe": direction["probe"], "image_path": str(image_path), "image_size": [width, height],
            "display_size": [640, 360], "hfov_deg": 85., "focal_pixels": float(focal),
            "camera_forward": forward.tolist(), "camera_right": right.tolist(),
            "listener_m": listener.tolist(), "source_m": source.tolist(), "marker_radius_m": .15,
            "marker_visible_pixels": len(ids), "projected_source_px": expected,
            "marker_centroid_px": measured, "projection_error_pixels": error,
            "projected_foa_px": projected_foa,
            "status": "pass" if len(ids) >= 3 and error is not None and error <= 3 else "review"}


class NativeRawRayIntersector:
    """CPU BVH on raw visual geometry, with approximate face colors for display."""
    def __init__(self, mesh, result, manifest, output):
        from dataclasses import replace
        from scipy.spatial import cKDTree
        from avengine.acoustics.runtime import (
            RLRSimulationConfig, load_compiled_acoustic_scene, load_habitat_runtime,
            _native_configuration, _upload_report)
        from avengine.spatial_audio.runtime import _native_layout
        source_scene = load_compiled_acoustic_scene(result.get("acoustic_package_manifest", output / "package" / "manifest.json"),
                                                   allow_nonpassing_research_qa=True)
        hs, runtime = load_habitat_runtime(runtime_mode="current-installed", **manifest["runtime"])
        sim = replace(RLRSimulationConfig.from_mapping(manifest["simulation"]),
                      thread_count=1, max_ir_seconds=.1)
        configuration, readback = _native_configuration(hs, sim)
        self.context = hs.RLRAcousticContext(configuration)
        from avengine.acoustics.house_acceptance_geometry import valid_native_triangles
        keep = valid_native_triangles(mesh.vertices, mesh.faces)
        faces = mesh.faces[keep]
        if not len(faces):
            raise ValueError("Raw GLB has no nondegenerate triangle geometry")
        objects = [{"object_id": "raw_visual_scan_ray_diagnostic",
                    "vertices": np.ascontiguousarray(mesh.vertices, dtype="<f4"),
                    "triangles": np.ascontiguousarray(faces, dtype="<u4"),
                    "triangle_material_ids": np.zeros(len(faces), dtype="<u4"),
                    "position": (0.,0.,0.), "orientation_wxyz": (1.,0.,0.,0.)}]
        upload = self.context.load_acoustic_scene(str(source_scene.material_database_path),
                                                 [source_scene.material_categories[0]], objects)
        self.receipt = {"degenerate_faces_omitted_from_raw_ray_bvh": int(np.count_nonzero(~keep)),
                        "geometry_policy": "in-memory zero-surface faces only; raw GLB unchanged",
                        "runtime": runtime, "configuration": readback,
                        "upload": _upload_report(upload),
                        "purpose": "raw-GLB CPU ray BVH only; warmup IR is discarded, not a house acceptance RIR",
                        "color_lookup": "nearest face centroid, approximate display texture"}
        d = result["foa_checks"][0]
        self.context.add_source("ray_bvh_source", d["source_m"], 0.)
        self.context.add_listener("ray_bvh_listener", d["listener_m"], [1.,0.,0.,0.],
                                  _native_layout(hs, "ambisonics"), 4, 0., "")
        self.context.simulate_owned()
        self.nearest_face = cKDTree(mesh.triangles_center)
        self.maximum = max(30., float(np.linalg.norm(mesh.extents)) * 2)

    def intersects_location(self, origins, directions, multiple_hits=False):
        hits, ray_ids = [], []
        for i, (origin, direction) in enumerate(zip(origins, directions, strict=True)):
            value = self.context.trace_ray_first_hit(origin.tolist(), direction.tolist(), .005, self.maximum)
            if value.hit:
                hits.append(origin + float(value.distance) * direction)
                ray_ids.append(i)
        points = np.asarray(hits, float).reshape(-1,3)
        ids = self.nearest_face.query(points)[1] if len(points) else np.empty(0,dtype=int)
        return points, np.asarray(ray_ids,dtype=int), ids



def inspect_visual(result, output, manifest=None):
    started = time.monotonic()
    raw = Path(result["input"]["scene_directory"]) / (result["input"]["scan_id"] + ".glb")
    mesh, colors, modes = load_raw_scan(raw)
    intersector = NativeRawRayIntersector(mesh, result, manifest, output) if manifest else RayMeshIntersector(mesh)
    directions = result["foa_checks"]
    checks = [render_probe(mesh, colors, intersector, d, output, i+1)
              for i, d in enumerate(directions)]
    print("CPU_VIEWS_FINISHED", result["house"], time.monotonic()-started, flush=True)
    holes = boundary_locators(mesh, result["leakage"], output)
    return {"status": "pass" if len(checks) == 3 and all(c["status"] == "pass" for c in checks) else "review",
            "raw_visual_asset": str(raw), "coordinate_transform": "raw (x,y,z) -> Habitat (x,z,-y)",
            "ray_engine": "native RLR CPU BVH on raw visual scan" if manifest else "trimesh CPU",
            "ray_bvh_setup": intersector.receipt if manifest else None,
            "triangle_count": len(mesh.faces), "bounds_m": mesh.bounds.tolist(), "color_modes": modes,
            "checks": checks, "hole_locators": {k: v for k,v in holes.items() if k != "escaped_ray_boundary_candidates"},
            "wall_seconds": time.monotonic() - started, "compute_device": "CPU",
            "native_habitat_rgb_verified": False,
            "claim": "CPU raw GLB software views plus v6 FOA rotation/sign check; native Habitat RGB capture not run"}

def inspect_boundary_geometry(result, output):
    """Write a CPU boundary plot even when native propagation or views fail."""
    output = Path(output)
    npz, metadata = output / "raw_boundary_edges.npz", output / "hole_locators.json"
    if npz.exists() != metadata.exists():
        raise ValueError("Partial existing boundary evidence; use a fresh directory")
    if metadata.exists():
        holes = json.loads(metadata.read_text())
        with np.load(npz) as data:
            segments = data["segments_m"]
    else:
        raw = Path(result["input"]["scene_directory"]) / (result["input"]["scan_id"] + ".glb")
        mesh, _, _ = load_raw_scan(raw)
        holes = boundary_locators(mesh, result.get("leakage", []), output)
        with np.load(npz) as data:
            segments = data["segments_m"]
    image = output / "hole_locators.png"
    if image.exists():
        raise FileExistsError("Boundary plot already exists; use a fresh directory")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7))
    displayed = segments[::max(1, len(segments)//5000)]
    from matplotlib.collections import LineCollection
    if len(displayed):
        ax.add_collection(LineCollection(displayed[:, :, [0, 2]], colors="gray", linewidths=.5, alpha=.35))
        flat = displayed[:, :, [0, 2]].reshape(-1, 2)
        ax.update_datalim(flat)
    points = holes.get("escaped_ray_boundary_candidates", [])
    if points:
        p = np.array([v["candidate_boundary_midpoint_m"] for v in points])
        ax.scatter(p[:, 0], p[:, 2], c="red", s=8, label="escape-ray boundary candidates")
    origins = result.get("leakage", [])
    if origins:
        p = np.array([v["origin_m"] for v in origins])
        ax.scatter(p[:, 0], p[:, 2], marker="x", c="blue", label="RLR probe origins")
    else:
        witness = result.get("input", {}).get("acoustic_rooms", [])
        p = [r["placement"][k] for r in witness for k in ("camera_m", "source_1_m", "source_2_m")]
        if p:
            p = np.asarray(p)
            ax.scatter(p[:, 0], p[:, 2], marker="x", c="orange", label="curation witness; RLR rays unavailable")
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set(title=result["house"] + " / raw scan boundary candidates",
           xlabel="world X (m)", ylabel="world Z (m); all floors projected")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    fig.text(.04, .02, f"Raw boundary edges: {holes['boundary_edge_count']}. Potential scan borders/openings; not verified holes or repaired surfaces.", fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(image, dpi=140)
    plt.close(fig)
    return {**holes, "image_path": str(image), "compute_device": "CPU",
            "native_rir_or_rgb_claim": False,
            "ray_origins_available": bool(result.get("leakage"))}
