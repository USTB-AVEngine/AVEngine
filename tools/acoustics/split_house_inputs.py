"""External inputs and placement geometry for split-room escape measurements."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import trimesh

from tools.rooms.room_selection.geometry import collision_mesh

FAMILIES = ("hm3d", "mp3d", "kujiale")


def read_json(path):
    return json.loads(Path(path).read_text())


def package_matrix(manifest):
    """Read the package's reviewed source transform; never infer a USD axis map."""
    transform = manifest["geometry"]["source_to_canonical"]
    matrix = np.asarray(transform["matrix_row_major"], dtype=float).reshape(4, 4)
    if (transform.get("reviewed") is not True or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or abs(np.linalg.det(matrix[:3, :3])) <= 1e-12):
        raise ValueError("Package source_to_canonical must be a reviewed nonsingular affine matrix")
    if (manifest["geometry"]["transform_policy"] != "baked_to_canonical_world"
            or manifest["coordinate_system"]["up_axis"] != "+Y"
            or manifest["coordinate_system"]["linear_unit"] != "meter"
            or manifest["unit_scale_to_m"] != 1):
        raise ValueError("Package geometry must already be baked into canonical metre coordinates")
    return matrix, transform


def _descriptors(value, prefix):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str):
            yield prefix, value["path"]
        else:
            for key, item in value.items():
                yield from _descriptors(item, prefix + "." + key)


def validate_house_input(h):
    """Check mandatory external files and house/package identity without changing them."""
    house, family = h["house"], h["family"]
    if family not in FAMILIES or not house.startswith(family + "_") or Path(house).name != house:
        raise ValueError(f"Invalid house/family identity: {house!r}, {family!r}")
    paths = {key: h[key] for key in ("annotation_source", "navmesh_source")}
    if family != "kujiale":
        paths["semantic_source"] = h["semantic_source"]
    if family == "kujiale":
        for key in ("placement_mesh_vertices", "placement_mesh_triangles", "placement_geometry_manifest",
                    "placement_reference_package_manifest"):
            paths[key] = h[key]
    else:
        paths["whole_house_visual_source"] = str(Path(h["scene_directory"]) / (h["scan_id"] + ".glb"))
    package = h.get("acoustic_package_manifest")
    if family == "kujiale" and (not package or not Path(package).parent.name.endswith("package_rlr")):
        raise ValueError(f"{house}: Kujiale RLR requires the historical package_rlr manifest")
    if package:
        paths["acoustic_package_manifest"] = package
    records = []
    for field, path in paths.items():
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"{house}: missing {field}: {path}")
        records.append({"field": field, "path": str(path), "exists": True, "size_bytes": path.stat().st_size})
    matrix_record = None
    if package:
        p = Path(package)
        manifest = read_json(p)
        if manifest["source_room"]["room_id"] != house:
            raise ValueError(f"{house}: acoustic package belongs to {manifest['source_room']['room_id']}")
        for section in ("arrays", "materials", "qa", "debug_mesh"):
            for field, relative in _descriptors(manifest.get(section), section):
                artifact = p.parent / relative
                if not artifact.is_file():
                    raise FileNotFoundError(f"{house}: missing package {field}: {artifact}")
                records.append({"field": field, "path": str(artifact), "exists": True,
                                "size_bytes": artifact.stat().st_size})
        if family == "kujiale":
            _, matrix_record = package_matrix(manifest)
            reference = read_json(h["placement_reference_package_manifest"])
            if reference["source_room"]["room_id"] != house:
                raise ValueError(f"{house}: placement reference package belongs to a different house")
            _, reference_matrix = package_matrix(reference)
            if reference_matrix != matrix_record:
                raise ValueError(f"{house}: placement and RLR package transforms disagree")
            geometry = read_json(h["placement_geometry_manifest"])
            if geometry["room_id"] != house or geometry["coordinate_system"] != manifest["coordinate_system"]:
                raise ValueError(f"{house}: original placement geometry coordinate declaration disagrees with package")
            # Some historical unfiltered array files are missing. The original
            # placement cache is the surviving byte-identical copy; validate it
            # against the existing package descriptor, never invent a new hash.
            for key in ("vertices", "triangles"):
                verify_canonical_cache(h["placement_mesh_" + key], reference["arrays"][key], house)
    return {"house": house, "family": family, "required_files": records,
            "source_to_canonical": matrix_record, "compile_on_demand": package is None,
            "historical_provenance_files": [{"field": "semantic_source", "path": h["semantic_source"],
                "exists": Path(h["semantic_source"]).is_file(), "required_for_cpu_measurement": False}] if family == "kujiale" else []}


def require_houses(houses, selected):
    missing = sorted(set(selected) - set(houses))
    if missing:
        raise ValueError(
            "Missing house inputs: " + ", ".join(missing) + ". Add each house's original navmesh, "
            "whole-house geometry and package provenance to --house-inputs, or regenerate the file with "
            "tools/acoustics/prepare_split_house_inputs.py (see --help for the source lists/roots). "
            "Existing input files are never extended in place."
        )


def verify_canonical_cache(path, descriptor, house):
    path = Path(path)
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if sha != descriptor["sha256"] or path.stat().st_size != descriptor["byte_size"]:
        raise ValueError(f"{house}: placement cache differs from the original package canonical array: {path}")
    return sha


def placement_mesh(h):
    if h["family"] in ("hm3d", "mp3d"):
        mesh, path = collision_mesh(h["scene_directory"])
        return mesh, {"path": path, "coordinate_transform": "original adapter: raw Z-up (x,y,z) -> Habitat (x,z,-y)"}
    # These are the original USD-derived placement arrays used to build the
    # declared navmesh. They already contain the package matrix's baked result.
    # Compare with the original unfiltered acoustic arrays before using them;
    # applying the source matrix again would transform canonical coordinates twice.
    manifest = read_json(h["acoustic_package_manifest"])
    _, transform = package_matrix(manifest)
    refpath = Path(h["placement_reference_package_manifest"])
    reference = read_json(refpath)
    _, reference_transform = package_matrix(reference)
    if transform != reference_transform:
        raise ValueError(f"{h['house']}: package transforms disagree")
    vertices = np.load(h["placement_mesh_vertices"], allow_pickle=False)
    triangles = np.load(h["placement_mesh_triangles"], allow_pickle=False)
    for field in ("vertices", "triangles"):
        verify_canonical_cache(h["placement_mesh_" + field], reference["arrays"][field], h["house"])
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    return mesh, {"path": h["placement_mesh_vertices"], "triangles_path": h["placement_mesh_triangles"],
                  "coordinate_transform": transform, "reference_package_manifest": str(refpath),
                  "canonical_arrays_equal_reference": True,
                  "validation": "surviving original placement .npy files match the retained original package byte_size and SHA256 descriptors",
                  "transform_applied_this_run": False,
                  "reason": "original placement arrays already equal the package matrix's canonical baked output"}
