#!/usr/bin/env python3
"""Build complete split-acoustics inputs from retained HM3D and historical family sources.

HM3D house records are copied unchanged. MP3D reuses the original measured
whole-house package where available, otherwise the runner uses the original
compiler/material seed. Kujiale always uses its historical package_rlr and
original placement navmesh/canonical mesh. All paths come from explicit inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import os
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from tools.acoustics.room_split_escape import save
from tools.acoustics.split_house_inputs import read_json, validate_house_input, placement_mesh


def mp3d_package(house, rows, root):
    candidates = []
    cache = root / "followup_per_room_20261003" / "inputs" / ("cache_" + house + ".json")
    if cache.is_file():
        package = read_json(cache)["input"].get("reuse_package")
        if package:
            candidates.append((Path(package), str(cache)))
    for row in rows:
        result = row.get("unpatched_result_path") or row.get("result_path")
        if result and Path(result).is_file():
            package = read_json(result).get("acoustic_package_manifest")
            if package:
                candidates.append((Path(package), result))
            break
    candidates.extend((p, str(root)) for p in sorted(root.glob("**/houses/" + house + "/package/manifest.json")))
    for path, provenance in candidates:
        if path.is_file():
            if read_json(path)["source_room"]["room_id"] != house:
                raise ValueError(f"{house}: candidate package belongs to another house: {path}")
            return str(path), provenance
    return None, str(root)


def rlr_package_alias(package, root, house):
    """Name the already compatible historical no-op case package_rlr, without editing it."""
    if root is None:
        raise ValueError(f"{house}: historical package has no package_rlr sibling; supply --kujiale-package-alias-root for a verified no-op alias")
    manifest = read_json(package)
    vertices = np.load(package.parent / manifest["arrays"]["vertices"]["path"], mmap_mode="r", allow_pickle=False)
    triangles = np.load(package.parent / manifest["arrays"]["triangles"]["path"], mmap_mode="r", allow_pickle=False)
    zero, incompatible = 0, 0
    for offset in range(0, len(triangles), 100000):
        points = np.asarray(vertices[triangles[offset:offset + 100000]], dtype=np.float32)
        doubles = points.astype(np.float64)
        cross = np.cross(doubles[:, 1] - doubles[:, 0], doubles[:, 2] - doubles[:, 0])
        zero += int(np.count_nonzero(np.all(cross == 0, axis=1)))
        # The original native uploader forms float32 edges then double products.
        ab = (points[:, 1] - points[:, 0]).astype(np.float64)
        ac = (points[:, 2] - points[:, 0]).astype(np.float64)
        native_cross = np.cross(ab, ac)
        incompatible += int(np.count_nonzero(np.einsum("ij,ij->i", native_cross, native_cross) <= 1e-20))
    if zero or incompatible:
        raise ValueError(f"{house}: unsuffixed historical package contains {zero} zero-area/{incompatible} native-incompatible faces; no no-op alias is permitted")
    destination = root / house
    destination.mkdir(parents=True, exist_ok=False)
    alias = destination / "package_rlr"
    alias.symlink_to(package.parent.resolve(), target_is_directory=True)
    receipt = destination / "alias_receipt.json"
    save(receipt, {"house": house, "source_package_manifest": str(package),
                   "package_rlr_manifest": str(alias / package.name), "triangle_count": len(triangles),
                   "zero_area_triangles": zero, "native_incompatible_triangles": incompatible,
                   "operation": "no-op symlink alias of original already RLR-compatible package; no geometry or materials modified"})
    return alias / package.name, receipt


def prepare(args):
    if os.getpriority(os.PRIO_PROCESS, 0) < 15:
        os.nice(15 - os.getpriority(os.PRIO_PROCESS, 0))
    for path in (args.output, args.validation_output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    base = read_json(args.hm3d_inputs)
    houses = dict(base["houses"])
    if any(h["family"] != "hm3d" for h in houses.values()):
        raise ValueError("--hm3d-inputs must contain only HM3D house records")
    with args.mp3d_rooms.open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["family"] == "mp3d"]
    for house in sorted({r["house"] for r in rows}):
        sid = house.removeprefix("mp3d_")
        scene = args.mp3d_data_root / sid
        package, provenance = mp3d_package(house, [r for r in rows if r["house"] == house], args.measurement_root)
        houses[house] = {
            "house": house, "family": "mp3d", "scan_id": sid, "scene_directory": str(scene),
            "semantic_source": str(scene / (sid + "_semantic.ply")),
            "annotation_source": str(scene / (sid + ".house")),
            "navmesh_source": str(scene / (sid + ".navmesh")),
            "registry_source": str(args.mp3d_rooms), "acoustic_package_manifest": package,
            "package_provenance": provenance,
        }
    inventory = read_json(args.kujiale_root / "inventory.json")
    for item in sorted(inventory, key=lambda x: x["house"]):
        house = item["house"]
        result_path = args.kujiale_root / "acoustics_batch" / "houses" / house / "result.json"
        result = read_json(result_path)
        original = result["input"]
        house_input = {key: original[key] for key in (
            "house", "family", "scan_id", "scene_directory", "semantic_source", "annotation_source",
            "navmesh_source", "registry_source",
        )}
        original_package = Path(result["acoustic_package_manifest"])
        package = original_package
        alias_receipt = None
        if not package.parent.name.endswith("package_rlr"):
            package, alias_receipt = rlr_package_alias(package, args.kujiale_package_alias_root, house)
        surface = Path(original["scene_directory"])
        house_input.update(
            acoustic_package_manifest=str(package), historical_acoustic_result=str(result_path),
            placement_mesh_vertices=str(surface / "vertices.npy"),
            placement_mesh_triangles=str(surface / "triangles.npy"),
            placement_geometry_manifest=str(surface.parent / "geometry_room_manifest.json"),
            placement_reference_package_manifest=str(original_package.parent.parent / "acoustic_package" / "manifest.json"),
        )
        if alias_receipt:
            house_input["package_rlr_alias_receipt"] = str(alias_receipt)
        houses[house] = house_input
    checks = []
    for house, h in sorted(houses.items()):
        record = validate_house_input(h)
        if h["family"] == "kujiale":
            mesh, receipt = placement_mesh(h)
            record["placement_coordinate_validation"] = receipt
            del mesh
        checks.append(record)
    counts = dict(Counter(h["family"] for h in houses.values()))
    result = dict(base)
    result.update(houses=houses, family_sources={
        "hm3d": str(args.hm3d_inputs), "mp3d": str(args.mp3d_rooms), "kujiale": str(args.kujiale_root / "inventory.json"),
    }, family_counts=counts)
    assert all(result["houses"][h] == original for h, original in base["houses"].items())
    save(args.validation_output, {
        "status": "pass", "family_counts": counts, "hm3d_records_unchanged": True,
        "validation_scope": "mandatory files exist; house/package IDs and coordinate declarations; Kujiale original canonical placement arrays equal the package output. Native loader checks package contents when measuring.",
        "required_file_checks": sum(len(x["required_files"]) for x in checks), "houses": checks,
    })
    save(args.output, result)
    print("HOUSE_INPUTS", counts, "required_file_checks", sum(len(x["required_files"]) for x in checks), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hm3d-inputs", type=Path, required=True)
    parser.add_argument("--mp3d-rooms", type=Path, required=True, help="historical training-room CSV; all MP3D houses in this list")
    parser.add_argument("--mp3d-data-root", type=Path, required=True)
    parser.add_argument("--measurement-root", type=Path, required=True, help="original whole-house measurement root")
    parser.add_argument("--kujiale-root", type=Path, required=True)
    parser.add_argument("--kujiale-package-alias-root", type=Path, help="new output root for historical packages requiring no cleanup but lacking a package_rlr directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
