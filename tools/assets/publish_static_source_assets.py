#!/usr/bin/env python3
"""Publish admitted static sound sources into the shared asset tree.

The layout is the one publish_animal_assets.py established and this writes into
the same index, so a consumer still reads one file:

    <root>/index.json                the only file a consumer must read
    <root>/<category>/               what kind of thing this is
              <object_type>/         the product
                  <variant>/         one realized attribute combination
                      asset.json     the same record, kept beside the mesh
                      finalized.glb  scaled, grounded, front on +X
                      watertight.glb the closed proxy the finalizer scaled
                      emitter_marker.glb
                      evidence/      stage manifests and the review sheet

Statics differ from animals in three ways that matter to a consumer:

  * They are rigid.  There is no rig, no walk, and the two animal gates -
    retopology starvation and skinned tearing - do not apply.  What replaces
    them is watertightness, the front axis, grounding, the physical extent on
    all three axes, and where the emitter anchor landed.
  * The emitter anchor is a reviewed point on a visible feature, not a
    measured muzzle, so its authority and the image it was read off travel
    with the asset.
  * The finalizer applies yaw only.  Nothing in the static chain levels pitch
    or roll, and image-to-3D output arrives tilted, so the measured tilt is
    recorded on every asset rather than being silently carried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path

import numpy as np

ASSET_SCHEMA = "avengine_sound_source_asset_v1"
INDEX_SCHEMA = "avengine_sound_source_asset_index_v1"
PIPELINE = "flux2_pixal3d_static_v1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mesh_extent_and_tilt(glb: Path) -> dict:
    """Extent in metres and how far the shell is from standing straight.

    Node transforms carry the finalizer's rotation and scale, so they have to
    be applied: reading POSITION alone reports the untransformed proxy.
    """

    data = glb.read_bytes()
    json_length = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20 : 20 + json_length].decode("utf-8"))
    offset = 20 + json_length + 8

    def accessor(index):
        item = gltf["accessors"][index]
        view = gltf["bufferViews"][item["bufferView"]]
        start = offset + view.get("byteOffset", 0) + item.get("byteOffset", 0)
        dtype = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}[
            item["componentType"]
        ]
        width = {"SCALAR": 1, "VEC3": 3}[item["type"]]
        flat = np.frombuffer(data, dtype=dtype, count=item["count"] * width, offset=start)
        return flat.reshape(-1, width) if width > 1 else flat

    def matrix_of(node):
        result = np.eye(4)
        if "matrix" in node:
            return np.array(node["matrix"], dtype=float).reshape(4, 4).T
        if "scale" in node:
            result = np.diag(list(node["scale"]) + [1.0]) @ result
        if "rotation" in node:
            x, y, z, w = node["rotation"]
            rotation = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )
            block = np.eye(4)
            block[:3, :3] = rotation
            result = block @ result
        if "translation" in node:
            block = np.eye(4)
            block[:3, 3] = node["translation"]
            result = block @ result
        return result

    points, triangles = [], []
    def walk(index, parent):
        node = gltf["nodes"][index]
        world = parent @ matrix_of(node)
        if "mesh" in node:
            for primitive in gltf["meshes"][node["mesh"]]["primitives"]:
                local = accessor(primitive["attributes"]["POSITION"]).astype(float)
                base = len(np.concatenate(points)) if points else 0
                points.append((world[:3, :3] @ local.T).T + world[:3, 3])
                triangles.append(
                    accessor(primitive["indices"]).astype(np.int64).reshape(-1, 3) + base
                )
        for child in node.get("children", []):
            walk(child, world)

    for root in gltf["scenes"][gltf.get("scene", 0)]["nodes"]:
        walk(root, np.eye(4))
    vertices = np.concatenate(points)
    faces = np.concatenate(triangles)
    extent = vertices.max(0) - vertices.min(0)

    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    centroid = ((a + b + c) / 3.0)
    mean = (centroid * area[:, None]).sum(0) / area.sum()
    delta = centroid - mean
    covariance = (delta * area[:, None]).T @ delta / area.sum()
    values, vectors = np.linalg.eigh(covariance)
    long_axis = vectors[:, int(np.argmax(values))]
    return {
        # glTF is y-up; the finalizer puts the reviewed front on +x.
        "depth_forward_m": round(float(extent[0]), 4),
        "height_up_m": round(float(extent[1]), 4),
        "width_right_m": round(float(extent[2]), 4),
        "faces": int(len(faces)),
        "long_axis_elevation_deg": round(
            float(np.degrees(np.arcsin(min(1.0, abs(long_axis[1]))))), 1
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--admission-batch", required=True, type=Path)
    parser.add_argument("--admission-root", required=True, type=Path)
    parser.add_argument("--flux-root", required=True, type=Path)
    parser.add_argument("--review-root", required=True, type=Path)
    parser.add_argument("--admission-state", default="research")
    parser.add_argument("--revision", default="v1")
    args = parser.parse_args()

    batch = load(args.admission_batch)
    root = args.root
    root.mkdir(parents=True, exist_ok=True)

    records = []
    for entry in batch["jobs"]:
        instance_id = entry["instance_id"]
        instance_root = args.admission_root / "instances" / instance_id
        finalization = load(instance_root / "02_finalization/finalization_manifest.json")
        emitter = load(instance_root / "03_emitter/emitter_measurement.json")
        anchor = load(instance_root / "03_emitter/bound_anchor_spec.json")
        candidate = None
        for path in sorted((args.flux_root / "candidates").glob("*/candidate_manifest.json")):
            payload = load(path)
            if payload["instance_id"] == instance_id:
                candidate = payload
                break
        if candidate is None:
            raise SystemExit(f"{instance_id}: no flux candidate manifest")

        taxonomy = candidate["taxonomy"]
        category = taxonomy["category"]
        object_type = taxonomy["object_type"]
        variant = "_".join(
            candidate["sampled_attributes"][key]
            for key in sorted(candidate["sampled_attributes"])
        )
        asset_id = (
            f"generated_{object_type}_{variant}_"
            f"{args.admission_state}_{args.revision}"
        )
        relative = Path(category) / object_type / variant
        destination = root / relative
        if destination.exists():
            raise SystemExit(f"{relative} already published; bump --revision")
        destination.mkdir(parents=True)
        (destination / "evidence").mkdir()

        for source, name in (
            (instance_root / "02_finalization/finalized.glb", "finalized.glb"),
            (instance_root / "01_watertight/watertight.glb", "watertight.glb"),
            (instance_root / "03_emitter/emitter_marker.glb", "emitter_marker.glb"),
        ):
            shutil.copy2(source, destination / name)
        for source, name in (
            (instance_root / "01_watertight/watertight_manifest.json", "watertight_manifest.json"),
            (instance_root / "02_finalization/finalization_manifest.json", "finalization_manifest.json"),
            (instance_root / "03_emitter/emitter_measurement.json", "emitter_measurement.json"),
            (instance_root / "03_emitter/bound_anchor_spec.json", "bound_anchor_spec.json"),
            (args.review_root / instance_id / "static_object_review_manifest.json", "static_object_review_manifest.json"),
        ):
            if source.is_file():
                shutil.copy2(source, destination / "evidence" / name)
        sheet = args.review_root / instance_id / "contact_sheet.png"
        if sheet.is_file():
            shutil.copy2(sheet, destination / "evidence" / "review_contact_sheet.png")

        geometry = mesh_extent_and_tilt(destination / "finalized.glb")
        physical = finalization["physical_scale"]
        anchor_record = emitter["emitter_anchor"]
        sample = anchor_record["resolved_surface_samples"][0]

        record = {
            "schema": ASSET_SCHEMA,
            "asset_id": asset_id,
            "path": str(relative),
            "category": category,
            "entity_class": "rigid_static_object",
            "pipeline": PIPELINE,
            "identity": {
                "object_type": object_type,
                "fixed_attributes": candidate["fixed_attributes"],
                "profile_schema_id": candidate["profile_schema_id"],
                "profile_sha256": candidate["profile_sha256"],
                "request_sha256": candidate["request_sha256"],
                "instance_id": instance_id,
            },
            "realized_attributes": candidate["sampled_attributes"],
            "generation": {
                "effective_prompt": candidate["generation"].get("effective_prompt"),
                "generation": {
                    key: value
                    for key, value in candidate["generation"].items()
                    if key != "effective_prompt"
                },
                "one_shot_execution": candidate["one_shot_execution"],
                "candidate_image": candidate["output"],
                "lineage_group_id": candidate["lineage_group_id"],
            },
            "geometry": {
                "finalized_glb": "finalized.glb",
                "finalized_glb_sha256": digest(destination / "finalized.glb"),
                "watertight_glb_sha256": digest(destination / "watertight.glb"),
                "coordinate_system": finalization["coordinate_system"],
                **geometry,
                "target_height_cm": round(physical["target_height_m"] * 100, 2),
                "height_tolerance_cm": round(physical["tolerance_m"] * 100, 2),
                "height_provenance": (
                    "provisional typical retail dimension; the measured extent "
                    "above is the mesh as published"
                ),
            },
            "emitter": {
                "anchor_id": anchor_record["anchor_id"],
                "anchor_type": anchor_record["anchor_type"],
                "semantic_role": anchor_record["semantic_role"],
                "offset_m": anchor_record["offset_m"],
                "offset_space": anchor_record["offset_space"],
                "selection_method": anchor_record["method"],
                "reviewed_target_fraction_xyz": sample["target_fraction_xyz"],
                "target_to_surface_distance_m": sample["target_to_surface_distance_m"],
                "authority_sha256": anchor["review_evidence"]["sha256"],
            },
            "acceptance": {
                "watertight": True,
                "boundary_edges": 0,
                "front_axis": finalization["heading"]["target_front_axis"],
                "reviewed_source_front_yaw_deg": finalization["heading"][
                    "reviewed_source_front_yaw_deg"
                ],
                "grounded_minimum_up_m": finalization["grounding"][
                    "minimum_up_after_export_readback_m"
                ],
                "height_absolute_error_m": physical["absolute_error_m"],
                "known_defect_tilt": (
                    "the static chain applies yaw only; long-axis elevation is "
                    f"{geometry['long_axis_elevation_deg']} degrees where an "
                    "upright object wants 90 and a bar wants 0"
                ),
            },
            "admission_state": args.admission_state,
            "formal_dataset_registration_authorized": False,
        }
        (destination / "asset.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        records.append(record)

    index_path = root / "index.json"
    index = load(index_path) if index_path.is_file() else {
        "schema": INDEX_SCHEMA,
        "formal_dataset_registration_authorized": False,
        "assets": [],
    }
    known = {record["asset_id"] for record in records}
    index["assets"] = sorted(
        records + [item for item in index.get("assets", []) if item["asset_id"] not in known],
        key=lambda item: item["asset_id"],
    )
    gates = dict(index.get("acceptance_gates") or {})
    gates[PIPELINE] = {
        "note": (
            "rigid objects are not rigged and do not walk, so the two animal "
            "gates do not apply here; these are what replaces them"
        ),
        "watertight": {
            "tool": "SPEAR tools/blender_create_watertight_textured_proxy_mesh.py",
            "criterion": "boundary, wire and non-manifold edges all zero",
        },
        "heading": {
            "criterion": (
                "the reviewed front maps to +X and a render of the finalized "
                "mesh with front-axis positive-x shows the sound outlet"
            ),
        },
        "grounding": {"criterion": "lowest point at zero within 1e-5 m"},
        "physical_extent": {
            "criterion": (
                "all three axes plausible for the real product, not only the "
                "height the finalizer scaled: uniform height scaling amplifies "
                "any aspect error from reconstruction"
            ),
        },
        "debris": {
            "criterion": (
                "surface area outside the largest connected component under 2 "
                "percent and no debris visible in the five review views, judged "
                "on the raw reconstruction; the watertight remesh then removes "
                "what is left"
            ),
        },
        "known_gap_no_levelling": (
            "nothing in the static chain corrects pitch or roll, and measured "
            "reconstructions arrive 5.7 to 22.5 degrees off upright, so every "
            "asset carries its long_axis_elevation_deg"
        ),
    }
    index["acceptance_gates"] = gates
    axes = dict(index.get("instance_axes") or {})
    axes["rigid_static_object"] = sorted(
        {key for record in records for key in record["realized_attributes"]}
    )
    index["instance_axes"] = axes
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        "PUBLISHED "
        + json.dumps(
            {"root": str(root), "assets": [record["asset_id"] for record in records]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
