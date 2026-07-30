from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    resolve_declared_path,
    sha256_file,
    write_json,
)
from avengine.m1.contracts import validate_room_manifest
from avengine.m3.contracts import (
    AcousticSceneContractError,
    load_and_validate_package,
    validate_canary_request,
    validate_mapping_document,
    validate_material_database_document,
)
from avengine.m3.gltf import ExpandedGltfScene, extract_triangle_scene
from avengine.m3.materials import MaterialContractError, compile_materials
from avengine.m3.materials import (
    MATERIAL_QUALIFICATION_CLAIM,
    controlled_counterfactual_errors,
    controlled_counterfactual_proof,
    production_admission_errors,
)
from avengine.m3.qa import (
    compiler_source_to_package_parity_report,
    geometry_report,
    material_coverage_report,
    ray_leakage_report,
    write_debug_obj,
)
from avengine.m3.rlr_material_import import (
    RLRMaterialImportError,
    compile_rlr_semantic_material_documents,
)
from avengine.m3.semantic import ExpandedSemanticScene, load_mp3d_semantic_scene
from avengine.m3.semantic_materials import (
    SemanticMaterialRuleError,
    SemanticSurfaceIdentity,
    compile_semantic_material_documents,
)
from avengine.m3.usd_snapshot import (
    UsdAcousticSnapshotError,
    load_usd_acoustic_snapshot,
)


COMPILER_VERSION = "1"


class AcousticSceneCompileError(ValueError):
    pass


def _snapshot_json(path: str | Path) -> tuple[Path, bytes, dict[str, Any], str]:
    resolved = Path(path).resolve()
    try:
        payload = resolved.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise AcousticSceneCompileError(f"unable to snapshot JSON {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise AcousticSceneCompileError(f"JSON input must be an object: {resolved}")
    return resolved, payload, value, hashlib.sha256(payload).hexdigest()


def _source_geometry_path(
    room_path: Path,
    room: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
) -> Path:
    matches = [
        asset
        for asset in room.get("assets", [])
        if isinstance(asset, Mapping) and asset.get("role") == "render_surface_mesh"
    ]
    if len(matches) != 1:
        raise AcousticSceneCompileError(
            "room must declare exactly one render_surface_mesh asset"
        )
    return resolve_declared_path(
        matches[0]["path"],
        manifest_dir=room_path.parent,
        environment=environment,
    )


def _source_asset_path(
    room_path: Path,
    room: Mapping[str, Any],
    *,
    role: str,
    environment: Mapping[str, str] | None = None,
) -> Path:
    matches = [
        asset
        for asset in room.get("assets", [])
        if isinstance(asset, Mapping) and asset.get("role") == role
    ]
    if len(matches) != 1:
        raise AcousticSceneCompileError(
            f"room must declare exactly one {role} asset"
        )
    return resolve_declared_path(
        matches[0]["path"],
        manifest_dir=room_path.parent,
        environment=environment,
    )


def _canonical_matrix(mapping: Mapping[str, Any]) -> np.ndarray:
    transform = mapping["source_to_canonical"]
    matrix = np.asarray(transform["matrix_row_major"], dtype=np.float64).reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise AcousticSceneCompileError("source_to_canonical matrix must be finite")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise AcousticSceneCompileError("source_to_canonical matrix must be affine")
    if abs(float(np.linalg.det(matrix[:3, :3]))) <= 1e-12:
        raise AcousticSceneCompileError("source_to_canonical matrix must be nonsingular")
    return matrix


def _apply_source_to_canonical(
    scene: ExpandedGltfScene, mapping: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    matrix = _canonical_matrix(mapping)
    source = scene.vertices.astype(np.float64, copy=False)
    homogeneous = np.concatenate(
        (source, np.ones((len(source), 1), dtype=np.float64)), axis=1
    )
    transformed = (matrix @ homogeneous.T).T[:, :3]
    vertices = np.ascontiguousarray(transformed, dtype="<f4")
    if not np.isfinite(vertices).all():
        raise AcousticSceneCompileError(
            "source_to_canonical transform overflows float32 package vertices"
        )
    triangles = np.ascontiguousarray(scene.triangles, dtype="<u4")
    if float(np.linalg.det(matrix[:3, :3])) < 0:
        triangles = triangles.copy()
        triangles[:, [1, 2]] = triangles[:, [2, 1]]
    return vertices, triangles


def _npy_record(path: Path, *, relative_to: Path, array: np.ndarray) -> dict[str, Any]:
    record = file_record(path, relative_to=relative_to)
    return {
        **record,
        "format": "npy",
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "memory_order": "C",
    }


def _json_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {**file_record(path, relative_to=relative_to), "format": "json"}


def _obj_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {**file_record(path, relative_to=relative_to), "format": "obj"}


def _compiler_identity() -> tuple[str, dict[str, str]]:
    module_directory = Path(__file__).resolve().parent
    components = {
        name: sha256_file(module_directory / name)
        for name in (
            "compiler.py",
            "contracts.py",
            "gltf.py",
            "materials.py",
            "qa.py",
            "rlr_material_import.py",
            "semantic.py",
            "semantic_materials.py",
            "usd_snapshot.py",
        )
    }
    return canonical_json_sha256(components), components


def _verify_input_snapshots(
    snapshots: list[tuple[Path, str]], *, geometry: tuple[Path, str]
) -> None:
    changed = [str(path) for path, expected in snapshots if sha256_file(path) != expected]
    geometry_path, geometry_hash = geometry
    if sha256_file(geometry_path) != geometry_hash:
        changed.append(str(geometry_path))
    if changed:
        raise AcousticSceneCompileError(
            "compiler input changed during snapshot-bound compilation: " + ", ".join(changed)
        )


def _build_explicit_glb_package(
    *,
    room_path: Path,
    room_bytes: bytes,
    room: dict[str, Any],
    room_sha256: str,
    mapping_path: Path,
    mapping_bytes: bytes,
    mapping: dict[str, Any],
    mapping_sha256: str,
    database_path: Path,
    database_bytes: bytes,
    database: dict[str, Any],
    database_sha256: str,
    output: Path,
    package_id: str,
    environment: Mapping[str, str] | None,
    expected_room_kind: str | None,
    package_mode: str,
    expected_material_semantics: str | None = None,
    source_scene: ExpandedGltfScene | ExpandedSemanticScene | None = None,
    source_geometry_path: Path | None = None,
    acoustic_profile_binding: Mapping[str, Any] | None = None,
    automatic_leakage_origins: Sequence[Sequence[float]] | None = None,
    automatic_leakage_direction_count: int = 32,
) -> Path:
    room_errors = validate_room_manifest(room)
    if room_errors:
        raise AcousticSceneCompileError("invalid source room: " + "; ".join(room_errors))
    if expected_room_kind is not None and room.get("room_kind") != expected_room_kind:
        raise AcousticSceneCompileError(
            f"compiler expected room_kind={expected_room_kind!r}, got "
            f"{room.get('room_kind')!r}"
        )
    if room.get("geometry_representation") != "real_surface_mesh":
        raise AcousticSceneCompileError("explicit GLB input must be a real surface mesh")
    mapping_errors = validate_mapping_document(mapping, room_id=room["room_id"])
    if mapping_errors:
        raise AcousticSceneCompileError("invalid material mapping: " + "; ".join(mapping_errors))
    database_errors = validate_material_database_document(database)
    if database_errors:
        raise AcousticSceneCompileError(
            "invalid acoustic material database: " + "; ".join(database_errors)
        )
    if package_mode == "production":
        admission_errors = production_admission_errors(
            mapping,
            database,
            expected_material_semantics=expected_material_semantics,
        )
        if admission_errors:
            raise AcousticSceneCompileError(
                "production material admission rejected: "
                + "; ".join(admission_errors)
            )
    try:
        compiled_materials = compile_materials(mapping, database, room_id=room["room_id"])
    except MaterialContractError as exc:
        raise AcousticSceneCompileError(str(exc)) from exc

    geometry_path = (
        source_geometry_path
        if source_geometry_path is not None
        else _source_geometry_path(room_path, room, environment=environment)
    )
    scene = source_scene if source_scene is not None else extract_triangle_scene(
        geometry_path
    )
    vertices, triangles = _apply_source_to_canonical(scene, mapping)
    used_source_materials = {
        str(item["source_material_name"]) for item in scene.objects
    }
    declared_source_materials = set(compiled_materials.source_material_to_id)
    if used_source_materials != declared_source_materials:
        raise AcousticSceneCompileError(
            "material mapping must exactly cover source GLB material names; "
            f"missing={sorted(used_source_materials - declared_source_materials)}, "
            f"unused={sorted(declared_source_materials - used_source_materials)}"
        )
    material_ids = np.full(
        len(triangles), np.iinfo(np.uint32).max, dtype="<u4"
    )
    for item in scene.objects:
        start = int(item["triangle_offset"])
        stop = start + int(item["triangle_count"])
        source_name = str(item["source_material_name"])
        if start < 0 or stop > len(material_ids):
            raise AcousticSceneCompileError(
                f"source object {item['object_id']} has an invalid triangle range"
            )
        material_ids[start:stop] = compiled_materials.source_material_to_id[
            source_name
        ]
    if np.any(material_ids == np.iinfo(np.uint32).max):
        raise AcousticSceneCompileError(
            "source object ranges do not assign every triangle material"
        )

    output.mkdir(parents=True, exist_ok=False)
    acoustic = output / "acoustic"
    provenance_directory = output / "provenance"
    qa_directory = output / "qa"
    acoustic.mkdir()
    provenance_directory.mkdir()
    qa_directory.mkdir()
    vertices_path = acoustic / "vertices.npy"
    triangles_path = acoustic / "triangles.npy"
    material_ids_path = acoustic / "triangle_material_ids.npy"
    np.save(vertices_path, vertices, allow_pickle=False)
    np.save(triangles_path, triangles, allow_pickle=False)
    np.save(material_ids_path, material_ids, allow_pickle=False)

    categories_path = acoustic / "material_categories.json"
    rlr_database_path = acoustic / "material_database.json"
    write_json(categories_path, compiled_materials.categories_document)
    write_json(rlr_database_path, compiled_materials.rlr_database)

    source_mapping_path = provenance_directory / "source_material_mapping.json"
    source_database_path = provenance_directory / "source_material_database.json"
    source_mapping_path.write_bytes(mapping_bytes)
    source_database_path.write_bytes(database_bytes)

    source_to_canonical = dict(mapping["source_to_canonical"])
    geometry_qa = geometry_report(
        vertices,
        triangles,
        source_sha256=scene.source_sha256,
        representation=room["geometry_representation"],
        source_to_canonical=source_to_canonical,
        objects=scene.objects,
    )
    coverage_qa = material_coverage_report(
        vertices,
        triangles,
        material_ids,
        compiled_materials.categories_document,
    )
    debug_path = qa_directory / "compiler_acoustic_mesh.obj"
    write_debug_obj(
        debug_path,
        vertices,
        triangles,
        material_ids,
        compiled_materials.categories_document,
        scene.objects,
    )
    emitted_vertices = np.load(vertices_path, allow_pickle=False)
    emitted_triangles = np.load(triangles_path, allow_pickle=False)
    parity_qa = compiler_source_to_package_parity_report(
        source_vertices=scene.vertices,
        source_triangles=scene.triangles,
        package_vertices=emitted_vertices,
        package_triangles=emitted_triangles,
        source_geometry_sha256=scene.source_sha256,
        openings=room["openings"],
        source_to_canonical=source_to_canonical,
        debug_obj_path=debug_path,
        declared_surface_audit=(
            room.get("surface_audit")
            if isinstance(room.get("surface_audit"), Mapping)
            else None
        ),
    )
    leakage_qa = ray_leakage_report(
        vertices,
        triangles,
        room["ray_checks"],
        automatic_origins=automatic_leakage_origins,
        automatic_direction_count=automatic_leakage_direction_count,
    )
    qa_values = {
        "geometry_report": geometry_qa,
        "material_coverage": coverage_qa,
        "ray_leakage": leakage_qa,
        "compiler_source_to_package_parity": parity_qa,
    }
    failed = [name for name, value in qa_values.items() if value["status"] != "pass"]
    if package_mode == "production" and failed:
        raise AcousticSceneCompileError(
            "custom production compiler QA did not pass: " + ", ".join(failed)
        )
    qa_paths: dict[str, Path] = {}
    for name, value in qa_values.items():
        qa_path = qa_directory / f"{name}.json"
        write_json(qa_path, value)
        qa_paths[name] = qa_path
    geometry_hash = scene.source_sha256
    compiler_sha256, compiler_components = _compiler_identity()
    manifest: dict[str, Any] = {
        "schema": "avengine_acoustic_scene_package_v1",
        "package_id": package_id,
        "package_mode": package_mode,
        "room_kind": room["room_kind"],
        "source_room": {
            "room_id": room["room_id"],
            "manifest_sha256": room_sha256,
            "source_revision": room["provenance"]["source_revision"],
            "geometry_asset_sha256": geometry_hash,
        },
        "coordinate_system": dict(room["coordinate_system"]),
        "unit_scale_to_m": 1.0,
        "geometry": {
            "representation": room["geometry_representation"],
            "transform_policy": "baked_to_canonical_world",
            "index_space": "global_vertex_array",
            "source_to_canonical": source_to_canonical,
            "vertex_count": int(len(vertices)),
            "triangle_count": int(len(triangles)),
            "source_primitive_count": scene.source_primitive_count,
            "source_node_instance_count": scene.source_node_instance_count,
        },
        "arrays": {
            "vertices": _npy_record(vertices_path, relative_to=output, array=vertices),
            "triangles": _npy_record(triangles_path, relative_to=output, array=triangles),
            "triangle_material_ids": _npy_record(
                material_ids_path, relative_to=output, array=material_ids
            ),
        },
        "objects": list(scene.objects),
        "materials": {
            "mapping_id": mapping["mapping_id"],
            "mapping_sha256": mapping_sha256,
            "mapping_source_kind": mapping["mapping_source_kind"],
            "database_id": database["database_id"],
            "database_source_sha256": database_sha256,
            "material_semantics": database["provenance"]["material_semantics"],
            "qualification_claim": MATERIAL_QUALIFICATION_CLAIM[
                database["provenance"]["material_semantics"]
            ],
            "category_count": len(compiled_materials.categories_document["categories"]),
            "source_mapping": _json_record(
                source_mapping_path, relative_to=output
            ),
            "source_database": _json_record(
                source_database_path, relative_to=output
            ),
            "categories": _json_record(categories_path, relative_to=output),
            "rlr_database": _json_record(rlr_database_path, relative_to=output),
        },
        "qa": {
            name: _json_record(path, relative_to=output)
            for name, path in qa_paths.items()
        },
        "debug_mesh": _obj_record(debug_path, relative_to=output),
        "compiler": {
            "name": "avengine.m3.compiler",
            "version": COMPILER_VERSION,
            "implementation_sha256": compiler_sha256,
            "components": compiler_components,
        },
    }
    if acoustic_profile_binding is not None:
        manifest["materials"]["acoustic_profile_binding"] = copy.deepcopy(
            dict(acoustic_profile_binding)
        )
    manifest["package_content_sha256"] = canonical_json_sha256(manifest)
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    _verify_input_snapshots(
        [
            (room_path, room_sha256),
            (mapping_path, mapping_sha256),
            (database_path, database_sha256),
        ],
        geometry=(geometry_path, scene.source_sha256),
    )
    try:
        load_and_validate_package(manifest_path)
    except AcousticSceneContractError as exc:
        raise AcousticSceneCompileError(
            "compiler produced an invalid package: " + "; ".join(exc.errors)
        ) from exc
    return manifest_path


def _staging_directory(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise AcousticSceneCompileError(f"output already exists: {output}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    ).resolve()
    staging.chmod(0o700)
    return staging


def compile_custom_acoustic_scene(
    *,
    room_manifest: str | Path,
    material_mapping: str | Path,
    material_database: str | Path,
    output: str | Path,
    package_id: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> Path:
    room_path, room_bytes, room, room_sha256 = _snapshot_json(room_manifest)
    mapping_path, mapping_bytes, mapping, mapping_sha256 = _snapshot_json(
        material_mapping
    )
    database_path, database_bytes, database, database_sha256 = _snapshot_json(
        material_database
    )
    destination = Path(output).resolve()
    staging = _staging_directory(destination)
    package_directory = staging / "package"
    try:
        manifest_path = _build_explicit_glb_package(
            room_path=room_path,
            room_bytes=room_bytes,
            room=room,
            room_sha256=room_sha256,
            mapping_path=mapping_path,
            mapping_bytes=mapping_bytes,
            mapping=mapping,
            mapping_sha256=mapping_sha256,
            database_path=database_path,
            database_bytes=database_bytes,
            database=database,
            database_sha256=database_sha256,
            output=package_directory,
            package_id=package_id
            or f"{room.get('room_id', 'unknown')}_{database.get('database_id', 'unknown')}",
            environment=environment,
            expected_room_kind="blender_custom",
            package_mode="production",
        )
        os.rename(package_directory, destination)
        shutil.rmtree(staging)
        return destination / manifest_path.name
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def compile_canary_request(
    request_path: str | Path,
    output: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> Path:
    request_file, request_bytes, request, request_sha256 = _snapshot_json(request_path)
    request_errors = validate_canary_request(request)
    if request_errors:
        raise AcousticSceneCompileError("invalid canary request: " + "; ".join(request_errors))
    effective_environment = dict(os.environ if environment is None else environment)
    effective_environment.setdefault(
        "AVENGINE_REPOSITORY_ROOT", str(Path(__file__).resolve().parents[3])
    )

    def request_input(raw_path: str) -> Path:
        return resolve_declared_path(
            raw_path,
            manifest_dir=request_file.parent,
            environment=effective_environment,
        )

    room_path, room_bytes, room, room_sha256 = _snapshot_json(
        request_input(request["room_manifest"])
    )
    mapping_path, mapping_bytes, mapping, mapping_sha256 = _snapshot_json(
        request_input(request["material_mapping"])
    )
    database_snapshots: dict[str, tuple[Path, bytes, dict[str, Any], str]] = {}
    for condition in ("low_absorption", "high_absorption"):
        database_snapshots[condition] = _snapshot_json(
            request_input(request["material_databases"][condition])
        )
    counterfactual_errors = controlled_counterfactual_errors(
        database_snapshots["low_absorption"][2],
        database_snapshots["high_absorption"][2],
    )
    if counterfactual_errors:
        raise AcousticSceneCompileError(
            "invalid absorption-only controlled counterfactual: "
            + "; ".join(counterfactual_errors)
        )
    material_counterfactual_proof = controlled_counterfactual_proof(
        database_snapshots["low_absorption"][2],
        database_snapshots["high_absorption"][2],
    )
    geometry_path = _source_geometry_path(
        room_path, room, environment=effective_environment
    )

    destination = Path(output).resolve()
    staging = _staging_directory(destination)
    try:
        inputs_directory = staging / "source_inputs"
        inputs_directory.mkdir()
        input_paths = {
            "request": inputs_directory / "canary_request.json",
            "room_manifest": inputs_directory / "room_manifest.json",
            "material_mapping": inputs_directory / "material_mapping.json",
            "low_database": inputs_directory / "materials_low.json",
            "high_database": inputs_directory / "materials_high.json",
            "source_geometry": inputs_directory / f"source_geometry{geometry_path.suffix}",
        }
        input_paths["request"].write_bytes(request_bytes)
        input_paths["room_manifest"].write_bytes(room_bytes)
        input_paths["material_mapping"].write_bytes(mapping_bytes)
        input_paths["low_database"].write_bytes(
            database_snapshots["low_absorption"][1]
        )
        input_paths["high_database"].write_bytes(
            database_snapshots["high_absorption"][1]
        )
        shutil.copyfile(geometry_path, input_paths["source_geometry"])
        package_manifests: dict[str, Path] = {}
        for condition in ("low_absorption", "high_absorption"):
            database_path, database_bytes, database, database_sha256 = (
                database_snapshots[condition]
            )
            package_manifests[condition] = _build_explicit_glb_package(
                room_path=room_path,
                room_bytes=room_bytes,
                room=room,
                room_sha256=room_sha256,
                mapping_path=mapping_path,
                mapping_bytes=mapping_bytes,
                mapping=mapping,
                mapping_sha256=mapping_sha256,
                database_path=database_path,
                database_bytes=database_bytes,
                database=database,
                database_sha256=database_sha256,
                output=staging / condition,
                package_id=f"{request['request_id']}_{condition}",
                environment=effective_environment,
                expected_room_kind="blender_custom",
                package_mode="production",
                expected_material_semantics="controlled_canary",
            )
        low = json.loads(package_manifests["low_absorption"].read_text(encoding="utf-8"))
        high = json.loads(package_manifests["high_absorption"].read_text(encoding="utf-8"))
        invariant_names = ("vertices", "triangles", "triangle_material_ids")
        invariants = {
            name: {
                "low_sha256": low["arrays"][name]["sha256"],
                "high_sha256": high["arrays"][name]["sha256"],
                "identical": low["arrays"][name]["sha256"]
                == high["arrays"][name]["sha256"],
            }
            for name in invariant_names
        }
        if not all(record["identical"] for record in invariants.values()):
            raise AcousticSceneCompileError(
                "low/high packages changed geometry or triangle material IDs"
            )
        evidence: dict[str, Any] = {
            "schema": "avengine_m3_compile_evidence_v1",
            "status": "pass",
            "request_id": request["request_id"],
            "request_sha256": request_sha256,
            "source_inputs": {
                name: file_record(path, relative_to=staging)
                for name, path in input_paths.items()
            },
            "packages": {
                condition: {
                    **file_record(path, relative_to=staging),
                    "package_content_sha256": json.loads(
                        path.read_text(encoding="utf-8")
                    )["package_content_sha256"],
                }
                for condition, path in package_manifests.items()
            },
            "frozen_variable_proof": invariants,
            "material_counterfactual_proof": material_counterfactual_proof,
            "runtime_material_activation": {
                "status": "not_run",
                "reason": (
                    "Compiler evidence is complete; RLR ingestion/RIR metrics are a "
                    "separate executable canary."
                ),
            },
        }
        evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
        evidence_path = staging / "compile_evidence.json"
        write_json(evidence_path, evidence)
        # The staging root is mode 0700.  Verify the completed bundle there
        # before publication so package manifests and source snapshots are
        # consumed from one immutable byte snapshot per file.
        from avengine.m3.evidence import load_and_verify_compile_evidence

        verified = load_and_verify_compile_evidence(evidence_path)
        if verified.status != "pass":
            failed_checks = [
                check["check_id"]
                for check in verified.checks
                if check.get("status") != "pass"
            ]
            raise AcousticSceneCompileError(
                "self-verification of compile evidence failed: "
                + ", ".join(failed_checks)
            )
        if sha256_file(request_file) != request_sha256:
            raise AcousticSceneCompileError("canary request changed during compilation")
        os.rename(staging, destination)
        return destination / evidence_path.name
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def compile_explicit_glb_research_scene(
    *,
    room_manifest: str | Path,
    material_mapping: str | Path,
    material_database: str | Path,
    output: str | Path,
    package_id: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Compile any M1 real-surface GLB as an explicitly non-admitted package.

    This path is how Habitat visual scenes and migrated UE surfaces exercise the
    same transform/material compiler before their acoustic mappings and opening
    rays are reviewed.  It never emits a production package.
    """

    room_path, room_bytes, room, room_sha256 = _snapshot_json(room_manifest)
    mapping_path, mapping_bytes, mapping, mapping_sha256 = _snapshot_json(
        material_mapping
    )
    database_path, database_bytes, database, database_sha256 = _snapshot_json(
        material_database
    )
    destination = Path(output).resolve()
    staging = _staging_directory(destination)
    package_directory = staging / "package"
    try:
        manifest_path = _build_explicit_glb_package(
            room_path=room_path,
            room_bytes=room_bytes,
            room=room,
            room_sha256=room_sha256,
            mapping_path=mapping_path,
            mapping_bytes=mapping_bytes,
            mapping=mapping,
            mapping_sha256=mapping_sha256,
            database_path=database_path,
            database_bytes=database_bytes,
            database=database,
            database_sha256=database_sha256,
            output=package_directory,
            package_id=package_id
            or f"{room.get('room_id', 'unknown')}_explicit_glb_research_v1",
            environment=environment,
            expected_room_kind=None,
            package_mode="research_candidate",
        )
        os.rename(package_directory, destination)
        shutil.rmtree(staging)
        return destination / manifest_path.name
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


_KNOWN_SOURCE_TRANSFORMS: dict[str, tuple[list[float], str]] = {
    "identity_y_up": (
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "Reviewed identity: source GLB is already right-handed +Y up, -Z forward, metres",
    ),
    "mp3d_z_up_y_front_to_habitat": (
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "Reviewed MP3D dataset-config stage frame: source +Z up/+Y front -> canonical +Y up/-Z front",
    ),
}


def _safe_category_token(source_name: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", source_name.lower()).strip("_")
    slug = slug[:24] or "unnamed"
    suffix = hashlib.sha256(source_name.encode("utf-8")).hexdigest()[:10]
    return f"avm3_slot_{index:03d}_{slug}_{suffix}"


def propose_visual_slot_research_materials(
    *,
    room_manifest: str | Path,
    output: str | Path,
    transform_profile: str,
    transform_reviewed: bool = False,
    environment: Mapping[str, str] | None = None,
) -> tuple[Path, Path, Path]:
    if transform_profile not in _KNOWN_SOURCE_TRANSFORMS:
        raise AcousticSceneCompileError(
            f"unknown transform profile {transform_profile!r}; expected one of "
            f"{sorted(_KNOWN_SOURCE_TRANSFORMS)}"
        )
    room_path, _room_bytes, room, room_sha256 = _snapshot_json(room_manifest)
    room_errors = validate_room_manifest(room)
    if room_errors:
        raise AcousticSceneCompileError("invalid source room: " + "; ".join(room_errors))
    geometry_path = _source_geometry_path(room_path, room, environment=environment)
    scene = extract_triangle_scene(geometry_path)
    material_names = sorted(set(scene.triangle_source_material_names))
    matrix, transform_source = _KNOWN_SOURCE_TRANSFORMS[transform_profile]
    entries: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    for index, source_name in enumerate(material_names):
        token = _safe_category_token(source_name, index)
        key = f"research_slot_{index:03d}_{hashlib.sha256(source_name.encode()).hexdigest()[:10]}"
        entries.append(
            {
                "source_material_name": source_name,
                "material_id": index,
                "category_name": token,
                "material_key": key,
                "mapping_source": (
                    "Unreviewed visual material-slot identity; this is not a "
                    "physical acoustic-material inference"
                ),
                "mapping_confidence": 0.0,
                "human_override": False,
                "randomized": False,
                "fallback": False,
            }
        )
        materials.append(
            {
                "material_key": key,
                "name": f"AVEngine unreviewed research slot {index:03d}",
                "labels": [token],
                "absorption": [0.2, 0.2, 0.2, 0.2],
                "scattering": [0.05, 0.05, 0.05, 0.05],
                "transmission": [0.0, 0.0, 0.0, 0.0],
                "damping": [0.0, 0.0, 0.0, 0.0],
                "density": 1.225,
                "speed": 343.0,
                "source": "Synthetic neutral research placeholder; no acoustic truth claim",
                "confidence": 0.0,
            }
        )
    mapping = {
        "schema": "avengine_m3_acoustic_material_mapping_v1",
        "mapping_id": f"{room['room_id']}_unreviewed_visual_slots_v1",
        "room_id": room["room_id"],
        "mapping_source_kind": "visual_material_slot_proposal",
        "source_to_canonical": {
            "matrix_row_major": matrix,
            "source": transform_source,
            "reviewed": bool(transform_reviewed),
        },
        "entries": entries,
    }
    database = {
        "schema": "avengine_m3_acoustic_material_database_v1",
        "database_id": f"{room['room_id']}_unreviewed_neutral_slots_v1",
        "version": "1",
        "bands_hz": [125.0, 500.0, 2000.0, 8000.0],
        "coefficient_units": {
            "absorption": "fraction_of_incident_sound_pressure",
            "scattering": "fraction_of_incident_sound_pressure",
            "transmission": "fraction_of_incident_sound_pressure",
            "damping": "decibels_per_meter",
            "density": "kilograms_per_cubic_meter",
            "speed": "meters_per_second",
        },
        "provenance": {
            "source": (
                "AVEngine generated neutral placeholder; visual PBR slots are not "
                "treated as acoustic truth"
            ),
            "confidence": 0.0,
            "material_semantics": "research_placeholder",
            "intended_use": "research_compiler_diagnostics",
        },
        "materials": materials,
    }
    destination = Path(output).resolve()
    staging = _staging_directory(destination)
    try:
        mapping_path = staging / "mapping.json"
        database_path = staging / "materials_research.json"
        report_path = staging / "proposal_report.json"
        write_json(mapping_path, mapping)
        write_json(database_path, database)
        report = {
            "schema": "avengine_m3_visual_slot_proposal_report_v1",
            "status": "research_candidate",
            "qualification_claim": False,
            "physical_acoustic_material_claim": False,
            "room_id": room["room_id"],
            "room_manifest_sha256": room_sha256,
            "source_geometry_sha256": scene.source_sha256,
            "source_material_slot_count": len(material_names),
            "source_material_names": material_names,
            "transform_profile": transform_profile,
            "source_to_canonical_reviewed": bool(transform_reviewed),
            "mapping_sha256": sha256_file(mapping_path),
            "database_sha256": sha256_file(database_path),
        }
        write_json(report_path, report)
        mapping_errors = validate_mapping_document(mapping, room_id=room["room_id"])
        database_errors = validate_material_database_document(database)
        if mapping_errors or database_errors:
            raise AcousticSceneCompileError(
                "generated research proposal is invalid: "
                + "; ".join([*mapping_errors, *database_errors])
            )
        if report.get("mapping_sha256") != sha256_file(mapping_path) or report.get(
            "database_sha256"
        ) != sha256_file(database_path):
            raise AcousticSceneCompileError("generated proposal report hashes are invalid")
        _verify_input_snapshots(
            [(room_path, room_sha256)], geometry=(geometry_path, scene.source_sha256)
        )
        os.rename(staging, destination)
        return (
            destination / mapping_path.name,
            destination / database_path.name,
            destination / report_path.name,
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _default_semantic_probe_origins(
    room: Mapping[str, Any],
) -> list[list[float]]:
    origins: list[list[float]] = []
    for pair in room.get("connectivity_pairs", []):
        if not isinstance(pair, Mapping):
            continue
        for field in ("start_m", "end_m"):
            value = pair.get(field)
            if (
                isinstance(value, list)
                and len(value) == 3
                and all(
                    not isinstance(item, bool) and isinstance(item, (int, float))
                    for item in value
                )
            ):
                candidate = [float(item) for item in value]
                if candidate not in origins:
                    origins.append(candidate)
            if len(origins) >= 2:
                return origins
    return origins


def compile_mp3d_semantic_research_scene(
    *,
    room_manifest: str | Path,
    material_rules: str | Path,
    output: str | Path,
    seed: int,
    package_id: str | None = None,
    probe_origins: Sequence[Sequence[float]] | None = None,
    probe_direction_count: int = 32,
    environment: Mapping[str, str] | None = None,
) -> tuple[Path, Path]:
    """Compile MP3D semantic PLY labels into the existing M3/RLR package.

    Candidate material selection is deterministic for ``seed`` and remains a
    research proposal until a separate physical calibration reviews the room.
    """

    room_path, room_bytes, room, room_sha256 = _snapshot_json(room_manifest)
    rules_path, _rules_bytes, rules, rules_sha256 = _snapshot_json(material_rules)
    room_errors = validate_room_manifest(room)
    if room_errors:
        raise AcousticSceneCompileError("invalid source room: " + "; ".join(room_errors))
    if room.get("room_kind") != "habitat_native":
        raise AcousticSceneCompileError(
            "MP3D semantic compilation requires room_kind='habitat_native'"
        )
    effective_environment = dict(os.environ if environment is None else environment)
    effective_environment.setdefault(
        "AVENGINE_REPOSITORY_ROOT", str(Path(__file__).resolve().parents[3])
    )
    semantic_path = _source_asset_path(
        room_path,
        room,
        role="semantic_surface_mesh",
        environment=effective_environment,
    )
    descriptor_path = _source_asset_path(
        room_path,
        room,
        role="semantic_descriptor",
        environment=effective_environment,
    )
    try:
        scene = load_mp3d_semantic_scene(semantic_path, descriptor_path)
        matrix, transform_source = _KNOWN_SOURCE_TRANSFORMS[
            "mp3d_z_up_y_front_to_habitat"
        ]
        surfaces = [
            SemanticSurfaceIdentity(
                source_material_name=category,
                semantic_category=category,
                identity_key=f"{room['room_id']}/{category}",
                object_name=category,
            )
            for category in scene.semantic_categories
        ]
        compiled = compile_semantic_material_documents(
            room_id=room["room_id"],
            surfaces=surfaces,
            rules=rules,
            seed=seed,
            source_to_canonical={
                "matrix_row_major": matrix,
                "source": transform_source,
                "reviewed": True,
            },
        )
    except (OSError, SemanticMaterialRuleError, ValueError) as exc:
        raise AcousticSceneCompileError(
            f"unable to compile MP3D semantic materials: {exc}"
        ) from exc
    mapping_errors = validate_mapping_document(
        compiled.mapping, room_id=room["room_id"]
    )
    database_errors = validate_material_database_document(compiled.database)
    if mapping_errors or database_errors:
        raise AcousticSceneCompileError(
            "generated semantic material documents are invalid: "
            + "; ".join([*mapping_errors, *database_errors])
        )
    effective_probe_origins = (
        [list(map(float, origin)) for origin in probe_origins]
        if probe_origins is not None
        else _default_semantic_probe_origins(room)
    )

    destination = Path(output).resolve()
    staging = _staging_directory(destination)
    package_directory = staging / "package"
    generated_inputs = staging / "generated_inputs"
    generated_inputs.mkdir()
    mapping_path = generated_inputs / "mapping.json"
    database_path = generated_inputs / "materials_research.json"
    write_json(mapping_path, compiled.mapping)
    write_json(database_path, compiled.database)
    mapping_bytes = mapping_path.read_bytes()
    database_bytes = database_path.read_bytes()
    mapping_sha256 = sha256_file(mapping_path)
    database_sha256 = sha256_file(database_path)
    descriptor_sha256 = sha256_file(descriptor_path)
    try:
        manifest_path = _build_explicit_glb_package(
            room_path=room_path,
            room_bytes=room_bytes,
            room=room,
            room_sha256=room_sha256,
            mapping_path=mapping_path,
            mapping_bytes=mapping_bytes,
            mapping=compiled.mapping,
            mapping_sha256=mapping_sha256,
            database_path=database_path,
            database_bytes=database_bytes,
            database=compiled.database,
            database_sha256=database_sha256,
            output=package_directory,
            package_id=package_id
            or f"{room['room_id']}_semantic_seed{seed}_research_v1",
            environment=effective_environment,
            expected_room_kind=None,
            package_mode="research_candidate",
            source_scene=scene,
            source_geometry_path=semantic_path,
            automatic_leakage_origins=effective_probe_origins,
            automatic_leakage_direction_count=probe_direction_count,
        )
        report = copy.deepcopy(compiled.report)
        report.update(
            {
                "source_kind": "mp3d_semantic_ply_house",
                "semantic_mesh_path": str(semantic_path),
                "semantic_mesh_sha256": scene.source_sha256,
                "semantic_descriptor_path": str(descriptor_path),
                "semantic_descriptor_sha256": scene.descriptor_sha256,
                "source_vertex_count": scene.source_vertex_count,
                "source_triangle_count": scene.source_triangle_count,
                "compiled_vertex_count": int(len(scene.vertices)),
                "compiled_triangle_count": int(len(scene.triangles)),
                "category_triangle_counts": scene.category_triangle_counts,
                "automatic_leakage_probe_origins_m": effective_probe_origins,
                "automatic_leakage_direction_count": probe_direction_count,
            }
        )
        report_path = package_directory / "semantic_material_coverage.json"
        write_json(report_path, report)
        changed = []
        if sha256_file(rules_path) != rules_sha256:
            changed.append(str(rules_path))
        if sha256_file(descriptor_path) != descriptor_sha256:
            changed.append(str(descriptor_path))
        if changed:
            raise AcousticSceneCompileError(
                "semantic compiler input changed during compilation: "
                + ", ".join(changed)
            )
        os.rename(package_directory, destination)
        shutil.rmtree(staging)
        return destination / manifest_path.name, destination / report_path.name
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def compile_mp3d_soundspaces_research_scene(
    *,
    room_manifest: str | Path,
    material_config: str | Path,
    output: str | Path,
    database_id: str,
    source_description: str,
    version: str = "1",
    source_uri: str | None = None,
    package_id: str | None = None,
    probe_origins: Sequence[Sequence[float]] | None = None,
    probe_direction_count: int = 32,
    environment: Mapping[str, str] | None = None,
) -> tuple[Path, Path]:
    """Compile official SoundSpaces/RLR MP3D materials into an M3 package.

    MP3D semantic category tokens are resolved by faithfully replaying the
    public RLR greatest-label-substring-match-count rule.  A zero-score token
    is assigned to that config's official ``Default`` material; a highest-score
    tie fails closed.  Every resolved winner then receives a generated exact
    runtime alias.  This keeps native ingestion fail-closed while the returned
    coverage report preserves the official selection evidence.
    """

    room_path, room_bytes, room, room_sha256 = _snapshot_json(room_manifest)
    (
        material_config_path,
        _material_config_bytes,
        material_config_document,
        material_config_sha256,
    ) = _snapshot_json(material_config)
    if source_uri is not None and (
        not isinstance(source_uri, str) or not source_uri
    ):
        raise AcousticSceneCompileError("source_uri must be a non-empty string")
    room_errors = validate_room_manifest(room)
    if room_errors:
        raise AcousticSceneCompileError("invalid source room: " + "; ".join(room_errors))
    if room.get("room_kind") != "habitat_native":
        raise AcousticSceneCompileError(
            "SoundSpaces MP3D compilation requires room_kind='habitat_native'"
        )
    effective_environment = dict(os.environ if environment is None else environment)
    effective_environment.setdefault(
        "AVENGINE_REPOSITORY_ROOT", str(Path(__file__).resolve().parents[3])
    )
    semantic_path = _source_asset_path(
        room_path,
        room,
        role="semantic_surface_mesh",
        environment=effective_environment,
    )
    descriptor_path = _source_asset_path(
        room_path,
        room,
        role="semantic_descriptor",
        environment=effective_environment,
    )
    try:
        scene = load_mp3d_semantic_scene(semantic_path, descriptor_path)
        matrix, transform_source = _KNOWN_SOURCE_TRANSFORMS[
            "mp3d_z_up_y_front_to_habitat"
        ]
        source_identity = (
            f"{source_description}; source_sha256={material_config_sha256}"
        )
        if source_uri is not None:
            source_identity += f"; source_uri={source_uri}"
        compiled = compile_rlr_semantic_material_documents(
            room_id=room["room_id"],
            semantic_categories=scene.semantic_categories,
            raw_semantic_category_labels=scene.raw_semantic_category_labels,
            source=material_config_document,
            database_id=database_id,
            version=version,
            source_description=source_identity,
            source_to_canonical={
                "matrix_row_major": matrix,
                "source": transform_source,
                "reviewed": True,
            },
        )
    except (OSError, RLRMaterialImportError, ValueError) as exc:
        raise AcousticSceneCompileError(
            f"unable to compile SoundSpaces/RLR MP3D materials: {exc}"
        ) from exc

    mapping_errors = validate_mapping_document(
        compiled.mapping, room_id=room["room_id"]
    )
    database_errors = validate_material_database_document(compiled.database)
    if mapping_errors or database_errors:
        raise AcousticSceneCompileError(
            "generated SoundSpaces/RLR material documents are invalid: "
            + "; ".join([*mapping_errors, *database_errors])
        )
    effective_probe_origins = (
        [list(map(float, origin)) for origin in probe_origins]
        if probe_origins is not None
        else _default_semantic_probe_origins(room)
    )

    destination = Path(output).resolve()
    staging = _staging_directory(destination)
    package_directory = staging / "package"
    generated_inputs = staging / "generated_inputs"
    generated_inputs.mkdir()
    mapping_path = generated_inputs / "mapping.json"
    database_path = generated_inputs / "materials_soundspaces_rlr.json"
    write_json(mapping_path, compiled.mapping)
    write_json(database_path, compiled.database)
    mapping_bytes = mapping_path.read_bytes()
    database_bytes = database_path.read_bytes()
    mapping_sha256 = sha256_file(mapping_path)
    database_sha256 = sha256_file(database_path)
    descriptor_sha256 = sha256_file(descriptor_path)
    try:
        manifest_path = _build_explicit_glb_package(
            room_path=room_path,
            room_bytes=room_bytes,
            room=room,
            room_sha256=room_sha256,
            mapping_path=mapping_path,
            mapping_bytes=mapping_bytes,
            mapping=compiled.mapping,
            mapping_sha256=mapping_sha256,
            database_path=database_path,
            database_bytes=database_bytes,
            database=compiled.database,
            database_sha256=database_sha256,
            output=package_directory,
            package_id=package_id
            or f"{room['room_id']}_soundspaces_rlr_research_v1",
            environment=effective_environment,
            expected_room_kind="habitat_native",
            package_mode="research_candidate",
            source_scene=scene,
            source_geometry_path=semantic_path,
            acoustic_profile_binding={
                "schema": "avengine_m3_acoustic_profile_binding_v1",
                "profile_id": database_id,
                "profile_revision": version,
                "adapter_id": "soundspaces2_mp3d_semantic_labels_v1",
                "resources": [
                    {
                        "role": "soundspaces2_public_material_config",
                        "sha256": material_config_sha256,
                    }
                ],
            },
            automatic_leakage_origins=effective_probe_origins,
            automatic_leakage_direction_count=probe_direction_count,
        )
        report = copy.deepcopy(compiled.report)
        decisions_by_label = {
            decision["source_semantic_label"]: decision
            for decision in report["decisions"]
        }
        substring_triangles = 0
        default_triangles = 0
        for category, triangle_count in scene.category_triangle_counts.items():
            decision = decisions_by_label[category]
            decision["triangle_count"] = triangle_count
            if decision["official_default_applied"]:
                default_triangles += triangle_count
            else:
                substring_triangles += triangle_count
        triangle_count = int(len(scene.triangles))
        report.update(
            {
                "source_kind": "soundspaces_rlr_mp3d_semantic_ply_house",
                "source_material_config": {
                    "path": str(material_config_path),
                    "byte_size": len(_material_config_bytes),
                    "sha256": material_config_sha256,
                    "canonical_sha256": canonical_json_sha256(
                        material_config_document
                    ),
                    **({"uri": source_uri} if source_uri is not None else {}),
                },
                "semantic_mesh_path": str(semantic_path),
                "semantic_mesh_sha256": scene.source_sha256,
                "semantic_descriptor_path": str(descriptor_path),
                "semantic_descriptor_sha256": scene.descriptor_sha256,
                "source_vertex_count": scene.source_vertex_count,
                "source_triangle_count": scene.source_triangle_count,
                "compiled_vertex_count": int(len(scene.vertices)),
                "compiled_triangle_count": triangle_count,
                "category_triangle_counts": scene.category_triangle_counts,
                "triangle_coverage": {
                    "triangle_count": triangle_count,
                    "official_substring_match_triangle_count": (
                        substring_triangles
                    ),
                    "official_default_triangle_count": default_triangles,
                    "official_substring_match_triangle_fraction": (
                        substring_triangles / triangle_count
                    ),
                    "official_default_triangle_fraction": (
                        default_triangles / triangle_count
                    ),
                },
                "automatic_leakage_probe_origins_m": effective_probe_origins,
                "automatic_leakage_direction_count": probe_direction_count,
            }
        )
        report_path = package_directory / "semantic_material_coverage.json"
        write_json(report_path, report)
        changed = []
        if sha256_file(material_config_path) != material_config_sha256:
            changed.append(str(material_config_path))
        if sha256_file(descriptor_path) != descriptor_sha256:
            changed.append(str(descriptor_path))
        if changed:
            raise AcousticSceneCompileError(
                "SoundSpaces material compiler input changed during compilation: "
                + ", ".join(changed)
            )
        os.rename(package_directory, destination)
        shutil.rmtree(staging)
        return destination / manifest_path.name, destination / report_path.name
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def compile_usd_snapshot_semantic_research_scene(
    *,
    room_manifest: str | Path,
    material_rules: str | Path,
    output: str | Path,
    seed: int,
    package_id: str | None = None,
    probe_origins: Sequence[Sequence[float]] | None = None,
    probe_direction_count: int = 32,
    environment: Mapping[str, str] | None = None,
) -> tuple[Path, Path]:
    """Compile a strictly extracted USD snapshot into the existing M3 package.

    USD composition remains an optional offline authoring step.  This function
    consumes one exact NPZ snapshot, resolves its object/material identities
    through the shared residential rules, and emits the same Acoustic Scene
    Package used by GLB and MP3D.  The result remains a research candidate.
    """

    room_path, room_bytes, room, room_sha256 = _snapshot_json(room_manifest)
    rules_path, _rules_bytes, rules, rules_sha256 = _snapshot_json(material_rules)
    room_errors = validate_room_manifest(room)
    if room_errors:
        raise AcousticSceneCompileError("invalid source room: " + "; ".join(room_errors))
    if room.get("room_kind") != "external_usd_real_surface":
        raise AcousticSceneCompileError(
            "USD snapshot compilation requires "
            "room_kind='external_usd_real_surface'"
        )
    effective_environment = dict(os.environ if environment is None else environment)
    effective_environment.setdefault(
        "AVENGINE_REPOSITORY_ROOT", str(Path(__file__).resolve().parents[3])
    )
    snapshot_path = _source_geometry_path(
        room_path,
        room,
        environment=effective_environment,
    )
    try:
        loaded = load_usd_acoustic_snapshot(snapshot_path)
        if loaded.metadata.get("room_id") != room["room_id"]:
            raise AcousticSceneCompileError(
                "USD snapshot room_id differs from its room manifest"
            )
        source_to_canonical = copy.deepcopy(
            loaded.metadata["source_to_canonical"]
        )
        compiled = compile_semantic_material_documents(
            room_id=room["room_id"],
            surfaces=loaded.surfaces,
            rules=rules,
            seed=seed,
            source_to_canonical=source_to_canonical,
        )
    except (
        OSError,
        SemanticMaterialRuleError,
        UsdAcousticSnapshotError,
        ValueError,
    ) as exc:
        if isinstance(exc, AcousticSceneCompileError):
            raise
        raise AcousticSceneCompileError(
            f"unable to compile USD snapshot semantic materials: {exc}"
        ) from exc
    mapping_errors = validate_mapping_document(
        compiled.mapping, room_id=room["room_id"]
    )
    database_errors = validate_material_database_document(compiled.database)
    if mapping_errors or database_errors:
        raise AcousticSceneCompileError(
            "generated USD semantic material documents are invalid: "
            + "; ".join([*mapping_errors, *database_errors])
        )
    if probe_origins is None:
        raw_origins = loaded.metadata.get("reviewed_interior_origins_m", [])
        effective_probe_origins = [
            [float(value) for value in origin] for origin in raw_origins
        ]
    else:
        effective_probe_origins = [
            [float(value) for value in origin] for origin in probe_origins
        ]
    if not effective_probe_origins:
        raise AcousticSceneCompileError(
            "USD snapshot compilation requires reviewed interior probe origins"
        )

    destination = Path(output).resolve()
    staging = _staging_directory(destination)
    package_directory = staging / "package"
    generated_inputs = staging / "generated_inputs"
    generated_inputs.mkdir()
    mapping_path = generated_inputs / "mapping.json"
    database_path = generated_inputs / "materials_research.json"
    write_json(mapping_path, compiled.mapping)
    write_json(database_path, compiled.database)
    mapping_bytes = mapping_path.read_bytes()
    database_bytes = database_path.read_bytes()
    mapping_sha256 = sha256_file(mapping_path)
    database_sha256 = sha256_file(database_path)
    try:
        manifest_path = _build_explicit_glb_package(
            room_path=room_path,
            room_bytes=room_bytes,
            room=room,
            room_sha256=room_sha256,
            mapping_path=mapping_path,
            mapping_bytes=mapping_bytes,
            mapping=compiled.mapping,
            mapping_sha256=mapping_sha256,
            database_path=database_path,
            database_bytes=database_bytes,
            database=compiled.database,
            database_sha256=database_sha256,
            output=package_directory,
            package_id=package_id
            or f"{room['room_id']}_usd_semantic_seed{seed}_research_v1",
            environment=effective_environment,
            expected_room_kind="external_usd_real_surface",
            package_mode="research_candidate",
            source_scene=loaded.scene,
            source_geometry_path=snapshot_path,
            automatic_leakage_origins=effective_probe_origins,
            automatic_leakage_direction_count=probe_direction_count,
        )
        report = copy.deepcopy(compiled.report)
        resolution_counts: dict[str, int] = {}
        for decision in report["decisions"]:
            resolution = decision["resolution"]
            resolution_counts[resolution] = resolution_counts.get(resolution, 0) + 1
        report.update(
            {
                "source_kind": "composed_usd_acoustic_snapshot",
                "source_stage": loaded.metadata.get("source_stage"),
                "source_stage_sha256": loaded.metadata.get("source_stage_sha256"),
                "snapshot_path": str(snapshot_path),
                "snapshot_sha256": loaded.scene.source_sha256,
                "source_mesh_prim_count": loaded.metadata.get(
                    "source_mesh_prim_count"
                ),
                "visible_mesh_prim_count": loaded.metadata.get(
                    "visible_mesh_prim_count"
                ),
                "hidden_mesh_prim_count": loaded.metadata.get(
                    "hidden_mesh_prim_count"
                ),
                "compiled_object_partition_count": len(loaded.scene.objects),
                "compiled_vertex_count": int(len(loaded.scene.vertices)),
                "compiled_triangle_count": int(len(loaded.scene.triangles)),
                "surface_identity_count": len(loaded.surfaces),
                "semantic_category_triangle_counts": (
                    loaded.scene.category_triangle_counts
                ),
                "resolution_counts": dict(sorted(resolution_counts.items())),
                "automatic_leakage_probe_origins_m": effective_probe_origins,
                "automatic_leakage_direction_count": probe_direction_count,
                "geometry_claim": loaded.metadata.get("geometry_claim"),
                "physical_material_claim": False,
                "hole_repair": "not_performed",
            }
        )
        report_path = package_directory / "semantic_material_coverage.json"
        write_json(report_path, report)
        if sha256_file(rules_path) != rules_sha256:
            raise AcousticSceneCompileError(
                "semantic material rules changed during USD snapshot compilation"
            )
        os.rename(package_directory, destination)
        shutil.rmtree(staging)
        return destination / manifest_path.name, destination / report_path.name
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def compile_visual_slot_semantic_research_scene(
    *,
    room_manifest: str | Path,
    material_rules: str | Path,
    output: str | Path,
    seed: int,
    transform_profile: str,
    transform_reviewed: bool = False,
    package_id: str | None = None,
    probe_origins: Sequence[Sequence[float]] | None = None,
    probe_direction_count: int = 32,
    environment: Mapping[str, str] | None = None,
    acoustic_profile_binding: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Resolve visual material-slot names through semantic rules into an M3 package.

    Visual slots carry no reviewed semantic category, so every surface resolves
    through explicit overrides, name hints or the declared default candidates.
    The result is the same uncalibrated research proposal contract as the MP3D
    and USD semantic routes; it never claims physical material truth.
    """

    if transform_profile not in _KNOWN_SOURCE_TRANSFORMS:
        raise AcousticSceneCompileError(
            f"unknown transform profile {transform_profile!r}; expected one of "
            f"{sorted(_KNOWN_SOURCE_TRANSFORMS)}"
        )
    room_path, room_bytes, room, room_sha256 = _snapshot_json(room_manifest)
    rules_path, _rules_bytes, rules, rules_sha256 = _snapshot_json(material_rules)
    room_errors = validate_room_manifest(room)
    if room_errors:
        raise AcousticSceneCompileError("invalid source room: " + "; ".join(room_errors))
    effective_environment = dict(os.environ if environment is None else environment)
    effective_environment.setdefault(
        "AVENGINE_REPOSITORY_ROOT", str(Path(__file__).resolve().parents[3])
    )
    geometry_path = _source_geometry_path(
        room_path, room, environment=effective_environment
    )
    try:
        scene = extract_triangle_scene(geometry_path)
        material_names = sorted(set(scene.triangle_source_material_names))
        matrix, transform_source = _KNOWN_SOURCE_TRANSFORMS[transform_profile]
        surfaces = [
            SemanticSurfaceIdentity(
                source_material_name=name,
                semantic_category="",
                identity_key=f"{room['room_id']}/{name}",
                material_slot=name,
            )
            for name in material_names
        ]
        compiled = compile_semantic_material_documents(
            room_id=room["room_id"],
            surfaces=surfaces,
            rules=rules,
            seed=seed,
            source_to_canonical={
                "matrix_row_major": matrix,
                "source": transform_source,
                "reviewed": bool(transform_reviewed),
            },
        )
    except (OSError, SemanticMaterialRuleError, ValueError) as exc:
        raise AcousticSceneCompileError(
            f"unable to compile visual-slot semantic materials: {exc}"
        ) from exc
    mapping_errors = validate_mapping_document(
        compiled.mapping, room_id=room["room_id"]
    )
    database_errors = validate_material_database_document(compiled.database)
    if mapping_errors or database_errors:
        raise AcousticSceneCompileError(
            "generated semantic material documents are invalid: "
            + "; ".join([*mapping_errors, *database_errors])
        )
    effective_probe_origins = (
        [list(map(float, origin)) for origin in probe_origins]
        if probe_origins is not None
        else _default_semantic_probe_origins(room)
    )

    destination = Path(output).resolve()
    staging = _staging_directory(destination)
    package_directory = staging / "package"
    generated_inputs = staging / "generated_inputs"
    generated_inputs.mkdir()
    mapping_path = generated_inputs / "mapping.json"
    database_path = generated_inputs / "materials_research.json"
    write_json(mapping_path, compiled.mapping)
    write_json(database_path, compiled.database)
    mapping_bytes = mapping_path.read_bytes()
    database_bytes = database_path.read_bytes()
    mapping_sha256 = sha256_file(mapping_path)
    database_sha256 = sha256_file(database_path)
    try:
        manifest_path = _build_explicit_glb_package(
            room_path=room_path,
            room_bytes=room_bytes,
            room=room,
            room_sha256=room_sha256,
            mapping_path=mapping_path,
            mapping_bytes=mapping_bytes,
            mapping=compiled.mapping,
            mapping_sha256=mapping_sha256,
            database_path=database_path,
            database_bytes=database_bytes,
            database=compiled.database,
            database_sha256=database_sha256,
            output=package_directory,
            package_id=package_id
            or f"{room['room_id']}_visual_slot_semantic_seed{seed}_research_v1",
            environment=effective_environment,
            expected_room_kind=None,
            package_mode="research_candidate",
            source_scene=scene,
            source_geometry_path=geometry_path,
            acoustic_profile_binding=acoustic_profile_binding,
            automatic_leakage_origins=effective_probe_origins,
            automatic_leakage_direction_count=probe_direction_count,
        )
        report = copy.deepcopy(compiled.report)
        resolution_counts: dict[str, int] = {}
        for decision in report.get("decisions", []):
            key = str(decision.get("resolution"))
            resolution_counts[key] = resolution_counts.get(key, 0) + 1
        report.update(
            {
                "source_kind": "visual_material_slots",
                "source_geometry_sha256": scene.source_sha256,
                "source_material_slot_count": len(material_names),
                "source_material_names": material_names,
                "transform_profile": transform_profile,
                "source_to_canonical_reviewed": bool(transform_reviewed),
                "compiled_vertex_count": int(len(scene.vertices)),
                "compiled_triangle_count": int(len(scene.triangles)),
                "resolution_counts": dict(sorted(resolution_counts.items())),
                "automatic_leakage_probe_origins_m": effective_probe_origins,
                "automatic_leakage_direction_count": probe_direction_count,
                "physical_material_claim": False,
            }
        )
        report_path = package_directory / "semantic_material_coverage.json"
        write_json(report_path, report)
        if sha256_file(rules_path) != rules_sha256:
            raise AcousticSceneCompileError(
                "semantic material rules changed during visual-slot compilation"
            )
        os.rename(package_directory, destination)
        shutil.rmtree(staging)
        return destination / manifest_path.name, destination / report_path.name
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
