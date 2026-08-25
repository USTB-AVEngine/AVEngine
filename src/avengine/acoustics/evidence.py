from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from avengine.contracts.json_io import canonical_json_sha256
from avengine.rooms.contracts import validate_room_manifest
from avengine.acoustics.compiler import _apply_source_to_canonical
from avengine.acoustics.contracts import (
    COMPILE_EVIDENCE_SCHEMA,
    AcousticSceneContractError,
    ImmutableFileSnapshot,
    ValidatedAcousticScenePackage,
    json_schema_errors,
    load_and_validate_acoustic_scene_package,
    read_immutable_file_snapshot,
    validate_canary_request,
)
from avengine.acoustics.gltf import extract_triangle_scene_bytes
from avengine.acoustics.materials import (
    MaterialContractError,
    compile_materials,
    controlled_counterfactual_proof,
)


@dataclass(frozen=True)
class VerifiedCompileEvidence:
    evidence_path: Path
    evidence_snapshot: ImmutableFileSnapshot | None
    evidence: dict[str, Any]
    status: str
    checks: tuple[dict[str, Any], ...]
    packages: dict[str, ValidatedAcousticScenePackage]


def _check(
    check_id: str,
    passed: bool,
    *,
    measured: Any,
    threshold: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "status": "pass" if passed else "fail",
        "measured": measured,
        "threshold": threshold,
        **({} if passed else {"failure_reason": failure_reason}),
    }


def _confined_snapshot(
    base: Path,
    record: Any,
    *,
    cache: dict[Path, ImmutableFileSnapshot],
) -> tuple[ImmutableFileSnapshot | None, str | None]:
    if not isinstance(record, Mapping):
        return None, "record must be an object"
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None, "path is missing"
    declared = Path(raw_path)
    if declared.is_absolute() or ".." in declared.parts:
        return None, "path is absolute or escaping"
    path = (base / declared).resolve()
    try:
        path.relative_to(base)
    except ValueError:
        return None, "path or symlink escapes"
    try:
        snapshot = read_immutable_file_snapshot(path, cache=cache)
    except OSError as exc:
        return None, f"file is missing or unreadable: {exc}"
    if snapshot.byte_size != record.get("byte_size"):
        return None, "byte_size mismatch"
    if snapshot.sha256 != record.get("sha256"):
        return None, "sha256 mismatch"
    return snapshot, None


def _json_snapshot_object(
    snapshot: ImmutableFileSnapshot, owner: str
) -> dict[str, Any]:
    try:
        value = json.loads(snapshot.payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{owner} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{owner} JSON root must be an object")
    return value


def load_and_verify_compile_evidence(
    evidence_path: str | Path,
    *,
    evidence_snapshot: ImmutableFileSnapshot | None = None,
    snapshot_cache: dict[Path, ImmutableFileSnapshot] | None = None,
) -> VerifiedCompileEvidence:
    path = Path(evidence_path).resolve()
    cache = snapshot_cache if snapshot_cache is not None else {}
    try:
        if evidence_snapshot is None:
            evidence_snapshot = read_immutable_file_snapshot(path, cache=cache)
        elif evidence_snapshot.path.resolve() != path:
            raise ValueError(
                "provided compile-evidence snapshot path does not match evidence_path"
            )
        else:
            cached = cache.get(path)
            if cached is not None and cached.payload != evidence_snapshot.payload:
                raise ValueError(
                    "snapshot cache contains conflicting compile-evidence bytes"
                )
            cache[path] = evidence_snapshot
        evidence = _json_snapshot_object(evidence_snapshot, "compile evidence")
    except (OSError, ValueError) as exc:
        return VerifiedCompileEvidence(
            evidence_path=path,
            evidence_snapshot=evidence_snapshot,
            evidence={},
            status="fail",
            checks=(
                _check(
                    "compile_evidence_snapshot",
                    False,
                    measured=str(exc),
                    threshold="single readable JSON byte snapshot",
                    failure_reason=(
                        "Compile evidence cannot be read and parsed atomically"
                    ),
                ),
            ),
            packages={},
        )
    checks: list[dict[str, Any]] = []
    schema_errors = json_schema_errors(evidence, COMPILE_EVIDENCE_SCHEMA)
    checks.append(
        _check(
            "compile_evidence_schema",
            not schema_errors,
            measured=schema_errors,
            threshold=[],
            failure_reason="Compile evidence does not satisfy its JSON Schema",
        )
    )
    try:
        content_hash = canonical_json_sha256(
            {
                key: value
                for key, value in evidence.items()
                if key != "evidence_content_sha256"
            }
        )
    except (TypeError, ValueError):
        content_hash = None
    checks.append(
        _check(
            "compile_evidence_content_hash",
            content_hash == evidence.get("evidence_content_sha256"),
            measured=content_hash,
            threshold=evidence.get("evidence_content_sha256"),
            failure_reason="Compile evidence canonical content hash changed",
        )
    )

    base = path.parent.resolve()
    source_input_names = (
        "request",
        "room_manifest",
        "material_mapping",
        "low_database",
        "high_database",
        "source_geometry",
    )
    source_snapshots: dict[str, ImmutableFileSnapshot] = {}
    source_record_errors: list[str] = []
    source_records = evidence.get("source_inputs", {})
    for name in source_input_names:
        record = source_records.get(name, {}) if isinstance(source_records, Mapping) else {}
        source_snapshot, snapshot_error = _confined_snapshot(
            base, record, cache=cache
        )
        if source_snapshot is None:
            source_record_errors.append(f"{name}: {snapshot_error}")
        else:
            source_snapshots[name] = source_snapshot
    checks.append(
        _check(
            "compile_source_input_records",
            len(source_snapshots) == len(source_input_names) and not source_record_errors,
            measured=source_record_errors,
            threshold=[],
            failure_reason="Compiler source-input snapshots are missing, changed, or escaping",
        )
    )

    request: dict[str, Any] | None = None
    room: dict[str, Any] | None = None
    mapping: dict[str, Any] | None = None
    source_databases: dict[str, dict[str, Any]] = {}
    source_scene = None
    source_replay_errors: list[str] = []
    if len(source_snapshots) == len(source_input_names):
        try:
            request = _json_snapshot_object(source_snapshots["request"], "request")
            room = _json_snapshot_object(
                source_snapshots["room_manifest"], "room_manifest"
            )
            mapping = _json_snapshot_object(
                source_snapshots["material_mapping"], "material_mapping"
            )
            source_databases = {
                "low_absorption": _json_snapshot_object(
                    source_snapshots["low_database"], "low_database"
                ),
                "high_absorption": _json_snapshot_object(
                    source_snapshots["high_database"], "high_database"
                ),
            }
            source_scene = extract_triangle_scene_bytes(
                source_snapshots["source_geometry"].payload,
                source_path=source_snapshots["source_geometry"].path,
            )
        except (OSError, TypeError, ValueError) as exc:
            source_replay_errors.append(str(exc))
    if request is not None:
        source_replay_errors.extend(validate_canary_request(request))
        if source_snapshots["request"].sha256 != evidence.get("request_sha256"):
            source_replay_errors.append("request snapshot hash differs from evidence")
        if request.get("request_id") != evidence.get("request_id"):
            source_replay_errors.append("request_id differs from evidence")
    if room is not None:
        source_replay_errors.extend(validate_room_manifest(room))
    checks.append(
        _check(
            "compile_source_input_contracts",
            bool(
                request is not None
                and room is not None
                and mapping is not None
                and len(source_databases) == 2
                and source_scene is not None
                and not source_replay_errors
            ),
            measured=source_replay_errors,
            threshold=[],
            failure_reason="Hash-bound compiler source inputs cannot be parsed or validated",
        )
    )

    packages: dict[str, Any] = {}
    for condition in ("low_absorption", "high_absorption"):
        record = evidence.get("packages", {}).get(condition, {})
        manifest_snapshot, manifest_snapshot_error = _confined_snapshot(
            base, record, cache=cache
        )
        record_passed = manifest_snapshot is not None
        package_error: str | None = None
        if record_passed and manifest_snapshot is not None:
            try:
                package = load_and_validate_acoustic_scene_package(
                    manifest_snapshot.path,
                    manifest_snapshot=manifest_snapshot,
                    snapshot_cache=cache,
                )
                if (
                    package.manifest.get("package_content_sha256")
                    != record.get("package_content_sha256")
                ):
                    package_error = "package content hash differs from evidence"
                else:
                    packages[condition] = package
            except (AcousticSceneContractError, OSError, ValueError) as exc:
                package_error = str(exc)
        elif manifest_snapshot_error is not None:
            package_error = manifest_snapshot_error
        passed = record_passed and package_error is None and condition in packages
        checks.append(
            _check(
                f"compile_package_{condition}",
                passed,
                measured={
                    "path": record.get("path"),
                    "record_hash_match": record_passed,
                    "validation_error": package_error,
                },
                threshold={"record_hash_match": True, "strict_package_validation": "pass"},
                failure_reason="Compiled package is missing, changed, or semantically invalid",
            )
        )

    glb_replay_records: dict[str, Any] = {}
    glb_replay_errors: list[str] = []
    if (
        set(packages) == {"low_absorption", "high_absorption"}
        and source_scene is not None
        and mapping is not None
        and room is not None
        and len(source_databases) == 2
    ):
        try:
            expected_vertices, expected_triangles = _apply_source_to_canonical(
                source_scene, mapping
            )
            source_geometry_hash = source_snapshots["source_geometry"].sha256
            room_hash = source_snapshots["room_manifest"].sha256
            mapping_hash = source_snapshots["material_mapping"].sha256
            for condition, database_input_name in (
                ("low_absorption", "low_database"),
                ("high_absorption", "high_database"),
            ):
                package = packages[condition]
                database = source_databases[condition]
                compiled = compile_materials(
                    mapping, database, room_id=str(room["room_id"])
                )
                expected_material_ids = np.asarray(
                    [
                        compiled.source_material_to_id[source_name]
                        for source_name in source_scene.triangle_source_material_names
                    ],
                    dtype="<u4",
                )
                manifest = package.manifest
                database_hash = source_snapshots[database_input_name].sha256
                record = {
                    "source_geometry_hash_bound": (
                        source_geometry_hash
                        == source_scene.source_sha256
                        == manifest["source_room"]["geometry_asset_sha256"]
                    ),
                    "room_manifest_hash_bound": (
                        room_hash == manifest["source_room"]["manifest_sha256"]
                    ),
                    "mapping_hash_bound": (
                        mapping_hash == manifest["materials"]["mapping_sha256"]
                    ),
                    "database_hash_bound": (
                        database_hash
                        == manifest["materials"]["database_source_sha256"]
                    ),
                    "expected_vertices_identical": np.array_equal(
                        expected_vertices, package.vertices
                    ),
                    "expected_triangles_identical": np.array_equal(
                        expected_triangles, package.triangles
                    ),
                    "expected_material_ids_identical": np.array_equal(
                        expected_material_ids, package.triangle_material_ids
                    ),
                    "expanded_objects_identical": (
                        list(source_scene.objects) == manifest["objects"]
                    ),
                    "recompiled_categories_identical": (
                        compiled.categories_document == package.material_categories
                    ),
                    "recompiled_rlr_database_identical": (
                        compiled.rlr_database == package.rlr_material_database
                    ),
                }
                record["all_pass"] = all(record.values())
                glb_replay_records[condition] = record
        except (KeyError, MaterialContractError, TypeError, ValueError) as exc:
            glb_replay_errors.append(str(exc))
    glb_replay_passed = bool(
        len(glb_replay_records) == 2
        and not glb_replay_errors
        and all(record.get("all_pass") for record in glb_replay_records.values())
    )
    checks.append(
        _check(
            "compile_source_glb_to_package_replay",
            glb_replay_passed,
            measured={"conditions": glb_replay_records, "errors": glb_replay_errors},
            threshold={
                "source_hashes_bound": True,
                "transformed_array_bytes_identical": True,
                "objects_and_material_ids_recompiled_identical": True,
            },
            failure_reason=(
                "Hash-bound source GLB/mapping/databases do not replay exactly to "
                "the compiled package"
            ),
        )
    )

    invariant_records = evidence.get("frozen_variable_proof", {})
    for name in ("vertices", "triangles", "triangle_material_ids"):
        declared = invariant_records.get(name, {})
        if set(packages) == {"low_absorption", "high_absorption"}:
            low_hash = packages["low_absorption"].manifest["arrays"][name]["sha256"]
            high_hash = packages["high_absorption"].manifest["arrays"][name]["sha256"]
        else:
            low_hash = high_hash = None
        passed = bool(
            low_hash is not None
            and low_hash == high_hash
            and declared.get("low_sha256") == low_hash
            and declared.get("high_sha256") == high_hash
            and declared.get("identical") is True
        )
        checks.append(
            _check(
                f"compile_frozen_{name}",
                passed,
                measured={
                    "actual_low_sha256": low_hash,
                    "actual_high_sha256": high_hash,
                    "declared": declared,
                },
                threshold={"identical_and_bound": True},
                failure_reason=f"Low/high {name} bytes are not frozen and hash-bound",
            )
        )
    actual_counterfactual: dict[str, Any] | None = None
    counterfactual_error: str | None = None
    if set(packages) == {"low_absorption", "high_absorption"}:
        try:
            low_package = packages["low_absorption"]
            high_package = packages["high_absorption"]
            actual_counterfactual = controlled_counterfactual_proof(
                low_package.source_material_database,
                high_package.source_material_database,
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            counterfactual_error = str(exc)
    declared_counterfactual = evidence.get("material_counterfactual_proof")
    counterfactual_passed = bool(
        actual_counterfactual is not None
        and actual_counterfactual.get("status") == "pass"
        and declared_counterfactual == actual_counterfactual
    )
    checks.append(
        _check(
            "compile_absorption_only_counterfactual",
            counterfactual_passed,
            measured={
                "actual": actual_counterfactual,
                "declared": declared_counterfactual,
                "error": counterfactual_error,
            },
            threshold={
                "non_absorption_structure_identical": True,
                "high_absorption_strictly_greater_every_band": True,
                "proof_hash_bound": True,
            },
            failure_reason=(
                "Low/high package source databases do not prove an absorption-only "
                "controlled counterfactual"
            ),
        )
    )
    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return VerifiedCompileEvidence(
        evidence_path=path,
        evidence_snapshot=evidence_snapshot,
        evidence=evidence,
        status=status,
        checks=tuple(checks),
        packages=packages,
    )


def verify_compile_evidence(
    evidence_path: str | Path,
    *,
    evidence_snapshot: ImmutableFileSnapshot | None = None,
    snapshot_cache: dict[Path, ImmutableFileSnapshot] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Compatibility wrapper around the reusable single-snapshot result."""

    result = load_and_verify_compile_evidence(
        evidence_path,
        evidence_snapshot=evidence_snapshot,
        snapshot_cache=snapshot_cache,
    )
    return result.status, list(result.checks)
