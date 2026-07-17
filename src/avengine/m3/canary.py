"""Executable high/low absorption activation canary for M3.

The canary consumes compiler evidence, creates a fresh modern RLR context for
every condition repeat, copies every raw IR, and admits material activation
only when EDT, DRR and late energy all move in their declared directions by
more than both the requested minimum and within-condition repeat spread.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Callable, Mapping

from jsonschema import Draft202012Validator
import numpy as np

from avengine.contracts.json_io import (
    canonical_json_bytes,
    canonical_json_sha256,
    load_json,
    resolve_declared_path,
    write_json,
)
from avengine.m1.contracts import validate_room_manifest
from avengine.m3.contracts import (
    ImmutableFileSnapshot,
    read_immutable_file_snapshot,
    validate_canary_request,
    validate_mapping_document,
    validate_material_database_document,
)
from avengine.m3.evidence import load_and_verify_compile_evidence
from avengine.m3.metrics import AcousticMetricError, MetricConfig, analyze_ir
from avengine.m3.materials import MaterialContractError, compile_materials
from avengine.m3.runtime import (
    CompiledAcousticScene,
    _NATIVE_CONFIG_FIELDS,
    _cpu_first_hit_distance,
    _expected_native_scene_readback_report,
    _expected_scene_readback,
    _parse_scene_obj_bytes,
    _verify_upload_report,
    RLRSimulationConfig,
    RUNTIME_IMPORT_WORKAROUND,
    RuntimeAnchor,
    RuntimeContractError,
    RuntimeExecutionError,
    RuntimeIRResult,
    RuntimeUnavailableError,
    load_compiled_acoustic_scene,
    simulate_compiled_acoustic_scene,
)


CANARY_EVIDENCE_SCHEMA = "avengine_m3_acoustic_canary_evidence_v1"
_CONDITIONS = ("low_absorption", "high_absorption")
_METRICS = ("edt_seconds", "drr_db", "late_energy_ratio")


class AcousticCanaryError(ValueError):
    """The canary request or compiler lineage is invalid."""


@dataclass(frozen=True)
class VerifiedCanaryEvidence:
    """One canary-evidence snapshot and the verification derived from it."""

    evidence_path: Path
    evidence_snapshot: ImmutableFileSnapshot | None
    evidence: dict[str, Any]
    errors: tuple[str, ...]


SimulationRunner = Callable[..., RuntimeIRResult]


def _check(
    check_id: str,
    passed: bool,
    *,
    measured: Any,
    threshold: Any,
    failure_reason: str,
    blocked: bool = False,
) -> dict[str, Any]:
    status = "pass" if passed else ("blocked" if blocked else "fail")
    return {
        "check_id": check_id,
        "required": True,
        "status": status,
        "measured": measured,
        "threshold": threshold,
        **({} if passed else {"failure_reason": failure_reason}),
    }


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
                "memory_order": "C",
            }
        )
    )
    digest.update(b"\x00")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _external_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    payload = _read_file_once(resolved)
    return {
        "path": str(resolved),
        "byte_size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _read_file_once(path: Path) -> bytes:
    resolved = path.resolve()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise AcousticCanaryError(f"unable to snapshot {resolved}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise AcousticCanaryError(f"snapshot source is not a file: {resolved}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
    finally:
        os.close(descriptor)
    return payload


def _load_json_snapshot(
    path: Path,
    *,
    snapshot_cache: dict[Path, ImmutableFileSnapshot] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve()
    try:
        snapshot = read_immutable_file_snapshot(resolved, cache=snapshot_cache)
    except OSError as exc:
        raise AcousticCanaryError(f"unable to snapshot {resolved}: {exc}") from exc
    payload = snapshot.payload
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcousticCanaryError(f"invalid JSON snapshot {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AcousticCanaryError(f"JSON snapshot is not an object: {path}")
    return value, {
        "path": str(resolved),
        "byte_size": snapshot.byte_size,
        "sha256": snapshot.sha256,
    }


def _load_json_once(path: Path) -> dict[str, Any]:
    return _load_json_snapshot(path)[0]


def _package_input(
    scene: CompiledAcousticScene, record: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **dict(record),
        "package_content_sha256": scene.package_content_sha256,
        "package_id": scene.package_id,
    }


def _resolve_request_input(
    request_path: Path,
    raw_path: str,
    *,
    environment: Mapping[str, str],
) -> Path:
    return resolve_declared_path(
        raw_path,
        manifest_dir=request_path.parent,
        environment=environment,
    )


def _load_inputs(
    request_path: Path,
    compile_evidence_path: Path,
    *,
    environment: Mapping[str, str] | None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Path],
    dict[str, CompiledAcousticScene],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    snapshot_cache: dict[Path, ImmutableFileSnapshot] = {}
    request, request_record = _load_json_snapshot(
        request_path, snapshot_cache=snapshot_cache
    )
    snapshots: dict[str, dict[str, Any]] = {
        "request": {"document": request, "record": request_record}
    }
    request_errors = validate_canary_request(request)
    if request_errors:
        raise AcousticCanaryError("invalid canary request: " + "; ".join(request_errors))
    effective_environment = dict(os.environ if environment is None else environment)
    effective_environment.setdefault(
        "AVENGINE_REPOSITORY_ROOT", str(Path(__file__).resolve().parents[3])
    )
    paths = {
        "room_manifest": _resolve_request_input(
            request_path,
            request["room_manifest"],
            environment=effective_environment,
        ),
        "material_mapping": _resolve_request_input(
            request_path,
            request["material_mapping"],
            environment=effective_environment,
        ),
        "low_absorption_database": _resolve_request_input(
            request_path,
            request["material_databases"]["low_absorption"],
            environment=effective_environment,
        ),
        "high_absorption_database": _resolve_request_input(
            request_path,
            request["material_databases"]["high_absorption"],
            environment=effective_environment,
        ),
    }
    room, room_record = _load_json_snapshot(
        paths["room_manifest"], snapshot_cache=snapshot_cache
    )
    snapshots["room_manifest"] = {"document": room, "record": room_record}
    room_errors = validate_room_manifest(room)
    if room_errors:
        raise AcousticCanaryError("invalid source room: " + "; ".join(room_errors))
    mapping, mapping_record = _load_json_snapshot(
        paths["material_mapping"], snapshot_cache=snapshot_cache
    )
    snapshots["material_mapping"] = {
        "document": mapping,
        "record": mapping_record,
    }
    mapping_errors = validate_mapping_document(mapping, room_id=room["room_id"])
    if mapping_errors:
        raise AcousticCanaryError(
            "invalid acoustic material mapping: " + "; ".join(mapping_errors)
        )
    for condition in _CONDITIONS:
        database, database_record = _load_json_snapshot(
            paths[f"{condition}_database"], snapshot_cache=snapshot_cache
        )
        snapshots[f"{condition}_database"] = {
            "document": database,
            "record": database_record,
        }
        database_errors = validate_material_database_document(database)
        if database_errors:
            raise AcousticCanaryError(
                f"invalid {condition} material database: "
                + "; ".join(database_errors)
            )

    compile_result = load_and_verify_compile_evidence(
        compile_evidence_path,
        snapshot_cache=snapshot_cache,
    )
    compile_checks = list(compile_result.checks)
    if compile_result.status != "pass":
        failures = [
            str(check.get("failure_reason", check["check_id"]))
            for check in compile_checks
            if check["status"] != "pass"
        ]
        raise AcousticCanaryError(
            "compiler evidence did not verify: " + "; ".join(failures)
        )
    if compile_result.evidence_snapshot is None:
        raise AcousticCanaryError("compiler evidence has no immutable byte snapshot")
    compile_evidence = compile_result.evidence
    compile_record = {
        "path": str(compile_result.evidence_snapshot.path),
        "byte_size": compile_result.evidence_snapshot.byte_size,
        "sha256": compile_result.evidence_snapshot.sha256,
    }
    snapshots["compile_evidence"] = {
        "document": compile_evidence,
        "record": compile_record,
    }
    if compile_evidence.get("request_id") != request["request_id"]:
        raise AcousticCanaryError("compiler evidence request_id differs from request")
    if compile_evidence.get("request_sha256") != request_record["sha256"]:
        raise AcousticCanaryError("compiler evidence is not bound to the request bytes")
    scenes: dict[str, CompiledAcousticScene] = {}
    compile_root = compile_evidence_path.parent.resolve()
    for condition in _CONDITIONS:
        raw_manifest = compile_evidence["packages"][condition]["path"]
        declared = Path(raw_manifest)
        if declared.is_absolute() or ".." in declared.parts:
            raise AcousticCanaryError("compiler package path is not confined")
        manifest_path = (compile_root / declared).resolve()
        try:
            manifest_path.relative_to(compile_root)
        except ValueError as exc:
            raise AcousticCanaryError("compiler package symlink escapes evidence") from exc
        validated_package = compile_result.packages.get(condition)
        if validated_package is None:
            raise AcousticCanaryError(
                f"compiler evidence has no validated {condition} package"
            )
        scenes[condition] = load_compiled_acoustic_scene(
            manifest_path,
            validated_package=validated_package,
        )
        manifest_record = {
            "path": str(manifest_path),
            "byte_size": validated_package.manifest_byte_size,
            "sha256": validated_package.manifest_file_sha256,
        }
        snapshots[f"{condition}_package"] = {
            "document": scenes[condition].manifest,
            "record": manifest_record,
        }
    return request, room, paths, scenes, compile_checks, snapshots


def _input_invariant_checks(
    request: Mapping[str, Any],
    room: Mapping[str, Any],
    scenes: Mapping[str, CompiledAcousticScene],
    compile_checks: list[dict[str, Any]],
    snapshots: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    low = scenes["low_absorption"]
    high = scenes["high_absorption"]
    checks: list[dict[str, Any]] = []
    checks.append(
        _check(
            "compile_evidence_verified",
            all(item["status"] == "pass" for item in compile_checks),
            measured=compile_checks,
            threshold={"all_compile_checks": "pass"},
            failure_reason="Compiler evidence did not pass readback verification",
        )
    )
    room_hash = snapshots["room_manifest"]["record"]["sha256"]
    room_bound = all(
        scene.manifest["source_room"]["room_id"] == room["room_id"]
        and scene.manifest["source_room"]["manifest_sha256"] == room_hash
        for scene in scenes.values()
    )
    checks.append(
        _check(
            "room_manifest_bound",
            room_bound,
            measured={
                condition: scene.manifest["source_room"]
                for condition, scene in scenes.items()
            },
            threshold={"room_id": room["room_id"], "manifest_sha256": room_hash},
            failure_reason="Runtime package is not bound to the requested room manifest",
        )
    )
    mapping_hash = snapshots["material_mapping"]["record"]["sha256"]
    mapping_bound = all(
        scene.manifest["materials"]["mapping_sha256"] == mapping_hash
        for scene in scenes.values()
    )
    checks.append(
        _check(
            "material_mapping_bound",
            mapping_bound,
            measured={
                condition: scene.manifest["materials"]["mapping_sha256"]
                for condition, scene in scenes.items()
            },
            threshold=mapping_hash,
            failure_reason="Runtime packages are not bound to the requested mapping",
        )
    )
    database_hashes = {
        condition: snapshots[f"{condition}_database"]["record"]["sha256"]
        for condition in _CONDITIONS
    }
    database_bound = all(
        scenes[condition].manifest["materials"]["database_source_sha256"]
        == database_hashes[condition]
        for condition in _CONDITIONS
    )
    checks.append(
        _check(
            "material_databases_bound",
            database_bound,
            measured={
                condition: scenes[condition].manifest["materials"][
                    "database_source_sha256"
                ]
                for condition in _CONDITIONS
            },
            threshold=database_hashes,
            failure_reason="Runtime package material database lineage is wrong",
        )
    )
    for array_name in ("vertices", "triangles", "triangle_material_ids"):
        identical = (
            low.geometry_records[array_name]["sha256"]
            == high.geometry_records[array_name]["sha256"]
        )
        checks.append(
            _check(
                f"frozen_{array_name}",
                identical,
                measured={
                    "low_sha256": low.geometry_records[array_name]["sha256"],
                    "high_sha256": high.geometry_records[array_name]["sha256"],
                },
                threshold={"identical": True},
                failure_reason=f"Low/high {array_name} bytes differ",
            )
        )
    objects_identical = low.manifest["objects"] == high.manifest["objects"]
    checks.append(
        _check(
            "frozen_object_partitions",
            objects_identical,
            measured={"identical": objects_identical},
            threshold={"identical": True},
            failure_reason="Low/high object partitions or transforms differ",
        )
    )
    coverage_threshold = request["thresholds"]["material_coverage_fraction"]
    fallback_threshold = request["thresholds"]["fallback_triangle_count"]
    for condition, scene in scenes.items():
        report = scene.qa_reports["material_coverage"]
        coverage_passed = (
            report.get("coverage_fraction") == coverage_threshold
            and report.get("fallback_triangle_count") == fallback_threshold
        )
        checks.append(
            _check(
                f"{condition}_material_coverage",
                coverage_passed,
                measured={
                    "coverage_fraction": report.get("coverage_fraction"),
                    "fallback_triangle_count": report.get("fallback_triangle_count"),
                },
                threshold={
                    "coverage_fraction": coverage_threshold,
                    "fallback_triangle_count": fallback_threshold,
                },
                failure_reason=f"{condition} package material coverage is incomplete",
            )
        )
    low_database = snapshots["low_absorption_database"]["document"]
    high_database = snapshots["high_absorption_database"]["document"]
    low_materials = low_database["materials"]
    high_materials = high_database["materials"]
    def without_condition_fields(database: Mapping[str, Any]) -> dict[str, Any]:
        value = copy.deepcopy(dict(database))
        value.pop("database_id", None)
        for material in value.get("materials", []):
            material.pop("absorption", None)
        return value

    non_absorption_identical = without_condition_fields(
        low_database
    ) == without_condition_fields(high_database)
    checks.append(
        _check(
            "counterfactual_non_absorption_fields_frozen",
            non_absorption_identical,
            measured={
                "allowed_differences": ["database_id", "materials[*].absorption"],
                "material_count_low": len(low_materials),
                "material_count_high": len(high_materials),
                "identical": non_absorption_identical,
            },
            threshold={"identical": True},
            failure_reason="Low/high databases change fields other than absorption",
        )
    )
    absorption_pairs = [
        {
            "material_key": low_material.get("material_key"),
            "low": low_material.get("absorption"),
            "high": high_material.get("absorption"),
            "all_high_strictly_greater": (
                isinstance(low_material.get("absorption"), list)
                and isinstance(high_material.get("absorption"), list)
                and len(low_material["absorption"])
                == len(high_material["absorption"])
                == len(low_database["bands_hz"])
                and all(
                    float(high) > float(low)
                    for low, high in zip(
                        low_material["absorption"], high_material["absorption"]
                    )
                )
            ),
        }
        for low_material, high_material in zip(low_materials, high_materials)
    ]
    absorption_ordered = bool(absorption_pairs) and len(low_materials) == len(
        high_materials
    ) and all(pair["all_high_strictly_greater"] for pair in absorption_pairs)
    checks.append(
        _check(
            "counterfactual_absorption_strictly_ordered",
            absorption_ordered,
            measured=absorption_pairs,
            threshold={"high_absorption_gt_low_at_every_material_and_band": True},
            failure_reason="High absorption is not greater at every material and band",
        )
    )
    mapping = snapshots["material_mapping"]["document"]
    for condition, database in (
        ("low_absorption", low_database),
        ("high_absorption", high_database),
    ):
        try:
            rebuilt = compile_materials(mapping, database, room_id=room["room_id"])
            categories_equal = (
                rebuilt.categories_document
                == scenes[condition].material_categories_document
            )
            rlr_database_equal = (
                rebuilt.rlr_database == scenes[condition].rlr_material_database
            )
        except MaterialContractError:
            categories_equal = rlr_database_equal = False
        checks.append(
            _check(
                f"{condition}_packaged_material_recompile",
                categories_equal and rlr_database_equal,
                measured={
                    "categories_exact": categories_equal,
                    "rlr_database_exact": rlr_database_equal,
                },
                threshold={"categories_exact": True, "rlr_database_exact": True},
                failure_reason=(
                    f"{condition} packaged material files do not reproduce from sources"
                ),
            )
        )
    ray_required = request["thresholds"]["ray_checks"]["require_nonempty"]
    ray_nonempty = bool(room.get("ray_checks"))
    checks.append(
        _check(
            "room_ray_checks_declared",
            ray_nonempty or not ray_required,
            measured={"declared_count": len(room.get("ray_checks", []))},
            threshold={"require_nonempty": ray_required},
            failure_reason="Formal M3 canary requires room opening/control rays",
        )
    )
    return checks


def _expected_configuration_readback(
    simulation: RLRSimulationConfig,
) -> dict[str, Any]:
    return {
        public_name: getattr(simulation, public_name)
        for public_name in _NATIVE_CONFIG_FIELDS
    }


def _ray_report_verification_errors(
    scene: CompiledAcousticScene,
    declarations: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    reports: Any,
    *,
    distance_tolerance_m: float,
    owner: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(reports, (list, tuple)):
        return [f"{owner} is not an array"]
    if len(reports) != len(declarations):
        errors.append(
            f"{owner} count differs from the declared room ray-check count"
        )
    expected_keys = {
        "check_id",
        "expectation",
        "maximum_distance_m",
        "cpu_first_hit_distance_m",
        "rlr_any_hit",
        "rlr_first_hit",
        "cpu_rlr_hit_consistent",
        "cpu_rlr_distance_consistent",
        "distance_tolerance_m",
        "passed",
    }
    result_keys = {"hit", "has_hit_details", "distance_m", "normal"}
    for index, declaration in enumerate(declarations):
        if index >= len(reports):
            break
        report = reports[index]
        prefix = f"{owner}[{index}]"
        if not isinstance(report, Mapping):
            errors.append(f"{prefix} is not an object")
            continue
        if set(report) != expected_keys:
            errors.append(f"{prefix} fields differ from the ray evidence contract")
            continue
        try:
            origin = np.asarray(declaration["origin_m"], dtype=np.float64)
            direction = np.asarray(declaration["direction"], dtype=np.float64)
            maximum_distance = float(declaration["distance_m"])
            direction_norm = float(np.linalg.norm(direction))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{prefix} declaration cannot be evaluated: {exc}")
            continue
        if (
            origin.shape != (3,)
            or direction.shape != (3,)
            or not np.all(np.isfinite(origin))
            or not np.all(np.isfinite(direction))
            or not math.isclose(direction_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6)
            or not math.isfinite(maximum_distance)
            or maximum_distance <= 0.0
        ):
            errors.append(f"{prefix} declaration is geometrically invalid")
            continue
        direction = direction / direction_norm
        cpu_distance = _cpu_first_hit_distance(
            scene.objects,
            origin=origin,
            direction=direction,
            minimum_distance_m=0.0,
            maximum_distance_m=maximum_distance,
        )
        if report.get("check_id") != declaration.get("check_id"):
            errors.append(f"{prefix}.check_id differs from the room declaration")
        if report.get("expectation") != declaration.get("expectation"):
            errors.append(f"{prefix}.expectation differs from the room declaration")
        if report.get("maximum_distance_m") != maximum_distance:
            errors.append(f"{prefix}.maximum_distance_m differs from the room declaration")
        if report.get("distance_tolerance_m") != distance_tolerance_m:
            errors.append(f"{prefix}.distance_tolerance_m differs from the request")
        if report.get("cpu_first_hit_distance_m") != cpu_distance:
            errors.append(f"{prefix}.cpu_first_hit_distance_m differs from recomputation")

        native_results: dict[str, Mapping[str, Any]] = {}
        for result_name in ("rlr_any_hit", "rlr_first_hit"):
            raw_result = report.get(result_name)
            if not isinstance(raw_result, Mapping) or set(raw_result) != result_keys:
                errors.append(f"{prefix}.{result_name} fields are invalid")
                continue
            native_results[result_name] = raw_result
            if not isinstance(raw_result.get("hit"), bool) or not isinstance(
                raw_result.get("has_hit_details"), bool
            ):
                errors.append(f"{prefix}.{result_name} hit flags are not boolean")
            distance_value = raw_result.get("distance_m")
            normal = raw_result.get("normal")
            if (
                isinstance(distance_value, bool)
                or not isinstance(distance_value, (int, float))
                or not math.isfinite(float(distance_value))
                or float(distance_value) < 0.0
            ):
                errors.append(f"{prefix}.{result_name}.distance_m is invalid")
            if (
                not isinstance(normal, list)
                or len(normal) != 3
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in normal
                )
            ):
                errors.append(f"{prefix}.{result_name}.normal is invalid")

        if len(native_results) != 2:
            continue
        any_hit = native_results["rlr_any_hit"]
        first_hit = native_results["rlr_first_hit"]
        if any_hit["has_hit_details"] is not False:
            errors.append(f"{prefix}.rlr_any_hit unexpectedly claims hit details")
        if first_hit["has_hit_details"] != first_hit["hit"]:
            errors.append(f"{prefix}.rlr_first_hit detail flag differs from hit")
        any_normal = any_hit.get("normal")
        any_zero_sentinels = (
            any_hit.get("distance_m") == 0.0
            and isinstance(any_normal, list)
            and len(any_normal) == 3
            and all(value == 0.0 for value in any_normal)
        )
        if not any_zero_sentinels:
            errors.append(
                f"{prefix}.rlr_any_hit must use zero distance/normal sentinels"
            )
        first_normal = first_hit.get("normal")
        if first_hit.get("hit") is True:
            first_normal_is_finite_nonzero = (
                isinstance(first_normal, list)
                and len(first_normal) == 3
                and all(
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    for value in first_normal
                )
                and any(value != 0.0 for value in first_normal)
            )
            if not first_normal_is_finite_nonzero:
                errors.append(
                    f"{prefix}.rlr_first_hit.normal must be finite and non-zero"
                )
        else:
            first_zero_sentinels = (
                first_hit.get("distance_m") == 0.0
                and isinstance(first_normal, list)
                and len(first_normal) == 3
                and all(value == 0.0 for value in first_normal)
            )
            if not first_zero_sentinels:
                errors.append(
                    f"{prefix}.rlr_first_hit miss must use zero "
                    "distance/normal sentinels"
                )
        cpu_hit = cpu_distance is not None
        expected_hit = declaration.get("expectation") == "hit_within_m"
        hit_consistent = cpu_hit == any_hit["hit"] == first_hit["hit"]
        distance_consistent = (
            not cpu_hit
            or abs(float(first_hit["distance_m"]) - float(cpu_distance))
            <= distance_tolerance_m
        )
        passed = hit_consistent and distance_consistent and cpu_hit == expected_hit
        expected_flags = {
            "cpu_rlr_hit_consistent": hit_consistent,
            "cpu_rlr_distance_consistent": distance_consistent,
            "passed": passed,
        }
        for name, expected in expected_flags.items():
            if report.get(name) is not expected:
                errors.append(f"{prefix}.{name} differs from recomputation")
    return errors


def _validate_runtime_result(
    result: RuntimeIRResult,
    *,
    scene: CompiledAcousticScene,
    simulation: RLRSimulationConfig,
    source: RuntimeAnchor,
    listener: RuntimeAnchor,
    readback_path: Path,
    ray_declarations: list[Mapping[str, Any]],
    ray_distance_tolerance_m: float,
) -> None:
    mismatches: list[str] = []
    if result.source_id != source.anchor_id:
        mismatches.append("source_id")
    if result.listener_id != listener.anchor_id:
        mismatches.append("listener_id")
    if result.package_manifest_sha256 != scene.manifest_sha256:
        mismatches.append("package_manifest_sha256")
    if result.package_content_sha256 != scene.package_content_sha256:
        mismatches.append("package_content_sha256")
    if not math.isclose(
        float(result.sample_rate_hz),
        float(simulation.sample_rate_hz),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        mismatches.append("sample_rate_hz")
    samples = result.samples
    maximum_samples = int(
        math.ceil(simulation.max_ir_seconds * simulation.sample_rate_hz)
    )
    if (
        not isinstance(samples, np.ndarray)
        or samples.dtype.str != "<f4"
        or samples.ndim != 2
        or samples.shape[0] != simulation.channel_layout.channel_count
        or samples.shape[1] < 2
        or samples.shape[1] > maximum_samples
        or not samples.flags.c_contiguous
        or not np.all(np.isfinite(samples))
    ):
        mismatches.append("owned_ir_array")
    if mismatches:
        raise RuntimeContractError(
            "runtime result differs from request/package: " + ", ".join(mismatches)
        )
    _verify_upload_report(scene, result.upload_report)
    if (
        not math.isfinite(float(result.indirect_ray_efficiency))
        or not 0.0 <= float(result.indirect_ray_efficiency) <= 1.0
    ):
        raise RuntimeContractError("runtime indirect-ray efficiency is outside [0, 1]")
    runtime = result.runtime
    if not isinstance(runtime, Mapping):
        raise RuntimeContractError("runtime report is not an object")
    if runtime.get("import_workaround") != RUNTIME_IMPORT_WORKAROUND:
        raise RuntimeContractError("runtime import workaround record differs")
    if runtime.get("configuration_readback") != _expected_configuration_readback(
        simulation
    ):
        raise RuntimeContractError("native configuration readback differs from request")
    readback = runtime.get("scene_mesh_readback")
    if not isinstance(readback, Mapping):
        raise RuntimeContractError("runtime scene mesh readback record is missing")
    try:
        reported_path = Path(str(readback["path"])).resolve()
    except KeyError as exc:
        raise RuntimeContractError("runtime scene mesh readback path is missing") from exc
    if reported_path != readback_path.resolve():
        raise RuntimeContractError("runtime scene mesh readback path differs from request")
    try:
        readback_payload = _read_file_once(reported_path)
    except AcousticCanaryError as exc:
        raise RuntimeContractError(
            f"runtime scene mesh readback artifact is missing: {exc}"
        ) from exc
    parsed = _parse_scene_obj_bytes(readback_payload)
    expected_geometry = _expected_scene_readback(scene)
    expected_readback = {
        "path": str(reported_path),
        "byte_size": len(readback_payload),
        "sha256": hashlib.sha256(readback_payload).hexdigest(),
        "native_report": _expected_native_scene_readback_report(
            expected_geometry, result.upload_report
        ),
        **expected_geometry,
    }
    if dict(readback) != expected_readback or parsed != expected_geometry:
        raise RuntimeContractError(
            "runtime scene mesh readback differs from package geometry/materials"
        )
    ray_errors = _ray_report_verification_errors(
        scene,
        ray_declarations,
        result.ray_checks,
        distance_tolerance_m=ray_distance_tolerance_m,
        owner="runtime.ray_checks",
    )
    if ray_errors:
        raise RuntimeContractError("; ".join(ray_errors))


def _metric_comparison(
    metric_name: str,
    low_values: list[float],
    high_values: list[float],
    threshold: Mapping[str, Any],
) -> dict[str, Any]:
    if len(low_values) < 3 or len(high_values) < 3:
        raise AcousticCanaryError(
            f"{metric_name} comparison requires at least three values per condition"
        )
    if any(not math.isfinite(float(value)) for value in [*low_values, *high_values]):
        raise AcousticCanaryError(f"{metric_name} comparison contains non-finite values")
    low_median = float(np.median(np.asarray(low_values, dtype=np.float64)))
    high_median = float(np.median(np.asarray(high_values, dtype=np.float64)))
    direction = threshold["direction"]
    oriented_effect = (
        low_median - high_median
        if direction == "high_lt_low"
        else high_median - low_median
    )
    low_spread = float(max(low_values) - min(low_values))
    high_spread = float(max(high_values) - min(high_values))
    within_spread = max(low_spread, high_spread)
    relative_floor = 1.0e-12
    relative_spread = max(
        low_spread / max(abs(low_median), relative_floor),
        high_spread / max(abs(high_median), relative_floor),
    )
    effect_to_spread = (
        None if within_spread == 0.0 else oriented_effect / within_spread
    )
    checks = {
        "direction": oriented_effect > 0.0,
        "minimum_absolute_effect": oriented_effect
        >= float(threshold["minimum_absolute_effect"]),
        "maximum_relative_repeat_spread": relative_spread
        <= float(threshold["maximum_relative_repeat_spread"]),
        "effect_strictly_exceeds_within_spread": oriented_effect > within_spread,
        "minimum_effect_to_within_spread_ratio": (
            oriented_effect > 0.0
            if effect_to_spread is None
            else effect_to_spread
            >= float(threshold["minimum_effect_to_within_spread_ratio"])
        ),
        "finite_inputs_and_effect": all(
            math.isfinite(value)
            for value in (
                low_median,
                high_median,
                oriented_effect,
                low_spread,
                high_spread,
                relative_spread,
            )
        ),
    }
    result: dict[str, Any] = {
        "direction": direction,
        "low_median": low_median,
        "high_median": high_median,
        "oriented_effect": oriented_effect,
        "minimum_absolute_effect": float(threshold["minimum_absolute_effect"]),
        "low_absolute_repeat_spread": low_spread,
        "high_absolute_repeat_spread": high_spread,
        "maximum_within_condition_spread": within_spread,
        "maximum_relative_repeat_spread_measured": relative_spread,
        "maximum_relative_repeat_spread_threshold": float(
            threshold["maximum_relative_repeat_spread"]
        ),
        "effect_to_within_spread_ratio": effect_to_spread,
        "minimum_effect_to_within_spread_ratio": float(
            threshold["minimum_effect_to_within_spread_ratio"]
        ),
        "checks": checks,
    }
    if metric_name == "late_energy_ratio":
        denominator_positive = high_median > 0.0
        effect_ratio = low_median / high_median if denominator_positive else 0.0
        checks["positive_effect_ratio_denominator"] = denominator_positive
        checks["minimum_effect_ratio"] = denominator_positive and effect_ratio >= float(
            threshold["minimum_effect_ratio"]
        )
        result["minimum_effect_ratio"] = float(threshold["minimum_effect_ratio"])
        result["measured_effect_ratio"] = effect_ratio
    result["passed"] = all(checks.values())
    return result


def _finalize_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    required = [check for check in evidence["checks"] if check["required"]]
    statuses = {check["status"] for check in required}
    if "fail" in statuses:
        status = "fail"
    elif "blocked" in statuses:
        status = "blocked"
    else:
        status = "pass"
    evidence["overall_status"] = status
    evidence["failure_reasons"] = [
        check["failure_reason"]
        for check in required
        if check["status"] != "pass"
    ]
    evidence.pop("evidence_content_sha256", None)
    evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
    return evidence


def run_material_activation_canary(
    request_path: str | Path,
    compile_evidence_path: str | Path,
    output_directory: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
    runner: SimulationRunner = simulate_compiled_acoustic_scene,
) -> Path:
    """Run the strict repeated M3 material canary and publish atomic evidence."""

    request_file = Path(request_path).resolve()
    compile_file = Path(compile_evidence_path).resolve()
    runtime_lock = Path(__file__).resolve().parents[3] / "runtime.lock.yaml"
    request, room, paths, scenes, compile_checks, snapshots = _load_inputs(
        request_file, compile_file, environment=environment
    )
    simulation = RLRSimulationConfig.from_mapping(request["simulation"])
    source = RuntimeAnchor.from_mapping(request["source"], listener=False)
    listener = RuntimeAnchor.from_mapping(request["listener"], listener=True)
    destination = Path(output_directory).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise AcousticCanaryError(f"refusing to replace existing output: {destination}")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-", dir=destination.parent
        )
    ).resolve()
    conditions: dict[str, dict[str, Any]] = {
        condition: {
            "attempted_run_count": 0,
            "completed_run_count": 0,
            "runs": [],
        }
        for condition in _CONDITIONS
    }
    checks = _input_invariant_checks(
        request, room, scenes, compile_checks, snapshots
    )
    runtime_lock_record = _external_file(runtime_lock)
    planned_order: list[str] = []
    for repeat_index in range(request["repeat_count"]):
        pair = _CONDITIONS if repeat_index % 2 == 0 else tuple(reversed(_CONDITIONS))
        planned_order.extend(pair)
    evidence: dict[str, Any] = {
        "schema": CANARY_EVIDENCE_SCHEMA,
        "request_id": request["request_id"],
        "overall_status": "fail",
        "failure_reasons": [],
        "request": {
            "source": snapshots["request"]["record"],
            "canonical_content_sha256": canonical_json_sha256(request),
            "snapshot": copy.deepcopy(request),
        },
        "inputs": {
            "room_manifest": snapshots["room_manifest"]["record"],
            "material_mapping": snapshots["material_mapping"]["record"],
            "low_absorption_database": snapshots["low_absorption_database"][
                "record"
            ],
            "high_absorption_database": snapshots["high_absorption_database"][
                "record"
            ],
            "compile_evidence": snapshots["compile_evidence"]["record"],
            "runtime_lock": runtime_lock_record,
            "low_absorption_package": _package_input(
                scenes["low_absorption"],
                snapshots["low_absorption_package"]["record"],
            ),
            "high_absorption_package": _package_input(
                scenes["high_absorption"],
                snapshots["high_absorption_package"]["record"],
            ),
        },
        "execution": {
            "repeat_count": request["repeat_count"],
            "condition_order": planned_order,
            "fresh_context_per_run": True,
            "temporal_coherence": False,
            "runtime_import_workaround": dict(RUNTIME_IMPORT_WORKAROUND),
        },
        "checks": checks,
        "conditions": conditions,
        "comparisons": {},
        "evidence_content_sha256": "0" * 64,
    }
    try:
        inputs_passed = all(check["status"] == "pass" for check in checks)
        stop = not inputs_passed
        execution_index = 0
        metric_config = MetricConfig()
        distance_m = math.dist(source.position_m, listener.position_m)
        expected_arrival = (
            distance_m
            / simulation.speed_of_sound_m_s
            * simulation.sample_rate_hz
        )
        arrival_threshold = float(
            request["thresholds"]["direct_arrival"][
                "maximum_absolute_error_samples"
            ]
        )
        ray_distance_tolerance = float(
            request["thresholds"]["ray_checks"][
                "maximum_first_hit_distance_error_m"
            ]
        )
        for repeat_index in range(request["repeat_count"]):
            if stop:
                break
            pair = _CONDITIONS if repeat_index % 2 == 0 else tuple(reversed(_CONDITIONS))
            for condition in pair:
                condition_evidence = conditions[condition]
                condition_evidence["attempted_run_count"] += 1
                run_id = f"{condition}_repeat_{repeat_index:03d}"
                readback_path = staging / "runtime_readback" / f"{run_id}.obj"
                try:
                    runtime_result = runner(
                        scenes[condition],
                        simulation,
                        source=source,
                        listener=listener,
                        scene_readback_obj=readback_path,
                        ray_checks=tuple(room["ray_checks"]),
                        ray_distance_tolerance_m=ray_distance_tolerance,
                    )
                    _validate_runtime_result(
                        runtime_result,
                        scene=scenes[condition],
                        simulation=simulation,
                        source=source,
                        listener=listener,
                        readback_path=readback_path,
                        ray_declarations=list(room["ray_checks"]),
                        ray_distance_tolerance_m=ray_distance_tolerance,
                    )
                    samples = np.array(
                        runtime_result.samples,
                        dtype="<f4",
                        order="C",
                        copy=True,
                    )
                    metrics = analyze_ir(
                        samples,
                        runtime_result.sample_rate_hz,
                        channel_axis=0,
                        configuration=metric_config,
                    )
                except RuntimeUnavailableError as exc:
                    checks.append(
                        _check(
                            "runtime_available",
                            False,
                            measured=str(exc),
                            threshold="modern audio-enabled RLR binding available",
                            failure_reason=str(exc),
                            blocked=True,
                        )
                    )
                    stop = True
                    break
                except (RuntimeContractError, RuntimeExecutionError, AcousticMetricError) as exc:
                    checks.append(
                        _check(
                            f"runtime_{run_id}",
                            False,
                            measured=str(exc),
                            threshold="complete valid RLR IR and metric result",
                            failure_reason=str(exc),
                        )
                    )
                    stop = True
                    break

                ir_directory = staging / "raw_ir" / condition
                ir_directory.mkdir(parents=True, exist_ok=True)
                ir_path = ir_directory / f"repeat_{repeat_index:03d}.npy"
                np.save(ir_path, samples, allow_pickle=False)
                ir_payload = _read_file_once(ir_path)
                ir_record = {
                    "path": ir_path.relative_to(staging).as_posix(),
                    "byte_size": len(ir_payload),
                    "sha256": hashlib.sha256(ir_payload).hexdigest(),
                }
                metric_value = metrics.to_dict()
                arrival_error = abs(metrics.direct_arrival_sample - expected_arrival)
                arrival = {
                    "distance_m": distance_m,
                    "speed_of_sound_m_s": simulation.speed_of_sound_m_s,
                    "expected_sample": expected_arrival,
                    "detected_sample": metrics.direct_arrival_sample,
                    "absolute_error_samples": arrival_error,
                    "maximum_absolute_error_samples": arrival_threshold,
                    "passed": arrival_error <= arrival_threshold,
                }
                runtime_payload = copy.deepcopy(runtime_result.runtime)
                readback = runtime_payload.get("scene_mesh_readback")
                if isinstance(readback, dict):
                    raw_readback_path = Path(str(readback.pop("path"))).resolve()
                    try:
                        confined_readback_path = raw_readback_path.relative_to(staging)
                    except ValueError as exc:
                        raise RuntimeContractError(
                            "runtime readback path escapes private canary staging"
                        ) from exc
                    readback["artifact"] = {
                        "path": confined_readback_path.as_posix(),
                        "byte_size": readback["byte_size"],
                        "sha256": readback["sha256"],
                    }
                run = {
                    "run_id": run_id,
                    "condition": condition,
                    "repeat_index": repeat_index,
                    "execution_index": execution_index,
                    "runtime_result_identity": {
                        "listener_id": runtime_result.listener_id,
                        "source_id": runtime_result.source_id,
                        "sample_rate_hz": runtime_result.sample_rate_hz,
                        "package_manifest_sha256": (
                            runtime_result.package_manifest_sha256
                        ),
                        "package_content_sha256": (
                            runtime_result.package_content_sha256
                        ),
                    },
                    "ir_artifact": ir_record,
                    "ir_array": {
                        "dtype": samples.dtype.str,
                        "shape": list(samples.shape),
                        "raw_array_sha256": _array_sha256(samples),
                    },
                    "metrics": metric_value,
                    "direct_arrival": arrival,
                    "runtime": runtime_payload,
                    "upload_report": copy.deepcopy(runtime_result.upload_report),
                    "indirect_ray_efficiency": runtime_result.indirect_ray_efficiency,
                    "ray_checks": [copy.deepcopy(item) for item in runtime_result.ray_checks],
                }
                condition_evidence["runs"].append(run)
                condition_evidence["completed_run_count"] += 1
                execution_index += 1
                checks.append(
                    _check(
                        f"{run_id}_direct_arrival",
                        arrival["passed"],
                        measured=arrival,
                        threshold={"maximum_absolute_error_samples": arrival_threshold},
                        failure_reason="Detected direct arrival disagrees with metric geometry",
                    )
                )
                edt_threshold = request["thresholds"]["metrics"]["edt_seconds"]
                edt_quality = (
                    metrics.edt_fit_r2 >= float(edt_threshold["minimum_fit_r2"])
                    and metrics.edt_decay_span_db
                    >= float(edt_threshold["minimum_decay_span_db"])
                )
                checks.append(
                    _check(
                        f"{run_id}_edt_quality",
                        edt_quality,
                        measured={
                            "fit_r2": metrics.edt_fit_r2,
                            "decay_span_db": metrics.edt_decay_span_db,
                        },
                        threshold={
                            "minimum_fit_r2": edt_threshold["minimum_fit_r2"],
                            "minimum_decay_span_db": edt_threshold[
                                "minimum_decay_span_db"
                            ],
                        },
                        failure_reason="EDT fit quality is below the declared threshold",
                    )
                )
                ray_passed = bool(runtime_result.ray_checks) and all(
                    item.get("passed") is True for item in runtime_result.ray_checks
                )
                checks.append(
                    _check(
                        f"{run_id}_rlr_ray_checks",
                        ray_passed,
                        measured=run["ray_checks"],
                        threshold={
                            "declared_count": len(room["ray_checks"]),
                            "all_cpu_rlr_consistent_and_expected": True,
                            "maximum_first_hit_distance_error_m": ray_distance_tolerance,
                        },
                        failure_reason="RLR opening/control rays disagree with CPU reference",
                    )
                )

        for condition in _CONDITIONS:
            complete = (
                conditions[condition]["completed_run_count"]
                == request["repeat_count"]
            )
            checks.append(
                _check(
                    f"{condition}_repeat_count",
                    complete,
                    measured=conditions[condition]["completed_run_count"],
                    threshold=request["repeat_count"],
                    failure_reason=f"{condition} did not complete every independent repeat",
                    blocked=any(check["status"] == "blocked" for check in checks),
                )
            )
        repeats_complete = all(
            conditions[condition]["completed_run_count"] == request["repeat_count"]
            for condition in _CONDITIONS
        )
        completed_runs = [
            run
            for condition in _CONDITIONS
            for run in conditions[condition]["runs"]
        ]
        binary_records = [
            run.get("runtime", {}).get("native_binaries") for run in completed_runs
        ]
        binary_identity = bool(binary_records) and all(
            record == binary_records[0] and isinstance(record, dict)
            for record in binary_records
        )
        checks.append(
            _check(
                "runtime_native_binary_identity",
                binary_identity,
                measured={
                    "completed_run_count": len(completed_runs),
                    "unique_record_count": len(
                        {
                            canonical_json_sha256(record)
                            for record in binary_records
                            if isinstance(record, dict)
                        }
                    ),
                    "native_binaries": binary_records[0] if binary_identity else None,
                },
                threshold={
                    "same_hash_bound_native_binaries_every_repeat": True,
                    "runtime_lock_sha256": runtime_lock_record["sha256"],
                },
                failure_reason="Runs did not use one hash-bound Habitat/RLR binary pair",
                blocked=any(check["status"] == "blocked" for check in checks),
            )
        )
        if repeats_complete:
            for metric_name in _METRICS:
                low_values = [
                    float(run["metrics"][metric_name])
                    for run in conditions["low_absorption"]["runs"]
                ]
                high_values = [
                    float(run["metrics"][metric_name])
                    for run in conditions["high_absorption"]["runs"]
                ]
                comparison = _metric_comparison(
                    metric_name,
                    low_values,
                    high_values,
                    request["thresholds"]["metrics"][metric_name],
                )
                evidence["comparisons"][metric_name] = comparison
                checks.append(
                    _check(
                        f"material_effect_{metric_name}",
                        comparison["passed"],
                        measured=comparison,
                        threshold=request["thresholds"]["metrics"][metric_name],
                        failure_reason=(
                            f"{metric_name} did not prove a directional material "
                            "effect larger than repeat spread"
                        ),
                    )
                )

        _finalize_evidence(evidence)
        evidence_path = staging / "canary_evidence.json"
        write_json(evidence_path, evidence)
        verification_errors = verify_canary_evidence(evidence_path)
        if verification_errors:
            raise AcousticCanaryError(
                "generated canary evidence failed self-verification: "
                + "; ".join(verification_errors)
            )
        os.rename(staging, destination)
        return destination / evidence_path.name
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _schema_errors(value: Any) -> list[str]:
    schema_path = (
        Path(__file__).resolve().parents[3]
        / "schemas"
        / "m3_acoustic_canary_evidence_v1.schema.json"
    )
    schema = load_json(schema_path)
    validator = Draft202012Validator(schema)
    errors = sorted(
        validator.iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    return [
        f"{'.'.join(str(part) for part in error.absolute_path) or '$'}: {error.message}"
        for error in errors
    ]


def _all_finite(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    return False


def _confined_artifact(base: Path, record: Any, owner: str) -> tuple[Path | None, str | None]:
    if not isinstance(record, Mapping):
        return None, f"{owner} is not a file record"
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None, f"{owner}.path is invalid"
    declared = Path(raw_path)
    if declared.is_absolute() or ".." in declared.parts:
        return None, f"{owner}.path is not confined"
    path = (base / declared).resolve()
    try:
        path.relative_to(base)
    except ValueError:
        return None, f"{owner}.path or symlink escapes evidence"
    return path, None


def _verified_file_snapshot(
    path: Path,
    record: Mapping[str, Any],
    owner: str,
    *,
    cache: dict[Path, bytes] | None = None,
) -> tuple[bytes | None, str | None]:
    """Bind record verification and every downstream parse to one byte read."""

    resolved = path.resolve()
    payload = None if cache is None else cache.get(resolved)
    if payload is None:
        try:
            payload = _read_file_once(resolved)
        except AcousticCanaryError as exc:
            return None, f"{owner} is missing or unreadable: {exc}"
        if cache is not None:
            cache[resolved] = payload
    if len(payload) != record.get("byte_size"):
        return None, f"{owner}.byte_size does not match"
    if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
        return None, f"{owner}.sha256 does not match"
    return payload, None


def _verify_file_record(path: Path, record: Mapping[str, Any], owner: str) -> str | None:
    return _verified_file_snapshot(path, record, owner)[1]


def _runtime_evidence_errors(
    run: Mapping[str, Any],
    *,
    scene: CompiledAcousticScene,
    request: Mapping[str, Any],
    room: Mapping[str, Any],
    base: Path,
    owner: str,
    snapshot_cache: dict[Path, bytes],
) -> list[str]:
    errors: list[str] = []
    try:
        simulation = RLRSimulationConfig.from_mapping(request["simulation"])
    except (KeyError, RuntimeContractError) as exc:
        return [f"{owner} cannot reconstruct simulation config: {exc}"]
    expected_identity = {
        "listener_id": request["listener"]["id"],
        "source_id": request["source"]["id"],
        "sample_rate_hz": simulation.sample_rate_hz,
        "package_manifest_sha256": scene.manifest_sha256,
        "package_content_sha256": scene.package_content_sha256,
    }
    if run.get("runtime_result_identity") != expected_identity:
        errors.append(f"{owner}.runtime_result_identity differs from request/package")

    upload = run.get("upload_report")
    if not isinstance(upload, Mapping):
        errors.append(f"{owner}.upload_report is not an object")
    else:
        try:
            _verify_upload_report(scene, upload)
        except RuntimeContractError as exc:
            errors.append(f"{owner}.upload_report: {exc}")

    runtime = run.get("runtime")
    if not isinstance(runtime, Mapping):
        return [*errors, f"{owner}.runtime is not an object"]
    if runtime.get("import_workaround") != RUNTIME_IMPORT_WORKAROUND:
        errors.append(f"{owner}.runtime import workaround differs")
    if runtime.get("configuration_readback") != _expected_configuration_readback(
        simulation
    ):
        errors.append(f"{owner}.runtime configuration readback differs")

    readback = runtime.get("scene_mesh_readback")
    if not isinstance(readback, Mapping):
        errors.append(f"{owner}.scene_mesh_readback is missing")
    else:
        artifact = readback.get("artifact")
        obj_path, obj_error = _confined_artifact(
            base, artifact, f"{owner}.scene_mesh_readback"
        )
        if obj_error:
            errors.append(obj_error)
        elif obj_path is not None and isinstance(artifact, Mapping):
            obj_payload, record_error = _verified_file_snapshot(
                obj_path,
                artifact,
                f"{owner}.scene_mesh_readback",
                cache=snapshot_cache,
            )
            if record_error:
                errors.append(record_error)
            else:
                assert obj_payload is not None
                try:
                    parsed = _parse_scene_obj_bytes(obj_payload)
                except RuntimeContractError as exc:
                    errors.append(f"{owner}.scene_mesh_readback cannot be parsed: {exc}")
                else:
                    expected_geometry = _expected_scene_readback(scene)
                    if parsed != expected_geometry:
                        errors.append(
                            f"{owner}.scene_mesh_readback fingerprint differs from package"
                        )
                    if isinstance(upload, Mapping):
                        expected_readback = {
                            "artifact": dict(artifact),
                            "byte_size": len(obj_payload),
                            "sha256": hashlib.sha256(obj_payload).hexdigest(),
                            "native_report": (
                                _expected_native_scene_readback_report(
                                    expected_geometry, upload
                                )
                            ),
                            **expected_geometry,
                        }
                        if dict(readback) != expected_readback:
                            errors.append(
                                f"{owner}.scene_mesh_readback record differs "
                                "from artifact"
                            )

    native_binaries = runtime.get("native_binaries")
    if not isinstance(native_binaries, Mapping):
        errors.append(f"{owner}.native_binaries is missing")
    else:
        for binary_name in ("habitat_sim_bindings", "rlr_audio_propagation"):
            binary_record = native_binaries.get(binary_name)
            if not isinstance(binary_record, Mapping):
                errors.append(f"{owner}.{binary_name} record is missing")
                continue
            raw_binary_path = binary_record.get("path")
            if (
                not isinstance(raw_binary_path, str)
                or not Path(raw_binary_path).is_absolute()
            ):
                errors.append(f"{owner}.{binary_name} path is not absolute")
                continue
            _binary_payload, binary_error = _verified_file_snapshot(
                Path(raw_binary_path),
                binary_record,
                f"{owner}.{binary_name}",
                cache=snapshot_cache,
            )
            if binary_error:
                errors.append(binary_error)

    errors.extend(
        _ray_report_verification_errors(
            scene,
            list(room.get("ray_checks", [])),
            run.get("ray_checks"),
            distance_tolerance_m=float(
                request["thresholds"]["ray_checks"][
                    "maximum_first_hit_distance_error_m"
                ]
            ),
            owner=f"{owner}.ray_checks",
        )
    )
    return errors


def _verify_canary_evidence_document(
    path: Path, evidence: Mapping[str, Any]
) -> list[str]:
    """Recompute claims from a caller-owned immutable evidence snapshot."""

    errors = _schema_errors(evidence)
    if not _all_finite(evidence):
        errors.append("evidence contains a non-finite number")
    try:
        content_hash = canonical_json_sha256(
            {
                key: value
                for key, value in evidence.items()
                if key != "evidence_content_sha256"
            }
        )
    except (TypeError, ValueError) as exc:
        content_hash = None
        errors.append(f"unable to canonicalize evidence: {exc}")
    if content_hash != evidence.get("evidence_content_sha256"):
        errors.append("evidence_content_sha256 does not match canonical content")

    request_value = evidence.get("request", {})
    request = request_value.get("snapshot", {})
    if canonical_json_sha256(request) != request_value.get("canonical_content_sha256"):
        errors.append("request snapshot canonical hash does not match")
    base = path.parent.resolve()
    inputs = evidence.get("inputs", {})

    scenes: dict[str, CompiledAcousticScene] = {}
    room: dict[str, Any] = {}
    verified_runtime_lock_sha256: str | None = None
    artifact_snapshot_cache: dict[Path, bytes] = {}
    try:
        request_source = Path(request_value["source"]["path"])
        compile_source = Path(inputs["compile_evidence"]["path"])
        (
            loaded_request,
            room,
            _loaded_paths,
            scenes,
            compile_checks,
            snapshots,
        ) = _load_inputs(request_source, compile_source, environment=None)
        if loaded_request != request:
            errors.append("request source bytes do not decode to the evidence snapshot")
        if request_value.get("source") != snapshots["request"]["record"]:
            errors.append("request source record differs from its immutable snapshot")
        expected_inputs = {
            "room_manifest": snapshots["room_manifest"]["record"],
            "material_mapping": snapshots["material_mapping"]["record"],
            "low_absorption_database": snapshots["low_absorption_database"][
                "record"
            ],
            "high_absorption_database": snapshots["high_absorption_database"][
                "record"
            ],
            "compile_evidence": snapshots["compile_evidence"]["record"],
            "runtime_lock": _external_file(
                Path(__file__).resolve().parents[3] / "runtime.lock.yaml"
            ),
            "low_absorption_package": _package_input(
                scenes["low_absorption"],
                snapshots["low_absorption_package"]["record"],
            ),
            "high_absorption_package": _package_input(
                scenes["high_absorption"],
                snapshots["high_absorption_package"]["record"],
            ),
        }
        verified_runtime_lock_sha256 = expected_inputs["runtime_lock"]["sha256"]
        if inputs != expected_inputs:
            errors.append("input/package records differ from compiler-bound sources")
        expected_input_checks = _input_invariant_checks(
            request, room, scenes, compile_checks, snapshots
        )
    except (OSError, ValueError, KeyError, AcousticCanaryError) as exc:
        errors.append(f"unable to revalidate compiler/input lineage: {exc}")
        expected_input_checks = []

    repeat_count = request.get("repeat_count")
    execution = evidence.get("execution", {})
    order = execution.get("condition_order", [])
    conditions = evidence.get("conditions", {})
    overall_status = evidence.get("overall_status")
    if isinstance(repeat_count, int):
        expected_order: list[str] = []
        for repeat_index in range(repeat_count):
            pair = _CONDITIONS if repeat_index % 2 == 0 else tuple(reversed(_CONDITIONS))
            expected_order.extend(pair)
        if order != expected_order:
            errors.append("condition_order is not the complete alternating 2N schedule")
        expected_execution = {
            "repeat_count": repeat_count,
            "condition_order": expected_order,
            "fresh_context_per_run": True,
            "temporal_coherence": False,
            "runtime_import_workaround": dict(RUNTIME_IMPORT_WORKAROUND),
        }
        if execution != expected_execution:
            errors.append("execution record differs from the fixed canary schedule")
    all_runs: list[Mapping[str, Any]] = []
    for condition in _CONDITIONS:
        condition_value = conditions.get(condition, {})
        runs = condition_value.get("runs", [])
        if not isinstance(runs, list):
            errors.append(f"conditions.{condition}.runs is not an array")
            continue
        all_runs.extend(run for run in runs if isinstance(run, Mapping))
        if condition_value.get("completed_run_count") != len(runs):
            errors.append(f"conditions.{condition}.completed_run_count is inconsistent")
        attempted = condition_value.get("attempted_run_count")
        if not isinstance(attempted, int) or attempted < len(runs):
            errors.append(f"conditions.{condition}.attempted_run_count is inconsistent")
        if overall_status == "pass" and len(runs) != repeat_count:
            errors.append(f"conditions.{condition} does not contain repeat_count runs")
        repeat_indices = sorted(
            run.get("repeat_index") for run in runs if isinstance(run, Mapping)
        )
        if overall_status == "pass" and repeat_indices != list(range(repeat_count)):
            errors.append(f"conditions.{condition} repeat indices are not complete")
    execution_indices = sorted(
        run.get("execution_index") for run in all_runs if isinstance(run, Mapping)
    )
    if execution_indices != list(range(len(all_runs))):
        errors.append("run execution indices are not unique and contiguous")
    actual_order = [
        run["condition"]
        for run in sorted(all_runs, key=lambda item: item["execution_index"])
    ]
    if isinstance(order, list) and actual_order != order[: len(all_runs)]:
        errors.append("completed runs are not a prefix of condition_order")

    check_by_id = {
        check.get("check_id"): check
        for check in evidence.get("checks", [])
        if isinstance(check, Mapping)
    }
    if len(check_by_id) != len(evidence.get("checks", [])):
        errors.append("check_id values must be unique")
    for expected in expected_input_checks:
        observed = check_by_id.get(expected["check_id"])
        if observed != expected:
            errors.append(f"check {expected['check_id']} differs from recomputed input check")

    metric_config = MetricConfig()
    source_position = request.get("source", {}).get("position_m", [])
    listener_position = request.get("listener", {}).get("position_m", [])
    try:
        distance_m = math.dist(source_position, listener_position)
        expected_arrival = (
            distance_m
            / float(request["simulation"]["speed_of_sound_m_s"])
            * float(request["simulation"]["sample_rate_hz"])
        )
    except (KeyError, TypeError, ValueError):
        expected_arrival = math.nan
        distance_m = math.nan
    for run in all_runs:
        run_id = str(run.get("run_id"))
        condition = run.get("condition")
        if condition not in _CONDITIONS or condition not in scenes:
            errors.append(f"{run_id}.condition has no validated package")
            continue
        repeat_index_value = run.get("repeat_index")
        expected_run_id = (
            f"{condition}_repeat_{repeat_index_value:03d}"
            if isinstance(repeat_index_value, int)
            and not isinstance(repeat_index_value, bool)
            else None
        )
        if expected_run_id is None or run_id != expected_run_id:
            errors.append(f"{run_id}.run_id differs from condition/repeat_index")
        errors.extend(
            _runtime_evidence_errors(
                run,
                scene=scenes[condition],
                request=request,
                room=room,
                base=base,
                owner=run_id,
                snapshot_cache=artifact_snapshot_cache,
            )
        )
        artifact = run.get("ir_artifact")
        ir_path, artifact_error = _confined_artifact(base, artifact, f"{run_id}.ir")
        if artifact_error:
            errors.append(artifact_error)
            continue
        assert ir_path is not None and isinstance(artifact, Mapping)
        ir_payload, record_error = _verified_file_snapshot(
            ir_path,
            artifact,
            f"{run_id}.ir",
            cache=artifact_snapshot_cache,
        )
        if record_error:
            errors.append(record_error)
            continue
        assert ir_payload is not None
        try:
            samples = np.load(io.BytesIO(ir_payload), allow_pickle=False)
        except (OSError, ValueError) as exc:
            errors.append(f"{run_id}.ir cannot be loaded: {exc}")
            continue
        array_record = run.get("ir_array", {})
        if (
            samples.dtype.str != "<f4"
            or list(samples.shape) != array_record.get("shape")
            or samples.dtype.str != array_record.get("dtype")
            or not samples.flags.c_contiguous
            or not np.all(np.isfinite(samples))
        ):
            errors.append(f"{run_id}.ir dtype/shape/finite contract failed")
            continue
        if _array_sha256(samples) != array_record.get("raw_array_sha256"):
            errors.append(f"{run_id}.ir raw array hash changed")
        try:
            measured = analyze_ir(
                samples,
                float(request["simulation"]["sample_rate_hz"]),
                configuration=metric_config,
            ).to_dict()
        except (AcousticMetricError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{run_id}.metrics cannot be recomputed: {exc}")
            continue
        if canonical_json_sha256(measured) != canonical_json_sha256(run.get("metrics")):
            errors.append(f"{run_id}.metrics differ from raw IR recomputation")
        arrival = run.get("direct_arrival", {})
        arrival_error = abs(measured["direct_arrival_sample"] - expected_arrival)
        expected_direct = {
            "distance_m": distance_m,
            "speed_of_sound_m_s": request["simulation"]["speed_of_sound_m_s"],
            "expected_sample": expected_arrival,
            "detected_sample": measured["direct_arrival_sample"],
            "absolute_error_samples": arrival_error,
            "maximum_absolute_error_samples": request["thresholds"]["direct_arrival"][
                "maximum_absolute_error_samples"
            ],
            "passed": arrival_error
            <= request["thresholds"]["direct_arrival"][
                "maximum_absolute_error_samples"
            ],
        }
        if canonical_json_sha256(arrival) != canonical_json_sha256(expected_direct):
            errors.append(f"{run_id}.direct_arrival differs from geometry recomputation")
        direct_check = check_by_id.get(f"{run_id}_direct_arrival")
        expected_direct_check = _check(
            f"{run_id}_direct_arrival",
            expected_direct["passed"],
            measured=expected_direct,
            threshold={
                "maximum_absolute_error_samples": request["thresholds"][
                    "direct_arrival"
                ]["maximum_absolute_error_samples"]
            },
            failure_reason=(
                "Detected direct arrival disagrees with metric geometry"
            ),
        )
        if direct_check != expected_direct_check:
            errors.append(f"{run_id} direct-arrival check differs from recomputation")
        edt_threshold = request["thresholds"]["metrics"]["edt_seconds"]
        edt_passed = (
            measured["edt_fit_r2"] >= edt_threshold["minimum_fit_r2"]
            and measured["edt_decay_span_db"] >= edt_threshold["minimum_decay_span_db"]
        )
        edt_check = check_by_id.get(f"{run_id}_edt_quality")
        expected_edt_check = _check(
            f"{run_id}_edt_quality",
            edt_passed,
            measured={
                "fit_r2": measured["edt_fit_r2"],
                "decay_span_db": measured["edt_decay_span_db"],
            },
            threshold={
                "minimum_fit_r2": edt_threshold["minimum_fit_r2"],
                "minimum_decay_span_db": edt_threshold[
                    "minimum_decay_span_db"
                ],
            },
            failure_reason="EDT fit quality is below the declared threshold",
        )
        if edt_check != expected_edt_check:
            errors.append(f"{run_id} EDT-quality check differs from recomputation")
        ray_passed = bool(run.get("ray_checks")) and all(
            report.get("passed") is True for report in run.get("ray_checks", [])
        )
        ray_check = check_by_id.get(f"{run_id}_rlr_ray_checks")
        expected_ray_check = _check(
            f"{run_id}_rlr_ray_checks",
            ray_passed,
            measured=run.get("ray_checks"),
            threshold={
                "declared_count": len(room.get("ray_checks", [])),
                "all_cpu_rlr_consistent_and_expected": True,
                "maximum_first_hit_distance_error_m": request["thresholds"][
                    "ray_checks"
                ]["maximum_first_hit_distance_error_m"],
            },
            failure_reason=(
                "RLR opening/control rays disagree with CPU reference"
            ),
        )
        if ray_check != expected_ray_check:
            errors.append(f"{run_id} RLR-ray check differs from recomputation")

    comparisons = evidence.get("comparisons", {})
    repeats_complete = all(
        len(conditions.get(condition, {}).get("runs", [])) == repeat_count
        for condition in _CONDITIONS
    )
    if repeats_complete:
        for metric_name in _METRICS:
            expected_comparison = _metric_comparison(
                metric_name,
                [
                    float(run["metrics"][metric_name])
                    for run in conditions["low_absorption"]["runs"]
                ],
                [
                    float(run["metrics"][metric_name])
                    for run in conditions["high_absorption"]["runs"]
                ],
                request["thresholds"]["metrics"][metric_name],
            )
            if canonical_json_sha256(comparisons.get(metric_name)) != canonical_json_sha256(
                expected_comparison
            ):
                errors.append(f"comparison {metric_name} differs from run metrics")
            comparison_check = check_by_id.get(f"material_effect_{metric_name}")
            expected_comparison_check = _check(
                f"material_effect_{metric_name}",
                expected_comparison["passed"],
                measured=expected_comparison,
                threshold=request["thresholds"]["metrics"][metric_name],
                failure_reason=(
                    f"{metric_name} did not prove a directional material effect "
                    "larger than repeat spread"
                ),
            )
            if comparison_check != expected_comparison_check:
                errors.append(
                    f"comparison check {metric_name} differs from recomputation"
                )
    elif comparisons:
        errors.append("comparisons must be empty when repeats are incomplete")

    execution_blocked = any(
        check.get("status") == "blocked"
        and check.get("check_id")
        not in {
            "low_absorption_repeat_count",
            "high_absorption_repeat_count",
            "runtime_native_binary_identity",
        }
        for check in evidence.get("checks", [])
        if isinstance(check, Mapping)
    )
    for condition in _CONDITIONS:
        repeat_check = check_by_id.get(f"{condition}_repeat_count")
        completed = len(conditions.get(condition, {}).get("runs", []))
        expected_repeat_check = _check(
            f"{condition}_repeat_count",
            completed == repeat_count,
            measured=completed,
            threshold=repeat_count,
            failure_reason=(
                f"{condition} did not complete every independent repeat"
            ),
            blocked=execution_blocked,
        )
        if repeat_check != expected_repeat_check:
            errors.append(f"{condition} repeat-count check differs from recomputation")

    binary_records = [
        run.get("runtime", {}).get("native_binaries")
        for run in all_runs
        if isinstance(run.get("runtime"), Mapping)
    ]
    binary_identity = bool(binary_records) and all(
        isinstance(record, Mapping) and record == binary_records[0]
        for record in binary_records
    )
    expected_binary_check = _check(
        "runtime_native_binary_identity",
        binary_identity,
        measured={
            "completed_run_count": len(all_runs),
            "unique_record_count": len(
                {
                    canonical_json_sha256(record)
                    for record in binary_records
                    if isinstance(record, Mapping)
                }
            ),
            "native_binaries": binary_records[0] if binary_identity else None,
        },
        threshold={
            "same_hash_bound_native_binaries_every_repeat": True,
            "runtime_lock_sha256": verified_runtime_lock_sha256,
        },
        failure_reason=(
            "Runs did not use one hash-bound Habitat/RLR binary pair"
        ),
        blocked=execution_blocked,
    )
    if check_by_id.get("runtime_native_binary_identity") != expected_binary_check:
        errors.append(
            "runtime_native_binary_identity check differs from recomputation"
        )

    expected_check_ids = {
        check["check_id"] for check in expected_input_checks
    }
    expected_check_ids.update(
        f"{run.get('run_id')}_{suffix}"
        for run in all_runs
        for suffix in ("direct_arrival", "edt_quality", "rlr_ray_checks")
    )
    expected_check_ids.update(
        {
            "low_absorption_repeat_count",
            "high_absorption_repeat_count",
            "runtime_native_binary_identity",
        }
    )
    if repeats_complete:
        expected_check_ids.update(
            f"material_effect_{metric_name}" for metric_name in _METRICS
        )

    attempted_counts = {
        condition: conditions.get(condition, {}).get("attempted_run_count")
        for condition in _CONDITIONS
    }
    attempted_total = (
        sum(attempted_counts.values())
        if all(isinstance(value, int) for value in attempted_counts.values())
        else -1
    )
    inputs_passed = bool(expected_input_checks) and all(
        check["status"] == "pass" for check in expected_input_checks
    )
    expected_attempted_total = (
        len(all_runs)
        if repeats_complete
        else (len(all_runs) + 1 if inputs_passed else 0)
    )
    if attempted_total != expected_attempted_total:
        errors.append(
            "attempted_run_count does not match the deterministic stop schedule"
        )
    elif isinstance(order, list):
        attempted_prefix = order[:attempted_total]
        for condition in _CONDITIONS:
            if attempted_counts[condition] != attempted_prefix.count(condition):
                errors.append(
                    f"conditions.{condition}.attempted_run_count differs from schedule"
                )

    if attempted_total == len(all_runs) + 1 and not repeats_complete:
        failed_execution_index = len(all_runs)
        failed_condition = (
            order[failed_execution_index]
            if isinstance(order, list) and failed_execution_index < len(order)
            else None
        )
        failed_repeat_index = failed_execution_index // len(_CONDITIONS)
        expected_runtime_failure_id = (
            f"runtime_{failed_condition}_repeat_{failed_repeat_index:03d}"
            if failed_condition in _CONDITIONS
            else None
        )
        unavailable = check_by_id.get("runtime_available")
        execution_failure = (
            check_by_id.get(expected_runtime_failure_id)
            if expected_runtime_failure_id is not None
            else None
        )
        if unavailable is not None and execution_failure is not None:
            errors.append("runtime failure has both unavailable and execution checks")
        elif unavailable is not None:
            expected_check_ids.add("runtime_available")
            measured = unavailable.get("measured")
            expected_unavailable = _check(
                "runtime_available",
                False,
                measured=measured,
                threshold="modern audio-enabled RLR binding available",
                failure_reason=str(measured),
                blocked=True,
            )
            if (
                not isinstance(measured, str)
                or not measured
                or unavailable != expected_unavailable
            ):
                errors.append("runtime_available check is structurally invalid")
        elif execution_failure is not None and expected_runtime_failure_id is not None:
            expected_check_ids.add(expected_runtime_failure_id)
            measured = execution_failure.get("measured")
            expected_failure = _check(
                expected_runtime_failure_id,
                False,
                measured=measured,
                threshold="complete valid RLR IR and metric result",
                failure_reason=str(measured),
            )
            if (
                not isinstance(measured, str)
                or not measured
                or execution_failure != expected_failure
            ):
                errors.append(
                    f"check {expected_runtime_failure_id} is structurally invalid"
                )
        else:
            errors.append("attempted incomplete execution has no runtime failure check")

    actual_check_ids = set(check_by_id)
    if expected_input_checks and actual_check_ids != expected_check_ids:
        errors.append(
            "check_id set differs from deterministic evidence contract: "
            f"missing={sorted(expected_check_ids - actual_check_ids)}, "
            f"unexpected={sorted(actual_check_ids - expected_check_ids)}"
        )

    checks = [item for item in evidence.get("checks", []) if item.get("required") is True]
    for check in checks:
        has_reason = isinstance(check.get("failure_reason"), str) and bool(
            check.get("failure_reason")
        )
        if (check.get("status") == "pass" and has_reason) or (
            check.get("status") != "pass" and not has_reason
        ):
            errors.append(f"check {check.get('check_id')} failure_reason is inconsistent")
    statuses = {check.get("status") for check in checks}
    expected_overall = (
        "fail" if "fail" in statuses else ("blocked" if "blocked" in statuses else "pass")
    )
    if overall_status != expected_overall:
        errors.append("overall_status does not match required check aggregation")
    expected_reasons = [
        check["failure_reason"] for check in checks if check.get("status") != "pass"
    ]
    if evidence.get("failure_reasons") != expected_reasons:
        errors.append("failure_reasons do not match failed/blocked checks")
    return errors


def load_and_verify_canary_evidence(
    evidence_path: str | Path,
    *,
    evidence_snapshot: ImmutableFileSnapshot | None = None,
) -> VerifiedCanaryEvidence:
    """Read evidence once, then parse, verify and expose those exact bytes.

    Command-line callers must use this result for both the declared status and
    verification outcome.  Reopening the JSON after verification would allow
    those two decisions to observe different file contents.
    """

    path = Path(evidence_path).resolve()
    try:
        if evidence_snapshot is None:
            evidence_snapshot = read_immutable_file_snapshot(path)
        elif evidence_snapshot.path.resolve() != path:
            raise ValueError(
                "provided canary-evidence snapshot path does not match evidence_path"
            )
        value = json.loads(evidence_snapshot.payload)
        if not isinstance(value, dict):
            raise ValueError("canary evidence JSON root must be an object")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return VerifiedCanaryEvidence(
            evidence_path=path,
            evidence_snapshot=evidence_snapshot,
            evidence={},
            errors=(f"unable to load canary evidence: {exc}",),
        )

    try:
        errors = _verify_canary_evidence_document(path, value)
    except (
        AcousticCanaryError,
        AcousticMetricError,
        KeyError,
        OSError,
        RuntimeContractError,
        TypeError,
        ValueError,
    ) as exc:
        errors = [f"unable to verify malformed canary evidence: {exc}"]
    return VerifiedCanaryEvidence(
        evidence_path=path,
        evidence_snapshot=evidence_snapshot,
        evidence=value,
        errors=tuple(errors),
    )


def verify_canary_evidence(evidence_path: str | Path) -> list[str]:
    """Compatibility wrapper returning errors for one immutable snapshot."""

    return list(load_and_verify_canary_evidence(evidence_path).errors)


__all__ = [
    "AcousticCanaryError",
    "CANARY_EVIDENCE_SCHEMA",
    "VerifiedCanaryEvidence",
    "load_and_verify_canary_evidence",
    "run_material_activation_canary",
    "verify_canary_evidence",
]
