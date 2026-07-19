from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image

from avengine.contracts.json_io import (
    canonical_json_bytes,
    canonical_json_sha256,
    file_record,
    load_json,
    resolve_declared_path,
    sha256_file,
)
from avengine.contracts.transforms import (
    compose_transforms,
    invert_transform,
    round_trip_via_parent,
    transform_error,
)
from avengine.m1.contracts import (
    EVIDENCE_SCHEMA,
    ROOM_KINDS,
    STATUS_VALUES,
    ValidatedM1Inputs,
    aggregate_status,
    validate_capture_request,
    validate_recorded_scene_asset_graph,
    validate_room_manifest,
    validate_scene_asset_graph,
)
from avengine.runtime_lock import RuntimeLockError, resolve_runtime_profile


def make_check(
    check_id: str,
    status: str,
    *,
    measured: Any,
    threshold: Any,
    required: bool = True,
    artifact: str | None = None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "check_id": check_id,
        "required": required,
        "status": status,
        "measured": measured,
        "threshold": threshold,
    }
    if artifact is not None:
        value["artifact"] = artifact
    if failure_reason is not None:
        value["failure_reason"] = failure_reason
    return value


def array_sha256(sensor_uuid: str, array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = canonical_json_bytes(
        {
            "sensor_uuid": sensor_uuid,
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "memory_order": "C",
        }
    )
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\x00")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _save_depth_preview(path: Path, depth: np.ndarray) -> None:
    finite_positive = np.isfinite(depth) & (depth > 0)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if finite_positive.any():
        high = float(np.percentile(depth[finite_positive], 99.0))
        high = max(high, 1e-6)
        normalized = np.clip(depth / high, 0.0, 1.0)
        preview[finite_positive] = np.asarray(
            (1.0 - normalized[finite_positive]) * 255.0, dtype=np.uint8
        )
    Image.fromarray(preview, mode="L").save(path)


def _semantic_color(semantic_id: int) -> tuple[int, int, int]:
    if semantic_id == 0:
        return (0, 0, 0)
    digest = hashlib.sha256(str(semantic_id).encode("ascii")).digest()
    return tuple(64 + (channel % 192) for channel in digest[:3])


def _save_semantic_preview(path: Path, semantic: np.ndarray) -> None:
    preview = np.zeros((*semantic.shape, 3), dtype=np.uint8)
    for semantic_id in np.unique(semantic):
        preview[semantic == semantic_id] = _semantic_color(int(semantic_id))
    Image.fromarray(preview, mode="RGB").save(path)


def save_observations(
    observations: dict[str, np.ndarray],
    modality_to_uuid: dict[str, str],
    output_dir: str | Path,
) -> dict[str, dict[str, Any]]:
    output = Path(output_dir)
    observation_dir = output / "observations"
    observation_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}

    rgb = np.ascontiguousarray(observations[modality_to_uuid["rgb"]]).copy()
    if rgb.ndim != 3 or rgb.shape[-1] not in {3, 4}:
        raise ValueError(f"RGB observation must be HxWx3/4, got {rgb.shape}")
    rgb_path = observation_dir / "rgb.png"
    Image.fromarray(rgb).save(rgb_path)
    rgb_color = rgb[..., :3].astype(np.float64)
    spatial_standard_deviation = float(rgb_color.std())
    per_channel_standard_deviation = [
        float(rgb_color[..., channel].std()) for channel in range(3)
    ]
    records["rgb"] = {
        "sensor_uuid": modality_to_uuid["rgb"],
        "dtype": str(rgb.dtype),
        "shape": list(rgb.shape),
        "raw_array_sha256": array_sha256(modality_to_uuid["rgb"], rgb),
        "artifact": file_record(rgb_path, relative_to=output),
        "statistics": {
            "color_minimum": int(rgb[..., :3].min()),
            "color_maximum": int(rgb[..., :3].max()),
            "color_standard_deviation": spatial_standard_deviation,
            "per_channel_standard_deviation": per_channel_standard_deviation,
            "alpha_present": rgb.shape[-1] == 4,
            "alpha_unique_values": (
                [int(value) for value in np.unique(rgb[..., 3])]
                if rgb.shape[-1] == 4
                else []
            ),
        },
    }

    depth = np.ascontiguousarray(observations[modality_to_uuid["depth"]]).copy()
    depth_path = observation_dir / "depth.npy"
    depth_preview_path = observation_dir / "depth_preview.png"
    np.save(depth_path, depth, allow_pickle=False)
    _save_depth_preview(depth_preview_path, depth)
    finite_positive = np.isfinite(depth) & (depth > 0)
    records["depth"] = {
        "sensor_uuid": modality_to_uuid["depth"],
        "dtype": str(depth.dtype),
        "shape": list(depth.shape),
        "raw_array_sha256": array_sha256(modality_to_uuid["depth"], depth),
        "artifact": file_record(depth_path, relative_to=output),
        "preview_artifact": file_record(depth_preview_path, relative_to=output),
        "statistics": {
            "finite_fraction": float(np.isfinite(depth).mean()),
            "finite_positive_fraction": float(finite_positive.mean()),
            "minimum_positive_m": (
                float(depth[finite_positive].min()) if finite_positive.any() else None
            ),
            "maximum_finite_m": (
                float(depth[np.isfinite(depth)].max())
                if np.isfinite(depth).any()
                else None
            ),
        },
    }

    semantic = np.ascontiguousarray(observations[modality_to_uuid["semantic"]]).copy()
    semantic_path = observation_dir / "semantic.npy"
    semantic_preview_path = observation_dir / "semantic_preview.png"
    np.save(semantic_path, semantic, allow_pickle=False)
    _save_semantic_preview(semantic_preview_path, semantic)
    unique_ids = [int(value) for value in np.unique(semantic)]
    records["semantic"] = {
        "sensor_uuid": modality_to_uuid["semantic"],
        "dtype": str(semantic.dtype),
        "shape": list(semantic.shape),
        "raw_array_sha256": array_sha256(modality_to_uuid["semantic"], semantic),
        "artifact": file_record(semantic_path, relative_to=output),
        "preview_artifact": file_record(semantic_preview_path, relative_to=output),
        "statistics": {
            "unique_id_count": len(unique_ids),
            "unique_ids": unique_ids,
        },
    }
    return records


BASE_REQUIRED_CHECKS = {
    "single_formal_view",
    "runtime_commit_matches_lock",
    "runtime_worktree_clean",
    "runtime_binary_origin",
    "avengine_worktree_clean",
    "capture_state_unchanged",
    "repeatability_same_process",
    "independent_process_repeatability",
    "rig_visual_listener_alignment",
    "rgb_nonconstant",
    "depth_valid",
    "semantic_nontrivial_raw_ids",
    "named_source_transform_roundtrip",
    "scene_asset_closure",
    "scene_load_graph_closure",
}


@lru_cache(maxsize=1)
def _evidence_validator() -> Draft202012Validator:
    source_path = (
        Path(__file__).resolve().parents[3]
        / "schemas"
        / "m1_visual_evidence_v1.schema.json"
    )
    installed_path = (
        Path(sys.prefix)
        / "share"
        / "avengine"
        / "schemas"
        / "m1_visual_evidence_v1.schema.json"
    )
    schema_path = source_path if source_path.is_file() else installed_path
    schema = load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _schema_errors(value: Any) -> list[str]:
    errors = sorted(
        _evidence_validator().iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    reports: list[str] = []
    for error in errors:
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        reports.append(f"{location}: {error.message}")
    return reports


def _all_json_numbers_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_all_json_numbers_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_json_numbers_finite(item) for item in value.values())
    return False


def _safe_artifact_path(base: Path, raw_path: Any) -> tuple[Path | None, str | None]:
    if not isinstance(raw_path, str) or not raw_path:
        return None, "artifact path is not a non-empty string"
    declared = Path(raw_path)
    if declared.is_absolute() or ".." in declared.parts:
        return None, "artifact path must be a confined relative path"
    candidate = (base / declared).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None, "artifact path or symlink escapes the evidence directory"
    return candidate, None


def _artifact_check(
    *,
    base: Path,
    owner: str,
    key: str,
    artifact: Any,
) -> tuple[dict[str, Any], Path | None]:
    expected = artifact if isinstance(artifact, dict) else {}
    path, path_error = _safe_artifact_path(base, expected.get("path"))
    exists = path is not None and path.is_file()
    actual_hash = sha256_file(path) if exists and path is not None else None
    actual_size = path.stat().st_size if exists and path is not None else None
    passed = bool(
        path_error is None
        and exists
        and actual_hash == expected.get("sha256")
        and actual_size == expected.get("byte_size")
    )
    return (
        make_check(
            f"artifact_{owner}_{key}",
            "pass" if passed else "fail",
            measured={
                "exists": exists,
                "sha256": actual_hash,
                "byte_size": actual_size,
                "path_error": path_error,
            },
            threshold={
                "sha256": expected.get("sha256"),
                "byte_size": expected.get("byte_size"),
                "confined_to_evidence_directory": True,
            },
            artifact=(
                expected.get("path") if isinstance(expected.get("path"), str) else None
            ),
            failure_reason=None
            if passed
            else "Artifact is missing, changed, or outside the evidence directory",
        ),
        path if passed else None,
    )


def _is_within(path: Path, roots: list[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def _declared_file_check(
    check_id: str,
    record: Any,
    *,
    allowed_roots: list[Path],
    path_key: str = "path",
) -> tuple[dict[str, Any], Path | None]:
    declared = record if isinstance(record, dict) else {}
    raw_path = declared.get(path_key)
    path = Path(raw_path).resolve() if isinstance(raw_path, str) and raw_path else None
    path_allowed = path is not None and _is_within(path, allowed_roots)
    exists = bool(path_allowed and path is not None and path.is_file())
    actual_hash = sha256_file(path) if exists and path is not None else None
    actual_size = path.stat().st_size if exists and path is not None else None
    expected_size = declared.get("byte_size")
    passed = bool(
        exists
        and actual_hash == declared.get("sha256")
        and (expected_size is None or actual_size == expected_size)
    )
    return (
        make_check(
            check_id,
            "pass" if passed else "fail",
            measured={
                "path": raw_path,
                "path_allowed": path_allowed,
                "exists": exists,
                "sha256": actual_hash,
                "byte_size": actual_size,
            },
            threshold={
                "sha256": declared.get("sha256"),
                "byte_size": expected_size,
                "path_within_evidence_repo_or_runtime": True,
            },
            failure_reason=None
            if passed
            else "Declared input/scene asset changed or is outside allowed roots",
        ),
        path if passed else None,
    )


def _git_output(repository: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if getattr(result, "returncode", 0) != 0:
        return None
    return result.stdout.strip()


def _content_hash_check(evidence: dict[str, Any]) -> dict[str, Any]:
    expected = evidence.get("evidence_content_sha256")
    try:
        actual = hashlib.sha256(
            canonical_json_bytes(
                {
                    key: value
                    for key, value in evidence.items()
                    if key != "evidence_content_sha256"
                }
            )
        ).hexdigest()
    except (TypeError, ValueError):
        actual = None
    passed = isinstance(expected, str) and actual == expected
    return make_check(
        "evidence_content_hash",
        "pass" if passed else "fail",
        measured=actual,
        threshold=expected,
        failure_reason=None if passed else "Evidence content hash changed",
    )


def _close(left: Any, right: Any, *, atol: float = 1e-9) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    try:
        return bool(math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=atol))
    except (TypeError, ValueError, OverflowError):
        return False


def _vec_close(left: Any, right: Any, *, atol: float = 1e-6) -> bool:
    try:
        return bool(
            np.allclose(
                np.asarray(left, dtype=np.float64),
                np.asarray(right, dtype=np.float64),
                rtol=0.0,
                atol=atol,
            )
        )
    except (TypeError, ValueError):
        return False


def _required_evidence_check_ids(evidence: dict[str, Any]) -> set[str]:
    required = set(BASE_REQUIRED_CHECKS)
    room_kind = evidence.get("room_kind")
    if room_kind == "blender_custom":
        required.add("blender_authored_surface_provenance")
    elif room_kind == "legacy_ue_real_surface_export":
        required.add("legacy_real_surface_provenance")
    for report in evidence.get("connectivity", []):
        required.add(f"connectivity_{report['pair_id']}")
    for report in evidence.get("ray_checks", []):
        required.add(f"ray_{report['check_id']}")
    for report in evidence.get("qa_observations", []):
        required.add(f"qa_{report['qa_id']}")
    return required


def _check_declared_profile(
    evidence: dict[str, Any], *, allow_reference: bool
) -> tuple[list[dict[str, Any]], bool]:
    verification: list[dict[str, Any]] = []
    declared = evidence["checks"]
    identifiers = [check["check_id"] for check in declared]
    unique = len(identifiers) == len(set(identifiers))
    indexed = {check["check_id"]: check for check in declared}
    required_ids = _required_evidence_check_ids(evidence)
    missing = sorted(required_ids - set(indexed))
    wrong_required = sorted(
        check_id
        for check_id in required_ids & set(indexed)
        if indexed[check_id].get("required") is not True
    )
    wrong_status = sorted(
        check_id
        for check_id in required_ids & set(indexed)
        if indexed[check_id].get("status") != "pass"
        and not (
            check_id == "independent_process_repeatability"
            and indexed[check_id].get("status") == "not_run"
        )
    )
    independent = indexed.get("independent_process_repeatability", {})
    required_nonpass = [
        check
        for check in declared
        if check.get("required") is True and check.get("status") != "pass"
    ]
    first_run_profile = bool(
        len(required_nonpass) == 1
        and required_nonpass[0].get("check_id") == "independent_process_repeatability"
        and required_nonpass[0].get("status") == "not_run"
        and independent.get("required") is True
    )
    all_required_pass = not required_nonpass
    profile_passed = bool(unique and not missing and not wrong_required)
    verification.append(
        make_check(
            "evidence_check_profile",
            "pass" if profile_passed else "fail",
            measured={
                "declared_check_count": len(declared),
                "unique_check_ids": unique,
                "missing_required_check_ids": missing,
                "required_check_ids_marked_optional": wrong_required,
                "required_check_ids_with_invalid_status": wrong_status,
                "all_required_pass": all_required_pass,
                "first_run_profile": first_run_profile,
            },
            threshold={
                "unique_check_ids": True,
                "missing_required_check_ids": [],
                "all_profile_checks_required": True,
                "statuses": "reported separately through overall_status",
            },
            failure_reason=None
            if profile_passed
            else "Declared checks do not satisfy the executable M1 profile",
        )
    )
    recomputed = aggregate_status(declared)
    declared_status = evidence["overall_status"]
    self_consistent = recomputed == declared_status
    final_pass = self_consistent and all_required_pass and declared_status == "pass"
    reference_pass = (
        self_consistent and first_run_profile and declared_status == "not_run"
    )
    if final_pass or (allow_reference and reference_pass):
        verification_status = "pass"
    elif self_consistent and declared_status in STATUS_VALUES:
        verification_status = declared_status
    else:
        verification_status = "fail"
    verification.append(
        make_check(
            "evidence_overall_status",
            verification_status,
            measured={
                "declared": declared_status,
                "recomputed": recomputed,
                "final_pass": final_pass,
                "valid_first_run_reference": reference_pass,
            },
            threshold={
                "declared_equals_recomputed": True,
                "completion_status": "pass",
            },
            failure_reason=None
            if verification_status == "pass"
            else "Evidence is not a self-consistent passing completion run",
        )
    )
    return verification, first_run_profile


def _load_validated_inputs(
    evidence: dict[str, Any], room_path: Path | None, request_path: Path | None
) -> tuple[dict[str, Any], ValidatedM1Inputs | None]:
    measured: dict[str, Any] = {}
    inputs: ValidatedM1Inputs | None = None
    try:
        if room_path is None or request_path is None:
            raise ValueError("Input file integrity checks did not produce usable paths")
        room = load_json(room_path)
        request = load_json(request_path)
        room_errors = validate_room_manifest(room)
        request_errors = validate_capture_request(request, room_id=room.get("room_id"))
        identity_matches = bool(
            room.get("room_id") == evidence.get("room_id")
            and room.get("room_kind") == evidence.get("room_kind")
            and request.get("request_id") == evidence.get("request_id")
        )
        measured = {
            "room_errors": room_errors,
            "request_errors": request_errors,
            "identity_matches": identity_matches,
        }
        if not room_errors and not request_errors and identity_matches:
            inputs = ValidatedM1Inputs(
                room_path=room_path,
                request_path=request_path,
                room=room,
                request=request,
            )
    except (OSError, ValueError, TypeError) as error:
        measured = {"exception": f"{type(error).__name__}: {error}"}
    passed = inputs is not None
    return (
        make_check(
            "evidence_input_contracts",
            "pass" if passed else "fail",
            measured=measured,
            threshold={"schemas_and_semantics_valid": True, "identity_matches": True},
            failure_reason=None
            if passed
            else "Room/request inputs no longer satisfy their executable contracts",
        ),
        inputs,
    )


def _sensor_source_state_checks(
    evidence: dict[str, Any], inputs: ValidatedM1Inputs
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    rig = inputs.request["primary_camera_rig"]
    sensor_contract = evidence["sensor_contract"]
    contract_passed = bool(
        sensor_contract.get("rig_id") == rig["rig_id"] == "camera_rig_0"
        and sensor_contract.get("view_id") == rig["view_id"] == "view0"
        and sensor_contract.get("world_from_rig") == rig["world_from_rig"]
        and sensor_contract.get("shared_calibration") == rig["shared_calibration"]
        and sensor_contract.get("modalities") == rig["modalities"]
        and sensor_contract.get("listener") == inputs.request["listener"]
        and sensor_contract.get("audio_propagation_status") == "not_run"
        and "M4" in sensor_contract.get("audio_propagation_reason", "")
    )
    checks.append(
        make_check(
            "evidence_sensor_contract",
            "pass" if contract_passed else "fail",
            measured={
                "rig_id": sensor_contract.get("rig_id"),
                "view_id": sensor_contract.get("view_id"),
                "modalities": sensor_contract.get("modalities"),
                "listener": sensor_contract.get("listener"),
                "audio_propagation_status": sensor_contract.get(
                    "audio_propagation_status"
                ),
            },
            threshold={
                "exact_request_contract": True,
                "one_view_three_colocated_modalities": True,
                "listener_colocated": True,
                "audio_deferred_to_m4": True,
            },
            failure_reason=None
            if contract_passed
            else "Sensor evidence diverges from the single-view request contract",
        )
    )

    modality_to_uuid = {
        item["modality"]: item["sensor_uuid"] for item in rig["modalities"]
    }
    expected_uuids = {
        *modality_to_uuid.values(),
        inputs.request["listener"]["listener_id"],
    }
    expected_pose = compose_transforms(
        rig["world_from_rig"], rig["shared_calibration"]["rig_from_sensor"]
    )
    state = evidence["capture_state"]
    pose_errors: dict[str, dict[str, float]] = {}
    state_passed = True
    for snapshot_name in ("before", "after"):
        snapshot = state[snapshot_name]
        sensors = snapshot["sensors"]
        item_errors: dict[str, float] = {}
        if set(sensors) != expected_uuids:
            state_passed = False
        for uuid in expected_uuids & set(sensors):
            item_errors[uuid] = transform_error(sensors[uuid], expected_pose)
        agent_error = transform_error(snapshot["agent"], rig["world_from_rig"])
        item_errors["agent"] = agent_error
        pose_errors[snapshot_name] = item_errors
        state_passed = bool(
            state_passed
            and snapshot["world_time_seconds"] == 0.0
            and item_errors
            and max(item_errors.values()) <= 1e-7
        )
    checks.append(
        make_check(
            "evidence_sensor_state_alignment",
            "pass" if state_passed else "fail",
            measured={
                "expected_sensor_uuids": sorted(expected_uuids),
                "pose_errors": pose_errors,
            },
            threshold={
                "exact_sensor_set": True,
                "maximum_transform_error": 1e-7,
                "world_time_seconds": 0.0,
            },
            failure_reason=None
            if state_passed
            else "Read-back sensor/listener state is not the shared formal viewpoint",
        )
    )

    source_reports = evidence["sources"]
    request_sources = inputs.request["sources"]
    source_passed = len(source_reports) == len(request_sources) >= 2
    source_errors: dict[str, Any] = {}
    rig_from_world = invert_transform(rig["world_from_rig"])
    for expected, report in zip(request_sources, source_reports, strict=False):
        source_id = expected["source_id"]
        expected_rig = compose_transforms(rig_from_world, expected["world_from_source"])
        recovered, roundtrip_error = round_trip_via_parent(
            rig["world_from_rig"], expected["world_from_source"]
        )
        item_passed = bool(
            report.get("source_id") == source_id
            and transform_error(
                report["world_from_source"], expected["world_from_source"]
            )
            <= 1e-9
            and transform_error(report["rig_from_source"], expected_rig) <= 1e-9
            and transform_error(report["recovered_world_from_source"], recovered)
            <= 1e-9
            and _close(report.get("roundtrip_max_error"), roundtrip_error)
            and roundtrip_error <= 1e-9
        )
        source_errors[source_id] = {
            "reported_id": report.get("source_id"),
            "roundtrip_error": roundtrip_error,
            "passed": item_passed,
        }
        source_passed = source_passed and item_passed
    checks.append(
        make_check(
            "evidence_named_source_roundtrip",
            "pass" if source_passed else "fail",
            measured={"source_count": len(source_reports), "sources": source_errors},
            threshold={"minimum_source_count": 2, "exact_request_roundtrip": True},
            failure_reason=None
            if source_passed
            else "Named source evidence is missing, reordered, or geometrically invalid",
        )
    )
    return checks


def _runtime_check(evidence: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    runtime = evidence["runtime"]
    repository_root = Path(__file__).resolve().parents[3]
    runtime_root = Path(runtime["habitat_runtime_root"]).resolve()
    module_path = Path(runtime["habitat_module_path"]).resolve()
    binding_path = Path(runtime["native_binding_path"]).resolve()
    paths_within_runtime = bool(
        _is_within(module_path, [runtime_root])
        and _is_within(binding_path, [runtime_root])
        and module_path.is_file()
        and binding_path.is_file()
    )
    current_runtime_commit = _git_output(runtime_root, "rev-parse", "HEAD")
    runtime_status = _git_output(runtime_root, "status", "--porcelain")
    current_avengine_commit = _git_output(repository_root, "rev-parse", "HEAD")
    avengine_status = _git_output(repository_root, "status", "--porcelain")
    current_binding_hash = sha256_file(binding_path) if binding_path.is_file() else None
    try:
        lock_path = resolve_runtime_profile(repository_root, "m1")
    except RuntimeLockError:
        lock_path = repository_root / "runtime.lock.yaml"
    current_lock_hash = sha256_file(lock_path) if lock_path.is_file() else None
    lock_text = lock_path.read_text(encoding="utf-8") if lock_path.is_file() else ""
    match = re.search(
        r"^habitat_runtime:\s*$.*?^\s+fork_governance_commit:\s+([0-9a-f]{40})\s*$",
        lock_text,
        flags=re.MULTILINE | re.DOTALL,
    )
    lock_commit = match.group(1) if match else None
    passed = bool(
        paths_within_runtime
        and current_runtime_commit
        == runtime.get("habitat_runtime_commit")
        == runtime.get("locked_habitat_runtime_commit")
        == lock_commit
        and runtime_status == ""
        and runtime.get("habitat_runtime_worktree_dirty") is False
        and current_binding_hash == runtime.get("native_binding_sha256")
        and current_avengine_commit == runtime.get("avengine_commit")
        and avengine_status == ""
        and runtime.get("avengine_worktree_dirty") is False
        and current_lock_hash == runtime.get("runtime_lock_sha256")
    )
    return (
        make_check(
            "evidence_runtime_identity",
            "pass" if passed else "fail",
            measured={
                "paths_within_runtime": paths_within_runtime,
                "current_runtime_commit": current_runtime_commit,
                "locked_runtime_commit": lock_commit,
                "runtime_status": runtime_status,
                "current_native_binding_sha256": current_binding_hash,
                "current_avengine_commit": current_avengine_commit,
                "avengine_status": avengine_status,
                "current_runtime_lock_sha256": current_lock_hash,
            },
            threshold={
                "locked_clean_runtime_and_binary_match": True,
                "clean_avengine_commit_matches": True,
                "runtime_lock_hash_matches": True,
            },
            failure_reason=None
            if passed
            else "Current runtime/main checkout or runtime lock differs from evidence",
        ),
        runtime_root if runtime_root.is_dir() else None,
    )


def _scene_asset_checks(
    evidence: dict[str, Any],
    inputs: ValidatedM1Inputs,
    *,
    base: Path,
    runtime_root: Path | None,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    repository_root = Path(__file__).resolve().parents[3]
    allowed = [base, repository_root, inputs.room_path.parent]
    if runtime_root is not None:
        allowed.append(runtime_root)
    source_root_raw = inputs.room.get("provenance", {}).get("source_repository_root")
    if isinstance(source_root_raw, str) and source_root_raw:
        allowed.append(Path(source_root_raw).resolve())
    records = evidence["scene_assets"]
    roles = [record["role"] for record in records]
    declared = [(asset["role"], asset["path"]) for asset in inputs.room["assets"]]
    recorded = [(record["role"], record["declared_path"]) for record in records]
    resolution_environment = dict(os.environ)
    if runtime_root is not None:
        resolution_environment["AVENGINE_HABITAT_RUNTIME_ROOT"] = str(runtime_root)
    resolved_path_reports: list[dict[str, Any]] = []
    resolved_paths_match = len(records) == len(inputs.room["assets"])
    for asset, record in zip(inputs.room["assets"], records, strict=False):
        try:
            expected_path = resolve_declared_path(
                asset["path"],
                manifest_dir=inputs.room_path.parent,
                environment=resolution_environment,
            )
            recorded_path = Path(record["resolved_path"]).resolve()
            item_matches = expected_path == recorded_path
            resolved_path_reports.append(
                {
                    "role": asset["role"],
                    "declared_path": asset["path"],
                    "expected_resolved_path": str(expected_path),
                    "recorded_resolved_path": str(recorded_path),
                    "matches": item_matches,
                }
            )
            resolved_paths_match = resolved_paths_match and item_matches
        except (KeyError, OSError, TypeError, ValueError) as error:
            resolved_paths_match = False
            resolved_path_reports.append(
                {
                    "role": asset.get("role"),
                    "declared_path": asset.get("path"),
                    "error": f"{type(error).__name__}: {error}",
                    "matches": False,
                }
            )
    closure_passed = bool(
        records
        and len(roles) == len(set(roles))
        and recorded == declared
        and resolved_paths_match
        and all(record.get("exists") is True for record in records)
    )
    for index, record in enumerate(records):
        check, _ = _declared_file_check(
            f"scene_asset_{index}_{record['role']}",
            record,
            allowed_roots=allowed,
            path_key="resolved_path",
        )
        checks.append(check)
        closure_passed = closure_passed and check["status"] == "pass"
    checks.append(
        make_check(
            "evidence_scene_asset_closure",
            "pass" if closure_passed else "fail",
            measured={
                "count": len(records),
                "roles": roles,
                "matches_room_asset_order": recorded == declared,
                "resolved_paths_match_declarations": resolved_paths_match,
                "resolved_path_reports": resolved_path_reports,
            },
            threshold={
                "exact_room_asset_closure": True,
                "declared_paths_re_resolve_exactly": True,
                "unique_roles": True,
                "all_hashes_current": True,
            },
            failure_reason=None
            if closure_passed
            else "Scene asset closure is incomplete, reordered, or changed",
        )
    )
    declared_scene_graph_check = next(
        (
            check
            for check in evidence["checks"]
            if check.get("check_id") == "scene_load_graph_closure"
        ),
        None,
    )
    loaded_snapshot = (
        declared_scene_graph_check.get("measured", {}).get("loaded_graph")
        if isinstance(declared_scene_graph_check, dict)
        and isinstance(declared_scene_graph_check.get("measured"), dict)
        else None
    )
    if runtime_root is not None:
        static_scene_graph_errors = validate_scene_asset_graph(inputs, runtime_root)
        recorded_scene_graph_errors = validate_recorded_scene_asset_graph(
            inputs, runtime_root, loaded_snapshot
        )
    else:
        static_scene_graph_errors = ["current Habitat runtime root is unavailable"]
        recorded_scene_graph_errors = [
            "loaded Habitat graph cannot be replayed without the runtime root"
        ]
    scene_graph_errors = static_scene_graph_errors + recorded_scene_graph_errors
    checks.append(
        make_check(
            "evidence_scene_load_graph_closure",
            "pass" if not scene_graph_errors else "fail",
            measured={
                "errors": scene_graph_errors,
                "static_errors": static_scene_graph_errors,
                "recorded_loaded_graph_errors": recorded_scene_graph_errors,
            },
            threshold={
                "errors": [],
                "actual_habitat_scene_resolves_to_declared_assets": True,
            },
            failure_reason=None
            if not scene_graph_errors
            else "Habitat scene graph no longer resolves to the hashed room assets",
        )
    )
    if evidence["room_kind"] in {
        "blender_custom",
        "legacy_ue_real_surface_export",
    }:
        from avengine.m1.habitat_capture import _surface_provenance_check

        replay = _surface_provenance_check(inputs, records)
        replay_passed = replay is not None and replay["status"] == "pass"
        checks.append(
            make_check(
                "evidence_surface_provenance_replay",
                "pass" if replay_passed else "fail",
                measured=replay.get("measured") if replay is not None else None,
                threshold={"capture_provenance_gate_replayed": True},
                failure_reason=None
                if replay_passed
                else "Custom/legacy real-surface provenance no longer validates",
            )
        )
    return checks


def _observation_checks(
    evidence: dict[str, Any], inputs: ValidatedM1Inputs, *, base: Path
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    observations = evidence["observations"]
    paths: dict[str, Path] = {}
    for modality in ("rgb", "depth", "semantic"):
        record = observations[modality]
        keys = ["artifact"]
        if modality in {"depth", "semantic"}:
            keys.append("preview_artifact")
        for key in keys:
            check, path = _artifact_check(
                base=base,
                owner=modality,
                key=key,
                artifact=record[key],
            )
            checks.append(check)
            if key == "artifact" and path is not None:
                paths[modality] = path

    arrays: dict[str, np.ndarray] = {}
    measured: dict[str, Any] = {}
    raw_passed = set(paths) == {"rgb", "depth", "semantic"}
    if raw_passed:
        try:
            with Image.open(paths["rgb"]) as image:
                arrays["rgb"] = np.asarray(image).copy()
            arrays["depth"] = np.load(paths["depth"], allow_pickle=False)
            arrays["semantic"] = np.load(paths["semantic"], allow_pickle=False)
            for modality, array in arrays.items():
                record = observations[modality]
                digest = array_sha256(record["sensor_uuid"], array)
                item_passed = bool(
                    digest == record["raw_array_sha256"]
                    and str(array.dtype) == record["dtype"]
                    and list(array.shape) == record["shape"]
                )
                measured[modality] = {
                    "raw_array_sha256": digest,
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "passed": item_passed,
                }
                raw_passed = raw_passed and item_passed
        except (OSError, ValueError, TypeError) as error:
            raw_passed = False
            measured["exception"] = f"{type(error).__name__}: {error}"
    checks.append(
        make_check(
            "evidence_raw_observation_hashes",
            "pass" if raw_passed else "fail",
            measured=measured,
            threshold={"all_raw_array_hashes_match": True},
            failure_reason=None if raw_passed else "Raw observation hash mismatch",
        )
    )

    semantic_passed = raw_passed
    semantic_measured: dict[str, Any] = {}
    if raw_passed:
        rig = inputs.request["primary_camera_rig"]
        calibration = rig["shared_calibration"]
        height, width = calibration["resolution_hw"]
        modality_to_uuid = {
            item["modality"]: item["sensor_uuid"] for item in rig["modalities"]
        }
        rgb = arrays["rgb"]
        rgb_color = rgb[..., :3].astype(np.float64)
        rgb_stats = observations["rgb"]["statistics"]
        actual_rgb = {
            "color_minimum": int(rgb[..., :3].min()),
            "color_maximum": int(rgb[..., :3].max()),
            "color_standard_deviation": float(rgb_color.std()),
            "per_channel_standard_deviation": [
                float(rgb_color[..., index].std()) for index in range(3)
            ],
            "alpha_present": rgb.shape[-1] == 4,
            "alpha_unique_values": (
                [int(value) for value in np.unique(rgb[..., 3])]
                if rgb.shape[-1] == 4
                else []
            ),
        }
        rgb_passed = bool(
            rgb.dtype == np.uint8
            and list(rgb.shape[:2]) == [height, width]
            and rgb.ndim == 3
            and rgb.shape[-1] in {3, 4}
            and observations["rgb"]["sensor_uuid"] == modality_to_uuid["rgb"]
            and actual_rgb["color_standard_deviation"] > 1.0
            and max(actual_rgb["per_channel_standard_deviation"]) > 1.0
            and rgb_stats.get("color_minimum") == actual_rgb["color_minimum"]
            and rgb_stats.get("color_maximum") == actual_rgb["color_maximum"]
            and _close(
                rgb_stats.get("color_standard_deviation"),
                actual_rgb["color_standard_deviation"],
            )
            and all(
                _close(left, right)
                for left, right in zip(
                    rgb_stats.get("per_channel_standard_deviation", []),
                    actual_rgb["per_channel_standard_deviation"],
                    strict=False,
                )
            )
            and len(rgb_stats.get("per_channel_standard_deviation", [])) == 3
            and rgb_stats.get("alpha_present") == actual_rgb["alpha_present"]
            and rgb_stats.get("alpha_unique_values")
            == actual_rgb["alpha_unique_values"]
        )

        depth = arrays["depth"]
        finite = np.isfinite(depth)
        positive = finite & (depth > 0)
        actual_depth = {
            "finite_fraction": float(finite.mean()),
            "finite_positive_fraction": float(positive.mean()),
            "minimum_positive_m": (
                float(depth[positive].min()) if positive.any() else None
            ),
            "maximum_finite_m": (float(depth[finite].max()) if finite.any() else None),
        }
        depth_stats = observations["depth"]["statistics"]
        depth_passed = bool(
            np.issubdtype(depth.dtype, np.floating)
            and list(depth.shape) == [height, width]
            and observations["depth"]["sensor_uuid"] == modality_to_uuid["depth"]
            and actual_depth["finite_positive_fraction"] > 0.05
            and actual_depth["maximum_finite_m"] is not None
            and actual_depth["maximum_finite_m"] <= calibration["far_m"] + 1e-4
            and all(
                _close(depth_stats.get(key), value)
                for key, value in actual_depth.items()
            )
        )

        semantic = arrays["semantic"]
        unique_ids = [int(value) for value in np.unique(semantic)]
        declared_ids = {
            int(value)
            for value in inputs.room.get("semantics", {}).get("id_to_label", {})
            if str(value).lstrip("-").isdigit()
        }
        expected_nonzero = declared_ids - {0}
        semantic_stats = observations["semantic"]["statistics"]
        semantic_array_passed = bool(
            np.issubdtype(semantic.dtype, np.integer)
            and list(semantic.shape) == [height, width]
            and observations["semantic"]["sensor_uuid"] == modality_to_uuid["semantic"]
            and len(unique_ids) > 1
            and all(value >= 0 for value in unique_ids)
            and (not expected_nonzero or bool(expected_nonzero & set(unique_ids)))
            and semantic_stats.get("unique_id_count") == len(unique_ids)
            and semantic_stats.get("unique_ids") == unique_ids
        )
        semantic_measured = {
            "rgb": {"actual": actual_rgb, "passed": rgb_passed},
            "depth": {"actual": actual_depth, "passed": depth_passed},
            "semantic": {
                "unique_ids": unique_ids,
                "declared_nonzero_ids": sorted(expected_nonzero),
                "passed": semantic_array_passed,
            },
        }
        semantic_passed = bool(
            semantic_passed and rgb_passed and depth_passed and semantic_array_passed
        )
    checks.append(
        make_check(
            "evidence_observation_semantics",
            "pass" if semantic_passed else "fail",
            measured=semantic_measured,
            threshold={
                "rgb_color_std_excluding_alpha_gt": 1.0,
                "depth_finite_positive_fraction_gt": 0.05,
                "semantic_unique_nonnegative_ids_gt": 1,
                "declared_statistics_recomputed": True,
            },
            failure_reason=None
            if semantic_passed
            else "Observation values or declared statistics fail M1 thresholds",
        )
    )

    repeats = evidence["repeat_observation_hashes"]
    first_hashes = {
        modality: observations[modality]["raw_array_sha256"]
        for modality in ("rgb", "depth", "semantic")
    }
    repeat_passed = bool(
        len(repeats) >= 2
        and all(item == repeats[0] for item in repeats)
        and repeats[0] == first_hashes
    )
    checks.append(
        make_check(
            "evidence_repeat_hashes",
            "pass" if repeat_passed else "fail",
            measured={
                "repeat_count": len(repeats),
                "first_matches_saved_observations": repeats[0] == first_hashes,
            },
            threshold={"minimum_repeat_count": 2, "all_repeats_identical": True},
            failure_reason=None
            if repeat_passed
            else "Same-process repeat hashes are inconsistent",
        )
    )
    return checks


def _topology_qa_checks(
    evidence: dict[str, Any], inputs: ValidatedM1Inputs, *, base: Path
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    connectivity = evidence["connectivity"]
    connectivity_by_id = {report["pair_id"]: report for report in connectivity}
    expected_pairs = inputs.room["connectivity_pairs"]
    connectivity_passed = bool(
        len(connectivity_by_id) == len(connectivity) == len(expected_pairs)
        and set(connectivity_by_id) == {pair["pair_id"] for pair in expected_pairs}
    )
    connectivity_measured: dict[str, Any] = {}
    for pair in expected_pairs:
        report = connectivity_by_id.get(pair["pair_id"])
        item_passed = bool(
            report is not None
            and _vec_close(report["requested_start_m"], pair["start_m"])
            and _vec_close(report["requested_end_m"], pair["end_m"])
            and report["found"] is True
            and report["start_snap_distance_m"] <= 0.30
            and report["end_snap_distance_m"] <= 0.30
            and report["geodesic_distance_m"] >= 0.0
            and report["path_point_count"] >= 2
        )
        connectivity_measured[pair["pair_id"]] = {
            "report": report,
            "passed": item_passed,
        }
        connectivity_passed = connectivity_passed and item_passed
    checks.append(
        make_check(
            "evidence_connectivity_semantics",
            "pass" if connectivity_passed else "fail",
            measured=connectivity_measured,
            threshold={
                "exact_declared_pairs": True,
                "found": True,
                "maximum_snap_distance_m": 0.30,
                "minimum_path_points": 2,
            },
            failure_reason=None
            if connectivity_passed
            else "Connectivity reports do not satisfy declared room pairs",
        )
    )

    ray_reports = evidence["ray_checks"]
    ray_by_id = {report["check_id"]: report for report in ray_reports}
    expected_rays = inputs.room["ray_checks"]
    rays_passed = bool(
        len(ray_by_id) == len(ray_reports) == len(expected_rays)
        and set(ray_by_id) == {ray["check_id"] for ray in expected_rays}
    )
    ray_measured: dict[str, Any] = {}
    for expected in expected_rays:
        report = ray_by_id.get(expected["check_id"])
        threshold_passed = False
        if report is not None:
            nearest = report["nearest_hit_m"]
            if expected["expectation"] == "clear_until_m":
                threshold_passed = (
                    nearest is None or nearest >= expected["distance_m"] - 1e-6
                )
            else:
                threshold_passed = (
                    nearest is not None and nearest <= expected["distance_m"] + 1e-6
                )
        item_passed = bool(
            report is not None
            and report["expectation"] == expected["expectation"]
            and _close(report["distance_m"], expected["distance_m"])
            and report["passed"] is True
            and threshold_passed
        )
        ray_measured[expected["check_id"]] = {
            "report": report,
            "passed": item_passed,
        }
        rays_passed = rays_passed and item_passed
    if inputs.room["room_kind"] == "blender_custom":
        rays_passed = bool(
            rays_passed
            and {item["expectation"] for item in expected_rays}
            == {"clear_until_m", "hit_within_m"}
        )
    checks.append(
        make_check(
            "evidence_ray_semantics",
            "pass" if rays_passed else "fail",
            measured=ray_measured,
            threshold={
                "exact_declared_rays": True,
                "all_thresholds_pass": True,
                "custom_has_clear_and_solid_controls": True,
            },
            failure_reason=None
            if rays_passed
            else "Opening/control ray reports do not satisfy declared thresholds",
        )
    )

    qa_reports = evidence["qa_observations"]
    qa_by_id = {report["qa_id"]: report for report in qa_reports}
    expected_qa = inputs.request["qa_views"]
    qa_passed = bool(
        len(qa_by_id) == len(qa_reports) == len(expected_qa)
        and set(qa_by_id) == {view["qa_id"] for view in expected_qa}
    )
    qa_measured: dict[str, Any] = {}
    for expected in expected_qa:
        report = qa_by_id.get(expected["qa_id"])
        artifact_path: Path | None = None
        if report is not None:
            artifact_check, artifact_path = _artifact_check(
                base=base,
                owner=f"qa_{expected['qa_id']}",
                key="artifact",
                artifact=report["artifact"],
            )
            checks.append(artifact_check)
        actual_shape: list[int] | None = None
        navigable_pixels: int | None = None
        if artifact_path is not None:
            try:
                with Image.open(artifact_path) as image:
                    topdown = np.asarray(image.convert("L"))
                actual_shape = list(topdown.shape)
                navigable_pixels = int(np.count_nonzero(topdown))
            except OSError:
                pass
        item_passed = bool(
            report is not None
            and report["kind"] == expected["kind"] == "topdown"
            and report["formal_view"] is False
            and _close(report["meters_per_pixel"], expected["meters_per_pixel"])
            and _close(report["height_m"], expected["height_m"])
            and report["shape"] == actual_shape
            and report["navigable_pixel_count"] == navigable_pixels
            and navigable_pixels is not None
            and navigable_pixels > 0
        )
        qa_measured[expected["qa_id"]] = {
            "actual_shape": actual_shape,
            "actual_navigable_pixel_count": navigable_pixels,
            "passed": item_passed,
        }
        qa_passed = qa_passed and item_passed
    checks.append(
        make_check(
            "evidence_qa_semantics",
            "pass" if qa_passed else "fail",
            measured=qa_measured,
            threshold={
                "exact_qa_only_views": True,
                "formal_view": False,
                "minimum_navigable_pixels": 1,
            },
            failure_reason=None
            if qa_passed
            else "QA top-down records or pixels do not match the request",
        )
    )
    return checks


def _state_and_batch_checks(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    state = evidence["capture_state"]
    before_hash = canonical_json_sha256(state["before"])
    after_hash = canonical_json_sha256(state["after"])
    state_passed = bool(
        before_hash == state["before_sha256"]
        and after_hash == state["after_sha256"]
        and before_hash == after_hash
    )
    checks.append(
        make_check(
            "evidence_capture_state_hashes",
            "pass" if state_passed else "fail",
            measured={"before": before_hash, "after": after_hash},
            threshold={"before_equals_after": True, "declared_hashes_match": True},
            failure_reason=None
            if state_passed
            else "Capture state hashes are invalid or state advanced",
        )
    )
    runtime = evidence["runtime"]
    repeats = evidence["repeat_observation_hashes"]
    expected_batch = canonical_json_sha256(
        {
            "room_manifest_sha256": evidence["room_manifest"]["sha256"],
            "capture_request_sha256": evidence["capture_request"]["sha256"],
            "scene_assets": evidence["scene_assets"],
            "avengine_commit": runtime["avengine_commit"],
            "habitat_runtime_commit": runtime["habitat_runtime_commit"],
            "native_binding_sha256": runtime["native_binding_sha256"],
            "state": state["before"],
            "repeat_count": len(repeats),
        }
    )
    batch_passed = evidence["capture_batch_id"] == expected_batch
    checks.append(
        make_check(
            "evidence_capture_batch_id",
            "pass" if batch_passed else "fail",
            measured=expected_batch,
            threshold=evidence["capture_batch_id"],
            failure_reason=None if batch_passed else "Capture batch identity changed",
        )
    )
    return checks


def _independent_reference_check(
    evidence: dict[str, Any], *, base: Path, allow_reference: bool
) -> list[dict[str, Any]]:
    record = evidence["independent_reference"]
    if record is None:
        status = "pass" if allow_reference else "not_run"
        return [
            make_check(
                "evidence_independent_reference",
                status,
                measured={"reference": None},
                threshold={"self_contained_valid_reference": True},
                failure_reason=None
                if allow_reference
                else "Completion requires a self-contained first-run reference",
            )
        ]
    artifact_check, reference_path = _artifact_check(
        base=base,
        owner="independent_reference",
        key="artifact",
        artifact=record["artifact"],
    )
    checks = [artifact_check]
    passed = bool(record["path"] == record["artifact"]["path"])
    measured: dict[str, Any] = {
        "path_matches_artifact": passed,
        "reference_verification_status": None,
        "comparisons": {},
    }
    if reference_path is not None:
        try:
            reference_status, reference_checks = verify_evidence_artifacts(
                reference_path, _allow_reference=True
            )
            reference = load_json(reference_path)
            current_hashes = {
                modality: evidence["observations"][modality]["raw_array_sha256"]
                for modality in ("rgb", "depth", "semantic")
            }
            reference_hashes = {
                modality: reference["observations"][modality]["raw_array_sha256"]
                for modality in ("rgb", "depth", "semantic")
            }
            comparisons = {
                "reference_verified": reference_status == "pass",
                "content_hash_matches_record": reference.get("evidence_content_sha256")
                == record["evidence_content_sha256"],
                "reference_is_first_run": reference.get("overall_status") == "not_run"
                and reference.get("independent_reference") is None,
                "room_and_request_match": reference.get("room_id")
                == evidence["room_id"]
                and reference.get("request_id") == evidence["request_id"],
                "input_hashes_match": reference.get("room_manifest", {}).get("sha256")
                == evidence["room_manifest"]["sha256"]
                and reference.get("capture_request", {}).get("sha256")
                == evidence["capture_request"]["sha256"],
                "batch_matches": reference.get("capture_batch_id")
                == evidence["capture_batch_id"],
                "scene_assets_match": reference.get("scene_assets")
                == evidence["scene_assets"],
                "runtime_identity_matches": all(
                    reference.get("runtime", {}).get(key) == evidence["runtime"][key]
                    for key in (
                        "avengine_commit",
                        "habitat_runtime_commit",
                        "native_binding_sha256",
                        "runtime_lock_sha256",
                    )
                ),
                "initial_state_matches": reference.get("capture_state", {}).get(
                    "before_sha256"
                )
                == evidence["capture_state"]["before_sha256"],
                "observation_hashes_match": reference_hashes == current_hashes,
                "fresh_process_instance": reference.get("producer_process", {}).get(
                    "process_instance_id"
                )
                != evidence.get("producer_process", {}).get("process_instance_id"),
            }
            measured = {
                "path_matches_artifact": passed,
                "reference_verification_status": reference_status,
                "reference_verification_checks": reference_checks,
                "comparisons": comparisons,
            }
            passed = passed and all(comparisons.values())
        except (OSError, ValueError, TypeError, KeyError) as error:
            passed = False
            measured["exception"] = f"{type(error).__name__}: {error}"
    else:
        passed = False
    checks.append(
        make_check(
            "evidence_independent_reference",
            "pass" if passed else "fail",
            measured=measured,
            threshold={
                "self_contained_valid_first_run": True,
                "all_identity_asset_state_and_observation_comparisons": True,
            },
            failure_reason=None
            if passed
            else "Independent-process reference is missing, mutable, or inconsistent",
        )
    )
    return checks


def _verify_blocked_attempt(
    evidence: dict[str, Any], checks: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    declared = evidence["checks"]
    valid_declared = bool(
        len(declared) == 1
        and declared[0]["check_id"] == "capture_execution"
        and declared[0]["required"] is True
        and declared[0]["status"] == "blocked"
        and evidence["overall_status"] == "blocked"
        and aggregate_status(declared) == "blocked"
    )
    checks.append(
        make_check(
            "blocked_attempt_contract",
            "pass" if valid_declared else "fail",
            measured={
                "declared_status": evidence["overall_status"],
                "checks": declared,
            },
            threshold={"one_required_capture_execution_check": "blocked"},
            failure_reason=None
            if valid_declared
            else "Blocked attempt is not self-consistent",
        )
    )
    checks.append(
        make_check(
            "capture_attempt_result",
            "blocked",
            measured=evidence["exception"],
            threshold={"capture_completed": True},
            failure_reason=evidence["exception"]["message"],
        )
    )
    return aggregate_status(checks), checks


def verify_evidence_artifacts(
    evidence_path: str | Path, *, _allow_reference: bool = False
) -> tuple[str, list[dict[str, Any]]]:
    resolved = Path(evidence_path).resolve()
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            evidence = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return (
            "fail",
            [
                make_check(
                    "evidence_json",
                    "fail",
                    measured=f"{type(error).__name__}: {error}",
                    threshold="readable JSON object",
                    failure_reason="Evidence could not be loaded",
                )
            ],
        )
    if not isinstance(evidence, dict):
        return (
            "fail",
            [
                make_check(
                    "evidence_object",
                    "fail",
                    measured=type(evidence).__name__,
                    threshold="object",
                    failure_reason="Evidence root must be a JSON object",
                )
            ],
        )

    checks: list[dict[str, Any]] = []
    schema_name_passed = evidence.get("schema") == EVIDENCE_SCHEMA
    checks.append(
        make_check(
            "evidence_schema_name",
            "pass" if schema_name_passed else "fail",
            measured=evidence.get("schema"),
            threshold=EVIDENCE_SCHEMA,
            failure_reason=None
            if schema_name_passed
            else "Unexpected evidence schema name",
        )
    )
    schema_errors = _schema_errors(evidence)
    checks.append(
        make_check(
            "evidence_json_schema",
            "pass" if not schema_errors else "fail",
            measured={"errors": schema_errors},
            threshold={"errors": []},
            failure_reason=None
            if not schema_errors
            else "Evidence fails Draft 2020-12 schema validation",
        )
    )
    finite = _all_json_numbers_finite(evidence)
    checks.append(
        make_check(
            "evidence_finite_numbers",
            "pass" if finite else "fail",
            measured={"all_json_numbers_finite": finite},
            threshold={"all_json_numbers_finite": True},
            failure_reason=None if finite else "Evidence contains NaN or infinity",
        )
    )
    if not schema_name_passed or schema_errors or not finite:
        return aggregate_status(checks), checks

    checks.append(_content_hash_check(evidence))
    base = resolved.parent
    repository_root = Path(__file__).resolve().parents[3]
    room_raw = Path(evidence["room_manifest"]["path"]).resolve()
    request_raw = Path(evidence["capture_request"]["path"]).resolve()
    input_roots = [
        base,
        repository_root,
        room_raw.parent,
        request_raw.parent,
    ]
    room_check, room_path = _declared_file_check(
        "room_manifest_file",
        evidence["room_manifest"],
        allowed_roots=input_roots,
    )
    request_check, request_path = _declared_file_check(
        "capture_request_file",
        evidence["capture_request"],
        allowed_roots=input_roots,
    )
    checks.extend([room_check, request_check])

    if evidence["evidence_kind"] == "blocked_attempt":
        return _verify_blocked_attempt(evidence, checks)

    identity_passed = bool(
        evidence["room_kind"] in ROOM_KINDS and evidence["formal_view_ids"] == ["view0"]
    )
    checks.append(
        make_check(
            "evidence_identity",
            "pass" if identity_passed else "fail",
            measured={
                "room_id": evidence["room_id"],
                "request_id": evidence["request_id"],
                "room_kind": evidence["room_kind"],
                "formal_view_ids": evidence["formal_view_ids"],
            },
            threshold={"room_kind": sorted(ROOM_KINDS), "formal_view_ids": ["view0"]},
            failure_reason=None if identity_passed else "Evidence identity is invalid",
        )
    )
    profile_checks, _ = _check_declared_profile(
        evidence, allow_reference=_allow_reference
    )
    checks.extend(profile_checks)
    input_check, inputs = _load_validated_inputs(evidence, room_path, request_path)
    checks.append(input_check)
    runtime_check, runtime_root = _runtime_check(evidence)
    checks.append(runtime_check)
    if inputs is not None:
        checks.extend(_sensor_source_state_checks(evidence, inputs))
        checks.extend(
            _scene_asset_checks(
                evidence,
                inputs,
                base=base,
                runtime_root=runtime_root,
            )
        )
        checks.extend(_observation_checks(evidence, inputs, base=base))
        checks.extend(_topology_qa_checks(evidence, inputs, base=base))
    else:
        checks.append(
            make_check(
                "evidence_semantic_replay",
                "fail",
                measured={"validated_inputs": False},
                threshold={"validated_inputs": True},
                failure_reason="Cannot replay M1 semantics without valid inputs",
            )
        )
    checks.extend(_state_and_batch_checks(evidence))
    checks.extend(
        _independent_reference_check(
            evidence,
            base=base,
            allow_reference=_allow_reference,
        )
    )
    return aggregate_status(checks), checks


def finalize_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    evidence["overall_status"] = aggregate_status(evidence["checks"])
    evidence["evidence_content_sha256"] = hashlib.sha256(
        canonical_json_bytes(
            {
                key: value
                for key, value in evidence.items()
                if key != "evidence_content_sha256"
            }
        )
    ).hexdigest()
    return evidence
