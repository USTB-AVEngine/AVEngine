"""Read-only M6 qualification attempts for the representative room set.

The runner intentionally separates three things that are easy to conflate:

* a resource exists and matches its registry hash;
* a retained package/evidence bundle can be revalidated now; and
* Habitat/RLR/Blender was actually executed during this attempt.

Only the first two are performed here.  Native execution stays ``not_run`` unless
another canary supplies measured evidence and builds a new qualification report.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    load_json,
    sha256_file,
    write_json,
)
from avengine.acoustics.contracts import (
    AcousticSceneContractError,
    ValidatedAcousticScenePackage,
    load_and_validate_acoustic_scene_package,
)
from avengine.acoustics.gltf import extract_triangle_scene
from avengine.acoustics.qa import geometry_report
from avengine.timeline.canary import verify_m5_canary_evidence
from avengine.m6.qualification import (
    build_qualification_report,
    qualify_corrupted_acoustic_fixture,
    validate_qualification_report,
)
from avengine.m6.room_providers import provider_for_id
from avengine.m6.rooms import (
    find_acoustic_representation,
    find_room_record,
    load_room_registry,
    room_revision_key,
)
from avengine.security.path_policy import (
    PathPolicyError,
    WorkspacePathPolicy,
    atomic_publish_directory,
)


ATTEMPT_MANIFEST_SCHEMA = "avengine_m6_room_qualification_attempt_v1"
OBSERVATION_SCHEMA = "avengine_m6_room_qualification_observation_v1"
DERIVED_PROXY_SCHEMA = "avengine_m6_derived_acoustic_proxy_v1"
CANONICAL_ROOM_REGISTRY_PATH = "examples/m6/rooms/room_registry.json"
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
ATTEMPT_CASE_IDS = (
    "blender_custom_two_zone",
    "replicacad_apt_0",
    "legacy_ue_apartment",
    "mp3d_17DRP5sb8fy_raw",
    "mp3d_17DRP5sb8fy_derived",
    "independent_corrupted_fixture",
)


class RoomAttemptError(ValueError):
    """Raised when an attempt bundle is malformed or cannot be published."""


def _aggregate_attempt_status(*statuses: str | None) -> str:
    """Combine independent evidence states without weakening a hard failure."""

    observed = [status for status in statuses if status is not None]
    if not observed:
        return "not_run"
    if any(status not in {"pass", "fail", "blocked", "not_run"} for status in observed):
        return "fail"
    for candidate in ("fail", "blocked", "not_run"):
        if candidate in observed:
            return candidate
    return "pass"


def _proxy_descriptor_assessment(
    descriptor_path: Path | None,
    package_manifest_path: str | Path | None,
    raw_package_manifest_path: str | Path | None,
    *,
    descriptor_resolution_status: str | None = None,
    provider_manifest_path: Path | None = None,
    provider_manifest_status: str | None = None,
    provider_representation_status: str | None = None,
    room_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate one materialized package against its committed proxy descriptor."""

    errors: list[str] = []
    prerequisite_status = _aggregate_attempt_status(
        descriptor_resolution_status
        if descriptor_resolution_status is not None
        else ("pass" if descriptor_path is not None else "not_run"),
        "pass" if package_manifest_path is not None else "not_run",
        "pass" if raw_package_manifest_path is not None else "not_run",
        provider_manifest_status,
        provider_representation_status,
    )
    if descriptor_path is None:
        return {
            "status": prerequisite_status,
            "proxy_id": "mp3d_17DRP5sb8fy_acoustic_proxy_v2",
            "errors": ["committed proxy descriptor did not resolve"],
            "artifact_count": 0,
        }
    if package_manifest_path is None:
        return {
            "status": prerequisite_status,
            "proxy_id": "mp3d_17DRP5sb8fy_acoustic_proxy_v2",
            "errors": ["materialized proxy package manifest was not supplied"],
            "artifact_count": 0,
        }
    if raw_package_manifest_path is None:
        return {
            "status": prerequisite_status,
            "proxy_id": "mp3d_17DRP5sb8fy_acoustic_proxy_v2",
            "errors": ["immutable raw package manifest was not supplied"],
            "artifact_count": 0,
        }
    if provider_manifest_status is not None and provider_manifest_status != "pass":
        return {
            "status": prerequisite_status,
            "proxy_id": "mp3d_17DRP5sb8fy_acoustic_proxy_v2",
            "errors": [
                "provider output manifest did not resolve with status pass: "
                f"{provider_manifest_status}"
            ],
            "artifact_count": 0,
        }
    if (
        provider_representation_status is not None
        and provider_representation_status != "pass"
    ):
        return {
            "status": prerequisite_status,
            "proxy_id": "mp3d_17DRP5sb8fy_acoustic_proxy_v2",
            "errors": [
                "provider acoustic representation did not resolve with status pass: "
                f"{provider_representation_status}"
            ],
            "artifact_count": 0,
        }
    try:
        descriptor = load_json(descriptor_path)
        schema = load_json(
            Path(__file__).resolve().parents[3]
            / "schemas"
            / "m6_derived_acoustic_proxy_v1.schema.json"
        )
        errors.extend(
            f"descriptor schema {'.'.join(str(part) for part in error.path) or '$'}: "
            f"{error.message}"
            for error in sorted(
                Draft202012Validator(schema).iter_errors(descriptor),
                key=lambda item: list(item.path),
            )
        )
        descriptor_core = deepcopy(descriptor)
        declared_descriptor_hash = descriptor_core.pop(
            "descriptor_content_sha256", None
        )
        if declared_descriptor_hash != canonical_json_sha256(descriptor_core):
            errors.append("proxy descriptor content hash mismatch")

        manifest_path = Path(package_manifest_path).resolve(strict=True)
        package_root = manifest_path.parent
        package = load_json(manifest_path)
        declared_package = descriptor["package"]
        raw_manifest_path = Path(raw_package_manifest_path).resolve(strict=True)
        provider_path = (
            provider_manifest_path.resolve(strict=True)
            if provider_manifest_path is not None
            else None
        )
        if provider_manifest_status == "pass" and provider_path is None:
            errors.append("provider passed without a materialized output manifest path")
        if provider_path is not None and provider_path != manifest_path:
            errors.append(
                "provider output and qualification candidate are different manifests"
            )
        if descriptor.get("proxy_id") != "mp3d_17DRP5sb8fy_acoustic_proxy_v2":
            errors.append("proxy descriptor stable ID differs")
        if descriptor.get("representation_id") != descriptor.get("proxy_id"):
            errors.append("proxy descriptor representation ID differs")
        if package.get("source_room", {}).get("room_id") != descriptor.get("room_id"):
            errors.append("materialized package room identity differs from descriptor")
        if package.get("package_id") != declared_package.get("package_id"):
            errors.append("materialized package ID differs from descriptor")
        if package.get("package_content_sha256") != declared_package.get(
            "package_content_sha256"
        ):
            errors.append("materialized package content identity differs")
        source = descriptor.get("source", {})
        if package.get("source_room", {}).get("geometry_asset_sha256") != source.get(
            "sha256"
        ):
            errors.append("materialized package raw geometry identity differs")
        if sha256_file(raw_manifest_path) != source.get(
            "raw_package_manifest_sha256"
        ):
            errors.append("immutable raw package manifest identity differs")

        if room_record is not None:
            try:
                representation = find_acoustic_representation(
                    room_record, descriptor["representation_id"]
                )
                raw_representation = find_acoustic_representation(
                    room_record, source["representation_id"]
                )
                descriptor_resources = [
                    resource
                    for resource in room_record["resources"]
                    if resource["resource_type"] == "derived_proxy_descriptor"
                ]
                if descriptor.get("room_id") != room_record.get("room_id"):
                    errors.append("proxy descriptor room differs from registry record")
                if representation.get("role") != "derived_proxy":
                    errors.append("registry representation is not a derived proxy")
                if representation.get("resource_id") != declared_package.get(
                    "resource_id"
                ):
                    errors.append("proxy output resource differs from registry")
                if representation.get("producer") != descriptor.get(
                    "derivation", {}
                ).get("producer"):
                    errors.append("proxy producer differs from registry")
                if representation.get("derived_from") != raw_representation.get(
                    "representation_id"
                ):
                    errors.append("proxy raw representation lineage differs")
                if raw_representation.get("resource_id") != source.get("resource_id"):
                    errors.append("proxy raw resource differs from registry")
                if source.get("resource_id") not in representation.get(
                    "input_resource_ids", []
                ):
                    errors.append("proxy raw resource is absent from registry inputs")
                if len(descriptor_resources) != 1 or descriptor_resources[0].get(
                    "resource_id"
                ) not in representation.get("input_resource_ids", []):
                    errors.append("proxy descriptor resource is absent from registry inputs")
                if package.get("materials", {}).get(
                    "material_semantics"
                ) != representation.get("material_semantics"):
                    errors.append("proxy material semantics differ from registry")
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"proxy registry contract is malformed: {error}")

        entries = sorted(package_root.rglob("*"))
        if any(path.is_symlink() for path in entries):
            errors.append("materialized proxy artifact closure contains a symlink")
        actual_records = [
            file_record(path, relative_to=package_root)
            for path in entries
            if path.is_file() and not path.is_symlink()
        ]
        if actual_records != declared_package.get("artifacts"):
            errors.append("materialized proxy artifact closure differs from descriptor")
        if canonical_json_sha256(actual_records) != declared_package.get(
            "artifact_set_sha256"
        ):
            errors.append("materialized proxy artifact-set hash differs")
        manifest_record = file_record(manifest_path, relative_to=package_root)
        if manifest_record != declared_package.get("manifest"):
            errors.append("materialized proxy manifest record differs")

        geometry = load_json(package_root / "qa" / "geometry_report.json")
        cleanup = geometry.get("research_cleanup", {})
        declared_cleanup = descriptor.get("derivation", {})
        cleanup_fields = {
            "policy": cleanup.get("policy"),
            "record_content_sha256": cleanup.get("record_content_sha256"),
            "removed_triangle_count": cleanup.get("removed_triangle_count"),
            "removed_triangle_indices_sha256": cleanup.get(
                "removed_triangle_indices_sha256"
            ),
            "removed_triangle_area_max_m2": cleanup.get(
                "removed_triangle_area_max_m2"
            ),
            "source_triangle_count": cleanup.get("source_triangle_count"),
            "derived_triangle_count": cleanup.get("derived_triangle_count"),
            "source_array_hashes": cleanup.get("source_arrays"),
            "derived_array_hashes": cleanup.get("derived_arrays"),
        }
        for key, observed in cleanup_fields.items():
            if declared_cleanup.get(key) != observed:
                errors.append(f"proxy derivation field {key} differs")
        if descriptor.get("qualification_claim") is not False:
            errors.append("proxy descriptor makes a qualification claim")
        if descriptor.get("dataset_admission") is not False:
            errors.append("proxy descriptor admits an unqualified package")
        return {
            "status": "pass" if not errors else "fail",
            "proxy_id": descriptor.get("proxy_id"),
            "descriptor_sha256": sha256_file(descriptor_path),
            "package_manifest_sha256": sha256_file(manifest_path),
            "provider_manifest_sha256": (
                sha256_file(provider_path) if provider_path is not None else None
            ),
            "same_materialized_manifest": (
                provider_path == manifest_path if provider_path is not None else None
            ),
            "package_content_sha256": package.get("package_content_sha256"),
            "artifact_count": len(actual_records),
            "artifact_set_sha256": canonical_json_sha256(actual_records),
            "errors": errors,
        }
    except (KeyError, OSError, TypeError, ValueError) as error:
        return {
            "status": "fail",
            "proxy_id": "mp3d_17DRP5sb8fy_acoustic_proxy_v2",
            "errors": [str(error)],
            "artifact_count": 0,
        }


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _git_state(repository_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
        clean = not bool(run("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "worktree_clean": None}
    return {"commit": commit, "branch": branch, "worktree_clean": clean}


def _redactions(repository_root: Path, environment: Mapping[str, str]) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = [(str(repository_root), "${AVENGINE_REPOSITORY_ROOT}")]
    for name in (
        "AVENGINE_REPLICACAD_ROOT",
        "AVENGINE_LEGACY_APARTMENT_EXPORT_ROOT",
        "AVENGINE_LEGACY_APARTMENT_PACKAGE_ROOT",
        "AVENGINE_HABITAT_RUNTIME_ROOT",
        "AVENGINE_MP3D_PROXY_V2_ROOT",
    ):
        raw = environment.get(name)
        if raw:
            values.append((str(Path(raw).expanduser().resolve()), f"${{{name}}}"))
    return sorted(values, key=lambda item: len(item[0]), reverse=True)


def _portable_text(value: str | None, redactions: Sequence[tuple[str, str]]) -> str | None:
    if value is None:
        return None
    result = value
    for private, replacement in redactions:
        result = result.replace(private, replacement)
    return result


def _repository_locator(path: Path, repository_root: Path) -> dict[str, str]:
    try:
        relative = path.resolve().relative_to(repository_root)
    except ValueError as error:
        raise RoomAttemptError(
            "qualification candidate artifacts must be inside repository_root"
        ) from error
    return {"kind": "repository_relative", "path": relative.as_posix()}


def _resource_observations(
    record: Mapping[str, Any],
    resolution: Any,
    *,
    redactions: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    declared = {item["resource_id"]: item for item in record["resources"]}
    observations: list[dict[str, Any]] = []
    for resource_id in sorted(resolution.resources):
        observed = resolution.resources[resource_id]
        source = declared[resource_id]
        actual_hash = None
        byte_size = None
        if observed.status == "pass" and observed.path is not None:
            actual_hash = sha256_file(observed.path)
            byte_size = observed.path.stat().st_size
        observations.append(
            {
                "resource_id": resource_id,
                "resource_type": source["resource_type"],
                "required_for": list(source["required_for"]),
                "declared_location": deepcopy(source["location"]),
                "status": observed.status,
                "reason": _portable_text(observed.reason, redactions),
                "byte_size": byte_size,
                "sha256": actual_hash,
                "declared_sha256": source.get("sha256"),
            }
        )
    return observations


def _provider_observation(
    registry: Mapping[str, Any],
    room_id: str,
    *,
    repository_root: Path,
    environment: Mapping[str, str],
    redactions: Sequence[tuple[str, str]],
) -> tuple[Mapping[str, Any], Any, dict[str, Any]]:
    record = find_room_record(registry, room_id)
    provider = provider_for_id(record["provider_id"])
    resolution = provider.resolve_room(
        record,
        repository_root=repository_root,
        environment=environment,
        verify_hash=True,
    )
    representation_results: list[dict[str, Any]] = []
    for representation in record["acoustic_representations"]:
        result = provider.acoustic_representation(
            record,
            representation["representation_id"],
            repository_root=repository_root,
            environment=environment,
            verify_hash=True,
        )
        representation_results.append(
            {
                "representation_id": result.representation_id,
                "role": representation["role"],
                "geometry_kind": representation["geometry_kind"],
                "build_mode": result.build_mode,
                "producer": result.producer,
                "status": result.status,
                "reason": _portable_text(result.reason, redactions),
                "input_resources": list(result.input_resources),
            }
        )
    value = {
        "room_key": room_revision_key(record),
        "provider_id": record["provider_id"],
        "registry_admission_state": record["admission_state"],
        "status": resolution.status,
        "dimension_resource_statuses": dict(resolution.dimension_statuses),
        "resources": _resource_observations(
            record, resolution, redactions=redactions
        ),
        "acoustic_representations": representation_results,
        "blockers": [
            _portable_text(item, redactions) for item in resolution.blockers
        ],
    }
    return record, resolution, value


def _topology_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    topology = report["topology"]
    return {
        "status": report["status"],
        "source_geometry_sha256": report["source_geometry_sha256"],
        "vertex_count": report["vertex_count"],
        "triangle_count": report["triangle_count"],
        "object_count": report["object_count"],
        "bounds_m": deepcopy(report["bounds_m"]),
        "source_to_canonical": deepcopy(report["source_to_canonical"]),
        "array_hashes": deepcopy(report["array_hashes"]),
        "topology": {
            name: topology[name]
            for name in (
                "degenerate_triangle_count",
                "duplicate_triangle_count",
                "boundary_edge_count_after_exact_weld",
                "nonmanifold_edge_count_after_exact_weld",
                "per_object_boundary_edge_count_after_exact_weld",
                "per_object_nonmanifold_edge_count_after_exact_weld",
            )
        },
    }


def _package_probe(
    raw_path: str | Path | None,
    *,
    candidate_id: str,
    repository_root: Path,
    redactions: Sequence[tuple[str, str]],
) -> tuple[dict[str, Any], ValidatedAcousticScenePackage | None]:
    if raw_path is None:
        return (
            {
                "candidate_id": candidate_id,
                "status": "not_run",
                "reason": "no candidate manifest was supplied",
            },
            None,
        )
    policy = WorkspacePathPolicy.from_roots([repository_root])
    try:
        path = policy.resolve_input(
            raw_path,
            owner=f"{candidate_id} manifest",
            kind="file",
        )
    except PathPolicyError as error:
        return (
            {
                "candidate_id": candidate_id,
                "status": "blocked",
                "reason": _portable_text(str(error), redactions),
            },
            None,
        )
    try:
        package = load_and_validate_acoustic_scene_package(path)
    except (AcousticSceneContractError, OSError, ValueError) as error:
        return (
            {
                "candidate_id": candidate_id,
                "status": "fail",
                "manifest": {
                    **_repository_locator(path, repository_root),
                    "byte_size": path.stat().st_size,
                    "sha256": sha256_file(path),
                },
                "reason": _portable_text(str(error), redactions),
            },
            None,
        )

    geometry = package.qa_reports["geometry_report"]
    coverage = package.qa_reports["material_coverage"]
    rays = package.qa_reports["ray_leakage"]
    parity = package.qa_reports["compiler_source_to_package_parity"]
    qa_records: dict[str, Any] = {}
    for name, record in package.manifest["qa"].items():
        qa_path = package.package_root / record["path"]
        qa_records[name] = {
            "path": record["path"],
            "byte_size": qa_path.stat().st_size,
            "sha256": sha256_file(qa_path),
        }
    observation = {
        "candidate_id": candidate_id,
        "status": "pass",
        "strict_package_validation": "pass",
        "manifest": {
            **_repository_locator(path, repository_root),
            "byte_size": package.manifest_byte_size,
            "sha256": package.manifest_file_sha256,
        },
        "source_geometry_sha256": geometry["source_geometry_sha256"],
        "geometry": _topology_summary(geometry),
        "material": {
            "semantics": package.manifest["materials"]["material_semantics"],
            "qualification_claim": package.manifest["materials"][
                "qualification_claim"
            ],
            "coverage_status": coverage["status"],
            "coverage_fraction": coverage["coverage_fraction"],
            "fallback_triangle_count": coverage["fallback_triangle_count"],
            "unmatched_triangle_count": coverage["unmatched_triangle_count"],
        },
        "ray_checks": {
            "status": rays["status"],
            "declared_check_count": rays["declared_check_count"],
            "rlr_runtime_ray_check_status": rays["rlr_runtime_ray_check_status"],
        },
        "compiler_source_to_package_parity": {
            "status": parity["status"],
            "comparison_scope": parity["comparison_scope"],
            "bounds_identical_within_m": parity["bounds_identical_within_m"],
            "source_geometry_sha256": parity["source_geometry_sha256"],
            "visual_runtime_parity_claim": parity.get(
                "visual_runtime_parity_claim"
            ),
        },
        "qa_artifacts": qa_records,
    }
    cleanup = geometry.get("research_cleanup")
    if cleanup is not None:
        observation["research_cleanup"] = {
            "policy": cleanup["policy"],
            "qualification_claim": cleanup["qualification_claim"],
            "record_content_sha256": cleanup["record_content_sha256"],
            "source_triangle_count": cleanup["source_triangle_count"],
            "source_vertex_count": cleanup["source_vertex_count"],
            "derived_triangle_count": cleanup["derived_triangle_count"],
            "derived_vertex_count": cleanup["derived_vertex_count"],
            "removed_triangle_count": cleanup["removed_triangle_count"],
            "removed_vertex_count": cleanup["removed_vertex_count"],
            "removed_triangle_area_max_m2": cleanup["removed_triangle_area_max_m2"],
            "source_arrays": deepcopy(cleanup["source_arrays"]),
            "derived_arrays": deepcopy(cleanup["derived_arrays"]),
        }
    return observation, package


def _declared_derivation_assessment(
    raw: ValidatedAcousticScenePackage | None,
    derived: ValidatedAcousticScenePackage | None,
) -> dict[str, Any]:
    if raw is None or derived is None:
        return {
            "status": "not_run",
            "reason": "both a strict-valid raw package and derived package are required",
            "checks": {},
        }
    raw_geometry = raw.qa_reports["geometry_report"]
    derived_geometry = derived.qa_reports["geometry_report"]
    cleanup = derived_geometry.get("research_cleanup")
    if not isinstance(cleanup, Mapping):
        return {
            "status": "fail",
            "reason": "derived package has no declared research_cleanup record",
            "checks": {"cleanup_record_present": False},
        }
    checks = {
        "source_geometry_identity_preserved": (
            raw_geometry["source_geometry_sha256"]
            == derived_geometry["source_geometry_sha256"]
        ),
        "source_transform_preserved": (
            raw_geometry["source_to_canonical"]
            == derived_geometry["source_to_canonical"]
        ),
        "source_bounds_preserved": (
            raw_geometry["bounds_m"] == derived_geometry["bounds_m"]
        ),
        "source_array_hashes_bound": (
            cleanup["source_arrays"]["vertices"]
            == raw_geometry["array_hashes"]["vertices"]
            and cleanup["source_arrays"]["triangles"]
            == raw_geometry["array_hashes"]["triangles"]
        ),
        "derived_array_hashes_bound": (
            cleanup["derived_arrays"]["vertices"]
            == derived_geometry["array_hashes"]["vertices"]
            and cleanup["derived_arrays"]["triangles"]
            == derived_geometry["array_hashes"]["triangles"]
        ),
        "triangle_count_delta_declared": (
            raw.triangle_count - derived.triangle_count
            == cleanup["removed_triangle_count"]
        ),
        "vertex_count_delta_declared": (
            raw.vertex_count - derived.vertex_count
            == cleanup["removed_vertex_count"]
        ),
        "removed_triangles_are_zero_area": (
            cleanup["removed_triangle_area_max_m2"] == 0
        ),
        "not_claimed_as_qualified": cleanup["qualification_claim"] is False,
    }
    passed = all(checks.values())
    return {
        "status": "pass" if passed else "fail",
        "reason": (
            "declared zero-area cleanup is internally bound to raw and derived arrays"
            if passed
            else "one or more declared cleanup lineage checks failed"
        ),
        "checks": checks,
        "legacy_byte_parity_status": derived.qa_reports[
            "compiler_source_to_package_parity"
        ]["status"],
        "legacy_byte_parity_interpretation": (
            "not reused as the derivation gate; a declared derived proxy is expected "
            "to differ from raw bytes"
        ),
    }


def _replicacad_topology_probe(
    resolution: Any,
    *,
    redactions: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    resource = resolution.resources.get("replicacad_stage_surface")
    if resource is None or resource.status != "pass" or resource.path is None:
        return {
            "status": "not_run",
            "reason": "hash-verified ReplicaCAD stage surface is unavailable",
        }
    try:
        scene = extract_triangle_scene(resource.path)
        report = geometry_report(
            scene.vertices,
            scene.triangles,
            source_sha256=scene.source_sha256,
            representation="raw_visual_surface_topology_probe",
            source_to_canonical={
                "matrix_row_major": [
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1,
                ],
                "reviewed": False,
                "source": "topology-only probe; acoustic coordinate transform not qualified",
            },
            objects=scene.objects,
        )
    except (OSError, ValueError) as error:
        return {
            "status": "fail",
            "reason": _portable_text(str(error), redactions),
        }
    return _topology_summary(report)


def _m5_evidence_probe(
    raw_path: str | Path | None,
    *,
    repository_root: Path,
    redactions: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    if raw_path is None:
        return {"status": "not_run", "reason": "no M5 evidence was supplied"}
    policy = WorkspacePathPolicy.from_roots([repository_root])
    try:
        path = policy.resolve_input(raw_path, owner="M5 evidence", kind="file")
        status, checks = verify_m5_canary_evidence(path)
    except (OSError, ValueError, PathPolicyError) as error:
        return {
            "status": "fail",
            "reason": _portable_text(str(error), redactions),
        }
    return {
        "status": status,
        "evidence": {
            **_repository_locator(path, repository_root),
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "check_count": len(checks),
        "failed_check_ids": [
            check["check_id"] for check in checks if check["status"] != "pass"
        ],
        "interpretation": (
            "current semantic re-verification of retained M5 evidence; Habitat and "
            "RLR were not re-executed by this room attempt"
        ),
    }


def _unverified_artifact_probe(
    raw_path: str | Path | None,
    *,
    artifact_id: str,
    repository_root: Path,
    redactions: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    if raw_path is None:
        return {"status": "not_run", "reason": f"no {artifact_id} was supplied"}
    policy = WorkspacePathPolicy.from_roots([repository_root])
    try:
        path = policy.resolve_input(raw_path, owner=artifact_id, kind="file")
    except PathPolicyError as error:
        return {
            "status": "blocked",
            "reason": _portable_text(str(error), redactions),
        }
    return {
        "status": "observed_unverified",
        "artifact": {
            **_repository_locator(path, repository_root),
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "interpretation": "hash observation only; no semantic pass is inferred",
    }


def _check(
    status: str,
    summary: str,
    blocker_code: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "status": status,
        "summary": summary,
        "evidence_refs": ["current_observation"],
    }
    if blocker_code is not None:
        value["blocker_code"] = blocker_code
    return value


def _placement_not_run(code: str, summary: str) -> dict[str, Any]:
    return {
        **_check("not_run", summary, code),
        "checks": [],
        "failure_reasons": [code],
    }


def _base_observation(
    *,
    case_id: str,
    record: Mapping[str, Any],
    registry_sha256: str,
    provider_observation: Mapping[str, Any],
    git_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": OBSERVATION_SCHEMA,
        "case_id": case_id,
        "room_key": room_revision_key(record),
        "registry_sha256": registry_sha256,
        "execution_mode": "read_only_qualification_attempt",
        "native_execution": {
            "habitat_sim": "not_run",
            "rlr_audio_propagation": "not_run",
            "blender": "not_run",
            "media_render": "not_run",
        },
        "provider_resolution": deepcopy(dict(provider_observation)),
        "code_provenance": deepcopy(dict(git_state)),
        "notes": (
            "Passes apply only to the named read-only hash/contract/topology check; "
            "they do not imply a native runtime pass."
        ),
    }


def _subject(record: Mapping[str, Any], representation_id: str, kind: str) -> dict[str, Any]:
    return {
        "room_id": record["room_id"],
        "revision": record["revision"],
        "qualification_scope": "acoustic_representation",
        "acoustic_representation_id": representation_id,
        "acoustic_representation_kind": kind,
    }


def _write_observation(staging: Path, case_id: str, value: dict[str, Any]) -> dict[str, Any]:
    core = deepcopy(value)
    core.pop("content_sha256", None)
    value["content_sha256"] = canonical_json_sha256(core)
    path = staging / "observations" / f"{case_id}.json"
    write_json(path, value)
    return {
        "artifact_id": "current_observation",
        "path": path.relative_to(staging).as_posix(),
        "sha256": sha256_file(path),
    }


def _build_report(
    *,
    report_id: str,
    subject: Mapping[str, Any],
    observation_artifact: Mapping[str, Any],
    dimensions: Mapping[str, Any],
    placement: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    source_records: Sequence[str],
    notes: str,
) -> dict[str, Any]:
    return build_qualification_report(
        report_id=report_id,
        subject=subject,
        evidence_basis="current_execution",
        evidence_artifacts=[observation_artifact],
        dimensions=dimensions,
        placement_feasibility=placement,
        acoustic_diagnostics=diagnostics,
        provenance={"source_records": list(source_records), "notes": notes},
        promote_if_eligible=False,
    )


def _write_report(staging: Path, case_id: str, report: Mapping[str, Any]) -> dict[str, Any]:
    path = staging / "reports" / f"{case_id}.json"
    write_json(path, report)
    return {
        "case_id": case_id,
        "path": path.relative_to(staging).as_posix(),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
        "dataset_admission": report["dataset_admission"],
        "admission_blockers": list(report["admission_blockers"]),
    }


def run_room_qualification_attempt(
    *,
    registry_path: str | Path,
    corrupted_fixture_path: str | Path,
    output_directory: str | Path,
    repository_root: str | Path,
    environment: Mapping[str, str] | None = None,
    custom_package_manifest: str | Path | None = None,
    custom_m5_evidence: str | Path | None = None,
    legacy_package_manifest: str | Path | None = None,
    legacy_delivery_evidence: str | Path | None = None,
    mp3d_raw_package_manifest: str | Path | None = None,
    mp3d_derived_package_manifest: str | Path | None = None,
    attempt_id: str = "m6_representative_rooms_current_attempt_v1",
) -> Path:
    """Run read-only probes and publish an immutable six-report evidence bundle."""

    root = Path(repository_root).resolve(strict=True)
    env = dict(os.environ if environment is None else environment)
    redactions = _redactions(root, env)
    input_policy = WorkspacePathPolicy.from_roots([root])
    registry_source = input_policy.resolve_input(
        registry_path, owner="M6 room registry", kind="file"
    )
    fixture_source = input_policy.resolve_input(
        corrupted_fixture_path, owner="M6 corrupted fixture", kind="file"
    )
    registry = load_room_registry(registry_source)
    registry_sha = sha256_file(registry_source)
    git_state = _git_state(root)

    destination = Path(output_directory).expanduser()
    if not destination.is_absolute():
        destination = root / destination
    destination_parent = destination.parent.resolve(strict=True)
    output_policy = WorkspacePathPolicy.from_roots([destination_parent])
    output_policy.resolve_output(destination, owner="M6 room attempt")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination_parent)
    )
    report_records: list[dict[str, Any]] = []
    try:
        # Controlled procedural room.
        custom_record, custom_resolution, custom_provider = _provider_observation(
            registry,
            "blender_custom_two_zone_v1",
            repository_root=root,
            environment=env,
            redactions=redactions,
        )
        custom_package_observation, custom_package = _package_probe(
            custom_package_manifest,
            candidate_id="custom_controlled_acoustic_package",
            repository_root=root,
            redactions=redactions,
        )
        custom_m5 = _m5_evidence_probe(
            custom_m5_evidence, repository_root=root, redactions=redactions
        )
        custom_observation = _base_observation(
            case_id="blender_custom_two_zone",
            record=custom_record,
            registry_sha256=registry_sha,
            provider_observation=custom_provider,
            git_state=git_state,
        )
        custom_observation["candidate_package"] = custom_package_observation
        custom_observation["retained_m5_evidence_verification"] = custom_m5
        custom_artifact = _write_observation(
            staging, "blender_custom_two_zone", custom_observation
        )
        custom_geometry = (
            custom_package.qa_reports["geometry_report"]
            if custom_package is not None
            else None
        )
        custom_material = (
            custom_package.qa_reports["material_coverage"]
            if custom_package is not None
            else None
        )
        custom_rays = (
            custom_package.qa_reports["ray_leakage"]
            if custom_package is not None
            else None
        )
        custom_parity = (
            custom_package.qa_reports["compiler_source_to_package_parity"]
            if custom_package is not None
            else None
        )
        custom_identity_pass = (
            custom_package is not None
            and custom_resolution.resources["custom_visual_surface"].status == "pass"
            and custom_geometry["source_geometry_sha256"]
            == custom_resolution.resources["custom_visual_surface"].sha256
        )
        custom_report = _build_report(
            report_id=f"{attempt_id}_blender_custom_two_zone",
            subject=_subject(
                custom_record, "custom_two_zone_acoustic_v1", "production_authority"
            ),
            observation_artifact=custom_artifact,
            dimensions={
                "visual_runtime_status": _check(
                    "not_run",
                    (
                        "retained M5 visual evidence passed an integrity/semantic check, "
                        "but Habitat was not rerun, so current runtime status remains not_run"
                        if custom_m5.get("status") == "pass"
                        else "no current Habitat visual runtime execution was performed"
                    ),
                    "custom_native_visual_not_run",
                ),
                "navigation_status": _check(
                    "not_run",
                    "navmesh bytes resolve, but no native pathfinder probe ran in this attempt",
                    "custom_native_navigation_not_run",
                ),
                "acoustic_geometry_status": _check(
                    "pass"
                    if custom_geometry is not None and custom_geometry["status"] == "pass"
                    else "not_run",
                    "strict-valid controlled surface package passed topology"
                    if custom_geometry is not None and custom_geometry["status"] == "pass"
                    else "controlled acoustic package was not supplied and validated",
                    None
                    if custom_geometry is not None and custom_geometry["status"] == "pass"
                    else "custom_acoustic_package_not_verified",
                ),
                "material_binding_status": _check(
                    "pass"
                    if custom_material is not None and custom_material["status"] == "pass"
                    else "not_run",
                    "controlled material coverage is complete"
                    if custom_material is not None and custom_material["status"] == "pass"
                    else "controlled material coverage was not verified",
                    None
                    if custom_material is not None and custom_material["status"] == "pass"
                    else "custom_material_binding_not_verified",
                ),
                "ray_leakage_status": _check(
                    "not_run",
                    (
                        "compiler CPU opening checks were observed, but the required current "
                        "native RLR ray readback was not run"
                    ),
                    "custom_native_ray_readback_not_run",
                ),
                "physical_material_truth_status": _check(
                    "controlled_profile",
                    "synthetic controlled profile; no measured real-room truth is claimed",
                ),
                "episode_feasibility_status": _check(
                    "not_run",
                    (
                        "retained controlled-room M5 episode passed an integrity/semantic "
                        "check, but no current episode execution or placement probe ran"
                        if custom_m5.get("status") == "pass"
                        else "no current controlled episode execution was performed"
                    ),
                    "custom_current_episode_not_run",
                ),
            },
            placement=_placement_not_run(
                "custom_five_point_placement_not_run",
                "M5 fixed positions are retained, but five support, clearance, and frustum rays were not measured",
            ),
            diagnostics={
                "raw_source_identity": _check(
                    "pass" if custom_identity_pass else "not_run",
                    "strict package source hash is bound to the checked-in GLB"
                    if custom_identity_pass
                    else "custom package source identity was not rebound",
                    None if custom_identity_pass else "custom_identity_not_run",
                ),
                "declared_derivation_integrity": _check(
                    "pass" if custom_package is not None else "not_run",
                    "strict compiler package dependencies and hashes revalidated"
                    if custom_package is not None
                    else "custom compilation integrity was not revalidated",
                    None if custom_package is not None else "custom_compile_integrity_not_run",
                ),
                "visual_to_acoustic_spatial_parity": _check(
                    "pass"
                    if custom_parity is not None and custom_parity["status"] == "pass"
                    else "not_run",
                    "compiler source-to-package parity passed"
                    if custom_parity is not None and custom_parity["status"] == "pass"
                    else "custom spatial parity was not verified",
                    None
                    if custom_parity is not None and custom_parity["status"] == "pass"
                    else "custom_spatial_parity_not_run",
                ),
                "solver_loadability": _check(
                    "not_run",
                    "retained RIR evidence may verify, but RLR package upload was not rerun here",
                    "custom_current_solver_load_not_run",
                ),
                "topology_diagnostics": _check(
                    "pass"
                    if custom_geometry is not None and custom_geometry["status"] == "pass"
                    else "not_run",
                    "controlled surface topology passed"
                    if custom_geometry is not None and custom_geometry["status"] == "pass"
                    else "controlled topology was not verified",
                    None
                    if custom_geometry is not None and custom_geometry["status"] == "pass"
                    else "custom_topology_not_run",
                ),
                "opening_and_enclosure_checks": _check(
                    "pass"
                    if custom_rays is not None and custom_rays["status"] == "pass"
                    else "not_run",
                    "declared compiler CPU opening/enclosure checks passed"
                    if custom_rays is not None and custom_rays["status"] == "pass"
                    else "opening and enclosure checks were not verified",
                    None
                    if custom_rays is not None and custom_rays["status"] == "pass"
                    else "custom_opening_checks_not_run",
                ),
            },
            source_records=[
                "examples/m6/rooms/room_registry.json",
                "current_observation",
            ],
            notes="Read-only current attempt. Native Habitat/RLR/Blender was not re-executed.",
        )
        report_records.append(
            _write_report(staging, "blender_custom_two_zone", custom_report)
        )

        # ReplicaCAD apt_0.
        replica_record, replica_resolution, replica_provider = _provider_observation(
            registry,
            "replicacad_apt_0",
            repository_root=root,
            environment=env,
            redactions=redactions,
        )
        replica_topology = _replicacad_topology_probe(
            replica_resolution, redactions=redactions
        )
        replica_observation = _base_observation(
            case_id="replicacad_apt_0",
            record=replica_record,
            registry_sha256=registry_sha,
            provider_observation=replica_provider,
            git_state=git_state,
        )
        replica_observation["raw_stage_topology_probe"] = replica_topology
        replica_artifact = _write_observation(
            staging, "replicacad_apt_0", replica_observation
        )
        replica_report = _build_report(
            report_id=f"{attempt_id}_replicacad_apt_0",
            subject=_subject(
                replica_record,
                "replicacad_apt_0_acoustic_proxy_v1",
                "derived_proxy",
            ),
            observation_artifact=replica_artifact,
            dimensions={
                "visual_runtime_status": _check(
                    "not_run",
                    "dataset bytes resolved, but Habitat RGB/PBR loading was not executed",
                    "replicacad_native_visual_not_run",
                ),
                "navigation_status": _check(
                    "not_run",
                    "navmesh bytes resolved, but native pathfinder/route checks were not executed",
                    "replicacad_native_navigation_not_run",
                ),
                "acoustic_geometry_status": _check(
                    "not_run",
                    "the declared apt_0 derived acoustic proxy has not been built",
                    "replicacad_proxy_not_built",
                ),
                "material_binding_status": _check(
                    "not_run",
                    "no controlled or semantic material package/readback exists",
                    "replicacad_material_binding_not_run",
                ),
                "ray_leakage_status": _check(
                    "not_run",
                    "scene-specific opening and enclosure rays have not run",
                    "replicacad_rays_not_run",
                ),
                "physical_material_truth_status": _check(
                    "unqualified",
                    "the target controlled/semantic approximation has not been realized",
                    "replicacad_material_truth_unqualified",
                ),
                "episode_feasibility_status": _check(
                    "not_run",
                    "listener/source/actor placement and source program have not run",
                    "replicacad_episode_not_run",
                ),
            },
            placement=_placement_not_run(
                "replicacad_placement_not_run",
                "five-point support, clearance, and frustum probes were not executed",
            ),
            diagnostics={
                "raw_source_identity": _check(
                    "pass"
                    if replica_resolution.resources["replicacad_stage_surface"].status
                    == "pass"
                    else "blocked",
                    "ReplicaCAD stage hash matches the registry"
                    if replica_resolution.resources["replicacad_stage_surface"].status
                    == "pass"
                    else "ReplicaCAD stage is unavailable or failed its hash",
                    None
                    if replica_resolution.resources["replicacad_stage_surface"].status
                    == "pass"
                    else "replicacad_raw_source_unavailable",
                ),
                "declared_derivation_integrity": _check(
                    "not_run",
                    "the declared proxy derivation has not executed",
                    "replicacad_derivation_not_run",
                ),
                "visual_to_acoustic_spatial_parity": _check(
                    "not_run",
                    "no derived proxy exists for spatial parity",
                    "replicacad_spatial_parity_not_run",
                ),
                "solver_loadability": _check(
                    "not_run",
                    "no apt_0 package was uploaded to RLR",
                    "replicacad_solver_load_not_run",
                ),
                "topology_diagnostics": _check(
                    "fail" if replica_topology.get("status") == "fail" else "not_run",
                    (
                        "current topology-only raw stage probe found geometry defects"
                        if replica_topology.get("status") == "fail"
                        else "raw stage topology probe did not run"
                    ),
                    "replicacad_raw_topology_failed"
                    if replica_topology.get("status") == "fail"
                    else "replicacad_topology_not_run",
                ),
                "opening_and_enclosure_checks": _check(
                    "not_run",
                    "apt_0 openings and enclosure expectations are not declared",
                    "replicacad_openings_not_run",
                ),
            },
            source_records=[
                "examples/m6/rooms/room_registry.json",
                "current_observation",
            ],
            notes="Current read-only resource/hash/topology attempt; no native runtime claim.",
        )
        report_records.append(
            _write_report(staging, "replicacad_apt_0", replica_report)
        )

        # Legacy Apartment migration surface/package.
        legacy_record, legacy_resolution, legacy_provider = _provider_observation(
            registry,
            "legacy_ue_apartment_0000_v1",
            repository_root=root,
            environment=env,
            redactions=redactions,
        )
        legacy_package_observation, legacy_package = _package_probe(
            legacy_package_manifest,
            candidate_id="legacy_real_surface_acoustic_package",
            repository_root=root,
            redactions=redactions,
        )
        legacy_delivery = _unverified_artifact_probe(
            legacy_delivery_evidence,
            artifact_id="legacy_delivery_evidence",
            repository_root=root,
            redactions=redactions,
        )
        legacy_observation = _base_observation(
            case_id="legacy_ue_apartment",
            record=legacy_record,
            registry_sha256=registry_sha,
            provider_observation=legacy_provider,
            git_state=git_state,
        )
        legacy_observation["candidate_package"] = legacy_package_observation
        legacy_observation["retained_delivery_artifact"] = legacy_delivery
        legacy_observation["aabb_authority"] = {
            "status": "diagnostic_only",
            "representation_id": "legacy_route_aabb_diagnostic",
            "acoustic_authority": False,
        }
        legacy_artifact = _write_observation(
            staging, "legacy_ue_apartment", legacy_observation
        )
        legacy_geometry = (
            legacy_package.qa_reports["geometry_report"]
            if legacy_package is not None
            else None
        )
        legacy_parity = (
            legacy_package.qa_reports["compiler_source_to_package_parity"]
            if legacy_package is not None
            else None
        )
        legacy_identity_pass = (
            legacy_package is not None
            and legacy_resolution.resources["legacy_real_surface"].status == "pass"
            and legacy_geometry["source_geometry_sha256"]
            == legacy_resolution.resources["legacy_real_surface"].sha256
        )
        legacy_report = _build_report(
            report_id=f"{attempt_id}_legacy_ue_apartment",
            subject=_subject(
                legacy_record,
                "legacy_real_surface_acoustic_v1",
                "production_authority",
            ),
            observation_artifact=legacy_artifact,
            dimensions={
                "visual_runtime_status": _check(
                    "not_run",
                    "real surface and visual package resolve, but Habitat was not rerun",
                    "legacy_native_visual_not_run",
                ),
                "navigation_status": _check(
                    "not_run",
                    "navmesh resolves, but native pathfinder was not rerun",
                    "legacy_native_navigation_not_run",
                ),
                "acoustic_geometry_status": _check(
                    "fail" if legacy_geometry is not None else "not_run",
                    "current strict package validation succeeded but topology remains failed"
                    if legacy_geometry is not None
                    else "legacy real-surface acoustic package was not supplied",
                    "legacy_topology_failed"
                    if legacy_geometry is not None
                    else "legacy_acoustic_package_not_run",
                ),
                "material_binding_status": _check(
                    "blocked",
                    "coverage exists only for unreviewed neutral visual-slot placeholders",
                    "legacy_material_semantics_placeholder",
                ),
                "ray_leakage_status": _check(
                    "not_run",
                    "no real-surface opening/enclosure ray suite was declared or run",
                    "legacy_rays_not_run",
                ),
                "physical_material_truth_status": _check(
                    "unqualified",
                    "legacy material slots have no controlled or measured truth",
                    "legacy_material_truth_unqualified",
                ),
                "episode_feasibility_status": _check(
                    "not_run",
                    "retained delivery was hash-observed only; no semantic verifier/native rerun",
                    "legacy_episode_current_verification_not_run",
                ),
            },
            placement=_placement_not_run(
                "legacy_placement_not_run",
                "five-point support, clearance, and frustum probes were not executed",
            ),
            diagnostics={
                "raw_source_identity": _check(
                    "pass" if legacy_identity_pass else "blocked",
                    "strict package source hash matches the declared real surface"
                    if legacy_identity_pass
                    else "legacy package/source identity was not rebound",
                    None if legacy_identity_pass else "legacy_raw_identity_blocked",
                ),
                "declared_derivation_integrity": _check(
                    "pass" if legacy_package is not None else "not_run",
                    "strict package dependencies and hashes revalidated"
                    if legacy_package is not None
                    else "legacy compilation integrity was not revalidated",
                    None
                    if legacy_package is not None
                    else "legacy_compile_integrity_not_run",
                ),
                "visual_to_acoustic_spatial_parity": _check(
                    "fail"
                    if legacy_parity is not None and legacy_parity["status"] == "fail"
                    else "not_run",
                    "declared surface audit differs by one triangle; visual parity claim remains false"
                    if legacy_parity is not None and legacy_parity["status"] == "fail"
                    else "legacy spatial parity was not checked",
                    "legacy_declared_surface_parity_failed"
                    if legacy_parity is not None and legacy_parity["status"] == "fail"
                    else "legacy_spatial_parity_not_run",
                ),
                "solver_loadability": _check(
                    "not_run",
                    "RLR package upload was not rerun in this attempt",
                    "legacy_current_solver_load_not_run",
                ),
                "topology_diagnostics": _check(
                    "fail" if legacy_geometry is not None else "not_run",
                    "current package topology report remains failed"
                    if legacy_geometry is not None
                    else "legacy topology report was not supplied",
                    "legacy_topology_failed"
                    if legacy_geometry is not None
                    else "legacy_topology_not_run",
                ),
                "opening_and_enclosure_checks": _check(
                    "not_run",
                    "real-surface opening/enclosure expectations have not run",
                    "legacy_openings_not_run",
                ),
            },
            source_records=[
                "examples/m6/rooms/room_registry.json",
                "current_observation",
            ],
            notes=(
                "The route AABBs remain center-point diagnostics only and are not an "
                "acoustic authority."
            ),
        )
        report_records.append(
            _write_report(staging, "legacy_ue_apartment", legacy_report)
        )

        # MP3D immutable raw source and separately declared cleanup-derived proxy.
        mp3d_record, mp3d_resolution, mp3d_provider = _provider_observation(
            registry,
            "habitat_mp3d_example_17DRP5sb8fy",
            repository_root=root,
            environment=env,
            redactions=redactions,
        )
        raw_observed, raw_package = _package_probe(
            mp3d_raw_package_manifest,
            candidate_id="mp3d_raw_compiled_diagnostic_package",
            repository_root=root,
            redactions=redactions,
        )
        derived_observed, derived_package = _package_probe(
            mp3d_derived_package_manifest,
            candidate_id="mp3d_declared_cleanup_proxy_v2_candidate",
            repository_root=root,
            redactions=redactions,
        )
        derivation = _declared_derivation_assessment(raw_package, derived_package)

        raw_observation = _base_observation(
            case_id="mp3d_17DRP5sb8fy_raw",
            record=mp3d_record,
            registry_sha256=registry_sha,
            provider_observation=mp3d_provider,
            git_state=git_state,
        )
        raw_observation["candidate_package"] = raw_observed
        raw_artifact = _write_observation(
            staging, "mp3d_17DRP5sb8fy_raw", raw_observation
        )
        raw_geometry = (
            raw_package.qa_reports["geometry_report"]
            if raw_package is not None
            else None
        )
        raw_parity = (
            raw_package.qa_reports["compiler_source_to_package_parity"]
            if raw_package is not None
            else None
        )
        raw_identity_pass = (
            raw_package is not None
            and mp3d_resolution.resources["mp3d_raw_visual_surface"].status == "pass"
            and raw_geometry["source_geometry_sha256"]
            == mp3d_resolution.resources["mp3d_raw_visual_surface"].sha256
        )
        raw_report = _build_report(
            report_id=f"{attempt_id}_mp3d_raw",
            subject=_subject(
                mp3d_record,
                "mp3d_17DRP5sb8fy_raw_source_v1",
                "raw_source",
            ),
            observation_artifact=raw_artifact,
            dimensions={
                "visual_runtime_status": _check(
                    "not_run",
                    "raw visual assets resolve, but Habitat visual loading was not rerun",
                    "mp3d_raw_native_visual_not_run",
                ),
                "navigation_status": _check(
                    "not_run",
                    "navmesh resolves, but native pathfinder was not rerun",
                    "mp3d_raw_native_navigation_not_run",
                ),
                "acoustic_geometry_status": _check(
                    "fail" if raw_geometry is not None else "not_run",
                    "raw topology contains zero-area and other defects"
                    if raw_geometry is not None
                    else "raw diagnostic package was not supplied",
                    "mp3d_raw_topology_failed"
                    if raw_geometry is not None
                    else "mp3d_raw_package_not_run",
                ),
                "material_binding_status": _check(
                    "blocked",
                    "raw package uses an unreviewed research-placeholder mapping",
                    "mp3d_raw_material_placeholder",
                ),
                "ray_leakage_status": _check(
                    "not_run",
                    "raw package has no declared ray suite",
                    "mp3d_raw_rays_not_run",
                ),
                "physical_material_truth_status": _check(
                    "unqualified",
                    "raw visual material slots are not physical acoustic truth",
                    "mp3d_raw_material_truth_unqualified",
                ),
                "episode_feasibility_status": _check(
                    "not_run",
                    "no current native MP3D source episode ran",
                    "mp3d_raw_episode_not_run",
                ),
            },
            placement=_placement_not_run(
                "mp3d_raw_placement_not_run",
                "five-point support, clearance, and frustum probes were not executed",
            ),
            diagnostics={
                "raw_source_identity": _check(
                    "pass" if raw_identity_pass else "blocked",
                    "immutable raw GLB hash matches registry and compiled source identity"
                    if raw_identity_pass
                    else "raw source identity could not be fully rebound",
                    None if raw_identity_pass else "mp3d_raw_identity_blocked",
                ),
                "declared_derivation_integrity": _check(
                    "not_run",
                    "not applicable to the immutable raw representation",
                    "mp3d_raw_derivation_not_applicable",
                ),
                "visual_to_acoustic_spatial_parity": _check(
                    "pass"
                    if raw_parity is not None and raw_parity["status"] == "pass"
                    else "not_run",
                    "raw compiler source-to-package parity passed"
                    if raw_parity is not None and raw_parity["status"] == "pass"
                    else "raw compiler parity was not verified",
                    None
                    if raw_parity is not None and raw_parity["status"] == "pass"
                    else "mp3d_raw_spatial_parity_not_run",
                ),
                "solver_loadability": _check(
                    "not_run",
                    "RLR upload was deliberately not executed by this read-only attempt",
                    "mp3d_raw_current_solver_load_not_run",
                ),
                "topology_diagnostics": _check(
                    "fail" if raw_geometry is not None else "not_run",
                    "current strict package topology report remains failed"
                    if raw_geometry is not None
                    else "raw topology package was not supplied",
                    "mp3d_raw_topology_failed"
                    if raw_geometry is not None
                    else "mp3d_raw_topology_not_run",
                ),
                "opening_and_enclosure_checks": _check(
                    "not_run",
                    "MP3D opening/enclosure expectations were not authored and run",
                    "mp3d_raw_openings_not_run",
                ),
            },
            source_records=[
                "examples/m6/rooms/room_registry.json",
                "current_observation",
            ],
            notes="Raw scan identity remains immutable; the report does not modify MP3D.",
        )
        report_records.append(
            _write_report(staging, "mp3d_17DRP5sb8fy_raw", raw_report)
        )

        derived_observation = _base_observation(
            case_id="mp3d_17DRP5sb8fy_derived",
            record=mp3d_record,
            registry_sha256=registry_sha,
            provider_observation=mp3d_provider,
            git_state=git_state,
        )
        derived_observation["raw_candidate_package"] = raw_observed
        derived_observation["derived_candidate_package"] = derived_observed
        derived_observation["declared_derivation_assessment"] = derivation
        proxy_representation_status = next(
            (
                item["status"]
                for item in mp3d_provider["acoustic_representations"]
                if item["representation_id"]
                == "mp3d_17DRP5sb8fy_acoustic_proxy_v2"
            ),
            "not_run",
        )
        proxy_output = mp3d_resolution.resources["mp3d_declared_proxy_v2"]
        proxy_descriptor = mp3d_resolution.resources["mp3d_proxy_v2_descriptor"]
        proxy_binding = _proxy_descriptor_assessment(
            proxy_descriptor.path,
            mp3d_derived_package_manifest,
            mp3d_raw_package_manifest,
            descriptor_resolution_status=proxy_descriptor.status,
            provider_manifest_path=proxy_output.path,
            provider_manifest_status=proxy_output.status,
            provider_representation_status=proxy_representation_status,
            room_record=mp3d_record,
        )
        binding_components = {
            "descriptor_and_closure": proxy_binding["status"],
            "raw_candidate_package": raw_observed["status"],
            "derived_candidate_package": derived_observed["status"],
            "provider_output_manifest": proxy_output.status,
            "provider_representation": proxy_representation_status,
        }
        binding_status = _aggregate_attempt_status(*binding_components.values())
        proxy_binding = {
            **proxy_binding,
            "status": binding_status,
            "component_statuses": binding_components,
        }
        derived_observation["materialized_proxy_binding"] = proxy_binding
        derived_artifact = _write_observation(
            staging, "mp3d_17DRP5sb8fy_derived", derived_observation
        )
        derived_geometry = (
            derived_package.qa_reports["geometry_report"]
            if derived_package is not None
            else None
        )
        materialized_proxy_bound = (
            derived_package is not None and binding_status == "pass"
        )
        bound_geometry = derived_geometry if materialized_proxy_bound else None
        topology_status = (
            bound_geometry.get("status", "fail")
            if bound_geometry is not None
            else binding_status
        )
        derivation_status = _aggregate_attempt_status(
            binding_status, derivation["status"]
        )
        derived_report = _build_report(
            report_id=f"{attempt_id}_mp3d_derived",
            subject=_subject(
                mp3d_record,
                "mp3d_17DRP5sb8fy_acoustic_proxy_v2",
                "derived_proxy",
            ),
            observation_artifact=derived_artifact,
            dimensions={
                "visual_runtime_status": _check(
                    "not_run",
                    "raw visual assets resolve, but Habitat visual loading was not rerun",
                    "mp3d_derived_native_visual_not_run",
                ),
                "navigation_status": _check(
                    "not_run",
                    "navmesh resolves, but native pathfinder was not rerun",
                    "mp3d_derived_native_navigation_not_run",
                ),
                "acoustic_geometry_status": _check(
                    topology_status,
                    (
                        "materialized proxy is bound and its topology gates pass"
                        if topology_status == "pass"
                        else (
                            "materialized proxy removes zero-area faces, but duplicate/"
                            "boundary/nonmanifold topology gates still fail"
                            if materialized_proxy_bound
                            else "topology was not assessed because the versioned proxy "
                            f"binding is {binding_status}"
                        )
                    ),
                    None
                    if topology_status == "pass"
                    else (
                        "mp3d_derived_topology_failed"
                        if materialized_proxy_bound
                        else f"mp3d_derived_proxy_binding_{binding_status}"
                    ),
                ),
                "material_binding_status": _check(
                    "blocked",
                    (
                        "triangle coverage passes, but materials remain an unreviewed "
                        "research-placeholder rather than semantic mapping"
                    ),
                    "mp3d_derived_material_semantics_placeholder",
                ),
                "ray_leakage_status": _check(
                    "not_run",
                    "derived candidate has no declared opening/enclosure ray suite",
                    "mp3d_derived_rays_not_run",
                ),
                "physical_material_truth_status": _check(
                    "unqualified",
                    "semantic_mapping_approximation has not been implemented/reviewed",
                    "mp3d_derived_material_truth_unqualified",
                ),
                "episode_feasibility_status": _check(
                    "not_run",
                    "no current native episode/placement run used the candidate proxy",
                    "mp3d_derived_episode_not_run",
                ),
            },
            placement=_placement_not_run(
                "mp3d_derived_placement_not_run",
                "five-point support, clearance, and frustum probes were not executed",
            ),
            diagnostics={
                "raw_source_identity": _check(
                    "pass" if raw_identity_pass else "blocked",
                    "derived candidate remains bound to the immutable raw GLB identity"
                    if raw_identity_pass
                    else "raw source identity could not be rebound",
                    None if raw_identity_pass else "mp3d_derived_raw_identity_blocked",
                ),
                "declared_derivation_integrity": _check(
                    derivation_status,
                    (
                        "committed proxy descriptor, immutable raw source, complete "
                        "materialized package closure, and declared cleanup all match"
                        if derivation_status == "pass"
                        else "; ".join(proxy_binding.get("errors", []))
                        or derivation["reason"]
                    ),
                    None
                    if derivation_status == "pass"
                    else "mp3d_materialized_proxy_binding_not_verified",
                ),
                "visual_to_acoustic_spatial_parity": _check(
                    "pass" if derivation_status == "pass" else "not_run",
                    (
                        "source transform/bounds identity is preserved by the declared "
                        "zero-area cleanup; byte equality is not required"
                        if derivation_status == "pass"
                        else "derived spatial parity could not be assessed"
                    ),
                    None
                    if derivation_status == "pass"
                    else "mp3d_derived_spatial_parity_not_run",
                ),
                "solver_loadability": _check(
                    "not_run",
                    "RLR upload was deliberately not executed by this read-only attempt",
                    "mp3d_derived_current_solver_load_not_run",
                ),
                "topology_diagnostics": _check(
                    topology_status,
                    (
                        "bound derived proxy topology passed"
                        if topology_status == "pass"
                        else (
                            "zero-area faces are gone, but other topology gates remain failed"
                            if materialized_proxy_bound
                            else "topology was not inspected because the materialized proxy "
                            f"binding is {binding_status}"
                        )
                    ),
                    None
                    if topology_status == "pass"
                    else (
                        "mp3d_derived_topology_failed"
                        if materialized_proxy_bound
                        else f"mp3d_derived_proxy_binding_{binding_status}"
                    ),
                ),
                "opening_and_enclosure_checks": _check(
                    "not_run",
                    "scene-specific MP3D opening/enclosure expectations have not run",
                    "mp3d_derived_openings_not_run",
                ),
            },
            source_records=[
                "examples/m6/rooms/room_registry.json",
                "examples/m6/rooms/proxies/mp3d_17DRP5sb8fy_acoustic_proxy_v2.json",
                "current_observation",
            ],
            notes=(
                "The materialized generated-local package is authenticated by the "
                "committed proxy descriptor and assessed separately from raw byte parity. "
                "No qualified revision is created while topology/material/ray/native "
                "gates fail or remain not_run."
            ),
        )
        report_records.append(
            _write_report(staging, "mp3d_17DRP5sb8fy_derived", derived_report)
        )

        # Independent, deliberately corrupted fail-closed fixture.
        fixture = load_json(fixture_source)
        fixture_result = qualify_corrupted_acoustic_fixture(fixture)
        fixture_observation = {
            "schema": OBSERVATION_SCHEMA,
            "case_id": "independent_corrupted_fixture",
            "room_key": None,
            "registry_sha256": registry_sha,
            "execution_mode": "contract_fixture_evaluation",
            "native_execution": {
                "habitat_sim": "not_run",
                "rlr_audio_propagation": "not_run",
                "blender": "not_run",
                "media_render": "not_run",
            },
            "fixture": {
                **_repository_locator(fixture_source, root),
                "byte_size": fixture_source.stat().st_size,
                "sha256": sha256_file(fixture_source),
            },
            "findings": list(fixture_result.findings),
            "code_provenance": deepcopy(git_state),
            "notes": "Independent negative fixture; never a real room or repair target.",
        }
        fixture_artifact = _write_observation(
            staging, "independent_corrupted_fixture", fixture_observation
        )
        fixture_report = deepcopy(fixture_result.report)
        fixture_report["evidence_artifacts"] = [fixture_artifact]
        fixture_report["provenance"]["source_records"].append(
            "current_observation"
        )
        errors = validate_qualification_report(fixture_report)
        if errors:
            raise RoomAttemptError("invalid corrupted fixture report: " + "; ".join(errors))
        report_records.append(
            _write_report(
                staging, "independent_corrupted_fixture", fixture_report
            )
        )

        artifact_records: list[dict[str, Any]] = []
        for path in sorted(staging.rglob("*.json")):
            if path.name == "attempt_manifest.json":
                continue
            artifact_records.append(file_record(path, relative_to=staging))
        manifest: dict[str, Any] = {
            "schema": ATTEMPT_MANIFEST_SCHEMA,
            "attempt_id": attempt_id,
            "generated_at_utc": _now_utc(),
            "execution_mode": "read_only_qualification_attempt",
            "native_execution": {
                "habitat_sim": "not_run",
                "rlr_audio_propagation": "not_run",
                "blender": "not_run",
                "media_render": "not_run",
            },
            "registry": {
                **_repository_locator(registry_source, root),
                "byte_size": registry_source.stat().st_size,
                "sha256": registry_sha,
            },
            "reports": report_records,
            "artifacts": artifact_records,
            "case_ids": list(ATTEMPT_CASE_IDS),
            "code_provenance": git_state,
            "claims": {
                "current_native_runtime_pass": False,
                "dataset_admission_count": sum(
                    1 for report in report_records if report["dataset_admission"]
                ),
                "historical_artifact_statuses_promoted_to_current_native_pass": False,
                "mp3d_raw_modified": False,
            },
        }
        core = deepcopy(manifest)
        manifest["content_sha256"] = canonical_json_sha256(core)
        write_json(staging / "attempt_manifest.json", manifest)

        published = atomic_publish_directory(output_policy, staging, destination)
        staging = Path()
        return published / "attempt_manifest.json"
    except Exception:
        if staging and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _first_symlink_component(path: Path) -> Path | None:
    """Return the first existing symlink in one lexical absolute path.

    Calling ``resolve()`` before this check would erase the evidence that an
    intermediate directory was a symlink.  The trusted-workspace policy does
    not claim race-free resolution, but a retained bundle must still reject
    symlinks that are observable during verification.
    """

    absolute = path.expanduser()
    if not absolute.is_absolute():
        absolute = Path.cwd() / absolute
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            return current
    return None


def _resolve_bundle_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("artifact path is not portable")
    portable = PurePosixPath(relative)
    if portable.is_absolute() or any(
        part in {"", ".", ".."} for part in portable.parts
    ):
        raise ValueError(f"artifact path is not confined: {relative}")
    lexical = root.joinpath(*portable.parts)
    symlink = _first_symlink_component(lexical)
    if symlink is not None:
        raise ValueError(f"bundle path contains symlink component: {relative}")
    candidate = lexical.resolve(strict=True)
    candidate.relative_to(root)
    if not candidate.is_file():
        raise ValueError(f"bundle artifact is not a regular file: {relative}")
    return candidate


def _discover_git_root(bundle_root: Path) -> Path | None:
    """Locate the checkout that owns a retained attempt bundle.

    Formal bundles live below the repository.  The source-tree fallback keeps
    development/unit bundles written to a temporary directory verifiable
    without recording a private absolute checkout path in evidence.
    """

    candidates = (bundle_root, Path(__file__).resolve().parents[3])
    for candidate in candidates:
        completed = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            return Path(completed.stdout.strip()).resolve(strict=True)
    return None


def _git_provenance_observation(
    manifest: Mapping[str, Any], *, bundle_root: Path
) -> dict[str, Any]:
    provenance = manifest.get("code_provenance")
    commit = provenance.get("commit") if isinstance(provenance, Mapping) else None
    claimed_clean = (
        provenance.get("worktree_clean")
        if isinstance(provenance, Mapping)
        else None
    )
    repository = _discover_git_root(bundle_root)
    observed: dict[str, Any] = {
        "commit": commit,
        "commit_format_valid": isinstance(commit, str)
        and _COMMIT_PATTERN.fullmatch(commit) is not None,
        "commit_exists": False,
        "commit_is_ancestor_of_head": False,
        "claimed_worktree_clean": claimed_clean,
        "current_worktree_clean": None,
        "repository_available": repository is not None,
    }
    if repository is None or not observed["commit_format_valid"]:
        return observed

    exists = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "-e", f"{commit}^{{commit}}"],
        check=False,
        capture_output=True,
    )
    observed["commit_exists"] = exists.returncode == 0
    if observed["commit_exists"]:
        ancestor = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "merge-base",
                "--is-ancestor",
                commit,
                "HEAD",
            ],
            check=False,
            capture_output=True,
        )
        observed["commit_is_ancestor_of_head"] = ancestor.returncode == 0

    status = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ],
        check=False,
        capture_output=True,
    )
    if status.returncode == 0:
        observed["current_worktree_clean"] = not bool(status.stdout.strip())
    observed["repository"] = repository
    return observed


def _formal_registry_git_binding(
    manifest: Mapping[str, Any], *, git_observation: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """Bind a clean/formal attempt to the canonical registry blob in Git."""

    provenance = manifest.get("code_provenance")
    claimed_clean = (
        provenance.get("worktree_clean")
        if isinstance(provenance, Mapping)
        else None
    )
    registry = manifest.get("registry")
    measured: dict[str, Any] = {
        "enforced": claimed_clean is True,
        "canonical_path": CANONICAL_ROOM_REGISTRY_PATH,
        "declared": deepcopy(dict(registry))
        if isinstance(registry, Mapping)
        else registry,
    }
    if claimed_clean is not True:
        measured["reason"] = (
            "dirty development evidence remains verifiable but is not eligible "
            "for formal/release use"
        )
        return True, measured

    repository = git_observation.get("repository")
    commit = git_observation.get("commit")
    if not isinstance(repository, Path) or not isinstance(commit, str):
        measured["reason"] = "Git repository or commit is unavailable"
        return False, measured
    if not isinstance(registry, Mapping):
        measured["reason"] = "registry record is missing"
        return False, measured

    fixed_locator = (
        registry.get("kind") == "repository_relative"
        and registry.get("path") == CANONICAL_ROOM_REGISTRY_PATH
    )
    shown = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "show",
            f"{commit}:{CANONICAL_ROOM_REGISTRY_PATH}",
        ],
        check=False,
        capture_output=True,
    )
    git_blob_available = shown.returncode == 0
    git_blob = shown.stdout if git_blob_available else b""
    git_sha256 = hashlib.sha256(git_blob).hexdigest() if git_blob_available else None
    git_byte_size = len(git_blob) if git_blob_available else None
    measured.update(
        {
            "fixed_repository_locator": fixed_locator,
            "git_blob_available": git_blob_available,
            "git_blob_sha256": git_sha256,
            "git_blob_byte_size": git_byte_size,
        }
    )
    passed = (
        fixed_locator
        and git_blob_available
        and registry.get("sha256") == git_sha256
        and registry.get("byte_size") == git_byte_size
    )
    return passed, measured


def verify_room_qualification_attempt(
    manifest_path: str | Path,
) -> tuple[str, list[dict[str, Any]]]:
    """Rehash and semantically validate one published room-attempt bundle."""

    lexical_path = Path(manifest_path).expanduser()
    if not lexical_path.is_absolute():
        lexical_path = Path.cwd() / lexical_path
    symlink = _first_symlink_component(lexical_path)
    if symlink is not None:
        return "fail", [
            {
                "check_id": "bundle_path_no_symlinks",
                "status": "fail",
                "measured": f"manifest path contains symlink component: {symlink}",
            }
        ]
    try:
        path = lexical_path.resolve(strict=True)
    except OSError as error:
        return "fail", [
            {
                "check_id": "manifest_load",
                "status": "fail",
                "measured": str(error),
            }
        ]
    root = path.parent
    checks: list[dict[str, Any]] = []

    def add(check_id: str, passed: bool, measured: Any) -> None:
        checks.append(
            {
                "check_id": check_id,
                "status": "pass" if passed else "fail",
                "measured": measured,
            }
        )

    try:
        manifest = load_json(path)
    except (OSError, ValueError) as error:
        return "fail", [
            {"check_id": "manifest_load", "status": "fail", "measured": str(error)}
        ]
    declared_content_hash = manifest.get("content_sha256")
    core = deepcopy(manifest)
    core.pop("content_sha256", None)
    add(
        "manifest_content_hash",
        declared_content_hash == canonical_json_sha256(core),
        {
            "declared": declared_content_hash,
            "recomputed": canonical_json_sha256(core),
        },
    )
    add(
        "manifest_schema",
        manifest.get("schema") == ATTEMPT_MANIFEST_SCHEMA,
        manifest.get("schema"),
    )
    add(
        "case_set",
        manifest.get("case_ids") == list(ATTEMPT_CASE_IDS),
        manifest.get("case_ids"),
    )
    add(
        "no_native_pass_claim",
        manifest.get("claims", {}).get("current_native_runtime_pass") is False
        and manifest.get("claims", {}).get(
            "historical_artifact_statuses_promoted_to_current_native_pass"
        )
        is False,
        manifest.get("claims"),
    )

    git_observation = _git_provenance_observation(manifest, bundle_root=root)
    add(
        "code_provenance_commit",
        bool(
            git_observation["commit_format_valid"]
            and git_observation["commit_exists"]
            and git_observation["commit_is_ancestor_of_head"]
        ),
        {
            key: value
            for key, value in git_observation.items()
            if key != "repository"
        },
    )
    claimed_clean = git_observation["claimed_worktree_clean"]
    formal_worktree_pass = isinstance(claimed_clean, bool) and (
        claimed_clean is False
        or git_observation["current_worktree_clean"] is True
    )
    add(
        "formal_worktree_state",
        formal_worktree_pass,
        {
            "formal_release_eligible": claimed_clean is True,
            "claimed_worktree_clean": claimed_clean,
            "current_worktree_clean": git_observation["current_worktree_clean"],
            "interpretation": (
                "clean evidence must still be verified from a currently clean checkout"
                if claimed_clean is True
                else "dirty development evidence is not formal/release eligible"
            ),
        },
    )
    registry_binding_pass, registry_binding = _formal_registry_git_binding(
        manifest, git_observation=git_observation
    )
    add(
        "formal_registry_git_binding",
        registry_binding_pass,
        registry_binding,
    )

    artifact_errors: list[str] = []
    records = manifest.get("artifacts", [])
    declared_paths: set[str] = set()
    for record in records if isinstance(records, list) else []:
        try:
            relative = record["path"]
            if relative in declared_paths:
                raise ValueError(f"duplicate artifact record: {relative}")
            declared_paths.add(relative)
            candidate = _resolve_bundle_file(root, relative)
            if candidate.stat().st_size != record["byte_size"]:
                artifact_errors.append(f"{record['path']}: byte_size mismatch")
            if sha256_file(candidate) != record["sha256"]:
                artifact_errors.append(f"{record['path']}: sha256 mismatch")
        except (KeyError, OSError, TypeError, ValueError) as error:
            artifact_errors.append(str(error))
    actual_paths = set()
    for item in root.rglob("*"):
        if item == path:
            continue
        if item.is_file() or item.is_symlink():
            actual_paths.add(item.relative_to(root).as_posix())
    if declared_paths != actual_paths:
        artifact_errors.append(
            "artifact records are not the exact bundle file closure; "
            f"missing={sorted(declared_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - declared_paths)}"
        )
    add(
        "artifact_hashes",
        isinstance(records, list) and not artifact_errors,
        artifact_errors,
    )

    report_errors: list[str] = []
    report_case_ids: list[str] = []
    derived_binding_observation: Mapping[str, Any] | None = None
    manifest_registry = manifest.get("registry")
    manifest_registry_sha = (
        manifest_registry.get("sha256")
        if isinstance(manifest_registry, Mapping)
        else None
    )
    for record in manifest.get("reports", []):
        try:
            report_case_ids.append(record["case_id"])
            report_path = _resolve_bundle_file(root, record["path"])
            report = load_json(report_path)
            declared_report_artifact = next(
                (
                    item
                    for item in records
                    if item.get("path") == record["path"]
                ),
                None,
            )
            if (
                not isinstance(declared_report_artifact, Mapping)
                or record.get("byte_size") != report_path.stat().st_size
                or record.get("sha256") != sha256_file(report_path)
                or declared_report_artifact != {
                    "path": record["path"],
                    "byte_size": record["byte_size"],
                    "sha256": record["sha256"],
                }
            ):
                report_errors.append(
                    f"{record['case_id']}: report record/artifact binding mismatch"
                )
            report_errors.extend(
                f"{record['case_id']}: {error}"
                for error in validate_qualification_report(report)
            )
            if report["dataset_admission"]:
                report_errors.append(
                    f"{record['case_id']}: read-only attempt unexpectedly admitted room"
                )
            for artifact in report["evidence_artifacts"]:
                expected_observation = (
                    f"observations/{record['case_id']}.json"
                )
                if artifact["path"] != expected_observation:
                    report_errors.append(
                        f"{record['case_id']}: report evidence does not bind its "
                        "case observation"
                    )
                artifact_path = _resolve_bundle_file(root, artifact["path"])
                if sha256_file(artifact_path) != artifact["sha256"]:
                    report_errors.append(
                        f"{record['case_id']}: evidence artifact hash mismatch"
                    )
                observation = load_json(artifact_path)
                if record["case_id"] == "mp3d_17DRP5sb8fy_derived":
                    derived_binding_observation = observation
                observation_core = deepcopy(observation)
                observation_hash = observation_core.pop("content_sha256", None)
                if observation_hash != canonical_json_sha256(observation_core):
                    report_errors.append(
                        f"{record['case_id']}: observation content hash mismatch"
                    )
                if observation.get("code_provenance") != manifest.get(
                    "code_provenance"
                ):
                    report_errors.append(
                        f"{record['case_id']}: observation code provenance differs "
                        "from attempt manifest"
                    )
                if observation.get("registry_sha256") != manifest_registry_sha:
                    report_errors.append(
                        f"{record['case_id']}: observation registry SHA-256 differs "
                        "from attempt manifest"
                    )
        except (KeyError, OSError, TypeError, ValueError) as error:
            report_errors.append(f"malformed report record: {error}")
    add(
        "qualification_reports",
        report_case_ids == list(ATTEMPT_CASE_IDS) and not report_errors,
        {"case_ids": report_case_ids, "errors": report_errors},
    )

    try:
        fixture_report_path = _resolve_bundle_file(
            root, "reports/independent_corrupted_fixture.json"
        )
        fixture_report = load_json(fixture_report_path)
        fixture_fail_closed = (
            fixture_report["dataset_admission"] is False
            and fixture_report["dimensions"]["acoustic_geometry_status"]["status"]
            == "fail"
            and fixture_report["dimensions"]["material_binding_status"]["status"]
            == "fail"
            and fixture_report["dimensions"]["ray_leakage_status"]["status"]
            == "fail"
        )
    except (OSError, ValueError, KeyError):
        fixture_fail_closed = False
    add("corrupted_fixture_fail_closed", fixture_fail_closed, fixture_fail_closed)

    try:
        derived_report_path = _resolve_bundle_file(
            root, "reports/mp3d_17DRP5sb8fy_derived.json"
        )
        derived_report = load_json(derived_report_path)
        separate_derivation = (
            derived_report["acoustic_diagnostics"]["raw_source_identity"]["status"]
            in {"pass", "blocked"}
            and derived_report["acoustic_diagnostics"][
                "declared_derivation_integrity"
            ]["status"]
            in {"pass", "fail", "blocked", "not_run"}
            and derived_report["acoustic_diagnostics"][
                "visual_to_acoustic_spatial_parity"
            ]["status"]
            in {"pass", "fail", "blocked", "not_run"}
            and derived_report["dataset_admission"] is False
        )
    except (OSError, ValueError, KeyError):
        separate_derivation = False
    add("mp3d_split_diagnostics", separate_derivation, separate_derivation)

    binding_errors: list[str] = []
    binding_measured: dict[str, Any] = {
        "formal_release_eligible": claimed_clean is True,
    }
    try:
        if not isinstance(derived_binding_observation, Mapping):
            raise ValueError("derived MP3D observation is unavailable")
        binding = derived_binding_observation["materialized_proxy_binding"]
        components = binding["component_statuses"]
        provider = derived_binding_observation["provider_resolution"]
        derived_candidate = derived_binding_observation["derived_candidate_package"]
        raw_candidate = derived_binding_observation["raw_candidate_package"]
        declared_derivation = derived_binding_observation[
            "declared_derivation_assessment"
        ]
        derived_report_path = _resolve_bundle_file(
            root, "reports/mp3d_17DRP5sb8fy_derived.json"
        )
        derived_report = load_json(derived_report_path)
        provider_representation = next(
            item
            for item in provider["acoustic_representations"]
            if item["representation_id"]
            == "mp3d_17DRP5sb8fy_acoustic_proxy_v2"
        )
        provider_output = next(
            item
            for item in provider["resources"]
            if item["resource_id"] == "mp3d_declared_proxy_v2"
        )
        descriptor_resource = next(
            item
            for item in provider["resources"]
            if item["resource_id"] == "mp3d_proxy_v2_descriptor"
        )
        expected_binding_status = _aggregate_attempt_status(
            *components.values()
        )
        expected_derivation_status = _aggregate_attempt_status(
            binding["status"], declared_derivation["status"]
        )
        expected_topology_status = (
            derived_candidate.get("geometry", {}).get("status", "fail")
            if binding["status"] == "pass"
            else binding["status"]
        )
        binding_consistent = (
            binding["status"] == expected_binding_status
            and derived_report["acoustic_diagnostics"]
            ["declared_derivation_integrity"]["status"]
            == expected_derivation_status
            and derived_report["dimensions"]["acoustic_geometry_status"]["status"]
            == expected_topology_status
            and derived_report["acoustic_diagnostics"]["topology_diagnostics"]
            ["status"]
            == expected_topology_status
        )
        exact_binding_pass = (
            binding["status"] == "pass"
            and not binding.get("errors")
            and binding.get("same_materialized_manifest") is True
            and binding.get("package_manifest_sha256")
            == binding.get("provider_manifest_sha256")
            and raw_candidate.get("status") == "pass"
            and derived_candidate.get("status") == "pass"
            and provider_representation.get("status") == "pass"
            and provider_output.get("status") == "pass"
            and descriptor_resource.get("status") == "pass"
            and declared_derivation.get("status") == "pass"
            and derived_report["acoustic_diagnostics"]
            ["declared_derivation_integrity"]["status"]
            == "pass"
        )
        if not binding_consistent:
            binding_errors.append(
                "MP3D binding, derivation, and topology report statuses are inconsistent"
            )
        if claimed_clean is True and not exact_binding_pass:
            binding_errors.append(
                "formal room evidence requires an exact materialized MP3D proxy binding"
            )
        binding_measured.update(
            {
                "binding_status": binding.get("status"),
                "component_statuses": deepcopy(dict(components)),
                "same_materialized_manifest": binding.get(
                    "same_materialized_manifest"
                ),
                "provider_manifest_sha256": binding.get(
                    "provider_manifest_sha256"
                ),
                "candidate_manifest_sha256": binding.get(
                    "package_manifest_sha256"
                ),
                "binding_consistent": binding_consistent,
                "exact_binding_pass": exact_binding_pass,
                "errors": binding_errors,
            }
        )
    except (KeyError, OSError, StopIteration, TypeError, ValueError) as error:
        binding_errors.append(str(error))
        binding_measured["errors"] = binding_errors
    add(
        "mp3d_materialized_proxy_binding",
        not binding_errors,
        binding_measured,
    )

    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return status, checks


__all__ = [
    "ATTEMPT_CASE_IDS",
    "ATTEMPT_MANIFEST_SCHEMA",
    "OBSERVATION_SCHEMA",
    "RoomAttemptError",
    "run_room_qualification_attempt",
    "verify_room_qualification_attempt",
]
