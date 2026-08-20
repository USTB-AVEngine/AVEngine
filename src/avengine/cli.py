from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import shutil
import sys
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Sequence

from avengine.appearance import (
    AppearanceContractError,
    generate_l9_batch,
    write_l9_batch_exclusive,
)
from avengine.acoustic_profiles import (
    AcousticProfileError,
    load_acoustic_profile_registry,
    load_default_acoustic_profile_registry,
)
from avengine.contracts.json_io import file_record, load_json, sha256_file, write_json
from avengine.m1.contracts import (
    ContractError,
    EVIDENCE_SCHEMA_V2,
    ValidatedM1Inputs,
    aggregate_status,
    load_and_validate_inputs,
    validate_capture_request,
    validate_room_manifest,
)
from avengine.m1.evidence import (
    finalize_evidence,
    make_check,
    verify_evidence_artifacts,
)
from avengine.m1.habitat_capture import build_navmesh, capture_m1
from avengine.m3.canary import (
    load_and_verify_canary_evidence,
    run_material_activation_canary,
)
from avengine.m3.compiler import (
    AcousticSceneCompileError,
    compile_canary_request,
    compile_custom_acoustic_scene,
    compile_explicit_glb_research_scene,
    compile_mp3d_semantic_research_scene,
    compile_mp3d_soundspaces_research_scene,
    compile_usd_snapshot_semantic_research_scene,
    compile_visual_slot_semantic_research_scene,
    propose_visual_slot_research_materials,
)
from avengine.m3.contracts import (
    load_and_validate_acoustic_scene_package,
    read_immutable_file_snapshot,
    validate_package,
)
from avengine.m3.evidence import verify_compile_evidence
from avengine.m3.materials import MaterialContractError, resolve_material_profile
from avengine.m3.profiled_compiler import compile_registered_acoustic_scene
from avengine.m3.qa import automatic_mesh_leakage_report
from avengine.m3.real_rir_reference import (
    RealRIRReferenceError,
    verify_soundspaces2_real_rir_reference,
)
from avengine.m3.rlr_material_import import (
    RLRMaterialImportError,
    build_rlr_material_import_report,
    import_rlr_material_database,
)
from avengine.m3.runtime import RuntimeUnavailableError
from avengine.m4.canary import M4CanaryError, run_m4_canary
from avengine.m4.contracts import (
    M4ContractError,
    load_and_validate_multi_source_canary_request,
    validate_audio_bundle,
)
from avengine.m4.evidence import M4EvidenceError, verify_m4_canary_evidence
from avengine.m5.canary import M5CanaryError, run_m5_canary, verify_m5_canary_evidence
from avengine.m5.current_visual import CurrentVisualError, capture_current_visual
from avengine.m5.current_mp3d_route import (
    CurrentMP3DRouteError,
    author_current_mp3d_two_beagle_route,
)
from avengine.m5.timeline import validate_episode_request
from avengine.m6.rooms import load_room_registry
from avengine.m6.canary import (
    M6CanaryError,
    load_controlled_canary_request,
    run_controlled_canary,
    verify_controlled_canary_evidence,
)


EXIT_BY_STATUS = {"pass": 0, "fail": 1, "blocked": 3, "not_run": 3}


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


@contextmanager
def _temporary_native_audio_environment(
    *,
    runtime_prefix: str | None,
    rlr_sdk_root: str | None,
    magnum_python_site: str | None = None,
):
    """Apply explicit M3/M4 native-runtime inputs only for one CLI invocation."""

    updates = {
        "AVENGINE_HABITAT_RUNTIME_PREFIX": runtime_prefix,
        "AVENGINE_RLR_SDK_ROOT": rlr_sdk_root,
        "AVENGINE_HABITAT_MAGNUM_PYTHON_SITE": magnum_python_site,
    }
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is not None:
                os.environ[key] = str(Path(value).resolve())
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _require_current_installed_runtime_inputs(args: argparse.Namespace) -> None:
    if getattr(args, "runtime_mode", "historical") != "current-installed":
        return
    missing = [
        option
        for option, value in (
            ("--runtime-prefix", getattr(args, "runtime_prefix", None)),
            ("--rlr-sdk-root", getattr(args, "rlr_sdk_root", None)),
            ("--magnum-python-site", getattr(args, "magnum_python_site", None)),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "current-installed mode requires explicit " + ", ".join(missing)
        )


def _require_ignored_or_external_output(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    repository_root = Path(__file__).resolve().parents[2]
    try:
        relative = resolved.relative_to(repository_root)
    except ValueError:
        return resolved
    result = subprocess.run(
        ["git", "-C", str(repository_root), "check-ignore", "--quiet", str(relative)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            "Aggregate output inside the repository must be Git-ignored so "
            "writing it cannot invalidate clean evidence"
        )
    return resolved


def _require_ignored_repository_output(path: str | Path) -> Path:
    """Select a Git-ignored output under the trusted AVEngine workspace."""

    resolved = Path(path).resolve()
    repository_root = Path(__file__).resolve().parents[2]
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(
            "M6 controlled output must remain inside the trusted AVEngine "
            "repository and be Git-ignored"
        ) from exc
    return _require_ignored_or_external_output(resolved)


def _validate_room(args: argparse.Namespace) -> int:
    room = load_json(args.room)
    errors = validate_room_manifest(room)
    result = {
        "status": "pass" if not errors else "fail",
        "room": str(Path(args.room).resolve()),
        "errors": errors,
    }
    _print(result)
    return 0 if not errors else 2


def _validate_request(args: argparse.Namespace) -> int:
    request = load_json(args.request)
    room_id = None
    if args.room is not None:
        room_id = load_json(args.room).get("room_id")
    errors = validate_capture_request(request, room_id=room_id)
    result = {
        "status": "pass" if not errors else "fail",
        "request": str(Path(args.request).resolve()),
        "errors": errors,
    }
    _print(result)
    return 0 if not errors else 2


def _blocked_evidence(
    output: Path,
    inputs: ValidatedM1Inputs,
    error: Exception,
    *,
    schema_name: str = EVIDENCE_SCHEMA_V2,
) -> dict[str, Any]:
    message = str(error) or repr(error)
    evidence: dict[str, Any] = {
        "schema": schema_name,
        "evidence_kind": "blocked_attempt",
        "room_id": inputs.room["room_id"],
        "room_kind": inputs.room["room_kind"],
        "request_id": inputs.request["request_id"],
        "room_manifest": {
            "path": str(inputs.room_path),
            "sha256": sha256_file(inputs.room_path),
        },
        "capture_request": {
            "path": str(inputs.request_path),
            "sha256": sha256_file(inputs.request_path),
        },
        "output_directory": str(output),
        "exception": {
            "type": type(error).__name__,
            "message": message,
        },
        "checks": [
            make_check(
                "capture_execution",
                "blocked",
                measured={
                    "exception_type": type(error).__name__,
                    "message": message,
                },
                threshold={"capture_completed": True},
                failure_reason=message,
            )
        ],
    }
    finalize_evidence(evidence)
    write_json(output / "evidence.json", evidence)
    return evidence


def _capture(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    try:
        inputs = load_and_validate_inputs(args.room, args.request)
    except ContractError as error:
        _print({"status": "fail", "errors": error.errors})
        return 2
    try:
        evidence = capture_m1(
            inputs,
            output,
            runtime_prefix=args.runtime_prefix,
            repeat_count=args.repeat,
            reference_evidence=args.reference_evidence,
        )
    except Exception as error:
        evidence = _blocked_evidence(
            output, inputs, error, schema_name=EVIDENCE_SCHEMA_V2
        )
    summary = {
        "status": evidence["overall_status"],
        "evidence": str(output / "evidence.json"),
        "failed_or_blocked_checks": [
            check
            for check in evidence["checks"]
            if check["status"] != "pass" and check.get("required", True)
        ],
    }
    _print(summary)
    return EXIT_BY_STATUS[evidence["overall_status"]]


def _build_navmesh(args: argparse.Namespace) -> int:
    try:
        inputs = load_and_validate_inputs(args.room, args.request)
        result = build_navmesh(
            inputs,
            runtime_prefix=args.runtime_prefix,
            output_path=args.output,
        )
    except ContractError as error:
        _print({"status": "fail", "errors": error.errors})
        return 2
    except Exception as error:
        _print(
            {
                "status": "blocked",
                "exception_type": type(error).__name__,
                "message": str(error),
            }
        )
        return 3
    _print(result)
    return 0


def _verify(args: argparse.Namespace) -> int:
    status, checks = verify_evidence_artifacts(args.evidence)
    _print({"status": status, "checks": checks})
    return EXIT_BY_STATUS[status]


def _aggregate(args: argparse.Namespace) -> int:
    entries: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for path_value in args.evidence:
        path = Path(path_value).resolve()
        evidence = load_json(path)
        verification_status, verification_checks = verify_evidence_artifacts(path)
        entry = {
            "path": str(path),
            "room_id": evidence.get("room_id"),
            "room_kind": evidence.get("room_kind"),
            "request_id": evidence.get("request_id"),
            "declared_status": evidence.get("overall_status", "fail"),
            "verification_status": verification_status,
            "evidence_content_sha256": evidence.get("evidence_content_sha256"),
        }
        entries.append(entry)
        checks.append(
            make_check(
                f"room_{entry['room_id'] or path.stem}",
                verification_status,
                measured={
                    **entry,
                    "verification_checks": verification_checks,
                },
                threshold={"verification_status": "pass"},
                failure_reason=None
                if verification_status == "pass"
                else "Room evidence did not pass full semantic/artifact verification",
            )
        )
    kinds = [entry["room_kind"] for entry in entries]
    required_kinds = {
        "habitat_native",
        "blender_custom",
        "legacy_ue_real_surface_export",
    }
    room_ids = [entry["room_id"] for entry in entries]
    request_ids = [entry["request_id"] for entry in entries]
    kinds_are_strings = all(isinstance(value, str) and value for value in kinds)
    profile_passed = (
        len(entries) == 3
        and kinds_are_strings
        and set(kinds) == required_kinds
        and len(kinds) == len(set(kinds))
        and all(isinstance(value, str) and value for value in room_ids)
        and len(room_ids) == len(set(room_ids))
        and all(isinstance(value, str) and value for value in request_ids)
        and len(request_ids) == len(set(request_ids))
    )
    checks.append(
        make_check(
            "three_room_profile",
            "pass" if profile_passed else "fail",
            measured={
                "count": len(entries),
                "room_kinds": sorted(str(kind) for kind in kinds),
                "room_ids": room_ids,
                "request_ids": request_ids,
            },
            threshold={
                "count": 3,
                "room_kinds": sorted(required_kinds),
                "unique_room_and_request_ids": True,
            },
            failure_reason=None
            if profile_passed
            else "M1 requires exactly one verified evidence file for each room kind",
        )
    )
    status = aggregate_status(checks)
    result = {"status": status, "rooms": entries, "checks": checks}
    if args.output:
        write_json(_require_ignored_or_external_output(args.output), result)
    _print(result)
    return EXIT_BY_STATUS[status]


def _build_appearance_l9(args: argparse.Namespace) -> int:
    try:
        batch = generate_l9_batch(args.request, args.source)
        output = write_l9_batch_exclusive(args.output, batch)
    except AppearanceContractError as error:
        _print({"status": "fail", "errors": error.errors})
        return 2
    _print(
        {
            "status": "pass",
            "output": str(output),
            "batch_id": batch["batch_id"],
            "batch_content_sha256": batch["batch_content_sha256"],
            "request_count": len(batch["requests"]),
            "pairwise_orthogonal": batch["balance_audit"]["pairwise_orthogonal"],
            "ofat_status": batch["ofat_validation"]["status"],
        }
    )
    return 0


def _m3_validate_package(args: argparse.Namespace) -> int:
    errors = validate_package(args.manifest)
    result = {
        "status": "pass" if not errors else "fail",
        "manifest": str(Path(args.manifest).resolve()),
        "errors": errors,
    }
    _print(result)
    return 0 if not errors else 2


def _m3_compile_custom(args: argparse.Namespace) -> int:
    try:
        output = _require_ignored_or_external_output(args.output)
        manifest = compile_custom_acoustic_scene(
            room_manifest=args.room,
            material_mapping=args.mapping,
            material_database=args.materials,
            output=output,
            package_id=args.package_id,
        )
    except (AcousticSceneCompileError, OSError, ValueError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    _print({"status": "pass", "manifest": str(manifest)})
    return 0


def _m3_compile_canary(args: argparse.Namespace) -> int:
    try:
        output = _require_ignored_or_external_output(args.output)
        evidence = compile_canary_request(args.request, output)
    except (AcousticSceneCompileError, OSError, ValueError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    _print({"status": "pass", "compile_evidence": str(evidence)})
    return 0


def _m3_verify_compile(args: argparse.Namespace) -> int:
    try:
        status, checks = verify_compile_evidence(args.evidence)
    except (OSError, ValueError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    _print({"status": status, "checks": checks})
    return 0 if status == "pass" else 1


def _m3_run_canary(args: argparse.Namespace) -> int:
    try:
        _require_current_installed_runtime_inputs(args)
        output = _require_ignored_or_external_output(args.output)
        with _temporary_native_audio_environment(
            runtime_prefix=args.runtime_prefix,
            rlr_sdk_root=args.rlr_sdk_root,
            magnum_python_site=args.magnum_python_site,
        ):
            evidence_path = run_material_activation_canary(
                args.request,
                args.compile_evidence,
                output,
                runtime_mode=args.runtime_mode,
                runtime_prefix=args.runtime_prefix,
                rlr_sdk_root=args.rlr_sdk_root,
                magnum_python_site=args.magnum_python_site,
            )
        result = load_and_verify_canary_evidence(evidence_path)
        if result.errors:
            _print(
                {
                    "status": "fail",
                    "canary_evidence": str(evidence_path),
                    "errors": list(result.errors),
                }
            )
            return 2
        status = result.evidence["overall_status"]
    except RuntimeUnavailableError as error:
        _print({"status": "blocked", "error": str(error)})
        return 3
    except (OSError, ValueError, RuntimeError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    _print({"status": status, "canary_evidence": str(evidence_path)})
    return EXIT_BY_STATUS[status]


def _m3_verify_canary(args: argparse.Namespace) -> int:
    try:
        result = load_and_verify_canary_evidence(args.evidence)
    except (OSError, ValueError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    errors = list(result.errors)
    declared_status = result.evidence.get("overall_status", "fail")
    verification_status = "pass" if not errors else "fail"
    _print(
        {
            "status": declared_status if not errors else "fail",
            "verification_status": verification_status,
            "declared_status": declared_status,
            "errors": errors,
        }
    )
    if errors:
        return 2
    return EXIT_BY_STATUS.get(str(declared_status), 2)


def _m3_environment(runtime_root: str | None) -> dict[str, str]:
    environment = dict(os.environ)
    if runtime_root is not None:
        environment["AVENGINE_HABITAT_RUNTIME_ROOT"] = str(Path(runtime_root).resolve())
    return environment


def _m3_propose_visual_slots(args: argparse.Namespace) -> int:
    try:
        output = _require_ignored_or_external_output(args.output)
        mapping, materials, report = propose_visual_slot_research_materials(
            room_manifest=args.room,
            output=output,
            transform_profile=args.transform_profile,
            transform_reviewed=args.confirm_reviewed_transform,
            environment=_m3_environment(args.runtime_root),
        )
    except (AcousticSceneCompileError, OSError, ValueError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    _print(
        {
            "status": "research_candidate",
            "mapping": str(mapping),
            "materials": str(materials),
            "proposal_report": str(report),
            "qualification_claim": False,
        }
    )
    return 0


def _m3_compile_research(args: argparse.Namespace) -> int:
    try:
        output = _require_ignored_or_external_output(args.output)
        manifest = compile_explicit_glb_research_scene(
            room_manifest=args.room,
            material_mapping=args.mapping,
            material_database=args.materials,
            output=output,
            package_id=args.package_id,
            environment=_m3_environment(args.runtime_root),
        )
    except (AcousticSceneCompileError, OSError, ValueError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    _print(
        {
            "status": "research_candidate",
            "manifest": str(manifest),
            "qualification_claim": False,
        }
    )
    return 0


def _m3_compile_mp3d_semantic(args: argparse.Namespace) -> int:
    try:
        output = _require_ignored_or_external_output(args.output)
        manifest, coverage = compile_mp3d_semantic_research_scene(
            room_manifest=args.room,
            material_rules=args.rules,
            output=output,
            seed=args.seed,
            package_id=args.package_id,
            probe_origins=args.probe_origin,
            probe_direction_count=args.probe_directions,
            environment=_m3_environment(args.runtime_root),
        )
    except (AcousticSceneCompileError, OSError, ValueError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    _print(
        {
            "status": "research_candidate",
            "manifest": str(manifest),
            "semantic_material_coverage": str(coverage),
            "qualification_claim": False,
        }
    )
    return 0


def _m3_compile_mp3d_rlr_materials(args: argparse.Namespace) -> int:
    try:
        output = _require_ignored_or_external_output(args.output)
        manifest, coverage = compile_mp3d_soundspaces_research_scene(
            room_manifest=args.room,
            material_config=args.materials,
            output=output,
            database_id=args.database_id,
            version=args.version,
            source_description=args.source_description,
            source_uri=args.source_uri,
            package_id=args.package_id,
            probe_origins=args.probe_origin,
            probe_direction_count=args.probe_directions,
            environment=_m3_environment(args.runtime_root),
        )
    except (AcousticSceneCompileError, OSError, ValueError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    _print(
        {
            "status": "research_candidate",
            "manifest": str(manifest),
            "semantic_material_coverage": str(coverage),
            "material_source": "soundspaces_rlr_native_config",
            "qualification_claim": False,
        }
    )
    return 0


def _m3_compile_usd_snapshot_semantic(args: argparse.Namespace) -> int:
    try:
        output = _require_ignored_or_external_output(args.output)
        manifest, coverage = compile_usd_snapshot_semantic_research_scene(
            room_manifest=args.room,
            material_rules=args.rules,
            output=output,
            seed=args.seed,
            package_id=args.package_id,
            probe_origins=args.probe_origin,
            probe_direction_count=args.probe_directions,
            environment=_m3_environment(args.runtime_root),
        )
    except (AcousticSceneCompileError, OSError, ValueError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    _print(
        {
            "status": "research_candidate",
            "manifest": str(manifest),
            "semantic_material_coverage": str(coverage),
            "qualification_claim": False,
        }
    )
    return 0


def _m3_compile_visual_slots_semantic(args: argparse.Namespace) -> int:
    try:
        output = _require_ignored_or_external_output(args.output)
        manifest, coverage = compile_visual_slot_semantic_research_scene(
            room_manifest=args.room,
            material_rules=args.rules,
            output=output,
            seed=args.seed,
            transform_profile=args.transform_profile,
            transform_reviewed=args.confirm_reviewed_transform,
            package_id=args.package_id,
            probe_origins=args.probe_origin,
            probe_direction_count=args.probe_directions,
            environment=_m3_environment(args.runtime_root),
        )
    except (AcousticSceneCompileError, OSError, ValueError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    _print(
        {
            "status": "research_candidate",
            "manifest": str(manifest),
            "semantic_material_coverage": str(coverage),
            "qualification_claim": False,
        }
    )
    return 0


def _m3_compile_registered_scene(args: argparse.Namespace) -> int:
    try:
        output = _require_ignored_or_external_output(args.output)
        environment = _m3_environment(args.runtime_root)
        acoustic_registry = (
            load_acoustic_profile_registry(args.acoustic_profile_registry)
            if args.acoustic_profile_registry is not None
            else load_default_acoustic_profile_registry()
        )
        room_registry = load_room_registry(args.room_registry)
        result = compile_registered_acoustic_scene(
            acoustic_registry,
            room_registry,
            {
                "registry_id": room_registry["registry_id"],
                "room_id": args.room_id,
                "revision": args.room_revision,
            },
            room_manifest=args.room,
            output=output,
            repository_root=Path(__file__).resolve().parents[2],
            environment=environment,
            seed=args.seed,
            package_id=args.package_id,
            probe_origins=args.probe_origin,
            probe_direction_count=args.probe_directions,
        )
        receipt = result.receipt()
    except (
        AcousticProfileError,
        AcousticSceneCompileError,
        OSError,
        ValueError,
    ) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    _print(
        {
            "status": "research_candidate",
            "manifest": str(result.manifest_path),
            "semantic_material_coverage": str(result.coverage_path),
            "compiler_route": result.compiler_route,
            "room_ref": receipt["room_ref"],
            "profile_ref": receipt["profile_ref"],
            "geometry_resource_id": receipt["geometry_resource_id"],
            "solver_backend_id": receipt["solver_backend_id"],
            "receipt_content_sha256": receipt["receipt_content_sha256"],
            "qualification_claim": False,
        }
    )
    return 0


def _m3_inspect_mesh_leakage(args: argparse.Namespace) -> int:
    try:
        package = load_and_validate_acoustic_scene_package(args.package)
        report = automatic_mesh_leakage_report(
            package.vertices,
            package.triangles,
            origins=args.origin,
            direction_count=args.directions,
            maximum_distance_m=args.maximum_distance,
        )
        report["source_package"] = {
            "manifest": str(package.manifest_path),
            "package_id": package.manifest["package_id"],
            "room_id": package.manifest["source_room"]["room_id"],
            "vertex_count": package.vertex_count,
            "triangle_count": package.triangle_count,
        }
        topology = package.qa_reports["geometry_report"]["topology"]
        report["topology_context"] = {
            "geometry_status": package.qa_reports["geometry_report"]["status"],
            "boundary_edge_count_after_exact_weld": topology[
                "boundary_edge_count_after_exact_weld"
            ],
            "nonmanifold_edge_count_after_exact_weld": topology[
                "nonmanifold_edge_count_after_exact_weld"
            ],
            "degenerate_triangle_count": topology["degenerate_triangle_count"],
            "duplicate_triangle_count": topology["duplicate_triangle_count"],
        }
        output = _require_ignored_or_external_output(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to replace existing output: {output}")
        write_json(output, report)
    except (OSError, ValueError, KeyError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    _print(
        {
            "status": report["status"],
            "report": str(output),
            "room_id": report["source_package"]["room_id"],
            "escape_fraction": report["escape_fraction"],
            "escaped_ray_count": report["escaped_ray_count"],
        }
    )
    return 0


def _m3_resolve_materials(args: argparse.Namespace) -> int:
    """Materialize one deterministic global/per-material profile bundle."""

    destination: Path | None = None
    staging: Path | None = None
    try:
        destination = _require_ignored_or_external_output(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError(f"refusing to replace existing output: {destination}")

        snapshots = {
            name: read_immutable_file_snapshot(path)
            for name, path in (
                ("mapping", args.mapping),
                ("base_materials", args.base_materials),
                ("profile", args.profile),
            )
        }
        documents: dict[str, dict[str, Any]] = {}
        for name, snapshot in snapshots.items():
            value = json.loads(snapshot.payload)
            if not isinstance(value, dict):
                raise ValueError(f"{name} input must be a JSON object")
            documents[name] = value

        mapping = documents["mapping"]
        resolved = resolve_material_profile(
            mapping,
            documents["base_materials"],
            documents["profile"],
            room_id=str(mapping.get("room_id", "")),
        )
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.staging-", dir=destination.parent
            )
        ).resolve()
        mapping_path = staging / "mapping.json"
        materials_path = staging / "materials.json"
        report_path = staging / "resolution_report.json"
        write_json(mapping_path, resolved.effective_mapping)
        write_json(materials_path, resolved.effective_database)
        report = {
            **resolved.report,
            "input_files": {
                name: {
                    "byte_size": snapshot.byte_size,
                    "sha256": snapshot.sha256,
                }
                for name, snapshot in snapshots.items()
            },
            "outputs": {
                "mapping": file_record(mapping_path, relative_to=staging),
                "materials": file_record(materials_path, relative_to=staging),
            },
        }
        write_json(report_path, report)
        os.rename(staging, destination)
        staging = None
    except (MaterialContractError, OSError, ValueError) as error:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        _print({"status": "fail", "error": str(error)})
        return 2

    _print(
        {
            "status": "pass",
            "mapping": str(destination / "mapping.json"),
            "materials": str(destination / "materials.json"),
            "resolution_report": str(destination / "resolution_report.json"),
        }
    )
    return 0


def _m3_verify_soundspaces_reference(args: argparse.Namespace) -> int:
    """Bind and reproduce the public SoundSpaces 2 real-RIR summary."""

    staging: Path | None = None
    try:
        destination = _require_ignored_or_external_output(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError(f"refusing to replace existing output: {destination}")
        report = verify_soundspaces2_real_rir_reference(args.reference_root)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.staging-", dir=destination.parent
            )
        ).resolve()
        staged_report = staging / "report.json"
        write_json(staged_report, report)
        os.link(staged_report, destination)
        shutil.rmtree(staging, ignore_errors=True)
        staging = None
    except (OSError, RealRIRReferenceError, ValueError) as error:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        _print({"status": "fail", "error": str(error)})
        return 2
    _print(
        {
            "status": "pass",
            "report": str(destination),
            "reference_verified": report["reference_verified"],
            "pinned_snapshot_identity_verified": report[
                "pinned_snapshot_identity_verified"
            ],
            "published_summary_reproduced": report["published_summary_reproduced"],
            "qualification_claim": report["qualification_claim"],
            "coordinate_binding": report["coordinate_binding"],
            "computed_summary": report["computed_summary"],
        }
    )
    return 0


def _m3_import_rlr_materials(args: argparse.Namespace) -> int:
    """Import an upstream RLR database without resampling its curves."""

    destination: Path | None = None
    staging: Path | None = None
    try:
        destination = _require_ignored_or_external_output(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError(f"refusing to replace existing output: {destination}")

        source_snapshot = read_immutable_file_snapshot(args.source)
        source = json.loads(source_snapshot.payload)
        if not isinstance(source, dict):
            raise ValueError("RLR material source must be a JSON object")
        database = import_rlr_material_database(
            source,
            database_id=args.database_id,
            version=args.version,
            source_description=args.source_description,
        )

        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.staging-", dir=destination.parent
            )
        ).resolve()
        materials_path = staging / "materials.json"
        report_path = staging / "import_report.json"
        write_json(materials_path, database)
        report = build_rlr_material_import_report(
            source_snapshot.path,
            database,
            output_path=materials_path,
            source_uri=args.source_uri,
        )
        if (
            report["source"]["byte_size"] != source_snapshot.byte_size
            or report["source"]["sha256"] != source_snapshot.sha256
        ):
            raise ValueError("RLR material source changed during import")
        report["output"]["path"] = "materials.json"
        write_json(report_path, report)
        os.rename(staging, destination)
        staging = None
    except (
        json.JSONDecodeError,
        OSError,
        RLRMaterialImportError,
        ValueError,
    ) as error:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        _print({"status": "fail", "error": str(error)})
        return 2

    _print(
        {
            "status": "pass",
            "materials": str(destination / "materials.json"),
            "import_report": str(destination / "import_report.json"),
            "material_count": report["statistics"]["material_count"],
            "preserved_exactly": report["roundtrip"]["preserved_exactly"],
            "frl_measurement_fitted": report["claims"]["frl_measurement_fitted"],
        }
    )
    return 0


def _m4_validate_request(args: argparse.Namespace) -> int:
    try:
        validated = load_and_validate_multi_source_canary_request(args.request)
    except M4ContractError as error:
        _print({"status": "fail", "errors": list(error.errors)})
        return 2
    _print(
        {
            "status": "pass",
            "request": str(validated.request_path),
            "request_id": validated.request["request_id"],
            "listener_id": validated.listener["listener_id"],
            "canonical_source_ids": list(validated.canonical_source_ids),
            "all_m2_anchor_evidence_available": (
                validated.all_m2_anchor_evidence_available
            ),
            "identity_position_authority": "formal_m1_source_pose",
        }
    )
    return 0


def _m4_run_canary(args: argparse.Namespace) -> int:
    missing_dependencies = [
        str(Path(path).resolve())
        for path in (args.hrtf, args.hrtf_license)
        if not Path(path).resolve().is_file()
    ]
    if missing_dependencies:
        _print(
            {
                "status": "blocked",
                "reason": "explicit HRTF or its license evidence is unavailable",
                "missing": missing_dependencies,
            }
        )
        return 3
    try:
        _require_current_installed_runtime_inputs(args)
        m4_runtime_kwargs: dict[str, Any] = {}
        runtime_lock: str | None = args.runtime_lock
        if args.runtime_mode == "current-installed":
            missing_hrtf_metadata = [
                option
                for option, value in (
                    ("--current-hrtf-sample-rate-hz", args.current_hrtf_sample_rate_hz),
                    ("--current-hrtf-license-id", args.current_hrtf_license_id),
                    ("--current-hrtf-citation", args.current_hrtf_citation),
                )
                if value is None or not str(value).strip()
            ]
            if missing_hrtf_metadata:
                raise ValueError(
                    "current-installed mode requires explicit "
                    + ", ".join(missing_hrtf_metadata)
                )
            runtime_lock = None
            m4_runtime_kwargs = {
                "runtime_mode": args.runtime_mode,
                "runtime_prefix": args.runtime_prefix,
                "rlr_sdk_root": args.rlr_sdk_root,
                "magnum_python_site": args.magnum_python_site,
                "current_hrtf_sample_rate_hz": args.current_hrtf_sample_rate_hz,
                "current_hrtf_license_id": args.current_hrtf_license_id,
                "current_hrtf_citation": args.current_hrtf_citation,
            }
        output = _require_ignored_or_external_output(args.output)
        with _temporary_native_audio_environment(
            runtime_prefix=args.runtime_prefix,
            rlr_sdk_root=args.rlr_sdk_root,
            magnum_python_site=args.magnum_python_site,
        ):
            evidence = run_m4_canary(
                args.request,
                args.package_manifest,
                runtime_lock,
                output,
                hrtf_path=args.hrtf,
                hrtf_license_path=args.hrtf_license,
                **m4_runtime_kwargs,
            )
    except RuntimeUnavailableError as error:
        _print({"status": "blocked", "error": str(error)})
        return 3
    except (M4CanaryError, M4EvidenceError, OSError, ValueError, RuntimeError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    status, checks = verify_m4_canary_evidence(evidence)
    _print(
        {
            "status": status,
            "canary_evidence": str(evidence),
            "failed_checks": [
                check for check in checks if check.get("status") != "pass"
            ],
        }
    )
    return EXIT_BY_STATUS.get(status, 2)


def _m4_verify_canary(args: argparse.Namespace) -> int:
    try:
        status, checks = verify_m4_canary_evidence(args.evidence)
    except (OSError, ValueError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    _print({"status": status, "checks": checks})
    return 0 if status == "pass" else 1


def _m4_verify_bundle(args: argparse.Namespace) -> int:
    try:
        bundle = load_json(args.bundle)
        errors = validate_audio_bundle(bundle, bundle_path=args.bundle)
    except (OSError, ValueError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    _print({"status": "pass" if not errors else "fail", "errors": errors})
    return 0 if not errors else 1


def _m5_validate_request(args: argparse.Namespace) -> int:
    try:
        request = load_json(args.request)
        errors = validate_episode_request(request)
    except (OSError, ValueError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    _print(
        {
            "status": "pass" if not errors else "fail",
            "request": str(Path(args.request).resolve()),
            "request_id": request.get("request_id"),
            "simultaneous_source_count": len(request.get("sources", [])),
            "formal_view_ids": request.get("timeline_profile", {})
            .get("video", {})
            .get("view_ids"),
            "errors": errors,
        }
    )
    return 0 if not errors else 2


def _m5_run_canary(args: argparse.Namespace) -> int:
    try:
        output = _require_ignored_or_external_output(args.output)
        evidence = run_m5_canary(
            request_path=args.request,
            animal_manifest_path=args.animal_manifest,
            m2_request_path=args.m2_request,
            room_manifest_path=args.room_manifest,
            m1_request_path=args.m1_request,
            acoustic_package_manifest_path=args.acoustic_package_manifest,
            m4_request_path=args.m4_request,
            output_directory=output,
            runtime_root=args.runtime_root,
            hrtf_path=args.hrtf,
            hrtf_license_path=args.hrtf_license,
            beagle_dry_path=args.beagle_dry,
            golden_dry_path=args.golden_dry,
            sensor_rig_trajectory_path=args.sensor_rig_trajectory,
        )
        status, checks = verify_m5_canary_evidence(evidence)
    except RuntimeUnavailableError as error:
        _print({"status": "blocked", "error": str(error)})
        return 3
    except (M5CanaryError, OSError, ValueError, RuntimeError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    _print(
        {
            "status": status,
            "canary_evidence": str(evidence),
            "failed_checks": [
                check for check in checks if check.get("status") != "pass"
            ],
        }
    )
    return 0 if status == "pass" else 1


def _m5_capture_current_visual(args: argparse.Namespace) -> int:
    try:
        output = _require_ignored_or_external_output(args.output)
        receipt = capture_current_visual(
            animal_manifest_path=args.animal_manifest,
            m2_request_path=args.m2_request,
            room_manifest_path=args.room_manifest,
            m1_request_path=args.m1_request,
            runtime_prefix=args.runtime_prefix,
            mp3d_root=args.mp3d_root,
            magnum_python_site=args.magnum_python_site,
            output_directory=output,
        )
    except (CurrentVisualError, OSError, ValueError, RuntimeError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    status = receipt["status"]
    _print(
        {
            "status": status,
            "research_only": receipt["research_only"],
            "episode_counted": receipt["episode_counted"],
            "output": str(output),
            "receipt": str(output / "research_receipt.json"),
            **(
                {"reason": receipt["reason"]}
                if status == "not_run"
                else {"frame_count": receipt["capture"]["frame_count"]}
            ),
        }
    )
    return 0 if status == "research_only" else EXIT_BY_STATUS.get(status, 2)


def _m5_author_current_mp3d_two_beagle_route(args: argparse.Namespace) -> int:
    try:
        result = author_current_mp3d_two_beagle_route(
            source_animal_manifest_path=args.source_animal_manifest,
            source_m2_request_path=args.source_m2_request,
            runtime_prefix=args.runtime_prefix,
            mp3d_root=args.mp3d_root,
            magnum_python_site=args.magnum_python_site,
            output_directory=args.output,
            seed=args.seed,
            distance_tolerance_m=args.distance_tolerance_m,
            minimum_center_separation_m=args.minimum_center_separation_m,
        )
    except (CurrentMP3DRouteError, OSError, ValueError, RuntimeError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    _print(result)
    return 0


def _m5_verify_canary(args: argparse.Namespace) -> int:
    try:
        status, checks = verify_m5_canary_evidence(args.evidence)
    except (OSError, ValueError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    _print({"status": status, "checks": checks})
    return 0 if status == "pass" else 1


def _m6_validate_controlled_request(args: argparse.Namespace) -> int:
    try:
        request = load_controlled_canary_request(args.request)
    except (M6CanaryError, OSError, ValueError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    _print(
        {
            "status": "pass",
            "request": str(Path(args.request).resolve()),
            "run_id": request["run_id"],
            "research_only": request["research_only"],
            "qualification_claim": request["qualification_claim"],
        }
    )
    return 0


def _m6_run_controlled(args: argparse.Namespace) -> int:
    try:
        output = _require_ignored_repository_output(args.output)
        evidence = run_controlled_canary(
            request_path=args.request,
            upstream_evidence_path=args.upstream_evidence,
            output_directory=output,
            implementation_commit=args.implementation_commit,
            registry_directory=args.registry_directory,
            room_registry_path=args.room_registry,
            room_qualification_path=args.room_qualification,
            program_path=args.program,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
        )
    except (M6CanaryError, OSError, ValueError, RuntimeError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    generated = load_json(evidence)
    generated_status = str(generated.get("overall_status"))
    _print(
        {
            "status": generated_status,
            "research_only": True,
            "qualification_claim": False,
            "canary_evidence": str(evidence),
        }
    )
    return EXIT_BY_STATUS.get(generated_status, 2)


def _m6_verify_controlled(args: argparse.Namespace) -> int:
    try:
        status, checks = verify_controlled_canary_evidence(
            args.evidence, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe
        )
    except (M6CanaryError, OSError, ValueError) as error:
        _print({"status": "fail", "error": str(error)})
        return 2
    _print({"status": status, "checks": checks})
    return 0 if status == "pass" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="avengine")
    commands = parser.add_subparsers(dest="command", required=True)
    m1 = commands.add_parser("m1", help="M1 visual/room canary commands")
    m1_commands = m1.add_subparsers(dest="m1_command", required=True)

    validate_room = m1_commands.add_parser("validate-room")
    validate_room.add_argument("room")
    validate_room.set_defaults(handler=_validate_room)

    validate_request = m1_commands.add_parser("validate-request")
    validate_request.add_argument("request")
    validate_request.add_argument("--room")
    validate_request.set_defaults(handler=_validate_request)

    capture = m1_commands.add_parser("capture")
    capture.add_argument("--room", required=True)
    capture.add_argument("--request", required=True)
    capture.add_argument("--output", required=True)
    capture.add_argument("--runtime-prefix", required=True)
    capture.add_argument("--repeat", type=int, default=3)
    capture.add_argument(
        "--reference-evidence",
        help="First-run evidence from a separate process for deterministic rerun proof",
    )
    capture.set_defaults(handler=_capture)

    navmesh = m1_commands.add_parser("build-navmesh")
    navmesh.add_argument("--room", required=True)
    navmesh.add_argument("--request", required=True)
    navmesh.add_argument("--runtime-prefix", required=True)
    navmesh.add_argument("--output")
    navmesh.set_defaults(handler=_build_navmesh)

    verify = m1_commands.add_parser("verify")
    verify.add_argument("evidence")
    verify.set_defaults(handler=_verify)

    aggregate = m1_commands.add_parser("aggregate")
    aggregate.add_argument("evidence", nargs="+")
    aggregate.add_argument("--output")
    aggregate.set_defaults(handler=_aggregate)

    m3 = commands.add_parser("m3", help="M3 explicit acoustic-scene commands")
    m3_commands = m3.add_subparsers(dest="m3_command", required=True)

    m3_validate = m3_commands.add_parser("validate-package")
    m3_validate.add_argument("manifest")
    m3_validate.set_defaults(handler=_m3_validate_package)

    m3_compile = m3_commands.add_parser("compile-custom")
    m3_compile.add_argument("--room", required=True)
    m3_compile.add_argument("--mapping", required=True)
    m3_compile.add_argument("--materials", required=True)
    m3_compile.add_argument("--output", required=True)
    m3_compile.add_argument("--package-id")
    m3_compile.set_defaults(handler=_m3_compile_custom)

    m3_canary = m3_commands.add_parser("compile-canary")
    m3_canary.add_argument("--request", required=True)
    m3_canary.add_argument("--output", required=True)
    m3_canary.set_defaults(handler=_m3_compile_canary)

    m3_verify = m3_commands.add_parser("verify-compile")
    m3_verify.add_argument("evidence")
    m3_verify.set_defaults(handler=_m3_verify_compile)

    m3_run_canary = m3_commands.add_parser("run-canary")
    m3_run_canary.add_argument("--request", required=True)
    m3_run_canary.add_argument("--compile-evidence", required=True)
    m3_run_canary.add_argument("--output", required=True)
    m3_run_canary.add_argument(
        "--runtime-prefix",
        help=(
            "Explicit installed non-checkout Habitat prefix required with "
            "--runtime-mode current-installed"
        ),
    )
    m3_run_canary.add_argument(
        "--rlr-sdk-root",
        help=(
            "Explicit external non-checkout RLRAudioPropagationPkg required with "
            "--runtime-mode current-installed"
        ),
    )
    m3_run_canary.add_argument(
        "--runtime-mode",
        choices=["historical", "current-installed"],
        default="historical",
        help="historical verifies the v1 lock; current-installed records only one fresh external runtime identity",
    )
    m3_run_canary.add_argument(
        "--magnum-python-site",
        help="External Corrade/Magnum Python site; required with --runtime-mode current-installed",
    )
    m3_run_canary.set_defaults(handler=_m3_run_canary)

    m3_verify_canary = m3_commands.add_parser("verify-canary")
    m3_verify_canary.add_argument("evidence")
    m3_verify_canary.set_defaults(handler=_m3_verify_canary)

    m3_propose = m3_commands.add_parser("propose-visual-slots")
    m3_propose.add_argument("--room", required=True)
    m3_propose.add_argument(
        "--transform-profile",
        required=True,
        choices=["identity_y_up", "mp3d_z_up_y_front_to_habitat"],
    )
    m3_propose.add_argument("--runtime-root")
    m3_propose.add_argument(
        "--confirm-reviewed-transform",
        action="store_true",
        help="Record that the selected source-to-canonical transform was reviewed for this exact room",
    )
    m3_propose.add_argument("--output", required=True)
    m3_propose.set_defaults(handler=_m3_propose_visual_slots)

    m3_research = m3_commands.add_parser("compile-explicit-research")
    m3_research.add_argument("--room", required=True)
    m3_research.add_argument("--mapping", required=True)
    m3_research.add_argument("--materials", required=True)
    m3_research.add_argument("--runtime-root")
    m3_research.add_argument("--output", required=True)
    m3_research.add_argument("--package-id")
    m3_research.set_defaults(handler=_m3_compile_research)

    m3_semantic = m3_commands.add_parser(
        "compile-mp3d-semantic",
        help="Compile MP3D semantic PLY/.house labels into a research RLR package",
    )
    m3_semantic.add_argument("--room", required=True)
    m3_semantic.add_argument("--rules", required=True)
    m3_semantic.add_argument("--seed", type=int, default=917)
    m3_semantic.add_argument("--runtime-root")
    m3_semantic.add_argument(
        "--probe-origin",
        nargs=3,
        type=float,
        action="append",
        metavar=("X", "Y", "Z"),
        help=(
            "Canonical interior point for automatic enclosure probes; may be "
            "repeated. Defaults to up to two room connectivity anchors."
        ),
    )
    m3_semantic.add_argument("--probe-directions", type=int, default=32)
    m3_semantic.add_argument("--output", required=True)
    m3_semantic.add_argument("--package-id")
    m3_semantic.set_defaults(handler=_m3_compile_mp3d_semantic)

    m3_soundspaces = m3_commands.add_parser(
        "compile-mp3d-rlr-materials",
        help=(
            "Compile MP3D semantic PLY/.house geometry by replaying official "
            "RLR label matching over native SoundSpaces material curves"
        ),
    )
    m3_soundspaces.add_argument("--room", required=True)
    m3_soundspaces.add_argument("--materials", required=True)
    m3_soundspaces.add_argument("--database-id", required=True)
    m3_soundspaces.add_argument("--version", default="1")
    m3_soundspaces.add_argument("--source-description", required=True)
    m3_soundspaces.add_argument("--source-uri")
    m3_soundspaces.add_argument("--runtime-root")
    m3_soundspaces.add_argument(
        "--probe-origin",
        nargs=3,
        type=float,
        action="append",
        metavar=("X", "Y", "Z"),
        help=(
            "Canonical interior point for automatic enclosure probes; may be "
            "repeated. Defaults to up to two room connectivity anchors."
        ),
    )
    m3_soundspaces.add_argument("--probe-directions", type=int, default=32)
    m3_soundspaces.add_argument("--output", required=True)
    m3_soundspaces.add_argument("--package-id")
    m3_soundspaces.set_defaults(handler=_m3_compile_mp3d_rlr_materials)

    m3_usd_semantic = m3_commands.add_parser(
        "compile-usd-snapshot-semantic",
        help=(
            "Compile an externally expanded USD acoustic snapshot into a "
            "research RLR package"
        ),
    )
    m3_usd_semantic.add_argument("--room", required=True)
    m3_usd_semantic.add_argument("--rules", required=True)
    m3_usd_semantic.add_argument("--seed", type=int, default=917)
    m3_usd_semantic.add_argument("--runtime-root")
    m3_usd_semantic.add_argument(
        "--probe-origin",
        nargs=3,
        type=float,
        action="append",
        metavar=("X", "Y", "Z"),
        help=(
            "Canonical interior point for automatic enclosure probes; may be "
            "repeated. Defaults to reviewed origins stored in the snapshot."
        ),
    )
    m3_usd_semantic.add_argument("--probe-directions", type=int, default=32)
    m3_usd_semantic.add_argument("--output", required=True)
    m3_usd_semantic.add_argument("--package-id")
    m3_usd_semantic.set_defaults(handler=_m3_compile_usd_snapshot_semantic)

    m3_slots_semantic = m3_commands.add_parser(
        "compile-visual-slots-semantic",
        help=(
            "Resolve visual material-slot names through semantic rules into a "
            "research RLR package"
        ),
    )
    m3_slots_semantic.add_argument("--room", required=True)
    m3_slots_semantic.add_argument("--rules", required=True)
    m3_slots_semantic.add_argument("--seed", type=int, default=917)
    m3_slots_semantic.add_argument(
        "--transform-profile",
        required=True,
        choices=["identity_y_up", "mp3d_z_up_y_front_to_habitat"],
    )
    m3_slots_semantic.add_argument(
        "--confirm-reviewed-transform",
        action="store_true",
        help=(
            "Record that the selected source-to-canonical transform was "
            "reviewed for this exact room"
        ),
    )
    m3_slots_semantic.add_argument("--runtime-root")
    m3_slots_semantic.add_argument(
        "--probe-origin",
        nargs=3,
        type=float,
        action="append",
        metavar=("X", "Y", "Z"),
        help=(
            "Canonical interior point for automatic enclosure probes; may be "
            "repeated. Defaults to up to two room connectivity anchors."
        ),
    )
    m3_slots_semantic.add_argument("--probe-directions", type=int, default=32)
    m3_slots_semantic.add_argument("--output", required=True)
    m3_slots_semantic.add_argument("--package-id")
    m3_slots_semantic.set_defaults(handler=_m3_compile_visual_slots_semantic)

    m3_registered = m3_commands.add_parser(
        "compile-registered-scene",
        help=(
            "Select a SoundSpaces, Habitat, or SPEAR/UE acoustic profile from "
            "an exact room identity and compile the common RLR scene package"
        ),
    )
    m3_registered.add_argument("--room", required=True)
    m3_registered.add_argument("--room-id", required=True)
    m3_registered.add_argument("--room-revision", required=True)
    m3_registered.add_argument(
        "--room-registry",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2] / "examples/m6/rooms/room_registry.json"
        ),
    )
    m3_registered.add_argument(
        "--acoustic-profile-registry",
        type=Path,
        help="Override the installed/default acoustic profile registry",
    )
    m3_registered.add_argument("--seed", type=int, default=917)
    m3_registered.add_argument("--runtime-root")
    m3_registered.add_argument(
        "--probe-origin",
        nargs=3,
        type=float,
        action="append",
        metavar=("X", "Y", "Z"),
        help=(
            "Canonical interior point for automatic enclosure probes; may be "
            "repeated. Defaults to room connectivity anchors."
        ),
    )
    m3_registered.add_argument("--probe-directions", type=int, default=32)
    m3_registered.add_argument("--output", required=True)
    m3_registered.add_argument("--package-id")
    m3_registered.set_defaults(handler=_m3_compile_registered_scene)

    m3_leakage = m3_commands.add_parser(
        "inspect-mesh-leakage",
        help="Run automatic interior-ray leakage diagnostics on an existing package",
    )
    m3_leakage.add_argument("--package", required=True)
    m3_leakage.add_argument(
        "--origin",
        nargs=3,
        type=float,
        action="append",
        required=True,
        metavar=("X", "Y", "Z"),
        help="Reviewed canonical interior point; repeat for multiple room zones",
    )
    m3_leakage.add_argument("--directions", type=int, default=32)
    m3_leakage.add_argument("--maximum-distance", type=float)
    m3_leakage.add_argument("--output", required=True)
    m3_leakage.set_defaults(handler=_m3_inspect_mesh_leakage)

    m3_resolve = m3_commands.add_parser(
        "resolve-materials",
        help="Resolve global and exact per-material coefficient overrides",
    )
    m3_resolve.add_argument("--mapping", required=True)
    m3_resolve.add_argument("--base-materials", required=True)
    m3_resolve.add_argument("--profile", required=True)
    m3_resolve.add_argument("--output", required=True)
    m3_resolve.set_defaults(handler=_m3_resolve_materials)

    m3_reference = m3_commands.add_parser(
        "verify-soundspaces-reference",
        help="Verify the published SoundSpaces 2 seven-RIR comparison bundle",
    )
    m3_reference.add_argument("--reference-root", required=True)
    m3_reference.add_argument("--output", required=True)
    m3_reference.set_defaults(handler=_m3_verify_soundspaces_reference)

    m3_import_rlr = m3_commands.add_parser(
        "import-rlr-materials",
        help="Import an upstream RLR material database without interpolation",
    )
    m3_import_rlr.add_argument("--source", required=True)
    m3_import_rlr.add_argument("--database-id", required=True)
    m3_import_rlr.add_argument("--version", default="1")
    m3_import_rlr.add_argument("--source-description", required=True)
    m3_import_rlr.add_argument("--source-uri")
    m3_import_rlr.add_argument("--output", required=True)
    m3_import_rlr.set_defaults(handler=_m3_import_rlr_materials)

    m4 = commands.add_parser("m4", help="M4 named multi-source spatial-audio commands")
    m4_commands = m4.add_subparsers(dest="m4_command", required=True)

    m4_validate = m4_commands.add_parser("validate-request")
    m4_validate.add_argument("request")
    m4_validate.set_defaults(handler=_m4_validate_request)

    repository_root = Path(__file__).resolve().parents[2]
    m4_run = m4_commands.add_parser("run-canary")
    m4_run.add_argument("--request", required=True)
    m4_run.add_argument("--package-manifest", required=True)
    m4_run.add_argument(
        "--runtime-lock",
        default=str(repository_root / "locks" / "m4_runtime_v1.json"),
    )
    m4_run.add_argument(
        "--hrtf", default="/usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa"
    )
    m4_run.add_argument("--hrtf-license", default="/usr/share/doc/libmysofa1/copyright")
    m4_run.add_argument("--output", required=True)
    m4_run.add_argument(
        "--runtime-prefix",
        help=(
            "Explicit installed non-checkout Habitat prefix required with "
            "--runtime-mode current-installed"
        ),
    )
    m4_run.add_argument(
        "--rlr-sdk-root",
        help=(
            "Explicit external non-checkout RLRAudioPropagationPkg required with "
            "--runtime-mode current-installed"
        ),
    )
    m4_run.add_argument(
        "--runtime-mode",
        choices=("historical", "current-installed"),
        default="historical",
        help="historical verifies m4_runtime_v1; current-installed writes a fresh v2 receipt",
    )
    m4_run.add_argument(
        "--magnum-python-site",
        help="Explicit external Magnum Python site required by current-installed mode",
    )
    m4_run.add_argument(
        "--current-hrtf-sample-rate-hz",
        type=int,
        help="Explicit HRTF sample rate required by current-installed mode",
    )
    m4_run.add_argument(
        "--current-hrtf-license-id",
        help="Explicit HRTF license identifier required by current-installed mode",
    )
    m4_run.add_argument(
        "--current-hrtf-citation",
        help="Explicit HRTF citation required by current-installed mode",
    )
    m4_run.set_defaults(handler=_m4_run_canary)

    m4_verify = m4_commands.add_parser("verify-canary")
    m4_verify.add_argument("evidence")
    m4_verify.set_defaults(handler=_m4_verify_canary)

    m4_bundle = m4_commands.add_parser("verify-bundle")
    m4_bundle.add_argument("bundle")
    m4_bundle.set_defaults(handler=_m4_verify_bundle)

    m5 = commands.add_parser(
        "m5", help="M5 exact-timeline dynamic counterfactual commands"
    )
    m5_commands = m5.add_subparsers(dest="m5_command", required=True)

    m5_validate = m5_commands.add_parser("validate-request")
    m5_validate.add_argument("request")
    m5_validate.set_defaults(handler=_m5_validate_request)

    m5_run = m5_commands.add_parser("run-canary")
    m5_run.add_argument("--request", required=True)
    m5_run.add_argument("--animal-manifest", required=True)
    m5_run.add_argument("--m2-request", required=True)
    m5_run.add_argument("--room-manifest", required=True)
    m5_run.add_argument("--m1-request", required=True)
    m5_run.add_argument("--acoustic-package-manifest", required=True)
    m5_run.add_argument("--m4-request", required=True)
    m5_run.add_argument("--runtime-root")
    m5_run.add_argument(
        "--hrtf", default="/usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa"
    )
    m5_run.add_argument("--hrtf-license", default="/usr/share/doc/libmysofa1/copyright")
    m5_run.add_argument("--beagle-dry", required=True)
    m5_run.add_argument("--golden-dry", required=True)
    m5_run.add_argument(
        "--sensor-rig-trajectory",
        help="Optional SensorRigTrajectory v1 sidecar for a moving camera/listener",
    )
    m5_run.add_argument("--output", required=True)
    m5_run.set_defaults(handler=_m5_run_canary)

    m5_current_visual = m5_commands.add_parser(
        "capture-current-visual",
        help="Run a visual-only installed-prefix MP3D research capture",
    )
    m5_current_visual.add_argument("--animal-manifest", required=True)
    m5_current_visual.add_argument("--m2-request", required=True)
    m5_current_visual.add_argument("--room-manifest", required=True)
    m5_current_visual.add_argument("--m1-request", required=True)
    m5_current_visual.add_argument("--runtime-prefix", required=True)
    m5_current_visual.add_argument("--mp3d-root", required=True)
    m5_current_visual.add_argument("--magnum-python-site", required=True)
    m5_current_visual.add_argument("--output", required=True)
    m5_current_visual.set_defaults(handler=_m5_capture_current_visual)

    m5_current_route = m5_commands.add_parser(
        "author-current-mp3d-two-beagle-route",
        help="Author one fresh current-MP3D two-Beagle research route",
    )
    m5_current_route.add_argument("--source-animal-manifest", required=True)
    m5_current_route.add_argument("--source-m2-request", required=True)
    m5_current_route.add_argument("--runtime-prefix", required=True)
    m5_current_route.add_argument("--mp3d-root", required=True)
    m5_current_route.add_argument("--magnum-python-site", required=True)
    m5_current_route.add_argument("--output", required=True)
    m5_current_route.add_argument("--seed", type=int, default=20_260_820)
    m5_current_route.add_argument("--distance-tolerance-m", type=float, default=0.15)
    m5_current_route.add_argument(
        "--minimum-center-separation-m", type=float, default=0.75
    )
    m5_current_route.set_defaults(handler=_m5_author_current_mp3d_two_beagle_route)

    m5_verify = m5_commands.add_parser("verify-canary")
    m5_verify.add_argument("evidence")
    m5_verify.set_defaults(handler=_m5_verify_canary)

    m6 = commands.add_parser(
        "m6", help="M6 extensibility, room qualification, and feasibility canaries"
    )
    m6_commands = m6.add_subparsers(dest="m6_command", required=True)

    m6_validate = m6_commands.add_parser("validate-controlled-request")
    m6_validate.add_argument("request")
    m6_validate.set_defaults(handler=_m6_validate_controlled_request)

    m6_run = m6_commands.add_parser("run-controlled-canary")
    m6_run.add_argument("--request", required=True)
    m6_run.add_argument("--upstream-evidence", required=True)
    m6_run.add_argument("--output", required=True)
    m6_run.add_argument("--implementation-commit", required=True)
    m6_run.add_argument("--registry-directory")
    m6_run.add_argument("--room-registry")
    m6_run.add_argument("--room-qualification")
    m6_run.add_argument("--program")
    m6_run.add_argument("--ffmpeg", default="ffmpeg")
    m6_run.add_argument("--ffprobe", default="ffprobe")
    m6_run.set_defaults(handler=_m6_run_controlled)

    m6_verify = m6_commands.add_parser("verify-controlled-canary")
    m6_verify.add_argument("evidence")
    m6_verify.add_argument("--ffmpeg", default="ffmpeg")
    m6_verify.add_argument("--ffprobe", default="ffprobe")
    m6_verify.set_defaults(handler=_m6_verify_controlled)

    appearance = commands.add_parser(
        "appearance", help="Animal appearance contract/design commands"
    )
    appearance_commands = appearance.add_subparsers(
        dest="appearance_command", required=True
    )
    build_l9 = appearance_commands.add_parser(
        "build-l9", help="Build an immutable OA L9(3^4) appearance request batch"
    )
    build_l9.add_argument("--request", required=True)
    build_l9.add_argument("--source", required=True)
    build_l9.add_argument("--output", required=True)
    build_l9.set_defaults(handler=_build_appearance_l9)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        json.JSONDecodeError,
    ) as error:
        _print(
            {
                "status": "fail",
                "exception_type": type(error).__name__,
                "message": str(error),
            }
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
