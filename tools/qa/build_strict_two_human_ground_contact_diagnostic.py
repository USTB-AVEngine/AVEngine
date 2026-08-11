#!/usr/bin/env python3
"""Build a fail-closed f0/f37/f74 live foot-floor diagnostic request.

The request is CPU-only evidence.  It never launches SPEAR and deliberately
does not invent a contact tolerance before live bone and floor measurements
exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

FRAME_INDICES = [0, 37, 74]
FRAME_COUNT = 75
GROUND_CONTACT_BONES = {
    "left": {"foot": "Bip01 L Foot", "toe": "Bip01 L Toe0"},
    "right": {"foot": "Bip01 R Foot", "toe": "Bip01 R Toe0"},
}
REQUIRED_BONES = {
    bone_name for side in GROUND_CONTACT_BONES.values() for bone_name in side.values()
}
REQUEST_SCHEMA = "avengine_strict_two_human_ground_contact_diagnostic_request_v2"
MUTATION_SCHEMA = "avengine_strict_two_human_ground_contact_diagnostic_mutation_v2"
FLOOR_TRACE_IDENTITY_SCHEMA = "ue_fhitresult_component_owner_floor_identity_v2"
FLOOR_TRACE_IDENTITY_AUTHORITY = "OutHit.Component_to_UActorComponent.GetOwner"
FAILURE_LEDGER_SCHEMA = "avengine_strict_two_human_ground_contact_failure_ledger_v1"
FAILED_ATTEMPT_FIRST_BLOCKER = "Unreal result is missing actor"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "byte_size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _materialize_failure_ledger(
    *, final_receipt_path: Path, output: Path, new_capture_output: Path
) -> Path:
    _require(not output.exists(), f"refusing to overwrite failure ledger: {output}")
    final_receipt_path = final_receipt_path.resolve()
    final = _load(final_receipt_path)
    _require(
        final.get("schema")
        == "avengine_strict_two_human_ground_contact_gpu_launch_receipt_v1"
        and final.get("status") == "failed"
        and final.get("capture_process_exit_code") == 1
        and final.get("attempt_consumed") is True
        and final.get("gpu_started") is True
        and final.get("frame_indices") == FRAME_INDICES
        and final.get("release_authorized") is False
        and final.get("formal_dataset_count") == 0,
        "superseded ground attempt is not the closed failed attempt01",
    )
    attempt_root = final_receipt_path.parent
    request_path = attempt_root / "request.json"
    dry_receipt_path = attempt_root / "dry_run_receipt.json"
    running_receipt_path = attempt_root / "running_receipt.json"
    for owner, path in (
        ("failed request", request_path),
        ("failed dry receipt", dry_receipt_path),
        ("failed running receipt", running_receipt_path),
    ):
        _require(path.is_file(), f"{owner} is missing: {path}")
    failed_capture_output = Path(str(final.get("capture_output", ""))).resolve()
    _require(
        failed_capture_output.is_dir(),
        "failed attempt capture root is missing",
    )
    files = sorted(path for path in failed_capture_output.rglob("*") if path.is_file())
    _require(not files, "failed attempt unexpectedly produced capture evidence")
    _require(
        new_capture_output.resolve() != failed_capture_output,
        "revision_v2 must use a fresh capture candidate/output",
    )
    ledger = {
        "schema": FAILURE_LEDGER_SCHEMA,
        "status": "closed_failed_attempt_no_same_candidate_retry",
        "failed_attempt": {
            "request": _file_record(request_path),
            "dry_run_receipt": _file_record(dry_receipt_path),
            "running_receipt": _file_record(running_receipt_path),
            "final_receipt": _file_record(final_receipt_path),
            "required_repo_commit": final.get("required_repo_commit"),
            "capture_output": str(failed_capture_output),
            "capture_process_exit_code": 1,
            "attempt_consumed": True,
            "captured_file_count": 0,
            "live_trace_count": 0,
            "snap_measurement_count": 0,
            "pixel_frame_count": 0,
        },
        "first_blocker": {
            "stage": "f0_first_visual_root_floor_trace_identity_decode",
            "message": FAILED_ATTEMPT_FIRST_BLOCKER,
            "code_precondition": "required_OutHit.Actor",
            "machine_receipt_error": final.get("error"),
            "evidence_boundary": (
                "exact inner exception observed in attempt01 terminal traceback; "
                "immutable final receipt records the enclosing capture exit 1"
            ),
        },
        "disposition": {
            "same_candidate_retry_forbidden": True,
            "failed_attempt_preserved": True,
            "new_capture_output": str(new_capture_output.resolve()),
        },
        "revision_v2": {
            "floor_identity_schema": FLOOR_TRACE_IDENTITY_SCHEMA,
            "floor_identity_authority": FLOOR_TRACE_IDENTITY_AUTHORITY,
            "component_required": True,
            "owner_derived_via_get_owner": True,
            "raw_out_hit_shape_required": True,
            "legacy_actor_field_identity_authority": False,
            "hit_object_handle_identity_authority": False,
        },
        "release_authorized": False,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }
    _write(output, ledger)
    return output


def _glb_node_names(path: Path) -> list[str]:
    _require(path.is_file(), f"runtime GLB is missing: {path}")
    with path.open("rb") as stream:
        header = stream.read(12)
        _require(len(header) == 12, f"runtime GLB header is truncated: {path}")
        magic, version, total_length = struct.unpack("<4sII", header)
        _require(magic == b"glTF" and version == 2, f"not a GLB v2 file: {path}")
        chunk_header = stream.read(8)
        _require(
            len(chunk_header) == 8,
            f"runtime GLB JSON chunk header is truncated: {path}",
        )
        chunk_length, chunk_type = struct.unpack("<I4s", chunk_header)
        _require(chunk_type == b"JSON", f"first runtime GLB chunk is not JSON: {path}")
        document = json.loads(stream.read(chunk_length))
    _require(total_length == path.stat().st_size, f"runtime GLB length drift: {path}")
    nodes = document.get("nodes")
    _require(isinstance(nodes, list), f"runtime GLB nodes are missing: {path}")
    return [str(node["name"]) for node in nodes if isinstance(node.get("name"), str)]


def _runtime_authorities_for_actor(
    actor: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    expectation = actor.get("runtime_asset_expectation")
    _require(
        isinstance(expectation, Mapping),
        f"{actor.get('actor_id')} runtime asset expectation is missing",
    )
    import_manifest_path = expectation.get("ue_import_manifest")
    _require(
        isinstance(import_manifest_path, str) and bool(import_manifest_path),
        f"{actor.get('actor_id')} UE import manifest is missing",
    )
    manifest = _load(Path(import_manifest_path))
    source_glb = manifest.get("source_glb")
    _require(
        isinstance(source_glb, str) and bool(source_glb),
        f"{actor.get('actor_id')} UE import manifest lacks source_glb",
    )
    source_manifest = manifest.get("source_manifest")
    _require(
        isinstance(source_manifest, str) and bool(source_manifest),
        f"{actor.get('actor_id')} UE import manifest lacks source_manifest",
    )
    source_manifest_path = Path(source_manifest)
    normalization = _load(source_manifest_path)
    return Path(source_glb), source_manifest_path, normalization


def build_request(
    *,
    suite_plan: Path,
    scenario_id: str,
    audio_wav: Path,
    spear_root: Path,
    capture_output: Path,
    instrumented_suite_output: Path,
    output: Path,
    failed_attempt_final_receipt: Path,
    failure_ledger_output: Path,
    rpc_port: int,
    graphics_adapter: int,
) -> Path:
    _require(not output.exists(), f"refusing to overwrite request: {output}")
    _require(
        not instrumented_suite_output.exists(),
        f"refusing to overwrite instrumented suite: {instrumented_suite_output}",
    )
    suite = _load(suite_plan)
    scenarios = [
        scenario
        for scenario in suite.get("scenarios", [])
        if scenario.get("scenario_id") == scenario_id
    ]
    _require(len(scenarios) == 1, "diagnostic scenario must resolve exactly once")
    scenario = scenarios[0]
    plan = scenario.get("plan")
    _require(isinstance(plan, Mapping), "diagnostic scenario plan is missing")
    frames = plan.get("frames")
    actors = plan.get("actors")
    _require(
        isinstance(frames, list)
        and len(frames) == FRAME_COUNT
        and [frame.get("frame_index") for frame in frames] == list(range(FRAME_COUNT)),
        "diagnostic requires the exact full75 Timeline",
    )
    _require(
        isinstance(actors, list)
        and [actor.get("actor_id") for actor in actors]
        == ["source1_actor", "source2_actor"],
        "diagnostic requires the ordered strict two-human actor pair",
    )
    asset_evidence = []
    profile_mutations: dict[str, Any] = {}
    for actor in actors:
        _require(
            actor.get("body_plan_id") == "biped_human",
            f"{actor.get('actor_id')} is not a declared biped human",
        )
        runtime_glb, normalization_path, normalization = _runtime_authorities_for_actor(
            actor
        )
        node_names = _glb_node_names(runtime_glb)
        missing = sorted(REQUIRED_BONES - set(node_names))
        _require(
            not missing,
            f"{actor.get('actor_id')} runtime GLB lacks contact bones: {missing}",
        )
        expected_qa = normalization.get("expected_ue_qa", {})
        runtime_motion = normalization.get("runtime_motion_contract", {})
        maximum_abs_correction_cm = float(
            expected_qa.get("ground_snap_max_abs_correction_cm", -1.0)
        )
        residual_tolerance_cm = float(
            expected_qa.get("ground_snap_residual_tolerance_cm", -1.0)
        )
        _require(
            runtime_motion.get("dynamic_ground_snap_to_floor_required") is True
            and expected_qa.get("ground_snap_to_floor") is True
            and 0.0 < maximum_abs_correction_cm <= 15.0
            and 0.0 <= residual_tolerance_cm <= 0.1,
            f"{actor.get('actor_id')} normalization lacks dynamic ground snap",
        )
        actor_id = str(actor["actor_id"])
        asset_evidence.append(
            {
                "actor_id": actor_id,
                "asset_id": actor["asset_id"],
                "asset_revision": actor["asset_revision"],
                "runtime_glb": str(runtime_glb.resolve()),
                "normalization_manifest": str(normalization_path.resolve()),
                "joint_name_count": len(node_names),
                "required_contact_bones": sorted(REQUIRED_BONES),
                "required_contact_bones_present": True,
                "socket_claim": False,
                "dynamic_ground_snap_required": True,
                "maximum_abs_correction_cm": maximum_abs_correction_cm,
                "residual_tolerance_cm": residual_tolerance_cm,
            }
        )
        profile_mutations[actor_id] = {
            "schema": "avengine_strict_two_human_ground_contact_release_profile_v1",
            "status": "diagnostic_pending_not_release_qualified",
            "ue_length_unit": "centimeter",
            "bone_names": GROUND_CONTACT_BONES,
            "clearance_interval_authority": {
                "derived_from_live_diagnostic": False,
                "artifact": None,
            },
            "expected_floor_hit_actor": None,
            "expected_floor_hit_components": [],
            "support_anchor_clearance_interval_cm_by_action": None,
            "minimum_individual_anchor_clearance_cm": None,
            "minimum_floor_normal_z": None,
            "runtime_visual_ground_snap": {
                "schema": "ue_dynamic_ground_snap_v1",
                "target": "attached_visual_actor_root_component",
                "timeline_anchor_mutation_allowed": False,
                "emitter_or_rir_mutation_allowed": False,
                "maximum_abs_correction_cm": maximum_abs_correction_cm,
                "residual_tolerance_cm": residual_tolerance_cm,
                "normalization_manifest_authority": str(normalization_path.resolve()),
            },
        }
    failure_ledger_path = _materialize_failure_ledger(
        final_receipt_path=failed_attempt_final_receipt,
        output=failure_ledger_output,
        new_capture_output=capture_output,
    )
    instrumented_suite = deepcopy(suite)
    instrumented_scenario = next(
        item
        for item in instrumented_suite["scenarios"]
        if item["scenario_id"] == scenario_id
    )
    for actor in instrumented_scenario["plan"]["actors"]:
        actor["ground_contact_release_profile"] = profile_mutations[actor["actor_id"]]
    instrumented_suite["ground_contact_diagnostic_mutation"] = {
        "schema": MUTATION_SCHEMA,
        "status": "cpu_materialized_pending_one_sparse_capture",
        "source_suite_plan": str(suite_plan.resolve()),
        "failure_ledger": str(failure_ledger_path.resolve()),
        "timeline_actor_root_mutation": False,
        "emitter_or_rir_mutation": False,
        "visual_root_dynamic_ground_snap_only": True,
        "floor_trace_identity_schema": FLOOR_TRACE_IDENTITY_SCHEMA,
        "floor_trace_identity_authority": FLOOR_TRACE_IDENTITY_AUTHORITY,
        "raw_out_hit_shape_required": True,
        "legacy_actor_field_identity_authority": False,
        "hit_object_handle_identity_authority": False,
        "formal": False,
        "qualification_claim": False,
    }
    _require(audio_wav.is_file(), "diagnostic audio WAV is missing")
    capture_script = Path(__file__).with_name("capture_spear_native_pixel_episode.py")
    _require(capture_script.is_file(), "instrumented capture script is missing")
    _write(instrumented_suite_output, instrumented_suite)
    argv = [
        str(capture_script.resolve()),
        "--suite-plan",
        str(instrumented_suite_output.resolve()),
        "--scenario-id",
        scenario_id,
        "--audio-wav",
        str(audio_wav.resolve()),
        "--spear-root",
        str(spear_root.resolve()),
        "--output",
        str(capture_output.resolve()),
        "--rpc-port",
        str(rpc_port),
        "--graphics-adapter",
        str(graphics_adapter),
    ]
    for frame_index in FRAME_INDICES:
        argv.extend(["--frame-index", str(frame_index)])
    request = {
        "schema": REQUEST_SCHEMA,
        "status": "cpu_ready_not_authorized_for_execution",
        "scenario_id": scenario_id,
        "frame_indices": FRAME_INDICES,
        "sample_purpose": "begin_midpoint_end_live_foot_floor_measurement",
        "formal": False,
        "qualification_claim": False,
        "gpu_launch_authorized": False,
        "one_attempt_policy": {
            "maximum_attempts": 1,
            "same_candidate_retry_forbidden": True,
            "launch_requires_separate_authorization": True,
        },
        "measurement_contract": {
            "bone_names": GROUND_CONTACT_BONES,
            "bone_authority": "USkeletalMeshComponent.GetBoneTransform_RTS_World",
            "floor_authority": (
                "UKismetSystemLibrary.LineTraceSingleByProfile_"
                "BlockAll_complex_runtime_map"
            ),
            "floor_identity_schema": FLOOR_TRACE_IDENTITY_SCHEMA,
            "floor_identity_authority": FLOOR_TRACE_IDENTITY_AUTHORITY,
            "raw_out_hit_shape_required": True,
            "legacy_actor_field_identity_authority": False,
            "hit_object_handle_identity_authority": False,
            "actors_to_ignore": "both_runtime_anchor_and_visual_actors",
            "required_hit_fields": [
                "component",
                "location",
                "normal",
            ],
            "derived_identity_fields": [
                "hit_actor",
                "hit_actor_class",
                "hit_component",
                "hit_component_class",
            ],
            "ue_length_unit": "centimeter",
        },
        "threshold_policy": {
            "contact_clearance_interval_cm": None,
            "status": "must_be_derived_after_live_diagnostic",
            "actor_root_z_revision_cm": None,
            "bounds_only_release_forbidden": True,
            "plan_root_only_release_forbidden": True,
        },
        "asset_evidence": asset_evidence,
        "diagnostic_profile_mutations": profile_mutations,
        "capture_argv_without_python": argv,
        "artifacts": {
            "source_suite_plan": str(suite_plan.resolve()),
            "instrumented_suite_plan": str(instrumented_suite_output.resolve()),
            "audio_wav": str(audio_wav.resolve()),
            "spear_root": str(spear_root.resolve()),
            "capture_output": str(capture_output.resolve()),
            "failure_ledger": str(failure_ledger_path.resolve()),
            "supersedes_failed_final_receipt": str(
                failed_attempt_final_receipt.resolve()
            ),
        },
    }
    _write(output, request)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-plan", required=True, type=Path)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--audio-wav", required=True, type=Path)
    parser.add_argument("--spear-root", required=True, type=Path)
    parser.add_argument("--capture-output", required=True, type=Path)
    parser.add_argument("--instrumented-suite-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--failed-attempt-final-receipt", required=True, type=Path)
    parser.add_argument("--failure-ledger-output", required=True, type=Path)
    parser.add_argument("--rpc-port", type=int, default=39583)
    parser.add_argument("--graphics-adapter", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1024 <= args.rpc_port <= 65535:
        parser.error("--rpc-port must be in [1024,65535]")
    if args.graphics_adapter < 0:
        parser.error("--graphics-adapter must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path = build_request(
        suite_plan=args.suite_plan.resolve(),
        scenario_id=args.scenario_id,
        audio_wav=args.audio_wav.resolve(),
        spear_root=args.spear_root.resolve(),
        capture_output=args.capture_output.resolve(),
        instrumented_suite_output=args.instrumented_suite_output.resolve(),
        output=args.output.resolve(),
        failed_attempt_final_receipt=args.failed_attempt_final_receipt.resolve(),
        failure_ledger_output=args.failure_ledger_output.resolve(),
        rpc_port=args.rpc_port,
        graphics_adapter=args.graphics_adapter,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
