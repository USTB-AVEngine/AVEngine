#!/usr/bin/env python3
"""Prepare and launch one fail-closed f0/f37/f74 ground-contact diagnostic.

Preparation binds a clean repository HEAD and every CPU input.  A dry run checks
the physical-GPU1 and RPC-port gates without consuming the single attempt.  A real
launch additionally requires an explicit one-attempt authorization flag and a
previous immutable dry-run receipt.  This workflow never qualifies an Episode or
increments the formal dataset count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from avengine.qa.spear_unreal_capabilities import normalize_unreal_name

REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE_REQUEST_SCHEMA = "avengine_strict_two_human_ground_contact_diagnostic_request_v3"
REQUEST_SCHEMA = "avengine_strict_two_human_ground_contact_gpu_launch_request_v3"
RECEIPT_SCHEMA = "avengine_strict_two_human_ground_contact_gpu_launch_receipt_v3"
READBACK_SCHEMA = "avengine_native_live_ground_contact_readback_v1"
MUTATION_SCHEMA = "avengine_strict_two_human_ground_contact_diagnostic_mutation_v3"
FLOOR_TRACE_IDENTITY_SCHEMA = "ue_fhitresult_component_owner_floor_identity_v3"
FLOOR_TRACE_IDENTITY_AUTHORITY = (
    "UGameplayStatics.BreakHitResult(HitComponent)_to_UActorComponent.GetOwner"
)
BONE_NAME_RESOLUTION_SCHEMA = "avengine_spear_bone_name_resolution_v1"
BONE_NAME_NORMALIZATION = "unicode_casefold_then_alphanumeric_only"
FAILURE_LEDGER_SCHEMA = "avengine_strict_two_human_ground_contact_failure_ledger_v2"
PREVIOUS_FAILURE_LEDGER_SCHEMA = (
    "avengine_strict_two_human_ground_contact_failure_ledger_v1"
)
FAILED_V2_OUTER_ERROR = "RuntimeError: ground-contact capture exited 1"
FAILED_V2_AUDITED_INNER_EXCEPTION = (
    "source1_actor root-floor authority Component could not resolve to a UObject"
)
GROUND_HIT_RAW_JOURNAL_NAME = "ground_contact_raw_hit_journal.json"
GROUND_HIT_RAW_JOURNAL_SCHEMA = "avengine_ue_fhitresult_raw_component_journal_v1"
EPISODE_ID = "strict2h_dynamic_canary_04_camera_pan_both_static_v2"
GPU1_UUID = "GPU-6d3e273e-58c6-2a5b-480a-4816fef6c581"
CAPTURE_PYTHON_LOGICAL = Path("/data/jzy/miniconda3/envs/spear-env/bin/python")
FRAME_INDICES = [0, 37, 74]
ACTOR_IDS = ["source1_actor", "source2_actor"]
INSTANCE_IDS = ["source1", "source2"]
GROUND_BONES = {
    "left": {"foot": "Bip01 L Foot", "toe": "Bip01 L Toe0"},
    "right": {"foot": "Bip01 R Foot", "toe": "Bip01 R Toe0"},
}
FLAT_GROUND_BONES = [
    "Bip01 L Foot",
    "Bip01 L Toe0",
    "Bip01 R Foot",
    "Bip01 R Toe0",
]
SOURCE_ATTEMPT_POLICY = {
    "launch_requires_separate_authorization": True,
    "maximum_attempts": 1,
    "same_candidate_retry_forbidden": True,
}
ATTEMPT_POLICY = {
    "attempt_index": 1,
    "maximum_attempts_for_candidate": 1,
    "retry_same_candidate_forbidden": True,
    "failure_disposition": "reject_candidate_without_same_candidate_retry",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _file_record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "byte_size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _validate_file_record(record: Mapping[str, Any], *, owner: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    _require(path.is_file(), f"{owner} is missing: {path}")
    _require(_file_record(path) == dict(record), f"{owner} artifact binding drift")
    return path


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _tracked_status(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _nvidia_csv(query_kind: str, fields: str) -> list[list[str]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--query-{query_kind}={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return [
        [field.strip() for field in line.split(",")]
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def _gpu_snapshot() -> dict[str, Any]:
    gpus = _nvidia_csv("gpu", "index,uuid,name,memory.used,utilization.gpu")
    apps = _nvidia_csv("compute-apps", "gpu_uuid,pid,process_name,used_memory")
    return {
        "captured_at_utc": _utc_now(),
        "gpus": [
            {
                "physical_index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_used_mib": int(memory),
                "utilization_percent": int(utilization),
            }
            for index, uuid, name, memory, utilization in gpus
        ],
        "compute_apps": [
            {
                "gpu_uuid": uuid,
                "pid": int(pid),
                "process_name": name,
                "used_memory_mib": int(memory),
            }
            for uuid, pid, name, memory in apps
        ],
    }


def _validate_gpu1_idle(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    gpus = [
        item for item in snapshot.get("gpus", []) if item.get("physical_index") == 1
    ]
    _require(len(gpus) == 1, "physical GPU1 did not resolve exactly once")
    gpu = gpus[0]
    _require(gpu.get("uuid") == GPU1_UUID, "physical GPU1 UUID drift")
    apps = [
        item
        for item in snapshot.get("compute_apps", [])
        if item.get("gpu_uuid") == GPU1_UUID
    ]
    _require(not apps, f"physical GPU1 is not idle: {apps}")
    return dict(gpu)


def _assert_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def _is_authoritative_capture_python(path: Path) -> bool:
    return path.resolve() == CAPTURE_PYTHON_LOGICAL.resolve()


def _scenario(suite: Mapping[str, Any], scenario_id: str) -> Mapping[str, Any]:
    scenarios = suite.get("scenarios")
    _require(isinstance(scenarios, list), "suite scenarios are missing")
    matches = [item for item in scenarios if item.get("scenario_id") == scenario_id]
    _require(len(matches) == 1, "diagnostic scenario did not resolve exactly once")
    return matches[0]


def _expected_capture_argv_without_python(source: Mapping[str, Any]) -> list[str]:
    artifacts = source["artifacts"]
    return [
        str((REPOSITORY / "tools/qa/capture_spear_native_pixel_episode.py").resolve()),
        "--suite-plan",
        str(Path(artifacts["instrumented_suite_plan"]).resolve()),
        "--scenario-id",
        EPISODE_ID,
        "--audio-wav",
        str(Path(artifacts["audio_wav"]).resolve()),
        "--spear-root",
        str(Path(artifacts["spear_root"]).resolve()),
        "--output",
        str(Path(artifacts["capture_output"]).resolve()),
        "--rpc-port",
        str(
            source["capture_argv_without_python"][
                source["capture_argv_without_python"].index("--rpc-port") + 1
            ]
        ),
        "--graphics-adapter",
        "1",
        "--frame-index",
        "0",
        "--frame-index",
        "37",
        "--frame-index",
        "74",
    ]


def _validate_source_cpu_request(
    source_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    source_path = source_path.resolve()
    source = _load(source_path)
    _require(
        source.get("schema") == SOURCE_REQUEST_SCHEMA, "source request schema drift"
    )
    _require(
        source.get("status") == "cpu_ready_not_authorized_for_execution"
        and source.get("scenario_id") == EPISODE_ID
        and source.get("frame_indices") == FRAME_INDICES,
        "source diagnostic identity/frame closure drift",
    )
    _require(
        source.get("gpu_launch_authorized") is False
        and source.get("formal") is False
        and source.get("qualification_claim") is False,
        "source diagnostic crossed its CPU-only/formal boundary",
    )
    _require(
        source.get("one_attempt_policy") == SOURCE_ATTEMPT_POLICY,
        "source one-attempt/separate-authorization policy drift",
    )
    _require(
        source.get("sample_purpose")
        == "begin_midpoint_end_live_foot_floor_measurement",
        "source diagnostic purpose drift",
    )

    artifacts = source.get("artifacts")
    _require(isinstance(artifacts, Mapping), "source diagnostic artifacts are missing")
    diagnostic_root = source_path.parent
    try:
        diagnostic_root.relative_to((REPOSITORY / "tmp").resolve())
    except ValueError as exc:
        raise RuntimeError("diagnostic root must stay inside repository tmp") from exc
    _require(
        source_path == diagnostic_root / "request.json", "source request path drift"
    )
    capture_output = Path(str(artifacts.get("capture_output", ""))).resolve()
    _require(
        capture_output == diagnostic_root / "capture_attempt_01",
        "diagnostic capture output is not attempt 01",
    )
    _require(not capture_output.exists(), "diagnostic capture output must be new")
    instrumented_path = Path(
        str(artifacts.get("instrumented_suite_plan", ""))
    ).resolve()
    source_suite_path = Path(str(artifacts.get("source_suite_plan", ""))).resolve()
    audio_path = Path(str(artifacts.get("audio_wav", ""))).resolve()
    spear_root = Path(str(artifacts.get("spear_root", ""))).resolve()
    failure_ledger_path = Path(str(artifacts.get("failure_ledger", ""))).resolve()
    frozen_failure_ledger_path = Path(
        str(artifacts.get("frozen_revision_v2_failure_ledger", ""))
    ).resolve()
    failed_final_receipt_path = Path(
        str(artifacts.get("supersedes_failed_final_receipt", ""))
    ).resolve()
    for owner, path in (
        ("instrumented suite", instrumented_path),
        ("source suite", source_suite_path),
        ("audio WAV", audio_path),
        ("failure ledger", failure_ledger_path),
        ("frozen revision_v2 failure ledger", frozen_failure_ledger_path),
        ("superseded final receipt", failed_final_receipt_path),
    ):
        _require(path.is_file(), f"{owner} is missing: {path}")
    _require(spear_root.is_dir(), f"SPEAR root is missing: {spear_root}")

    source_suite = _load(source_suite_path)
    instrumented_suite = _load(instrumented_path)
    source_scenario = _scenario(source_suite, EPISODE_ID)
    instrumented_scenario = _scenario(instrumented_suite, EPISODE_ID)
    source_plan = source_scenario.get("plan", {})
    instrumented_plan = instrumented_scenario.get("plan", {})
    _require(
        source_plan.get("frames") == instrumented_plan.get("frames")
        and len(instrumented_plan.get("frames", [])) == 75,
        "instrumentation changed Timeline/acoustic frame anchors",
    )
    source_actors = source_plan.get("actors")
    instrumented_actors = instrumented_plan.get("actors")
    _require(
        isinstance(source_actors, list)
        and isinstance(instrumented_actors, list)
        and [actor.get("actor_id") for actor in source_actors] == ACTOR_IDS
        and [actor.get("actor_id") for actor in instrumented_actors] == ACTOR_IDS,
        "instrumented actor closure drift",
    )
    for original, mutated in zip(source_actors, instrumented_actors, strict=True):
        without_profile = dict(mutated)
        without_profile.pop("ground_contact_release_profile", None)
        _require(
            without_profile == original,
            f"instrumentation changed non-visual actor declaration: {original['actor_id']}",
        )
    mutation = instrumented_suite.get("ground_contact_diagnostic_mutation", {})
    _require(
        mutation.get("schema") == MUTATION_SCHEMA
        and mutation.get("status") == "cpu_materialized_pending_one_sparse_capture"
        and mutation.get("visual_root_dynamic_ground_snap_only") is True
        and mutation.get("timeline_actor_root_mutation") is False
        and mutation.get("emitter_or_rir_mutation") is False
        and mutation.get("floor_trace_identity_schema") == FLOOR_TRACE_IDENTITY_SCHEMA
        and mutation.get("floor_trace_identity_authority")
        == FLOOR_TRACE_IDENTITY_AUTHORITY
        and mutation.get("raw_out_hit_shape_required") is True
        and mutation.get("raw_component_key_type_literal_journal_required") is True
        and mutation.get("raw_component_non_handle_string_required") is True
        and mutation.get("break_hit_result_required") is True
        and mutation.get("legacy_actor_field_identity_authority") is False
        and mutation.get("hit_object_handle_identity_authority") is False
        and Path(str(mutation.get("failure_ledger", ""))).resolve()
        == failure_ledger_path
        and mutation.get("qualification_claim") is False
        and mutation.get("formal") is False,
        "instrumented suite visual-only mutation contract drift",
    )

    failure_ledger = _load(failure_ledger_path)
    failed_attempt = failure_ledger.get("failed_revision_v2_attempt", {})
    first_blocker = failure_ledger.get("first_blocker", {})
    disposition = failure_ledger.get("disposition", {})
    revision = failure_ledger.get("revision_v3", {})
    frozen_previous = failure_ledger.get("frozen_revision_v2_failure_ledger", {})
    _require(
        failure_ledger.get("schema") == FAILURE_LEDGER_SCHEMA
        and failure_ledger.get("status")
        == "closed_two_failed_attempts_no_same_candidate_retry"
        and failed_attempt.get("capture_process_exit_code") == 1
        and failed_attempt.get("attempt_consumed") is True
        and failed_attempt.get("captured_file_count") == 0
        and failed_attempt.get("live_trace_count") == 0
        and failed_attempt.get("snap_measurement_count") == 0
        and failed_attempt.get("pixel_frame_count") == 0
        and first_blocker.get("machine_receipt_message") == FAILED_V2_OUTER_ERROR
        and first_blocker.get("audited_inner_exception")
        == FAILED_V2_AUDITED_INNER_EXCEPTION
        and first_blocker.get("audited_inner_exception_machine_persisted") is False
        and first_blocker.get("code_precondition")
        == "treated_raw_OutHit.Component_string_as_0x_handle"
        and disposition.get("same_candidate_retry_forbidden") is True
        and disposition.get("failed_attempt_preserved") is True
        and Path(str(disposition.get("new_capture_output", ""))).resolve()
        == capture_output
        and revision.get("floor_identity_schema") == FLOOR_TRACE_IDENTITY_SCHEMA
        and revision.get("floor_identity_authority") == FLOOR_TRACE_IDENTITY_AUTHORITY
        and revision.get("raw_component_required") is True
        and revision.get("raw_component_non_handle_string_required") is True
        and revision.get("raw_component_key_type_literal_journal_required") is True
        and revision.get("raw_component_identity_authority") is False
        and revision.get("break_hit_result_required") is True
        and revision.get("break_hit_result_hit_component_handle_required") is True
        and revision.get("owner_derived_via_get_owner") is True
        and revision.get("raw_out_hit_shape_required") is True
        and revision.get("legacy_actor_field_identity_authority") is False
        and revision.get("hit_object_handle_identity_authority") is False
        and failure_ledger.get("release_authorized") is False
        and failure_ledger.get("qualification_claim") is False
        and failure_ledger.get("formal_dataset_count") == 0,
        "failed-attempt ledger or revision_v3 disposition drift",
    )
    frozen_previous_path = _validate_file_record(
        frozen_previous,
        owner="frozen revision_v2 failure ledger",
    )
    expected_frozen_previous = Path(
        str(artifacts.get("frozen_revision_v2_failure_ledger", ""))
    ).resolve()
    _require(
        frozen_previous_path == expected_frozen_previous,
        "frozen revision_v2 failure-ledger path drift",
    )
    previous_failure_ledger = _load(frozen_previous_path)
    failed_capture_output = Path(
        str(failed_attempt.get("capture_output", ""))
    ).resolve()
    _require(
        frozen_previous_path
        == failed_final_receipt_path.parent.parent / "failure_ledger.json"
        and previous_failure_ledger.get("schema") == PREVIOUS_FAILURE_LEDGER_SCHEMA
        and previous_failure_ledger.get("status")
        == "closed_failed_attempt_no_same_candidate_retry",
        "frozen revision_v2 failure ledger content drift",
    )
    _require(
        Path(
            str(
                previous_failure_ledger.get("disposition", {}).get(
                    "new_capture_output", ""
                )
            )
        ).resolve()
        == failed_capture_output,
        "frozen revision_v2 ledger does not bind the failed revision_v2 capture",
    )
    observed_failed_final = _validate_file_record(
        failed_attempt.get("final_receipt", {}),
        owner="superseded final receipt",
    )
    _require(
        observed_failed_final == failed_final_receipt_path,
        "superseded final receipt path drift",
    )

    assets = source.get("asset_evidence")
    _require(
        isinstance(assets, list)
        and [asset.get("actor_id") for asset in assets] == ACTOR_IDS,
        "two-actor asset evidence closure drift",
    )
    for asset in assets:
        actor_id = str(asset["actor_id"])
        _require(
            asset.get("required_contact_bones") == FLAT_GROUND_BONES
            and asset.get("required_contact_bones_present") is True
            and asset.get("dynamic_ground_snap_required") is True
            and asset.get("socket_claim") is False,
            f"{actor_id} foot-bone evidence drift",
        )
        _require(
            float(asset.get("maximum_abs_correction_cm", 16.0)) <= 15.0
            and float(asset.get("maximum_abs_correction_cm", 0.0)) > 0.0
            and 0.0 <= float(asset.get("residual_tolerance_cm", 1.0)) <= 0.1,
            f"{actor_id} normalization snap limits drift",
        )
        _require(
            Path(str(asset.get("runtime_glb", ""))).is_file()
            and Path(str(asset.get("normalization_manifest", ""))).is_file(),
            f"{actor_id} runtime GLB/normalization evidence is missing",
        )

    profiles = source.get("diagnostic_profile_mutations")
    _require(
        isinstance(profiles, Mapping) and list(profiles) == ACTOR_IDS,
        "diagnostic profile closure drift",
    )
    for actor_id, profile in profiles.items():
        snap = profile.get("runtime_visual_ground_snap", {})
        _require(
            profile.get("status") == "diagnostic_pending_not_release_qualified"
            and profile.get("bone_names") == GROUND_BONES
            and profile.get("ue_length_unit") == "centimeter"
            and profile.get("support_anchor_clearance_interval_cm_by_action") is None
            and profile.get("minimum_individual_anchor_clearance_cm") is None
            and profile.get("minimum_floor_normal_z") is None,
            f"{actor_id} diagnostic profile invented release thresholds",
        )
        _require(
            snap.get("target") == "attached_visual_actor_root_component"
            and snap.get("timeline_anchor_mutation_allowed") is False
            and snap.get("emitter_or_rir_mutation_allowed") is False
            and float(snap.get("maximum_abs_correction_cm", 16.0)) <= 15.0
            and float(snap.get("residual_tolerance_cm", 1.0)) <= 0.1,
            f"{actor_id} visual-root snap contract drift",
        )
    measurement = source.get("measurement_contract", {})
    _require(
        measurement.get("bone_authority")
        == "USkeletalMeshComponent.GetBoneTransform_RTS_World"
        and measurement.get("floor_authority")
        == "UKismetSystemLibrary.LineTraceSingleByProfile_BlockAll_complex_runtime_map"
        and measurement.get("actors_to_ignore")
        == "both_runtime_anchor_and_visual_actors"
        and measurement.get("bone_names") == GROUND_BONES
        and measurement.get("required_hit_fields")
        == ["component", "location", "normal"]
        and measurement.get("derived_identity_fields")
        == ["hit_actor", "hit_actor_class", "hit_component", "hit_component_class"]
        and measurement.get("floor_identity_schema") == FLOOR_TRACE_IDENTITY_SCHEMA
        and measurement.get("floor_identity_authority")
        == FLOOR_TRACE_IDENTITY_AUTHORITY
        and measurement.get("raw_out_hit_shape_required") is True
        and measurement.get("raw_component_key_type_literal_journal_required") is True
        and measurement.get("raw_component_non_handle_string_required") is True
        and measurement.get("raw_component_identity_authority") is False
        and measurement.get("break_hit_result_required") is True
        and measurement.get("break_hit_result_hit_component_handle_required") is True
        and measurement.get("legacy_actor_field_identity_authority") is False
        and measurement.get("hit_object_handle_identity_authority") is False
        and measurement.get("ue_length_unit") == "centimeter",
        "live foot/floor measurement contract drift",
    )
    threshold = source.get("threshold_policy", {})
    _require(
        threshold.get("status") == "must_be_derived_after_live_diagnostic"
        and threshold.get("actor_root_z_revision_cm") is None
        and threshold.get("contact_clearance_interval_cm") is None
        and threshold.get("bounds_only_release_forbidden") is True
        and threshold.get("plan_root_only_release_forbidden") is True,
        "source request guessed a release threshold",
    )

    argv = source.get("capture_argv_without_python")
    _require(
        isinstance(argv, list) and all(isinstance(value, str) for value in argv),
        "source capture argv is missing",
    )
    _require(
        argv == _expected_capture_argv_without_python(source), "capture argv drift"
    )
    return source, argv


def _artifact_paths(source_path: Path, source: Mapping[str, Any]) -> dict[str, Path]:
    artifacts = source["artifacts"]
    paths = {
        "source_request": source_path.resolve(),
        "source_suite_plan": Path(artifacts["source_suite_plan"]).resolve(),
        "instrumented_suite_plan": Path(artifacts["instrumented_suite_plan"]).resolve(),
        "audio_wav": Path(artifacts["audio_wav"]).resolve(),
        "capture_script": (
            REPOSITORY / "tools/qa/capture_spear_native_pixel_episode.py"
        ).resolve(),
        "failure_ledger": Path(artifacts["failure_ledger"]).resolve(),
        "frozen_revision_v2_failure_ledger": Path(
            artifacts["frozen_revision_v2_failure_ledger"]
        ).resolve(),
        "superseded_final_receipt": Path(
            artifacts["supersedes_failed_final_receipt"]
        ).resolve(),
    }
    for asset in source["asset_evidence"]:
        actor_id = str(asset["actor_id"])
        paths[f"{actor_id}_runtime_glb"] = Path(asset["runtime_glb"]).resolve()
        paths[f"{actor_id}_normalization_manifest"] = Path(
            asset["normalization_manifest"]
        ).resolve()
    return paths


def prepare_request(*, source_request: Path, capture_python: Path) -> Path:
    source_path = source_request.resolve()
    source, _ = _validate_source_cpu_request(source_path)
    _require(not _tracked_status(REPOSITORY), "tracked worktree must be clean")
    _require(capture_python.is_file(), "authoritative SPEAR Python is missing")
    _require(
        _is_authoritative_capture_python(capture_python),
        "capture Python is not the pinned SPEAR runtime",
    )
    diagnostic_root = source_path.parent
    attempt_root = diagnostic_root / "gpu_launch_attempt_01"
    capture_output = Path(source["artifacts"]["capture_output"]).resolve()
    _require(
        not attempt_root.exists(), "ground-contact launch attempt 01 already exists"
    )
    _require(
        not capture_output.exists(), "ground-contact capture output already exists"
    )
    artifacts = _artifact_paths(source_path, source)
    artifact_records = {name: _file_record(path) for name, path in artifacts.items()}
    request = {
        "schema": REQUEST_SCHEMA,
        "status": "prepared_not_launched",
        "episode_id": EPISODE_ID,
        "required_repo_commit": _git_head(REPOSITORY),
        "required_clean_tracked_worktree": True,
        "repo_root": str(REPOSITORY.resolve()),
        "diagnostic_root": str(diagnostic_root),
        "attempt_root": str(attempt_root),
        "source_request": str(source_path),
        "capture_output": str(capture_output),
        "capture_python": str(capture_python.resolve()),
        "spear_root": str(Path(source["artifacts"]["spear_root"]).resolve()),
        "artifact_records": artifact_records,
        "attempt_policy": ATTEMPT_POLICY,
        "frame_indices": FRAME_INDICES,
        "full75_allowed": False,
        "physical_gpu_index": 1,
        "physical_gpu_uuid": GPU1_UUID,
        "graphics_adapter_argument": 1,
        "forbidden_physical_gpu_indices": [0, 3],
        "required_idle_compute_process_count": 0,
        "real_launch_requires_flag": "--authorize-one-attempt",
        "source_gpu_launch_authorized": False,
        "manual_pixel_ground_contact_review_required": True,
        "release_authorized": False,
        "qualification_claim": False,
        "formal_dataset_count": 0,
        "created_at_utc": _utc_now(),
    }
    attempt_root.mkdir(parents=True)
    request_path = attempt_root / "request.json"
    _write_json_exclusive(request_path, request)
    return request_path


def _capture_argv(request: Mapping[str, Any], source: Mapping[str, Any]) -> list[str]:
    return [str(request["capture_python"]), *source["capture_argv_without_python"]]


def _validate_request(request_path: Path) -> tuple[dict[str, Any], list[str]]:
    request_path = request_path.resolve()
    request = _load(request_path)
    _require(request.get("schema") == REQUEST_SCHEMA, "launch request schema drift")
    _require(
        request.get("status") == "prepared_not_launched"
        and request.get("episode_id") == EPISODE_ID,
        "launch request identity drift",
    )
    repo_root = Path(str(request.get("repo_root", ""))).resolve()
    _require(repo_root == REPOSITORY.resolve(), "launch request repository drift")
    _require(
        request.get("required_repo_commit") == _git_head(repo_root),
        "repository HEAD differs from request-bound commit",
    )
    _require(not _tracked_status(repo_root), "tracked worktree is no longer clean")
    diagnostic_root = Path(str(request.get("diagnostic_root", ""))).resolve()
    attempt_root = diagnostic_root / "gpu_launch_attempt_01"
    _require(
        request_path == attempt_root / "request.json"
        and Path(str(request.get("attempt_root", ""))).resolve() == attempt_root,
        "request is not bound to ground-contact attempt 01",
    )
    source_path = Path(str(request.get("source_request", ""))).resolve()
    _require(
        source_path == diagnostic_root / "request.json", "source request path drift"
    )
    source, _ = _validate_source_cpu_request(source_path)
    capture_output = Path(str(request.get("capture_output", ""))).resolve()
    _require(
        capture_output == diagnostic_root / "capture_attempt_01"
        and not capture_output.exists(),
        "diagnostic capture output must be the new attempt-01 path",
    )
    _require(
        request.get("attempt_policy") == ATTEMPT_POLICY
        and request.get("frame_indices") == FRAME_INDICES
        and request.get("full75_allowed") is False,
        "one-attempt three-frame policy drift",
    )
    _require(
        request.get("physical_gpu_index") == 1
        and request.get("physical_gpu_uuid") == GPU1_UUID
        and request.get("graphics_adapter_argument") == 1
        and request.get("forbidden_physical_gpu_indices") == [0, 3]
        and request.get("required_idle_compute_process_count") == 0,
        "physical GPU1/adapter1 binding drift",
    )
    _require(
        request.get("real_launch_requires_flag") == "--authorize-one-attempt"
        and request.get("source_gpu_launch_authorized") is False
        and request.get("manual_pixel_ground_contact_review_required") is True
        and request.get("release_authorized") is False
        and request.get("qualification_claim") is False
        and request.get("formal_dataset_count") == 0,
        "diagnostic authorization/formal boundary drift",
    )
    _require(
        _is_authoritative_capture_python(Path(request["capture_python"])),
        "capture Python is not the pinned SPEAR runtime",
    )
    for key in ("capture_python", "spear_root"):
        _require(Path(request[key]).exists(), f"missing runtime input: {key}")
    expected_artifacts = _artifact_paths(source_path, source)
    records = request.get("artifact_records")
    _require(
        isinstance(records, Mapping) and set(records) == set(expected_artifacts),
        "launch artifact-record closure drift",
    )
    for name, expected_path in expected_artifacts.items():
        observed = _validate_file_record(records[name], owner=name)
        _require(observed == expected_path, f"{name} path drift")
    argv = _capture_argv(request, source)
    _require(
        argv.count("--frame-index") == 3
        and [
            int(argv[index + 1])
            for index, value in enumerate(argv)
            if value == "--frame-index"
        ]
        == FRAME_INDICES,
        "capture must select exactly f0/f37/f74",
    )
    _require(
        argv.count("--graphics-adapter") == 1
        and argv[argv.index("--graphics-adapter") + 1] == "1",
        "capture must use graphics adapter 1",
    )
    return request, argv


def _finite_xyz(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in value
        )
    )


def _nonzero_hex_handle(value: object) -> bool:
    if not isinstance(value, str) or not value.strip().lower().startswith("0x"):
        return False
    try:
        return int(value, 16) != 0
    except ValueError:
        return False


def _validate_bone_name_resolution(
    value: object,
    *,
    expected_owner: str,
    owner: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    _require(isinstance(value, Mapping), f"{owner} bone-name resolution is missing")
    resolution = value
    bone_count = resolution.get("bone_count")
    inventory = resolution.get("inventory")
    resolutions = resolution.get("resolutions")
    _require(
        resolution.get("schema") == BONE_NAME_RESOLUTION_SCHEMA
        and resolution.get("status") == "pass"
        and resolution.get("owner") == expected_owner
        and resolution.get("normalization") == BONE_NAME_NORMALIZATION
        and isinstance(bone_count, int)
        and not isinstance(bone_count, bool)
        and bone_count > 0
        and isinstance(inventory, list)
        and len(inventory) == bone_count
        and isinstance(resolutions, list),
        f"{owner} bone-name resolution schema/owner/inventory header failed",
    )

    inventory_records: list[dict[str, Any]] = []
    for expected_index, item in enumerate(inventory):
        _require(
            isinstance(item, Mapping),
            f"{owner} bone inventory entry {expected_index} is invalid",
        )
        actual_name = item.get("actual_name")
        normalized_name = item.get("normalized_name")
        inventory_index = item.get("inventory_index")
        _require(
            isinstance(inventory_index, int)
            and not isinstance(inventory_index, bool)
            and inventory_index == expected_index
            and isinstance(actual_name, str)
            and bool(actual_name)
            and isinstance(normalized_name, str)
            and normalized_name == normalize_unreal_name(actual_name),
            f"{owner} bone inventory entry {expected_index} name/index failed",
        )
        inventory_records.append(
            {
                "inventory_index": inventory_index,
                "actual_name": actual_name,
                "normalized_name": normalized_name,
            }
        )
    _require(
        len({item["actual_name"] for item in inventory_records}) == bone_count,
        f"{owner} live bone inventory contains duplicate actual names",
    )

    _require(
        len(resolutions) == len(FLAT_GROUND_BONES),
        f"{owner} must resolve exactly four requested contact bones",
    )
    by_requested: dict[str, Mapping[str, Any]] = {}
    stable_resolutions: list[dict[str, Any]] = []
    for expected_name, item in zip(FLAT_GROUND_BONES, resolutions, strict=True):
        _require(
            isinstance(item, Mapping),
            f"{owner} resolution for {expected_name!r} is invalid",
        )
        requested_name = item.get("requested_name")
        requested_normalized = item.get("requested_normalized_name")
        actual_name = item.get("actual_live_name")
        actual_normalized = item.get("actual_normalized_name")
        inventory_index = item.get("inventory_index")
        requested_probe = item.get("requested_probe_index")
        actual_probe = item.get("actual_probe_index")
        mode = item.get("resolution_mode")
        _require(
            requested_name == expected_name
            and requested_normalized == normalize_unreal_name(expected_name)
            and isinstance(actual_name, str)
            and bool(actual_name)
            and actual_normalized == normalize_unreal_name(actual_name)
            and requested_normalized == actual_normalized
            and isinstance(inventory_index, int)
            and not isinstance(inventory_index, bool)
            and 0 <= inventory_index < bone_count
            and isinstance(requested_probe, int)
            and not isinstance(requested_probe, bool)
            and requested_probe in (-1, inventory_index)
            and isinstance(actual_probe, int)
            and not isinstance(actual_probe, bool)
            and actual_probe == inventory_index,
            f"{owner} requested/actual/index/probe closure failed for {expected_name!r}",
        )
        inventory_match = inventory_records[inventory_index]
        normalized_matches = [
            candidate
            for candidate in inventory_records
            if candidate["normalized_name"] == requested_normalized
        ]
        _require(
            len(normalized_matches) == 1
            and inventory_match == normalized_matches[0]
            and inventory_match["actual_name"] == actual_name,
            f"{owner} actual live name does not close against a unique inventory entry",
        )
        mode_is_closed = bool(
            (
                mode == "direct_exact_fname"
                and requested_name == actual_name
                and requested_probe == inventory_index
            )
            or (
                mode == "requested_alias_fname_accepted"
                and requested_name != actual_name
                and requested_probe == inventory_index
            )
            or (
                mode == "sanitized_live_fname_required"
                and requested_name != actual_name
                and requested_probe == -1
            )
        )
        _require(
            mode_is_closed,
            f"{owner} resolution probe/mode semantics failed for {expected_name!r}",
        )
        _require(
            requested_name not in by_requested,
            f"{owner} repeats requested bone {requested_name!r}",
        )
        by_requested[str(requested_name)] = item
        stable_resolutions.append(dict(item))

    return by_requested, {
        "schema": resolution["schema"],
        "status": resolution["status"],
        "owner": resolution["owner"],
        "normalization": resolution["normalization"],
        "bone_count": bone_count,
        "inventory": inventory_records,
        "resolutions": stable_resolutions,
    }


def _validate_floor_trace_identity(trace: Mapping[str, Any], *, owner: str) -> None:
    shape = trace.get("raw_out_hit_shape")
    keys = shape.get("keys") if isinstance(shape, Mapping) else None
    value_types = shape.get("value_types") if isinstance(shape, Mapping) else None
    raw_component = trace.get("raw_component_diagnostic")
    break_component = trace.get("break_hit_result_component")
    break_actor = trace.get("break_hit_result_actor_auxiliary")
    raw_literal = (
        raw_component.get("literal") if isinstance(raw_component, Mapping) else None
    )
    break_literal = (
        break_component.get("literal") if isinstance(break_component, Mapping) else None
    )
    _require(
        trace.get("schema") == FLOOR_TRACE_IDENTITY_SCHEMA
        and trace.get("authority") == FLOOR_TRACE_IDENTITY_AUTHORITY
        and trace.get("hit_actor") not in (None, "")
        and trace.get("hit_actor_class") not in (None, "")
        and trace.get("hit_component") not in (None, "")
        and trace.get("hit_component_class") not in (None, "")
        and isinstance(keys, list)
        and any(str(key).lower() == "component" for key in keys)
        and isinstance(value_types, Mapping)
        and set(value_types) == set(keys),
        f"{owner} BreakHitResult/Component/GetOwner identity or raw shape failed",
    )
    _require(
        isinstance(trace.get("raw_hit_journal_sequence"), int)
        and int(trace["raw_hit_journal_sequence"]) >= 0
        and isinstance(raw_component, Mapping)
        and raw_component.get("present") is True
        and str(raw_component.get("key", "")).lower() == "component"
        and raw_component.get("python_type") == "builtins.str"
        and isinstance(raw_literal, str)
        and bool(raw_literal.strip())
        and not raw_literal.strip().lower().startswith("0x")
        and raw_component.get("literal_persisted_exactly") is True
        and raw_component.get("identity_authority") is False
        and isinstance(break_component, Mapping)
        and break_component.get("present") is True
        and str(break_component.get("key", "")).lower() == "hitcomponent"
        and break_component.get("python_type") == "builtins.str"
        and _nonzero_hex_handle(break_literal)
        and break_component.get("literal_persisted_exactly") is True
        and break_component.get("identity_authority") is True
        and isinstance(break_actor, Mapping)
        and break_actor.get("identity_authority") is False,
        f"{owner} raw Component or BreakHitResult HitComponent evidence failed",
    )
    raw_actor = trace.get("raw_actor_field")
    hit_object = trace.get("hit_object_handle_auxiliary")
    _require(
        isinstance(raw_actor, Mapping)
        and raw_actor.get("identity_authority") is False
        and isinstance(hit_object, Mapping)
        and hit_object.get("identity_authority") is False,
        f"{owner} treated Actor/HitObjectHandle as floor identity authority",
    )


def _validate_trace_journal_binding(
    trace: Mapping[str, Any],
    *,
    entries: Mapping[int, Mapping[str, Any]],
    owner: str,
) -> None:
    sequence = int(trace["raw_hit_journal_sequence"])
    _require(sequence in entries, f"{owner} raw-hit journal sequence is missing")
    entry = entries[sequence]
    stable_identity = entry.get("stable_identity")
    break_result = entry.get("break_hit_result")
    _require(
        entry.get("raw_component") == trace.get("raw_component_diagnostic")
        and isinstance(break_result, Mapping)
        and break_result.get("hit_component") == trace.get("break_hit_result_component")
        and break_result.get("hit_actor_auxiliary")
        == trace.get("break_hit_result_actor_auxiliary")
        and isinstance(stable_identity, Mapping)
        and stable_identity.get("authority") == FLOOR_TRACE_IDENTITY_AUTHORITY
        and stable_identity.get("hit_actor") == trace.get("hit_actor")
        and stable_identity.get("hit_actor_class") == trace.get("hit_actor_class")
        and stable_identity.get("hit_component") == trace.get("hit_component")
        and stable_identity.get("hit_component_class")
        == trace.get("hit_component_class"),
        f"{owner} trace differs from pre-conversion raw-hit journal",
    )


def _raw_hit_journal_entry_is_closed(
    entry: Mapping[str, Any], *, sequence: int
) -> bool:
    raw_component = entry.get("raw_component")
    break_result = entry.get("break_hit_result")
    stable_identity = entry.get("stable_identity")
    if not (
        entry.get("sequence") == sequence
        and isinstance(raw_component, Mapping)
        and isinstance(break_result, Mapping)
        and isinstance(stable_identity, Mapping)
    ):
        return False
    raw_literal = raw_component.get("literal")
    break_component = break_result.get("hit_component")
    break_actor = break_result.get("hit_actor_auxiliary")
    return bool(
        raw_component.get("present") is True
        and str(raw_component.get("key", "")).lower() == "component"
        and raw_component.get("python_type") == "builtins.str"
        and isinstance(raw_literal, str)
        and raw_literal.strip()
        and not raw_literal.strip().lower().startswith("0x")
        and raw_component.get("literal_persisted_exactly") is True
        and raw_component.get("identity_authority") is False
        and break_result.get("method") == "UGameplayStatics.BreakHitResult"
        and isinstance(break_component, Mapping)
        and break_component.get("present") is True
        and str(break_component.get("key", "")).lower() == "hitcomponent"
        and break_component.get("python_type") == "builtins.str"
        and _nonzero_hex_handle(break_component.get("literal"))
        and break_component.get("literal_persisted_exactly") is True
        and break_component.get("identity_authority") is True
        and isinstance(break_actor, Mapping)
        and break_actor.get("identity_authority") is False
        and (
            (
                _nonzero_hex_handle(break_actor.get("literal"))
                and break_actor.get("stable_name") == stable_identity.get("hit_actor")
            )
            or (
                not _nonzero_hex_handle(break_actor.get("literal"))
                and break_actor.get("stable_name") is None
            )
        )
        and stable_identity.get("authority") == FLOOR_TRACE_IDENTITY_AUTHORITY
        and all(
            stable_identity.get(key) not in (None, "")
            for key in (
                "hit_actor",
                "hit_actor_class",
                "hit_component",
                "hit_component_class",
            )
        )
    )


def _validate_capture(request: Mapping[str, Any]) -> dict[str, Any]:
    capture_root = Path(request["capture_output"])
    manifest_path = capture_root / "manifest.json"
    readback_path = capture_root / "runtime_asset_readbacks.json"
    raw_hit_journal_path = capture_root / GROUND_HIT_RAW_JOURNAL_NAME
    manifest = _load(manifest_path)
    raw_hit_journal = _load(raw_hit_journal_path)
    raw_entries = raw_hit_journal.get("entries")
    _require(
        raw_hit_journal.get("schema") == GROUND_HIT_RAW_JOURNAL_SCHEMA
        and raw_hit_journal.get("status") == "complete"
        and isinstance(raw_entries, list)
        and raw_hit_journal.get("entry_count") == len(raw_entries)
        and len(raw_entries) >= 30
        and raw_hit_journal.get("raw_component_literal_identity_claim") is False
        and raw_hit_journal.get("stable_identity_authority")
        == FLOOR_TRACE_IDENTITY_AUTHORITY
        and raw_hit_journal.get("release_authorized") is False
        and raw_hit_journal.get("qualification_claim") is False
        and raw_hit_journal.get("formal_dataset_count") == 0,
        "raw FHitResult Component journal closure failed",
    )
    journal_entries = {
        int(entry.get("sequence", -1)): entry
        for entry in raw_entries
        if isinstance(entry, Mapping)
    }
    _require(
        set(journal_entries) == set(range(len(raw_entries)))
        and all(
            _raw_hit_journal_entry_is_closed(entry, sequence=sequence)
            for sequence, entry in journal_entries.items()
        ),
        "raw FHitResult Component journal sequence/resolution closure failed",
    )
    frame = manifest.get("frame_contract", {})
    _require(
        manifest.get("schema") == "avengine_qa_native_spear_pixel_episode_v1"
        and manifest.get("status") == "pass"
        and manifest.get("scenario_id") == EPISODE_ID,
        "ground-contact capture manifest drift",
    )
    _require(
        Path(
            str(manifest.get("artifacts", {}).get("ground_contact_raw_hit_journal", ""))
        ).resolve()
        == raw_hit_journal_path.resolve(),
        "capture manifest does not bind the raw hit journal",
    )
    _require(
        frame.get("frame_count") == 3
        and frame.get("formal_episode_frame_count") == 75
        and frame.get("captured_frame_indices") == FRAME_INDICES,
        "capture is not exactly sparse f0/f37/f74",
    )
    rgb_paths = sorted((capture_root / "rgb_frames").glob("frame_*.png"))
    _require(len(rgb_paths) == 3, "fresh sparse RGB frame closure failed")
    readback = _load(readback_path)
    samples = readback.get("sampled_frames")
    _require(
        readback.get("schema") == "avengine_native_spear_runtime_asset_readbacks_v1"
        and readback.get("status") == "pass"
        and isinstance(samples, list)
        and [sample.get("frame_index") for sample in samples] == FRAME_INDICES,
        "runtime readback does not close f0/f37/f74",
    )
    trace_count = 0
    clearances: list[float] = []
    snap_corrections: list[float] = []
    snap_residuals: list[float] = []
    floor_actors: set[str] = set()
    floor_components: set[str] = set()
    measurements: list[dict[str, Any]] = []
    stable_bone_resolution_by_instance: dict[str, dict[str, Any]] = {}
    for sample in samples:
        frame_index = int(sample["frame_index"])
        records = sample.get("per_instance")
        _require(
            isinstance(records, Mapping) and set(records) == set(INSTANCE_IDS),
            f"frame {frame_index}: instance readback closure failed",
        )
        for slot, record in records.items():
            ground = record.get("live_ground_contact_readback")
            expected_owner = f"{slot}_actor"
            _require(
                record.get("actor_id") == expected_owner,
                f"{slot} runtime actor owner drift at frame {frame_index}",
            )
            _require(
                isinstance(ground, Mapping)
                and ground.get("schema") == READBACK_SCHEMA
                and ground.get("status") == "pass_instrumented_measurement_only"
                and ground.get("ue_length_unit") == "centimeter",
                f"{slot} live ground readback is missing at frame {frame_index}",
            )
            resolution_by_requested, stable_resolution = _validate_bone_name_resolution(
                ground.get("bone_name_resolution"),
                expected_owner=expected_owner,
                owner=f"{slot} frame {frame_index}",
            )
            previous_resolution = stable_bone_resolution_by_instance.get(slot)
            if previous_resolution is None:
                stable_bone_resolution_by_instance[slot] = stable_resolution
            else:
                _require(
                    stable_resolution == previous_resolution,
                    f"{slot} bone-name resolution is not stable across f0/f37/f74",
                )
            snap = ground.get("runtime_visual_ground_snap")
            _require(
                isinstance(snap, Mapping)
                and snap.get("schema") == "ue_dynamic_ground_snap_v1"
                and snap.get("status") == "passed"
                and snap.get("target") == "attached_visual_actor_root_component"
                and snap.get("timeline_anchor_mutated") is False
                and snap.get("emitter_or_rir_mutated") is False
                and snap.get("bounds_role") == "action_only_not_release_evidence",
                f"{slot} visual-only ground snap failed at frame {frame_index}",
            )
            snap_floor_trace = snap.get("floor_trace")
            _require(
                isinstance(snap_floor_trace, Mapping)
                and snap_floor_trace.get("status") == "hit"
                and snap_floor_trace.get("profile_name") == "BlockAll"
                and snap_floor_trace.get("trace_complex") is True
                and _finite_xyz(snap_floor_trace.get("hit_point_ue_cm"))
                and _finite_xyz(snap_floor_trace.get("hit_normal_ue")),
                f"{slot} visual ground snap floor trace failed at frame {frame_index}",
            )
            _validate_floor_trace_identity(
                snap_floor_trace,
                owner=f"{slot} visual snap frame {frame_index}",
            )
            _validate_trace_journal_binding(
                snap_floor_trace,
                entries=journal_entries,
                owner=f"{slot} visual snap frame {frame_index}",
            )
            correction = float(snap.get("applied_z_correction_cm", math.nan))
            residual = float(snap.get("residual_clearance_cm", math.nan))
            anchor_error = float(snap.get("maximum_timeline_anchor_error_cm", math.nan))
            bounds_before = snap.get("visual_bounds_before")
            bounds_after = snap.get("visual_bounds_after")
            anchor_before = snap.get("timeline_anchor_before_ue_cm")
            anchor_after = snap.get("timeline_anchor_after_ue_cm")
            floor_z = float(snap.get("floor_z_cm", math.nan))
            _require(
                isinstance(bounds_before, Mapping)
                and isinstance(bounds_after, Mapping)
                and isinstance(bounds_before.get("bottom_z_ue_cm"), (int, float))
                and isinstance(bounds_after.get("bottom_z_ue_cm"), (int, float))
                and _finite_xyz(anchor_before)
                and _finite_xyz(anchor_after)
                and math.isfinite(floor_z),
                f"{slot} visual snap bounds/anchor ledger failed at frame {frame_index}",
            )
            bottom_before = float(bounds_before["bottom_z_ue_cm"])
            bottom_after = float(bounds_after["bottom_z_ue_cm"])
            _require(
                math.isfinite(correction)
                and abs(correction) <= 15.0
                and math.isfinite(residual)
                and abs(residual) <= 0.1
                and math.isfinite(anchor_error)
                and anchor_error <= 1.0e-6,
                f"{slot} visual ground snap metric failed at frame {frame_index}",
            )
            _require(
                abs((bottom_after - bottom_before) - correction) <= 1.0e-4
                and abs((bottom_after - floor_z) - residual) <= 1.0e-4
                and max(
                    abs(float(before) - float(after))
                    for before, after in zip(anchor_before, anchor_after, strict=True)
                )
                <= 1.0e-6,
                f"{slot} visual snap pre/post geometry or anchor invariant failed",
            )
            snap_corrections.append(correction)
            snap_residuals.append(residual)
            actor_measurement = {
                "frame_index": frame_index,
                "instance_id": slot,
                "snap_floor_actor": snap_floor_trace["hit_actor"],
                "snap_floor_component": snap_floor_trace["hit_component"],
                "visual_bounds_bottom_before_cm": bottom_before,
                "visual_bounds_bottom_after_cm": bottom_after,
                "applied_z_correction_cm": correction,
                "residual_clearance_cm": residual,
                "timeline_anchor_before_ue_cm": anchor_before,
                "timeline_anchor_after_ue_cm": anchor_after,
                "maximum_timeline_anchor_error_cm": anchor_error,
                "emitter_or_rir_mutated": False,
                "anchors": [],
            }
            sides = ground.get("sides")
            _require(
                isinstance(sides, Mapping) and set(sides) == set(GROUND_BONES),
                f"{slot} left/right foot readback closure failed at frame {frame_index}",
            )
            for side, expected in GROUND_BONES.items():
                anchors = sides[side].get("anchors")
                _require(
                    isinstance(anchors, Mapping) and set(anchors) == set(expected),
                    f"{slot} {side} foot/toe closure failed at frame {frame_index}",
                )
                for kind, bone_name in expected.items():
                    anchor = anchors[kind]
                    resolution = resolution_by_requested[bone_name]
                    position = anchor.get("world_position_ue_cm")
                    clearance = anchor.get("bone_to_floor_clearance_cm")
                    trace = anchor.get("floor_trace")
                    _require(
                        anchor.get("bone_name") == bone_name
                        and anchor.get("requested_bone_name") == bone_name
                        and anchor.get("actual_live_bone_name")
                        == resolution["actual_live_name"]
                        and anchor.get("bone_name_resolution_mode")
                        == resolution["resolution_mode"]
                        and isinstance(anchor.get("bone_index"), int)
                        and not isinstance(anchor.get("bone_index"), bool)
                        and int(anchor["bone_index"])
                        == int(resolution["inventory_index"])
                        and _finite_xyz(position)
                        and isinstance(clearance, (int, float))
                        and not isinstance(clearance, bool)
                        and math.isfinite(float(clearance)),
                        f"{slot} {side} {kind} live bone readback failed",
                    )
                    _require(
                        isinstance(trace, Mapping)
                        and trace.get("status") == "hit"
                        and trace.get("profile_name") == "BlockAll"
                        and trace.get("trace_complex") is True
                        and _finite_xyz(trace.get("hit_point_ue_cm"))
                        and _finite_xyz(trace.get("hit_normal_ue")),
                        f"{slot} {side} {kind} exact floor trace failed",
                    )
                    _validate_floor_trace_identity(
                        trace,
                        owner=f"{slot} {side} {kind} frame {frame_index}",
                    )
                    _validate_trace_journal_binding(
                        trace,
                        entries=journal_entries,
                        owner=f"{slot} {side} {kind} frame {frame_index}",
                    )
                    hit_point = trace["hit_point_ue_cm"]
                    _require(
                        abs(float(position[0]) - float(hit_point[0])) <= 1.0e-4
                        and abs(float(position[1]) - float(hit_point[1])) <= 1.0e-4,
                        f"{slot} {side} {kind} floor trace XY drift",
                    )
                    trace_count += 1
                    clearances.append(float(clearance))
                    floor_actors.add(str(trace["hit_actor"]))
                    floor_components.add(str(trace["hit_component"]))
                    actor_measurement["anchors"].append(
                        {
                            "side": side,
                            "anchor_kind": kind,
                            "bone_name": bone_name,
                            "actual_live_bone_name": resolution["actual_live_name"],
                            "bone_name_resolution_mode": resolution["resolution_mode"],
                            "bone_index": int(anchor["bone_index"]),
                            "world_position_ue_cm": position,
                            "bone_to_floor_clearance_cm": float(clearance),
                            "floor_actor": trace["hit_actor"],
                            "floor_component": trace["hit_component"],
                            "floor_hit_point_ue_cm": hit_point,
                            "floor_hit_normal_ue": trace["hit_normal_ue"],
                        }
                    )
            _require(
                len(actor_measurement["anchors"]) == 4,
                f"{slot} frame {frame_index} does not contain four bone traces",
            )
            measurements.append(actor_measurement)
    _require(trace_count == 24, "ground diagnostic must contain exactly 24 traces")
    _require(
        set(stable_bone_resolution_by_instance) == set(INSTANCE_IDS),
        "stable two-instance bone-name resolution closure failed",
    )
    return {
        "status": "pass_live_measurements_manual_visual_review_pending",
        "manifest": _file_record(manifest_path),
        "runtime_asset_readbacks": _file_record(readback_path),
        "ground_contact_raw_hit_journal": _file_record(raw_hit_journal_path),
        "rgb_frames": [_file_record(path) for path in rgb_paths],
        "trace_count": trace_count,
        "minimum_bone_to_floor_clearance_cm": min(clearances),
        "maximum_bone_to_floor_clearance_cm": max(clearances),
        "maximum_abs_visual_root_correction_cm": max(map(abs, snap_corrections)),
        "maximum_abs_snap_residual_cm": max(map(abs, snap_residuals)),
        "observed_floor_actors": sorted(floor_actors),
        "observed_floor_components": sorted(floor_components),
        "per_actor_frame_measurements": measurements,
        "bone_name_resolution_by_instance": stable_bone_resolution_by_instance,
        "bounds_only_release_forbidden": True,
        "clearance_threshold_derivation_pending": True,
        "manual_pixel_ground_contact_review_required": True,
        "release_authorized": False,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def run(
    request_path: Path,
    *,
    dry_run: bool,
    authorize_one_attempt: bool,
) -> int:
    _require(
        dry_run != authorize_one_attempt,
        "select dry-run or explicitly authorize the one real attempt",
    )
    request, argv = _validate_request(request_path)
    attempt_root = Path(request["attempt_root"])
    dry_receipt = attempt_root / "dry_run_receipt.json"
    running_receipt = attempt_root / "running_receipt.json"
    final_receipt = attempt_root / "final_receipt.json"
    _require(not final_receipt.exists(), "attempt 01 already has a final receipt")
    _require(not running_receipt.exists(), "real attempt already started")
    if dry_run:
        _require(not dry_receipt.exists(), "dry-run receipt already exists")
    else:
        _require(dry_receipt.is_file(), "real launch requires a passed dry-run receipt")
        dry = _load(dry_receipt)
        _require(
            dry.get("schema") == RECEIPT_SCHEMA
            and dry.get("status") == "dry_run_pass_not_launched"
            and dry.get("request") == str(request_path.resolve())
            and dry.get("required_repo_commit") == request["required_repo_commit"]
            and dry.get("gpu_started") is False
            and dry.get("attempt_consumed") is False,
            "dry-run receipt binding drift",
        )
    snapshot = _gpu_snapshot()
    gpu = _validate_gpu1_idle(snapshot)
    rpc_index = argv.index("--rpc-port")
    _assert_port_available(int(argv[rpc_index + 1]))
    common = {
        "schema": RECEIPT_SCHEMA,
        "episode_id": EPISODE_ID,
        "attempt_policy": ATTEMPT_POLICY,
        "required_repo_commit": request["required_repo_commit"],
        "request": str(request_path.resolve()),
        "capture_argv": argv,
        "capture_output": request["capture_output"],
        "frame_indices": FRAME_INDICES,
        "full75_allowed": False,
        "physical_gpu_index": 1,
        "physical_gpu_uuid": GPU1_UUID,
        "graphics_adapter_argument": 1,
        "prelaunch_gpu": gpu,
        "prelaunch_snapshot": snapshot,
        "source_gpu_launch_authorized": False,
        "release_authorized": False,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }
    if dry_run:
        _write_json_exclusive(
            dry_receipt,
            {
                **common,
                "status": "dry_run_pass_not_launched",
                "gpu_started": False,
                "attempt_consumed": False,
                "captured_at_utc": _utc_now(),
            },
        )
        return 0

    started_at = _utc_now()
    _write_json_exclusive(
        running_receipt,
        {
            **common,
            "status": "running",
            "gpu_started": True,
            "attempt_consumed": True,
            "started_at_utc": started_at,
            "capture_process_exit_code": None,
        },
    )
    exit_code = 1
    final: dict[str, Any] = {
        **common,
        "status": "failed",
        "gpu_started": True,
        "attempt_consumed": True,
        "started_at_utc": started_at,
        "ended_at_utc": None,
        "capture_process_exit_code": None,
    }
    try:
        completed = subprocess.run(argv, cwd=REPOSITORY, check=False)
        exit_code = int(completed.returncode)
        final["capture_process_exit_code"] = exit_code
        _require(exit_code == 0, f"ground-contact capture exited {exit_code}")
        final["validation"] = _validate_capture(request)
        final["status"] = "pass_live_measurements_manual_visual_review_pending"
    except Exception as exc:  # noqa: BLE001
        final["error"] = f"{type(exc).__name__}: {exc}"
        exit_code = exit_code or 1
    finally:
        final["ended_at_utc"] = _utc_now()
        raw_hit_journal_path = (
            Path(request["capture_output"]) / GROUND_HIT_RAW_JOURNAL_NAME
        )
        if raw_hit_journal_path.is_file():
            final["ground_contact_raw_hit_journal"] = _file_record(raw_hit_journal_path)
        try:
            final["postlaunch_snapshot"] = _gpu_snapshot()
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            final["postlaunch_snapshot_error"] = f"{type(exc).__name__}: {exc}"
        _write_json_exclusive(final_receipt, final)
    return exit_code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-request", required=True, type=Path)
    prepare.add_argument(
        "--capture-python",
        type=Path,
        default=CAPTURE_PYTHON_LOGICAL,
    )
    launch = subparsers.add_parser("launch")
    launch.add_argument("--request", required=True, type=Path)
    mode = launch.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--authorize-one-attempt", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        path = prepare_request(
            source_request=args.source_request,
            capture_python=args.capture_python,
        )
        print(f"GROUND_CONTACT_REQUEST_PREPARED request={path} formal=0", flush=True)
        return 0
    return run(
        args.request,
        dry_run=args.dry_run,
        authorize_one_attempt=args.authorize_one_attempt,
    )


if __name__ == "__main__":
    raise SystemExit(main())
