from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    sha256_file,
    write_json,
)
from avengine.m1.contracts import EVIDENCE_SCHEMA, aggregate_status
from avengine.m1.evidence import (
    BASE_REQUIRED_CHECKS,
    array_sha256,
    finalize_evidence,
    make_check,
    save_observations,
    verify_evidence_artifacts,
)


def _check(status: str, *, required: bool = True) -> dict:
    return make_check(
        f"check_{status}_{required}",
        status,
        measured=status,
        threshold="pass",
        required=required,
    )


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (["pass", "pass"], "pass"),
        (["pass", "not_run"], "not_run"),
        (["pass", "not_run", "blocked"], "blocked"),
        (["pass", "not_run", "blocked", "fail"], "fail"),
    ],
)
def test_aggregate_status_uses_deterministic_severity_precedence(
    statuses: list[str], expected: str
) -> None:
    assert aggregate_status([_check(status) for status in statuses]) == expected


def test_aggregate_status_ignores_non_required_failure() -> None:
    checks = [_check("pass"), _check("fail", required=False)]

    assert aggregate_status(checks) == "pass"


def test_aggregate_status_rejects_unknown_required_status() -> None:
    with pytest.raises(ValueError, match="status vocabulary"):
        aggregate_status([_check("unknown")])


def test_aggregate_status_rejects_empty_check_list() -> None:
    with pytest.raises(ValueError, match="At least one required check"):
        aggregate_status([])


def test_array_hash_is_stable_across_memory_layouts() -> None:
    contiguous = np.arange(12, dtype=np.float32).reshape(3, 4)
    fortran_order = np.asfortranarray(contiguous)

    assert not fortran_order.flags.c_contiguous
    assert array_sha256("rig_depth", contiguous) == array_sha256(
        "rig_depth", fortran_order
    )


def test_array_hash_binds_sensor_dtype_shape_and_values() -> None:
    values = np.arange(6, dtype=np.uint16).reshape(2, 3)
    baseline = array_sha256("rig_semantic", values)

    assert array_sha256("other_sensor", values) != baseline
    assert array_sha256("rig_semantic", values.astype(np.uint32)) != baseline
    assert array_sha256("rig_semantic", values.reshape(3, 2)) != baseline
    changed = values.copy()
    changed[0, 0] = 99
    assert array_sha256("rig_semantic", changed) != baseline


def test_finalize_evidence_is_deterministic_and_idempotent() -> None:
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "room_id": "room0",
        "checks": [_check("pass"), _check("fail", required=False)],
        "payload": {"z": 1, "a": [2, 3]},
    }

    finalized = finalize_evidence(evidence)
    first_hash = finalized["evidence_content_sha256"]
    finalized_again = finalize_evidence(finalized)

    assert finalized is evidence
    assert finalized["overall_status"] == "pass"
    assert len(first_hash) == 64
    assert finalized_again["evidence_content_sha256"] == first_hash

    changed = copy.deepcopy(finalized)
    changed["payload"]["a"].append(4)
    assert finalize_evidence(changed)["evidence_content_sha256"] != first_hash


@pytest.fixture
def complete_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Path]]:
    room_manifest_path = tmp_path / "room_manifest.json"
    capture_request_path = tmp_path / "capture_request.json"
    dataset_config = {
        "stages": {
            "paths": {".glb": ["*.glb"]},
            "default_attributes": {
                "nav_asset": "%%CONFIG_NAME_AS_ASSET_FILENAME%%.navmesh",
                "semantic_asset": "%%CONFIG_NAME_AS_ASSET_FILENAME%%_semantic.ply",
                "semantic_descriptor_filename": (
                    "%%CONFIG_NAME_AS_ASSET_FILENAME%%.house"
                ),
            },
        }
    }
    asset_definitions = [
        ("render_surface_mesh", "scene.glb", b"real surface mesh bytes"),
        (
            "scene_dataset_config",
            "dataset.json",
            json.dumps(dataset_config).encode("utf-8"),
        ),
        ("semantic_surface_mesh", "scene_semantic.ply", b"ply\n"),
        ("semantic_descriptor", "scene.house", b"ASCII 1.1\n"),
        ("navmesh", "scene.navmesh", b"navmesh bytes"),
    ]
    for _, relative_path, payload in asset_definitions:
        (tmp_path / relative_path).write_bytes(payload)

    identity = {
        "translation_m": [0.0, 0.0, 0.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    room_manifest = {
        "schema": "avengine_room_package_v1",
        "room_id": "room0",
        "room_kind": "habitat_native",
        "geometry_representation": "real_surface_mesh",
        "coordinate_system": {
            "handedness": "right",
            "up_axis": "+Y",
            "forward_axis": "-Z",
            "linear_unit": "meter",
            "quaternion_order": "xyzw",
        },
        "scene": {
            "scene_id_kind": "path",
            "scene_id": "scene.glb",
            "dataset_config_path": "dataset.json",
            "navmesh_path": "scene.navmesh",
            "navmesh_policy": "load_declared",
            "load_semantic_mesh": True,
            "enable_physics": False,
        },
        "assets": [
            {"role": role, "path": relative_path}
            for role, relative_path, _ in asset_definitions
        ],
        "semantics": {"interpretation": "raw controlled test IDs"},
        "navigation": {
            "agent_height_m": 1.5,
            "agent_radius_m": 0.2,
            "include_static_objects": False,
        },
        "openings": [],
        "connectivity_pairs": [
            {
                "pair_id": "test_pair",
                "start_m": [0.0, 0.0, 0.0],
                "end_m": [1.0, 0.0, 0.0],
            }
        ],
        "ray_checks": [],
        "acoustics": {
            "status": "deferred_to_m3",
            "reason": "M1 visual-only test fixture",
        },
        "provenance": {
            "source": "unit-test",
            "source_revision": "fixture-v1",
        },
    }
    request = {
        "schema": "avengine_m1_capture_request_v1",
        "request_id": "request0",
        "room_id": "room0",
        "seed": 7,
        "primary_camera_rig": {
            "rig_id": "camera_rig_0",
            "view_id": "view0",
            "world_from_rig": copy.deepcopy(identity),
            "shared_calibration": {
                "projection": "pinhole",
                "resolution_hw": [2, 3],
                "hfov_degrees": 90.0,
                "near_m": 0.05,
                "far_m": 10.0,
                "rig_from_sensor": copy.deepcopy(identity),
            },
            "modalities": [
                {"modality": "rgb", "sensor_uuid": "rig_rgb"},
                {"modality": "depth", "sensor_uuid": "rig_depth"},
                {"modality": "semantic", "sensor_uuid": "rig_semantic"},
            ],
        },
        "listener": {
            "listener_id": "listener0",
            "attached_to": "camera_rig_0",
            "rig_from_listener": copy.deepcopy(identity),
        },
        "sources": [
            {
                "source_id": "source0",
                "world_from_source": {
                    "translation_m": [1.0, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            {
                "source_id": "source1",
                "world_from_source": {
                    "translation_m": [2.0, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        ],
        "qa_views": [
            {
                "qa_id": "test_topdown",
                "kind": "topdown",
                "meters_per_pixel": 0.1,
                "height_m": 0.0,
            }
        ],
    }
    write_json(room_manifest_path, room_manifest)
    write_json(capture_request_path, request)

    runtime_root = tmp_path / "runtime"
    habitat_module = runtime_root / "habitat_sim" / "__init__.py"
    native_binding = runtime_root / "habitat_sim" / "_ext" / "bindings.so"
    native_binding.parent.mkdir(parents=True)
    habitat_module.parent.mkdir(parents=True, exist_ok=True)
    habitat_module.write_text("# test habitat module\n", encoding="utf-8")
    native_binding.write_bytes(b"test native binding")
    runtime_commit = "425fe084eb680844b2b01d86904b9a72c4896d7a"
    avengine_commit = "a" * 40

    def fake_git_run(arguments: list[str], **_: object) -> SimpleNamespace:
        repository = Path(arguments[2]).resolve()
        operation = arguments[3]
        if operation == "rev-parse":
            output = runtime_commit if repository == runtime_root else avengine_commit
            return SimpleNamespace(stdout=f"{output}\n")
        if operation == "status":
            return SimpleNamespace(stdout="")
        raise AssertionError(f"unexpected git invocation: {arguments}")

    monkeypatch.setattr("avengine.m1.evidence.subprocess.run", fake_git_run)

    native_binding_hash = sha256_file(native_binding)
    runtime_lock_hash = sha256_file(
        Path(__file__).resolve().parents[2] / "runtime.lock.yaml"
    )
    runtime = {
        "avengine_commit": avengine_commit,
        "avengine_worktree_dirty": False,
        "habitat_runtime_root": str(runtime_root),
        "habitat_runtime_commit": runtime_commit,
        "habitat_runtime_worktree_dirty": False,
        "locked_habitat_runtime_commit": runtime_commit,
        "habitat_module_path": str(habitat_module),
        "native_binding_path": str(native_binding),
        "native_binding_sha256": native_binding_hash,
        "runtime_lock_sha256": runtime_lock_hash,
        "habitat_python_version": "0.3.3-test",
        "habitat_audio_enabled": True,
        "habitat_bullet_enabled": True,
        "habitat_cuda_enabled": False,
        "python": "3.12-test",
        "platform": "unit-test",
        "numpy": np.__version__,
        "pillow": "test",
    }
    runtime["locked_habitat_runtime_commit"] = runtime_commit

    scene_assets = []
    for role, relative_path, _ in asset_definitions:
        path = tmp_path / relative_path
        scene_assets.append(
            {
                "role": role,
                "declared_path": relative_path,
                "resolved_path": str(path),
                "exists": True,
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    state = {
        "world_time_seconds": 0.0,
        "agent": copy.deepcopy(identity),
        "sensors": {
            uuid: copy.deepcopy(identity)
            for uuid in ("rig_rgb", "rig_depth", "rig_semantic", "listener0")
        },
    }
    state_hash = canonical_json_sha256(state)
    sensor_contract = {
        "rig_id": "camera_rig_0",
        "view_id": "view0",
        "world_from_rig": copy.deepcopy(identity),
        "shared_calibration": copy.deepcopy(
            request["primary_camera_rig"]["shared_calibration"]
        ),
        "modalities": copy.deepcopy(request["primary_camera_rig"]["modalities"]),
        "listener": copy.deepcopy(request["listener"]),
        "audio_propagation_status": "not_run",
        "audio_propagation_reason": "M1 pose anchor only; propagation is M4",
    }
    source_reports = [
        {
            "source_id": source["source_id"],
            "world_from_source": copy.deepcopy(source["world_from_source"]),
            "rig_from_source": copy.deepcopy(source["world_from_source"]),
            "recovered_world_from_source": copy.deepcopy(source["world_from_source"]),
            "roundtrip_max_error": 0.0,
        }
        for source in request["sources"]
    ]
    check_ids = set(BASE_REQUIRED_CHECKS) | {
        "connectivity_test_pair",
        "qa_test_topdown",
    }

    rgb = np.array(
        [
            [[1, 2, 3], [40, 50, 60], [100, 120, 140]],
            [[10, 20, 30], [80, 90, 100], [200, 210, 220]],
        ],
        dtype=np.uint8,
    )
    depth = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    semantic = np.array([[0, 101, 101], [0, 102, 102]], dtype=np.int32)
    modality_to_uuid = {
        "rgb": "rig_rgb",
        "depth": "rig_depth",
        "semantic": "rig_semantic",
    }

    def build_capture(
        output: Path,
        *,
        reference_record: dict | None,
        process_instance_id: str,
    ) -> tuple[Path, dict[str, dict]]:
        output.mkdir(parents=True, exist_ok=True)
        observations = save_observations(
            {
                "rig_rgb": rgb,
                "rig_depth": depth,
                "rig_semantic": semantic,
            },
            modality_to_uuid,
            output,
        )
        hashes = {
            modality: observations[modality]["raw_array_sha256"]
            for modality in ("rgb", "depth", "semantic")
        }
        repeats = [copy.deepcopy(hashes), copy.deepcopy(hashes)]
        qa_output = output / "qa" / "test_topdown.png"
        qa_output.parent.mkdir()
        topdown = np.array([[0, 255, 255], [0, 255, 0]], dtype=np.uint8)
        Image.fromarray(topdown, mode="L").save(qa_output)
        declared_checks = []
        pathfinder_fingerprint = {
            "settings": {
                "agent_height": 1.5,
                "agent_radius": 0.2,
                "include_static_objects": False,
            },
            "vertices_sha256": "a" * 64,
            "indices_sha256": "b" * 64,
        }
        loaded_graph = {
            "active_dataset": str(tmp_path / "dataset.json"),
            "current_scene": "scene",
            "scene_handle_matches": [str(tmp_path / "scene.glb")],
            "stage_template_matches": [str(tmp_path / "scene.glb")],
            "stage": {
                "handle": str(tmp_path / "scene.glb"),
                "render_asset": str(tmp_path / "scene.glb"),
                "collision_asset": str(tmp_path / "scene.glb"),
                "navmesh_asset": str(tmp_path / "scene.navmesh"),
                "semantic_asset": str(tmp_path / "scene_semantic.ply"),
                "semantic_descriptor": str(tmp_path / "scene.house"),
            },
            "navmesh": {
                "declared_path": str(tmp_path / "scene.navmesh"),
                "explicit_load_succeeded": True,
                "requested_agent_settings": {
                    "agent_height": 1.5,
                    "agent_radius": 0.2,
                    "include_static_objects": False,
                },
                "active_fingerprint": copy.deepcopy(pathfinder_fingerprint),
                "declared_fingerprint": copy.deepcopy(pathfinder_fingerprint),
            },
            "object_template_matches": {},
            "objects": [],
            "lighting": {"template_matches": [], "current_light_count": 0},
        }
        for check_id in sorted(check_ids):
            status = (
                "not_run"
                if check_id == "independent_process_repeatability"
                and reference_record is None
                else "pass"
            )
            declared_checks.append(
                make_check(
                    check_id,
                    status,
                    measured=(
                        {
                            "errors": [],
                            "static_errors": [],
                            "loaded_errors": [],
                            "loaded_graph": copy.deepcopy(loaded_graph),
                        }
                        if check_id == "scene_load_graph_closure"
                        else True
                    ),
                    threshold=True,
                    failure_reason=(
                        "first run has no independent reference"
                        if status == "not_run"
                        else None
                    ),
                )
            )
        evidence = {
            "schema": EVIDENCE_SCHEMA,
            "evidence_kind": "completed_capture",
            "room_id": "room0",
            "room_kind": "habitat_native",
            "request_id": "request0",
            "producer_process": {
                "process_instance_id": process_instance_id,
                "pid": 1001 if process_instance_id.startswith("1") else 1002,
                "initial_pid": 1001 if process_instance_id.startswith("1") else 1002,
                "started_at_utc": "2026-07-16T00:00:00+00:00",
            },
            "formal_view_ids": ["view0"],
            "room_manifest": {
                "path": str(room_manifest_path),
                "sha256": sha256_file(room_manifest_path),
            },
            "capture_request": {
                "path": str(capture_request_path),
                "sha256": sha256_file(capture_request_path),
            },
            "runtime": copy.deepcopy(runtime),
            "scene_assets": copy.deepcopy(scene_assets),
            "sensor_contract": copy.deepcopy(sensor_contract),
            "capture_state": {
                "before": copy.deepcopy(state),
                "after": copy.deepcopy(state),
                "before_sha256": state_hash,
                "after_sha256": state_hash,
            },
            "repeat_observation_hashes": repeats,
            "observations": observations,
            "sources": copy.deepcopy(source_reports),
            "connectivity": [
                {
                    "pair_id": "test_pair",
                    "requested_start_m": [0.0, 0.0, 0.0],
                    "requested_end_m": [1.0, 0.0, 0.0],
                    "snapped_start_m": [0.0, 0.0, 0.0],
                    "snapped_end_m": [1.0, 0.0, 0.0],
                    "start_snap_distance_m": 0.0,
                    "end_snap_distance_m": 0.0,
                    "found": True,
                    "geodesic_distance_m": 1.0,
                    "path_point_count": 2,
                }
            ],
            "ray_checks": [],
            "qa_observations": [
                {
                    "qa_id": "test_topdown",
                    "kind": "topdown",
                    "formal_view": False,
                    "meters_per_pixel": 0.1,
                    "height_m": 0.0,
                    "shape": [2, 3],
                    "navigable_pixel_count": 3,
                    "artifact": file_record(qa_output, relative_to=output),
                }
            ],
            "independent_reference": reference_record,
            "known_runtime_failures_carried_forward": [],
            "checks": declared_checks,
        }
        evidence["capture_batch_id"] = canonical_json_sha256(
            {
                "room_manifest_sha256": evidence["room_manifest"]["sha256"],
                "capture_request_sha256": evidence["capture_request"]["sha256"],
                "scene_assets": scene_assets,
                "avengine_commit": avengine_commit,
                "habitat_runtime_commit": runtime_commit,
                "native_binding_sha256": native_binding_hash,
                "state": state,
                "repeat_count": len(repeats),
            }
        )
        finalize_evidence(evidence)
        evidence_path = output / "evidence.json"
        write_json(evidence_path, evidence)
        return evidence_path, observations

    reference_source = tmp_path / "reference_source"
    reference_path, _ = build_capture(
        reference_source,
        reference_record=None,
        process_instance_id="11111111-1111-4111-8111-111111111111",
    )
    final_output = tmp_path / "final"
    final_output.mkdir()
    copied_reference = final_output / "independent_reference"
    shutil.copytree(reference_source, copied_reference)
    copied_evidence = copied_reference / "evidence.json"
    reference_record = {
        "path": copied_evidence.relative_to(final_output).as_posix(),
        "evidence_content_sha256": json.loads(
            copied_evidence.read_text(encoding="utf-8")
        )["evidence_content_sha256"],
        "artifact": file_record(copied_evidence, relative_to=final_output),
    }
    evidence_path, observation_records = build_capture(
        final_output,
        reference_record=reference_record,
        process_instance_id="22222222-2222-4222-8222-222222222222",
    )
    return evidence_path, {
        "depth": final_output / observation_records["depth"]["artifact"]["path"],
    }


def _checks_by_id(checks: list[dict]) -> dict[str, dict]:
    return {check["check_id"]: check for check in checks}


def test_verify_evidence_accepts_complete_artifacts_and_raw_hashes(
    complete_evidence: tuple[Path, dict[str, Path]],
) -> None:
    evidence_path, _ = complete_evidence

    status, checks = verify_evidence_artifacts(evidence_path)

    indexed = _checks_by_id(checks)
    assert status == "pass"
    assert indexed["evidence_json_schema"]["status"] == "pass"
    assert indexed["evidence_content_hash"]["status"] == "pass"
    assert indexed["evidence_raw_observation_hashes"]["status"] == "pass"


def test_verify_evidence_rejects_structurally_incomplete_document(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps({"schema": EVIDENCE_SCHEMA, "observations": {}}),
        encoding="utf-8",
    )

    status, checks = verify_evidence_artifacts(evidence_path)

    assert status == "fail"
    assert _checks_by_id(checks)["evidence_json_schema"]["status"] == "fail"


def test_verify_evidence_rejects_content_hash_tampering(
    complete_evidence: tuple[Path, dict[str, Path]],
) -> None:
    evidence_path, _ = complete_evidence
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["room_id"] = "tampered-room"
    write_json(evidence_path, evidence)

    status, checks = verify_evidence_artifacts(evidence_path)

    assert status == "fail"
    assert _checks_by_id(checks)["evidence_content_hash"]["status"] == "fail"


def test_verify_evidence_rejects_parent_directory_artifact_path(
    complete_evidence: tuple[Path, dict[str, Path]],
) -> None:
    evidence_path, _ = complete_evidence
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["observations"]["rgb"]["artifact"]["path"] = "../escape.png"
    finalize_evidence(evidence)
    write_json(evidence_path, evidence)

    status, checks = verify_evidence_artifacts(evidence_path)

    artifact_check = _checks_by_id(checks)["artifact_rgb_artifact"]
    assert status == "fail"
    assert artifact_check["status"] == "fail"
    assert "confined relative path" in artifact_check["measured"]["path_error"]


def test_verify_evidence_artifacts_detects_file_tampering(
    complete_evidence: tuple[Path, dict[str, Path]],
) -> None:
    evidence_path, artifacts = complete_evidence
    artifacts["depth"].write_bytes(b"tampered bytes")

    status, checks = verify_evidence_artifacts(evidence_path)

    assert status == "fail"
    assert _checks_by_id(checks)["artifact_depth_artifact"]["status"] == "fail"


def test_verify_rejects_required_profile_check_marked_optional(
    complete_evidence: tuple[Path, dict[str, Path]],
) -> None:
    evidence_path, _ = complete_evidence
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    next(
        check for check in evidence["checks"] if check["check_id"] == "rgb_nonconstant"
    )["required"] = False
    finalize_evidence(evidence)
    write_json(evidence_path, evidence)

    status, checks = verify_evidence_artifacts(evidence_path)

    assert status == "fail"
    assert _checks_by_id(checks)["evidence_check_profile"]["status"] == "fail"


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("sensor_contract", {}),
        ("sources", []),
        ("qa_observations", None),
        ("ray_checks", None),
    ],
)
def test_verify_rejects_malformed_core_evidence_without_traceback(
    complete_evidence: tuple[Path, dict[str, Path]],
    field: str,
    replacement: object,
) -> None:
    evidence_path, _ = complete_evidence
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence[field] = replacement
    finalize_evidence(evidence)
    write_json(evidence_path, evidence)

    status, checks = verify_evidence_artifacts(evidence_path)

    assert status == "fail"
    assert _checks_by_id(checks)["evidence_json_schema"]["status"] == "fail"


def test_verify_recomputes_connectivity_instead_of_trusting_declared_pass(
    complete_evidence: tuple[Path, dict[str, Path]],
) -> None:
    evidence_path, _ = complete_evidence
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["connectivity"][0]["found"] = False
    evidence["connectivity"][0]["geodesic_distance_m"] = None
    evidence["connectivity"][0]["path_point_count"] = 0
    finalize_evidence(evidence)
    write_json(evidence_path, evidence)

    status, checks = verify_evidence_artifacts(evidence_path)

    assert status == "fail"
    assert _checks_by_id(checks)["evidence_json_schema"]["status"] == "pass"
    assert _checks_by_id(checks)["evidence_connectivity_semantics"]["status"] == "fail"


def test_verify_keeps_zero_pixel_qa_as_schema_valid_semantic_failure(
    complete_evidence: tuple[Path, dict[str, Path]],
) -> None:
    evidence_path, _ = complete_evidence
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["qa_observations"][0]["navigable_pixel_count"] = 0
    finalize_evidence(evidence)
    write_json(evidence_path, evidence)

    status, checks = verify_evidence_artifacts(evidence_path)

    assert status == "fail"
    assert _checks_by_id(checks)["evidence_json_schema"]["status"] == "pass"
    assert _checks_by_id(checks)["evidence_qa_semantics"]["status"] == "fail"


def test_verify_re_resolves_declared_scene_asset_paths(
    complete_evidence: tuple[Path, dict[str, Path]],
) -> None:
    evidence_path, _ = complete_evidence
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    original = Path(evidence["scene_assets"][0]["resolved_path"])
    substituted = evidence_path.parent / "same_bytes_different_asset.glb"
    shutil.copyfile(original, substituted)
    evidence["scene_assets"][0]["resolved_path"] = str(substituted)
    finalize_evidence(evidence)
    write_json(evidence_path, evidence)

    status, checks = verify_evidence_artifacts(evidence_path)

    assert status == "fail"
    closure = _checks_by_id(checks)["evidence_scene_asset_closure"]
    assert closure["status"] == "fail"
    assert closure["measured"]["resolved_paths_match_declarations"] is False


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("stage", "semantic_asset"),
        ("stage", "semantic_descriptor"),
        ("navmesh", "declared_path"),
    ],
)
def test_verify_replays_loaded_scene_graph_snapshot(
    complete_evidence: tuple[Path, dict[str, Path]],
    section: str,
    field: str,
) -> None:
    evidence_path, _ = complete_evidence
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    scene_check = next(
        check
        for check in evidence["checks"]
        if check["check_id"] == "scene_load_graph_closure"
    )
    scene_check["measured"]["loaded_graph"][section][field] = str(
        evidence_path.parent / "same-bytes-alternate.asset"
    )
    finalize_evidence(evidence)
    write_json(evidence_path, evidence)

    status, checks = verify_evidence_artifacts(evidence_path)

    assert status == "fail"
    replay = _checks_by_id(checks)["evidence_scene_load_graph_closure"]
    assert replay["status"] == "fail"
    assert replay["measured"]["recorded_loaded_graph_errors"]


def test_verify_requires_self_contained_independent_reference(
    complete_evidence: tuple[Path, dict[str, Path]],
) -> None:
    evidence_path, _ = complete_evidence
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    reference_path = evidence_path.parent / evidence["independent_reference"]["path"]
    reference_path.unlink()

    status, checks = verify_evidence_artifacts(evidence_path)

    assert status == "fail"
    assert (
        _checks_by_id(checks)["artifact_independent_reference_artifact"]["status"]
        == "fail"
    )


def test_verify_rejects_same_process_instance_as_independent_reference(
    complete_evidence: tuple[Path, dict[str, Path]],
) -> None:
    evidence_path, _ = complete_evidence
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    reference_path = evidence_path.parent / evidence["independent_reference"]["path"]
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    evidence["producer_process"] = copy.deepcopy(reference["producer_process"])
    finalize_evidence(evidence)
    write_json(evidence_path, evidence)

    status, checks = verify_evidence_artifacts(evidence_path)

    assert status == "fail"
    independent = _checks_by_id(checks)["evidence_independent_reference"]
    assert independent["status"] == "fail"
    assert independent["measured"]["comparisons"]["fresh_process_instance"] is False


def test_verify_preserves_well_formed_blocked_attempt_status(
    complete_evidence: tuple[Path, dict[str, Path]],
) -> None:
    completed_path, _ = complete_evidence
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    blocked = {
        "schema": EVIDENCE_SCHEMA,
        "evidence_kind": "blocked_attempt",
        "room_id": completed["room_id"],
        "room_kind": completed["room_kind"],
        "request_id": completed["request_id"],
        "room_manifest": completed["room_manifest"],
        "capture_request": completed["capture_request"],
        "output_directory": str(completed_path.parent / "blocked"),
        "exception": {
            "type": "RuntimeError",
            "message": "controlled missing runtime capability",
        },
        "checks": [
            make_check(
                "capture_execution",
                "blocked",
                measured={"available": False},
                threshold={"available": True},
                failure_reason="controlled missing runtime capability",
            )
        ],
    }
    finalize_evidence(blocked)
    blocked_path = completed_path.parent / "blocked.json"
    write_json(blocked_path, blocked)

    status, checks = verify_evidence_artifacts(blocked_path)

    assert status == "blocked"
    assert _checks_by_id(checks)["blocked_attempt_contract"]["status"] == "pass"
