from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.m2 import admission
from avengine.m2.admission import (
    CanaryPromotionError,
    ExpectedArtifact,
    LEGACY_M2_HUMAN_REVIEW_DECISION,
    USER_DECISION_STATEMENT,
    build_human_review_decision,
    promote_research_candidate,
    validate_human_review_decision,
    write_human_review_decision_exclusive,
    write_legacy_m2_human_review_decision,
)
from avengine.m2.contracts import (
    compute_applied_state_hash,
    compute_pose_hash,
    validate_animal_asset_package,
)
from test_m2_contracts import (
    _asset_fixture,
    _request_fixture,
    _write_json,
)


def _rehash_record(asset_path: Path, asset: dict, role: str) -> None:
    record = next(value for value in asset["files"] if value["role"] == role)
    path = asset_path.parent / record["path"]
    record["byte_size"] = path.stat().st_size
    record["sha256"] = sha256_file(path)


CONTACT_IDS = (
    "paw_front_left",
    "paw_front_right",
    "paw_hind_left",
    "paw_hind_right",
)
FIXED_SCALE_REFERENCE = {
    "mode": "fixed_reference_unit_v1",
    "linear_scale": 1.0,
    "caller_supplied_linear_scale_allowed": False,
}
FIXTURE_DECISION_STATEMENT = "Fixture reviewer accepted this exact rear-leg video."


def _contact_report_fixture(asset: dict) -> tuple[dict, dict[str, list[int]]]:
    roles = {record["role"]: record for record in asset["files"]}
    stance = {
        contact_id: list(range(contact_index * 10, contact_index * 10 + 6))
        for contact_index, contact_id in enumerate(CONTACT_IDS)
    }
    actions = []
    for action_id in ("idle", "walk"):
        frames = []
        for index in range(41):
            frames.append(
                {
                    "sample_index": index,
                    "sample_tick": index * 3200,
                    "source_time_seconds": index / 15.0,
                    "contacts": [
                        {
                            "contact_id": contact_id,
                            "in_contact": (
                                True
                                if action_id == "idle"
                                else index in stance[contact_id]
                            ),
                        }
                        for contact_id in CONTACT_IDS
                    ],
                }
            )
        metrics = []
        for contact_id in CONTACT_IDS:
            contact_count = 41 if action_id == "idle" else len(stance[contact_id])
            metrics.append(
                {
                    "contact_id": contact_id,
                    "inference_mode": (
                        "forced_idle_contact"
                        if action_id == "idle"
                        else "height_backward_velocity_world_locked"
                    ),
                    "confidence": "high",
                    "idle_reference_height_m": 0.0,
                    "contact_height_threshold_m": 0.01,
                    "minimum_height_m": 0.0,
                    "maximum_height_m": 0.001 if action_id == "idle" else 0.02,
                    "vertical_range_m": 0.001 if action_id == "idle" else 0.02,
                    "maximum_step_displacement_m": (
                        0.001 if action_id == "idle" else 0.02
                    ),
                    "maximum_horizontal_step_m": 0.01,
                    "maximum_contact_horizontal_step_m": (
                        0.0 if action_id == "idle" else 0.01
                    ),
                    "contact_frame_count": contact_count,
                    "swing_frame_count": 41 - contact_count,
                }
            )
        actions.append(
            {
                "semantic_action_id": action_id,
                "source_action_name": f"fixture_{action_id}",
                "sample_count": 41,
                "frames": frames,
                "metrics": metrics,
            }
        )
    report = {
        "schema": "avengine_m2_contact_phases_v1",
        "source_glb_sha256": roles["visual"]["sha256"],
        "baked_actions_sha256": roles["walk_poses"]["sha256"],
        "runtime_joint_order": asset["skeleton"]["runtime_joint_order"],
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "coordinate_system": asset["coordinate_system"],
        "sample_rate_hz": 15,
        "time_base_hz": 48000,
        "contact_order": list(CONTACT_IDS),
        "anchor_definitions": [
            anchor for anchor in asset["anchors"] if anchor["anchor_id"] in CONTACT_IDS
        ],
        "thresholds": {
            "minimum_dynamic_vertical_range_m": 0.005,
            "contact_height_fraction": 0.35,
            "maximum_idle_vertical_range_m": 0.015,
            "maximum_idle_step_displacement_m": 0.003,
            "maximum_contact_horizontal_step_m": 0.015,
        },
        "uniform_linear_scale": 1.0,
        "actions": actions,
        "warnings": [],
        "notes": [
            "Walk stance requires both low height and rearward actor-relative velocity.",
            "The hash-bound world audit fits root cadence and gates stance residuals.",
        ],
        "scale_reference": FIXED_SCALE_REFERENCE,
    }
    return report, stance


def _world_contact_fixture(
    asset: dict, request: dict, *, contact_sha256: str, stance: dict[str, list[int]]
) -> dict:
    roles = {record["role"]: record for record in asset["files"]}
    walk_states = [state for state in request["states"] if state["action_id"] == "walk"]
    idle_contacts = {
        contact_id: {
            "vertical_range_m": 0.001,
            "maximum_vertical_range_m": 0.015,
            "maximum_step_displacement_m": 0.001,
            "maximum_allowed_step_displacement_m": 0.003,
            "passed": True,
        }
        for contact_id in CONTACT_IDS
    }
    dynamic_contacts = {
        contact_id: {
            "vertical_range_m": 0.02,
            "minimum_vertical_range_m": 0.005,
            "passed": True,
        }
        for contact_id in CONTACT_IDS
    }
    return {
        "schema": "avengine_m2_world_contact_audit_v1",
        "status": "pass",
        "qualification_claim": False,
        "source_glb_sha256": roles["visual"]["sha256"],
        "baked_actions_sha256": roles["walk_poses"]["sha256"],
        "contact_phases_sha256": contact_sha256,
        "solver": {
            "solver_id": "height_backward_velocity_constant_root_minimax_v1",
            "contact_height_fraction": 0.35,
            "root_step_search_m": {
                "minimum": 0.005,
                "maximum": 0.04,
                "increment": 0.0001,
            },
        },
        "root_step_fit": {
            "step_m": 0.01,
            "direction_world": [1.0, 0.0, 0.0],
            "maximum_contact_horizontal_step_m": 0.01,
            "mean_contact_horizontal_step_m": 0.005,
            "contact_pair_count": 20,
        },
        "contacts": {
            contact_id: {
                "contact_pair_count": 5,
                "maximum_contact_horizontal_step_m": 0.01,
                "mean_contact_horizontal_step_m": 0.005,
            }
            for contact_id in CONTACT_IDS
        },
        "trajectory": {
            "start_translation_m": walk_states[0]["root_transform"]["translation_m"],
            "end_translation_m": walk_states[-1]["root_transform"]["translation_m"],
            "rotation_xyzw": walk_states[0]["root_transform"]["rotation_xyzw"],
            "walk_frame_count": 45,
            "sample_rate_hz": 15,
            "path_length_m": 0.44,
            "root_speed_m_per_second": 0.15,
        },
        "gate": {
            "maximum_contact_horizontal_step_m": 0.015,
            "measured_maximum_contact_horizontal_step_m": 0.01,
            "passed": True,
        },
        "idle_gate": {"passed": True, "contacts": idle_contacts},
        "walk_dynamic_gate": {"passed": True, "contacts": dynamic_contacts},
        "overall_passed": True,
        "uniform_linear_scale": {
            "reference": 1.0,
            "target": 1.0,
            "normalized_measured_maximum_contact_horizontal_step_m": 0.01,
            "all_dimensional_solver_parameters_scaled": True,
        },
        "stance_frames_by_contact": stance,
        "scale_reference": FIXED_SCALE_REFERENCE,
    }


def _admission_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    revision = "fixture-rocketbox-revision"
    rocketbox_root = tmp_path / "Microsoft-Rocketbox"
    rocketbox_root.mkdir()
    license_path = rocketbox_root / "LICENSE.md"
    readme_path = rocketbox_root / "README.md"
    license_path.write_text("MIT License fixture\n", encoding="utf-8")
    readme_path.write_text("MICROSOFT ROCKETBOX AVATAR LIBRARY\n", encoding="utf-8")
    monkeypatch.setattr(admission, "ROCKETBOX_REVISION", revision)
    monkeypatch.setattr(
        admission, "ROCKETBOX_LICENSE_SHA256", sha256_file(license_path)
    )
    monkeypatch.setattr(admission, "ROCKETBOX_README_SHA256", sha256_file(readme_path))
    monkeypatch.setattr(admission, "verify_saved_capture_arrays", lambda *_: [])

    asset_path, asset = _asset_fixture(tmp_path)
    asset["admission_state"] = "research_candidate"
    asset["files"] = [
        record for record in asset["files"] if record["role"] != "human_visual_review"
    ]
    asset["qualification"] = {
        "automatic_qa_status": "pass",
        "human_visual_review_status": "not_run",
        "human_review_binding_sha256": None,
        "decision_reason": "Automatic fixture QA passed; human review has not run.",
    }
    asset["provenance"] = {
        "source": "Microsoft Rocketbox Dog_Beagle_01",
        "source_revision": revision,
        "source_sha256": "3" * 64,
        "license": "MIT",
        "allowed_use": "review_required",
        "redistribution": "review_required",
    }
    provenance_record = next(
        record for record in asset["files"] if record["role"] == "provenance_manifest"
    )
    provenance_path = asset_path.parent / provenance_record["path"]
    _write_json(
        provenance_path,
        {
            "source_manifest": {
                "snapshot": {
                    "source_repository": {
                        "url": admission.ROCKETBOX_REPOSITORY,
                        "revision": revision,
                    },
                    "source_artifacts": [
                        {
                            "path": "LICENSE.md",
                            "sha256": sha256_file(license_path),
                        },
                        {
                            "path": "README.md",
                            "sha256": sha256_file(readme_path),
                        },
                    ],
                }
            }
        },
    )
    _rehash_record(asset_path, asset, "provenance_manifest")
    contact_report, stance = _contact_report_fixture(asset)
    contact_record = next(
        record for record in asset["files"] if record["role"] == "contact_phases"
    )
    contact_path = asset_path.parent / contact_record["path"]
    _write_json(contact_path, contact_report)
    _rehash_record(asset_path, asset, "contact_phases")
    _write_json(asset_path, asset)
    candidate_sha256 = sha256_file(asset_path)
    request_path, request = _request_fixture(
        tmp_path,
        asset,
        asset_manifest_sha256=candidate_sha256,
    )
    for frame_index, state in enumerate(request["states"]):
        if frame_index >= 60:
            state["action_id"] = "idle"
            state["action_time_ticks"] = (frame_index - 60) * 3200
            state["pose_hash"] = compute_pose_hash(asset, state)
            state["applied_state_hash"] = compute_applied_state_hash(
                asset,
                state,
                asset_manifest_sha256=candidate_sha256,
            )
    _write_json(request_path, request)

    room_manifest = tmp_path / "room_manifest.json"
    room_request = tmp_path / "room_request.json"
    _write_json(room_manifest, {"fixture": "room"})
    _write_json(room_request, {"fixture": "camera"})
    evidence_root = tmp_path / "capture_evidence"
    media_root = evidence_root / "review_media"
    media_root.mkdir(parents=True)
    videos: dict[str, dict] = {}
    for modality in ("rgb", "depth", "semantic"):
        path = media_root / f"view0_{modality}_review.mp4"
        path.write_bytes(f"fixture {modality} review video\n".encode())
        videos[modality] = {
            "artifact": {
                "path": path.relative_to(evidence_root).as_posix(),
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
            },
            "frame_count": 75,
            "frame_rate_hz": 15,
            "view_id": "view0",
            "review_only": True,
            "qualification_claim": False,
        }
    frames = []
    for index, state in enumerate(request["states"]):
        frames.append(
            {
                "frame_index": index,
                "pts_ticks": state["pts_ticks"],
                "action_id": state["action_id"],
                "action_time_ticks": state["action_time_ticks"],
                "hashes": {
                    "declared_pose_hash": state["pose_hash"],
                    "recomputed_pose_hash": state["pose_hash"],
                    "declared_applied_state_hash": state["applied_state_hash"],
                    "recomputed_applied_state_hash": state["applied_state_hash"],
                },
                "animal_semantic_visibility": {
                    "visible": True,
                    "pixel_count": 10,
                },
                "modalities": {name: {} for name in ("rgb", "depth", "semantic")},
            }
        )
    evidence = {
        "schema": "avengine_m2_habitat_capture_evidence_v1",
        "status": "review_only",
        "evidence_kind": "research_candidate_habitat_review",
        "review_only": True,
        "qualification_claim": False,
        "request_id": request["request_id"],
        "asset_id": asset["asset_id"],
        "asset_admission_state": "research_candidate",
        "room_id": request["room_id"],
        "formal_view_ids": [],
        "formal_modalities": [],
        "review_view_ids": ["view0"],
        "review_modalities": ["rgb", "depth", "semantic"],
        "inputs": {
            "animal_asset_package": {
                "path": str(asset_path),
                "sha256": candidate_sha256,
            },
            "m2_capture_request": {
                "path": str(request_path),
                "sha256": sha256_file(request_path),
            },
            "m1_room_manifest": {
                "path": str(room_manifest),
                "sha256": sha256_file(room_manifest),
            },
            "m1_camera_request": {
                "path": str(room_request),
                "sha256": sha256_file(room_request),
            },
        },
        "runtime_application": {
            "initial_world_time_seconds": 0.0,
            "final_world_time_seconds": 0.0,
        },
        "frames": frames,
        "array_artifacts": {},
        "review_media": {"videos": videos},
    }
    evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
    evidence_path = evidence_root / "evidence.json"
    _write_json(evidence_path, evidence)

    world_contact = _world_contact_fixture(
        asset,
        request,
        contact_sha256=sha256_file(contact_path),
        stance=stance,
    )
    world_contact_path = tmp_path / "world_contact_audit.json"
    _write_json(world_contact_path, world_contact)
    canonical_contact = copy.deepcopy(contact_report)
    canonical_audit = copy.deepcopy(world_contact)
    monkeypatch.setattr(
        admission,
        "_reconstruct_world_contact_artifacts",
        lambda *_: (copy.deepcopy(canonical_contact), copy.deepcopy(canonical_audit)),
    )
    diagnostic_path = tmp_path / "walk_side.mp4"
    diagnostic_path.write_bytes(b"fixture approved walk-side diagnostic\n")
    decision = build_human_review_decision(
        candidate_manifest=asset_path,
        diagnostic_videos=[
            ExpectedArtifact(diagnostic_path, sha256_file(diagnostic_path))
        ],
        reviewer_id="fixture_reviewer",
        decision_date="2026-07-17",
        statement=FIXTURE_DECISION_STATEMENT,
        overall_canary_visual_acceptance="pass",
        rear_leg_motion_naturalness="pass",
    )
    decision_path = tmp_path / "human_review_decision.json"
    write_human_review_decision_exclusive(decision_path, decision)
    return {
        "asset_path": asset_path,
        "decision_path": decision_path,
        "request_path": request_path,
        "evidence_path": evidence_path,
        "world_contact_path": world_contact_path,
        "diagnostic_path": diagnostic_path,
        "rocketbox_root": rocketbox_root,
        "output": tmp_path / "canary_package",
    }


def _promote(paths: dict) -> admission.PromotionResult:
    return promote_research_candidate(
        candidate_manifest=paths["asset_path"],
        human_review_decision=paths["decision_path"],
        review_request=paths["request_path"],
        capture_evidence=paths["evidence_path"],
        world_contact_audit=ExpectedArtifact(
            paths["world_contact_path"], sha256_file(paths["world_contact_path"])
        ),
        diagnostic_videos=[
            ExpectedArtifact(
                paths["diagnostic_path"], sha256_file(paths["diagnostic_path"])
            )
        ],
        rocketbox_root=paths["rocketbox_root"],
        output_directory=paths["output"],
    )


def _bind_fixture_as_legacy_migration(
    paths: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = json.loads(paths["asset_path"].read_text())
    contact_record = next(
        record for record in asset["files"] if record["role"] == "contact_phases"
    )
    contact_path = paths["asset_path"].parent / contact_record["path"]
    decision = json.loads(paths["decision_path"].read_text())
    monkeypatch.setattr(
        admission,
        "_LEGACY_M2_HUMAN_REVIEW_DECISION_SHA256",
        decision["decision_content_sha256"],
    )
    monkeypatch.setattr(
        admission,
        "_LEGACY_M2_CANDIDATE_MANIFEST_SHA256",
        sha256_file(paths["asset_path"]),
    )
    monkeypatch.setattr(
        admission,
        "_LEGACY_M2_CONTACT_REPORT_SHA256",
        sha256_file(contact_path),
    )
    monkeypatch.setattr(
        admission,
        "_LEGACY_M2_WORLD_CONTACT_AUDIT_SHA256",
        sha256_file(paths["world_contact_path"]),
    )
    monkeypatch.setattr(
        admission,
        "_LEGACY_M2_REVIEW_REQUEST_SHA256",
        sha256_file(paths["request_path"]),
    )
    monkeypatch.setattr(
        admission,
        "_LEGACY_M2_CAPTURE_EVIDENCE_SHA256",
        sha256_file(paths["evidence_path"]),
    )


def _move_walk_trajectory_and_rebind_evidence(paths: dict) -> None:
    asset = json.loads(paths["asset_path"].read_text())
    candidate_sha256 = sha256_file(paths["asset_path"])
    request = json.loads(paths["request_path"].read_text())
    for state in request["states"]:
        if state["action_id"] != "walk":
            continue
        state["root_transform"]["translation_m"][0] += 1.0
        state["pose_hash"] = compute_pose_hash(asset, state)
        state["applied_state_hash"] = compute_applied_state_hash(
            asset,
            state,
            asset_manifest_sha256=candidate_sha256,
        )
    _write_json(paths["request_path"], request)

    evidence = json.loads(paths["evidence_path"].read_text())
    evidence["inputs"]["m2_capture_request"]["sha256"] = sha256_file(
        paths["request_path"]
    )
    for frame, state in zip(evidence["frames"], request["states"], strict=True):
        frame["hashes"] = {
            "declared_pose_hash": state["pose_hash"],
            "recomputed_pose_hash": state["pose_hash"],
            "declared_applied_state_hash": state["applied_state_hash"],
            "recomputed_applied_state_hash": state["applied_state_hash"],
        }
    evidence.pop("evidence_content_sha256")
    evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
    _write_json(paths["evidence_path"], evidence)


def test_promotion_is_new_hash_closed_canary_without_manifest_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _admission_fixture(tmp_path, monkeypatch)
    candidate_before = paths["asset_path"].read_bytes()

    result = _promote(paths)

    assert paths["asset_path"].read_bytes() == candidate_before
    final = json.loads(result.manifest_path.read_text())
    assert final["admission_state"] == "canary_qualified"
    assert final["qualification"]["human_visual_review_status"] == "pass"
    assert final["provenance"]["allowed_use"] == "research_canary"
    assert final["provenance"]["redistribution"] == "allowed"
    assert (
        validate_animal_asset_package(final, manifest_path=result.manifest_path) == []
    )
    review = json.loads(result.human_review_path.read_text())
    assert review["reviewer_decision"]["statement"] == FIXTURE_DECISION_STATEMENT
    provenance = json.loads(result.provenance_path.read_text())
    assert provenance["human_review_decision"]["sha256"] == sha256_file(
        paths["decision_path"]
    )
    assert (
        review["candidate"]["asset_manifest"]["sha256"]
        == hashlib.sha256(candidate_before).hexdigest()
    )
    assert review["capture"]["state_count"] == 75
    assert set(review["review_media"]) == {"rgb", "depth", "semantic"}
    assert review["world_contact_audit"]["sha256"] == sha256_file(
        paths["world_contact_path"]
    )
    assert result.manifest_sha256 not in result.human_review_path.read_text()
    assert result.manifest_sha256 not in result.provenance_path.read_text()
    assert all(record["path"] != "asset_manifest.json" for record in final["files"])


def test_promotion_refuses_existing_output_without_touching_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _admission_fixture(tmp_path, monkeypatch)
    paths["output"].mkdir()
    marker = paths["output"] / "owned.txt"
    marker.write_text("keep\n", encoding="utf-8")

    with pytest.raises(CanaryPromotionError, match="refusing to overwrite"):
        _promote(paths)

    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_promotion_refuses_dangling_output_symlink_without_replacing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _admission_fixture(tmp_path, monkeypatch)
    dangling_target = tmp_path / "must-not-be-created"
    paths["output"].symlink_to(dangling_target, target_is_directory=True)

    with pytest.raises(CanaryPromotionError, match="refusing to overwrite"):
        _promote(paths)

    assert paths["output"].is_symlink()
    assert not dangling_target.exists()


def test_promotion_rejects_symlinked_output_ancestor_without_writing_through_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _admission_fixture(tmp_path, monkeypatch)
    outside = tmp_path / "outside-output-parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked-output-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)
    paths["output"] = linked_parent / "canary-package"

    with pytest.raises(CanaryPromotionError, match="symbolic link"):
        _promote(paths)

    assert linked_parent.is_symlink()
    assert not (outside / "canary-package").exists()


def test_atomic_promotion_never_replaces_racing_empty_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _admission_fixture(tmp_path, monkeypatch)
    original = admission._publish_directory_no_replace
    raced_inode: int | None = None

    def race(staging: Path, destination: Path) -> None:
        nonlocal raced_inode
        destination.mkdir()
        raced_inode = destination.stat().st_ino
        original(staging, destination)

    monkeypatch.setattr(admission, "_publish_directory_no_replace", race)
    with pytest.raises(
        CanaryPromotionError, match="appeared during atomic publication"
    ):
        _promote(paths)

    assert raced_inode is not None
    assert paths["output"].is_dir()
    assert paths["output"].stat().st_ino == raced_inode
    assert list(paths["output"].iterdir()) == []
    assert not list(tmp_path.glob(".canary_package.staging-*"))


def test_atomic_promotion_fails_closed_when_renameat2_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _admission_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(admission.ctypes, "CDLL", lambda *_args, **_kwargs: object())

    with pytest.raises(CanaryPromotionError, match="no-replace.*unavailable"):
        _promote(paths)

    assert not paths["output"].exists()
    assert not list(tmp_path.glob(".canary_package.staging-*"))


def test_promotion_rejects_symlink_anywhere_in_capture_evidence_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _admission_fixture(tmp_path, monkeypatch)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    (paths["evidence_path"].parent / "unrelated-link").symlink_to(outside)

    with pytest.raises(CanaryPromotionError, match="capture evidence tree"):
        _promote(paths)

    assert not paths["output"].exists()


def test_promotion_fails_closed_on_diagnostic_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _admission_fixture(tmp_path, monkeypatch)

    with pytest.raises(CanaryPromotionError, match="diagnostic video hash mismatch"):
        promote_research_candidate(
            candidate_manifest=paths["asset_path"],
            human_review_decision=paths["decision_path"],
            review_request=paths["request_path"],
            capture_evidence=paths["evidence_path"],
            world_contact_audit=ExpectedArtifact(
                paths["world_contact_path"], sha256_file(paths["world_contact_path"])
            ),
            diagnostic_videos=[ExpectedArtifact(paths["diagnostic_path"], "0" * 64)],
            rocketbox_root=paths["rocketbox_root"],
            output_directory=paths["output"],
        )

    assert not paths["output"].exists()


def test_promotion_fails_closed_on_world_contact_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _admission_fixture(tmp_path, monkeypatch)

    with pytest.raises(CanaryPromotionError, match="world-contact audit hash mismatch"):
        promote_research_candidate(
            candidate_manifest=paths["asset_path"],
            human_review_decision=paths["decision_path"],
            review_request=paths["request_path"],
            capture_evidence=paths["evidence_path"],
            world_contact_audit=ExpectedArtifact(paths["world_contact_path"], "0" * 64),
            diagnostic_videos=[
                ExpectedArtifact(
                    paths["diagnostic_path"], sha256_file(paths["diagnostic_path"])
                )
            ],
            rocketbox_root=paths["rocketbox_root"],
            output_directory=paths["output"],
        )

    assert not paths["output"].exists()


def test_promotion_rejects_self_declared_relaxed_world_contact_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _admission_fixture(tmp_path, monkeypatch)
    audit = json.loads(paths["world_contact_path"].read_text())
    audit["gate"]["maximum_contact_horizontal_step_m"] = 100.0
    audit["gate"]["measured_maximum_contact_horizontal_step_m"] = 99.0
    _write_json(paths["world_contact_path"], audit)

    with pytest.raises(CanaryPromotionError, match="fixed M2 0.015 m gate"):
        _promote(paths)

    assert not paths["output"].exists()


@pytest.mark.parametrize(
    "field",
    ("root_step_fit", "contacts", "stance_frames_by_contact", "uniform_linear_scale"),
)
def test_promotion_rejects_incomplete_normative_world_contact_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    paths = _admission_fixture(tmp_path, monkeypatch)
    audit = json.loads(paths["world_contact_path"].read_text())
    del audit[field]
    _write_json(paths["world_contact_path"], audit)

    with pytest.raises(CanaryPromotionError, match="field set is incomplete"):
        _promote(paths)

    assert not paths["output"].exists()


def test_promotion_rejects_conflicting_solver_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _admission_fixture(tmp_path, monkeypatch)
    audit = json.loads(paths["world_contact_path"].read_text())
    audit["solver"]["root_step_search_m"]["increment"] = 0.001
    _write_json(paths["world_contact_path"], audit)

    with pytest.raises(CanaryPromotionError, match="complete fixed M2 configuration"):
        _promote(paths)


def test_promotion_rejects_consistent_but_fabricated_contact_means(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _admission_fixture(tmp_path, monkeypatch)
    audit = json.loads(paths["world_contact_path"].read_text())
    audit["root_step_fit"]["mean_contact_horizontal_step_m"] = 0.004
    for record in audit["contacts"].values():
        record["mean_contact_horizontal_step_m"] = 0.004
    _write_json(paths["world_contact_path"], audit)

    with pytest.raises(CanaryPromotionError, match="independent reconstruction"):
        _promote(paths)


def test_promotion_rehashes_evidence_declared_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _admission_fixture(tmp_path, monkeypatch)
    evidence = json.loads(paths["evidence_path"].read_text())
    room_path = Path(evidence["inputs"]["m1_room_manifest"]["path"])
    room_path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(CanaryPromotionError, match="evidence input hash mismatch"):
        _promote(paths)

    assert not paths["output"].exists()


def _reauthenticate_decision(value: dict) -> dict:
    core = copy.deepcopy(value)
    core.pop("decision_content_sha256", None)
    value["decision_content_sha256"] = canonical_json_sha256(core)
    return value


@pytest.mark.parametrize(
    ("binding", "message"),
    (
        ("asset_manifest", "different candidate/visual/actions"),
        ("visual", "different candidate/visual/actions"),
        ("idle_poses", "different candidate/visual/actions"),
        ("walk_poses", "different candidate/visual/actions"),
        ("diagnostic", "different diagnostic video bytes"),
    ),
)
def test_promotion_rejects_reauthenticated_decision_for_different_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding: str,
    message: str,
) -> None:
    paths = _admission_fixture(tmp_path, monkeypatch)
    decision = json.loads(paths["decision_path"].read_text())
    if binding == "diagnostic":
        decision["reviewed_diagnostics"][0]["sha256"] = "0" * 64
    else:
        decision["candidate"][binding]["sha256"] = "0" * 64
    _reauthenticate_decision(decision)
    replacement = tmp_path / f"decision-{binding}.json"
    write_human_review_decision_exclusive(replacement, decision)
    paths["decision_path"] = replacement

    with pytest.raises(CanaryPromotionError, match=message):
        _promote(paths)

    assert not paths["output"].exists()


def test_promotion_rejects_new_video_even_when_caller_supplies_its_new_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _admission_fixture(tmp_path, monkeypatch)
    paths["diagnostic_path"].write_bytes(b"different unreviewed video bytes\n")

    with pytest.raises(CanaryPromotionError, match="different diagnostic video bytes"):
        _promote(paths)

    assert not paths["output"].exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "field set is incomplete"),
        ("tampered_hash", "content hash differs"),
    ),
)
def test_promotion_rejects_incomplete_or_tampered_decision_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    paths = _admission_fixture(tmp_path, monkeypatch)
    decision = json.loads(paths["decision_path"].read_text())
    if mutation == "missing":
        decision.pop("reviewer_decision")
    else:
        decision["reviewer_decision"]["statement"] = "tampered after review"
    bad = tmp_path / f"bad-decision-{mutation}.json"
    _write_json(bad, decision)
    paths["decision_path"] = bad

    with pytest.raises(CanaryPromotionError, match=message):
        _promote(paths)

    assert not paths["output"].exists()


def test_decision_writer_is_no_replace_and_rejects_dangling_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _admission_fixture(tmp_path, monkeypatch)
    decision = json.loads(paths["decision_path"].read_text())
    occupied = tmp_path / "occupied-decision.json"
    occupied.write_bytes(b"keep existing bytes\n")
    original = occupied.read_bytes()

    with pytest.raises(CanaryPromotionError, match="refusing to replace"):
        write_human_review_decision_exclusive(occupied, decision)
    assert occupied.read_bytes() == original

    outside = tmp_path / "must-not-be-created.json"
    dangling = tmp_path / "dangling-decision.json"
    dangling.symlink_to(outside)
    with pytest.raises(CanaryPromotionError, match="refusing to replace"):
        write_human_review_decision_exclusive(dangling, decision)
    assert dangling.is_symlink()
    assert not outside.exists()


def test_promotion_rejects_symlinked_decision_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _admission_fixture(tmp_path, monkeypatch)
    link = tmp_path / "decision-link.json"
    link.symlink_to(paths["decision_path"])
    paths["decision_path"] = link

    with pytest.raises(CanaryPromotionError, match="symbolic link"):
        _promote(paths)

    assert not paths["output"].exists()


def test_legacy_migration_rejects_a_new_self_consistent_walk_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _admission_fixture(tmp_path, monkeypatch)
    _bind_fixture_as_legacy_migration(paths, monkeypatch)
    _move_walk_trajectory_and_rebind_evidence(paths)

    with pytest.raises(CanaryPromotionError, match="intermediate walk state"):
        _promote(paths)

    assert not paths["output"].exists()


def test_legacy_migration_requires_exact_capture_evidence_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _admission_fixture(tmp_path, monkeypatch)
    _bind_fixture_as_legacy_migration(paths, monkeypatch)

    def normal_validation_was_required(*_args: object, **_kwargs: object) -> object:
        raise CanaryPromotionError("normal world-contact validation was required")

    monkeypatch.setattr(
        admission,
        "_reconstruct_world_contact_artifacts",
        normal_validation_was_required,
    )
    candidate = json.loads(paths["asset_path"].read_text())
    request = json.loads(paths["request_path"].read_text())
    decision = admission.load_human_review_decision(paths["decision_path"])
    exact_audit = admission._validate_world_contact_audit(
        paths["world_contact_path"],
        candidate=candidate,
        candidate_manifest=paths["asset_path"],
        request=request,
        request_path=paths["request_path"],
        evidence_path=paths["evidence_path"],
        decision=decision,
    )
    assert exact_audit["schema"] == "avengine_m2_world_contact_audit_v1"

    evidence = json.loads(paths["evidence_path"].read_text())
    evidence["runtime_application"]["test_marker"] = "different capture bytes"
    evidence.pop("evidence_content_sha256")
    evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
    _write_json(paths["evidence_path"], evidence)
    with pytest.raises(CanaryPromotionError, match="normal world-contact validation"):
        admission._validate_world_contact_audit(
            paths["world_contact_path"],
            candidate=candidate,
            candidate_manifest=paths["asset_path"],
            request=request,
            request_path=paths["request_path"],
            evidence_path=paths["evidence_path"],
            decision=decision,
        )


def test_legacy_decision_is_exact_and_quote_cannot_be_rebound(tmp_path: Path) -> None:
    assert (
        validate_human_review_decision(copy.deepcopy(LEGACY_M2_HUMAN_REVIEW_DECISION))[
            "reviewer_decision"
        ]["statement"]
        == USER_DECISION_STATEMENT
    )
    migrated = tmp_path / "legacy_m2_human_review_decision.json"
    write_legacy_m2_human_review_decision(migrated)
    assert json.loads(migrated.read_text()) == LEGACY_M2_HUMAN_REVIEW_DECISION

    rebound = copy.deepcopy(LEGACY_M2_HUMAN_REVIEW_DECISION)
    rebound["candidate"]["visual"]["sha256"] = "0" * 64
    _reauthenticate_decision(rebound)
    with pytest.raises(CanaryPromotionError, match="legacy M2 user statement"):
        validate_human_review_decision(rebound)
