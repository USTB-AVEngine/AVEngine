from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.m2 import admission
from avengine.m2.admission import (
    CanaryPromotionError,
    ExpectedArtifact,
    USER_DECISION_STATEMENT,
    promote_research_candidate,
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

    roles = {record["role"]: record for record in asset["files"]}
    walk_states = [state for state in request["states"] if state["action_id"] == "walk"]
    world_contact = {
        "schema": "avengine_m2_world_contact_audit_v1",
        "status": "pass",
        "qualification_claim": False,
        "source_glb_sha256": roles["visual"]["sha256"],
        "baked_actions_sha256": roles["walk_poses"]["sha256"],
        "contact_phases_sha256": roles["contact_phases"]["sha256"],
        "gate": {
            "maximum_contact_horizontal_step_m": 0.015,
            "measured_maximum_contact_horizontal_step_m": 0.01,
            "passed": True,
        },
        "trajectory": {
            "walk_frame_count": 45,
            "sample_rate_hz": 15,
            "start_translation_m": walk_states[0]["root_transform"]["translation_m"],
            "end_translation_m": walk_states[-1]["root_transform"]["translation_m"],
        },
    }
    world_contact_path = tmp_path / "world_contact_audit.json"
    _write_json(world_contact_path, world_contact)
    diagnostic_path = tmp_path / "walk_side.mp4"
    diagnostic_path.write_bytes(b"fixture approved walk-side diagnostic\n")
    return {
        "asset_path": asset_path,
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
    assert review["reviewer_decision"]["statement"] == USER_DECISION_STATEMENT
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


def test_promotion_fails_closed_on_diagnostic_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _admission_fixture(tmp_path, monkeypatch)

    with pytest.raises(CanaryPromotionError, match="diagnostic video hash mismatch"):
        promote_research_candidate(
            candidate_manifest=paths["asset_path"],
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
