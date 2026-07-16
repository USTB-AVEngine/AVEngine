from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.m2.compile_animal_package import _validate_motion_evidence


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _evidence(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    profile = tmp_path / "profile.json"
    profile.write_text('{"schema":"profile"}\n', encoding="utf-8")
    profile_sha256 = hashlib.sha256(profile.read_bytes()).hexdigest()
    fields = {
        "profile_id": "profile-v1",
        "adapter_id": "adapter-v1",
        "body_plan_id": "body-plan-v1",
        "motion_family_id": "walk-v1",
    }
    retarget = tmp_path / "retarget.json"
    _write_json(
        retarget,
        {
            "schema": "avengine_motion_retarget_evidence_v1",
            "status": "pass",
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "formal_dataset_registration_authorized": False,
            "profile": {"sha256": profile_sha256, **fields},
            "output": {"sha256": "retarget-output"},
        },
    )
    motion_qa = tmp_path / "motion_qa.json"
    _write_json(
        motion_qa,
        {
            "schema": "avengine_motion_retarget_audit_v1",
            "status": "pass",
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "formal_dataset_registration_authorized": False,
            "bindings": {
                "motion_profile_sha256": profile_sha256,
                "visual_glb_sha256": "package-visual",
                "baked_actions_sha256": "package-actions",
                **fields,
            },
            "qa": {"status": "pass"},
        },
    )
    return profile, retarget, motion_qa, {"source": {"sha256": "retarget-output"}}


def test_motion_evidence_accepts_one_hash_bound_lineage(tmp_path: Path) -> None:
    profile, retarget, motion_qa, rebase = _evidence(tmp_path)

    _validate_motion_evidence(
        visual_sha256="package-visual",
        actions_sha256="package-actions",
        rebase_report=rebase,
        motion_profile=profile,
        retarget_report=retarget,
        motion_qa_report=motion_qa,
    )


def test_motion_evidence_rejects_mismatched_package_actions(tmp_path: Path) -> None:
    profile, retarget, motion_qa, rebase = _evidence(tmp_path)

    with pytest.raises(ValueError, match="action hash"):
        _validate_motion_evidence(
            visual_sha256="package-visual",
            actions_sha256="tampered-actions",
            rebase_report=rebase,
            motion_profile=profile,
            retarget_report=retarget,
            motion_qa_report=motion_qa,
        )


def test_motion_evidence_rejects_failed_qa(tmp_path: Path) -> None:
    profile, retarget, motion_qa, rebase = _evidence(tmp_path)
    value = json.loads(motion_qa.read_text(encoding="utf-8"))
    value["qa"]["status"] = "fail"
    _write_json(motion_qa, value)

    with pytest.raises(ValueError, match="must pass"):
        _validate_motion_evidence(
            visual_sha256="package-visual",
            actions_sha256="package-actions",
            rebase_report=rebase,
            motion_profile=profile,
            retarget_report=retarget,
            motion_qa_report=motion_qa,
        )


def test_motion_evidence_rejects_malformed_nested_binding(tmp_path: Path) -> None:
    profile, retarget, motion_qa, rebase = _evidence(tmp_path)
    value = json.loads(motion_qa.read_text(encoding="utf-8"))
    value["bindings"] = []
    _write_json(motion_qa, value)

    with pytest.raises(ValueError, match="motion QA bindings"):
        _validate_motion_evidence(
            visual_sha256="package-visual",
            actions_sha256="package-actions",
            rebase_report=rebase,
            motion_profile=profile,
            retarget_report=retarget,
            motion_qa_report=motion_qa,
        )
