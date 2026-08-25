from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from avengine.contracts.json_io import load_json
from avengine.rooms.qualification import (
    PlacementProbeError,
    build_qualification_report,
    compute_dataset_admission,
    evaluate_placement_feasibility,
    qualify_corrupted_acoustic_fixture,
    validate_qualification_report,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ROOM_EXAMPLES = REPOSITORY_ROOT / "examples/m6/rooms/qualification"
BAD_FIXTURE = (
    REPOSITORY_ROOT / "tests/fixtures/m6/corrupted_acoustic_package/fixture.json"
)


def _passing_probe() -> dict:
    return {
        "thresholds": {
            "maximum_support_gap_m": 0.15,
            "minimum_horizontal_clearance_m": 0.25,
        },
        "support_samples": [
            {"sample_id": sample_id, "hit": True, "distance_m": 0.05}
            for sample_id in (
                "center",
                "front_left",
                "front_right",
                "rear_left",
                "rear_right",
            )
        ],
        "horizontal_clearance_samples": [
            {"direction_id": "front", "hit_distance_m": None},
            {"direction_id": "back", "hit_distance_m": 0.8},
            {"direction_id": "left", "hit_distance_m": 0.4},
            {"direction_id": "right", "hit_distance_m": 0.3},
        ],
        "frustum_samples": [
            {"ray_id": "center", "outcome": "surface_hit", "opening_id": None},
            {
                "ray_id": "window_corner",
                "outcome": "opening_exit",
                "opening_id": "window_north",
            },
        ],
        "allowed_opening_ids": ["window_north"],
    }


def test_placement_probe_accepts_supported_clear_route_and_whitelisted_window() -> None:
    result = evaluate_placement_feasibility(_passing_probe())

    assert result["status"] == "pass"
    assert result["failure_reasons"] == []
    assert len(result["checks"]) == 11


def test_placement_probe_fails_closed_on_corner_gap_close_wall_and_scene_escape() -> None:
    probe = _passing_probe()
    probe["support_samples"][1] = {
        "sample_id": "front_left",
        "hit": False,
    }
    probe["horizontal_clearance_samples"][0]["hit_distance_m"] = 0.1
    probe["frustum_samples"][0]["outcome"] = "scene_escape"

    result = evaluate_placement_feasibility(probe)

    assert result["status"] == "fail"
    assert result["blocker_code"] == "placement_probe_failed"
    assert any(reason.startswith("support:front_left") for reason in result["failure_reasons"])
    assert any(reason.startswith("clearance:front") for reason in result["failure_reasons"])
    assert any(reason.startswith("frustum:center") for reason in result["failure_reasons"])


def test_placement_probe_rejects_unwhitelisted_opening_and_missing_support_ray() -> None:
    probe = _passing_probe()
    probe["allowed_opening_ids"] = []
    assert evaluate_placement_feasibility(probe)["status"] == "fail"

    malformed = _passing_probe()
    malformed["support_samples"].pop()
    with pytest.raises(PlacementProbeError, match="exactly"):
        evaluate_placement_feasibility(malformed)


def test_corrupted_acoustic_fixture_generates_valid_fail_closed_report() -> None:
    qualification = qualify_corrupted_acoustic_fixture(load_json(BAD_FIXTURE))

    assert validate_qualification_report(qualification.report) == []
    assert qualification.report["dataset_admission"] is False
    assert qualification.report["dimensions"]["acoustic_geometry_status"]["status"] == "fail"
    assert qualification.report["dimensions"]["material_binding_status"]["status"] == "fail"
    assert qualification.report["dimensions"]["ray_leakage_status"]["status"] == "fail"
    assert "source_identity_mismatch" in qualification.findings
    assert any(item.startswith("zero_area_triangles") for item in qualification.findings)


def test_visual_success_cannot_override_corrupted_acoustic_fixture() -> None:
    report = qualify_corrupted_acoustic_fixture(load_json(BAD_FIXTURE)).report
    report["dimensions"]["visual_runtime_status"] = {
        "status": "pass",
        "summary": "synthetic host visual shell loaded",
        "evidence_refs": ["contract_fixture"],
    }
    report["dimensions"]["navigation_status"] = {
        "status": "pass",
        "summary": "synthetic host navmesh loaded",
        "evidence_refs": ["contract_fixture"],
    }

    assert compute_dataset_admission(report).eligible is False


def test_report_builder_keeps_eligibility_separate_from_promotion() -> None:
    source = load_json(ROOM_EXAMPLES / "blender_custom_two_zone.json")
    passing = deepcopy(source)
    passing["dimensions"]["episode_feasibility_status"] = {
        "status": "pass",
        "summary": "M6 episode passed",
        "evidence_refs": ["fresh_evidence.json"],
    }
    passing["placement_feasibility"] = evaluate_placement_feasibility(_passing_probe())

    unpromoted = build_qualification_report(
        report_id="controlled_room_fresh_v1",
        subject=passing["subject"],
        evidence_basis="current_execution",
        evidence_artifacts=[{
            "artifact_id": "fresh_controlled_room_evidence",
            "path": "evidence/fresh_controlled_room.json",
            "sha256": "1" * 64,
        }],
        dimensions=passing["dimensions"],
        placement_feasibility=passing["placement_feasibility"],
        acoustic_diagnostics=passing["acoustic_diagnostics"],
        provenance=passing["provenance"],
    )
    promoted = build_qualification_report(
        report_id="controlled_room_fresh_v1",
        subject=passing["subject"],
        evidence_basis="current_execution",
        evidence_artifacts=[{
            "artifact_id": "fresh_controlled_room_evidence",
            "path": "evidence/fresh_controlled_room.json",
            "sha256": "1" * 64,
        }],
        dimensions=passing["dimensions"],
        placement_feasibility=passing["placement_feasibility"],
        acoustic_diagnostics=passing["acoustic_diagnostics"],
        provenance=passing["provenance"],
        promote_if_eligible=True,
    )

    assert compute_dataset_admission(unpromoted).eligible is True
    assert unpromoted["dataset_admission"] is False
    assert unpromoted["admission_blockers"] == ["release_promotion_not_requested"]
    assert promoted["dataset_admission"] is True
    assert promoted["admission_blockers"] == []
