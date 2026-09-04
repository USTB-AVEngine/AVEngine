"""Tests for the room-centric QA-v3 scene x profile scheduler."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from run_qa_v3_room_profile_scheduler import (  # noqa: E402
    SceneSpec,
    classify_manifest,
    main,
    run_scheduler,
)


def _manifest(candidates=1, *, proof=None):
    value = {
        "evidence_class": "geometry_candidate",
        "counts": {
            "cells_requested": max(1, candidates),
            "geometry_candidates": candidates,
            "rejected": max(1, candidates) - candidates,
        },
        "search": {
            "combinations_evaluated": 7,
            "budget_exhausted": 1 if candidates == 0 else 0,
            "by_reason": {
                "target_outside_fov_at_binding_instant": 3,
            },
        },
    }
    if proof is not None:
        value["feasibility_proof"] = proof
    return value


def test_finite_failure_is_not_mislabeled_scene_infeasible():
    record = classify_manifest(_manifest(candidates=0))
    assert record["attempt_status"] == "not_found_within_budget"
    assert "not proof" in record["boundary"]
    assert record["quota_status"] == "empty"
    assert record["quota_shortfall"] == 1


def test_resource_unavailable_is_not_profile_or_scene_failure():
    manifest = _manifest(candidates=0)
    manifest["resource_status"] = {
        "status": "unavailable",
        "method": "registry_preflight",
        "missing": ["four_transcribed_speech_assets"],
    }
    record = classify_manifest(manifest)
    assert record["attempt_status"] == "resource_unavailable"
    assert record["evidence_class"] == "resource_unavailable"
    assert record["resource_status"]["missing"] == [
        "four_transcribed_speech_assets"]


def test_scene_infeasible_requires_explicit_exhaustive_proof():
    record = classify_manifest(_manifest(
        candidates=0,
        proof={"status": "infeasible", "method": "exhaustive",
               "detail": "finite route-camera domain exhausted"}))
    assert record["quota_status"] == "empty"
    assert record["attempt_status"] == "scene_infeasible"
    assert record["infeasibility_proof"]["method"] == "exhaustive"


def test_partial_generation_keeps_quota_shortfall_visible():
    manifest = _manifest(candidates=1)
    manifest["counts"].update(
        {"cells_requested": 2, "geometry_candidates": 1, "rejected": 1})
    record = classify_manifest(manifest)
    assert record["attempt_status"] == "generated"
    assert record["quota_status"] == "partial"
    assert record["quota_shortfall"] == 1


def test_complete_pixel_rejection_is_distinct_from_geometry_failure():
    record = classify_manifest(
        _manifest(candidates=2),
        {"complete_for_geometry_candidates": True,
         "attempted": 2, "passed": 0, "rejected": 2,
         "rejection_reasons": {"visible_pixels_below_minimum": 2}})
    assert record["attempt_status"] == "pixel_rejected"
    assert record["geometry_candidates"] == 2
    assert record["pixel"]["rejected"] == 2
    assert record["quota_status"] == "filled"


def test_partial_pixel_results_do_not_overstate_rejection():
    record = classify_manifest(
        _manifest(candidates=2),
        {"complete_for_geometry_candidates": False,
         "attempted": 1, "passed": 0, "rejected": 1})
    assert record["attempt_status"] == "generated"
    assert record["evidence_class"] == "geometry_candidate"
    assert record["pixel"]["status"] == "partial"


def test_complete_pixel_counts_must_cover_all_candidates():
    with pytest.raises(ValueError, match="does not cover every"):
        classify_manifest(
            _manifest(candidates=2),
            {"complete_for_geometry_candidates": True,
             "attempted": 1, "passed": 0, "rejected": 1})


def test_batch_counts_must_close():
    manifest = _manifest(candidates=1)
    manifest["counts"]["cells_requested"] = 2
    with pytest.raises(ValueError, match="counts do not close"):
        classify_manifest(manifest)

def test_scheduler_attempts_full_matrix_and_isolates_pair_failure(tmp_path):
    scenes = [
        SceneSpec(tmp_path / "s1.json", "room_A", {"scene_id": "room_A"}),
        SceneSpec(tmp_path / "s2.json", "room_B", {"scene_id": "room_B"}),
    ]
    profiles = {
        "card1F": {"id": "card1F"},
        "card7": {"id": "card7"},
    }
    out = tmp_path / "out"
    out.mkdir()

    def fake_runner(**kwargs):
        scene = json.loads(kwargs["scene_config"].read_text())["scene_id"]
        profile = json.loads(kwargs["profile_config"].read_text())[0]["id"]
        if scene == "room_A" and profile == "card7":
            raise RuntimeError("synthetic pair failure")
        result = _manifest(
            candidates=0 if (scene, profile) == ("room_B", "card1F") else 1)
        kwargs["batch_root"].mkdir(parents=True)
        (kwargs["batch_root"] / "batch_manifest.json").write_text(
            json.dumps(result))
        return result

    matrix = run_scheduler(
        scene_specs=scenes,
        profile_catalog=profiles,
        requested_profiles=["card1F", "card7", "card16"],
        params_value={"T_HALF": 1.0},
        params_source=tmp_path / "params.json",
        out_root=out,
        cells=1,
        seed="test-seed",
        snapshot_content="/unused",
        pixel_results={},
        runner=fake_runner)

    assert matrix["expected_matrix_cells"] == 6
    assert matrix["observed_matrix_cells"] == 6
    assert matrix["attempted_every_requested_profile_per_scene"] is True
    assert matrix["counts_by_status"] == {
        "generated": 2,
        "pipeline_error": 1,
        "profile_not_implemented": 2,
        "not_found_within_budget": 1,
    }
    assert matrix["counts_by_quota_status"] == {
        "filled": 2,
        "not_run": 3,
        "empty": 1,
    }
    rows = {(row["scene_id"], row["profile_id"]): row
            for row in matrix["matrix"]}
    assert rows[("room_A", "card7")]["attempt_status"] == "pipeline_error"
    assert rows[("room_A", "card16")][
        "attempt_status"] == "profile_not_implemented"
    assert rows[("room_B", "card7")]["attempt_status"] == "generated"
    for room in ("room_A", "room_B"):
        manifest = json.loads(
            (out / "rooms" / room / "room_attempt_manifest.json").read_text())
        assert manifest["attempted_all_requested_profiles"] is True
        assert len(manifest["attempts"]) == 3


def test_existing_root_is_no_clobber_even_with_missing_inputs(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    assert main([
        "--scene-config", str(tmp_path / "missing-scene.json"),
        "--profiles", str(tmp_path / "missing-profiles.json"),
        "--params", str(tmp_path / "missing-params.json"),
        "--seed", "seed",
        "--out-root", str(out),
    ]) == 2


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("attempted,passed,rejected", [
    (3, 3, 0), (2, 1, 0), (1, 2, -1), (-1, 0, -1),
])
def test_pixel_counts_close_even_for_partial_results(complete, attempted, passed, rejected):
    with pytest.raises(ValueError, match="pixel counts do not close"):
        classify_manifest(_manifest(2), {
            "complete_for_geometry_candidates": complete,
            "attempted": attempted, "passed": passed, "rejected": rejected,
        })
