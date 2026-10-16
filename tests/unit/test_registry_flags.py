from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from avengine.contracts.json_io import load_json
from avengine.capture.source_contracts import (
    ALL_FLAG_IDS,
    AND_AGGREGATED_FLAGS,
    OR_AGGREGATED_FLAGS,
    PAIR_FLAG_IDS,
    SOURCE_FLAG_IDS,
)
from avengine.registry.flags import (
    LEGACY_THRESHOLDS,
    aggregate_legacy_status,
    evaluate_legacy_flags,
    legacy_flag_access,
    load_legacy_flag_registry,
    provider_assessment,
    validate_legacy_flag_registry,
)
from avengine.registry.registry import bind_content_hash


ROOT = Path(__file__).resolve().parents[2]
FLAG_REGISTRY = ROOT / "examples" / "registry" / "registries" / "legacy_m5_1_flags_v1.json"
LEGACY_SOURCE_MANIFEST = ROOT / "examples" / "capture" / "legacy_apartment" / "source_manifest.json"


def test_flag_registry_is_an_exact_public_view_of_m5_1_v1() -> None:
    registry = load_legacy_flag_registry(FLAG_REGISTRY)
    access = legacy_flag_access(registry)
    assert access.source_flag_ids == SOURCE_FLAG_IDS
    assert access.pair_flag_ids == PAIR_FLAG_IDS
    assert tuple(item.flag_id for item in access.definitions) == ALL_FLAG_IDS
    assert dict(access.thresholds) == dict(LEGACY_THRESHOLDS)
    assert {
        item.flag_id for item in access.definitions if item.clip_aggregation == "or"
    } == set(OR_AGGREGATED_FLAGS)
    assert {
        item.flag_id for item in access.definitions if item.clip_aggregation == "and"
    } == set(AND_AGGREGATED_FLAGS)


def test_flag_registry_rejects_threshold_or_order_drift() -> None:
    registry = load_json(FLAG_REGISTRY)
    registry["thresholds"]["passes_close_to_mic_m"] = 1.01
    registry = bind_content_hash(registry)
    assert any("frozen M5.1 v1 thresholds" in item for item in validate_legacy_flag_registry(registry))

    registry = load_json(FLAG_REGISTRY)
    registry["flags"][0], registry["flags"][1] = registry["flags"][1], registry["flags"][0]
    registry = bind_content_hash(registry)
    assert any("ID membership and order" in item for item in validate_legacy_flag_registry(registry))


@pytest.mark.parametrize("flag_id", sorted(OR_AGGREGATED_FLAGS))
def test_or_aggregation_preserves_unknown(flag_id: str) -> None:
    assert aggregate_legacy_status(flag_id, ["absent", "not_evaluated"]) == "not_evaluated"
    assert aggregate_legacy_status(flag_id, ["not_evaluated", "present"]) == "present"
    assert aggregate_legacy_status(flag_id, ["absent", "absent"]) == "absent"


@pytest.mark.parametrize("flag_id", sorted(AND_AGGREGATED_FLAGS))
def test_and_aggregation_preserves_unknown(flag_id: str) -> None:
    assert aggregate_legacy_status(flag_id, ["present", "not_evaluated"]) == "not_evaluated"
    assert aggregate_legacy_status(flag_id, ["not_evaluated", "absent"]) == "absent"
    assert aggregate_legacy_status(flag_id, ["present", "present"]) == "present"


def test_provider_missing_fact_becomes_not_evaluated_never_absent() -> None:
    assessment = provider_assessment(
        flag_id="occluded_by_wall",
        scope="per_source",
        status=None,
        reason_code="raycast_missing",
        reason="No room raycast provider was available.",
        evidence=[
            {
                "evidence_id": "missing_raycast",
                "kind": "missing_dependency",
                "uri": "memory://missing",
                "sha256": "0" * 64,
                "summary": "No raycast facts.",
            }
        ],
    )
    assert assessment["status"] == "not_evaluated"
    assert assessment["value"] is None


def test_registry_evaluator_matches_checked_in_m5_1_source_pair_and_clip_statuses() -> None:
    manifest = load_json(LEGACY_SOURCE_MANIFEST)
    positions = {
        source["source_id"]: [
            keyframe["position_m"] for keyframe in source["trajectory"]["keyframes"]
        ]
        for source in manifest["sources"]
    }
    report = evaluate_legacy_flags(
        observer_position_m=manifest["observer"]["position_m"],
        observer_yaw_deg=manifest["observer"]["yaw_deg"],
        fps=manifest["clip"]["fps_num"] / manifest["clip"]["fps_den"],
        positions_by_source=positions,
        visibility_facts_by_source=None,
    )
    for source in manifest["sources"]:
        source_id = source["source_id"]
        for flag_id in SOURCE_FLAG_IDS:
            assert report["source_flags"][source_id][flag_id]["status"] == source["flags"][flag_id]["status"]
            assert report["source_flags"][source_id][flag_id]["value"] is source["flags"][flag_id]["value"]
    assert report["pair_flags"][0]["flags"]["sources_pass_each_other"]["status"] == manifest["relationships"][0]["flags"]["sources_pass_each_other"]["status"]
    for flag_id in ALL_FLAG_IDS:
        assert report["clip_flags"][flag_id]["status"] == manifest["clip_flags"][flag_id]["status"]
        assert report["clip_flags"][flag_id]["value"] is manifest["clip_flags"][flag_id]["value"]


def test_visibility_provider_facts_are_evaluated_and_missing_source_stays_unknown() -> None:
    positions = {
        "source0": [[-1.0, 0.0, -2.0], [-0.5, 0.0, -2.0], [0.5, 0.0, -2.0], [1.0, 0.0, -2.0]],
        "source1": [[2.0, 0.0, -3.0], [2.0, 0.0, -3.0], [2.0, 0.0, -3.0], [2.0, 0.0, -3.0]],
    }
    report = evaluate_legacy_flags(
        observer_position_m=[0.0, 0.0, 0.0],
        observer_yaw_deg=0.0,
        fps=15.0,
        positions_by_source=positions,
        visibility_facts_by_source={
            "source0": {
                "in_fov_by_frame": [True, True, False, False],
                "occlusion_by_frame": ["clear", "furniture", "clear", "wall"],
            }
        },
    )
    source0 = report["source_flags"]["source0"]
    assert source0["occluded_by_furniture"]["status"] == "present"
    assert source0["occluded_by_wall"]["status"] == "present"
    assert source0["never_occluded"]["status"] == "absent"
    assert source0["leaves_camera_fov"]["status"] == "present"
    assert source0["stays_in_camera_fov"]["status"] == "absent"
    assert all(
        report["source_flags"]["source1"][flag_id]["status"] == "not_evaluated"
        for flag_id in (
            "occluded_by_furniture",
            "occluded_by_wall",
            "never_occluded",
            "leaves_camera_fov",
            "stays_in_camera_fov",
        )
    )

