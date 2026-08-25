from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from avengine.contracts.json_io import load_json
from avengine.rooms.qualification import validate_qualification_report
from avengine.rooms.rooms import (
    find_room_record,
    index_room_records,
    validate_room_registry,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ROOM_EXAMPLES = REPOSITORY_ROOT / "examples/m6/rooms"
REGISTRY_PATH = ROOM_EXAMPLES / "room_registry.json"


def test_checked_in_registry_has_four_complementary_room_providers() -> None:
    registry = load_json(REGISTRY_PATH)

    assert validate_room_registry(registry) == []
    assert len(registry["records"]) == 4
    assert {record["provider_id"] for record in registry["records"]} == {
        "blender_custom",
        "replica_cad",
        "legacy_ue_apartment",
        "matterport3d",
    }
    assert len(index_room_records(registry)) == 4


def test_room_lineage_keeps_geometry_layout_acoustics_and_episode_separate() -> None:
    registry = load_json(REGISTRY_PATH)

    for record in registry["records"]:
        lineage = record["lineage"]
        assert lineage["room_geometry_id"] != lineage["acoustic_profile_id"]
        assert lineage["layout_variant_id"]
        assert lineage["episode_layout_id"]


def test_mp3d_raw_and_declared_proxy_are_independent_representations() -> None:
    registry = load_json(REGISTRY_PATH)
    record = find_room_record(registry, "habitat_mp3d_example_17DRP5sb8fy")
    representations = {
        item["role"]: item for item in record["acoustic_representations"]
    }

    raw = representations["raw_source"]
    derived = representations["derived_proxy"]
    assert raw["immutable"] is True
    assert derived["representation_id"] != raw["representation_id"]
    assert derived["derived_from"] == raw["representation_id"]
    assert len(record["qualification_reports"]) == 2


def test_aabb_cannot_be_promoted_to_acoustic_authority() -> None:
    registry = load_json(REGISTRY_PATH)
    mutated = deepcopy(registry)
    legacy = next(
        record
        for record in mutated["records"]
        if record["provider_id"] == "legacy_ue_apartment"
    )
    diagnostic = next(
        item
        for item in legacy["acoustic_representations"]
        if item["geometry_kind"] == "debug_aabb_proxy"
    )
    diagnostic["role"] = "production_authority"

    errors = validate_room_registry(mutated)

    assert any("AABB" in error or "diagnostic_only" in error for error in errors)


def test_public_registry_contains_no_private_absolute_resource_paths() -> None:
    registry = load_json(REGISTRY_PATH)

    for record in registry["records"]:
        for resource in record["resources"]:
            location = resource["location"]
            assert not any(
                isinstance(value, str) and value.startswith("/data/")
                for value in location.values()
            )
        for report in record["qualification_reports"]:
            assert not Path(report["path"]).is_absolute()


def test_all_checked_in_room_reports_validate_without_overall_status() -> None:
    report_paths = sorted((ROOM_EXAMPLES / "qualification").glob("*.json"))

    assert len(report_paths) == 5
    for path in report_paths:
        report = load_json(path)
        assert "overall_status" not in report
        assert validate_qualification_report(report) == [], path


def test_room_report_cannot_hide_nonpass_dimension_behind_admission() -> None:
    report = load_json(
        ROOM_EXAMPLES / "qualification/blender_custom_two_zone.json"
    )
    report["dataset_admission"] = True
    report["admission_blockers"] = []

    errors = validate_qualification_report(report)

    assert any("dataset_admission=true" in error for error in errors)
