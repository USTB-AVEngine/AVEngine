from __future__ import annotations

import copy
from pathlib import Path

import pytest

from avengine.contracts.json_io import load_json
from avengine.m3.contracts import (
    validate_mapping_document,
    validate_material_database_document,
)
from avengine.m3.materials import MaterialContractError, compile_materials


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples/m3/blender_custom"
ROOM_ID = "blender_custom_two_zone_v1"


def test_checked_in_material_inputs_are_strict_and_uniquely_match_rlr() -> None:
    mapping = load_json(EXAMPLE_ROOT / "mapping.json")
    low = load_json(EXAMPLE_ROOT / "materials_low.json")
    high = load_json(EXAMPLE_ROOT / "materials_high.json")

    assert validate_mapping_document(mapping, room_id=ROOM_ID) == []
    assert validate_material_database_document(low) == []
    assert validate_material_database_document(high) == []

    low_compiled = compile_materials(mapping, low, room_id=ROOM_ID)
    high_compiled = compile_materials(mapping, high, room_id=ROOM_ID)
    assert low_compiled.source_material_to_id == high_compiled.source_material_to_id
    assert [
        category["category_name"]
        for category in low_compiled.categories_document["categories"]
    ] == [
        category["category_name"]
        for category in high_compiled.categories_document["categories"]
    ]
    assert all(
        category["rlr_match"]["tie_count"] == 1
        and category["rlr_match"]["score"] == 1
        and category["fallback"] is False
        for category in low_compiled.categories_document["categories"]
    )
    assert low_compiled.rlr_database != high_compiled.rlr_database


def test_controlled_pair_uses_declared_edt_capable_absorption_values() -> None:
    """Keep the tracked canary distinct without invalidating true EDT.

    A 0.90 exploratory condition made the direct impulse dominate the first
    10 dB of the Schroeder curve.  The tracked 0.02/0.60 pair retains a real
    0 to -10 dB fit; the metric implementation must not silently substitute a
    direct-removed T10 estimate to make a more extreme fixture pass.
    """

    low = load_json(EXAMPLE_ROOT / "materials_low.json")
    high = load_json(EXAMPLE_ROOT / "materials_high.json")

    assert {
        tuple(material["absorption"])
        for material in low["materials"]
    } == {(0.02, 0.02, 0.02, 0.02)}
    assert {
        tuple(material["absorption"])
        for material in high["materials"]
    } == {(0.6, 0.6, 0.6, 0.6)}
    assert low["provenance"] == high["provenance"]


def test_mapping_rejects_noncontiguous_ids_and_unreviewed_production_is_visible() -> None:
    mapping = load_json(EXAMPLE_ROOT / "mapping.json")
    mapping["entries"][0]["material_id"] = 9
    mapping["source_to_canonical"]["reviewed"] = False

    errors = validate_mapping_document(mapping, room_id=ROOM_ID)

    assert any("contiguous" in error for error in errors)
    # Research mappings may deliberately remain unreviewed; package-mode
    # validation, rather than this reusable mapping contract, blocks promotion.
    assert not any("reviewed" in error for error in errors)


def test_rlr_substring_tie_is_a_hard_error() -> None:
    mapping = load_json(EXAMPLE_ROOT / "mapping.json")
    database = load_json(EXAMPLE_ROOT / "materials_low.json")
    database["materials"][1]["labels"] = ["avm3_floor"]

    with pytest.raises(MaterialContractError, match="does not uniquely match"):
        compile_materials(mapping, database, room_id=ROOM_ID)


def test_rlr_substring_only_match_is_rejected_before_cpp_ingestion() -> None:
    mapping = load_json(EXAMPLE_ROOT / "mapping.json")
    database = load_json(EXAMPLE_ROOT / "materials_low.json")
    database["materials"][0]["labels"] = ["avm3_floor"]

    with pytest.raises(MaterialContractError, match="exact lower-case label"):
        compile_materials(mapping, database, room_id=ROOM_ID)


def test_material_database_rejects_band_length_and_nonfinite_values() -> None:
    database = load_json(EXAMPLE_ROOT / "materials_low.json")
    malformed = copy.deepcopy(database)
    malformed["materials"][0]["absorption"] = [0.1]
    malformed["materials"][1]["speed"] = float("nan")

    errors = validate_material_database_document(malformed)

    assert any("exactly 4 band values" in error for error in errors)
    assert any("speed" in error and "finite" in error for error in errors)
