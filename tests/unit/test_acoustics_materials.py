from __future__ import annotations

import copy
from pathlib import Path

import pytest

from avengine.contracts.json_io import load_json
from avengine.acoustics.contracts import (
    validate_mapping_document,
    validate_material_database_document,
)
from avengine.acoustics.materials import MaterialContractError, compile_materials
from avengine.acoustics.materials import (
    MATERIAL_PROFILE_SCHEMA,
    resolve_material_profile,
    validate_material_database,
    validate_material_mapping,
    validate_material_profile,
)


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


def _material_profile(
    *,
    global_override: dict[str, object] | None = None,
    material_overrides: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    profile: dict[str, object] = {
        "schema": MATERIAL_PROFILE_SCHEMA,
        "profile_id": "unit_explicit_controls_v1",
        "room_id": ROOM_ID,
        "material_overrides": (
            [] if material_overrides is None else material_overrides
        ),
    }
    if global_override is not None:
        profile["global_override"] = global_override
    return profile


def test_profile_resolution_broadcasts_global_and_applies_per_material_last() -> None:
    mapping = load_json(EXAMPLE_ROOT / "mapping.json")
    database = load_json(EXAMPLE_ROOT / "materials_low.json")
    profile = _material_profile(
        global_override={
            "absorption": 0.2,
            "scattering": 0.15,
            "density": 1.3,
        },
        material_overrides=[
            {
                "selector": {"source_material_name": "FloorWarmGray"},
                "absorption": [0.1, 0.2, 0.3, 0.4],
                "speed": 340.0,
            }
        ],
    )
    original_mapping = copy.deepcopy(mapping)
    original_database = copy.deepcopy(database)
    original_profile = copy.deepcopy(profile)

    resolved = resolve_material_profile(
        mapping, database, profile, room_id=ROOM_ID
    )

    assert mapping == original_mapping
    assert database == original_database
    assert profile == original_profile
    assert resolved.effective_mapping == mapping
    assert resolved.effective_mapping is not mapping
    effective_by_key = {
        material["material_key"]: material
        for material in resolved.effective_database["materials"]
    }
    floor = effective_by_key["floor_extreme_c91f"]
    assert floor["absorption"] == [0.1, 0.2, 0.3, 0.4]
    assert floor["scattering"] == [0.15] * 4
    assert floor["density"] == 1.3
    assert floor["speed"] == 340.0
    assert effective_by_key["wall_extreme_d42a"]["absorption"] == [0.2] * 4
    assert effective_by_key["wall_extreme_d42a"]["speed"] == 343.0
    assert resolved.effective_database["database_id"].startswith(
        f"{database['database_id']}__profile__{profile['profile_id']}__"
    )
    assert profile["profile_id"] in resolved.effective_database["provenance"][
        "source"
    ]
    assert validate_material_mapping(resolved.effective_mapping, room_id=ROOM_ID) == []
    assert validate_material_database(resolved.effective_database) == []
    assert resolved.report["status"] == "pass"
    assert resolved.report["precedence"] == [
        "base_database",
        "global_override",
        "material_override",
    ]
    assert resolved.report["input_hashes"]["mapping_sha256"] == resolved.report[
        "output_hashes"
    ]["mapping_sha256"]
    assert resolved.report["selector_resolutions"] == [
        {
            "override_index": 0,
            "selector": {"source_material_name": "FloorWarmGray"},
            "material_key": "floor_extreme_c91f",
            "source_material_names": ["FloorWarmGray"],
        }
    ]


def test_profile_material_key_selector_can_override_all_physical_fields() -> None:
    mapping = load_json(EXAMPLE_ROOT / "mapping.json")
    database = load_json(EXAMPLE_ROOT / "materials_low.json")
    profile = _material_profile(
        material_overrides=[
            {
                "selector": {"material_key": "wall_extreme_d42a"},
                "transmission": 0.25,
                "damping": 0.5,
                "density": 2.0,
                "speed": 320.0,
            }
        ]
    )

    resolved = resolve_material_profile(
        mapping, database, profile, room_id=ROOM_ID
    )
    wall = next(
        material
        for material in resolved.effective_database["materials"]
        if material["material_key"] == "wall_extreme_d42a"
    )

    assert wall["transmission"] == [0.25] * 4
    assert wall["damping"] == [0.5] * 4
    assert wall["density"] == 2.0
    assert wall["speed"] == 320.0
    assert resolved.report["materials"][1]["field_lineage"]["speed"] == (
        "profile.material_overrides[0]"
    )


@pytest.mark.parametrize(
    ("material_overrides", "message"),
    [
        (
            [
                {
                    "selector": {"material_key": "not_in_the_room"},
                    "absorption": 0.2,
                }
            ],
            "unknown material_key",
        ),
        (
            [
                {
                    "selector": {"source_material_name": "MissingSlot"},
                    "absorption": 0.2,
                }
            ],
            "unknown source_material_name",
        ),
        (
            [
                {
                    "selector": {"material_key": "floor_extreme_c91f"},
                    "absorption": 0.2,
                },
                {
                    "selector": {"material_key": "floor_extreme_c91f"},
                    "scattering": 0.2,
                },
            ],
            "duplicates",
        ),
        (
            [
                {
                    "selector": {"material_key": "floor_extreme_c91f"},
                    "absorption": 0.2,
                },
                {
                    "selector": {"source_material_name": "FloorWarmGray"},
                    "scattering": 0.2,
                },
            ],
            "conflicts",
        ),
    ],
)
def test_profile_rejects_unknown_duplicate_and_conflicting_selectors(
    material_overrides: list[dict[str, object]], message: str
) -> None:
    mapping = load_json(EXAMPLE_ROOT / "mapping.json")
    database = load_json(EXAMPLE_ROOT / "materials_low.json")
    profile = _material_profile(material_overrides=material_overrides)

    with pytest.raises(MaterialContractError, match=message):
        resolve_material_profile(mapping, database, profile, room_id=ROOM_ID)


def test_profile_rejects_source_selector_when_material_key_is_shared() -> None:
    mapping = load_json(EXAMPLE_ROOT / "mapping.json")
    database = load_json(EXAMPLE_ROOT / "materials_low.json")
    floor_key = mapping["entries"][0]["material_key"]
    wall_category = mapping["entries"][1]["category_name"]
    mapping["entries"][1]["material_key"] = floor_key
    database["materials"][0]["labels"].append(wall_category)
    database["materials"][1]["labels"] = ["unused_wall_material"]
    assert compile_materials(mapping, database, room_id=ROOM_ID)
    profile = _material_profile(
        material_overrides=[
            {
                "selector": {"source_material_name": "FloorWarmGray"},
                "absorption": 0.2,
            }
        ]
    )

    with pytest.raises(MaterialContractError, match="is shared by source materials"):
        resolve_material_profile(mapping, database, profile, room_id=ROOM_ID)


def test_profile_contract_rejects_wrong_band_count_and_nonfinite_scalar() -> None:
    wrong_bands = _material_profile(global_override={"absorption": [0.1, 0.2]})
    nonfinite = _material_profile(global_override={"damping": float("nan")})
    overflowing = _material_profile(global_override={"density": 10**4000})

    assert any(
        "exactly 4 band values" in error
        for error in validate_material_profile(
            wrong_bands, room_id=ROOM_ID, band_count=4
        )
    )
    assert any(
        "must be finite" in error
        for error in validate_material_profile(
            nonfinite, room_id=ROOM_ID, band_count=4
        )
    )
    assert any(
        "must be finite" in error
        for error in validate_material_profile(
            overflowing, room_id=ROOM_ID, band_count=4
        )
    )


def test_profile_effective_database_id_binds_all_resolution_inputs() -> None:
    mapping = load_json(EXAMPLE_ROOT / "mapping.json")
    database = load_json(EXAMPLE_ROOT / "materials_low.json")
    profile = _material_profile(global_override={"absorption": 0.2})

    baseline = resolve_material_profile(
        mapping, database, profile, room_id=ROOM_ID
    )
    changed_database = copy.deepcopy(database)
    changed_database["materials"][0]["absorption"][0] = 0.021
    database_result = resolve_material_profile(
        mapping, changed_database, profile, room_id=ROOM_ID
    )
    changed_mapping = copy.deepcopy(mapping)
    changed_mapping["mapping_id"] += "_same_semantics_different_bytes"
    mapping_result = resolve_material_profile(
        changed_mapping, database, profile, room_id=ROOM_ID
    )

    assert len(
        {
            baseline.effective_database["database_id"],
            database_result.effective_database["database_id"],
            mapping_result.effective_database["database_id"],
        }
    ) == 3


def test_profile_contract_accepts_global_only_and_requires_an_override() -> None:
    global_only = _material_profile(global_override={"absorption": 0.4})
    global_only.pop("material_overrides")
    empty = _material_profile()
    omitted_both = copy.deepcopy(empty)
    omitted_both.pop("material_overrides")

    assert validate_material_profile(
        global_only, room_id=ROOM_ID, band_count=4
    ) == []
    mapping = load_json(EXAMPLE_ROOT / "mapping.json")
    database = load_json(EXAMPLE_ROOT / "materials_low.json")
    resolved = resolve_material_profile(
        mapping, database, global_only, room_id=ROOM_ID
    )
    assert all(
        material["absorption"] == [0.4] * 4
        for material in resolved.effective_database["materials"]
    )
    assert validate_material_profile(empty, room_id=ROOM_ID, band_count=4)
    assert validate_material_profile(
        omitted_both, room_id=ROOM_ID, band_count=4
    )


def test_profile_cannot_inherit_reviewed_physical_claim_after_modification() -> None:
    mapping = load_json(EXAMPLE_ROOT / "mapping.json")
    database = load_json(EXAMPLE_ROOT / "materials_low.json")
    database["provenance"].update(
        {
            "material_semantics": "reviewed_physical",
            "intended_use": "reviewed_production_profile",
        }
    )
    profile = _material_profile(global_override={"absorption": 0.2})

    resolved = resolve_material_profile(
        mapping, database, profile, room_id=ROOM_ID
    )

    assert resolved.effective_database["provenance"]["material_semantics"] == (
        "research_placeholder"
    )
    assert resolved.effective_database["provenance"]["intended_use"] == (
        "research_compiler_diagnostics"
    )
    assert resolved.effective_database["provenance"]["confidence"] == 0.0
    assert all(
        material["confidence"] == 0.0
        for material in resolved.effective_database["materials"]
    )
    assert resolved.report["material_semantics"] == {
        "base": "reviewed_physical",
        "effective": "research_placeholder",
        "profile_grants_physical_review": False,
    }
