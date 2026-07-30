from __future__ import annotations

import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from avengine.cli import main
from avengine.contracts.json_io import canonical_json_sha256, load_json
from avengine.m3.contracts import (
    validate_mapping_document,
    validate_material_database_document,
)
from avengine.m3.materials import compile_materials, validate_material_database
from avengine.m3.rlr_material_import import (
    RLRMaterialImportError,
    RLR_NATIVE_MATERIAL_DATABASE_SCHEMA,
    build_rlr_material_import_report,
    compile_rlr_semantic_material_documents,
    import_rlr_material_database,
    rlr_document_from_native_database,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPOSITORY_ROOT / "schemas/m3_acoustic_material_database_v2.schema.json"


def _curve(
    frequencies: tuple[float, ...],
    values: tuple[float, ...],
) -> list[float]:
    return [item for pair in zip(frequencies, values) for item in pair]


def _source_document() -> dict[str, object]:
    return {
        "materials": [
            {
                "name": "Brick, Painted",
                "absorption": _curve((125.0, 500.0), (0.02, 0.04)),
                "scattering": _curve((100.0, 1000.0, 8000.0), (0.1, 0.2, 0.3)),
                "transmission": _curve((80.0,), (0.01,)),
                "labels": [],
                "damping": _curve((22.0, 220.0, 2200.0), (1e-9, 1e-7, 1e-5)),
                "density": 998.6546630859375,
                "speed": 1483.9610595703125,
            },
            {
                "name": "Brick Painted",
                "absorption": _curve((125.0, 250.0), (0.03, 0.05)),
                "scattering": _curve((125.0, 250.0), (0.15, 0.25)),
                "transmission": _curve((125.0, 250.0), (0.02, 0.01)),
                "labels": ["wall", "wall"],
                "damping": _curve((20.0, 20000.0), (0.0, 0.001)),
                "density": 1.2,
                "speed": 343.0,
            },
        ]
    }


def _import(source: dict[str, object] | None = None) -> dict[str, object]:
    return import_rlr_material_database(
        _source_document() if source is None else source,
        database_id="unit_rlr_native_v1",
        source_description="unit RLR material fixture; not measurement-fitted",
    )


def _semantic_source_document() -> dict[str, object]:
    def material(name: str, labels: list[str], absorption: float) -> dict[str, object]:
        return {
            "name": name,
            "absorption": _curve((125.0, 500.0), (absorption, absorption + 0.01)),
            "scattering": _curve((125.0, 500.0), (0.1, 0.2)),
            "transmission": _curve((125.0, 500.0), (0.01, 0.0)),
            "labels": labels,
            "damping": _curve((20.0, 20_000.0), (0.0, 0.001)),
            "density": 1.2,
            "speed": 343.0,
        }

    return {
        "materials": [
            material("Default", ["default"], 0.10),
            material("Carpet", ["floor", "floor"], 0.20),
            material("Wood, Thick", ["chair", "table"], 0.30),
            material("Steel", ["major-appliance"], 0.40),
        ]
    }


def test_semantic_compiler_replays_substring_rule_and_discloses_default() -> None:
    source = _semantic_source_document()
    compiled = compile_rlr_semantic_material_documents(
        room_id="unit_mp3d",
        semantic_categories=(
            "chair",
            "floor",
            "table_lamp",
            "major_appliance",
            "mystery",
        ),
        source=source,
        database_id="unit_soundspaces_mp3d_v1",
        source_description="unit SoundSpaces/RLR MP3D material fixture",
        source_to_canonical={
            "matrix_row_major": [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            "source": "unit identity",
            "reviewed": True,
        },
    )

    assert validate_mapping_document(compiled.mapping, room_id="unit_mp3d") == []
    assert validate_material_database_document(compiled.database) == []
    runtime = compile_materials(
        compiled.mapping, compiled.database, room_id="unit_mp3d"
    )

    decisions = {
        decision["source_semantic_label"]: decision
        for decision in compiled.report["decisions"]
    }
    assert decisions["chair"]["assignment_kind"] == "official_substring_match"
    assert decisions["chair"]["selected_material_name"] == "Wood, Thick"
    assert decisions["chair"]["official_exact_label_present"] is True
    assert decisions["floor"]["assignment_kind"] == "official_substring_match"
    assert decisions["floor"]["selected_material_name"] == "Carpet"
    assert decisions["floor"]["official_match_score"] == 2
    assert decisions["floor"]["official_matched_labels"] == ["floor", "floor"]
    assert decisions["table_lamp"]["assignment_kind"] == (
        "official_substring_match"
    )
    assert decisions["table_lamp"]["selected_material_name"] == "Wood, Thick"
    assert decisions["table_lamp"]["official_matched_labels"] == ["table"]
    assert decisions["table_lamp"]["official_exact_label_present"] is False
    # Hyphen/underscore rewriting is deliberately not treated as an official
    # substring match.
    assert decisions["major_appliance"]["assignment_kind"] == "official_default"
    assert decisions["mystery"]["assignment_kind"] == "official_default"

    entries = {
        entry["source_material_name"]: entry for entry in compiled.mapping["entries"]
    }
    materials_by_key = {
        material["material_key"]: material
        for material in compiled.database["materials"]
    }
    imported = import_rlr_material_database(
        source,
        database_id="unit_soundspaces_mp3d_v1",
        source_description="unit SoundSpaces/RLR MP3D material fixture",
    )
    imported_by_key = {
        material["material_key"]: material for material in imported["materials"]
    }
    mapping_keys = [entry["material_key"] for entry in compiled.mapping["entries"]]
    assert len(compiled.database["materials"]) == len(decisions)
    assert len(mapping_keys) == len(set(mapping_keys)) == len(decisions)
    assert set(mapping_keys) == set(materials_by_key)
    for source_label in decisions:
        decision = decisions[source_label]
        alias = decision["rlr_category_name"]
        assert alias.startswith("avengine_rlr_alias_")
        runtime_material = materials_by_key[decision["runtime_material_key"]]
        source_material = imported_by_key[decision["selected_source_material_key"]]
        assert runtime_material["labels"] == [alias]
        assert entries[source_label]["material_key"] == decision["runtime_material_key"]
        assert entries[source_label]["category_name"] == alias
        assert decision["source_parameter_sha256"] == (
            decision["runtime_parameter_sha256"]
        )
        assert decision["parameters_preserved_exactly"] is True
        for field in (
            "absorption",
            "scattering",
            "transmission",
            "damping",
            "density",
            "speed",
        ):
            assert runtime_material[field] == source_material[field]
        # The runtime category is fully resolved; the semantic fallback remains
        # visible in the separate coverage report.
        assert entries[source_label]["fallback"] is False

    # Both categories select the same official Wood material but must be
    # represented by different native entries with unchanged parameters.
    assert decisions["chair"]["selected_source_material_key"] == (
        decisions["table_lamp"]["selected_source_material_key"]
    )
    assert decisions["chair"]["runtime_material_key"] != (
        decisions["table_lamp"]["runtime_material_key"]
    )

    # Mirror the pinned native greatest-substring-score lookup and its
    # one-to-one requirement.  Every category must select exactly one distinct
    # runtime database index; sharing a selected index is the native failure
    # this compiler contract prevents.
    resolved_runtime_indices: list[int] = []
    for category in runtime.categories_document["categories"]:
        lowered = category["category_name"].casefold()
        scores = [
            sum(
                1
                for label in material["labels"]
                if label.casefold() in lowered
            )
            for material in runtime.rlr_database["materials"]
        ]
        highest_score = max(scores)
        winners = [
            index for index, score in enumerate(scores) if score == highest_score
        ]
        assert highest_score == 1
        assert len(winners) == 1
        resolved_runtime_indices.append(winners[0])
    assert len(runtime.rlr_database["materials"]) == len(decisions)
    assert len(resolved_runtime_indices) == len(set(resolved_runtime_indices))

    assert compiled.report["runtime_one_to_one"] == {
        "semantic_category_count": 5,
        "runtime_material_count": 5,
        "unique_runtime_material_key_count": 5,
        "unique_runtime_label_count": 5,
        "mapping_and_database_order_identical": True,
        "passed": True,
    }
    assert compiled.report["coefficient_preservation"]["preserved_exactly"] is True
    assert compiled.report["coefficient_preservation"]["comparison_count"] == 5
    assert compiled.report["coefficient_preservation"]["imported_sha256"] == (
        compiled.report["coefficient_preservation"]["compiled_sha256"]
    )
    assert compiled.report["coverage"] == {
        "semantic_category_count": 5,
        "resolved_category_count": 5,
        "unresolved_category_count": 0,
        "official_substring_match_category_count": 3,
        "official_default_category_count": 2,
        "official_substring_match_category_fraction": 0.6,
        "official_default_category_fraction": 0.4,
    }


def test_semantic_compiler_matches_raw_hyphen_and_space_labels_but_maps_canonical_tokens(
) -> None:
    source = _semantic_source_document()
    source["materials"][2]["labels"].extend(["chopping-board", "side table"])

    compiled = compile_rlr_semantic_material_documents(
        room_id="unit_mp3d",
        semantic_categories=("chopping_board", "side_table"),
        raw_semantic_category_labels=("chopping-board", "side table"),
        source=source,
        database_id="unit_soundspaces_mp3d_v1",
        source_description="unit SoundSpaces/RLR MP3D material fixture",
        source_to_canonical={
            "matrix_row_major": [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            "source": "unit identity",
            "reviewed": True,
        },
    )

    decisions = {
        decision["canonical_semantic_category"]: decision
        for decision in compiled.report["decisions"]
    }
    entries = {
        entry["source_material_name"]: entry
        for entry in compiled.mapping["entries"]
    }
    for canonical, raw in (
        ("chopping_board", "chopping-board"),
        ("side_table", "side table"),
    ):
        decision = decisions[canonical]
        assert decision["raw_semantic_category_label"] == raw
        assert decision["source_semantic_label"] == canonical
        assert decision["assignment_kind"] == "official_substring_match"
        assert raw in decision["official_matched_labels"]
        assert decision["official_exact_label_present"] is True
        assert decision["official_default_applied"] is False
        assert entries[canonical]["source_material_name"] == canonical
        assert entries[canonical]["category_name"].startswith(
            "avengine_rlr_alias_"
        )
    assert compiled.report["coverage"][
        "official_substring_match_category_count"
    ] == 2
    assert compiled.report["coverage"]["official_default_category_count"] == 0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda source: source["materials"][0].__setitem__("labels", []),
            "exactly one material with the exact label 'default'",
        ),
        (
            lambda source: source["materials"][2]["labels"].extend(
                ["floor", "floor"]
            ),
            "official substring match tie for category 'floor'",
        ),
    ],
)
def test_semantic_compiler_rejects_missing_default_or_ambiguous_exact_labels(
    mutate: object,
    message: str,
) -> None:
    source = _semantic_source_document()
    mutate(source)

    with pytest.raises(RLRMaterialImportError, match=message):
        compile_rlr_semantic_material_documents(
            room_id="unit_mp3d",
            semantic_categories=("floor", "mystery"),
            source=source,
            database_id="unit_soundspaces_mp3d_v1",
            source_description="unit SoundSpaces/RLR MP3D material fixture",
            source_to_canonical={
                "matrix_row_major": [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "source": "unit identity",
                "reviewed": True,
            },
        )


def test_import_preserves_native_curves_empty_and_duplicate_labels() -> None:
    source = _source_document()
    database = _import(source)

    assert database["schema"] == RLR_NATIVE_MATERIAL_DATABASE_SCHEMA
    assert "bands_hz" not in database
    assert database["provenance"] == {
        "source": "unit RLR material fixture; not measurement-fitted",
        "confidence": 0.0,
        "material_semantics": "research_placeholder",
        "intended_use": "research_compiler_diagnostics",
    }
    assert database["materials"][0]["labels"] == []
    assert database["materials"][1]["labels"] == ["wall", "wall"]
    assert (
        database["materials"][0]["absorption"] == source["materials"][0]["absorption"]
    )
    assert (
        database["materials"][0]["scattering"] == source["materials"][0]["scattering"]
    )
    assert (
        database["materials"][0]["transmission"]
        == source["materials"][0]["transmission"]
    )
    assert database["materials"][0]["damping"] == source["materials"][0]["damping"]
    assert (
        database["materials"][0]["absorption"]
        is not source["materials"][0]["absorption"]
    )
    assert database["materials"][0]["labels"] is not source["materials"][0]["labels"]

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert list(Draft202012Validator(schema).iter_errors(database)) == []


def test_slug_collisions_get_stable_unique_keys() -> None:
    source = _source_document()

    first = _import(source)
    second = _import(copy.deepcopy(source))
    keys = [material["material_key"] for material in first["materials"]]

    assert keys == [material["material_key"] for material in second["materials"]]
    assert len(keys) == len(set(keys))
    assert all(key.startswith("brick_painted_") for key in keys)


def test_duplicate_names_still_get_stable_unique_keys() -> None:
    source = _source_document()
    source["materials"][1]["name"] = source["materials"][0]["name"]

    database = _import(source)
    keys = [material["material_key"] for material in database["materials"]]

    assert len(keys) == len(set(keys))
    assert keys[0].endswith("_1")
    assert keys[1].endswith("_2")


def test_roundtrip_is_canonically_exact() -> None:
    source = _source_document()
    database = _import(source)

    roundtrip = rlr_document_from_native_database(database)

    assert roundtrip == source
    assert roundtrip is not source
    assert canonical_json_sha256(roundtrip) == canonical_json_sha256(source)


def test_manual_validator_matches_v2_units_and_semantic_schema() -> None:
    wrong_units = _import()
    wrong_units["coefficient_units"]["speed"] = "centimeters_per_second"
    assert "database.coefficient_units must use the RLR-native units" in (
        validate_material_database(wrong_units)
    )

    wrong_semantics = _import()
    wrong_semantics["provenance"]["material_semantics"] = "controlled_canary"
    wrong_semantics["provenance"]["intended_use"] = (
        "synthetic_material_activation_canary"
    )
    errors = validate_material_database(wrong_semantics)
    assert any("material_semantics must be one of" in error for error in errors)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda source: source["materials"][0].__setitem__(
                "absorption", [125.0, 0.2, 250.0]
            ),
            "even number",
        ),
        (
            lambda source: source["materials"][0].__setitem__(
                "scattering", [125.0, 0.2, 100.0, 0.3]
            ),
            "strictly increasing",
        ),
        (
            lambda source: source["materials"][0].__setitem__(
                "transmission", [125.0, float("nan")]
            ),
            "must be finite",
        ),
        (
            lambda source: source["materials"][0].__setitem__(
                "absorption", [125.0, 1.01]
            ),
            "must be in \\[0, 1\\]",
        ),
        (
            lambda source: source["materials"][0].__setitem__(
                "damping", [125.0, -0.01]
            ),
            "must be non-negative",
        ),
        (
            lambda source: source["materials"][0].__setitem__("speed", False),
            "speed must be finite",
        ),
        (
            lambda source: source.__setitem__("unexpected", True),
            "unsupported fields",
        ),
    ],
)
def test_malformed_or_nonfinite_sources_fail_closed(
    mutation: object,
    message: str,
) -> None:
    source = _source_document()
    mutation(source)

    with pytest.raises(RLRMaterialImportError, match=message):
        _import(source)


def test_report_binds_file_hashes_statistics_and_non_calibration_claim(
    tmp_path: Path,
) -> None:
    source = _source_document()
    source_path = tmp_path / "rlr.json"
    source_path.write_text(
        json.dumps(source, ensure_ascii=False),
        encoding="utf-8",
    )
    database = _import(source)
    output_path = tmp_path / "native.json"
    output_path.write_text(
        json.dumps(database, ensure_ascii=False),
        encoding="utf-8",
    )

    report = build_rlr_material_import_report(
        source_path,
        database,
        output_path=output_path,
        source_uri="https://example.invalid/materials.json",
    )

    assert report["status"] == "pass"
    assert report["source"]["byte_size"] == source_path.stat().st_size
    assert len(report["source"]["sha256"]) == 64
    assert report["source"]["uri"].startswith("https://")
    assert report["output"]["canonical_sha256"] == canonical_json_sha256(database)
    assert report["output"]["byte_size"] == output_path.stat().st_size
    assert report["statistics"] == {
        "material_count": 2,
        "curve_count": 8,
        "pair_counts_by_field": {
            "absorption": 4,
            "scattering": 5,
            "transmission": 3,
            "damping": 5,
        },
        "total_pair_count": 17,
        "unique_frequency_grid_count": 6,
    }
    assert report["roundtrip"]["preserved_exactly"] is True
    assert (
        report["roundtrip"]["canonical_sha256"] == report["source"]["canonical_sha256"]
    )
    assert report["claims"] == {
        "frl_measurement_fitted": False,
        "physical_calibration": False,
    }


def test_report_rejects_database_from_a_different_source(tmp_path: Path) -> None:
    source = _source_document()
    source_path = tmp_path / "rlr.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    database = _import(source)
    database["materials"][0]["absorption"][1] = 0.9

    with pytest.raises(RLRMaterialImportError, match="does not canonically round-trip"):
        build_rlr_material_import_report(source_path, database)


def test_cli_imports_one_exclusive_replayable_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_document()
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    output = tmp_path / "imported"

    arguments = [
        "m3",
        "import-rlr-materials",
        "--source",
        str(source_path),
        "--database-id",
        "unit_cli_rlr_native_v1",
        "--source-description",
        "unit upstream RLR fixture; not physically calibrated",
        "--source-uri",
        "https://example.invalid/materials.json",
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    assert load_json(output / "materials.json")["schema"] == (
        RLR_NATIVE_MATERIAL_DATABASE_SCHEMA
    )
    report = load_json(output / "import_report.json")
    assert report["roundtrip"]["preserved_exactly"] is True
    assert report["output"]["path"] == "materials.json"
    assert report["claims"]["frl_measurement_fitted"] is False
    assert main(arguments) == 2
    assert "refusing to replace existing output" in capsys.readouterr().out
