from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
import unicodedata

from avengine.contracts.json_io import (
    canonical_json_sha256,
)


RLR_NATIVE_MATERIAL_DATABASE_SCHEMA = "avengine_m3_acoustic_material_database_v2"
RLR_MATERIAL_IMPORT_REPORT_SCHEMA = "avengine_m3_rlr_material_import_report_v1"
RLR_SEMANTIC_MATERIAL_COVERAGE_SCHEMA = (
    "avengine_m3_rlr_semantic_material_coverage_v1"
)

_CURVE_FIELDS = ("absorption", "scattering", "transmission", "damping")
_PARAMETER_FIELDS = (*_CURVE_FIELDS, "density", "speed")
_SOURCE_MATERIAL_FIELDS = {
    "name",
    "absorption",
    "scattering",
    "transmission",
    "labels",
    "damping",
    "density",
    "speed",
}
_NATIVE_MATERIAL_FIELDS = {
    "material_key",
    *_SOURCE_MATERIAL_FIELDS,
    "source",
    "confidence",
}
_COEFFICIENT_UNITS = {
    "absorption": ("frequency_hz_fraction_of_incident_sound_pressure_interleaved"),
    "scattering": ("frequency_hz_fraction_of_incident_sound_pressure_interleaved"),
    "transmission": ("frequency_hz_fraction_of_incident_sound_pressure_interleaved"),
    "damping": "frequency_hz_decibels_per_meter_interleaved",
    "density": "kilograms_per_cubic_meter",
    "speed": "meters_per_second",
}


class RLRMaterialImportError(ValueError):
    """A fail-closed RLR material import or round-trip contract error."""

    def __init__(self, errors: str | Iterable[str]):
        if isinstance(errors, str):
            self.errors = [errors]
        else:
            self.errors = list(errors)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class CompiledRLRSemanticMaterials:
    """RLR-native database plus an explicit semantic-category assignment."""

    mapping: dict[str, Any]
    database: dict[str, Any]
    report: dict[str, Any]


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _validate_curve(
    value: Any,
    *,
    location: str,
    bounded_fraction: bool,
    errors: list[str],
) -> None:
    if not isinstance(value, list):
        errors.append(f"{location} must be an interleaved JSON array")
        return
    if len(value) < 2 or len(value) % 2:
        errors.append(
            f"{location} must contain an even number of values and at least one "
            "frequency/value pair"
        )
        return

    previous_frequency: float | None = None
    for pair_index in range(len(value) // 2):
        frequency = value[2 * pair_index]
        coefficient = value[2 * pair_index + 1]
        frequency_location = f"{location}[{2 * pair_index}]"
        coefficient_location = f"{location}[{2 * pair_index + 1}]"
        if not _is_finite_number(frequency):
            errors.append(f"{frequency_location} frequency must be finite")
        else:
            numeric_frequency = float(frequency)
            if numeric_frequency <= 0:
                errors.append(f"{frequency_location} frequency must be positive")
            if (
                previous_frequency is not None
                and numeric_frequency <= previous_frequency
            ):
                errors.append(
                    f"{frequency_location} frequency must be strictly increasing"
                )
            previous_frequency = numeric_frequency

        if not _is_finite_number(coefficient):
            errors.append(f"{coefficient_location} value must be finite")
            continue
        numeric_coefficient = float(coefficient)
        if bounded_fraction and not 0.0 <= numeric_coefficient <= 1.0:
            errors.append(f"{coefficient_location} value must be in [0, 1]")
        if not bounded_fraction and numeric_coefficient < 0.0:
            errors.append(f"{coefficient_location} value must be non-negative")


def _validate_source_material(
    material: Any,
    *,
    index: int,
    errors: list[str],
) -> None:
    location = f"materials[{index}]"
    if not isinstance(material, Mapping):
        errors.append(f"{location} must be an object")
        return
    unexpected = sorted(set(material) - _SOURCE_MATERIAL_FIELDS)
    missing = sorted(_SOURCE_MATERIAL_FIELDS - set(material))
    if missing:
        errors.append(f"{location} is missing fields: {', '.join(missing)}")
    if unexpected:
        errors.append(
            f"{location} contains unsupported fields: {', '.join(unexpected)}"
        )

    name = material.get("name")
    if not isinstance(name, str) or not name:
        errors.append(f"{location}.name must be a non-empty string")
    labels = material.get("labels")
    if not isinstance(labels, list):
        errors.append(f"{location}.labels must be a JSON array")
    else:
        for label_index, label in enumerate(labels):
            if not isinstance(label, str) or not label:
                errors.append(
                    f"{location}.labels[{label_index}] must be a non-empty string"
                )

    for field in _CURVE_FIELDS:
        _validate_curve(
            material.get(field),
            location=f"{location}.{field}",
            bounded_fraction=field != "damping",
            errors=errors,
        )
    for field in ("density", "speed"):
        physical_value = material.get(field)
        if not _is_finite_number(physical_value):
            errors.append(f"{location}.{field} must be finite")
        elif float(physical_value) <= 0.0:
            errors.append(f"{location}.{field} must be positive")


def _validate_rlr_document(source: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(source, Mapping):
        return ["RLR material document must be an object"]
    unexpected = sorted(set(source) - {"materials"})
    missing = sorted({"materials"} - set(source))
    if missing:
        errors.append("RLR material document is missing field: materials")
    if unexpected:
        errors.append(
            "RLR material document contains unsupported fields: "
            + ", ".join(unexpected)
        )
    materials = source.get("materials")
    if not isinstance(materials, list) or not materials:
        errors.append("RLR material document materials must be a non-empty array")
        return errors
    for index, material in enumerate(materials):
        _validate_source_material(material, index=index, errors=errors)
    return errors


def _slug(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii").lower()
    value = re.sub(r"[^a-z0-9]+", "_", ascii_name).strip("_")
    return (value or "material")[:64].rstrip("_") or "material"


def _stable_material_keys(names: list[str]) -> list[str]:
    bases = [_slug(name) for name in names]
    base_counts = Counter(bases)
    candidates = [
        (
            base
            if base_counts[base] == 1
            else f"{base}_{hashlib.sha256(name.encode('utf-8')).hexdigest()[:10]}"
        )
        for name, base in zip(names, bases)
    ]
    occurrence_counts: dict[str, int] = {}
    total_counts = Counter(candidates)
    keys: list[str] = []
    for candidate in candidates:
        occurrence_counts[candidate] = occurrence_counts.get(candidate, 0) + 1
        if total_counts[candidate] == 1:
            keys.append(candidate)
        else:
            keys.append(f"{candidate}_{occurrence_counts[candidate]}")
    return keys


def import_rlr_material_database(
    source: Mapping[str, Any],
    *,
    database_id: str,
    version: str = "1",
    source_description: str,
) -> dict[str, Any]:
    """Import an RLR-native material document without resampling any curve."""

    argument_errors: list[str] = []
    for name, value in (
        ("database_id", database_id),
        ("version", version),
        ("source_description", source_description),
    ):
        if not isinstance(value, str) or not value:
            argument_errors.append(f"{name} must be a non-empty string")
    argument_errors.extend(_validate_rlr_document(source))
    if argument_errors:
        raise RLRMaterialImportError(argument_errors)

    source_materials = source["materials"]
    material_keys = _stable_material_keys(
        [material["name"] for material in source_materials]
    )
    materials: list[dict[str, Any]] = []
    for material_key, source_material in zip(material_keys, source_materials):
        imported = {
            "material_key": material_key,
            **copy.deepcopy(dict(source_material)),
            "source": source_description,
            "confidence": 0.0,
        }
        materials.append(imported)

    return {
        "schema": RLR_NATIVE_MATERIAL_DATABASE_SCHEMA,
        "database_id": database_id,
        "version": version,
        "coefficient_units": copy.deepcopy(_COEFFICIENT_UNITS),
        "provenance": {
            "source": source_description,
            "confidence": 0.0,
            "material_semantics": "research_placeholder",
            "intended_use": "research_compiler_diagnostics",
        },
        "materials": materials,
    }


def _coefficient_projection(database: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(material["material_key"]): {
            field: copy.deepcopy(material[field]) for field in _PARAMETER_FIELDS
        }
        for material in database["materials"]
    }


def _material_parameter_projection(material: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: copy.deepcopy(material[field]) for field in _PARAMETER_FIELDS
    }


def _runtime_alias(
    *,
    room_id: str,
    source_semantic_label: str,
    selected_material_key: str,
    occupied_labels: set[str],
    occupied_categories: set[str],
) -> str:
    """Create an exact routing label with no pre-existing substring match."""

    for nonce in range(10_000):
        digest = hashlib.sha256(
            (
                f"{room_id}\0{source_semantic_label}\0"
                f"{selected_material_key}\0{nonce}"
            ).encode("utf-8")
        ).hexdigest()
        candidate = f"avengine_rlr_alias_{digest[:24]}"
        if candidate in occupied_labels or candidate in occupied_categories:
            continue
        lowered_candidate = candidate.casefold()
        if not any(
            label.casefold() in lowered_candidate for label in occupied_labels
        ):
            return candidate
    raise RLRMaterialImportError(
        f"unable to allocate a collision-free RLR alias for "
        f"{source_semantic_label!r}"
    )


def compile_rlr_semantic_material_documents(
    *,
    room_id: str,
    semantic_categories: Iterable[str],
    raw_semantic_category_labels: Iterable[str] | None = None,
    source: Mapping[str, Any],
    database_id: str,
    source_description: str,
    source_to_canonical: Mapping[str, Any],
    version: str = "1",
) -> CompiledRLRSemanticMaterials:
    """Replay official RLR label matching, then emit exact runtime aliases.

    The public RLR rule lower-cases each source category label and selects the
    material with the greatest number of label substring matches.  The raw
    labels are kept separate from AVEngine's canonical semantic-category
    tokens so punctuation normalization cannot change official material
    selection.  Duplicate source labels count independently, as they do in the
    upstream data.  A zero-score category is assigned to the one material
    carrying the exact ``default`` label; a highest-score tie fails closed.

    AVEngine's runtime ingestion is intentionally stricter than that public
    selection rule.  After resolving the official winner offline, this compiler
    creates one distinct runtime material entry per semantic category.  Its
    only label is a collision-free exact alias and its mapping entry points to
    its own unique material key.  This satisfies the pinned native ingestion
    contract that every runtime category resolve one-to-one to a material
    database entry.

    Every derived entry copies all coefficient curves and physical scalars
    byte-canonically from its selected official source material.  Official
    identity and matching evidence remain explicit in the coverage report;
    use of the official ``Default`` is therefore never a silent fallback.
    """

    if not isinstance(room_id, str) or not room_id:
        raise RLRMaterialImportError("room_id must be a non-empty string")
    if not isinstance(source_to_canonical, Mapping):
        raise RLRMaterialImportError("source_to_canonical must be an object")
    categories = list(semantic_categories)
    if not categories:
        raise RLRMaterialImportError("semantic_categories must be non-empty")
    invalid_categories = [
        category
        for category in categories
        if not isinstance(category, str)
        or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", category) is None
    ]
    if invalid_categories:
        raise RLRMaterialImportError(
            "semantic categories must be unique lower-case AVEngine category "
            f"tokens: {invalid_categories!r}"
        )
    if len(categories) != len(set(categories)):
        raise RLRMaterialImportError("semantic categories must be unique")
    raw_category_labels = (
        list(categories)
        if raw_semantic_category_labels is None
        else list(raw_semantic_category_labels)
    )
    if len(raw_category_labels) != len(categories):
        raise RLRMaterialImportError(
            "raw_semantic_category_labels must have exactly one label for "
            "each canonical semantic category"
        )
    invalid_raw_category_labels = [
        label
        for label in raw_category_labels
        if not isinstance(label, str) or not label
    ]
    if invalid_raw_category_labels:
        raise RLRMaterialImportError(
            "raw semantic category labels must be non-empty strings: "
            f"{invalid_raw_category_labels!r}"
        )

    imported = import_rlr_material_database(
        source,
        database_id=database_id,
        version=version,
        source_description=source_description,
    )
    imported_materials_by_key = {
        str(material["material_key"]): material for material in imported["materials"]
    }
    label_owners: dict[str, set[str]] = {}
    for material in imported["materials"]:
        material_key = str(material["material_key"])
        for label in material["labels"]:
            label_owners.setdefault(str(label), set()).add(material_key)

    default_owners = label_owners.get("default", set())
    if len(default_owners) != 1:
        raise RLRMaterialImportError(
            "RLR semantic compilation requires exactly one material with the "
            "exact label 'default'"
        )
    default_key = next(iter(default_owners))
    default_material = imported_materials_by_key[default_key]

    entries: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    runtime_materials: list[dict[str, Any]] = []
    selected_source_parameters: list[dict[str, Any]] = []
    derived_runtime_parameters: list[dict[str, Any]] = []
    occupied_labels = set(label_owners)
    occupied_categories: set[str] = set()
    occupied_runtime_keys: set[str] = set()
    substring_count = 0
    default_count = 0
    for material_id, (category, raw_category_label) in enumerate(
        zip(categories, raw_category_labels, strict=True)
    ):
        lowered_category = raw_category_label.casefold()
        scores: list[dict[str, Any]] = []
        # Score only the immutable imported labels.  Aliases added for earlier
        # categories are runtime routing data and must never influence official
        # SoundSpaces/RLR winner selection.
        for candidate in imported["materials"]:
            matched_labels = [
                str(label)
                for label in candidate["labels"]
                if str(label).casefold() in lowered_category
            ]
            scores.append(
                {
                    "material_key": str(candidate["material_key"]),
                    "material_name": str(candidate["name"]),
                    "score": len(matched_labels),
                    "matched_labels": matched_labels,
                }
            )
        highest_score = max(score["score"] for score in scores)
        winners = [
            score for score in scores if score["score"] == highest_score
        ]
        if highest_score > 0 and len(winners) != 1:
            winner_summary = [
                {
                    "material_key": winner["material_key"],
                    "material_name": winner["material_name"],
                    "matched_labels": winner["matched_labels"],
                }
                for winner in winners
            ]
            raise RLRMaterialImportError(
                f"official substring match tie for category {category!r}"
                + (
                    ""
                    if raw_category_label == category
                    else f" with raw label {raw_category_label!r}"
                )
                + ": "
                f"highest_score={highest_score}, winners={winner_summary!r}"
            )
        if highest_score > 0:
            winner = winners[0]
            selected_source_material_key = str(winner["material_key"])
            selected_source_material = imported_materials_by_key[
                selected_source_material_key
            ]
            assignment_kind = "official_substring_match"
            mapping_source = (
                "Unique winner from the public RLR greatest-label-substring-"
                "match-count rule; coefficients copied exactly into a unique "
                "per-category runtime material"
            )
            matched_labels = list(winner["matched_labels"])
            substring_count += 1
        else:
            selected_source_material_key = default_key
            selected_source_material = default_material
            assignment_kind = "official_default"
            mapping_source = (
                "Zero public RLR substring-match score; explicitly assigned to "
                "an exact-parameter clone of the imported SoundSpaces/RLR "
                "Default material"
            )
            matched_labels = []
            default_count += 1
        rlr_category_name = _runtime_alias(
            room_id=room_id,
            source_semantic_label=category,
            selected_material_key=selected_source_material_key,
            occupied_labels=occupied_labels,
            occupied_categories=occupied_categories,
        )
        alias_digest = rlr_category_name.removeprefix("avengine_rlr_alias_")
        runtime_material_key = f"avengine_rlr_runtime_{alias_digest}"
        if runtime_material_key in occupied_runtime_keys:
            raise RLRMaterialImportError(
                "generated duplicate runtime material key "
                f"{runtime_material_key!r} for category {category!r}"
            )
        occupied_runtime_keys.add(runtime_material_key)
        occupied_labels.add(rlr_category_name)
        occupied_categories.add(rlr_category_name)

        source_parameters = _material_parameter_projection(
            selected_source_material
        )
        runtime_parameters = copy.deepcopy(source_parameters)
        source_parameter_sha256 = canonical_json_sha256(source_parameters)
        runtime_parameter_sha256 = canonical_json_sha256(runtime_parameters)
        parameters_preserved_exactly = (
            source_parameter_sha256 == runtime_parameter_sha256
            and source_parameters == runtime_parameters
        )
        if not parameters_preserved_exactly:
            raise RLRMaterialImportError(
                "semantic material compilation changed selected source "
                f"parameters for category {category!r}"
            )
        runtime_material_name = (
            f"{selected_source_material['name']} "
            f"[AVEngine derived runtime category: {category}]"
        )
        runtime_materials.append(
            {
                "material_key": runtime_material_key,
                "name": runtime_material_name,
                "labels": [rlr_category_name],
                **runtime_parameters,
                "source": (
                    f"{source_description}; exact parameter clone of official "
                    f"material_key={selected_source_material_key}; "
                    f"semantic_category={category}"
                ),
                "confidence": 0.0,
            }
        )
        selected_source_parameters.append(source_parameters)
        derived_runtime_parameters.append(runtime_parameters)
        entries.append(
            {
                "source_material_name": category,
                "material_id": material_id,
                "category_name": rlr_category_name,
                "material_key": runtime_material_key,
                "mapping_source": mapping_source,
                "mapping_confidence": 1.0,
                "human_override": False,
                "randomized": False,
                # No unresolved RLR fallback remains: Default use is an exact
                # assignment disclosed in ``decisions`` below.
                "fallback": False,
            }
        )
        decisions.append(
            {
                "canonical_semantic_category": category,
                "raw_semantic_category_label": raw_category_label,
                "source_semantic_label": category,
                "rlr_category_name": rlr_category_name,
                "assignment_kind": assignment_kind,
                "official_match_rule": (
                    "greatest_label_substring_match_count_on_lowercase_raw_category"
                ),
                "official_match_score": highest_score,
                "official_matched_labels": matched_labels,
                "official_exact_label_present": raw_category_label in matched_labels,
                "official_default_applied": assignment_kind == "official_default",
                # Retain selected_* as the official source identity; the
                # runtime clone has separate explicit fields.
                "selected_material_key": selected_source_material_key,
                "selected_material_name": selected_source_material["name"],
                "selected_source_material_key": selected_source_material_key,
                "selected_source_material_name": selected_source_material["name"],
                "runtime_material_key": runtime_material_key,
                "runtime_material_name": runtime_material_name,
                "source_parameter_sha256": source_parameter_sha256,
                "runtime_parameter_sha256": runtime_parameter_sha256,
                "parameters_preserved_exactly": parameters_preserved_exactly,
            }
        )

    database = copy.deepcopy(imported)
    database["materials"] = runtime_materials
    database["provenance"]["source"] = (
        f"{source_description}; public RLR substring material selection replayed "
        "offline, then copied without parameter changes into one unique AVEngine "
        "runtime material per semantic category"
    )
    mapping_runtime_keys = [entry["material_key"] for entry in entries]
    database_runtime_keys = [
        material["material_key"] for material in runtime_materials
    ]
    if (
        len(mapping_runtime_keys) != len(set(mapping_runtime_keys))
        or mapping_runtime_keys != database_runtime_keys
    ):
        raise RLRMaterialImportError(
            "semantic categories must map one-to-one, in material-id order, "
            "to unique derived runtime material entries"
        )
    mapping = {
        "schema": "avengine_m3_acoustic_material_mapping_v1",
        "mapping_id": f"{room_id}_soundspaces_rlr_semantic_v1",
        "room_id": room_id,
        "mapping_source_kind": "semantic_proposal",
        "source_to_canonical": copy.deepcopy(dict(source_to_canonical)),
        "entries": entries,
    }

    selected_source_parameter_hash = canonical_json_sha256(
        selected_source_parameters
    )
    derived_runtime_parameter_hash = canonical_json_sha256(
        derived_runtime_parameters
    )
    if selected_source_parameter_hash != derived_runtime_parameter_hash:
        raise RLRMaterialImportError(
            "semantic material compilation changed imported RLR coefficients"
        )
    category_count = len(categories)
    report = {
        "schema": RLR_SEMANTIC_MATERIAL_COVERAGE_SCHEMA,
        "status": "research_candidate",
        "qualification_claim": False,
        "physical_acoustic_material_claim": False,
        "room_id": room_id,
        "database_id": database_id,
        "label_match_policy": (
            "lowercase_raw_category_greatest_label_substring_match_count_v1"
        ),
        "tie_policy": "fail_closed_with_winner_evidence_v1",
        "unmatched_policy": "zero_score_official_default_v1",
        "runtime_binding_policy": (
            "unique_derived_runtime_material_per_semantic_category_v1"
        ),
        "runtime_silent_fallback_allowed": False,
        "runtime_one_to_one": {
            "semantic_category_count": len(categories),
            "runtime_material_count": len(runtime_materials),
            "unique_runtime_material_key_count": len(set(database_runtime_keys)),
            "unique_runtime_label_count": len(occupied_categories),
            "mapping_and_database_order_identical": (
                mapping_runtime_keys == database_runtime_keys
            ),
            "passed": True,
        },
        "official_default": {
            "label": "default",
            "material_key": default_key,
            "material_name": default_material["name"],
            "generated_aliases": [
                decision["rlr_category_name"]
                for decision in decisions
                if decision["official_default_applied"]
            ],
        },
        "runtime_exact_aliases": [
            {
                "canonical_semantic_category": decision[
                    "canonical_semantic_category"
                ],
                "raw_semantic_category_label": decision[
                    "raw_semantic_category_label"
                ],
                "source_semantic_label": decision["source_semantic_label"],
                "rlr_category_name": decision["rlr_category_name"],
                "selected_material_key": decision["selected_material_key"],
                "selected_source_material_key": (
                    decision["selected_source_material_key"]
                ),
                "runtime_material_key": decision["runtime_material_key"],
            }
            for decision in decisions
        ],
        "coefficient_preservation": {
            "fields": [
                "absorption",
                "scattering",
                "transmission",
                "damping",
                "density",
                "speed",
            ],
            "comparison_policy": (
                "per_category_selected_source_to_derived_runtime_exact_v1"
            ),
            "comparison_count": len(categories),
            "official_source_catalog_sha256": canonical_json_sha256(
                _coefficient_projection(imported)
            ),
            "imported_sha256": selected_source_parameter_hash,
            "compiled_sha256": derived_runtime_parameter_hash,
            "preserved_exactly": True,
        },
        "coverage": {
            "semantic_category_count": category_count,
            "resolved_category_count": category_count,
            "unresolved_category_count": 0,
            "official_substring_match_category_count": substring_count,
            "official_default_category_count": default_count,
            "official_substring_match_category_fraction": (
                substring_count / category_count
            ),
            "official_default_category_fraction": default_count / category_count,
        },
        "decisions": decisions,
    }
    return CompiledRLRSemanticMaterials(
        mapping=mapping,
        database=database,
        report=report,
    )


def _validate_native_database(database: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(database, Mapping):
        return ["RLR-native material database must be an object"]
    expected_fields = {
        "schema",
        "database_id",
        "version",
        "coefficient_units",
        "provenance",
        "materials",
    }
    missing = sorted(expected_fields - set(database))
    unexpected = sorted(set(database) - expected_fields)
    if missing:
        errors.append("database is missing fields: " + ", ".join(missing))
    if unexpected:
        errors.append("database contains unsupported fields: " + ", ".join(unexpected))
    if database.get("schema") != RLR_NATIVE_MATERIAL_DATABASE_SCHEMA:
        errors.append(
            f"database.schema must be {RLR_NATIVE_MATERIAL_DATABASE_SCHEMA!r}"
        )
    for field in ("database_id", "version"):
        if not isinstance(database.get(field), str) or not database[field]:
            errors.append(f"database.{field} must be a non-empty string")
    if database.get("coefficient_units") != _COEFFICIENT_UNITS:
        errors.append("database.coefficient_units must use the RLR-native units")

    provenance = database.get("provenance")
    expected_provenance = {
        "confidence": 0.0,
        "material_semantics": "research_placeholder",
        "intended_use": "research_compiler_diagnostics",
    }
    if not isinstance(provenance, Mapping):
        errors.append("database.provenance must be an object")
    else:
        if set(provenance) != {*expected_provenance, "source"}:
            errors.append("database.provenance has an invalid field set")
        for field, expected in expected_provenance.items():
            if provenance.get(field) != expected:
                errors.append(f"database.provenance.{field} must be {expected!r}")
        if not isinstance(provenance.get("source"), str) or not provenance.get(
            "source"
        ):
            errors.append("database.provenance.source must be non-empty")

    materials = database.get("materials")
    if not isinstance(materials, list) or not materials:
        errors.append("database.materials must be a non-empty array")
        return errors
    keys: list[str] = []
    for index, material in enumerate(materials):
        location = f"database.materials[{index}]"
        if not isinstance(material, Mapping):
            errors.append(f"{location} must be an object")
            continue
        missing_material = sorted(_NATIVE_MATERIAL_FIELDS - set(material))
        unexpected_material = sorted(set(material) - _NATIVE_MATERIAL_FIELDS)
        if missing_material:
            errors.append(
                f"{location} is missing fields: {', '.join(missing_material)}"
            )
        if unexpected_material:
            errors.append(
                f"{location} contains unsupported fields: "
                + ", ".join(unexpected_material)
            )
        material_key = material.get("material_key")
        if not isinstance(material_key, str) or not re.fullmatch(
            r"[a-z0-9]+(?:_[a-z0-9]+)*", material_key
        ):
            errors.append(f"{location}.material_key is invalid")
        else:
            keys.append(material_key)
        if not isinstance(material.get("source"), str) or not material.get("source"):
            errors.append(f"{location}.source must be non-empty")
        if material.get("confidence") != 0.0:
            errors.append(f"{location}.confidence must be 0")
        source_projection = {
            field: material.get(field) for field in _SOURCE_MATERIAL_FIELDS
        }
        _validate_source_material(
            source_projection,
            index=index,
            errors=errors,
        )
    if len(keys) != len(set(keys)):
        errors.append("database.materials material_key values must be unique")
    return errors


def rlr_document_from_native_database(
    database: Mapping[str, Any],
) -> dict[str, Any]:
    """Recover the upstream RLR document canonically, with no interpolation."""

    errors = _validate_native_database(database)
    if errors:
        raise RLRMaterialImportError(errors)
    return {
        "materials": [
            {
                field: copy.deepcopy(material[field])
                for field in (
                    "name",
                    "absorption",
                    "scattering",
                    "transmission",
                    "labels",
                    "damping",
                    "density",
                    "speed",
                )
            }
            for material in database["materials"]
        ]
    }


def _load_json_object(path: Path, *, purpose: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RLRMaterialImportError(f"unable to read {purpose} {path}: {exc}") from exc
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RLRMaterialImportError(
            f"{purpose} {path} is not valid UTF-8 JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise RLRMaterialImportError(f"{purpose} {path} must contain an object")
    return value, payload


def build_rlr_material_import_report(
    source_path: Path,
    database: Mapping[str, Any],
    output_path: Path | None = None,
    source_uri: str | None = None,
) -> dict[str, Any]:
    """Build a replayable exact-preservation report for a completed import."""

    source_path = Path(source_path)
    if source_uri is not None and (not isinstance(source_uri, str) or not source_uri):
        raise RLRMaterialImportError("source_uri must be a non-empty string")
    source_document, source_payload = _load_json_object(
        source_path, purpose="RLR source"
    )
    source_errors = _validate_rlr_document(source_document)
    if source_errors:
        raise RLRMaterialImportError(source_errors)

    roundtrip = rlr_document_from_native_database(database)
    source_canonical_hash = canonical_json_sha256(source_document)
    roundtrip_canonical_hash = canonical_json_sha256(roundtrip)
    if roundtrip_canonical_hash != source_canonical_hash:
        raise RLRMaterialImportError(
            "RLR-native database does not canonically round-trip to the "
            f"declared source {source_path}"
        )

    output_record: dict[str, Any] = {
        "canonical_sha256": canonical_json_sha256(database),
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_record["path"] = str(output_path)
        if output_path.is_file():
            output_document, output_payload = _load_json_object(
                output_path, purpose="imported database"
            )
            if (
                canonical_json_sha256(output_document)
                != output_record["canonical_sha256"]
            ):
                raise RLRMaterialImportError(
                    f"imported database file {output_path} does not match the "
                    "provided database"
                )
            output_record.update(
                {
                    "byte_size": len(output_payload),
                    "sha256": hashlib.sha256(output_payload).hexdigest(),
                }
            )

    pair_counts = {
        field: sum(len(material[field]) // 2 for material in database["materials"])
        for field in _CURVE_FIELDS
    }
    frequency_grids = {
        tuple(material[field][::2])
        for material in database["materials"]
        for field in _CURVE_FIELDS
    }
    source_record: dict[str, Any] = {
        "path": str(source_path),
        "byte_size": len(source_payload),
        "sha256": hashlib.sha256(source_payload).hexdigest(),
        "canonical_sha256": source_canonical_hash,
    }
    if source_uri is not None:
        source_record["uri"] = source_uri
    return {
        "schema": RLR_MATERIAL_IMPORT_REPORT_SCHEMA,
        "status": "pass",
        "source": source_record,
        "output": output_record,
        "statistics": {
            "material_count": len(database["materials"]),
            "curve_count": len(database["materials"]) * len(_CURVE_FIELDS),
            "pair_counts_by_field": pair_counts,
            "total_pair_count": sum(pair_counts.values()),
            "unique_frequency_grid_count": len(frequency_grids),
        },
        "roundtrip": {
            "canonical_sha256": roundtrip_canonical_hash,
            "preserved_exactly": True,
        },
        "claims": {
            "frl_measurement_fitted": False,
            "physical_calibration": False,
        },
    }
