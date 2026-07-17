from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

from avengine.contracts.json_io import canonical_json_sha256


MAPPING_SCHEMA = "avengine_m3_acoustic_material_mapping_v1"
MATERIAL_DATABASE_SCHEMA = "avengine_m3_acoustic_material_database_v1"

MATERIAL_SEMANTIC_INTENDED_USE = {
    "controlled_canary": "controlled_material_activation_canary",
    "reviewed_physical": "reviewed_production_profile",
    "research_placeholder": "research_compiler_diagnostics",
}

MATERIAL_QUALIFICATION_CLAIM = {
    "controlled_canary": "synthetic_activation_test_only",
    "reviewed_physical": "reviewed_physical_material_profile",
    "research_placeholder": "unqualified_research_placeholder",
}


class MaterialContractError(ValueError):
    def __init__(self, errors: Iterable[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class CompiledMaterials:
    source_material_to_id: dict[str, int]
    categories_document: dict[str, Any]
    rlr_database: dict[str, Any]


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _unique(values: Iterable[Any]) -> bool:
    items = list(values)
    return len(items) == len(set(items))


def _validate_coefficients(
    material: Mapping[str, Any],
    *,
    index: int,
    band_count: int,
    errors: list[str],
) -> None:
    for field in ("absorption", "scattering", "transmission", "damping"):
        values = material.get(field)
        prefix = f"materials[{index}].{field}"
        if not isinstance(values, list) or len(values) != band_count:
            errors.append(f"{prefix} must contain exactly {band_count} band values")
            continue
        for value_index, value in enumerate(values):
            if not _is_finite_number(value):
                errors.append(f"{prefix}[{value_index}] must be finite")
                continue
            number = float(value)
            if field == "damping":
                if number < 0:
                    errors.append(f"{prefix}[{value_index}] must be non-negative")
            elif not 0.0 <= number <= 1.0:
                errors.append(f"{prefix}[{value_index}] must be in [0, 1]")


def validate_material_mapping(
    mapping: Mapping[str, Any], *, room_id: str | None = None
) -> list[str]:
    errors: list[str] = []
    if mapping.get("schema") != MAPPING_SCHEMA:
        errors.append(f"mapping.schema must be {MAPPING_SCHEMA!r}")
    if room_id is not None and mapping.get("room_id") != room_id:
        errors.append("mapping.room_id must match the source room")
    for field in ("mapping_id", "room_id", "mapping_source_kind"):
        if not isinstance(mapping.get(field), str) or not mapping[field]:
            errors.append(f"mapping.{field} must be a non-empty string")
    source_to_canonical = mapping.get("source_to_canonical")
    if not isinstance(source_to_canonical, Mapping):
        errors.append("mapping.source_to_canonical must be an object")
    else:
        matrix = source_to_canonical.get("matrix_row_major")
        if (
            not isinstance(matrix, list)
            or len(matrix) != 16
            or any(not _is_finite_number(value) for value in matrix)
        ):
            errors.append(
                "mapping.source_to_canonical.matrix_row_major must contain 16 finite numbers"
            )
        else:
            rows = [matrix[offset : offset + 4] for offset in range(0, 16, 4)]
            if any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(rows[3], [0, 0, 0, 1])):
                errors.append("mapping.source_to_canonical matrix must be affine")
        if not isinstance(source_to_canonical.get("source"), str) or not source_to_canonical.get(
            "source"
        ):
            errors.append("mapping.source_to_canonical.source must be non-empty")
        if not isinstance(source_to_canonical.get("reviewed"), bool):
            errors.append("mapping.source_to_canonical.reviewed must be a boolean")

    entries = mapping.get("entries")
    if not isinstance(entries, list) or not entries:
        return [*errors, "mapping.entries must be a non-empty array"]
    material_ids: list[int] = []
    source_names: list[str] = []
    categories: list[str] = []
    material_keys: list[str] = []
    for index, entry in enumerate(entries):
        prefix = f"mapping.entries[{index}]"
        if not isinstance(entry, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        source_name = entry.get("source_material_name")
        if not isinstance(source_name, str) or not source_name:
            errors.append(f"{prefix}.source_material_name must be non-empty")
        else:
            source_names.append(source_name)
        material_id = entry.get("material_id")
        if (
            not isinstance(material_id, int)
            or isinstance(material_id, bool)
            or material_id < 0
        ):
            errors.append(f"{prefix}.material_id must be a non-negative integer")
        else:
            material_ids.append(material_id)
        category = entry.get("category_name")
        if not isinstance(category, str) or not category:
            errors.append(f"{prefix}.category_name must be non-empty")
        elif category != category.lower():
            errors.append(f"{prefix}.category_name must be lowercase")
        else:
            categories.append(category)
        material_key = entry.get("material_key")
        if not isinstance(material_key, str) or not material_key:
            errors.append(f"{prefix}.material_key must be non-empty")
        else:
            material_keys.append(material_key)
        if not isinstance(entry.get("mapping_source"), str) or not entry.get(
            "mapping_source"
        ):
            errors.append(f"{prefix}.mapping_source must be non-empty")
        confidence = entry.get("mapping_confidence")
        if not _is_finite_number(confidence) or not 0 <= float(confidence) <= 1:
            errors.append(f"{prefix}.mapping_confidence must be finite in [0, 1]")
        for field in ("human_override", "randomized"):
            if not isinstance(entry.get(field), bool):
                errors.append(f"{prefix}.{field} must be a boolean")
        if entry.get("fallback") is not False:
            errors.append(f"{prefix}.fallback must be false")

    for values, label in (
        (material_ids, "material_id"),
        (source_names, "source_material_name"),
        (categories, "category_name"),
    ):
        if not _unique(values):
            errors.append(f"mapping entries must have unique {label} values")
    if material_ids and sorted(material_ids) != list(range(len(entries))):
        errors.append("mapping material_id values must be contiguous from zero")
    if len(material_keys) != len(entries):
        # The field-specific errors above are more useful, but this prevents a
        # misleading downstream set comparison.
        return errors
    return errors


def validate_material_database(database: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if database.get("schema") != MATERIAL_DATABASE_SCHEMA:
        errors.append(f"database.schema must be {MATERIAL_DATABASE_SCHEMA!r}")
    for field in ("database_id", "version"):
        if not isinstance(database.get(field), str) or not database[field]:
            errors.append(f"database.{field} must be a non-empty string")

    bands = database.get("bands_hz")
    if not isinstance(bands, list) or not bands:
        errors.append("database.bands_hz must be a non-empty array")
        band_count = 0
    else:
        band_count = len(bands)
        if any(not _is_finite_number(value) or float(value) <= 0 for value in bands):
            errors.append("database.bands_hz must contain positive finite numbers")
        elif any(float(left) >= float(right) for left, right in zip(bands, bands[1:])):
            errors.append("database.bands_hz must be strictly increasing")

    provenance = database.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append("database.provenance must be an object")
    else:
        if not isinstance(provenance.get("source"), str) or not provenance.get(
            "source"
        ):
            errors.append("database.provenance.source must be non-empty")
        confidence = provenance.get("confidence")
        if not _is_finite_number(confidence) or not 0 <= float(confidence) <= 1:
            errors.append("database.provenance.confidence must be finite in [0, 1]")
        semantics = provenance.get("material_semantics")
        intended_use = provenance.get("intended_use")
        expected_use = MATERIAL_SEMANTIC_INTENDED_USE.get(semantics)
        if expected_use is None:
            errors.append(
                "database.provenance.material_semantics must be one of "
                + ", ".join(sorted(MATERIAL_SEMANTIC_INTENDED_USE))
            )
        elif intended_use != expected_use:
            errors.append(
                "database.provenance.intended_use does not match "
                f"material_semantics={semantics!r}"
            )

    materials = database.get("materials")
    if not isinstance(materials, list) or not materials:
        return [*errors, "database.materials must be a non-empty array"]
    keys: list[str] = []
    labels: list[str] = []
    for index, material in enumerate(materials):
        prefix = f"database.materials[{index}]"
        if not isinstance(material, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        for field in ("material_key", "name", "source"):
            if not isinstance(material.get(field), str) or not material[field]:
                errors.append(f"{prefix}.{field} must be a non-empty string")
        if isinstance(material.get("material_key"), str):
            keys.append(material["material_key"])
        raw_labels = material.get("labels")
        if not isinstance(raw_labels, list) or not raw_labels:
            errors.append(f"{prefix}.labels must be a non-empty array")
        else:
            for label_index, label in enumerate(raw_labels):
                if not isinstance(label, str) or not label:
                    errors.append(f"{prefix}.labels[{label_index}] must be non-empty")
                elif label != label.lower():
                    errors.append(f"{prefix}.labels[{label_index}] must be lowercase")
                else:
                    labels.append(label)
        _validate_coefficients(
            material, index=index, band_count=band_count, errors=errors
        )
        for physical_field in ("density", "speed"):
            value = material.get(physical_field)
            if not _is_finite_number(value) or float(value) <= 0:
                errors.append(
                    f"{prefix}.{physical_field} must be a positive finite number"
                )
        confidence = material.get("confidence")
        if not _is_finite_number(confidence) or not 0 <= float(confidence) <= 1:
            errors.append(f"{prefix}.confidence must be finite in [0, 1]")

    if not _unique(keys):
        errors.append("database material_key values must be unique")
    if not _unique(labels):
        errors.append("database labels must be globally unique")
    return errors


def production_admission_errors(
    mapping: Mapping[str, Any],
    database: Mapping[str, Any],
    *,
    expected_material_semantics: str | None = None,
) -> list[str]:
    """Return fail-closed errors for a package claiming production admission.

    ``controlled_canary`` is deliberately synthetic: it qualifies only the
    material-activation path.  ``reviewed_physical`` is the separate claim for
    a reviewed physical material profile.  Visual/semantic proposals cannot
    enter either path, even if their geometry QA happens to pass.
    """

    errors: list[str] = []
    if mapping.get("mapping_source_kind") != "explicit_author_slot":
        errors.append(
            "production mapping_source_kind must be 'explicit_author_slot'"
        )
    transform = mapping.get("source_to_canonical")
    if not isinstance(transform, Mapping) or transform.get("reviewed") is not True:
        errors.append("production source_to_canonical.reviewed must be true")
    entries = mapping.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append("production mapping.entries must be non-empty")
    else:
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                errors.append(f"production mapping.entries[{index}] must be an object")
                continue
            prefix = f"production mapping.entries[{index}]"
            if entry.get("mapping_confidence") != 1.0:
                errors.append(f"{prefix}.mapping_confidence must equal 1.0")
            if entry.get("human_override") is not True:
                errors.append(f"{prefix}.human_override must be true")
            if entry.get("randomized") is not False:
                errors.append(f"{prefix}.randomized must be false")
            if entry.get("fallback") is not False:
                errors.append(f"{prefix}.fallback must be false")

    provenance = database.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append("production database.provenance must be an object")
        semantics = None
    else:
        semantics = provenance.get("material_semantics")
        if provenance.get("confidence") != 1.0:
            errors.append("production database.provenance.confidence must equal 1.0")
        if semantics not in {"controlled_canary", "reviewed_physical"}:
            errors.append(
                "production material_semantics must be controlled_canary or "
                "reviewed_physical"
            )
        expected_use = MATERIAL_SEMANTIC_INTENDED_USE.get(semantics)
        if expected_use is None or provenance.get("intended_use") != expected_use:
            errors.append(
                "production database.provenance.intended_use does not match its "
                "material semantics"
            )
    if (
        expected_material_semantics is not None
        and semantics != expected_material_semantics
    ):
        errors.append(
            "production material_semantics must equal "
            f"{expected_material_semantics!r} for this compiler path"
        )
    materials = database.get("materials")
    if not isinstance(materials, list) or not materials:
        errors.append("production database.materials must be non-empty")
    else:
        for index, material in enumerate(materials):
            if not isinstance(material, Mapping) or material.get("confidence") != 1.0:
                errors.append(
                    f"production database.materials[{index}].confidence must equal 1.0"
                )
    return errors


def controlled_counterfactual_errors(
    low: Mapping[str, Any], high: Mapping[str, Any]
) -> list[str]:
    """Prove that a controlled low/high pair changes absorption only.

    Database IDs identify the conditions and are the only metadata difference
    permitted.  Every material structure and every non-absorption field must be
    byte-canonically equal; every high absorption band must be strictly larger.
    """

    errors: list[str] = []
    for label, database in (("low", low), ("high", high)):
        semantics = database.get("provenance", {}).get("material_semantics")
        if semantics != "controlled_canary":
            errors.append(f"{label} database must use controlled_canary semantics")

    def without_allowed_differences(database: Mapping[str, Any]) -> dict[str, Any]:
        normalized = {
            key: value for key, value in database.items() if key != "database_id"
        }
        normalized["materials"] = [
            {key: value for key, value in material.items() if key != "absorption"}
            for material in database.get("materials", [])
            if isinstance(material, Mapping)
        ]
        return normalized

    if without_allowed_differences(low) != without_allowed_differences(high):
        errors.append(
            "low/high databases differ outside database_id and absorption coefficients"
        )
    low_materials = low.get("materials")
    high_materials = high.get("materials")
    if not isinstance(low_materials, list) or not isinstance(high_materials, list):
        return [*errors, "low/high materials must be arrays"]
    if len(low_materials) != len(high_materials):
        return [*errors, "low/high material counts must match"]
    for index, (low_material, high_material) in enumerate(
        zip(low_materials, high_materials)
    ):
        if not isinstance(low_material, Mapping) or not isinstance(
            high_material, Mapping
        ):
            errors.append(f"low/high materials[{index}] must be objects")
            continue
        low_values = low_material.get("absorption")
        high_values = high_material.get("absorption")
        if not isinstance(low_values, list) or not isinstance(high_values, list):
            errors.append(f"low/high materials[{index}].absorption must be arrays")
            continue
        if len(low_values) != len(high_values) or not low_values:
            errors.append(
                f"low/high materials[{index}].absorption band counts must match"
            )
            continue
        if any(
            not _is_finite_number(low_value)
            or not _is_finite_number(high_value)
            or float(high_value) <= float(low_value)
            for low_value, high_value in zip(low_values, high_values)
        ):
            errors.append(
                f"high materials[{index}].absorption must be strictly greater "
                "than low in every band"
            )
    return errors


def controlled_counterfactual_proof(
    low: Mapping[str, Any], high: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a canonical, independently replayable absorption-only proof."""

    def without_allowed_differences(database: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **{
                key: value
                for key, value in database.items()
                if key not in {"database_id", "materials"}
            },
            "materials": [
                {
                    key: value
                    for key, value in material.items()
                    if key != "absorption"
                }
                for material in database.get("materials", [])
                if isinstance(material, Mapping)
            ],
        }

    low_projection = without_allowed_differences(low)
    high_projection = without_allowed_differences(high)
    material_records: list[dict[str, Any]] = []
    low_materials = low.get("materials", [])
    high_materials = high.get("materials", [])
    if isinstance(low_materials, list) and isinstance(high_materials, list):
        for low_material, high_material in zip(low_materials, high_materials):
            if not isinstance(low_material, Mapping) or not isinstance(
                high_material, Mapping
            ):
                continue
            low_values = list(low_material.get("absorption", []))
            high_values = list(high_material.get("absorption", []))
            material_records.append(
                {
                    "material_key": low_material.get("material_key"),
                    "low_absorption": low_values,
                    "high_absorption": high_values,
                    "high_strictly_greater_every_band": bool(
                        len(low_values) == len(high_values)
                        and bool(low_values)
                        and all(
                            _is_finite_number(low_value)
                            and _is_finite_number(high_value)
                            and float(high_value) > float(low_value)
                            for low_value, high_value in zip(
                                low_values, high_values
                            )
                        )
                    ),
                }
            )
    errors = controlled_counterfactual_errors(low, high)
    low_hash = canonical_json_sha256(low_projection)
    high_hash = canonical_json_sha256(high_projection)
    return {
        "status": "pass" if not errors else "fail",
        "allowed_differences": ["database_id", "materials[*].absorption"],
        "non_absorption_structure": {
            "low_sha256": low_hash,
            "high_sha256": high_hash,
            "identical": low_hash == high_hash,
        },
        "every_high_absorption_band_strictly_greater": bool(
            material_records
            and all(
                record["high_strictly_greater_every_band"]
                for record in material_records
            )
        ),
        "materials": material_records,
        "errors": errors,
    }


def rlr_match_scores(
    category_name: str, materials: Iterable[Mapping[str, Any]]
) -> list[tuple[str, int]]:
    """Replay the pinned RLR label-substring matching rule.

    The pinned header states that the human-readable material name is ignored;
    the selected material is the one with the greatest number of labels that
    occur as substrings in the lower-cased category string.
    """

    lowered = category_name.lower()
    scores: list[tuple[str, int]] = []
    for material in materials:
        key = str(material.get("material_key", ""))
        raw_labels = material.get("labels", [])
        labels = raw_labels if isinstance(raw_labels, list) else []
        score = sum(
            1
            for label in labels
            if isinstance(label, str) and label.lower() in lowered
        )
        scores.append((key, score))
    return scores


def _interleaved(bands: list[Any], values: list[Any]) -> list[float]:
    result: list[float] = []
    for frequency, value in zip(bands, values):
        result.extend((float(frequency), float(value)))
    return result


def compile_materials(
    mapping: Mapping[str, Any], database: Mapping[str, Any], *, room_id: str
) -> CompiledMaterials:
    errors = validate_material_mapping(mapping, room_id=room_id)
    errors.extend(validate_material_database(database))
    if errors:
        raise MaterialContractError(errors)

    material_by_key = {
        material["material_key"]: material for material in database["materials"]
    }
    categories: list[dict[str, Any]] = []
    source_to_id: dict[str, int] = {}
    used_material_keys: set[str] = set()
    for entry in sorted(mapping["entries"], key=lambda item: item["material_id"]):
        key = entry["material_key"]
        intended = material_by_key.get(key)
        if intended is None:
            errors.append(
                f"mapping category {entry['category_name']!r} references missing "
                f"material_key {key!r}"
            )
            continue
        scores = rlr_match_scores(entry["category_name"], database["materials"])
        highest = max(score for _, score in scores)
        winners = sorted(candidate for candidate, score in scores if score == highest)
        exact_labels = {
            str(label).lower()
            for label in intended.get("labels", [])
            if isinstance(label, str)
        }
        if highest <= 0:
            errors.append(
                f"category {entry['category_name']!r} matches no RLR material label"
            )
        elif winners != [key]:
            errors.append(
                f"category {entry['category_name']!r} does not uniquely match "
                f"{key!r}; winners={winners}, score={highest}"
            )
        if entry["category_name"].lower() not in exact_labels:
            errors.append(
                f"category {entry['category_name']!r} requires an exact lower-case "
                f"label on intended material {key!r}; substring-only matches are "
                "not accepted by the pinned C++ ingestion contract"
            )
        source_to_id[entry["source_material_name"]] = entry["material_id"]
        used_material_keys.add(key)
        categories.append(
            {
                "material_id": entry["material_id"],
                "category_name": entry["category_name"],
                "source_material_name": entry["source_material_name"],
                "material_key": key,
                "rlr_material_name": intended["name"],
                "mapping_source": entry["mapping_source"],
                "mapping_confidence": float(entry["mapping_confidence"]),
                "human_override": entry["human_override"],
                "randomized": entry["randomized"],
                "fallback": False,
                "rlr_match": {
                    "rule": "greatest_label_substring_match_count",
                    "score": highest,
                    "matched_material_key": key,
                    "tie_count": len(winners),
                },
            }
        )
    if errors:
        raise MaterialContractError(errors)

    bands = list(database["bands_hz"])
    rlr_materials: list[dict[str, Any]] = []
    for material in database["materials"]:
        if material["material_key"] not in used_material_keys:
            continue
        rlr_materials.append(
            {
                "name": material["name"],
                "labels": list(material["labels"]),
                "absorption": _interleaved(bands, material["absorption"]),
                "scattering": _interleaved(bands, material["scattering"]),
                "transmission": _interleaved(bands, material["transmission"]),
                "damping": _interleaved(bands, material["damping"]),
                "density": float(material["density"]),
                "speed": float(material["speed"]),
            }
        )
    categories_document = {
        "schema": "avengine_acoustic_material_categories_v1",
        "mapping_id": mapping["mapping_id"],
        "room_id": room_id,
        "mapping_source_kind": mapping["mapping_source_kind"],
        "fallback_category": None,
        "categories": categories,
    }
    return CompiledMaterials(
        source_material_to_id=source_to_id,
        categories_document=categories_document,
        rlr_database={"materials": rlr_materials},
    )
