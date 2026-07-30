from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator

from avengine.contracts.json_io import canonical_json_sha256


MAPPING_SCHEMA = "avengine_m3_acoustic_material_mapping_v1"
MATERIAL_DATABASE_SCHEMA = "avengine_m3_acoustic_material_database_v1"
RLR_NATIVE_MATERIAL_DATABASE_SCHEMA = "avengine_m3_acoustic_material_database_v2"
MATERIAL_PROFILE_SCHEMA = "avengine_m3_acoustic_material_profile_v1"
_MATERIAL_PROFILE_SCHEMA_FILE = "avengine_m3_acoustic_material_profile_v1.schema.json"
_CURVE_FIELDS = ("absorption", "scattering", "transmission", "damping")
_PHYSICAL_FIELDS = ("density", "speed")
_OVERRIDE_FIELDS = (*_CURVE_FIELDS, *_PHYSICAL_FIELDS)
_RLR_NATIVE_COEFFICIENT_UNITS = {
    "absorption": "frequency_hz_fraction_of_incident_sound_pressure_interleaved",
    "scattering": "frequency_hz_fraction_of_incident_sound_pressure_interleaved",
    "transmission": "frequency_hz_fraction_of_incident_sound_pressure_interleaved",
    "damping": "frequency_hz_decibels_per_meter_interleaved",
    "density": "kilograms_per_cubic_meter",
    "speed": "meters_per_second",
}

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


@dataclass(frozen=True)
class ResolvedMaterialProfile:
    """Immutable-by-convention products of one explicit profile resolution."""

    effective_mapping: dict[str, Any]
    effective_database: dict[str, Any]
    report: dict[str, Any]


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _unique(values: Iterable[Any]) -> bool:
    items = list(values)
    return len(items) == len(set(items))


def _material_profile_schema_path() -> Path:
    source = (
        Path(__file__).resolve().parents[3] / "schemas" / _MATERIAL_PROFILE_SCHEMA_FILE
    )
    installed = (
        Path(sys.prefix)
        / "share"
        / "avengine"
        / "schemas"
        / _MATERIAL_PROFILE_SCHEMA_FILE
    )
    path = source if source.is_file() else installed
    if not path.is_file():
        raise FileNotFoundError(
            "AVEngine acoustic material profile schema is unavailable: "
            f"{_MATERIAL_PROFILE_SCHEMA_FILE}"
        )
    return path


def _profile_schema_document() -> dict[str, Any]:
    import json

    with _material_profile_schema_path().open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("acoustic material profile schema must be an object")
    return value


def validate_material_profile(
    profile: Mapping[str, Any],
    *,
    room_id: str | None = None,
    band_count: int | None = None,
) -> list[str]:
    """Validate a room-scoped coefficient override profile.

    Array lengths depend on the selected base database, so callers that know
    the database must pass ``band_count``.  Selector existence and overlap are
    intentionally checked by :func:`resolve_material_profile`, where the
    mapping is available.
    """

    errors: list[str] = []
    validator = Draft202012Validator(_profile_schema_document())
    for error in sorted(
        validator.iter_errors(profile), key=lambda item: list(item.absolute_path)
    ):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"profile JSON Schema {location}: {error.message}")
    if room_id is not None and profile.get("room_id") != room_id:
        errors.append("profile.room_id must match the source room")

    override_owners: list[tuple[str, Any]] = [
        ("profile.global_override", profile.get("global_override"))
    ]
    raw_material_overrides = profile.get("material_overrides")
    if isinstance(raw_material_overrides, list):
        override_owners.extend(
            (f"profile.material_overrides[{index}]", override)
            for index, override in enumerate(raw_material_overrides)
        )
    for owner, override in override_owners:
        if not isinstance(override, Mapping):
            continue
        for field in _OVERRIDE_FIELDS:
            if field not in override:
                continue
            value = override[field]
            values = value if isinstance(value, list) else [value]
            for value_index, item in enumerate(values):
                if not _is_finite_number(item):
                    suffix = f"[{value_index}]" if isinstance(value, list) else ""
                    errors.append(f"{owner}.{field}{suffix} must be finite")
            if (
                field in _CURVE_FIELDS
                and isinstance(value, list)
                and band_count is not None
                and len(value) != band_count
            ):
                errors.append(
                    f"{owner}.{field} must contain exactly {band_count} band values"
                )
    return errors


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
            if any(
                abs(float(a) - float(b)) > 1e-9 for a, b in zip(rows[3], [0, 0, 0, 1])
            ):
                errors.append("mapping.source_to_canonical matrix must be affine")
        if not isinstance(
            source_to_canonical.get("source"), str
        ) or not source_to_canonical.get("source"):
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


def _validate_material_database_v1(database: Mapping[str, Any]) -> list[str]:
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


def _validate_native_curve(
    value: Any,
    *,
    owner: str,
    coefficient: bool,
    errors: list[str],
) -> None:
    if not isinstance(value, list) or len(value) < 2 or len(value) % 2:
        errors.append(f"{owner} must contain interleaved frequency/value pairs")
        return
    if any(not _is_finite_number(item) for item in value):
        errors.append(f"{owner} must contain finite numbers")
        return
    frequencies = [float(item) for item in value[0::2]]
    coefficients = [float(item) for item in value[1::2]]
    if any(frequency <= 0.0 for frequency in frequencies):
        errors.append(f"{owner} frequencies must be positive")
    elif any(left >= right for left, right in zip(frequencies, frequencies[1:])):
        errors.append(f"{owner} frequencies must be strictly increasing")
    if coefficient:
        if any(not 0.0 <= item <= 1.0 for item in coefficients):
            errors.append(f"{owner} values must be in [0, 1]")
    elif any(item < 0.0 for item in coefficients):
        errors.append(f"{owner} values must be non-negative")


def _validate_material_database_v2(database: Mapping[str, Any]) -> list[str]:
    """Validate lossless RLR-native material curves.

    Unlike v1, every coefficient field owns its frequency grid.  Empty and
    repeated labels are legal because upstream RLR databases contain both;
    package compilation still proves that every selected category has one
    unique winning material.
    """

    errors: list[str] = []
    if database.get("schema") != RLR_NATIVE_MATERIAL_DATABASE_SCHEMA:
        errors.append(
            f"database.schema must be {RLR_NATIVE_MATERIAL_DATABASE_SCHEMA!r}"
        )
    for field in ("database_id", "version"):
        if not isinstance(database.get(field), str) or not database[field]:
            errors.append(f"database.{field} must be a non-empty string")
    if database.get("coefficient_units") != _RLR_NATIVE_COEFFICIENT_UNITS:
        errors.append("database.coefficient_units must use the RLR-native units")

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
        expected_use = (
            MATERIAL_SEMANTIC_INTENDED_USE.get(semantics)
            if semantics in {"reviewed_physical", "research_placeholder"}
            else None
        )
        if expected_use is None:
            errors.append(
                "database.provenance.material_semantics must be one of "
                "'research_placeholder', 'reviewed_physical'"
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
        if not isinstance(raw_labels, list):
            errors.append(f"{prefix}.labels must be an array")
        else:
            for label_index, label in enumerate(raw_labels):
                if not isinstance(label, str) or not label:
                    errors.append(f"{prefix}.labels[{label_index}] must be non-empty")
                elif label != label.lower():
                    errors.append(f"{prefix}.labels[{label_index}] must be lowercase")
        for field in ("absorption", "scattering", "transmission"):
            _validate_native_curve(
                material.get(field),
                owner=f"{prefix}.{field}",
                coefficient=True,
                errors=errors,
            )
        _validate_native_curve(
            material.get("damping"),
            owner=f"{prefix}.damping",
            coefficient=False,
            errors=errors,
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
    return errors


def validate_material_database(database: Mapping[str, Any]) -> list[str]:
    schema = database.get("schema")
    if schema == MATERIAL_DATABASE_SCHEMA:
        return _validate_material_database_v1(database)
    if schema == RLR_NATIVE_MATERIAL_DATABASE_SCHEMA:
        return _validate_material_database_v2(database)
    return [
        "database.schema must be one of "
        f"{MATERIAL_DATABASE_SCHEMA!r}, {RLR_NATIVE_MATERIAL_DATABASE_SCHEMA!r}"
    ]


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
        errors.append("production mapping_source_kind must be 'explicit_author_slot'")
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
                {key: value for key, value in material.items() if key != "absorption"}
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
                            for low_value, high_value in zip(low_values, high_values)
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


def _normalized_profile_value(
    field: str, value: Any, *, band_count: int
) -> float | list[float]:
    if field in _CURVE_FIELDS:
        values = value if isinstance(value, list) else [value] * band_count
        return [float(item) for item in values]
    return float(value)


def resolve_material_profile(
    mapping: Mapping[str, Any],
    database: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    room_id: str,
) -> ResolvedMaterialProfile:
    """Resolve explicit global/per-material controls without mutating inputs.

    Precedence is strictly ``base database < global override < material
    override``.  Material overrides must resolve to disjoint database material
    keys.  A source-name override is rejected when its mapping entry shares a
    material key because one RLR database record cannot encode two different
    coefficient sets for that shared key.
    """

    if not isinstance(profile, Mapping):
        raise MaterialContractError(["profile must be an object"])
    if database.get("schema") != MATERIAL_DATABASE_SCHEMA:
        raise MaterialContractError(
            [
                "material profile v1 requires acoustic material database v1; "
                "RLR-native v2 curves must be authored or imported explicitly"
            ]
        )

    # This validates both input documents and their exact mapping/database
    # relationship before any derived document is created.
    compile_materials(mapping, database, room_id=room_id)
    band_count = len(database["bands_hz"])
    errors = validate_material_profile(profile, room_id=room_id, band_count=band_count)
    if errors:
        raise MaterialContractError(errors)

    ordered_entries = sorted(mapping["entries"], key=lambda item: item["material_id"])
    entry_by_source_name = {
        entry["source_material_name"]: entry for entry in ordered_entries
    }
    source_names_by_key: dict[str, list[str]] = {}
    for entry in ordered_entries:
        source_names_by_key.setdefault(entry["material_key"], []).append(
            entry["source_material_name"]
        )

    seen_selectors: dict[tuple[str, str], int] = {}
    claimed_material_keys: dict[str, int] = {}
    resolved_overrides: list[tuple[int, str, Mapping[str, Any]]] = []
    selector_resolutions: list[dict[str, Any]] = []
    for index, override in enumerate(profile.get("material_overrides", [])):
        selector = override["selector"]
        if "material_key" in selector:
            selector_kind = "material_key"
            selector_value = selector["material_key"]
            material_key = selector_value
            if material_key not in source_names_by_key:
                errors.append(
                    f"profile.material_overrides[{index}].selector references "
                    f"unknown material_key {material_key!r}"
                )
                continue
        else:
            selector_kind = "source_material_name"
            selector_value = selector["source_material_name"]
            entry = entry_by_source_name.get(selector_value)
            if entry is None:
                errors.append(
                    f"profile.material_overrides[{index}].selector references "
                    f"unknown source_material_name {selector_value!r}"
                )
                continue
            material_key = entry["material_key"]
            shared_names = source_names_by_key[material_key]
            if len(shared_names) != 1:
                errors.append(
                    f"profile.material_overrides[{index}] cannot target "
                    f"source_material_name {selector_value!r}: material_key "
                    f"{material_key!r} is shared by source materials "
                    f"{shared_names}; select the material_key to override all "
                    "of them or split the base database material"
                )
                continue

        selector_identity = (selector_kind, selector_value)
        duplicate_index = seen_selectors.get(selector_identity)
        if duplicate_index is not None:
            errors.append(
                f"profile.material_overrides[{index}].selector duplicates "
                f"material_overrides[{duplicate_index}]"
            )
            continue
        seen_selectors[selector_identity] = index

        conflict_index = claimed_material_keys.get(material_key)
        if conflict_index is not None:
            errors.append(
                f"profile.material_overrides[{index}].selector conflicts with "
                f"material_overrides[{conflict_index}]: both resolve to "
                f"material_key {material_key!r}"
            )
            continue
        claimed_material_keys[material_key] = index
        resolved_overrides.append((index, material_key, override))
        selector_resolutions.append(
            {
                "override_index": index,
                "selector": copy.deepcopy(selector),
                "material_key": material_key,
                "source_material_names": list(source_names_by_key[material_key]),
            }
        )
    if errors:
        raise MaterialContractError(errors)

    profile_hash = canonical_json_sha256(profile)
    base_mapping_hash = canonical_json_sha256(mapping)
    base_database_hash = canonical_json_sha256(database)
    resolution_hash = canonical_json_sha256(
        {
            "mapping_sha256": base_mapping_hash,
            "database_sha256": base_database_hash,
            "profile_sha256": profile_hash,
        }
    )
    profile_id = profile["profile_id"]
    effective_mapping = copy.deepcopy(dict(mapping))
    effective_database = copy.deepcopy(dict(database))
    effective_database["database_id"] = (
        f"{database['database_id']}__profile__{profile_id}__{resolution_hash[:12]}"
    )
    provenance = effective_database["provenance"]
    base_material_semantics = str(provenance["material_semantics"])
    if base_material_semantics == "reviewed_physical":
        # Changing even one reviewed coefficient creates a new, unreviewed
        # database.  A convenience profile must never inherit a physical-truth
        # claim from bytes that it has changed.
        provenance["confidence"] = 0.0
        provenance["material_semantics"] = "research_placeholder"
        provenance["intended_use"] = "research_compiler_diagnostics"
        for material in effective_database["materials"]:
            material["confidence"] = 0.0
    provenance["source"] = (
        f"{database['provenance']['source']}; explicit acoustic controls "
        f"resolved from profile {profile_id!r} (sha256:{profile_hash})"
    )

    global_override = profile.get("global_override")
    global_fields = (
        [field for field in _OVERRIDE_FIELDS if field in global_override]
        if isinstance(global_override, Mapping)
        else []
    )
    override_by_key = {
        material_key: (index, override)
        for index, material_key, override in resolved_overrides
    }
    material_lineage: list[dict[str, Any]] = []
    for material in effective_database["materials"]:
        material_key = material["material_key"]
        field_lineage = {
            field: f"base_database:{database['database_id']}"
            for field in _OVERRIDE_FIELDS
        }
        applied_layers = ["base_database"]
        modified_fields: list[str] = []
        if isinstance(global_override, Mapping):
            for field in global_fields:
                material[field] = _normalized_profile_value(
                    field, global_override[field], band_count=band_count
                )
                field_lineage[field] = "profile.global_override"
                modified_fields.append(field)
            applied_layers.append("global_override")

        resolved_override = override_by_key.get(material_key)
        if resolved_override is not None:
            override_index, override = resolved_override
            per_material_fields = [
                field for field in _OVERRIDE_FIELDS if field in override
            ]
            for field in per_material_fields:
                material[field] = _normalized_profile_value(
                    field, override[field], band_count=band_count
                )
                field_lineage[field] = f"profile.material_overrides[{override_index}]"
                if field not in modified_fields:
                    modified_fields.append(field)
            applied_layers.append(f"material_override[{override_index}]")

        if modified_fields:
            material["source"] = (
                f"{material['source']}; fields {sorted(modified_fields)} "
                f"overridden by profile {profile_id!r}"
            )
        material_lineage.append(
            {
                "material_key": material_key,
                "source_material_names": list(
                    source_names_by_key.get(material_key, [])
                ),
                "applied_layers": applied_layers,
                "field_lineage": field_lineage,
                "effective_values": {
                    field: copy.deepcopy(material[field]) for field in _OVERRIDE_FIELDS
                },
            }
        )

    # Revalidate the derived database and its RLR selection contract.  This is
    # deliberately fail-closed even though all mutations above are constrained.
    derived_errors = validate_material_database(effective_database)
    if derived_errors:
        raise MaterialContractError(derived_errors)
    compile_materials(effective_mapping, effective_database, room_id=room_id)

    report = {
        "schema": "avengine_m3_acoustic_material_resolution_report_v1",
        "status": "pass",
        "profile_id": profile_id,
        "room_id": room_id,
        "bands_hz": [float(value) for value in database["bands_hz"]],
        "precedence": [
            "base_database",
            "global_override",
            "material_override",
        ],
        "input_hashes": {
            "profile_sha256": profile_hash,
            "mapping_sha256": base_mapping_hash,
            "database_sha256": base_database_hash,
            "resolution_sha256": resolution_hash,
        },
        "output_hashes": {
            "mapping_sha256": canonical_json_sha256(effective_mapping),
            "database_sha256": canonical_json_sha256(effective_database),
        },
        "base_database_id": database["database_id"],
        "effective_database_id": effective_database["database_id"],
        "material_semantics": {
            "base": base_material_semantics,
            "effective": effective_database["provenance"]["material_semantics"],
            "profile_grants_physical_review": False,
        },
        "selector_resolutions": selector_resolutions,
        "materials": material_lineage,
    }
    return ResolvedMaterialProfile(
        effective_mapping=effective_mapping,
        effective_database=effective_database,
        report=report,
    )


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
            1 for label in labels if isinstance(label, str) and label.lower() in lowered
        )
        scores.append((key, score))
    return scores


def _interleaved(bands: list[Any], values: list[Any]) -> list[float]:
    result: list[float] = []
    for frequency, value in zip(bands, values):
        result.extend((float(frequency), float(value)))
    return result


def _stable_unique_labels(values: Any) -> tuple[list[str], list[str]]:
    labels = values if isinstance(values, list) else []
    seen: set[str] = set()
    retained: list[str] = []
    removed: list[str] = []
    for label in labels:
        if not isinstance(label, str):
            continue
        identity = label.casefold()
        if identity in seen:
            removed.append(label)
        else:
            seen.add(identity)
            retained.append(label)
    return retained, removed


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
    native_curves = database.get("schema") == RLR_NATIVE_MATERIAL_DATABASE_SCHEMA
    matching_materials = (
        [
            {
                **material,
                "labels": _stable_unique_labels(material["labels"])[0],
            }
            for material in database["materials"]
        ]
        if native_curves
        else database["materials"]
    )
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
        scores = rlr_match_scores(entry["category_name"], matching_materials)
        highest = max(score for _, score in scores)
        winners = sorted(candidate for candidate, score in scores if score == highest)
        runtime_labels, removed_labels = _stable_unique_labels(
            intended.get("labels", [])
        )
        exact_labels = {
            str(label).lower() for label in runtime_labels if isinstance(label, str)
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
        category = {
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
        if native_curves:
            category["rlr_label_normalization"] = {
                "policy": "stable_case_insensitive_exact_duplicate_removal_v1",
                "source_label_count": len(intended["labels"]),
                "runtime_label_count": len(runtime_labels),
                "removed_exact_duplicates": removed_labels,
            }
        categories.append(category)

    runtime_label_owners: dict[str, str] = {}
    for material in database["materials"]:
        key = material["material_key"]
        if key not in used_material_keys:
            continue
        runtime_labels, _removed = _stable_unique_labels(material["labels"])
        for label in runtime_labels:
            identity = label.casefold()
            previous = runtime_label_owners.get(identity)
            if previous is not None and previous != key:
                errors.append(
                    f"used RLR label {label!r} is shared by material_key "
                    f"{previous!r} and {key!r}"
                )
            else:
                runtime_label_owners[identity] = key
    if errors:
        raise MaterialContractError(errors)

    bands = [] if native_curves else list(database["bands_hz"])
    rlr_materials: list[dict[str, Any]] = []
    for material in database["materials"]:
        if material["material_key"] not in used_material_keys:
            continue
        if native_curves:
            curves = {
                field: list(material[field])
                for field in (
                    "absorption",
                    "scattering",
                    "transmission",
                    "damping",
                )
            }
            density = material["density"]
            speed = material["speed"]
        else:
            curves = {
                field: _interleaved(bands, material[field])
                for field in (
                    "absorption",
                    "scattering",
                    "transmission",
                    "damping",
                )
            }
            density = float(material["density"])
            speed = float(material["speed"])
        rlr_materials.append(
            {
                "name": material["name"],
                "labels": (
                    _stable_unique_labels(material["labels"])[0]
                    if native_curves
                    else list(material["labels"])
                ),
                **curves,
                "density": density,
                "speed": speed,
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
