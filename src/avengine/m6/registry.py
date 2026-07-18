"""Shared, fail-closed helpers for the small M6 registry contracts.

M6 deliberately introduces registries without turning them into a final
dataset-item schema.  The helpers in this module validate one checked-in JSON
document at a time, preserve canonical record ordering, and bind content
hashes without resolving runtime assets or creating simulator state.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator

from avengine.contracts.json_io import canonical_json_sha256, load_json


ENTITY_ASSET_REGISTRY_SCHEMA = "avengine_m6_entity_asset_registry_v1"
ANIMAL_TEMPLATE_REGISTRY_SCHEMA = "avengine_m6_animal_template_registry_v1"
SOURCE_ENDPOINT_REGISTRY_SCHEMA = "avengine_m6_source_endpoint_registry_v1"
SOUND_ASSET_REGISTRY_SCHEMA = "avengine_m6_sound_asset_registry_v1"
FLAG_DEFINITION_REGISTRY_SCHEMA = "avengine_m6_flag_definition_registry_v1"
AUDIO_PROGRAM_SCHEMA = "avengine_m6_audio_program_v1"

SCHEMA_FILES = {
    ENTITY_ASSET_REGISTRY_SCHEMA: "m6_entity_asset_registry_v1.schema.json",
    ANIMAL_TEMPLATE_REGISTRY_SCHEMA: "m6_animal_template_registry_v1.schema.json",
    SOURCE_ENDPOINT_REGISTRY_SCHEMA: "m6_source_endpoint_registry_v1.schema.json",
    SOUND_ASSET_REGISTRY_SCHEMA: "m6_sound_asset_registry_v1.schema.json",
    FLAG_DEFINITION_REGISTRY_SCHEMA: "m6_flag_definition_registry_v1.schema.json",
    AUDIO_PROGRAM_SCHEMA: "m6_audio_program_v1.schema.json",
}

_STABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class M6RegistryError(ValueError):
    """One or more M6 registry invariants failed."""

    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


def schema_path(schema_name: str) -> Path:
    """Return a source-tree or installed M6 schema path."""

    try:
        filename = SCHEMA_FILES[schema_name]
    except KeyError as exc:
        raise ValueError(f"unknown M6 schema: {schema_name!r}") from exc
    source = Path(__file__).resolve().parents[3] / "schemas" / filename
    installed = Path(sys.prefix) / "share" / "avengine" / "schemas" / filename
    path = source if source.is_file() else installed
    if not path.is_file():
        raise FileNotFoundError(f"AVEngine schema is unavailable: {filename}")
    return path


def json_schema_errors(value: Any, schema_name: str) -> list[str]:
    """Return stable, human-readable Draft 2020-12 validation errors."""

    schema = load_json(schema_path(schema_name))
    errors: list[str] = []
    for error in sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    ):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"JSON Schema {location}: {error.message}")
    return errors


def is_stable_id(value: Any) -> bool:
    return isinstance(value, str) and _STABLE_ID.fullmatch(value) is not None


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def all_numbers_finite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(all_numbers_finite(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(all_numbers_finite(item) for item in value)
    return False


def bind_content_hash(
    value: Mapping[str, Any], *, field: str = "registry_content_sha256"
) -> dict[str, Any]:
    """Return a detached document with its canonical outer hash rebound."""

    result = deepcopy(dict(value))
    result.pop(field, None)
    result[field] = canonical_json_sha256(result)
    return result


def content_hash_errors(
    value: Mapping[str, Any], *, field: str = "registry_content_sha256"
) -> list[str]:
    declared = value.get(field)
    if not is_sha256(declared):
        return [f"{field} must be a lowercase SHA-256"]
    payload = {key: item for key, item in value.items() if key != field}
    expected = canonical_json_sha256(payload)
    return [] if declared == expected else [f"{field} does not match canonical content"]


def registry_semantic_errors(
    value: Any,
    *,
    records_field: str,
    record_id_field: str,
    record_revision_field: str = "revision",
    require_sorted: bool = True,
) -> list[str]:
    """Check common ordering, uniqueness, finiteness, and outer-hash rules."""

    if not isinstance(value, Mapping):
        return ["registry must be a mapping"]
    errors: list[str] = []
    if not all_numbers_finite(value):
        errors.append("registry must contain only finite JSON numbers")
    errors.extend(content_hash_errors(value))
    records = value.get(records_field)
    if not isinstance(records, list):
        return errors

    keys: list[tuple[str, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        record_id = record.get(record_id_field)
        revision = record.get(record_revision_field)
        if is_stable_id(record_id) and is_stable_id(revision):
            keys.append((str(record_id), str(revision)))
        else:
            errors.append(
                f"{records_field}[{index}] requires stable {record_id_field} and "
                f"{record_revision_field}"
            )
    if len(keys) != len(set(keys)):
        errors.append(
            f"{records_field} must not repeat ({record_id_field}, "
            f"{record_revision_field})"
        )
    if require_sorted and keys != sorted(keys):
        errors.append(
            f"{records_field} must use canonical bytewise ID/revision order"
        )
    return errors


def record_index(
    registry: Mapping[str, Any],
    *,
    records_field: str,
    record_id_field: str,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Index a previously validated registry without choosing a fallback revision."""

    return {
        (str(record[record_id_field]), str(record["revision"])): record
        for record in registry[records_field]
    }


def resolve_record(
    registry: Mapping[str, Any],
    *,
    records_field: str,
    record_id_field: str,
    record_id: str,
    revision: str,
) -> Mapping[str, Any]:
    """Resolve one exact ID/revision pair; never select a latest revision."""

    try:
        return record_index(
            registry,
            records_field=records_field,
            record_id_field=record_id_field,
        )[(record_id, revision)]
    except KeyError as exc:
        raise KeyError(
            f"unregistered {record_id_field} revision: {record_id}@{revision}"
        ) from exc


def load_validated_document(
    path: str | Path,
    *,
    validator: Callable[[Any], list[str]],
) -> dict[str, Any]:
    """Load one registry/program and fail closed on every reported error."""

    value = load_json(path)
    errors = validator(value)
    if errors:
        raise M6RegistryError(errors)
    return value
