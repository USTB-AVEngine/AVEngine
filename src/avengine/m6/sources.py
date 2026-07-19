"""Named source-endpoint and dry-sound registries for M6."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from avengine.m6.entities import validate_entity_asset_registry
from avengine.m6.registry import (
    M6RegistryError,
    SOUND_ASSET_REGISTRY_SCHEMA,
    SOURCE_ENDPOINT_REGISTRY_SCHEMA,
    json_schema_errors,
    load_validated_document,
    registry_semantic_errors,
)


def _canonical_string_list(value: Any, owner: str) -> list[str]:
    if not isinstance(value, list):
        return []
    return [] if value == sorted(set(value)) else [f"{owner} must be unique and canonical"]


def validate_source_endpoint_registry(value: Any) -> list[str]:
    errors = json_schema_errors(value, SOURCE_ENDPOINT_REGISTRY_SCHEMA)
    errors.extend(
        registry_semantic_errors(
            value,
            records_field="source_endpoints",
            record_id_field="source_endpoint_id",
        )
    )
    if not isinstance(value, Mapping) or not isinstance(value.get("source_endpoints"), list):
        return errors
    for index, endpoint in enumerate(value["source_endpoints"]):
        if not isinstance(endpoint, Mapping):
            continue
        errors.extend(
            _canonical_string_list(
                endpoint.get("allowed_sound_class_ids"),
                f"source_endpoints[{index}].allowed_sound_class_ids",
            )
        )
    return errors


def validate_sound_asset_registry(value: Any) -> list[str]:
    errors = json_schema_errors(value, SOUND_ASSET_REGISTRY_SCHEMA)
    errors.extend(
        registry_semantic_errors(
            value,
            records_field="sound_assets",
            record_id_field="sound_asset_id",
        )
    )
    if not isinstance(value, Mapping) or not isinstance(value.get("sound_assets"), list):
        return errors
    for index, sound in enumerate(value["sound_assets"]):
        if not isinstance(sound, Mapping):
            continue
        taxonomy_path = sound.get("taxonomy_path")
        if isinstance(taxonomy_path, list) and len(taxonomy_path) != len(
            set(taxonomy_path)
        ):
            errors.append(
                f"sound_assets[{index}].taxonomy_path must not repeat a node"
            )
        errors.extend(
            _canonical_string_list(
                sound.get("tags"), f"sound_assets[{index}].tags"
            )
        )
        errors.extend(
            _canonical_string_list(
                sound.get("permitted_event_usage"),
                f"sound_assets[{index}].permitted_event_usage",
            )
        )
    return errors


def load_source_endpoint_registry(path: str | Path) -> dict[str, Any]:
    return load_validated_document(path, validator=validate_source_endpoint_registry)


def load_sound_asset_registry(path: str | Path) -> dict[str, Any]:
    return load_validated_document(path, validator=validate_sound_asset_registry)


@dataclass(frozen=True)
class ResolvedSourceEndpoint:
    source_endpoint_id: str
    revision: str
    entity_instance_id: str | None
    entity_asset_id: str | None
    entity_asset_revision: str | None
    emitter_anchor_id: str
    source_visibility_mode: str
    allowed_sound_class_ids: tuple[str, ...]
    persistent_when_silent: bool


def resolve_source_endpoint_bindings(
    endpoint_registry: Mapping[str, Any], entity_registry: Mapping[str, Any]
) -> tuple[ResolvedSourceEndpoint, ...]:
    """Resolve every entity-anchor binding without creating runtime sources."""

    errors = validate_source_endpoint_registry(endpoint_registry)
    errors.extend(validate_entity_asset_registry(entity_registry))
    if errors:
        raise M6RegistryError(errors)
    entities = {
        (item["entity_asset_id"], item["revision"]): item
        for item in entity_registry["entities"]
    }
    resolved: list[ResolvedSourceEndpoint] = []
    binding_errors: list[str] = []
    for index, endpoint in enumerate(endpoint_registry["source_endpoints"]):
        binding = endpoint["binding"]
        if binding["kind"] == "entity_anchor":
            key = (binding["entity_asset_id"], binding["entity_asset_revision"])
            entity = entities.get(key)
            if entity is None:
                binding_errors.append(f"source_endpoints[{index}] references an unregistered entity asset")
                continue
            anchor_ids = {item["anchor_id"] for item in entity["emitter_anchors"]}
            if binding["emitter_anchor_id"] not in anchor_ids:
                binding_errors.append(f"source_endpoints[{index}] references an unregistered emitter anchor")
                continue
            entity_instance_id = binding["entity_instance_id"]
            entity_asset_id = binding["entity_asset_id"]
            entity_asset_revision = binding["entity_asset_revision"]
        else:
            entity_instance_id = None
            entity_asset_id = None
            entity_asset_revision = None
        resolved.append(
            ResolvedSourceEndpoint(
                source_endpoint_id=endpoint["source_endpoint_id"],
                revision=endpoint["revision"],
                entity_instance_id=entity_instance_id,
                entity_asset_id=entity_asset_id,
                entity_asset_revision=entity_asset_revision,
                emitter_anchor_id=binding["emitter_anchor_id"],
                source_visibility_mode=endpoint["source_visibility_mode"],
                allowed_sound_class_ids=tuple(endpoint["allowed_sound_class_ids"]),
                persistent_when_silent=endpoint["persistent_when_silent"],
            )
        )
    if binding_errors:
        raise M6RegistryError(binding_errors)
    return tuple(resolved)


def endpoint_index(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    errors = validate_source_endpoint_registry(registry)
    if errors:
        raise M6RegistryError(errors)
    return {item["source_endpoint_id"]: item for item in registry["source_endpoints"]}


def sound_index(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    errors = validate_sound_asset_registry(registry)
    if errors:
        raise M6RegistryError(errors)
    return {item["sound_asset_id"]: item for item in registry["sound_assets"]}


def source_sound_compatibility_errors(
    endpoint: Mapping[str, Any], sound: Mapping[str, Any], *, owner: str
) -> list[str]:
    errors: list[str] = []
    if sound["semantic_sound_class"] not in endpoint["allowed_sound_class_ids"]:
        errors.append(
            f"{owner} sound class {sound['semantic_sound_class']!r} is not allowed "
            f"by endpoint {endpoint['source_endpoint_id']!r}"
        )
    if sound["admissibility"] in {"unavailable", "rejected"}:
        errors.append(
            f"{owner} sound asset {sound['sound_asset_id']!r} is not usable: "
            f"{sound['admissibility']}"
        )
    return errors
