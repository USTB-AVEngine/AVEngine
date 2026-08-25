"""M6 extensibility interfaces; not a final dataset or QA contract."""

from avengine.registry.entities import (
    AnimalTemplateSelection,
    load_animal_template_registry,
    load_entity_asset_registry,
    select_animal_template,
    validate_animal_template_registry,
    validate_entity_asset_registry,
    validate_entity_template_bindings,
)
from avengine.registry.exporter_interface import (
    EvidenceArtifactRef,
    ReadOnlyEvidenceBundle,
    TaskExporter,
)
from avengine.registry.flags import (
    aggregate_legacy_status,
    evaluate_legacy_flags,
    legacy_flag_access,
    load_legacy_flag_registry,
    provider_assessment,
    validate_legacy_flag_registry,
)
from avengine.registry.registry import M6RegistryError
from avengine.registry.sources import (
    load_sound_asset_registry,
    load_source_endpoint_registry,
    resolve_source_endpoint_bindings,
    validate_sound_asset_registry,
    validate_source_endpoint_registry,
)
from avengine.registry.static_objects import (
    StaticObjectRegistrationError,
    publish_static_object_entity_registry,
    resolve_static_object_emitter_world,
    validate_static_object_admission,
    verify_static_object_entity_registry,
)

__all__ = [
    "AnimalTemplateSelection",
    "EvidenceArtifactRef",
    "M6RegistryError",
    "ReadOnlyEvidenceBundle",
    "TaskExporter",
    "StaticObjectRegistrationError",
    "aggregate_legacy_status",
    "evaluate_legacy_flags",
    "legacy_flag_access",
    "load_animal_template_registry",
    "load_entity_asset_registry",
    "load_legacy_flag_registry",
    "load_sound_asset_registry",
    "load_source_endpoint_registry",
    "provider_assessment",
    "publish_static_object_entity_registry",
    "resolve_static_object_emitter_world",
    "resolve_source_endpoint_bindings",
    "select_animal_template",
    "validate_animal_template_registry",
    "validate_entity_asset_registry",
    "validate_entity_template_bindings",
    "validate_legacy_flag_registry",
    "validate_sound_asset_registry",
    "validate_source_endpoint_registry",
    "validate_static_object_admission",
    "verify_static_object_entity_registry",
]
