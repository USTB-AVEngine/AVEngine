"""M6 extensibility interfaces; not a final dataset or QA contract."""

from avengine.timeline.audio_program import (
    AudioProgramError,
    CompiledAudioProgram,
    bind_audio_program_hash,
    compile_audio_program,
    compile_audio_program_variant,
    load_audio_program,
    materialize_audio_program_variant,
    validate_audio_program,
)
from avengine.timeline.audio_render import (
    AudioProgramDryAssembly,
    assemble_audio_program_dry_buses,
)
from avengine.m6.canary import (
    M6CanaryError,
    bind_controlled_canary_request_hash,
    load_controlled_canary_request,
    run_controlled_canary,
    validate_controlled_canary_request,
    verify_controlled_canary_evidence,
)
from avengine.m6.entities import (
    AnimalTemplateSelection,
    load_animal_template_registry,
    load_entity_asset_registry,
    select_animal_template,
    validate_animal_template_registry,
    validate_entity_asset_registry,
    validate_entity_template_bindings,
)
from avengine.m6.exporter_interface import (
    EvidenceArtifactRef,
    ReadOnlyEvidenceBundle,
    TaskExporter,
)
from avengine.m6.flags import (
    aggregate_legacy_status,
    evaluate_legacy_flags,
    legacy_flag_access,
    load_legacy_flag_registry,
    provider_assessment,
    validate_legacy_flag_registry,
)
from avengine.m6.registry import M6RegistryError
from avengine.m6.sources import (
    load_sound_asset_registry,
    load_source_endpoint_registry,
    resolve_source_endpoint_bindings,
    validate_sound_asset_registry,
    validate_source_endpoint_registry,
)
from avengine.m6.static_objects import (
    StaticObjectRegistrationError,
    publish_static_object_entity_registry,
    resolve_static_object_emitter_world,
    validate_static_object_admission,
    verify_static_object_entity_registry,
)

__all__ = [
    "AnimalTemplateSelection",
    "AudioProgramDryAssembly",
    "AudioProgramError",
    "CompiledAudioProgram",
    "EvidenceArtifactRef",
    "M6RegistryError",
    "M6CanaryError",
    "ReadOnlyEvidenceBundle",
    "TaskExporter",
    "StaticObjectRegistrationError",
    "aggregate_legacy_status",
    "assemble_audio_program_dry_buses",
    "bind_audio_program_hash",
    "bind_controlled_canary_request_hash",
    "compile_audio_program",
    "compile_audio_program_variant",
    "evaluate_legacy_flags",
    "legacy_flag_access",
    "load_animal_template_registry",
    "load_audio_program",
    "load_controlled_canary_request",
    "load_entity_asset_registry",
    "load_legacy_flag_registry",
    "load_sound_asset_registry",
    "load_source_endpoint_registry",
    "materialize_audio_program_variant",
    "provider_assessment",
    "publish_static_object_entity_registry",
    "resolve_static_object_emitter_world",
    "resolve_source_endpoint_bindings",
    "run_controlled_canary",
    "select_animal_template",
    "validate_animal_template_registry",
    "validate_audio_program",
    "validate_controlled_canary_request",
    "validate_entity_asset_registry",
    "validate_entity_template_bindings",
    "validate_legacy_flag_registry",
    "validate_sound_asset_registry",
    "validate_source_endpoint_registry",
    "validate_static_object_admission",
    "verify_controlled_canary_evidence",
    "verify_static_object_entity_registry",
]
