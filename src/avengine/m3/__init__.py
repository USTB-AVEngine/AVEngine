"""M3 explicit acoustic-scene packages, compiler QA, and evidence."""

from avengine.m3.calibration import (
    BroadbandEDTCalibrationError,
    BroadbandEDTCalibrationResult,
    calibrate_broadband_edt_seconds,
)

from avengine.m3.compiler import (
    AcousticSceneCompileError,
    compile_canary_request,
    compile_custom_acoustic_scene,
    compile_explicit_glb_research_scene,
    compile_mp3d_soundspaces_research_scene,
    compile_usd_snapshot_semantic_research_scene,
    propose_visual_slot_research_materials,
)
from avengine.m3.contracts import (
    AcousticSceneContractError,
    ImmutableFileSnapshot,
    ValidatedAcousticScenePackage,
    load_and_validate_acoustic_scene_package,
    load_and_validate_package,
    read_immutable_file_snapshot,
    validate_package,
)
from avengine.m3.evidence import (
    VerifiedCompileEvidence,
    load_and_verify_compile_evidence,
    verify_compile_evidence,
)
from avengine.m3.materials import (
    ResolvedMaterialProfile,
    resolve_material_profile,
    validate_material_profile,
)
from avengine.m3.real_rir_reference import (
    RealRIRReferenceError,
    verify_soundspaces2_real_rir_reference,
)
from avengine.m3.rlr_material_import import (
    CompiledRLRSemanticMaterials,
    RLRMaterialImportError,
    build_rlr_material_import_report,
    compile_rlr_semantic_material_documents,
    import_rlr_material_database,
    rlr_document_from_native_database,
)

__all__ = [
    "AcousticSceneCompileError",
    "AcousticSceneContractError",
    "BroadbandEDTCalibrationError",
    "BroadbandEDTCalibrationResult",
    "ImmutableFileSnapshot",
    "ValidatedAcousticScenePackage",
    "VerifiedCompileEvidence",
    "compile_canary_request",
    "compile_custom_acoustic_scene",
    "compile_explicit_glb_research_scene",
    "compile_mp3d_soundspaces_research_scene",
    "compile_usd_snapshot_semantic_research_scene",
    "calibrate_broadband_edt_seconds",
    "load_and_validate_acoustic_scene_package",
    "load_and_validate_package",
    "load_and_verify_compile_evidence",
    "propose_visual_slot_research_materials",
    "read_immutable_file_snapshot",
    "RealRIRReferenceError",
    "RLRMaterialImportError",
    "CompiledRLRSemanticMaterials",
    "ResolvedMaterialProfile",
    "resolve_material_profile",
    "validate_package",
    "validate_material_profile",
    "verify_compile_evidence",
    "verify_soundspaces2_real_rir_reference",
    "build_rlr_material_import_report",
    "compile_rlr_semantic_material_documents",
    "import_rlr_material_database",
    "rlr_document_from_native_database",
]
