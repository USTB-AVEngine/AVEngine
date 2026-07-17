"""Research-only assembly of hash-closed M2 quadruped animal packages.

This boundary supports only the explicitly registered research quadruped body
plans below.  It binds taxonomy, identity, quadruped semantic anchors and real
technical evidence without fabricating QA reports, and it can only emit
``research_candidate``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

from avengine.appearance import AppearanceContractError, validate_l9_batch
from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    sha256_file,
)
from avengine.m2.contracts import CONTACT_ORDER
from avengine.m2.cross_species_lineage import (
    CROSS_SPECIES_APPEARANCE_LINEAGE_SCHEMA,
    CrossSpeciesLineageError,
    validate_cross_species_appearance_lineage,
    validate_repository_file_identity,
)
from avengine.m2.glb import GlbError, load_glb
from avengine.m2.materials import (
    MAXIMUM_SPECULAR_COLOR_FACTOR,
    MAXIMUM_SPECULAR_FACTOR,
    MINIMUM_ROUGHNESS_FACTOR,
    ZERO_EMISSIVE_FACTOR,
    load_and_validate_material_normalization_report,
)
from avengine.m2.package import (
    AnimalPackageIdentity,
    PackageCompileError,
    compile_research_candidate_animal_package,
)


VARIANT_PACKAGE_SPEC_SCHEMA = "avengine_m2_variant_package_spec_v1"
VARIANT_SOURCE_BINDING_SCHEMA = "avengine_m2_variant_source_binding_v2"
APPEARANCE_LINEAGE_SCHEMA = "avengine_m2_appearance_variant_lineage_v1"
_APPEARANCE_BATCH_SCHEMA = "avengine_animal_appearance_batch_v1"
_APPEARANCE_REALIZATION_SCHEMA = "avengine_animal_appearance_realization_v1"
_L9_PRODUCER_RELATIVE_PATH = "tools/m2/build_appearance_variant_inputs.py"
_SPECULAR_EXTENSION = "KHR_materials_specular"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_L9_OUTPUT_FLOAT_TOLERANCE = 5.0e-5
_L9_EXPORT_PROFILE = {
    "format": "GLB",
    "animation_mode": "ACTIONS",
    "force_sampling": True,
    "skins": True,
    "texcoords": True,
    "normals": True,
    "image_format": "AUTO",
}
# This profile is deliberately quadruped-specific: one muzzle emitter plus the
# four named paw contacts.  A different anatomy needs a new reviewed profile,
# not inference through this fixed contract.
_RESEARCH_QUADRUPED_REQUIRED_ANCHORS = ("body", "head", "muzzle", *CONTACT_ORDER)
_RESEARCH_QUADRUPED_PROFILE_BY_TAXONOMY = {
    ("canis_lupus_familiaris", "beagle"): {
        "body_plan_id": "quadruped_mammal_canid_v1",
        "coat_profiles": {
            "light_tricolor",
            "standard_tricolor",
            "dark_tricolor",
        },
    },
    ("canis_lupus_familiaris", "golden_retriever"): {
        "body_plan_id": "quadruped_mammal_canid_v1",
        "coat_profiles": {"light_golden", "classic_golden", "dark_golden"},
    },
    ("felis_catus", "generic"): {
        "body_plan_id": "quadruped_mammal_felid_v1",
        "coat_profiles": {"black", "charcoal_gray", "silver_gray"},
    },
    ("equus_caballus", "generic"): {
        "body_plan_id": "quadruped_mammal_equid_v1",
        "coat_profiles": {"black", "dark_bay", "bay"},
    },
}
_APPEARANCE_FIELDS = {"size", "body_build", "coat", "life_stage"}
_APPEARANCE_DOMAINS = {
    "size": {"small", "medium", "large"},
    "body_build": {"slim", "standard", "stocky"},
    "life_stage": {"young", "adult", "senior"},
}
_IDENTITY_FIELDS = {
    "asset_id",
    "template_id",
    "body_plan_id",
    "morphotype_id",
    "skeleton_revision",
    "weights_revision",
    "collision_revision",
    "action_revision",
    "source",
    "source_revision",
    "license",
    "allowed_use",
    "redistribution",
    "semantic_id",
}
_REQUIRED_EVIDENCE_SCHEMAS = {
    "rebase_report": "avengine_m2_skin_root_rebase_v1",
    "rebase_deformation_report": ("avengine_m2_rebase_deformation_verification_v1"),
    "action_report": "avengine_m2_action_bake_report_v1",
    "static_qa": "avengine_m2_static_geometry_qa_v1",
    "deformation_qa": "avengine_m2_deformation_qa_v1",
    "animation_qa": "avengine_m2_animation_qa_v1",
    "habitat_static_probe": "avengine_m2_habitat_skin_rest_probe_v1",
    "habitat_animation_review": "avengine_m2_habitat_action_review_v1",
}


class VariantPackageError(RuntimeError):
    """The research quadruped variant package inputs are incomplete or unsafe."""


@dataclass(frozen=True)
class VariantPackageSpec:
    """Validated identity, taxonomy and semantic anchors for one variant."""

    path: Path
    sha256: str
    value: Mapping[str, Any]
    identity: AnimalPackageIdentity
    anchors: tuple[Mapping[str, Any], ...]
    shader_type: str

    @property
    def semantic_joint_map(self) -> dict[str, str]:
        by_id = {
            str(anchor["anchor_id"]): str(anchor["joint_id"]) for anchor in self.anchors
        }
        return {
            anchor_id: by_id[anchor_id]
            for anchor_id in _RESEARCH_QUADRUPED_REQUIRED_ANCHORS
        }


@dataclass(frozen=True)
class VariantPackageEvidence:
    """All independently produced reports required by the strict compiler."""

    visual_glb: Path
    rebase_report: Path
    rebase_deformation_report: Path
    action_report: Path
    static_qa: Path
    deformation_qa: Path
    animation_qa: Path
    habitat_static_probe: Path
    habitat_animation_review: Path
    baked_actions: Path
    contacts: Path
    source_manifest: Path
    license_snapshot: Path
    appearance_lineage: Path
    material_normalization_report: Path


def _regular_file(path: str | Path, *, owner: str) -> Path:
    raw = Path(path)
    absolute = Path(os.path.abspath(raw))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise VariantPackageError(f"{owner} must not contain a symbolic link")
    if not absolute.is_file():
        raise VariantPackageError(f"{owner} is not a regular file: {absolute}")
    return absolute


def _nonempty_string(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VariantPackageError(f"{owner} must be a non-empty string")
    return value


def _anchor(value: Any, *, index: int) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VariantPackageError(f"anchors[{index}] must be an object")
    if set(value) != {"anchor_id", "joint_id", "joint_from_anchor"}:
        raise VariantPackageError(
            f"anchors[{index}] must contain exactly anchor_id, joint_id, "
            "and joint_from_anchor"
        )
    _nonempty_string(value.get("anchor_id"), owner=f"anchors[{index}].anchor_id")
    _nonempty_string(value.get("joint_id"), owner=f"anchors[{index}].joint_id")
    transform = value.get("joint_from_anchor")
    if not isinstance(transform, Mapping) or set(transform) != {
        "translation_m",
        "rotation_xyzw",
    }:
        raise VariantPackageError(
            f"anchors[{index}].joint_from_anchor must contain translation_m "
            "and rotation_xyzw"
        )
    return dict(value)


def load_variant_package_spec(path: str | Path) -> VariantPackageSpec:
    """Load a strict registered-quadruped spec without inferring anatomy."""

    resolved = _regular_file(path, owner="variant package spec")
    value = load_json(resolved)
    allowed_keys = {
        "schema",
        "identity",
        "taxonomy",
        "appearance",
        "rendering",
        "anchors",
    }
    if set(value) - allowed_keys:
        raise VariantPackageError(
            f"variant package spec has unknown keys: {sorted(set(value) - allowed_keys)}"
        )
    if value.get("schema") != VARIANT_PACKAGE_SPEC_SCHEMA:
        raise VariantPackageError(
            f"variant package spec schema must be {VARIANT_PACKAGE_SPEC_SCHEMA!r}"
        )
    taxonomy = value.get("taxonomy")
    if not isinstance(taxonomy, Mapping) or set(taxonomy) != {
        "species_id",
        "breed_id",
    }:
        raise VariantPackageError(
            "taxonomy must contain exactly non-empty species_id and breed_id"
        )
    species_id = _nonempty_string(
        taxonomy.get("species_id"), owner="taxonomy.species_id"
    )
    breed_id = _nonempty_string(taxonomy.get("breed_id"), owner="taxonomy.breed_id")
    identity_value = value.get("identity")
    if (
        not isinstance(identity_value, Mapping)
        or set(identity_value) != _IDENTITY_FIELDS
    ):
        raise VariantPackageError(
            "identity must contain exactly the AnimalPackageIdentity fields"
        )
    try:
        identity = AnimalPackageIdentity(**dict(identity_value))
    except (TypeError, ValueError) as exc:
        raise VariantPackageError(f"invalid package identity: {exc}") from exc
    profile = _RESEARCH_QUADRUPED_PROFILE_BY_TAXONOMY.get((species_id, breed_id))
    if profile is None:
        raise VariantPackageError(
            f"taxonomy pair {(species_id, breed_id)!r} has no registered research "
            "quadruped appearance/body-plan profile; add and review a new profile"
        )
    expected_body_plan_id = profile["body_plan_id"]
    if identity.body_plan_id != expected_body_plan_id:
        raise VariantPackageError(
            f"identity.body_plan_id must be {expected_body_plan_id!r} for "
            f"taxonomy.species_id {species_id!r}"
        )
    raw_anchors = value.get("anchors")
    if not isinstance(raw_anchors, list):
        raise VariantPackageError("anchors must be an explicit array")
    anchors = tuple(
        _anchor(item, index=index) for index, item in enumerate(raw_anchors)
    )
    ids = [str(anchor["anchor_id"]) for anchor in anchors]
    if len(ids) != len(set(ids)):
        raise VariantPackageError("anchors contain duplicate anchor IDs")
    missing = sorted(set(_RESEARCH_QUADRUPED_REQUIRED_ANCHORS) - set(ids))
    if missing:
        raise VariantPackageError(f"anchors are missing required IDs: {missing}")
    appearance = value.get("appearance")
    if not isinstance(appearance, Mapping) or set(appearance) != _APPEARANCE_FIELDS:
        raise VariantPackageError(
            "appearance must contain exactly size, body_build, coat, and life_stage"
        )
    for field_name, domain in _APPEARANCE_DOMAINS.items():
        field_value = appearance.get(field_name)
        if not isinstance(field_value, str) or field_value not in domain:
            raise VariantPackageError(
                f"appearance.{field_name} is not registered for research variants"
            )
    coat = appearance.get("coat")
    if not isinstance(coat, str) or coat not in profile["coat_profiles"]:
        raise VariantPackageError(
            "appearance.coat is not registered for this exact species/breed pair"
        )
    rendering = value.get("rendering")
    if not isinstance(rendering, Mapping) or set(rendering) != {"shader_type"}:
        raise VariantPackageError(
            "rendering must contain exactly the explicit shader_type"
        )
    shader_type = rendering.get("shader_type")
    if not isinstance(shader_type, str) or shader_type not in {"phong", "pbr"}:
        raise VariantPackageError("rendering.shader_type must be 'phong' or 'pbr'")
    return VariantPackageSpec(
        path=resolved,
        sha256=sha256_file(resolved),
        value=value,
        identity=identity,
        anchors=anchors,
        shader_type=shader_type,
    )


def _load_evidence_json(path: Path, *, owner: str) -> Mapping[str, Any]:
    resolved = _regular_file(path, owner=owner)
    try:
        return load_json(resolved)
    except (OSError, ValueError) as exc:
        raise VariantPackageError(f"{owner} is invalid JSON: {exc}") from exc


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validated_file_binding(value: Any, *, owner: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "byte_size",
        "sha256",
    }:
        raise VariantPackageError(
            f"{owner} must contain exactly path, byte_size, and sha256"
        )
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise VariantPackageError(f"{owner}.path must be a non-empty string")
    path = _regular_file(raw_path, owner=owner)
    actual = _file_record(path)
    if value.get("byte_size") != actual["byte_size"]:
        raise VariantPackageError(f"{owner}.byte_size does not bind its file")
    if value.get("sha256") != actual["sha256"]:
        raise VariantPackageError(f"{owner}.sha256 does not bind its file")
    return path, actual


def _validated_json_binding(
    value: Any,
    *,
    owner: str,
) -> tuple[Path, Mapping[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "byte_size",
        "sha256",
        "canonical_content_sha256",
        "snapshot",
    }:
        raise VariantPackageError(f"{owner} JSON binding fields are invalid")
    path, _ = _validated_file_binding(
        {
            "path": value.get("path"),
            "byte_size": value.get("byte_size"),
            "sha256": value.get("sha256"),
        },
        owner=owner,
    )
    current = _load_evidence_json(path, owner=owner)
    if value.get("snapshot") != current:
        raise VariantPackageError(f"{owner}.snapshot differs from its actual file")
    if value.get("canonical_content_sha256") != canonical_json_sha256(current):
        raise VariantPackageError(
            f"{owner}.canonical_content_sha256 does not bind its actual file"
        )
    return path, current


def _require_json_binding_matches_path(
    value: Any,
    expected_path: str | Path,
    *,
    owner: str,
) -> None:
    """Close a lineage JSON input over the independently supplied evidence file."""

    _, bound_value = _validated_json_binding(value, owner=owner)
    resolved = _regular_file(expected_path, owner=f"expected {owner}")
    expected_value = _load_evidence_json(resolved, owner=f"expected {owner}")
    binding = value
    assert isinstance(binding, Mapping)
    if (
        binding.get("byte_size") != resolved.stat().st_size
        or binding.get("sha256") != sha256_file(resolved)
        or binding.get("canonical_content_sha256")
        != canonical_json_sha256(expected_value)
        or binding.get("snapshot") != expected_value
        or bound_value != expected_value
    ):
        raise VariantPackageError(
            f"{owner} differs from the separately supplied file identity"
        )


def _same_file_identity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    owner: str,
) -> None:
    if left.get("byte_size") != right.get("byte_size") or left.get(
        "sha256"
    ) != right.get("sha256"):
        raise VariantPackageError(
            f"{owner} does not close over identical visual bytes and SHA-256"
        )


def _validate_l9_tool_identity(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "material_normalizer",
        "blender_version",
        "export_profile",
        "output_readback_float_tolerance",
    }:
        raise VariantPackageError("L9 appearance tool_identity fields are invalid")
    try:
        validate_repository_file_identity(
            value,
            repository_root=_REPOSITORY_ROOT,
            repository_relative_path=("tools/blender/realize_animal_appearance.py"),
            owner="L9 appearance realizer",
            require_byte_size=False,
        )
        material = value.get("material_normalizer")
        if not isinstance(material, Mapping) or set(material) != {"path", "sha256"}:
            raise VariantPackageError(
                "L9 appearance material_normalizer fields are invalid"
            )
        validate_repository_file_identity(
            material,
            repository_root=_REPOSITORY_ROOT,
            repository_relative_path="src/avengine/m2/materials.py",
            owner="L9 appearance material normalizer",
            require_byte_size=False,
        )
    except CrossSpeciesLineageError as exc:
        raise VariantPackageError(
            f"L9 canonical tool identity is invalid: {exc}"
        ) from exc
    if (
        not isinstance(value.get("blender_version"), str)
        or not value["blender_version"].strip()
        or value.get("export_profile") != _L9_EXPORT_PROFILE
    ):
        raise VariantPackageError("L9 appearance realizer profile is invalid")
    tolerance = value.get("output_readback_float_tolerance")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) != _L9_OUTPUT_FLOAT_TOLERANCE
    ):
        raise VariantPackageError("L9 appearance output tolerance is invalid")
    return value


def _l9_batch_request(
    batch: Mapping[str, Any],
    instance: Any,
) -> Mapping[str, Any]:
    if not isinstance(instance, Mapping) or set(instance) != {
        "instance_request_id",
        "request_sha256",
        "ordinal",
        "taxonomy",
        "attributes",
    }:
        raise VariantPackageError("appearance_lineage.instance_request is invalid")
    ordinal = instance.get("ordinal")
    requests = batch.get("requests")
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or not isinstance(requests, list)
        or not 1 <= ordinal <= len(requests)
    ):
        raise VariantPackageError("appearance lineage L9 ordinal is invalid")
    request = requests[ordinal - 1]
    if not isinstance(request, Mapping):
        raise VariantPackageError("appearance lineage L9 request is invalid")
    expected_instance = {
        "instance_request_id": request.get("instance_request_id"),
        "request_sha256": request.get("request_sha256"),
        "ordinal": request.get("ordinal"),
        "taxonomy": request.get("taxonomy"),
        "attributes": request.get("attributes"),
    }
    if instance != expected_instance:
        raise VariantPackageError(
            "appearance lineage instance differs from its authenticated L9 batch"
        )
    return request


def _validate_l9_realization(
    realization: Mapping[str, Any],
    *,
    batch: Mapping[str, Any],
    batch_path: Path,
    request: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    expected_digest = canonical_json_sha256(
        {
            key: value
            for key, value in realization.items()
            if key != "report_content_sha256"
        }
    )
    if realization.get("report_content_sha256") != expected_digest:
        raise VariantPackageError(
            "appearance realization report_content_sha256 is invalid"
        )
    if (
        realization.get("schema") != _APPEARANCE_REALIZATION_SCHEMA
        or realization.get("status") != "pass"
        or realization.get("state_classification") != "research_candidate"
        or realization.get("qualification_claim") is not False
        or realization.get("formal_dataset_registration_authorized") is not False
    ):
        raise VariantPackageError(
            "appearance realization input must be passing, non-qualifying evidence"
        )
    report_batch = realization.get("batch")
    if not isinstance(report_batch, Mapping) or set(report_batch) != {
        "path",
        "sha256",
        "batch_id",
        "batch_content_sha256",
    }:
        raise VariantPackageError("appearance realization batch binding is invalid")
    raw_batch_path = report_batch.get("path")
    if (
        not isinstance(raw_batch_path, str)
        or _regular_file(raw_batch_path, owner="appearance realization batch")
        != batch_path
        or report_batch.get("sha256") != sha256_file(batch_path)
        or report_batch.get("batch_id") != batch.get("batch_id")
        or report_batch.get("batch_content_sha256") != batch.get("batch_content_sha256")
    ):
        raise VariantPackageError(
            "appearance realization does not bind the exact L9 batch"
        )
    expected_report_instance = {
        "ordinal": request.get("ordinal"),
        "instance_request_id": request.get("instance_request_id"),
        "request_sha256": request.get("request_sha256"),
        "taxonomy": request.get("taxonomy"),
        "attributes": request.get("attributes"),
        "realization_operations": request.get("realization_operations"),
    }
    if realization.get("instance_request") != expected_report_instance:
        raise VariantPackageError(
            "appearance realization does not bind the exact L9 request"
        )
    _, report_source = _validated_file_binding(
        realization.get("source"), owner="appearance realization source"
    )
    batch_source = batch.get("source_asset")
    if not isinstance(batch_source, Mapping):
        raise VariantPackageError("appearance batch source_asset is invalid")
    batch_source_path, batch_source_record = _validated_file_binding(
        {
            "path": batch_source.get("path"),
            "byte_size": batch_source.get("byte_size"),
            "sha256": batch_source.get("sha256"),
        },
        owner="appearance batch source_asset",
    )
    report_source_path = _regular_file(
        str(realization["source"]["path"]), owner="appearance realization source"
    )
    _same_file_identity(
        report_source,
        batch_source_record,
        owner="appearance realization source to L9 batch source",
    )
    if report_source_path != batch_source_path:
        raise VariantPackageError(
            "appearance realization source path differs from the L9 batch source"
        )
    tool_identity = _validate_l9_tool_identity(realization.get("tool_identity"))
    output = realization.get("output")
    if not isinstance(output, Mapping):
        raise VariantPackageError("appearance realization output is invalid")
    _, visual = _validated_file_binding(
        output.get("glb"), owner="appearance realization output.glb"
    )
    return visual, tool_identity


def _load_l9_producer_validator() -> tuple[type[Exception], Any]:
    """Load the exact canonical producer bytes without trusting ``sys.path``."""

    producer_path = _regular_file(
        _REPOSITORY_ROOT / _L9_PRODUCER_RELATIVE_PATH,
        owner="canonical L9 producer validator",
    )
    try:
        payload = producer_path.read_bytes()
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        validate_repository_file_identity(
            {
                "path": str(producer_path),
                "byte_size": len(payload),
                "sha256": payload_sha256,
            },
            repository_root=_REPOSITORY_ROOT,
            repository_relative_path=_L9_PRODUCER_RELATIVE_PATH,
            owner="canonical L9 producer validator",
            require_byte_size=True,
        )
    except (CrossSpeciesLineageError, OSError) as exc:
        raise VariantPackageError(
            f"the canonical L9 producer validator is unavailable: {exc}"
        ) from exc

    module_name = f"_avengine_m2_l9_producer_{payload_sha256}"
    module_spec = importlib.util.spec_from_file_location(module_name, producer_path)
    if module_spec is None:
        raise VariantPackageError(
            "unable to create a module spec for the canonical L9 producer validator"
        )
    module = importlib.util.module_from_spec(module_spec)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        # Execute the authenticated bytes directly.  ``exec_module`` may use a
        # timestamp-valid ``__pycache__`` file, which would not be the bytes
        # whose SHA-256 was checked above.
        code = compile(payload, str(producer_path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except Exception as exc:
        raise VariantPackageError(
            f"unable to load the canonical L9 producer validator: {exc}"
        ) from exc
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module

    if (
        producer_path.stat().st_size != len(payload)
        or sha256_file(producer_path) != payload_sha256
    ):
        raise VariantPackageError(
            "canonical L9 producer validator changed while it was being loaded"
        )
    error_type = getattr(module, "AppearanceVariantInputError", None)
    validator = getattr(module, "validate_spec_lineage_binding", None)
    if (
        not isinstance(error_type, type)
        or not issubclass(error_type, Exception)
        or not callable(validator)
    ):
        raise VariantPackageError("canonical L9 producer validator exports are invalid")
    return error_type, validator


def _validate_l9_producer_contract(
    spec: VariantPackageSpec,
    lineage: Mapping[str, Any],
) -> None:
    """Run the producer's complete L9 spec/lineage validator, fail closed."""

    error_type, validate_spec_lineage_binding = _load_l9_producer_validator()
    try:
        validate_spec_lineage_binding(
            spec.value,
            lineage,
            serialized_spec=spec.path.read_bytes(),
        )
    except (error_type, OSError, TypeError, ValueError) as exc:
        raise VariantPackageError(
            f"L9 producer contract failed full validation: {exc}"
        ) from exc


def _validate_l9_appearance_lineage(
    *,
    spec: VariantPackageSpec,
    evidence: VariantPackageEvidence,
    visual: Path,
    rebase_report: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> Mapping[str, Any]:
    required_root = {
        "schema",
        "status",
        "qualification_state",
        "qualification_claim",
        "formal_dataset_registration_authorized",
        "instance_request",
        "inputs",
        "derivative",
        "decision_reason",
        "lineage_content_sha256",
    }
    if set(lineage) != required_root:
        raise VariantPackageError("appearance_lineage root fields are invalid")
    if lineage.get("schema") != APPEARANCE_LINEAGE_SCHEMA:
        raise VariantPackageError(
            f"appearance_lineage schema must be {APPEARANCE_LINEAGE_SCHEMA!r}"
        )
    if (
        lineage.get("status") != "pass"
        or lineage.get("qualification_state") != "research_candidate"
        or lineage.get("qualification_claim") is not False
        or lineage.get("formal_dataset_registration_authorized") is not False
    ):
        raise VariantPackageError(
            "appearance_lineage must be passing, research-only evidence"
        )
    reason = lineage.get("decision_reason")
    if not isinstance(reason, str) or not reason.strip():
        raise VariantPackageError("appearance_lineage.decision_reason is invalid")
    expected_digest = canonical_json_sha256(
        {
            key: value
            for key, value in lineage.items()
            if key != "lineage_content_sha256"
        }
    )
    if lineage.get("lineage_content_sha256") != expected_digest:
        raise VariantPackageError(
            "appearance_lineage.lineage_content_sha256 is invalid"
        )

    inputs = lineage.get("inputs")
    required_inputs = {
        "appearance_batch",
        "appearance_realization_report",
        "template_variant_spec",
        "upstream_source_manifest",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != required_inputs:
        raise VariantPackageError("appearance_lineage input fields are invalid")
    input_values: dict[str, Mapping[str, Any]] = {}
    input_paths: dict[str, Path] = {}
    for name in sorted(required_inputs):
        input_paths[name], input_values[name] = _validated_json_binding(
            inputs[name],
            owner=f"appearance_lineage.inputs.{name}",
        )
    _require_json_binding_matches_path(
        inputs["upstream_source_manifest"],
        evidence.source_manifest,
        owner="appearance_lineage upstream source manifest",
    )
    _validate_l9_producer_contract(spec, lineage)
    if input_values["appearance_batch"].get("schema") != _APPEARANCE_BATCH_SCHEMA:
        raise VariantPackageError("appearance lineage batch schema is invalid")
    try:
        validate_l9_batch(input_values["appearance_batch"])
    except (AppearanceContractError, OSError, ValueError) as exc:
        raise VariantPackageError(
            f"appearance lineage L9 batch failed full validation: {exc}"
        ) from exc
    if (
        input_values["template_variant_spec"].get("schema")
        != VARIANT_PACKAGE_SPEC_SCHEMA
    ):
        raise VariantPackageError("appearance lineage template spec schema is invalid")
    upstream_schema = input_values["upstream_source_manifest"].get("schema")
    if not isinstance(upstream_schema, str) or not upstream_schema:
        raise VariantPackageError("appearance lineage upstream schema is invalid")

    instance = lineage.get("instance_request")
    request = _l9_batch_request(input_values["appearance_batch"], instance)
    realization = input_values["appearance_realization_report"]
    realization_visual, realization_tool = _validate_l9_realization(
        realization,
        batch=input_values["appearance_batch"],
        batch_path=input_paths["appearance_batch"],
        request=request,
    )

    derivative = lineage.get("derivative")
    if not isinstance(derivative, Mapping) or set(derivative) != {
        "pre_rebase_visual_glb",
        "tool_identity",
        "derived_variant_spec",
    }:
        raise VariantPackageError("appearance_lineage derivative fields are invalid")
    pre_rebase_path, pre_rebase_record = _validated_file_binding(
        derivative.get("pre_rebase_visual_glb"),
        owner="appearance_lineage.derivative.pre_rebase_visual_glb",
    )
    _same_file_identity(
        pre_rebase_record,
        realization_visual,
        owner="appearance lineage to realization visual binding",
    )
    if derivative.get("tool_identity") != realization_tool:
        raise VariantPackageError(
            "appearance lineage tool identity differs from its realization report"
        )

    spec_binding = derivative.get("derived_variant_spec")
    expected_spec_binding = {
        "schema": VARIANT_PACKAGE_SPEC_SCHEMA,
        "asset_id": spec.identity.asset_id,
        "byte_size": spec.path.stat().st_size,
        "sha256": spec.sha256,
        "canonical_content_sha256": canonical_json_sha256(spec.value),
    }
    if spec_binding != expected_spec_binding:
        raise VariantPackageError(
            "appearance_lineage does not bind the exact derived spec bytes"
        )

    assert isinstance(instance, Mapping)
    attributes = instance.get("attributes")
    appearance = spec.value.get("appearance")
    if not isinstance(attributes, Mapping) or appearance != {
        "size": attributes.get("size"),
        "body_build": attributes.get("body_build"),
        "coat": attributes.get("coat_profile"),
        "life_stage": attributes.get("life_stage"),
    }:
        raise VariantPackageError(
            "appearance_lineage attributes differ from the derived spec"
        )

    rebase_source_path, rebase_source = _validated_file_binding(
        rebase_report.get("source"),
        owner="rebase_report.source",
    )
    rebase_output_path, rebase_output = _validated_file_binding(
        rebase_report.get("output"),
        owner="rebase_report.output",
    )
    _same_file_identity(
        pre_rebase_record,
        rebase_source,
        owner="appearance lineage to rebase source binding",
    )
    actual_visual = _file_record(visual)
    _same_file_identity(
        rebase_output,
        actual_visual,
        owner="rebase output to actual visual binding",
    )
    # Keep these local variables deliberate: resolving every declared record
    # through ``_regular_file`` is part of the no-symlink evidence boundary.
    del pre_rebase_path, rebase_source_path, rebase_output_path
    return lineage


def _validate_appearance_lineage(
    *,
    spec: VariantPackageSpec,
    evidence: VariantPackageEvidence,
    visual: Path,
    rebase_report: Mapping[str, Any],
) -> Mapping[str, Any]:
    lineage = _load_evidence_json(
        evidence.appearance_lineage,
        owner="appearance_lineage",
    )
    schema = lineage.get("schema")
    if schema == APPEARANCE_LINEAGE_SCHEMA:
        return _validate_l9_appearance_lineage(
            spec=spec,
            evidence=evidence,
            visual=visual,
            rebase_report=rebase_report,
            lineage=lineage,
        )
    if schema == CROSS_SPECIES_APPEARANCE_LINEAGE_SCHEMA:
        try:
            return validate_cross_species_appearance_lineage(
                lineage,
                expected_spec_path=spec.path,
                expected_upstream_source_manifest=evidence.source_manifest,
                expected_material_normalization_report=(
                    evidence.material_normalization_report
                ),
                expected_rebase_report=evidence.rebase_report,
                expected_final_visual=visual,
                expected_repository_root=_REPOSITORY_ROOT,
            )
        except CrossSpeciesLineageError as exc:
            raise VariantPackageError(
                f"cross-species appearance lineage is invalid: {exc}"
            ) from exc
    raise VariantPackageError(
        "appearance_lineage schema must be either "
        f"{APPEARANCE_LINEAGE_SCHEMA!r} or "
        f"{CROSS_SPECIES_APPEARANCE_LINEAGE_SCHEMA!r}"
    )


def _finite_material_number(value: Any, *, owner: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise VariantPackageError(f"{owner} must be a finite number")
    return float(value)


def _material_color(value: Any, *, length: int, owner: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise VariantPackageError(
            f"{owner} must contain exactly {length} finite factors"
        )
    result = [
        _finite_material_number(component, owner=f"{owner}[{index}]")
        for index, component in enumerate(value)
    ]
    if any(component < 0.0 or component > 1.0 for component in result):
        raise VariantPackageError(f"{owner} factors must be in [0, 1]")
    return result


def _validate_current_matte_visual(visual: Path) -> None:
    try:
        document = load_glb(visual).json
    except (OSError, GlbError) as exc:
        raise VariantPackageError(
            f"actual visual GLB failed strict material readback: {exc}"
        ) from exc
    asset = document.get("asset")
    if not isinstance(asset, Mapping) or asset.get("version") != "2.0":
        raise VariantPackageError("actual visual GLB asset schema must be glTF 2.0")
    materials = document.get("materials")
    if (
        not isinstance(materials, list)
        or not materials
        or any(not isinstance(material, Mapping) for material in materials)
    ):
        raise VariantPackageError(
            "actual visual GLB must contain a non-empty material object array"
        )
    extensions_used = document.get("extensionsUsed")
    if (
        not isinstance(extensions_used, list)
        or any(not isinstance(item, str) for item in extensions_used)
        or len(extensions_used) != len(set(extensions_used))
        or _SPECULAR_EXTENSION not in extensions_used
    ):
        raise VariantPackageError(
            "actual visual GLB must uniquely declare KHR_materials_specular"
        )

    for index, material in enumerate(materials):
        owner = f"actual visual materials[{index}]"
        pbr = material.get("pbrMetallicRoughness")
        if not isinstance(pbr, Mapping):
            raise VariantPackageError(f"{owner}.pbrMetallicRoughness is required")
        metallic = _finite_material_number(
            pbr.get("metallicFactor"),
            owner=f"{owner}.pbrMetallicRoughness.metallicFactor",
        )
        if metallic != 0.0 or "metallicRoughnessTexture" in pbr:
            raise VariantPackageError(f"{owner} has a metallic material bypass")
        roughness = _finite_material_number(
            pbr.get("roughnessFactor"),
            owner=f"{owner}.pbrMetallicRoughness.roughnessFactor",
        )
        if not MINIMUM_ROUGHNESS_FACTOR <= roughness <= 1.0:
            raise VariantPackageError(f"{owner} violates the matte roughness bound")
        base_color = _material_color(
            pbr.get("baseColorFactor"),
            length=4,
            owner=f"{owner}.pbrMetallicRoughness.baseColorFactor",
        )
        if material.get("alphaMode") != "OPAQUE" or base_color[3] != 1.0:
            raise VariantPackageError(f"{owner} has an alpha material bypass")
        emissive = _material_color(
            material.get("emissiveFactor"),
            length=3,
            owner=f"{owner}.emissiveFactor",
        )
        if emissive != [float(value) for value in ZERO_EMISSIVE_FACTOR] or (
            "emissiveTexture" in material
        ):
            raise VariantPackageError(f"{owner} has an emissive material bypass")

        extensions = material.get("extensions")
        if not isinstance(extensions, Mapping) or set(extensions) != {
            _SPECULAR_EXTENSION
        }:
            raise VariantPackageError(
                f"{owner} has an unsupported specular/material extension bypass"
            )
        specular = extensions.get(_SPECULAR_EXTENSION)
        allowed_specular_keys = {
            "specularFactor",
            "specularTexture",
            "specularColorFactor",
            "specularColorTexture",
        }
        if not isinstance(specular, Mapping) or set(specular) - allowed_specular_keys:
            raise VariantPackageError(f"{owner} specular controls are invalid")
        specular_factor = _finite_material_number(
            specular.get("specularFactor"),
            owner=f"{owner}.extensions.{_SPECULAR_EXTENSION}.specularFactor",
        )
        if not 0.0 <= specular_factor <= MAXIMUM_SPECULAR_FACTOR:
            raise VariantPackageError(f"{owner} has a specular material bypass")
        specular_color = _material_color(
            specular.get("specularColorFactor"),
            length=3,
            owner=(f"{owner}.extensions.{_SPECULAR_EXTENSION}.specularColorFactor"),
        )
        if any(
            component > MAXIMUM_SPECULAR_COLOR_FACTOR for component in specular_color
        ):
            raise VariantPackageError(f"{owner} has a specular color bypass")

    meshes = document.get("meshes")
    if not isinstance(meshes, list) or not meshes:
        raise VariantPackageError("actual visual GLB must contain rendered meshes")
    for mesh_index, mesh in enumerate(meshes):
        primitives = mesh.get("primitives") if isinstance(mesh, Mapping) else None
        if not isinstance(primitives, list) or not primitives:
            raise VariantPackageError(
                f"actual visual meshes[{mesh_index}] has no primitives"
            )
        for primitive_index, primitive in enumerate(primitives):
            material_index = (
                primitive.get("material") if isinstance(primitive, Mapping) else None
            )
            if (
                isinstance(material_index, bool)
                or not isinstance(material_index, int)
                or not 0 <= material_index < len(materials)
            ):
                raise VariantPackageError(
                    "actual visual mesh primitive lacks an explicit bounded material: "
                    f"meshes[{mesh_index}].primitives[{primitive_index}]"
                )


def _validate_material_evidence(
    *,
    evidence: VariantPackageEvidence,
    visual: Path,
) -> Mapping[str, Any]:
    report_path = _regular_file(
        evidence.material_normalization_report,
        owner="material_normalization_report",
    )
    try:
        report = load_and_validate_material_normalization_report(
            report_path,
            verify_files=True,
        )
    except (OSError, ValueError) as exc:
        raise VariantPackageError(
            f"material_normalization_report failed strict validation: {exc}"
        ) from exc
    policy = report.get("policy")
    if not isinstance(policy, Mapping) or policy.get("force_opaque") is not True:
        raise VariantPackageError(
            "material_normalization_report must enforce force_opaque=true"
        )
    _, report_output = _validated_file_binding(
        report.get("output"),
        owner="material_normalization_report.output",
    )
    _same_file_identity(
        report_output,
        _file_record(visual),
        owner="material normalization output to actual visual binding",
    )
    _validate_current_matte_visual(visual)
    return report


def _validate_real_evidence(
    spec: VariantPackageSpec,
    evidence: VariantPackageEvidence,
) -> None:
    for field_name in (
        "visual_glb",
        "baked_actions",
        "contacts",
        "source_manifest",
        "license_snapshot",
        "appearance_lineage",
        "material_normalization_report",
    ):
        _regular_file(getattr(evidence, field_name), owner=field_name)
    evidence_values: dict[str, Mapping[str, Any]] = {}
    for field_name, schema in _REQUIRED_EVIDENCE_SCHEMAS.items():
        value = _load_evidence_json(getattr(evidence, field_name), owner=field_name)
        evidence_values[field_name] = value
        if value.get("schema") != schema:
            raise VariantPackageError(
                f"{field_name} schema must be {schema!r}; placeholder QA is not accepted"
            )
        if value.get("status") != "pass":
            raise VariantPackageError(
                f"{field_name} must contain a real passing report"
            )
        if value.get("qualification_claim") is not False:
            raise VariantPackageError(
                f"{field_name} must remain non-qualifying research evidence"
            )
    source = _load_evidence_json(evidence.source_manifest, owner="source_manifest")
    if source.get("formal_dataset_registration_authorized") is not False:
        raise VariantPackageError(
            "source_manifest.formal_dataset_registration_authorized must be "
            "exactly false"
        )
    _load_evidence_json(evidence.license_snapshot, owner="license_snapshot")
    visual = _regular_file(evidence.visual_glb, owner="visual_glb")
    _validate_appearance_lineage(
        spec=spec,
        evidence=evidence,
        visual=visual,
        rebase_report=evidence_values["rebase_report"],
    )
    _validate_material_evidence(evidence=evidence, visual=visual)


def _input_binding(path: Path, *, snapshot: bool) -> dict[str, Any]:
    resolved = _regular_file(path, owner="source binding input")
    value: dict[str, Any] = {
        "path": str(resolved),
        "byte_size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if snapshot:
        value["snapshot"] = load_json(resolved)
    return value


def _bound_source_manifest(
    *,
    spec: VariantPackageSpec,
    evidence: VariantPackageEvidence,
) -> dict[str, Any]:
    """Bind registered quadruped taxonomy/appearance into package provenance."""

    source_manifest = _regular_file(
        evidence.source_manifest,
        owner="source_manifest",
    )
    visual = _regular_file(evidence.visual_glb, owner="visual_glb")
    source = load_json(source_manifest)
    return {
        "schema": VARIANT_SOURCE_BINDING_SCHEMA,
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "formal_dataset_registration_authorized": False,
        "variant_package_spec": _input_binding(spec.path, snapshot=True),
        "actual_visual_glb": _input_binding(visual, snapshot=False),
        "appearance_lineage": _input_binding(
            evidence.appearance_lineage,
            snapshot=True,
        ),
        "material_normalization_report": _input_binding(
            evidence.material_normalization_report,
            snapshot=True,
        ),
        "upstream_source_manifest": _input_binding(source_manifest, snapshot=True),
        "upstream_formal_dataset_registration_authorized": source.get(
            "formal_dataset_registration_authorized"
        ),
        "decision_reason": (
            "This research-only wrapper hash-binds explicit taxonomy, appearance, "
            "identity and semantic anchors to separately supplied appearance "
            "lineage, final-visual material normalization, actual GLB bytes and "
            "upstream source provenance. An upstream JSON snapshot cannot replace "
            "either required evidence file."
        ),
    }


def assemble_variant_package(
    *,
    spec: VariantPackageSpec,
    evidence: VariantPackageEvidence,
    output_directory: str | Path,
) -> Path:
    """Assemble one quadruped package from real compiler-verified evidence."""

    if not isinstance(spec, VariantPackageSpec):
        raise VariantPackageError("spec must come from load_variant_package_spec")
    if not isinstance(evidence, VariantPackageEvidence):
        raise VariantPackageError("evidence must be VariantPackageEvidence")
    raw_output = Path(output_directory)
    output = Path(os.path.abspath(raw_output))
    if raw_output.exists() or raw_output.is_symlink():
        raise VariantPackageError(f"refusing to replace package output: {output}")
    _validate_real_evidence(spec, evidence)
    bound_source = _bound_source_manifest(spec=spec, evidence=evidence)
    try:
        with tempfile.TemporaryDirectory(prefix="avengine-variant-package-") as temp:
            bound_source_path = Path(temp) / "source_manifest.json"
            bound_source_path.write_text(
                json.dumps(
                    bound_source,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = compile_research_candidate_animal_package(
                output_directory=output,
                identity=spec.identity,
                visual_glb=evidence.visual_glb,
                rebase_report=evidence.rebase_report,
                rebase_deformation_report=evidence.rebase_deformation_report,
                action_report=evidence.action_report,
                static_qa=evidence.static_qa,
                deformation_qa=evidence.deformation_qa,
                animation_qa=evidence.animation_qa,
                habitat_static_probe=evidence.habitat_static_probe,
                habitat_animation_review=evidence.habitat_animation_review,
                baked_actions=evidence.baked_actions,
                contacts=evidence.contacts,
                anchor_definitions=spec.anchors,
                source_manifest=bound_source_path,
                license_snapshot=evidence.license_snapshot,
                shader_type=spec.shader_type,
            )
    except PackageCompileError as exc:
        raise VariantPackageError(f"strict package compilation failed: {exc}") from exc
    return manifest


__all__ = [
    "APPEARANCE_LINEAGE_SCHEMA",
    "CROSS_SPECIES_APPEARANCE_LINEAGE_SCHEMA",
    "VARIANT_PACKAGE_SPEC_SCHEMA",
    "VARIANT_SOURCE_BINDING_SCHEMA",
    "VariantPackageError",
    "VariantPackageEvidence",
    "VariantPackageSpec",
    "assemble_variant_package",
    "load_variant_package_spec",
]
