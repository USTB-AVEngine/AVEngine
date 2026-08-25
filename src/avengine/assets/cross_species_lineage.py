"""Hash-closed appearance lineage for one cross-species diagnostic variant.

This contract deliberately does not model an appearance L9 experiment.  It
authenticates one already selected research diagnostic and the concrete
material/rebase path used to produce its final visual.  In particular it may
not be used for the Beagle L9 assets and it never authorizes qualification or
formal dataset registration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from avengine.contracts.json_io import canonical_json_sha256, load_json, sha256_file
from avengine.assets.materials import (
    MaterialNormalizationError,
    load_and_validate_material_normalization_report,
)


CROSS_SPECIES_APPEARANCE_LINEAGE_SCHEMA = (
    "avengine_m2_cross_species_appearance_lineage_v1"
)
VARIANT_PACKAGE_SPEC_SCHEMA = "avengine_m2_variant_package_spec_v1"
_UPSTREAM_SCHEMA_PREFIX = "avengine_m2_cross_species_research_lineage_v"
_REALIZATION_SCHEMA = "avengine_m2_force_matte_materials_v2"
_NORMALIZATION_SCHEMA = "avengine_m2_material_normalization_v2"
_REBASE_SCHEMA = "avengine_m2_skin_root_rebase_v1"
_BEAGLE_TAXONOMY = {
    "species_id": "canis_lupus_familiaris",
    "breed_id": "beagle",
}
_LIMITATIONS = [
    "This is one selected research diagnostic, not an L9 or OFAT experiment.",
    "Material lineage does not qualify mesh deformation, motion, contacts, anchors, rights, or source-to-target fidelity.",
    "The lineage cannot authorize formal dataset registration.",
]
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_TOOL_PATHS = {
    "lineage_producer": "tools/assets/build_cross_species_appearance_lineage.py",
    "lineage_contract": "src/avengine/assets/cross_species_lineage.py",
    "material_realization": "tools/assets/force_matte_materials.py",
    "material_normalization": "tools/assets/normalize_materials.py",
    "material_algorithm": "src/avengine/assets/materials.py",
    "skin_root_rebase": "tools/assets/rebase_skin_root.py",
    "rebase_algorithm": "src/avengine/assets/rebase.py",
}


class CrossSpeciesLineageError(ValueError):
    """A generic diagnostic lineage is incomplete, inconsistent, or unsafe."""


def _regular_file(path: str | Path, *, owner: str) -> Path:
    raw = Path(path)
    absolute = Path(os.path.abspath(raw))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise CrossSpeciesLineageError(f"{owner} must not contain a symbolic link")
    if not absolute.is_file():
        raise CrossSpeciesLineageError(f"{owner} is not a regular file: {absolute}")
    return absolute


def _mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossSpeciesLineageError(f"{owner} must be an object")
    return value


def _file_record(path: str | Path, *, owner: str) -> dict[str, Any]:
    resolved = _regular_file(path, owner=owner)
    return {
        "path": str(resolved),
        "byte_size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _json_binding(path: str | Path, *, owner: str) -> dict[str, Any]:
    resolved = _regular_file(path, owner=owner)
    try:
        value = load_json(resolved)
    except (OSError, ValueError) as exc:
        raise CrossSpeciesLineageError(f"{owner} is invalid JSON: {exc}") from exc
    return {
        **_file_record(resolved, owner=owner),
        "canonical_content_sha256": canonical_json_sha256(value),
        "snapshot": value,
    }


def _validated_file_binding(
    value: Any,
    *,
    owner: str,
) -> tuple[Path, dict[str, Any]]:
    binding = _mapping(value, owner=owner)
    if set(binding) != {"path", "byte_size", "sha256"}:
        raise CrossSpeciesLineageError(
            f"{owner} must contain exactly path, byte_size, and sha256"
        )
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise CrossSpeciesLineageError(f"{owner}.path must be non-empty")
    actual = _file_record(raw_path, owner=owner)
    if binding.get("byte_size") != actual["byte_size"]:
        raise CrossSpeciesLineageError(f"{owner}.byte_size does not bind its file")
    if binding.get("sha256") != actual["sha256"]:
        raise CrossSpeciesLineageError(f"{owner}.sha256 does not bind its file")
    return Path(actual["path"]), actual


def _validated_json_binding(
    value: Any,
    *,
    owner: str,
) -> tuple[Path, Mapping[str, Any], Mapping[str, Any]]:
    binding = _mapping(value, owner=owner)
    if set(binding) != {
        "path",
        "byte_size",
        "sha256",
        "canonical_content_sha256",
        "snapshot",
    }:
        raise CrossSpeciesLineageError(f"{owner} JSON binding fields are invalid")
    path, _ = _validated_file_binding(
        {
            "path": binding.get("path"),
            "byte_size": binding.get("byte_size"),
            "sha256": binding.get("sha256"),
        },
        owner=owner,
    )
    try:
        current = load_json(path)
    except (OSError, ValueError) as exc:
        raise CrossSpeciesLineageError(f"{owner} is invalid JSON: {exc}") from exc
    if binding.get("snapshot") != current:
        raise CrossSpeciesLineageError(f"{owner}.snapshot differs from its actual file")
    digest = canonical_json_sha256(current)
    if binding.get("canonical_content_sha256") != digest:
        raise CrossSpeciesLineageError(
            f"{owner}.canonical_content_sha256 does not bind its actual file"
        )
    return path, current, binding


def _same_file_identity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    owner: str,
) -> None:
    if left.get("byte_size") != right.get("byte_size") or left.get(
        "sha256"
    ) != right.get("sha256"):
        raise CrossSpeciesLineageError(
            f"{owner} does not bind identical bytes and SHA-256"
        )


def _require_json_input_identity(
    binding: Mapping[str, Any],
    expected_path: str | Path,
    *,
    owner: str,
) -> None:
    expected = _json_binding(expected_path, owner=f"expected {owner}")
    for field in (
        "byte_size",
        "sha256",
        "canonical_content_sha256",
        "snapshot",
    ):
        if binding.get(field) != expected[field]:
            raise CrossSpeciesLineageError(
                f"{owner} differs from the separately supplied file identity"
            )


def _require_file_input_identity(
    binding: Mapping[str, Any],
    expected_path: str | Path,
    *,
    owner: str,
) -> None:
    expected = _file_record(expected_path, owner=f"expected {owner}")
    _same_file_identity(binding, expected, owner=owner)


def validate_repository_file_identity(
    value: Any,
    *,
    repository_root: str | Path,
    repository_relative_path: str,
    owner: str,
    require_byte_size: bool,
) -> Path:
    """Bind a declared tool record to one canonical repository file."""

    record = _mapping(value, owner=owner)
    required = {"path", "sha256"}
    if require_byte_size:
        required.add("byte_size")
    if not required.issubset(record):
        raise CrossSpeciesLineageError(
            f"{owner} is missing canonical repository file identity fields"
        )
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise CrossSpeciesLineageError(f"{owner}.path must be non-empty")
    declared = _regular_file(raw_path, owner=owner)
    canonical = _regular_file(
        Path(repository_root) / repository_relative_path,
        owner=f"canonical {owner}",
    )
    expected = _file_record(canonical, owner=f"canonical {owner}")
    if (
        declared != canonical
        or record.get("sha256") != expected["sha256"]
        or (require_byte_size and record.get("byte_size") != expected["byte_size"])
    ):
        raise CrossSpeciesLineageError(
            f"{owner} does not bind canonical repository file "
            f"{repository_relative_path}"
        )
    return canonical


def _validate_spec_for_generic(
    spec: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if spec.get("schema") != VARIANT_PACKAGE_SPEC_SCHEMA:
        raise CrossSpeciesLineageError("variant spec schema is invalid")
    taxonomy = _mapping(spec.get("taxonomy"), owner="variant spec taxonomy")
    appearance = _mapping(spec.get("appearance"), owner="variant spec appearance")
    if set(taxonomy) != {"species_id", "breed_id"} or any(
        not isinstance(value, str) or not value for value in taxonomy.values()
    ):
        raise CrossSpeciesLineageError("variant spec taxonomy fields are invalid")
    if set(appearance) != {"size", "body_build", "coat", "life_stage"} or any(
        not isinstance(value, str) or not value for value in appearance.values()
    ):
        raise CrossSpeciesLineageError("variant spec appearance fields are invalid")
    identity = _mapping(spec.get("identity"), owner="variant spec identity")
    asset_id = identity.get("asset_id")
    if not isinstance(asset_id, str) or not asset_id:
        raise CrossSpeciesLineageError("variant spec identity.asset_id is invalid")
    if dict(taxonomy) == _BEAGLE_TAXONOMY or "_appearance_l9_" in asset_id:
        raise CrossSpeciesLineageError(
            "generic diagnostic lineage cannot be used to bypass the Beagle L9 lineage"
        )
    return taxonomy, appearance


def _validate_upstream(
    upstream: Mapping[str, Any],
    *,
    taxonomy: Mapping[str, Any],
    appearance: Mapping[str, Any],
) -> None:
    schema = upstream.get("schema")
    if (
        not isinstance(schema, str)
        or not schema.startswith(_UPSTREAM_SCHEMA_PREFIX)
        or not schema[len(_UPSTREAM_SCHEMA_PREFIX) :].isdigit()
    ):
        raise CrossSpeciesLineageError(
            "upstream source manifest is not a cross-species research lineage"
        )
    if (
        upstream.get("qualification_state") != "research_candidate"
        or upstream.get("qualification_claim") is not False
        or upstream.get("formal_dataset_registration_authorized") is not False
    ):
        raise CrossSpeciesLineageError(
            "upstream source manifest must remain non-qualifying research evidence"
        )
    if upstream.get("taxonomy") != taxonomy:
        raise CrossSpeciesLineageError(
            "upstream source manifest taxonomy differs from the variant spec"
        )
    if upstream.get("appearance") != appearance:
        raise CrossSpeciesLineageError(
            "upstream source manifest appearance differs from the variant spec"
        )


def _validate_upstream_material_terminal(
    upstream: Mapping[str, Any],
    *,
    realization_output: Mapping[str, Any],
) -> None:
    records = upstream.get("reused_hash_closed_inputs")
    if not isinstance(records, list) or not records:
        raise CrossSpeciesLineageError(
            "upstream source manifest lacks reused hash-closed inputs"
        )
    terminals: list[Mapping[str, Any]] = []
    for index, raw_record in enumerate(records):
        record = _mapping(
            raw_record,
            owner=f"upstream reused_hash_closed_inputs[{index}]",
        )
        path = record.get("path")
        if not isinstance(path, str) or not path:
            raise CrossSpeciesLineageError(
                "upstream reused hash-closed input path is invalid"
            )
        byte_size = record.get("byte_size")
        digest = record.get("sha256")
        if (
            isinstance(byte_size, bool)
            or not isinstance(byte_size, int)
            or byte_size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise CrossSpeciesLineageError(
                "upstream reused hash-closed input identity is invalid"
            )
        if Path(path).name == "projected_strict.glb":
            terminals.append(record)
    if len(terminals) != 1:
        raise CrossSpeciesLineageError(
            "upstream source manifest must identify exactly one projected_strict.glb terminal"
        )
    _same_file_identity(
        terminals[0],
        realization_output,
        owner="upstream projected visual to material realization output",
    )


def _validate_material_realization(
    report: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    required = {
        "schema",
        "status",
        "qualification_state",
        "qualification_claim",
        "material_policy_complete",
        "source",
        "output",
        "changes",
        "policy",
        "invariants",
        "notes",
        "report_content_sha256",
    }
    if set(report) != required:
        raise CrossSpeciesLineageError("material realization report fields are invalid")
    if (
        report.get("schema") != _REALIZATION_SCHEMA
        or report.get("status") != "pass"
        or report.get("qualification_state") != "research_candidate"
        or report.get("qualification_claim") is not False
        or report.get("material_policy_complete") is not True
    ):
        raise CrossSpeciesLineageError(
            "material realization report must be passing non-qualifying evidence"
        )
    expected_digest = canonical_json_sha256(
        {key: item for key, item in report.items() if key != "report_content_sha256"}
    )
    if report.get("report_content_sha256") != expected_digest:
        raise CrossSpeciesLineageError(
            "material realization report content digest is invalid"
        )
    policy = _mapping(report.get("policy"), owner="material realization policy")
    if (
        policy.get("alpha_mode") != "OPAQUE"
        or policy.get("base_color_alpha") != 1.0
        or policy.get("metallic_factor") != 0.0
        or policy.get("roughness_factor") != 1.0
        or policy.get("emissive_factor") != [0.0, 0.0, 0.0]
        or policy.get("specular_factor") != 0.0
    ):
        raise CrossSpeciesLineageError(
            "material realization report policy is not complete opaque matte"
        )
    invariants = _mapping(
        report.get("invariants"), owner="material realization invariants"
    )
    if (
        invariants.get("binary_unchanged") is not True
        or invariants.get("non_material_json_unchanged") is not True
    ):
        raise CrossSpeciesLineageError(
            "material realization invariants are not passing"
        )
    _, source = _validated_file_binding(
        report.get("source"), owner="material realization report.source"
    )
    _, output = _validated_file_binding(
        report.get("output"), owner="material realization report.output"
    )
    return source, output


def _validate_rebase(
    report: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if (
        report.get("schema") != _REBASE_SCHEMA
        or report.get("status") != "pass"
        or report.get("qualification_claim") is not False
    ):
        raise CrossSpeciesLineageError(
            "rebase report must be passing non-qualifying evidence"
        )
    _, source = _validated_file_binding(
        report.get("source"), owner="rebase report.source"
    )
    _, output = _validated_file_binding(
        report.get("output"), owner="rebase report.output"
    )
    return source, output


def validate_cross_species_appearance_lineage(
    value: Any,
    *,
    expected_spec_path: str | Path | None = None,
    expected_upstream_source_manifest: str | Path | None = None,
    expected_material_normalization_report: str | Path | None = None,
    expected_rebase_report: str | Path | None = None,
    expected_final_visual: str | Path | None = None,
    expected_repository_root: str | Path | None = None,
) -> Mapping[str, Any]:
    """Validate a diagnostic lineage and, optionally, external assembler inputs."""

    lineage = _mapping(value, owner="cross-species appearance lineage")
    required_root = {
        "schema",
        "status",
        "design_kind",
        "ofat_status",
        "qualification_state",
        "qualification_claim",
        "formal_dataset_registration_authorized",
        "taxonomy",
        "appearance",
        "inputs",
        "artifacts",
        "tool_identity",
        "limitations",
        "decision_reason",
        "lineage_content_sha256",
    }
    if set(lineage) != required_root:
        raise CrossSpeciesLineageError(
            "cross-species appearance lineage root fields are invalid"
        )
    if (
        lineage.get("schema") != CROSS_SPECIES_APPEARANCE_LINEAGE_SCHEMA
        or lineage.get("status") != "pass"
        or lineage.get("design_kind") != "single_research_diagnostic"
        or lineage.get("ofat_status") != "not_run"
        or lineage.get("qualification_state") != "research_candidate"
        or lineage.get("qualification_claim") is not False
        or lineage.get("formal_dataset_registration_authorized") is not False
    ):
        raise CrossSpeciesLineageError(
            "cross-species appearance lineage must be diagnostic-only and non-qualifying"
        )
    if lineage.get("limitations") != _LIMITATIONS:
        raise CrossSpeciesLineageError(
            "cross-species appearance lineage limitations are invalid"
        )
    reason = lineage.get("decision_reason")
    if not isinstance(reason, str) or not reason.strip():
        raise CrossSpeciesLineageError(
            "cross-species appearance lineage decision_reason is invalid"
        )
    expected_digest = canonical_json_sha256(
        {key: item for key, item in lineage.items() if key != "lineage_content_sha256"}
    )
    if lineage.get("lineage_content_sha256") != expected_digest:
        raise CrossSpeciesLineageError(
            "cross-species appearance lineage content digest is invalid"
        )

    inputs = _mapping(lineage.get("inputs"), owner="lineage.inputs")
    required_inputs = {
        "variant_spec",
        "upstream_source_manifest",
        "material_realization_report",
        "material_normalization_report",
        "rebase_report",
    }
    if set(inputs) != required_inputs:
        raise CrossSpeciesLineageError(
            "cross-species appearance lineage input fields are invalid"
        )
    input_values: dict[str, Mapping[str, Any]] = {}
    input_bindings: dict[str, Mapping[str, Any]] = {}
    for name in sorted(required_inputs):
        _, input_values[name], input_bindings[name] = _validated_json_binding(
            inputs[name], owner=f"lineage.inputs.{name}"
        )

    taxonomy, appearance = _validate_spec_for_generic(input_values["variant_spec"])
    if lineage.get("taxonomy") != taxonomy:
        raise CrossSpeciesLineageError(
            "lineage taxonomy differs from the exact variant spec"
        )
    if lineage.get("appearance") != appearance:
        raise CrossSpeciesLineageError(
            "lineage appearance differs from the exact variant spec"
        )
    _validate_upstream(
        input_values["upstream_source_manifest"],
        taxonomy=taxonomy,
        appearance=appearance,
    )
    realization_source, realization_output = _validate_material_realization(
        input_values["material_realization_report"]
    )
    _validate_upstream_material_terminal(
        input_values["upstream_source_manifest"],
        realization_output=realization_output,
    )
    try:
        normalization = load_and_validate_material_normalization_report(
            str(input_bindings["material_normalization_report"]["path"]),
            verify_files=True,
        )
    except (OSError, ValueError, MaterialNormalizationError) as exc:
        raise CrossSpeciesLineageError(
            f"material normalization report failed strict validation: {exc}"
        ) from exc
    if normalization.get("schema") != _NORMALIZATION_SCHEMA:
        raise CrossSpeciesLineageError("material normalization schema is invalid")
    policy = _mapping(
        normalization.get("policy"), owner="material normalization policy"
    )
    if policy.get("force_opaque") is not True:
        raise CrossSpeciesLineageError(
            "material normalization must enforce force_opaque=true"
        )
    _, normalization_source = _validated_file_binding(
        normalization.get("source"), owner="material normalization report.source"
    )
    _, normalization_output = _validated_file_binding(
        normalization.get("output"), owner="material normalization report.output"
    )
    rebase_source, rebase_output = _validate_rebase(input_values["rebase_report"])

    artifacts = _mapping(lineage.get("artifacts"), owner="lineage.artifacts")
    artifact_names = {
        "material_realization_before_glb",
        "material_realization_after_glb",
        "pre_rebase_visual_glb",
        "material_normalization_before_glb",
        "material_normalization_after_glb",
        "final_visual_glb",
    }
    if set(artifacts) != artifact_names:
        raise CrossSpeciesLineageError(
            "cross-species appearance lineage artifact fields are invalid"
        )
    artifact_records: dict[str, Mapping[str, Any]] = {}
    for name in sorted(artifact_names):
        _, artifact_records[name] = _validated_file_binding(
            artifacts[name], owner=f"lineage.artifacts.{name}"
        )
    for left, right, owner in (
        (
            artifact_records["material_realization_before_glb"],
            realization_source,
            "material realization before artifact",
        ),
        (
            artifact_records["material_realization_after_glb"],
            realization_output,
            "material realization after artifact",
        ),
        (
            artifact_records["pre_rebase_visual_glb"],
            realization_output,
            "material realization to pre-rebase visual",
        ),
        (
            artifact_records["pre_rebase_visual_glb"],
            rebase_source,
            "pre-rebase visual to rebase source",
        ),
        (
            rebase_output,
            normalization_source,
            "rebase output to material normalization source",
        ),
        (
            artifact_records["material_normalization_before_glb"],
            normalization_source,
            "material normalization before artifact",
        ),
        (
            artifact_records["material_normalization_after_glb"],
            normalization_output,
            "material normalization after artifact",
        ),
        (
            artifact_records["final_visual_glb"],
            normalization_output,
            "final visual artifact",
        ),
    ):
        _same_file_identity(left, right, owner=owner)

    tools = _mapping(lineage.get("tool_identity"), owner="lineage.tool_identity")
    if set(tools) != set(_TOOL_PATHS):
        raise CrossSpeciesLineageError(
            "cross-species appearance lineage tool identity fields are invalid"
        )
    repository_root = (
        _REPOSITORY_ROOT
        if expected_repository_root is None
        else Path(expected_repository_root)
    )
    for role, relative_path in _TOOL_PATHS.items():
        _validated_file_binding(tools[role], owner=f"lineage.tool_identity.{role}")
        validate_repository_file_identity(
            tools[role],
            repository_root=repository_root,
            repository_relative_path=relative_path,
            owner=f"lineage.tool_identity.{role}",
            require_byte_size=True,
        )

    if expected_spec_path is not None:
        _require_json_input_identity(
            input_bindings["variant_spec"],
            expected_spec_path,
            owner="lineage variant spec",
        )
    if expected_upstream_source_manifest is not None:
        _require_json_input_identity(
            input_bindings["upstream_source_manifest"],
            expected_upstream_source_manifest,
            owner="lineage upstream source manifest",
        )
    if expected_material_normalization_report is not None:
        _require_json_input_identity(
            input_bindings["material_normalization_report"],
            expected_material_normalization_report,
            owner="lineage material normalization report",
        )
    if expected_rebase_report is not None:
        _require_json_input_identity(
            input_bindings["rebase_report"],
            expected_rebase_report,
            owner="lineage rebase report",
        )
    if expected_final_visual is not None:
        _require_file_input_identity(
            artifact_records["final_visual_glb"],
            expected_final_visual,
            owner="lineage final visual",
        )
    return lineage


def build_cross_species_appearance_lineage(
    *,
    variant_spec: str | Path,
    upstream_source_manifest: str | Path,
    material_realization_report: str | Path,
    material_normalization_report: str | Path,
    rebase_report: str | Path,
    pre_rebase_visual_glb: str | Path,
    lineage_producer: str | Path,
    lineage_contract: str | Path,
    material_realization_tool: str | Path,
    material_normalization_tool: str | Path,
    material_algorithm: str | Path,
    skin_root_rebase_tool: str | Path,
    rebase_algorithm: str | Path,
) -> dict[str, Any]:
    """Build and read back one exact diagnostic lineage in memory."""

    spec_binding = _json_binding(variant_spec, owner="variant_spec")
    spec = spec_binding["snapshot"]
    assert isinstance(spec, Mapping)
    taxonomy, appearance = _validate_spec_for_generic(spec)
    realization_binding = _json_binding(
        material_realization_report, owner="material_realization_report"
    )
    normalization_binding = _json_binding(
        material_normalization_report, owner="material_normalization_report"
    )
    rebase_binding = _json_binding(rebase_report, owner="rebase_report")
    realization = realization_binding["snapshot"]
    normalization = normalization_binding["snapshot"]
    assert isinstance(realization, Mapping)
    assert isinstance(normalization, Mapping)
    pre_rebase = _file_record(pre_rebase_visual_glb, owner="pre_rebase_visual_glb")
    realization_source = _mapping(
        realization.get("source"), owner="material realization source"
    )
    realization_output = _mapping(
        realization.get("output"), owner="material realization output"
    )
    normalization_source = _mapping(
        normalization.get("source"), owner="material normalization source"
    )
    normalization_output = _mapping(
        normalization.get("output"), owner="material normalization output"
    )
    core: dict[str, Any] = {
        "schema": CROSS_SPECIES_APPEARANCE_LINEAGE_SCHEMA,
        "status": "pass",
        "design_kind": "single_research_diagnostic",
        "ofat_status": "not_run",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "formal_dataset_registration_authorized": False,
        "taxonomy": dict(taxonomy),
        "appearance": dict(appearance),
        "inputs": {
            "variant_spec": spec_binding,
            "upstream_source_manifest": _json_binding(
                upstream_source_manifest, owner="upstream_source_manifest"
            ),
            "material_realization_report": realization_binding,
            "material_normalization_report": normalization_binding,
            "rebase_report": rebase_binding,
        },
        "artifacts": {
            "material_realization_before_glb": dict(realization_source),
            "material_realization_after_glb": dict(realization_output),
            "pre_rebase_visual_glb": pre_rebase,
            "material_normalization_before_glb": dict(normalization_source),
            "material_normalization_after_glb": dict(normalization_output),
            "final_visual_glb": dict(normalization_output),
        },
        "tool_identity": {
            "lineage_producer": _file_record(
                lineage_producer, owner="lineage_producer"
            ),
            "lineage_contract": _file_record(
                lineage_contract, owner="lineage_contract"
            ),
            "material_realization": _file_record(
                material_realization_tool, owner="material_realization_tool"
            ),
            "material_normalization": _file_record(
                material_normalization_tool, owner="material_normalization_tool"
            ),
            "material_algorithm": _file_record(
                material_algorithm, owner="material_algorithm"
            ),
            "skin_root_rebase": _file_record(
                skin_root_rebase_tool, owner="skin_root_rebase_tool"
            ),
            "rebase_algorithm": _file_record(
                rebase_algorithm, owner="rebase_algorithm"
            ),
        },
        "limitations": list(_LIMITATIONS),
        "decision_reason": (
            "This hash-closed record authenticates one cross-species research "
            "diagnostic from its exact taxonomy/appearance spec through material "
            "realization, skin-root rebase and final opaque material normalization."
        ),
    }
    core["lineage_content_sha256"] = canonical_json_sha256(core)
    validate_cross_species_appearance_lineage(
        core,
        expected_spec_path=variant_spec,
        expected_upstream_source_manifest=upstream_source_manifest,
        expected_material_normalization_report=material_normalization_report,
        expected_rebase_report=rebase_report,
        expected_final_visual=normalization_output["path"],
    )
    return core


__all__ = [
    "CROSS_SPECIES_APPEARANCE_LINEAGE_SCHEMA",
    "CrossSpeciesLineageError",
    "build_cross_species_appearance_lineage",
    "validate_repository_file_identity",
    "validate_cross_species_appearance_lineage",
]
