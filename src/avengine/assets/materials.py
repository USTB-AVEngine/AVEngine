"""Bounded, hash-bound PBR material normalization for GLB 2.0 assets.

This compiler changes entries in the glTF ``materials`` array and, when
needed, adds the corresponding root ``extensionsUsed`` declaration. Embedded
binary data and every other JSON section are retained byte-for-byte or
value-for-value and verified after serialization. The result remains a
research candidate: material normalization is not asset qualification.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Mapping

from avengine.contracts.json_io import canonical_json_sha256, load_json, sha256_file
from avengine.assets.glb import GlbError, load_glb, parse_glb
from avengine.assets.glb_write import build_glb


MATERIAL_NORMALIZATION_SCHEMA = "avengine_m2_material_normalization_v2"
MINIMUM_ROUGHNESS_FACTOR = 0.72
MAXIMUM_SPECULAR_FACTOR = 0.25
MAXIMUM_SPECULAR_COLOR_FACTOR = 1.0
ZERO_EMISSIVE_FACTOR = [0.0, 0.0, 0.0]
_SPECULAR_EXTENSION = "KHR_materials_specular"
_ALLOWED_MATERIAL_EXTENSIONS = frozenset({_SPECULAR_EXTENSION})
_REQUIRED_UNCHANGED_SECTIONS = (
    "buffers",
    "accessors",
    "meshes",
    "skins",
    "animations",
    "textures",
    "images",
    "samplers",
)


class MaterialNormalizationError(ValueError):
    """The input or requested write is outside the bounded material policy."""


def _objects(
    value: Any, owner: str, *, require_nonempty: bool = False
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise MaterialNormalizationError(f"{owner} must be an array of objects")
    if require_nonempty and not value:
        raise MaterialNormalizationError(f"{owner} must contain at least one object")
    return value


def _object(value: Any, owner: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MaterialNormalizationError(f"{owner} must be an object")
    return value


def _finite_number(value: Any, owner: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise MaterialNormalizationError(f"{owner} must be a finite number")
    return float(value)


def _unit_factor(value: Any, owner: str) -> float:
    result = _finite_number(value, owner)
    if not 0.0 <= result <= 1.0:
        raise MaterialNormalizationError(f"{owner} must be in [0, 1]")
    return result


def _color_factor(value: Any, owner: str, *, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise MaterialNormalizationError(
            f"{owner} must contain exactly {length} finite numbers"
        )
    return [
        _unit_factor(component, f"{owner}[{index}]")
        for index, component in enumerate(value)
    ]


def _clamped_specular_color(value: Any, owner: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise MaterialNormalizationError(
            f"{owner} must contain exactly three finite numbers"
        )
    return [
        min(
            MAXIMUM_SPECULAR_COLOR_FACTOR,
            max(0.0, _finite_number(component, f"{owner}[{index}]")),
        )
        for index, component in enumerate(value)
    ]


def _declared_change(
    path: str,
    *,
    declared_before: bool,
    before: Any,
    effective_before: Any,
    after: Any,
) -> dict[str, Any]:
    return {
        "path": path,
        "declared_before": declared_before,
        "before": copy.deepcopy(before),
        "effective_before": copy.deepcopy(effective_before),
        "after": copy.deepcopy(after),
        "changed": not declared_before or before != after,
    }


def _normalize_one_material(
    material: Mapping[str, Any],
    *,
    material_index: int,
    force_opaque: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = copy.deepcopy(dict(material))
    owner = f"materials[{material_index}]"
    changes: list[dict[str, Any]] = []
    if "name" in result and not isinstance(result["name"], str):
        raise MaterialNormalizationError(f"{owner}.name must be a string")

    if "pbrMetallicRoughness" not in result:
        pbr: dict[str, Any] = {}
        result["pbrMetallicRoughness"] = pbr
    else:
        pbr = _object(
            result["pbrMetallicRoughness"],
            f"{owner}.pbrMetallicRoughness",
        )

    # A metallic-roughness texture multiplies the scalar factors.  Keeping it
    # would therefore make ``roughnessFactor >= 0.72`` an ineffective lower
    # bound whenever the texture's green channel is dark.  This bounded
    # normalizer deliberately removes that texture reference; image/texture
    # objects and the BIN chunk remain untouched and auditable.
    if "metallicRoughnessTexture" in pbr:
        texture_before = copy.deepcopy(pbr["metallicRoughnessTexture"])
        if not isinstance(texture_before, dict) or not isinstance(
            texture_before.get("index"), int
        ):
            raise MaterialNormalizationError(
                f"{owner}.pbrMetallicRoughness.metallicRoughnessTexture "
                "must be a textureInfo object"
            )
        del pbr["metallicRoughnessTexture"]
        changes.append(
            _declared_change(
                "pbrMetallicRoughness.metallicRoughnessTexture",
                declared_before=True,
                before=texture_before,
                effective_before=texture_before,
                after=None,
            )
        )

    metallic_declared = "metallicFactor" in pbr
    metallic_before = pbr.get("metallicFactor")
    metallic_effective = (
        _finite_number(metallic_before, f"{owner}.pbrMetallicRoughness.metallicFactor")
        if metallic_declared
        else 1.0
    )
    pbr["metallicFactor"] = 0.0
    changes.append(
        _declared_change(
            "pbrMetallicRoughness.metallicFactor",
            declared_before=metallic_declared,
            before=metallic_before,
            effective_before=metallic_effective,
            after=0.0,
        )
    )

    roughness_declared = "roughnessFactor" in pbr
    roughness_before = pbr.get("roughnessFactor")
    roughness_effective = (
        _finite_number(
            roughness_before, f"{owner}.pbrMetallicRoughness.roughnessFactor"
        )
        if roughness_declared
        else 1.0
    )
    if roughness_effective > 1.0:
        raise MaterialNormalizationError(
            f"{owner}.pbrMetallicRoughness.roughnessFactor must be <= 1"
        )
    roughness_after = max(MINIMUM_ROUGHNESS_FACTOR, roughness_effective)
    pbr["roughnessFactor"] = roughness_after
    changes.append(
        _declared_change(
            "pbrMetallicRoughness.roughnessFactor",
            declared_before=roughness_declared,
            before=roughness_before,
            effective_before=roughness_effective,
            after=roughness_after,
        )
    )

    if "baseColorFactor" in pbr:
        base_color = _color_factor(
            pbr["baseColorFactor"],
            f"{owner}.pbrMetallicRoughness.baseColorFactor",
            length=4,
        )
    else:
        base_color = [1.0, 1.0, 1.0, 1.0]

    alpha_mode = result.get("alphaMode", "OPAQUE")
    if alpha_mode not in {"OPAQUE", "MASK", "BLEND"}:
        raise MaterialNormalizationError(
            f"{owner}.alphaMode must be OPAQUE, MASK, or BLEND"
        )
    if force_opaque:
        alpha_declared = "alphaMode" in result
        alpha_before = result.get("alphaMode")
        result["alphaMode"] = "OPAQUE"
        changes.append(
            _declared_change(
                "alphaMode",
                declared_before=alpha_declared,
                before=alpha_before,
                effective_before=alpha_mode,
                after="OPAQUE",
            )
        )
        color_declared = "baseColorFactor" in pbr
        color_before = copy.deepcopy(pbr.get("baseColorFactor"))
        opaque_color = [*base_color[:3], 1.0]
        pbr["baseColorFactor"] = opaque_color
        changes.append(
            _declared_change(
                "pbrMetallicRoughness.baseColorFactor",
                declared_before=color_declared,
                before=color_before,
                effective_before=base_color,
                after=opaque_color,
            )
        )

    # Emission bypasses the roughness/specular matte bounds.  A strict matte
    # material therefore has to zero the factor and remove the multiplier
    # texture instead of relying on a renderer-specific interpretation.
    if "emissiveTexture" in result:
        emissive_texture_before = copy.deepcopy(result["emissiveTexture"])
        if (
            not isinstance(emissive_texture_before, dict)
            or isinstance(emissive_texture_before.get("index"), bool)
            or not isinstance(emissive_texture_before.get("index"), int)
        ):
            raise MaterialNormalizationError(
                f"{owner}.emissiveTexture must be a textureInfo object"
            )
        del result["emissiveTexture"]
        changes.append(
            _declared_change(
                "emissiveTexture",
                declared_before=True,
                before=emissive_texture_before,
                effective_before=emissive_texture_before,
                after=None,
            )
        )

    emissive_declared = "emissiveFactor" in result
    emissive_before = copy.deepcopy(result.get("emissiveFactor"))
    emissive_effective = (
        _color_factor(emissive_before, f"{owner}.emissiveFactor", length=3)
        if emissive_declared
        else ZERO_EMISSIVE_FACTOR
    )
    result["emissiveFactor"] = list(ZERO_EMISSIVE_FACTOR)
    changes.append(
        _declared_change(
            "emissiveFactor",
            declared_before=emissive_declared,
            before=emissive_before,
            effective_before=emissive_effective,
            after=ZERO_EMISSIVE_FACTOR,
        )
    )

    extension_declared = "extensions" in result
    extension_map = (
        _object(result["extensions"], f"{owner}.extensions")
        if extension_declared
        else {}
    )
    unsupported = set(extension_map) - _ALLOWED_MATERIAL_EXTENSIONS
    if unsupported:
        raise MaterialNormalizationError(
            f"{owner}.extensions contains unsupported material extensions: "
            f"{sorted(unsupported)}"
        )

    # The effective glTF default is specularFactor=1.0 when this extension is
    # absent.  A report claiming a global 0.25 ceiling must therefore emit an
    # explicit bounded extension for every material, not only clamp materials
    # that happened to declare it already.  The outer compiler adds and then
    # independently verifies the matching root extensionsUsed declaration.
    specular_declared = _SPECULAR_EXTENSION in extension_map
    specular_map = (
        _object(
            extension_map[_SPECULAR_EXTENSION],
            f"{owner}.extensions.{_SPECULAR_EXTENSION}",
        )
        if specular_declared
        else {}
    )
    extension_map[_SPECULAR_EXTENSION] = specular_map
    result["extensions"] = extension_map

    factor_declared = "specularFactor" in specular_map
    factor_before = specular_map.get("specularFactor")
    factor_effective = (
        _finite_number(
            factor_before,
            f"{owner}.extensions.{_SPECULAR_EXTENSION}.specularFactor",
        )
        if factor_declared
        else 1.0
    )
    if factor_effective < 0.0:
        raise MaterialNormalizationError(
            f"{owner}.extensions.{_SPECULAR_EXTENSION}.specularFactor must be >= 0"
        )
    factor_after = min(MAXIMUM_SPECULAR_FACTOR, factor_effective)
    specular_map["specularFactor"] = factor_after
    changes.append(
        _declared_change(
            f"extensions.{_SPECULAR_EXTENSION}.specularFactor",
            declared_before=factor_declared,
            before=factor_before,
            effective_before=factor_effective,
            after=factor_after,
        )
    )

    color_declared = "specularColorFactor" in specular_map
    color_before = copy.deepcopy(specular_map.get("specularColorFactor"))
    color_effective = color_before if color_declared else [1.0, 1.0, 1.0]
    color_after = _clamped_specular_color(
        color_effective,
        f"{owner}.extensions.{_SPECULAR_EXTENSION}.specularColorFactor",
    )
    specular_map["specularColorFactor"] = color_after
    changes.append(
        _declared_change(
            f"extensions.{_SPECULAR_EXTENSION}.specularColorFactor",
            declared_before=color_declared,
            before=color_before,
            effective_before=color_effective,
            after=color_after,
        )
    )

    return result, changes


def _section_record(document: Mapping[str, Any], section: str) -> dict[str, Any]:
    present = section in document
    payload = {"present": present, "value": document.get(section)}
    return {"present": present, "canonical_sha256": canonical_json_sha256(payload)}


def _without_material_controls(document: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: copy.deepcopy(value)
        for key, value in document.items()
        if key != "materials"
    }
    extensions_used = result.get("extensionsUsed")
    if isinstance(extensions_used, list):
        retained = [value for value in extensions_used if value != _SPECULAR_EXTENSION]
        if retained:
            result["extensionsUsed"] = retained
        else:
            result.pop("extensionsUsed", None)
    return result


def _report_digest(report: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {
            key: copy.deepcopy(value)
            for key, value in report.items()
            if key != "report_content_sha256"
        }
    )


def normalize_glb_materials(
    source_path: str | Path,
    output_path: str | Path,
    *,
    force_opaque: bool = False,
) -> dict[str, Any]:
    """Normalize PBR materials and exclusively create one verified GLB output."""

    if not isinstance(force_opaque, bool):
        raise MaterialNormalizationError("force_opaque must be a boolean")
    source_resolved = Path(source_path).resolve()
    output_argument = Path(output_path)
    output_resolved = output_argument.resolve()
    if source_resolved == output_resolved:
        raise MaterialNormalizationError("output must not overwrite the source GLB")
    if output_argument.exists() or output_argument.is_symlink():
        raise MaterialNormalizationError(f"output already exists: {output_argument}")

    try:
        source = load_glb(source_resolved)
    except (OSError, GlbError) as exc:
        raise MaterialNormalizationError(
            f"input GLB failed strict parsing: {exc}"
        ) from exc
    source_document = source.json
    source_materials = _objects(
        source_document.get("materials"), "materials", require_nonempty=True
    )

    extensions_used = source_document.get("extensionsUsed", [])
    if not isinstance(extensions_used, list) or any(
        not isinstance(item, str) for item in extensions_used
    ):
        raise MaterialNormalizationError("extensionsUsed must be an array of strings")
    if len(set(extensions_used)) != len(extensions_used):
        raise MaterialNormalizationError("extensionsUsed must not contain duplicates")
    output_document = copy.deepcopy(source_document)
    if _SPECULAR_EXTENSION not in extensions_used:
        output_document["extensionsUsed"] = [
            *extensions_used,
            _SPECULAR_EXTENSION,
        ]
    normalized_materials: list[dict[str, Any]] = []
    material_reports: list[dict[str, Any]] = []
    for index, before in enumerate(source_materials):
        after, changes = _normalize_one_material(
            before, material_index=index, force_opaque=force_opaque
        )
        normalized_materials.append(after)
        material_reports.append(
            {
                "material_index": index,
                "name": before.get("name")
                if isinstance(before.get("name"), str)
                else None,
                "before": copy.deepcopy(before),
                "after": copy.deepcopy(after),
                "changed": before != after,
                "changes": changes,
            }
        )
    output_document["materials"] = normalized_materials

    payload = build_glb(output_document, source.binary)
    try:
        readback = parse_glb(payload)
    except GlbError as exc:
        raise MaterialNormalizationError(f"output GLB readback failed: {exc}") from exc
    readback_document = readback.json

    if readback.binary != source.binary:
        raise MaterialNormalizationError(
            "embedded BIN chunk changed during normalization"
        )
    expected_extensions_used = (
        extensions_used
        if _SPECULAR_EXTENSION in extensions_used
        else [*extensions_used, _SPECULAR_EXTENSION]
    )
    if readback_document.get("extensionsUsed") != expected_extensions_used:
        raise MaterialNormalizationError(
            "root extensionsUsed differs from the bounded material declaration"
        )
    if _without_material_controls(readback_document) != _without_material_controls(
        source_document
    ):
        raise MaterialNormalizationError(
            "glTF JSON outside bounded material controls changed during normalization"
        )
    if readback_document.get("materials") != normalized_materials:
        raise MaterialNormalizationError(
            "material JSON readback differs from normalized values"
        )

    invariant_sections: dict[str, Any] = {}
    for section in _REQUIRED_UNCHANGED_SECTIONS:
        before_record = _section_record(source_document, section)
        after_record = _section_record(readback_document, section)
        invariant_sections[section] = {
            "before": before_record,
            "after": after_record,
            "unchanged": before_record == after_record,
        }
        if before_record != after_record:
            raise MaterialNormalizationError(f"invariant section changed: {section}")

    source_binary_sha256 = hashlib.sha256(source.binary).hexdigest()
    output_binary_sha256 = hashlib.sha256(readback.binary).hexdigest()
    source_non_material_control_sha256 = canonical_json_sha256(
        _without_material_controls(source_document)
    )
    output_non_material_control_sha256 = canonical_json_sha256(
        _without_material_controls(readback_document)
    )
    report: dict[str, Any] = {
        "schema": MATERIAL_NORMALIZATION_SCHEMA,
        "status": "pass",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "source": {
            "path": str(source_resolved),
            "sha256": source.sha256,
            "byte_size": source.byte_length,
        },
        "output": {
            "path": str(output_resolved),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_size": len(payload),
        },
        "policy": {
            "metallic_factor": 0.0,
            "minimum_roughness_factor": MINIMUM_ROUGHNESS_FACTOR,
            "metallic_roughness_texture": "removed",
            "emissive_factor": ZERO_EMISSIVE_FACTOR,
            "emissive_texture": "removed",
            "maximum_specular_factor": MAXIMUM_SPECULAR_FACTOR,
            "specular_color_factor_range": [
                0.0,
                MAXIMUM_SPECULAR_COLOR_FACTOR,
            ],
            "allowed_material_extensions": sorted(_ALLOWED_MATERIAL_EXTENSIONS),
            "root_specular_extension_declaration": "preserve_or_add",
            "force_opaque": force_opaque,
            "alpha_policy": "force_opaque" if force_opaque else "preserve",
        },
        "material_count": len(material_reports),
        "materials": material_reports,
        "invariants": {
            "binary_chunk": {
                "before_sha256": source_binary_sha256,
                "after_sha256": output_binary_sha256,
                "unchanged": source_binary_sha256 == output_binary_sha256,
            },
            "required_json_sections": invariant_sections,
            "all_non_material_control_json": {
                "before_sha256": source_non_material_control_sha256,
                "after_sha256": output_non_material_control_sha256,
                "unchanged": source_non_material_control_sha256
                == output_non_material_control_sha256,
            },
            "only_material_control_json_changed": True,
        },
        "notes": [
            "This compiler changes only glTF material controls (materials and the required extensionsUsed declaration) and preserves embedded binary data.",
            "Materials, motion, deformation, contacts, provenance, Habitat playback, and human review remain separate gates.",
            "A passing normalization report does not qualify or register the output asset.",
        ],
    }
    report["report_content_sha256"] = _report_digest(report)
    validate_material_normalization_report(report, verify_files=False)

    output_resolved.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with output_resolved.open("xb") as handle:
            created = True
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        validate_material_normalization_report(report, verify_files=True)
    except FileExistsError as exc:
        raise MaterialNormalizationError(
            f"output already exists: {output_resolved}"
        ) from exc
    except Exception:
        if created:
            try:
                output_resolved.unlink()
            except OSError:
                pass
        raise
    return report


def validate_material_normalization_report(
    value: Any, *, verify_files: bool = True
) -> None:
    """Verify report self-integrity and, by default, source/output file closure."""

    if not isinstance(value, dict):
        raise MaterialNormalizationError("report must be an object")
    required = {
        "schema",
        "status",
        "qualification_state",
        "qualification_claim",
        "source",
        "output",
        "policy",
        "material_count",
        "materials",
        "invariants",
        "notes",
        "report_content_sha256",
    }
    if set(value) != required:
        raise MaterialNormalizationError(
            "report keys differ: "
            f"missing={sorted(required - set(value))}, "
            f"extra={sorted(set(value) - required)}"
        )
    if value["schema"] != MATERIAL_NORMALIZATION_SCHEMA or value["status"] != "pass":
        raise MaterialNormalizationError("report schema/status is invalid")
    if (
        value["qualification_state"] != "research_candidate"
        or value["qualification_claim"] is not False
    ):
        raise MaterialNormalizationError("report must not claim asset qualification")
    if value["report_content_sha256"] != _report_digest(value):
        raise MaterialNormalizationError(
            "report_content_sha256 does not authenticate report content"
        )
    if not isinstance(value["materials"], list) or value["material_count"] != len(
        value["materials"]
    ):
        raise MaterialNormalizationError("material_count does not match materials")
    policy = _object(value["policy"], "policy")
    expected_policy_keys = {
        "metallic_factor",
        "minimum_roughness_factor",
        "metallic_roughness_texture",
        "emissive_factor",
        "emissive_texture",
        "maximum_specular_factor",
        "specular_color_factor_range",
        "allowed_material_extensions",
        "root_specular_extension_declaration",
        "force_opaque",
        "alpha_policy",
    }
    if set(policy) != expected_policy_keys:
        raise MaterialNormalizationError("report policy keys are invalid")
    force_opaque = policy["force_opaque"]
    if not isinstance(force_opaque, bool):
        raise MaterialNormalizationError("policy.force_opaque must be a boolean")
    expected_policy = {
        "metallic_factor": 0.0,
        "minimum_roughness_factor": MINIMUM_ROUGHNESS_FACTOR,
        "metallic_roughness_texture": "removed",
        "emissive_factor": ZERO_EMISSIVE_FACTOR,
        "emissive_texture": "removed",
        "maximum_specular_factor": MAXIMUM_SPECULAR_FACTOR,
        "specular_color_factor_range": [0.0, MAXIMUM_SPECULAR_COLOR_FACTOR],
        "allowed_material_extensions": sorted(_ALLOWED_MATERIAL_EXTENSIONS),
        "root_specular_extension_declaration": "preserve_or_add",
        "force_opaque": force_opaque,
        "alpha_policy": "force_opaque" if force_opaque else "preserve",
    }
    if policy != expected_policy:
        raise MaterialNormalizationError("report policy values are invalid")
    if (
        not isinstance(value["notes"], list)
        or not value["notes"]
        or any(not isinstance(note, str) or not note for note in value["notes"])
    ):
        raise MaterialNormalizationError("report notes must be non-empty strings")
    invariants = _object(value["invariants"], "invariants")
    if (
        invariants.get("only_material_control_json_changed") is not True
        or _object(invariants.get("binary_chunk"), "invariants.binary_chunk").get(
            "unchanged"
        )
        is not True
        or _object(
            invariants.get("all_non_material_control_json"),
            "invariants.all_non_material_control_json",
        ).get("unchanged")
        is not True
    ):
        raise MaterialNormalizationError("report invariants are not all passing")
    section_reports = _object(
        invariants.get("required_json_sections"),
        "invariants.required_json_sections",
    )
    if set(section_reports) != set(_REQUIRED_UNCHANGED_SECTIONS) or any(
        not isinstance(record, dict) or record.get("unchanged") is not True
        for record in section_reports.values()
    ):
        raise MaterialNormalizationError(
            "required JSON-section invariants are incomplete"
        )

    if verify_files:
        records: dict[str, dict[str, Any]] = {}
        for label in ("source", "output"):
            record = _object(value[label], label)
            if set(record) != {"path", "sha256", "byte_size"}:
                raise MaterialNormalizationError(f"{label} record keys are invalid")
            path = Path(record.get("path", ""))
            if not path.is_file():
                raise MaterialNormalizationError(f"{label} file is missing: {path}")
            if record.get("byte_size") != path.stat().st_size:
                raise MaterialNormalizationError(f"{label} byte_size mismatch")
            if record.get("sha256") != sha256_file(path):
                raise MaterialNormalizationError(f"{label} sha256 mismatch")
            records[label] = record

        try:
            source = load_glb(records["source"]["path"])
            output = load_glb(records["output"]["path"])
        except (OSError, GlbError) as exc:
            raise MaterialNormalizationError(
                f"report GLB closure failed strict parsing: {exc}"
            ) from exc
        source_document = source.json
        output_document = output.json
        source_extensions_used = source_document.get("extensionsUsed", [])
        if not isinstance(source_extensions_used, list) or any(
            not isinstance(item, str) for item in source_extensions_used
        ):
            raise MaterialNormalizationError(
                "source extensionsUsed must be an array of strings"
            )
        if len(set(source_extensions_used)) != len(source_extensions_used):
            raise MaterialNormalizationError(
                "source extensionsUsed must not contain duplicates"
            )
        expected_extensions_used = (
            source_extensions_used
            if _SPECULAR_EXTENSION in source_extensions_used
            else [*source_extensions_used, _SPECULAR_EXTENSION]
        )
        if output_document.get("extensionsUsed") != expected_extensions_used:
            raise MaterialNormalizationError(
                "output extensionsUsed differs from the bounded material declaration"
            )
        source_materials = _objects(
            source_document.get("materials"), "source materials", require_nonempty=True
        )
        output_materials = _objects(
            output_document.get("materials"), "output materials", require_nonempty=True
        )
        if source.binary != output.binary:
            raise MaterialNormalizationError("report GLB binary chunks differ")
        if _without_material_controls(source_document) != _without_material_controls(
            output_document
        ):
            raise MaterialNormalizationError(
                "report GLBs differ outside bounded material controls"
            )

        expected_materials: list[dict[str, Any]] = []
        expected_material_reports: list[dict[str, Any]] = []
        for index, before in enumerate(source_materials):
            after, changes = _normalize_one_material(
                before, material_index=index, force_opaque=force_opaque
            )
            expected_materials.append(after)
            expected_material_reports.append(
                {
                    "material_index": index,
                    "name": before.get("name")
                    if isinstance(before.get("name"), str)
                    else None,
                    "before": copy.deepcopy(before),
                    "after": copy.deepcopy(after),
                    "changed": before != after,
                    "changes": changes,
                }
            )
        if output_materials != expected_materials:
            raise MaterialNormalizationError(
                "output materials do not match the declared normalization policy"
            )
        if value["materials"] != expected_material_reports:
            raise MaterialNormalizationError(
                "report material before/after records do not match source/output"
            )

        binary_sha256 = hashlib.sha256(source.binary).hexdigest()
        expected_sections: dict[str, Any] = {}
        for section in _REQUIRED_UNCHANGED_SECTIONS:
            before_record = _section_record(source_document, section)
            after_record = _section_record(output_document, section)
            expected_sections[section] = {
                "before": before_record,
                "after": after_record,
                "unchanged": before_record == after_record,
            }
        non_material_control_before = canonical_json_sha256(
            _without_material_controls(source_document)
        )
        non_material_control_after = canonical_json_sha256(
            _without_material_controls(output_document)
        )
        expected_invariants = {
            "binary_chunk": {
                "before_sha256": binary_sha256,
                "after_sha256": hashlib.sha256(output.binary).hexdigest(),
                "unchanged": source.binary == output.binary,
            },
            "required_json_sections": expected_sections,
            "all_non_material_control_json": {
                "before_sha256": non_material_control_before,
                "after_sha256": non_material_control_after,
                "unchanged": non_material_control_before == non_material_control_after,
            },
            "only_material_control_json_changed": True,
        }
        if invariants != expected_invariants:
            raise MaterialNormalizationError(
                "report invariants do not match source/output readback"
            )


def load_and_validate_material_normalization_report(
    path: str | Path, *, verify_files: bool = True
) -> dict[str, Any]:
    """Load and verify a material-normalization report."""

    value = load_json(path)
    validate_material_normalization_report(value, verify_files=verify_files)
    return value
