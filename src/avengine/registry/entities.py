"""Entity and animal-template registries for the bounded M6 interface layer."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

from avengine.contracts.json_io import canonical_json_sha256
from avengine.registry.registry import (
    ANIMAL_TEMPLATE_REGISTRY_SCHEMA,
    ENTITY_ASSET_REGISTRY_SCHEMA,
    M6RegistryError,
    json_schema_errors,
    load_validated_document,
    registry_semantic_errors,
    resolve_record,
)


SIZE_VALUES = ("small", "medium", "large")
BODY_BUILD_VALUES = ("slim", "standard", "stocky")
LIFE_STAGE_VALUES = ("young", "adult", "senior")
TEMPLATE_STATES = ("formal", "research", "unavailable", "rejected")
STATIC_OBJECT_EVIDENCE_KINDS = (
    "emitter_marker_glb",
    "marker_visual_approval",
    "spear_static_admission_batch",
    "spear_static_admission_job_receipt",
    "spear_static_emitter_stage_receipt",
    "spear_static_finalization_stage_receipt",
    "spear_static_watertight_stage_receipt",
    "visual_asset_glb",
)


def _canonical_ids(items: Any, field: str, owner: str) -> list[str]:
    if not isinstance(items, list):
        return []
    values = [str(item[field]) for item in items if isinstance(item, Mapping) and field in item]
    errors: list[str] = []
    if len(values) != len(set(values)):
        errors.append(f"{owner} must have unique {field} values")
    if values != sorted(values):
        errors.append(f"{owner} must use canonical bytewise {field} order")
    return errors


def validate_entity_asset_registry(value: Any) -> list[str]:
    errors = json_schema_errors(value, ENTITY_ASSET_REGISTRY_SCHEMA)
    errors.extend(
        registry_semantic_errors(
            value,
            records_field="entities",
            record_id_field="entity_asset_id",
        )
    )
    if not isinstance(value, Mapping) or not isinstance(value.get("entities"), list):
        return errors
    for index, entity in enumerate(value["entities"]):
        if not isinstance(entity, Mapping):
            continue
        owner = f"entities[{index}]"
        errors.extend(_canonical_ids(entity.get("emitter_anchors"), "anchor_id", f"{owner}.emitter_anchors"))
        raw_capabilities = entity.get("capabilities")
        action_ids = (
            raw_capabilities.get("action_ids", [])
            if isinstance(raw_capabilities, Mapping)
            else []
        )
        if isinstance(action_ids, list) and action_ids != sorted(set(action_ids)):
            errors.append(f"{owner}.capabilities.action_ids must be unique and canonical")
        admission_evidence = entity.get("admission_evidence")
        if isinstance(admission_evidence, Mapping):
            artifacts = admission_evidence.get("artifacts")
            kinds = (
                [item.get("kind") for item in artifacts if isinstance(item, Mapping)]
                if isinstance(artifacts, list)
                else []
            )
            if tuple(kinds) != STATIC_OBJECT_EVIDENCE_KINDS:
                errors.append(
                    f"{owner}.admission_evidence.artifacts must contain the "
                    "canonical static-object evidence closure"
                )
            payload = {
                key: item
                for key, item in admission_evidence.items()
                if key != "evidence_content_sha256"
            }
            evidence_hash = admission_evidence.get("evidence_content_sha256")
            if evidence_hash != canonical_json_sha256(payload):
                errors.append(
                    f"{owner}.admission_evidence.evidence_content_sha256 "
                    "does not match canonical content"
                )
            provenance = entity.get("provenance")
            if (
                isinstance(provenance, Mapping)
                and provenance.get("evidence_sha256") != evidence_hash
            ):
                errors.append(
                    f"{owner}.provenance.evidence_sha256 does not bind "
                    "admission_evidence"
                )
            entity_id = entity.get("entity_asset_id")
            identity = admission_evidence.get("identity")
            if (
                not isinstance(identity, Mapping)
                or identity.get("instance_id") != entity_id
            ):
                errors.append(
                    f"{owner}.admission_evidence.identity.instance_id must "
                    "match entity_asset_id"
                )
            if (
                admission_evidence.get(
                    "formal_dataset_registration_authorized"
                )
                is not False
            ):
                errors.append(
                    f"{owner}.admission_evidence cannot claim formal "
                    "dataset registration"
                )
            if (
                entity.get("entity_class") != "rigid_object"
                or entity.get("admission_state") != "research"
            ):
                errors.append(
                    f"{owner} static admission evidence is valid only for a "
                    "research rigid_object"
                )
            capabilities = entity.get("capabilities")
            if not isinstance(capabilities, Mapping) or any(
                capabilities.get(field) != expected
                for field, expected in (
                    ("articulated", False),
                    ("skeleton_revision", None),
                    ("skeleton_sha256", None),
                    ("action_ids", []),
                )
            ):
                errors.append(
                    f"{owner}.capabilities must be exactly joint-free rigid "
                    "static capabilities"
                )
            anchors = entity.get("emitter_anchors")
            if not isinstance(anchors, list) or len(anchors) != 1:
                errors.append(
                    f"{owner}.emitter_anchors must contain exactly one "
                    "static object_speaker"
                )
            else:
                anchor = anchors[0]
                if (
                    not isinstance(anchor, Mapping)
                    or anchor.get("anchor_type") != "object_speaker"
                    or anchor.get("joint_id") is not None
                    or anchor.get("anchor_id")
                    != admission_evidence.get("emitter_anchor_id")
                ):
                    errors.append(
                        f"{owner}.emitter_anchors[0] must be the joint-free "
                        "speaker named by admission_evidence"
                    )
            attributes = entity.get("realized_visual_attributes")
            if (
                not isinstance(attributes, Mapping)
                or attributes.get("source_instance_id") != entity_id
                or attributes.get(
                    "formal_dataset_registration_authorized"
                )
                is not False
            ):
                errors.append(
                    f"{owner}.realized_visual_attributes must bind the "
                    "research instance and keep formal authorization false"
                )
            visual = entity.get("visual_asset")
            visual_records = (
                [
                    item
                    for item in artifacts
                    if isinstance(item, Mapping)
                    and item.get("kind") == "visual_asset_glb"
                ]
                if isinstance(artifacts, list)
                else []
            )
            if (
                not isinstance(visual, Mapping)
                or len(visual_records) != 1
                or visual_records[0].get("path") != visual.get("uri")
                or visual_records[0].get("sha256") != visual.get("sha256")
            ):
                errors.append(
                    f"{owner}.visual_asset must match the authenticated "
                    "visual_asset_glb evidence record"
                )
            if (
                not isinstance(evidence_hash, str)
                or entity.get("revision") != f"spear_static_{evidence_hash}"
            ):
                errors.append(
                    f"{owner}.revision must bind the complete static "
                    "admission-evidence hash"
                )
        if entity.get("entity_class") == "articulated_animal":
            attributes = entity.get("realized_visual_attributes", {})
            for field, allowed in (
                ("size", SIZE_VALUES),
                ("body_build", BODY_BUILD_VALUES),
                ("life_stage", LIFE_STAGE_VALUES),
            ):
                if attributes.get(field) not in allowed:
                    errors.append(f"{owner}.realized_visual_attributes.{field} is not canonical")
            coat = attributes.get("coat_profile")
            if not isinstance(coat, Mapping) or not all(
                isinstance(coat.get(field), str) and coat.get(field)
                for field in ("profile_id", "value")
            ):
                errors.append(f"{owner}.realized_visual_attributes.coat_profile is required")
    return errors


def validate_animal_template_registry(value: Any) -> list[str]:
    errors = json_schema_errors(value, ANIMAL_TEMPLATE_REGISTRY_SCHEMA)
    errors.extend(
        registry_semantic_errors(
            value,
            records_field="templates",
            record_id_field="template_id",
        )
    )
    if not isinstance(value, Mapping) or not isinstance(value.get("templates"), list):
        return errors

    dimensions_by_body_plan: dict[str, tuple[str, ...]] = {}
    for index, template in enumerate(value["templates"]):
        if not isinstance(template, Mapping):
            continue
        owner = f"templates[{index}]"
        errors.extend(_canonical_ids(template.get("emitter_anchors"), "anchor_id", f"{owner}.emitter_anchors"))
        errors.extend(_canonical_ids(template.get("action_families"), "action_family_id", f"{owner}.action_families"))
        errors.extend(_canonical_ids(template.get("contact_semantics"), "contact_id", f"{owner}.contact_semantics"))

        appearance = template.get("appearance_domains", {})
        for field, expected in (
            ("size", SIZE_VALUES),
            ("body_build", BODY_BUILD_VALUES),
            ("life_stage", LIFE_STAGE_VALUES),
        ):
            if tuple(appearance.get(field, ())) != expected:
                errors.append(f"{owner}.appearance_domains.{field} must be {list(expected)}")
        coat = appearance.get("coat_profile", {})
        coat_values = coat.get("values", []) if isinstance(coat, Mapping) else []
        if not isinstance(coat_values, list) or len(coat_values) != 3 or len(set(coat_values)) != 3:
            errors.append(f"{owner}.appearance_domains.coat_profile requires exactly three distinct breed-scoped values")
        taxonomy = template.get("taxonomy", {})
        expected_prefix = f"{taxonomy.get('species_id', '')}_{taxonomy.get('breed_id', '')}_"
        if not str(coat.get("profile_id", "")).startswith(expected_prefix):
            errors.append(f"{owner}.appearance_domains.coat_profile.profile_id must be species/breed scoped")

        reference = template.get("morphology_reference", {})
        ranges = template.get("allowed_morphology_range", {})
        if isinstance(reference, Mapping) and isinstance(ranges, Mapping):
            reference_keys = tuple(sorted(reference))
            range_keys = tuple(sorted(ranges))
            if reference_keys != range_keys:
                errors.append(f"{owner} morphology reference/range dimensions must match")
            body_plan_id = str(template.get("body_plan_id", ""))
            previous = dimensions_by_body_plan.setdefault(body_plan_id, reference_keys)
            if previous != reference_keys:
                errors.append(f"{owner} body-plan templates must share morphology dimensions")
            for dimension in reference_keys:
                bounds = ranges.get(dimension, {})
                if not isinstance(bounds, Mapping):
                    continue
                low = bounds.get("minimum")
                high = bounds.get("maximum")
                center = reference.get(dimension)
                if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in (low, high, center)):
                    if not float(low) < float(high):
                        errors.append(f"{owner}.allowed_morphology_range.{dimension} requires minimum < maximum")
                    elif not float(low) <= float(center) <= float(high):
                        errors.append(f"{owner}.morphology_reference.{dimension} is outside its allowed range")
    return errors


def load_entity_asset_registry(path: str | Path) -> dict[str, Any]:
    return load_validated_document(path, validator=validate_entity_asset_registry)


def load_animal_template_registry(path: str | Path) -> dict[str, Any]:
    return load_validated_document(path, validator=validate_animal_template_registry)


def resolve_entity_asset(
    registry: Mapping[str, Any], entity_asset_id: str, revision: str
) -> Mapping[str, Any]:
    errors = validate_entity_asset_registry(registry)
    if errors:
        raise M6RegistryError(errors)
    return resolve_record(
        registry,
        records_field="entities",
        record_id_field="entity_asset_id",
        record_id=entity_asset_id,
        revision=revision,
    )


@dataclass(frozen=True)
class MorphologyCandidate:
    template_id: str
    revision: str
    normalized_distance: float
    within_allowed_range: bool
    exceeded_dimensions: tuple[str, ...]


@dataclass(frozen=True)
class AnimalTemplateSelection:
    status: str
    selected_template_id: str | None
    selected_revision: str | None
    nearest_template_id: str | None
    normalized_distance: float | None
    candidates: tuple[MorphologyCandidate, ...]
    rejection_code: str | None
    rejection_reason: str | None


def _rejected(
    code: str,
    reason: str,
    *,
    candidates: tuple[MorphologyCandidate, ...] = (),
) -> AnimalTemplateSelection:
    nearest = candidates[0] if candidates else None
    return AnimalTemplateSelection(
        status="rejected",
        selected_template_id=None,
        selected_revision=None,
        nearest_template_id=None if nearest is None else nearest.template_id,
        normalized_distance=None if nearest is None else nearest.normalized_distance,
        candidates=candidates,
        rejection_code=code,
        rejection_reason=reason,
    )


def select_animal_template(
    registry: Mapping[str, Any], request: Mapping[str, Any]
) -> AnimalTemplateSelection:
    """Select an in-range template or return an explicit, structured rejection.

    Body-plan and morphotype are hard filters.  The nearest out-of-range
    template is reported for audit only and is never silently selected.
    """

    errors = validate_animal_template_registry(registry)
    if errors:
        raise M6RegistryError(errors)
    body_plan_id = request.get("body_plan_id")
    morphotype_id = request.get("morphotype_id")
    measurements = request.get("measurements")
    if not isinstance(body_plan_id, str) or not body_plan_id:
        raise ValueError("request.body_plan_id must be a non-empty string")
    if not isinstance(morphotype_id, str) or not morphotype_id:
        raise ValueError("request.morphotype_id must be a non-empty string")
    if not isinstance(measurements, Mapping) or not measurements:
        raise ValueError("request.measurements must be a non-empty mapping")
    for dimension, number in measurements.items():
        if (
            not isinstance(dimension, str)
            or not dimension
            or isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
        ):
            raise ValueError("request.measurements must contain finite numeric dimensions")

    body_candidates = [
        item for item in registry["templates"] if item["body_plan_id"] == body_plan_id
    ]
    if not body_candidates:
        return _rejected(
            "no_body_plan_candidate",
            f"no registered template has body_plan_id={body_plan_id!r}",
        )
    morphotype_candidates = [
        item for item in body_candidates if item["morphotype_id"] == morphotype_id
    ]
    if not morphotype_candidates:
        return _rejected(
            "no_morphotype_candidate",
            f"body plan {body_plan_id!r} has no morphotype {morphotype_id!r}",
        )

    expected_dimensions = set(morphotype_candidates[0]["morphology_reference"])
    if set(measurements) != expected_dimensions:
        missing = sorted(expected_dimensions - set(measurements))
        extra = sorted(set(measurements) - expected_dimensions)
        return _rejected(
            "morphology_dimension_mismatch",
            f"measurement dimensions mismatch; missing={missing}, extra={extra}",
        )

    candidates: list[MorphologyCandidate] = []
    records: dict[tuple[str, str], Mapping[str, Any]] = {}
    for template in morphotype_candidates:
        exceeded: list[str] = []
        squared = 0.0
        for dimension in sorted(expected_dimensions):
            value = float(measurements[dimension])
            center = float(template["morphology_reference"][dimension])
            bounds = template["allowed_morphology_range"][dimension]
            low = float(bounds["minimum"])
            high = float(bounds["maximum"])
            if not low <= value <= high:
                exceeded.append(dimension)
            squared += ((value - center) / (high - low)) ** 2
        distance = math.sqrt(squared / len(expected_dimensions))
        candidate = MorphologyCandidate(
            template_id=template["template_id"],
            revision=template["revision"],
            normalized_distance=distance,
            within_allowed_range=not exceeded,
            exceeded_dimensions=tuple(exceeded),
        )
        candidates.append(candidate)
        records[(candidate.template_id, candidate.revision)] = template
    candidates.sort(key=lambda item: (item.normalized_distance, item.template_id, item.revision))
    ordered = tuple(candidates)
    selected = next((item for item in ordered if item.within_allowed_range), None)
    if selected is None:
        return _rejected(
            "morphology_out_of_distribution",
            "nearest registered template exceeds one or more allowed morphology dimensions",
            candidates=ordered,
        )
    if records[(selected.template_id, selected.revision)]["status"] != "formal":
        return _rejected(
            "template_not_formal",
            "nearest in-range template is not formally admitted",
            candidates=ordered,
        )
    return AnimalTemplateSelection(
        status="selected",
        selected_template_id=selected.template_id,
        selected_revision=selected.revision,
        nearest_template_id=selected.template_id,
        normalized_distance=selected.normalized_distance,
        candidates=ordered,
        rejection_code=None,
        rejection_reason=None,
    )


def validate_entity_template_bindings(
    entity_registry: Mapping[str, Any], template_registry: Mapping[str, Any]
) -> list[str]:
    """Validate explicit animal entity-to-template and appearance bindings."""

    errors = validate_entity_asset_registry(entity_registry)
    errors.extend(validate_animal_template_registry(template_registry))
    if errors:
        return errors
    templates = {
        (item["template_id"], item["revision"]): item
        for item in template_registry["templates"]
    }
    for index, entity in enumerate(entity_registry["entities"]):
        if entity["entity_class"] != "articulated_animal":
            continue
        ref = entity.get("animal_template_ref")
        key = (
            ref.get("template_id") if isinstance(ref, Mapping) else None,
            ref.get("revision") if isinstance(ref, Mapping) else None,
        )
        template = templates.get(key)
        if template is None:
            errors.append(f"entities[{index}].animal_template_ref is not registered")
            continue
        attributes = entity["realized_visual_attributes"]
        domains = template["appearance_domains"]
        for field in ("size", "body_build", "life_stage"):
            if attributes[field] not in domains[field]:
                errors.append(f"entities[{index}] {field} is outside template domain")
        coat = attributes["coat_profile"]
        template_coat = domains["coat_profile"]
        if coat["profile_id"] != template_coat["profile_id"] or coat["value"] not in template_coat["values"]:
            errors.append(f"entities[{index}] coat profile is outside its breed template")
    return errors
