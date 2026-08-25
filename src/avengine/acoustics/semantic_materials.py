"""SoundSpaces-style semantic-to-acoustic material compilation.

This module intentionally produces the existing M3 mapping/database documents.
It is a deterministic research proposal layer, not a second runtime material
format and not a claim that semantic labels recover physical room materials.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np


class SemanticMaterialRuleError(ValueError):
    pass


@dataclass(frozen=True)
class SemanticSurfaceIdentity:
    source_material_name: str
    semantic_category: str
    identity_key: str = ""
    material_slot: str = ""
    object_name: str = ""


@dataclass(frozen=True)
class CompiledSemanticMaterials:
    mapping: dict[str, Any]
    database: dict[str, Any]
    report: dict[str, Any]


_CURVE_FIELDS = ("absorption", "scattering", "transmission", "damping")


def _token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _category_token(source_name: str) -> str:
    slug = _token(source_name)[:36] or "unknown"
    suffix = hashlib.sha256(source_name.encode("utf-8")).hexdigest()[:10]
    return f"avm3_sem_{slug}_{suffix}"


def _stable_uint64(seed: int, identity: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{identity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _candidate_records(value: Any, *, owner: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise SemanticMaterialRuleError(f"{owner} must be a non-empty array")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            material = item
            weight = 1.0
        elif isinstance(item, Mapping):
            material = item.get("material")
            weight = item.get("weight", 1.0)
        else:
            raise SemanticMaterialRuleError(
                f"{owner}[{index}] must be a material name or candidate object"
            )
        if not isinstance(material, str) or not material:
            raise SemanticMaterialRuleError(
                f"{owner}[{index}].material must be non-empty"
            )
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) <= 0
        ):
            raise SemanticMaterialRuleError(
                f"{owner}[{index}].weight must be positive and finite"
            )
        records.append({"material": material, "weight": float(weight)})
    return records


def _validate_rules(rules: Mapping[str, Any]) -> tuple[list[float], dict[str, Any]]:
    if rules.get("schema") != "avengine_m3_semantic_material_rules_v1":
        raise SemanticMaterialRuleError(
            "rules.schema must be 'avengine_m3_semantic_material_rules_v1'"
        )
    if not isinstance(rules.get("ruleset_id"), str) or not rules["ruleset_id"]:
        raise SemanticMaterialRuleError("rules.ruleset_id must be non-empty")
    raw_bands = rules.get("bands_hz")
    if not isinstance(raw_bands, list) or not raw_bands:
        raise SemanticMaterialRuleError("rules.bands_hz must be a non-empty array")
    bands = [float(value) for value in raw_bands]
    if (
        any(not math.isfinite(value) or value <= 0 for value in bands)
        or any(left >= right for left, right in zip(bands, bands[1:]))
    ):
        raise SemanticMaterialRuleError(
            "rules.bands_hz must contain increasing positive finite frequencies"
        )
    materials = rules.get("materials")
    if not isinstance(materials, Mapping) or not materials:
        raise SemanticMaterialRuleError("rules.materials must be a non-empty object")
    normalized_materials: dict[str, Any] = {}
    for key, raw in materials.items():
        if not isinstance(key, str) or not key or not isinstance(raw, Mapping):
            raise SemanticMaterialRuleError(
                "every rules.materials entry must have a non-empty key and object value"
            )
        material = copy.deepcopy(dict(raw))
        for field in ("name", "source"):
            if not isinstance(material.get(field), str) or not material[field]:
                raise SemanticMaterialRuleError(
                    f"rules.materials.{key}.{field} must be non-empty"
                )
        for field in _CURVE_FIELDS:
            curve = material.get(field)
            if not isinstance(curve, list) or len(curve) != len(bands):
                raise SemanticMaterialRuleError(
                    f"rules.materials.{key}.{field} must have {len(bands)} values"
                )
            values = [float(value) for value in curve]
            if any(not math.isfinite(value) for value in values):
                raise SemanticMaterialRuleError(
                    f"rules.materials.{key}.{field} must contain finite values"
                )
            if field == "damping":
                invalid = any(value < 0 for value in values)
            else:
                invalid = any(not 0 <= value <= 1 for value in values)
            if invalid:
                raise SemanticMaterialRuleError(
                    f"rules.materials.{key}.{field} values are outside their range"
                )
            material[field] = values
        for field in ("density", "speed"):
            value = material.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise SemanticMaterialRuleError(
                    f"rules.materials.{key}.{field} must be positive and finite"
                )
            material[field] = float(value)
        confidence = material.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0 <= float(confidence) <= 1
        ):
            raise SemanticMaterialRuleError(
                f"rules.materials.{key}.confidence must be in [0, 1]"
            )
        material["confidence"] = float(confidence)
        normalized_materials[key] = material

    default_candidates = _candidate_records(
        rules.get("default_candidates"), owner="rules.default_candidates"
    )
    categories = rules.get("categories", {})
    if not isinstance(categories, Mapping):
        raise SemanticMaterialRuleError("rules.categories must be an object")
    normalized_categories: dict[str, list[dict[str, Any]]] = {}
    for category, raw in categories.items():
        if not isinstance(category, str) or not isinstance(raw, Mapping):
            raise SemanticMaterialRuleError(
                "rules.categories entries must map strings to objects"
            )
        normalized_categories[_token(category)] = _candidate_records(
            raw.get("candidates"),
            owner=f"rules.categories.{category}.candidates",
        )

    name_hints = rules.get("name_hints", [])
    if not isinstance(name_hints, list):
        raise SemanticMaterialRuleError("rules.name_hints must be an array")
    normalized_hints: list[dict[str, Any]] = []
    for index, raw in enumerate(name_hints):
        if not isinstance(raw, Mapping):
            raise SemanticMaterialRuleError(
                f"rules.name_hints[{index}] must be an object"
            )
        contains = raw.get("contains")
        if (
            not isinstance(contains, list)
            or not contains
            or any(not isinstance(value, str) or not _token(value) for value in contains)
        ):
            raise SemanticMaterialRuleError(
                f"rules.name_hints[{index}].contains must contain strings"
            )
        normalized_hints.append(
            {
                "contains": [_token(value) for value in contains],
                "candidates": _candidate_records(
                    raw.get("candidates"),
                    owner=f"rules.name_hints[{index}].candidates",
                ),
            }
        )
    explicit = rules.get("explicit_overrides", {})
    if not isinstance(explicit, Mapping) or any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or not value
        for key, value in explicit.items()
    ):
        raise SemanticMaterialRuleError(
            "rules.explicit_overrides must map non-empty identity strings to materials"
        )
    all_candidate_sets = [
        default_candidates,
        *normalized_categories.values(),
        *(hint["candidates"] for hint in normalized_hints),
    ]
    referenced = {
        candidate["material"]
        for candidates in all_candidate_sets
        for candidate in candidates
    } | set(explicit.values())
    missing = sorted(referenced - set(normalized_materials))
    if missing:
        raise SemanticMaterialRuleError(
            "rules reference undefined materials: " + ", ".join(missing)
        )
    jitter_std = rules.get("coefficient_jitter_std", 0.0)
    if (
        isinstance(jitter_std, bool)
        or not isinstance(jitter_std, (int, float))
        or not math.isfinite(float(jitter_std))
        or not 0 <= float(jitter_std) <= 0.25
    ):
        raise SemanticMaterialRuleError(
            "rules.coefficient_jitter_std must be finite in [0, 0.25]"
        )
    raw_jitter_fields = rules.get(
        "coefficient_jitter_fields", ["absorption", "scattering"]
    )
    if (
        not isinstance(raw_jitter_fields, list)
        or any(field not in ("absorption", "scattering", "transmission") for field in raw_jitter_fields)
        or len(raw_jitter_fields) != len(set(raw_jitter_fields))
    ):
        raise SemanticMaterialRuleError(
            "rules.coefficient_jitter_fields must be unique coefficient field names"
        )
    return bands, {
        "materials": normalized_materials,
        "default_candidates": default_candidates,
        "categories": normalized_categories,
        "name_hints": normalized_hints,
        "explicit_overrides": dict(explicit),
        "jitter_std": float(jitter_std),
        "jitter_fields": list(raw_jitter_fields),
    }


def _select_candidate(
    candidates: Sequence[Mapping[str, Any]], *, seed: int, identity: str
) -> str:
    total = sum(float(candidate["weight"]) for candidate in candidates)
    unit = _stable_uint64(seed, identity) / float(1 << 64)
    threshold = unit * total
    cumulative = 0.0
    for candidate in candidates:
        cumulative += float(candidate["weight"])
        if threshold < cumulative:
            return str(candidate["material"])
    return str(candidates[-1]["material"])


def _resolve_surface(
    surface: SemanticSurfaceIdentity,
    normalized: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]], bool]:
    explicit = normalized["explicit_overrides"]
    if surface.identity_key and surface.identity_key in explicit:
        return (
            "explicit_override",
            [{"material": explicit[surface.identity_key], "weight": 1.0}],
            True,
        )
    haystack = _token(
        " ".join(
            (
                surface.material_slot,
                surface.object_name,
                surface.source_material_name,
            )
        )
    )
    for hint in normalized["name_hints"]:
        if any(token in haystack for token in hint["contains"]):
            return "name_hint", hint["candidates"], False
    category = _token(surface.semantic_category)
    if category in normalized["categories"]:
        return (
            "semantic_category",
            normalized["categories"][category],
            False,
        )
    return "default_candidate", normalized["default_candidates"], False


def compile_semantic_material_documents(
    *,
    room_id: str,
    surfaces: Sequence[SemanticSurfaceIdentity],
    rules: Mapping[str, Any],
    seed: int,
    source_to_canonical: Mapping[str, Any],
) -> CompiledSemanticMaterials:
    """Resolve semantic identities into existing M3 mapping/database documents."""

    if not isinstance(room_id, str) or not room_id:
        raise SemanticMaterialRuleError("room_id must be non-empty")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SemanticMaterialRuleError("seed must be a non-negative integer")
    if not surfaces:
        raise SemanticMaterialRuleError("at least one semantic surface is required")
    names = [surface.source_material_name for surface in surfaces]
    if any(not isinstance(name, str) or not name for name in names):
        raise SemanticMaterialRuleError(
            "every semantic surface source_material_name must be non-empty"
        )
    if len(names) != len(set(names)):
        raise SemanticMaterialRuleError(
            "semantic surface source_material_name values must be unique"
        )
    bands, normalized = _validate_rules(rules)

    mapping_entries: list[dict[str, Any]] = []
    database_materials: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    unknown_categories: set[str] = set()
    for material_id, surface in enumerate(
        sorted(surfaces, key=lambda value: value.source_material_name)
    ):
        resolution, candidates, human_override = _resolve_surface(surface, normalized)
        choice_identity = (
            f"{room_id}\0{surface.semantic_category}\0{surface.identity_key}"
            f"\0{surface.source_material_name}\0{resolution}"
        )
        selected = _select_candidate(
            candidates, seed=seed, identity=choice_identity
        )
        base = copy.deepcopy(normalized["materials"][selected])
        jitter_applied = bool(
            not human_override
            and normalized["jitter_std"] > 0
            and normalized["jitter_fields"]
        )
        if jitter_applied:
            rng = np.random.default_rng(
                _stable_uint64(seed, f"jitter\0{choice_identity}\0{selected}")
            )
            for field in normalized["jitter_fields"]:
                values = np.asarray(base[field], dtype=np.float64)
                values += rng.normal(
                    loc=0.0,
                    scale=normalized["jitter_std"],
                    size=len(values),
                )
                base[field] = np.clip(values, 0.0, 1.0).astype(float).tolist()
        category_name = _category_token(surface.source_material_name)
        material_key = (
            f"semantic_{selected}_"
            f"{hashlib.sha256(surface.source_material_name.encode('utf-8')).hexdigest()[:12]}"
        )
        confidence_cap = {
            "explicit_override": 0.9,
            "name_hint": 0.75,
            "semantic_category": 0.65,
            "default_candidate": 0.2,
        }[resolution]
        confidence = min(float(base["confidence"]), confidence_cap)
        randomized = bool(len(candidates) > 1 or jitter_applied)
        mapping_entries.append(
            {
                "source_material_name": surface.source_material_name,
                "material_id": material_id,
                "category_name": category_name,
                "material_key": material_key,
                "mapping_source": (
                    f"{rules['ruleset_id']}:{resolution}:{selected}; "
                    "semantic research proposal, not measured physical truth"
                ),
                "mapping_confidence": confidence,
                "human_override": human_override,
                "randomized": randomized,
                "fallback": False,
            }
        )
        database_materials.append(
            {
                "material_key": material_key,
                "name": f"{base['name']} ({surface.source_material_name})",
                "labels": [category_name],
                "absorption": list(base["absorption"]),
                "scattering": list(base["scattering"]),
                "transmission": list(base["transmission"]),
                "damping": list(base["damping"]),
                "density": float(base["density"]),
                "speed": float(base["speed"]),
                "source": (
                    f"{base['source']}; selected by {rules['ruleset_id']} "
                    f"as {resolution}; uncalibrated research candidate"
                ),
                "confidence": confidence,
            }
        )
        if resolution == "default_candidate":
            unknown_categories.add(_token(surface.semantic_category) or "unknown")
        decisions.append(
            {
                "source_material_name": surface.source_material_name,
                "semantic_category": _token(surface.semantic_category) or "unknown",
                "identity_key": surface.identity_key,
                "resolution": resolution,
                "candidate_materials": [
                    {
                        "material": candidate["material"],
                        "weight": float(candidate["weight"]),
                    }
                    for candidate in candidates
                ],
                "selected_material": selected,
                "coefficient_jitter_applied": jitter_applied,
                "mapping_confidence": confidence,
            }
        )

    mapping = {
        "schema": "avengine_m3_acoustic_material_mapping_v1",
        "mapping_id": f"{room_id}_{rules['ruleset_id']}_seed{seed}",
        "room_id": room_id,
        "mapping_source_kind": "semantic_proposal",
        "source_to_canonical": copy.deepcopy(dict(source_to_canonical)),
        "entries": mapping_entries,
    }
    database = {
        "schema": "avengine_m3_acoustic_material_database_v1",
        "database_id": f"{room_id}_{rules['ruleset_id']}_seed{seed}",
        "version": "1",
        "bands_hz": bands,
        "coefficient_units": {
            "absorption": "fraction_of_incident_sound_pressure",
            "scattering": "fraction_of_incident_sound_pressure",
            "transmission": "fraction_of_incident_sound_pressure",
            "damping": "decibels_per_meter",
            "density": "kilograms_per_cubic_meter",
            "speed": "meters_per_second",
        },
        "provenance": {
            "source": (
                f"AVEngine SoundSpaces-style semantic rules {rules['ruleset_id']}; "
                "plausible candidate randomization without physical room calibration"
            ),
            "confidence": max(
                (entry["mapping_confidence"] for entry in mapping_entries),
                default=0.0,
            ),
            "material_semantics": "research_placeholder",
            "intended_use": "research_compiler_diagnostics",
        },
        "materials": database_materials,
    }
    report = {
        "schema": "avengine_m3_semantic_material_coverage_v1",
        "status": "research_candidate",
        "qualification_claim": False,
        "physical_acoustic_material_claim": False,
        "room_id": room_id,
        "ruleset_id": rules["ruleset_id"],
        "seed": seed,
        "coefficient_jitter_std": normalized["jitter_std"],
        "coefficient_jitter_fields": normalized["jitter_fields"],
        "precedence": [
            "explicit_override",
            "material_slot_or_object_name_hint",
            "semantic_category",
            "plausible_default_candidate_set",
        ],
        "surface_count": len(surfaces),
        "unknown_semantic_category_count": len(unknown_categories),
        "unknown_semantic_categories": sorted(unknown_categories),
        "decisions": decisions,
    }
    return CompiledSemanticMaterials(
        mapping=mapping,
        database=database,
        report=report,
    )
