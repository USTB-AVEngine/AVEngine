"""Animal appearance request contracts and a deterministic L9 generator.

The appearance vocabulary is intentionally profile scoped.  In particular,
coat values do not belong to a global color enum: a coat profile is bound to
one species/breed pair and a concrete request carries that binding forward.

L9 is a balanced combination design, not an isolation test.  The emitted
contract therefore records that separate OFAT evidence is still required.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    sha256_file,
)


ANIMAL_APPEARANCE_REQUEST_SCHEMA = "avengine_animal_appearance_request_v1"
ANIMAL_APPEARANCE_INSTANCE_REQUEST_SCHEMA = (
    "avengine_animal_appearance_instance_request_v1"
)
ANIMAL_APPEARANCE_BATCH_SCHEMA = "avengine_animal_appearance_batch_v1"
L9_ALGORITHM = "orthogonal_array_l9_3_level_4_factor_v1"
APPEARANCE_AXES = ("size", "body_build", "coat_profile", "life_stage")
CANONICAL_DOMAINS = {
    "size": ("small", "medium", "large"),
    "body_build": ("slim", "standard", "stocky"),
    "life_stage": ("young", "adult", "senior"),
}
BEAGLE_COAT_VALUES = (
    "light_tricolor",
    "standard_tricolor",
    "dark_tricolor",
)
ABYSSINIAN_COAT_VALUES = (
    "light_ruddy",
    "standard_ruddy",
    "dark_ruddy",
)
BORDER_COLLIE_COAT_VALUES = (
    "light_black_white",
    "standard_black_white",
    "dark_black_white",
)
LABRADOR_COAT_VALUES = (
    "light_yellow",
    "standard_yellow",
    "dark_yellow",
)
SHIBA_INU_COAT_VALUES = (
    "light_red",
    "standard_red",
    "dark_red",
)

# Reviewed coat vocabularies are registered by their complete taxonomic and
# profile identity.  A namespaced-looking ``profile_id`` is not registration:
# adding another breed (including cats, horses, or birds) requires an explicit
# entry with that breed's reviewed, exact three-level domain.
COAT_PROFILE_DOMAINS: Mapping[tuple[str, str, str], tuple[str, str, str]] = (
    MappingProxyType(
        {
            (
                "dog",
                "beagle",
                "dog_beagle_tricolor_v1",
            ): BEAGLE_COAT_VALUES,
            (
                "cat",
                "abyssinian",
                "cat_abyssinian_coat_v1",
            ): ABYSSINIAN_COAT_VALUES,
            (
                "dog",
                "border_collie",
                "dog_border_collie_coat_v1",
            ): BORDER_COLLIE_COAT_VALUES,
            (
                "dog",
                "labrador_retriever",
                "dog_labrador_retriever_coat_v1",
            ): LABRADOR_COAT_VALUES,
            (
                "dog",
                "shiba_inu",
                "dog_shiba_inu_coat_v1",
            ): SHIBA_INU_COAT_VALUES,
        }
    )
)
# Realization semantics are registered with the exact profile.  Adding a cat,
# horse, bird, or another dog coat only extends this table; core validation
# never needs breed-specific conditionals.
COAT_PROFILE_REALIZATION_RULES: Mapping[
    tuple[str, str, str], Mapping[str, str]
] = MappingProxyType(
    {
        (
            "dog",
            "beagle",
            "dog_beagle_tricolor_v1",
        ): MappingProxyType(
            {
                "light_level": "light_tricolor",
                "neutral_level": "standard_tricolor",
                "dark_level": "dark_tricolor",
                "baseline_level": "standard_tricolor",
                "preserve_pattern": "tricolor",
            }
        ),
        (
            "cat",
            "abyssinian",
            "cat_abyssinian_coat_v1",
        ): MappingProxyType(
            {
                "light_level": "light_ruddy",
                "neutral_level": "standard_ruddy",
                "dark_level": "dark_ruddy",
                "baseline_level": "standard_ruddy",
                "preserve_pattern": "ticked",
            }
        ),
        (
            "dog",
            "border_collie",
            "dog_border_collie_coat_v1",
        ): MappingProxyType(
            {
                "light_level": "light_black_white",
                "neutral_level": "standard_black_white",
                "dark_level": "dark_black_white",
                "baseline_level": "standard_black_white",
                "preserve_pattern": "black_white",
            }
        ),
        (
            "dog",
            "labrador_retriever",
            "dog_labrador_retriever_coat_v1",
        ): MappingProxyType(
            {
                "light_level": "light_yellow",
                "neutral_level": "standard_yellow",
                "dark_level": "dark_yellow",
                "baseline_level": "standard_yellow",
                "preserve_pattern": "solid_yellow",
            }
        ),
        (
            "dog",
            "shiba_inu",
            "dog_shiba_inu_coat_v1",
        ): MappingProxyType(
            {
                "light_level": "light_red",
                "neutral_level": "standard_red",
                "dark_level": "dark_red",
                "baseline_level": "standard_red",
                "preserve_pattern": "urajiro",
            }
        ),
    }
)

# These are the canonical bounds enforced by the Blender realizer.  Keeping
# the request contract at least as strict prevents a request from validating
# here only to fail later (or from relying on an implementation clamp).
REALIZER_PARAMETER_BOUNDS: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        "scale_ratio": (0.70, 1.30),
        "torso_girth_scale": (0.75, 1.25),
        "luminance_gain": (0.65, 1.35),
        "head_scale": (0.85, 1.20),
    }
)
_OPERATION_BY_AXIS = {
    "size": "uniform_actor_scale_v1",
    "body_build": "semantic_torso_girth_scale_v1",
    "coat_profile": "breed_scoped_coat_luminance_v1",
    "life_stage": "semantic_life_stage_cues_v1",
}

# Standard OA(9, 4, 3, 2): for every two columns, all 3 x 3 ordered level
# pairs occur exactly once.  Request-provided level orders map the integers to
# breed/profile-specific semantic values.
_L9_LEVEL_ROWS: tuple[tuple[int, int, int, int], ...] = (
    (0, 0, 0, 0),
    (0, 1, 1, 1),
    (0, 2, 2, 2),
    (1, 0, 1, 2),
    (1, 1, 2, 0),
    (1, 2, 0, 1),
    (2, 0, 2, 1),
    (2, 1, 0, 2),
    (2, 2, 1, 0),
)


class AppearanceContractError(ValueError):
    """An appearance request or generated batch violates the strict contract."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _schema_path(filename: str) -> Path:
    source = Path(__file__).resolve().parents[3] / "schemas" / filename
    installed = Path(sys.prefix) / "share" / "avengine" / "schemas" / filename
    return source if source.is_file() else installed


def _schema_errors(value: Any, filename: str) -> list[str]:
    schema = load_json(_schema_path(filename))
    validator = Draft202012Validator(schema)
    errors: list[str] = []
    for error in sorted(validator.iter_errors(value), key=lambda item: list(item.path)):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"JSON Schema {location}: {error.message}")
    return errors


def _domain_values(request: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    domains = request["attribute_domains"]
    return {
        "size": tuple(domains["size"]),
        "body_build": tuple(domains["body_build"]),
        "coat_profile": tuple(domains["coat_profile"]["values"]),
        "life_stage": tuple(domains["life_stage"]),
    }


def _semantic_errors(request: Mapping[str, Any]) -> list[str]:
    """Check rules that JSON Schema cannot express without custom code."""

    errors: list[str] = []
    domains = _domain_values(request)
    for axis, expected in CANONICAL_DOMAINS.items():
        if domains[axis] != expected:
            errors.append(
                f"attribute_domains.{axis} must be {list(expected)} in canonical order"
            )

    taxonomy = request["taxonomy"]
    coat = request["attribute_domains"]["coat_profile"]
    scope = coat["scope"]
    for field in ("species", "breed"):
        if scope[field] != taxonomy[field]:
            errors.append(
                f"attribute_domains.coat_profile.scope.{field} must equal "
                f"taxonomy.{field}"
            )

    expected_prefix = f"{taxonomy['species']}_{taxonomy['breed']}_"
    if not coat["profile_id"].startswith(expected_prefix):
        errors.append(
            "attribute_domains.coat_profile.profile_id must be namespaced by "
            f"the exact species/breed prefix {expected_prefix!r}"
        )

    coat_registry_key = (
        taxonomy["species"],
        taxonomy["breed"],
        coat["profile_id"],
    )
    registered_coat_domain = COAT_PROFILE_DOMAINS.get(coat_registry_key)
    if registered_coat_domain is None:
        errors.append(
            "no registered coat profile for exact "
            "(species, breed, profile_id) key "
            f"{coat_registry_key!r}; register a reviewed exact three-level domain"
        )
    elif domains["coat_profile"] != registered_coat_domain:
        errors.append(
            f"{taxonomy['species']}/{taxonomy['breed']} coat_profile values for "
            f"{coat['profile_id']!r} must be "
            f"{list(registered_coat_domain)} in canonical order"
        )

    registered_coat_rule = COAT_PROFILE_REALIZATION_RULES.get(coat_registry_key)
    if registered_coat_domain is not None and registered_coat_rule is None:
        errors.append(
            "registered coat profile lacks reviewed realization role metadata"
        )
    elif registered_coat_domain is not None and registered_coat_rule is not None:
        expected_rule_keys = {
            "light_level",
            "neutral_level",
            "dark_level",
            "baseline_level",
            "preserve_pattern",
        }
        roles_valid = (
            set(registered_coat_rule) == expected_rule_keys
            and all(
                isinstance(item, str) and bool(item)
                for item in registered_coat_rule.values()
            )
        )
        role_levels = (
            {
                registered_coat_rule["light_level"],
                registered_coat_rule["neutral_level"],
                registered_coat_rule["dark_level"],
            }
            if roles_valid
            else set()
        )
        if (
            not roles_valid
            or role_levels != set(registered_coat_domain)
            or registered_coat_rule.get("baseline_level")
            != registered_coat_rule.get("neutral_level")
        ):
            errors.append(
                "registered coat realization metadata must declare exact distinct "
                "light/neutral/dark roles, a neutral baseline, and a non-empty pattern"
            )
            registered_coat_rule = None

    level_order = request["l9_level_order"]
    baseline = request["baseline_attributes"]
    bindings = request["realization_bindings"]
    for axis in APPEARANCE_AXES:
        order = tuple(level_order[axis])
        if len(order) != 3 or set(order) != set(domains[axis]):
            errors.append(
                f"l9_level_order.{axis} must be a permutation of its exact domain"
            )
        elif baseline[axis] != order[0]:
            errors.append(
                f"baseline_attributes.{axis} must equal l9_level_order.{axis}[0]"
            )
        if baseline[axis] not in domains[axis]:
            errors.append(f"baseline_attributes.{axis} is outside its domain")

        parameter_keys = set(bindings[axis]["parameters_by_value"])
        if parameter_keys != set(domains[axis]):
            errors.append(
                f"realization_bindings.{axis}.parameters_by_value keys must equal "
                "its exact domain"
            )
        expected_operation = _OPERATION_BY_AXIS[axis]
        if bindings[axis]["operation_id"] != expected_operation:
            errors.append(
                f"realization_bindings.{axis}.operation_id must be "
                f"{expected_operation!r}"
            )
        for value, parameters in bindings[axis]["parameters_by_value"].items():
            errors.extend(_parameter_errors(axis, value, parameters))

    if not errors:
        errors.extend(
            _profile_parameter_errors(
                request,
                registered_coat_rule=registered_coat_rule,
            )
        )

    if request["source_asset"]["asset_id"] == request["request_id"]:
        errors.append(
            "source_asset.asset_id and request_id must identify different objects"
        )
    return errors


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _semantic_joint_errors(value: Any, owner: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or len(value) != len(set(value))
        or any(not isinstance(item, str) or not item for item in value)
    ):
        return [f"{owner} must be a non-empty unique string list"]
    return []


def _bounded_realizer_parameter_errors(
    value: Any, *, owner: str, parameter: str
) -> list[str]:
    lower, upper = REALIZER_PARAMETER_BOUNDS[parameter]
    if not _finite_number(value) or not lower <= float(value) <= upper:
        return [
            f"{owner}.{parameter} must be a finite number in "
            f"[{lower:.2f}, {upper:.2f}]"
        ]
    return []


def _parameter_errors(axis: str, value: str, parameters: Any) -> list[str]:
    owner = f"realization_bindings.{axis}.parameters_by_value.{value}"
    if not isinstance(parameters, dict):
        return [f"{owner} must be an object"]
    expected_keys = {
        "size": {"scale_ratio"},
        "body_build": {"torso_girth_scale", "semantic_joint_names"},
        "coat_profile": {"luminance_gain", "preserve_pattern"},
        "life_stage": {
            "head_scale",
            "muzzle_gray_mix",
            "muzzle_gray_target",
            "coat_desaturation",
            "semantic_joint_names",
        },
    }[axis]
    if set(parameters) != expected_keys:
        return [
            f"{owner} keys differ: missing={sorted(expected_keys - set(parameters))}, "
            f"extra={sorted(set(parameters) - expected_keys)}"
        ]

    errors: list[str] = []
    if axis == "size":
        errors.extend(
            _bounded_realizer_parameter_errors(
                parameters["scale_ratio"], owner=owner, parameter="scale_ratio"
            )
        )
    elif axis == "body_build":
        errors.extend(
            _bounded_realizer_parameter_errors(
                parameters["torso_girth_scale"],
                owner=owner,
                parameter="torso_girth_scale",
            )
        )
        errors.extend(
            _semantic_joint_errors(
                parameters["semantic_joint_names"],
                f"{owner}.semantic_joint_names",
            )
        )
    elif axis == "coat_profile":
        errors.extend(
            _bounded_realizer_parameter_errors(
                parameters["luminance_gain"],
                owner=owner,
                parameter="luminance_gain",
            )
        )
        if (
            not isinstance(parameters["preserve_pattern"], str)
            or not parameters["preserve_pattern"]
        ):
            errors.append(f"{owner}.preserve_pattern must be a non-empty string")
    else:
        errors.extend(
            _bounded_realizer_parameter_errors(
                parameters["head_scale"], owner=owner, parameter="head_scale"
            )
        )
        for name in ("muzzle_gray_mix", "muzzle_gray_target", "coat_desaturation"):
            if (
                not _finite_number(parameters[name])
                or not 0.0 <= float(parameters[name]) <= 1.0
            ):
                errors.append(f"{owner}.{name} must be a finite number in [0, 1]")
        errors.extend(
            _semantic_joint_errors(
                parameters["semantic_joint_names"],
                f"{owner}.semantic_joint_names",
            )
        )
    return errors


def _profile_parameter_errors(
    request: Mapping[str, Any], *, registered_coat_rule: Mapping[str, str] | None
) -> list[str]:
    """Enforce the reviewed three-level semantics, not merely valid numbers."""

    errors: list[str] = []
    bindings = request["realization_bindings"]
    size = bindings["size"]["parameters_by_value"]
    body = bindings["body_build"]["parameters_by_value"]
    coat = bindings["coat_profile"]["parameters_by_value"]
    life = bindings["life_stage"]["parameters_by_value"]

    size_values = {name: float(size[name]["scale_ratio"]) for name in CANONICAL_DOMAINS["size"]}
    if not size_values["small"] < size_values["medium"] < size_values["large"]:
        errors.append("size scale_ratio levels must satisfy small < medium < large")
    if size_values["medium"] != 1.0:
        errors.append("size medium baseline scale_ratio must be exactly 1.0")

    body_values = {
        name: float(body[name]["torso_girth_scale"])
        for name in CANONICAL_DOMAINS["body_build"]
    }
    if not body_values["slim"] < body_values["standard"] < body_values["stocky"]:
        errors.append(
            "body_build torso_girth_scale levels must satisfy slim < standard < stocky"
        )
    if body_values["standard"] != 1.0:
        errors.append(
            "body_build standard baseline torso_girth_scale must be exactly 1.0"
        )
    body_joint_sets = {
        tuple(body[name]["semantic_joint_names"])
        for name in CANONICAL_DOMAINS["body_build"]
    }
    if len(body_joint_sets) != 1:
        errors.append("body_build levels must use the same semantic_joint_names")

    coat_domain = tuple(request["attribute_domains"]["coat_profile"]["values"])
    if registered_coat_rule is not None:
        light = registered_coat_rule["light_level"]
        standard = registered_coat_rule["neutral_level"]
        dark = registered_coat_rule["dark_level"]
        coat_values = {
            name: float(coat[name]["luminance_gain"]) for name in coat_domain
        }
        if not coat_values[dark] < coat_values[standard] < coat_values[light]:
            errors.append(
                "coat luminance_gain levels must satisfy dark < standard < light"
            )
        if coat_values[standard] != 1.0:
            errors.append(
                "coat standard baseline luminance_gain must be exactly 1.0"
            )
        registered_pattern = registered_coat_rule["preserve_pattern"]
        for name in coat_domain:
            if coat[name]["preserve_pattern"] != registered_pattern:
                errors.append(
                    "realization_bindings.coat_profile.parameters_by_value."
                    f"{name}.preserve_pattern must equal the registered profile "
                    f"value {registered_pattern!r}"
                )

    adult = life["adult"]
    young = life["young"]
    senior = life["senior"]
    if not (
        float(adult["head_scale"]) == 1.0
        and float(adult["muzzle_gray_mix"]) == 0.0
        and float(adult["coat_desaturation"]) == 0.0
    ):
        errors.append(
            "life_stage adult baseline must be neutral: head_scale=1, "
            "muzzle_gray_mix=0, coat_desaturation=0"
        )
    if not (
        float(young["head_scale"]) > float(adult["head_scale"])
        and float(young["muzzle_gray_mix"]) == 0.0
        and float(young["coat_desaturation"]) == 0.0
    ):
        errors.append(
            "life_stage young must enlarge the head while keeping age texture cues neutral"
        )
    if not (
        float(senior["head_scale"]) < float(adult["head_scale"])
        and float(senior["muzzle_gray_mix"]) > 0.0
        and float(senior["coat_desaturation"]) > 0.0
    ):
        errors.append(
            "life_stage senior must reduce head_scale and enable both reviewed age texture cues"
        )
    if len(
        {
            float(young["head_scale"]),
            float(adult["head_scale"]),
            float(senior["head_scale"]),
        }
    ) != 3:
        errors.append("life_stage head_scale levels must be distinct")
    if len(
        {
            float(life[name]["muzzle_gray_target"])
            for name in CANONICAL_DOMAINS["life_stage"]
        }
    ) != 1:
        errors.append("life_stage levels must share one muzzle_gray_target")
    life_joint_sets = {
        tuple(life[name]["semantic_joint_names"])
        for name in CANONICAL_DOMAINS["life_stage"]
    }
    if len(life_joint_sets) != 1:
        errors.append("life_stage levels must use the same semantic_joint_names")

    expected_baseline = {
        "size": "medium",
        "body_build": "standard",
        "coat_profile": (
            registered_coat_rule["baseline_level"]
            if registered_coat_rule is not None
            else request["baseline_attributes"]["coat_profile"]
        ),
        "life_stage": "adult",
    }
    if request["baseline_attributes"] != expected_baseline:
        errors.append(
            "baseline_attributes must select the neutral medium/standard/"
            "standard_tricolor/adult levels"
        )
    return errors


def validate_appearance_request(value: Any) -> dict[str, Any]:
    """Return a detached request after schema and semantic validation."""

    errors = _schema_errors(value, "animal_appearance_request_v1.schema.json")
    if not errors and isinstance(value, dict):
        errors.extend(_semantic_errors(value))
    if errors:
        raise AppearanceContractError(errors)
    return deepcopy(value)


def _file_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise AppearanceContractError([f"source is not a regular file: {resolved}"])
    return {
        "path": str(resolved),
        "byte_size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _balance_audit(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    level_counts = {
        axis: dict(sorted(Counter(row[axis] for row in rows).items()))
        for axis in APPEARANCE_AXES
    }
    pair_counts: dict[str, dict[str, int]] = {}
    pairwise_orthogonal = True
    for left_index, left in enumerate(APPEARANCE_AXES):
        for right in APPEARANCE_AXES[left_index + 1 :]:
            key = f"{left}__{right}"
            counts = Counter(f"{row[left]}__{row[right]}" for row in rows)
            pair_counts[key] = dict(sorted(counts.items()))
            if len(counts) != 9 or set(counts.values()) != {1}:
                pairwise_orthogonal = False

    every_level_three_times = all(
        len(counts) == 3 and set(counts.values()) == {3}
        for counts in level_counts.values()
    )
    return {
        "every_level_three_times": every_level_three_times,
        "level_counts": level_counts,
        "pair_counts": pair_counts,
        "pairwise_orthogonal": pairwise_orthogonal,
    }


def _instance_request(
    request: Mapping[str, Any],
    *,
    request_file_sha256: str,
    source_file_sha256: str,
    ordinal: int,
    levels: tuple[int, int, int, int],
) -> dict[str, Any]:
    level_order = request["l9_level_order"]
    attributes = {
        axis: level_order[axis][level]
        for axis, level in zip(APPEARANCE_AXES, levels, strict=True)
    }
    operations = []
    for axis in APPEARANCE_AXES:
        binding = request["realization_bindings"][axis]
        selected = attributes[axis]
        operations.append(
            {
                "attribute": axis,
                "operation_id": binding["operation_id"],
                "parameters": deepcopy(binding["parameters_by_value"][selected]),
                "selected_value": selected,
            }
        )

    core: dict[str, Any] = {
        "schema": ANIMAL_APPEARANCE_INSTANCE_REQUEST_SCHEMA,
        "parent_request_id": request["request_id"],
        "parent_request_file_sha256": request_file_sha256,
        "source_asset_id": request["source_asset"]["asset_id"],
        "source_asset_sha256": source_file_sha256,
        "ordinal": ordinal,
        "l9_levels": {
            axis: level for axis, level in zip(APPEARANCE_AXES, levels, strict=True)
        },
        "taxonomy": deepcopy(request["taxonomy"]),
        "coat_profile_id": request["attribute_domains"]["coat_profile"]["profile_id"],
        "attributes": attributes,
        "realization_operations": operations,
        "design_role": "l9_combination_point_not_ofat",
        "state_classification": "research_candidate",
    }
    digest = canonical_json_sha256(core)
    core["instance_request_id"] = (
        f"{request['request_id']}_l9_{ordinal:02d}_{digest[:12]}"
    )
    core["request_sha256"] = digest
    return core


def _instance_schema_errors(value: Any) -> list[str]:
    schema = load_json(_schema_path("animal_appearance_batch_v1.schema.json"))
    instance_schema = {
        **schema["$defs"]["instanceRequest"],
        "$defs": schema["$defs"],
    }
    validator = Draft202012Validator(instance_schema)
    errors: list[str] = []
    for error in sorted(validator.iter_errors(value), key=lambda item: list(item.path)):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"JSON Schema {location}: {error.message}")
    return errors


def verify_instance_request_integrity(value: Any) -> None:
    """Fail if an emitted instance request is malformed or was edited."""

    errors = _instance_schema_errors(value)
    if errors:
        raise AppearanceContractError(errors)
    assert isinstance(value, dict)
    core = {
        key: deepcopy(item)
        for key, item in value.items()
        if key not in {"instance_request_id", "request_sha256"}
    }
    declared = value["request_sha256"]
    actual = canonical_json_sha256(core)
    if declared != actual:
        raise AppearanceContractError(
            ["request_sha256 does not authenticate the instance request content"]
        )
    expected_suffix = f"_{value['ordinal']:02d}_{declared[:12]}"
    if not value["instance_request_id"].endswith(expected_suffix):
        raise AppearanceContractError(
            ["instance_request_id does not match ordinal/request_sha256"]
        )


def build_l9_batch(
    request: Any,
    *,
    request_file: str | Path,
    source_file: str | Path,
) -> dict[str, Any]:
    """Build a hash-bound, pairwise-orthogonal nine-request batch."""

    validated = validate_appearance_request(request)
    request_record = _file_record(request_file)
    request_from_file = validate_appearance_request(load_json(request_file))
    if canonical_json_sha256(request_from_file) != canonical_json_sha256(validated):
        raise AppearanceContractError(
            ["request object does not match the exact request_file content"]
        )
    source_record = _file_record(source_file)
    expected_source_sha256 = validated["source_asset"]["expected_sha256"]
    if source_record["sha256"] != expected_source_sha256:
        raise AppearanceContractError(
            [
                "source file SHA-256 mismatch: "
                f"expected {expected_source_sha256}, measured {source_record['sha256']}"
            ]
        )

    requests = [
        _instance_request(
            validated,
            request_file_sha256=request_record["sha256"],
            source_file_sha256=source_record["sha256"],
            ordinal=index + 1,
            levels=levels,
        )
        for index, levels in enumerate(_L9_LEVEL_ROWS)
    ]
    rows = [item["attributes"] for item in requests]
    audit = _balance_audit(rows)
    if not audit["every_level_three_times"] or not audit["pairwise_orthogonal"]:
        raise AppearanceContractError(["internal L9 balance audit failed closed"])

    core: dict[str, Any] = {
        "schema": ANIMAL_APPEARANCE_BATCH_SCHEMA,
        "algorithm": L9_ALGORITHM,
        "parent_request": {
            "request_id": validated["request_id"],
            **request_record,
            "canonical_content_sha256": canonical_json_sha256(validated),
        },
        "source_asset": {
            "asset_id": validated["source_asset"]["asset_id"],
            **source_record,
        },
        "taxonomy": deepcopy(validated["taxonomy"]),
        "coat_profile": deepcopy(validated["attribute_domains"]["coat_profile"]),
        "axis_order": list(APPEARANCE_AXES),
        "level_order": deepcopy(validated["l9_level_order"]),
        "l9_level_rows": [list(row) for row in _L9_LEVEL_ROWS],
        "balance_audit": audit,
        "ofat_validation": {
            "strategy": validated["ofat_validation"]["strategy"],
            "status": "not_run",
            "required_before_formal_promotion": True,
            "l9_substitution_allowed": False,
        },
        "state_classification": "research_candidate",
        "requests": requests,
    }
    digest = canonical_json_sha256(core)
    core["batch_content_sha256"] = digest
    core["batch_id"] = f"{validated['request_id']}_batch_{digest[:12]}"
    validate_l9_batch(core)
    return core


def generate_l9_batch(
    request_file: str | Path, source_file: str | Path
) -> dict[str, Any]:
    """Load an immutable request file and generate its L9 batch in memory."""

    request_path = Path(request_file).resolve(strict=True)
    return build_l9_batch(
        load_json(request_path),
        request_file=request_path,
        source_file=source_file,
    )


def validate_l9_batch(value: Any) -> None:
    """Validate structure, hashes, balance, and the explicit OFAT boundary."""

    errors = _schema_errors(value, "animal_appearance_batch_v1.schema.json")
    if errors:
        raise AppearanceContractError(errors)
    assert isinstance(value, dict)

    if value["l9_level_rows"] != [list(row) for row in _L9_LEVEL_ROWS]:
        raise AppearanceContractError(["l9_level_rows is not the canonical OA matrix"])

    parent_record = value["parent_request"]
    source_record = value["source_asset"]
    try:
        measured_parent = _file_record(parent_record["path"])
        measured_source = _file_record(source_record["path"])
    except (OSError, AppearanceContractError) as error:
        raise AppearanceContractError(
            [f"batch file closure could not be verified: {error}"]
        ) from error
    for name, declared, measured in (
        ("parent_request", parent_record, measured_parent),
        ("source_asset", source_record, measured_source),
    ):
        for field in ("byte_size", "sha256"):
            if declared[field] != measured[field]:
                raise AppearanceContractError(
                    [f"{name}.{field} does not match the file at its declared path"]
                )

    parent_request = validate_appearance_request(load_json(parent_record["path"]))
    if parent_record["request_id"] != parent_request["request_id"]:
        raise AppearanceContractError(
            ["parent_request.request_id does not match the request file"]
        )
    if parent_record["canonical_content_sha256"] != canonical_json_sha256(
        parent_request
    ):
        raise AppearanceContractError(
            ["parent_request.canonical_content_sha256 does not match the request file"]
        )
    if source_record["asset_id"] != parent_request["source_asset"]["asset_id"]:
        raise AppearanceContractError(
            ["source_asset.asset_id does not match the parent request"]
        )
    if source_record["sha256"] != parent_request["source_asset"]["expected_sha256"]:
        raise AppearanceContractError(
            ["source_asset.sha256 does not match the parent request"]
        )
    if value["taxonomy"] != parent_request["taxonomy"]:
        raise AppearanceContractError(
            ["batch taxonomy does not match the parent request"]
        )
    if value["coat_profile"] != parent_request["attribute_domains"]["coat_profile"]:
        raise AppearanceContractError(
            ["batch coat_profile does not match the parent request"]
        )
    if value["level_order"] != parent_request["l9_level_order"]:
        raise AppearanceContractError(
            ["batch level_order does not match the parent request"]
        )

    requests = value["requests"]
    for index, request in enumerate(requests):
        verify_instance_request_integrity(request)
        ordinal = index + 1
        expected_levels = {
            axis: level
            for axis, level in zip(APPEARANCE_AXES, _L9_LEVEL_ROWS[index], strict=True)
        }
        expected_attributes = {
            axis: value["level_order"][axis][expected_levels[axis]]
            for axis in APPEARANCE_AXES
        }
        if request["ordinal"] != ordinal or request["l9_levels"] != expected_levels:
            raise AppearanceContractError(
                [f"requests[{index}] does not occupy canonical L9 ordinal {ordinal}"]
            )
        if request["attributes"] != expected_attributes:
            raise AppearanceContractError(
                [f"requests[{index}].attributes does not match its L9 levels"]
            )
        expected_links = {
            "parent_request_id": parent_record["request_id"],
            "parent_request_file_sha256": parent_record["sha256"],
            "source_asset_id": source_record["asset_id"],
            "source_asset_sha256": source_record["sha256"],
            "taxonomy": value["taxonomy"],
            "coat_profile_id": value["coat_profile"]["profile_id"],
        }
        for field, expected in expected_links.items():
            if request[field] != expected:
                raise AppearanceContractError(
                    [f"requests[{index}].{field} does not match the batch"]
                )
        operations = request["realization_operations"]
        if [operation["attribute"] for operation in operations] != list(
            APPEARANCE_AXES
        ):
            raise AppearanceContractError(
                [f"requests[{index}] realization operations are not axis-complete"]
            )
        for operation in operations:
            axis = operation["attribute"]
            selected = expected_attributes[axis]
            binding = parent_request["realization_bindings"][axis]
            if (
                operation["selected_value"] != selected
                or operation["operation_id"] != binding["operation_id"]
                or operation["parameters"] != binding["parameters_by_value"][selected]
            ):
                raise AppearanceContractError(
                    [
                        f"requests[{index}] realization operation for {axis} "
                        "does not match the parent request"
                    ]
                )
    audit = _balance_audit([request["attributes"] for request in requests])
    if audit != value["balance_audit"]:
        raise AppearanceContractError(
            ["balance_audit does not match the emitted instance requests"]
        )
    if not audit["every_level_three_times"] or not audit["pairwise_orthogonal"]:
        raise AppearanceContractError(["batch is not an OA L9(3^4) design"])

    core = {
        key: deepcopy(item)
        for key, item in value.items()
        if key not in {"batch_id", "batch_content_sha256"}
    }
    actual = canonical_json_sha256(core)
    if value["batch_content_sha256"] != actual:
        raise AppearanceContractError(
            ["batch_content_sha256 does not authenticate batch content"]
        )
    if not value["batch_id"].endswith(f"_{actual[:12]}"):
        raise AppearanceContractError(["batch_id does not match batch_content_sha256"])


def write_l9_batch_exclusive(path: str | Path, value: Any) -> Path:
    """Validate then create one JSON output without ever replacing a file."""

    validate_l9_batch(value)
    output = Path(os.path.abspath(Path(path)))
    if output.is_symlink():
        raise FileExistsError(f"refusing symbolic-link L9 output: {output}")
    cursor = Path(output.anchor)
    for part in output.parent.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise AppearanceContractError(
                ["L9 output parent must not contain a symbolic link"]
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    cursor = Path(output.anchor)
    for part in output.parent.parts[1:]:
        cursor /= part
        if cursor.is_symlink() or not cursor.is_dir():
            raise AppearanceContractError(
                ["L9 output parent must be a real directory tree"]
            )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(output, flags, 0o644)
    succeeded = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        succeeded = True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not succeeded:
            try:
                output.unlink()
            except OSError:
                pass
    return output
