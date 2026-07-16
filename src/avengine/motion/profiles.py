"""Strict profiles for body-plan-aware motion adapters.

Profiles are intentionally target-template specific.  They make semantic
mapping, action-family choice, attribute compatibility and unsupported body
plans explicit instead of silently applying a canine gait to every animal.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from avengine.motion.math import (
    QuaternionXYZW,
    canonical_quaternion_xyzw,
)
from avengine.motion.qa import (
    ChainMotionThresholds,
    ChainSymmetryThreshold,
    GroupExcursionRatioThreshold,
    JointMotionThresholds,
    MotionQAContract,
    SemanticChainGroup,
)


MOTION_RETARGET_PROFILE_SCHEMA = "avengine_motion_retarget_profile_v1"


class MotionProfileError(ValueError):
    """A motion profile is ambiguous, incomplete, or unsupported."""


@dataclass(frozen=True)
class AdapterCapability:
    adapter_id: str
    body_plan_family: str
    production_supported: bool
    contact_model: str
    notes: str


ADAPTER_CAPABILITIES: Mapping[str, AdapterCapability] = {
    "quadruped_mammal_locomotion_v1": AdapterCapability(
        adapter_id="quadruped_mammal_locomotion_v1",
        body_plan_family="quadruped_mammal",
        production_supported=True,
        contact_model="four_paw_contact",
        notes="Rest-aware rotation transfer with quadruped gait/contact QA.",
    ),
    "avian_biped_locomotion_v1": AdapterCapability(
        adapter_id="avian_biped_locomotion_v1",
        body_plan_family="avian_biped",
        production_supported=False,
        contact_model="two_foot_contact",
        notes="Reserved for a separately audited bird walking/hopping adapter.",
    ),
    "avian_flight_v1": AdapterCapability(
        adapter_id="avian_flight_v1",
        body_plan_family="avian_flight",
        production_supported=False,
        contact_model="no_ground_contact",
        notes="Reserved for wing-cycle, aerodynamic-axis and flight-path QA.",
    ),
    "fish_swim_v1": AdapterCapability(
        adapter_id="fish_swim_v1",
        body_plan_family="fish_swim",
        production_supported=False,
        contact_model="no_ground_contact",
        notes="Reserved for axial-wave and fin-phase QA.",
    ),
}


def _object(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise MotionProfileError(f"{owner} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], *, owner: str, expected: set[str]) -> None:
    actual = set(value)
    if actual != expected:
        raise MotionProfileError(
            f"{owner} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _only_keys(value: Mapping[str, Any], *, owner: str, allowed: set[str]) -> None:
    extra = set(value) - allowed
    if extra:
        raise MotionProfileError(f"{owner} has unsupported keys: {sorted(extra)}")


def _string(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise MotionProfileError(f"{owner} must be a non-empty string")
    return value


def _string_tuple(
    value: Any, *, owner: str, minimum: int = 1, maximum: int | None = None
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MotionProfileError(f"{owner} must be an array of unique strings")
    result = tuple(_string(item, owner=f"{owner}[]") for item in value)
    if len(result) < minimum or (maximum is not None and len(result) > maximum):
        upper = "unbounded" if maximum is None else str(maximum)
        raise MotionProfileError(
            f"{owner} must contain between {minimum} and {upper} values"
        )
    if len(set(result)) != len(result):
        raise MotionProfileError(f"{owner} must contain unique values")
    return result


@dataclass(frozen=True)
class AttributeDomain:
    size: tuple[str, ...]
    body_build: tuple[str, ...]
    coat_profile_id: str
    coat_values: tuple[str, ...]
    life_stage: tuple[str, ...]


@dataclass(frozen=True)
class SemanticChain:
    chain_id: str
    chain_kind: str
    side: str
    semantic_joint_ids: tuple[str, ...]
    end_effector_role: str
    target_end_effector_joint_id: str


@dataclass(frozen=True)
class MotionQACoordinateFrame:
    forward_axis: str
    lateral_axis: str
    vertical_axis: str


@dataclass(frozen=True)
class JointMapping:
    semantic_joint_id: str
    source_joint_id: str
    target_joint_id: str


@dataclass(frozen=True)
class ActionMapping:
    semantic_action_id: str
    source_action_hint: str
    output_action_name: str


@dataclass(frozen=True)
class MotionRetargetProfile:
    profile_id: str
    adapter_id: str
    body_plan_id: str
    motion_family_id: str
    source_skeleton_id: str
    target_template_id: str
    solver_id: str
    motion_basis_xyzw: QuaternionXYZW
    motion_amplitude: float
    output_sample_rate_hz: int
    time_mapping: str
    root_joint_semantic_id: str
    root_rotation_policy: str
    root_translation_policy: str
    unmapped_target_joint_policy: str
    semantic_chains: tuple[SemanticChain, ...]
    joint_mappings: tuple[JointMapping, ...]
    actions: tuple[ActionMapping, ...]
    attribute_domain: AttributeDomain
    qa_semantic_action_id: str
    qa_coordinate_frame: MotionQACoordinateFrame
    qa_contract: MotionQAContract

    @property
    def capability(self) -> AdapterCapability:
        return ADAPTER_CAPABILITIES[self.adapter_id]


def _parse_attribute_domain(value: Any) -> AttributeDomain:
    item = _object(value, owner="attribute_domain")
    _exact_keys(
        item,
        owner="attribute_domain",
        expected={
            "size",
            "body_build",
            "coat_profile_id",
            "coat_values",
            "life_stage",
        },
    )
    size = _string_tuple(item["size"], owner="attribute_domain.size", maximum=3)
    body_build = _string_tuple(
        item["body_build"], owner="attribute_domain.body_build", maximum=3
    )
    coat_values = _string_tuple(
        item["coat_values"], owner="attribute_domain.coat_values", maximum=3
    )
    life_stage = _string_tuple(
        item["life_stage"], owner="attribute_domain.life_stage", maximum=3
    )
    if size != ("small", "medium", "large"):
        raise MotionProfileError(
            "attribute_domain.size must be small/medium/large in canonical order"
        )
    if body_build != ("slim", "standard", "stocky"):
        raise MotionProfileError(
            "attribute_domain.body_build must be slim/standard/stocky"
        )
    if life_stage != ("young", "adult", "senior"):
        raise MotionProfileError(
            "attribute_domain.life_stage must be young/adult/senior"
        )
    if len(coat_values) != 3:
        raise MotionProfileError(
            "attribute_domain.coat_values must declare three breed-specific values"
        )
    return AttributeDomain(
        size=size,
        body_build=body_build,
        coat_profile_id=_string(
            item["coat_profile_id"], owner="attribute_domain.coat_profile_id"
        ),
        coat_values=coat_values,
        life_stage=life_stage,
    )


def _optional_nonnegative_number(value: Any, *, owner: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise MotionProfileError(f"{owner} must be a finite non-negative number")
    return float(value)


_CHAIN_THRESHOLD_FIELDS = {
    "minimum_forward_excursion_m",
    "maximum_forward_excursion_m",
    "minimum_vertical_excursion_m",
    "maximum_vertical_excursion_m",
    "maximum_lateral_excursion_m",
    "minimum_forward_excursion_normalized",
    "maximum_forward_excursion_normalized",
    "minimum_vertical_excursion_normalized",
    "maximum_vertical_excursion_normalized",
    "maximum_lateral_excursion_normalized",
    "maximum_lateral_to_forward_ratio",
}
_JOINT_THRESHOLD_FIELDS = {
    "minimum_angular_excursion_degrees",
    "maximum_angular_excursion_degrees",
    "maximum_angular_speed_degrees_per_second",
}


def _parse_chain_threshold(value: Any, *, owner: str) -> ChainMotionThresholds:
    item = _object(value, owner=owner)
    _only_keys(item, owner=owner, allowed=_CHAIN_THRESHOLD_FIELDS)
    if not item:
        raise MotionProfileError(f"{owner} must declare at least one threshold")
    return ChainMotionThresholds(
        **{
            key: _optional_nonnegative_number(raw, owner=f"{owner}.{key}")
            for key, raw in item.items()
        }
    )


def _parse_joint_threshold(value: Any, *, owner: str) -> JointMotionThresholds:
    item = _object(value, owner=owner)
    _only_keys(item, owner=owner, allowed=_JOINT_THRESHOLD_FIELDS)
    if not item:
        raise MotionProfileError(f"{owner} must declare at least one threshold")
    return JointMotionThresholds(
        **{
            key: _optional_nonnegative_number(raw, owner=f"{owner}.{key}")
            for key, raw in item.items()
        }
    )


def _parse_qa_contract(
    value: Any,
    *,
    chains: tuple[SemanticChain, ...],
    actions: tuple[ActionMapping, ...],
    sample_rate_hz: int,
) -> tuple[str, MotionQACoordinateFrame, MotionQAContract]:
    owner = "qa_contract"
    item = _object(value, owner=owner)
    _exact_keys(
        item,
        owner=owner,
        expected={
            "semantic_action_id",
            "coordinate_frame",
            "sample_rate_hz",
            "minimum_sample_count",
            "cyclic",
            "required_chain_ids",
            "chain_thresholds",
            "joint_thresholds_by_chain",
            "chain_groups",
            "group_ratio_thresholds",
            "symmetry_thresholds",
        },
    )
    semantic_action_id = _string(
        item["semantic_action_id"], owner=f"{owner}.semantic_action_id"
    )
    if semantic_action_id not in {action.semantic_action_id for action in actions}:
        raise MotionProfileError(
            "qa_contract.semantic_action_id must name one declared action"
        )

    frame_value = _object(item["coordinate_frame"], owner=f"{owner}.coordinate_frame")
    _exact_keys(
        frame_value,
        owner=f"{owner}.coordinate_frame",
        expected={"forward_axis", "lateral_axis", "vertical_axis"},
    )
    axes = tuple(
        _string(frame_value[key], owner=f"{owner}.coordinate_frame.{key}")
        for key in ("forward_axis", "lateral_axis", "vertical_axis")
    )
    valid_axes = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}
    if (
        any(axis not in valid_axes for axis in axes)
        or len({axis[-1] for axis in axes}) != 3
    ):
        raise MotionProfileError(
            "qa_contract coordinate axes must be signed, orthogonal X/Y/Z axes"
        )
    coordinate_frame = MotionQACoordinateFrame(*axes)

    qa_sample_rate_hz = item["sample_rate_hz"]
    if (
        isinstance(qa_sample_rate_hz, bool)
        or not isinstance(qa_sample_rate_hz, int)
        or qa_sample_rate_hz <= 0
        or sample_rate_hz % qa_sample_rate_hz
        or 48_000 % qa_sample_rate_hz
    ):
        raise MotionProfileError(
            "qa_contract.sample_rate_hz must divide the retarget and 48 kHz rates"
        )

    minimum_sample_count = item["minimum_sample_count"]
    if (
        isinstance(minimum_sample_count, bool)
        or not isinstance(minimum_sample_count, int)
        or minimum_sample_count < 3
    ):
        raise MotionProfileError(
            "qa_contract.minimum_sample_count must be an integer of at least three"
        )
    if item["cyclic"] is not True:
        raise MotionProfileError("qa_contract.cyclic must explicitly be true")

    required_chain_ids = _string_tuple(
        item["required_chain_ids"], owner=f"{owner}.required_chain_ids"
    )
    locomotion_chain_ids = tuple(
        chain.chain_id for chain in chains if chain.chain_kind == "locomotion_limb"
    )
    if required_chain_ids != locomotion_chain_ids:
        raise MotionProfileError(
            "qa_contract.required_chain_ids must exactly follow locomotion chains"
        )
    chain_by_id = {chain.chain_id: chain for chain in chains}

    raw_chain_thresholds = _object(
        item["chain_thresholds"], owner=f"{owner}.chain_thresholds"
    )
    if set(raw_chain_thresholds) != set(required_chain_ids):
        raise MotionProfileError(
            "qa_contract.chain_thresholds must cover every required chain"
        )
    chain_thresholds = {
        chain_id: _parse_chain_threshold(
            raw_chain_thresholds[chain_id],
            owner=f"{owner}.chain_thresholds.{chain_id}",
        )
        for chain_id in required_chain_ids
    }

    raw_joint_thresholds = _object(
        item["joint_thresholds_by_chain"],
        owner=f"{owner}.joint_thresholds_by_chain",
    )
    if set(raw_joint_thresholds) != set(required_chain_ids):
        raise MotionProfileError(
            "qa_contract.joint_thresholds_by_chain must cover every required chain"
        )
    joint_thresholds: dict[str, dict[str, JointMotionThresholds]] = {}
    for chain_id in required_chain_ids:
        raw_by_joint = _object(
            raw_joint_thresholds[chain_id],
            owner=f"{owner}.joint_thresholds_by_chain.{chain_id}",
        )
        semantic_joint_ids = chain_by_id[chain_id].semantic_joint_ids
        if set(raw_by_joint) != set(semantic_joint_ids):
            raise MotionProfileError(
                f"qa_contract joint thresholds for {chain_id!r} must cover its "
                "semantic joints"
            )
        joint_thresholds[chain_id] = {
            joint_id: _parse_joint_threshold(
                raw_by_joint[joint_id],
                owner=(f"{owner}.joint_thresholds_by_chain.{chain_id}.{joint_id}"),
            )
            for joint_id in semantic_joint_ids
        }

    raw_groups = item["chain_groups"]
    if not isinstance(raw_groups, list) or not raw_groups:
        raise MotionProfileError("qa_contract.chain_groups must be a non-empty array")
    groups: list[SemanticChainGroup] = []
    for index, raw_group in enumerate(raw_groups):
        group = _object(raw_group, owner=f"{owner}.chain_groups[{index}]")
        _exact_keys(
            group,
            owner=f"{owner}.chain_groups[{index}]",
            expected={"group_id", "chain_ids"},
        )
        groups.append(
            SemanticChainGroup(
                group_id=_string(
                    group["group_id"], owner=f"{owner}.chain_groups[{index}].group_id"
                ),
                chain_ids=_string_tuple(
                    group["chain_ids"],
                    owner=f"{owner}.chain_groups[{index}].chain_ids",
                ),
            )
        )

    raw_ratios = item["group_ratio_thresholds"]
    if not isinstance(raw_ratios, list) or not raw_ratios:
        raise MotionProfileError(
            "qa_contract.group_ratio_thresholds must be a non-empty array"
        )
    ratios: list[GroupExcursionRatioThreshold] = []
    for index, raw_ratio in enumerate(raw_ratios):
        ratio_owner = f"{owner}.group_ratio_thresholds[{index}]"
        ratio = _object(raw_ratio, owner=ratio_owner)
        _exact_keys(
            ratio,
            owner=ratio_owner,
            expected={
                "ratio_id",
                "numerator_group_id",
                "numerator_axis",
                "denominator_group_id",
                "denominator_axis",
                "metric_space",
                "minimum_ratio",
                "maximum_ratio",
            },
        )
        ratios.append(
            GroupExcursionRatioThreshold(
                ratio_id=_string(ratio["ratio_id"], owner=f"{ratio_owner}.ratio_id"),
                numerator_group_id=_string(
                    ratio["numerator_group_id"],
                    owner=f"{ratio_owner}.numerator_group_id",
                ),
                numerator_axis=_string(
                    ratio["numerator_axis"], owner=f"{ratio_owner}.numerator_axis"
                ),
                denominator_group_id=_string(
                    ratio["denominator_group_id"],
                    owner=f"{ratio_owner}.denominator_group_id",
                ),
                denominator_axis=_string(
                    ratio["denominator_axis"],
                    owner=f"{ratio_owner}.denominator_axis",
                ),
                metric_space=_string(
                    ratio["metric_space"], owner=f"{ratio_owner}.metric_space"
                ),
                minimum_ratio=_optional_nonnegative_number(
                    ratio["minimum_ratio"], owner=f"{ratio_owner}.minimum_ratio"
                ),
                maximum_ratio=_optional_nonnegative_number(
                    ratio["maximum_ratio"], owner=f"{ratio_owner}.maximum_ratio"
                ),
            )
        )

    raw_symmetries = item["symmetry_thresholds"]
    if not isinstance(raw_symmetries, list) or not raw_symmetries:
        raise MotionProfileError(
            "qa_contract.symmetry_thresholds must be a non-empty array"
        )
    symmetries: list[ChainSymmetryThreshold] = []
    for index, raw_symmetry in enumerate(raw_symmetries):
        symmetry_owner = f"{owner}.symmetry_thresholds[{index}]"
        symmetry = _object(raw_symmetry, owner=symmetry_owner)
        _exact_keys(
            symmetry,
            owner=symmetry_owner,
            expected={
                "symmetry_id",
                "first_chain_id",
                "second_chain_id",
                "maximum_relative_difference",
                "axes",
                "metric_space",
            },
        )
        maximum = _optional_nonnegative_number(
            symmetry["maximum_relative_difference"],
            owner=f"{symmetry_owner}.maximum_relative_difference",
        )
        assert maximum is not None
        symmetries.append(
            ChainSymmetryThreshold(
                symmetry_id=_string(
                    symmetry["symmetry_id"], owner=f"{symmetry_owner}.symmetry_id"
                ),
                first_chain_id=_string(
                    symmetry["first_chain_id"],
                    owner=f"{symmetry_owner}.first_chain_id",
                ),
                second_chain_id=_string(
                    symmetry["second_chain_id"],
                    owner=f"{symmetry_owner}.second_chain_id",
                ),
                maximum_relative_difference=maximum,
                axes=_string_tuple(symmetry["axes"], owner=f"{symmetry_owner}.axes"),
                metric_space=_string(
                    symmetry["metric_space"],
                    owner=f"{symmetry_owner}.metric_space",
                ),
            )
        )

    contract = MotionQAContract(
        required_chain_ids=required_chain_ids,
        required_joint_ids_by_chain={
            chain_id: chain_by_id[chain_id].semantic_joint_ids
            for chain_id in required_chain_ids
        },
        sample_rate_hz=float(qa_sample_rate_hz),
        cyclic=True,
        minimum_sample_count=minimum_sample_count,
        chain_groups=tuple(groups),
        chain_thresholds=chain_thresholds,
        joint_thresholds_by_chain=joint_thresholds,
        group_ratio_thresholds=tuple(ratios),
        symmetry_thresholds=tuple(symmetries),
    )
    return semantic_action_id, coordinate_frame, contract


def _parse_profile(value: Any) -> MotionRetargetProfile:
    root = _object(value, owner="motion profile")
    _exact_keys(
        root,
        owner="motion profile",
        expected={
            "schema",
            "profile_id",
            "adapter_id",
            "body_plan_id",
            "motion_family_id",
            "source_skeleton_id",
            "target_template_id",
            "solver",
            "semantic_chains",
            "joint_mappings",
            "actions",
            "attribute_domain",
            "qa_contract",
        },
    )
    if root["schema"] != MOTION_RETARGET_PROFILE_SCHEMA:
        raise MotionProfileError(
            f"motion profile schema must equal {MOTION_RETARGET_PROFILE_SCHEMA}"
        )
    adapter_id = _string(root["adapter_id"], owner="adapter_id")
    if adapter_id not in ADAPTER_CAPABILITIES:
        raise MotionProfileError(f"unknown body-plan adapter: {adapter_id}")
    capability = ADAPTER_CAPABILITIES[adapter_id]
    if not capability.production_supported:
        raise MotionProfileError(
            f"body-plan adapter {adapter_id} is unavailable: {capability.notes}"
        )

    solver = _object(root["solver"], owner="solver")
    _exact_keys(
        solver,
        owner="solver",
        expected={
            "solver_id",
            "motion_basis_xyzw",
            "motion_amplitude",
            "output_sample_rate_hz",
            "time_mapping",
            "root_joint_semantic_id",
            "root_rotation_policy",
            "root_translation_policy",
            "unmapped_target_joint_policy",
        },
    )
    solver_id = _string(solver["solver_id"], owner="solver.solver_id")
    if solver_id != "world_left_delta_v2":
        raise MotionProfileError("solver.solver_id must equal world_left_delta_v2")
    amplitude = solver["motion_amplitude"]
    if (
        isinstance(amplitude, bool)
        or not isinstance(amplitude, (int, float))
        or not 0.0 <= float(amplitude) <= 1.0
    ):
        raise MotionProfileError("solver.motion_amplitude must be in [0, 1]")
    output_sample_rate_hz = solver["output_sample_rate_hz"]
    if (
        isinstance(output_sample_rate_hz, bool)
        or not isinstance(output_sample_rate_hz, int)
        or output_sample_rate_hz <= 0
        or 48_000 % output_sample_rate_hz
    ):
        raise MotionProfileError(
            "solver.output_sample_rate_hz must be a positive divisor of 48000"
        )
    time_mapping = _string(solver["time_mapping"], owner="solver.time_mapping")
    if time_mapping != "preserve_source_seconds":
        raise MotionProfileError(
            "solver.time_mapping must equal preserve_source_seconds"
        )
    unmapped_policy = _string(
        solver["unmapped_target_joint_policy"],
        owner="solver.unmapped_target_joint_policy",
    )
    if unmapped_policy != "target_rest_local":
        raise MotionProfileError(
            "unmapped target joints must explicitly use target_rest_local"
        )
    root_semantic_id = _string(
        solver["root_joint_semantic_id"], owner="solver.root_joint_semantic_id"
    )
    root_rotation_policy = _string(
        solver["root_rotation_policy"], owner="solver.root_rotation_policy"
    )
    root_translation_policy = _string(
        solver["root_translation_policy"], owner="solver.root_translation_policy"
    )
    if root_rotation_policy not in {"target_rest", "retarget_world_delta"}:
        raise MotionProfileError(
            "solver.root_rotation_policy must be target_rest/retarget_world_delta"
        )
    if root_translation_policy != "target_rest":
        raise MotionProfileError(
            "solver.root_translation_policy currently supports only target_rest"
        )

    raw_chains = root["semantic_chains"]
    if not isinstance(raw_chains, list) or not raw_chains:
        raise MotionProfileError("semantic_chains must be a non-empty array")
    chains: list[SemanticChain] = []
    for index, raw_chain in enumerate(raw_chains):
        item = _object(raw_chain, owner=f"semantic_chains[{index}]")
        _exact_keys(
            item,
            owner=f"semantic_chains[{index}]",
            expected={
                "chain_id",
                "chain_kind",
                "side",
                "semantic_joint_ids",
                "end_effector_role",
                "target_end_effector_joint_id",
            },
        )
        side = _string(item["side"], owner=f"semantic_chains[{index}].side")
        if side not in {"center", "left", "right"}:
            raise MotionProfileError(
                f"semantic_chains[{index}].side must be center/left/right"
            )
        chains.append(
            SemanticChain(
                chain_id=_string(
                    item["chain_id"], owner=f"semantic_chains[{index}].chain_id"
                ),
                chain_kind=_string(
                    item["chain_kind"],
                    owner=f"semantic_chains[{index}].chain_kind",
                ),
                side=side,
                semantic_joint_ids=_string_tuple(
                    item["semantic_joint_ids"],
                    owner=f"semantic_chains[{index}].semantic_joint_ids",
                ),
                end_effector_role=_string(
                    item["end_effector_role"],
                    owner=f"semantic_chains[{index}].end_effector_role",
                ),
                target_end_effector_joint_id=_string(
                    item["target_end_effector_joint_id"],
                    owner=(f"semantic_chains[{index}].target_end_effector_joint_id"),
                ),
            )
        )
    chain_ids = tuple(chain.chain_id for chain in chains)
    if len(set(chain_ids)) != len(chain_ids):
        raise MotionProfileError("semantic chain IDs must be unique")
    end_effector_roles = tuple(chain.end_effector_role for chain in chains)
    if len(set(end_effector_roles)) != len(end_effector_roles):
        raise MotionProfileError("semantic chain end-effector roles must be unique")
    semantic_joint_ids = tuple(
        joint_id for chain in chains for joint_id in chain.semantic_joint_ids
    )
    if len(set(semantic_joint_ids)) != len(semantic_joint_ids):
        raise MotionProfileError(
            "semantic joint IDs must belong to exactly one declared chain"
        )
    if root_semantic_id not in semantic_joint_ids:
        raise MotionProfileError(
            "solver.root_joint_semantic_id must name one declared semantic joint"
        )

    raw_mappings = root["joint_mappings"]
    if not isinstance(raw_mappings, list) or not raw_mappings:
        raise MotionProfileError("joint_mappings must be a non-empty array")
    mappings: list[JointMapping] = []
    for index, raw_mapping in enumerate(raw_mappings):
        item = _object(raw_mapping, owner=f"joint_mappings[{index}]")
        _exact_keys(
            item,
            owner=f"joint_mappings[{index}]",
            expected={"semantic_joint_id", "source_joint_id", "target_joint_id"},
        )
        mappings.append(
            JointMapping(
                semantic_joint_id=_string(
                    item["semantic_joint_id"],
                    owner=f"joint_mappings[{index}].semantic_joint_id",
                ),
                source_joint_id=_string(
                    item["source_joint_id"],
                    owner=f"joint_mappings[{index}].source_joint_id",
                ),
                target_joint_id=_string(
                    item["target_joint_id"],
                    owner=f"joint_mappings[{index}].target_joint_id",
                ),
            )
        )
    mapped_semantics = tuple(mapping.semantic_joint_id for mapping in mappings)
    if set(mapped_semantics) != set(semantic_joint_ids) or len(mapped_semantics) != len(
        set(mapped_semantics)
    ):
        raise MotionProfileError(
            "joint_mappings must map every semantic joint exactly once"
        )
    for label, values in (
        ("source joint", tuple(mapping.source_joint_id for mapping in mappings)),
        ("target joint", tuple(mapping.target_joint_id for mapping in mappings)),
    ):
        if len(values) != len(set(values)):
            raise MotionProfileError(f"{label} mappings must be one-to-one")

    raw_actions = root["actions"]
    if not isinstance(raw_actions, list) or not raw_actions:
        raise MotionProfileError("actions must be a non-empty array")
    actions: list[ActionMapping] = []
    for index, raw_action in enumerate(raw_actions):
        item = _object(raw_action, owner=f"actions[{index}]")
        _exact_keys(
            item,
            owner=f"actions[{index}]",
            expected={
                "semantic_action_id",
                "source_action_hint",
                "output_action_name",
            },
        )
        actions.append(
            ActionMapping(
                semantic_action_id=_string(
                    item["semantic_action_id"],
                    owner=f"actions[{index}].semantic_action_id",
                ),
                source_action_hint=_string(
                    item["source_action_hint"],
                    owner=f"actions[{index}].source_action_hint",
                ),
                output_action_name=_string(
                    item["output_action_name"],
                    owner=f"actions[{index}].output_action_name",
                ),
            )
        )
    for label, values in (
        ("semantic action", tuple(action.semantic_action_id for action in actions)),
        ("source action hint", tuple(action.source_action_hint for action in actions)),
        ("output action", tuple(action.output_action_name for action in actions)),
    ):
        if len(values) != len(set(values)):
            raise MotionProfileError(f"{label} values must be unique")

    body_plan_id = _string(root["body_plan_id"], owner="body_plan_id")
    if not body_plan_id.startswith(capability.body_plan_family):
        raise MotionProfileError(
            f"body_plan_id {body_plan_id!r} is incompatible with adapter "
            f"family {capability.body_plan_family!r}"
        )
    qa_semantic_action_id, qa_coordinate_frame, qa_contract = _parse_qa_contract(
        root["qa_contract"],
        chains=tuple(chains),
        actions=tuple(actions),
        sample_rate_hz=output_sample_rate_hz,
    )
    return MotionRetargetProfile(
        profile_id=_string(root["profile_id"], owner="profile_id"),
        adapter_id=adapter_id,
        body_plan_id=body_plan_id,
        motion_family_id=_string(root["motion_family_id"], owner="motion_family_id"),
        source_skeleton_id=_string(
            root["source_skeleton_id"], owner="source_skeleton_id"
        ),
        target_template_id=_string(
            root["target_template_id"], owner="target_template_id"
        ),
        solver_id=solver_id,
        motion_basis_xyzw=canonical_quaternion_xyzw(
            solver["motion_basis_xyzw"], owner="solver.motion_basis_xyzw"
        ),
        motion_amplitude=float(amplitude),
        output_sample_rate_hz=output_sample_rate_hz,
        time_mapping=time_mapping,
        root_joint_semantic_id=root_semantic_id,
        root_rotation_policy=root_rotation_policy,
        root_translation_policy=root_translation_policy,
        unmapped_target_joint_policy=unmapped_policy,
        semantic_chains=tuple(chains),
        joint_mappings=tuple(mappings),
        actions=tuple(actions),
        attribute_domain=_parse_attribute_domain(root["attribute_domain"]),
        qa_semantic_action_id=qa_semantic_action_id,
        qa_coordinate_frame=qa_coordinate_frame,
        qa_contract=qa_contract,
    )


def load_motion_retarget_profile(path: str | Path) -> MotionRetargetProfile:
    """Load a duplicate-safe UTF-8 profile and validate its semantic contract."""

    source = Path(path).resolve()
    if source.is_symlink() or not source.is_file() or source.stat().st_size <= 0:
        raise MotionProfileError(f"missing or unsafe motion profile: {source}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MotionProfileError(
                    f"motion profile contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                MotionProfileError(f"non-finite profile number: {constant}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MotionProfileError(
            f"unable to load motion profile {source}: {exc}"
        ) from exc
    return _parse_profile(value)
