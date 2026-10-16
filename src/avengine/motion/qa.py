"""Body-plan-neutral quality gates for retargeted motion arrays.

The core deliberately knows nothing about dogs, paws, wings, or concrete rig
joint names.  A versioned body-plan/profile layer supplies semantic chain IDs,
group membership, required semantic joint IDs, and thresholds.  Callers also
perform forward kinematics and express each chain terminal in the canonical
``(forward, lateral, vertical)`` actor frame before invoking this module.

Invalid dimensions, non-finite values, missing semantic data, and malformed
threshold contracts produce a deterministic failing report.  They are never
coerced into a passing result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np


MOTION_QA_SCHEMA = "avengine_retarget_motion_qa_v1"
_AXES = ("forward", "lateral", "vertical")
_METRIC_SPACES = ("meters", "rest_length_normalized")
_ZERO_EXCURSION_TOLERANCE = 1.0e-12
_QUATERNION_NORM_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class SemanticChainSamples:
    """One semantic chain sampled on one shared action timeline.

    ``terminal_positions_flv_m`` has shape ``(samples, 3)`` and coordinate
    order forward, lateral, vertical.  Every joint rotation sequence has shape
    ``(samples, 4)`` in xyzw order and represents an absolute or local rotation
    chosen consistently by the caller/profile.
    """

    chain_id: str
    rest_length_m: float
    terminal_positions_flv_m: Any
    joint_rotations_xyzw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChainMotionThresholds:
    """Optional terminal-trajectory limits for one semantic chain."""

    minimum_forward_excursion_m: float | None = None
    maximum_forward_excursion_m: float | None = None
    minimum_vertical_excursion_m: float | None = None
    maximum_vertical_excursion_m: float | None = None
    maximum_lateral_excursion_m: float | None = None
    minimum_forward_excursion_normalized: float | None = None
    maximum_forward_excursion_normalized: float | None = None
    minimum_vertical_excursion_normalized: float | None = None
    maximum_vertical_excursion_normalized: float | None = None
    maximum_lateral_excursion_normalized: float | None = None
    maximum_lateral_to_forward_ratio: float | None = None


@dataclass(frozen=True)
class JointMotionThresholds:
    """Optional geodesic rotation limits for one semantic joint."""

    minimum_angular_excursion_degrees: float | None = None
    maximum_angular_excursion_degrees: float | None = None
    maximum_angular_speed_degrees_per_second: float | None = None


@dataclass(frozen=True)
class SemanticChainGroup:
    """A caller-defined collection such as fore/hind, left/right, or wings."""

    group_id: str
    chain_ids: tuple[str, ...]


@dataclass(frozen=True)
class GroupExcursionRatioThreshold:
    """Bound a ratio between caller-defined group/axis excursion means."""

    ratio_id: str
    numerator_group_id: str
    numerator_axis: str
    denominator_group_id: str
    denominator_axis: str
    metric_space: str = "rest_length_normalized"
    minimum_ratio: float | None = None
    maximum_ratio: float | None = None


@dataclass(frozen=True)
class ChainSymmetryThreshold:
    """Bound an unordered pair's excursion asymmetry on declared axes.

    Relative difference is ``abs(a - b) / mean(a, b)``.  Two zero excursions
    have zero asymmetry; exactly one zero excursion has the maximum value 2.
    """

    symmetry_id: str
    first_chain_id: str
    second_chain_id: str
    maximum_relative_difference: float
    axes: tuple[str, ...] = _AXES
    metric_space: str = "rest_length_normalized"


@dataclass(frozen=True)
class MotionQAContract:
    """Strict semantic and numeric contract supplied by a motion profile."""

    required_chain_ids: tuple[str, ...]
    required_joint_ids_by_chain: Mapping[str, tuple[str, ...]]
    sample_rate_hz: float
    cyclic: bool
    minimum_sample_count: int = 2
    chain_groups: tuple[SemanticChainGroup, ...] = ()
    chain_thresholds: Mapping[str, ChainMotionThresholds] = field(default_factory=dict)
    joint_thresholds_by_chain: Mapping[str, Mapping[str, JointMotionThresholds]] = (
        field(default_factory=dict)
    )
    group_ratio_thresholds: tuple[GroupExcursionRatioThreshold, ...] = ()
    symmetry_thresholds: tuple[ChainSymmetryThreshold, ...] = ()


@dataclass(frozen=True, order=True)
class MotionQAIssue:
    """One stable rejection reason."""

    code: str
    owner: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "owner": self.owner,
        }


@dataclass(frozen=True)
class JointMotionMetrics:
    joint_id: str
    angular_excursion_degrees: float
    maximum_angular_speed_degrees_per_second: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "angular_excursion_degrees": self.angular_excursion_degrees,
            "joint_id": self.joint_id,
            "maximum_angular_speed_degrees_per_second": (
                self.maximum_angular_speed_degrees_per_second
            ),
        }


@dataclass(frozen=True)
class ChainMotionMetrics:
    chain_id: str
    rest_length_m: float
    forward_excursion_m: float
    lateral_excursion_m: float
    vertical_excursion_m: float
    forward_excursion_normalized: float
    lateral_excursion_normalized: float
    vertical_excursion_normalized: float
    lateral_to_forward_ratio: float | None
    joints: tuple[JointMotionMetrics, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "joint_rotation_metrics": [item.to_dict() for item in self.joints],
            "lateral_to_forward_ratio": self.lateral_to_forward_ratio,
            "rest_length_m": self.rest_length_m,
            "terminal_excursion_m": {
                "forward": self.forward_excursion_m,
                "lateral": self.lateral_excursion_m,
                "vertical": self.vertical_excursion_m,
            },
            "terminal_excursion_normalized_by_rest_length": {
                "forward": self.forward_excursion_normalized,
                "lateral": self.lateral_excursion_normalized,
                "vertical": self.vertical_excursion_normalized,
            },
        }


@dataclass(frozen=True)
class GroupMotionMetrics:
    group_id: str
    chain_ids: tuple[str, ...]
    mean_forward_excursion_m: float
    mean_lateral_excursion_m: float
    mean_vertical_excursion_m: float
    mean_forward_excursion_normalized: float
    mean_lateral_excursion_normalized: float
    mean_vertical_excursion_normalized: float
    lateral_to_forward_ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregation": "arithmetic_mean_of_member_chain_excursions",
            "chain_ids": list(self.chain_ids),
            "group_id": self.group_id,
            "lateral_to_forward_ratio": self.lateral_to_forward_ratio,
            "lateral_to_forward_ratio_metric_space": "rest_length_normalized",
            "mean_terminal_excursion_m": {
                "forward": self.mean_forward_excursion_m,
                "lateral": self.mean_lateral_excursion_m,
                "vertical": self.mean_vertical_excursion_m,
            },
            "mean_terminal_excursion_normalized_by_rest_length": {
                "forward": self.mean_forward_excursion_normalized,
                "lateral": self.mean_lateral_excursion_normalized,
                "vertical": self.mean_vertical_excursion_normalized,
            },
        }


@dataclass(frozen=True)
class GroupRatioMetrics:
    ratio_id: str
    numerator_group_id: str
    numerator_axis: str
    denominator_group_id: str
    denominator_axis: str
    metric_space: str
    value: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "denominator_axis": self.denominator_axis,
            "denominator_group_id": self.denominator_group_id,
            "metric_space": self.metric_space,
            "numerator_axis": self.numerator_axis,
            "numerator_group_id": self.numerator_group_id,
            "ratio_id": self.ratio_id,
            "value": self.value,
        }


@dataclass(frozen=True)
class ChainSymmetryMetrics:
    symmetry_id: str
    first_chain_id: str
    second_chain_id: str
    metric_space: str
    relative_difference_by_axis: tuple[tuple[str, float], ...]
    maximum_relative_difference: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_chain_id": self.first_chain_id,
            "maximum_relative_difference": self.maximum_relative_difference,
            "metric_space": self.metric_space,
            "relative_difference_by_axis": dict(self.relative_difference_by_axis),
            "second_chain_id": self.second_chain_id,
            "symmetry_id": self.symmetry_id,
        }


@dataclass(frozen=True)
class MotionQAReport:
    """Deterministic, JSON-safe result of one motion audit."""

    status: str
    sample_count: int | None
    sample_rate_hz: float | None
    cyclic: bool | None
    chains: tuple[ChainMotionMetrics, ...]
    groups: tuple[GroupMotionMetrics, ...]
    group_ratios: tuple[GroupRatioMetrics, ...]
    symmetries: tuple[ChainSymmetryMetrics, ...]
    issues: tuple[MotionQAIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chains": [item.to_dict() for item in self.chains],
            "coordinate_order": list(_AXES),
            "cyclic": self.cyclic,
            "formal_dataset_registration_authorized": False,
            "group_ratios": [item.to_dict() for item in self.group_ratios],
            "groups": [item.to_dict() for item in self.groups],
            "issues": [item.to_dict() for item in self.issues],
            "sample_count": self.sample_count,
            "sample_rate_hz": self.sample_rate_hz,
            "schema": MOTION_QA_SCHEMA,
            "status": self.status,
            "symmetries": [item.to_dict() for item in self.symmetries],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize with stable key ordering and no non-standard NaN values."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
            sort_keys=True,
        )


def _canonical_float(value: float) -> float:
    number = float(value)
    return 0.0 if number == 0.0 else number


def _is_finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _valid_id(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value.strip() == value


def _issue(issues: list[MotionQAIssue], code: str, owner: str, message: str) -> None:
    issues.append(MotionQAIssue(code=code, owner=owner, message=message))


def _validate_optional_threshold(
    value: Any,
    *,
    owner: str,
    issues: list[MotionQAIssue],
) -> None:
    if value is not None and not _is_finite_nonnegative(value):
        _issue(
            issues,
            "invalid_threshold",
            owner,
            "threshold must be null or a finite non-negative number",
        )


def _validate_chain_thresholds(
    value: Any,
    *,
    owner: str,
    issues: list[MotionQAIssue],
) -> None:
    if not isinstance(value, ChainMotionThresholds):
        _issue(
            issues,
            "invalid_chain_threshold_contract",
            owner,
            "chain threshold must be ChainMotionThresholds",
        )
        return
    for name in (
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
    ):
        _validate_optional_threshold(
            getattr(value, name), owner=f"{owner}/{name}", issues=issues
        )
    if (
        _is_finite_nonnegative(value.minimum_forward_excursion_m)
        and _is_finite_nonnegative(value.maximum_forward_excursion_m)
        and float(value.minimum_forward_excursion_m)
        > float(value.maximum_forward_excursion_m)
    ):
        _issue(
            issues,
            "invalid_threshold_range",
            owner,
            "minimum forward excursion exceeds maximum",
        )
    if (
        _is_finite_nonnegative(value.minimum_vertical_excursion_m)
        and _is_finite_nonnegative(value.maximum_vertical_excursion_m)
        and float(value.minimum_vertical_excursion_m)
        > float(value.maximum_vertical_excursion_m)
    ):
        _issue(
            issues,
            "invalid_threshold_range",
            owner,
            "minimum vertical excursion exceeds maximum",
        )
    if (
        _is_finite_nonnegative(value.minimum_forward_excursion_normalized)
        and _is_finite_nonnegative(value.maximum_forward_excursion_normalized)
        and float(value.minimum_forward_excursion_normalized)
        > float(value.maximum_forward_excursion_normalized)
    ):
        _issue(
            issues,
            "invalid_threshold_range",
            owner,
            "minimum normalized forward excursion exceeds maximum",
        )
    if (
        _is_finite_nonnegative(value.minimum_vertical_excursion_normalized)
        and _is_finite_nonnegative(value.maximum_vertical_excursion_normalized)
        and float(value.minimum_vertical_excursion_normalized)
        > float(value.maximum_vertical_excursion_normalized)
    ):
        _issue(
            issues,
            "invalid_threshold_range",
            owner,
            "minimum normalized vertical excursion exceeds maximum",
        )


def _validate_joint_thresholds(
    value: Any,
    *,
    owner: str,
    issues: list[MotionQAIssue],
) -> None:
    if not isinstance(value, JointMotionThresholds):
        _issue(
            issues,
            "invalid_joint_threshold_contract",
            owner,
            "joint threshold must be JointMotionThresholds",
        )
        return
    for name in (
        "minimum_angular_excursion_degrees",
        "maximum_angular_excursion_degrees",
        "maximum_angular_speed_degrees_per_second",
    ):
        _validate_optional_threshold(
            getattr(value, name), owner=f"{owner}/{name}", issues=issues
        )
    if (
        _is_finite_nonnegative(value.minimum_angular_excursion_degrees)
        and _is_finite_nonnegative(value.maximum_angular_excursion_degrees)
        and float(value.minimum_angular_excursion_degrees)
        > float(value.maximum_angular_excursion_degrees)
    ):
        _issue(
            issues,
            "invalid_threshold_range",
            owner,
            "minimum angular excursion exceeds maximum",
        )


def _validate_identifier_tuple(
    value: Any,
    *,
    owner: str,
    allow_empty: bool,
    issues: list[MotionQAIssue],
) -> tuple[str, ...] | None:
    if not isinstance(value, tuple) or (not allow_empty and len(value) == 0):
        _issue(
            issues,
            "invalid_identifier_sequence",
            owner,
            "identifier contract must be a non-empty immutable tuple"
            if not allow_empty
            else "identifier contract must be an immutable tuple",
        )
        return None
    if not all(_valid_id(item) for item in value):
        _issue(
            issues,
            "invalid_identifier",
            owner,
            "identifiers must be non-empty canonical strings",
        )
        return None
    if len(set(value)) != len(value):
        _issue(
            issues,
            "duplicate_identifier",
            owner,
            "identifiers must be unique",
        )
        return None
    return tuple(value)


def _validate_contract(
    contract: Any, issues: list[MotionQAIssue]
) -> tuple[str, ...] | None:
    if not isinstance(contract, MotionQAContract):
        _issue(
            issues,
            "invalid_qa_contract",
            "contract",
            "contract must be MotionQAContract",
        )
        return None
    required = _validate_identifier_tuple(
        contract.required_chain_ids,
        owner="contract/required_chain_ids",
        allow_empty=False,
        issues=issues,
    )
    if (
        isinstance(contract.sample_rate_hz, bool)
        or not isinstance(contract.sample_rate_hz, Real)
        or not math.isfinite(float(contract.sample_rate_hz))
        or float(contract.sample_rate_hz) <= 0.0
    ):
        _issue(
            issues,
            "invalid_sample_rate",
            "contract/sample_rate_hz",
            "sample rate must be a finite positive number",
        )
    if not isinstance(contract.cyclic, bool):
        _issue(
            issues,
            "invalid_cyclic_flag",
            "contract/cyclic",
            "cyclic must be a boolean",
        )
    if (
        isinstance(contract.minimum_sample_count, bool)
        or not isinstance(contract.minimum_sample_count, int)
        or contract.minimum_sample_count < 2
    ):
        _issue(
            issues,
            "invalid_minimum_sample_count",
            "contract/minimum_sample_count",
            "minimum sample count must be an integer of at least two",
        )
    if required is None:
        return None
    required_set = set(required)
    if not isinstance(contract.required_joint_ids_by_chain, Mapping):
        _issue(
            issues,
            "invalid_joint_contract",
            "contract/required_joint_ids_by_chain",
            "required joint IDs must be a mapping",
        )
    else:
        joint_keys = set(contract.required_joint_ids_by_chain)
        if joint_keys != required_set:
            _issue(
                issues,
                "incomplete_joint_contract",
                "contract/required_joint_ids_by_chain",
                "joint contract keys must exactly match required chains",
            )
        for chain_id in sorted(joint_keys & required_set):
            _validate_identifier_tuple(
                contract.required_joint_ids_by_chain[chain_id],
                owner=f"contract/required_joint_ids_by_chain/{chain_id}",
                allow_empty=True,
                issues=issues,
            )

    if not isinstance(contract.chain_thresholds, Mapping):
        _issue(
            issues,
            "invalid_chain_threshold_mapping",
            "contract/chain_thresholds",
            "chain thresholds must be a mapping",
        )
    else:
        unknown = sorted(set(contract.chain_thresholds) - required_set)
        if unknown:
            _issue(
                issues,
                "unknown_threshold_chain",
                "contract/chain_thresholds",
                f"unknown chain IDs: {unknown}",
            )
        for chain_id in sorted(set(contract.chain_thresholds) & required_set):
            _validate_chain_thresholds(
                contract.chain_thresholds[chain_id],
                owner=f"contract/chain_thresholds/{chain_id}",
                issues=issues,
            )

    if not isinstance(contract.joint_thresholds_by_chain, Mapping):
        _issue(
            issues,
            "invalid_joint_threshold_mapping",
            "contract/joint_thresholds_by_chain",
            "joint thresholds must be a nested mapping",
        )
    else:
        unknown = sorted(set(contract.joint_thresholds_by_chain) - required_set)
        if unknown:
            _issue(
                issues,
                "unknown_threshold_chain",
                "contract/joint_thresholds_by_chain",
                f"unknown chain IDs: {unknown}",
            )
        for chain_id in sorted(set(contract.joint_thresholds_by_chain) & required_set):
            values = contract.joint_thresholds_by_chain[chain_id]
            if not isinstance(values, Mapping):
                _issue(
                    issues,
                    "invalid_joint_threshold_mapping",
                    f"contract/joint_thresholds_by_chain/{chain_id}",
                    "per-chain joint thresholds must be a mapping",
                )
                continue
            required_joints = set(
                contract.required_joint_ids_by_chain.get(chain_id, ())
            )
            unknown_joints = sorted(set(values) - required_joints)
            if unknown_joints:
                _issue(
                    issues,
                    "unknown_threshold_joint",
                    f"contract/joint_thresholds_by_chain/{chain_id}",
                    f"unknown joint IDs: {unknown_joints}",
                )
            for joint_id in sorted(set(values) & required_joints):
                _validate_joint_thresholds(
                    values[joint_id],
                    owner=(f"contract/joint_thresholds_by_chain/{chain_id}/{joint_id}"),
                    issues=issues,
                )

    if not isinstance(contract.chain_groups, tuple):
        _issue(
            issues,
            "invalid_group_contract",
            "contract/chain_groups",
            "chain groups must be an immutable tuple",
        )
        groups: dict[str, SemanticChainGroup] = {}
    else:
        groups = {}
        for index, group in enumerate(contract.chain_groups):
            owner = f"contract/chain_groups/{index}"
            if not isinstance(group, SemanticChainGroup):
                _issue(
                    issues,
                    "invalid_group_contract",
                    owner,
                    "group must be SemanticChainGroup",
                )
                continue
            if not _valid_id(group.group_id):
                _issue(
                    issues,
                    "invalid_identifier",
                    f"{owner}/group_id",
                    "group ID must be a non-empty canonical string",
                )
                continue
            if group.group_id in groups:
                _issue(
                    issues,
                    "duplicate_identifier",
                    "contract/chain_groups",
                    f"duplicate group ID: {group.group_id}",
                )
                continue
            members = _validate_identifier_tuple(
                group.chain_ids,
                owner=f"{owner}/chain_ids",
                allow_empty=False,
                issues=issues,
            )
            if members is not None:
                unknown_members = sorted(set(members) - required_set)
                if unknown_members:
                    _issue(
                        issues,
                        "unknown_group_chain",
                        f"{owner}/chain_ids",
                        f"unknown chain IDs: {unknown_members}",
                    )
            groups[group.group_id] = group

    if not isinstance(contract.group_ratio_thresholds, tuple):
        _issue(
            issues,
            "invalid_group_ratio_contract",
            "contract/group_ratio_thresholds",
            "group ratios must be an immutable tuple",
        )
    else:
        ratio_ids: set[str] = set()
        for index, rule in enumerate(contract.group_ratio_thresholds):
            owner = f"contract/group_ratio_thresholds/{index}"
            if not isinstance(rule, GroupExcursionRatioThreshold):
                _issue(
                    issues,
                    "invalid_group_ratio_contract",
                    owner,
                    "ratio rule must be GroupExcursionRatioThreshold",
                )
                continue
            valid_ratio_id = _valid_id(rule.ratio_id)
            if not valid_ratio_id:
                _issue(
                    issues,
                    "invalid_identifier",
                    f"{owner}/ratio_id",
                    "ratio ID must be a non-empty canonical string",
                )
            elif rule.ratio_id in ratio_ids:
                _issue(
                    issues,
                    "duplicate_identifier",
                    "contract/group_ratio_thresholds",
                    f"duplicate ratio ID: {rule.ratio_id}",
                )
            if valid_ratio_id:
                ratio_ids.add(rule.ratio_id)
            for label, group_id in (
                ("numerator_group_id", rule.numerator_group_id),
                ("denominator_group_id", rule.denominator_group_id),
            ):
                if not _valid_id(group_id) or group_id not in groups:
                    _issue(
                        issues,
                        "unknown_ratio_group",
                        f"{owner}/{label}",
                        f"unknown group ID: {group_id}",
                    )
            for label, axis in (
                ("numerator_axis", rule.numerator_axis),
                ("denominator_axis", rule.denominator_axis),
            ):
                if axis not in _AXES:
                    _issue(
                        issues,
                        "invalid_ratio_axis",
                        f"{owner}/{label}",
                        f"axis must be one of {_AXES}",
                    )
            if rule.metric_space not in _METRIC_SPACES:
                _issue(
                    issues,
                    "invalid_metric_space",
                    f"{owner}/metric_space",
                    f"metric space must be one of {_METRIC_SPACES}",
                )
            _validate_optional_threshold(
                rule.minimum_ratio,
                owner=f"{owner}/minimum_ratio",
                issues=issues,
            )
            _validate_optional_threshold(
                rule.maximum_ratio,
                owner=f"{owner}/maximum_ratio",
                issues=issues,
            )
            if rule.minimum_ratio is None and rule.maximum_ratio is None:
                _issue(
                    issues,
                    "unbounded_group_ratio",
                    owner,
                    "ratio rule must declare at least one bound",
                )
            if (
                _is_finite_nonnegative(rule.minimum_ratio)
                and _is_finite_nonnegative(rule.maximum_ratio)
                and float(rule.minimum_ratio) > float(rule.maximum_ratio)
            ):
                _issue(
                    issues,
                    "invalid_threshold_range",
                    owner,
                    "minimum ratio exceeds maximum",
                )

    if not isinstance(contract.symmetry_thresholds, tuple):
        _issue(
            issues,
            "invalid_symmetry_contract",
            "contract/symmetry_thresholds",
            "symmetry thresholds must be an immutable tuple",
        )
    else:
        symmetry_ids: set[str] = set()
        for index, rule in enumerate(contract.symmetry_thresholds):
            owner = f"contract/symmetry_thresholds/{index}"
            if not isinstance(rule, ChainSymmetryThreshold):
                _issue(
                    issues,
                    "invalid_symmetry_contract",
                    owner,
                    "symmetry rule must be ChainSymmetryThreshold",
                )
                continue
            valid_symmetry_id = _valid_id(rule.symmetry_id)
            if not valid_symmetry_id:
                _issue(
                    issues,
                    "invalid_identifier",
                    f"{owner}/symmetry_id",
                    "symmetry ID must be a non-empty canonical string",
                )
            elif rule.symmetry_id in symmetry_ids:
                _issue(
                    issues,
                    "duplicate_identifier",
                    "contract/symmetry_thresholds",
                    f"duplicate symmetry ID: {rule.symmetry_id}",
                )
            if valid_symmetry_id:
                symmetry_ids.add(rule.symmetry_id)
            for label, chain_id in (
                ("first_chain_id", rule.first_chain_id),
                ("second_chain_id", rule.second_chain_id),
            ):
                if not _valid_id(chain_id) or chain_id not in required_set:
                    _issue(
                        issues,
                        "unknown_symmetry_chain",
                        f"{owner}/{label}",
                        f"unknown chain ID: {chain_id}",
                    )
            if (
                _valid_id(rule.first_chain_id)
                and rule.first_chain_id == rule.second_chain_id
            ):
                _issue(
                    issues,
                    "invalid_symmetry_pair",
                    owner,
                    "symmetry pair must contain two distinct chains",
                )
            axes = _validate_identifier_tuple(
                rule.axes,
                owner=f"{owner}/axes",
                allow_empty=False,
                issues=issues,
            )
            invalid_axes = sorted(set(axes) - set(_AXES)) if axes is not None else []
            if axes is not None and invalid_axes:
                _issue(
                    issues,
                    "invalid_symmetry_axis",
                    f"{owner}/axes",
                    f"invalid axes: {invalid_axes}",
                )
            _validate_optional_threshold(
                rule.maximum_relative_difference,
                owner=f"{owner}/maximum_relative_difference",
                issues=issues,
            )
            if rule.metric_space not in _METRIC_SPACES:
                _issue(
                    issues,
                    "invalid_metric_space",
                    f"{owner}/metric_space",
                    f"metric space must be one of {_METRIC_SPACES}",
                )
    return required


def _numeric_array(
    value: Any,
    *,
    columns: int,
    owner: str,
    code: str,
    issues: list[MotionQAIssue],
) -> np.ndarray | None:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError):
        raw = np.asarray(())
    if raw.dtype.kind not in "iuf" or raw.ndim != 2 or raw.shape[1:] != (columns,):
        _issue(
            issues,
            code,
            owner,
            f"samples must have numeric shape (N, {columns})",
        )
        return None
    result = raw.astype(np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        _issue(
            issues,
            code,
            owner,
            "samples must contain only finite numbers",
        )
        return None
    return result


def _validated_samples(
    samples: Any,
    contract: MotionQAContract,
    required_chain_ids: tuple[str, ...],
    issues: list[MotionQAIssue],
) -> tuple[dict[str, tuple[float, np.ndarray, dict[str, np.ndarray]]], int] | None:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        _issue(
            issues,
            "invalid_chain_samples",
            "samples",
            "samples must be a sequence of SemanticChainSamples",
        )
        return None
    indexed: dict[str, SemanticChainSamples] = {}
    for index, item in enumerate(samples):
        if not isinstance(item, SemanticChainSamples):
            _issue(
                issues,
                "invalid_chain_samples",
                f"samples/{index}",
                "item must be SemanticChainSamples",
            )
            continue
        if not _valid_id(item.chain_id):
            _issue(
                issues,
                "invalid_identifier",
                f"samples/{index}/chain_id",
                "chain ID must be a non-empty canonical string",
            )
            continue
        if item.chain_id in indexed:
            _issue(
                issues,
                "duplicate_chain",
                "samples",
                f"duplicate chain ID: {item.chain_id}",
            )
            continue
        indexed[item.chain_id] = item
    required = set(required_chain_ids)
    available = set(indexed)
    missing = sorted(required - available)
    extra = sorted(available - required)
    if missing:
        _issue(
            issues,
            "missing_required_chain",
            "samples",
            f"missing chain IDs: {missing}",
        )
    if extra:
        _issue(
            issues,
            "unexpected_chain",
            "samples",
            f"unexpected chain IDs: {extra}",
        )
    if issues:
        return None

    validated: dict[str, tuple[float, np.ndarray, dict[str, np.ndarray]]] = {}
    sample_counts: set[int] = set()
    for chain_id in sorted(required):
        item = indexed[chain_id]
        rest_length: float | None = None
        if (
            isinstance(item.rest_length_m, bool)
            or not isinstance(item.rest_length_m, Real)
            or not math.isfinite(float(item.rest_length_m))
            or float(item.rest_length_m) <= 0.0
        ):
            _issue(
                issues,
                "invalid_rest_length",
                f"samples/{chain_id}/rest_length_m",
                "rest length must be a finite positive number in meters",
            )
        else:
            rest_length = float(item.rest_length_m)
        positions = _numeric_array(
            item.terminal_positions_flv_m,
            columns=3,
            owner=f"samples/{chain_id}/terminal_positions_flv_m",
            code="invalid_terminal_trajectory",
            issues=issues,
        )
        if not isinstance(item.joint_rotations_xyzw, Mapping):
            _issue(
                issues,
                "invalid_joint_rotation_mapping",
                f"samples/{chain_id}/joint_rotations_xyzw",
                "joint rotations must be a mapping",
            )
            continue
        required_joints = set(contract.required_joint_ids_by_chain[chain_id])
        available_joints = set(item.joint_rotations_xyzw)
        missing_joints = sorted(required_joints - available_joints)
        extra_joints = sorted(available_joints - required_joints)
        if missing_joints:
            _issue(
                issues,
                "missing_required_joint",
                f"samples/{chain_id}/joint_rotations_xyzw",
                f"missing joint IDs: {missing_joints}",
            )
        if extra_joints:
            _issue(
                issues,
                "unexpected_joint",
                f"samples/{chain_id}/joint_rotations_xyzw",
                f"unexpected joint IDs: {extra_joints}",
            )
        rotations: dict[str, np.ndarray] = {}
        for joint_id in sorted(required_joints & available_joints):
            rotation = _numeric_array(
                item.joint_rotations_xyzw[joint_id],
                columns=4,
                owner=(f"samples/{chain_id}/joint_rotations_xyzw/{joint_id}"),
                code="invalid_joint_rotations",
                issues=issues,
            )
            if rotation is None:
                continue
            norms = np.linalg.norm(rotation, axis=1)
            if np.any(np.abs(norms - 1.0) > _QUATERNION_NORM_TOLERANCE):
                _issue(
                    issues,
                    "invalid_joint_rotations",
                    (f"samples/{chain_id}/joint_rotations_xyzw/{joint_id}"),
                    "quaternions must already be unit normalized",
                )
                continue
            rotations[joint_id] = rotation / norms[:, None]
            sample_counts.add(len(rotation))
        if positions is not None and rest_length is not None:
            sample_counts.add(len(positions))
            validated[chain_id] = (
                rest_length,
                positions,
                rotations,
            )
    if len(sample_counts) > 1:
        _issue(
            issues,
            "sample_count_mismatch",
            "samples",
            f"all trajectories and rotations must share one count: {sorted(sample_counts)}",
        )
    sample_count = next(iter(sample_counts), 0)
    if sample_count < contract.minimum_sample_count:
        _issue(
            issues,
            "insufficient_samples",
            "samples",
            (
                f"sample count {sample_count} is below required minimum "
                f"{contract.minimum_sample_count}"
            ),
        )
    if issues:
        return None
    return validated, sample_count


def _ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= _ZERO_EXCURSION_TOLERANCE:
        return None
    return _canonical_float(numerator / denominator)


def _geodesic_degrees_from_dots(dots: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.abs(dots), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(clipped))


def _joint_metrics(
    joint_id: str,
    rotations: np.ndarray,
    *,
    sample_rate_hz: float,
    cyclic: bool,
) -> JointMotionMetrics:
    pairwise = rotations @ rotations.T
    excursion = float(np.max(_geodesic_degrees_from_dots(pairwise)))
    next_rotations = np.roll(rotations, -1, axis=0)
    dots = np.sum(rotations * next_rotations, axis=1)
    if not cyclic:
        dots = dots[:-1]
    speeds = _geodesic_degrees_from_dots(dots) * sample_rate_hz
    return JointMotionMetrics(
        joint_id=joint_id,
        angular_excursion_degrees=_canonical_float(excursion),
        maximum_angular_speed_degrees_per_second=_canonical_float(
            float(np.max(speeds))
        ),
    )


def _is_below(value: float, threshold: float) -> bool:
    tolerance = 1.0e-12 * max(1.0, abs(threshold))
    return value < threshold - tolerance


def _is_above(value: float, threshold: float) -> bool:
    tolerance = 1.0e-12 * max(1.0, abs(threshold))
    return value > threshold + tolerance


def _format_number(value: float) -> str:
    return format(float(value), ".12g")


def _check_chain_thresholds(
    metrics: ChainMotionMetrics,
    threshold: ChainMotionThresholds | None,
    issues: list[MotionQAIssue],
) -> None:
    if threshold is None:
        return
    owner = f"chains/{metrics.chain_id}"
    checks = (
        (
            "chain_forward_excursion_below_minimum",
            metrics.forward_excursion_m,
            threshold.minimum_forward_excursion_m,
            _is_below,
        ),
        (
            "chain_forward_excursion_above_maximum",
            metrics.forward_excursion_m,
            threshold.maximum_forward_excursion_m,
            _is_above,
        ),
        (
            "chain_vertical_excursion_below_minimum",
            metrics.vertical_excursion_m,
            threshold.minimum_vertical_excursion_m,
            _is_below,
        ),
        (
            "chain_vertical_excursion_above_maximum",
            metrics.vertical_excursion_m,
            threshold.maximum_vertical_excursion_m,
            _is_above,
        ),
        (
            "chain_lateral_excursion_above_maximum",
            metrics.lateral_excursion_m,
            threshold.maximum_lateral_excursion_m,
            _is_above,
        ),
        (
            "chain_forward_excursion_normalized_below_minimum",
            metrics.forward_excursion_normalized,
            threshold.minimum_forward_excursion_normalized,
            _is_below,
        ),
        (
            "chain_forward_excursion_normalized_above_maximum",
            metrics.forward_excursion_normalized,
            threshold.maximum_forward_excursion_normalized,
            _is_above,
        ),
        (
            "chain_vertical_excursion_normalized_below_minimum",
            metrics.vertical_excursion_normalized,
            threshold.minimum_vertical_excursion_normalized,
            _is_below,
        ),
        (
            "chain_vertical_excursion_normalized_above_maximum",
            metrics.vertical_excursion_normalized,
            threshold.maximum_vertical_excursion_normalized,
            _is_above,
        ),
        (
            "chain_lateral_excursion_normalized_above_maximum",
            metrics.lateral_excursion_normalized,
            threshold.maximum_lateral_excursion_normalized,
            _is_above,
        ),
    )
    for code, value, limit, predicate in checks:
        if limit is not None and predicate(value, limit):
            _issue(
                issues,
                code,
                owner,
                f"value {_format_number(value)} violates threshold {_format_number(limit)}",
            )
    ratio_limit = threshold.maximum_lateral_to_forward_ratio
    if ratio_limit is not None:
        if metrics.lateral_to_forward_ratio is None:
            _issue(
                issues,
                "chain_lateral_to_forward_undefined",
                owner,
                "forward excursion is zero, so lateral/forward is undefined",
            )
        elif _is_above(metrics.lateral_to_forward_ratio, ratio_limit):
            _issue(
                issues,
                "chain_lateral_to_forward_above_maximum",
                owner,
                (
                    f"value {_format_number(metrics.lateral_to_forward_ratio)} "
                    f"violates threshold {_format_number(ratio_limit)}"
                ),
            )


def _check_joint_thresholds(
    chain_id: str,
    metrics: JointMotionMetrics,
    threshold: JointMotionThresholds | None,
    issues: list[MotionQAIssue],
) -> None:
    if threshold is None:
        return
    owner = f"chains/{chain_id}/joints/{metrics.joint_id}"
    checks = (
        (
            "joint_angular_excursion_below_minimum",
            metrics.angular_excursion_degrees,
            threshold.minimum_angular_excursion_degrees,
            _is_below,
        ),
        (
            "joint_angular_excursion_above_maximum",
            metrics.angular_excursion_degrees,
            threshold.maximum_angular_excursion_degrees,
            _is_above,
        ),
        (
            "joint_angular_speed_above_maximum",
            metrics.maximum_angular_speed_degrees_per_second,
            threshold.maximum_angular_speed_degrees_per_second,
            _is_above,
        ),
    )
    for code, value, limit, predicate in checks:
        if limit is not None and predicate(value, limit):
            _issue(
                issues,
                code,
                owner,
                f"value {_format_number(value)} violates threshold {_format_number(limit)}",
            )


def _invalid_report(
    contract: Any,
    issues: list[MotionQAIssue],
) -> MotionQAReport:
    sample_rate = None
    cyclic = None
    if isinstance(contract, MotionQAContract):
        if (
            isinstance(contract.sample_rate_hz, Real)
            and not isinstance(contract.sample_rate_hz, bool)
            and math.isfinite(float(contract.sample_rate_hz))
            and float(contract.sample_rate_hz) > 0.0
        ):
            sample_rate = _canonical_float(float(contract.sample_rate_hz))
        if isinstance(contract.cyclic, bool):
            cyclic = contract.cyclic
    return MotionQAReport(
        status="fail",
        sample_count=None,
        sample_rate_hz=sample_rate,
        cyclic=cyclic,
        chains=(),
        groups=(),
        group_ratios=(),
        symmetries=(),
        issues=tuple(sorted(set(issues))),
    )


def evaluate_motion_qa(
    samples: Sequence[SemanticChainSamples],
    contract: MotionQAContract,
) -> MotionQAReport:
    """Evaluate one sampled action and return a fail-closed JSON-safe report.

    Input/contract defects are represented as ``status == "fail"`` with no
    partial metrics.  Valid inputs always produce all declared chain, joint,
    group, and ratio metrics, even when one or more thresholds reject them.
    """

    issues: list[MotionQAIssue] = []
    required_chain_ids = _validate_contract(contract, issues)
    if issues or required_chain_ids is None:
        return _invalid_report(contract, issues)
    assert isinstance(contract, MotionQAContract)
    validated = _validated_samples(samples, contract, required_chain_ids, issues)
    if validated is None:
        return _invalid_report(contract, issues)
    arrays, sample_count = validated
    sample_rate = float(contract.sample_rate_hz)

    chain_metrics: list[ChainMotionMetrics] = []
    for chain_id in sorted(arrays):
        rest_length, positions, rotations = arrays[chain_id]
        excursion = np.ptp(positions, axis=0)
        normalized = excursion / rest_length
        joints = tuple(
            _joint_metrics(
                joint_id,
                rotations[joint_id],
                sample_rate_hz=sample_rate,
                cyclic=contract.cyclic,
            )
            for joint_id in sorted(rotations)
        )
        metrics = ChainMotionMetrics(
            chain_id=chain_id,
            rest_length_m=_canonical_float(rest_length),
            forward_excursion_m=_canonical_float(float(excursion[0])),
            lateral_excursion_m=_canonical_float(float(excursion[1])),
            vertical_excursion_m=_canonical_float(float(excursion[2])),
            forward_excursion_normalized=_canonical_float(float(normalized[0])),
            lateral_excursion_normalized=_canonical_float(float(normalized[1])),
            vertical_excursion_normalized=_canonical_float(float(normalized[2])),
            lateral_to_forward_ratio=_ratio(float(excursion[1]), float(excursion[0])),
            joints=joints,
        )
        chain_metrics.append(metrics)
        _check_chain_thresholds(
            metrics, contract.chain_thresholds.get(chain_id), issues
        )
        joint_thresholds = contract.joint_thresholds_by_chain.get(chain_id, {})
        for joint in joints:
            _check_joint_thresholds(
                chain_id, joint, joint_thresholds.get(joint.joint_id), issues
            )

    chain_by_id = {item.chain_id: item for item in chain_metrics}
    group_metrics: list[GroupMotionMetrics] = []
    for group in sorted(contract.chain_groups, key=lambda item: item.group_id):
        members = [chain_by_id[chain_id] for chain_id in group.chain_ids]
        forward = sum(item.forward_excursion_m for item in members) / len(members)
        lateral = sum(item.lateral_excursion_m for item in members) / len(members)
        vertical = sum(item.vertical_excursion_m for item in members) / len(members)
        forward_normalized = sum(
            item.forward_excursion_normalized for item in members
        ) / len(members)
        lateral_normalized = sum(
            item.lateral_excursion_normalized for item in members
        ) / len(members)
        vertical_normalized = sum(
            item.vertical_excursion_normalized for item in members
        ) / len(members)
        group_metrics.append(
            GroupMotionMetrics(
                group_id=group.group_id,
                chain_ids=tuple(sorted(group.chain_ids)),
                mean_forward_excursion_m=_canonical_float(forward),
                mean_lateral_excursion_m=_canonical_float(lateral),
                mean_vertical_excursion_m=_canonical_float(vertical),
                mean_forward_excursion_normalized=_canonical_float(forward_normalized),
                mean_lateral_excursion_normalized=_canonical_float(lateral_normalized),
                mean_vertical_excursion_normalized=_canonical_float(
                    vertical_normalized
                ),
                lateral_to_forward_ratio=_ratio(lateral_normalized, forward_normalized),
            )
        )

    group_by_id = {item.group_id: item for item in group_metrics}

    def group_axis(group: GroupMotionMetrics, axis: str, metric_space: str) -> float:
        if metric_space == "meters":
            return {
                "forward": group.mean_forward_excursion_m,
                "lateral": group.mean_lateral_excursion_m,
                "vertical": group.mean_vertical_excursion_m,
            }[axis]
        return {
            "forward": group.mean_forward_excursion_normalized,
            "lateral": group.mean_lateral_excursion_normalized,
            "vertical": group.mean_vertical_excursion_normalized,
        }[axis]

    ratio_metrics: list[GroupRatioMetrics] = []
    for rule in sorted(contract.group_ratio_thresholds, key=lambda item: item.ratio_id):
        numerator = group_axis(
            group_by_id[rule.numerator_group_id],
            rule.numerator_axis,
            rule.metric_space,
        )
        denominator = group_axis(
            group_by_id[rule.denominator_group_id],
            rule.denominator_axis,
            rule.metric_space,
        )
        value = _ratio(numerator, denominator)
        ratio_metrics.append(
            GroupRatioMetrics(
                ratio_id=rule.ratio_id,
                numerator_group_id=rule.numerator_group_id,
                numerator_axis=rule.numerator_axis,
                denominator_group_id=rule.denominator_group_id,
                denominator_axis=rule.denominator_axis,
                metric_space=rule.metric_space,
                value=value,
            )
        )
        owner = f"group_ratios/{rule.ratio_id}"
        if value is None:
            _issue(
                issues,
                "group_ratio_undefined",
                owner,
                "denominator excursion is zero",
            )
        elif rule.minimum_ratio is not None and _is_below(value, rule.minimum_ratio):
            _issue(
                issues,
                "group_ratio_below_minimum",
                owner,
                (
                    f"value {_format_number(value)} violates threshold "
                    f"{_format_number(rule.minimum_ratio)}"
                ),
            )
        elif rule.maximum_ratio is not None and _is_above(value, rule.maximum_ratio):
            _issue(
                issues,
                "group_ratio_above_maximum",
                owner,
                (
                    f"value {_format_number(value)} violates threshold "
                    f"{_format_number(rule.maximum_ratio)}"
                ),
            )

    def chain_axis(chain: ChainMotionMetrics, axis: str, metric_space: str) -> float:
        if metric_space == "meters":
            return {
                "forward": chain.forward_excursion_m,
                "lateral": chain.lateral_excursion_m,
                "vertical": chain.vertical_excursion_m,
            }[axis]
        return {
            "forward": chain.forward_excursion_normalized,
            "lateral": chain.lateral_excursion_normalized,
            "vertical": chain.vertical_excursion_normalized,
        }[axis]

    symmetry_metrics: list[ChainSymmetryMetrics] = []
    for rule in sorted(contract.symmetry_thresholds, key=lambda item: item.symmetry_id):
        first = chain_by_id[rule.first_chain_id]
        second = chain_by_id[rule.second_chain_id]
        differences: list[tuple[str, float]] = []
        for axis in sorted(rule.axes):
            left = chain_axis(first, axis, rule.metric_space)
            right = chain_axis(second, axis, rule.metric_space)
            mean = (left + right) / 2.0
            difference = (
                0.0 if mean <= _ZERO_EXCURSION_TOLERANCE else abs(left - right) / mean
            )
            differences.append((axis, _canonical_float(difference)))
        maximum = max(value for _, value in differences)
        symmetry_metrics.append(
            ChainSymmetryMetrics(
                symmetry_id=rule.symmetry_id,
                first_chain_id=rule.first_chain_id,
                second_chain_id=rule.second_chain_id,
                metric_space=rule.metric_space,
                relative_difference_by_axis=tuple(differences),
                maximum_relative_difference=_canonical_float(maximum),
            )
        )
        if _is_above(maximum, rule.maximum_relative_difference):
            _issue(
                issues,
                "chain_symmetry_above_maximum",
                f"symmetries/{rule.symmetry_id}",
                (
                    f"value {_format_number(maximum)} violates threshold "
                    f"{_format_number(rule.maximum_relative_difference)}"
                ),
            )

    return MotionQAReport(
        status="fail" if issues else "pass",
        sample_count=sample_count,
        sample_rate_hz=_canonical_float(sample_rate),
        cyclic=contract.cyclic,
        chains=tuple(chain_metrics),
        groups=tuple(group_metrics),
        group_ratios=tuple(ratio_metrics),
        symmetries=tuple(symmetry_metrics),
        issues=tuple(sorted(set(issues))),
    )


__all__ = [
    "MOTION_QA_SCHEMA",
    "ChainMotionMetrics",
    "ChainMotionThresholds",
    "ChainSymmetryMetrics",
    "ChainSymmetryThreshold",
    "GroupExcursionRatioThreshold",
    "GroupMotionMetrics",
    "GroupRatioMetrics",
    "JointMotionMetrics",
    "JointMotionThresholds",
    "MotionQAContract",
    "MotionQAIssue",
    "MotionQAReport",
    "SemanticChainGroup",
    "SemanticChainSamples",
    "evaluate_motion_qa",
]
