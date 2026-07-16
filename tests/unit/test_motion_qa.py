from __future__ import annotations

from dataclasses import replace
import json
import math

import numpy as np
import pytest

from avengine.motion.qa import (
    ChainMotionThresholds,
    ChainSymmetryThreshold,
    GroupExcursionRatioThreshold,
    JointMotionThresholds,
    MotionQAContract,
    SemanticChainGroup,
    SemanticChainSamples,
    evaluate_motion_qa,
)


def _rotations(degrees: tuple[float, ...]) -> np.ndarray:
    radians = np.radians(np.asarray(degrees, dtype=np.float64)) / 2.0
    return np.column_stack(
        (
            np.zeros(len(degrees)),
            np.zeros(len(degrees)),
            np.sin(radians),
            np.cos(radians),
        )
    )


def _chain(
    chain_id: str,
    joint_id: str,
    *,
    rest_length_m: float,
    forward_normalized: float,
    lateral_normalized: float = 0.05,
    vertical_normalized: float = 0.2,
    angles: tuple[float, ...] = (0.0, 10.0, 20.0, 10.0),
) -> SemanticChainSamples:
    scale = rest_length_m
    positions = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (
                forward_normalized * scale,
                lateral_normalized * scale,
                vertical_normalized * scale,
            ),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    return SemanticChainSamples(
        chain_id=chain_id,
        rest_length_m=rest_length_m,
        terminal_positions_flv_m=positions,
        joint_rotations_xyzw={joint_id: _rotations(angles)},
    )


def _contract() -> MotionQAContract:
    chain_ids = ("alpha_a", "alpha_b", "beta_a", "beta_b")
    joint_ids = {chain_id: (f"joint_{chain_id}",) for chain_id in chain_ids}
    chain_thresholds = {
        chain_id: ChainMotionThresholds(
            minimum_forward_excursion_normalized=0.35,
            maximum_lateral_excursion_normalized=0.1,
            maximum_lateral_to_forward_ratio=0.2,
        )
        for chain_id in chain_ids
    }
    joint_thresholds = {
        chain_id: {
            f"joint_{chain_id}": JointMotionThresholds(
                minimum_angular_excursion_degrees=10.0,
                maximum_angular_excursion_degrees=30.0,
                maximum_angular_speed_degrees_per_second=150.0,
            )
        }
        for chain_id in chain_ids
    }
    return MotionQAContract(
        required_chain_ids=chain_ids,
        required_joint_ids_by_chain=joint_ids,
        sample_rate_hz=10.0,
        cyclic=True,
        chain_groups=(
            SemanticChainGroup("alpha", ("alpha_a", "alpha_b")),
            SemanticChainGroup("beta", ("beta_a", "beta_b")),
        ),
        chain_thresholds=chain_thresholds,
        joint_thresholds_by_chain=joint_thresholds,
        group_ratio_thresholds=(
            GroupExcursionRatioThreshold(
                ratio_id="beta_to_alpha_forward",
                numerator_group_id="beta",
                numerator_axis="forward",
                denominator_group_id="alpha",
                denominator_axis="forward",
                minimum_ratio=0.7,
                maximum_ratio=0.9,
            ),
        ),
        symmetry_thresholds=(
            ChainSymmetryThreshold(
                symmetry_id="alpha_pair",
                first_chain_id="alpha_a",
                second_chain_id="alpha_b",
                maximum_relative_difference=0.01,
            ),
            ChainSymmetryThreshold(
                symmetry_id="beta_pair",
                first_chain_id="beta_a",
                second_chain_id="beta_b",
                maximum_relative_difference=0.01,
            ),
        ),
    )


def _passing_samples() -> tuple[SemanticChainSamples, ...]:
    return (
        _chain(
            "alpha_a",
            "joint_alpha_a",
            rest_length_m=2.0,
            forward_normalized=0.5,
        ),
        _chain(
            "alpha_b",
            "joint_alpha_b",
            rest_length_m=1.0,
            forward_normalized=0.5,
        ),
        _chain(
            "beta_a",
            "joint_beta_a",
            rest_length_m=1.6,
            forward_normalized=0.4,
        ),
        _chain(
            "beta_b",
            "joint_beta_b",
            rest_length_m=0.8,
            forward_normalized=0.4,
        ),
    )


def _issue_codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def test_motion_qa_reports_scale_normalized_metrics_and_stable_json() -> None:
    samples = _passing_samples()
    report = evaluate_motion_qa(samples, _contract())

    assert report.status == "pass"
    assert report.sample_count == 4
    assert tuple(item.chain_id for item in report.chains) == (
        "alpha_a",
        "alpha_b",
        "beta_a",
        "beta_b",
    )
    alpha_a, alpha_b = report.chains[:2]
    assert alpha_a.forward_excursion_m == pytest.approx(1.0)
    assert alpha_b.forward_excursion_m == pytest.approx(0.5)
    assert alpha_a.forward_excursion_normalized == pytest.approx(0.5)
    assert alpha_b.forward_excursion_normalized == pytest.approx(0.5)
    assert alpha_a.joints[0].angular_excursion_degrees == pytest.approx(20.0)
    assert alpha_a.joints[0].maximum_angular_speed_degrees_per_second == pytest.approx(
        100.0
    )
    assert report.group_ratios[0].value == pytest.approx(0.8)
    assert report.group_ratios[0].metric_space == "rest_length_normalized"
    assert all(item.maximum_relative_difference == 0.0 for item in report.symmetries)

    serialized = report.to_json()
    assert (
        serialized
        == evaluate_motion_qa(tuple(reversed(samples)), _contract()).to_json()
    )
    payload = json.loads(serialized)
    assert payload["coordinate_order"] == ["forward", "lateral", "vertical"]
    assert payload["formal_dataset_registration_authorized"] is False
    assert payload["status"] == "pass"


def test_normalized_chain_group_and_symmetry_thresholds_reject_bad_motion() -> None:
    samples = list(_passing_samples())
    samples[2] = _chain(
        "beta_a",
        "joint_beta_a",
        rest_length_m=1.6,
        forward_normalized=0.1,
        lateral_normalized=0.08,
    )
    report = evaluate_motion_qa(tuple(samples), _contract())

    assert report.status == "fail"
    assert {
        "chain_forward_excursion_normalized_below_minimum",
        "chain_lateral_to_forward_above_maximum",
        "chain_symmetry_above_maximum",
        "group_ratio_below_minimum",
    } <= _issue_codes(report)
    assert report.group_ratios[0].value == pytest.approx(0.5)


def test_missing_or_unexpected_chain_fails_without_partial_metrics() -> None:
    samples = list(_passing_samples())
    samples.pop()
    samples.append(
        _chain(
            "unknown",
            "joint_unknown",
            rest_length_m=1.0,
            forward_normalized=0.4,
        )
    )
    report = evaluate_motion_qa(tuple(samples), _contract())

    assert report.status == "fail"
    assert report.sample_count is None
    assert report.chains == ()
    assert {"missing_required_chain", "unexpected_chain"} <= _issue_codes(report)


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    (
        (
            lambda item: replace(
                item,
                terminal_positions_flv_m=np.asarray(
                    ((0.0, 0.0, 0.0), (math.nan, 0.0, 0.0))
                ),
            ),
            "invalid_terminal_trajectory",
        ),
        (
            lambda item: replace(
                item,
                terminal_positions_flv_m=np.zeros((4, 2), dtype=np.float64),
            ),
            "invalid_terminal_trajectory",
        ),
        (
            lambda item: replace(item, rest_length_m=0.0),
            "invalid_rest_length",
        ),
        (
            lambda item: replace(
                item,
                joint_rotations_xyzw={
                    "joint_alpha_a": np.ones((4, 4), dtype=np.float64)
                },
            ),
            "invalid_joint_rotations",
        ),
        (
            lambda item: replace(
                item,
                joint_rotations_xyzw={"joint_alpha_a": _rotations((0.0, 10.0, 20.0))},
            ),
            "sample_count_mismatch",
        ),
    ),
)
def test_numeric_contract_defects_fail_closed(mutate, expected_code: str) -> None:
    samples = list(_passing_samples())
    samples[0] = mutate(samples[0])
    report = evaluate_motion_qa(tuple(samples), _contract())

    assert report.status == "fail"
    assert expected_code in _issue_codes(report)
    assert report.chains == ()


def test_joint_geodesic_excursion_and_speed_threshold_are_explicit() -> None:
    samples = (
        SemanticChainSamples(
            chain_id="chain",
            rest_length_m=2.0,
            terminal_positions_flv_m=np.asarray(
                ((0.0, 0.0, 0.0), (1.0, 0.1, 0.2), (0.0, 0.0, 0.0))
            ),
            joint_rotations_xyzw={"joint": _rotations((0.0, 30.0, 60.0))},
        ),
    )
    contract = MotionQAContract(
        required_chain_ids=("chain",),
        required_joint_ids_by_chain={"chain": ("joint",)},
        sample_rate_hz=10.0,
        cyclic=False,
        joint_thresholds_by_chain={
            "chain": {
                "joint": JointMotionThresholds(
                    maximum_angular_speed_degrees_per_second=299.0
                )
            }
        },
    )
    report = evaluate_motion_qa(samples, contract)

    assert report.status == "fail"
    metrics = report.chains[0].joints[0]
    assert metrics.angular_excursion_degrees == pytest.approx(60.0)
    assert metrics.maximum_angular_speed_degrees_per_second == pytest.approx(300.0)
    assert "joint_angular_speed_above_maximum" in _issue_codes(report)


def test_zero_forward_excursion_uses_json_null_and_fails_ratio_gate() -> None:
    sample = SemanticChainSamples(
        chain_id="chain",
        rest_length_m=1.0,
        terminal_positions_flv_m=np.asarray(((0.0, 0.0, 0.0), (0.0, 0.2, 0.1))),
    )
    contract = MotionQAContract(
        required_chain_ids=("chain",),
        required_joint_ids_by_chain={"chain": ()},
        sample_rate_hz=30.0,
        cyclic=True,
        chain_thresholds={
            "chain": ChainMotionThresholds(maximum_lateral_to_forward_ratio=0.2)
        },
    )
    report = evaluate_motion_qa((sample,), contract)

    assert report.status == "fail"
    assert report.chains[0].lateral_to_forward_ratio is None
    assert "chain_lateral_to_forward_undefined" in _issue_codes(report)
    assert json.loads(report.to_json())["chains"][0]["lateral_to_forward_ratio"] is None


def test_non_finite_threshold_contract_fails_before_metrics() -> None:
    contract = replace(
        _contract(),
        chain_thresholds={
            "alpha_a": ChainMotionThresholds(
                maximum_lateral_excursion_normalized=math.inf
            )
        },
    )
    report = evaluate_motion_qa(_passing_samples(), contract)

    assert report.status == "fail"
    assert report.chains == ()
    assert "invalid_threshold" in _issue_codes(report)
