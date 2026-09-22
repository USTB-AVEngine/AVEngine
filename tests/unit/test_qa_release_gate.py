"""The release gate: a declared standard, and a decision a production run can act on."""
import json

import pytest

from avengine.qa.release_gate import (
    DEFAULT_POLICY,
    evaluate_release,
    measure_spatial_support,
)


def _answer(question_id, azimuth=None):
    row = {"question_id": question_id, "qa_id": "QA-04", "evidence": {}}
    if azimuth is not None:
        row["evidence"]["azimuth_deg"] = azimuth
    return row


def _audit(*, constant=0.2, majority=0.3, binary=(), under_powered=()):
    by_qa = {}
    for qa in ("QA-04", "QA-06"):
        by_qa[qa] = {
            "mcq_questions": 10,
            "binary_mcq_warning": qa in binary,
            "validation_quota_status": "under_powered" if qa in under_powered else "pass",
            "by_split": {
                split: {"open_answers": {"majority_share": majority}}
                for split in ("train", "valid", "test")
            },
        }
    return {
        "schema": "avengine_answer_prior_audit_v1",
        "split_status": "complete",
        "by_qa": by_qa,
        "aggregate": {
            split: {"train_constant_micro_score": constant}
            for split in ("train", "valid", "test")
        },
    }


def test_a_source_outside_the_camera_angle_counts_as_off_screen():
    rows = [_answer("q1", 10.0), _answer("q2", -80.0), _answer("q3", 150.0), _answer("q4", None)]
    result = measure_spatial_support(rows, camera_half_fov_deg=42.5)
    # q4 carries no bearing, so it cannot be counted either way
    assert result["bank"]["bearing_questions"] == 3
    assert result["bank"]["off_screen_questions"] == 2
    assert result["bank"]["off_screen_share"] == pytest.approx(2 / 3)


def test_the_share_is_reported_per_split_with_its_own_denominator():
    rows = [_answer("q1", 10.0), _answer("q2", 90.0), _answer("q3", 5.0)]
    splits = {"q1": "valid", "q2": "valid", "q3": "test"}
    result = measure_spatial_support(rows, splits=splits)
    assert result["by_split"]["valid"]["off_screen_share"] == pytest.approx(0.5)
    assert result["by_split"]["test"]["off_screen_share"] == pytest.approx(0.0)
    assert result["by_split"]["train"]["off_screen_share"] is None


def test_a_bank_that_meets_every_rule_is_released():
    rows = [_answer(f"q{i}", 90.0) for i in range(8)] + [_answer("q8", 5.0)]
    splits = {f"q{i}": "valid" for i in range(9)} | {"q8": "valid"}
    result = evaluate_release(_audit(), rows, splits={**splits, "q0": "valid"})
    # every bearing row is in valid, so the test split has no denominator and blocks
    assert "off_screen_share:test" in result["blocked_rules"]
    assert "off_screen_share:valid" not in result["blocked_rules"]


def test_a_blind_constant_answer_above_the_ceiling_blocks_the_bank():
    rows = [_answer("q0", 90.0)]
    result = evaluate_release(_audit(constant=0.601), rows, splits={"q0": "valid"})
    assert result["status"] == "blocked"
    assert "blind_baseline:valid" in result["blocked_rules"]
    rule = next(r for r in result["rules"] if r["rule"] == "blind_baseline:valid")
    assert rule["measured"] == 0.601 and rule["limit"] == 0.35


def test_the_2026_09_22_bank_would_be_blocked_on_exactly_the_things_that_were_wrong():
    """The numbers this gate exists because of, as a regression against sliding back."""
    rows = [_answer(f"q{i}", 10.0) for i in range(81)] + [_answer(f"p{i}", 90.0) for i in range(19)]
    splits = {**{f"q{i}": "valid" for i in range(81)}, **{f"p{i}": "valid" for i in range(19)}}
    result = evaluate_release(
        _audit(constant=0.601, majority=1.0, binary=("QA-04",)), rows, splits=splits
    )
    assert result["status"] == "blocked"
    for expected in ("blind_baseline:valid", "off_screen_share:valid", "answer_majority:valid"):
        assert expected in result["blocked_rules"], expected
    spatial = result["spatial_support"]["by_split"]["valid"]
    assert spatial["off_screen_share"] == pytest.approx(0.19)


def test_a_policy_with_an_unknown_section_is_refused_rather_than_ignored():
    with pytest.raises(ValueError):
        evaluate_release(_audit(), [], policy={**DEFAULT_POLICY, "made_up": {}})


def test_a_bank_without_complete_splits_cannot_be_gated():
    audit = {**_audit(), "split_status": "partial"}
    with pytest.raises(ValueError):
        evaluate_release(audit, [])


def test_an_impossible_camera_angle_is_refused():
    for bad in (0, 180, -1, True, "42"):
        with pytest.raises(ValueError):
            measure_spatial_support([], camera_half_fov_deg=bad)
