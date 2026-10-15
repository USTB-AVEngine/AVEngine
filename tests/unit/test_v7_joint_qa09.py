"""Synthetic preflight regressions; only native production counts as delivery."""
from copy import deepcopy

import pytest

from avengine.construction.joint.families.qa09 import (
    ReappearanceDemand, certify_mask_sequence, requested_answer,
)

pytestmark = pytest.mark.fast_unit


def sequence(answer="yes"):
    states = ["visible_clear"] * 45 + ["fully_occluded"] * 30
    states += ["visible_clear" if answer == "yes" else "fully_occluded"] * 75
    frames = [{"frame_index": i, "state": state, "target_pixels": 1000,
               "visible_pixels": 0 if state == "fully_occluded" else 1000}
              for i, state in enumerate(states)]
    occ = [{"frame_index": f["frame_index"], "target_instance_id": "person_id",
            "occluder_instance_ids": ["speaker_id"], "unknown_occluder_pixels": 0,
            "target_only_pixels": 1000, "context_pixels": 0, "occluded_pixels": 1000}
           for f in frames if f["state"] == "fully_occluded"]
    return frames, occ


def certify(frames, occ, answer="yes", **kwargs):
    return certify_mask_sequence(frames, occ, ReappearanceDemand("person", "speaker", answer, **kwargs),
        target_instance_id="person_id", occluder_instance_id="speaker_id",
        registered_instance_ids=["person_id", "speaker_id"])


@pytest.mark.parametrize("answer", ["yes", "no"])
def test_complete_registered_sequence_certifies_its_actual_branch(answer):
    f, o = sequence(answer)
    result = certify(f, o, answer)
    assert result["status"] == "pass"
    assert result["observed_answer"] == answer
    assert result["clear_before_intervals_frames"] == [[0, 44]]


def test_no_branch_rejects_even_one_late_partial_visible_frame():
    f, o = sequence("no")
    f[149].update(state="visible_occluded", visible_pixels=1)
    result = certify(f, o, "no")
    assert result["status"] == "fail"
    assert result["observed_answer"] == "yes"
    assert "observed_branch_differs_from_requested" in result["errors"]


def test_transient_early_hide_already_changes_the_original_no_answer():
    f, o = sequence("no")
    f[1].update(state="fully_occluded", visible_pixels=0)
    result = certify(f, o, "no")
    assert result["observed_answer"] == "yes"
    assert result["status"] == "fail"


@pytest.mark.parametrize("change", ["missing_end", "duplicate", "reordered"])
def test_sparse_or_duplicate_stream_cannot_certify_absence(change):
    f, o = sequence("no")
    if change == "missing_end":
        f.pop()
    elif change == "duplicate":
        f[60] = deepcopy(f[59])
    else:
        f[59], f[60] = f[60], f[59]
    result = certify(f, o, "no")
    assert result["status"] == "fail"
    assert result["observed_answer"] is None


@pytest.mark.parametrize("field,value", [
    ("occluder_instance_ids", []), ("occluder_instance_ids", ["stage"]),
    ("occluder_instance_ids", ["speaker_id", "other"]),
    ("unknown_occluder_pixels", 1), ("target_only_pixels", 999),
    ("context_pixels", 1), ("occluded_pixels", 999),
])
def test_full_occlusion_needs_unique_registered_mask_evidence(field, value):
    f, o = sequence()
    for r in o:
        r[field] = value
    result = certify(f, o)
    assert result["status"] == "fail"
    assert "no_stable_registered_full_occlusion" in result["errors"]


def test_out_of_view_is_never_full_occlusion():
    f, o = sequence()
    for r in f[45:75]:
        r.update(state="out_of_view", target_pixels=0)
    result = certify(f, o)
    assert "no_fully_occluded_frame" in result["errors"]


def test_exactly_one_second_passes_but_fourteen_frames_do_not():
    f, o = sequence()
    for r in f[60:75]:
        r.update(state="visible_clear", visible_pixels=1000)
    assert certify(f, o)["status"] == "pass"
    f[59].update(state="visible_clear", visible_pixels=1000)
    assert "no_stable_registered_full_occlusion" in certify(f, o)["errors"]


def test_visibility_requires_a_stable_clear_interval_not_one_reappearance_pixel():
    f, o = sequence()
    for r in f[75:]:
        r.update(state="visible_occluded", visible_pixels=1)
    result = certify(f, o)
    assert result["observed_answer"] == "yes"
    assert "no_stable_clear_reappearance" in result["errors"]


def test_no_branch_must_have_been_seen_before_becoming_hidden():
    f, o = sequence("no")
    for r in f[:45]:
        r.update(state="out_of_view", visible_pixels=0, target_pixels=0)
    assert "no_stable_visibility_before_first_occlusion" in certify(f, o, "no")["errors"]


def test_no_branch_stays_in_the_registered_shadow_through_clip_end():
    f, o = sequence("no")
    f[149].update(state="out_of_view", visible_pixels=0, target_pixels=0)
    result = certify(f, o, "no")
    assert result["observed_answer"] == "no"
    assert "negative_branch_not_hidden_through_window_end" in result["errors"]


def test_requests_balance_actual_counts_and_keep_negative_branch_available():
    assert requested_answer({}, 0) == "yes"
    assert requested_answer({}, 1) == "no"
    assert requested_answer({"answer_counts": {"QA-09": {"yes": 8, "no": 2}}}) == "no"
    assert requested_answer({"family_answer_counts": {"QA-09": {"yes": 0, "no": 2}}}) == "yes"
    state = {"answer_branch_distribution": {"families": {"QA-09": {"answers": {"yes": 4, "no": 1}}}}}
    assert requested_answer(state) == "no"


def test_negative_relation_does_not_require_a_later_clear_frame():
    relation = ReappearanceDemand("person", "speaker", "no").relation()
    assert relation["qa09_demand"]["answer"] == "no"
    assert "later_clear_time_s" not in relation


@pytest.mark.parametrize("kwargs", [{"window_start_s": 21}, {"window_start_s": .01},
    {"minimum_hidden_s": 0}, {"frame_rate_hz": 0}, {"window_duration_s": float("nan")}])
def test_invalid_request_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ReappearanceDemand("person", "speaker", "no", **kwargs)
