"""A failed target has to stay in the accounting, and say which kind of failure it was.

Three things can end a request: the sampler legally refuses it, the request states
conditions that contradict each other, or the controller hits a defect. They call
for different work -- sample elsewhere, fix the config, fix the code -- so they
have to be distinguishable, and none of them may quietly leave the denominator.

The cases here are shaped the way tools/dataset/run_qa_batch.py produces them from
a real plan-only output directory, and are checked against that module's own
classifier and against the coverage provider's own failed-episode index. They are
hermetic: no controller is launched, nothing is rendered and no native budget is
touched.

The measured versions of these, taken from output directories real runs already
wrote, are in
tmp/binding_v1_parallel_20260910/TAKEOVER/CLAUDE_C06_R2/attempt_20260911T095703Z_pid1807042/
(FAILURE_CLASSIFICATION.json and PROVIDER_DENOMINATOR_READBACK.json).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from avengine.qa.batch_coverage import _failed_episode_index
from avengine.qa.failure_accounting import (
    EVIDENCE_GAP_STATE,
    INTERFACE_GAP_STATE,
    KNOWN_GAP_STATES,
    classify_failure,
)

REPOSITORY = Path(__file__).resolve().parents[2]

EXHAUSTION_HISTOGRAM = {
    "routes:initial_source_separation_below_0.95_m": 152,
    "camera:visibility_solver_candidate_budget_exhausted": 38,
}
CONFLICT_REASON = (
    "ConditionedRequestConflict: this request cannot be planned as one Episode: "
    "anchor_visibility: the request profile states 'in_fov' but QA-08 needs 'off_screen'"
)
ILLEGAL_VALUE_REASON = "ValueError: unsupported pixel_occlusion_transition"
CODE_DEFECT_REASON = "NameError: name 'condition_profile' is not defined"


@pytest.fixture(scope="module")
def batch():
    """tools/dataset/run_qa_batch.py, loaded by path like the repository's other tests."""
    path = REPOSITORY / "tools/dataset/run_qa_batch.py"
    spec = importlib.util.spec_from_file_location("run_qa_batch_failure_reporting", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def output_dir(tmp_path, name, *, planning_result=None, log_text=""):
    """One controller output directory, with only what such a failure leaves behind."""
    root = tmp_path / name
    root.mkdir(parents=True)
    if planning_result is not None:
        (root / "planning_result.json").write_text(
            json.dumps(planning_result), encoding="utf-8")
    log = root / "controller.log"
    log.write_text(log_text, encoding="utf-8")
    return root, log


def planning_result(reason, *, histogram=None, requested_profile=None):
    value = {
        "status": "failed",
        "condition_profile": None,
        "room_attempts": [{"room_id": "hm3d_val_00800_TEEsavR23oF",
                           "status": "not_selected", "reason": reason}],
        "gap_category": EVIDENCE_GAP_STATE,
    }
    if histogram is not None:
        value["failure_histogram"] = dict(histogram)
    if requested_profile is not None:
        value["requested_profile"] = dict(requested_profile)
    return value


def classify(batch, root, log):
    return batch.classify_controller_failure(
        episode_output_root=root, stderr_path=log, stdout_path=log, returncode=1)


def test_every_kind_of_failure_keeps_a_gap_state_the_accounting_recognises(batch, tmp_path):
    """Whatever ended the request, the target stays countable."""
    cases = {
        "exhausted": (planning_result("fixed condition profile exhausted",
                                      histogram=EXHAUSTION_HISTOGRAM), ""),
        "conflict": (planning_result(CONFLICT_REASON,
                                     requested_profile={"anchor_visibility": "in_fov"}),
                     CONFLICT_REASON),
        "illegal_value": (planning_result(ILLEGAL_VALUE_REASON), ILLEGAL_VALUE_REASON),
        "code_defect": (None, f"Traceback (most recent call last):\n{CODE_DEFECT_REASON}\n"),
    }
    for name, (result, log_text) in cases.items():
        root, log = output_dir(tmp_path, name, planning_result=result, log_text=log_text)
        classified = classify(batch, root, log)
        assert classified["gap_state"] in KNOWN_GAP_STATES, name
        assert classified["failure_stage"], name
        assert classified["reason_code"], name
        assert classified["diagnostic"], name


def test_a_missing_planning_result_does_not_take_the_target_out_of_the_accounting(
        batch, tmp_path):
    """The file carries the histogram and the exact reason, not the target's membership.

    An earlier controller defect left this file unwritten on every refusal, and it
    was tempting to read that as the whole target vanishing. It does not: an output
    with neither execution nor capture is a planning failure either way.
    """
    result = planning_result(CONFLICT_REASON,
                             requested_profile={"anchor_visibility": "in_fov"})
    with_file, log_a = output_dir(tmp_path, "with_file", planning_result=result,
                                  log_text=CONFLICT_REASON)
    without_file, log_b = output_dir(tmp_path, "without_file", planning_result=None,
                                     log_text=CONFLICT_REASON)

    before, after = classify(batch, with_file, log_a), classify(batch, without_file, log_b)
    assert before["gap_state"] == after["gap_state"]
    assert before["failure_stage"] == after["failure_stage"]
    assert before["reason_code"] == after["reason_code"]
    assert after["gap_state"] in KNOWN_GAP_STATES


def test_the_histogram_is_what_the_planning_result_adds(batch, tmp_path):
    """With the file the refusal names its own counts; without it, only the log line."""
    result = planning_result("fixed condition profile exhausted",
                             histogram=EXHAUSTION_HISTOGRAM)
    with_file, log_a = output_dir(tmp_path, "hist_with", planning_result=result)
    without_file, log_b = output_dir(
        tmp_path, "hist_without", planning_result=None,
        log_text="QAPlanningError: no existing room could realize the request")

    before = classify(batch, with_file, log_a)
    assert before["reason_code"] == "planning_exhausted"
    assert before["diagnostic"]["classification_reason"] == "planning_exhaustion"
    for key, count in EXHAUSTION_HISTOGRAM.items():
        assert key in before["failure_reason"]
        assert str(count) in before["failure_reason"]

    after = classify(batch, without_file, log_b)
    assert after["gap_state"] in KNOWN_GAP_STATES
    assert "initial_source_separation_below_0.95_m" not in after["failure_reason"]


def test_a_legal_refusal_and_a_broken_interface_are_not_the_same_record(batch, tmp_path):
    """Sampling elsewhere and fixing code are different jobs, so they read differently."""
    exhausted, log_a = output_dir(
        tmp_path, "legal_refusal",
        planning_result=planning_result("fixed condition profile exhausted",
                                        histogram=EXHAUSTION_HISTOGRAM))
    illegal, log_b = output_dir(tmp_path, "broken_interface",
                                planning_result=planning_result(ILLEGAL_VALUE_REASON),
                                log_text=ILLEGAL_VALUE_REASON)

    refusal, interface = classify(batch, exhausted, log_a), classify(batch, illegal, log_b)
    assert refusal["gap_state"] == EVIDENCE_GAP_STATE
    assert refusal["reason_code"] == "planning_exhausted"
    assert interface["gap_state"] == INTERFACE_GAP_STATE
    assert interface["diagnostic"]["classification_reason"] == "code_or_interface_exception"
    assert refusal["reason_code"] != interface["reason_code"]


def test_an_unrecognised_failure_is_kept_rather_than_dropped():
    """A reason nobody wrote a rule for still counts, and says so."""
    classified = classify_failure(failure_stage="planning",
                                  reason="a refusal phrased in a way no rule matches")
    assert classified["gap_state"] == EVIDENCE_GAP_STATE
    assert classified["gap_state"] in KNOWN_GAP_STATES
    assert classified["diagnostic"]["classification"] == "unclassified"
    assert classified["diagnostic"]["classification_reason"] == "no_matching_failure_rule"
    # It is kept and labelled, not silently folded into a recognised kind.
    assert classified["reason_code"] == "unclassified_failure"
    assert classified["failure_reason"]


def failed_row(episode_id, gap_state, *, room_id="hm3d_val_00800_TEEsavR23oF",
               asset_ids=("rocketbox_human_male_adult_01_top_blue_research_v1",)):
    row = {"episode_id": episode_id, "gap_state": gap_state,
           "failure_stage": "planning", "failure_reason": "measured elsewhere"}
    if room_id is not None:
        row["room_id"] = room_id
    if asset_ids is not None:
        row["asset_ids"] = list(asset_ids)
    return row


def test_a_complete_failure_row_reaches_the_coverage_table():
    rows = [failed_row("a", EVIDENCE_GAP_STATE, room_id="room_a", asset_ids=("asset_a",)),
            failed_row("b", INTERFACE_GAP_STATE, room_id="room_b", asset_ids=("asset_b",))]
    index = _failed_episode_index({"failed_episodes": rows})
    assert {record["episode_id"] for record in index.values()} == {"a", "b"}


def test_what_actually_removes_a_target_is_a_row_without_a_room_or_an_asset():
    """This, not a missing file, is how a failure leaves the per-asset table."""
    for name, row in {
        "no_room_id": failed_row("x", EVIDENCE_GAP_STATE, room_id=None),
        "no_asset_ids": failed_row("x", EVIDENCE_GAP_STATE, asset_ids=None),
        "empty_asset_ids": failed_row("x", EVIDENCE_GAP_STATE, asset_ids=()),
        "unknown_gap_state": failed_row("x", "something_else"),
    }.items():
        assert _failed_episode_index({"failed_episodes": [row]}) == {}, name


def test_one_cell_keeps_its_worst_failure_rather_than_all_of_them():
    """Several failures on one (room, asset) collapse by rank; that is a view, not a ledger."""
    rows = [failed_row("evidence", EVIDENCE_GAP_STATE),
            failed_row("interface", INTERFACE_GAP_STATE)]
    index = _failed_episode_index({"failed_episodes": rows})
    assert len(index) == 1
    assert next(iter(index.values()))["episode_id"] == "interface"


def test_a_stated_config_conflict_and_a_controller_defect_are_told_apart(batch, tmp_path):
    conflict, log_a = output_dir(
        tmp_path, "stated_conflict",
        planning_result=planning_result(
            CONFLICT_REASON, requested_profile={"anchor_visibility": "in_fov"}),
        log_text=CONFLICT_REASON)
    defect, log_b = output_dir(
        tmp_path, "controller_defect", planning_result=None,
        log_text=f"Traceback (most recent call last):\n{CODE_DEFECT_REASON}\n")

    stated, broken = classify(batch, conflict, log_a), classify(batch, defect, log_b)
    assert stated["reason_code"] == "conditioned_request_conflict"
    assert broken["reason_code"] == "controller_code_error"
    assert stated["reason_code"] != broken["reason_code"], (
        stated["reason_code"], broken["reason_code"])
    assert broken["gap_state"] == INTERFACE_GAP_STATE
