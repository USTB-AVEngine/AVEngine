"""Compilation from QA type and key branch to actual generation conditions.

The negative cases run on retained native facts, so a refusal here is the
refusal the shipped question generator will make on the same input.  The
positive controls are activity and geometry fixtures derived from those same
facts: only the motion, the pixel visibility states or the source positions
change, every other field stays as it was measured.  Deriving them this way
keeps the fact schema real and proves the checks are not simply always failing,
which a fixture-free negative-only suite could not show.

A fixture is not a rendered episode and is never counted as dataset coverage.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from avengine.dataset import source_capabilities as capabilities
from avengine.qa import generation_conditions as gc
from avengine.qa import unified_catalog as catalog
from avengine.qa.batch_coverage import COVERAGE_STATES
from avengine.qa.unified_catalog import QA_IDS

pytestmark = pytest.mark.fast_unit

REPOSITORY = Path(__file__).resolve().parents[2]
RETAINED = REPOSITORY / (
    "tmp/binding_dataset_20260909_v2/binding_delivery_initial16_v1/native/facts/facts_0009.json"
)
RETAINED_STATIC = REPOSITORY / (
    "tmp/binding_dataset_20260909_v2/binding_delivery_v4/native/facts/facts_0001.json"
)

# The retained episode this suite reads: two speaking humans, a fixed camera, one
# 10 second clip at 15 fps, events at frames 3..39 and 70..103, and a single
# moving run at frames 43..67 that lies inside the first event's measured wet
# tail rather than after it.
ACTORS = {"human_blue": "source1", "human_green": "source2"}
INSTANCES = [
    {
        "instance_id": "human_blue",
        "source_class": "articulated_human",
        "asset_id": "rocketbox_human_male_adult_01_top_blue_research_v1",
        "role": "anchor",
    },
    {
        "instance_id": "human_green",
        "source_class": "articulated_human",
        "asset_id": "rocketbox_human_male_adult_01_top_green_research_v1",
    },
]
DEVICE_INSTANCES = [
    {"instance_id": "speaker01", "source_class": "rigid_static_object", "asset_id": "device_a"},
    {"instance_id": "human_green", "source_class": "articulated_human", "asset_id": "human_b"},
]

FIRST_EVENT_FRAMES = (3, 39)
SECOND_EVENT_FRAMES = (70, 103)
# ceil(8.217625 * 15) + 1, the first frame a cross_time_state recipe may move in.
SECOND_EVENT_FIRST_LEGAL_MOTION_FRAME = 125


def _retained(path: Path = RETAINED) -> dict:
    if not path.is_file():
        pytest.skip(f"retained facts are not present in this checkout: {path}")
    return json.loads(path.read_text())


def _target(qa_id: str, *, instance: str = "human_blue", event: dict | None = None,
            others: tuple[str, ...] = ()) -> dict:
    """Name as many instances as this question is actually about."""

    scope = gc.subject_scope(qa_id)
    named = [instance]
    if scope != "single_target":
        named += [value for value in others if value != instance]
    row = {"qa_id": qa_id, "target_instance_ids": named}
    if event is not None:
        row["event"] = event
    return row


def _compile(qa_id, branch=None, *, instances=None, instance="human_blue", task_family=None,
             event=None, capabilities=None, backend=None):
    rows = INSTANCES if instances is None else instances
    others = tuple(row["instance_id"] for row in rows)
    return gc.compile_generation_conditions(
        _target(qa_id, instance=instance, event=event, others=others),
        branch=branch,
        instances=rows,
        task_family=task_family,
        capabilities=capabilities,
        backend=backend,
    )


def _check(compiled, facts, *, actors=None):
    return gc.check_conditions(compiled, facts, actor_by_instance=actors or ACTORS)


def _failed(result) -> set[str]:
    return {
        row["key"]
        for attempt in result["attempts"]
        for row in attempt["checks"]
        if row["status"] == "fail"
    }


# ----------------------------------------------------------------- fixture helpers


def _recompute_motion(facts: dict, actor: str) -> None:
    """Rebuild the motion flags from the positions the way the catalog does."""

    fps = float(facts["time"]["frame_rate_hz"])
    positions = facts["actors"][actor]["root_positions_m"]
    speeds, moving = catalog._motion_series(positions, frame_rate_hz=fps)
    assert moving is not None
    facts["actors"][actor]["moving"] = moving
    facts["actors"][actor]["speed_mps"] = speeds


def _displace_towards_listener(
    facts: dict, actor: str, first: int, last: int, total_m: float
) -> None:
    """Ramp one actor along the listener direction over [first, last] positions.

    A negative total moves the actor away.  Positions after ``last`` hold the
    final offset, so the derived motion flags are true exactly on the ramp.
    """

    listener = facts["listener"]["positions_m"]
    root = facts["actors"][actor]["root_positions_m"]
    emitter = facts["actors"][actor].get("emitter_positions_m")
    origin = list(root[first])
    reference = listener[min(first, len(listener) - 1)]
    vector = [float(reference[i]) - float(origin[i]) for i in range(3)]
    vector[1] = 0.0
    length = math.sqrt(sum(value * value for value in vector))
    assert length > 1.0e-6, "the actor already stands on the listener"
    unit = [value / length for value in vector]
    steps = max(1, last - first)
    for index in range(first, len(root)):
        travelled = total_m * min(1.0, max(0, index - first) / steps)
        offset = [unit[axis] * travelled for axis in range(3)]
        for series in (root, emitter):
            if series is None:
                continue
            base = series[first]
            series[index] = [float(base[axis]) + offset[axis] for axis in range(3)]
    _recompute_motion(facts, actor)


def _hold_still(facts: dict, actor: str) -> None:
    root = facts["actors"][actor]["root_positions_m"]
    emitter = facts["actors"][actor].get("emitter_positions_m")
    for series in (root, emitter):
        if series is None:
            continue
        first = list(series[0])
        for index in range(len(series)):
            series[index] = list(first)
    _recompute_motion(facts, actor)


def _set_visibility(facts: dict, actor: str, frames: range, state: str, *, centroid_x=None) -> None:
    rows = facts["visibility"][actor]
    for frame in frames:
        key = str(frame) if str(frame) in rows else frame
        row = rows.get(key)
        assert row is not None, f"frame {frame} has no retained visibility row"
        row["state"] = state
        row["frame_index"] = int(frame)
        if state == "out_of_view":
            row["in_fov"] = False
            row["visible_pixels"] = 0
            row["visible_fraction"] = 0.0
            row.pop("target_centroid_xy_px", None)
            row.pop("visible_centroid_xy_px", None)
        elif state == "fully_occluded":
            row["in_fov"] = True
            row["visible_pixels"] = 0
            row["visible_fraction"] = 0.0
            row["occlusion_fraction"] = 1.0
            row.pop("visible_centroid_xy_px", None)
        else:
            row["in_fov"] = True
            row["occlusion_fraction"] = 0.0 if state == "visible_clear" else 0.5
            row["visible_fraction"] = 1.0 if state == "visible_clear" else 0.5
            if centroid_x is not None:
                centroid = [float(centroid_x), 471.0]
                row["target_centroid_xy_px"] = centroid
                row["visible_centroid_xy_px"] = list(centroid)


@pytest.fixture
def retained() -> dict:
    return _retained()


@pytest.fixture
def moving_during_first_event() -> dict:
    """The target walks 1.2 m towards the listener across its first event."""

    facts = _retained()
    start, end = FIRST_EVENT_FRAMES
    _displace_towards_listener(facts, "source1", start, end, 1.2)
    _hold_still(facts, "source2")
    return facts


@pytest.fixture
def moving_after_second_tail() -> dict:
    """The target moves only after the second event's measured wet tail."""

    facts = _retained()
    _hold_still(facts, "source1")
    _hold_still(facts, "source2")
    _displace_towards_listener(
        facts, "source1", SECOND_EVENT_FIRST_LEGAL_MOTION_FRAME + 1, 135, -1.2
    )
    return facts


@pytest.fixture
def enters_from_the_left() -> dict:
    """The target is off frame for two seconds, then enters on the left."""

    facts = _retained()
    _set_visibility(facts, "source1", range(0, 30), "out_of_view")
    _set_visibility(facts, "source1", range(30, 150), "visible_clear", centroid_x=200.0)
    return facts


def _fully_occluded_fixture(*, reappears: bool) -> dict:
    facts = _retained()
    if reappears:
        _set_visibility(facts, "source1", range(0, 40), "visible_clear", centroid_x=600.0)
        _set_visibility(facts, "source1", range(40, 70), "fully_occluded")
        _set_visibility(facts, "source1", range(70, 150), "visible_clear", centroid_x=600.0)
    else:
        _set_visibility(facts, "source1", range(0, 100), "visible_clear", centroid_x=600.0)
        _set_visibility(facts, "source1", range(100, 150), "fully_occluded")
    return facts


# ------------------------------------------------------------------ traceable table


def test_every_qa_type_and_branch_compiles_to_a_traceable_result():
    matrix = gc.compile_catalog_matrix(instances=INSTANCES)
    rows = matrix["compiled"]
    seen = {row["qa_id"] for row in rows}
    assert seen == set(QA_IDS), "every catalog type needs a compiled result"

    expected_rows = sum(len(gc.branches_for(qa_id)) or 1 for qa_id in QA_IDS)
    assert len(rows) == expected_rows

    for row in rows:
        assert row["state"] in gc.APPLICABILITY_STATES
        assert row["conditions"], f"{row['qa_id']} compiled to no conditions at all"
        if row["state"] != capabilities.STATE_AVAILABLE:
            assert row["reason"], f"{row['qa_id']} is blocked without a stated reason"
        # A compiled row names scene conditions, not a candidate id list.
        kinds = {item["kind"] for item in row["conditions"]}
        assert kinds - {"candidate_domain", "required_modality"}, (
            f"{row['qa_id']} compiled to nothing but a requirement restatement"
        )


def test_gap_states_share_one_vocabulary_with_coverage_accounting():
    for state in (
        capabilities.STATE_NOT_APPLICABLE,
        capabilities.STATE_NOT_IMPLEMENTED,
        capabilities.STATE_EVIDENCE_MISSING,
    ):
        assert state in COVERAGE_STATES
    assert gc.APPLICABILITY_STATES == capabilities.CAPABILITY_STATES


def test_every_condition_kind_has_a_readback_check():
    assert not [kind for kind in gc.CONDITION_KINDS if kind not in gc._CHECKERS]


def test_the_five_identities_stay_apart():
    compiled = _compile("QA-06", "still")
    subject = next(item for item in compiled.subjects if item.role == "target")
    assert subject.entity_instance_id == "human_blue"
    assert subject.asset_id == INSTANCES[0]["asset_id"]
    assert subject.entity_instance_id != subject.asset_id
    payload = subject.to_dict()
    assert "entity_instance_id" in payload and "asset_id" in payload
    # A role attaches to the instance, and the anchor role is not the first event.
    event = next(item for item in compiled.conditions if item.kind == "event_selection")
    assert event.detail["selector"]["kind"] == "target_audible_window"
    assert "not the first" in event.detail["note"]


# -------------------------------------------------------- the branch changes things


@pytest.mark.parametrize("qa_id", sorted(gc.BRANCHES))
def test_each_key_branch_compiles_to_a_different_condition_set(qa_id):
    seen: dict[str, tuple] = {}
    for branch in gc.branches_for(qa_id):
        compiled = _compile(qa_id, branch)
        signature = tuple(
            sorted(
                (item.key, json.dumps(item.detail, sort_keys=True, default=str),
                 json.dumps(item.planning, sort_keys=True, default=str),
                 json.dumps(item.evidence, sort_keys=True, default=str))
                for item in compiled.conditions
            )
        )
        for other, previous in seen.items():
            assert signature != previous, (
                f"{qa_id} branches {branch} and {other} compile to identical conditions"
            )
        seen[branch] = signature
    assert len(seen) == len(gc.branches_for(qa_id))


def test_changing_the_qa_type_changes_the_necessary_conditions():
    during = _compile("QA-06", "moving")
    after = _compile("QA-17", "yes", task_family="cross_time_state")
    during_placement = {
        item.detail.get("placement")
        for item in during.conditions
        if item.kind == "motion_window_placement"
    }
    after_placement = {
        item.detail.get("placement")
        for item in after.conditions
        if item.kind == "motion_window_placement"
    }
    assert during_placement == {gc.MOTION_DURING_EVENT}
    assert after_placement == {gc.MOTION_AFTER_TAIL}
    assert during.sampler_profile()["speech_motion"] == "speaker_moving"
    assert after.sampler_profile()["speech_motion"] == "all_still"


def test_both_opposite_motion_conditions_survive_and_neither_is_rewritten():
    """QA-06 moving and cross_time_state must keep their own requirement."""

    moving = _compile("QA-06", "moving")
    still = _compile("QA-06", "still")
    cross = _compile("QA-13", task_family="cross_time_state")
    plain = _compile("QA-13")

    assert moving.sampler_profile()["speech_motion"] == "speaker_moving"
    assert still.sampler_profile()["speech_motion"] == "all_still"
    # The recipe requirement belongs to the recipe, not to QA-13 in general.
    assert cross.sampler_profile()["speech_motion"] == "all_still"
    assert "speech_motion" not in plain.sampler_profile()
    assert any(
        item.detail.get("placement") == gc.MOTION_AFTER_TAIL
        for item in cross.conditions
    )
    assert not any(item.kind == "motion_window_placement" for item in plain.conditions)


# ------------------------------------------------------------- conflicts are refused


def test_two_branches_of_one_type_on_one_instance_conflict():
    conflicts = gc.reject_conflicts([_compile("QA-06", "moving"), _compile("QA-06", "still")])
    knobs = {row["knob"] for row in conflicts}
    assert "speech_motion" in knobs
    subjects = {row["subject"] for row in conflicts}
    assert "human_blue" in subjects


def test_distance_trend_and_a_still_target_conflict():
    conflicts = gc.reject_conflicts([_compile("QA-15", "nearer"), _compile("QA-06", "still")])
    assert any(row["knob"] == "speech_motion" for row in conflicts)


def test_a_plain_episode_may_carry_motion_during_and_after_the_sound():
    """These are compatible without a recipe that places the motion window."""

    conflicts = gc.reject_conflicts([_compile("QA-06", "moving"), _compile("QA-17", "yes")])
    assert conflicts == []


def test_the_cross_time_recipe_refuses_motion_during_the_sound():
    compiled = _compile("QA-06", "moving", task_family="cross_time_state")
    assert compiled.state == capabilities.STATE_NOT_APPLICABLE
    reasons = {row["reason"] for row in compiled.conflicts}
    assert "recipe_places_motion_after_the_measured_tail" in reasons

    pair = gc.reject_conflicts([
        _compile("QA-06", "moving", task_family="cross_time_state"),
        _compile("QA-17", "yes", task_family="cross_time_state"),
    ])
    assert {row["knob"] for row in pair} >= {"motion_window_placement", "speech_motion"}


def test_a_counting_question_refuses_a_single_event_selector():
    counting = _compile("QA-23", event={"kind": "target_audible_window"})
    assert counting.state == capabilities.STATE_NOT_APPLICABLE
    assert "whole clip" in counting.reason
    whole = _compile("QA-23", event={"kind": "whole_clip"})
    assert whole.state == capabilities.STATE_AVAILABLE


def test_an_anchored_question_refuses_a_whole_clip_selector():
    compiled = _compile("QA-17", "yes", event={"kind": "whole_clip"})
    assert compiled.state == capabilities.STATE_NOT_APPLICABLE
    assert "anchor" in compiled.reason


# ------------------------------------------------------ a device keeps its other roles


@pytest.mark.parametrize("qa_id,branch", [("QA-06", "moving"), ("QA-15", "nearer"), ("QA-17", "yes")])
def test_a_device_is_not_a_self_motion_target(qa_id, branch):
    compiled = _compile(qa_id, branch, instances=DEVICE_INSTANCES, instance="speaker01")
    assert compiled.state == capabilities.STATE_NOT_APPLICABLE
    assert "never walks by itself" in compiled.reason


@pytest.mark.parametrize("qa_id", ["QA-01", "QA-12", "QA-21"])
def test_a_device_still_holds_every_other_role(qa_id):
    compiled = _compile(qa_id, instances=DEVICE_INSTANCES, instance="speaker01")
    assert compiled.state == capabilities.STATE_AVAILABLE, compiled.reason


def test_an_inapplicable_candidate_does_not_swallow_a_sibling():
    device = _compile("QA-06", "moving", instances=DEVICE_INSTANCES, instance="speaker01")
    human = _compile("QA-06", "moving", instances=DEVICE_INSTANCES, instance="human_green")
    report = gc.condition_report([device, human])
    row = report["qa_ids"]["QA-06"]
    assert len(row["candidates"]) == 2
    by_target = {candidate["targets"][0]: candidate for candidate in row["candidates"]}
    assert by_target["speaker01"]["state"] == capabilities.STATE_NOT_APPLICABLE
    assert "never walks by itself" in by_target["speaker01"]["reason"]
    # The human candidate keeps its own verdict rather than inheriting the device's.
    assert by_target["human_green"]["state"] == capabilities.STATE_AVAILABLE
    assert row["any_available"] is True


def test_a_derived_target_naming_two_speakers_fans_out_per_candidate():
    """production_spec names every speaking instance; each keeps its own row."""

    derived = {"qa_id": "QA-06", "target_instance_ids": ["human_blue", "human_green"]}
    with pytest.raises(gc.GenerationConditionError) as error:
        gc.compile_generation_conditions(derived, branch="moving", instances=INSTANCES)
    assert "one instance at a time" in str(error.value)

    rows = gc.compile_target_candidates(derived, branch="moving", instances=INSTANCES)
    assert [row.target_instance_ids for row in rows] == [("human_blue",), ("human_green",)]
    for row in rows:
        competitors = {item.entity_instance_id for item in row.subjects if item.role == "competitor"}
        assert competitors == {"human_blue", "human_green"} - set(row.target_instance_ids)


def test_a_pairwise_question_fans_out_over_pairs():
    three = INSTANCES + [
        {"instance_id": "human_red", "source_class": "articulated_human", "asset_id": "human_c"}
    ]
    derived = {"qa_id": "QA-14", "target_instance_ids": [row["instance_id"] for row in three]}
    rows = gc.compile_target_candidates(derived, instances=three)
    assert [row.target_instance_ids for row in rows] == [
        ("human_blue", "human_green"),
        ("human_blue", "human_red"),
        ("human_green", "human_red"),
    ]


def test_a_candidate_set_question_keeps_one_row_for_the_whole_set():
    derived = {"qa_id": "QA-18", "target_instance_ids": ["human_blue", "human_green"]}
    rows = gc.compile_target_candidates(derived, instances=INSTANCES)
    assert len(rows) == 1
    assert rows[0].target_instance_ids == ("human_blue", "human_green")


def test_the_report_names_the_branches_that_were_never_attempted():
    report = gc.condition_report([_compile("QA-17", "yes")])
    row = report["qa_ids"]["QA-17"]
    assert row["branches_expected"] == ["yes", "no"]
    assert row["branches_missing"] == ["no"]


# --------------------------------------------------- the integer second query window


def test_integer_second_window_matches_the_catalog_quantisation(retained):
    fps = float(retained["time"]["frame_rate_hz"])
    for window in ([63, 71], [124, 150], [0, 150], [63, 64]):
        mine = gc.integer_second_window(window[0] / fps, window[1] / fps, precision=0)
        theirs = catalog._display_time_bounds(retained, window)
        if theirs is None:
            assert mine is None, window
        else:
            assert mine is not None and mine == pytest.approx(theirs), window


def test_a_short_post_sound_window_cannot_be_published(retained):
    """The measured QA-17 yes interval on this episode is under one second."""

    result = _check(_compile("QA-17", "yes"), retained)
    rows = [
        row
        for attempt in result["attempts"]
        if attempt["event_id"] == "event_001"
        for row in attempt["checks"]
        if row["key"] == "motion_after_sound"
    ]
    assert rows and rows[0]["status"] == "fail"
    assert "too short to state at the public time precision" in rows[0]["reason"]
    runs = rows[0]["measured"]["stable_runs"]
    assert [run["value"] for run in runs] == ["yes"]
    assert runs[0]["public_s"] is None
    assert runs[0]["start"] == 63 and runs[0]["end"] == 71


# ----------------------------------------------- the five zero output types, measured


def test_qa06_on_retained_facts_is_still_and_shares_its_answer(retained):
    moving = _check(_compile("QA-06", "moving"), retained)
    assert moving["status"] == "fail"
    assert {"motion_during_event", "answer_distinguishable"} <= _failed(moving)

    # The still branch used to pass here, which is what the name of this test
    # already denied: every retained candidate is still, so the answer is shared.
    # It now states a competitor whose motion answer differs, so the shared
    # answer is refused while it is still a condition rather than after
    # generation, where the catalog gate deferred 137 of 148 QA-06 candidates.
    still = _check(_compile("QA-06", "still"), retained)
    assert still["status"] == "fail"
    assert "answer_distinguishable" in _failed(still)
    shared = [
        row
        for attempt in still["attempts"]
        for row in attempt["checks"]
        if row["key"] == "answer_distinguishable" and row["status"] == "fail"
    ]
    assert "share the target's answer" in shared[0]["reason"]


def test_qa07_has_no_entry_transition_in_the_retained_pixels(retained):
    result = _check(_compile("QA-07", "left"), retained)
    assert result["status"] == "fail"
    rows = [
        row
        for attempt in result["attempts"]
        for row in attempt["checks"]
        if row["key"] == "entry_transition"
    ]
    assert rows[0]["status"] == "fail"
    assert "out_of_view" in rows[0]["reason"]
    assert "out_of_view" not in rows[0]["measured"]["observed_states_in_episode"]


@pytest.mark.parametrize("branch", ["yes", "no"])
def test_qa09_has_no_full_occlusion_in_the_retained_pixels(retained, branch):
    result = _check(_compile("QA-09", branch), retained)
    assert result["status"] == "fail"
    rows = [
        row
        for attempt in result["attempts"]
        for row in attempt["checks"]
        if row["key"] == "occlusion_transition"
    ]
    assert rows[0]["status"] == "fail"
    # A blocked ray is a different measurement and cannot stand in for a pixel state.
    assert "line-of-sight ray" in rows[0]["reason"]
    assert rows[0]["measured"]["fully_occluded_frames"] == []


def test_qa15_measures_a_zero_net_change_on_retained_facts(retained):
    result = _check(_compile("QA-15", "nearer"), retained)
    assert result["status"] == "fail"
    rows = [
        row
        for attempt in result["attempts"]
        for row in attempt["checks"]
        if row["key"] == "distance_net_change"
    ]
    measured = rows[0]["measured"]
    assert rows[0]["status"] == "fail"
    assert measured["endpoint_delta_m"] == pytest.approx(0.0, abs=1.0e-9)
    assert measured["margin_m"] == gc.DISTANCE_MARGIN_M
    # The shipped judge compares endpoints, so monotonicity is reported beside it
    # rather than in place of it.
    assert measured["judge_is_endpoint_only"] is True
    assert "monotonic_over_interval" in measured


def test_qa17_both_branches_exist_in_one_retained_episode_for_different_reasons(retained):
    """The yes branch is blocked by the display precision, the no branch by the gate."""

    yes = _check(_compile("QA-17", "yes"), retained)
    no = _check(_compile("QA-17", "no"), retained)
    assert yes["status"] == "fail" and no["status"] == "fail"

    def blocking(result, event_id):
        return {
            row["key"]: row
            for attempt in result["attempts"]
            if attempt["event_id"] == event_id
            for row in attempt["checks"]
            if row["status"] == "fail"
        }

    # The first event does carry a distinguishable yes answer.
    first_yes = blocking(yes, "event_001")
    assert set(first_yes) == {"motion_after_sound", "legal_integer_query_window"}
    distinguishable = [
        row
        for attempt in yes["attempts"]
        if attempt["event_id"] == "event_001"
        for row in attempt["checks"]
        if row["key"] == "answer_distinguishable"
    ]
    assert distinguishable[0]["status"] == "pass"
    assert distinguishable[0]["measured"]["target"] == {"human_blue": "yes"}
    assert distinguishable[0]["measured"]["competitors"] == {"human_green": "no"}

    # The second event answers no, and so does its only competitor.
    second_no = blocking(no, "event_002")
    assert set(second_no) == {"answer_distinguishable"}
    assert "share the target" in second_no["answer_distinguishable"]["reason"]


# --------------------------------------------------------------- positive controls


def test_a_moving_target_satisfies_qa06_moving(moving_during_first_event):
    compiled = _compile("QA-06", "moving")
    # speech_motion=speaker_moving is accepted today, so planning is available.
    # The competitor's own state is not requestable, which is a retry yield and
    # is recorded as such rather than as an unreachable condition.
    assert compiled.state == capabilities.STATE_AVAILABLE
    distinguishable = next(
        item for item in compiled.conditions if item.kind == "answer_distinguishable"
    )
    assert "competitor_motion" in distinguishable.support["not_guaranteed"]
    assert "140 of" in distinguishable.support["not_guaranteed"]["competitor_motion"]

    result = _check(compiled, moving_during_first_event)
    assert result["status"] == "pass", sorted(_failed(result))
    assert result["selected_event_id"] == "event_001"

    # The target walks during its first event and stands still during its
    # second, so the still branch is answerable on the other event rather than
    # refused. The event a branch selects is part of the answer.
    # The target stands still during its second event, so the motion predicate
    # picks that event. The competitor in this fixture never moves, so the still
    # branch now needs a differing competitor and says so on that event instead
    # of passing with a shared answer.
    still = _check(_compile("QA-06", "still"), moving_during_first_event)
    assert still["status"] == "fail"
    first = next(
        attempt for attempt in still["attempts"] if attempt["event_id"] == "event_001"
    )
    assert first["status"] == "fail"
    assert "motion_during_event" in first["failed"]
    second = next(
        attempt for attempt in still["attempts"] if attempt["event_id"] == "event_002"
    )
    assert "motion_during_event" not in second["failed"]
    assert "answer_distinguishable" in second["failed"]


def test_a_walking_target_satisfies_qa15_nearer(moving_during_first_event):
    result = _check(_compile("QA-15", "nearer"), moving_during_first_event)
    assert result["status"] == "pass", sorted(_failed(result))
    rows = [
        row
        for attempt in result["attempts"]
        for row in attempt["checks"]
        if row["key"] == "distance_net_change" and row["status"] == "pass"
    ]
    measured = rows[0]["measured"]
    assert measured["endpoint_delta_m"] < -gc.DISTANCE_MARGIN_M
    assert measured["monotonic_over_interval"] is True

    farther = _check(_compile("QA-15", "farther"), moving_during_first_event)
    assert farther["status"] == "fail"


def test_the_same_walk_does_not_satisfy_the_cross_time_recipe(moving_during_first_event):
    """Motion during the sound is the wrong window for a post-sound question."""

    compiled = _compile("QA-17", "yes", task_family="cross_time_state")
    result = _check(compiled, moving_during_first_event)
    rows = [
        row
        for attempt in result["attempts"]
        for row in attempt["checks"]
        if row["key"] == "motion_window_placement"
    ]
    assert rows and all(row["status"] == "fail" for row in rows)
    assert any("inside the measured wet tail" in row["reason"] for row in rows)


def test_motion_after_the_tail_satisfies_qa17_yes(moving_after_second_tail):
    result = _check(_compile("QA-17", "yes"), moving_after_second_tail)
    assert result["status"] == "pass", sorted(_failed(result))
    assert result["selected_event_id"] == "event_002"
    rows = [
        row
        for attempt in result["attempts"]
        if attempt["event_id"] == "event_002"
        for row in attempt["checks"]
        if row["key"] == "motion_after_sound"
    ]
    run = rows[0]["measured"]["selected_run"]
    assert run["value"] == "yes"
    assert run["public_s"] == [9.0, 10.0]


def test_motion_after_the_tail_also_satisfies_the_cross_time_recipe(moving_after_second_tail):
    compiled = _compile("QA-17", "yes", task_family="cross_time_state")
    result = _check(compiled, moving_after_second_tail)
    rows = [
        row
        for attempt in result["attempts"]
        if attempt["event_id"] == "event_002"
        for row in attempt["checks"]
        if row["key"] == "motion_window_placement"
    ]
    assert rows[0]["status"] == "pass", rows[0].get("reason")
    measured = rows[0]["measured"]
    assert measured["first_allowed_motion_frame"] == SECOND_EVENT_FIRST_LEGAL_MOTION_FRAME
    assert measured["runs_before_allowed"] == []


def test_motion_after_the_tail_satisfies_qa16(moving_after_second_tail):
    result = _check(_compile("QA-16"), moving_after_second_tail)
    assert result["status"] == "pass", sorted(_failed(result))


@pytest.mark.parametrize("branch,centroid_x", [("left", 200.0), ("right", 1080.0)])
def test_an_entry_transition_satisfies_qa07(branch, centroid_x):
    facts = _retained()
    _set_visibility(facts, "source1", range(0, 30), "out_of_view")
    _set_visibility(facts, "source1", range(30, 150), "visible_clear", centroid_x=centroid_x)
    compiled = _compile("QA-07", branch)
    assert compiled.state == capabilities.STATE_NOT_IMPLEMENTED
    result = _check(compiled, facts)
    assert result["status"] == "pass", sorted(_failed(result))
    rows = [
        row
        for attempt in result["attempts"]
        for row in attempt["checks"]
        if row["key"] == "entry_transition" and row["status"] == "pass"
    ]
    entry = rows[0]["measured"]["selected_entry"]
    assert entry["entry_frame"] == 30
    assert entry["side"] == branch
    assert entry["public_s"] == [2.0, 10.0]

    other = "right" if branch == "left" else "left"
    wrong = _check(_compile("QA-07", other), facts)
    assert wrong["status"] == "fail"


def test_a_reappearance_satisfies_qa09_yes_and_refuses_no():
    facts = _fully_occluded_fixture(reappears=True)
    yes = _check(_compile("QA-09", "yes"), facts)
    assert yes["status"] == "pass", sorted(_failed(yes))
    no = _check(_compile("QA-09", "no"), facts)
    assert no["status"] == "fail"


def test_a_permanent_occlusion_satisfies_qa09_no_only_with_complete_coverage():
    facts = _fully_occluded_fixture(reappears=False)
    no = _check(_compile("QA-09", "no"), facts)
    assert no["status"] == "pass", sorted(_failed(no))

    # Remove one frame of visibility truth and the negative answer is refused.
    incomplete = deepcopy(facts)
    rows = incomplete["visibility"]["source1"]
    rows.pop("120", rows.pop(120, None))
    refused = _check(_compile("QA-09", "no"), incomplete)
    assert refused["status"] == "fail"
    reasons = [
        row["reason"]
        for attempt in refused["attempts"]
        for row in attempt["checks"]
        if row["key"] == "visibility_coverage_complete"
    ]
    assert any("every frame" in reason for reason in reasons)


# ------------------------------------------------------------------- QA-25 subsets


def test_qa25_subsets_compile_to_different_conditions_and_modalities():
    rows = {branch: _compile("QA-25", branch) for branch in ("A", "V", "AV")}
    modalities = {}
    for branch, compiled in rows.items():
        condition = next(item for item in compiled.conditions if item.kind == "required_modality")
        modalities[branch] = tuple(condition.detail["modalities"])
    assert modalities == {"A": ("audio",), "V": ("video",), "AV": ("audio", "video")}

    keys = {branch: {item.key for item in compiled.conditions} for branch, compiled in rows.items()}
    assert "wet_tail_readback" not in keys["A"] and "wet_tail_readback" not in keys["V"]
    assert {"wet_tail_readback", "visibility_state", "hidden_motion_changed"} <= keys["AV"]
    assert "source_activity_readback" not in keys["V"]

    av = rows["AV"]
    bearing = next(item for item in av.conditions if item.kind == "bearing_reference")
    assert bearing.detail["min_query_sources"] == 2
    assert bearing.detail["rival_sound_asset_must_differ"] is True


def test_qa25_av_needs_a_hidden_query_frame(retained):
    a = _check(_compile("QA-25", "A"), retained)
    v = _check(_compile("QA-25", "V"), retained)
    av = _check(_compile("QA-25", "AV"), retained)
    assert a["status"] == "pass", sorted(_failed(a))
    assert v["status"] == "pass", sorted(_failed(v))
    assert av["status"] == "fail"
    assert "visibility_state" in _failed(av)
    rows = [
        row
        for attempt in av["attempts"]
        for row in attempt["checks"]
        if row["key"] == "visibility_state"
    ]
    assert rows[0]["measured"]["searched_whole_clip"] is True
    assert rows[0]["measured"]["frames_in_allowed_states"] == []


def test_qa25_av_is_satisfied_when_the_target_hides_and_changes_course():
    facts = _retained()
    # Visible while walking, then hidden while it turns towards the listener.
    _displace_towards_listener(facts, "source1", 60, 80, 1.0)
    _displace_towards_listener(facts, "source1", 90, 110, -1.4)
    _set_visibility(facts, "source1", range(0, 90), "visible_clear", centroid_x=600.0)
    _set_visibility(facts, "source1", range(90, 150), "fully_occluded")
    result = _check(_compile("QA-25", "AV"), facts)
    rows = [
        row
        for attempt in result["attempts"]
        for row in attempt["checks"]
        if row["key"] in {"visibility_state", "hidden_motion_changed"}
    ]
    assert all(row["status"] == "pass" for row in rows), [
        (row["key"], row.get("reason")) for row in rows
    ]
    assert result["status"] == "pass", sorted(_failed(result))


# -------------------------------------------------------- consuming a P01 request


def test_compiles_the_qa_targets_of_a_real_production_request():
    from avengine.dataset.production_spec import parse_production_config

    config = {
        "batch_id": "p02_condition_smoke",
        "seed": 7,
        "defaults": {
            "room_family": "authored",
            "room_id": "authored_room",
            "clock": {"frame_count": 150, "frame_rate_hz": 15, "sample_rate_hz": 16000},
            "reserve_tail_s": 3.0,
            "rig": {"fov_deg": 85, "height_above_floor_m": 1.55, "motion": "static",
                    "resolution_hw": [720, 1280]},
            "post_assembly_convolution_gain": 0.5,
        },
        "episodes": [
            {
                "request_id": "episode_0001",
                "instances": INSTANCES,
                "qa_targets": [
                    {"qa_id": "QA-06", "target_instance_ids": ["human_blue"], "branch": "still"},
                    {"qa_id": "QA-17", "target_instance_ids": ["human_blue"], "branch": "yes"},
                ],
            }
        ],
    }
    parsed = parse_production_config(config)
    request = parsed.episodes[0]
    result = gc.compile_request_conditions(request)
    assert [row["qa_id"] for row in result["compiled"]] == ["QA-06", "QA-17"]
    # QA-06 still needs a competitor that moves and QA-17 yes needs one that
    # stays still, so one Episode cannot serve both. This used to compile clean
    # and then plan a world that answered neither.
    assert {row["knob"] for row in result["request_conflicts"]} == {"competitor_motion"}
    assert set(result["report"]["qa_ids"]) == {"QA-06", "QA-17"}
    # Branches survive the production request; no out-of-band map is needed.
    assert [row["branch"] for row in result["compiled"]] == ["still", "yes"]
    with_branches = gc.compile_request_conditions(
        request, branches={"QA-06": "still", "QA-17": "yes"}
    )
    assert [row["branch"] for row in with_branches["compiled"]] == ["still", "yes"]


def test_compiles_every_request_of_a_parsed_configuration():
    """Both plain Episodes and core-group members compile from one config."""

    from avengine.dataset.production_spec import parse_production_config

    config = {
        "batch_id": "p02_condition_group",
        "seed": 11,
        "defaults": {
            "room_family": "authored",
            "room_id": "authored_room",
            "clock": {"frame_count": 150, "frame_rate_hz": 15, "sample_rate_hz": 16000},
            "reserve_tail_s": 3.0,
            "rig": {"fov_deg": 85, "height_above_floor_m": 1.55, "motion": "static",
                    "resolution_hw": [720, 1280]},
            "post_assembly_convolution_gain": 0.5,
            "instances": INSTANCES,
            "qa_targets": [{"qa_id": "QA-13", "target_instance_ids": ["human_blue"]}],
        },
        "episodes": [{"request_id": "episode_0001"}],
    }
    parsed = parse_production_config(config)
    requests = parsed.all_requests()
    assert requests
    for request in requests:
        result = gc.compile_request_conditions(
            request, branches={} if request.task_family else {}
        )
        assert result["task_family"] == request.task_family
        for row in result["compiled"]:
            assert row["state"] in gc.APPLICABILITY_STATES
            assert row["conditions"]
            # The compiled row states scene conditions, not a candidate id list.
            assert row["sampler_profile"] or row["gaps"]


def test_a_request_naming_two_contradictory_branches_is_refused():
    request = {
        "request_id": "episode_conflict",
        "instances": INSTANCES,
        "qa_targets": [
            {"qa_id": "QA-06", "target_instance_ids": ["human_blue"], "branch": "moving"},
            {"qa_id": "QA-06", "target_instance_ids": ["human_blue"], "branch": "still"},
        ],
    }
    result = gc.compile_request_conditions(request)
    assert result["request_conflicts"], "one episode cannot carry both branches at once"
    assert {row["knob"] for row in result["request_conflicts"]} == {
        "speech_motion", "competitor_motion"}


# ---------------------------------------------------------------- input validation


def test_an_unknown_branch_is_refused():
    with pytest.raises(gc.GenerationConditionError):
        _compile("QA-06", "walking")
    with pytest.raises(gc.GenerationConditionError):
        _compile("QA-01", "moving")


def test_an_unknown_qa_id_is_refused():
    with pytest.raises(gc.GenerationConditionError):
        _compile("QA-99")


def test_a_target_naming_an_absent_instance_still_compiles_but_cannot_be_proven(retained):
    compiled = _compile("QA-06", "still", instance="human_blue")
    result = gc.check_conditions(compiled, retained, actor_by_instance={})
    assert result["unresolved_instances"] == []
    other = gc.check_conditions(
        compiled, retained, actor_by_instance={"human_blue": "source9"}
    )
    assert other["unresolved_instances"] == ["human_blue"]
    assert other["status"] == "fail"


def test_facts_without_a_clock_are_refused():
    with pytest.raises(gc.GenerationConditionError):
        gc.check_conditions(_compile("QA-06", "still"), {"actors": {}})


# ------------------------------------------------- the generator capability layer


SOLVER_DECLARATION = {
    "source": "avengine.rooms.conditioned_visibility",
    "version": "p04_test_declaration_v1",
    "knobs": ["visibility_transition", "pixel_occlusion_transition"],
    "declared_at": "a test declaration, standing in for the real solver",
}


def test_every_planning_key_the_compiler_emits_has_an_owning_layer():
    """A new condition cannot leak a key that looks like a sampler gap."""

    matrix = gc.compile_catalog_matrix(instances=INSTANCES)
    emitted = set()
    for row in matrix["compiled"]:
        for condition in row["conditions"]:
            emitted.update(condition.get("planning") or {})
    assert emitted, "the catalog matrix emitted no planning keys at all"
    unmapped = sorted(key for key in emitted if key not in gc.PLANNING_KEY_LAYER)
    assert unmapped == []
    for key in emitted:
        assert gc.planning_layer(key) in gc.PLANNING_LAYERS


def test_an_unknown_planning_key_is_refused_rather_than_guessed():
    with pytest.raises(gc.GenerationConditionError) as error:
        gc.planning_layer("some_new_knob")
    assert "owning layer" in str(error.value)


def test_every_capability_layer_key_is_classified_against_a_measured_record():
    for key, layer in gc.PLANNING_KEY_LAYER.items():
        if layer not in gc.CAPABILITY_LAYERS:
            continue
        enforcement, basis = gc.goal_enforcement(key)
        assert enforcement in gc.GOAL_ENFORCEMENTS
        assert basis, f"{key} is classified without saying how we know"
    # An unclassified key fails closed rather than silently becoming optional.
    assert gc.goal_enforcement("reserve_tail_s")[0] == "guarantee_required"


def test_a_key_owned_by_another_layer_is_not_a_planner_question():
    with pytest.raises(gc.GenerationConditionError) as error:
        gc.goal_enforcement("public_time_precision")
    assert "publication" in str(error.value)


def test_the_fallback_declaration_matches_what_the_real_sampler_honours():
    """Measured, not hand-listed: each baseline knob survives resolution."""

    from avengine.rooms import conditioned_sampler

    registry = json.loads(
        (REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json").read_text()
    )
    probes = {
        "speech_motion": "speaker_moving",
        "anchor_visibility": "off_screen",
        "competitor_visibility": "off_screen",
        "anchor_line_of_sight": "occluded",
        "event_relation": "overlap",
        "anchor_count": 1,
        "reserve_tail_s": 2.5,
        "minimum_overlap_s": 0.4,
        "min_gap_between_audible_windows_s": 0.75,
        "distance_range_m": [1.0, 4.0],
        "separation_bin_deg": [30.0, 60.0],
        "separation_target_policy": conditioned_sampler.SEPARATION_TARGET_ANY_LEGAL,
        "retry_budget_within_profile": 50,
    }
    assert set(probes) == set(gc.BASELINE_SAMPLER_KNOBS)
    base = {
        "seed": 7,
        "camera": {"motion": "static"},
        "entities": {"total_count": 2, "silent_count": 0, "min_articulated_count": 2},
        "profile": {},
    }
    for knob, value in probes.items():
        request = deepcopy(base)
        request["profile"][knob] = value
        resolved = conditioned_sampler.resolve_condition_profile(request, registry)
        assert knob in resolved, f"the sampler never reads {knob}"
        carried = resolved[knob] == value or list(resolved[knob] or []) == list(
            value if isinstance(value, (list, tuple)) else [value]
        )
        assert carried, f"the sampler accepted {knob} and then dropped it: {resolved[knob]!r}"


def test_the_measured_trap_knob_is_refused_in_a_declaration():
    """separation_floor_deg is accepted by the sampler and then ignored."""

    assert "separation_floor_deg" in gc.ACCEPTED_BUT_IGNORED_KNOBS
    with pytest.raises(gc.GenerationConditionError) as error:
        gc.resolve_generator_capabilities(
            {"source": "s", "version": "v", "knobs": ["separation_floor_deg"]}
        )
    assert "accept" in str(error.value) and "ignore" in str(error.value)


def test_qa04_asks_for_a_median_plane_offset_not_an_inter_source_separation():
    compiled = _compile("QA-04")
    bearing = next(item for item in compiled.conditions if item.kind == "bearing_reference")
    assert "separation_floor_deg" not in bearing.planning
    assert bearing.planning["anchor_median_plane_offset_deg"] == gc.QA04_SIDE_DEAD_ZONE_DEG
    assert "_event_start_side_window" in bearing.detail["judge"]
    # A missing knob here is a yield cost, because QA-04 did produce items.
    assert compiled.state == capabilities.STATE_AVAILABLE


def test_a_declaration_turns_its_supported_knobs_green_without_touching_the_rest():
    before = _compile("QA-09", "yes")
    assert before.state == capabilities.STATE_NOT_IMPLEMENTED
    blocked = {row["key"] for row in before.gaps()}
    assert "occlusion_transition" in blocked
    assert "pixel_occlusion_transition" in gc.unsupported_by_layer([before])["solver"]

    after = _compile("QA-09", "yes", capabilities=SOLVER_DECLARATION)
    assert after.state == capabilities.STATE_AVAILABLE, after.reason
    occlusion = next(item for item in after.conditions if item.kind == "occlusion_transition")
    label = f"{SOLVER_DECLARATION['source']}@{SOLVER_DECLARATION['version']}"
    assert occlusion.support["declared_by_knob"]["pixel_occlusion_transition"] == label
    assert gc.unsupported_by_layer([after]) == {}

    # A knob the declaration does not name is still refused.
    still_blocked = _compile("QA-15", "nearer", capabilities=SOLVER_DECLARATION)
    assert still_blocked.state == capabilities.STATE_NOT_IMPLEMENTED
    assert "distance_trend_during_event" in gc.unsupported_by_layer([still_blocked])["solver"]


def test_a_solver_declaration_does_not_erase_the_sampler_baseline():
    """Declarations add up, because the planners are layered."""

    merged = gc.resolve_generator_capabilities(SOLVER_DECLARATION)
    assert merged.supports("pixel_occlusion_transition")
    assert merged.supports("speech_motion"), "the sampler baseline must survive"
    assert merged.declared_by("speech_motion").startswith("avengine.rooms.conditioned_sampler")
    assert merged.declared_by("pixel_occlusion_transition").startswith(
        SOLVER_DECLARATION["source"]
    )

    # Two declarations merge, and a planner that dropped a knob can say so.
    both = gc.resolve_generator_capabilities([
        SOLVER_DECLARATION,
        {"source": "avengine.rooms.conditioned_motion", "version": "p03_v1",
         "knobs": ["distance_trend_during_event", "competitor_motion"]},
    ])
    assert both.supports("distance_trend_during_event")
    assert both.supports("visibility_transition")
    replaced = gc.resolve_generator_capabilities(
        {"source": "stripped", "version": "v1", "knobs": ["speech_motion"],
         "replaces_baseline": True}
    )
    assert replaced.supports("speech_motion")
    assert not replaced.supports("anchor_count")


def test_a_module_can_declare_its_own_capabilities():
    """The planner declares itself; this module never imports the planner."""

    class Planner:
        @staticmethod
        def describe_generator_capabilities():
            return dict(SOLVER_DECLARATION)

    resolved = gc.resolve_generator_capabilities(Planner)
    assert resolved.supports("visibility_transition")
    assert resolved.declared_by("visibility_transition").startswith(
        SOLVER_DECLARATION["source"]
    )
    assert not resolved.supports("distance_trend_during_event")


def test_a_declaration_may_restrict_a_knob_to_one_room_route():
    declaration = {
        "source": "avengine.rooms.conditioned_visibility",
        "version": "p04_route_scoped_v1",
        "knobs": {"visibility_transition": ["spear_unreal"],
                  "pixel_occlusion_transition": ["spear_unreal"]},
        "backends": ["spear_unreal", "habitat"],
    }
    on_ue = _compile("QA-09", "yes", capabilities=declaration)
    assert on_ue.state == capabilities.STATE_AVAILABLE
    on_habitat = gc.compile_generation_conditions(
        _target("QA-09", others=("human_green",)),
        branch="yes",
        instances=INSTANCES,
        capabilities=declaration,
        backend="habitat",
    )
    assert on_habitat.state == capabilities.STATE_NOT_IMPLEMENTED
    assert "habitat" in on_habitat.reason


def test_an_unknown_route_is_refused():
    with pytest.raises(gc.GenerationConditionError) as error:
        gc.compile_generation_conditions(
            _target("QA-09", others=("human_green",)),
            branch="yes",
            instances=INSTANCES,
            backend="unreal_engine_7",
        )
    assert "unknown room family route" in str(error.value)


def test_a_replacing_declaration_can_narrow_the_routes():
    with pytest.raises(gc.GenerationConditionError) as error:
        gc.compile_generation_conditions(
            _target("QA-09", others=("human_green",)),
            branch="yes",
            instances=INSTANCES,
            capabilities={"source": "s", "version": "v", "knobs": [],
                          "backends": ["spear_unreal"], "replaces_baseline": True},
            backend="habitat",
        )
    assert "does not declare the 'habitat' route" in str(error.value)


@pytest.mark.parametrize(
    "declaration,expected",
    [
        ({"source": "s", "version": "v", "state": "available"}, "not an outcome"),
        ({"source": "s", "version": "v", "evidence": {}}, "not an outcome"),
        ({"source": "s", "version": "v", "condition_met": True}, "not an outcome"),
        ({"version": "v"}, "name its source"),
        ({"source": "s"}, "capability version"),
        ({"source": "s", "version": "v", "wat": 1}, "unknown capability declaration keys"),
        ({"source": "s", "version": "v", "knobs": ["nope"]}, "never emits"),
        ({"source": "s", "version": "v", "knobs": ["public_time_precision"]},
         "not a planner interface"),
    ],
)
def test_a_declaration_cannot_assert_an_outcome_or_an_unknown_knob(declaration, expected):
    with pytest.raises(gc.GenerationConditionError) as error:
        gc.resolve_generator_capabilities(declaration)
    assert expected in str(error.value)


def test_a_declaration_cannot_make_evidence_or_a_met_condition_green(retained):
    """The three states stay three states."""

    generous = {
        "source": "avengine.rooms.conditioned_visibility",
        "version": "generous_v1",
        "knobs": ["visibility_transition", "pixel_occlusion_transition",
                  "distance_trend_during_event", "competitor_motion"],
    }
    compiled = _compile("QA-09", "yes", capabilities=generous)
    assert compiled.state == capabilities.STATE_AVAILABLE
    layers = compiled.state_layers()
    assert layers["planning_support"] == capabilities.STATE_AVAILABLE
    assert layers["native_evidence"] == "not_checked"
    assert layers["condition_met"] == "not_checked"

    # Reading a real episode still refuses it: nothing was rendered.
    result = _check(compiled, retained)
    assert result["state_layers"]["planning_support"] == capabilities.STATE_AVAILABLE
    assert result["state_layers"]["condition_met"] == "fail"
    assert result["status"] == "fail"
    assert "occlusion_transition" in _failed(result)


def test_the_layer_summary_separates_blocking_gaps_from_yield_costs():
    matrix = gc.compile_catalog_matrix(instances=INSTANCES)
    rows = matrix["unsupported_by_layer"]
    blocking = {
        key
        for keys in rows.values()
        for key, row in keys.items()
        if row["enforcement"] == "guarantee_required"
    }
    # These are the knobs a planner must guarantee rather than merely verify.
    # first_speaker_instance_id joined them once QA-03 and QA-24 started naming
    # which entity has to own the earliest audible window: the sampler used to
    # enumerate every feasible actor order and draw one uniformly, so the gold
    # answer of a "who spoke first" question was whoever the draw picked.
    assert blocking == {
        "visibility_transition",
        "pixel_occlusion_transition",
        "distance_trend_during_event",
        "first_speaker_instance_id",
    }
    yields = {
        key
        for keys in rows.values()
        for key, row in keys.items()
        if row["enforcement"] == "verify_only"
    }
    assert "competitor_motion" in yields
    for keys in rows.values():
        for key, row in keys.items():
            assert row["basis"], f"{key} is reported without a measured basis"
            assert row["wanted_by"]


def test_sampler_profile_keeps_every_key_the_sampler_actually_accepts():
    """The old hand-kept list dropped knobs the sampler honours."""

    compiled = _compile("QA-13")
    profile = compiled.sampler_profile()
    assert profile["min_gap_between_audible_windows_s"] == 0.5
    overlap = _compile("QA-05", "overlap")
    assert overlap.sampler_profile()["minimum_overlap_s"] == 0.3
    assert overlap.sampler_profile()["event_relation"] == "overlap"
    for knob in profile:
        assert gc.planning_layer(knob) == "sampler_profile"


def test_planning_keys_are_grouped_by_the_layer_that_has_to_act():
    compiled = _compile("QA-17", "yes")
    grouped = compiled.planning_by_layer()
    assert set(grouped) <= set(gc.PLANNING_LAYERS)
    assert "readback" in grouped and "require_measured_wet_tail" in grouped["readback"]
    assert "publication" in grouped and "public_time_precision" in grouped["publication"]
    assert compiled.solver_goals()["target_moved_after_sound"] is True
    # A readback demand is never presented as a sampler knob.
    assert "require_measured_wet_tail" not in compiled.sampler_profile()


# ------------------------------------------------------------ save and restore


def test_a_compiled_condition_set_survives_save_and_restore():
    compiled = gc.compile_generation_conditions(
        {
            "qa_id": "QA-17",
            "target_instance_ids": ["human_blue"],
            "branch": "yes",
            "event": {"kind": "event_ordinal", "ordinal": 2},
        },
        instances=INSTANCES,
        task_family="cross_time_state",
        capabilities=SOLVER_DECLARATION,
    )
    payload = compiled.to_dict()
    restored = gc.restore_compiled_conditions(payload)
    assert restored.to_dict() == payload
    assert restored.qa_id == "QA-17"
    assert restored.branch == "yes"
    assert restored.event == {"kind": "event_ordinal", "ordinal": 2}
    assert restored.task_family == "cross_time_state"
    # The saved set is the merged set, so the per-knob owner is what survives.
    assert restored.capabilities.declared_by("pixel_occlusion_transition").startswith(
        SOLVER_DECLARATION["source"]
    )
    assert restored.capabilities.supports("speech_motion")
    assert restored.target_instance_ids == ("human_blue",)
    # Round-tripping through JSON is what a runner actually does.
    again = gc.restore_compiled_conditions(json.loads(json.dumps(payload)))
    assert again.to_dict() == payload


def test_restoring_an_incomplete_saved_set_is_refused():
    with pytest.raises(gc.GenerationConditionError):
        gc.restore_compiled_conditions({"qa_id": "QA-17"})
    with pytest.raises(gc.GenerationConditionError):
        gc.restore_compiled_conditions("not a mapping")


def test_recompiling_the_same_inputs_gives_the_same_conditions():
    kwargs = dict(instances=INSTANCES, task_family="cross_time_state",
                  capabilities=SOLVER_DECLARATION)
    target = {
        "qa_id": "QA-13",
        "target_instance_ids": ["human_blue"],
        "event": {"kind": "event_id", "event_id": "event_002"},
    }
    first = gc.compile_generation_conditions(dict(target), **kwargs)
    second = gc.compile_generation_conditions(dict(target), **kwargs)
    assert first.to_dict() == second.to_dict()


# ------------------------------------------------- the judges belong to P08


def test_every_catalog_judge_this_module_calls_still_exists_with_its_parameters():
    """unified_catalog and angular_questions belong to another owner."""

    import inspect

    from avengine.qa import angular_questions

    for module, expected in ((catalog, gc.CATALOG_JUDGES),
                             (angular_questions, gc.ANGULAR_JUDGES)):
        for name, parameters in expected.items():
            judge = getattr(module, name, None)
            assert callable(judge), f"{module.__name__}.{name} is gone"
            actual = tuple(inspect.signature(judge).parameters)
            missing = [item for item in parameters if item not in actual]
            assert not missing, (
                f"{module.__name__}.{name} no longer takes {missing}; its signature is "
                f"{actual}. Check the new one rather than approximating it here."
            )


# --------------------------------- the live handshake with the real sampler


@pytest.fixture
def source_registry() -> dict:
    path = REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"
    if not path.is_file():
        pytest.skip("the runtime source registry is not present in this checkout")
    return json.loads(path.read_text())


def test_the_real_sampler_declares_itself_and_the_compiler_consumes_it():
    """P05 owns conditioned_sampler; this reads its own declaration."""

    from avengine.rooms import conditioned_sampler

    describe = getattr(conditioned_sampler, "describe_generator_capabilities", None)
    if not callable(describe):
        pytest.skip("the sampler in this checkout does not declare its capabilities yet")
    resolved = gc.resolve_generator_capabilities(conditioned_sampler)
    assert resolved.knobs, "the sampler declared no knobs at all"
    assert resolved.supports("speech_motion")
    for knob in resolved.knobs:
        # A declaration may only name keys this compiler actually asks for.
        assert gc.planning_layer(knob) in gc.CAPABILITY_LAYERS


def test_a_declaration_is_checked_against_the_sampler_rather_than_trusted(source_registry):
    """Advertised is not implemented until a probe carries the value through."""

    from avengine.rooms import conditioned_sampler

    if not callable(getattr(conditioned_sampler, "describe_generator_capabilities", None)):
        pytest.skip("the sampler in this checkout does not declare its capabilities yet")
    measured = gc.measure_sampler_capabilities(source_registry, sampler=conditioned_sampler)
    assert measured["verified_knobs"], "nothing was measurable"
    assert measured["agrees_with_declaration"], (
        f"these knobs are advertised but not honoured: {measured['unverified_knobs']} "
        f"({ {k: measured['outcomes'][k] for k in measured['unverified_knobs']} })"
    )
    # Every declared knob got a real probe, not a shrug.
    declared = set(measured["declared"]["knobs"])
    assert set(measured["outcomes"]) == declared


def test_measuring_refuses_a_planner_with_no_resolver():
    class Empty:
        __name__ = "empty_planner"

        @staticmethod
        def describe_generator_capabilities():
            return {"source": "empty_planner", "version": "v1", "knobs": []}

    with pytest.raises(gc.GenerationConditionError) as error:
        gc.measure_sampler_capabilities({}, sampler=Empty)
    assert "resolve_condition_profile" in str(error.value)


def test_an_uncheckable_declaration_is_not_taken_on_trust(source_registry):
    """A declared knob with no published values cannot be verified, so it is not."""

    from avengine.rooms import conditioned_sampler

    class Lying:
        __name__ = "lying_planner"
        ENUM_KNOB_VALUES = dict(getattr(conditioned_sampler, "ENUM_KNOB_VALUES", {}))
        resolve_condition_profile = staticmethod(
            conditioned_sampler.resolve_condition_profile
        )

        @staticmethod
        def describe_generator_capabilities():
            return {
                "source": "lying_planner",
                "version": "v1",
                # The sampler was measured to accept this and then drop it.
                "knobs": ["speech_motion", "pixel_occlusion_transition"],
            }

    # A trap knob is refused at declaration time; an unread one at measurement.
    measured = gc.measure_sampler_capabilities(source_registry, sampler=Lying)
    assert measured["agrees_with_declaration"] is False
    assert measured["unverified_knobs"] == ["pixel_occlusion_transition"]
    row = measured["outcomes"]["pixel_occlusion_transition"]
    # Not "the planner ignores it": "the planner publishes no way to test it".
    assert row["outcome"] == "not_probed"
    assert "publishes no legal values" in row["reason"]
    assert "speech_motion" in measured["verified_knobs"]


def test_an_advertised_but_dropped_knob_is_reported_as_ignored(source_registry):
    """The trap knob the sampler accepts and then drops, measured end to end."""

    from avengine.rooms import conditioned_sampler

    class Dropping:
        __name__ = "dropping_planner"
        ENUM_KNOB_VALUES = {}
        resolve_condition_profile = staticmethod(
            conditioned_sampler.resolve_condition_profile
        )

        @staticmethod
        def describe_generator_capabilities():
            return {"source": "dropping_planner", "version": "v1",
                    "knobs": ["separation_bin_deg", "anchor_count"]}

    measured = gc.measure_sampler_capabilities(source_registry, sampler=Dropping)
    assert measured["agrees_with_declaration"] is True
    assert set(measured["verified_knobs"]) == {"separation_bin_deg", "anchor_count"}


def test_the_real_declaration_leaves_only_the_pixel_and_distance_solvers_open(source_registry):
    """What is actually left to build, measured rather than asserted."""

    from avengine.rooms import conditioned_sampler

    if not callable(getattr(conditioned_sampler, "describe_generator_capabilities", None)):
        pytest.skip("the sampler in this checkout does not declare its capabilities yet")
    matrix = gc.compile_catalog_matrix(instances=INSTANCES, capabilities=conditioned_sampler)
    blocking = {
        key
        for keys in matrix["unsupported_by_layer"].values()
        for key, row in keys.items()
        if row["enforcement"] == "guarantee_required"
    }
    assert blocking <= {"pixel_occlusion_transition", "distance_trend_during_event"}, blocking
    # Whatever remains has to say who wants it and how we know it blocks.
    for keys in matrix["unsupported_by_layer"].values():
        for row in keys.values():
            assert row["basis"] and row["wanted_by"]
