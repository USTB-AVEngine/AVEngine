"""Joint solving of the audible window and the motion window.

The positive cases run against the same retained native episode
``test_v1_generation_conditions`` reads: a real ten second, 15 fps clip with
two humans, two programmed events on one of them and measured binaural wet
tails.  Its single moving run, frames 43 through 67, starts *inside* the first
event's measured tail.  That one property makes it a legal ``QA-17`` candidate
and an illegal ``cross_time_state`` one, so the same delivered trajectory is
used to prove that this solver keeps those two apart instead of collapsing
every post-sound question into the stricter recipe.

The hermetic cases build the smallest layout that exercises one rule.  A
fixture is not a rendered episode and none of this counts as dataset coverage;
the native side is proved by ``check_conditions`` on delivered facts, not here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from avengine.qa import generation_conditions as gc
from avengine.qa import unified_catalog as catalog
from avengine.rooms import conditioned_motion as cm

pytestmark = pytest.mark.fast_unit

REPOSITORY = Path(__file__).resolve().parents[2]
RETAINED = REPOSITORY / (
    "tmp/binding_dataset_20260909_v2/binding_delivery_initial16_v1/native/facts/facts_0009.json"
)
# The retained episode, read once so the numbers in this suite are checkable.
RETAINED_MOVING_RUN = (43, 68)
RETAINED_ANCHOR_TAIL_END_S = 4.1574375
RETAINED_SECOND_EVENT_START_S = 4.6666875
# ceil(4.1574375 * 15) + 1: the first frame a cross_time_state recipe may move in.
RETAINED_FIRST_LEGAL_MOTION_FRAME = 64

CLOCK = cm.EpisodeClock(frame_count=150, frame_rate_hz=15.0, sample_rate_hz=16000)
LISTENER = [0.0, 1.5, 0.0]

REGISTRY = {
    "assets": [
        {
            "asset_id": "human_a",
            "entity_class": "articulated_human",
            "revision": 1,
            "timeline": {
                "idle_action_id": "idle",
                "walking_action_id": "walk",
                "body_plan_id": "biped",
                "walk_phase_period_frames": 30,
            },
        },
        {
            "asset_id": "dog_a",
            "entity_class": "articulated_animal",
            "revision": 1,
            "timeline": {
                "idle_action_id": "idle",
                "walking_action_id": "walk",
                "body_plan_id": "quadruped",
                "walk_phase_period_frames": 24,
            },
        },
        {"asset_id": "speaker_a", "entity_class": "rigid_static_object", "revision": 1},
    ]
}
TWO_HUMANS = [
    {"entity_instance_id": "inst_1", "asset_id": "human_a",
     "source_class": "articulated_human", "role": "anchor"},
    {"entity_instance_id": "inst_2", "asset_id": "human_a",
     "source_class": "articulated_human"},
]
HUMAN_DOG_DEVICE = [
    {"entity_instance_id": "inst_1", "asset_id": "human_a",
     "source_class": "articulated_human", "role": "anchor"},
    {"entity_instance_id": "inst_dog", "asset_id": "dog_a",
     "source_class": "articulated_animal"},
    {"entity_instance_id": "inst_dev", "asset_id": "speaker_a",
     "source_class": "rigid_static_object"},
]
HUMAN_AND_DEVICE = [
    {"entity_instance_id": "inst_1", "asset_id": "human_a",
     "source_class": "articulated_human", "role": "anchor"},
    {"entity_instance_id": "inst_dev", "asset_id": "speaker_a",
     "source_class": "rigid_static_object"},
]


def _retained() -> dict:
    if not RETAINED.is_file():
        pytest.skip(f"retained facts are not present in this checkout: {RETAINED}")
    return json.loads(RETAINED.read_text())


@pytest.fixture
def retained() -> dict:
    return _retained()


def _pool_row(name: str, seconds: float, *, lead: float = 0.1, trail: float = 0.1,
              gap: float | None = None, measured: bool = True) -> dict:
    """One prepared-segment row in the shape ``sound_segments.pool_row`` emits."""

    rate = 16000
    count = int(seconds * rate)
    first, last = int(lead * rate), count - int(trail * rate)
    intervals = [[first, last]]
    if gap:
        middle, half = (first + last) // 2, int(gap * rate / 2)
        intervals = [[first, middle - half], [middle + half, last]]
    row = {
        "sound_asset_id": name,
        "sample_rate_hz": rate,
        "sample_count": count,
        "audible_start_sample": intervals[0][0],
        "audible_end_sample_exclusive": intervals[-1][1],
        "max_internal_silence_s": gap or 0.0,
        "activity_coverage": 0.9,
        "source_origin": f"/library/{name}.wav",
        "linear_gain": 1.0,
    }
    if measured:
        row["source_activity_intervals_samples"] = intervals
    return row


def _compile(qa_id: str, branch: str | None = None, *, instances=TWO_HUMANS,
             task_family: str | None = None, target: str = "inst_1",
             anchors: tuple[str, ...] | None = None):
    spec: dict = {
        "qa_id": qa_id,
        "target_instance_ids": [target],
        "event": {"kind": "target_audible_window"},
    }
    if anchors is not None:
        spec["anchor_instance_ids"] = list(anchors)
    return gc.compile_generation_conditions(
        spec, branch=branch, instances=instances, registry=REGISTRY,
        task_family=task_family)


def _solve(qa_id, branch=None, *, instances=TWO_HUMANS, task_family=None,
           sounds=None, budget=None, target="inst_1", anchors=None, **kwargs):
    compiled = _compile(qa_id, branch, instances=instances, task_family=task_family,
                        target=target, anchors=anchors)
    if sounds is None:
        sounds = {target: _pool_row("seg_target", 2.0)}
        for row in instances:
            if row["entity_instance_id"] != target:
                sounds[row["entity_instance_id"]] = _pool_row(
                    "seg_" + row["entity_instance_id"], 1.5)
    return cm.solve_motion_windows(
        compiled, clock=CLOCK, budget=budget, sounds=sounds, registry=REGISTRY, **kwargs)


def _codes(solution) -> list[str]:
    return [row["code"] for row in solution.rejections]


# ------------------------------------------------------------------ semantics


def test_the_three_motion_meanings_are_read_from_the_compiled_conditions():
    """Which requirement applies comes from P02's compiled set, not from a QA id."""

    during = cm.motion_semantics(_compile("QA-06", "moving"))
    post = cm.motion_semantics(_compile("QA-17", "yes"))
    recipe = cm.motion_semantics(_compile("QA-17", "yes", task_family="cross_time_state"))

    assert during["semantics"] == "during_audible_window"
    assert post["semantics"] == "post_sound_query_only"
    assert recipe["semantics"] == "after_wet_tail"
    # The reading names the compiled condition it came from, so a change in
    # generation_conditions shows up here rather than silently disagreeing.
    assert "motion_during_event(moving=True)" in during["because"]
    assert recipe["because"] == ["motion_window_placement=" + gc.MOTION_AFTER_TAIL]
    assert set(post["because"]) <= {
        "post_sound_silent_window", "motion_after_sound", "wet_tail_readback"}


def test_a_post_sound_question_outside_a_recipe_does_not_place_the_movement():
    """QA-13/QA-16/QA-17 constrain the query moment; only cross_time_state moves."""

    plain = _solve("QA-17", "yes", budget=cm.MotionBudget(minimum_motion_s=1.0))
    recipe = _solve("QA-17", "yes", task_family="cross_time_state",
                    budget=cm.MotionBudget(minimum_motion_s=1.0))
    plain_target = plain.requirement_for("inst_1")
    recipe_target = recipe.requirement_for("inst_1")

    # Plain: the motion may open the moment the event ends, tail or no tail.
    assert plain_target.permitted_moving_frames[0] == plain.placement_for(
        "inst_1").end_frame
    assert plain_target.still_frames == ()
    # Recipe: the motion opens only after the tail, and the clip before it is still.
    tail_end_s = recipe.query_window["tail_end_s"]
    assert recipe_target.permitted_moving_frames[0] == int(
        math.ceil(tail_end_s * CLOCK.frame_rate_hz)) + 1
    assert recipe_target.still_frames[0][0] == 0


def test_qa13_is_not_forced_to_stand_still_while_it_sounds():
    """The over-constraint that would delete every legal QA-13 candidate."""

    solution = _solve("QA-13")
    target = solution.requirement_for("inst_1")

    assert solution.status == "solved"
    assert target.semantics == "unconstrained"
    assert target.still_frames == ()
    assert "speech_motion" not in solution.sampler_profile()
    assert cm.motion_semantics(_compile("QA-13")).get("speech_motion") is None


def test_a_cross_time_state_recipe_does_state_all_still():
    solution = _solve("QA-13", task_family="cross_time_state")
    assert solution.sampler_profile()["speech_motion"] == "all_still"
    assert solution.sampler_profile()["motion_window_placement"] == gc.MOTION_AFTER_TAIL


# ------------------------------------------------------- the audible window


def test_the_whole_audible_window_has_to_move_and_the_built_walk_does():
    """n moving frames need n+1 positions; one short and the last frame is still."""

    solution = _solve("QA-06", "moving")
    target = solution.requirement_for("inst_1")
    assert solution.status == "solved"
    first, last = target.moving_frames
    assert (first, last) == (solution.placement_for("inst_1").start_frame,
                             solution.placement_for("inst_1").end_frame)
    assert target.required_moving_steps == last - first

    built = cm.build_motion_trajectory(
        polyline_m=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]]),
        moving_frames=target.moving_frames, clock=CLOCK, budget=cm.MotionBudget())
    assert built["window_reads_moving"] is True
    assert built["position_frames"] == [first, last + 1]
    flags = cm.moving_flags_from_path(
        built["path_m"], frame_rate_hz=CLOCK.frame_rate_hz)
    assert flags[first:last].all()
    # and it stops afterwards rather than running to the end of the clip
    assert not flags[last + 1:].any()


def test_a_walk_one_position_short_leaves_the_final_audible_frame_still():
    """The off-by-one this solver's first run actually produced."""

    from avengine.routes.trajectory import resample_polyline_by_arc_length

    solution = _solve("QA-06", "moving")
    first, last = solution.requirement_for("inst_1").moving_frames
    short = np.repeat(np.zeros((1, 3)), CLOCK.frame_count, axis=0)
    short[first:last] = resample_polyline_by_arc_length(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]]), last - first)
    short[last:] = short[last - 1]
    flags = cm.moving_flags_from_path(short, frame_rate_hz=CLOCK.frame_rate_hz)

    assert not flags[last - 1]
    report = cm.verify_motion_candidate(
        solution, positions_m={"inst_1": short,
                               "inst_2": cm.static_trajectory([3, 0, 0], CLOCK.frame_count)})
    assert report["status"] == "fail"
    assert "motion_during_event" in report["failed_checks"]


def test_natural_pauses_stay_inside_the_audible_window():
    """The owner authorised pauses, so the window spans them and says it did."""

    solution = _solve("QA-06", "moving",
                      sounds={"inst_1": _pool_row("seg_gap", 2.0, gap=0.4),
                              "inst_2": _pool_row("seg_c", 1.5)})
    placement = solution.placement_for("inst_1")
    kinds = [row["qualification"] for row in solution.qualifications]

    assert solution.status == "solved"
    assert "audible_window_contains_natural_pauses" in kinds
    assert placement.internal_silence_kept_s == pytest.approx(0.4)
    # The motion requirement covers the whole span, pauses included, rather than
    # being cut into one window per active island.
    assert solution.requirement_for("inst_1").moving_frames == (
        placement.start_frame, placement.end_frame)


def test_an_event_bounding_box_is_not_a_measured_activity_window():
    solution = _solve("QA-15", "farther",
                      sounds={"inst_1": _pool_row("seg_bb", 2.0, measured=False),
                              "inst_2": _pool_row("seg_c", 1.5)})
    assert "audible_window_missing_activity_measurement" in _codes(solution)


def test_a_plan_record_interval_is_refused_rather_than_silently_relocated():
    """A record's whole-recording samples are not a pool row's segment samples."""

    with pytest.raises(cm.ConditionedMotionError, match="prepared segment"):
        cm.SoundCandidate.from_pool_row({
            "sound_asset_id": "seg_x", "sample_rate_hz": 16000, "sample_count": 32000,
            "source_activity_intervals_samples": [[441000, 480000]]})


def test_five_seconds_is_not_an_implicit_cap_but_the_reserved_tail_is_real():
    fits = _solve("QA-06", "moving",
                  sounds={"inst_1": _pool_row("seg_six", 6.0),
                          "inst_2": _pool_row("seg_c", 1.5)})
    too_long = _solve("QA-06", "moving",
                      sounds={"inst_1": _pool_row("seg_eight", 8.0),
                              "inst_2": _pool_row("seg_c", 1.5)})

    assert "audio_does_not_fit_reserved_tail" not in _codes(fits)
    assert "audio_does_not_fit_reserved_tail" in _codes(too_long)
    # and a caller that does declare a cap gets it
    capped = _solve("QA-06", "moving", budget=cm.MotionBudget(max_clip_s=3.0),
                    sounds={"inst_1": _pool_row("seg_six", 6.0),
                            "inst_2": _pool_row("seg_c", 1.5)})
    assert "audio_does_not_fit_reserved_tail" in _codes(capped)


# --------------------------------------------------------------- roles


def test_a_device_target_cannot_walk_but_a_device_competitor_blocks_nothing():
    """A mixed group's legal human or animal slot stays usable."""

    as_target = _solve("QA-06", "moving", instances=HUMAN_DOG_DEVICE, target="inst_dev",
                       sounds={"inst_dev": _pool_row("seg_d", 2.0),
                               "inst_1": _pool_row("seg_a", 1.5),
                               "inst_dog": _pool_row("seg_b", 1.5)})
    assert "target_cannot_self_locomote" in _codes(as_target)

    # QA-17 "no" is the branch that asks a competitor to move.
    mixed = _solve("QA-17", "no", instances=HUMAN_DOG_DEVICE,
                   budget=cm.MotionBudget(minimum_motion_s=1.0))
    assert mixed.status == "solved", _codes(mixed)
    assert mixed.requirement_for("inst_dog").must_move is True
    device = mixed.requirement_for("inst_dev")
    assert device.must_move is False and device.semantics == "unconstrained"
    assert "cannot" in device.reason

    only_device = _solve("QA-17", "no", instances=HUMAN_AND_DEVICE,
                         budget=cm.MotionBudget(minimum_motion_s=1.0))
    assert "no_locomotion_capable_competitor" in _codes(only_device)


def test_the_anchor_is_a_declared_role_and_is_never_guessed():
    # The instance rows carry no anchor role of their own here, so the anchor is
    # whatever the target spec names - and it names the competitor.
    plain = [{key: value for key, value in row.items() if key != "role"}
             for row in TWO_HUMANS]
    named = _solve("QA-06", "moving", instances=plain, anchors=("inst_2",))
    default = _solve("QA-06", "moving")

    assert "anchor_role_not_declared" in _codes(named)
    assert named.anchor["entity_instance_id"] is None
    assert named.anchor["required_by_question"] is True
    # generation_conditions marks every named target as the anchor when the spec
    # names none, which is a statement about the target and not about the first
    # programmed event.
    assert default.anchor["entity_instance_id"] == "inst_1"
    assert "anchor_role_not_declared" not in _codes(default)


def test_the_competitor_is_given_the_opposite_answer_not_a_random_flag():
    moving_branch = _solve("QA-06", "moving")
    assert moving_branch.sampler_profile()["competitor_motion"] == "still"
    assert moving_branch.requirement_for("inst_2").must_move is False

    yes = _solve("QA-17", "yes", budget=cm.MotionBudget(minimum_motion_s=1.0))
    no = _solve("QA-17", "no", budget=cm.MotionBudget(minimum_motion_s=1.0))
    assert yes.requirement_for("inst_1").must_move is True
    assert yes.requirement_for("inst_2").must_move is False
    assert no.requirement_for("inst_1").must_move is False
    assert no.requirement_for("inst_2").must_move is True


def test_the_qa06_still_branch_now_compiles_a_competitor_separation():
    """It used to compile none, which is why every candidate shared its answer."""

    still = _solve("QA-06", "still")
    kinds = [row["qualification"] for row in still.qualifications]

    assert still.status == "solved"
    assert "no_compiled_competitor_separation" not in kinds
    assert still.requirement_for("inst_1").must_move is False
    assert still.requirement_for("inst_2").must_move is True


def test_two_instances_of_one_asset_stay_two_bodies():
    solution = _solve("QA-06", "moving")
    ids = [row.entity_instance_id for row in solution.requirements]

    assert ids == ["inst_1", "inst_2"]
    assert solution.requirement_for("inst_1").must_move is True
    assert solution.requirement_for("inst_2").must_move is False


# ------------------------------------------------------------ distance trend


def test_a_net_change_alone_does_not_prove_a_trend():
    """The endpoint judge accepts both walks; only one held its direction."""

    solution = _solve("QA-15", "nearer")
    window = solution.requirement_for("inst_1").moving_frames
    straight = cm.build_motion_trajectory(
        polyline_m=np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 2.8]]),
        moving_frames=window, clock=CLOCK, budget=cm.MotionBudget())["path_m"]
    out_and_back = cm.build_motion_trajectory(
        polyline_m=np.array([[0.0, 0.0, 4.0], [0.5, 0.0, 4.2], [0.0, 0.0, 3.4]]),
        moving_frames=window, clock=CLOCK, budget=cm.MotionBudget())["path_m"]

    good = cm.distance_trend(straight, LISTENER, window)
    bad = cm.distance_trend(out_and_back, LISTENER, window)

    assert good["sign"] == bad["sign"] == "negative"
    assert good["endpoint_only_judge_would_accept"] is True
    assert bad["endpoint_only_judge_would_accept"] is True
    assert good["monotone_within_tolerance"] is True
    assert bad["monotone_within_tolerance"] is False
    assert bad["max_reversal_m"] > good["max_reversal_m"]

    still = cm.static_trajectory([3.0, 0.0, 3.0], CLOCK.frame_count)
    assert cm.verify_motion_candidate(
        solution, positions_m={"inst_1": straight, "inst_2": still},
        listener_position_m=LISTENER)["status"] == "pass"
    rejected = cm.verify_motion_candidate(
        solution, positions_m={"inst_1": out_and_back, "inst_2": still},
        listener_position_m=LISTENER)
    assert rejected["status"] == "fail"
    assert "distance_trend" in rejected["failed_checks"]


def test_the_trend_criterion_names_its_own_source_until_the_catalog_has_one():
    criterion = cm.resolve_distance_trend_criterion()
    assert criterion["net_change_at_least_m"] == gc.DISTANCE_MARGIN_M
    published = any(getattr(catalog, attribute, None) is not None
                    for _, attribute in cm.DISTANCE_TREND_CRITERION_HOOKS)
    if published:
        assert criterion["criterion_source"].startswith("avengine/qa/unified_catalog.py")
    else:
        assert "planning default" in criterion["criterion_source"]
    override = cm.resolve_distance_trend_criterion({"max_reversal_m": 0.0})
    assert override["max_reversal_m"] == 0.0
    assert override["criterion_source"] == "caller_override"


# --------------------------------------------------------------- the clock


def test_the_two_moving_flag_conventions_disagree_on_the_last_frame():
    path = np.zeros((10, 3))
    path[:6, 2] = np.linspace(0.0, 1.0, 6)
    path[6:, 2] = 1.0
    walking = np.zeros((10, 3))
    walking[:, 2] = np.linspace(0.0, 2.0, 10)

    hold = cm.moving_flags_from_path(walking, frame_rate_hz=15.0)
    still = cm.moving_flags_from_path(
        walking, frame_rate_hz=15.0, convention="forward_difference_last_still")

    assert hold[-1] and not still[-1]
    assert list(hold[:-1]) == list(still[:-1])
    with pytest.raises(cm.ConditionedMotionError, match="final frame still"):
        cm.build_motion_trajectory(
            polyline_m=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            moving_frames=(140, CLOCK.frame_count), clock=CLOCK,
            budget=cm.MotionBudget(
                moving_flag_convention="forward_difference_last_still"))


def test_a_walk_that_would_leave_the_clip_or_break_the_speed_band_is_refused():
    with pytest.raises(cm.ConditionedMotionError, match="escapes"):
        cm.build_motion_trajectory(
            polyline_m=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            moving_frames=(140, 200), clock=CLOCK, budget=cm.MotionBudget())
    fast = cm.build_motion_trajectory(
        polyline_m=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 9.0]]),
        moving_frames=(0, 30), clock=CLOCK, budget=cm.MotionBudget())
    assert fast["speed_within_declared_range"] is False
    assert fast["time_stretch_applied"] is False


def test_an_any_movement_branch_will_not_invent_its_own_minimum():
    bare = _solve("QA-17", "yes")
    declared = _solve("QA-17", "yes", budget=cm.MotionBudget(minimum_motion_s=1.0))

    assert "unsatisfiable_motion_semantics" in _codes(bare)
    assert declared.status == "solved"
    assert declared.requirement_for("inst_1").required_moving_steps == 15


def test_another_programmed_event_closes_the_post_sound_window():
    alone = _solve("QA-13")
    blocked = _solve("QA-13", other_event_windows_s=[(4.0, 6.0)])

    assert alone.query_window["legal_frames"][1] == CLOCK.frame_count
    assert alone.query_window["blocking_event_start_s"] is None
    assert blocked.query_window["blocking_event_start_s"] == pytest.approx(4.0)
    assert blocked.query_window["status"] == "empty"


def test_the_measured_tail_replaces_the_reservation_and_says_what_moved():
    planned = _solve("QA-13")
    assert planned.query_window["tail_basis"] == "planned_reserve_s"
    assert planned.query_window["requires_recheck_after_measurement"] is True

    shorter = cm.requery_after_measured_tail(
        planned, 3.0, query_frame=planned.query_window["earliest_query_frame"])
    assert shorter["tail_basis"] == "measured_wet_tail_intervals"
    assert shorter["earliest_query_frame_moved"] is True
    assert shorter["truth_recompute_required"] is True
    assert shorter["still_satisfiable"] is True

    same = cm.requery_after_measured_tail(planned, planned.query_window["tail_end_s"])
    assert same["earliest_query_frame_moved"] is False
    assert same["truth_recompute_required"] is False


def test_a_recipe_motion_window_follows_the_measured_tail():
    planned = _solve("QA-17", "yes", task_family="cross_time_state",
                     budget=cm.MotionBudget(minimum_motion_s=1.0))
    longer = cm.requery_after_measured_tail(planned, 6.0)

    assert longer["measured_first_motion_frame"] == int(math.ceil(6.0 * 15)) + 1
    assert longer["planned_first_motion_frame"] != longer["measured_first_motion_frame"]
    assert longer["motion_window_must_move"] is True


def test_the_solver_signature_names_the_knobs_p02_reports_as_gaps():
    signature = cm.solver_signature()
    compiled = _compile("QA-15", "nearer")
    gaps = {row["key"] for row in compiled.gaps()}

    assert "distance_trend_during_event" in signature["solves_knobs"]
    assert "competitor_motion" in signature["solves_knobs"]
    assert {"distance_net_change", "answer_distinguishable"} & gaps
    assert set(signature["motion_semantics"]) == set(cm.MOTION_SEMANTICS)


def test_qa16_states_its_motion_through_the_planning_knob():
    """QA-16 never says "it moved"; it says target_moved_after_sound."""

    meaning = cm.motion_semantics(_compile("QA-16"))
    solution = _solve("QA-16", budget=cm.MotionBudget(minimum_motion_s=1.0))
    target = solution.requirement_for("inst_1")

    assert meaning["target_moves"] is True
    assert meaning["stable_after_sound"] is True
    assert solution.status == "solved", _codes(solution)
    assert target.must_move is True
    assert solution.sampler_profile()["target_moved_after_sound"] is True


def test_qa16_is_judged_as_publishable_runs_not_as_a_net_change():
    """A source that only creeps past the margin has no publishable run."""

    solution = _solve("QA-16", budget=cm.MotionBudget(minimum_motion_s=1.0))
    window = solution.requirement_for("inst_1").moving_frames
    legal = solution.query_window["legal_frames"]
    anchor_end = solution.placement_for("inst_1").end_frame
    listener = [0.0, 0.0, 0.0]

    creeping = cm.build_motion_trajectory(
        polyline_m=np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.15]]),
        moving_frames=window, clock=CLOCK,
        budget=cm.MotionBudget(walk_speed_range_mps=(0.05001, 0.8)))["path_m"]
    decisive = cm.build_motion_trajectory(
        polyline_m=np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]),
        moving_frames=window, clock=CLOCK,
        budget=cm.MotionBudget(walk_speed_range_mps=(0.5, 1.5)))["path_m"]

    weak = cm.post_sound_distance_stability(
        creeping, listener, anchor_end_frame=anchor_end, windows=[legal],
        frame_rate_hz=CLOCK.frame_rate_hz)
    strong = cm.post_sound_distance_stability(
        decisive, listener, anchor_end_frame=anchor_end, windows=[legal],
        frame_rate_hz=CLOCK.frame_rate_hz)

    assert weak["publishable"] is False
    assert {run["value"] for run in weak["stable_runs"]} == {"under_margin"}
    assert strong["publishable"] is True
    assert strong["selected_run"]["value"] == "farther"
    assert strong["selected_run"]["public_s"] is not None


def test_a_moving_camera_is_refused_and_an_unstated_one_is_reported():
    assert cm.assert_static_camera({"motion": "static"})["state"] == "available"
    assert cm.assert_static_camera(None)["state"].startswith("evidence_missing")
    assert cm.assert_static_camera({})["state"].startswith("evidence_missing")
    with pytest.raises(cm.ConditionedMotionError, match="fixed rig"):
        cm.assert_static_camera({"motion": "orbit"})
    solved = _solve("QA-06", "moving", camera={"motion": "static"})
    assert solved.camera["camera_motion"] == "static"
    with pytest.raises(cm.ConditionedMotionError, match="fixed rig"):
        _solve("QA-06", "moving", camera={"motion": "dolly"})


def test_several_recordings_are_reported_and_none_is_quietly_chosen():
    report = cm.feasible_sound_candidates(
        _compile("QA-06", "moving"), clock=CLOCK,
        candidates=[_pool_row("short", 1.0), _pool_row("medium", 4.0),
                    _pool_row("too_long", 8.0)],
        sounds={"inst_2": _pool_row("seg_c", 1.5)}, registry=REGISTRY)
    by_id = {row["sound_asset_id"]: row for row in report["candidates"]}

    assert report["considered"] == 3
    assert report["eligible_count"] == 2
    assert report["selection"] == "none_made_here"
    assert by_id["short"]["status"] == "solved"
    assert by_id["medium"]["status"] == "solved"
    assert by_id["too_long"]["rejection_codes"] == ["audio_does_not_fit_reserved_tail"]
    # the shortest is not marked, preferred or reordered
    assert [row["sound_asset_id"] for row in report["candidates"]] == [
        "short", "medium", "too_long"]


# ------------------------------------------------------- retained native facts


def test_the_legal_query_window_agrees_with_the_catalog_on_retained_facts(retained):
    """Two independent derivations of the same window, on one real episode."""

    events = retained["events"]
    anchor, second = events[0], events[1]
    clock = cm.EpisodeClock(
        frame_count=int(retained["time"]["frame_count"]),
        frame_rate_hz=float(retained["time"]["frame_rate_hz"]),
        sample_rate_hz=int(retained["time"]["sample_rate_hz"]))
    tail = max(float(row["end_s"]) for row in retained["audio"]["wet_tail_intervals"]
               if row["event_id"] == anchor["event_id"])
    assert tail == pytest.approx(RETAINED_ANCHOR_TAIL_END_S)

    solution = cm.solve_motion_windows(
        _compile("QA-13", instances=_retained_instances()),
        clock=clock, budget=cm.MotionBudget(minimum_motion_s=1.0),
        sounds={"inst_1": _sound_for(anchor, clock)},
        event_start_s={"inst_1": float(anchor["start_s"])},
        other_event_windows_s=[(float(second["start_s"]), float(second["end_s"]))],
        registry=REGISTRY, measured_wet_tail_s=tail)

    theirs = catalog._derived_legal_query_windows(retained, "QA-13", event=anchor)
    assert theirs == [solution.query_window["legal_frames"]]
    assert solution.query_window["blocking_event_start_s"] == pytest.approx(
        RETAINED_SECOND_EVENT_START_S)
    # and that window is too short to be published at whole-second precision
    assert "no_displayable_integer_query_window" in _codes(solution)


def test_the_retained_walk_is_legal_for_qa17_and_illegal_for_the_recipe(retained):
    """One delivered trajectory, two verdicts, because they ask different things."""

    events = retained["events"]
    anchor, second = events[0], events[1]
    clock = cm.EpisodeClock(
        frame_count=int(retained["time"]["frame_count"]),
        frame_rate_hz=float(retained["time"]["frame_rate_hz"]),
        sample_rate_hz=int(retained["time"]["sample_rate_hz"]))
    tail = max(float(row["end_s"]) for row in retained["audio"]["wet_tail_intervals"]
               if row["event_id"] == anchor["event_id"])
    moving = np.asarray(retained["actors"]["source1"]["moving"], dtype=bool)
    run = np.flatnonzero(moving)
    assert (int(run[0]), int(run[-1]) + 1) == RETAINED_MOVING_RUN
    assert run[0] < int(math.ceil(tail * clock.frame_rate_hz))  # inside the tail

    positions = {inst: np.asarray(retained["actors"][actor]["emitter_positions_m"],
                                  dtype=float)
                 for inst, actor in (("inst_1", "source1"), ("inst_2", "source2"))}
    roots = {inst: np.asarray(retained["actors"][actor]["root_positions_m"], dtype=float)
             for inst, actor in (("inst_1", "source1"), ("inst_2", "source2"))}
    flags = {inst: np.asarray(retained["actors"][actor]["moving"], dtype=bool)
             for inst, actor in (("inst_1", "source1"), ("inst_2", "source2"))}
    steps = np.linalg.norm(np.diff(roots["inst_1"], axis=0), axis=1) * clock.frame_rate_hz
    walked = steps[steps > cm.DEFAULT_MOVING_THRESHOLD_MPS]
    band = (float(walked.min()), float(walked.max()))

    verdicts = {}
    for label, family in (("plain", None), ("recipe", "cross_time_state")):
        solution = cm.solve_motion_windows(
            _compile("QA-17", "yes", instances=_retained_instances(),
                     task_family=family),
            clock=clock,
            budget=cm.MotionBudget(minimum_motion_s=1.0, walk_speed_range_mps=band),
            sounds={"inst_1": _sound_for(anchor, clock)},
            event_start_s={"inst_1": float(anchor["start_s"])},
            other_event_windows_s=[(float(second["start_s"]), float(second["end_s"]))],
            registry=REGISTRY, measured_wet_tail_s=tail)
        verdicts[label] = cm.verify_motion_candidate(
            solution, positions_m=positions, root_positions_m=roots, moving_flags=flags)

    assert verdicts["plain"]["status"] == "pass"
    assert verdicts["recipe"]["status"] == "fail"
    assert verdicts["recipe"]["failed_checks"] == ["still_window"]
    inside = verdicts["recipe"]["checks"][0]["measured"]
    assert inside["window"] == [0, RETAINED_FIRST_LEGAL_MOTION_FRAME]
    assert min(inside["moving_frames_inside"]) == RETAINED_MOVING_RUN[0]


def test_the_locomotion_basis_is_recorded_and_defaults_are_not_hidden(retained):
    clock = cm.EpisodeClock(
        frame_count=int(retained["time"]["frame_count"]),
        frame_rate_hz=float(retained["time"]["frame_rate_hz"]),
        sample_rate_hz=int(retained["time"]["sample_rate_hz"]))
    anchor = retained["events"][0]
    solution = cm.solve_motion_windows(
        _compile("QA-13", instances=_retained_instances()), clock=clock,
        sounds={"inst_1": _sound_for(anchor, clock)},
        event_start_s={"inst_1": float(anchor["start_s"])}, registry=REGISTRY)
    positions = {inst: np.asarray(retained["actors"][actor]["emitter_positions_m"],
                                  dtype=float)
                 for inst, actor in (("inst_1", "source1"), ("inst_2", "source2"))}
    roots = {inst: np.asarray(retained["actors"][actor]["root_positions_m"], dtype=float)
             for inst, actor in (("inst_1", "source1"), ("inst_2", "source2"))}

    separate = cm.verify_motion_candidate(solution, positions_m=positions,
                                          root_positions_m=roots)
    merged = cm.verify_motion_candidate(solution, positions_m=positions)

    assert separate["locomotion_position_basis"] == "caller_supplied root_positions_m"
    assert "no separate root track" in merged["locomotion_position_basis"]
    assert separate["distance_position_basis"] == "caller_supplied positions_m"


def _retained_instances() -> list[dict]:
    return [
        {"entity_instance_id": "inst_1", "asset_id": "human_a",
         "source_class": "articulated_human", "role": "anchor"},
        {"entity_instance_id": "inst_2", "asset_id": "human_a",
         "source_class": "articulated_human"},
    ]


def _sound_for(event, clock: cm.EpisodeClock) -> dict:
    """The prepared segment one retained event emitted, in segment samples."""

    rate = clock.sample_rate_hz
    start = int(round(float(event["start_s"]) * rate))
    end = int(round(float(event["end_s"]) * rate))
    count = end - start
    audible = event.get("planned_audible_interval_samples")
    first, last = ((int(audible[0]) - start, int(audible[1]) - start) if audible
                   else (0, count))
    first, last = max(0, first), min(count, last)
    return {
        "sound_asset_id": str(event.get("sound_asset_id") or event["event_id"]),
        "sample_rate_hz": rate,
        "sample_count": count,
        "audible_start_sample": first,
        "audible_end_sample_exclusive": last,
        "source_activity_intervals_samples": [[first, last]],
    }
