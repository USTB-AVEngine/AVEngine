"""Transition to conditioned sampling preserves old calls and real silent endpoints."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

import avengine.rooms.qa_episode as qa


def load_tool(relative):
    path = Path(__file__).resolve().parents[2] / relative
    spec = importlib.util.spec_from_file_location(path.stem + "_p2_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def camera_scene(monkeypatch):
    candidates = [
        {"candidate_id": f"candidate_{i:02d}", "position_authoring_m": [0, i / 100, 1.55],
         "position_habitat_m": [0, 1.55, -i / 100], "horizontal_fov_deg": 85}
        for i in range(60)]
    monkeypatch.setattr(qa, "generate_camera_candidates",
                        lambda *a, **k: {"candidates": deepcopy(candidates), "generation": {}})
    monkeypatch.setattr(qa, "_load_static_triangle_geometry",
                        lambda *a: {"vertices": np.zeros((0, 3)), "triangles": np.zeros((0, 3), dtype=int)})
    monkeypatch.setattr(qa, "_mesh_ray_occluded", lambda *a: False)
    class Pathfinder:
        def is_navigable(self, point):
            return True
    routes = {"source1": np.repeat([[3.0, 0, 0]], 10, axis=0),
              "source2": np.repeat([[3.0, 0, -2]], 10, axis=0)}
    actors = [{"actor_id": aid, "emitter_local_ue_cm": [0, 0, 150]} for aid in routes]
    return Pathfinder(), routes, actors


def test_camera_samples_complete_legal_pool_and_reproduces_seed(monkeypatch):
    pf, routes, actors = camera_scene(monkeypatch)
    selected = []
    for seed in range(20):
        kwargs = dict(rng=np.random.default_rng(seed), camera_motion="static", qa_ids=["QA-01"],
                      sampling_policy="conditioned_static_v2")
        first = qa.select_question_camera({}, pf, routes, actors, **kwargs)
        second = qa.select_question_camera({}, pf, routes, actors,
                      **{**kwargs, "rng": np.random.default_rng(seed)})
        assert first == second
        record = first[2]
        assert record["checked_candidate_count"] == 60
        assert len(record["legal_candidate_ids"]) == 60
        assert record["selected_candidate_id"] in record["legal_candidate_ids"]
        selected.append(int(record["selected_candidate_id"].split("_")[-1]))
    assert len(set(selected)) > 1
    assert max(selected) >= 40


def test_legacy_camera_call_and_explicit_none_are_identical(monkeypatch):
    pf, routes, actors = camera_scene(monkeypatch)
    old = qa.select_question_camera({}, pf, routes, actors, rng=np.random.default_rng(42),
                                     camera_motion="follow_group", qa_ids=["QA-01"])
    explicit = qa.select_question_camera({}, pf, routes, actors, rng=np.random.default_rng(42),
                  camera_motion="follow_group", qa_ids=["QA-01"], sampling_policy=None)
    assert old == explicit
    assert old[2]["selection"] == "question_target_torso_geometry_and_path_clearance"
    with pytest.raises(qa.QAPlanningError, match="static camera"):
        qa.select_question_camera({}, pf, routes, actors, rng=np.random.default_rng(42),
                  camera_motion="follow_group", qa_ids=["QA-01"], sampling_policy="conditioned_static_v2")


def test_repeat_uniform_among_clips_that_fit_and_old_repeat_keeps_shortest(tmp_path):
    import soundfile as sf
    sounds = []
    for i, count in enumerate((1600, 3200, 4800, 6400)):
        path = tmp_path / f"voice{i}.wav"
        sf.write(path, np.ones(count, dtype=np.float32) * 0.1, 16000, subtype="PCM_16")
        sounds.append({"sound_asset_id": f"clip{i}", "path": str(path), "transcript": f"sentence {i}"})
    actors = [{"actor_id": f"source{i}"} for i in (1, 2)]
    clock = {"sample_rate_hz": 16000, "duration_seconds": 8}
    chose_longer = False
    for seed in range(20):
        kwargs = dict(clock=clock, rng=np.random.default_rng(seed), mode="repeat")
        events, _ = qa.schedule_audio(actors, sounds, sampling_policy="conditioned_static_v2", **kwargs)
        repeat = next(e for e in events if e["event_id"] == "event_003")
        first = [e for e in events if e["event_id"] != "event_003"]
        assert repeat["sound_asset_id"] in {e["sound_asset_id"] for e in first}
        assert repeat["end_sample"] < (8 - 1.6) * 16000
        chose_longer |= repeat["sample_count"] > min(e["sample_count"] for e in first)
        old, _ = qa.schedule_audio(actors, sounds, **{**kwargs, "rng": np.random.default_rng(seed)})
        assert old[-1]["sample_count"] == min(e["sample_count"] for e in old[:-1])
    assert chose_longer


def test_native_request_forwarding_only_new_policy_changes_defaults():
    controller = load_tool("tools/studio/run_qa_episode.py")
    assert controller._native_sampling_arguments({"camera_fov_deg": 85, "silent_actor_count": 1}) == {
        "camera_motion": "follow_group"}
    assert controller._native_sampling_arguments({"sampling_policy": "conditioned_static_v2",
        "camera": {"fov_deg": 85}, "silent_actor_count": 1}) == {
            "camera_motion": "static", "camera_fov_deg": 85,
            "sampling_policy": "conditioned_static_v2", "silent_actor_count": 1}


def test_silent_real_endpoint_is_present_without_a_fake_event():
    audio = load_tool("tools/acoustics/render_frame_readback_sequential_speech.py")
    events = [{"event_id": "e1", "actor_id": "source1", "source_endpoint_id": "source1_mouth",
               "path": "clip.wav"}]
    plan = {"visual_plan": {"actors": [{"actor_id": "source1"}, {"actor_id": "source2"}]}}
    endpoints = audio._plan_source_endpoints(plan, events)
    assert endpoints == {"source1_mouth": {"actor_id": "source1", "path": "clip.wav"},
                         "source2_mouth": {"actor_id": "source2", "path": None}}
    assert len(events) == 1
    with pytest.raises(ValueError, match="absent from the plan"):
        audio._plan_source_endpoints({"visual_plan": {"actors": [{"actor_id": "different"}]}}, events)
    with pytest.raises(ValueError, match="share a source endpoint"):
        audio._plan_source_endpoints({"visual_plan": {"actors": [
            {"actor_id": "source1", "source_endpoint_id": "source1_mouth"},
            {"actor_id": "source2", "source_endpoint_id": "source1_mouth"}]}}, events)


# ------------------------------------------------- P05: compiled conditions in the plan

TIMELINE = {"idle_action_id": "idle", "walking_action_id": "walk", "walk_phase_period_frames": 30,
            "body_plan_id": "biped_v1", "template_id": "human_v1",
            "local_anatomical_forward_axis": [1.0, 0.0, 0.0]}


def condition_registry():
    assets = [{"asset_id": f"human_{i}", "revision": "v1", "entity_class": "articulated_human",
               "identity": {"species_id": "human"}, "display_label": f"person {i}",
               "realized_attributes": {"sex_or_gender_label": "male", "top_color": color},
               "timeline": dict(TIMELINE), "default_emitter_anchor_id": "mouth",
               "emitter_anchors": [{"anchor_id": "mouth", "offset_m": [0.0, 1.6, 0.0],
                                    "offset_space": "final_scaled_asset_root"}]}
              for i, color in enumerate(["blue", "green"])]
    assets.append({"asset_id": "desk_phone_0", "revision": "v1", "entity_class": "rigid_object",
                   "identity": {"object_type": "desk_telephone", "category": "appliance"},
                   "display_label": "desk telephone", "realized_attributes": {},
                   "default_emitter_anchor_id": "body",
                   "emitter_anchors": [{"anchor_id": "body", "offset_m": [0.0, 0.75, 0.0],
                                        "offset_space": "final_scaled_asset_root"}]})
    return {"assets": assets}


def condition_sounds():
    """Prepared pool rows, including the measured activity a real row carries.

    ``source_activity_intervals_samples`` is what the P25 effective-sound
    cropping writes, in prepared-segment samples. QA-06 compiles
    ``require_source_activity_measurement``, so a row that carries only an event
    bounding box is refused as ``audible_window_missing_activity_measurement``.
    Leaving it off made this fixture unlike every row the sampler is handed in
    production, and the two tests below never reached what they were written to
    check.
    """
    rows = [{"sound_asset_id": f"speech_{i}", "sound_class": "speech", "gender": "M",
             "transcript": f"utterance {i}", "sample_count": 32000, "sample_rate_hz": 16000,
             "audible_start_sample": 800, "audible_end_sample_exclusive": 31200,
             "source_activity_intervals_samples": [[800, 31200]],
             "active_duration_s": 1.9, "path": f"/prepared/{i}.wav"} for i in range(2)]
    rows.append({"sound_asset_id": "ring_0", "sound_class": "telephone_bell_ringing",
                 "compatible_asset_ids": ["desk_phone_0"], "sample_count": 24000,
                 "sample_rate_hz": 16000, "audible_start_sample": 200,
                 "audible_end_sample_exclusive": 22600,
                 "source_activity_intervals_samples": [[200, 22600]],
                 "active_duration_s": 1.4,
                 "path": "/prepared/ring.wav"})
    return rows


def condition_plan(body):
    from avengine.qa.answerability import MeshHandle
    from avengine.rooms import conditioned_sampler as cs
    from avengine.rooms.furniture_layout import clock_config
    from avengine.rooms.walkable_space import RasterWalkableSpace
    from avengine.routes.raster_pathfinder import RasterPathfinder

    pathfinder = RasterPathfinder(np.ones((40, 40), dtype=bool), bounds_m=[[0, -1, 0], [10, 1, 10]],
                                  floor_height_m=0.0)
    space = RasterWalkableSpace(pathfinder, {"floor_height_m": 0.0, "resolution_m": 0.25,
                                             "authority": "fixture_retained_grid"})
    return cs.build_conditioned_plan(
        room={"room_id": "fixture"}, request=body, source_registry=condition_registry(),
        sounds=condition_sounds(), space=space,
        mesh=MeshHandle(np.zeros((0, 3)), np.zeros((0, 3), dtype=int)),
        clock=clock_config(frame_count=150, frame_rate_hz=15, sample_rate_hz=16000))


def branch_request(branch):
    return {"episode_id": f"branch_{branch}", "seed": 7, "sampling_policy": "conditioned_static_v2",
            "source_asset_ids": ["human_0", "human_1"], "qa_ids": ["QA-06"],
            "question_branches": {"QA-06": branch},
            "profile": {"separation_bin_deg": [15, 90], "reserve_tail_s": 1.0,
                        "retry_budget_within_profile": 60}}


def test_changing_the_answer_branch_changes_the_knobs_the_plan_was_solved_with():
    moving = condition_plan(branch_request("moving"))
    still = condition_plan(branch_request("still"))

    assert moving["condition_profile"]["speech_motion"] == "speaker_moving"
    assert moving["condition_profile"]["competitor_motion"] == "still"
    assert still["condition_profile"]["speech_motion"] == "all_still"
    # Both branches now state a competitor, and they state opposite ones: the
    # gold answer has to be separable from the alternative on either side.
    assert still["condition_profile"]["competitor_motion"] == "moving"
    assert moving["condition_profile"]["knob_sources"]["speech_motion"] == "compiled_question_condition"
    assert moving["question_condition_match"]["knob_application"] == "applied_to_condition_profile"

    def motion(plan):
        actors = plan["visual_plan"]["actors"]
        anchor = actors[plan["condition_profile"]["anchor_indices"][0]]["actor_id"]
        competitor = next(a["actor_id"] for a in actors if a["actor_id"] != anchor)
        frames = {}
        for frame in plan["visual_plan"]["frames"]:
            for state in frame["actor_states"]:
                frames.setdefault(state["actor_id"], []).append(bool(state["moving"]))
        event = next(e for e in plan["audio_events"] if e["actor_id"] == anchor)
        low, high = event["planned_audible_interval_samples"]
        fps, sr = plan["clock"]["frame_rate_hz"], plan["clock"]["sample_rate_hz"]
        window = range(int(low * fps / sr), int(math.ceil(high * fps / sr)))
        return (all(frames[anchor][f] for f in window),
                any(frames[competitor][f] for f in window))

    # (target moves through its whole audible window, any competitor motion in it).
    # The branch flips both halves: on "moving" the target walks and the
    # competitor holds still, on "still" the target holds still and the
    # competitor walks so the two answers are distinguishable. The still branch
    # used to leave the competitor still as well, which is the shape
    # unified_catalog._p8_apply_distractor_gate defers as distractors_equal_gold.
    assert motion(moving) == (True, False)
    assert motion(still) == (False, True)
    # The native facts and the question record differ, not only the candidate list.
    assert moving["visual_plan"]["camera"] != still["visual_plan"]["camera"]
    assert (moving["question_condition_match"]["sampler_profile"]
            != still["question_condition_match"]["sampler_profile"])
    assert json.dumps(moving, sort_keys=True) != json.dumps(still, sort_keys=True)


def test_each_candidate_keeps_its_own_verdict_in_a_mixed_group():
    plan = condition_plan({
        "episode_id": "mixed", "seed": 5, "sampling_policy": "conditioned_static_v2",
        "source_asset_ids": ["human_0", "desk_phone_0"],
        "entities": {"instances": [
            {"instance_id": "talker", "asset_id": "human_0", "role": "target", "speaking": True},
            {"instance_id": "phone", "asset_id": "desk_phone_0", "role": "competitor", "speaking": True}]},
        "qa_ids": ["QA-06"], "question_branches": {"QA-06": "moving"},
        "sound_class_config": {"object_sound_classes": {"desk_telephone": ["telephone_bell_ringing"]}},
        "profile": {"separation_bin_deg": [15, 90], "reserve_tail_s": 1.0,
                    "retry_budget_within_profile": 80}})
    states = {tuple(row["target_instance_ids"]): row
              for row in plan["question_condition_match"]["candidates"]}
    assert states[("talker",)]["state"] == "available"
    assert states[("phone",)]["state"] == "not_applicable_by_definition"
    # One inapplicable device candidate does not hide the human candidate.
    assert plan["question_condition_match"]["status"] == "candidate"
    assert plan["question_condition_match"]["candidate_qa_ids"] == ["QA-06"]
    assert any(row["state"] != "available" for row in plan["question_condition_match"]["blocked"])
    families = {row["entity_instance_id"]: row["entity_class"] for row in plan["entity_instances"]}
    assert families == {"talker": "articulated_human", "phone": "rigid_object"}


def test_match_question_conditions_keeps_its_room_screen_without_instances():
    capabilities = {"potential": {"min_entities": 2, "events": True, "motion": True,
                                  "appearance": True, "speech_content": True,
                                  "after_sound": True, "pixel_visibility": True,
                                  "occlusion": True, "entry": True,
                                  "distinct_sound_classes": 2, "seated": 0}}
    screen = qa.match_question_conditions(["QA-01", "QA-06"], capabilities)
    assert screen == qa.room_potential_match(["QA-01", "QA-06"], capabilities)
    assert screen["status"] == "candidate"
    assert "compiled" not in screen

    instances = [{"entity_instance_id": "i1", "source_slot_id": "source1", "asset_id": "human_0",
                  "source_class": "articulated_human", "role": "target", "speaking": True},
                 {"entity_instance_id": "i2", "source_slot_id": "source2", "asset_id": "human_1",
                  "source_class": "articulated_human", "role": "competitor", "speaking": True}]
    compiled = qa.match_question_conditions(
        ["QA-06"], capabilities, instances=instances, branches={"QA-06": "moving"},
        registry=condition_registry(), generator=__import__(
            "avengine.rooms.conditioned_sampler", fromlist=["x"]))
    assert compiled["sampler_profile"]["speech_motion"] == "speaker_moving"
    assert compiled["sampler_profile"]["competitor_motion"] == "still"
    assert compiled["room_potential"]["status"] == "candidate"
    assert compiled["compiler"].endswith("compile_target_candidates")
