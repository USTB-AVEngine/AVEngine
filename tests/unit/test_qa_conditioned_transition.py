"""Transition to conditioned sampling preserves old calls and real silent endpoints."""
from copy import deepcopy
import importlib.util
import json
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
