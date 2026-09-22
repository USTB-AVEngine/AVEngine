"""The off-screen request has to survive the trip from the caller to the camera planner.

The planner's own behaviour is covered in test_qa_offscreen_camera; what is under test
here is only that a request carrying the key still carries it when it arrives.
"""
from avengine.rooms import native_qa_room as nq
from tests.unit.test_native_qa_room import _resources, _sounds


def _capture(monkeypatch):
    captured = {}

    def recorder(*, room, request, source_registry, sounds):
        captured.update(request)
        return ({"visual_plan": {"frames": [], "actors": []}, "resources": {}}, {}, object())

    monkeypatch.setattr("avengine.rooms.qa_episode.build_qa_episode_plan", recorder)
    return captured


def test_an_off_screen_request_reaches_the_delegated_episode_request(tmp_path, monkeypatch):
    captured = _capture(monkeypatch)
    nq.build_native_apartment_qa_plan(
        resources=_resources(tmp_path), source_registry={"assets": []}, sounds=_sounds(),
        episode_id="offscreen", source_asset_ids=["human_0", "human_1"],
        sampling_policy="conditioned_static_v2", offscreen_actor_ids=["source2"])
    assert captured["offscreen_actor_ids"] == ["source2"]


def test_an_episode_that_asks_for_nothing_builds_the_request_it_always_built(tmp_path, monkeypatch):
    captured = _capture(monkeypatch)
    nq.build_native_apartment_qa_plan(
        resources=_resources(tmp_path), source_registry={"assets": []}, sounds=_sounds(),
        episode_id="plain", source_asset_ids=["human_0", "human_1"],
        sampling_policy="conditioned_static_v2")
    assert "offscreen_actor_ids" not in captured
