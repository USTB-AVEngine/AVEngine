"""A camera request can demand a source the camera never shows.

A direction question about a source inside the frustum is answerable by looking, so this
is what separates a question that needs spatial hearing from one that does not.
"""
from copy import deepcopy

import numpy as np
import pytest

from avengine.rooms import qa_episode as qa


def _scene(monkeypatch):
    """Two actors: source1 always visible, source2 visible only on even candidates."""
    candidates = [
        {"candidate_id": f"candidate_{i:02d}", "position_authoring_m": [0, i / 100, 1.55],
         "position_habitat_m": [0, 1.55, -i / 100], "horizontal_fov_deg": 85}
        for i in range(60)
    ]
    monkeypatch.setattr(qa, "generate_camera_candidates",
                        lambda *a, **k: {"candidates": deepcopy(candidates), "generation": {}})
    monkeypatch.setattr(qa, "_load_static_triangle_geometry",
                        lambda *a: {"vertices": np.zeros((0, 3)), "triangles": np.zeros((0, 3), dtype=int)})
    monkeypatch.setattr(qa, "_mesh_ray_occluded", lambda *a: False)

    def visible(camera, point):
        if float(point[1]) < 1.0:          # source1 sits at y = 0
            return True
        return int(str(camera["candidate_id"]).split("_")[-1]) % 2 == 0

    monkeypatch.setattr(qa, "_projected_visible", visible)

    class Pathfinder:
        def is_navigable(self, point):
            return True

    routes = {"source1": np.repeat([[3.0, 0, 0]], 10, axis=0),
              "source2": np.repeat([[3.0, 0, -2]], 10, axis=0)}
    actors = [{"actor_id": aid, "emitter_local_ue_cm": [0, 0, 150]} for aid in routes]
    return Pathfinder(), routes, actors


def _call(pf, routes, actors, **extra):
    return qa.select_question_camera(
        {}, pf, routes, actors, rng=np.random.default_rng(7), camera_motion="static",
        qa_ids=["QA-04"], sampling_policy="conditioned_static_v2", **extra,
    )


def test_without_a_request_every_candidate_stays_legal(monkeypatch):
    pf, routes, actors = _scene(monkeypatch)
    record = _call(pf, routes, actors)[2]
    assert len(record["legal_candidate_ids"]) == 60
    assert record["offscreen_actor_ids"] == []


def test_requesting_a_source_off_screen_keeps_only_cameras_that_never_show_it(monkeypatch):
    pf, routes, actors = _scene(monkeypatch)
    record = _call(pf, routes, actors, offscreen_actor_ids=["source2"])[2]
    legal = record["legal_candidate_ids"]
    assert len(legal) == 30
    assert all(int(name.split("_")[-1]) % 2 for name in legal), "an even camera can see source2"
    assert record["offscreen_actor_ids"] == ["source2"]


def test_a_source_the_camera_always_shows_cannot_be_requested_off_screen(monkeypatch):
    pf, routes, actors = _scene(monkeypatch)
    with pytest.raises(qa.QAPlanningError, match="out of view"):
        _call(pf, routes, actors, offscreen_actor_ids=["source1"])


def test_naming_an_actor_the_episode_does_not_have_is_refused(monkeypatch):
    pf, routes, actors = _scene(monkeypatch)
    with pytest.raises(qa.QAPlanningError, match="no route for"):
        _call(pf, routes, actors, offscreen_actor_ids=["source9"])


def test_the_request_is_reproducible_under_one_seed(monkeypatch):
    pf, routes, actors = _scene(monkeypatch)
    first = _call(pf, routes, actors, offscreen_actor_ids=["source2"])
    second = _call(pf, routes, actors, offscreen_actor_ids=["source2"])
    assert first == second
