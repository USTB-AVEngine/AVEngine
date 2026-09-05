"""Planning correctness checks; these do not replace native render evidence."""
import json

import numpy as np
import pytest

from avengine.rooms.furniture_layout import FurnitureLayoutError, load_room_layout
from avengine.rooms.qa_episode import (
    QAPlanningError, build_room_navigation, navigation_points,
    sample_activity_routes, schedule_audio,
)


def object_record(name, category, low, high):
    return {"object_id": name, "semantic_class": category, "bounds_xyz_m": [low, high]}


def layout():
    return {
        "room_id": "different_name", "geometry": {"bounds_xy_m": [0, 0, 8, 8]},
        "objects": [
            object_record("slab1", "floor", [0, 0, -0.1], [8, 3, 0]),
            object_record("slab2", "floor", [0, 3, -0.1], [3, 8, 0]),
            object_record("outside_slab", "floor", [-1, -1, -0.3], [9, 9, -0.2]),
            object_record("wall", "wall", [3.9, 0, 0], [4.1, 2, 2.8]),
        ],
    }


def test_l_shaped_floor_does_not_turn_envelope_or_exterior_into_navigation():
    pf, nav = build_room_navigation(layout(), resolution_m=0.1, clearance_m=0.3)
    assert nav["floor_height_m"] == 0
    assert not pf.is_navigable([6, 0, -6])
    assert not pf.is_navigable([4, 0, -1])
    assert not pf.is_navigable([3.7, 0, -1])
    assert pf.is_navigable([2, 0, -6])
    assert "outside_slab" not in nav["floor_object_ids"]


def test_missing_floor_is_unknown_not_invented():
    data = layout()
    data["objects"] = [x for x in data["objects"] if x["semantic_class"] != "floor"]
    with pytest.raises(QAPlanningError, match="registered floor"):
        build_room_navigation(data)


def test_no_seat_room_is_loadable_for_walk_and_keeps_seated_rejection(tmp_path):
    data = {
        "room_id": "not_a_furnished_room",
        "envelope": {"bounds_xy_m": [0, 0, 8, 8]},
        "objects": [{"object_id": "floor", "category": "floor",
                     "bounds_xyz_m": [[0, 0, -0.1], [8, 8, 0]]}],
    }
    path = tmp_path / "room.json"
    path.write_text(json.dumps(data))
    with pytest.raises(FurnitureLayoutError, match="no seated affordances"):
        load_room_layout(path)
    result = load_room_layout(path, require_seats=False)
    assert result["seats"] == []
    assert result["coordinate_contract"]["habitat"] == "[authoring_x, authoring_z, -authoring_y]"


def test_routes_stay_on_floor_and_separate_after_resampling():
    pf, nav = build_room_navigation(layout(), resolution_m=0.1, clearance_m=0.3)
    routes, record = sample_activity_routes(
        pf, nav, actor_count=2, frame_count=150, fps=15,
        rng=np.random.default_rng(97), activity="walking")
    assert record["native_body_collision_status"] == "not_run"
    assert all(pf.is_navigable(p) for path in routes.values() for p in path)
    a, b = routes.values()
    assert np.min(np.linalg.norm(a - b, axis=1)) >= 0.78
    assert any(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() > 1 for p in routes.values())


def test_schedule_preserves_full_pcm_and_does_not_fix_identity_to_first_event(tmp_path):
    sf = pytest.importorskip("soundfile")
    sounds = []
    for i in range(4):
        path = tmp_path / f"voice{i}.wav"
        sf.write(path, np.ones(1600 + i * 80, dtype=np.float32) * 0.1, 16000, subtype="PCM_16")
        sounds.append({"sound_asset_id": f"voice{i}", "path": str(path),
                       "transcript": f"statement {i}", "linear_gain": 0.15})
    actors = [{"actor_id": f"source{i}"} for i in range(1, 5)]
    clock = {"sample_rate_hz": 16000, "duration_seconds": 6}
    firsts = set()
    for seed in range(8):
        events, _ = schedule_audio(actors, sounds, clock=clock,
                                  rng=np.random.default_rng(seed))
        firsts.add(events[0]["actor_id"])
        assert all(e["end_sample"] - e["start_sample"] == e["sample_count"] for e in events)
        assert max(e["end_sample"] for e in events) <= (6 - 1.2) * 16000
        assert len({e["actor_id"] for e in events}) == 4
    assert len(firsts) > 1


def test_unfit_full_speech_is_rejected_not_truncated(tmp_path):
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "long.wav"
    sf.write(path, np.ones(16000, dtype=np.float32) * 0.1, 16000, subtype="PCM_16")
    sounds = [{"sound_asset_id": str(i), "path": str(path)} for i in range(2)]
    with pytest.raises(QAPlanningError, match="do not fit"):
        schedule_audio([{"actor_id": "source1"}, {"actor_id": "source2"}], sounds,
                       clock={"sample_rate_hz": 16000, "duration_seconds": 2},
                       rng=np.random.default_rng(0))


def test_sound_preparation_can_preserve_gain_for_physical_mixing():
    from avengine.assets.sound_prepare import prepare_samples
    samples = np.sin(np.arange(16000) * 2 * np.pi * 200 / 16000) * 0.07
    prepared, facts = prepare_samples(samples, 16000, normalize_peak=False)
    assert facts["applied_gain_db"] == 0
    assert facts["peak_normalization"] is False
    assert np.max(np.abs(prepared)) < 0.071


def test_occluder_identity_requires_native_foreground_pixels(tmp_path):
    from avengine.rooms.qa_evidence import derive_actor_occluders
    modal = np.array([[[1, 1, 0, 0], [1, 1, 2, 2]]], dtype=np.uint8)
    target = np.array([[[2, 2, 0, 0], [2, 2, 2, 2]]], dtype=np.uint8)
    path = tmp_path / "masks.npz"
    np.savez_compressed(path, depth_derived_modal_semantic=modal,
                        target_only_source1=(modal == 1), target_only_source2=target)
    truth = {"status": "pass", "per_instance": {
        "source1": {"semantic_id": 1, "frames": [{"frame_index": 0, "state": "visible_clear"}]},
        "source2": {"semantic_id": 2, "frames": [{"frame_index": 0, "state": "visible_occluded"}]},
    }}
    result = derive_actor_occluders(path, truth, minimum_covered_pixels=2)
    assert result["frame_records"][0]["occluder_instance_ids"] == ["source1"]
    # An unlabeled static blocker may not acquire a guessed actor identity.
    modal[0, :1, :2] = 0
    np.savez_compressed(path, depth_derived_modal_semantic=modal,
                        target_only_source1=(modal == 1), target_only_source2=target)
    result = derive_actor_occluders(path, truth, minimum_covered_pixels=2)
    assert not result["frame_records"]


def test_pixel_color_diagnostic_rejects_tiny_visibility_and_uses_rgb():
    from avengine.rooms.qa_evidence import inspect_coarse_top_color
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[:] = [30, 75, 200]
    mask = np.ones((100, 100), dtype=bool)
    result = inspect_coarse_top_color(rgb, mask, [0, 0, 100, 100])
    assert result["observed_color"] == "blue"
    mask[:] = False
    mask[20:30, 20:30] = True
    result = inspect_coarse_top_color(rgb, mask, [0, 0, 100, 100])
    assert result["status"] == "not_observable"
