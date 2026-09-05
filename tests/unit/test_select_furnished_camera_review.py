from pathlib import Path
import importlib.util
import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[2] / "tools/rooms/select_furnished_camera_review.py"
SPEC = importlib.util.spec_from_file_location("furnished_camera_select", PATH)
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def test_select_balanced_visibility_over_large_occluding_foreground_person():
    segments = [{"candidate_id": "one_large", "frame_start": 0, "frame_end": 3},
                {"candidate_id": "both_visible", "frame_start": 4, "frame_end": 7}]
    visible = {"a": np.array([900]*4 + [200]*4), "b": np.array([0]*4 + [180]*4)}
    target = {"a": np.array([1000]*8), "b": np.array([1000]*8)}
    ranking = TOOL.rank_segments(segments, visible, target)
    assert ranking[0]["candidate_id"] == "both_visible"
    assert ranking[0]["minimum_visible_pixels"] == 180
    assert ranking[0]["evaluated_frames"] == [6, 7]


def test_reject_missing_participant_and_frame_mismatch():
    segments = [{"candidate_id": "a", "frame_start": 0, "frame_end": 1}]
    with pytest.raises(ValueError, match="every actor"):
        TOOL.rank_segments(segments, {"a": [2, 2], "b": [0, 0]},
                           {"a": [2, 2], "b": [4, 4]})
    with pytest.raises(ValueError, match="counts differ"):
        TOOL.rank_segments(segments, {"a": [1, 1]}, {"a": [1]})


def test_native_target_footprints_carry_actor_ids_rather_than_boolean_values():
    visible = np.array([[[True, False, False]]])
    target = np.array([[[4, 4, 0]]], dtype=np.uint8)
    counts, footprints = TOOL.count_native_masks(visible, target, semantic_id=4, frame_count=1)
    assert counts.tolist() == [1]
    assert footprints.tolist() == [2]
    with pytest.raises(ValueError, match="another actor"):
        TOOL.count_native_masks(visible, target, semantic_id=3, frame_count=1)
