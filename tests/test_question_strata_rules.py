"""第二轮补上的候选数口径：运动、距离、遮挡转场、听者与相机同位、遮挡者、连续角度。

这些口径都建立在从出题函数里拆出来的那几条判据上，所以这里钉的是"口径怎么用判据"，
判据本身的语义在 tests/test_observation_predicates.py。另外钉一条覆盖性：25 个题型里，
每一个要么有口径，要么有一句写明的理由，不许有第三种状态。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from avengine.dataset.question_strata import (
    CANDIDATE_RULES,
    EVIDENCE_OPTION_DOMAINS,
    QUERY_FRAME_RULES,
    UNDERIVED_REASONS,
    listener_camera_colocation,
    unimodal_candidates,
)
from avengine.qa.unified_catalog import camera_position_series

FRAMES = 30
RATE = 15.0
POLICY = {"mode": "noticeable_motion", "min_moving_duration_s": 0.8, "min_travel_m": 0.2,
          "max_still_travel_m": 0.05, "speed_threshold_mps": 0.05}


def walk(step, frames=FRAMES):
    return [[index * step, 0.0, 0.0] for index in range(frames)]


def track(*states):
    return {str(index): {"frame_index": index, "state": state} for index, state in enumerate(states)}


def facts(*, actors=None, visibility=None, listener=None, events=None, extra=None):
    body = {
        "time": {"frame_count": FRAMES, "frame_rate_hz": RATE},
        "actors": actors if actors is not None else {
            "a1": {"actor_id": "a1", "root_positions_m": walk(0.05)},
            "a2": {"actor_id": "a2", "root_positions_m": walk(0.0)},
        },
        "visibility": visibility if visibility is not None else {},
        "listener": {"status": "pass",
                     "positions_m": listener if listener is not None else [[0.0, 0.0, 0.0]] * FRAMES},
        "events": events if events is not None else [],
        "appearance_review": {},
        "sampling": {"acceptance_policy": {"motion": dict(POLICY)}},
    }
    body.update(extra or {})
    return body


def item(qa_id, evidence, options=None):
    forms = {"open": {}}
    if options is not None:
        forms["mcq"] = {"options": [{"value": value} for value in options]}
    return {"qa_id": qa_id, "truth": {"value": None, "evidence": evidence}, "forms": forms}


# --------------------------------------------------------------------------
# 覆盖性
# --------------------------------------------------------------------------


def test_每个题型要么有口径要么有一句理由():
    for qa_id in QUERY_FRAME_RULES:
        assert (qa_id in CANDIDATE_RULES) != (qa_id in UNDERIVED_REASONS), (
            f"{qa_id} 既没有口径也没有理由，或者两样都有")
    assert set(UNDERIVED_REASONS) == {"QA-07"}


def test_没有选项的题型也得说得出答案域从哪来():
    for qa_id in EVIDENCE_OPTION_DOMAINS:
        assert qa_id in CANDIDATE_RULES


# --------------------------------------------------------------------------
# 运动
# --------------------------------------------------------------------------


def test_QA06_只听按相对听者的轨迹判只看按可达性判():
    got = unimodal_candidates(
        item("QA-06", {"actor_id": "a1", "start_frame": 0, "end_frame": FRAMES - 1},
             ["moving", "still"]), facts())
    # a1 整段走了 1.45 m，听者不动，所以相对轨迹和世界轨迹判定一致 -> 只听唯一
    assert got["audio_only"] == 1
    # a1 动过、a2 整段不动，两个判定都够得着 -> 只看不唯一
    assert got["video_only"] == 2
    assert got["joint"] == 1


def test_QA06_全场没人动时只看也唯一():
    still = {"a1": {"actor_id": "a1", "root_positions_m": walk(0.0)},
             "a2": {"actor_id": "a2", "root_positions_m": walk(0.0)}}
    got = unimodal_candidates(
        item("QA-06", {"actor_id": "a1", "start_frame": 0, "end_frame": FRAMES - 1},
             ["moving", "still"]), facts(actors=still))
    assert got["video_only"] == 1


def test_QA06_听者跟着一起动时只听就判不出来():
    # 听者与声源同步平移，相对位置纹丝不动：世界坐标说动了，只听说没动，两边不一致
    got = unimodal_candidates(
        item("QA-06", {"actor_id": "a1", "start_frame": 0, "end_frame": FRAMES - 1},
             ["moving", "still"]), facts(listener=walk(0.05)))
    assert got["audio_only"] == 2


# --------------------------------------------------------------------------
# 遮挡转场
# --------------------------------------------------------------------------


def test_QA11_只看按每条轨迹的转场可达性():
    visibility = {
        "a1": track(*(["visible_occluded"] * 15 + ["visible_clear"] * 15)),
        "a2": track(*(["visible_occluded"] * 30)),
    }
    got = unimodal_candidates(item("QA-11", {"target_actor_id": "a1"}, ["yes", "no"]),
                              facts(visibility=visibility))
    assert got["audio_only"] == 2     # 声音里没有遮挡信息
    assert got["video_only"] == 2     # 一条能转、一条不能，两个答案都够得着


def test_QA09_所有候选都重新露面时只看唯一():
    visibility = {
        "a1": track(*(["fully_occluded"] * 10 + ["visible_clear"] * 20)),
        "a2": track(*(["fully_occluded"] * 5 + ["visible_clear"] * 25)),
    }
    got = unimodal_candidates(item("QA-09", {"target_actor_id": "a1"}, ["yes", "no"]),
                              facts(visibility=visibility))
    assert (got["audio_only"], got["video_only"]) == (2, 1)


def test_没有候选轨迹时写_null_加理由():
    visibility = {"a1": track(*(["visible_clear"] * 30))}
    got = unimodal_candidates(item("QA-09", {"target_actor_id": "a1"}, ["yes", "no"]),
                              facts(visibility=visibility))
    assert got["audio_only"] is None and got["reason"]


# --------------------------------------------------------------------------
# 听者与相机同位
# --------------------------------------------------------------------------


def readbacks(tmp_path, name="readbacks.json"):
    """一份 UE 形状的相机读数，外加它解出来的逐帧位置。"""

    document = {"camera": [{"frame_index": index,
                            "location_cm": [100.0 + index, 200.0, 300.0],
                            "rotation_deg": [0.0, 0.0, 0.0]} for index in range(FRAMES)]}
    path = tmp_path / name
    path.write_text(json.dumps(document))
    series = camera_position_series(document, frame_count=FRAMES)
    assert series is not None, "仓库自己的相机读法应当认得这份读数"
    return path, series["positions_m"]


def test_听者正好坐在相机上就声明同位(tmp_path):
    path, positions = readbacks(tmp_path)
    got = listener_camera_colocation(
        facts(listener=positions, extra={"source_paths": {"frame_readbacks": str(path)}}))
    assert got["colocated"] is True
    assert got["max_separation_m"] == 0.0
    assert got["frames_compared"] == FRAMES
    assert got["camera_pose_source"] == "frame_readbacks.camera"
    assert got["reason"] is None


def test_听者不在相机上就拒绝声明并给出差多少(tmp_path):
    path, positions = readbacks(tmp_path, "moved.json")
    shifted = [[x + 1.0, y, z] for x, y, z in positions]
    got = listener_camera_colocation(
        facts(listener=shifted, extra={"source_paths": {"frame_readbacks": str(path)}}))
    assert got["colocated"] is False
    assert got["max_separation_m"] == pytest.approx(1.0)
    assert "不是同一个点" in got["reason"]


def test_读不到相机位姿就写_null_加理由():
    got = listener_camera_colocation(facts(extra={"source_paths": {}}))
    assert got["colocated"] is None
    assert "frame_readbacks" in got["reason"]


def test_QA14_同位了只看就唯一(tmp_path):
    path, positions = readbacks(tmp_path, "qa14.json")
    got = unimodal_candidates(
        item("QA-14", {"query_frame": 5, "appearance_reviews": {"a1": {}, "a2": {}}},
             ["a1", "a2"]),
        facts(listener=positions, extra={"source_paths": {"frame_readbacks": str(path)}}))
    assert (got["audio_only"], got["video_only"], got["joint"]) == (2, 1, 1)
    assert "同位" in got["basis"] or "相等" in got["basis"]


def test_QA14_没同位就写_null_加理由(tmp_path):
    path, positions = readbacks(tmp_path, "qa14b.json")
    shifted = [[x, y + 2.0, z] for x, y, z in positions]
    got = unimodal_candidates(
        item("QA-14", {"query_frame": 5, "appearance_reviews": {"a1": {}, "a2": {}}},
             ["a1", "a2"]),
        facts(listener=shifted, extra={"source_paths": {"frame_readbacks": str(path)}}))
    assert got["audio_only"] is None
    assert "同位" in got["reason"]


# --------------------------------------------------------------------------
# 遮挡者与连续角度：没有 MCQ 选项，答案域从证据或题面推
# --------------------------------------------------------------------------


def test_QA10_候选域是查询时刻画面里除目标外可见的个体():
    visibility = {"a1": track(*(["visible_occluded"] * 30)),
                  "a2": track(*(["visible_clear"] * 30)),
                  "a3": track(*(["out_of_view"] * 30))}
    got = unimodal_candidates(
        item("QA-10", {"target_actor_id": "a1", "query_frame": 5,
                       "option_instance_ids": ["a2"]}),
        facts(visibility=visibility))
    # a3 整段不在画面里，不是候选；a1 是目标自己
    assert got["option_count"] == 1
    assert got["option_domain_source"] == "derived"
    assert got["video_only"] == 1


def test_QA10_画面里没有别人就写_null_加理由():
    visibility = {"a1": track(*(["visible_occluded"] * 30))}
    got = unimodal_candidates(
        item("QA-10", {"target_actor_id": "a1", "query_frame": 5}), facts(visibility=visibility))
    assert got["audio_only"] is None and got["reason"]


def test_QA25_看的那一支只看唯一听的那一支只听唯一():
    visual = unimodal_candidates(
        item("QA-25", {"actor_id": "a1", "query_frame": 5, "subset": "V",
                       "angle_target": "visible_pixel_centroid"}), facts())
    assert (visual["audio_only"], visual["video_only"], visual["option_count"]) == (360, 1, 360)
    audio = unimodal_candidates(
        item("QA-25", {"actor_id": "a1", "query_frame": 5, "subset": "A",
                       "angle_target": "sound_emitter"}), facts())
    assert (audio["audio_only"], audio["video_only"]) == (1, 360)


def test_QA13_静音里声源没动过时只听还能唯一():
    evidence = {"actor_id": "a2", "end_frame": 0, "query_frame": 20,
                "azimuth_at_query_deg": 0.0, "fov_band_boundaries_deg": [-40.0, -13.5, 13.5, 40.0]}
    got = unimodal_candidates(
        item("QA-13", evidence, ["fov_band_0", "fov_band_1", "fov_band_2"]), facts())
    assert got["audio_only"] == 1     # a2 整段不动
    assert got["video_only"] == 3
    assert "没有明显移动" in got["basis"]


def test_QA13_静音里声源动过就每条带都排除不掉():
    evidence = {"actor_id": "a1", "end_frame": 0, "query_frame": 29,
                "azimuth_at_query_deg": 0.0, "fov_band_boundaries_deg": [-40.0, -13.5, 13.5, 40.0]}
    got = unimodal_candidates(
        item("QA-13", evidence, ["fov_band_0", "fov_band_1", "fov_band_2"]), facts())
    assert got["audio_only"] == 3     # a1 走了 1.45 m
    assert "动过" in got["basis"]


def test_QA13_开放作答那一支两边都定不了():
    got = unimodal_candidates(
        item("QA-13", {"actor_id": "a1", "end_frame": 0, "query_frame": 29}), facts())
    assert (got["audio_only"], got["video_only"], got["option_count"]) == (360, 360, 360)
    assert got["necessary_multimodal"] is True
