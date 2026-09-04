"""Listener-orientation cross-check (renderer defensive gate).

背景:音频渲染的听者**朝向**只来自 M1 请求,而画面相机 yaw 来自捕获;
原先只交叉校验位置,朝向不校验 —— 逐点改相机 yaw 会让画面转、双耳声
不转,而且不报错。这里证明新校验:真实批(yaw −145°)通过,任何朝向
失配被抓,带俯仰的四元数被拒。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from avengine.dataset.apartment_dynamic_audio import (  # noqa: E402
    assert_listener_matches_capture_yaw,
    listener_ue_yaw_deg,
)
from avengine.timeline.current_mp3d_dynamic_audio import (  # noqa: E402
    CurrentMP3DDynamicAudioError,
    listener_pose_from_m1_request,
)

M1 = REPO / "examples/routes/fixed_apartment/m1_capture_request_review_720p.json"
CAPTURE_YAW = -145.0      # 本批全部捕获使用的相机 yaw


def yaw_quaternion(yaw_deg: float) -> list[float]:
    """habitat 世界系下绕竖轴 (y) 的 yaw-only 四元数,使前向落到
    (cos yaw, 0, sin yaw)。

    绕 y 转 θ 把 −Z 前向送到 (−sin θ, 0, −cos θ),要它等于
    (cos yaw, sin yaw) 得 θ = −(yaw + 90°)。仓库里真实 M1 请求的
    wxyz=[0.88701,0,0.46175,0] 对应 θ=55°、yaw=−145°,与此式一致。
    """
    half = math.radians(-(yaw_deg + 90.0)) / 2.0
    return [math.cos(half), 0.0, math.sin(half), 0.0]


def test_real_m1_request_matches_capture_yaw():
    _, wxyz = listener_pose_from_m1_request(json.loads(M1.read_text()))
    resolved = assert_listener_matches_capture_yaw(wxyz, CAPTURE_YAW)
    assert abs(((resolved - CAPTURE_YAW + 180) % 360) - 180) < 1e-6


def test_roundtrip_of_yaw_helper():
    for yaw in (-180.0, -145.0, -90.0, 0.0, 37.5, 179.0):
        got = listener_ue_yaw_deg(yaw_quaternion(yaw))
        assert abs(((got - yaw + 180) % 360) - 180) < 1e-6, (yaw, got)


@pytest.mark.parametrize("wrong_yaw", [-144.0, -145.5, -35.0, 35.0, 180.0])
def test_mismatched_yaw_is_rejected(wrong_yaw):
    # 听者朝 wrong_yaw、画面相机朝 −145° —— 静默失配的正是这种情形
    with pytest.raises(CurrentMP3DDynamicAudioError) as exc:
        assert_listener_matches_capture_yaw(yaw_quaternion(wrong_yaw),
                                            CAPTURE_YAW)
    assert "does not match" in str(exc.value)


def test_tilted_orientation_is_rejected():
    # 绕 x 轴 30°:前向离开水平面,单个 UE yaw 表达不了
    half = math.radians(30.0) / 2.0
    with pytest.raises(CurrentMP3DDynamicAudioError) as exc:
        listener_ue_yaw_deg([math.cos(half), math.sin(half), 0.0, 0.0])
    assert "yaw-only" in str(exc.value)


def test_malformed_orientation_is_rejected():
    with pytest.raises(CurrentMP3DDynamicAudioError):
        listener_ue_yaw_deg([1.0, 0.0, 0.0])


def test_tolerance_is_tight_enough_to_matter():
    # 0.5° 的失配在 5 米处就是几厘米级的方位错位,必须被抓
    with pytest.raises(CurrentMP3DDynamicAudioError):
        assert_listener_matches_capture_yaw(yaw_quaternion(-145.5),
                                            CAPTURE_YAW)
    # 而数值噪声级别(1e-4 度)不该误报
    assert_listener_matches_capture_yaw(yaw_quaternion(-145.0 + 1e-4),
                                        CAPTURE_YAW, tolerance_deg=1e-3)
