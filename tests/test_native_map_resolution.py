"""The captured map is a declared fact, not a module constant.

地图曾经是模块常量:渲染层因此绑死一个房间,而且更糟——为某个房间创作
的时间线可以被拿到另一张地图上捕获而不报警。这里证明新解析:时间线
自带 room.map_path、显式参数只在与之一致时才允许,失配 fail-closed。
与相机 yaw 那条缺陷是同一种形状:声明的事实与执行的事实不能悄悄分叉。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from avengine.timeline.current_apartment_visual import (  # noqa: E402
    NATIVE_APARTMENT_MAP,
    CurrentApartmentVisualError,
    resolve_native_map,
)


def test_default_preserves_the_existing_apartment_behaviour():
    assert resolve_native_map({}) == NATIVE_APARTMENT_MAP
    assert resolve_native_map(None) == NATIVE_APARTMENT_MAP
    assert resolve_native_map({"room": {}}) == NATIVE_APARTMENT_MAP


def test_timeline_declares_its_own_room():
    timeline = {"room": {"map_path": "/Game/SPEAR/Scenes/debug_0000/Maps/debug_0000"}}
    assert resolve_native_map(timeline) == \
        "/Game/SPEAR/Scenes/debug_0000/Maps/debug_0000"


def test_explicit_argument_must_agree_with_the_timeline():
    timeline = {"room": {"map_path": "/Game/A/Map"}}
    assert resolve_native_map(timeline, "/Game/A/Map") == "/Game/A/Map"
    with pytest.raises(CurrentApartmentVisualError) as exc:
        resolve_native_map(timeline, "/Game/B/Other")
    assert "authored for map" in str(exc.value)


def test_explicit_argument_alone_is_honoured_for_timelines_without_a_room():
    assert resolve_native_map({}, "/Game/C/Third") == "/Game/C/Third"


@pytest.mark.parametrize("bad", ["debug_0000", "/Content/Foo", "Game/X"])
def test_non_package_paths_are_refused(bad):
    with pytest.raises(CurrentApartmentVisualError) as exc:
        resolve_native_map({}, bad)
    assert "/Game package path" in str(exc.value)


def test_declared_but_empty_map_is_an_error_not_a_fallback():
    """声明了却是空的,不能悄悄回落到默认房间 —— 那是静默分叉。"""
    with pytest.raises(CurrentApartmentVisualError) as exc:
        resolve_native_map({"room": {"map_path": ""}})
    assert "declared-but-blank" in str(exc.value)
    with pytest.raises(CurrentApartmentVisualError):
        resolve_native_map({"room": {"map_path": None}})
