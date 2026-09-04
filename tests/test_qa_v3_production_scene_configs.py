"""一个房间一份配置,而且这一份必须是完整的。

2026-09-04 之前 apartment 这一个房间在 /data/jzy/tmp 下有六份场景配置并存,名字看不出
哪份是当前的:codex_v3 听着比 floor_v2 新,synthesis_20260903_v1 和 floor_20260903_v2
还是同一天。六份是严格累积的,所以合并本身不难;难的是**挑错了会怎样**——
codex_v3 和 synthesis_v1 里 render.ground_z_ue_cm 是手写的 0.0,就是让所有渲染的狗
陷进地板 27 cm 的那个值;floor_v1 把 camera_clearance_table 弄丢了。它们不是过期,是错的。

这个文件把"生产房间"这件事变成可执行的:examples/qa/scenes 下的每一份都要过完整校验,
而单元测试自己拼的三键配置不受影响——那不是生产房间。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

import scene_sampler as SS  # noqa: E402

SCENES = sorted((REPO / SS.PRODUCTION_SCENE_DIR).glob("*.json"))


def test_there_is_at_least_one_production_room():
    assert SCENES, f"no scene config under {SS.PRODUCTION_SCENE_DIR}"


@pytest.mark.parametrize("path", SCENES, ids=lambda p: p.stem)
def test_every_production_scene_config_is_complete(path):
    SS.require_production_scene_config(json.loads(path.read_text()), path)


@pytest.mark.parametrize("path", SCENES, ids=lambda p: p.stem)
def test_a_room_appears_exactly_once(path):
    ids = [json.loads(p.read_text())["scene_id"] for p in SCENES]
    assert ids.count(json.loads(path.read_text())["scene_id"]) == 1


@pytest.mark.parametrize("path", SCENES, ids=lambda p: p.stem)
def test_render_facts_come_with_a_measured_floor(path):
    """挑错配置最贵的那一种后果:手写的地板高度。"""
    doc = json.loads(path.read_text())
    if not doc.get("render"):
        pytest.skip("no render facts in this config")
    assert "floor_reference" in doc
    ground = float(doc["render"]["ground_z_ue_cm"])
    measured = json.loads(
        (Path(doc["floor_reference"]) / "floor_reference.json").read_text())
    assert measured["scene_id"] == doc["scene_id"]
    assert measured["status"] == "measured"
    assert abs(ground - float(measured["ground_z_ue_cm"])) <= 1.0, (
        f"declared {ground} vs measured {measured['ground_z_ue_cm']}")


# ── 每一条规矩都配一个反例,否则校验器可能什么都没在拦 ──────────────────────
def _good():
    return json.loads(SCENES[0].read_text())


def test_a_missing_key_is_refused():
    doc = _good(); doc.pop("walkable_grid")
    with pytest.raises(ValueError, match="missing"):
        SS.require_production_scene_config(doc, SCENES[0])


def test_a_foreign_schema_is_refused():
    doc = _good(); doc["schema"] = "something_else"
    with pytest.raises(ValueError, match="expected"):
        SS.require_production_scene_config(doc, SCENES[0])


def test_a_file_name_that_disagrees_with_scene_id_is_refused():
    doc = _good()
    with pytest.raises(ValueError, match="one room, one file"):
        SS.require_production_scene_config(doc, Path("/x/some_other_name.json"))


def test_render_facts_without_a_floor_reference_are_refused():
    """floor_v1 那次悄悄丢掉 camera_clearance_table,就是这一条要拦的形状。"""
    doc = _good(); doc.pop("floor_reference")
    with pytest.raises(ValueError, match="render facts"):
        SS.require_production_scene_config(doc, SCENES[0])
    doc = _good(); doc.pop("camera_clearance_table")
    with pytest.raises(ValueError, match="render facts"):
        SS.require_production_scene_config(doc, SCENES[0])


# ── 一个房间一份，而且以后也不许有新的（owner 2026-09-04）──────────────────
def test_the_canonical_config_is_accepted():
    got = SS.resolve_production_scene_config(SCENES[0], repo_root=REPO)
    assert got == SCENES[0].resolve()


@pytest.mark.parametrize("stale", [
    "/data/jzy/tmp/qa_v3_scene_apartment_codex_v3.json",
    "/data/jzy/tmp/qa_v3_scene_configs_floor_20260903_v2/apartment_0000.json",
    "/data/jzy/tmp/somewhere_else/apartment_0000.json",
])
def test_a_config_from_anywhere_else_is_refused(stale):
    """标死旧文件不够:标死的文件照样能被指着加载。

    floor_20260903_v2 也在这张表里,尽管仓库里那份的内容就是从它合来的——
    内容合进来之后再从 /tmp 加载就是走回头路,这条要一并拦掉。
    """
    with pytest.raises(ValueError, match="one scene config|not under"):
        SS.resolve_production_scene_config(stale, repo_root=REPO)


def test_replaying_an_old_batch_still_has_a_door():
    """历史复现要能走,但必须明说,不能是默认。"""
    stale = "/data/jzy/tmp/qa_v3_scene_apartment_codex_v3.json"
    got = SS.resolve_production_scene_config(
        stale, repo_root=REPO, historical_reproduction=True)
    assert got == Path(stale).resolve()


def test_a_name_under_the_canonical_directory_that_does_not_exist_is_refused():
    missing = REPO / SS.PRODUCTION_SCENE_DIR / "no_such_room.json"
    with pytest.raises(ValueError, match="does not exist"):
        SS.resolve_production_scene_config(missing, repo_root=REPO)


# ── 一个房间一份，而且以后也不许有新的（owner 2026-09-04）──────────────────
def test_the_canonical_config_is_accepted():
    got = SS.resolve_production_scene_config(SCENES[0], repo_root=REPO)
    assert got == SCENES[0].resolve()


@pytest.mark.parametrize("stale", [
    "/data/jzy/tmp/qa_v3_scene_apartment_codex_v3.json",
    "/data/jzy/tmp/qa_v3_scene_configs_floor_20260903_v2/apartment_0000.json",
    "/data/jzy/tmp/somewhere_else/apartment_0000.json",
])
def test_a_config_from_anywhere_else_is_refused(stale):
    """标死旧文件不够:标死的文件照样能被指着加载。

    floor_20260903_v2 也在这张表里,尽管仓库里那份的内容就是从它合来的——
    内容合进来之后再从 /tmp 加载就是走回头路,这条要一并拦掉。
    """
    with pytest.raises(ValueError, match="one scene config|not under"):
        SS.resolve_production_scene_config(stale, repo_root=REPO)


def test_replaying_an_old_batch_still_has_a_door():
    """历史复现要能走,但必须明说,不能是默认。"""
    stale = "/data/jzy/tmp/qa_v3_scene_apartment_codex_v3.json"
    got = SS.resolve_production_scene_config(
        stale, repo_root=REPO, historical_reproduction=True)
    assert got == Path(stale).resolve()


def test_a_name_under_the_canonical_directory_that_does_not_exist_is_refused():
    missing = REPO / SS.PRODUCTION_SCENE_DIR / "no_such_room.json"
    with pytest.raises(ValueError, match="does not exist"):
        SS.resolve_production_scene_config(missing, repo_root=REPO)
