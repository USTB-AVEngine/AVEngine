"""答案范围按场景的相机推导,不再手写度数。

手写的表只对一台相机成立:card1 那张写到 52.5 是因为这套机身 HFOV 是 105。换一个镜头
它就错,而且没有任何东西会报警——已经因此产生过外侧带各 5.0 度的死区。这里钉住的是
"同一个 profile 在不同房间里自动是对的",以及"手写值与推导值不一致会被拒绝"。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

import scene_sampler as SS  # noqa: E402

PARAMS = {"VISUAL_FOV_MARGIN_DEG": 5.0}


def scene(hfov):
    return SimpleNamespace(hfov_deg=float(hfov))


def test_camera_cone_follows_the_scene_camera():
    assert SS.answer_domain_arcs("camera_cone", scene(105), PARAMS) == \
        pytest.approx([(-47.5, 47.5)])
    # 换一个更宽的镜头,同一个域自己变宽,没人重算
    assert SS.answer_domain_arcs("camera_cone", scene(120), PARAMS) == \
        pytest.approx([(-55.0, 55.0)])


def test_rear_cone_bound_is_derived_not_written_down():
    """前后二元题的下界来自"前向镜像必须落在可信视锥内",即 |180-theta| <= H。

    2026-09-03 我是手算出 132.5 交给 owner 的;现在它是推导量,换镜头自己跟着动。
    """
    lo, hi = SS.answer_domain_arcs("rear_cone", scene(105), PARAMS)[1]
    assert (lo, hi) == pytest.approx((132.5, 180.0))
    lo, hi = SS.answer_domain_arcs("rear_cone", scene(120), PARAMS)[1]
    assert (lo, hi) == pytest.approx((125.0, 180.0))


def test_full_circle_ignores_the_camera():
    for hfov in (60, 105, 120):
        assert SS.answer_domain_arcs("full_circle", scene(hfov), PARAMS) == \
            pytest.approx([(-180.0, 180.0)])


def test_unknown_domain_is_refused():
    with pytest.raises(ValueError, match="unknown answer_domain"):
        SS.answer_domain_arcs("whatever", scene(105), PARAMS)


def test_equal_bands_across_the_cone_are_actually_equal():
    profile = {"id": "p", "answer_domain": "camera_cone",
               "answer_shape": {"equal_bands": 3}}
    bands = SS.derive_answer_bands(profile, scene(105), PARAMS)
    assert len(bands) == 3
    widths = [hi - lo for lo, hi in bands]
    assert widths == pytest.approx([95.0 / 3] * 3)
    assert bands[0][0] == pytest.approx(-47.5)
    assert bands[-1][1] == pytest.approx(47.5)


def test_eight_sectors_across_the_full_circle():
    profile = {"id": "p", "answer_domain": "full_circle",
               "answer_shape": {"equal_bands": 8}}
    bands = SS.derive_answer_bands(profile, scene(105), PARAMS)
    assert len(bands) == 8
    assert [hi - lo for lo, hi in bands] == pytest.approx([45.0] * 8)


def test_a_hand_written_table_that_disagrees_with_its_domain_is_refused():
    profile = {"id": "card1F", "answer_domain": "camera_cone",
               "answer_shape": {"equal_bands": 3},
               "answer_bands_deg": [[-52.5, -17.5], [-17.5, 17.5], [17.5, 52.5]]}
    with pytest.raises(ValueError, match="disagrees with what domain"):
        SS.audit_answer_bands(scene(105), PARAMS, [profile])


def test_a_hand_written_table_that_agrees_is_accepted():
    third = 95.0 / 3
    profile = {"id": "card1F", "answer_domain": "camera_cone",
               "answer_shape": {"equal_bands": 3},
               "answer_bands_deg": [[-47.5, -47.5 + third],
                                    [-47.5 + third, -47.5 + 2 * third],
                                    [-47.5 + 2 * third, 47.5]]}
    report = SS.audit_answer_bands(scene(105), PARAMS, [profile])
    assert "card1F" in report["derived_from_domain"]


def test_legacy_profiles_keep_working_and_get_measured_instead_of_refused():
    """21 个现存 profile 只带手写度数。它们照旧能跑——改它们生成什么是另一个决定。

    这个审计做的是把到不了的度数量出来写进 manifest,而不是拦下来。
    """
    legacy = {"id": "card1F",
              "answer_bands_deg": [[-52.5, -17.5], [-17.5, 17.5], [17.5, 52.5]]}
    report = SS.audit_answer_bands(scene(105), PARAMS, [legacy])
    assert report["derived_from_domain"] == {}
    row = report["legacy_declared_degrees"]["card1F"]
    assert row["declared_total_deg"] == pytest.approx(105.0)
    # 外侧两条带各有 52.5-47.5 = 5.0 度到不了
    assert row["unreachable_deg"] == pytest.approx(10.0)


def test_a_profile_with_no_answer_bands_is_simply_absent():
    report = SS.audit_answer_bands(scene(105), PARAMS, [{"id": "card9"}])
    assert report["derived_from_domain"] == {}
    assert report["legacy_declared_degrees"] == {}


def test_domain_profile_reaches_the_actual_cell_allocator():
    import design_qa_v3_scene_batch as batch
    profile = {"id": "card1F", "temporal": "forward", "anchor_binding": "target",
               "anchor_frame": 40, "idle_choices": [0], "answer_kind": "azimuth_band",
               "answer_domain": "camera_cone", "answer_shape": {"equal_bands": 3}}
    batch.validate_profiles([profile])
    effective, audit = SS.materialize_answer_domains(scene(120), PARAMS, [profile])
    cells = batch.build_cell_plan(18, effective, list(batch.COAT_WORDS), PARAMS, "domain")
    assert "answer_bands_deg" not in profile
    assigned = {tuple(row["answer_band"]) for row in cells}
    assert len(assigned) == 3
    assert min(lo for lo, hi in assigned) == pytest.approx(-55)
    assert max(hi for lo, hi in assigned) == pytest.approx(55)
    assert all(hi - lo == pytest.approx(110 / 3) for lo, hi in assigned)
    assert effective[0]["answer_bands_deg"] == audit["derived_from_domain"]["card1F"]


def test_outer_band_selection_preserves_the_deliberate_center_gap():
    profile = {"id": "outer", "answer_domain": "camera_cone",
               "answer_shape": {"equal_bands": 3, "band_indices": [0, 2]}}
    bands = SS.derive_answer_bands(profile, scene(120), PARAMS)
    assert len(bands) == 2
    assert bands[0] == pytest.approx((-55, -55 / 3))
    assert bands[1] == pytest.approx((55 / 3, 55))


@pytest.mark.parametrize("shape", [{"equal_bands": 2.5}, {"equal_bands": True},
                                  {"equal_bands": 3, "band_indices": [0, 0]},
                                  {"equal_bands": 3, "band_indices": [3]}])
def test_band_shape_cannot_silently_change_the_requested_options(shape):
    with pytest.raises(ValueError):
        SS.derive_answer_bands({"id": "bad", "answer_domain": "camera_cone",
                               "answer_shape": shape}, scene(105), PARAMS)
