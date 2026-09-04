"""Unit tests for the MCQ option builder (pilot item 1.4).

阳性对照:同扇区/同时间带/同可见性状态/未标记的负样本锚点都必须被拒;
拒出率超阈整批非零退出。位置均衡以 batch2d 的位置先验事故为回归锚:
任何 题型×切分 桶内正确位置计数差必须 ≤1。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qa"))

from build_mcq_options import (  # noqa: E402
    DEFAULT_BANDS,
    band_of,
    build,
    main,
    norm_deg,
    sector_of,
)


@pytest.mark.parametrize("deg,expected", [
    (0, "front"), (44.9, "front"), (-45, "front"), (45, "right"), (134.9, "right"),
    (135, "back"), (180, "back"), (-180, "back"), (190, "back"), (-135, "left"),
    (-45.1, "left"), (-85, "left"), (359, "front"),
])
def test_sector_boundaries(deg, expected):
    assert sector_of(deg) == expected


def test_norm_deg_half_open():
    assert norm_deg(-180) == 180.0 and norm_deg(540) == 180.0


@pytest.mark.parametrize("t,expected", [
    (0.0, 0), (1.24, 0), (1.25, 1), (2.5, 2), (3.75, 3), (5.0, 3), (5.01, None), (-0.1, None),
])
def test_band_boundaries(t, expected):
    assert band_of(t, DEFAULT_BANDS) == expected


def _card1(i, truth=-85.0, other=100.0, split="train"):
    return {"question_id": f"c1_{i:03d}", "card": "card1", "split": split,
            "truth_deg": truth, "other_ending_deg": other}


def _card9(i, truth="black_white", split="train"):
    return {"question_id": f"c9_{i:03d}", "card": "card9", "split": split,
            "truth_label": truth}


def test_position_balance_four_way_and_binary():
    items = [_card1(i) for i in range(80)] + \
            [_card9(i, truth=("black_white" if i % 2 else "yellow")) for i in range(30)]
    presented, rejected, balance, rate = build(items, seed="s", bands=DEFAULT_BANDS,
                                               reject_threshold=0.05)
    assert not rejected
    c1 = balance["card1/train/main"]["correct_position_counts"]
    assert set(c1.values()) == {20}  # 80 题四位置各 20
    c9 = balance["card9/train/main"]["correct_position_counts"]
    assert sorted(c9.values()) == [15, 15]


def test_balance_holds_within_each_split_separately():
    items = [_card1(i, split="train") for i in range(7)] + \
            [_card1(100 + i, split="eval") for i in range(5)]
    _, _, balance, _ = build(items, seed="s", bands=DEFAULT_BANDS, reject_threshold=0.05)
    for key in ("card1/train/main", "card1/eval/main"):
        counts = list(balance[key]["correct_position_counts"].values())
        assert max(counts) - min(counts) <= 1


def test_positive_control_same_sector_rejected():
    items = [_card1(0, truth=-85.0, other=-100.0)]  # 两只都在左区
    presented, rejected, _, _ = build(items, seed="s", bands=DEFAULT_BANDS,
                                      reject_threshold=1.0)
    assert not presented and rejected[0]["reason"].startswith("both endings")


def test_positive_control_same_band_rejected():
    item = {"question_id": "c8_0", "card": "card8", "split": "train",
            "truth_s": 2.6, "other_first_bark_s": 3.0}  # 同在 band_2
    presented, rejected, _, _ = build([item], seed="s", bands=DEFAULT_BANDS,
                                      reject_threshold=1.0)
    assert not presented and "same band" in rejected[0]["reason"]


def test_card8_ok_and_truth_band_label():
    item = {"question_id": "c8_1", "card": "card8", "split": "train",
            "truth_s": 2.4, "other_first_bark_s": 3.8}
    presented, rejected, _, _ = build([item], seed="s", bands=DEFAULT_BANDS,
                                      reject_threshold=1.0)
    assert not rejected and presented[0]["truth_label"] == "band_1"


def test_positive_control_same_visibility_state_rejected():
    item = {"question_id": "c16_0", "card": "card16", "split": "train",
            "truth_label": "fully_occluded", "other_ending_state": "fully_occluded"}
    presented, rejected, _, _ = build([item], seed="s", bands=DEFAULT_BANDS,
                                      reject_threshold=1.0)
    assert not presented and "same visibility" in rejected[0]["reason"]


def test_card16_out_of_view_true_rate_disclosed_in_main_bucket():
    items = [{"question_id": f"c16_{i}", "card": "card16", "split": "train",
              "truth_label": ("out_of_view" if i < 2 else "fully_occluded"),
              "other_ending_state": "visible_clear"} for i in range(8)]
    _, _, balance, _ = build(items, seed="s", bands=DEFAULT_BANDS, reject_threshold=0.05)
    assert balance["card16/train/main"]["negative_label_true_rate"] == pytest.approx(0.25)


def test_positive_control_unmarked_negative_anchor_rejected():
    bad = {"question_id": "c7_bad", "card": "card7", "split": "train",
           "truth_label": "none_calling"}
    ok = {"question_id": "c7_ok", "card": "card7", "split": "train",
          "truth_label": "none_calling", "negative_control": True}
    presented, rejected, balance, _ = build([bad, ok], seed="s", bands=DEFAULT_BANDS,
                                            reject_threshold=1.0)
    assert rejected and rejected[0]["question_id"] == "c7_bad"
    assert presented[0]["bucket"] == "control"
    assert "card7/train/control" in balance


def test_deterministic_under_same_seed():
    items = [_card1(i) for i in range(12)]
    a = build(items, seed="fixed", bands=DEFAULT_BANDS, reject_threshold=0.05)
    b = build(items, seed="fixed", bands=DEFAULT_BANDS, reject_threshold=0.05)
    assert a[0] == b[0]


def test_cli_reject_threshold_and_no_clobber(tmp_path):
    items = [_card1(0, truth=-85.0, other=-100.0)] + [_card1(i) for i in range(1, 4)]
    items_p = tmp_path / "items.json"
    items_p.write_text(json.dumps(items))
    out = tmp_path / "mcq.json"
    # 拒出率 1/4 = 25% > 5% → 非零退出,但 manifest 已写出供排查
    assert main(["--items", str(items_p), "--seed", "s", "--out", str(out)]) == 1
    assert out.exists()
    assert main(["--items", str(items_p), "--seed", "s", "--out", str(out)]) == 2


def test_cli_rejects_bad_bands(tmp_path):
    items_p = tmp_path / "i.json"
    items_p.write_text("[]")
    assert main(["--items", str(items_p), "--seed", "s", "--bands", "5,1,0",
                 "--out", str(tmp_path / "o.json")]) == 2
