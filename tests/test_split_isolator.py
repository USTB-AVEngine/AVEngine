"""Unit tests for the split isolator (pilot item 1.3).

阳性对照:人为构造跨集孪生与跨集说话人泄漏,扫描器必须抓到;分配器
自出的计划必须过自己的扫描;单值维(整批同房间)必须被显式报告而不是
静默跳过或误报违规。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qa"))

from split_isolator import assign, check, main  # noqa: E402


def _pt(pid, twin_of=None, room="apartment_0000", traj=None, speakers=None, scripts=None):
    return {"point_id": pid, "episode_id": pid, "twin_of": twin_of, "room_id": room,
            "trajectory_id": traj or f"traj_{pid}",
            "speaker_voice_ids": speakers or [], "transcript_ids": scripts or []}


def _batch(n=40, twin_every=4):
    pts = []
    for i in range(n):
        pid = f"p{i:03d}"
        pts.append(_pt(pid))
        if i % twin_every == 0:
            pts.append(_pt(pid + "_twin", twin_of=pid, traj=f"traj_{pid}"))
    return pts


def test_twins_stay_on_the_same_side():
    pts = _batch()
    plan, _, _ = assign(pts, {"train": 0.8, "eval": 0.2}, seed="s1")
    for p in pts:
        if p["twin_of"]:
            assert plan[p["point_id"]] == plan[p["twin_of"]]


def test_shared_speaker_locks_points_together():
    pts = [_pt("a", speakers=["voice_1"]), _pt("b", speakers=["voice_1"]),
           _pt("c", speakers=["voice_2"])] + _batch(20, twin_every=99)
    plan, _, _ = assign(pts, {"train": 0.5, "eval": 0.5}, seed="s2")
    assert plan["a"] == plan["b"]


def test_soft_dim_whole_batch_reported_not_faked():
    pts = _batch()  # 全部同一个 room(软维全覆盖)
    plan, unisolated, _ = assign(pts, {"train": 0.8, "eval": 0.2}, seed="s3")
    assert any(u["dimension"] == "room_id" for u in unisolated)
    # 软维全覆盖不许把全批锁成一个分量:两个 split 都必须有点
    assert len(set(plan.values())) == 2


def test_empty_values_do_not_link_points():
    # 两个都没有说话人的点,不因"空"被锁同侧(能各自分配)
    pts = [_pt(f"q{i}") for i in range(10)]
    plan, _, comps = assign(pts, {"train": 0.5, "eval": 0.5}, seed="s4")
    assert len(comps) == 10  # 每点自成分量


def test_assign_ratio_roughly_met_and_deterministic():
    pts = _batch(60, twin_every=5)
    plan1, _, _ = assign(pts, {"train": 0.8, "eval": 0.2}, seed="same")
    plan2, _, _ = assign(pts, {"train": 0.8, "eval": 0.2}, seed="same")
    assert plan1 == plan2
    n = len(pts)
    train_share = sum(1 for s in plan1.values() if s == "train") / n
    assert 0.7 <= train_share <= 0.9


def test_check_catches_injected_twin_leak():
    pts = [_pt("m"), _pt("m_twin", twin_of="m", traj="traj_m"), _pt("x")]
    bad_plan = {"m": "train", "m_twin": "eval", "x": "train"}  # 人为拆开孪生
    violations, _ = check(pts, bad_plan)
    kinds = {(v.get("dimension"), v.get("kind")) for v in violations}
    assert ("twin_group", "cross_split_leak") in kinds


def test_check_catches_speaker_leak_even_when_value_covers_batch():
    # 语义修正的回归锚:硬维(说话人)即使该值覆盖全批,跨集也必须报违规
    # ——早先按"该维只有一个取值"豁免会把这种真泄漏漏掉
    pts = [_pt("a", speakers=["v1"]), _pt("b", speakers=["v1"])]
    violations, unisolated = check(pts, {"a": "train", "b": "eval"})
    assert any(v.get("dimension") == "speaker_voice_ids" for v in violations)
    assert any(u["dimension"] == "room_id" for u in unisolated)  # 软维:标注不违规
    assert not any(v.get("dimension") == "room_id" for v in violations)


def test_check_flags_unassigned_points():
    pts = [_pt("a"), _pt("b")]
    violations, _ = check(pts, {"a": "train"})
    assert any(v["kind"] == "unassigned_points" for v in violations)


def test_cli_end_to_end_and_no_clobber(tmp_path):
    pts = _batch(24)
    points_p = tmp_path / "points.json"
    points_p.write_text(json.dumps(pts))
    plan_p = tmp_path / "plan.json"
    assert main(["assign", "--points", str(points_p), "--ratios", "train=0.8,eval=0.2",
                 "--seed", "s", "--out", str(plan_p)]) == 0
    report_p = tmp_path / "report.json"
    assert main(["check", "--points", str(points_p), "--plan", str(plan_p),
                 "--out", str(report_p)]) == 0
    doc = json.loads(report_p.read_text())
    assert doc["violation_count"] == 0
    assert any(u["dimension"] == "room_id" for u in doc["unisolated"])
    # no-clobber
    assert main(["assign", "--points", str(points_p), "--ratios", "train=0.8,eval=0.2",
                 "--seed", "s", "--out", str(plan_p)]) == 2


def test_cli_check_fails_on_bad_plan(tmp_path):
    pts = [_pt("m"), _pt("m_twin", twin_of="m", traj="traj_m")]
    points_p = tmp_path / "p.json"
    points_p.write_text(json.dumps(pts))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"m": "train", "m_twin": "eval"}))
    out = tmp_path / "r.json"
    assert main(["check", "--points", str(points_p), "--plan", str(bad),
                 "--out", str(out)]) == 1


def test_cli_rejects_bad_ratios(tmp_path):
    points_p = tmp_path / "p.json"
    points_p.write_text(json.dumps([_pt("a")]))
    assert main(["assign", "--points", str(points_p), "--ratios", "train=0.8,eval=0.3",
                 "--seed", "s", "--out", str(tmp_path / "o.json")]) == 2
