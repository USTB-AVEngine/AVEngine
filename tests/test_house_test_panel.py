"""Owner-policy regressions for a whole-house panel without score selection."""
import importlib.util
from pathlib import Path
import pytest
pytestmark = pytest.mark.fast_unit

spec = importlib.util.spec_from_file_location("panel", Path(__file__).resolve().parents[1] /
                                            "tools/acoustics/build_house_test_panel.py")
panel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panel)


def fixture():
    names = ["train", "dev", "test_a", "test_b", "candidate_c", "candidate_d", "candidate_e"]
    results = [{"house": h, "status": "pass", "known_exposure": False, "batch_eligible": True,
                "rirs": [{"status": "valid"}] * 6, "rt60_mid_median_s": 1. - i * .1,
                "result_directory": "/fixture/" + h,
                "input": {"candidate_area_sum_m2": 50. - i, "candidate_room_count": 3,
                          "room_types": ["living"], "original_split": "val"}}
               for i, h in enumerate(names)]
    initial = {"houses": [{"house": h, "split": split} for h, split in
                          [("train", "train"), ("dev", "development"), ("test_a", "test"), ("test_b", "test")]]}
    frozen = {"tiers": [{"tier": 1, "order": [{"house": h} for h in
                                              ["candidate_d", "candidate_c", "candidate_e"]]}]}
    return initial, results, frozen


def test_completion_order_and_rt60_do_not_change_the_predeclared_allocation():
    initial, results, frozen = fixture()
    rec = panel.build_recommendations(initial, list(reversed(results)), frozen, target=4)
    assert [(r["house"], r["split"]) for r in rec["houses"][:4]] == [
        ("train", "train"), ("dev", "development"), ("test_a", "test"), ("test_b", "test")]
    assert [r["house"] for r in rec["houses"] if r["split"] == "test"] == [
        "test_a", "test_b", "candidate_d", "candidate_c"]
    assert rec["target_met"] and rec["independent_test_quadruplets_generated"] == 0


def test_review_exposure_and_missing_valid_rirs_cannot_fill_a_shortfall():
    initial, results, frozen = fixture()
    by_house = {r["house"]: r for r in results}
    by_house["candidate_d"].update(status="review", batch_eligible=False)
    by_house["candidate_c"]["known_exposure"] = True
    by_house["candidate_e"]["rirs"] = [{"status": "valid"}] * 5 + [{"status": "no_energy"}]
    rec = panel.build_recommendations(initial, results, frozen, target=4)
    assert rec["house_counts"]["test"] == 2
    assert not rec["target_met"] and rec["test_house_shortfall"] == 2


def test_seed_plan_contains_sixty_independent_groups_per_house_and_four_members():
    initial, results, frozen = fixture()
    rec = panel.build_recommendations(initial, results, frozen, target=4)
    plan = panel.reserve_quadruplets(rec, 2026100300)
    assert len(plan["groups"]) == 240 and plan["members_reserved"] == 960
    assert len({g["group_id"] for g in plan["groups"]}) == 240
    assert len({m for g in plan["groups"] for m in g["member_ids"]}) == 960
    assert all(len(g["member_ids"]) == 4 and g["status"] == "reserved_not_generated" for g in plan["groups"])
    assert plan["quadruplets_generated"] == 0


def test_failed_retained_house_cannot_be_silently_replaced():
    initial, results, frozen = fixture()
    results[2].update(status="fail", batch_eligible=False)
    with pytest.raises(ValueError, match="Retained house"):
        panel.build_recommendations(initial, results, frozen, target=4)


def test_unmeasured_earlier_candidate_blocks_admission_from_later_candidates():
    initial, results, frozen = fixture()
    results = [r for r in results if r["house"] != "candidate_d"]
    rec = panel.build_recommendations(initial, results, frozen, target=4)
    assert rec["house_counts"]["test"] == 2
    assert rec["test_house_shortfall"] == 2
