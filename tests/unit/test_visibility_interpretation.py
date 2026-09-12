from copy import deepcopy
import pytest
from avengine.qa.visibility_interpretation import prepare_facts_for_qa, visibility_policy

POLICY = {"policy_id":"unit_tolerance", "visibility":{
    "max_hidden_visible_fraction":0.01, "min_reappeared_visible_fraction":0.05,
    "max_edge_sliver_width_fraction":0.02}}


def facts(rows, *, enabled=True):
    return {"sampling":{"acceptance_policy":deepcopy(POLICY)} if enabled else {},
            "visibility_meta":{"resolution_hw":[100,100]},
            "visibility":{"actor":{i:r for i,r in enumerate(rows)}}}


def row(fraction, box=None):
    return {"state":"visible_occluded", "visible_pixels":int(fraction*10000),
            "target_pixels":10000, "visible_fraction":fraction, "in_fov":True,
            "target_bbox_xyxy_px":box or [10,10,50,90]}


def test_strict_default_and_other_qa_unchanged():
    raw=facts([row(.001)],enabled=False)
    assert prepare_facts_for_qa(raw,"QA-09") is raw
    raw=facts([row(.001)])
    assert prepare_facts_for_qa(raw,"QA-08") is raw


def test_hidden_threshold_and_reappearance_hysteresis_preserve_measurements():
    raw=facts([row(.02),row(.005),row(.03),row(.06)])
    original=deepcopy(raw)
    result=prepare_facts_for_qa(raw,"QA-09")
    assert [r["state"] for r in result["visibility"]["actor"].values()] == [
        "visible_occluded","fully_occluded","fully_occluded","visible_occluded"]
    assert raw==original
    for i,r in result["visibility"]["actor"].items():
        assert r["visible_pixels"]==raw["visibility"]["actor"][i]["visible_pixels"]
        assert r["raw_state"]=="visible_occluded"
    assert prepare_facts_for_qa(result,"QA-09")==result


def test_only_side_edge_slivers_delay_clear_entry():
    raw=facts([row(.8,[0,10,1,80]),row(.8,[0,10,3,80]),row(.8,[40,10,41,80])])
    result=prepare_facts_for_qa(raw,"QA-07")
    assert [r["state"] for r in result["visibility"]["actor"].values()] == [
        "out_of_view","visible_occluded","visible_occluded"]


def test_no_fake_occlusion_for_offscreen_or_missing_measurement():
    raw=facts([{"state":"out_of_view","target_pixels":0,"visible_pixels":0},
               {"state":"visible_occluded"}])
    result=prepare_facts_for_qa(raw,"QA-09")
    assert [r["state"] for r in result["visibility"]["actor"].values()] == [
        "out_of_view","visible_occluded"]


@pytest.mark.parametrize("value", [float("nan"),-1,1,True])
def test_invalid_config_rejected(value):
    raw=facts([])
    raw["sampling"]["acceptance_policy"]["visibility"]["max_hidden_visible_fraction"]=value
    with pytest.raises(ValueError):
        visibility_policy(raw)
