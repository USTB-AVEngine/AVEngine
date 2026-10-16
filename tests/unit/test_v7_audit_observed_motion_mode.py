import json
import pytest
from avengine.timeline.v7_audit_actor_kinds import observed_motion_modes,actor_kind
pytestmark=pytest.mark.fast_unit


def test_slide_category_comes_from_complete_actual_native_admission(tmp_path):
    p=tmp_path/'native.json';record={'asset_id':'animal','status':'pass','motion_mode':'slide_no_animation','final_complete_scene_recheck':{'status':'pass'},'native_30Hz_readbacks':[{'asset_id':'animal'}for _ in range(901)]}
    p.write_text(json.dumps({'native_motion_admissions':{'voice1':record}}));modes=observed_motion_modes(str(p))
    actor={'species_id':'dog','source':{'motion_mode':modes['animal']}}
    assert actor_kind(actor)=='slide'
    assert actor_kind({'species_id':'dog','speed_mps':[.45]})=='dog'
    record['native_30Hz_readbacks'].pop();q=tmp_path/'incomplete.json';q.write_text(json.dumps({'native_motion_admissions':{'voice1':record}}))
    assert observed_motion_modes(str(q))=={}
