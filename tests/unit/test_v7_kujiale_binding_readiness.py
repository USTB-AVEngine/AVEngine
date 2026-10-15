import copy
import pytest
from avengine.timeline.v7_kujiale_binding_readiness import native_binding_readiness
pytestmark=pytest.mark.fast_unit


def inputs():
    assets=[{'endpoint_id': 'voice'+str(i), 'asset_id': c} for i,c in enumerate(['blue','green','yellow'],1)]
    scene={'production_scenario': {'camera_policy':'binding_static','recipe_revision':'kujiale-fixed-natural-construction-v2'}}
    actors=[dict(a,actor_id=str(i),visible_pixels=40000) for i,a in enumerate(assets,1)]
    review={'review': {'actors': {str(i): {'status':'reviewed','frame_refs':[0,15,30],
              'review_method':'automatic_native_color_pixels'} for i in range(1,4)}}}
    return scene,{'frames':[{'actors':actors}]},review,assets


def test_all_required_native_appearances_pass_without_changing_the_review():
    args=inputs();before=copy.deepcopy(args);r=native_binding_readiness(*args)
    assert r['status']=='pass' and not r['question_thresholds_changed']
    assert args==before


def test_occluded_speaker_cannot_borrow_another_persons_appearance():
    scene,visual,review,assets=inputs();visual['frames'][0]['actors'][0]['visible_pixels']=1197
    del review['review']['actors']['1']
    r=native_binding_readiness(scene,visual,review,assets)
    assert r['status']=='needs_new_layout'
    assert [c['endpoint_id'] for c in r['checks'] if c['status']!='pass']==['voice1']
    assert len(r['checks'])==3 and 'retain all required' in r['action_on_missing']


def test_out_of_view_silent_comparison_body_requests_construction_before_audio():
    scene,visual,review,assets=inputs();del review['review']['actors']['3']
    r=native_binding_readiness(scene,visual,review,assets)
    assert r['status']=='needs_new_layout' and r['checks'][-1]['endpoint_id']=='voice3'


def test_other_explicit_offscreen_or_occlusion_profiles_remain_eligible():
    scene,visual,review,assets=inputs();scene['production_scenario']['recipe_revision']='explicit-occlusion-construction'
    assert native_binding_readiness(scene,visual,{'review':{}},assets)['status']=='not_requested'
