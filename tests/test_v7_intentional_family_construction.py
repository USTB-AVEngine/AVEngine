import importlib.util,runpy,copy
from pathlib import Path
import numpy as np
import pytest
W=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.fast_unit
from avengine.timeline import v7_specific_family_construction as static,v7_visibility_construction as moving
from avengine.capture import v7_camera_path_proposals as route,v7_dynamic_occlusion_proposals as occlusion

@pytest.fixture
def registry():
    original=runpy.run_path(str(W/'tests/test_v7_family_automation.py'))['inputs'].__wrapped__()
    rooms,houses,assets,semantic,contract=original
    for color in ('white','black'):
        for gender in ('female','male'):
            assets.append({**copy.deepcopy(assets[0]),'asset_id':gender+'_'+color,'color':color,'visual_gender':gender})
    records=[{'sound_asset_id':f'whole_{gender}_{voice}_{line}','sound_class':'speech_playback',
        'dataset':'VCTK-Corpus 0.92','legacy_library_split':'train','source_type':'recorded',
        'gender':gender,'speaker_id':f'external_{gender}_{voice}',
        'transcript':f'The complete sentence is number {line}.','duration_s':1.+line/10}
        for gender in ('M','F')for voice in range(4)for line in range(3)]
    return rooms[0],houses[rooms[0]['house']],assets,records,semantic,contract

@pytest.mark.parametrize('family',static.SPECIFIC_FAMILIES)
def test_static_explicit_branches_keep_real_people_whole_speech_and_fixed_camera(registry,family):
    room,house,catalog,records,semantic,contract=registry
    row=static.construct_specific_scene(room,house,{},catalog,records,semantic,contract,{},
        scene_id='new_explicit',seed=51,family=family)['row']
    assert 3<=len(row['asset_ids'])<=5
    assert row['automatic_construction']['family']==family
    assert row['native_view_roles']
    assert all(kinds==['speech_playback']for kinds in row['source_classes'].values())
    assert len({k['yaw_deg']for k in row['camera_yaw_knots']})==1
    if family=='QA-05':assert row['semantic_start_offsets_s']==[0.,.1]
    if family=='QA-14':
        assert len(row['native_distance_bands_m'])==3
        for aid,band in row['native_distance_bands_m'].items():
            p=np.array(row['native_placement_preferences_m'][aid]);listener=np.array(row['listener_m'])
            distance=np.linalg.norm((p-listener)[[0,2]])
            assert band[0]<=distance<=band[1]
    if family in {'QA-23','STARSS-CLASS13','STARSS-SPEAKERS'}:
        assert row['explicit_class13_count']['live_speakers']==len(row['asset_ids'])

@pytest.mark.parametrize('family',moving.VISIBILITY_FAMILIES)
def test_visibility_branches_request_native_route_and_never_add_shoe_sounds(registry,family):
    room,house,catalog,records,semantic,contract=registry
    row=moving.construct_visibility_scene(room,house,{},catalog,records,semantic,contract,{},
        scene_id='new_visibility',seed=51,family=family)['row']
    assert len(row['motion_requests'])==1
    target,request=next(iter(row['motion_requests'].items()))
    assert request['speed_m_s']==1.
    assert request['minimum_listener_horizontal_distance_m']==1.5
    assert row['source_classes'][target]==['speech_playback']
    assert row['explicit_whole_walking_speech']['footsteps_added']is False
    assert len(row['phase2_recorded_sound_ids'][target])==3
    if family in {'QA-07','QA-22'}:
        assert row['native_view_roles'][target]['required_visible_sometime']is True
        assert row['native_view_roles'][target]['static']is False
    else:
        relation=row['native_dynamic_occlusion_relations'][0]
        assert relation['target_asset_id']==target
        assert relation['occluder_asset_id']in row['silent_asset_ids']
        assert relation['minimum_stable_seconds']==.5
        assert relation['later_clear_time_s']>relation['query_times_s'][-1]


def test_entry_and_crossing_proposals_use_fixed_real_camera_geometry():
    forward=[0,0,-1]
    enter={'camera_path_condition':{'kind':'enter','camera_forward':forward,'required_entry_side':1,
        'initial_outside_minimum_abs_deg':60.,'final_inside_maximum_abs_deg':28.}}
    assert route.matches_camera_path([[3.2,0,-1.8],[0,0,-1.8]],[0,30],[0,1.5,0],enter)
    assert not route.matches_camera_path([[-3.2,0,-1.8],[0,0,-1.8]],[0,30],[0,1.5,0],enter)
    cross={'intervals_s':[[3,4.5],[6,7.5]],'camera_path_condition':{'kind':'cross_view','camera_forward':forward,
        'minimum_crossing_radius_m':4.2,'maximum_crossing_abs_deg':32.,'minimum_angular_span_deg':20.}}
    times=[0,3,4.5,6,7.5,30];points=[[-1.5,0,-5],[-1.5,0,-5],[0,0,-5],[0,0,-5],[1.5,0,-5],[1.5,0,-5]]
    assert route.matches_camera_path(points,times,[0,1.5,0],cross)
    assert not route.matches_camera_path(np.asarray(points)+[0,0,3],times,[0,1.5,0],cross)


def test_dynamic_mask_contract_requires_state_identity_and_no_unknown_blocker():
    relation={'required_state':'fully_occluded','minimum_occluded_fraction':1.}
    mask={'state':'fully_occluded','occlusion_fraction':1.}
    blocker={'occluder_instance_ids':['3'],'unknown_occluder_pixels':0}
    assert occlusion.matches_dynamic_observation(relation,mask,blocker,'3')
    assert not occlusion.matches_dynamic_observation(relation,mask,{**blocker,'unknown_occluder_pixels':1},'3')
    assert not occlusion.matches_dynamic_observation(relation,mask,{**blocker,'occluder_instance_ids':['3','4']},'3')
    assert not occlusion.matches_dynamic_observation(relation,{'state':'visible_occluded','occlusion_fraction':.99},blocker,'3')
