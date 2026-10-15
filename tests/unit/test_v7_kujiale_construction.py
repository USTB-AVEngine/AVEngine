import copy
import numpy as np
import pytest
from avengine.timeline.v7_kujiale_construction import authored_pair,registered_people,select_group,construct_request
from avengine.timeline.v7_kujiale_scheduler import can_start,assign_gpu,isolated_command
pytestmark=pytest.mark.fast_unit

def test_whole_authored_pair_is_staggered_from_measured_duration():
    sounds=[{'scenario_id':'s','sound_asset_id':str(i),'role_id':i,'voice_reference':{'gender':'female','speaker_id':'v'+str(i)},'synthesis_readback':{'duration_s':d}}for i,d in enumerate([1.7,2.2,8.9])]
    scenario,ids,offsets=authored_pair({'sounds':sounds},'female',0,latest_end_s=5.5)
    assert ids==['0','1'] and offsets[1]>=offsets[0]+1.7+.299 and offsets[1]+2.2<=5.5
    with pytest.raises(ValueError,match='no complete'):authored_pair({'sounds':sounds},'male',0)

def test_formally_measured_different_appearances_keep_silent_bystander():
    pool=[{'asset_id':str(i),'body_identity_id':'f'if i<2 else'm','visual_gender':'female'if i<2 else'male','color':color,'dialogue_voice_binding_eligible':True,'registered_native_ue':{'status':'pass'}}for i,color in enumerate(['blue','green','yellow'])]
    ids,gender=registered_people(pool,{'assets':[{'asset_id':a['asset_id']}for a in pool]},3)
    assert len(set(ids))==3 and ids[:2]==['1','0'] and ids[2]=='2' and gender=='female'
    with pytest.raises(ValueError,match='two same-type'):registered_people(pool,{'assets':[{'asset_id':'0'}]},0)

def test_position_retry_uses_new_geometry_without_removing_people():
    candidates=[{'position_m':[x,1.3,z],'root_m':[x,0,z],'body_yaw_deg':0}for x in [-2,-1,0,1,2]for z in [-4,-3,-2,-1]]
    a=select_group(candidates,[0,1.5,0],0);b=select_group(candidates,[0,1.5,0],1)
    assert len(a)==len(b)==3 and a!=b
    assert {tuple(p['root_m'])for p in a[:2]}.isdisjoint(tuple(p['root_m'])for p in b[:2])
    assert all(min(np.linalg.norm((np.array(p['root_m'])-q['root_m'])[[0,2]])for q in group if q is not p)>=.85 for group in [a,b]for p in group)

def test_resource_pressure_queues_only_new_parents_and_gpu0_is_excluded(tmp_path):
    snapshot={'MemAvailable_bytes':40e9,'PSS_bytes':20e9,'data_free_bytes':80e9,'minimum_disk_free_bytes':60e9}
    assert can_start(snapshot,active_parents=7) and not can_start(snapshot,active_parents=8)
    snapshot['MemAvailable_bytes']=29e9;assert not can_start(snapshot,active_parents=0)
    assert assign_gpu([1,2,3],{1:2,2:4,3:0})==3
    with pytest.raises(ValueError,match='GPU0'):assign_gpu([0,2],{})
    command=isolated_command(tmp_path/'plan.json',5,3,'python',tmp_path/'worktree',tmp_path/'out')
    assert '/dev/nvidia3'in command and '/dev/nvidia0'not in command

def test_sampling_distance_contrast_is_not_a_room_veto_when_a_real_group_fits(monkeypatch):
    from avengine.capture import kujiale_pilot_environment
    def unexpected(*args):raise AssertionError('third-person test called without any possible distance contrast')
    monkeypatch.setattr(kujiale_pilot_environment,'comparison_person_candidates',unexpected)
    candidates=[{'position_m':[2*np.sin(a),1.3,-2*np.cos(a)],'root_m':[2*np.sin(a),0.,-2*np.cos(a)],'body_yaw_deg':0.}for a in np.linspace(-1.,1.,500)]
    result=select_group(candidates,[0,1.5,0])
    assert len(result)==3 and all(p['group_construction_preference_fallback']['native_admission_required']for p in result)
    assert min(np.linalg.norm((np.array(result[i]['root_m'])-result[j]['root_m'])[[0,2]])for i in range(3)for j in range(i))>=1.

def test_camera_can_occupy_clear_air_above_furniture_without_inventing_a_human_floor_point():
    from shapely.geometry import box
    from avengine.timeline.v7_kujiale_construction import air_camera_proposals
    eyes=air_camera_proposals(box(0,0,2,2),.03,.5)
    assert any(p[0]>.8 and p[2]>.8 for p in eyes)
    assert all(p[1]==1.53 for p in eyes)


def test_small_room_keeps_its_only_legal_pair_when_diversity_is_exhausted():
    candidates=[{'position_m':[x,1.3,z],'root_m':[x,0.,z],'body_yaw_deg':0.}for x,z in [(-.8,-2.),(.8,-2.),(0.,-1.2)]]
    first=select_group(candidates,[0.,1.5,0.],0)
    assert select_group(candidates,[0.,1.5,0.],2)==first
    assert len(first)==3


def test_low_registered_body_constructs_a_fixed_lower_native_camera(tmp_path):
    import json
    from shapely.geometry import box
    from avengine.timeline.v7_kujiale_construction import requested_camera_eye_height,air_camera_proposals
    inventory=tmp_path/'native.json';inventory.write_text(json.dumps({'assets':[{'ue_native_physical_status':'pass','attachment_surface':'floor','dimensions_m':[.5,.3,.7],'allowed_v7_classes_candidate':['dog_bark']}]}));inputs={'kujiale_ue_asset_admission':str(inventory)}
    height=requested_camera_eye_height({'required_sound_classes':['dog_bark']},inputs)
    assert height==1.0 and all(p[1]==1.03 for p in air_camera_proposals(box(0,0,2,2),.03,.5,height))
    assert requested_camera_eye_height({},inputs)==1.5
    assert requested_camera_eye_height({'required_sound_classes':['ceiling_sound']},inputs)==1.5


def test_native_retry_reaches_finer_grids_instead_of_repeating_the_first_coarse_solution():
    from avengine.timeline.v7_kujiale_construction import construction_search_steps
    assert construction_search_steps(0)[0]==.5
    assert construction_search_steps(3)[0]==.35
    assert construction_search_steps(6)[0]==.25
    assert construction_search_steps(12)==(.1,)
    assert construction_search_steps(120)==(.1,)


def test_camera_batches_eventually_visit_every_legal_candidate_without_permanent_subsampling():
    from avengine.timeline.v7_kujiale_construction import construction_camera_batch
    views=list(range(700))
    visited=set()
    for attempt in range(0,18,3):visited.update(construction_camera_batch(views,attempt))
    assert visited==set(views)
    assert construction_camera_batch(views,0)!=construction_camera_batch(views,3)
    assert construction_camera_batch([],99)==[]


def test_retry_budget_never_silently_drops_a_room_but_deadline_pauses_new_jobs():
    import datetime
    from tools.timeline.run_v7_kujiale_rooms import construction_retry_allowed
    now=datetime.datetime.now(datetime.timezone.utc);deadline=now+datetime.timedelta(minutes=1)
    assert construction_retry_allowed({'status':'failed','construction_attempt':120},now,deadline)
    assert not construction_retry_allowed({'status':'passed'},now,deadline)
    assert not construction_retry_allowed({'status':'failed'},deadline,deadline)


def test_camera_preference_fallback_never_goes_below_the_unchanged_1_2m_gate():
    candidates=[{'position_m':[x,1.3,z],'root_m':[x,0.,z]}for x,z in [(-.7,-1.1),(.7,-1.1),(0.,-2.4)]]
    group=select_group(candidates,[0.,1.5,0.])
    assert len(group)==3
    assert all(np.linalg.norm(np.asarray(p['root_m'])[[0,2]])>=1.2 for p in group)
    assert any(p['group_construction_preference_fallback']['camera_minimum_m']==1.2 for p in group)
