"""Whole registered programs stay physical while their layout is jointly solved."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from avengine.timeline.v7_registered_content_joint import content_role_offsets,content_row
from avengine.timeline.v7_family_constructors import choose_authored,ConstructionUnavailable

pytestmark=pytest.mark.fast_unit

def bank(count=3):
    sounds=[]
    for epoch in range(count):
        for i,duration in enumerate([4.,3.,2.]):
            sounds.append({'scenario_id':f'scenario{epoch}','role_id':f'role{i}',
                'sound_asset_id':f'sound{epoch}_{i}','transcript':'The evening meal is ready.',
                'voice_reference':{'gender':'female','speaker_id':f'fixed_voice{i}'},
                'synthesis_readback':{'duration_s':duration,'sample_count':int(duration*16000),'sample_rate_hz':16000}})
    return {'sounds':sounds}

def people():
    return [{'asset_id':name,'visual_gender':'female','color':colour,'kind':'human','body_identity_id':'bodyA'if i<2 else f'body{i}',
        'mouth_local_glb_m':[0.,1.5,0.],'capability':{'asset_kind':'standing_human'}}for i,(name,colour)in enumerate([('personA','blue'),('personB','yellow'),('personC','white'),('personD','green')])]

def bind_roles(monkeypatch,required='standing_human'):
    monkeypatch.setattr('avengine.timeline.v7_production_inputs.bind_semantic_registry',
        lambda manifest,assets:{'dialogues':{'scenarios':[{'id':f'scenario{i}','roles':[{'role_id':f'role{j}','required_asset_capability':required}for j in range(3)]}for i in range(3)]}})

def test_whole_window_selection_accepts_long_admitted_roles_without_changing_default(monkeypatch):
    bind_roles(monkeypatch);manifest=bank();actors=people()[:3]
    with pytest.raises(ConstructionUnavailable):choose_authored(manifest,actors,{},17,epoch_count=3)
    chosen=choose_authored(manifest,actors,{},17,epoch_count=3,whole_window_s=10.)
    assert len(chosen)==3 and {e[0]['scenario_id']for e in chosen}=={'scenario0','scenario1','scenario2'}
    assert len({tuple(s['voice_reference']['speaker_id']for s in e)for e in chosen})==1
    assert {s['sound_asset_id']for e in chosen for s in e}=={s['sound_asset_id']for s in manifest['sounds']}

def test_whole_window_selection_preserves_real_role_capabilities(monkeypatch):
    bind_roles(monkeypatch,'phone_user')
    with pytest.raises(ConstructionUnavailable):choose_authored(bank(),people()[:3],{},17,whole_window_s=10.)

def test_single_authored_scenario_remains_available_under_owner_partial_construction_rule(monkeypatch):
    bind_roles(monkeypatch)
    chosen=choose_authored(bank(1),people()[:3],{},17,epoch_count=3,whole_window_s=10.)
    assert len(chosen)==1 and chosen[0][0]['scenario_id']=='scenario0'

@pytest.mark.parametrize('durations',[[4.,3.,2.],[8.,6.,5.],[1.,1.,1.]])
def test_every_unchanged_clip_fits_all_three_native_epochs(durations):
    epochs=[]
    for j in range(3):
        epoch=deepcopy(bank()['sounds'][j*3:(j+1)*3])
        for sound,duration in zip(epoch,durations,strict=True):sound['synthesis_readback'].update(sample_count=int(duration*16000),duration_s=duration)
        epochs.append(epoch)
    offsets=content_role_offsets(epochs)
    for epoch in epochs:
        for offset,sound in zip(offsets,epoch,strict=True):assert 0<=offset and offset+sound['synthesis_readback']['sample_count']/16000<=10.


def test_content_row_has_four_members_same_real_sources_and_solved_fixed_camera(monkeypatch):
    from avengine.construction.joint import CameraPose,ScenePlan,BodySpec,static_actor
    actors=people();ids=[a['asset_id']for a in actors];roots=[[-2.,0.,-4.],[1.5,0.,-6.],[0.,0.,4.],[0.,0.,-8.]]
    camera=CameraPose((0.,1.25,0.),0.,width=1280,height=720)
    plan=ScenePlan(camera,{aid:static_actor(aid,p,BodySpec('human',.2,1.6))for aid,p in zip(ids,roots,strict=True)},metadata={'roles':{aid:('outside'if aid==ids[2]else'inside')for aid in ids}})
    polygon={'type':'Polygon','coordinates':[[[-10,-10],[10,-10],[10,10],[-10,10],[-10,-10]]]};context=SimpleNamespace(room={'floor_polygon':polygon})
    manifest=bank();epochs=[manifest['sounds'][i*3:(i+1)*3]for i in range(3)]
    captured=[];monkeypatch.setattr('avengine.timeline.v7_production_inputs.validate_recipe',lambda row,*args:captured.append(deepcopy(row)))
    row=content_row({'house':'new_house','room_id':'new_room','family':'hm3d'}, {},{},actors,manifest,{},context,plan,actors,epochs,scene_id='fresh_content',seed=6)
    assert row['members']==['v0_a0','v0_a1','v1_a0','v1_a1'] and row['window_start_ticks']==[0,50,100,150,200]
    assert row['production_scenario']['asset_ids']==ids and row['production_scenario']['speaker_count']==3
    assert row['silent_asset_ids']==[ids[3]] and row['native_view_roles'][ids[2]]['view']=='outside'
    assert row['joint_prepared_scene_required']is True and row['automatic_construction']['scene_template_files_read']==[]
    assert all(k['yaw_deg']==camera.yaw_deg for k in row['camera_yaw_knots'])
    assert row['measured_static_layout']['listener_m']==list(camera.position_m)
    assert row['production_scenario']['semantic_epoch_sound_ids']==[[s['sound_asset_id']for s in e]for e in epochs]
    assert row['native_minimum_endpoint_distance_gaps'][0]['minimum_difference_m']==.6 and captured
