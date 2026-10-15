import numpy as np
import pytest
pytestmark=pytest.mark.fast_unit
from avengine.timeline.v7_export_scene_policy import camera_metrics,measured_actor_metrics,layout_metrics
from avengine.timeline.v7_english_export import validate_english_model_text


def test_wraparound_and_total_drift():
    frames=[{'position_m':[0,0,0],'yaw_deg':359.9},{'position_m':[0,0,0],'yaw_deg':.1}]
    assert camera_metrics(frames)['status']=='pass'
    frames[-1]['position_m']=[.011,0,0]
    assert camera_metrics(frames)['status']=='fail'
    frames[-1]['position_m']=[0,0,0];frames[-1]['yaw_deg']=.6
    assert camera_metrics(frames)['status']=='fail'


def actor(speed,species='human'):
    return {'species_id':species,'asset_id':'body','root_positions_m':[[speed*i/10,0,0]for i in range(101)]}


@pytest.mark.parametrize('speed,expected',[(.15,'fail'),(.8,'pass'),(1.4,'pass'),(1.41,'fail'),(0,'pass')])
def test_actual_root_speed_not_emitter_or_requested_speed(speed,expected):
    a=actor(speed);a['emitter_positions_m']=[[i,0,0]for i in range(101)]
    assert measured_actor_metrics({'a':a},10,{'body':{'speed_m_s':1}})[0]['status']==expected


def test_slide_has_its_own_range():
    assert measured_actor_metrics({'a':actor(.4,'dog')},10,{'body':{'intentional_slide':True}})[0]['status']=='pass'
    assert measured_actor_metrics({'a':actor(.4,'dog')},10)[0]['status']=='fail'


def test_spacing_and_heading_diversity():
    actors={str(i):{'species_id':'human','root_positions_m':[[x,0,z]]*2}for i,(x,z)in enumerate([(0,0),(1.2,0),(0,1.2)])}
    assert layout_metrics(actors,{'0':0,'1':90,'2':180})['status']=='pass'
    assert 'all_humans_share_one_heading'in layout_metrics(actors,{'0':0,'1':0,'2':0})['reasons']
    actors['1']['root_positions_m']=[[.99,0,0]]*2
    assert 'human_centres_under_1m'in layout_metrics(actors,{'0':0,'1':90,'2':180})['reasons']


@pytest.mark.parametrize('field',['question_en','choices','model_answer'])
def test_jargon_is_rejected_only_in_model_text(field):
    row={'question_en':'Who made the sound?','qa_id':'STARSS-SPEAKERS','protocol_panel':'STARSS'}
    validate_english_model_text(row)
    row[field]=[{'text':'admitted person'}]if field=='choices'else'Which STARSS class?'
    with pytest.raises(ValueError,match='internal terminology'):validate_english_model_text(row)


def test_invalidated_source_is_never_reused(tmp_path):
    from avengine.timeline.v7_export_scene_policy import inspect_parent_policy
    (tmp_path/'INVALID_OWNER.md').write_text('All old media are invalid.')
    pointer=tmp_path/'native/parent/seal.json';pointer.parent.mkdir(parents=True);pointer.write_text('{}')
    assert inspect_parent_policy(pointer)['reasons']==['source_output_explicitly_invalidated']


def test_complete_native_parent_and_cross_window_camera_gate(tmp_path):
    import json
    from avengine.timeline.v7_export_scene_policy import inspect_parent_policy
    root=tmp_path/'native/parents/new_parent';root.mkdir(parents=True)
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    assets=[{'asset_id':'body'+str(i),'kind':'human'}for i in range(3)]
    positions=[[0,0,-3],[2,0,-3],[-2,0,-4]];angles=[0,90,180]
    actors=[]
    for i,(position,angle)in enumerate(zip(positions,angles)):
        a=np.radians(angle);matrix=[[np.cos(a),0,np.sin(a),position[0]],[0,1,0,0],[-np.sin(a),0,np.cos(a),position[2]],[0,0,0,1]]
        actors.append({'actor_id':i,'asset_id':'body'+str(i),'position_m':position,'body_to_world_m':matrix})
    native={'frames':[{'time_s':i/10,'actors':actors}for i in range(301)],
        'camera_frame_readbacks':[{'time_s':i/10,'position_m':[0,1.3,0],'yaw_deg':0}for i in range(301)]}
    put(root/'visual/native_visible_visual_evidence_phase2.json',native)
    put(root/'visible_native_completed.json',{'attempt':'visual'})
    windows=[{'window_id':'scene__v0_a0__w'+str(i).zfill(2),'member':'v0_a0'}for i in range(5)]
    put(root/'selected_windows_v0_a0_completed.json',{'recipe':{'pool_directory':str(root/'pool')}})
    for w in windows:
        put(root/'pool'/w['window_id']/'candidate_audit.json',{'facts':{'actors':{},'time':{'frame_rate_hz':15},
            'listener':{'positions_m':[[0,1.3,0]]*150,'yaw_deg':[0]*150}}})
    seal=root/'integration_seal_completed.json'
    put(seal,{'recipe':{'input_recipe':{'row':{},'assets':assets}},'result':{'windows':windows}})
    assert inspect_parent_policy(seal)['status']=='pass'
    last=root/'pool'/windows[-1]['window_id']/'candidate_audit.json';data=json.loads(last.read_text());data['facts']['listener']['yaw_deg']=[1]*150;put(last,data)
    result=inspect_parent_policy(seal)
    assert result['status']=='fail'and 'v0_a0:listener_moved_across_windows'in result['reasons']
