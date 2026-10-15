from copy import deepcopy
import pytest
from avengine.timeline.v7_motion_quartet_planning import plan_room_motion_quartet,validate_template_request
pytestmark=pytest.mark.fast_unit


def fixture():
    assets=[{'asset_id':'walker','kind':'human','species_id':'human'},{'asset_id':'still','kind':'human','species_id':'human'}]
    row={'scene_id':'new','scene_seed':7,'asset_ids':['walker','still'],'source_classes':{'walker':['speech_playback'],'still':['speech_playback']},
        'motion_requests':{'walker':{'action':'walk','speed_m_s':1.0}},'motion_during_condition_requests':[
            {'qa_id':f,'target_asset_id':'walker','minimum_distance_change_m':1.5,'minimum_motion_s':2.0}for f in ['QA-06','QA-15','QA-16','QA-17']]}
    return row,assets


def test_room_constructor_gets_only_room_registries_seed_and_actual_deficits():
    row,assets=fixture();seen={}
    def constructor(base,room,catalog,semantic,records,pathfinder,*,attempt):
        seen.update(room=room,attempt=attempt,seed=base['scene_seed']);return deepcopy(row)
    state={'logical_questions':14,'family_counts':{'QA-06':14},'minimum_motion_independent_questions':30}
    result=plan_room_motion_quartet({'scene_seed':7},state,{'room_label':'unseen'},assets,{},[],object(),attempt=2,constructor=constructor)
    assert seen=={'room':{'room_label':'unseen'},'attempt':2,'seed':7}
    assert result['scene_demand']['independent_motion_deficits']=={'QA-06':16,'QA-15':30,'QA-16':30,'QA-17':30}
    assert result['scene_constructor']=='motion_quartet_template'


def test_template_cannot_silently_lower_the_real_human_speed_or_distance():
    row,assets=fixture();row['motion_requests']['walker']['speed_m_s']=.4
    with pytest.raises(ValueError,match='outside owner range'):validate_template_request(row,assets)
    row['motion_requests']['walker']['speed_m_s']=1.;row['motion_during_condition_requests'][1]['minimum_distance_change_m']=.4
    with pytest.raises(ValueError,match='0.5m'):validate_template_request(row,assets)


def test_post_sound_motion_speech_cannot_be_replaced_with_footsteps():
    row,assets=fixture();row['source_classes']['walker']=['footsteps']
    with pytest.raises(ValueError,match='without footsteps'):validate_template_request(row,assets)


def test_native_room_adapter_selects_the_explicit_installed_runtime(monkeypatch):
    from types import SimpleNamespace
    from avengine.timeline import v7_motion_quartet_planning as planning
    from avengine.acoustics import runtime
    calls=[]
    pf=SimpleNamespace(load_nav_mesh=lambda path:path=='registered.navmesh')
    def load(**kwargs):
        calls.append(kwargs);return SimpleNamespace(PathFinder=lambda:pf),None
    monkeypatch.setattr(runtime,'load_habitat_runtime',load)
    monkeypatch.setattr(planning,'plan_room_motion_quartet',lambda *args,**kwargs:'constructed')
    result=planning.plan_native_room_motion_quartet({'room_id':'R'},{},{'checked_rooms':[{'room_label':'R'}],'input':{'navmesh_source':'registered.navmesh'}},[],{},[],{'runtime':{'runtime_prefix':'registered-runtime'}})
    assert result=='constructed'
    assert calls==[{'runtime_mode':'current-installed','runtime_prefix':'registered-runtime'}]
