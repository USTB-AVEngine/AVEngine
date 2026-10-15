import copy
import pytest
from avengine.timeline.v7_registered_sampling import build_registered_scene,REGISTERED_INPUT_SCHEMA
pytestmark=pytest.mark.fast_unit


def test_lower_constructed_camera_does_not_change_frozen_room_input(monkeypatch):
    from avengine.timeline import v7_registered_sampling as sampling
    record={'sound_asset_id':'speech','dataset':'VCTK-Corpus 0.92','legacy_library_split':'train','gender':'F','duration_s':1.,'speaker_id':'voice','header':{'sample_count':16000}}
    monkeypatch.setattr(sampling,'filter_admission',lambda records:([record],[]))
    polygon={'type':'Polygon','coordinates':[[[-5,-5],[5,-5],[5,5],[-5,5],[-5,-5]]]}
    house={'house':'new_house','split':'train','acoustic_status':'pass','checked_room_ids':['R0'],'checked_rooms':[{'room_label':'R0','placement':{'camera_m':[2.375,1.51,1.]}}]}
    row={'schema':REGISTERED_INPUT_SCHEMA,'house':'new_house','split':'train','scene_id':'fresh','scene_seed':23,'room_id':'R0','seeds':{'audio':5},'window_start_ticks':[0,50,100,150,200],'window_count':5,'production_scenario':{'speaker_count':1,'camera_yaw_deg':0.,'environment_sound_classes':[],'non_speech_class':None},'measured_static_layout':{'listener_m':[2.125,1.,1.],'room_floor_polygon':polygon,'valid':[{'position_m':[x,1.3,z],'root_m':[x,0.,z],'body_yaw_deg':y}for x,z,y in [(-1.,-3.,30.),(1.,-3.,90.),(0.,-1.6,180.)]]}}
    assets=[{'asset_id':f'p{i}','endpoint_id':f'speaker{i}','visual_gender':'female'}for i in range(3)];before=copy.deepcopy(house);request=copy.deepcopy(row)
    first=build_registered_scene(row,house,[],assets);second=build_registered_scene(row,house,[],assets)
    assert house==before and row==request
    assert first==second
    assert first[0]['room']['placement']['camera_m']==[2.125,1.,1.]
    assert first[0]['checked_rooms'][0]['placement']['camera_m']==[2.375,1.51,1.]
