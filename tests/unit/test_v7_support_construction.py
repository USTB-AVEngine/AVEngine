import pytest
from shapely.geometry import box
from avengine.capture.v7_support_construction import retry_same_surface,alternate_survey_origins
pytestmark=pytest.mark.fast_unit


def test_object_retries_another_native_site_without_changing_mount_or_identity():
    asset={'asset_id':'clock','kind':'object','attachment_surface':'table'};scene={}
    native={'point':[2.,.8,2.],'floor_y':0.,'normal':[0,1,0],'surface':'table'}
    supports={'_native_geometric_candidates':{'table':[native]}};poly=box(0,0,6,6)
    first=retry_same_surface(scene,asset,'table',supports,poly,0.,[1,0,1],lambda p:{})
    assert first['stage']=='same_registered_surface_native_geometry'
    assert supports['table']==[native]and asset['asset_id']=='clock'
    calls=[]
    def survey(p):calls.append(p);return {'table':[dict(native,point=[3,.8,3])],'wall':[dict(native,surface='wall')]}
    second=retry_same_surface(scene,asset,'table',supports,poly,0.,[1,0,1],survey)
    assert calls and second['survey_origin_m']!=[1,0,1]
    assert all(p['surface']=='table'for p in supports['table'])


def test_required_people_are_never_converted_to_object_support_fallback():
    scene={};assert retry_same_surface(scene,{'asset_id':'p','kind':'human'},'floor',{},box(0,0,5,5),0,[1,0,1],lambda p:{})is None
    assert scene=={}


def test_reconstruction_is_bounded_and_stays_on_same_room_level():
    scene={};asset={'asset_id':'a','kind':'object','attachment_surface':'wall'};calls=[]
    def survey(p):calls.append(p);return {'wall':[{'surface':'wall','floor_y':3.,'point':[2,4,2]}]}
    assert retry_same_surface(scene,asset,'wall',{},box(0,0,5,5),0,[1,0,1],survey)is None
    assert len(calls)<=4
    assert retry_same_surface(scene,asset,'wall',{},box(0,0,5,5),0,[1,0,1],survey)is None
    assert len(calls)<=4
