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


@pytest.mark.parametrize('kind',['human','animal'])
def test_real_native_floor_support_does_not_require_a_scan_semantic_label(kind):
    asset={'asset_id':'required_body','kind':kind,'attachment_surface':'floor'};scene={}
    native={'surface':'floor','floor_y':0.,'point':[2.,0.,2.],'normal':[0,1,0]}
    supports={'_native_geometric_candidates':{'floor':[native]}}
    result=retry_same_surface(scene,asset,'floor',supports,box(0,0,5,5),0,[1,0,1],lambda p:pytest.fail('floor body must reuse actual floor probes, not move to another mount'))
    assert result['actual_support_contacts_and_acoustics_required']
    assert supports['floor']==[native]and asset['asset_id']=='required_body'
    assert retry_same_surface(scene,asset,'floor',supports,box(0,0,5,5),0,[1,0,1],lambda p:{})is None
