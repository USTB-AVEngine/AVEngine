"""The v13 phone merge changes runtime permits, never registry or STARSS IDs."""
import json
from avengine.timeline.v7_visible_sources import object_catalog, normalized_visible_classes


def test_phone_permits_are_one_active_class(tmp_path):
    glb=tmp_path/'phone.glb';glb.write_bytes(b'unchanged registered fixture')
    registry=tmp_path/'roster.json'
    source={'schema':'avengine_v7_static_object_roster_v1','assets':[{
        'asset_id':'registered_phone','status':'pass','derived_glb':str(glb),
        'object_type':'desk_telephone','attachment_surface':'table',
        'allowed_v7_classes_candidate':['telephone','telephone_bell_ringing','phone_ring'],
        'silent_decor':False,'label_color':'black','geometry_checks':{},
        'asset_metadata':{},'emitter':{'active':True,'local_m':[0,0,0],
        'physical_rationale':'original registered phone emitter'}}]}
    registry.write_text(json.dumps(source));original=registry.read_bytes()
    rows,excluded=object_catalog(registry)
    assert rows[0]['allowed_classes']==['telephone_bell_ringing']
    assert not excluded and registry.read_bytes()==original


def test_non_phone_permits_preserve_order():
    assert normalized_visible_classes(['speech_playback','telephone','laughter','telephone_bell_ringing'])==['speech_playback','telephone_bell_ringing','laughter']


def test_historical_taxonomy_is_retained():
    from avengine.qa.v7_protocol import CLASS13
    assert 'telephone' in CLASS13
