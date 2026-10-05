from copy import deepcopy
from pathlib import Path
import hashlib,json
import pytest
from avengine.qa.v7_roster_intake import intake_roster,bind_physical_actors

def package(tmp_path):
    root=tmp_path/'roster';(root/'registrations').mkdir(parents=True)
    (root/'evidence').mkdir();(root/'reports/licenses').mkdir(parents=True)
    license_source=tmp_path/'original_MIT';license_source.write_text('MIT fixture permission and copyright')
    copy=root/'reports/licenses/mit.txt';copy.write_bytes(license_source.read_bytes())
    digest=hashlib.sha256(copy.read_bytes()).hexdigest()
    (root/'evidence/license_sources.json').write_text(json.dumps([{'name':'rocketbox_mit','source_path':str(license_source),'retained_copy':'reports/licenses/mit.txt','sha256':digest,'license':'MIT','status':'verified_local_exact_producer'}]))
    for name in ['static.glb','native_mp3d.png','native_hm3d.png','native_ue.png']:
        (root/name).write_bytes(b'unit artifact')
    variant={'variant_id':'red','major_id':'body','color':'red','sound_class_requested':'human_speech',
       'mouth_local_glb_m':[0,1.5,0],'mouth_local_ue_cm':[0,0,150],
       'dual_backend_static_usable':True,'habitat':{'status':'pass','glb':'static.glb',
       'native_evidence':['native_mp3d.png','native_hm3d.png']},
       'ue':{'status':'pass','logical_actor_count':1,'blueprint_readback_status':'pass','native_evidence':'native_ue.png'}}
    m={'schema':'avengine_v7_human_roster_v1','artifact_root':str(root),
       'summary':{'major_count':1,'minor_count':1},
       'majors':[{'major_id':'body','source':'Rocketbox Adults','gender':'female','minor_variants':[variant]}]}
    path=root/'registrations/human_roster.json';path.write_text(json.dumps(m))
    return root,path,m

def test_static_mit_qualification_never_becomes_full_episode(tmp_path):
    root,path,m=package(tmp_path);result=intake_roster(path)
    assert result['static_research_qualified_minors']==1
    assert result['full_episode_admission_count']==0
    actors=bind_physical_actors([{'actor_uid':'a','native_actor_id':1,'asset_id':'red'},
                                {'actor_uid':'b','native_actor_id':2,'asset_id':'red'}],result)
    assert len(actors)==2 and actors[0]['body_identity_id']==actors[1]['body_identity_id']
    with pytest.raises(ValueError,match='unique'):
        bind_physical_actors([{'actor_uid':'a','native_actor_id':1,'asset_id':'red'},
                            {'actor_uid':'b','native_actor_id':1,'asset_id':'red'}],result)

def test_modified_license_copy_and_artifact_escape_are_rejected(tmp_path):
    root,path,m=package(tmp_path)
    (root/'reports/licenses/mit.txt').write_text('wrong licence')
    result=intake_roster(path)
    assert not result['records'][0]['static_research_qualified']
    assert 'Rocketbox_original_copy_MIT_binding_failed' in result['records'][0]['static_blockers']
    m['majors'][0]['minor_variants'][0]['habitat']['glb']='../outside.glb'
    path.write_text(json.dumps(m));result=intake_roster(path)
    assert 'native_artifact_missing_or_outside_root' in result['records'][0]['static_blockers']

def test_lx_unknown_gender_and_missing_rights_are_preserved(tmp_path):
    root,path,m=package(tmp_path)
    m['majors'][0].update(source='lx',gender='not_annotated')
    m['majors'][0]['minor_variants'][0]['sound_class_requested']='megaphone_speech'
    path.write_text(json.dumps(m));r=intake_roster(path,authority='legacy_local_rights')['records'][0]
    assert 'visual_gender_not_annotated' in r['static_blockers']
    assert 'lx_ISNet_checkpoint_rights_and_source_commit_unverified' in r['static_blockers']
    assert r['non_mouth_emitter_status']=='not_provided'

def test_summary_counts_cannot_inflate_distinct_bodies(tmp_path):
    root,path,m=package(tmp_path)
    m['summary']['major_count']=2;path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='counts differ'):
        intake_roster(path)

def test_formal_registry_admits_lx_static_but_does_not_guess_voice_gender(tmp_path):
    root,path,m=package(tmp_path)
    m['majors'][0].update(source='lx',gender='not_annotated')
    m['majors'][0]['minor_variants'][0]['sound_class_requested']='violin'
    path.write_text(json.dumps(m))
    r=intake_roster(path)['records'][0]
    assert r['standing_registered_qualified']
    assert not r['dialogue_voice_binding_eligible']
    assert not r['native_episode_admitted']
    assert r['admission_authority']=='formal_registry'
    with pytest.raises(ValueError,match='standing pose'):
        bind_physical_actors([{'actor_uid':'a','native_actor_id':1,'asset_id':'red','pose':'walk'}],intake_roster(path))

def test_mouth_comes_from_registered_point_and_rigid_metre_transform(tmp_path):
    import numpy as np
    from avengine.qa.v7_roster_intake import standing_mouth_world
    root,path,m=package(tmp_path);r=intake_roster(path)['records'][0]
    matrix=np.eye(4);matrix[:3,3]=[2,0,-4]
    assert standing_mouth_world(r,matrix)==[2.,1.5,-4.]
    matrix[0,0]=100
    with pytest.raises(ValueError,match='rigid metre'):
        standing_mouth_world(r,matrix)
