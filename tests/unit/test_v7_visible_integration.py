from copy import deepcopy
import numpy as np
import pytest
from avengine.timeline.v7_visible_sources import validate_scene_bindings,audit_native_bindings,point
from avengine.timeline.v7_s_batch_checks import expected_sound_classes
from avengine.capture.v7_visible_native import attachment_template
from avengine.capture.v7_visible_semantic_support import category_kind
from avengine.qa.v7_visible_truth import validate_additional_items


def binding():
    asset={'asset_id':'dog','endpoint_id':'voice1','native_instance_id':1,'allowed_classes':['dog_bark'],'silent_decor':False,'emitter_local_m':[0,1,0]}
    scene={'sources':{'voice1':{'asset_id':'dog','actor_id':1,'native_visual_kind':'registered_drawable','sound_class':'dog_bark'}},'events':[{'endpoint_id':'voice1','sound_class':'dog_bark'}]}
    return scene,[asset]


def test_registered_entity_is_required():
    scene,assets=binding();assert validate_scene_bindings(scene,assets)['invisible_sources']==0
    scene['sources']['hidden']={'native_visual_kind':'audio_only'}
    with pytest.raises(ValueError,match='registered drawable'):validate_scene_bindings(scene,assets)


@pytest.mark.parametrize('change',[{'sound_class':'footsteps'},{'sound_class':'cat_meow'}])
def test_no_phase_two_or_substitute_class(change):
    scene,assets=binding();scene['events'][0].update(change)
    with pytest.raises(ValueError):validate_scene_bindings(scene,assets)


def test_silent_ac_cannot_emit():
    scene,assets=binding();assets[0]['silent_decor']=True
    with pytest.raises(ValueError,match='permitted'):validate_scene_bindings(scene,assets)


def test_actor_identity_cannot_be_relabelled():
    scene,assets=binding();scene['sources']['voice1']['actor_id']=2
    with pytest.raises(ValueError,match='identity'):validate_scene_bindings(scene,assets)


@pytest.mark.parametrize('delta,passed',[(.00009,True),(.00011,False)])
def test_emitter_readback_tolerance(delta,passed):
    scene,assets=binding();row={'endpoint_id':'voice1','asset_id':'dog','actor_id':1,'body_to_world_m':np.eye(4).tolist(),'emitter_position_m':[delta,1,0],'visible_pixels':100,'native_object_id':7}
    result=audit_native_bindings(scene,assets,{'frames':[{'actors':[row]}]});assert (result['status']=='pass')==passed


def test_native_drawable_and_visible_pixels_cannot_be_omitted():
    scene,assets=binding();row={'endpoint_id':'voice1','asset_id':'dog','actor_id':1,'body_to_world_m':np.eye(4).tolist(),'emitter_position_m':[0,1,0],'visible_pixels':0,'native_object_id':None}
    result=audit_native_bindings(scene,assets,{'frames':[{'actors':[row]}]});assert result['status']=='fail';assert len(result['errors'])==2


@pytest.mark.parametrize('bad',[[0,0],[0,float('nan'),0],[0,float('inf'),0]])
def test_invalid_emitter_coordinates(bad):
    with pytest.raises(ValueError):point(bad)


@pytest.mark.parametrize('surface,axis,coordinate',[('floor',1,0),('table',1,0),('wall',0,-1),('ceiling',1,2)])
def test_attachment_plane_comes_from_original_geometry(surface,axis,coordinate):
    vertices=np.array([[x,y,z] for x in [-1,1] for y in [0,2] for z in [-.5,.5]],float)
    anchor,probes,proof=attachment_template(vertices,surface);assert anchor[axis]==coordinate;assert all(p[axis]==coordinate for p in probes);assert proof['axis']==axis


@pytest.mark.parametrize('n',[26,27,34])
def test_class_domain_is_dynamic(n):
    table={f'class{i}':1 for i in range(n)};assert expected_sound_classes({'actual_sound_class_emission_counts':table})==set(table)


def test_class_domain_retains_declared_missing_classes():
    assert expected_sound_classes({'admitted_sound_class_domain':['a','b'],'actual_sound_class_emission_counts':{'a':1}})=={'a','b'}
    assert expected_sound_classes({'admitted_sound_class_domain':[],'actual_sound_class_emission_counts':{'a':1}})==set()


@pytest.mark.parametrize('name,kind',[('coffee table','table'),('desk','table'),('bed',None),('sofa',None),('ceiling lamp',None),('ceiling','ceiling')])
def test_semantic_support_rejects_beds_and_lamps(name,kind):assert category_kind(name)==kind


def test_independent_truth_rejects_corrupted_class_answer():
    q={'question_id':'q','qa_id':'V7-CLASS34','protocol_panel':'v7_admitted_recording_classes','form':'open'}
    g={'question_id':'q','canonical_answer':'cat_meow','answer':'cat_meow','evidence':{'activity':{'endpoint':'v1','start_tick':2,'end_tick':4}}}
    proof={'protocol_scene':{'events':[]},'window_start_s':0.,'events':{'e':{'sound_class':'dog_bark','native_endpoint_id':'v1','activity_intervals_samples_global':[[3200,6400]]}}}
    r=validate_additional_items([q],[g],proof);assert r['checked_forms']==1;assert r['passed_forms']==0;assert r['mismatches'][0]['expected_canonical']=='dog_bark'


def test_independent_class_truth_rejects_ambiguous_onset():
    q={'question_id':'q','qa_id':'V7-CLASS34','protocol_panel':'v7_admitted_recording_classes','form':'open'}
    g={'question_id':'q','canonical_answer':'dog_bark','answer':'dog_bark','evidence':{'activity':{'endpoint':'v1','start_tick':2,'end_tick':4}}}
    event={'sound_class':'dog_bark','native_endpoint_id':'v1','activity_intervals_samples_global':[[3200,6400]]}
    proof={'protocol_scene':{'events':[]},'window_start_s':0.,'events':{'e':event,'f':{**event,'native_endpoint_id':'v2'}}}
    with pytest.raises(ValueError,match='ambiguous'):validate_additional_items([q],[g],proof)


def correction_glb(tmp_path,cycle=False):
    import json,struct,trimesh
    from avengine.assets.source_materials import read_glb
    raw=tmp_path/'original.glb';raw.write_bytes(trimesh.creation.box().export(file_type='glb'))
    document,blob=read_glb(raw);roots=document['scenes'][0]['nodes'];i=len(document['nodes'])
    document['nodes'].append({'name':'world','translation':[1,2,3],'scale':[.2,.3,.4],'children':roots if not cycle else [i]});document['scenes'][0]['nodes']=[i]
    payload=json.dumps(document,separators=(',',':')).encode();payload+=b' '*((-len(payload))%4);blob+=b'\0'*((-len(blob))%4)
    raw.write_bytes(struct.pack('<4sII',b'glTF',2,28+len(payload)+len(blob))+struct.pack('<I4s',len(payload),b'JSON')+payload+struct.pack('<I4s',len(blob),b'BIN\0')+blob);return raw


def test_original_index_graph_preserves_duplicate_world_correction(tmp_path):
    from avengine.capture.v7_visible_mesh import indexed_static_mesh
    mesh=indexed_static_mesh(correction_glb(tmp_path))
    assert np.allclose(np.ptp(mesh.vertices,axis=0),[.2,.3,.4])
    assert np.allclose(mesh.bounds.mean(0),[1,2,3])
    assert len(mesh.faces)==12


def test_original_index_graph_rejects_cycles(tmp_path):
    from avengine.capture.v7_visible_mesh import indexed_static_mesh
    with pytest.raises(ValueError,match='cycle'):indexed_static_mesh(correction_glb(tmp_path,cycle=True))


def test_thin_wall_control_stays_partly_in_the_wall():
    from avengine.capture.v7_visible_native import contact_control_policy
    vertices=np.array([[0,0,0],[.012,.1,.1]])
    p=contact_control_policy(vertices,'wall','object')
    assert p['burial_m']==.006
    assert p['negative_distance_threshold_m']==-.00101
    assert contact_control_policy(vertices,'floor','human')['burial_m']==.1


def test_both_human_foot_regions_are_measured():
    vertices=np.array([[-.1,0,0],[-.1,.001,.1],[.1,.02,0],[.1,.021,.1]])
    _,probes,_=attachment_template(vertices,'floor','human')
    assert any(np.allclose(p,[.1,.02,0]) for p in probes)
    assert any(np.allclose(p,[-.1,0,0]) for p in probes)


def test_english_guard_refuses_pre_audit_admission():
    from avengine.timeline.v7_visible_sources import validate_english_source_admission
    with pytest.raises(ValueError,match='v9 English'):
        validate_english_source_admission({'records':[]})


def test_english_guard_refuses_restored_non_english_clip():
    from avengine.timeline.v7_visible_sources import validate_english_source_admission
    source={'scope_change_5':{'evidence':'language_audit.json','removed_sound_asset_ids':['removed'],'records_after':1},'records':[{'sound_asset_id':'removed'}]}
    with pytest.raises(ValueError,match='non-English recording restored'):validate_english_source_admission(source)


def test_english_guard_checks_actual_admission_count():
    from avengine.timeline.v7_visible_sources import validate_english_source_admission
    source={'scope_change_5':{'evidence':'language_audit.json','removed_sound_asset_ids':['removed'],'records_after':1},'records':[{'sound_asset_id':'kept'}]}
    assert validate_english_source_admission(source)['admitted_record_count']==1
    source['scope_change_5']['records_after']=2
    with pytest.raises(ValueError,match='record count'):validate_english_source_admission(source)


@pytest.mark.parametrize('gender,visual,expected',[('F','female',0),('M','male',1),('M','unknown',1)])
def test_speech_taxonomy_comes_from_actual_voice_metadata(gender,visual,expected):
    from avengine.timeline.v7_visible_sources import speech_class_id
    assert speech_class_id({'gender':gender},visual)==expected


def test_speech_rejects_known_body_voice_gender_disagreement():
    from avengine.timeline.v7_visible_sources import speech_class_id
    with pytest.raises(ValueError,match='visual gender'):speech_class_id({'gender':'M'},'female')



def test_object_body_pixels_mismatch_cannot_be_reviewed():
    from avengine.assets.v7_visible_color import rebuild_appearance_review
    a={'asset_id':'microwave','native_instance_id':1,'kind':'object','label_color':'black','registered_label_color':'black','name_zh':'微波炉'}
    checks=[{'asset_id':'microwave','frame_index':0,'matched':True,'body_pixels':100},
            {'asset_id':'microwave','frame_index':60,'matched':False,'body_pixels':718}]
    review,report=rebuild_appearance_review(checks,[a]);assert not review['actors']
    assert report['excluded'][0]['measurable_mismatch_frames']==[60]
    assert report['excluded'][0]['asset_label_changed'] is False


def test_animal_insufficient_pixels_are_not_a_measured_mismatch():
    from avengine.assets.v7_visible_color import rebuild_appearance_review
    a={'asset_id':'cat','native_instance_id':1,'kind':'animal','label_color':'black','registered_label_color':'black','name_zh':'猫'}
    checks=[{'asset_id':'cat','frame_index':0,'matched':True,'roi_pixels':220},
            {'asset_id':'cat','frame_index':60,'matched':False,'roi_pixels':150}]
    review,report=rebuild_appearance_review(checks,[a]);assert review['actors']['1']['field']=='coat_color'
    assert not report['excluded']



def test_registered_animal_feet_override_density_split():
    import numpy as np
    from avengine.capture.v7_visible_native import attachment_template
    feet=np.array([[-1,0,-1],[-1,0,1],[1,0,-1],[1,0,1]],float)
    verts=np.vstack([feet,np.tile([3,1,3],(100,1))])
    landmarks={f'foot_{i}':p.tolist() for i,p in enumerate(feet)}
    _,probes,_=attachment_template(verts,'floor','animal',landmarks)
    assert np.array_equal(np.array(probes[-4:]),feet)


def test_registered_animal_foot_must_be_on_original_mesh():
    import numpy as np,pytest
    from avengine.capture.v7_visible_native import attachment_template
    verts=np.array([[-1,0,-1],[-1,0,1],[1,0,-1],[1,0,1]],float)
    landmarks={f'foot_{i}':p.tolist() for i,p in enumerate(verts)}
    landmarks['foot_3'][1]=.02
    with pytest.raises(ValueError,match='not near the original mesh'):
        attachment_template(verts,'floor','animal',landmarks)



def test_scan_category_rejects_tabletop_devices_and_accepts_original_counter():
    from avengine.capture.v7_visible_semantic_support import category_kind
    assert category_kind('table#lamp','lighting') is None
    assert category_kind('tabletop#object','objects') is None
    assert category_kind('bedside#table','chest_of_drawers')=='table'
    assert category_kind('kitchen#island','counter')=='table'
    assert category_kind('kitchen#cabinet','cabinet') is None



def test_door_on_another_floor_is_not_nearby():
    from avengine.capture.v7_visible_attachment import anchor_distance
    upstairs={'center_m':[0,4,0],'sizes_m':[1,2,.2]}
    assert anchor_distance([0,1.45,0],upstairs)>1.2
    downstairs={'center_m':[0,1,0],'sizes_m':[1,2,.2]}
    assert anchor_distance([.7,1.45,0],downstairs)<1.2

@pytest.mark.parametrize('label',['carpet','rug','doormat','floor#mat'])
def test_ground_coverings_are_floor_semantics(label):
    from avengine.capture.v7_visible_semantic_support import category_kind
    assert category_kind(label)=='floor'


def test_bed_and_furniture_are_never_ground_covering_aliases():
    from avengine.capture.v7_visible_semantic_support import category_kind
    for label in ['bed','chair','sofa','coffee table','cabinet','table lamp']:
        assert category_kind(label)!='floor'


def test_whole_semantic_role_timings_fit_both_windows():
    from avengine.timeline.v7_visible_sources import semantic_schedule
    assert semantic_schedule([0.1,4.0,2.2],2,3,7.8)==[2.2,17.2]


@pytest.mark.parametrize('offsets,role_count,duration', [([0.1],3,2), ([-0.1],1,2), ([8.1],1,2), ([0.15],1,2)])
def test_semantic_timing_refuses_missing_or_clipped_roles(offsets,role_count,duration):
    from avengine.timeline.v7_visible_sources import semantic_schedule
    with pytest.raises(ValueError):semantic_schedule(offsets,0,role_count,duration)


def test_stationary_native_reuse_rejects_camera_or_asset_changes():
    from copy import deepcopy
    from avengine.capture.v7_visible_native_reuse import validate_physical_reuse
    original={'house':{},'inputs':{},'assets':[{'allowed_pose_for_this_task':'stand'}],'row':{'scene_id':'old','listener_m':[0,1,0]}}
    new=deepcopy(original);new['row'].update(scene_id='new',semantic_start_offsets_s=[0.1])
    validate_physical_reuse(new,original)
    new['row']['listener_m']=[0,1,1]
    with pytest.raises(ValueError,match='camera'):validate_physical_reuse(new,original)
    new=deepcopy(original);new['assets'][0]['pilot_walk_allowed']=True
    with pytest.raises(ValueError,match='physical input'):validate_physical_reuse(new,original)


def test_paw_centroid_projection_does_not_replace_original_sole():
    import numpy as np
    from avengine.capture.v7_visible_native import attachment_template
    soles=np.array([[-1,0,-1],[-1,0,1],[1,0,-1],[1,0,1]],float)
    upper=soles+[0,.010,0];vertices=np.vstack([soles,upper])
    landmarks={f'foot_{i}':p.tolist() for i,p in enumerate(upper)}
    _,probes,definition=attachment_template(vertices,'floor','animal',landmarks)
    assert np.array_equal(np.array(probes[-4:]),soles)
    assert all(p['paw_region_radius_m']<=.05 for p in definition['registered_foot_original_surface_projection'])


def test_actually_raised_paw_sole_is_still_measured_as_raised():
    import numpy as np
    from avengine.capture.v7_visible_native import attachment_template
    soles=np.array([[-1,0,-1],[-1,0,1],[1,0,-1],[1,.025,1]],float)
    upper=soles+[0,.010,0];vertices=np.vstack([soles,upper])
    landmarks={f'foot_{i}':p.tolist() for i,p in enumerate(upper)}
    _,probes,_=attachment_template(vertices,'floor','animal',landmarks)
    assert np.array(probes)[-1,1]>.015
