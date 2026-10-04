from copy import deepcopy
import math
import pytest
from avengine.qa.v7_protocol import generate_questions
from avengine.qa.v7_panel_coverage import (
 canonical_speech_scene, speaker_state_sidecar, boundary_projection,
 panel_coverage, select_balanced_panels, validate_counterfactual_pair)

pytestmark = pytest.mark.fast_unit

def native_scene():
 track=[{'time_s':i/15,'position_m':[0.,1.5,-3.]} for i in range(151)]
 return {'listener_m':[0.,1.5,0.], 'camera_forward':[0.,0.,-1.],
         'sources':{'mouth':{'actor_id':7,'mechanism':'human_live_formant','trajectory':track}},
         'events':[], 'actor_speech_schedule':{
          'schema':'v7_actor_speech_schedule_v1','activity_complete':True,
          'actors':[{'actor_uid':'physical-person','native_actor_id':7,'native_endpoint_id':'mouth',
                     'video_track_id':'native-mask-1001','native_readback':'retained-native-readback.json'}],
          'event_bindings':[]}}

def add_speech(s,eid,lo,hi,slot=1,ep='mouth',asset='voice_a'):
 s['events'].append({'event_id':eid,'class_id':1,'source_id':slot,'endpoint_id':ep,'start_tick':lo,'end_tick':hi})
 s['actor_speech_schedule']['event_bindings'].append({
  'event_id':eid,'class_id':1,'source_slot_id':slot,'endpoint_id':ep,'actor_uid':'physical-person',
  'native_actor_id':7,'emission_mechanism':'human_live_formant','activity_observed':True,
  'activity_receipt':'observed-dry-intervals.json','sound_asset_id':asset})

def answer(s):
 p,g,a=generate_questions(s,0.,'opaque',{})
 byid={r['question_id']:r for r in g}
 return [byid[q['question_id']]['canonical_answer'] for q in p
         if q['qa_id']=='STARSS-SPEAKERS' and q['form']=='open']

def test_slots_voice_assets_repeated_motion_and_silent_body_are_not_identity():
 s=native_scene();s['sources']['alias']=deepcopy(s['sources']['mouth'])
 add_speech(s,'first',0,10,slot=11,asset='voice_a')
 add_speech(s,'second',30,40,slot=12,ep='alias',asset='voice_b')
 add_speech(s,'again',80,90,slot=13,asset='voice_a')
 # Actor moves between observed native samples, then speaks again.
 for source in s['sources'].values():
  for i,t in enumerate(source['trajectory']):t['position_m'][0]=i/150
 assert answer(s)==['1']
 sidecar=speaker_state_sidecar(s)
 assert sidecar['frames_10hz'][10]['actors']==[{'native_actor_id':7,'is_speaking':False}]
 assert sidecar['frames_10hz'][30]['actors']==[{'native_actor_id':7,'is_speaking':True}]
 canonical,_=canonical_speech_scene(s)
 assert {e['source_id'] for e in canonical['events']}=={7}
 assert {e['endpoint_id'] for e in canonical['events']}=={'mouth'}

def test_zero_includes_silent_visible_registry_and_environment_not_playback_people():
 s=native_scene()
 s['sources']['cabinet']={'actor_id':99,'mechanism':'loudspeaker_playback','class_id':8,
                         'trajectory':deepcopy(s['sources']['mouth']['trajectory'])}
 s['events']=[{'event_id':'music','source_id':99,'endpoint_id':'cabinet','class_id':8,
               'start_tick':20,'end_tick':60,'mechanism':'loudspeaker_playback'}]
 assert answer(s)==['0']
 assert speaker_state_sidecar(s)['frames_10hz'][30]['actors']==[{'native_actor_id':7,'is_speaking':False}]

@pytest.mark.parametrize('bad',['incomplete','unknown_actor','slot_conflict','duplicate_binding','playback','different_mouth'])
def test_native_identity_failures_reject_instead_of_silently_counting(bad):
 s=native_scene();add_speech(s,'event',0,10)
 binding=s['actor_speech_schedule']['event_bindings'][0]
 if bad=='incomplete':s['actor_speech_schedule']['activity_complete']=False
 elif bad=='unknown_actor':binding['actor_uid']='unknown'
 elif bad=='slot_conflict':binding['source_slot_id']=999
 elif bad=='duplicate_binding':s['actor_speech_schedule']['event_bindings'].append(deepcopy(binding))
 elif bad=='playback':binding['emission_mechanism']='loudspeaker_playback'
 else:
  s['sources']['alias']=deepcopy(s['sources']['mouth']);s['sources']['alias']['trajectory'][0]['position_m'][0]=3
  s['events'][0]['endpoint_id']='alias';binding['endpoint_id']='alias'
 with pytest.raises(ValueError):generate_questions(s,0.,'opaque',{})

def test_legacy_render_slot_is_not_counted_as_actor():
 s=native_scene();s.pop('actor_speech_schedule')
 s['events']=[{'class_id':1,'source_id':104,'endpoint_id':'mouth','start_tick':0,'end_tick':10}]
 assert answer(s)==[]

@pytest.mark.parametrize('u,v,label',[(0,360,'in_frame'),(-.25,360,'out_of_frame'),
 (.25,360,'in_frame'),(1279.75,360,'in_frame'),(1280,360,'out_of_frame'),
 (1280.25,360,'out_of_frame'),(640,0,'in_frame'),(640,-.25,'out_of_frame'),
 (640,719.75,'in_frame'),(640,720,'out_of_frame')])
def test_native_pixel_centers_convert_to_half_open_boundaries(u,v,label):
 s=native_scene();fx=1280/(2*math.tan(math.radians(42.5)))
 delta=[(u-640)/fx,(360-v)/fx,-1.]
 p=boundary_projection(s,delta)
 assert p['pixel_boundary_xy']==pytest.approx([u,v],abs=1e-9)
 assert p['frustum']==label
 assert p['frustum_dead_zone_px']==0
 s['camera_calibration']={'fx':fx,'fy':fx,'cx':640,'cy':360,'coordinate_convention':'pixel_boundaries'}
 assert boundary_projection(s,delta)['pixel_boundary_xy']==pytest.approx([u,v],abs=1e-9)

def test_vertical_fov_behind_camera_and_real_native_basis():
 s=native_scene()
 assert boundary_projection(s,[0,0,3])['frustum']=='out_of_frame'
 s['camera_forward']=[0,1,0];s['camera_right']=[1,0,0];s['camera_up']=[0,0,1]
 assert boundary_projection(s,[0,3,0])['pixel_boundary_xy']==[640,360]

def rows():
 return [{'qa_id':'STARSS-SPEAKERS','logical_question_id':f'{world}_{i}',
          'canonical_answer':'3','world_id':world}
         for world in ['world_a','world_b'] for i in range(3)]

def test_world_round_robin_and_forms_do_not_inflate_independent_coverage():
 r=rows()
 coverage=panel_coverage(r+r)
 n=coverage['speakers_exact'][3]
 assert n['logical_questions']==6 and n['independent_worlds']==2
 selection=select_balanced_panels(r,2,1)
 assert {x['world_id'] for x in selection['selected']}=={'world_a','world_b'}
 assert next(x for x in selection['deficits'] if x['stratum']==['speakers',0])['missing']==2
 assert panel_coverage([dict(x,world_id='same_world') for x in r])['speakers_exact'][3]['independent_worlds']==1

def pair():
 common=dict(pair_id='pair',world_id='world',split='train',source_actor_uid='person',
   dry_audio_identity='actual-dry',activity_identity='actual-activity',question_stem_id='same-wording',
   intervention='camera_orientation_only',native_media_receipt='actual-video-evidence',
   native_projection_receipt='actual-native-camera',source_trajectory_identity='same-native-mouth',
   listener_position_identity='same-native-listener',world_foa_identity='same-world-wave')
 return [dict(common,logical_question_id='a',canonical_answer='in_frame',camera_orientation=[0,0,-1]),
         dict(common,logical_question_id='b',canonical_answer='out_of_frame',camera_orientation=[0,0,1])]

def test_counterfactual_atomicity_same_source_native_receipts_and_actual_flip():
 p=pair();assert validate_counterfactual_pair(p)['status']=='pass'
 for field in ['source_actor_uid','activity_identity','source_trajectory_identity','listener_position_identity','world_foa_identity']:
  wrong=deepcopy(p);wrong[1][field]='changed'
  with pytest.raises(ValueError):validate_counterfactual_pair(wrong)
 with pytest.raises(ValueError):validate_counterfactual_pair(p[:1])
 wrong=deepcopy(p);wrong[1]['canonical_answer']='in_frame'
 with pytest.raises(ValueError):validate_counterfactual_pair(wrong)
 wrong=deepcopy(p);wrong[1]['native_projection_receipt']=None
 with pytest.raises(ValueError):validate_counterfactual_pair(wrong)

def test_source_position_pair_requires_recomputed_rir():
 p=pair()
 for i,m in enumerate(p):m.update(intervention='source_position_only',camera_orientation=[0,0,-1],
       source_trajectory_identity='position_'+str(i),rir_recomputed_receipt='actual-new-rir')
 assert validate_counterfactual_pair(p)['status']=='pass'
 p[1].pop('rir_recomputed_receipt')
 with pytest.raises(ValueError):validate_counterfactual_pair(p)

def test_actor_and_pair_supervision_never_enters_public_prompt_or_choices():
 s=native_scene();add_speech(s,'first',0,20)
 p,g,_=generate_questions(s,0.,'opaque',{})
 forbidden={'actor_uid','native_actor_id','video_track_id','event_id','source_slot_id','actor_speech_schedule',
            'pair_id','native_projection','identity_proof','is_speaking','evidence','canonical_answer'}
 def visit(x):
  if isinstance(x,dict):
   assert not forbidden.intersection(x)
   for v in x.values():visit(v)
  elif isinstance(x,list):
   for v in x:visit(v)
 visit(p)
 q,h,_=generate_questions(s,0.,'opaque',{},wording_variant='alternate')
 assert g==h
 assert [(x['answer_domain'],x.get('choices')) for x in p]==[(x['answer_domain'],x.get('choices')) for x in q]


def test_balanced_selector_never_splits_a_native_counterfactual_pair():
    members=pair()
    for m in members:m.update(qa_id='STARSS-FRUSTUM',world_azimuth_deg=0.)
    alternate=deepcopy(members)
    for i,m in enumerate(alternate):m.update(pair_id='second',logical_question_id='second_'+str(i))
    selection=select_balanced_panels(members+alternate,1,1)
    assert {r['pair_id'] for r in selection['selected']}=={'pair'}
    assert {r['pair_id'] for r in selection['not_selected']}=={'second'}
    assert len(selection['selected'])==2
    with pytest.raises(ValueError):select_balanced_panels(members[:1],1,1)


def test_native_bundle_uses_actor_identity_before_v6_fact_materialization():
    from avengine.qa.v7_native_adapter import native_bundle
    s=native_scene();s.update(video_fps=15,sample_rate_hz=16000)
    add_speech(s,'first',0,20,slot=113)
    actor=dict(actor_id=7,endpoint_id='mouth',asset_id='native_person',
               semantic_id=1001,position_m=[0.,0.,-3.],emitter_position_m=[0.,1.5,-3.],
               visible_pixels=200,mask_bbox_xyxy=[1,1,20,20],mask_centroid_px=[10,10])
    visual={'renderer':{'fps':15},'frames':[{'actors':[deepcopy(actor)]} for _ in range(150)]}
    raw=native_bundle(s,0.,'opaque',visual,{})
    assert raw['audio_program']['events'][0]['actor_id']=='7'
    assert set(raw['actors'])=={'7'}


def test_dense_native_visibility_reuses_v6_exact_predicate_and_refuses_unknown_identity():
    import numpy as np
    from avengine.qa.v7_dense_visibility import DenseNativeVisibility
    from avengine.qa.pixel_visibility import _frame_truth
    target=np.zeros((4,4),np.int32);target[1:3,1:3]=1001
    modal=target.copy();modal[1:3,1:3]=2001
    dense=DenseNativeVisibility({'7':1001},{'screen':{'semantic_id':2001}},(4,4))
    dense.add('7',0,modal,target,native_receipt='native-frame')
    dense.add('7',1,np.zeros_like(modal),target,native_receipt='native-stage-unknown')
    proof=dense.finish(2)
    actual=proof['pixel_visibility_truth']['per_instance']['7']['frames'][0].copy();actual.pop('native_receipt')
    assert actual==_frame_truth(modal_mask=modal,target_only_mask=target,target_semantic_id=1001,frame_index=0)
    assert proof['occluder_evidence']['frame_records'][0]['occluder_instance_ids']==['screen']
    assert len(proof['occluder_evidence']['frame_records'])==1
    assert proof['unattributed_occlusion_frames'][0]['unknown_occluder_pixels']==4
    with pytest.raises(ValueError):dense.finish(3)
    with pytest.raises(ValueError):dense.add('7',0,modal,target,native_receipt='duplicate')


def test_camera_frame_readback_must_be_complete_finite_and_on_the_native_clock():
    from avengine.qa.v7_native_adapter import native_bundle
    s=native_scene();s.update(video_fps=15,sample_rate_hz=16000)
    actor=dict(actor_id=7,endpoint_id='mouth',asset_id='native_person',
               semantic_id=1001,position_m=[0.,0.,-3.],emitter_position_m=[0.,1.5,-3.],
               visible_pixels=200,mask_bbox_xyxy=[1,1,20,20],mask_centroid_px=[10,10])
    visual={'renderer':{'fps':15},'frames':[{'actors':[deepcopy(actor)]} for _ in range(150)]}
    camera=[{'frame_index':i,'position_m':[0.,1.5,0.],'yaw_deg':i/15} for i in range(150)]
    raw=native_bundle(s,0.,'opaque',visual,{'camera_frame_readbacks':camera})
    assert raw['frame_readbacks']['camera'][90]['yaw_deg']==6
    for bad in [camera[:-1],camera+camera[:1],[dict(camera[0],frame_index=False)]+camera[1:],
                [dict(camera[0],yaw_deg=float('nan'))]+camera[1:]]:
        with pytest.raises(ValueError):native_bundle(s,0.,'opaque',visual,{'camera_frame_readbacks':bad})

def test_same_direction_different_people_count_two_despite_shared_utterance_asset():
    s=native_scene()
    second=deepcopy(s['sources']['mouth'])
    for t in second['trajectory']:t['position_m']=[0.,1.5,-6.]
    second['actor_id']=8;s['sources']['other_mouth']=second
    s['actor_speech_schedule']['actors'].append({'actor_uid':'second-person','native_actor_id':8,
       'native_endpoint_id':'other_mouth','video_track_id':'native-mask-1002','native_readback':'native-second-person'})
    add_speech(s,'first',0,20,slot=100,asset='same_utterance')
    s['events'].append({'event_id':'second','class_id':1,'source_id':101,'endpoint_id':'other_mouth','start_tick':10,'end_tick':30})
    s['actor_speech_schedule']['event_bindings'].append({'event_id':'second','class_id':1,'source_slot_id':101,
       'endpoint_id':'other_mouth','actor_uid':'second-person','native_actor_id':8,'emission_mechanism':'human_live_formant',
       'activity_observed':True,'activity_receipt':'native-second-dry-activity','sound_asset_id':'same_utterance'})
    assert answer(s)==['2']


def test_selection_never_uses_training_worlds_to_fill_development_targets():
    r=[dict(x,split='train') for x in rows()]
    train=select_balanced_panels(r,2,1,split='train')
    dev=select_balanced_panels(r,2,1,split='dev')
    assert len(train['selected'])==2
    assert not dev['selected'] and dev['status']=='blocked'
    assert not dev['may_publish_as_balanced_panel']

@pytest.mark.parametrize('declared_in',['profile','event'])
def test_playback_cannot_be_relabelled_as_a_live_speaker(declared_in):
 s=native_scene();add_speech(s,'event',0,10,slot=7)
 if declared_in=='profile':s['sources']['mouth']['mechanism']='loudspeaker_playback'
 else:s['events'][0]['mechanism']='loudspeaker_playback'
 with pytest.raises(ValueError):answer(s)
 s.pop('actor_speech_schedule')
 assert not canonical_speech_scene(s)[1]['identity_complete']
 if declared_in=='profile':assert answer(s)==[]
 else:
  with pytest.raises(ValueError):answer(s)

@pytest.mark.parametrize('field',['fx','fy','cx','cy'])
def test_invalid_intrinsic_receipt_is_rejected(field):
 s=native_scene();s['camera_calibration']={field:float('nan')}
 with pytest.raises(ValueError):boundary_projection(s,[0,0,-3])
