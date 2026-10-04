from copy import deepcopy
import json,math
from pathlib import Path
import numpy as np
import pytest
from avengine.qa.v7_protocol import generate_questions
from avengine.qa.v7_native_adapter import _crop_occluder_evidence, _playback_class_is_supported, _resolved_qa21_class
from avengine.qa import unified_catalog as u
from avengine.qa.v7_auto_admission import (provenance_blockers,qualify_dry_record,audio_fingerprint,fingerprint_alignment,fingerprint_index,calibrate_label_gate)
from avengine.qa.v7_group_selection import (MEMBERS,binding_reservation,select_counterfactual_group)
from avengine.qa.v7_visual_admission import measure_native_color
from avengine.qa.v7_human_admission import qualify_action_samples
pytestmark=pytest.mark.fast_unit
def static_scene(lo,hi):
 return {'events':[{'source_id':1,'class_id':1,'endpoint_id':'person','start_tick':lo,'end_tick':hi}],
 'listener_m':[0.,1.5,0.],'camera_forward':[0.,0.,-1.],
 'sources':{'person':{'actor_id':1,'mechanism':'human_live','trajectory':[{'time_s':0.,'position_m':[0.,1.5,-3.]},{'time_s':30.,'position_m':[0.,1.5,-3.]}]}}}
@pytest.mark.parametrize('anchor,end,query',[(0,1,0),(10,11,1),(20,21,2),(1,11,1),(1,10,None),(99,100,None)])
def test_authoritative_integer_anchor_and_half_open_endpoint(anchor,end,query):
 p,g,_=generate_questions(static_scene(anchor,end),0.,'integer',{})
 rows=[x for x in p if x['qa_id']=='QA-25']
 assert len(rows)==(query is not None)
 if rows:
  byid={x['question_id']:x for x in g}
  assert byid[rows[0]['question_id']]['evidence']['query_s']==query
def test_carry_in_integer_anchor_is_zero():
 p,g,_=generate_questions(static_scene(95,110),10.,'carry',{})
 row=next(x for x in p if x['qa_id']=='QA-25')
 gold=next(x for x in g if x['question_id']==row['question_id'])
 assert gold['evidence']['query_s']==0 and gold['evidence']['carry_in']
def test_occluder_cropping_uses_authority_target_instance_id():
 r={'frame_index_scope':'scene','frame_records':[
 {'target_instance_id':'1','frame_index':77,'occluder_instance_ids':['screen'],'unknown_occluder_pixels':0},
 {'target_instance_id':'1','frame_index':78,'occluder_instance_ids':['screen'],'unknown_occluder_pixels':0}]}
 c=_crop_occluder_evidence(r,78,150,{'screen':{}})
 assert c['frame_records'][0]['frame_index']==0
 assert u._occluder_ids({'visibility':{},'occluder_evidence':c},'1',0)==['screen']
 assert r['frame_records'][1]['frame_index']==78
@pytest.mark.parametrize('change,message',[
 ({'occluder_instance_ids':['unknown']},'absent'),({'occluder_instance_ids':['screen','screen']},'unique'),
 ({'unknown_occluder_pixels':1},'unattributed')])
def test_occluder_invalid_identity_cannot_be_published(change,message):
 row=dict(target_instance_id='1',frame_index=78,occluder_instance_ids=['screen'],unknown_occluder_pixels=0);row.update(change)
 with pytest.raises(ValueError,match=message):_crop_occluder_evidence({'frame_records':[row]},78,150,{'screen':{}})
def test_duplicate_target_frame_is_rejected():
 row=dict(target_instance_id='1',frame_index=78,occluder_instance_ids=['screen'])
 with pytest.raises(ValueError,match='duplicate'):_crop_occluder_evidence({'frame_records':[row,row]},78,150,{'screen':{}})
def passed_screens():
 d={k:{'status':'pass'} for k in ['license','provenance','deduplication','automatic_label','automatic_activity','normalization']}
 d['deduplication']['coverage_complete_for_listed_roots']=True;return d
@pytest.mark.parametrize('record',[
 {'source_dataset':'STARSS23'},{'source_uri':'DCASE2025/x'},{'split':'eval'},
 {'official_metadata':{'description':'Binaural recording of a room'}}])
def test_prohibited_spatial_or_evaluation_audio_stays_blocked(record):
 r=dict(sound_class='speech_playback',source_dataset='project_tts');r.update(record)
 assert not qualify_dry_record(r,passed_screens())['strict_v7_eligible']
def test_recorded_mono_is_allowed_without_human_review():
 r=qualify_dry_record({'sound_class':'cat_meow','source_dataset':'FSD50K'},passed_screens())
 assert r['strict_v7_eligible'] and r['human_review_required'] is False
def test_incomplete_reference_scan_cannot_admit_audio():
 screens=passed_screens();screens['deduplication']['coverage_complete_for_listed_roots']=False
 assert not qualify_dry_record({'sound_class':'speech_playback'},screens)['strict_v7_eligible']
def test_composite_placeholder_needs_observed_content_and_playback_mechanism():
 r={'sound_class':'any_audioset_class_playback'}
 assert not qualify_dry_record(r,passed_screens())['strict_v7_eligible']
 r.update(content_class='cat_meow',emission_mechanism='loudspeaker_playback')
 assert qualify_dry_record(r,passed_screens())['strict_v7_eligible']
def test_fingerprint_detects_gain_and_crop():
 sr=16000;t=np.arange(48000)/sr
 x=np.sin(2*np.pi*(220*t+140*t*t))*.3+np.sin(2*np.pi*(701*t+70*t*t))*.1
 base=audio_fingerprint(x,sr)
 gain=fingerprint_alignment(base,audio_fingerprint(x*.31,sr))
 crop=fingerprint_alignment(audio_fingerprint(x[8000:40000],sr),base)
 assert gain['aligned_fraction']>.9
 assert crop['matched_hashes']>=12 and crop['aligned_fraction']>.25
def test_calibration_excludes_overlapping_target_synonyms():
 config={'a':['x'],'b':['x'],'c':['y']}
 rows=[{'sound_asset_id':'known','sound_class':'a','scores_by_mid':{'x':.9,'y':.1}}]
 receipt=calibrate_label_gate(rows,config,calibration_ids={'known'})
 assert receipt['negative_count']==1
 assert receipt['threshold']<.9 and receipt['positive_recall']==1
def test_binding_reservation_covers_whole_seed_table():
 rows=[{'split':'train','window_count':2,'members':list(MEMBERS)}]*100+[
       {'split':'train','window_count':2,'members':['v0_a0']}]*300
 r=binding_reservation(rows,split='train')
 assert r['all_member_windows']==1400 and r['quartet_member_windows']==800
 assert r['binding_items_per_quartet_window']==5 and r['planned_binding_fraction']>=.2
def group(tmp_path,vision=4):
 public={};private={}
 for m in MEMBERS:
  v=tmp_path/(m[:2]+'.mp4');v.write_bytes(m[:2].encode())
  a=tmp_path/(m[3:]+'.wav');a.write_bytes(m[3:].encode())
  qs=[];gs=[]
  for i,mod in enumerate(['audiovisual']*6+['audio']*4+['vision']*vision):
   qid=m+'_'+str(i);qa='QA-02' if mod=='audiovisual' else 'QA-03'
   qs.append({'question_id':qid,'logical_question_id':qid,'qa_id':qa,'protocol_panel':'v6','question_zh':'问题'+str(i),'form':'open','modality':mod,'media':{'video':str(v),'audio_foa':str(a)}})
   gs.append({'question_id':qid,'logical_question_id':qid,'canonical_answer':str((m in ['v0_a0','v1_a1']) if mod=='audiovisual' else i),'answer':'value'})
  public[m]=qs;private[m]=gs
 return public,private
def test_binding_selector_preserves_modal_counts_and_measured_binding(tmp_path):
 p,g=group(tmp_path);alloc={m:{'vision':4,'audio':4,'audiovisual':6} for m in MEMBERS}
 q,a,r=select_counterfactual_group(p,g,alloc)
 assert r['binding_items_per_member']==5
 for m in MEMBERS:
  assert len(q[m])==len(a[m])==14
  assert sum(x['modality']=='audiovisual' for x in q[m])==6
def test_binding_quota_does_not_hide_visual_deficit(tmp_path):
 p,g=group(tmp_path,vision=1)
 with pytest.raises(ValueError,match='vision: 1 legal'):
  select_counterfactual_group(p,g,{m:{'vision':4,'audio':4,'audiovisual':6} for m in MEMBERS})
def test_binding_media_must_be_actual_audio_intervention(tmp_path):
 p,g=group(tmp_path);Path(p['v0_a1'][0]['media']['audio_foa']).write_bytes(b'a0')
 with pytest.raises(ValueError,match='identical audio'):
  select_counterfactual_group(p,g,{m:{'vision':4,'audio':4,'audiovisual':6} for m in MEMBERS})
def test_automatic_color_blocks_wrong_label_and_mask_mismatch():
 rgb=np.zeros((30,30,3),np.uint8);rgb[:]=[25,85,220];sem=np.full((30,30),1001)
 assert measure_native_color(rgb,sem,1001,'blue')['status']=='pass'
 assert measure_native_color(rgb,sem,1001,'yellow')['status']=='blocked'
 with pytest.raises(ValueError,match='dimensions'):measure_native_color(rgb,sem[:1],1001,'blue')
def test_human_grounding_does_not_allow_proxy_escape_or_floating():
 good={'left_sole_y_relative_m':.002,'right_sole_y_relative_m':.003,'body_min_y_relative_m':.002}
 assert qualify_action_samples([good],maximum_contact_error_m=.015,proxy_covers_pose=True)['status']=='pass'
 assert qualify_action_samples([good],maximum_contact_error_m=.015,proxy_covers_pose=False)['status']=='blocked'
 bad=dict(good,left_sole_y_relative_m=.05,right_sole_y_relative_m=.04,body_min_y_relative_m=.04)
 assert qualify_action_samples([bad],maximum_contact_error_m=.015,proxy_covers_pose=True)['status']=='blocked'

def test_cached_fingerprint_alignment_preserves_result():
    t = np.arange(48000) / 16000
    wave = np.sin(2 * np.pi * (277 * t + 71 * t * t))
    reference = audio_fingerprint(wave, 16000)
    query = audio_fingerprint(wave[4000:40000] * .23, 16000)
    assert fingerprint_alignment(query, reference) == fingerprint_alignment(
        query, reference, query_index=fingerprint_index(query, query=True),
        reference_index=fingerprint_index(reference))


def test_calibration_excludes_all_original_multilabel_targets():
    row = {'sound_asset_id': 'clip', 'sound_class': 'a',
           'trusted_original_mids': ['x', 'y'],
           'scores_by_mid': {'x': .9, 'y': .8, 'z': .01}}
    result = calibrate_label_gate([row], {'a': ['x'], 'b': ['y'], 'c': ['z']},
                                  calibration_ids={'clip'}, minimum_positive_recall=0)
    assert result['negative_count'] == 1


def test_unverified_original_label_cannot_calibrate_gate():
    known = {'sound_asset_id': 'known', 'sound_class': 'a',
             'scores_by_mid': {'x': .9, 'y': .01}}
    bad_metadata = {'sound_asset_id': 'unverified', 'sound_class': 'a',
                    'calibration_class_valid': False,
                    'scores_by_mid': {'x': .01, 'y': .9}}
    config = {'a': ['x'], 'b': ['y']}
    baseline = calibrate_label_gate([known], config, calibration_ids={'known'})
    combined = calibrate_label_gate([known, bad_metadata], config,
                                    calibration_ids={'known', 'unverified'})
    assert combined == baseline


def test_evaluation_original_id_overlap_is_rejected():
    record = {'source_dataset': 'FSD50K', 'source_asset_id': 'original_42'}
    assert 'evaluation_source_identity_overlap' in provenance_blockers(
        record, evaluation_source_ids={'original_42'})


def test_live_tts_cannot_be_relabelled_as_playback():
    proof = {'sound_class': 'speech_playback', 'emission_mechanism': 'loudspeaker_playback'}
    assert not _playback_class_is_supported(proof, {'emission_mechanism': 'human_live'})
    assert not _playback_class_is_supported(proof, {})
    assert _playback_class_is_supported(proof, {'emission_mechanism': 'loudspeaker_playback'})

def test_prohibited_source_cannot_change_label_calibration(tmp_path, monkeypatch):
    import hashlib
    import soundfile as sf
    from avengine.qa.v7_auto_admission import build_automatic_admission
    from avengine.qa import v7_asset_admission

    group_id = next(str(i) for i in range(100)
                    if int(hashlib.sha256(str(i).encode()).hexdigest()[:8], 16) % 4 == 0)
    wave = .2 * np.sin(2 * np.pi * 271 * np.arange(32000) / 16000)
    records = []
    model_rows = []
    for sid, dataset, scores in [('safe', 'FSD50K', {'x': .9, 'y': .01}),
                                 ('forbidden', 'STARSS23', {'x': .01, 'y': .99})]:
        path = tmp_path / (sid + '.wav')
        sf.write(path, wave, 16000, subtype='FLOAT')
        records.append({'sound_asset_id': sid, 'sound_class': 'a', 'path': str(path),
                        'source_dataset': dataset, 'license': 'CC0',
                        'license_candidate_eligible': True, 'fsd50k_id': group_id})
        model_rows.append({'sound_asset_id': sid, 'sound_class': 'a',
                           'scores_by_mid': scores, 'trusted_original_mids': ['x'],
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    monkeypatch.setattr(v7_asset_admission, '_loudness',
                        lambda path: {'input_i': -23., 'input_tp': -20.})
    result = build_automatic_admission(
        {'records': records}, {'sounds': records}, model_rows,
        {'checkpoint': 'test_cpu_head', 'checkpoint_sha256': 'test'},
        {'class_mids': {'a': ['x'], 'b': ['y']}},
        {'complete_for_listed_roots': True,
         'records': {r['sound_asset_id']: {'status': 'pass'} for r in records}},
        {'sounds': []}, tmp_path / 'automatic')
    assert result['label_calibration']['model_rows_excluded_by_source_screen'] == 1
    assert result['label_calibration']['positive_count'] == 1
    assert result['label_calibration']['calibration_sound_asset_ids'] == ['safe']
    admitted = {r['sound_asset_id']: r for r in result['records']}
    assert admitted['safe']['strict_v7_eligible']
    assert not admitted['forbidden']['strict_v7_eligible']

def test_composite_playback_preserves_fixed_qa21_domain():
    profile = {'emission_mechanism': 'loudspeaker_playback'}
    proof = {'sound_class': 'any_audioset_class_playback',
             'emission_mechanism': 'loudspeaker_playback', 'content_class': 'speech_playback'}
    assert _resolved_qa21_class(proof, profile) == 'speech_playback'
    assert _resolved_qa21_class(dict(proof, content_class='unknown_audioset_class'), profile) is None
    assert _resolved_qa21_class(proof, {'emission_mechanism': 'human_live'}) is None

def test_original_metadata_cannot_hide_prohibited_dataset_origin():
    record = {'source_dataset': 'FSD50K',
              'official_metadata': {'description': 'Extracted from a STARSS23 spatial recording'}}
    assert 'prohibited_real_spatial_dataset_provenance' in provenance_blockers(record)
