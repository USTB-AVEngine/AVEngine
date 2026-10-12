"""Join extended recorded inputs with real, ID-bound automatic receipts.

No audio is listened to by a human. A licence candidate, a model top label or
an incomplete reference scan is never promoted to a permitted input alone.
Existing QA-21 and STARSS answer domains and v6 truth remain unchanged.
"""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import soundfile as sf
from avengine.qa.v7_asset_admission import _loudness
from avengine.qa.v7_auto_admission import provenance_blockers
from avengine.qa.v7_recorded_audio import source_policy_blockers
from avengine.qa.v7_sound_extensions import (
    EXTENDED_CLASSES,STARSS_EXTENSION_MAP,recording_metadata_blockers,
    calibrate_extended,calibrate_caption_model,score_and_rival,caption_score)
from avengine.dataset.sound_segments import measure_activity
from avengine.timeline.continuous_scene import QA21_CLASSES

PERMITTED_LICENSES={'http://creativecommons.org/publicdomain/zero/1.0',
                    'http://creativecommons.org/licenses/by/3.0',
                    'http://creativecommons.org/licenses/by/4.0'}

def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_eval_originals(pool, old_inventory):
    by_id={r['sound_asset_id']:r for r in old_inventory}
    heldout=set()
    for r in pool:
        if r.get('split') not in {'eval','test','evaluation'}:
            continue
        original=by_id.get(r['sound_asset_id']) or {}
        if original.get('fsd50k_id'):
            heldout.add('fsd:'+str(original['fsd50k_id']))
        text=str(r.get('source_origin',''))+' '+str((r.get('source_normalization') or {}).get('source') or '')
        for match in re.finditer(r'fsd50k_(\d+)',text):
            heldout.add('fsd:'+match[1])
        for match in re.finditer(r'(?:vctk_)?((?:p|s)\d+)_(\d+)',text):
            heldout.add('vctk:'+match[1]+'_'+match[2])
    return heldout

def normalize_recorded(row, output):
    """Keep one contiguous crop's sample clock; only apply a constant gain."""
    source=Path(row['path']);x,rate=sf.read(source,dtype='float64',always_2d=True)
    if rate!=16000 or x.shape[1]!=1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('input is not finite nonempty mono16k')
    activity=measure_activity(x[:,0],rate,sound_class='unknown')
    if not activity['source_activity_intervals_samples']:
        raise ValueError('no automatically measured activity')
    loud=_loudness(source)
    if loud['input_i'] is None or loud['input_tp'] is None:
        raise ValueError('integrated LUFS or true peak not measurable')
    gain=min(-23-loud['input_i'],-3.02-loud['input_tp'])
    path=Path(output)/row['sound_class']/(row['sound_asset_id']+'.wav')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise ValueError('normalized output already exists')
    for attempt in range(3):
        sf.write(path,x*10**(gain/20),rate,subtype='FLOAT');observed=_loudness(path)
        if observed['input_tp'] is not None and observed['input_tp']<=-3:
            break
        gain+=-3.02-observed['input_tp']
    y,sr=sf.read(path,dtype='float64',always_2d=True)
    error=float(np.max(np.abs(y-x*10**(gain/20))))
    valid=(sr==rate and y.shape==x.shape and error<=1e-6
           and observed['input_tp'] is not None and observed['input_tp']<=-3)
    profile=measure_activity(y[:,0],sr,sound_class='unknown')
    return {'status':'pass' if valid and profile['source_activity_intervals_samples'] else 'blocked',
       'path':str(path),'sha256':_digest(path),'source_path':str(source),'source_sha256':_digest(source),
       'sample_rate_hz':rate,'sample_count':len(x),'channels':1,'duration_s':len(x)/rate,
       'constant_gain_db':gain,'maximum_constant_gain_error':error,'trimmed_or_repeated':False,
       'target_integrated_lufs':-23,'true_peak_ceiling_dbfs':-3,
       'readback_loudness':observed,
       'common_lufs_target_achieved':observed['input_i'] is not None and abs(observed['input_i']+23)<=.1,
       'true_peak_ceiling_limited':observed['input_i'] is not None and observed['input_i']<-23.1,
       'measured_activity':profile,'human_listening_used':False}

def build_recorded_admission(*,pool,old_inventory,old_admission,vctk_lineages,fsd_rows,
        beats_rows,clap_rows,ontology,collection_mids,class_targets,deduplications,output,
        existing_gate_classes=('dog_bark',),existing_subtype_metadata_groups=None,
        generated_source_authorization=None):
    """Receipt-driven admission. The caller loads read-only corpora explicitly.

    Generated audio requires an explicit owner authorization and bound
    multilingual no-speech/activity receipts. Its model gate is unchanged.
    New original recordings for an existing (pre-extension) class keep that
    class's unchanged BEATs gate, as dog_bark did; CLAP stays a new-class
    alternative only. existing_subtype_metadata_groups re-applies the original
    class's required source-metadata tokens (same substring rule as the
    automatic_audio_v1 screen). Defaults reproduce the round-4 behaviour.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    beat={r['sound_asset_id']:r for r in beats_rows}
    clap={r['sound_asset_id']:r for r in clap_rows}
    source={r['sound_asset_id']:r for r in fsd_rows}
    lineage={r['sound_asset_id']:r.get('license_lineage',r.get('receipt',r)) for r in vctk_lineages}
    evaluations=source_eval_originals(pool,old_inventory)
    dedup={}
    for report in deduplications:
        if report.get('complete_for_listed_roots') is True:
            for key,proof in report.get('records',{}).items():
                dedup[key]=proof
            for control in report.get('positive_controls',[]):
                if control.get('detected') is not True:
                    dedup[control['sound_asset_id']]={'status':'blocked',
                        'reason':'known_gain_crop_or_resample_positive_control_not_detected',
                        'failed_control':control}
    calibration_rows=[];caption_rows=[]
    for sid,r in source.items():
        group='fsd:'+r['original_id']
        if group in evaluations or dedup.get(sid,{}).get('status')!='pass':
            continue
        if not r.get('official_zip_crc_verified'):
            continue
        blockers=[]
        for cls in r['candidate_classes']:
            blockers+=recording_metadata_blockers({**r,'sound_class':cls})
        common={'original_group':group,'trusted_original_mids':collection_mids.get(r['original_id'],[]),
                'candidate_classes':r['candidate_classes'],'calibration_source_blockers':blockers}
        if sid in beat and beat[sid].get('sha256')==_digest(r['path']):
            calibration_rows.append({**beat[sid],**common})
        if sid in clap and clap[sid].get('sha256')==_digest(r['path']):
            caption_rows.append({**clap[sid],**common})
    speech_sources=[r for r in pool if r['sound_class']=='speech_playback' and r.get('split')=='train']
    for r in speech_sources:
        proof=lineage.get(r['sound_asset_id'])
        if (not proof or proof.get('license_chain_status')!='pass'
                or (proof.get('prepared_audio_lineage') or {}).get('status')!='pass'):
            continue
        sid=r['sound_asset_id']
        if dedup.get(sid,{}).get('status')!='pass':
            continue
        common={'original_group':'vctk:'+proof['original_id'],'trusted_original_mids':['/m/09x0r'],
                'candidate_classes':['speech_playback'],'calibration_source_blockers':[]}
        if sid in beat and beat[sid].get('sha256')==_digest(r['path']):
            calibration_rows.append({**beat[sid],**common})
        if sid in clap and clap[sid].get('sha256')==_digest(r['path']):
            caption_rows.append({**clap[sid],**common})
    targets={k:v['target_mids'] for k,v in class_targets.items()}
    extended_gate,relatives=calibrate_extended(calibration_rows,targets,ontology)
    caption_gate=calibrate_caption_model(caption_rows,targets,ontology)
    rows=[]
    # Retain old licensed recordings only after updated source metadata and a
    # fresh identical-query reference check. Old eSpeak assets are retired.
    for r in old_admission['records']:
        if r['sound_class']=='speech_playback':
            if 'author-tts:' in str(r.get('source_uri','')) or 'formant' in str(r.get('source_uri','')).lower():
                retired=dict(r,strict_v7_eligible=False,formal_status='blocked',
                    current_policy_status='blocked',current_policy_blockers=['retired_espeak_formant_not_permitted'],
                    original_group='retired:'+r['sound_asset_id'])
                rows.append(retired)
            continue
        sid=r['sound_asset_id']
        reasons=source_policy_blockers(r)+recording_metadata_blockers(r)
        if not r.get('strict_v7_eligible'):
            reasons+=list(r.get('formal_blockers') or ['previous_automatic_screen_did_not_pass'])
        if dedup.get(sid,{}).get('status')!='pass':
            reasons.append('fresh_reference_fingerprint_blocked_or_missing')
        copy=dict(r)
        copy['reference_dedup']=dedup.get(sid)
        copy['current_policy_status']='admitted' if not reasons else 'blocked'
        copy['current_policy_blockers']=reasons
        copy['strict_v7_eligible']=not reasons
        copy['formal_status']='admitted_under_listed_screens' if not reasons else 'blocked'
        copy['source_original_id']=str(r.get('fsd50k_id') or r.get('source_origin') or sid)
        copy['original_group']='fsd:'+copy['source_original_id']
        copy['updated_policy']='recorded_real_inputs_no_espeak_no_synthetic_effects_v3'
        copy['readback_reused_source']='previous recorded input receipts; fresh full-root fingerprint replay'
        rows.append(copy)
    for original in pool:
        if original['sound_class']!='speech_playback':
            continue
        sid=original['sound_asset_id'];proof=lineage.get(sid) or {}
        score=beat.get(sid);reasons=source_policy_blockers(original)
        if proof.get('license_chain_status')!='pass':
            reasons.append('VCTK_original_copy_licence_chain_missing')
        if (proof.get('prepared_audio_lineage') or {}).get('status')!='pass':
            reasons.append('prepared_clip_not_proven_original_recording_segment')
        group='vctk:'+str(proof.get('original_id','unknown'))
        if group in evaluations:
            reasons.append('evaluation_original_identity_overlap')
        if dedup.get(sid,{}).get('status')!='pass':
            reasons.append('complete_reference_fingerprint_missing_or_blocked')
        confidence=score['scores_by_mid'].get('/m/09x0r',0.) if score else None
        speech_margin=None
        if score:
            observed,rival,_=score_and_rival(score,['/m/09x0r'],relatives)
            speech_margin=observed-rival
        if not score or score['sha256']!=_digest(original['path']) or confidence<.1917129977 or speech_margin<.05:
            reasons.append('bound_BEATS_Speech_below_existing_confidence')
        row={**original,'source_dataset':'VCTK-0.92','source_original_id':proof.get('original_id'),
          'original_group':group,'license':'CC-BY-4.0','source_license_receipt':proof,
          'source_uri':proof.get('source_uri'),'attribution':proof.get('attribution'),
          'human_review_required':False,
          'voice_gender':proof.get('gender'),'content_class':'speech',
          'compatible_emission_mechanisms':['human_live_recorded_speech','loudspeaker_playback'],
          'effective_qa21_answer_class_if_playback':'speech_playback',
          'starss_live_class_id':0 if proof.get('gender')=='female' else 1 if proof.get('gender')=='male' else None,
          'starss_playback_class_id':None,
          'reference_dedup':dedup.get(sid),'automatic_label':{'model':'BEATs','confidence':confidence,'threshold':.1917129977,'margin':speech_margin,'minimum_margin':.05,
                             'proof':score,'semantic_ASR_not_claimed':True},
          'current_policy_blockers':sorted(set(reasons)),'current_policy_status':'blocked' if reasons else 'pending_normalization'}
        rows.append(row)
    for sid,r in source.items():
        for cls in r['candidate_classes']:
            generated=r.get('source_type')=='generated_audio'
            group=('sao:' if generated else 'fsd:')+r['original_id']
            if generated:
                from avengine.qa.v7_sao_admission import generated_source_blockers
                reasons=generated_source_blockers({**r,'sound_class':cls},generated_source_authorization)
            else:
                reasons=source_policy_blockers(r)
            reasons+=recording_metadata_blockers({**r,'sound_class':cls})
            groups=(existing_subtype_metadata_groups or {}).get(cls) or []
            if groups:
                subtype_text=' '.join(str(r['source_metadata'].get(k,'')) for k in ('title','description','tags')).lower()
                if not all(any(token.lower() in subtype_text for token in group) for group in groups):
                    reasons.append('original_metadata_does_not_confirm_specific_subtype')
            lic=r['source_license'].replace('https://','http://').rstrip('/')
            if not generated and (lic not in PERMITTED_LICENSES or not r.get('official_zip_crc_verified')):
                reasons.append('per_clip_license_or_official_waveform_association_missing')
            if group in evaluations:
                reasons.append('evaluation_original_identity_overlap')
            if dedup.get(sid,{}).get('status')!='pass':
                reasons.append('complete_reference_fingerprint_missing_or_blocked')
            scored=beat.get(sid);caption=clap.get(sid);model_proofs=[];accepted=[]
            bg=extended_gate.get(cls) or {}
            if cls in existing_gate_classes:
                bg={'status':'pass','threshold':.1917129977,'minimum_margin':.05,
                    'source':'unchanged existing dry target confidence, new original Bark recordings' if cls=='dog_bark'
                        else 'unchanged existing dry target confidence, new original recordings for an existing class'}
            if scored and scored['sha256']==_digest(r['path']) and bg.get('status')=='pass':
                confidence,rival,rival_mid=score_and_rival(scored,targets.get(cls,r['target_mids']),relatives)
                passed=confidence>=bg['threshold'] and confidence-rival>=bg['minimum_margin']
                accepted.append(passed)
                model_proofs.append({'model':'BEATs','status':'pass' if passed else 'blocked',
                    'confidence':confidence,'margin':confidence-rival,'rival_mid':rival_mid,'gate':bg,
                    'scores_source_asset_id':sid,'input_sha256':scored['sha256'],'top8':scored['top8']})
            else:
                model_proofs.append({'model':'BEATs','status':'blocked','reason':'bound_model_or_passing_positive_calibration_missing','gate':bg})
            cg=caption_gate.get(cls) or {}
            # Existing dog_bark screening is retained. CLAP is only a new-class
            # alternative, calibrated before applying it to new recordings.
            if cls not in existing_gate_classes and caption and caption['sha256']==_digest(r['path']) and cg.get('status')=='pass':
                confidence,margin=caption_score(caption,cls)
                passed=confidence>=cg['threshold_cosine'] and margin>=cg['minimum_cosine_margin']
                accepted.append(passed)
                model_proofs.append({'model':'CLAP','status':'pass' if passed else 'blocked',
                    'cosine':confidence,'margin':margin,'gate':cg,
                    'scores_source_asset_id':sid,'input_sha256':caption['sha256'],'top8':caption['top8']})
            if not any(accepted):
                reasons.append('no_source_calibrated_automatic_label_model_pass')
            text=' '.join(str(r['source_metadata'].get(k,'')) for k in ('title','description','tags')).lower()
            subtype=None
            if cls=='guitar_live':
                if 'Acoustic_guitar' in r['official_collection_labels']:subtype='acoustic'
                elif 'Electric_guitar' in r['official_collection_labels']:subtype='electric'
            elif cls=='hand_drum_live':
                subtype=next((n for n in ('tabla','bongo','conga','djembe','darbuka','doumbek','tambourine','frame_drum') if n.replace('_',' ') in text),None)
            row={**r,'sound_asset_id':sid+'__'+cls,'sound_class':cls,'dedup_query_asset_id':sid,
                 'model_query_asset_id':sid,'source_original_id':r['original_id'],'original_group':group,
                 'license':r['source_license'],'attribution':{k:r['source_metadata'].get(k) for k in ('title','uploader','license')},
                 'content_class':cls,'instrument_subtype':subtype,
                 'qa21_domain_extended':False,'effective_qa21_answer_class':None if cls in EXTENDED_CLASSES else cls,
                 'starss_class_id_if_live':STARSS_EXTENSION_MAP.get(cls),
                 'starss_class_id_if_music_playback':8 if cls.endswith('_live') else None,
                 'native_emission_capability_not_yet_admitted':True,
                 'reference_dedup':dedup.get(sid),'automatic_label':model_proofs,'current_policy_blockers':sorted(set(reasons)),
                 'current_policy_status':'blocked' if reasons else 'pending_normalization'}
            rows.append(row)
    pending=[r for r in rows if r.get('current_policy_status')=='pending_normalization']
    def normalize(row):
        try:
            receipt=normalize_recorded(row,output/'normalized')
            row['normalization']=receipt
            if receipt['status']!='pass':
                row['current_policy_blockers'].append('automatic_activity_or_normalization_failed')
        except Exception as error:
            row['current_policy_blockers'].append('automatic_activity_or_normalization:'+str(error))
        row['current_policy_status']='blocked' if row['current_policy_blockers'] else 'admitted'
        row['strict_v7_eligible']=not row['current_policy_blockers']
        row['formal_status']='admitted_under_listed_screens' if row['strict_v7_eligible'] else 'blocked'
        return row
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(normalize,pending))
    for row in rows:
        row['acquisition_provenance_status']=row.get('provenance_status')
        qualified=('qualified_generated_input_under_listed_screens' if row.get('source_type')=='generated_audio'
                   else 'qualified_recorded_input_under_listed_screens')
        row['provenance_status']=qualified if row.get('strict_v7_eligible') else 'blocked'
    class_names=list(dict.fromkeys(list(QA21_CLASSES)+['any_audioset_class_playback']+list(EXTENDED_CLASSES)))
    by_class=[]
    for cls in class_names:
        members=[r for r in rows if r['sound_class']==cls]
        passed=[r for r in members if r.get('strict_v7_eligible')]
        originals={r['original_group'] for r in passed}
        reasons=Counter(reason for r in members for reason in r.get('current_policy_blockers',[]))
        by_class.append({'sound_class':cls,'examined_records':len(members),'admitted_records':len(passed),
          'admitted_distinct_original_ids':len(originals),'target_distinct_originals':20 if cls in EXTENDED_CLASSES else None,
          'target_deficit':max(0,20-len(originals)) if cls in EXTENDED_CLASSES else None,
          'typical_rejections':dict(reasons.most_common(6)) if cls!='any_audioset_class_playback' else {'composite_placeholder_not_a_fixed_answer':1},
          'example_rejected_ids':[r['sound_asset_id'] for r in members if not r.get('strict_v7_eligible')][:5]})
    policy=('v7_recorded_plus_owner_authorized_sao_train_only_20261012'
            if any(r.get('source_type')=='generated_audio' for r in source.values())
            else 'recorded_real_inputs_no_espeak_no_synthetic_effects_v3')
    result={'policy':policy,'records':rows,'by_class':by_class,
       'admitted_records':sum(r.get('strict_v7_eligible',False) for r in rows),
       'admitted_distinct_original_ids':len({r['original_group'] for r in rows if r.get('strict_v7_eligible')}),
       'old_35_classes_nonempty':sum(r['admitted_records']>0 for r in by_class if r['sound_class'] not in EXTENDED_CLASSES),
       'new_14_classes_nonempty':sum(r['admitted_records']>0 for r in by_class if r['sound_class'] in EXTENDED_CLASSES),
       'new_14_classes_meeting_20_origins':sum(r['admitted_distinct_original_ids']>=20 for r in by_class if r['sound_class'] in EXTENDED_CLASSES),
       'retired_espeak_formant_assets':sum('retired_espeak_formant_not_permitted' in r.get('current_policy_blockers',[]) for r in rows),'full_pipeline_native_questions':0,
       'calibration':{'BEATs':extended_gate,'CLAP':caption_gate,'fusion':'pass either source-calibrated model plus every source/clock/normalization/reference screen'},
       'reference_scan_reports':[{'file_count':r['reference_file_count'],'waveform_count':r['reference_waveform_count'],
              'complete_for_listed_roots':r['complete_for_listed_roots'],'query_count':len(r['records']),
              'seconds':r['seconds'],'scope_limit':r['scope_limit']} for r in deduplications],
       'source_license_count_VCTK':sum(r.get('license_chain_status')=='pass' for r in lineage.values()),
       'source_heldout_VCTK_records':sum(r['sound_class']=='speech_playback' and r.get('split')=='eval' for r in pool),
       'original_eval_groups':sorted(evaluations),'human_listening_used':False,
       'limits':['automatic label retention is not classification accuracy',
         'spectral fingerprint scan cannot guarantee absence of masked or strongly reverberant shared source',
         'native human/contact/voice-visual/physical emitter admission remains separate',
         'instrument subtype must match final admitted roster before emission']}
    (output/'dry_recorded_admission.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return result
