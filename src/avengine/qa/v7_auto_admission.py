"""Automatic dry-input screens. Original data and classifiers remain read-only.

Recorded mono inputs are allowed. Spatial/evaluation provenance, rights,
machine label confidence, automatic activity and measured loudness are
separate mandatory screens. A classifier score is not a source license.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import numpy as np
from scipy import signal, ndimage
from avengine.timeline.continuous_scene import QA21_CLASSES

POLICY_ID = "v7_recorded_mono_inputs_no_real_spatial_or_eval_v2"
PROHIBITED_TOKENS = ("starss", "dcase", "tau-srir", "tau_srir", "tau_rir", "tau_test_rir")

def provenance_blockers(record, *, evaluation_source_ids=()):
    fields = [record.get(k, "") for k in ("source_origin", "source_pcm_path",
              "source_uri", "source_dataset", "source_relative")]
    text = " ".join(str(x) for x in fields).lower()
    source_metadata = record.get("official_metadata") or {}
    spatial = " ".join(str(source_metadata.get(k,"")) for k in ("title","description","tags")).lower()
    reasons = []
    if any(token in text or token in spatial for token in PROHIBITED_TOKENS):
        reasons.append("prohibited_real_spatial_dataset_provenance")
    if any(token in spatial for token in ("binaural", "ambisonic", "room impulse response")):
        reasons.append("original_metadata_describes_spatial_recording")
    if record.get("split") in {"eval", "test", "evaluation"}:
        reasons.append("evaluation_source_split")
    identities = set(record.get("sound_identity_keys") or ())
    identities.update(str(record.get(k, "")) for k in ("source_asset_id", "source_origin", "source_id"))
    if identities.intersection(evaluation_source_ids):
        reasons.append("evaluation_source_identity_overlap")
    return reasons

def audio_fingerprint(wave, rate):
    """Gain-invariant 8 kHz spectral peak constellations, plus canonical PCM."""
    x = np.asarray(wave, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("fingerprint expects a finite nonempty mono signal")
    rate = int(rate);divisor = math.gcd(rate, 8000)
    x = signal.resample_poly(x, 8000 // divisor, rate // divisor)
    peak = float(np.max(np.abs(x)))
    if peak < 1e-12:
        return {"pcm_sha256": None, "landmarks": [], "seconds": len(x)/8000}
    canonical = np.clip(np.rint(x / peak * 32767), -32767, 32767).astype("<i2")
    digest = hashlib.sha256(canonical.tobytes()).hexdigest()
    if len(x) < 512:
        return {"pcm_sha256": digest, "landmarks": [], "seconds": len(x)/8000}
    _,_,z = signal.stft(x, fs=8000, nperseg=512, noverlap=384, boundary=None, padded=False)
    db = 20*np.log10(np.maximum(np.abs(z), 1e-12))
    maxima = (db == ndimage.maximum_filter(db, size=(7, 5))) & (db > db.max()-45)
    maxima[:3] = False
    by_time = {}
    for t in range(db.shape[1]):
        f = np.flatnonzero(maxima[:,t])
        if len(f):by_time[t] = sorted(f, key=lambda k: -db[k,t])[:3]
    times=np.asarray(sorted(by_time),dtype=np.int64)
    if not len(times):
        return {"pcm_sha256":digest, "landmarks":[], "seconds":len(x)/8000}
    frequencies=np.full((len(times),3),-1,dtype=np.int64)
    target_indices=np.full((len(times),5),-1,dtype=np.int64)
    for i,t in enumerate(times):
        fs=by_time[int(t)];frequencies[i,:len(fs)]=fs
        targets=[j for j in range(i+1,min(i+16,len(times))) if 2<=times[j]-t<=32][:5]
        target_indices[i,:len(targets)]=targets
    safe=np.maximum(target_indices,0)
    first=frequencies[:,:,None,None]
    second=frequencies[safe,:2][:,None,:,:]
    delta=(times[safe]-times[:,None])[:,None,:,None]
    hashes=(first*257+second)*64+delta
    valid=(first>=0)&(second>=0)&(target_indices[:,None,:,None]>=0)
    anchor_times=np.broadcast_to(times[:,None,None,None],hashes.shape)
    landmarks=list(zip(hashes[valid].tolist(),anchor_times[valid].tolist()))
    return {"pcm_sha256":digest, "landmarks":landmarks, "seconds":len(x)/8000}

def fingerprint_index(fingerprint, *, query=False):
    result={}
    for i,(h,t) in enumerate(fingerprint['landmarks']):
        values=result.setdefault(h,[])
        if query:values.append((t,i))
        elif len(values)<128:values.append(t)
    return result

def fingerprint_alignment(query, reference, *, query_index=None, reference_index=None):
    if query["pcm_sha256"] and query["pcm_sha256"] == reference["pcm_sha256"]:
        return {"exact_normalized_pcm": True, "matched_hashes": len(query["landmarks"]), "aligned_fraction":1., "offset_frames":0}
    by_reference=fingerprint_index(reference) if reference_index is None else reference_index
    by_query=fingerprint_index(query,query=True) if query_index is None else query_index
    offsets=[];keys=[];ordinals=[]
    for h,qt in by_query.items():
        rt=by_reference.get(h)
        if not rt:continue
        rt=np.asarray(rt[:128],dtype=np.int64);qt=np.asarray(qt,dtype=np.int64)
        values=np.rint((rt[None,:]-qt[:,0,None])/2).astype(np.int64)
        offsets.append(values.ravel());keys.append(np.full(values.size,h,dtype=np.int64))
        ordinals.append((qt[:,1,None]*128+np.arange(len(rt))[None,:]).ravel())
    if not offsets:return {"exact_normalized_pcm":False,"matched_hashes":0,"aligned_fraction":0.,"offset_frames":None}
    offsets=np.concatenate(offsets);keys=np.concatenate(keys);ordinals=np.concatenate(ordinals)
    lower=int(offsets.min());votes=np.bincount(offsets-lower)
    winners=np.flatnonzero(votes==votes.max())+lower
    eligible=np.flatnonzero(np.isin(offsets,winners))
    first=eligible[int(np.argmin(ordinals[eligible]))]
    offset=int(offsets[first]);hashes=len(np.unique(keys[offsets==offset]))
    total=max(1,len(by_query))
    return {"exact_normalized_pcm":False,"matched_hashes":hashes,"aligned_fraction":hashes/total,"offset_frames":offset*2}

def label_score(row, target_mids):
    scores=row["scores_by_mid"]
    return max((float(scores.get(mid,0)) for mid in target_mids), default=0.)

def calibrate_label_gate(rows, class_mids, *, calibration_ids, minimum_positive_recall=.5):
    """Calibrate on metadata-backed positives and deliberately wrong labels.

    Incompatible targets sharing classifier labels are excluded as negative
    controls; such controls could not distinguish two synonyms.
    """
    positive=[];negative=[];positive_margins=[];negative_margins=[]
    for row in rows:
        if row["sound_asset_id"] not in calibration_ids or row.get("calibration_class_valid") is False:continue
        mids=set(class_mids.get(row["sound_class"],[]))
        trusted=set(row.get("trusted_original_mids") or mids)
        if not mids:continue
        score=label_score(row,mids)
        competitors=[label_score(row,v) for k,v in class_mids.items()
                     if k!=row["sound_class"] and v and not mids.intersection(v)]
        positive.append(score)
        positive_margins.append(score-max(competitors,default=0.))
        for other,other_mids in class_mids.items():
            other_mids=set(other_mids)
            if other!=row["sound_class"] and other_mids and not trusted.intersection(other_mids):
                wrong=label_score(row,other_mids)
                rival=[label_score(row,v) for k,v in class_mids.items()
                       if k!=other and v and not other_mids.intersection(v)]
                negative.append(wrong);negative_margins.append(wrong-max(rival,default=0.))
    if not positive or not negative:
        raise ValueError("calibration requires both positive and negative controls")
    # Bound the observed false acceptance of wrong, nonoverlapping classes.
    threshold=max(.05,float(np.quantile(negative,.99))+1e-6)
    margin=max(.05,float(np.quantile(negative_margins,.99))+1e-6)
    positive_ok=(np.asarray(positive)>=threshold)&(np.asarray(positive_margins)>=margin)
    negative_ok=(np.asarray(negative)>=threshold)&(np.asarray(negative_margins)>=margin)
    recall=float(np.mean(positive_ok))
    return {"threshold":threshold,"minimum_margin":margin,"positive_count":len(positive),"negative_count":len(negative),
            "positive_recall":recall,"negative_false_accept_rate":float(np.mean(negative_ok)),
            "minimum_positive_recall":minimum_positive_recall,
            "status":"pass" if recall>=minimum_positive_recall else "insufficient_positive_recall",
            "truth_source":"official original AudioSet metadata or authored TTS; negative controls exclude every known original label",
            "human_review_required":False}

def qualify_dry_record(record, screens):
    required=("license","provenance","deduplication","automatic_label","automatic_activity","normalization")
    blockers=provenance_blockers(record)
    for key in required:
        proof=screens.get(key) or {}
        if proof.get("status")!="pass":blockers.append(key+":"+str(proof.get("reason","missing_or_failed_receipt")))
    if (screens.get("deduplication") or {}).get("coverage_complete_for_listed_roots") is not True:
        blockers.append("deduplication:reference_coverage_not_complete")
    if record["sound_class"]=="any_audioset_class_playback":
        if not record.get("content_class") or record.get("emission_mechanism")!="loudspeaker_playback":
            blockers.append("composite_requires_content_class_and_loudspeaker_mechanism")
    return {"policy_id":POLICY_ID,"strict_v7_eligible":not blockers,
            "formal_status":"admitted" if not blockers else "blocked",
            "formal_blockers":sorted(set(blockers)),"screens":screens,
            "human_review_required":False}


def build_automatic_admission(inventory, pool, model_rows, model_receipt, class_config,
                              deduplication, tts, output):
    """Join measured, ID-bound screens; never turn an inventory into admission."""
    import json
    from pathlib import Path
    import soundfile as sf
    from avengine.qa.v7_asset_admission import _loudness, _write
    from avengine.dataset.sound_segments import measure_activity

    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    originals={r['sound_asset_id']:r for r in pool['sounds']}
    evaluation_ids=set(deduplication.get('evaluation_source_ids',()))
    for original in pool['sounds']:
        if original.get('split') in {'eval','evaluation','test'}:
            evaluation_ids.update(original.get('sound_identity_keys') or ())
            evaluation_ids.update(str(original[k]) for k in ('source_asset_id','source_origin','source_id') if original.get(k))
    original_groups={}
    for r in inventory['records']:
        sid=r['sound_asset_id'];original_groups[sid]=str(r.get('fsd50k_id') or r.get('source_origin') or sid)
    for r in tts['sounds']:
        original_groups[r['sound_asset_id']]=str(r.get('transcript') or r['sound_asset_id'])
    scores={r['sound_asset_id']:r for r in model_rows}
    calibration_ids={sid for sid,group in original_groups.items()
                     if int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%4==0}
    calibration_rows = model_rows
    complete_scan = deduplication.get('complete_for_listed_roots') is True
    if complete_scan:
        # The label gate itself must not use prohibited/evaluation source clips.
        inventory_by_id = {r['sound_asset_id']: r for r in inventory['records']}
        tts_ids = {r['sound_asset_id'] for r in tts['sounds']}
        allowed = set()
        for scored in model_rows:
            sid = scored['sound_asset_id']
            meta_record = inventory_by_id.get(sid, {})
            source = originals.get(sid, {})
            prohibited = provenance_blockers(
                {**source, 'official_metadata': meta_record.get('official_metadata', {})},
                evaluation_source_ids=evaluation_ids)
            official_eval = 'eval_clips' in str(
                meta_record.get('official_metadata', {}).get('metadata_member', ''))
            if ((meta_record.get('license_candidate_eligible') or sid in tts_ids)
                    and not prohibited and not official_eval
                    and deduplication.get('records', {}).get(sid, {}).get('status') == 'pass'):
                allowed.add(sid)
        calibration_rows = [r for r in model_rows if r['sound_asset_id'] in allowed]
    calibration=calibrate_label_gate(calibration_rows,class_config['class_mids'],
                                    calibration_ids=calibration_ids,minimum_positive_recall=0.)
    calibration['source_filter'] = ('licensed_nonspatial_nonevaluation_dedup_pass'
                                    if complete_scan else 'provisional_pending_reference_scan')
    calibration['model_rows_excluded_by_source_screen'] = len(model_rows) - len(calibration_rows)
    holdout=[r for r in calibration_rows if r['sound_asset_id'] not in calibration_ids and r.get('calibration_class_valid') is not False]
    calibration['source_group_disjoint']=True
    calibration['trusted_metadata_rows']=sum(r.get('calibration_class_valid',True) for r in calibration_rows)
    calibration['metadata_unverified_rows_excluded_from_calibration']=sum(r.get('calibration_class_valid') is False for r in calibration_rows)
    calibration['holdout_positive_count']=len(holdout)
    calibration['holdout_positive_recall']=float(np.mean([
        label_score(r,class_config['class_mids'].get(r['sound_class'],[]))>=calibration['threshold'] and
        label_score(r,class_config['class_mids'].get(r['sound_class'],[]))-max([label_score(r,v) for k,v in class_config['class_mids'].items() if k!=r['sound_class'] and v and not set(class_config['class_mids'].get(r['sound_class'],[])).intersection(v)],default=0.)>=calibration['minimum_margin']
        for r in holdout])) if holdout else None
    calibration['calibration_sound_asset_ids']=sorted(calibration_ids.intersection(r['sound_asset_id'] for r in calibration_rows))
    candidates=[{k:v for k,v in r.items() if k!='human_listening_status'} for r in inventory['records']]
    for sound in tts['sounds']:
        r=dict(sound,source_uri='author-tts:'+sound['sound_asset_id'],
               license='project_authored_text_and_formant_synthesis',
               license_candidate_eligible=True,emission_mechanism='unassigned_dry_input',
               content_class='speech',
               compatible_emission_mechanisms=['human_live_formant','loudspeaker_playback'])
        candidates.append(r);originals[r['sound_asset_id']]=r
    records=[]
    for r in candidates:
        sid=r['sound_asset_id'];source=originals.get(sid,r);screens={}
        screens['license']={'status':'pass' if r.get('license_candidate_eligible') else 'blocked',
                            'license':r.get('license'),'original_id':r.get('fsd50k_id') or sid,
                            'source_uri':r.get('source_uri'),'attribution':r.get('attribution'),
                            'reason':'no_permissive_per_clip_or_closed_source_license' if not r.get('license_candidate_eligible') else None}
        meta=provenance_blockers({**source,'official_metadata':r.get('official_metadata',{})}, evaluation_source_ids=evaluation_ids)
        if 'eval_clips' in str(r.get('official_metadata',{}).get('metadata_member','')):
            meta.append('official_FSD50K_evaluation_clip')
        screens['provenance']={'status':'pass' if not meta else 'blocked','reason':','.join(meta) or None,
                               'original_source_group':original_groups.get(sid),'pool_split':source.get('split')}
        proof=deduplication.get('records',{}).get(sid,{})
        screens['deduplication']={'status':proof.get('status','unverified') if deduplication.get('complete_for_listed_roots',False) else 'unverified',
            'reason':'aligned_prohibited_or_evaluation_audio_fingerprint' if proof.get('status')=='blocked' else 'reference_scan_missing_or_incomplete' if proof.get('status')!='pass' else None,
            'nearest':proof.get('nearest'),'reference_file_count':deduplication.get('reference_file_count'),
            'coverage_complete_for_listed_roots':deduplication.get('complete_for_listed_roots',False)}
        scored=scores.get(sid)
        label_class = r['sound_class']
        if label_class == 'any_audioset_class_playback':
            label_class = (source.get('content_class')
                           if source.get('emission_mechanism') == 'loudspeaker_playback' else None)
        target=class_config['class_mids'].get(label_class,[])
        if (r['sound_class'] == 'any_audioset_class_playback' and label_class and not target):
            target = source.get('content_audioset_mids') or []
        r['effective_qa21_answer_class'] = label_class if label_class in QA21_CLASSES else None
        confidence=label_score(scored,target) if scored else None
        bound=scored is not None and Path(r['path']).is_file() and hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==scored['sha256']
        metadata=r.get('official_metadata') or {}
        original_text=' '.join(str(metadata.get(k,'')) for k in ('title','description','tags')).lower()
        groups=class_config.get('specific_metadata_groups',{}).get(label_class,[])
        subtype_ok=all(any(token.lower() in original_text for token in group) for group in groups)
        competitors=[label_score(scored,v) for k,v in class_config['class_mids'].items() if k!=label_class and v and not set(target).intersection(v)] if scored else []
        label_margin=confidence-max(competitors,default=0.) if confidence is not None else None
        label_ok=bound and bool(target) and confidence>=calibration['threshold'] and label_margin>=calibration['minimum_margin'] and subtype_ok
        screens['automatic_label']={'status':'pass' if label_ok else 'blocked',
            'reason':None if label_ok else 'model_receipt_missing_or_audio_hash_changed' if not bound else 'classifier_domain_has_no_specific_target' if not target else 'original_metadata_does_not_confirm_specific_subtype' if not subtype_ok else 'competing_class_or_multiclass_audio' if confidence is not None and confidence>=calibration['threshold'] and label_margin<calibration['minimum_margin'] else 'target_class_below_calibrated_confidence',
            'confidence':confidence,'threshold':calibration['threshold'],
            'label_margin':label_margin,'minimum_margin':calibration['minimum_margin'],
            'audio_sha256_matches_model_receipt':bound,
            'calibration_metadata_source':scored.get('trusted_metadata_source') if scored else None,
            'target_mids':target,'top8':scored.get('top8') if scored else None,
            'checkpoint':model_receipt.get('checkpoint'),'checkpoint_sha256':model_receipt.get('checkpoint_sha256'),
            'calibration_role':'calibration' if sid in calibration_ids else 'holdout',
            'source_subtype_requires_original_metadata':bool(groups),
            'source_subtype_metadata_confirmed':subtype_ok,'source_subtype_required_groups':groups}
        if r.get('license_candidate_eligible'):
            try:
                x,rate=sf.read(r['path'],dtype='float64',always_2d=True)
                activity=measure_activity(x[:,0],rate,sound_class='unknown')
                intervals=activity['source_activity_intervals_samples']
                valid=rate==16000 and x.shape[1]==1 and np.isfinite(x).all() and bool(intervals)
                screens['automatic_activity']={'status':'pass' if valid else 'blocked',
                    'reason':None if valid else 'invalid_mono_or_no_measured_activity',
                    'activity_intervals_samples':intervals,'coordinate':'original_clip_samples',
                    'sample_count':len(x),'sample_rate_hz':rate,'duration_s':len(x)/rate,
                    'measurement':activity,'activity_is_qa_event_count':False}
                derivative=r.get('normalized_research_derivative')
                if derivative:
                    dest=Path(derivative['path'])
                    y,sr=sf.read(dest,always_2d=True)
                    if y.shape!=x.shape or sr!=rate:raise ValueError('normalized copy changed clip clock')
                    error=float(np.max(np.abs(y-x*10**(derivative['applied_gain_db']/20))))
                    if error>1e-6:raise ValueError('normalized copy is not the declared constant gain')
                    loud=_loudness(dest)
                    norm={'status':'pass' if loud['input_tp'] is not None and loud['input_tp']<=-3 else 'blocked',
                          'path':str(dest),'readback_loudness':loud,
                          'applied_gain_db':derivative['applied_gain_db'],'constant_gain_max_error':error,'method':'constant_gain_with_true_peak_cap',
                          'common_lufs_target_achieved':loud['input_i'] is not None and abs(loud['input_i']+23)<=.1,
                          'target_integrated_lufs':-23,'true_peak_ceiling_dbfs':-3,
                          'sample_count_preserved':len(x),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
                else:
                    loud=_loudness(r['path'])
                    if loud['input_i'] is None or loud['input_tp'] is None:raise ValueError('integrated loudness or true peak not measurable')
                    gain=min(-23-loud['input_i'],-3.02-loud['input_tp'])
                    dest=output/'normalized'/r['sound_class']/(sid+'.wav');dest.parent.mkdir(parents=True,exist_ok=True)
                    for attempt in range(3):
                        sf.write(dest,x*10**(gain/20),rate,subtype='FLOAT');post=_loudness(dest)
                        if post['input_tp'] is not None and post['input_tp']<=-3:break
                        gain+=-3.02-post['input_tp']
                    norm={'status':'pass' if post['input_tp'] is not None and post['input_tp']<=-3 else 'blocked',
                          'path':str(dest),'readback_loudness':post,'applied_gain_db':gain,
                          'method':'constant_gain_with_true_peak_cap',
                          'common_lufs_target_achieved':post['input_i'] is not None and abs(post['input_i']+23)<=.1,
                          'target_integrated_lufs':-23,'true_peak_ceiling_dbfs':-3,
                          'sample_count_preserved':len(x),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
                screens['normalization']=norm
            except Exception as e:
                screens.setdefault('automatic_activity',{'status':'blocked','reason':str(e)})
                screens['normalization']={'status':'blocked','reason':str(e)}
        else:
            screens['automatic_activity']={'status':'unverified','reason':'license_or_source_chain_not_admitted'}
            screens['normalization']={'status':'unverified','reason':'license_or_source_chain_not_admitted'}
        r.update(qualify_dry_record(source,screens))
        r['sound_class']=source['sound_class'];r['normalized_path']=screens.get('normalization',{}).get('path')
        records.append(r)
    classes=[]
    for cls in class_config['class_mids']:
        rows=[r for r in records if r['sound_class']==cls]
        classes.append({'sound_class':cls,'records':len(rows),'usable':sum(r['strict_v7_eligible'] for r in rows),
            'automatic_label_pass':sum(r['screens']['automatic_label']['status']=='pass' for r in rows),
            'rejection_reasons':dict(Counter(b for r in rows for b in r['formal_blockers'])),
            'typical_rejections':[{'sound_asset_id':r['sound_asset_id'],'reasons':r['formal_blockers'],
                'model_top8':r['screens']['automatic_label'].get('top8')} for r in rows if not r['strict_v7_eligible']][:3],
            'qa21_domain_member':cls in QA21_CLASSES,
            'starss_mapping':next((r.get('starss_mapping') for r in rows if r.get('starss_mapping')),None)})
    result={'policy_id':POLICY_ID,'classes':classes,'records':records,'usable':sum(r['strict_v7_eligible'] for r in records),
            'covered_classes':sum(c['usable']>0 for c in classes),'model_receipt':model_receipt,
            'label_calibration':calibration,'deduplication_scope':{k:v for k,v in deduplication.items() if k!='records' and k!='matches'},
            'human_listening_or_viewing_used':False,
            'scope_limit':'Admission covers explicitly enumerated prohibited/evaluation reference roots and pool evaluation identities; spectral screening cannot prove absence of inaudible/masked sources.'}
    _write(output/'dry_sound_admission.json',result)
    return result
