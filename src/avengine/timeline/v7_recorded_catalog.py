"""Adapt the QA owner's R4 recorded receipts without altering their decisions."""
from copy import deepcopy
from pathlib import Path
import hashlib

def recorded_catalog_entry(original,source_manifest):
    r=deepcopy(original)
    if r.get('strict_v7_eligible') is not True or r.get('formal_blockers'):raise ValueError('upstream source not admitted')
    if 'screens' not in r:
        if r.get('current_policy_status')!='admitted' or r.get('current_policy_blockers'):raise ValueError('recorded source policy not passed')
        models=r.get('automatic_label',[])
        if isinstance(models,dict):models=[models]
        passed=[x for x in models if x.get('status')=='pass' and x.get('input_sha256')==r.get('sha256')]
        if r.get('is_vctk') and isinstance(r.get('automatic_label'),dict):
            label=r['automatic_label'];proof=label.get('proof',{})
            if (label.get('confidence',-1)>=label.get('threshold',float('inf')) and label.get('margin',-1)>=label.get('minimum_margin',float('inf')) and proof.get('sha256')==r.get('prepared_sha256') and hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r.get('prepared_sha256')):
                passed=[label]
        if 'original_id' not in r:r['original_id']=r.get('source_original_id')
        if not r.get('original_id'):raise ValueError('recorded original ID absent')
        if not passed or r.get('reference_dedup',{}).get('status')!='pass':raise ValueError('recorded model or reference receipt incomplete')
        norm=r.get('normalization',{})
        if norm.get('status')!='pass' or not norm.get('path') or not norm.get('sha256'):raise ValueError('actual normalized recording not admitted')
        if hashlib.sha256(Path(norm['path']).read_bytes()).hexdigest()!=norm['sha256']:raise ValueError('normalized recorded input SHA changed')
        activity=norm.get('measured_activity',{})
        intervals=activity.get('source_activity_intervals_samples')
        if intervals is None:raise ValueError('recorded source activity measurement absent')
        r['screens']={'license':{'status':'pass','license':r['license'],'original_id':r['original_id'],'source_uri':r['source_uri'],'attribution':r.get('attribution'),'upstream_receipt':source_manifest},'provenance':{'status':'pass','upstream_receipt':source_manifest,'policy_status':r['current_policy_status'],'original_waveform':r.get('source_original_waveform')},'deduplication':deepcopy(r['reference_dedup']),'automatic_label':{'status':'pass','method':'upstream_calibrated_BEATs_or_CLAP','model_receipts':models,'passed_models':[x['model'] for x in passed],'upstream_receipt':source_manifest},'automatic_activity':{'status':'pass','activity_intervals_samples':intervals,'coordinate':'normalized_source_samples','sample_count':norm['sample_count'],'sample_rate_hz':norm['sample_rate_hz'],'measurement':activity},'normalization':deepcopy(norm)}
        r['normalized_path']=norm['path'];r['header']={'sample_rate_hz':norm['sample_rate_hz'],'sample_count':norm['sample_count'],'channels':norm['channels']};r['duration_s']=norm['duration_s']
    if 'original_id' not in r:r['original_id']=r['screens']['license']['original_id']
    if not r.get('normalized_path'):r['normalized_path']=r['screens'].get('normalization',{}).get('path')
    r['source_admission_manifest']=source_manifest
    return r
