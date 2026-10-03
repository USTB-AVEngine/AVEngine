#!/usr/bin/env python3
"""Compare admitted twenty-parent replay geometry, clock, media and existing QA metrics.

Pixel/PCM changes across versions are reported, never used for model selection.
The unchanged native verifier and scorer provide within-quartet quality gates.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))
from avengine.qa.frozen_panel_versions import read,digest,versions_identity


def certificate_groups(root):
    return sorted((root/'scoring_private/certificates').glob('*.json'),key=lambda p:int(p.stem.rsplit('_',1)[1]))


def check_proof(root, proof, paths):
    x=read(proof)
    if x['status']!='pass' or len(x['groups'])!=20 or not all(g['three_layers_verified'] for g in x['groups']):
        raise ValueError('canary requires twenty passing full three-layer proofs')
    if x.get('scoring_script_audit',{}).get('status')!='pass':raise ValueError('canary requires the unchanged scoring seal audit')
    by={g['group']:g for g in x['groups']}
    for path in paths:
        g=by[path.stem]
        if g['certificate_sha256']!=digest(path) or g['inconsistent_episode_relations'] or g['cache_counts']!={'M1':2,'M2':2,'cross_modal':4}:
            raise ValueError('canary certificate/input/cache identity mismatch')
    if len({g['physical_parent_sha256'] for g in x['groups']})!=20:raise ValueError('canary parents not independent')
    return by


def metrics(cert):
    joint=cert['joint_solvability']
    return {'foa_angle_max_deg':max(x['check']['angle_deg'] for x in joint['foa_direction']),
            'c50_min_db':min(x['c50_db'] for x in joint['rir_clarity']),
            'target_competitor_min_db':min(x['target_to_competitor_db'] for x in joint['target_stems']),
            'native_equivalence_max_delta':cert['construction']['native_equivalence']['max_numeric_delta']}


def compare(before, after, oldproof, newproof):
    oldpaths=certificate_groups(before);newpaths=certificate_groups(after)
    if len(oldpaths)!=20 or len(newpaths)!=20:raise ValueError('two houses x ten actual admitted parents required')
    oldrows=check_proof(before,oldproof,oldpaths);newrows=check_proof(after,newproof,newpaths)
    rows=[];quotas=Counter();houses=Counter()
    for oldpath,newpath in zip(oldpaths,newpaths):
        a,b=read(oldpath),read(newpath);fixed=[]
        for field in ['house','room','quota']:
            if a[field]!=b[field]:fixed.append(field)
        if a['construction']['question']!=b['construction']['question']:fixed.append('question')
        for field in ['camera_m','feet_m','mouths_planned_m','basis']:
            if a['construction']['geometry'][field]!=b['construction']['geometry'][field]:fixed.append('geometry.'+field)
        av=a['media']['members'][0]['media']['video'];bv=b['media']['members'][0]['media']['video']
        for field in ['frame_count','resolution_hw','pts_seconds','duration_seconds']:
            if av[field]!=bv[field]:fixed.append('clock.'+field)
        if oldrows[a['group']]['physical_parent_sha256']!=newrows[b['group']]['physical_parent_sha256']:fixed.append('physical_parent_sha256')
        changes=[]
        for ai in range(2):
            am=next(m for m in a['media']['members'] if m['audio_index']==ai and m['video_index']==0)
            bm=next(m for m in b['media']['members'] if m['audio_index']==ai and m['video_index']==0)
            pa=before/'public'/am['request']['foa'];pb=after/'public'/bm['request']['foa']
            aw,ar=sf.read(pa,dtype='float32',always_2d=True);bw,br=sf.read(pb,dtype='float32',always_2d=True)
            if aw.shape!=bw.shape or ar!=br:fixed.append('FOA.length/layout')
            changes.append({'audio_index':ai,'pcm_sha256_before':am['media']['foa']['pcm_sha256'],
                'pcm_sha256_after':bm['media']['foa']['pcm_sha256'],
                'rms_delta':float(np.sqrt(np.mean((aw.astype(float)-bw)**2))) if aw.shape==bw.shape else None})
        quotas[b['quota']['voice']+'/'+b['quota']['side']]+=1;houses[b['house']]+=1
        rows.append({'baseline_group':a['group'],'canary_group':b['group'],'status':'pass' if not fixed else 'fail',
                     'changed_fixed_fields':fixed,'before_metrics':metrics(a),'after_metrics':metrics(b),
                     'video_pixels_changed_across_versions':av['decoded_sha256']!=bv['decoded_sha256'],
                     'native_geometry_hashes_equal_across_versions':a['construction']['appearance_geometry_invariants']==b['construction']['appearance_geometry_invariants'],
                     'audio_across_versions':changes})
    valid=len(houses)==2 and set(houses.values())=={10} and len(quotas)==4 and set(quotas.values())=={5}
    config=read(after/'inputs/config.json') if (after/'inputs/config.json').exists() else read(after/'inputs/shard_config.json')
    version_key=versions_identity(config['production_versions']) if config.get('production_versions') else None
    return {'production_versions_sha256_after':version_key,'status':'pass' if valid and all(r['status']=='pass' for r in rows) else 'fail', 'groups':rows,
            'houses':dict(houses),'quota_counts':dict(quotas),'threshold_changes':False,'answers_opened':False,'models_run':False,
            'cross_version_pcm_delta_is_diagnostic_only':True,
            'required_visual_material_review':'inspect new twenty groups; full native QA gates must also pass',
            'required_new_size_measurement':'compact_frozen_panel.py measure on the newly accepted canary',
            'new_asset_or_runtime_delivery_verified_by_this_comparison':False}


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ['before','after','before-proof','after-proof','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();x=compare(a.before,a.after,a.before_proof,a.after_proof)
    with a.output.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    if x['status']!='pass':sys.exit(1)
if __name__=='__main__':main()
