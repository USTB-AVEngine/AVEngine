"""Native moving-emitter convolution with the previously frozen energy budget."""
from pathlib import Path
import hashlib,json
import numpy as np
from avengine.acoustics.moving_rir_acceptance import compare_rirs
from avengine.acoustics.moving_rir_calibration import CorrelatedRIRTimeline,effective_emission_filter,correlated_convolve,qualify_calibrated,CROSSFADE_REVISION
from avengine.acoustics.house_acceptance_metrics import foa_direction
from avengine.timeline.continuous_pilot import write_json


def position_at(trajectory,time_s):
    times=np.array([k['time_s']for k in trajectory]);points=np.array([k['position_m']for k in trajectory])
    return np.array([np.interp(time_s,times,points[:,j])for j in range(3)])


def moving_stem(spec,scene,inputs,cache,dry,attempt,endpoint):
    path=Path(inputs['moving_energy_budget']);budget=json.loads(path.read_text())
    if 'baseline_p95_upper' not in budget['metrics']['direct_energy_difference_db']:raise ValueError('frozen renderer-noise energy calibration missing')
    n=len(dry);trajectory=spec['trajectory'];sample_rate=16000
    key_samples=[0];last=np.asarray(trajectory[0]['position_m']);previous=0.
    for row in trajectory[1:]:
        sample=min(n-1,round(row['time_s']*sample_rate));point=np.asarray(row['position_m'])
        if sample<=key_samples[-1]:continue
        if row['time_s']-previous>=.25-1e-12 or np.linalg.norm(point-last)>=.5:
            key_samples.append(sample);last=point;previous=row['time_s']
    if key_samples[-1]!=n-1:key_samples.append(n-1)
    filters={};directions=[];checks=[];refinements=[]
    listener=np.asarray(scene['listener_m']);forward=np.asarray(scene['camera_forward'])
    def native(sample):
        if sample not in filters:
            p=position_at(trajectory,sample/sample_rate);rir=cache.get(p)
            direction=foa_direction(rir,listener,p,forward,sample_rate=sample_rate)
            error=direction['check']['angle_deg']
            if error is None or error>5.:raise ValueError('moving native RIR exceeds unchanged geometric FOA 5deg criterion')
            directions.append({'endpoint_id':endpoint,'emission_time_s':sample/sample_rate,'native_mouth_m':p.tolist(),
                'distance_m':float(np.linalg.norm(p-listener)),'foa_readback':direction,
                'angular_error_deg':error,'passed_5deg':True,'position_source':'interpolation of actual native 30Hz emitter readbacks'})
            filters[sample]=rir
        return filters[sample]
    for k in key_samples:native(k)
    keys=sorted(key_samples);accepted=[]
    for refinement_pass in range(16):
        timeline=CorrelatedRIRTimeline([native(k)for k in keys],keys)
        accepted=[];new_keys=[]
        for lo,hi in zip(keys[:-1],keys[1:]):
            mid=(lo+hi)//2
            if mid<=lo:raise ValueError('moving RIR interval too short for reference')
            reference=native(mid)
            estimate=effective_emission_filter(timeline,mid,n)
            check=compare_rirs(estimate,reference,source=position_at(trajectory,mid/sample_rate),listener=listener,forward=forward,sample_rate=sample_rate)
            verdict=qualify_calibrated([check],budget)
            if verdict['status']=='pass':accepted.append((lo,hi,mid,check));continue
            refinements.append({'interval_samples':[lo,hi],'midpoint':mid,'check':check,'failures':verdict['failures']})
            if (hi-lo)/sample_rate<=.1+1e-12:
                write_json(attempt/(endpoint+'_moving_energy_rejection.json'),{'status':'blocked','refinements':refinements,'frozen_budget':budget})
                raise ValueError('moving gate fails at minimum native interval; no budget changes')
            new_keys.append(mid)
        if not new_keys:break
        keys=sorted(set(keys+new_keys))
    else:raise ValueError('motion refinement did not converge under unchanged budget')
    final_checks=[c for lo,hi,mid,c in sorted(accepted)];qualification=qualify_calibrated(final_checks,budget)
    if qualification['status']!='pass':raise ValueError('moving interval aggregate qualification failed')
    # Deliberate wrong-pose control, kept entirely outside emitted training audio.
    first,last=0,n-1
    bad=compare_rirs(native(first),native(last),source=position_at(trajectory,last/sample_rate),listener=listener,forward=forward,sample_rate=sample_rate)
    bad_qualification=qualify_calibrated([bad],budget)
    if bad_qualification['status']=='pass':raise ValueError('wrong-pose control was not distinguishable; motion calibration not admitted')
    wet=correlated_convolve(dry,[native(k)for k in keys],keys,sample_rate=sample_rate,block_samples=160)
    receipt={'status':'pass','crossfade_revision':CROSSFADE_REVISION,'native_step_policy':'.25s or .5m; failing intervals bisected, minimum .1s',
        'key_samples':keys,'RIR_count':len(keys),'independent_midpoints':len(final_checks),
        'native_unique_pose_count_including_midpoints':len(filters),'refinements':refinements,
        'midpoint_checks':final_checks,'energy_qualification':qualification,
        'known_bad_wrong_pose':{'check':bad,'qualification':bad_qualification,'excluded_from_training_audio':True},
        'frozen_budget_source':str(path),'frozen_budget_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'calibration_refit':False,'thresholds_changed':False,'full_emission_tail':True}
    write_json(attempt/(endpoint+'_moving_energy.json'),receipt)
    return wet,receipt,directions
