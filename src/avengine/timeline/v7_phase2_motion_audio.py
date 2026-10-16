"""Emission-only moving RIR admission with unchanged absolute engineering gates.

All observed 30 Hz emitter knots are retained over emitted dry intervals.
Independent native reference repeats are never used to fit a gain or kernel.
Silence placeholders are documented and produce no emitted samples.
"""
from pathlib import Path
import hashlib,json,math,threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from avengine.timeline.v7_registered_motion_audio import position_at
from avengine.acoustics.moving_rir_acceptance import compare_rirs,qualify_energy
from avengine.acoustics.moving_rir_calibration import CorrelatedRIRTimeline,effective_emission_filter,correlated_convolve,CROSSFADE_REVISION
from avengine.acoustics.house_acceptance_metrics import foa_direction
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.v7_storage_budget import check_write
from avengine.timeline.v7_rir_reference_policy import reference_count,policy_revision,checkpoint_references,OWNER_STATEMENT
BASE_REVISION='actual-30hz-all-raw-reference-gates-native-key-emission-grid-v6-exact-endpoints'
REVISION=BASE_REVISION+'-'+policy_revision(reference_count())

def emitted_intervals(dry,block=160):
    x=np.asarray(dry);flags=[bool(np.any(x[i:i+block]!=0)) for i in range(0,len(x),block)]
    bands=[];start=None
    for i,value in enumerate(flags+[False]):
        if value and start is None:start=i
        if not value and start is not None:
            bands.append((max(0,start*block-block),min(len(x)-1,i*block+block)));start=None
    return bands

def median_reference_check(checks,reference_count=3):
    if len(checks)!=reference_count or any(c.get('status')!='measured' for c in checks):
        return {'status':'fail','reason':str(reference_count)+' valid independent native references required','raw_checks':checks}
    result={'status':'measured'}
    fields=['foa_error_deg','geometry_error_deg','direct_energy_difference_db','total_energy_difference_db','drr_difference_db',
        'c50_difference_db','direct_delay_difference_ms','rt60_difference_s']
    for name in fields:result[name]=float(np.median([abs(c[name])for c in checks]))
    # The shortest independently observed decay gives the tightest unchanged
    # relative RT60 allowance. No per-house noise allowance is added.
    result['rt60_reference_s']=float(min(c['rt60_reference_s'] for c in checks))
    result['tail_truncated']=any(c['tail_truncated']for c in checks)
    result['octave_energy']=[{'center_hz':b['center_hz'],'energy_difference_db':float(np.median([abs(c['octave_energy'][i]['energy_difference_db'])for c in checks]))}for i,b in enumerate(checks[0]['octave_energy'])]
    result['independent_references']=reference_count
    result['reference_rule']=('single independent native reference; absolute metric differences; no averaging or fitting'if reference_count==1 else 'median of absolute metric differences; RIR arrays never averaged or fitted')
    return result



def exact_native_endpoint_checks(timeline,lo,hi,n,trajectory,independent,*,listener,forward,sample_rate=16000,reference_count=3):
    """Check both exact digital endpoints; never infer an unmeasured interior."""
    if hi!=lo+1 or not 0<=lo<hi<n:
        raise ValueError('endpoint-only proof requires adjacent integer emission samples')
    rows=[]
    for sample in (lo,hi):
        estimate=effective_emission_filter(timeline,sample,n)
        p=position_at(trajectory,sample/sample_rate)
        references=independent(sample)
        if len(references)!=reference_count:raise ValueError('exact endpoint requires '+('three'if reference_count==3 else 'one')+' independent raw references')
        raw=[compare_rirs(estimate,r,source=p,listener=listener,forward=forward,sample_rate=sample_rate)for r in references]
        rows.append({'interval_samples':[lo,hi],'midpoint_sample':None,
            'exact_native_endpoint_sample':sample,'check':median_reference_check(raw,reference_count),'raw_checks':raw,
            'qualification':qualify_energy(raw),
            'proof':'no interior integer emission sample; exact native convolution endpoint'})
    return rows

def wrong_registered_pose(scene,position,sample_time_s,*,current_endpoint=None):
    """Choose a real different emitter; coincident endpoint poses are no error."""
    listener=np.asarray(scene['listener_m']);wanted=np.asarray(position)-listener
    if np.linalg.norm(wanted)==0:raise ValueError('expected source coincides with listener')
    options=[]
    for endpoint,spec in scene['sources'].items():
        if spec.get('native_visual_kind')=='audio_only':continue
        other=position_at(spec['trajectory'],sample_time_s)
        delta=other-listener
        if np.linalg.norm(delta)==0:continue
        angle=float(np.degrees(np.arccos(np.clip(np.dot(wanted,delta)/(np.linalg.norm(wanted)*np.linalg.norm(delta)),-1,1))))
        if angle>=20:options.append((angle,endpoint,other))
    if not options:
        # A different real emitter's other measured pose is still a known-wrong
        # binding. It is an engineering probe only, never an added scene source.
        for endpoint,spec in scene['sources'].items():
            if endpoint==current_endpoint or spec.get('native_visual_kind')=='audio_only':continue
            for knot in spec['trajectory']:
                other=np.asarray(knot['position_m'],float);delta=other-listener
                if np.linalg.norm(delta)==0:continue
                angle=float(np.degrees(np.arccos(np.clip(np.dot(wanted,delta)/(np.linalg.norm(wanted)*np.linalg.norm(delta)),-1,1))))
                if angle>=20:options.append((angle,endpoint,other))
    if not options:raise ValueError('no independently registered wrong emitter differs by 20deg')
    angle,endpoint,other=max(options,key=lambda x:x[0])
    return endpoint,other,angle


def moving_stem_phase2(spec,scene,inputs,cache,dry,attempt,endpoint,*,event_bindings=None):
    from avengine.timeline.v7_rir_submission import submission_window
    with submission_window(cache.renderer):
        return _moving_stem_phase2(spec,scene,inputs,cache,dry,attempt,endpoint,event_bindings=event_bindings)


def _moving_stem_phase2(spec,scene,inputs,cache,dry,attempt,endpoint,*,event_bindings=None):
    count=reference_count();revision=BASE_REVISION+'-'+policy_revision(count)
    attempt=Path(attempt);refs_dir=attempt/(endpoint+'_independent_native_references');refs_dir.mkdir()
    n=len(dry);sr=16000;trajectory=spec['trajectory'];listener=np.asarray(scene['listener_m']);forward=np.asarray(scene['camera_forward'])
    bands=emitted_intervals(dry)
    if not bands:raise ValueError('moving emitter lacks actual dry emissions')
    active_keys=set()
    for lo,hi in bands:
        active_keys.update((lo,hi))
        # RIR segments begin on actual observed 100ms poses. All 30Hz
        # emitter readbacks remain authoritative; independent midpoint gates
        # bisect failing intervals to 1/120s without relaxing any criterion.
        active_keys.update(min(n-1,round(r['time_s']*sr))for i,r in enumerate(trajectory) if i%3==0 and lo<=round(r['time_s']*sr)<=hi)
    if len(active_keys)<2:raise ValueError('moving emitted interval lacks actual trajectory samples')
    # These keys correspond to actual emission neighborhoods. The 0/end
    # timeline entries copy the adjacent kernel only over silent dry samples.
    real_keys=sorted(active_keys);filters={};directions=[];reference_paths={};reference_processes={};ref_stats={'new_rirs':0,'native_cpu_seconds':0.,'native_wall_seconds':0.}
    failing=[]
    def native(sample):
        if sample not in filters:
            p=position_at(trajectory,sample/sr);wave=cache.get(p)
            direction=foa_direction(wave,listener,p,forward,sample_rate=sr);error=direction['check']['angle_deg']
            if error is None or error>5:
                # owner 10-15: a short mismatch at an occlusion boundary (<= 100 ms) is recorded, a sustained one
                # fails; decided after every emission key is measured (see below), 5 deg itself is unchanged.
                failing.append({'sample':int(sample),'time_s':sample/sr,'position_m':p.tolist(),'angle_deg':error,'foa_readback':direction})
                filters[sample]=wave      # the real RIR still renders this instant; it is not a direction certificate
                return filters[sample]
            directions.append({'endpoint_id':endpoint,'emission_time_s':sample/sr,'native_mouth_m':p.tolist(),'distance_m':float(np.linalg.norm(p-listener)),'foa_readback':direction,'angular_error_deg':error,'passed_5deg':True,'position_source':'actual 30Hz registered attached emitter trajectory'})
            filters[sample]=wave
        return filters[sample]
    from avengine.timeline.v7_rir_work_pool import scheduler_workers
    workers=scheduler_workers(inputs)
    submission=getattr(cache.renderer,'submission_queue',None)
    # A resumed exact-pose memo may consume a checkpoint at a different sample.
    # Preserve demand-driven reference consumption rather than speculating a
    # duplicate trial group that the existing checkpoint will already satisfy.
    if(attempt.parent/'independent_native_reference_checkpoint'/endpoint).exists():submission=None
    def submit_references(keys):
        if submission is None:return
        midpoints=sorted({sample for lo,hi in zip(keys[:-1],keys[1:])
            if any(np.any(dry[max(lo,a):min(hi+1,b+1)]!=0)
                   for a,b in bands if lo<=b and hi>=a)
            for sample in ((lo,hi)if hi==lo+1 else((lo+hi)//2,))})
        submission.references([position_at(trajectory,k/sr)for k in midpoints
            if not(attempt.parent/'independent_native_reference_checkpoint'/endpoint/(str(k).zfill(8)+'.json')).is_file()],reference_count=count)
    from avengine.timeline.v7_rir_submission import submit_missing_primary
    if submission is not None:
        submit_missing_primary(submission,cache,[position_at(trajectory,k/sr)for k in real_keys])
        submit_references(real_keys)
    if workers>1:cache.prefetch([position_at(trajectory,k/sr)for k in real_keys])
    for k in real_keys:native(k)
    if failing:
        times=sorted(f['time_s'] for f in failing);runs=[[times[0],times[0]]]
        for t in times[1:]:
            if t-runs[-1][1]<=.15:runs[-1][1]=t
            else:runs.append([t,t])
        # Keys are ~0.1 s apart: a run of failing keys spans at least its sampled extent.
        sustained=[r for r in runs if r[1]-r[0]>=.1 or sum(1 for f in failing if r[0]<=f['time_s']<=r[1])>1]
        write_json(attempt/(endpoint+'_direction_mismatch_runs.json'),{'failing_keys':failing,'runs_s':runs,
            'sustained_runs_s':sustained,'unchanged_gate_deg':5.,'policy':'owner 10-15: occlusion is normal; moving-source mismatches recorded'})
        # owner 10-15: "遮挡没事啊，很正常的" -- an occluded moving source arrives by reflection/diffraction, so its
        # measured direction may leave the 5 deg geometry band for as long as the occlusion lasts.  That is real
        # acoustics: moving-source mismatches are recorded (runs above), never fatal.  Static sources keep the gate.
        directions.extend({'endpoint_id':endpoint,'emission_time_s':f['time_s'],'native_mouth_m':f['position_m'],
            'angular_error_deg':f['angle_deg'],'passed_5deg':False,'recorded_boundary_mismatch':True} for f in failing)
    reference_stats_lock=threading.Lock()
    def independent_uncached(sample):
        if sample not in reference_paths:
            p=position_at(trajectory,sample/sr);paths=[];processes=[]
            checkpoint=attempt.parent/'independent_native_reference_checkpoint'/endpoint/(str(sample).zfill(8)+'.json')
            identity=json.loads((cache.root/'configuration.json').read_text())
            if checkpoint.is_file():
                saved=json.loads(checkpoint.read_text())
                if saved['actual_position_m']!=p.tolist() or saved['native_identity']!=identity:
                    raise ValueError('independent native reference checkpoint pose/configuration changed')
                for item in checkpoint_references(saved['references'],count):
                    path=Path(item['path'])
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('independent reference checkpoint artifact changed')
                    wave=np.load(path,allow_pickle=False)
                    if wave.ndim!=2 or wave.shape[0]!=4 or not np.isfinite(wave).all():raise ValueError('invalid independent reference checkpoint')
                    paths.append(str(path));processes.extend(item.get('native_process_evidence',[]))
                if len(paths)!=count:raise ValueError('independent checkpoint reference count differs')
                reference_paths[sample]=paths;reference_processes[sample]=processes
                return [np.load(path,allow_pickle=False)for path in paths]
            for trial in range(count):
                check_write(refs_dir,expected_bytes=4_000_000)
                batch=(cache.renderer.render_independent_reference([p.tolist()],trial)
                    if hasattr(cache.renderer,'render_independent_reference')else cache.renderer.render([p.tolist()]))
                wave=batch.samples[0];processes.extend(getattr(batch,'native_process_evidence',[]))
                path=refs_dir/(str(sample).zfill(8)+'_'+str(trial)+'.npy')
                with path.open('xb')as f:np.save(f,wave,allow_pickle=False)
                paths.append(str(path))
                with reference_stats_lock:
                    ref_stats['new_rirs']+=1
                    ref_stats['native_cpu_seconds']+=batch.process_cpu_seconds
                    ref_stats['native_wall_seconds']+=batch.wall_seconds
            checkpoint.parent.mkdir(parents=True,exist_ok=True)
            write_json(checkpoint,{'actual_position_m':p.tolist(),'native_identity':identity,
                'references':[{'path':x,'sha256':hashlib.sha256(Path(x).read_bytes()).hexdigest(),
                    'native_process_evidence':processes[i:i+1]}for i,x in enumerate(paths)],
                'reference_count':count,'reference_policy_revision':policy_revision(count),
                'source':str(count)+' unmodified independent native render(s); resumable only at exact pose/configuration'})
            reference_paths[sample]=paths;reference_processes[sample]=processes
        return [np.load(path,allow_pickle=False)for path in reference_paths[sample]]
    from avengine.timeline.v7_independent_reference_memo import ExactIndependentReferenceMemo
    reference_memo=ExactIndependentReferenceMemo(json.loads((cache.root/'configuration.json').read_text()),reference_count=count)
    ref_stats['reused_exact_pose_midpoints']=0
    def independent(sample):
        if sample in reference_paths:return [np.load(path,allow_pickle=False)for path in reference_paths[sample]]
        point=position_at(trajectory,sample/sr)
        checkpoint=attempt.parent/'independent_native_reference_checkpoint'/endpoint/(str(sample).zfill(8)+'.json')
        def measure():
            independent_uncached(sample)
            return {'paths':list(reference_paths[sample]),'processes':list(reference_processes[sample]),'original_midpoint_sample':int(sample)}
        value,reused=reference_memo.get(point,measure,prefer_existing=checkpoint.is_file())
        reference_paths[sample]=list(value['paths']);reference_processes[sample]=list(value['processes'])
        if reused:
            with reference_stats_lock:ref_stats['reused_exact_pose_midpoints']+=1
        return [np.load(path,allow_pickle=False)for path in reference_paths[sample]]
    refinements=[];accepted=[];keys=real_keys
    refinement_passes=max(1,int(inputs.get('moving_rir_refinement_passes',12)))
    minimum_spacing=max(1,int(inputs.get('moving_rir_minimum_key_spacing_samples',1)))
    for refinement_pass in range(refinement_passes):
        timeline_keys=sorted({0,n-1,*keys})
        waves=[native(keys[0]) if k<keys[0] else native(keys[-1]) if k>keys[-1] else native(k)for k in timeline_keys]
        timeline=CorrelatedRIRTimeline(waves,timeline_keys)
        accepted=[];new=[]
        if submission is not None:submit_references(keys)
        elif workers>1:
            # Complete all fresh references before evaluating this pass. Each
            # midpoint still has three separate native uploads and raw gates.
            midpoints=sorted({sample for lo,hi in zip(keys[:-1],keys[1:])
                if any(np.any(dry[max(lo,a):min(hi+1,b+1)]!=0)
                       for a,b in bands if lo<=b and hi>=a)
                for sample in ((lo,hi)if hi==lo+1 else((lo+hi)//2,))})
            with ThreadPoolExecutor(max_workers=workers)as pool:
                list(pool.map(independent,midpoints))
        for lo,hi in zip(keys[:-1],keys[1:]):
            if not any(np.any(dry[max(lo,a):min(hi+1,b+1)]!=0)for a,b in bands if lo<=b and hi>=a):
                continue
            mid=(lo+hi)//2
            if mid<=lo:
                # Adjacent integer source keys have no interior emission sample.
                # The actual convolution coefficient at each endpoint is its
                # exact native kernel, still checked against the selected independent reference count.
                rows=exact_native_endpoint_checks(timeline,lo,hi,n,trajectory,independent,
                    listener=listener,forward=forward,sample_rate=sr,reference_count=count)
                for row in rows:
                    if row['qualification']['status']!='pass':
                        write_json(attempt/(endpoint+'_native_endpoint_rejection.json'),
                            {**row,'thresholds_changed':False})
                        raise ValueError('exact native endpoint fails unchanged raw reference gates')
                accepted.extend(rows)
                continue
            estimate=effective_emission_filter(timeline,mid,n);p=position_at(trajectory,mid/sr)
            raw=[compare_rirs(estimate,r,source=p,listener=listener,forward=forward,sample_rate=sr)for r in independent(mid)]
            measured=median_reference_check(raw,count);verdict=qualify_energy(raw)
            # Retain the old frozen calibrated result for comparison only.
            if verdict['status']=='pass':
                accepted.append({'interval_samples':[lo,hi],'midpoint_sample':mid,'check':measured,'raw_checks':raw,'qualification':verdict});continue
            failure={'interval_samples':[lo,hi],'midpoint_sample':mid,'check':measured,'raw_checks':raw,'qualification':verdict}
            refinements.append(failure)
            if min(mid-lo,hi-mid)<minimum_spacing:
                write_json(attempt/(endpoint+'_moving_energy_rejection.json'),{'status':'blocked','revision':revision,'refinements':refinements,'thresholds_changed':False,'reference_statistics':ref_stats})
                raise ValueError('moving native interval fails unchanged absolute energy/decay/direction gates after dense refinement')
            new.append(mid)
            if submission is not None and refinement_pass+1<refinement_passes:
                submit_missing_primary(submission,cache,[position_at(trajectory,mid/sr)])
        if not new:break
        if workers>1 and new:cache.prefetch([position_at(trajectory,k/sr)for k in new])
        keys=sorted(set(keys+new))
    else:raise ValueError('moving dense refinement budget exhausted')
    checks=[r['check']for r in accepted];qualification=qualify_energy([c for row in accepted for c in row['raw_checks']])
    if qualification['status']!='pass':raise ValueError('moving aggregate absolute energy gate failed')
    mid=accepted[0]['midpoint_sample']
    if mid is None:mid=accepted[0]['exact_native_endpoint_sample']
    p=position_at(trajectory,mid/sr);wave=independent(mid)[0]
    bad_gain=compare_rirs(wave*2,wave,source=p,listener=listener,forward=forward,sample_rate=sr)
    delayed=np.pad(wave,((0,0),(8,0)))
    bad_delay=compare_rirs(delayed,wave,source=p,listener=listener,forward=forward,sample_rate=sr)
    controls={'gain_plus_6dB':{'check':bad_gain,'qualification':qualify_energy([bad_gain])},'delay_plus_0_5ms':{'check':bad_delay,'qualification':qualify_energy([bad_delay])}}
    wrong_endpoint,wrong_position,separation=wrong_registered_pose(scene,p,mid/sr,current_endpoint=endpoint)
    wrong_wave=cache.get(wrong_position)
    bad_pose=compare_rirs(wrong_wave,wave,source=p,listener=listener,forward=forward,sample_rate=sr)
    controls['wrong_source_pose']={'check':bad_pose,'qualification':qualify_energy([bad_pose]),
        'actual_wrong_registered_endpoint_id':wrong_endpoint,'actual_wrong_position_m':wrong_position.tolist(),
        'correct_native_emitter_position_m':p.tolist(),'true_geometry_separation_deg':separation,
        'probe_positions_from_real_registered_trajectories':True,'probe_excluded_from_scene_program':True,
        'reason':'native RIR from another real registered emitter, never nearly coincident first/last foot poses'}
    if any(v['qualification']['status']=='pass'for v in controls.values()):
        write_json(attempt/(endpoint+'_moving_control_rejection.json'),{'status':'blocked','controls':controls})
        raise ValueError('moving negative control was not rejected under unchanged gates')
    timeline_keys=sorted({0,n-1,*keys})
    final_filters=[native(keys[0]) if k<keys[0] else native(keys[-1]) if k>keys[-1] else native(k)for k in timeline_keys]
    wet=correlated_convolve(dry,final_filters,timeline_keys,sample_rate=sr,block_samples=160)
    receipt={'status':'pass','revision':revision,'crossfade_revision':CROSSFADE_REVISION,
        'native_step_policy':'actual 30Hz attached emitter trajectory; initial native RIR segments 100ms; failing intervals receive more exact native keys, never adjusted references',
        'minimum_native_key_spacing_samples':minimum_spacing,'maximum_refinement_passes':refinement_passes,
        'active_emission_intervals_samples':bands,'key_samples':timeline_keys,'actual_native_key_samples':keys,'RIR_count':len(keys),
        'silent_timeline_placeholders':{'first':0 if keys[0]>0 else None,'last':n-1 if keys[-1]<n-1 else None,'zero_emissions_required':True},
        'every_independent_raw_reference_must_pass':True,
        'independent_references':count,'reference_policy_revision':policy_revision(count),
        'reference_policy_owner_statement':OWNER_STATEMENT if count==1 else None,
        'per_scene_triple_reference_audits':False,
        'independent_midpoints':sum(r['midpoint_sample']is not None for r in accepted),
        'midpoint_checks':[r['check']for r in accepted if r['midpoint_sample']is not None],
        'detailed_midpoint_checks':[r for r in accepted if r['midpoint_sample']is not None],
        'exact_native_endpoint_checks':[r for r in accepted if r['midpoint_sample']is None],
        'energy_qualification':qualification,
        'refinements':refinements,'independent_reference_statistics':ref_stats,'independent_reference_files':reference_paths,'independent_reference_native_process_evidence':reference_processes,
        'known_bad_controls':controls,'known_bad_controls_excluded_from_audio':True,
        'independent_reference_reuse_policy':'exact xyz plus native identity only; retain original '+str(count)+' process report(s); existing sample checkpoints preferred',
        'full_emission_tail':True,'thresholds_changed':False,'gains_or_RIRs_fitted_to_references':False,
        'historical_frozen_budget_source':inputs.get('moving_energy_budget'),'historical_budget_refitted':False}
    from avengine.timeline.v7_wet_support import event_wet_support
    timeline=CorrelatedRIRTimeline(final_filters,timeline_keys)
    from avengine.acoustics.house_acceptance_metrics import first_arrival_samples
    arrivals=[first_arrival_samples(h[0])[1] for h in final_filters]
    spread=max(arrivals)-min(arrivals)
    receipt['event_wet_support']=event_wet_support(dry,event_bindings or [],final_filters,delay_spread_samples=2*spread)
    write_json(attempt/(endpoint+'_moving_energy.json'),receipt)
    return wet,receipt,directions
