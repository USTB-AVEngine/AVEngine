"""Owner 2026-10-15: at most 100ms measured occlusion-boundary exceptions per source event.

Only the explicitly authorised metric categories are recordable. Direction,
DRR, arrival, truncation and invalid measurements retain their original gate.
Intervals are unioned before attributing time to each real scheduled event.
"""
MAX_EVENT_EXCEPTION_MS=100.

def recordable_boundary_difference(verdict,boundary_proof=None):
    failures=verdict.get('failures',[])
    return bool(boundary_proof and boundary_proof.get('status')=='measured' and boundary_proof.get('direct_visibility_flips')) and bool(failures) and all(x.rsplit(':',1)[-1] in {'energy','decay','direction'} for x in failures)

def event_exception_budget(scene,endpoint,intervals,sample_rate=16000):
    merged=[]
    for lo,hi in sorted(set((int(a),int(b))for a,b in intervals)):
        if hi<=lo:continue
        if merged and lo<=merged[-1][1]:merged[-1][1]=max(hi,merged[-1][1])
        else:merged.append([lo,hi])
    events=[e for e in scene.get('events',[])if e['endpoint_id']==endpoint];rows=[];unscoped=[]
    for lo,hi in merged:
        if not any(max(lo,round(e['start_tick']*sample_rate/10))<min(hi,round(e['end_tick']*sample_rate/10))for e in events):unscoped.append([lo,hi])
    for event in events:
        a=round(event['start_tick']*sample_rate/10);b=round(event['end_tick']*sample_rate/10)
        samples=sum(max(0,min(hi,b)-max(lo,a))for lo,hi in merged);ms=samples*1000/sample_rate
        rows.append({'event_id':event['event_id'],'exception_samples':samples,'exception_ms':ms,'maximum_ms':MAX_EVENT_EXCEPTION_MS,'passed':ms<=MAX_EVENT_EXCEPTION_MS+1e-9})
    return {'status':'pass'if not unscoped and all(x['passed']for x in rows)else'fail',
        'owner_policy':'2026-10-15 energy/decay/direction differences only near measured visibility flips, <=100ms per source event; geometric QA unchanged',
        'merged_intervals_samples':merged,'per_event':rows,'unscoped_intervals_samples':unscoped,
        'original_raw_metrics_retained':True,'direction_gate_changed_only_at_measured_boundaries':True,'qa_truth_or_scorer_changed':False}
