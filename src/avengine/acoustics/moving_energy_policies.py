"""Strict midpoint qualification and bounded adaptive native RIR refinement."""
from __future__ import annotations
import numpy as np
from avengine.acoustics.moving_rir_acceptance import compare_rirs,qualify_energy
from avengine.timeline.continuous_scene import position_at,interpolated_filter,SAMPLE_RATE

def summarize_checks(checks):
    admission=qualify_energy(checks)
    failed={int(s.split(':')[0]) for s in admission['failures'] if s.split(':')[0].isdigit()}
    measured=[c for c in checks if c['status']=='measured']
    fields=('foa_error_deg','geometry_error_deg','direct_energy_difference_db','total_energy_difference_db',
            'drr_difference_db','direct_delay_difference_ms','rt60_difference_s')
    return {**admission,'pass_count':len(checks)-len(failed),'failed_samples':len(failed),
        'max_abs':{k:max((abs(c[k]) for c in measured),default=None) for k in fields},
        'max_octave_energy_abs_db':max((abs(b['energy_difference_db']) for c in measured for b in c['octave_energy']),default=None)}

def midpoint_checks(track,keys,filters,times,reference,listener,forward):
    samples=[k['sample'] for k in keys]
    return [{'time_s':t,**compare_rirs(interpolated_filter(filters,samples,round(t*SAMPLE_RATE)),
        reference(position_at(track,t)),source=position_at(track,t),listener=np.asarray(listener),
        forward=np.asarray(forward))} for t in times]

def adaptive_keyframes(track,keys,get_native,listener,forward,*,minimum_step_s=.1):
    """Bisect failing intervals only when both children meet the lower bound.

    Native observed trajectory corners are never dropped to create an apparent
    refinement. An already finer track therefore legitimately has no eligible
    interval. Existing energy/delay/direction/decay thresholds remain fixed.
    """
    if minimum_step_s<=0:raise ValueError('minimum step must be positive')
    pending=[(a,b) for a,b in zip(keys[:-1],keys[1:],strict=True)]
    result={k['sample']:dict(k) for k in keys};checks=[];subdivisions=0
    while pending:
        a,b=pending.pop();width=(b['sample']-a['sample'])/SAMPLE_RATE
        if width/2 < minimum_step_s-1e-9:continue
        sample=(a['sample']+b['sample'])//2;t=sample/SAMPLE_RATE
        point=position_at(track,t);native=get_native(point)
        check=compare_rirs(interpolated_filter([get_native(a['position_m']),get_native(b['position_m'])],
            [a['sample'],b['sample']],sample),native,source=point,listener=np.asarray(listener),forward=np.asarray(forward))
        checks.append({'time_s':t,**check})
        if qualify_energy([check])['status']=='fail':
            m={'sample':sample,'time_s':t,'position_m':point.tolist()};result[sample]=m
            subdivisions+=1;pending.extend(((a,m),(m,b)))
    output=[result[k] for k in sorted(result)]
    return output,{'minimum_allowed_step_s':minimum_step_s,'added_keyframes':subdivisions,'adaptive_checks':checks,
        'initial_max_step_s':max(np.diff([k['sample'] for k in keys]),default=0)/SAMPLE_RATE,
        'eligible_initial_intervals':sum((b['sample']-a['sample'])/SAMPLE_RATE/2>=minimum_step_s-1e-9
            for a,b in zip(keys[:-1],keys[1:],strict=True)),
        'result_keyframes':len(output),'no_coarsening_of_native_trajectory':True}
