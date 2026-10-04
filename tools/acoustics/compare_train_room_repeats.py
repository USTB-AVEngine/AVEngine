#!/usr/bin/env python3
"""Report random room reruns against retained native measurements."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'src')]
from tools.acoustics.train_house_followup import read,save
from avengine.acoustics.house_acceptance_metrics import rir_metrics

def decisions(r,c):
    return dict(leak=r.get('max_escape_fraction') is not None and r['max_escape_fraction']<=c['leak_pass_max'],foa=r.get('max_foa_angle_deg') is not None and r['max_foa_angle_deg']<=c['foa_angle_pass_deg'],distance=bool(r.get('distance_attenuation')) and all(x['status']=='valid' and c['distance_exponent_pass'][0]<=x['pressure_exponent']<=c['distance_exponent_pass'][1] for x in r['distance_attenuation']),valid_rirs=sum(x['status']=='valid' for x in r['rirs'])>=c['minimum_rirs'],listeners=r.get('distinct_listener_positions',0)>=c['minimum_listeners'],direct_delay=bool(r['rirs']) and all(x.get('direct_delay_error_ms') is not None and abs(x['direct_delay_error_ms'])<=3 for x in r['rirs']),status=r['status'],reasons=r.get('reasons',[]))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();w=a.output
    selected=read(w/'inputs/repeat_rooms.json');rows=[]
    for r in selected['rooms']:
        h,label=r['house'],r['room_label'];family='hm3d' if h.startswith('hm3d') else 'mp3d'
        m=read(w/'inputs'/(family+'_settings.json'));base=read(w/'houses'/h/'rooms'/label/'result.json');repeat=read(w/'repeats'/h/label/'result.json');cache=read(w/'inputs'/('cache_'+h+'.json'))
        existing={x['probe']:x for x in cache['rirs'] if x['room_label']==label};again={x['probe']:x for x in repeat['rirs']};probes=[]
        for key,x in existing.items():
            if key not in again:continue
            y=again[key];oldpath,oldkey=x['rir_path'].split('#/',1);newpath,newkey=y['rir_path'].split('#/',1)
            with np.load(oldpath) as z:before=z[oldkey]
            with np.load(newpath) as z:after=z[newkey]
            corrected,_=rir_metrics(before,int(m['simulation']['sample_rate_hz']),x['distance_m'],m['simulation']['max_ir_seconds'])
            fields=('direct_peak_s','direct_delay_error_ms','direct_energy','direct_gate_samples','rt60_mid_s','drr_db')
            differences={k:(float(np.max(np.abs(np.asarray(corrected[k],dtype=float)-np.asarray(y[k],dtype=float)))) if corrected.get(k) is not None and y.get(k) is not None else None) for k in fields}
            probes.append(dict(probe=key,retained_waveform=x['rir_path'],independent_waveform=y['rir_path'],status_before=corrected['status'],status_after=y['status'],metrics_before={k:corrected.get(k) for k in fields},metrics_after={k:y.get(k) for k in fields},absolute_differences=differences,same_listener=np.array_equal(x['listener_m'],y['listener_m']),same_source=np.array_equal(x['source_m'],y['source_m']),waveform_shape_before=list(before.shape),waveform_shape_after=list(after.shape),waveforms_exactly_equal=np.array_equal(before,after),maximum_waveform_absolute_difference=float(np.abs(before.astype(float)-after.astype(float)).max()) if before.shape==after.shape else None))
        fields=('max_escape_fraction','max_foa_angle_deg','rt60_mid_median_s','rt60_valid_fraction')
        leak_old={(x['room_label'],x['role']):x['escape_fraction'] for x in cache['leakage'] if x['room_label']==label}
        leak_new={(x['room_label'],x['role']):x['escape_fraction'] for x in repeat['leakage']}
        leakage_exact=all(leak_new.get(k)==v for k,v in leak_old.items())
        same_decisions=decisions(base,m['criteria'])==decisions(repeat,m['criteria'])
        rows.append(dict(house=h,room_label=label,family=family,existing_probe_count=len(probes),leakage_existing_origins_exactly_equal=leakage_exact,acceptance_after_completion_unchanged=same_decisions,decisions_before=decisions(base,m['criteria']),decisions_after=decisions(repeat,m['criteria']),room_absolute_differences={k:abs(base[k]-repeat[k]) if base.get(k) is not None and repeat.get(k) is not None else None for k in fields},retained_probes=probes))
    result=dict(seed=selected['seed'],selected_before_new_results=True,rooms=len(rows),all_ten_acceptance_decisions_unchanged=all(x['acceptance_after_completion_unchanged'] for x in rows),all_existing_leakage_origins_exactly_equal=all(x['leakage_existing_origins_exactly_equal'] for x in rows),matched_retained_probes=sum(x['existing_probe_count'] for x in rows),maximum_direct_delay_difference_ms=max(x['absolute_differences']['direct_delay_error_ms'] for r in rows for x in r['retained_probes'] if x['absolute_differences']['direct_delay_error_ms'] is not None),maximum_rt60_difference_s=max(x['absolute_differences']['rt60_mid_s'] for r in rows for x in r['retained_probes'] if x['absolute_differences']['rt60_mid_s'] is not None),comparison_policy='unchanged thresholds and source/listener geometry; record all numeric differences; independent Monte Carlo late reverberation is not assumed bit-identical',rows=rows)
    save(w/'evidence/random_ten_room_comparison.json',result);print('TEN_ROOM_COMPARISON',json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
