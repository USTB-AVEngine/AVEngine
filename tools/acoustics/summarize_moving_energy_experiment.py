#!/usr/bin/env python3
"""Summarize fixed-threshold native diffraction/refinement controls and plot errors."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from avengine.timeline.continuous_pilot import write_json

def read(path):return json.loads(Path(path).read_text())
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arms',type=Path,required=True);p.add_argument('--diagnostics',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cases=read(a.arms/'summary.json')['results'];diagnostics=read(a.diagnostics/'summary.json')['results'];a.output.mkdir(parents=True,exist_ok=True)
    table=[]
    for label in 'abcd':
        arm=[next(x for x in h['arms'] if x['arm']==label) for h in cases]
        fields=arm[0]['qualification']['max_abs']
        maxima={k:max(x['qualification']['max_abs'][k] for x in arm) for k in fields}
        table.append({'arm':label,'passed':sum(x['qualification']['pass_count'] for x in arm),'samples':80,
            'max_abs':maxima,'max_octave_energy_db':max(x['qualification']['max_octave_energy_abs_db'] for x in arm),
            'house_costs':[{'house':h['house'],'RIRs_per_scene':x['production_RIRs_per_scene'],
                'production_native_CPU_seconds':x['production_native_CPU_seconds'],
                'production_stage_wall_seconds':x.get('production_stage_wall_seconds'),
                'independent_validation_native_stats':x.get('midpoint_reference_stats'),
                'cost_source':x['cost_source'],'adaptive':x['refinement']} for h,x in zip(cases,arm,strict=True)]})
    repeats=[c for h in diagnostics for c in h['repeatability']]
    native_repeat_pass=sum(c['native_repeat_qualification']['pass_count'] for c in repeats)
    report={'status':'blocked','formal_generation_ready':False,'arms':table,
        'selected_formal_policy':None,'reason':'all four requested arms failed unchanged admission',
        'prospective_diffraction_policy':'on/order10 preferred for physical coverage, unqualified',
        'blocked_source_listener_rays':sum(h['blocked_reference_samples'] for h in diagnostics),
        'tested_source_listener_rays':80,'same_pose_native_repeat_passed':native_repeat_pass,'same_pose_native_repeat_tests':8,
        'same_array_positive_controls':{'passed':sum(c['controls']['same_array']['pass_count'] for c in repeats),'tests':4},
        'known_bad_6dB_controls':{'correctly_failed':sum(c['controls']['known_bad_6dB_gain']['failed_samples'] for c in repeats),'tests':4},
        'thresholds_changed':False,'sources':{'arms':str((a.arms/'summary.json').resolve()),'diagnostics':str((a.diagnostics/'summary.json').resolve())}}
    write_json(a.output/'summary.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    data=[(max(abs(c['direct_energy_difference_db']),abs(c['total_energy_difference_db']),*(abs(b['energy_difference_db']) for b in c['octave_energy'])),
           abs(c['drr_difference_db']),abs(c['direct_delay_difference_ms']),c['geometry_error_deg'])
          for h in cases for x in h['arms'] if x['arm']=='a' for c in x['checks']]
    data_b=[(max(abs(c['direct_energy_difference_db']),abs(c['total_energy_difference_db']),*(abs(b['energy_difference_db']) for b in c['octave_energy'])),
           abs(c['drr_difference_db']),abs(c['direct_delay_difference_ms']),c['geometry_error_deg'])
          for h in cases for x in h['arms'] if x['arm']=='b' for c in x['checks']]
    for i,(title,threshold) in enumerate([('Max direct/total/octave energy error (dB)',1),('DRR error (dB)',1),('Arrival error (ms)',.25),('FOA geometry error (deg)',5)]):
        ax=axes.flat[i];ax.plot([x[i] for x in data],label='a = c: diffraction off',alpha=.8);ax.plot([x[i] for x in data_b],label='b = d: diffraction on',alpha=.8)
        ax.axhline(threshold,color='red',linestyle='--',label='unchanged threshold');ax.set_title(title);ax.set_xlabel('Fixed reference index (train 0-39; test 40-79)');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.savefig(a.output/'energy_four_arms.png',dpi=170);fig.savefig(a.output/'energy_four_arms.pdf');plt.close(fig)
    print(json.dumps({'status':report['status'],'passed':{x['arm']:x['passed'] for x in table},'same_pose_native_repeat_passed':native_repeat_pass}))
if __name__=='__main__':main()
