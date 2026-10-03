"""Independent energy qualification for moving-source RIR interpolation.

No target-domain waveform, material fit, gain compensation or RIR channel
normalization is used. Failed motion intervals require native refinement;
direction alone cannot admit an energy-distorting interpolation.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import signal
from avengine.acoustics.house_acceptance_metrics import rir_metrics,energy_db,foa_direction,OCTAVES_HZ
from avengine.timeline.continuous_scene import angular_distance


def compare_rirs(estimate, reference, *, source, listener, forward, sample_rate=16000):
    distance=float(np.linalg.norm(np.asarray(source)-listener))
    m,_=rir_metrics(estimate,sample_rate,distance,4.)
    r,_=rir_metrics(reference,sample_rate,distance,4.)
    check=foa_direction(estimate,listener,source,forward.copy())['check']
    refcheck=foa_direction(reference,listener,source,forward.copy())['check']
    if m['status']!='valid' or r['status']!='valid':
        return {'status':'fail','reason':'invalid_midband_decay','estimate':m,'reference':r}
    bands=[]
    for frequency in OCTAVES_HZ:
        sos=signal.butter(4,[frequency/math.sqrt(2),frequency*math.sqrt(2)],btype='bandpass',fs=sample_rate,output='sos')
        a=signal.sosfilt(sos,np.asarray(estimate)[0]);b=signal.sosfilt(sos,np.asarray(reference)[0])
        bands.append({'center_hz':frequency,'energy_difference_db':energy_db(np.square(a).sum(),np.square(b).sum())})
    return {'status':'measured','foa_error_deg':angular_distance(check['foa_world_direction'],refcheck['foa_world_direction']),
            'geometry_error_deg':check['angle_deg'],'direct_energy_difference_db':energy_db(m['direct_energy'],r['direct_energy']),
            'total_energy_difference_db':energy_db(m['total_energy'],r['total_energy']),
            'drr_difference_db':m['drr_db']-r['drr_db'],'c50_difference_db':m['c50_db']-r['c50_db'],
            'direct_delay_difference_ms':1000*(m['direct_peak_s']-r['direct_peak_s']),
            'rt60_difference_s':m['rt60_mid_s']-r['rt60_mid_s'],
            'rt60_reference_s':r['rt60_mid_s'],'tail_truncated':bool(m['tail_truncated']),
            'octave_energy':bands}


def qualify_energy(checks, *, max_direction_deg=5., max_energy_db=1., max_drr_db=1., max_delay_ms=.25):
    """New explicit engineering budgets; existing house/RLR gates stay intact."""
    failures=[]
    if not checks:failures.append('no_independent_native_reference')
    for i,c in enumerate(checks):
        if c['status']!='measured':failures.append(f'{i}:invalid_native_metric');continue
        if max(c['foa_error_deg'],c['geometry_error_deg'])>max_direction_deg:failures.append(f'{i}:direction')
        if max(abs(c['direct_energy_difference_db']),abs(c['total_energy_difference_db']),
               *(abs(b['energy_difference_db']) for b in c['octave_energy']))>max_energy_db:failures.append(f'{i}:energy')
        if abs(c['drr_difference_db'])>max_drr_db:failures.append(f'{i}:DRR')
        if abs(c['direct_delay_difference_ms'])>max_delay_ms:failures.append(f'{i}:arrival_delay')
        if c['tail_truncated']:failures.append(f'{i}:tail_truncation')
        if abs(c['rt60_difference_s'])>max(.05,.1*c['rt60_reference_s']):failures.append(f'{i}:decay')
    return {'status':'fail' if failures else 'pass','sample_count':len(checks),'failures':failures,
            'criteria':{'FOA_and_geometry_max_deg':max_direction_deg,'direct_total_octave_energy_max_abs_db':max_energy_db,
                        'DRR_max_abs_db':max_drr_db,'arrival_delay_max_abs_ms':max_delay_ms,
                        'RT60_max_abs_s':'max(0.05,0.1*reference)','tail_truncated_allowed':False}}
