"""Consume independently qualified animal RIRs in the engine's real stem renderer."""
import json
import sqlite3
from pathlib import Path

import numpy as np

from avengine.acoustics.house_acceptance_metrics import foa_direction
from avengine.acoustics.moving_rir_acceptance import compare_rirs, qualify_energy
from avengine.acoustics.moving_rir_calibration import effective_emission_filter
from avengine.assets.animal_motion.rir_crossfade import (
    FrequencyCoherentTimeline, frequency_correlated_convolve, REVISION,
)
from avengine.assets.animal_motion.service import completed_checks


def require_same_acoustic_world(actual, admitted):
    """Reuse the existing compiled-package and runtime identities, not a new hash."""
    for key in (
        'package_id', 'existing_package_content_sha256', 'package_manifest',
        'listener_position_m', 'listener_orientation_wxyz', 'source_radius_m',
        'listener_radius_m', 'runtime_identity',
    ):
        if key not in actual or actual[key] != admitted.get(key):
            raise ValueError('animal_audio_world_identity_differs:' + key)
    ray_keys = {'direct_ray_count', 'indirect_ray_count', 'source_ray_count'}
    a, b = actual['simulation'], admitted['simulation']
    if {k:v for k,v in a.items() if k not in ray_keys} != {k:v for k,v in b.items() if k not in ray_keys}:
        raise ValueError('animal_audio_physical_simulation_differs')
    if any(b[k] < a[k] for k in ray_keys):
        raise ValueError('animal_audio_admitted_sampling_below_requested_sampling')


def render_qualified_stem(spec, scene, inputs, cache, dry, attempt, endpoint):
    from avengine.timeline.v7_registered_motion_audio import position_at

    binding = spec['animal_motion_audio']
    native_path, audio_path = Path(binding['native']), Path(binding['audio'])
    native, audio = json.loads(native_path.read_text()), json.loads(audio_path.read_text())
    mode = spec['motion_metadata']['motion_mode']
    if mode not in ('walk_animation', 'slide_no_animation'):
        raise ValueError('unadmitted_animal_motion_mode')
    if spec['asset_id'] != native['asset_id']:
        raise ValueError('animal_audio_native_asset_identity_differs')
    if completed_checks(native, audio, mode)['status'] != 'pass':
        raise ValueError('animal_audio_native_motion_not_completely_admitted')
    expected = [{'time_s':f['time_s'], 'position_m':f['mouth_world_m']} for f in native['frames']]
    if spec['trajectory'] != expected or scene['listener_m'] != native['listener_m']:
        raise ValueError('animal_audio_actual_native_trajectory_or_listener_differs')
    if audio['crossfade_revision'] != REVISION:
        raise ValueError('animal_audio_crossfade_revision_differs')
    directory = audio_path.parent
    admitted_identity = json.loads((directory/'cache/configuration.json').read_text())
    current_identity = json.loads((Path(cache.root)/'configuration.json').read_text())
    require_same_acoustic_world(current_identity, admitted_identity)
    if admitted_identity.get('context_policy') != 'fresh_process_per_RIR_deterministic_native_replay':
        raise ValueError('animal_audio_independent_native_processes_missing')

    sr, n = 16000, len(dry)
    if round(expected[-1]['time_s']*sr)+1 != n:
        raise ValueError('animal_audio_dry_clock_differs_from_qualified_native_clock')
    segments = audio['segments']
    keys = sorted(set(round(s[k]*sr) for s in segments for k in ('start_s', 'end_s')))
    if not keys or keys[0] != 0 or keys[-1] != n-1:
        raise ValueError('animal_audio_interval_coverage_incomplete')
    if [(round(s['start_s']*sr), round(s['end_s']*sr)) for s in segments] != list(zip(keys[:-1],keys[1:])):
        raise ValueError('animal_audio_interval_gap_or_overlap')
    listener = np.asarray(scene['listener_m'], float)
    forward = np.asarray(scene['camera_forward'], float)
    forward[1] = 0
    forward /= np.linalg.norm(forward)
    filters, directions = [], []
    with sqlite3.connect((directory/'cache/positions.sqlite3').as_uri()+'?mode=ro', uri=True) as db:
        for key in keys:
            point = position_at(expected,key/sr)
            row = db.execute('SELECT id,completed FROM rirs WHERE pose=?',
                             (json.dumps(point.tolist(),separators=(',',':')),)).fetchone()
            if row is None or not row[1]:
                raise ValueError('animal_audio_native_endpoint_missing')
            wave = np.load(directory/'cache'/f'rir_{row[0]:06d}.npy',allow_pickle=False)
            measured = foa_direction(wave,listener,point,forward,sample_rate=sr)
            angle = measured['check']['angle_deg']
            if angle is None or angle > 5:
                raise ValueError('animal_audio_native_endpoint_direction_failed')
            filters.append(wave)
            directions.append({'endpoint_id':endpoint,'emission_time_s':key/sr,
                'native_mouth_m':point.tolist(),'foa_readback':measured,
                'angular_error_deg':angle,'passed_5deg':True})
    timeline = FrequencyCoherentTimeline(filters,keys)
    checks = []
    for segment, lo, hi in zip(segments,keys[:-1],keys[1:]):
        mid = (lo+hi)//2
        source = position_at(expected,mid/sr)
        estimate = effective_emission_filter(timeline,mid,n)
        for repetition in range(3):
            reference = np.load(directory/f'reference_{mid:06d}_{repetition}.npy',allow_pickle=False)
            checks.append(compare_rirs(estimate,reference,source=source,listener=listener,
                                      forward=forward,sample_rate=sr))
        points = np.array([position_at(expected,t) for t in
            [lo/sr,hi/sr,*[f['time_s'] for f in native['frames'] if lo/sr<f['time_s']<hi/sr]]])-listener
        points /= np.linalg.norm(points,axis=1,keepdims=True)
        sweep = float(np.degrees(np.arccos(np.clip((points@points.T).min(),-1,1))))
        if sweep > 5:
            raise ValueError('animal_audio_segment_direction_sweep_failed')
    qualification = qualify_energy(checks)
    negative = compare_rirs(filters[0]*2,filters[0],source=position_at(expected,0),
                           listener=listener,forward=forward,sample_rate=sr)
    control = qualify_energy([negative])
    if qualification['status'] != 'pass' or control['status'] != 'fail':
        raise ValueError('animal_audio_raw_reference_or_6dB_control_failed')
    wet = frequency_correlated_convolve(dry,filters,keys,sample_rate=sr,block_samples=160)
    result = {'status':'pass','crossfade_revision':REVISION,'RIR_count':len(keys),
        'key_samples':keys,'native_motion':str(native_path),'audio_admission':str(audio_path),
        'energy_qualification':qualification,'independent_raw_reference_count':len(checks),
        'amplitude_6dB_negative_control':control,'native_configuration':admitted_identity,
        'full_emission_tail':True,'reference_waveforms_modified':False,
        'thresholds_changed':False,'motion_metadata':spec['motion_metadata']}
    target = Path(attempt)/(endpoint+'_moving_energy.json')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as output:
        json.dump(result,output,ensure_ascii=False,indent=2)
    return wet,result,directions
