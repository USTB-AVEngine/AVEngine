#!/usr/bin/env python3
"""CPU repeatability diagnostic at unchanged poses and higher native ray counts.

The four requested arms, their fixed 80 reference times, and every acceptance
threshold stay unchanged. This diagnostic cannot select a production policy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from avengine.acoustics.moving_energy_policies import summarize_checks
from avengine.acoustics.moving_rir_acceptance import compare_rirs
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.continuous_scene import position_at
from tools.timeline.pilot_continuous_scenes import native_renderer


def read(path):
    return json.loads(Path(path).read_text())


def work(task):
    ip, previous, output, house_id = task
    inputs = read(ip)
    house = next(h for h in inputs['houses'] if h['house'] == house_id)
    scene = read(Path(previous) / 'native_windows_correct_clock' / (house_id + '__observed_scene.json'))
    track = scene['sources']['voice2']['trajectory']
    listener, forward = np.asarray(scene['listener_m']), np.asarray(scene['camera_forward'])
    root = Path(output) / house_id
    root.mkdir(parents=True, exist_ok=False)
    results = []
    for factor in (1, 4, 16):
        requested = deepcopy(inputs)
        requested['simulation'].update(diffraction=True, max_diffraction_order=10)
        for name, base in [('direct_ray_count', 500), ('indirect_ray_count', 5000), ('source_ray_count', 500)]:
            requested['simulation'][name] = inputs['simulation'].get(name, base) * factor
        renderer, _ = native_renderer(requested, house)
        write_json(root / f'quality_{factor}_native_setup.json', renderer.setup_report)
        native_cpu = native_wall = 0.
        checks, controls = [], []
        start = time.perf_counter()
        for t in (4.1, 7.7):
            position = position_at(track, t)
            waves = []
            for repeat in range(3):
                batch = renderer.render([position.tolist()])
                native_cpu += batch.process_cpu_seconds
                native_wall += batch.wall_seconds
                wave = batch.samples[0]
                np.save(root / f'quality_{factor}_time{t:.1f}_repeat{repeat}.npy', wave, allow_pickle=False)
                waves.append(wave)
            checks.extend({'time_s': t, **compare_rirs(waves[j], waves[0], source=position,
                listener=listener, forward=forward)} for j in (1, 2))
            positive = compare_rirs(waves[0], waves[0], source=position, listener=listener, forward=forward)
            bad_gain = compare_rirs(waves[0] * 2, waves[0], source=position, listener=listener, forward=forward)
            bad_delay = compare_rirs(np.pad(waves[0], ((0, 0), (8, 0))), waves[0],
                source=position, listener=listener, forward=forward)
            controls.append({'time_s': t, 'same_array': summarize_checks([positive]),
                'known_bad_6dB_gain': summarize_checks([bad_gain]),
                'known_bad_half_ms_delay': summarize_checks([bad_delay])})
        result = {'ray_count_factor': factor, 'requested_simulation': requested['simulation'],
            'native_calls': 6, 'native_CPU_seconds': native_cpu, 'native_wall_seconds': native_wall,
            'stage_wall_excluding_setup_seconds': time.perf_counter() - start,
            'repeat_qualification': summarize_checks(checks), 'checks': checks, 'controls': controls,
            'not_a_fixed80_interpolation_test': True, 'formal_dataset_count': 0}
        write_json(root / f'quality_{factor}_result.json', result)
        results.append(result)
        print(json.dumps({'house': house_id, 'factor': factor, 'passed': result['repeat_qualification']['pass_count'],
            'samples': len(checks), 'native_CPU_seconds': native_cpu}), flush=True)
    report = {'house': house_id, 'quality_levels': results, 'selected_formal_policy': None}
    write_json(root / 'summary.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True); p.add_argument('--round3', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU only')
    a.output.mkdir(parents=True, exist_ok=False)
    inputs = read(a.inputs)
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        rows = list(pool.map(work, [(str(a.inputs), str(a.round3), str(a.output), h['house']) for h in inputs['houses']]))
    write_json(a.output / 'summary.json', {'houses': rows, 'selected_formal_policy': None,
        'thresholds_changed': False, 'fixed80_arms_unchanged': True, 'formal_dataset_count': 0})


if __name__ == '__main__':
    main()
