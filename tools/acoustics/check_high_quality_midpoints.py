#!/usr/bin/env python3
"""Independent CPU fixed80 diagnostic using higher native ray quality.

The original four arms and thresholds remain unchanged. Only required endpoint
poses are rendered; full-scene production cost is explicitly an estimate.
Alternative waveform policies are diagnostics, not automatic production choices.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import time
import numpy as np
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.continuous_scene import plan_rir_keyframes, position_at, interpolated_filter
from avengine.acoustics.moving_rir_acceptance import compare_rirs
from avengine.acoustics.moving_energy_policies import summarize_checks
from tools.acoustics.check_moving_waveform_controls import shift
from tools.timeline.pilot_continuous_scenes import native_renderer


def read(path):
    return json.loads(Path(path).read_text())


def pose_key(position):
    # Sub-picometre aliases are far below the existing 1e-10m lookup tolerance
    # and the native float32 position precision. Native requests retain floats.
    return ','.join(f'{x:.12f}' for x in position)


def render_chunk(task):
    inputs_path, house_id, category, points, output, job = task
    inputs = read(inputs_path)
    inputs['simulation'].update(diffraction=True, max_diffraction_order=10)
    for name, default in [('direct_ray_count', 500), ('indirect_ray_count', 5000), ('source_ray_count', 500)]:
        inputs['simulation'][name] = inputs['simulation'].get(name, default) * 16
    house = next(h for h in inputs['houses'] if h['house'] == house_id)
    root = Path(output) / house_id / category
    renderer, _ = native_renderer(inputs, house)
    write_json(root / f'native_setup_job{job:03d}.json', renderer.setup_report)
    records = []
    for index, position in points:
        batch = renderer.render([position])
        path = root / f'pose_{index:04d}.npy'
        with path.open('xb') as f:
            np.save(f, batch.samples[0], allow_pickle=False)
        records.append({'index': index, 'position_m': position, 'CPU_seconds': batch.process_cpu_seconds,
            'wall_seconds': batch.wall_seconds, 'native_IR': str(path)})
    result = {'house': house_id, 'category': category, 'job': job, 'pid': os.getpid(),
        'native_calls': len(records), 'records': records,
        'native_CPU_seconds': sum(r['CPU_seconds'] for r in records)}
    write_json(root / f'job_{job:03d}.json', result)
    print(json.dumps({'house': house_id, 'category': category, 'job': job, 'completed_IRs': len(records)}), flush=True)
    return result


def evaluate(plan, output, native_root=None):
    house_id, scene, keys, times, points = plan
    root = output / house_id
    root.mkdir(parents=True, exist_ok=True)
    native = (native_root or output) / house_id
    track = scene['sources']['voice2']['trajectory']
    listener, forward = np.asarray(scene['listener_m']), np.asarray(scene['camera_forward'])
    samples = [k['sample'] for k in keys]
    wave = {category: {key: np.load(native / category / f'pose_{i:04d}.npy', allow_pickle=False)
             for i, key in enumerate(mapping)} for category, mapping in points.items()}
    methods = {k: [] for k in ('linear', 'arrival_aligned_linear', 'nearest', 'arrival_aligned_nearest', 'early_linear_late_equal_power')}
    for t in times:
        sample = round(t * 16000)
        i = min(len(keys) - 2, max(0, int(np.searchsorted(samples, sample, side='right')) - 1))
        u = (sample - samples[i]) / (samples[i+1] - samples[i])
        source = position_at(track, t)
        a, b = [wave['endpoints'][pose_key(keys[j]['position_m'])] for j in (i, i+1)]
        length = max(a.shape[1], b.shape[1])
        a = np.pad(a, ((0, 0), (0, length-a.shape[1])))
        b = np.pad(b, ((0, 0), (0, length-b.shape[1])))
        true = wave['references'][pose_key(source)]
        estimates = {'linear': interpolated_filter([a, b], [samples[i], samples[i+1]], sample),
            'nearest': a if u <= .5 else b}
        aligned = [shift(w, (np.linalg.norm(source-listener) - np.linalg.norm(np.asarray(keys[j]['position_m'])-listener)) / 343 * 16000)
                   for j, w in zip((i, i+1), (a, b), strict=True)]
        coherent = interpolated_filter(aligned, [samples[i], samples[i+1]], sample)
        estimates['arrival_aligned_linear'] = coherent
        estimates['arrival_aligned_nearest'] = aligned[0] if u <= .5 else aligned[1]
        if pose_key(keys[i]['position_m']) == pose_key(keys[i+1]['position_m']):
            hybrid = a  # A stationary source reuses one native RIR.
        else:
            # Coherent early paths use linear weights; the diffuse late field
            # uses an equal-power crossfade. No measured-reference gain fitting,
            # channel normalization or band compensation is performed.
            arrival = np.linalg.norm(source-listener) / 343
            x = np.clip((np.arange(a.shape[1]) / 16000 - arrival - .050) / .030, 0, 1)
            late = .5 - .5*np.cos(np.pi*x)
            diffuse = np.sqrt(1-u)*aligned[0] + np.sqrt(u)*aligned[1]
            hybrid = (1-late)[None, :]*coherent + late[None, :]*diffuse
        estimates['early_linear_late_equal_power'] = hybrid
        for name, estimate in estimates.items():
            methods[name].append({'time_s': t, **compare_rirs(estimate, true, source=source, listener=listener, forward=forward)})
    report = {'house': house_id, 'fixed_times_s': times, 'methods': {
        name: {'qualification': summarize_checks(checks), 'checks': checks} for name, checks in methods.items()},
        'new_profile': 'diffraction=true/order10; native ray counts x16; other physics unchanged',
        'production_policy_changed': False, 'formal_dataset_count': 0}
    write_json(root / 'comparison.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True); p.add_argument('--round3', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--workers', type=int, default=16)
    p.add_argument('--native-root', type=Path, help='Evaluate retained native jobs into a new output; no new simulations')
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not 1 <= a.workers <= 16:
        raise ValueError('CPU-only diagnostic with 1..16 workers')
    a.output.mkdir(parents=True, exist_ok=False)
    inputs = read(a.inputs)
    plans, tasks = [], []
    for house in inputs['houses']:
        hid = house['house']; scene = read(a.round3 / 'native_windows_correct_clock' / (hid + '__observed_scene.json'))
        track = scene['sources']['voice2']['trajectory']; listener = np.asarray(scene['listener_m'])
        keys = plan_rir_keyframes(track, 30, listener, max_seconds=.25, max_distance_m=.5)
        times = [c['time_s'] for c in read(a.round3 / 'native_mouth_energy' / hid / 'result.json')['checks']]
        assert len(times) == 40
        samples = [k['sample'] for k in keys]; points = {'endpoints': {}, 'references': {}}
        for t in times:
            i = min(len(keys)-2, max(0, int(np.searchsorted(samples, round(t*16000), side='right'))-1))
            for j in (i, i+1):
                points['endpoints'].setdefault(pose_key(keys[j]['position_m']), keys[j]['position_m'])
            source = position_at(track, t).tolist(); points['references'].setdefault(pose_key(source), source)
        for category, mapping in points.items():
            (a.output / hid / category).mkdir(parents=True)
            ordered = list(enumerate(mapping.values()))
            for offset in range(0, len(ordered), 6):
                tasks.append((str(a.inputs), hid, category, ordered[offset:offset+6], str(a.output), offset//6))
        write_json(a.output / hid / 'fixed_pose_plan.json', {'times_s': times, 'points': points,
            'independent_reference_contexts': True, 'full_track_keyframes': len(keys), 'only_required_endpoints_rendered': True})
        plans.append((hid, scene, keys, times, points))
    start = time.perf_counter()
    if a.native_root:
        costs = []
        for hid, _, _, times, points in plans:
            saved = read(a.native_root / hid / 'fixed_pose_plan.json')
            assert saved['times_s'] == times and saved['points'] == points
            for category in points:
                for path in sorted((a.native_root / hid / category).glob('job_*.json')):
                    costs.append(read(path))
                for path in (a.native_root / hid / category).glob('native_setup_job*.json'):
                    actual = read(path)['configuration_readback']
                    assert actual['diffraction'] and actual['max_diffraction_order'] == 10
                    assert actual['direct_ray_count'] == actual['source_ray_count'] == 8000
                    assert actual['indirect_ray_count'] == 80000 and actual['thread_count'] == 1
        expected = sum(len(mapping) for plan in plans for mapping in plan[-1].values())
        assert sum(c['native_calls'] for c in costs) == expected
        reports = [evaluate(plan, a.output, a.native_root) for plan in plans]
        write_json(a.output / 'summary.json', {'results': reports, 'native_jobs': costs,
            'native_calls': expected, 'native_CPU_seconds': sum(c['native_CPU_seconds'] for c in costs),
            'evaluation_wall_seconds': time.perf_counter()-start, 'native_root': str(a.native_root),
            'new_native_calls': 0, 'selected_formal_policy': None, 'thresholds_changed': False,
            'original_four_arms_unchanged': True, 'full_scene_CPU_measurement': 'not_run', 'formal_dataset_count': 0})
        return
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        costs = list(pool.map(render_chunk, tasks))
    reports = [evaluate(plan, a.output) for plan in plans]
    write_json(a.output / 'summary.json', {'results': reports, 'native_jobs': costs,
        'native_calls': sum(r['native_calls'] for r in costs),
        'native_CPU_seconds': sum(r['native_CPU_seconds'] for r in costs),
        'wall_seconds': time.perf_counter()-start, 'worker_count': a.workers,
        'selected_formal_policy': None, 'thresholds_changed': False, 'original_four_arms_unchanged': True,
        'full_scene_CPU_measurement': 'not_run; only fixed80 endpoint/reference poses rendered', 'formal_dataset_count': 0})


if __name__ == '__main__':
    main()
