#!/usr/bin/env python3
"""CPU-only frozen-house floor/pose preparation, without RGB or answer files.

Uses only the accepted room scopes and the unchanged pilot geometry thresholds.
Native Habitat PathFinder and the CPU RLR scan-mesh ray context are used once
per house. Final materials, native RGB and speech propagation are not admitted
by this preflight; they require the later production/version comparison.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
from scipy.signal import fftconvolve
from scipy.spatial.transform import Rotation
import soundfile as sf
from shapely.geometry import Point, shape

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
import produce_test_panel_pilot as native
from avengine.timeline.continuous_scene import camera_basis, published_bearing
from avengine.acoustics.house_acceptance_metrics import foa_direction, rir_metrics


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def identity(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def parent_identity(house, geometry):
    return identity({'house': house, 'camera_m': geometry['camera_m'],
                     'feet_m': sorted(geometry['feet_m']), 'basis': geometry['basis']})


def quota(index, house_index):
    cell = (index + 2 * house_index) % 4
    return ('same' if cell < 2 else 'different',
            'opposite' if cell % 2 == 0 else 'same_side')


def progress(root, message):
    line = '\n' + datetime.now(timezone.utc).isoformat() + '：' + message + '\n'
    for path in [root / 'PROGRESS_zh.md', root.parent / 'PROGRESS_zh.md']:
        if path.exists():
            with path.open('a') as f:
                f.write(line)


def floor_room(pf, room):
    # Measurement is the first operation for each new checked room.
    p = room['placement']
    queries = [np.asarray(p['camera_m'], float) - [0, 1.5, 0],
               np.asarray(p['source_1_m'], float) - [0, 1.2, 0],
               np.asarray(p['source_2_m'], float) - [0, 1.2, 0]]
    polygon = shape(room['floor_polygon'])
    measured = []
    for q in queries:
        snap = np.asarray(pf.snap_point(q), float)
        if (not np.isfinite(snap).all() or np.linalg.norm(snap - q) > .10 or
                not polygon.covers(Point(snap[0], snap[2]))):
            raise ValueError('accepted witness has no native floor support')
        measured.append({'query_m': q.tolist(), 'snap_m': snap.tolist(),
                         'offset_m': float(snap[1]), 'snap_error_m': float(np.linalg.norm(snap - q))})
    pool = []
    bounds = polygon.bounds
    for x in np.arange(math.floor(bounds[0] * 4) / 4 + .125, bounds[2], .25):
        for z in np.arange(math.floor(bounds[1] * 4) / 4 + .125, bounds[3], .25):
            if not polygon.covers(Point(x, z)):
                continue
            q = np.array([x, measured[0]['offset_m'], z])
            snap = np.asarray(pf.snap_point(q), float)
            if (not np.isfinite(snap).all() or np.linalg.norm(snap - q) > .10 or
                    not polygon.covers(Point(snap[0], snap[2]))):
                continue
            clearance = float(pf.distance_to_closest_obstacle(snap, .5))
            if clearance >= .15:
                pool.append(snap)
    pool = np.asarray(pool, dtype=float).reshape(-1, 3)
    cameras = [q for q in pool if pf.distance_to_closest_obstacle(q, .5) >= .25]
    if len(pool) < 3 or not cameras:
        raise ValueError('insufficient admitted source/camera floor candidates')
    return {'room': room, 'polygon': polygon, 'pool': pool,
            'cameras': np.asarray(cameras), 'floor_measurements': measured}


def find_pose(pf, ray_context, floor, side, rng, excluded, house):
    failures = Counter()
    for attempt in range(1600):
        feet = floor['pool'][rng.choice(len(floor['pool']), 2, replace=False)].copy()
        listener = floor['cameras'][int(rng.integers(len(floor['cameras'])))].copy()
        camera = listener + [0, 1.55, 0]
        mouths = feet + [0, 1.641311, 0]
        distance = [float(np.linalg.norm(q - camera)) for q in mouths]
        source_gap = float(np.linalg.norm(feet[0] - feet[1]))
        if source_gap < .8 or any(not 1.4 <= d <= 4.2 for d in distance):
            failures['distance'] += 1
            continue
        directions = mouths - camera
        directions[:, 1] = 0
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        gap = math.degrees(math.acos(np.clip(directions[0] @ directions[1], -1, 1)))
        forward = directions[0] + directions[1]
        if np.linalg.norm(forward) < 1e-8:
            failures['opposed_bisector'] += 1
            continue
        forward /= np.linalg.norm(forward)
        if side == 'same_side':
            if not 12 <= gap <= 34:
                failures['same_side_angle'] += 1
                continue
            forward = Rotation.from_rotvec([0, math.radians(gap / 2 + 5), 0]).apply(forward)
        elif gap < 18:
            failures['opposite_angle'] += 1
            continue
        f, right, up = camera_basis(forward)
        bearings = [published_bearing(q, camera, f) for q in mouths]
        if max(abs(x) for x in bearings) > 49:
            failures['fov_margin'] += 1
            continue
        if (side == 'same_side') != (bearings[0] * bearings[1] > 0):
            failures['sides'] += 1
            continue
        rays = []
        for foot in feet:
            for height in [1.10, 1.641311]:
                endpoint = foot + [0, height, 0]
                direction = endpoint - camera
                length = float(np.linalg.norm(direction))
                hit = ray_context.trace_ray_any_hit(camera.tolist(),
                      (direction / length).tolist(), .02, length - .08).hit
                rays.append({'endpoint_m': endpoint.tolist(), 'blocked': bool(hit)})
        if any(r['blocked'] for r in rays):
            failures['body_ray_occlusion'] += 1
            continue
        navigation = []
        for q in [listener, *feet]:
            snap = np.asarray(pf.snap_point(q), float)
            navigation.append({'query_m': q.tolist(), 'snap_m': snap.tolist(),
              'error_m': float(np.linalg.norm(snap - q)),
              'clearance_m': float(pf.distance_to_closest_obstacle(snap, .5))})
        if any(r['error_m'] > .10 for r in navigation):
            failures['native_navigation'] += 1
            continue
        geometry = {'camera_m': camera.tolist(), 'feet_m': feet.tolist(),
                    'mouths_planned_m': mouths.tolist(),
                    'basis': {'forward': f.tolist(), 'right': right.tolist(), 'up': up.tolist()},
                    'bearings_deg': bearings, 'navigation': navigation, 'clear_body_rays': rays,
                    'source_gap_m': source_gap, 'camera_source_distances_m': distance,
                    'checked_room_floor_polygon': floor['room']['floor_polygon'],
                    'checked_floor_pool_count': len(floor['pool']),
                    'floor_offsets_m': feet[:, 1].tolist(), 'floor_measured_first': True,
                    'raw_hm3d_to_common': '[x,y,z] -> [x,z,-y]', 'navmesh_common_up_axis': '+Y',
                    'attempt': attempt, 'failed_candidates': dict(failures)}
        key = parent_identity(house, geometry)
        if key in excluded:
            failures['duplicate_or_pilot_parent'] += 1
            continue
        return geometry, key
    raise ValueError('quota geometry unavailable: ' + str(dict(failures)))


def speech_inputs(config, ordinal, voice):
    speech = read(config['speech'])
    scenarios = speech['dialogues']['scenarios']
    scenario = scenarios[ordinal % len(scenarios)]
    clips = [s for s in speech['sounds'] if s['scenario_id'] == scenario['id']
             and s.get('voice_gender') == 'male']
    for first in clips:
        for second in clips:
            if (first['utterance_index'] == second['utterance_index'] or
                (first['voice_preset'] == second['voice_preset']) != (voice == 'same') or
                    first['duration_s'] + second['duration_s'] > 8.):
                continue
            rows = [first, second]
            for clip in rows:
                if not Path(clip['path']).is_file():
                    raise ValueError('missing admitted dry speech')
            return {'scenario_id': scenario['id'], 'clips': rows,
                    'start_seconds': [.4, 1.0 + first['duration_s']],
                    'full_clips_fit': True, 'target_utterance_index': 0}
    raise ValueError('voice quota lacks admitted full speech clips')


def planned_acoustics(config, renderer, geometry, speech):
    """Apply the existing 10 degree/C50/SNR gates to planned mouths on CPU."""
    camera = geometry['camera_m']
    filters = []
    reports = []
    for mouth in geometry['mouths_planned_m']:
        batch = renderer.render([mouth], listener_position_m=camera,
                                listener_orientation_wxyz=[1., 0., 0., 0.])
        rir = batch.samples[0]
        direction = foa_direction(rir, camera, mouth, geometry['basis']['forward'])
        distance = float(np.linalg.norm(np.asarray(mouth) - camera))
        metrics, _ = rir_metrics(rir, 16000, distance, config['simulation']['max_ir_seconds'])
        if direction['check']['angle_deg'] is None or direction['check']['angle_deg'] > 10:
            raise ValueError('planned FOA target bearing invalid')
        if metrics['c50_db'] < 0:
            raise ValueError('planned speech clarity C50 < 0 dB')
        filters.append(rir)
        reports.append({'rir_pcm_sha256': hashlib.sha256(rir.astype('<f4').tobytes()).hexdigest(),
                        'direction': direction, 'metrics': metrics})
    dry = []
    for clip in speech['clips']:
        wave, rate = sf.read(clip['path'], dtype='float32')
        if rate != 16000 or wave.ndim != 1:
            raise ValueError('admitted dry rate/layout invalid')
        dry.append(wave)
    target = []
    for ai in range(2):
        stems = []
        for di, wave in enumerate(dry):
            position = di if ai == 0 else 1 - di
            convolution = fftconvolve(wave, filters[position][0])
            stem = np.zeros(160000, float)
            lo = round(speech['start_seconds'][di] * 16000)
            hi = min(160000, lo + len(convolution))
            stem[lo:hi] = convolution[:hi-lo]
            stems.append(stem)
        lo = round(speech['start_seconds'][0] * 16000)
        hi = lo + len(dry[0])
        energy = float(np.square(stems[0][lo:hi]).sum())
        noise = float(np.square(stems[1][lo:hi]).sum())
        snr = float(10 * np.log10(max(energy, 1e-30) / max(noise, 1e-30)))
        if energy <= 1e-12 or snr < 12:
            raise ValueError('planned target sentence unclear/overlapping')
        target.append({'routing_position': ai, 'target_to_competitor_db': snr,
                       'target_stem_sha256': hashlib.sha256(stems[0].astype('<f4').tobytes()).hexdigest()})
    if target[0]['target_stem_sha256'] == target[1]['target_stem_sha256']:
        raise ValueError('planned physical swap leaves target stem unchanged')
    return {'status': 'pass_planned_mouth_cpu', 'positions': reports, 'target': target,
            'final_native_mouth_readback': 'not_run', 'binaural_final_tracks': 'not_run',
            'thresholds': {'foa_angle_max_deg': 10, 'c50_min_db': 0, 'target_competitor_min_db': 12}}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only preflight requires empty CUDA_VISIBLE_DEVICES')
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-8:])
    config = read(a.config)
    if len(config['houses']) != 12 or config['groups_per_house'] != 60:
        raise ValueError('formal preflight requires twelve houses x sixty parents')
    test = {h['house'] for h in config['houses']}
    if test & set(config['training_houses'] + config['development_houses']):
        raise ValueError('whole-house train/development overlap')
    rt = native.runtime(config)
    seeds = read(config['frozen_seed_plan'])['groups']
    seed_map = {(row['house'], int(row['group_id'].rsplit('_', 1)[1])): row for row in seeds}
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    excluded = set(config['excluded_physical_parent_sha256'])
    start = time.monotonic()
    for hi, house in enumerate(config['houses']):
        hstart = time.monotonic()
        cpu = time.process_time()
        target = root / house['house'] / 'prepared.json'
        if target.exists():
            if not a.resume:
                raise ValueError('existing preflight requires --resume')
            saved = read(target)
            if saved['config_sha256'] != native.sha(a.config):
                raise ValueError('preflight resume config differs')
            rows.extend(saved['groups'])
            excluded.update(g['physical_parent_sha256'] for g in saved['groups'])
            continue
        accepted = read(house['acceptance_result'])
        if accepted['status'] != 'pass' or accepted.get('known_exposure'):
            raise ValueError('house not admitted/known clean')
        accepted['acoustic_package_manifest'] = house['acoustic_package_manifest']
        pf = rt.habitat_sim.PathFinder()
        if not pf.load_nav_mesh(accepted['input']['navmesh_source']):
            raise ValueError('native navmesh unavailable')
        floors = [floor_room(pf, room) for room in house['rooms']]
        witness = house['rooms'][0]['placement']
        camera = np.asarray(witness['camera_m']) + [0, .05, 0]
        mouth = np.asarray(witness['source_1_m']) + [0, .441311, 0]
        renderer = native.renderer(config, accepted, camera)
        renderer.render([mouth.tolist()])  # CPU scan-mesh warmup, not production audio.
        groups = []
        for j in range(60):
            seed = seed_map[(house['house'], j)]
            voice, side = quota(j, hi)
            failures = []
            found = None
            rejected_poses = set()
            for offset in range(len(floors)):
                fi = (j + 1 + offset) % len(floors)
                rng = np.random.default_rng(seed['independent_scene_seed'] + offset * 100003)
                try:
                    geometry, key = find_pose(pf, renderer.context, floors[fi], side,
                                              rng, excluded | rejected_poses, house['house'])
                    speech = speech_inputs(config, j, voice)
                    acoustic = planned_acoustics(config, renderer, geometry, speech)
                    found = {'index': hi * 60 + j, 'house_index': hi, 'ordinal': j,
                             'house': house['house'], 'room': floors[fi]['room']['room_label'],
                             'frozen_group_id': seed['group_id'], 'scene_seed': seed['independent_scene_seed'],
                             'quota': {'voice': voice, 'side': side}, 'geometry': geometry,
                             'physical_parent_sha256': key, 'speech': speech,
                             'planned_acoustics': acoustic,
                             'earlier_room_failures': failures, 'native_rgb': 'not_run',
                             'production_audio': 'not_run', 'status': 'cpu_preflight_pass',
                             'answers_generated': False, 'model_selection': False}
                    excluded.add(key)
                    break
                except ValueError as exc:
                    failures.append({'room': floors[fi]['room']['room_label'], 'reason': str(exc)})
            if found is None:
                found = {'index': hi * 60 + j, 'house': house['house'], 'ordinal': j,
                         'quota': {'voice': voice, 'side': side}, 'status': 'quota_failed',
                         'reasons': failures, 'answers_generated': False}
            groups.append(found)
        del renderer
        result = {'house': house['house'], 'status': 'pass' if all(g['status'] == 'cpu_preflight_pass' for g in groups) else 'fail',
                  'config_sha256': native.sha(a.config), 'habitat_rgb_renderer_created': False,
                  'gpu_jobs': 'not_run', 'groups': groups,
                  'floor_measurements': {f['room']['room_label']: f['floor_measurements'] for f in floors},
                  'floor_candidate_counts': {f['room']['room_label']: {'source': len(f['pool']), 'camera': len(f['cameras'])} for f in floors},
                  'wall_seconds': time.monotonic() - hstart, 'cpu_seconds': time.process_time() - cpu}
        write(target, result)
        rows.extend(groups)
        count = sum(g['status'] == 'cpu_preflight_pass' for g in groups)
        print(json.dumps({'house': house['house'], 'cpu_preflight_groups': count, 'status': result['status']}, ensure_ascii=False), flush=True)
        progress(root.parent, f"房屋 {house['house']} CPU 阶段完成：{count}/60，地板偏移/合法摆位/净空/身体射线/四配额已查；RGB、完整传播及最终材质未验证。出处 preflight/{house['house']}/prepared.json。")
    report = {'status': 'pass' if all(g['status'] == 'cpu_preflight_pass' for g in rows) else 'fail',
              'group_count': len(rows), 'passed_groups': sum(g['status'] == 'cpu_preflight_pass' for g in rows),
              'house_count': len(config['houses']), 'groups': rows,
              'quota_counts': dict(Counter(g['quota']['voice'] + '/' + g['quota']['side'] for g in rows if g['status'] == 'cpu_preflight_pass')),
              'config_sha256': native.sha(a.config), 'recommendation_commit': config['recommendation_commit'],
              'wall_seconds': time.monotonic() - start, 'native_rgb': 'not_run',
              'formal_generation': 'blocked_pending_three_deliveries', 'answers_generated': False}
    write(root / 'prepared_panel.json', report)
    progress(root.parent, f"12 房 CPU 输入阶段结论：{report['passed_groups']}/720 通过，正式生成仍待三条交付；证据 preflight/prepared_panel.json。")
    if report['status'] != 'pass':
        sys.exit(1)


if __name__ == '__main__':
    main()
