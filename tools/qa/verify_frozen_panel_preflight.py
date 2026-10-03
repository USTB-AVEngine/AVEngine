#!/usr/bin/env python3
"""Independent CPU check of 720 frozen poses, seeds, admitted rooms and acoustic gates."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from shapely.geometry import Point, shape
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from avengine.qa.frozen_panel_versions import parent_hash, read
from avengine.timeline.continuous_scene import published_bearing
import produce_test_panel_pilot as native


def require(value, reason):
    if not value:
        raise ValueError(reason)


def review(config, prepared_root):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only checker')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-8:])
    seeds = {(g['house'], int(g['group_id'].rsplit('_', 1)[1])): g
             for g in read(config['frozen_seed_plan'])['groups']}
    test = {h['house'] for h in config['houses']}
    require(len(test) == 12 and not test & set(config['training_houses'] + config['development_houses']), 'house split overlap')
    runtime = native.runtime(config)
    identities = set(config['excluded_physical_parent_sha256'])
    reports = []
    start = time.monotonic()
    for hi, house in enumerate(config['houses']):
        saved = read(prepared_root / house['house'] / 'prepared.json')
        require(saved['status'] == 'pass' and len(saved['groups']) == 60, 'house does not cover sixty parents')
        source = read(house['acceptance_result'])
        require(source['status'] == 'pass', 'house acceptance changed')
        source['acoustic_package_manifest'] = house['acoustic_package_manifest']
        admitted = {r['room_label']: r for r in source['input']['acoustic_rooms']}
        checked = {r['room_label'] for r in house['rooms']}
        pf = runtime.habitat_sim.PathFinder()
        require(pf.load_nav_mesh(source['input']['navmesh_source']), 'native floor navmesh unavailable')
        witness = saved['groups'][0]['geometry']
        rays = native.renderer(config, source, witness['camera_m'])
        rays.render([witness['mouths_planned_m'][0]])
        quotas = Counter(); rooms = Counter()
        extrema = {'source_gap_min_m': float('inf'), 'source_camera_min_m': float('inf'),
                   'source_camera_max_m': 0., 'source_clearance_min_m': float('inf'),
                   'camera_clearance_min_m': float('inf'), 'foa_direction_max_deg': 0.,
                   'c50_min_db': float('inf'), 'target_competitor_min_db': float('inf')}
        for j, row in enumerate(saved['groups']):
            require(row['ordinal'] == j and row['index'] == hi * 60 + j and row['house'] == house['house'], 'parent index changed')
            seed = seeds[(house['house'], j)]
            require(row['scene_seed'] == seed['independent_scene_seed'] and row['frozen_group_id'] == seed['group_id'], 'frozen seed changed')
            cell = (j + 2 * hi) % 4
            voice, side = ('same' if cell < 2 else 'different', 'opposite' if cell % 2 == 0 else 'same_side')
            require(row['quota'] == {'voice': voice, 'side': side}, 'quota changed')
            require(row['answers_generated'] is False and row['status'] == 'cpu_preflight_pass', 'preflight scope changed')
            require(row['room'] in checked & set(admitted), 'room outside accepted scope')
            geometry = row['geometry']; polygon = shape(admitted[row['room']]['floor_polygon'])
            require(geometry['checked_room_floor_polygon'] == admitted[row['room']]['floor_polygon'], 'floor scope changed')
            key = parent_hash(house['house'], geometry)
            require(key == row['physical_parent_sha256'] and key not in identities, 'duplicate or pilot physical parent')
            identities.add(key)
            camera = np.asarray(geometry['camera_m']); feet = np.asarray(geometry['feet_m'])
            mouths = np.asarray(geometry['mouths_planned_m']); f = np.asarray(geometry['basis']['forward'])
            listener = camera - [0, 1.55, 0]
            for qi, q in enumerate([listener, *feet]):
                snap = np.asarray(pf.snap_point(q))
                clearance = float(pf.distance_to_closest_obstacle(q, .5))
                require(np.isfinite(snap).all() and np.linalg.norm(snap-q) <= .10 and polygon.covers(Point(q[0], q[2])), 'illegal live floor pose')
                require(clearance >= (.25 if qi == 0 else .15), 'insufficient native clearance')
                name = 'camera_clearance_min_m' if qi == 0 else 'source_clearance_min_m'
                extrema[name] = min(extrema[name], clearance)
            gap = float(np.linalg.norm(feet[0]-feet[1])); distances = np.linalg.norm(mouths-camera, axis=1)
            require(gap >= .8 and np.all((distances >= 1.4) & (distances <= 4.2)), 'distance rule failed')
            require(np.allclose(mouths-feet, [0, 1.641311, 0], atol=1e-9, rtol=0), 'planned body emitter height changed')
            angles = [published_bearing(m, camera, f) for m in mouths]
            require(np.allclose(angles, geometry['bearings_deg'], atol=1e-8, rtol=0), 'bearing metadata changed')
            separation = abs(angles[0]-angles[1])
            require(max(abs(x) for x in angles) <= 49, 'FOV margin failed')
            require((side == 'same_side' and angles[0]*angles[1] > 0 and 12 <= separation <= 34) or
                    (side == 'opposite' and angles[0]*angles[1] < 0 and separation >= 18), 'side quota failed')
            for foot in feet:
                for height in [1.1, 1.641311]:
                    endpoint = foot + [0, height, 0]; d = endpoint-camera; length = float(np.linalg.norm(d))
                    require(not rays.context.trace_ray_any_hit(camera.tolist(), (d/length).tolist(), .02, length-.08).hit, 'body ray newly blocked')
            a = row['planned_acoustics']
            require(a['status'] == 'pass_planned_mouth_cpu' and len(a['positions']) == 2, 'planned acoustic proof missing')
            for position in a['positions']:
                angle = position['direction']['check']['angle_deg']; c50 = position['metrics']['c50_db']
                require(angle is not None and angle <= 10 and c50 >= 0, 'existing FOA/C50 gate failed')
                extrema['foa_direction_max_deg'] = max(extrema['foa_direction_max_deg'], angle)
                extrema['c50_min_db'] = min(extrema['c50_min_db'], c50)
            require(a['target'][0]['target_stem_sha256'] != a['target'][1]['target_stem_sha256'], 'target physical evidence unchanged')
            for target in a['target']:
                require(target['target_to_competitor_db'] >= 12, 'existing target audibility gate failed')
                extrema['target_competitor_min_db'] = min(extrema['target_competitor_min_db'], target['target_to_competitor_db'])
            clips = row['speech']['clips']
            require((clips[0]['voice_preset'] == clips[1]['voice_preset']) == (voice == 'same') and
                    clips[0]['utterance_index'] != clips[1]['utterance_index'] and sum(c['duration_s'] for c in clips) <= 8., 'voice/full-clip rule failed')
            extrema['source_gap_min_m'] = min(extrema['source_gap_min_m'], gap)
            extrema['source_camera_min_m'] = min(extrema['source_camera_min_m'], float(distances.min()))
            extrema['source_camera_max_m'] = max(extrema['source_camera_max_m'], float(distances.max()))
            quotas[voice+'/'+side] += 1; rooms[row['room']] += 1
        require(len(quotas) == 4 and set(quotas.values()) == {15}, 'quota not fifteen per cell')
        del rays
        reports.append({'house': house['house'], 'status': 'pass', 'groups': 60,
                        'quotas': dict(quotas), 'rooms': dict(rooms), 'extrema': extrema,
                        'floor_measurements': saved['floor_measurements'], 'floor_candidate_counts': saved['floor_candidate_counts']})
        print(json.dumps({'house': house['house'], 'live_cpu_review': 'pass', 'groups': 60}), flush=True)
    return {'status': 'pass', 'groups': 720, 'houses': reports, 'unique_physical_parents': len(identities)-len(config['excluded_physical_parent_sha256']),
            'live_native_navigation_and_body_rays': True, 'planned_rir_pairs_previously_simulated': 720,
            'new_rir_simulations': 'twelve warmups only; full planned RIR hashes/gates retained from preflight',
            'native_rgb': 'not_run', 'final_materials': 'not_verified', 'answers_generated': False,
            'wall_seconds': time.monotonic()-start}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--prepared-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=review(read(a.config),a.prepared_root)
    with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

if __name__=='__main__':main()
