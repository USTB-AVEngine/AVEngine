"""Unattended two-speaker motion construction from a legal room inventory.

No scene, house, speaker, sound or task-output identifier is special-cased.
The original v3 solver owns event conditions; native admission and the original
unified catalog must independently validate the delivered parent.
"""
from collections import defaultdict, Counter
from copy import deepcopy
import hashlib
import math
import numpy as np
from shapely.geometry import LineString, Point, shape

REVISION = 'motion-quartet-template-native-tts-v1'
MOTION_FAMILIES = ('QA-06', 'QA-15', 'QA-16', 'QA-17')

class MotionTemplateUnavailable(ValueError):
    pass

def authored_pairs(manifest):
    groups = defaultdict(list)
    for sound in manifest['sounds']:
        groups[sound['scenario_id']].append(sound)
    pairs = []
    for sounds in groups.values():
        for a in sounds:
            duration_a = float(a['synthesis_readback']['duration_s'])
            if not 2.0 <= duration_a <= 2.4:
                continue
            for b in sounds:
                duration_b = float(b['synthesis_readback']['duration_s'])
                if (a['role_id'] != b['role_id'] and 1.0 <= duration_b <= 1.6
                        and a['voice_reference']['speaker_id'] != b['voice_reference']['speaker_id']):
                    pairs.append((a, b))
    return sorted(pairs, key=lambda pair: tuple(s['sound_asset_id'] for s in pair))

def choose_template_assets(catalog, pair, seed, attempt=0):
    rng = np.random.default_rng(seed)
    pool = [a for a in catalog if a.get('kind') == 'human' and a.get('controlled_shirt')
            and a.get('dialogue_voice_binding_eligible') is not False and a.get('standing_registered_qualified')]
    chosen = []
    for index, sound in enumerate(pair):
        gender = sound['voice_gender']
        eligible = [a for a in pool if a['visual_gender'] == gender
                    and (index != 0 or a.get('walk_allowed'))
                    and a['color'] not in {p['color'] for p in chosen}
                    and a['asset_id'] not in {p['asset_id'] for p in chosen}]
        eligible.sort(key=lambda a: a['asset_id'])
        if not eligible:
            raise MotionTemplateUnavailable('no admitted human for complete TTS voice/appearance/walk role')
        order = rng.permutation(len(eligible))
        chosen.append(eligible[int(order[attempt % len(order)])])
    for _ in range(2):
        eligible = [a for a in pool if a['color'] not in {p['color'] for p in chosen}
                    and a['asset_id'] not in {p['asset_id'] for p in chosen}]
        eligible.sort(key=lambda a: a['asset_id'])
        if not eligible:
            raise MotionTemplateUnavailable('no unique admitted silent same-type bystander')
        chosen.append(eligible[int(rng.integers(len(eligible)))])
    return chosen

def _snap(pathfinder, point):
    snapped = np.asarray(pathfinder.snap_point(np.asarray(point, float)), float)
    return snapped if np.isfinite(snapped).all() and np.linalg.norm((snapped-point)[[0, 2]]) <= .002 else None

def find_template_route(room, pathfinder, seed, attempt=0, *, speed_range=(.8,1.2),quiet_speed=.8):
    from avengine.assets.roster_motion.straight_window_routes import stadium_samples
    from avengine.assets.roster_motion.timed_straight_clock import timed_straight_clock
    rng = np.random.default_rng(seed)
    polygon = shape(room['floor_polygon'])
    free = polygon.convex_hull.buffer(-.34)
    if free.is_empty:
        raise MotionTemplateUnavailable('room has no eroded native walking floor')
    x0, z0, x1, z1 = free.bounds
    centres = [free.representative_point()] + [Point(x, z) for x in np.linspace(x0, x1, 15)
               for z in np.linspace(z0, z1, 15)]
    centres = [p for p in centres if free.covers(p)]
    rng.shuffle(centres)
    speeds = sorted(float(v) for v in rng.uniform(*speed_range, 8))
    floor = float(room['measured_floor_y_m'])
    times = np.arange(901)/30.
    tested = 0
    accepted = 0
    checks = Counter()
    for speed in []:
        # Emission and silent query are straight; slower smooth turns happen
        # only outside .7..7.5 s, repeated every ten seconds.
        distance, velocity, profile = timed_straight_clock(times,speed,straight_start_s=.1,straight_end_s=7.6,turn_speed_m_s=quiet_speed)
        length = profile['straight_length_m']
        radius = profile['turn_radius_m']
        local, tangent, _ = stadium_samples(distance+profile['phase_distance_offset_m'], length, radius)
        for angle in rng.permutation(np.arange(16)*math.pi/8):
            rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            for centre in centres:
                tested += 1
                xz = local@rotation.T + [centre.x, centre.y]
                if not free.covers(LineString(xz)):
                    continue
                checks['floor_polygon'] += 1
                listener_xz = np.array([length/2+radius+.65, 0.])@rotation.T + [centre.x, centre.y]
                listener_point=Point(*listener_xz)
                if not polygon.covers(listener_point) or listener_point.distance(polygon.boundary)<.25:
                    continue
                checks['listener_polygon'] += 1
                listener_floor = np.array([listener_xz[0], floor, listener_xz[1]])
                if _snap(pathfinder, listener_floor) is None:
                    continue
                checks['listener_native'] += 1
                positions = np.column_stack((xz[:, 0], np.full(901, floor), xz[:, 1]))
                if any(_snap(pathfinder, positions[i]) is None for i in [0, 75, 150, 225, 300, 450, 600, 750, 900]):
                    continue
                if any(_snap(pathfinder, p) is None for p in positions):
                    continue
                checks['all_901_native'] += 1
                listener = listener_floor + [0., 1.25, 0.]
                approximate_mouth = positions + [0., 1.5, 0.]
                radial = np.linalg.norm(approximate_mouth-listener, axis=1)
                tangent_xz = tangent@rotation.T
                to_listener = listener[[0, 2]]-positions[:, [0, 2]]
                radial_cosine = np.abs(np.sum(tangent_xz*to_listener, axis=1))/np.linalg.norm(to_listener, axis=1)
                windows = []
                valid = True
                for offset in [0., 10., 20.]:
                    lo, hi = round((offset+.3)*30), round((offset+2.7)*30)
                    values = radial[lo:hi+1]
                    sign = -1 if values[-1] < values[0] else 1
                    if abs(values[-1]-values[0]) < .4 or np.min(sign*np.diff(values)) < -.005 or np.min(radial_cosine[lo:hi+1]) < math.cos(math.radians(30)):
                        valid = False
                        break
                    windows.append({'offset_s': offset, 'branch': 'nearer' if sign < 0 else 'farther'})
                if not valid:
                    continue
                accepted += 1
                if accepted <= attempt:
                    continue
                delta = approximate_mouth[75]-listener
                yaw = math.degrees(math.atan2(delta[0], -delta[2]))
                return {**profile, 'speed_m_s': speed, 'speed_range_m_s': [quiet_speed, speed_range[1]],
                        'centre_xz_m': [centre.x, centre.y], 'heading_rad': float(angle),
                        'straight_length_m': length, 'turn_radius_m': radius,
                        'listener_m': listener.tolist(), 'camera_yaw_deg': yaw,
                        'distance_branches': windows, 'candidate_count': tested,
                        'candidate_native_root_samples': 901,
                        'minimum_emission_radial_cosine': float(min(np.min(radial_cosine[round((o+.3)*30):round((o+2.7)*30)+1]) for o in [0.,10.,20.])),
                        'maximum_radial_angle_deg': 30., 'emission_speed_range_m_s':list(speed_range),
                        'fallback_speed_used': False,
                        'proposal_floor_outline':'convex outline of the same registered room; native NavMesh, original skin/floor and complete collision gates independently required',
                        'geometry_status': 'native NavMesh proposal only; full skin/contact/pixels still required'}
    # Preserve all four conditions when a narrow room cannot contain a loop:
    # one complete voiced/silent straight pass, with a real planted finish.
    # Later windows are still rendered; only the first has this moving target.
    from avengine.assets.roster_motion.timed_straight_clock import single_pass_clock
    for speed in speeds:
        distance,_=single_pass_clock(times,speed,quiet_speed);length=float(distance[-1]*1.12+.02)
        for angle in rng.permutation(np.arange(16)*math.pi/8):
            direction=np.array([math.cos(angle),math.sin(angle)])
            for centre in centres:
                start=np.array([centre.x,centre.y])-direction*length/2
                end=start+direction*length
                if not free.covers(LineString([start,end])):continue
                camera=end+direction*.7;point=Point(*camera)
                if not polygon.covers(point) or point.distance(polygon.boundary)<.25:continue
                listener=np.array([camera[0],floor+1.25,camera[1]])
                if _snap(pathfinder,listener-[0,1.25,0]) is None:continue
                xz=start[None,:]+distance[:,None]*1.12*direction
                positions=np.column_stack((xz[:,0],np.full(901,floor),xz[:,1]))
                if any(_snap(pathfinder,positions[i])is None for i in [0,30,75,120,180,240,450,900]):continue
                if any(_snap(pathfinder,p)is None for p in positions):continue
                if hasattr(pathfinder,'distance_to_closest_obstacle'):
                    selected=np.linspace(0,len(positions)-1,17,dtype=int)
                    if any(pathfinder.distance_to_closest_obstacle(pathfinder.snap_point(positions[i]),1.)<.35 for i in selected):continue
                    if any(pathfinder.distance_to_closest_obstacle(pathfinder.snap_point(p),1.)<.35 for p in positions):continue
                radial=np.linalg.norm(positions+[0,1.5,0]-listener,axis=1)
                lo,hi=30,102
                if radial[lo]-radial[hi]<.4 or np.min(-np.diff(radial[lo:hi+1]))<-.005:continue
                accepted+=1
                if accepted<=attempt:continue
                delta=positions[75]+[0,1.5,0]-listener
                return {'single_pass':True,'speed_m_s':float(speed),'speed_range_m_s':[quiet_speed,speed_range[1]],
                    'emission_speed_range_m_s':list(speed_range),'centre_xz_m':[centre.x,centre.y],
                    'heading_rad':float(angle),'straight_length_m':length,'turn_radius_m':0.,
                    'phase_distance_offset_m':0.,'straight_start_s':0.,'straight_end_s':7.5,
                    'turn_speed_m_s':quiet_speed,'listener_m':listener.tolist(),
                    'camera_yaw_deg':math.degrees(math.atan2(delta[0],-delta[2])),
                    'distance_branches':[{'offset_s':0.,'branch':'nearer'}],
                    'minimum_emission_radial_cosine':1.,'maximum_radial_angle_deg':30.,
                    'candidate_count':tested,'candidate_native_root_samples':901,
                    'complete_step_distance_margin':1.12,'minimum_silent_straight_until_s':7.5,
                    'motion_template_windows_s':[0.], 'all_four_families_retained':True,
                    'geometry_status':'native NavMesh proposal; exact complete-step, full skin/contact/pixel gates still required'}
    raise MotionTemplateUnavailable('bounded native NavMesh straight/turn search exhausted: '+str(tested)+' proposals; stages='+str(dict(checks))+'; planner must choose another legal room')

def motion_quartet_template(base_row, room, catalog, semantic_manifest, source_records,
                            pathfinder, *, attempt=0, contract=None, source_type=None):
    from avengine.timeline.v7_registered_sampling import semantic_record
    from avengine.timeline.v7_motion_during_conditions import prepare_motion_during_row
    import soundfile as sf
    row = deepcopy(base_row)
    seed = int(row['scene_seed'])
    pairs = authored_pairs(semantic_manifest)
    if not pairs:
        raise MotionTemplateUnavailable('no complete reviewed 2..3s / 1..1.6s TTS pair')
    pair = pairs[(seed+attempt) % len(pairs)]
    from avengine.timeline.v7_motion_source_roles import select_motion_role,EMISSION_SPEEDS,QUIET_SPEEDS
    from avengine.timeline.v7_visible_sources import production_audio_record_views
    source_type=source_type or row.get('motion_source_type','human_speech')
    assets = choose_template_assets(catalog, pair, seed, attempt)
    role=select_motion_role(source_type,assets[0],catalog,production_audio_record_views(source_records),contract,seed,attempt)
    assets[0]=role['asset']
    route=find_template_route(room,pathfinder,seed,attempt,speed_range=EMISSION_SPEEDS[source_type],quiet_speed=QUIET_SPEEDS[source_type])
    a, b = [asset['asset_id'] for asset in assets[:2]]
    row.update(schema='v7-visible-registered-pilot-phase2-v1', members=['v0_a0'], trial_only=True,
               asset_ids=[asset['asset_id'] for asset in assets],
               silent_asset_ids=[asset['asset_id'] for asset in assets[2:]],
               source_classes={a: ['speech_playback'], b: ['speech_playback']},
               motion_quartet_template_revision=REVISION,
               semantic_scenario_id=pair[0]['scenario_id'],
               semantic_sound_ids=[sound['sound_asset_id'] for sound in pair],
               semantic_human_asset_ids=[a, b], semantic_start_offsets_s=[.3, 0.],
               proposed_sound_asset_ids={a: pair[0]['sound_asset_id'], b: pair[1]['sound_asset_id']},
               motion_source_start_s={a: .3, b: 0.},
               listener_m=route['listener_m'], camera_yaw_deg=route['camera_yaw_deg'],
               window_count=5, acceptance_window_starts_s=[0., 5., 10., 15., 20.])
    for key in ['motion_during_condition_plan', 'phase2_recorded_speech_schedule_s',
                'phase2_whole_animal_recordings', 'phase2_effect_start_offsets_s']:
        row.pop(key, None)
    row['motion_requests'] = {a: {'action': 'walk', 'speed_m_s': route['speed_m_s'],
        'scene_speed_range_m_s': [.8, 1.4], 'motion_execution_revision': 'species-scene-speeds-native-curved-waypoints-v1',
        'motion_quartet_route': route, 'allow_registered_walk_tts': True,
        'maximum_native_route_attempts': 32,'native_route_candidate_pool':96}}
    row['motion_source_type']=source_type
    row['phase2_source_event_schedule_s']={b:[0.,10.,20.]}
    if source_type=='human_speech':row['phase2_source_event_schedule_s'][a]=[.3,10.3,20.3]
    if source_type!='human_speech':
        row.update(semantic_sound_ids=[pair[1]['sound_asset_id']],semantic_human_asset_ids=[b],semantic_start_offsets_s=[0.])
        record=role['record'];row['source_classes'][a]=[record['sound_class']];row['proposed_sound_asset_ids'][a]=record['sound_asset_id']
        if role['program'] is not None:
            row['phase2_animal_program_samples']={a:role['program']}
            row['phase2_sound_program_schedule_s']={a:[.3,10.3,20.3]}
        else:row['phase2_whole_source_recordings']={a:[.3,10.3,20.3]}
        if assets[0]['kind']=='animal':
            from avengine.timeline.v7_production_inputs import bind_animal_motion
            row['motion_requests'][a]=bind_animal_motion(row['motion_requests'][a],a,contract)
        row['motion_requests'][a]['allow_registered_walk_tts']=False
    row['motion_requests'][a]['scene_speed_range_m_s']=route['speed_range_m_s']
    measured = []
    for sound, asset in (zip(pair,assets) if source_type=='human_speech' else [(pair[1],assets[1])]):
        record = semantic_record(sound, 'F' if asset['visual_gender']=='female' else 'M')
        samples, rate = sf.read(record['normalized_path'], dtype='float64')
        if rate != 16000 or samples.ndim != 1:
            raise MotionTemplateUnavailable('whole TTS must retain the actual 16kHz mono source')
        nonzero = np.flatnonzero(samples != 0.)
        if not len(nonzero):
            raise MotionTemplateUnavailable('whole TTS source is silent')
        record.setdefault('screens', {})['automatic_activity'] = {
            'activity_intervals_samples': [[int(nonzero[0]), int(nonzero[-1]+1)]],
            'method': 'measured exact nonzero waveform envelope; conservative full speech support'}
        measured.append(record)
    if role['record'] is not None:measured.append(role['record'])
    offsets=[0.] if route.get('single_pass') else [0.,10.,20.]
    requests = []
    for qa, branch in [('QA-06', 'moving'), ('QA-16', None), ('QA-17', 'yes')]:
        requests.append({'qa_id': qa, 'branch': branch, 'target_asset_id': a,
            'speed_m_s': route['speed_m_s'], 'speed_range_m_s': route['speed_range_m_s'],
            'minimum_motion_s': 2., 'end_hold_s': 2.5, 'offsets_s': offsets,
            'wet_tail_planning_bound_s': None})
    for item in route['distance_branches']:
        requests.append({'qa_id': 'QA-15', 'branch': item['branch'], 'target_asset_id': a,
            'speed_m_s': route['speed_m_s'], 'speed_range_m_s': route['speed_range_m_s'],
            'minimum_distance_change_m': .5, 'offsets_s': [item['offset_s']]})
    demand=deepcopy(row.get('scene_demand',{}))
    demand.update(focus='motion_quartet',requested_conditions=['motion_quartet'],
                  condition_families={'motion_quartet':list(MOTION_FAMILIES)},
                  constructor=REVISION)
    row['scene_demand']=demand
    row['motion_during_condition_requests'] = requests
    row = prepare_motion_during_row(row, catalog, list(source_records)+measured)
    row['motion_requests'][a]['intervals_s'] = [[0., 9.7]]
    row['motion_requests'][a]['straight_motion_windows_s'] = [[x+.3, x+7.5] for x in offsets]
    row['motion_requests'][a].pop('distance_trend', None)
    row['motion_requests'][a].pop('distance_trend_intervals_s', None)
    if route.get('single_pass'):
        row['motion_requests'][a]['distance_trend']='approaching'
        row['motion_requests'][a]['distance_trend_intervals_s']=row['motion_requests'][a]['sounding_straight_windows_s']
    row['motion_quartet_construction'] = {'revision': REVISION, 'route': route,
        'utterance_durations_s': [s['synthesis_readback']['duration_s'] for s in pair],
        'all_source_content_preserved': True, 'walking_human_has_footsteps': False,
        'constructed_source_type':source_type,'source_program':role.get('program'),
        'actual_wet_support_required_at_delivery': True, 'native_admission': 'pending'}
    return row
