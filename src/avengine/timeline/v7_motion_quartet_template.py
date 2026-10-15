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
from avengine.assets.roster_motion.distance_constraints import matches_radial_window
from shapely.geometry import LineString, Point, shape

REVISION = 'motion-reflection-per-family-v2'
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
            if not 1.2 <= duration_a <= 2.4:
                continue
            for b in sounds:
                duration_b = float(b['synthesis_readback']['duration_s'])
                if (a['role_id'] != b['role_id'] and 1.0 <= duration_b <= 2.4
                        and a['voice_reference']['speaker_id'] != b['voice_reference']['speaker_id']):
                    pairs.append((a, b))
    return sorted(pairs, key=lambda pair: (float(pair[0]['synthesis_readback']['duration_s'])>1.8,)+tuple(s['sound_asset_id'] for s in pair))

def choose_template_assets(catalog, pair, seed, attempt=0):
    rng = np.random.default_rng(seed)
    pool = [a for a in catalog if a.get('kind') == 'human' and a.get('controlled_shirt')
            and a.get('dialogue_voice_binding_eligible') is not False and a.get('standing_registered_qualified') is not False]
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

def find_legacy_template_route(room, pathfinder, seed, attempt=0, *, speed_range=(.82,1.2),quiet_speed=.81,allowed_speed_range=(.8,1.4),minimum_camera_distance_m=1.5):
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
                if not matches_radial_window(radial[lo:hi+1],-1,.5):continue
                accepted+=1
                if accepted<=attempt:continue
                delta=positions[75]+[0,1.5,0]-listener
                return {'single_pass':True,'speed_m_s':float(speed),'speed_range_m_s':list(allowed_speed_range),
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
    from avengine.assets.roster_motion.critical_hairpin import find_critical_hairpin
    compact=find_critical_hairpin(room,pathfinder,seed,speed_range,quiet_speed)
    if compact is not None:
        compact['speed_range_m_s']=list(allowed_speed_range)
        return compact
    raise MotionTemplateUnavailable('bounded native NavMesh straight/turn search exhausted: '+str(tested)+' proposals; stages='+str(dict(checks))+'; planner must choose another legal room')

def find_template_route(room,pathfinder,seed,attempt=0,*,speed_range=(.82,1.2),quiet_speed=.81,allowed_speed_range=(.8,1.4),minimum_camera_distance_m=1.5,species='human',utterance_s=1.5,focus='M1'):
    from avengine.assets.roster_motion.reflect_navmesh import find_reflected_route
    try:return find_reflected_route(room,pathfinder,seed,attempt,species=species,utterance_s=utterance_s,focus=focus,speed_range=speed_range)
    except ValueError as error:raise MotionTemplateUnavailable(str(error)) from error


def motion_quartet_template(base_row, room, catalog, semantic_manifest, source_records,
                            pathfinder, *, attempt=0, contract=None, source_type=None):
    from avengine.timeline.v7_registered_sampling import semantic_record
    from avengine.timeline.v7_motion_during_conditions import prepare_motion_during_row
    import soundfile as sf
    from avengine.timeline.v7_production_inputs import planning_catalog
    catalog=planning_catalog(base_row,catalog,contract)
    if (source_type or base_row.get('motion_source_type'))=='human_footsteps':
        from avengine.timeline.v7_footstep_motion_template import footstep_motion_template
        return footstep_motion_template(base_row,room,catalog,semantic_manifest,attempt=attempt,pathfinder=pathfinder)
    row = deepcopy(base_row)
    seed = int(row['scene_seed'])
    pairs = authored_pairs(semantic_manifest)
    if not pairs:
        raise MotionTemplateUnavailable('no complete reviewed 2..3s / 1..1.6s TTS pair')
    shorter=[p for p in pairs if float(p[0]['synthesis_readback']['duration_s'])<=1.8]
    if shorter:pairs=shorter
    pair = pairs[(seed+attempt) % len(pairs)]
    from avengine.timeline.v7_motion_source_roles import select_motion_role,EMISSION_SPEEDS,QUIET_SPEEDS
    from avengine.timeline.v7_visible_sources import production_audio_record_views
    source_type=source_type or row.get('motion_source_type','human_speech')
    assets = choose_template_assets(catalog, pair, seed, attempt)
    role=select_motion_role(source_type,assets[0],catalog,production_audio_record_views(source_records),contract,seed,attempt)
    assets[0]=role['asset']
    from avengine.timeline.v7_motion_speed import SPEED_RANGES
    speed_kind='human' if source_type.startswith('human') else 'slide' if source_type=='animal_slide' else assets[0]['species_id']
    emitted_duration=float(pair[0]['synthesis_readback']['duration_s'])
    if role['record'] is not None:
        from avengine.timeline.v7_environmental_sources import recording_header
        physical=recording_header(role['record'])
        emitted_duration=float((role.get('program') or {}).get('sample_count',physical['header']['sample_count']))/16000.
    focus=row.get('motion_template_focus','M2' if seed%2 else 'M1')
    # owner 10-15 approved judging distance trends on a 0.5 s smoothed mouth track, so walking animals carry the
    # distance-trend families again (v7_motion_during_conditions.smooth_track).
    route=find_template_route(room,pathfinder,seed,attempt,speed_range=EMISSION_SPEEDS[source_type],quiet_speed=QUIET_SPEEDS[source_type],allowed_speed_range=SPEED_RANGES[speed_kind],species=speed_kind,utterance_s=emitted_duration,focus=focus)
    slot=route['emission_slot'];offset=slot['offset_s'];start=slot['start_s'];local_start=start-offset

    a, b = [asset['asset_id'] for asset in assets[:2]]
    row.update(schema='v7-visible-registered-pilot-phase2-v1', members=['v0_a0'], trial_only=True,
               asset_ids=[asset['asset_id'] for asset in assets],
               silent_asset_ids=[asset['asset_id'] for asset in assets[2:]],
               source_classes={a: ['speech_playback'], b: ['speech_playback']},
               motion_quartet_template_revision=REVISION,
               semantic_scenario_id=pair[0]['scenario_id'],
               semantic_sound_ids=[sound['sound_asset_id'] for sound in pair],
               semantic_human_asset_ids=[a, b], semantic_start_offsets_s=[local_start, 0.],
               proposed_sound_asset_ids={a: pair[0]['sound_asset_id'], b: pair[1]['sound_asset_id']},
               motion_source_start_s={a: local_start, b: 0.},
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
    row['phase2_source_event_schedule_s']={b:[offset]}
    if source_type=='human_speech':row['phase2_source_event_schedule_s'][a]=[start]
    if source_type!='human_speech':
        row.update(semantic_sound_ids=[pair[1]['sound_asset_id']],semantic_human_asset_ids=[b],semantic_start_offsets_s=[0.])
        record=role['record'];row['source_classes'][a]=[record['sound_class']];row['proposed_sound_asset_ids'][a]=record['sound_asset_id']
        if role['program'] is not None:
            row['phase2_animal_program_samples']={a:role['program']}
            row['phase2_sound_program_schedule_s']={a:[start]}
        else:row['phase2_whole_source_recordings']={a:[start]}
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
    offsets=[offset]
    requests=[]
    motion_end=(route.get('short_window_walk') or {}).get('walk_end_s')
    if motion_end is None:motion_end=(slot['leg'][1] if slot.get('leg') is not None else slot['window_frames'][1])/30.
    end_local=min(9.8,float(motion_end)-offset-.1)
    for qa,branch,flag in [('QA-06','moving','qa06'),('QA-15',slot['branch'],'qa15'),('QA-16',None,'qa16'),('QA-17','yes','qa17')]:
        if not slot[flag]:continue
        requests.append({'qa_id':qa,'branch':branch,'target_asset_id':a,
            'speed_m_s':route['speed_m_s'],'speed_range_m_s':route['speed_range_m_s'],
            'minimum_motion_s':1.5,'end_hold_s':10.-end_local,'offsets_s':offsets,
            'reserve_tail_s':.5,'wet_tail_planning_bound_s':None,'minimum_distance_change_m':.5})
    demand=deepcopy(row.get('scene_demand',{}))
    demand.update(focus='motion_quartet',requested_conditions=['motion_quartet'],
                  condition_families={'motion_quartet':[r['qa_id']for r in requests]},
                  constructor=REVISION)
    row['scene_demand']=demand
    row['motion_during_condition_requests'] = requests
    try:
        row=prepare_motion_during_row(row,catalog,list(source_records)+measured)
    except ValueError as error:
        __import__('logging').getLogger(__name__).warning('Original motion solver rejected reflection schedule; reconstructing: %s',error)
        retries=int(base_row.get('_motion_solver_construction_retry',0))
        if retries>=31:raise MotionTemplateUnavailable('original solver rejected 32 automatic reflection schedules: '+str(error)) from error
        retry=deepcopy(base_row);retry['_motion_solver_construction_retry']=retries+1
        retry.setdefault('motion_solver_construction_rejections',[]).append(str(error))
        return motion_quartet_template(retry,room,catalog,semantic_manifest,source_records,pathfinder,attempt=attempt+1,contract=contract,source_type=source_type)

    # Claude 10-15: verify the planned track with the judge's own verifier and keep only the families it already
    # satisfies; a plan that satisfies none is rebuilt (owner: "为什么运动条件不一定成立").
    from avengine.timeline.v7_motion_during_conditions import verify_planned_trajectories
    listener_m=np.asarray(route['listener_m'],float)
    track=np.asarray(route['positions_xz_m'],float);mouth=float(listener_m[1])+.3
    tracks={aid:np.tile(listener_m+[25.,0.,25.+5.*i],(2,1)) for i,aid in enumerate(row['asset_ids'])}
    tracks[a]=np.column_stack((track[:,0],np.full(len(track),mouth),track[:,1]))
    verdict=verify_planned_trajectories(row['motion_during_condition_plan'],tracks,listener_m.tolist())
    passing=[r for r in requests if verdict.get(r['qa_id'])]
    row.setdefault('motion_quartet_planned_verification',{}).update(verdict)
    if not passing:
        raise MotionTemplateUnavailable('planned track satisfies no requested motion family under the judge: '+str(verdict))
    if len(passing)<len(requests):
        row['motion_during_condition_requests']=passing
        row.pop('motion_during_condition_plan',None)
        row=prepare_motion_during_row(row,catalog,list(source_records)+measured)
    row['motion_requests'][a]['motion_during_condition_plan_for_fit']=deepcopy(row['motion_during_condition_plan'])
    # owner 10-15: question quantity is not important; a window delivers the legal questions it has instead of the
    # whole rendered parent failing 'single parent has fewer actual legal questions than requested total' (G107).
    row['acceptance_allow_partial_families']=True
    row['motion_requests'][a]['intervals_s']=[[0.,30.]]
    # The original v3 solver validates selected windows; geometry follows the
    # fully prefiltered reflection track, including its turns outside a leg.
    row['motion_requests'][a].pop('straight_motion_windows_s',None)
    row['motion_requests'][a].pop('distance_trend',None)
    row['motion_requests'][a].pop('distance_trend_intervals_s',None)
    row['motion_quartet_construction'] = {'revision': REVISION, 'route': route,
        'utterance_durations_s': [s['synthesis_readback']['duration_s'] for s in pair],
        'all_source_content_preserved': True, 'walking_human_has_footsteps': False,
        'constructed_source_type':source_type,'source_program':role.get('program'),
        'actual_wet_support_required_at_delivery': True, 'native_admission': 'pending'}
    return row
