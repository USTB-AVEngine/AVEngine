"""Automatic continuous walking with one footstep source and a still speaker."""
from copy import deepcopy
import math
import numpy as np


def footstep_motion_template(base_row,room,catalog,semantic_manifest,*,attempt=0,pathfinder=None):
    from avengine.timeline.v7_motion_quartet_template import authored_pairs,choose_template_assets,MotionTemplateUnavailable,REVISION
    pairs=authored_pairs(semantic_manifest)
    if not pairs:raise MotionTemplateUnavailable('complete stationary TTS contrast is missing')
    row=deepcopy(base_row);seed=int(row['scene_seed']);pair=pairs[(seed+attempt)%len(pairs)]
    assets=choose_template_assets(catalog,pair,seed,attempt);a,b=[asset['asset_id']for asset in assets[:2]]
    placement=room['placement'];listener=np.asarray(placement['camera_m'],float)
    target=(np.asarray(placement['source_1_m'])+np.asarray(placement['source_2_m']))/2
    forward=target-listener;forward[1]=0.
    if np.linalg.norm(forward)<1e-6:raise MotionTemplateUnavailable('registered room camera has no forward placement ray')
    speed=float(np.random.default_rng(seed+33571).uniform(.82,1.38))
    row.update(schema='v7-visible-registered-pilot-phase2-v1',members=['v0_a0'],trial_only=True,
        asset_ids=[asset['asset_id']for asset in assets],silent_asset_ids=[asset['asset_id']for asset in assets[2:]],
        source_classes={a:['footsteps'],b:['speech_playback']},motion_source_type='human_footsteps',
        semantic_scenario_id=pair[1]['scenario_id'],semantic_sound_ids=[pair[1]['sound_asset_id']],
        semantic_human_asset_ids=[b],semantic_start_offsets_s=[0.],
        proposed_sound_asset_ids={b:pair[1]['sound_asset_id']},phase2_source_event_schedule_s={b:[0.,10.,20.]},
        listener_m=listener.tolist(),camera_yaw_deg=math.degrees(math.atan2(forward[0],-forward[2])),
        window_count=5,acceptance_window_starts_s=[0.,5.,10.,15.,20.],motion_quartet_template_revision=REVISION,
        motion_requests={a:{'action':'walk','speed_m_s':speed,'scene_speed_range_m_s':[.8,1.4],
            'motion_execution_revision':'species-scene-speeds-native-curved-waypoints-v1',
            'intervals_s':[[0.,30.]],'allow_registered_walk_tts':False,
            'maximum_native_route_attempts':32,'native_route_candidate_pool':96}})
    for key in ('motion_during_condition_plan','motion_during_condition_requests','motion_source_start_s'):
        row.pop(key,None)
    if pathfinder is not None:
        from avengine.assets.roster_motion.reflect_navmesh import find_reflected_route
        route=find_reflected_route(room,pathfinder,seed,attempt,species='human',utterance_s=1.5,focus='M1')
        row['motion_requests'][a].update(motion_quartet_route=route,speed_m_s=route['speed_m_s'])
        row.update(listener_m=route['listener_m'],camera_yaw_deg=route['camera_yaw_deg'])
    row['scene_demand']={'focus':'human_motion','requested_conditions':['human_motion'],
        'condition_families':{'human_motion':['QA-06','QA-15']},'constructor':'continuous-footstep-bout-motion-v1'}
    row['motion_quartet_construction']={'revision':'continuous-footstep-bout-motion-v1',
        'constructed_source_type':'human_footsteps','walking_human_has_footsteps':True,
        'speaker_contrast_stationary':True,'all_required_bodies_retained':True,
        'event_clock_authority':'actual native foot contact onsets; adjacent onsets <=0.6s form one bout; every raw step retained',
        'solver_scope':'continuous walking requested; sounding intervals become measurable only after native contact readback',
        'native_admission':'pending','actual_wet_support_required_at_delivery':True}
    return row
