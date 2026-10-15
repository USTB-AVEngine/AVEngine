"""Typed motion witnesses for the two-speaker native-NavMesh family adapter.

The common cached-map certificate remains distinct: a cached map rejection is
not relabelled as a native pass. The stock native floor/skin stage still owns
physical admission of every walking body.
"""
from dataclasses import asdict
import numpy as np
from avengine.construction.joint.types import ActorPlan,BodySpec,CameraPose,Certificate,ScenePlan,SoundEvent
from avengine.construction.joint.certification import motion_witness_checks


def scene_plan_for_row(row,room):
    camera=CameraPose(tuple(row['listener_m']),float(row['camera_yaw_deg']),width=1280,height=720)
    actors={};emitters={}
    for aid,request in row['motion_requests'].items():
        route=request['motion_quartet_route'];xz=np.asarray(route['positions_xz_m'],float)
        roots=np.column_stack((xz[:,0],np.full(len(xz),float(room['measured_floor_y_m'])),xz[:,1]))
        window=route['short_window_walk']
        actors[aid]=ActorPlan(aid,roots,BodySpec.for_species('human'),np.asarray(route['travel_yaws_rad']),
            metadata={'walk_intervals_s':[[window['walk_start_s'],window['walk_end_s']]],'ramp_s':window.get('ramp_s',.3),
                      'navigation_authority':'actual registered native NavMesh; cached map preference may be bypassed'})
        emitters[aid]=np.column_stack((xz[:,0],np.full(len(xz),camera.position_m[1]+.3),xz[:,1]))
    control=row.get('motion_wrong_emitter_control')
    if control:
        aid=control['asset_id'];roots=np.tile(np.asarray(control['root_m'],float),(901,1))
        actors[aid]=ActorPlan(aid,roots,BodySpec.for_species('human'),metadata={'silent_negative_control':True})
        emitters[aid]=roots
    sounds=[]
    durations=dict(zip(row['semantic_human_asset_ids'],row['motion_quartet_construction']['utterance_durations_s']))
    for aid,starts in row['phase2_source_event_schedule_s'].items():
        for start in starts:
            end=float(start)+float(durations[aid])
            sounds.append(SoundEvent(aid,row['proposed_sound_asset_ids'][aid],'speech_playback',float(start),end,end+.5))
    return ScenePlan(camera,actors,tuple(sounds),metadata={'motion_condition_plan':row['motion_during_condition_plan'],
        'planned_emitter_positions_by_asset':emitters,'native_readback_required':True})


def motion_certificate(condition_plan,emitter_tracks,listener_m):
    return Certificate(motion_witness_checks(condition_plan,emitter_tracks,listener_m))


def certificate_record(certificate):
    return {'authority':certificate.authority,'passed':certificate.passed,
            'checks':[asdict(c)for c in certificate.checks],
            'scope':'original motion witnesses only; common map certificate and native skin/floor/pixels are separate'}
