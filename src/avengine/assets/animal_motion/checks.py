"""Independent frame checks for animal direction, contact and foot sliding."""
from copy import deepcopy
import math
import numpy as np
from avengine.assets.animal_motion.profiles import PAWS

from avengine.assets.animal_motion.validation_policy import (
    DIRECTION_DEG,CONTACT_M,PLANT_DRIFT_M,PLANT_SPEED_M_S,
    validation_policy,VALIDATION_POLICY_NAMES,
)
FLOOR_PENETRATION_M=.00101
EMITTER_M=.0001
NONFLOOR_CLEARANCE_M=.02


def direction_check(times, roots, heads, tails, *, policy='strict'):
    settings=validation_policy(policy);direction_deg=settings['direction_deg']
    t=np.asarray(times,float);r=np.asarray(roots,float);h=np.asarray(heads,float);q=np.asarray(tails,float)
    if t.ndim!=1 or len(t)<2 or any(x.shape!=(len(t),3) for x in (r,h,q)):
        raise ValueError('direction_requires_aligned_nonempty_native_frames')
    if not all(np.isfinite(x).all() for x in (t,r,h,q)) or np.any(np.diff(t)<=0):
        raise ValueError('direction_requires_finite_increasing_native_time')
    velocity=np.diff(r[:,[0,2]],axis=0)/np.diff(t)[:,None];speed=np.linalg.norm(velocity,axis=1)
    angles=[];failures=[]
    for i in np.flatnonzero(speed>.005):
        # Check both ends of every moving interval, including the final frame.
        for j in (i,i+1):
            axis=(h[j]-q[j])[[0,2]];norm=np.linalg.norm(axis)
            if norm<1e-6:raise ValueError('head_tail_axis_degenerate')
            angle=float(np.degrees(np.arccos(np.clip(axis@velocity[i]/(norm*speed[i]),-1,1))))
            angles.append({'frame':int(j),'interval':int(i),'angle_deg':angle})
            if angle>direction_deg:failures.append({'frame':int(j),'reason':'backward_or_sideways','angle_deg':angle})
    if not angles:failures.append({'reason':'no_actual_moving_interval'})
    return {'status':'pass' if not failures else 'fail','maximum_angle_deg':max((a['angle_deg'] for a in angles),default=None),
            'checked_moving_intervals':int((speed>.005).sum()),'maximum_speed_m_s':float(speed.max()),'threshold_deg':direction_deg,'failures':failures,'samples':angles}


def frame_checks(frames, *, mode='walk_animation', native=False, policy='strict'):
    settings=validation_policy(policy)
    if not frames:raise ValueError('no_motion_frames')
    failures=[];anchors={};previous={};max_lock=0.;max_slip=0.;max_contact=0.;max_all=0.;paw_phases={s:set() for s in PAWS}
    for i,f in enumerate(frames):
        feet=f['feet']
        if set(feet)!=set(PAWS):raise ValueError('four_named_paws_required')
        if not any(x['in_contact'] for x in feet.values()):failures.append(f'{i}:no_support_phase')
        for side,x in feet.items():
            delta=float(x['floor_delta_m']);paw_phases[side].add(bool(x['in_contact']));max_all=max(max_all,abs(delta))
            if not math.isfinite(delta) or delta < -FLOOR_PENETRATION_M:failures.append(f'{i}:{side}:paw_penetration')
            # This bounded slow gait also keeps its 8 mm swing inside 12 mm.
            if abs(delta)>CONTACT_M:failures.append(f'{i}:{side}:four_paw_12mm')
            if x['in_contact']:
                max_contact=max(max_contact,abs(delta))
                if mode=='walk_animation':
                    key=(side,x['support_segment']);point=np.asarray(x['support_world_m'],float)
                    if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('nonfinite_support_point')
                    if key not in anchors:anchors[key]=point
                    drift=float(np.linalg.norm(point-anchors[key]));max_lock=max(max_lock,drift)
                    if drift>settings['plant_drift_m']:failures.append(f'{i}:{side}:stance_drift')
                    if side in previous and previous[side][0]==key:
                        _,pt,t=previous[side];speed=float(np.linalg.norm(point-pt)/(f['time_s']-t));max_slip=max(max_slip,speed)
                        if settings['plant_speed_m_s'] is not None and speed>settings['plant_speed_m_s']:failures.append(f'{i}:{side}:stance_speed')
                    previous[side]=(key,point,float(f['time_s']))
            else:previous.pop(side,None)
        if f['body_floor_min_m'] < -FLOOR_PENETRATION_M:failures.append(f'{i}:full_skin_penetration')
        if f['body_head_tail_pitch_deg']>45:failures.append(f'{i}:body_upright_instead_of_quadruped')
        if native:
            if f['native_emitter_readback_error_m']>EMITTER_M:failures.append(f'{i}:mouth_attachment')
            if f['minimum_nonfloor_clearance_m']<=NONFLOOR_CLEARANCE_M:failures.append(f'{i}:nonfloor_collision')
            if f['navmesh_horizontal_error_m']>.002:failures.append(f'{i}:navigation')
    if mode=='walk_animation':
        for s,flags in paw_phases.items():
            if flags!={False,True}:failures.append(s+':incomplete_stance_and_swing_cycle')
    direction=direction_check([f['time_s'] for f in frames],[f['root_world_m'] for f in frames],[f['head_world_m'] for f in frames],[f['tail_world_m'] for f in frames],policy=policy)
    if direction['status']!='pass':failures.append('heading_velocity_disagreement')
    return {'status':'pass' if not failures else 'fail','frame_count':len(frames),'failures':failures,'direction':direction,
        'max_contact_error_m':max_contact,'max_all_four_paw_gap_m':max_all,'max_stance_drift_m':max_lock,'max_stance_speed_m_s':max_slip,
        'min_full_skin_floor_m':min(f['body_floor_min_m'] for f in frames),'max_body_pitch_deg':max(f['body_head_tail_pitch_deg'] for f in frames),
        'validation_policy':settings,'native_checks_run':native,'motion_mode':mode,'slide_intentionally_exempt_from_gait_claim':mode=='slide_no_animation'}


def direction_negative_controls(frames, *, policy='strict'):
    times=[f['time_s'] for f in frames];roots=[f['root_world_m'] for f in frames];heads=np.array([f['head_world_m'] for f in frames]);tails=np.array([f['tail_world_m'] for f in frames])
    backwards=direction_check(times,roots,tails,heads,policy=policy)
    d=heads-tails;sideways=d.copy();sideways[:,0]=d[:,2];sideways[:,2]=-d[:,0]
    side=direction_check(times,roots,tails+sideways,tails,policy=policy)
    return {'status':'pass' if backwards['status']==side['status']=='fail' else 'fail','backward_180deg':backwards,'sideways_90deg':side,'excluded_from_admission':True}
