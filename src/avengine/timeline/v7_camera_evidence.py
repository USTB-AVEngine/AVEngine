"""Explicit producer camera evidence profile; actual roots and FOA share it."""
from __future__ import annotations
import math
import numpy as np

PROFILE='v7-native-camera-evidence-r17-v1'


def configure_camera_evidence(scene,row,policy):
    if policy is None:return
    if policy.get('profile')!=PROFILE:raise ValueError('unsupported registered camera evidence profile')
    yaw=math.degrees(math.atan2(scene['camera_forward'][0],-scene['camera_forward'][2]))
    quartet=set(row['members'])=={'v0_a0','v0_a1','v1_a0','v1_a1'}
    if quartet:
        hold=float(policy.get('binding_hold_s',8.));peak=float(policy.get('sweep_peak_deg',65.))
        if hold!=8. or not 0<peak<=120:raise ValueError('invalid explicit binding evidence camera schedule')
        schedule=[(0.,0.),(hold,0.),(8.6,peak),(9.3,-peak),(10.,0.)]
        track={0.:yaw,30.:yaw}
        for start in scene['window_starts_s']:
            for offset,delta in schedule:track[start+offset]=yaw+delta
        scene['camera_yaw_knots']=[{'time_s':t,'yaw_deg':v}for t,v in sorted(track.items())]
        realized='actual-root-centered hold8s plus measured entry/exit sweep'
    else:
        scene['camera_yaw_knots']=[{'time_s':float(i),'yaw_deg':yaw+36*i}for i in range(31)]
        realized='uniform original phase; 36deg/s continuous sweep; one turn per 10s'
    scene['camera_evidence_policy']={'profile':PROFILE,'requested_camera_policy':row['production_scenario'].get('camera_policy'),
        'realized_policy':realized,'quartet':quartet,'center_yaw_deg':yaw,
        'review_interval_s':.5,'FOA_rotation':'same complete measured native camera frames',
        'shared_between_counterfactual_members':True,'base_azimuth_resampled':False}


def recenter_on_actual_roots(scene,entries):
    policy=scene.get('camera_evidence_policy')
    if not policy or not policy['quartet']:return
    lookup={e['endpoint_id']:e for e in entries};roots=[np.asarray(lookup[ep]['applied_root_input'],float)[:3,3]for ep in ('voice1','voice2')]
    origin=np.asarray(scene['listener_m'],float);vectors=[(p-origin)*[1.,0.,1.]for p in roots]
    if any(np.linalg.norm(v)<1e-6 for v in vectors):raise ValueError('actual binding actor coincides with observer')
    forward=sum(v/np.linalg.norm(v)for v in vectors)
    if np.linalg.norm(forward)<1e-6:raise ValueError('actual binding roots span opposite hemispheres')
    forward/=np.linalg.norm(forward);yaw=math.degrees(math.atan2(forward[0],-forward[2]))+float(scene['production_scenario'].get('binding_camera_yaw_offset_deg',7.))
    old=float(policy['center_yaw_deg']);delta=(yaw-old+180)%360-180
    for knot in scene['camera_yaw_knots']:knot['yaw_deg']+=delta
    a=math.radians(yaw);scene['camera_forward']=[math.sin(a),0.,-math.cos(a)]
    policy.update(center_yaw_deg=yaw,observed_body_roots_m=[r.tolist()for r in roots],recentered_from_actual_native_placement=True)


def review_frame_indices(scene):
    if not scene.get('camera_evidence_policy'):return None
    fps=scene['video_fps'];count=round(scene['duration_s']*fps)
    return sorted({min(count-1,round(t*fps))for t in np.arange(0.,scene['duration_s'],.5)}|{count-1})
