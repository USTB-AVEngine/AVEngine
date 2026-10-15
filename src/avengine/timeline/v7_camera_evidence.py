"""A single camera pose is chosen before capture and retained for the full parent."""
from __future__ import annotations
import math
import numpy as np

PROFILE='v7-native-camera-evidence-r17-v1'


def set_static_camera(scene, yaw_deg=None):
    """Normalize legacy camera requests to the production fixed-camera policy."""
    yaw = (math.degrees(math.atan2(scene['camera_forward'][0], -scene['camera_forward'][2]))
           if yaw_deg is None else float(yaw_deg))
    if not math.isfinite(yaw):
        raise ValueError('fixed camera yaw must be finite')
    origin = np.asarray(scene['listener_m'], float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('fixed camera requires one finite listener position')
    angle = math.radians(yaw)
    scene['camera_forward'] = [math.sin(angle), 0., -math.cos(angle)]
    scene['camera_yaw_knots'] = [{'time_s': 0., 'yaw_deg': yaw},
                               {'time_s': float(scene['duration_s']), 'yaw_deg': yaw}]
    scene['static_camera_policy'] = {'position_m': origin.tolist(), 'yaw_deg': yaw,
        'camera_motion_allowed': False, 'selection_stage': 'before native capture'}
    return scene


def assert_static_camera(scene):
    """Refuse a later producer mutation before it can enter capture."""
    knots = scene['camera_yaw_knots']
    yaw = np.asarray([k['yaw_deg'] for k in knots], float)
    if not len(yaw) or not np.isfinite(yaw).all() or np.ptp(yaw) > .5:
        raise ValueError('production camera yaw changes by more than 0.5 degrees')
    anchor = scene.get('static_camera_policy', {}).get('position_m', scene['listener_m'])
    if np.linalg.norm(np.asarray(scene['listener_m'], float)-np.asarray(anchor, float)) > .01:
        raise ValueError('production camera position changes by more than 1 cm')


def configure_camera_evidence(scene,row,policy):
    set_static_camera(scene)
    if policy is None:return
    if policy.get('profile')!=PROFILE:raise ValueError('unsupported registered camera evidence profile')
    yaw=scene['camera_yaw_knots'][0]['yaw_deg']
    quartet=set(row['members'])=={'v0_a0','v0_a1','v1_a0','v1_a1'}
    scene['camera_evidence_policy']={'profile':PROFILE,
        'requested_camera_policy':row['production_scenario'].get('camera_policy'),
        'realized_policy':'one fixed camera pose for the complete parent',
        'quartet':quartet,'center_yaw_deg':yaw,'review_interval_s':.5,
        'FOA_rotation':'same fixed camera basis as the measured native frames',
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
    set_static_camera(scene,yaw)
    policy.update(center_yaw_deg=yaw,observed_body_roots_m=[r.tolist()for r in roots],recentered_from_actual_native_placement=True)


def review_frame_indices(scene):
    if not scene.get('camera_evidence_policy'):return None
    fps=scene['video_fps'];count=round(scene['duration_s']*fps)
    return sorted({min(count-1,round(t*fps))for t in np.arange(0.,scene['duration_s'],.5)}|{count-1})
