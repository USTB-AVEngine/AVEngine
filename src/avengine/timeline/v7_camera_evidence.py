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
    scene['camera_yaw_knots']=[{'time_s':0.,'yaw_deg':yaw},{'time_s':30.,'yaw_deg':yaw}]
    scene['camera_evidence_policy']={'profile':PROFILE,'realized_policy':'owner20261015 fixed camera throughout30s','quartet':quartet,'center_yaw_deg':yaw,'FOA_rotation':'same complete measured fixed native camera frames','shared_between_counterfactual_members':True,'base_azimuth_resampled':False}


def recenter_on_actual_roots(scene,entries):
    # The selected fixed pose is immutable, including after native placement.
    return


def review_frame_indices(scene):
    if not scene.get('camera_evidence_policy'):return None
    fps=scene['video_fps'];count=round(scene['duration_s']*fps)
    return sorted({min(count-1,round(t*fps))for t in np.arange(0.,scene['duration_s'],.5)}|{count-1})
