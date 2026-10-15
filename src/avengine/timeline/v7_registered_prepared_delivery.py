"""Consume both original solver-certified registered visual worlds."""
from pathlib import Path
import shutil
import numpy as np


def apply_registered_root_pin(scene,entry,root):
    if not scene.get('joint_strict_placement'):return False
    pin=scene.get('native_attachment_supports',{}).get(entry['declaration']['asset_id'],{})
    if not pin.get('joint_solved_pose'):raise ValueError('strict registered world is missing its solved root pin')
    from avengine.capture.v7_visible_attachment import rotation
    point=np.asarray(pin['joint_expected_root_m'],float);yaw=float(pin['yaw_rad'])
    if point.shape!=(3,)or not np.isfinite(point).all()or not np.isfinite(yaw):raise ValueError('invalid solved registered root')
    root[:3,:3]=rotation(yaw);root[:3,3]=point
    return True


def registered_scene_inputs(visuals,row):
    if set(visuals)!={'v0','v1'}:raise ValueError('both independently prepared registered visuals are required')
    for payload in visuals.values():
        if payload.closed or payload.video is None or not payload.certificate.passed:raise ValueError('registered prepared visual is closed or uncertified')
        if payload.row['scene_id']!=row['scene_id']:raise ValueError('registered prepared parent identity differs')
    p=visuals['v0']
    return p.row,p.house,p.scene,p.program,p.assets


def arm_registered_prepared(payload):
    video=payload.video
    if not getattr(video,'standing_checks',None):raise ValueError('prepared registered standing evidence missing')
    video._prepared_registered_scene=payload.scene
    video._prepared_registered_directory=Path(payload.directory)/'standing'
    return video


def consume_registered_qualification(video,scene,directory):
    if scene is not video._prepared_registered_scene:raise ValueError('registered encode must consume its original world object')
    checks=video.standing_checks
    if len(checks)!=len(video.actors)or any(c.get('status')!='pass'for c in checks):raise ValueError('registered prepared physical checks are incomplete')
    source=video._prepared_registered_directory;target=Path(directory);target.mkdir(parents=True,exist_ok=True)
    required=source/'standing_admission.json'
    if not required.is_file():raise ValueError('original registered standing receipt missing')
    for original in (required,source/'placement_search.json'):
        if not original.is_file():continue
        dest=target/original.name
        if original.resolve()==dest.resolve():continue
        if dest.exists():
            if original.read_bytes()!=dest.read_bytes():raise ValueError('registered standing evidence changed')
        else:shutil.copyfile(original,dest)
    admission=getattr(video,'acoustic_position_admission',None)
    if admission:
        for entry in video.actors:
            if entry['declaration'].get('silent_decor'):continue
            actual=video._apply_entry(entry,scene,frame_index=0)['emitter_position_m']
            if admission(entry['endpoint_id'],np.asarray(actual))['status']=='blocked':raise ValueError('prepared registered emitter fails original 5deg gate')
    return checks
