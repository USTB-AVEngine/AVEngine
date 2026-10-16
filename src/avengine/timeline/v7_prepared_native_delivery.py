"""Consume a solver-certified live world without building another native scene."""
from pathlib import Path
import shutil
import numpy as np


def scene_inputs(prepared,row):
    if prepared.closed or prepared.video is None:raise ValueError('prepared native world is already closed; solve again')
    if not prepared.certificate.passed:raise ValueError('prepared native certificate did not pass')
    if prepared.row['scene_id']!=row['scene_id']:raise ValueError('prepared native scene identity differs from requested scene')
    # Contract provenance is metadata, not a request to rebuild the solved world.
    receipt=row.get('production_sampling')
    if receipt is not None:
        from avengine.timeline.v7_production_inputs import validate_sampling_row
        if prepared.row.get('production_sampling') not in (None,receipt):
            raise ValueError('prepared native production contract differs from requested contract')
        validated={**prepared.row,'production_sampling':receipt}
        intake=validate_sampling_row(validated,prepared.assets)
        # Preserve all five physical objects and every planned pose/event.
        prepared.row['production_sampling']=dict(receipt)
        if intake:prepared.scene['production_sampling_intake']=intake
    return prepared.row,prepared.house,prepared.scene,prepared.program,prepared.assets


def arm_prepared_encode(prepared):
    video=prepared.video
    if not getattr(video,'standing_checks',None):raise ValueError('prepared native placement checks missing')
    video._prepared_native_encode=True
    video._prepared_qualification_directory=Path(prepared.directory)/'standing'
    return video


def consume_prepared_qualification(video,scene,directory):
    if scene is not video._phase2_scene:raise ValueError('prepared encode must consume its original scene object')
    checks=video.standing_checks
    if len(checks)!=len(video.actors)or any(c.get('status')!='pass'for c in checks):
        raise ValueError('prepared placement evidence is incomplete')
    source=Path(video._prepared_qualification_directory);target=Path(directory);target.mkdir(parents=True,exist_ok=True)
    for name in ('standing_admission.json','native_support_survey.json'):
        original=source/name;dest=target/name
        if original.resolve()!=dest.resolve():
            if dest.exists():
                if original.read_bytes()!=dest.read_bytes():raise ValueError('prepared qualification target differs')
            else:shutil.copyfile(original,dest)
    # The original acoustic position predicate remains in force. This checks
    # exact current emitters; it never searches for or changes a placement.
    admission=getattr(video,'acoustic_position_admission',None)
    if admission:
        for entry in video.actors:
            if entry['declaration'].get('silent_decor'):continue
            actual=video._apply_entry(entry,scene,frame_index=0)['emitter_position_m']
            result=admission(entry['endpoint_id'],np.asarray(actual))
            if result.get('status')=='blocked':raise ValueError('prepared exact emitter fails original native RIR direction gate')
    from avengine.timeline.v7_cpu_visual_runtime import configure_cpu_visual
    policy=configure_cpu_visual(getattr(video,'_prepared_cpu_visual_inputs',getattr(video,'motion_inputs',{})))
    if hasattr(video,'receipt'):video.receipt['visual_cpu_policy_after_RIR']=policy
    return checks
