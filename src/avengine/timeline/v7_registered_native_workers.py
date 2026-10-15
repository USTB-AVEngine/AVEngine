"""Prepare both full native visuals before entering the shared CPU QA phase."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import fcntl,gc,multiprocessing,os
from avengine.timeline.v7_kujiale_cpu_phases import cpu_phase,phase_state


def _native(job):
    from avengine.timeline.v7_registered_generation import generate_registered_parent
    from avengine.timeline.v7_position_admission import PositionAdmission
    root=Path(job['output'])/'parents'/job['row']['scene_id'];original=PositionAdmission.check
    def serialized(self,*args,**kwargs):
        with(root/'native_parallel_position.lock').open('a')as lease:
            fcntl.flock(lease,fcntl.LOCK_EX);return original(self,*args,**kwargs)
    PositionAdmission.check=serialized;os.environ['AVENGINE_UE_RPC_SHARE_RENDERER_CPU']='1'
    with cpu_phase(job['task_root'],'native_'+job['visual'],estimate_bytes=2500000000):
        return generate_registered_parent(job['row'],job['house'],job['inputs'],job['catalog'],job['source'],job['human'],job['output'],native_only_visual=job['visual'])


def prepare_native(row,house,inputs,catalog,source,human,output,task_root):
    from avengine.timeline.v7_registered_generation import generate_registered_parent
    with cpu_phase(task_root,'dry_inputs',estimate_bytes=1500000000):
        generate_registered_parent(row,house,inputs,catalog,source,human,output,prepare_native_inputs_only=True)
    root=Path(output)/'parents'/row['scene_id'];pending=[v for v in ['v0','v1']if not(root/('visual_'+v+'_completed.json')).exists()]
    if not pending:return
    phase_state(task_root,'idle','waiting_full_native_visuals');gc.collect()
    if inputs.get('registered_camera_evidence_policy')is not None:
        with cpu_phase(task_root,'sequential_native_camera_replay',estimate_bytes=2500000000):
            generate_registered_parent(row,house,inputs,catalog,source,human,output,native_only_visual='all')
        return
    jobs=[{'row':row,'house':house,'inputs':inputs,'catalog':catalog,'source':source,'human':human,'output':output,'task_root':str(task_root),'visual':v}for v in pending]
    with ProcessPoolExecutor(max_workers=len(jobs),mp_context=multiprocessing.get_context('spawn'))as pool:
        futures=[pool.submit(_native,job)for job in jobs]
        for result in futures:result.result()
