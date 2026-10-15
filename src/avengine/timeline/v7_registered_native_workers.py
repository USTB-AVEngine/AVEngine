"""Prepare full native visuals; waiting children receive small file references."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import fcntl,gc,json,multiprocessing,os,time
from avengine.timeline.v7_kujiale_cpu_phases import cpu_phase,phase_state
from avengine.timeline.v7_kujiale_manifest import atomic_write


def native_job(context_path,task_root,visual):return {'context_path':str(context_path),'task_root':str(task_root),'visual':visual}


def _native(job):
    with cpu_phase(job['task_root'],'native_'+job['visual'],estimate_bytes=6000000000):
        context=json.loads(Path(job['context_path']).read_text())if 'context_path'in job else job
        from avengine.timeline.v7_registered_generation import generate_registered_parent
        from avengine.timeline.v7_position_admission import PositionAdmission
        root=Path(context['output'])/'parents'/context['row']['scene_id'];original=PositionAdmission.check
        def serialized(self,*args,**kwargs):
            with(root/'native_parallel_position.lock').open('a')as lease:
                fcntl.flock(lease,fcntl.LOCK_EX);return original(self,*args,**kwargs)
        PositionAdmission.check=serialized;os.environ['AVENGINE_UE_RPC_SHARE_RENDERER_CPU']='1'
        load=lambda key:json.loads(Path(context[key]).read_text())
        catalog=load('catalog_path')if 'context_path'in job else job['catalog'];catalog=catalog['assets']if isinstance(catalog,dict)else catalog
        source=load('source_path')if 'context_path'in job else job['source'];human=load('human_path')if 'context_path'in job else job['human']
        return generate_registered_parent(context['row'],context['house'],context['inputs'],catalog,source,human,context['output'],native_only_visual=job['visual'])

def prepare_native(row,house,inputs,catalog,source,human,output,task_root,*,source_path=None,human_path=None,catalog_path=None):
    from avengine.timeline.v7_registered_generation import generate_registered_parent
    with cpu_phase(task_root,'dry_inputs',estimate_bytes=1500000000):
        generate_registered_parent(row,house,inputs,catalog,source,human,output,prepare_native_inputs_only=True)
    root=Path(output)/'parents'/row['scene_id'];pending=[v for v in ['v0','v1']if not(root/('visual_'+v+'_completed.json')).exists()]
    if not pending:return
    phase_state(task_root,'idle','waiting_full_native_visuals');gc.collect()
    if inputs.get('registered_camera_evidence_policy')is not None:
        with cpu_phase(task_root,'sequential_native_camera_replay',estimate_bytes=6000000000):
            generate_registered_parent(row,house,inputs,catalog,source,human,output,native_only_visual='all')
        return
    if source_path is None or human_path is None or catalog_path is None:
        declared=json.loads((Path(task_root)/'inputs/GAP_PLAN.json').read_text())
        contract=json.loads(Path(inputs['production_input_contract']).read_text())
        source_path=source_path or contract['inputs']['source_admission']['path'];human_path=human_path or declared['human_ue'];catalog_path=catalog_path or declared['assets_ue']
    context=root/f'native_worker_context_{os.getpid()}_{time.time_ns()}.json';atomic_write(context,{'row':row,'house':house,'inputs':inputs,'output':output,'source_path':str(source_path),'human_path':str(human_path),'catalog_path':str(catalog_path)})
    jobs=[native_job(context,task_root,v)for v in pending]
    with ProcessPoolExecutor(max_workers=len(jobs),mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1)as pool:
        futures=[pool.submit(_native,job)for job in jobs]
        for result in futures:result.result()
