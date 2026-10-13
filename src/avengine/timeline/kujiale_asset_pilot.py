"""Train-only asset pilot behind the existing Kujiale production coordinator."""
from __future__ import annotations
import argparse,json,os,subprocess,time,traceback
from pathlib import Path
from copy import deepcopy
from avengine.timeline.continuous_pilot import write_json

def live_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.pending')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def progress(root,message):
    with (root/'PROGRESS.md').open('a') as f:f.write('- '+time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+'：'+message+'\n')

def freeze_inputs(root):
    """Choose the newest authorized audio admission when input sealing starts."""
    old=root.parent/'ue_round14/inputs';prior=root.parent/'ue_round20_assets'
    src=Path(json.loads((root/'evidence/planning_summary.json').read_text())['source_admission_inspected'])
    candidate=src.with_name('source_admission_v11_classes27_english_sao.json')
    actual=candidate if candidate.is_file() else src
    source=json.loads(actual.read_text())
    from avengine.timeline.v7_audio_sources import filter_admission
    _,rejected=filter_admission(source['records'])
    if rejected:raise ValueError('authorized sound admission contains blocked records: '+str(rejected[:3]))
    receipt={'status':'sealed_inputs','source_admission':str(actual),'input_sealing_started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'v11_present_at_input_sealing':candidate.is_file(),'records':len(source['records']),'owner_rule':'use v11 if available when sealing starts'}
    write_json(root/'inputs/source_admission.json',source)
    inputs=json.loads((old/'runtime.json').read_text())
    inputs.update(render_backend='ue_spear',semantic_manifest=None,kujiale_ue_asset_admission=json.loads((root/'inputs/task_scope.json').read_text())['asset_admission_path'])
    write_json(root/'inputs/runtime.json',inputs)
    import shutil
    for name in ['assets.json','human_admission.json']:shutil.copy2(old/name,root/'inputs'/name)
    write_json(root/'evidence/input_seal.json',receipt)
    progress(root,'封存本轮输入：'+actual.name+'，'+str(len(source['records']))+' 个声音记录；不修改源名册。')

def worker(root,index,gpu,seed_path=None):
    import signal
    from avengine.timeline.v7_registered_generation import generate_registered_parent
    from avengine.timeline.v7_storage_budget import configure_kujiale_production
    if gpu not in (2,3) or os.nice(0)<10:raise ValueError('pilot needs physical GPU 2/3 and nice10')
    rows=[json.loads(l) for l in (Path(seed_path or json.loads((root/'inputs/task_scope.json').read_text())['seed_path'])).read_text().splitlines()]
    row=rows[index];houses=json.loads((root/'inputs/houses.json').read_text())['houses'];house=deepcopy(next(h for h in houses if h['house']==row['house']))
    if row['split']!='train' or house['split']!='train':raise ValueError('pilot accepts admitted training houses only')
    ue=house['input']['ue_config'];ue.update(graphics_adapter=0,physical_gpu_index=gpu,unreal_editor=str(root/'scripts'/f'ue_gpu{gpu}'))
    (root/'evidence/parent_results').mkdir(parents=True,exist_ok=True)
    inputs=json.loads((root/'inputs/runtime.json').read_text());assets=json.loads((root/'inputs/assets.json').read_text())['assets']
    source=json.loads((root/'inputs/source_admission.json').read_text());human=json.loads((root/'inputs/human_admission.json').read_text())
    configure_kujiale_production(root/'production',root,workers=4)
    start=time.time()
    try:
        result=generate_registered_parent(row,house,inputs,assets,source,human,root/'production')
        from avengine.qa.kujiale_pilot_truth import audit_sealed_parent
        truth=audit_sealed_parent(root/'production/parents'/row['scene_id'],result['delivery_directory'])
        result['independent_truth']={'status':truth['status'],'checked_forms':truth['checked_forms'],'mismatches':len(truth['mismatches'])}
        result['pilot_scope']='local native integration trial; no training table export or CCI'
    except BaseException as error:
        result={'scene_id':row['scene_id'],'house':row['house'],'status':'failed','error_type':type(error).__name__,'reason':str(error),'traceback':traceback.format_exc(),'wall_seconds':time.time()-start}
        write_json(root/'evidence/parent_results'/f'{row["scene_id"]}_{time.time_ns()}.json',result);live_json(root/'evidence/parent_results'/f'{row["scene_id"]}.json',result);progress(root,row['scene_id']+' 失败：'+str(error));raise
    write_json(root/'evidence/parent_results'/f'{row["scene_id"]}_{time.time_ns()}.json',result);live_json(root/'evidence/parent_results'/f'{row["scene_id"]}.json',result)
    progress(root,row['scene_id']+' 完成原生生成、声音和题目门禁。')
    print(json.dumps(result,ensure_ascii=False),flush=True)

def launch_worker(root,index,gpu,slot,seed_path=None):
    repo=Path(__file__).resolve().parents[3]
    env=os.environ.copy();env.update(json.loads((root/'ue_env.json').read_text()))
    env.update(PYTHONPATH=str(repo/'src')+':'+str(repo)+':'+str(root/'pydeps')+':'+env['PYTHONPATH'],
        PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(root/'tmp'))
    cpus=sorted(os.sched_getaffinity(0));offset=(slot*16)%len(cpus);affinity=cpus[offset:offset+16]
    cmd=['taskset','-c',','.join(map(str,affinity)),'/data/jzy/envs/room-selection-merged-20261003/bin/python',
        str(repo/'tools/timeline/v7_kujiale_production_v2.py'),'--pilot-root',str(root),'--parent-index',str(index),'--physical-gpu',str(gpu)]
    if seed_path is not None:cmd+=['--seed-path',str(seed_path)]
    sandbox=['/usr/bin/bwrap','--die-with-parent','--unshare-user','--ro-bind','/','/','--dev','/dev','--proc','/proc',
        '--bind',str(root),str(root),'--bind',str(root/'tmp'),'/tmp','--bind',str(root/'shm'),'/dev/shm',
        '--bind',str(root/'ue_config'),'/data/jzy/.config','--dev-bind','/dev/nvidiactl','/dev/nvidiactl',
        '--dev-bind','/dev/nvidia'+str(gpu),'/dev/nvidia'+str(gpu),'--dev-bind','/dev/nvidia-uvm','/dev/nvidia-uvm',
        '--dev-bind','/dev/nvidia-uvm-tools','/dev/nvidia-uvm-tools','--chdir',str(repo)]
    label=f'parent_{index:03d}_gpu{gpu}_{time.time_ns()}'
    f=(root/'logs'/f'{label}_supervisor.log').open('x')
    proc=subprocess.Popen(['nice','-n','10','python3',str(root/'scripts/tracked_job.py'),'--root',str(root),'--name',label,'--gpu',str(gpu),'--timeout','9000','--',*sandbox,*cmd],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);f.close()
    return proc

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot-root',type=Path,required=True);p.add_argument('--parent-index',type=int);p.add_argument('--seed-path',type=Path);p.add_argument('--physical-gpu',type=int);p.add_argument('--start-index',type=int,default=0);p.add_argument('--limit',type=int,default=12);p.add_argument('--workers-per-gpu',type=int,default=2)
    a=p.parse_args();root=a.pilot_root.resolve()
    if root.name!='ue_round21_pilot':raise ValueError('pilot output must be the authorized round21 directory')
    if a.parent_index is not None:worker(root,a.parent_index,a.physical_gpu,a.seed_path);return
    if not 1<=a.workers_per_gpu<=4:raise ValueError('one to four instances per assigned card')
    if not (root/'evidence/input_seal.json').exists():freeze_inputs(root)
    rows=[json.loads(l) for l in (Path(json.loads((root/'inputs/task_scope.json').read_text())['seed_path'])).read_text().splitlines()][:a.limit]
    pending=[]
    for i,row in enumerate(rows):
        if i<a.start_index:continue
        path=root/'evidence/parent_results'/f'{row["scene_id"]}.json'
        if path.exists() and json.loads(path.read_text()).get('status')=='formal_alpha_admitted':continue
        pending.append(i)
    active=[];deadline=json.loads((root/'inputs/task_scope.json').read_text())['deadline_epoch']
    import fcntl,signal
    lock=(root/'coordinator.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    def interrupted(signum,frame):
        raise RuntimeError('own coordinator interrupted by signal '+str(signum))
    for signum in (signal.SIGTERM,signal.SIGINT):
        signal.signal(signum,interrupted)
    try:
        while pending or active:
            if time.time()>deadline:raise RuntimeError('pilot six-hour deadline reached')
            if (root/'STOP').exists():raise RuntimeError('own STOP requested')
            for gpu in (2,3):
                count=sum(g==gpu for _,_,g,_ in active)
                while pending and count<a.workers_per_gpu:
                    # XML graphics processes and measured instance budget are checked
                    # again by launch_native before every actual UE launch.
                    import xml.etree.ElementTree as ET
                    xml=subprocess.check_output(['nvidia-smi','-q','-x'],text=True)
                    cards=ET.fromstring(xml).findall('gpu')
                    free=float(cards[gpu].findtext('fb_memory_usage/free').split()[0])
                    if free<10240:break
                    index=pending.pop(0);proc=launch_worker(root,index,gpu,len(active));active.append((proc,index,gpu,time.time()));count+=1
                    live_json(root/'evidence/coordinator_state.json',{'status':'running','active':[{'supervisor_pid':q.pid,'parent_index':i,'gpu':g} for q,i,g,_ in active],'pending':pending,'original_generator':'33b5b12df1a45ca265bf0c4fbe9b26b664577a7c'})
            for entry in list(active):
                if entry[0].poll() is not None:active.remove(entry)
            time.sleep(5)
    finally:
        # Only Popen children launched by this coordinator are stopped. Their
        # supervisors terminate the separately verified child tree.
        for proc,_,_,_ in active:
            if proc.poll() is None:proc.terminate()
        for proc,_,_,_ in active:
            try:proc.wait(timeout=40)
            except subprocess.TimeoutExpired:
                raise RuntimeError('own worker supervisor has not finished cleanup: '+str(proc.pid))
        lock.close()
    results=[json.loads((root/'evidence/parent_results'/f'{r["scene_id"]}.json').read_text()) for r in rows[a.start_index:]]
    live_json(root/'evidence/coordinator_state.json',{'status':'finished','parents':len(results),'passed':sum(r['status']=='formal_alpha_admitted' for r in results),'results':results,'training_exported':False,'CCI_transferred':False})
    progress(root,'试产协调器结束，所有子进程已退出，结果留在本轮目录。')
if __name__=='__main__':main()
