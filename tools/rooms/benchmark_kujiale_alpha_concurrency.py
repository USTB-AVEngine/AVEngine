#!/usr/bin/env python3
"""Measured 2/4/8/12 native UE recorded-alpha end-to-end worker rounds."""
from __future__ import annotations
import argparse,datetime,json,os,shutil,signal,subprocess,time
from pathlib import Path
from copy import deepcopy
import psutil
from tools.rooms.benchmark_kujiale_native_concurrency import choose_topology
from tools.rooms.package_kujiale_alpha import package_alpha
from tools.rooms.compact_kujiale_alpha import compact_alpha_parent
from tools.rooms.alpha_storage_contract import write_owned_json as write_json
from tools.rooms.alpha_live_status import write_live_status


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--canary',type=Path,required=True);p.add_argument('--rounds',type=int,nargs='+',default=[2,4,8,12]);p.add_argument('--output-name',default='concurrency_alpha_v1');p.add_argument('--footprint-calibration-round',type=Path);a=p.parse_args()
    o=a.root.resolve();r=o/'ue_round8';repo=Path(__file__).resolve().parents[2];py=os.sys.executable
    if not a.output_name.replace('_','').isalnum():raise ValueError('unsafe output name')
    canary=read(a.canary/'generation/result.json')
    if canary['status']!='formal_alpha_admitted':raise ValueError('formal native UE alpha canary must pass before measuring throughput')
    root=r/a.output_name;root.mkdir(exist_ok=False);rounds=[];peak=3908.
    if (a.canary/'guard/result.json').is_file():peak=max(peak,read(a.canary/'guard/result.json')['peak_mib'])
    reserve_per_worker=750_000_000;footprint_proof=None
    if a.footprint_calibration_round is not None:
        calibrated=a.footprint_calibration_round.resolve();calibrated.relative_to(r)
        rr=read(calibrated/'result.json')
        if rr['status']!='completed' or any(j['pipeline_status']!='formal_alpha_admitted' for j in rr['jobs']):raise ValueError('completed native calibration round required')
        sampled=read(r/'evidence/job_allocated_footprints_snapshot.json')
        peaks=[x['peak_allocated_bytes'] for x in sampled['jobs'] if Path(x['job']).parent==calibrated]
        if len(peaks)!=rr['parallel_instances'] or min(peaks)<=0:raise ValueError('all completed worker footprints must be measured')
        reserve_per_worker=max(250_000_000,2*max(peaks)+64_000_000)
        footprint_proof={'round_result':str(calibrated/'result.json'),'observed_unique_inode_worker_peaks':peaks,'startup_reserve_per_worker_bytes':reserve_per_worker,'rule':'twice maximum measured same-profile worker allocation plus 64MB, minimum 250MB; 80GiB+2GB floor and 40GB growth limit unchanged'}
    start=time.monotonic();stop=None;active_jobs=[]
    def terminate(number,_frame):
        nonlocal stop;stop='coordinator_signal_'+str(number)
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    identity=psutil.Process();write_json(root/'coordinator_identity.json',{'pid':identity.pid,'create_time':identity.create_time(),'command':identity.cmdline()})
    try:
        for n in a.rounds:
            if stop:break
            d=root/f'n{n}';d.mkdir(exist_ok=False);snapshot=read(r/'evidence/resource_snapshot.json')
            if time.time()-snapshot['sample_epoch']>8 or snapshot['disk_stop_reason']:raise RuntimeError('fresh task resource preflight failed')
            topology=choose_topology(snapshot['gpus'],n,peak);reserve=n*reserve_per_worker
            baseline=read(r/'evidence/baseline.json')
            if snapshot['shared_data_free_bytes']-reserve<baseline['minimum_shared_disk_free_bytes']+2_000_000_000 or snapshot['allocated_bytes']+reserve>baseline['maximum_total_task_allocated_bytes']-2_000_000_000 or snapshot['new_allocated_bytes']+reserve>baseline['maximum_new_allocated_bytes']-2_000_000_000:raise RuntimeError('round disk startup reserve insufficient')
            began=time.monotonic();jobs=[];cpu_io={};global_samples=0
            receipt={'parallel_instances':n,'topology_gpu_indices':topology,'previous_measured_peak_mib':peak,'initial_resources':snapshot,
                'scope':'fresh full 30s single-member recorded/static four-human alpha, 2x10s windows, 14 logical questions/window; canonical QA/select/seal + local portability/compaction; warm task DDC and per-job cold native RIR cache',
                'benchmark_inputs_are_replicas_not_independent_training_worlds':True,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':[],'startup_reserve_per_worker_bytes':reserve_per_worker,'footprint_calibration':footprint_proof}
            for i,gpu in enumerate(topology):
                job=d/f'job{i:02d}';job.mkdir();house=read(a.canary/'house.json');house['input']['ue_config'].update(graphics_adapter=gpu,rpc_port=0)
                row=read(a.canary/'seed_row.json');row.update(scene_id=f'v7_kujiale_0022_bench_{a.output_name}_n{n}_j{i:02d}',members=['v0_a0'],counterfactual_plan='single',allowed_member_changes=[],question_quota_base=2*i)
                write_json(job/'house.json',house);write_json(job/'seed_row.json',row)
                q=read(a.canary/'guard_request.json');command=q['command']
                command=[s.replace(str(a.canary/'house.json'),str(job/'house.json')).replace(str(a.canary/'seed_row.json'),str(job/'seed_row.json')).replace(str(a.canary/'generation'),str(job/'generation')) for s in command if s!='--resume']
                q.update(command=command,gpu_index=gpu,previous_peak_mib=peak,run_directory=str(job/'guard'),cpu_affinity=list(range(16+8*i,24+8*i)),reserve_bytes=reserve_per_worker,maximum_seconds=9000)
                write_json(job/'guard_request.json',q)
                log=(job/'coordinator.log').open('x');env=dict(os.environ);env.update(q['environment'])
                proc=subprocess.Popen([py,str(repo/'tools/rooms/v7_kujiale_ue_guard.py'),'--request',str(job/'guard_request.json')],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                jobs.append({'process':proc,'log':log,'directory':job,'create_time':psutil.Process(proc.pid).create_time(),'command':psutil.Process(proc.pid).cmdline(),'gpu':gpu,'index':i,'accounted':False})
                active_jobs=jobs;write_json(job/'guard_identity.json',{'jobs':[{'pid':j['process'].pid,'create_time':j['create_time'],'command':j['command'],'request':str(j['directory']/'guard_request.json')} for j in jobs]})
            write_json(d/'launch.json',{'jobs':[{'pid':j['process'].pid,'create_time':j['create_time'],'command':j['command'],'request':str(j['directory']/'guard_request.json')} for j in jobs]})
            with (d/'system_samples.jsonl').open('x') as stream:
                while any(not j['accounted'] for j in jobs):
                    if stop:raise RuntimeError(stop)
                    shared=read(r/'evidence/resource_snapshot.json');tick=time.monotonic()
                    sample={'sample_epoch':time.time(),'elapsed_seconds':tick-began,'system_cpu_percent':psutil.cpu_percent(None),'system_cpu_times_percent':psutil.cpu_times_percent(None)._asdict(),'memory':psutil.virtual_memory()._asdict(),'swap':psutil.swap_memory()._asdict(),'load_average':os.getloadavg(),'disk_io_system':psutil.disk_io_counters()._asdict(),'gpu_resource_sample':shared['gpus'],'shared_data_free_bytes':shared['shared_data_free_bytes'],'task_allocated_bytes':shared['allocated_bytes'],'new_task_allocated_bytes':shared['new_allocated_bytes'],'task_owned_process_io':[]}
                    for j in jobs:
                        try:
                            parent=psutil.Process(j['process'].pid)
                            if parent.create_time()!=j['create_time'] or parent.cmdline()!=j['command']:raise psutil.NoSuchProcess(parent.pid)
                            for proc in [parent]+parent.children(recursive=True):
                                try:
                                    io=proc.io_counters();cpu=proc.cpu_times();key=(proc.pid,proc.create_time())
                                    value={'pid':proc.pid,'create_time':key[1],'read_bytes':io.read_bytes,'write_bytes':io.write_bytes,'cpu_seconds':cpu.user+cpu.system,'rss_bytes':proc.memory_info().rss}
                                    cpu_io[key]=value;sample['task_owned_process_io'].append(value)
                                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                        if j['process'].poll() is not None and not j['accounted']:
                            ended=time.monotonic();j['log'].close();guard=read(j['directory']/'guard/result.json');result=read(j['directory']/'generation/result.json') if (j['directory']/'generation/result.json').exists() else {}
                            item={'job_index':j['index'],'gpu_index':j['gpu'],'guard_status':guard['status'],'pipeline_status':result.get('status'),'whole_pipeline_wall_seconds':result.get('whole_pipeline_wall_seconds'),'guard_machine_seconds':guard['elapsed_seconds'],'finished_group_seconds':ended-began,'owned_cpu_seconds_lower_bound':guard.get('owned_tree_cpu_seconds_lower_bound'),'peak_mib':guard['peak_mib'],'alpha_formal_windows':result.get('alpha_formal_windows',0),'guard_result':str(j['directory']/'guard/result.json'),'pipeline_result':str(j['directory']/'generation/result.json'),'reason':result.get('reason'),'stop_reason':guard.get('stop_reason')}
                            if guard['status']=='pass' and result.get('status')=='formal_alpha_admitted':
                                par=j['directory']/'generation/parents'/result['scene_id'];packaged=package_alpha(result,par,o);compaction=compact_alpha_parent(par,o)
                                item.update(delivery=packaged,compacted_intermediate_logical_bytes=compaction['logical_intermediate_bytes'])
                            receipt['jobs'].append(item);j['accounted']=True;write_live_status(d/'live_result.json',receipt)
                    stream.write(json.dumps(sample)+'\n');stream.flush();global_samples+=1;time.sleep(max(0,1-(time.monotonic()-tick)))
            receipt['jobs'].sort(key=lambda x:x['job_index']);wall=max(j['finished_group_seconds'] for j in receipt['jobs']);complete=sum(j['pipeline_status']=='formal_alpha_admitted' and j['guard_status']=='pass' for j in receipt['jobs'])
            receipt.update(status='completed' if complete==n else 'failed',group_generation_machine_seconds=wall,group_generation_and_compaction_seconds=time.monotonic()-began,completed_single_member_scenes=complete,member_scenes_per_hour=3600*complete/wall,alpha_formal_windows=sum(j['alpha_formal_windows'] for j in receipt['jobs']),system_sample_count=global_samples,
                mean_cpu_occupied_cores_lower_bound=sum(j['owned_cpu_seconds_lower_bound'] or 0 for j in receipt['jobs'])/wall,task_owned_io_lower_bound={'read_bytes':sum(v['read_bytes'] for v in cpu_io.values()),'write_bytes':sum(v['write_bytes'] for v in cpu_io.values())},mean_scene_wall_seconds=sum(j['whole_pipeline_wall_seconds'] or 0 for j in receipt['jobs'])/n)
            peak=max(peak,max(j['peak_mib'] for j in receipt['jobs']));write_json(d/'result.json',receipt);rounds.append(receipt);write_live_status(root/'live_result.json',{'rounds':rounds})
            with (o/'PROGRESS_zh.md').open('a') as f:f.write('\n\n## '+datetime.datetime.now(datetime.timezone.utc).isoformat()+f' 第8回合{n}实例正式alpha全链路实测\n\n'+f"状态{receipt['status']}，完成{complete}/{n}个单成员场景（同一控制输入的测速副本，不计独立训练世界）；拓扑{topology}，整批{wall:.3f}秒，{receipt['member_scenes_per_hour']:.3f}成员场景/小时，CPU占用下界{receipt['mean_cpu_occupied_cores_lower_bound']:.3f}核。来源{d}/result.json。已完成场景中间件逐个压缩删除，保留正式alpha交付物和3帧/静态接触见证；待CCI核SHA后删除48g交付副本。\n")
            print(json.dumps({k:receipt[k] for k in ['parallel_instances','status','member_scenes_per_hour','group_generation_machine_seconds','mean_cpu_occupied_cores_lower_bound']},ensure_ascii=False),flush=True)
            active_jobs=[]
            if receipt['status']!='completed':raise RuntimeError('failed complete pipeline; later rounds withheld')
        write_json(root/'result.json',{'status':'completed' if not stop else 'stopped','rounds':rounds,'controller_machine_seconds':time.monotonic()-start})
    finally:
        # Identity checked own guards only; guards stop their exact child trees.
        for j in active_jobs:
            if j['process'].poll() is None:
                try:
                    proc=psutil.Process(j['process'].pid)
                    if proc.create_time()==j['create_time'] and proc.cmdline()==j['command']:proc.terminate()
                except psutil.NoSuchProcess:pass
        for j in active_jobs:
            if j['process'].poll() is None:j['process'].wait(timeout=45)
        if not (root/'result.json').exists():write_json(root/'result.json',{'status':'incomplete','rounds':rounds,'stop_reason':stop,'controller_machine_seconds':time.monotonic()-start})

if __name__=='__main__':main()
