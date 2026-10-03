#!/usr/bin/env python3
"""Sequential one-GPU shard worker for 48g pilot or CCI production.

No remote actions, model inference, process preemption, or environment changes.
Use distinct --output per worker; global parent indices are disjoint modulo shards.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))
from avengine.qa.frozen_panel_versions import validate_versions, digest, require_canary_acceptance
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def stamp():return datetime.now(timezone.utc).isoformat()
def progress(root,message):
    with (root/'PROGRESS_zh.md').open('a',encoding='utf8') as f:f.write(f'\n{stamp()}：{message}\n')
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shard-index',type=int,required=True);p.add_argument('--shard-count',type=int,required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--start-index',type=int,default=0);p.add_argument('--end-index',type=int);p.add_argument('--resume',action='store_true');p.add_argument('--max-quality-attempts',type=int,default=3)
    a=p.parse_args();c=read(a.config);total=len(c['houses'])*int(c['groups_per_house'])
    if c.get('split') not in {'pilot','test'} or not re.fullmatch(r'[A-Za-z0-9_-]+',c.get('panel_id','pilot')):raise ValueError('invalid panel identity/split')
    if c['groups_per_house']<=0 or a.max_quality_attempts<=0:raise ValueError('invalid parent/attempt count')
    if c['split']=='test' and (len(c['houses'])!=12 or c['groups_per_house']!=60 or
        len(c.get('excluded_physical_parent_sha256',[]))!=20 or len(c.get('excluded_video_pair_sha256',[]))!=20):
        raise ValueError('formal production needs twelve accepted houses, sixty parents each and the twenty-pilot exclusion list')
    if not 0<=a.shard_index<a.shard_count:raise ValueError('invalid shard allocation')
    end=total if a.end_index is None else a.end_index
    if not 0<=a.start_index<end<=total:raise ValueError('invalid index interval')
    producer=validate_versions(c) or str(REPO/'tools/qa/produce_test_panel_pilot.py')
    require_canary_acceptance(c)
    if c.get('split')=='test' and (not c.get('prepared_geometry_root') or not c.get('retain_compact')):
        raise ValueError('formal production requires frozen CPU poses and per-group compact retention')
    root=a.output.resolve()
    if c.get('split')=='test' and (not c.get('allowed_output_root') or not root.is_relative_to(Path(c['allowed_output_root']).resolve())):
        raise ValueError('formal output must stay inside the explicitly configured test-only output directory')
    if (root/'public/COMMITMENTS.sha256').exists():raise ValueError('sealed output is read-only')
    if root.exists() and not a.resume:raise ValueError('output exists; explicit --resume required')
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for d in ('production','inputs','logs','evidence','scoring_private','public','rejected'): (root/d).mkdir(exist_ok=True)
    (root/'production').chmod(0o700);(root/'scoring_private').chmod(0o700);(root/'rejected').chmod(0o700)
    c['output']=str(root);shard_config=root/'inputs/shard_config.json'
    if shard_config.exists():
        if read(shard_config)!=c:raise ValueError('resume configuration differs')
    else:save(shard_config,c)
    indices=[i for i in range(a.start_index,end) if i%a.shard_count==a.shard_index]
    if not indices:raise ValueError('shard has no allocated parents')
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=f'{REPO}/src:{REPO}',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.monotonic();results=[]
    def retained_finish(cert,group,already_staged=False,accepted_attempt=None):
        cpuenv=dict(env);cpuenv['CUDA_VISIBLE_DEVICES']=''
        cmds=[]
        if not already_staged:
            cmds.append([sys.executable,str(REPO/'tools/qa/compact_frozen_panel.py'),'stage','--root',str(root),'--certificate',str(cert),'--output',str(root/'evidence'/f'retained_stage_{group}_{session}.json')])
        proof=root/'evidence'/f'retained_verify_{group}_{session}.json'
        cmds.append([sys.executable,str(REPO/'tools/qa/verify_retained_frozen_panel.py'),'--root',str(root),'--certificate',str(cert),'--output',str(proof)])
        if not (root/'retention/prune_receipts'/f'{group}.json').exists():
            cmds.append([sys.executable,str(REPO/'tools/qa/compact_frozen_panel.py'),'prune','--root',str(root),'--certificate',str(cert),'--verification',str(proof),'--commit-prune','--output',str(root/'evidence'/f'retained_prune_{group}_{session}.json')])
        for cmd in cmds:
            with (root/'logs'/f'{group}_retention_{session}.log').open('a') as f:
                subprocess.run(cmd,env=cpuenv,stdout=f,stderr=subprocess.STDOUT,check=True)
        if accepted_attempt is not None:
            # Only the successful attempt's known operational duplicates; failed attempts remain.
            files=[root/'inputs'/f'{group}_attempt_{accepted_attempt}.json']
            files += [root/'logs'/f'{group}_attempt{accepted_attempt}_{s}.log' for s in ('prepare','capture','audio','certify','verify')]
            ledger=[]
            for file in files:
                if file.is_symlink():raise ValueError('successful operational artifact is a symlink')
                if file.is_file():
                    ledger.append({'path':str(file.relative_to(root)),'sha256':digest(file),'bytes':file.stat().st_size,'allocated_bytes':file.stat().st_blocks*512})
            receipt=root/'evidence'/f'{group}_operational_prune_{session}.json'
            save(receipt,{'status':'audited_ready_to_prune','group':group,'certificate_sha256':digest(cert),'retained_verification':str(proof),'files':ledger})
            for row in ledger:
                file=root/row['path']
                if digest(file)!=row['sha256']:raise ValueError('operational artifact changed during cleanup')
                file.unlink()
            save(receipt,{'status':'pass','group':group,'certificate_sha256':digest(cert),'retained_verification':str(proof),'files':ledger})
        progress(root,f'{group} 最小见证验证通过；仅已认证父场景的审计中间件已清理。')
    session=str(time.time_ns())
    state={'pid':os.getpid(),'command':sys.argv,'gpu':a.gpu,'start_utc':stamp(),'indices':indices,'status':'running','results':results,
           'shard_index':a.shard_index,'shard_count':a.shard_count,'session':session}
    save(root/'run_state.json',state);print(json.dumps({k:v for k,v in state.items() if k!='results'}),flush=True)
    for i in indices:
        group=f'{c.get("panel_id","pilot")}_{i:03d}';g=root/'production'/group;cert=root/'scoring_private/certificates'/f'{group}.json'
        if c.get('retain_compact') and (root/'retention/groups'/f'{group}.json').exists():
            retained_finish(cert,group,already_staged=True)
            results.append({'group':group,'status':'certified_retained_resume'})
            save(root/'run_state.json',state)
            continue
        previous=[int(x.name.rsplit('_attempt',1)[1]) for x in (root/'rejected').glob(f'{group}_attempt*') if x.name.rsplit('_attempt',1)[1].isdigit()]
        first_attempt=max(previous,default=-1)+1;attempt=first_attempt;success=False
        while attempt<first_attempt+a.max_quality_attempts:
            ac=dict(c);ac['seed']=c['seed']+attempt*100003;ac['quality_retry_ordinal']=attempt;attempt_config=root/'inputs'/f'{group}_attempt_{attempt}.json';save(attempt_config,ac)
            # A retry replaces the whole parent, never individual members or answers.
            try:
                for stage in ('prepare','capture','audio','certify','verify'):
                    if stage=='prepare' and (g/'prepare_timing.json').exists():continue
                    if stage=='capture' and (g/'native_equivalence.json').exists():continue
                    if stage=='audio' and (g/'audio_timing.json').exists():continue
                    if stage=='certify' and cert.exists():continue
                    log=root/'logs'/f'{group}_attempt{attempt}_{stage}.log'
                    task_env=dict(env);task_env['CUDA_VISIBLE_DEVICES']=''
                    if stage in ('prepare','capture','audio'):
                        cmd=[sys.executable,producer,stage,'--config',str(attempt_config),'--index',str(i)]
                        if stage=='capture':task_env.pop('CUDA_VISIBLE_DEVICES',None);cmd+=['--gpu',str(a.gpu)]
                    elif stage=='certify':cmd=[sys.executable,str(REPO/'tools/qa/certify_frozen_panel.py'),'--config',str(attempt_config),'--index',str(i)]
                    else:cmd=[sys.executable,str(REPO/'tools/qa/verify_frozen_panel.py'),'--root',str(root),'--certificate',str(cert),
                              '--output',str(root/'evidence'/f'verify_{group}_attempt{attempt}_{session}.json')]
                    state.update(current_group=group,current_stage=stage,current_attempt=attempt);save(root/'run_state.json',state)
                    begin=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
                    with log.open('a') as f:rc=subprocess.call(cmd,env=task_env,stdout=f,stderr=subprocess.STDOUT)
                    after=resource.getrusage(resource.RUSAGE_CHILDREN)
                    save(root/'evidence'/f'{group}_attempt{attempt}_{stage}_{session}_execution.json',
                      {'group':group,'attempt':attempt,'stage':stage,'command':cmd,'returncode':rc,
                       'wall_seconds':time.monotonic()-begin,'cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime})
                    progress(root,f'{group} / {stage} 阶段结束，退出码 {rc}；日志 `{log.relative_to(root)}`。未运行模型。')
                    if rc:raise RuntimeError(f'{stage} exit {rc}; {log}')
                success=True;break
            except RuntimeError as exc:
                reason=str(exc);last_stage=state['current_stage']
                print(json.dumps({'group':group,'attempt':attempt,'failure':reason}),flush=True)
                # Resource shortages or certificate/verification failures need diagnosis, not random retries.
                tail=(root/'logs'/f'{group}_attempt{attempt}_{last_stage}.log').read_text()
                quality=last_stage in {'prepare','capture','certify'} and any(s in tail for s in ('quota geometry unavailable','joint visual answerability rejected','requested in_fov','requested visibility','requested visibility conditions','speech clarity C50','target sentence unclear'))
                if not quality or cert.exists():
                    results.append({'group':group,'status':'failed','reason':reason});state.update(status='stopped',end_utc=stamp(),wall_seconds=time.monotonic()-start);save(root/'run_state.json',state);sys.exit(1)
                rejected=root/'rejected'/f'{group}_attempt{attempt}';g.rename(rejected)
                save(rejected/'rejection.json',{'whole_parent_rejected':True,'reason':reason,'model_selection':False})
                attempt+=1
        if success and c.get('retain_compact'):
            retained_finish(cert,group,accepted_attempt=attempt)
        results.append({'group':group,'status':'certified' if success else 'quota_failed',
                        'quality_attempts':min(a.max_quality_attempts,attempt-first_attempt+1),
                        'quality_retry_ordinal':attempt if success else attempt-1})
        save(root/'run_state.json',state);print(json.dumps(results[-1]),flush=True)
        progress(root,f'{group} 整组结论：{results[-1]["status"]}；质量筛选依据为媒体/几何审核，未使用模型表现。')
    if c.get('retain_compact') and all(r['status']!='quota_failed' for r in results) and not (root/'retention/COMMITMENTS.sha256').exists():
        cpuenv=dict(env);cpuenv['CUDA_VISIBLE_DEVICES']=''
        subprocess.run([sys.executable,str(REPO/'tools/qa/compact_frozen_panel.py'),'freeze','--root',str(root),'--output',str(root/'evidence'/f'retention_freeze_{session}.json')],env=cpuenv,check=True)
    state.update(status='complete' if all(r['status']!='quota_failed' for r in results) else 'incomplete',end_utc=stamp(),wall_seconds=time.monotonic()-start)
    save(root/'run_state.json',state)
    save(root/'evidence'/f'worker_session_{session}.json',state)
    progress(root,f'分片 {a.shard_index}/{a.shard_count} 结束：{state["status"]}，{len(results)} 个父场景；出处 `run_state.json`。尚需评分脚本封存。')
    if state['status']!='complete':sys.exit(1)
if __name__=='__main__':main()
