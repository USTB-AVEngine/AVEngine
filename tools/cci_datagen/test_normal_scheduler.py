#!/usr/bin/env python3
"""CPU-only integration of the unchanged ordinary worker and one probe lease."""
import argparse,json,os,subprocess,sys,time,uuid
from pathlib import Path
from resources import write_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--base');p.add_argument('--launcher-python');p.add_argument('--worker');p.add_argument('--cpu-peer-receipt',action='store_true');p.add_argument('--generator-profile');p.add_argument('--output');a=p.parse_args()
 if a.cpu_peer_receipt:
  out=Path(a.output);out.mkdir(parents=True,exist_ok=False);time.sleep(12)
  (out/'input_manifest.jsonl').write_text('{"id":"cpu_peer_fixture"}\n');(out/'predictions.jsonl').write_text('{"id":"cpu_peer_fixture","status":"completed_cpu_only_fixture"}\n');write_json(out/'run.json',{'status':'complete','count':1,'is_training':False,'fixture':'keeps second CPU worker alive for read-only rank observation'});return
 base=Path(a.base);name='dev_ordinary_'+uuid.uuid4().hex[:8];evidence=base/'evidence'/name;evidence.mkdir(parents=True)
 source=base/'code/tools/cci_datagen/plans/cci_datagen_probe_shared_20261003_codex_r2.json';plan=json.loads(source.read_text());plan.update(name=name,run_root=str(evidence/'scheduler_output'))
 peer={'name':'cpu_peer_fixture','evals':[{'name':'job','checkpoint':str(base/'configs/generator_v6_pilot_r2.json'),'checkpoint_flag':'--generator-profile','args':[str(Path(__file__).resolve()),'--cpu-peer-receipt']}]};plan['experiments'].append(peer);plan_path=evidence/'plan.json';write_json(plan_path,plan)
 env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',LADDER_JOB_ID=name,LADDER_WORKERS_PER_GPU='2',OMP_NUM_THREADS='1',PYTHONPATH=str(base/'code'))
 cmd=[a.launcher_python,'-B','-m','torch.distributed.run','--standalone','--nnodes','1','--nproc-per-node','2',a.worker,'--plan',str(plan_path),'--job-id',name]
 with (evidence/'torchrun.log').open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=90)
 probe=list((evidence/'scheduler_output/datagen_probe').glob('eval_job_*/probe_summary.json'))
 if len(probe)!=1:raise RuntimeError('ordinary scheduler did not complete exactly one probe experiment')
 result=json.loads(probe[0].read_text());runs=list((evidence/'scheduler_output').glob('*/eval_job_*/run.json'))
 if len(runs)!=2 or not all(json.loads(p.read_text())['status']=='complete' for p in runs):raise RuntimeError('ordinary receipt ABI did not accept both CPU fixtures')
 if result['assignment']['generation_scheduler_ranks']!=1 or result['assignment']['torchrun_children_launched']!=0:raise RuntimeError('generation entry stole additional scheduler ranks')
 if result['observed_rank_count']!=2 or result['lease']['local_world_size']!=2:raise RuntimeError('probe failed to observe both colocated CPU workers')
 write_json(evidence/'SUMMARY.json',{'status':'pass','task':'unchanged ordinary worker CPU integration','world_size':2,'observed_ranks':2,'generation_claims':1,'generation_nested_torchrun':0,'usable_gpus':0,'native_worlds_generated':0,'probe_summary':str(probe[0]),'command':cmd,'log':str(evidence/'torchrun.log')});print(evidence/'SUMMARY.json')
if __name__=='__main__':main()
