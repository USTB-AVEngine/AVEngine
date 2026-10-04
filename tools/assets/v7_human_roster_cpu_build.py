"""Resumable, bounded CPU-only native builds in the caller's task directory."""
import argparse,concurrent.futures,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--blender',type=Path,required=True);p.add_argument('--legacy-root',type=Path,required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args();T=a.task_root.resolve();rows=json.loads((T/'registrations/selection.json').read_text())['avatars'];script=Path(__file__).with_name('v7_human_roster_blender.py')
assert 1<=a.workers<=4
(T/'evidence/cpu_supervisor.json').write_text(json.dumps({'pid':os.getpid(),'command':list(__import__('sys').argv),'machine':os.uname().nodename,'workers':a.workers},indent=2))
def run(row):
 if (T/'assets/rocketbox'/row['asset_id']/'build.json').exists():return {'asset_id':row['asset_id'],'status':'reused'}
 cmd=[str(a.blender),'--background','--python-exit-code','1','--factory-startup','--threads','2','--python',str(script),'--','--mode','build','--task-root',str(T),'--legacy-root',str(a.legacy_root),'--asset-id',row['asset_id']];log=T/'logs'/('build_'+row['asset_id']+'.log');env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='');start=time.monotonic()
 with log.open('w') as f:cp=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);(T/'evidence'/('cpu_'+row['asset_id']+'_process.json')).write_text(json.dumps({'pid':cp.pid,'command':cmd,'log':str(log)},indent=2));code=cp.wait()
 receipt={'asset_id':row['asset_id'],'returncode':code,'status':'pass' if code==0 and (T/'assets/rocketbox'/row['asset_id']/'build.json').exists() else 'fail','seconds':time.monotonic()-start,'log':str(log)};print(json.dumps(receipt),flush=True);(T/'evidence'/('cpu_'+row['asset_id']+'_result.json')).write_text(json.dumps(receipt,indent=2));return receipt
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:results=list(ex.map(run,rows))
(T/'evidence/cpu_build_results.json').write_text(json.dumps(results,indent=2));raise SystemExit(any(r['status']=='fail' for r in results))
