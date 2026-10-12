"""Use resident CPU workers and pipes to keep six-worker scheduling below32 threads."""
from _dialogue_voice_runtime import *
from expand_dialogue_voices_cpu import Resident,owned_processes
import multiprocessing
from multiprocessing.connection import wait
import sys,signal,traceback


def pipe_worker(index,connection):
 name='stage2_batch_pipe_worker_'+str(index);logfile=STAGE/'logs'/('batch_pipe_worker_'+str(index)+'.log')
 log=logfile.open('a',buffering=1);sys.stdout=log;sys.stderr=log
 tick=time.monotonic();register(name,logfile,threads=1);results=[];status='finished'
 try:
  plan=json.loads((STAGE/'expansion_plan.json').read_text());resident=Resident(plan,1)
  while True:
   task=connection.recv()
   if task is None:break
   row=resident.process(task)
   result={'id':row['sound_asset_id'],'status':row['status'],'attempts':row['attempts_performed'],'wall_seconds':row['wall_seconds']}
   results.append(result);connection.send({'kind':'result','row':result})
   if len(results)%10==0:progress('pipe worker '+str(index)+' 完成 '+str(len(results))+' 条，通过 '+str(sum(r['status']=='pass' for r in results))+'。')
 except Exception:
  status='interrupted';error=traceback.format_exc();print(error,flush=True)
  connection.send({'kind':'error','traceback':error})
 finally:
  finish(name,tick,status=status,checked=len(results),passed=sum(r['status']=='pass' for r in results),model_load_seconds=resident.load_seconds if 'resident' in locals() else None)
  connection.close()


def stop_verified(processes):
 stopped=[]
 for row in reversed(processes):
  if row['pid']==os.getpid():continue
  p=Path('/proc')/str(row['pid'])
  try:
   current=(p/'stat').read_text().rsplit(')',1)[1].split()
   if current[19]==row['start'] and (p/'cmdline').read_bytes()==row['command']:
    os.kill(row['pid'],signal.SIGTERM);stopped.append(row['pid'])
  except (FileNotFoundError,ProcessLookupError):pass
 return stopped


def main():
 guard();assert 'torch' not in sys.modules,'fork controller must not import torch'
 assert json.loads((ROOT/'language_gate_calibration.json').read_text())['status']=='pass'
 assert json.loads((ROOT/'license_receipt.json').read_text())['status']=='pass'
 plan=json.loads((STAGE/'expansion_plan.json').read_text())
 tasks=[t for t in plan['tasks'] if not (STAGE/'task_records'/(t['sound_asset_id']+'.json')).exists() or not json.loads((STAGE/'task_records'/(t['sound_asset_id']+'.json')).read_text()).get('terminal')]
 workers=min(6,len(tasks));tick=time.monotonic();name='stage2_batch_pipe_parent'
 register(name,STAGE/'logs/batch_pipe.log',workers=workers,threads_per_worker=1,controller='single native thread with pipes; no queue feeders or multiprocessing resource_tracker')
 conf=json.loads((STAGE/'config.json').read_text());context=multiprocessing.get_context('fork');pairs=[];processes=[];index=0;rows=[];peak=0;samples=0;interrupted=False
 try:
  for number in range(workers):
   parent_conn,child_conn=context.Pipe()
   process=context.Process(target=pipe_worker,args=(number,child_conn));process.start();child_conn.close()
   parent_conn.send(tasks[index]);index+=1;pairs.append(parent_conn);processes.append(process)
  active=set(pairs)
  while active:
   owned=owned_processes();threads=sum(p['threads'] for p in owned);peak=max(peak,threads);samples+=1
   if threads>32:
    stops=stop_verified(owned)
    save(STAGE/'evidence/batch_pipe_thread_stop.json',{'timestamp':stamp(),'threads':threads,'verified_stopped_pids':stops,'processes':[dict(p,command=p['command'].replace(bytes([0]),b' ').decode()) for p in owned]})
    raise RuntimeError('own complete process tree exceeded32 native threads')
   if time.time()>conf['deadline_epoch']-120:
    stop_verified(owned);raise RuntimeError('actual hard deadline reserve2 minutes')
   for connection in wait(active,timeout=.25):
    message=connection.recv()
    if message['kind']!='result':raise RuntimeError('worker error '+message['traceback'])
    rows.append(message['row'])
    if index<len(tasks):
     connection.send(tasks[index]);index+=1
    else:connection.send(None);active.remove(connection)
  for process in processes:process.join();assert process.exitcode==0
 except Exception:
  interrupted=True;stop_verified(owned_processes());raise
 finally:
  for process in processes:
   process.join(timeout=10)
   if process.is_alive():
    checked=[p for p in owned_processes() if p['pid']==process.pid]
    stop_verified(checked);process.join(timeout=10)
  for connection in pairs:connection.close()
  usage=resource.getrusage(resource.RUSAGE_SELF);child_usage=resource.getrusage(resource.RUSAGE_CHILDREN)
  save(STAGE/'evidence/batch_pipe_cpu_time.json',{'method':'getrusage SELF plus reaped CHILDREN after all fork workers joined; includes waited descendants','cpu_user_seconds':usage.ru_utime+child_usage.ru_utime,'cpu_system_seconds':usage.ru_stime+child_usage.ru_stime,'wall_seconds':time.monotonic()-tick,'finished_utc':stamp(),'interrupted':interrupted})
  save(STAGE/'evidence/batch_pipe_resources.json',{'process_tree_threads_peak':peak,'samples':samples,'workers':workers,'model_threads_each':1,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'parent_nice':os.getpriority(os.PRIO_PROCESS,0),'includes_ffmpeg_and_controller_threads':True,'controller_did_not_import_torch':True})
  finish(name,tick,status='interrupted' if interrupted else 'finished',processed=len(rows),process_tree_threads_peak=peak)
 summary={'phase':'batch_pipe','workers':workers,'threads_per_worker':1,'processed':len(rows),'passed':sum(r['status']=='pass' for r in rows),'dropped':sum(r['status']=='dropped' for r in rows),'rows':rows,'finished_utc':stamp()}
 save(STAGE/'batch_pipe_summary.json',summary);progress('6个单线程pipe worker收尾：'+str(len(rows))+'条，通过'+str(summary['passed'])+'，丢弃'+str(summary['dropped'])+'；进程树线程峰值'+str(peak)+'。')
 print('BATCH_PIPE_SUMMARY',json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
