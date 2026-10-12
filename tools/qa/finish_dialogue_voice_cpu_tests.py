"""Recheck baseline and final CPU scopes without accumulating studio daemon threads."""
from _dialogue_voice_runtime import *
from expand_dialogue_voices_cpu import owned_processes
import subprocess,signal,sys,argparse

def checked_stop(rows):
 for row in rows:
  if row['pid']==os.getpid():continue
  p=Path('/proc')/str(row['pid'])
  try:
   now=(p/'stat').read_text().rsplit(')',1)[1].split()
   if now[19]==row['start'] and (p/'cmdline').read_bytes()==row['command']:
    os.kill(row['pid'],signal.SIGTERM)
  except (FileNotFoundError,ProcessLookupError):pass

def run_groups(phase):
 tick=time.monotonic();register('stage2_'+phase+'_groups',STAGE/'logs'/(phase+'_groups.log'),groups=3)
 jobs=json.loads((ROOT/'JOBS.json').read_text())['jobs']
 active=[]
 for row in jobs:
  if 'worker' not in row['name'] or row.get('status')!='running':continue
  p=Path('/proc')/str(row['pid'])
  if p.exists() and (p/'stat').read_text().rsplit(')',1)[1].split()[19]==row.get('process_start_ticks'):active.append(row['name'])
 if active:raise RuntimeError('resident model workers must finish before full tests: '+str(active))
 conf=json.loads((STAGE/'config.json').read_text())
 groups=[
  (phase+'_server',['--test-path','tests/unit/test_studio_server.py']),
  (phase+'_tasks',['--test-path','tests/unit/test_studio_tasks.py']),
  (phase+'_main',['--ignore','tests/unit/test_studio_server.py','--ignore','tests/unit/test_studio_tasks.py']),
 ]
 receipts=[];peak=0
 wt=Path(conf['worktree'])
 for label,extra in groups:
  completed=STAGE/'evidence'/(label+'_pre_affinity.json')
  if phase=='final' and label in {'final_server','final_tasks'} and completed.exists():
   previous=json.loads(completed.read_text())
   assert not previous['failure_ids'] and not previous['error_ids']
   canonical=STAGE/'evidence'/(label+'.json')
   assert not canonical.exists()
   shutil.copyfile(completed,canonical)
   receipts.append(previous)
   progress('复用重启前已完成测试 '+label+'：通过 '+str(previous['passed'])+'；原收据不改，不重复跑。')
   continue
  if time.time()>conf['deadline_epoch']-300:raise RuntimeError('final test deadline reserve five minutes')
  log=STAGE/'logs'/(label+'.log');timer=STAGE/'evidence'/(label+'_time.txt')
  if log.exists() or timer.exists() or (STAGE/'evidence'/(label+'.json')).exists():raise RuntimeError('preserve existing final group evidence before a rerun')
  cmd=['/usr/bin/time','-v','-o',str(timer),'/data/jzy/envs/room-selection-merged-20261003/bin/python','-u',str(STAGE/'scripts/test_runner_guarded.py'),label,str(STAGE/'evidence')]+extra
  with log.open('w') as stream:
   child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,cwd=wt)
   p=Path('/proc')/str(child.pid)
   job('stage2_'+label+'_tests',pid=child.pid,process_start_ticks=(p/'stat').read_text().rsplit(')',1)[1].split()[19],command=cmd,log=str(log),status='running',started_utc=stamp(),device='cpu',nice=10)
   while child.poll() is None:
    processes=owned_processes();threads=sum(r['threads'] for r in processes);peak=max(peak,threads)
    if threads>32:
     save(STAGE/'evidence/final_groups_thread_stop.json',{'timestamp':stamp(),'threads':threads,'processes':[dict(r,command=r['command'].replace(b'\\0',b' ').decode()) for r in processes]})
     checked_stop(processes);child.wait();raise RuntimeError('whole final-test process tree above32; only verified descendants stopped')
    if time.time()>conf['deadline_epoch']-120:
     checked_stop(processes);child.wait();raise RuntimeError('actual deadline reserve two minutes')
    time.sleep(.2)
  path=STAGE/'evidence'/(label+'.json')
  if child.returncode not in (0,1) or not path.exists():
   job('stage2_'+label+'_tests',status='interrupted',finished_utc=stamp(),exit_code=child.returncode)
   raise RuntimeError('full-test group interrupted '+label+' '+str(child.returncode))
  receipt=json.loads(path.read_text());receipts.append(receipt)
  job('stage2_'+label+'_tests',status='finished',finished_utc=stamp(),exit_code=child.returncode,failures=len(receipt['failure_ids']),errors=len(receipt['error_ids']))
  progress('全量CPU测试分组 '+label+' 完成：失败 '+str(len(receipt['failure_ids']))+'，错误 '+str(len(receipt['error_ids']))+'。')
 baseline=json.loads((STAGE/'evidence/baseline.json').read_text())
 reports=[report for r in receipts for report in r['reports']]
 final_ids={r['nodeid'] for r in reports};baseline_ids={r['nodeid'] for r in baseline['reports']}
 missing=sorted(baseline_ids-final_ids)
 if missing:raise RuntimeError('final grouped scope omitted baseline test IDs: '+str(missing))
 failures=sorted({sid for r in receipts for sid in r['failure_ids']});errors=sorted({sid for r in receipts for sid in r['error_ids']})
 final={'label':phase,'commit':os.environ['TEST_COMMIT'],'command':['run_final_grouped.sh'],'pytest_args':receipts[0]['pytest_args'],'scope':'all repository CPU tests; same native/integration/media/canary exclusions as baseline; disjoint file groups isolate existing studio daemon queues','exit_code':1 if failures or errors else 0,'failure_ids':failures,'error_ids':errors,'passed':sum(r['passed'] for r in receipts),'skipped':sum(r['skipped'] for r in receipts),'wall_seconds':time.monotonic()-tick,'cpu_user_seconds':sum(r['cpu_user_seconds'] for r in receipts),'cpu_system_seconds':sum(r['cpu_system_seconds'] for r in receipts),'process_tree_threads_peak':peak,'os_threads_peak':max(r['os_threads_peak'] for r in receipts),'private_runtime_caps':receipts[-1]['private_runtime_caps'],'caps_by_group':{r['label']:r['private_runtime_caps'] for r in receipts},'reused_completed_groups':['final_server','final_tasks'] if phase=='final' else [],'group_receipts':[str(STAGE/'evidence'/(r['label']+'.json')) for r in receipts],'baseline_coverage_missing_ids':missing,'baseline_unique_report_ids':len(baseline_ids),'final_unique_report_ids':len(final_ids),'new_report_ids':sorted(final_ids-baseline_ids),'reports':reports,'finished_utc':stamp()}
 save(STAGE/'evidence'/(phase+'.json'),final)
 finish('stage2_'+phase+'_groups',tick,failures=len(failures),errors=len(errors),process_tree_threads_peak=peak)
 print('FULL_TEST_COUNTS',len(failures),len(errors),'NEW_FAILURES',len(set(failures)-set(baseline['failure_ids'])),'NEW_ERRORS',len(set(errors)-set(baseline['error_ids'])),flush=True)

 return final

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['baseline_recheck','final','both'],default='both');args=parser.parse_args()
 conf=json.loads((STAGE/'config.json').read_text());wt=Path(conf['worktree']);branch='codex/v7-dialogue-voices-20261012'
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.getpriority(os.PRIO_PROCESS,0)>=10
 if subprocess.check_output(['git','status','--porcelain'],cwd=wt,text=True).strip():raise RuntimeError('designated worktree must be clean before testing')
 original_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=wt,text=True).strip()
 current_branch=subprocess.check_output(['git','branch','--show-current'],cwd=wt,text=True).strip();assert current_branch==branch
 # Capture the committed observer before checking out the requested baseline.
 observer=wt/'tools/qa/run_dialogue_voice_test_suite.py'
 saved_observer=STAGE/'scripts/test_runner_guarded.py'
 if not saved_observer.exists() or digest(saved_observer)!=digest(observer):shutil.copyfile(observer,saved_observer)
 jobs=json.loads((ROOT/'JOBS.json').read_text())['jobs']
 for row in jobs:
  if 'worker' not in row['name'] or row.get('status')!='running':continue
  p=Path('/proc')/str(row['pid'])
  if p.exists() and (p/'stat').read_text().rsplit(')',1)[1].split()[19]==row.get('process_start_ticks'):raise RuntimeError('resident worker remains active before baseline checkout')
 if args.phase in ['baseline_recheck','both']:
  try:
   subprocess.run(['git','switch','--detach','9e4e63de893692883ba735190449edc70a4a5184'],cwd=wt,check=True)
   os.environ['TEST_COMMIT']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=wt,text=True).strip()
   run_groups('baseline_recheck')
  finally:
   subprocess.run(['git','switch',branch],cwd=wt,check=True)
   assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=wt,text=True).strip()==original_commit
 if args.phase in ['final','both']:
  os.environ['TEST_COMMIT']=original_commit
  run_groups('final')

if __name__=='__main__':main()

