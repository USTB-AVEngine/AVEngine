"""Run resumable resident CPU TTS workers with three seeds and calibrated readback."""
from _dialogue_voice_runtime import *
from copy import deepcopy
import argparse,multiprocessing,random,math,sys,threading
from concurrent.futures import ProcessPoolExecutor, wait

def level(source,output):
 import numpy as np,soundfile as sf
 from avengine.qa.v7_ready_semantic_assets import measure_loudness_one_thread
 from avengine.qa.v7_extended_admission import measure_activity
 x,rate=sf.read(source,dtype='float64',always_2d=True);initial=measure_loudness_one_thread(source);gain=min(-23-initial['input_i'],-3.02-initial['input_tp'])
 for index in range(4):
  sf.write(output,x*10**(gain/20),rate,subtype='FLOAT');observed=measure_loudness_one_thread(output)
  if observed['input_tp']>-3:gain+=-3.02-observed['input_tp'];continue
  if observed['input_i']>-22.9:gain+=-23-observed['input_i'];continue
  break
 y,sr=sf.read(output,dtype='float64',always_2d=True);error=float(np.max(np.abs(y-x*10**(gain/20))))
 assert sr==rate and x.shape==y.shape and error<=1e-6 and observed['input_tp']<=-3
 return {'status':'pass','path':str(output),'sha256':digest(output),'source_path':str(source),'source_sha256':digest(source),'sample_rate_hz':rate,'sample_count':len(x),'channels':1,'duration_s':len(x)/rate,'initial_loudness':initial,'constant_gain_db':gain,'gain_formula':'min(-23-I, -3.02-TP); same round5 safety refinements','maximum_constant_gain_error':error,'trimmed_or_repeated':False,'target_integrated_lufs':-23,'true_peak_ceiling_dbfs':-3,'readback_loudness':observed,'common_lufs_target_achieved':abs(observed['input_i']+23)<=.100000001,'true_peak_ceiling_limited':observed['input_i']<-23.100000001,'measured_activity':measure_activity(y[:,0],sr,sound_class='unknown'),'human_listening_used':False}

class Resident:
 def __init__(self,plan,threads):
  tick=time.monotonic();self.judge=WhisperJudge(threads)
  # Bound ORT locally, including both speech-tokenizer and speaker-embedding sessions.
  import onnxruntime as ort
  original=ort.InferenceSession
  def bounded(*args,**kwargs):
   options=kwargs.get('sess_options')
   if options is not None:options.inter_op_num_threads=1;options.intra_op_num_threads=1
   return original(*args,**kwargs)
  ort.InferenceSession=bounded
  from cosyvoice.cli.cosyvoice import CosyVoice2
  self.engine=CosyVoice2(plan['model_dir'],load_jit=False,load_trt=False,fp16=False);self.cached=set();self.plan=plan;self.load_seconds=time.monotonic()-tick
 def process(self,task):
  import numpy as np,soundfile as sf,torch
  from scipy.signal import resample_poly
  from avengine.qa.v7_reconcile_semantic_assets import _sound
  sid=task['sound_asset_id'];target=STAGE/'task_records'/(sid+'.json');tick=time.monotonic()
  if target.exists():
   row=json.loads(target.read_text());assert row['task']==task,'task changed after attempts began'
   if row.get('terminal'):
    if row['status']=='pass':assert row['sound']['sha256']==digest(row['sound']['path'])
    return row
   for a in row['attempts']:
    if a['status']=='running':a.update(status='interrupted',reason='previous process ended after seed reserved')
  else:row={'sound_asset_id':sid,'task':task,'attempts':[],'terminal':False,'status':'running','kind':task['expansion_kind']}
  ref=task['voice_reference'];speaker=ref['speaker_id']
  if speaker not in self.cached:
   self.engine.add_zero_shot_spk(ref['source_transcript'],ref['source_waveform'],speaker);self.cached.add(speaker)
  for number in range(len(row['attempts'])+1,4):
   guard();seed=int(task['synthesis_config']['seed'])+(number-1)*1000003
   a={'attempt':number,'seed':seed,'status':'running','started_utc':stamp()};row['attempts'].append(a);save(target,row);attempt_tick=time.monotonic()
   try:
    torch.manual_seed(seed);np.random.seed(seed%(2**32));random.seed(seed);chunks=[]
    with torch.inference_mode():
     for result in self.engine.inference_zero_shot(task['transcript'],ref['source_transcript'],ref['source_waveform'],zero_shot_spk_id=speaker,stream=False,speed=1.0,text_frontend=False):chunks.append(result['tts_speech'].detach().cpu().numpy().reshape(-1))
    if not chunks:raise ValueError('synthesis returned no complete speech')
    complete=np.concatenate(chunks);rate=int(self.engine.sample_rate);g=math.gcd(rate,16000);wave=resample_poly(complete,16000//g,rate//g)
    if len(wave)!=math.ceil(len(complete)*16000/rate) or not np.isfinite(wave).all():raise ValueError('full resampling frame count/finite check failed')
    raw=STAGE/'audio'/(sid+f'__attempt_{number}_raw.wav');norm=STAGE/'audio'/(sid+f'__attempt_{number}_normalized.wav');sf.write(raw,wave,16000,subtype='FLOAT');normalization=level(raw,norm)
    checked=self.judge.evaluate(norm,task['transcript'],task['semantic_answer']['critical_semantic_slots']);checked['ASR'].update(source=str(target),checked_after_constant_gain_normalization=True)
    passed=checked['language_gate']['status']=='pass';a.update(status='pass' if passed else 'blocked',reason=None if passed else '+'.join(checked['language_gate']['reasons']),path=str(norm),raw_path=str(raw),normalization=normalization,ASR=checked['ASR'],language_identification=checked['language_identification'],language_gate=checked['language_gate'],source_sample_rate_hz=rate,source_sample_count=len(complete),wall_seconds=time.monotonic()-attempt_tick,finished_utc=stamp())
    if passed:
     sound=_sound(task,{'path':str(raw),'seed':seed,'attempt':number},checked['ASR'],self.plan,None);sound['original_synthesis_readback']=deepcopy(sound['synthesis_readback']);sound['original_synthesis_readback'].update(source_sample_rate_hz=rate,source_sample_count=len(complete));sound['original_synthesis_readback'].pop('source_sample_count_provenance',None)
     sound['synthesis_readback'].update(path=str(norm),sample_count=len(wave),sample_rate_hz=16000,duration_s=len(wave)/16000,source_sample_rate_hz=rate,source_sample_count=len(complete));sound['synthesis_readback'].pop('source_sample_count_provenance',None)
     sound.update(path=str(norm),sha256=digest(norm),source_normalization=normalization,automatic_content_readback=checked['ASR'],language_identification=checked['language_identification'],language_gate=checked['language_gate'],attempts_performed=number,selected_seed=seed,selected_attempt=number,attempt_receipt=str(target),voice_accent=ref['accent'],authored_line_id=task['authored_line_id'])
     sound['synthesis_config'].update(actual_waveform=str(norm),raw_complete_synthesis_waveform=str(raw),model_license='Apache-2.0',model_weight_license_receipt=str(ROOT/'license_receipt.json'));sound['normalization_reconciliation']={'method':'fresh_synthesis_then_complete_constant_gain_normalization_and_fresh_blind_ASR','selected_attempt':number};row.update(sound=sound,status='pass',terminal=True)
   except Exception as error:a.update(status='blocked',reason='synthesis_or_quality_exception:'+type(error).__name__+':'+str(error),wall_seconds=time.monotonic()-attempt_tick,finished_utc=stamp())
   save(target,row);print('ATTEMPT',sid,number,seed,a['status'],a.get('reason'),a['wall_seconds'],flush=True)
   if row.get('terminal'):break
  if not row.get('terminal'):row.update(status='dropped',terminal=True,sound=None,reason=';'.join(a.get('reason') or a['status'] for a in row['attempts']))
  row.update(attempts_performed=len(row['attempts']),wall_seconds=time.monotonic()-tick,finished_utc=stamp());save(target,row);return row

def worker(index,workers,tasks,threads,phase):
 logfile=STAGE/'logs'/(f'{phase}_worker_{index}.log');log=logfile.open('a',buffering=1);sys.stdout=log;sys.stderr=log;name=f'stage2_{phase}_worker_{index}';tick=time.monotonic();register(name,logfile,threads=threads)
 peak=[len(list(Path('/proc/self/task').iterdir()))];done=threading.Event()
 def monitor():
  while not done.wait(.1):peak[0]=max(peak[0],len(list(Path('/proc/self/task').iterdir())))
 watcher=threading.Thread(target=monitor,daemon=True) if phase=='pilot' else None
 if watcher is not None:watcher.start()
 plan=json.loads((STAGE/'expansion_plan.json').read_text());resident=Resident(plan,threads);results=[]
 for t in tasks[index::workers]:
  r=resident.process(t);peak[0]=max(peak[0],len(list(Path('/proc/self/task').iterdir())));results.append({'id':r['sound_asset_id'],'status':r['status'],'wall_seconds':r['wall_seconds'],'attempts':r['attempts_performed'],'kind':r['kind']})
  if len(results)%10==0:progress(f'{phase} worker {index} 完成 {len(results)} 条，通过 {sum(r["status"]=="pass" for r in results)} 条。')
 done.set()
 if watcher is not None:watcher.join()
 finish(name,tick,checked=len(results),passed=sum(r['status']=='pass' for r in results),model_load_seconds=resident.load_seconds,os_threads_peak_sampled=peak[0]);return {'results':results,'model_load_seconds':resident.load_seconds,'os_threads_peak_sampled':peak[0]}

def owned_processes():
 records={};pending=[os.getpid()]
 while pending:
  pid=pending.pop()
  if pid in records:continue
  directory=Path('/proc')/str(pid)
  try:
   fields=(directory/'stat').read_text().rsplit(')',1)[1].split()
   if pid!=os.getpid() and int(fields[1]) not in records:continue
   records[pid]={'pid':pid,'parent':int(fields[1]),'start':fields[19],'threads':int(fields[17]),'command':(directory/'cmdline').read_bytes()}
   for thread in (directory/'task').iterdir():
    try:pending.extend(int(child) for child in (thread/'children').read_text().split())
    except (FileNotFoundError,ProcessLookupError):pass
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 return list(records.values())


def supervised_results(pool,workers,tasks,threads,phase):
 futures=[pool.submit(worker,i,workers,tasks,threads,phase) for i in range(workers)]
 peak=0;observations=0
 while True:
  processes=owned_processes();count=sum(row['threads'] for row in processes);peak=max(peak,count);observations+=1
  if count>32:
   import signal
   receipt={'threads':count,'timestamp':stamp(),'processes':[dict(r,command=r['command'].replace(b'\0',b' ').decode()) for r in processes],'verified_stops':[]}
   for row in processes:
    if row['pid']==os.getpid():continue
    directory=Path('/proc')/str(row['pid'])
    try:
     current=(directory/'stat').read_text().rsplit(')',1)[1].split()
     if current[19]!=row['start'] or (directory/'cmdline').read_bytes()!=row['command']:continue
     os.kill(row['pid'],signal.SIGTERM);receipt['verified_stops'].append({'pid':row['pid'],'start_ticks':row['start'],'command':row['command'].replace(b'\0',b' ').decode()})
    except (FileNotFoundError,ProcessLookupError):pass
   save(STAGE/'evidence'/(phase+'_thread_stop.json'),receipt)
   raise RuntimeError('own process tree exceeded 32 OS threads; only verified descendants stopped')
  if all(f.done() for f in futures):break
  wait(futures,timeout=1)
 save(STAGE/'evidence'/(phase+'_resources.json'),{'process_tree_threads_peak':peak,'samples':observations,'workers':workers,'model_threads_each':threads,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'parent_nice':os.getpriority(os.PRIO_PROCESS,0),'includes_ffmpeg_and_controller_threads':True})
 return [f.result() for f in futures]

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilot','batch','replacement'],required=True);p.add_argument('--workers',type=int,default=1);p.add_argument('--threads',type=int,default=2);a=p.parse_args();guard();assert json.loads((ROOT/'language_gate_calibration.json').read_text())['status']=='pass';assert json.loads((ROOT/'license_receipt.json').read_text())['status']=='pass';assert a.workers*a.threads<=32
 plan=json.loads((STAGE/'expansion_plan.json').read_text());tasks=plan['tasks'];pilot_ids=set(plan['pilot_ids'])
 if a.phase=='pilot':tasks=[plan['repair_task']]+[t for t in tasks if t['sound_asset_id'] in pilot_ids]
 elif a.phase=='replacement':tasks=json.loads((STAGE/'replacement_tasks.json').read_text())
 else:tasks=[t for t in tasks if not (STAGE/'task_records'/(t['sound_asset_id']+'.json')).exists() or not json.loads((STAGE/'task_records'/(t['sound_asset_id']+'.json')).read_text()).get('terminal')]
 tick=time.monotonic();register('stage2_'+a.phase+'_parent',STAGE/'logs'/(a.phase+'.log'),workers=a.workers,threads_per_worker=a.threads)
 with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:results=supervised_results(pool,a.workers,tasks,a.threads,a.phase)
 rows=[r for w in results for r in w['results']];summary={'phase':a.phase,'workers':a.workers,'threads_per_worker':a.threads,'processed':len(rows),'passed':sum(r['status']=='pass' for r in rows),'dropped':sum(r['status']=='dropped' for r in rows),'wall_seconds':time.monotonic()-tick,'workers_metrics':[{k:v for k,v in w.items() if k!='results'} for w in results],'rows':rows,'finished_utc':stamp()}
 if a.phase=='pilot':
  measured=[r['wall_seconds'] for r in rows if r['kind']=='new_voice'];summary.update(pilot_line_count=len(measured),mean_line_seconds=sum(measured)/len(measured),min_line_seconds=min(measured),max_line_seconds=max(measured),repair_result=next(r for r in rows if r['kind']=='old_language_repair'))
 save(STAGE/(a.phase+'_summary.json'),summary);finish('stage2_'+a.phase+'_parent',tick,processed=len(rows));progress(f'{a.phase} 阶段收尾：{len(rows)} 条，通过 {summary["passed"]}，丢弃 {summary["dropped"]}。');print('PHASE_SUMMARY',json.dumps(summary),flush=True)
if __name__=='__main__':main()
