"""Read-only audit of interrupted jobs, mouth qualifications and admitted audio links."""
import argparse,json,hashlib,datetime,subprocess,os
from pathlib import Path
from v7_roster_disk import check,task_bytes
def audio_bindings(index_path):
 index_path=Path(index_path);idx=json.loads(index_path.read_text());source=Path(idx['recorded_admission_read_only']);raw=source.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=idx['recorded_admission_sha256']:raise ValueError('families admission changed relative to its current delivery index')
 data=json.loads(raw);out={}
 for cls in ['dog_bark','pet_bird_parrot']:
  rows=[r for r in data['records']if r['sound_class']==cls and r.get('formal_status')=='admitted_under_listed_screens'and r.get('current_policy_status')=='admitted']
  selected=[]
  for r in rows:
   norm=r.get('normalization',{});path=Path(norm.get('path','missing'))
   if norm.get('status')!='pass'or not path.is_file()or hashlib.sha256(path.read_bytes()).hexdigest()!=norm.get('sha256'):raise ValueError('admitted normalized audio missing/changed: '+r['sound_asset_id'])
   selected.append({'sound_asset_id':r['sound_asset_id'],'original_id':r.get('source_original_id',r.get('original_id')),'sound_class':cls,'normalized_path':str(path),'normalized_sha256':norm['sha256'],'source_uri':r['source_uri'],'license':r.get('source_license',r.get('license')),'attribution':r.get('attribution',r.get('source_metadata',{})),'formal_status':r['formal_status'],'source_dataset':r.get('source_dataset'),'label_provenance':r.get('label_provenance',r.get('provenance_status')),'activity_intervals_samples':r.get('activity_intervals_samples',r.get('activity',{}).get('source_activity_intervals_samples'))})
  out[cls]={'count':len(selected),'records':selected,'source_species_scope':'dog bark generic, recording dog breed not certified'if cls=='dog_bark'else'parrot/parakeet metadata plus families listed screens; exact species not bound'}
 return {'family_index':str(index_path),'admission':str(source),'admission_sha256':hashlib.sha256(raw).hexdigest(),'classes':out,'audio_admission_owner':'v7-families','writes_to_families':False,'new_audio_generated':False,'full_spatial_episode':'not_run'}
def jobs(T):
 rows=[]
 for directory in [T/'round2/evidence',T/'round3/evidence']:
  for p in directory.glob('*.json'):
   try:d=json.loads(p.read_text())
   except (OSError,ValueError):continue
   if not isinstance(d,dict)or not isinstance(d.get('pid'),int):continue
   pid=d['pid'];proc=Path('/proc')/str(pid)
   try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');env=(proc/'environ').read_bytes();owned=str(T)in cmd or ('V7_HUMAN_TASK_ROOT='+str(T)).encode()in env
   except OSError:cmd=None;owned=False
   rows.append({'round':directory.parent.name,'pid':pid,'command_at_launch':d.get('command'),'launch_utc':d.get('started_utc',d.get('started_unix')),'receipt':str(p.relative_to(T)),'log':d.get('log',str(p.with_suffix('.log'))),'returncode_recorded':d.get('returncode'),'finished_recorded':d.get('finished_utc',d.get('finished_unix')),'measured_peak_mib':d.get('measured_peak_mib'),'currently_alive_owned':owned,'current_cmdline':cmd,'pid_state':'owned_alive'if owned else'not_alive'if cmd is None else'reused_by_another_process_do_not_touch','occupies_gpu_now':False if not owned else'needs_current_GPU_PID_check'})
 return rows
def run(T,index):
 R=T/'round4';check(T,4*1024**2);a=audio_bindings(index);(R/'evidence/animal_audio_binding.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
 j=jobs(T);gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
 q={'machine':os.uname().nodename,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'historical_jobs':j,'current_gpu_process_query':gpu,'round2_apparent_bytes_retained':task_bytes(T/'round2'),'disk_state':check(T),'qualification':'Historical receipt PIDs checked against /proc command and task-root environment. Missing returncode remains unknown; completed artifact receipts alone do not establish clean process exit.'}
 (R/'evidence/round2_jobs_audit.json').write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n')
 print('AUDIT',len(j),'ROUND2',sum(r['round']=='round2'for r in j),'ALIVE_OWNED',sum(r['currently_alive_owned']for r in j),'AUDIO',{k:v['count']for k,v in a['classes'].items()})
 return q,a
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--families-index',type=Path,required=True);args=p.parse_args();run(args.task_root,args.families_index)
