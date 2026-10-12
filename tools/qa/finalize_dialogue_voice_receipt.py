"""Final evidence receipt; run after synthesis and full-test processes have exited."""
from _dialogue_voice_runtime import *
import subprocess, re
from html.parser import HTMLParser

def read_json(path):return json.loads(Path(path).read_text())

def timer(path):
 text=Path(path).read_text()
 def field(label):
  match=re.search(re.escape(label)+r':\s*([0-9.]+)',text)
  if not match:raise RuntimeError('incomplete CPU timer '+str(path))
  return float(match.group(1))
 return {'source':str(path),'cpu_user_seconds':field('User time (seconds)'),'cpu_system_seconds':field('System time (seconds)')}

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.getpriority(os.PRIO_PROCESS,0)>=10
 conf=read_json(STAGE/'config.json');wt=Path(conf['worktree']);summary=read_json(ROOT/'expansion_summary.json');manifest=read_json(ROOT/'semantic_manifest.json')
 license_receipt=read_json(ROOT/'license_receipt.json');assert license_receipt['status']=='pass' and license_receipt['output_use_for_AI_training_restriction_found'] is False
 old=read_json(R/'natural_batch_ready/semantic_manifest.json')
 assert digest(R/'natural_batch_v1/plan.json')=='4041f4f8bd66d7f8d9e597a16c58c97da8d2cff7ddf61f2794775f2afd762322'
 assert digest(R/'natural_batch_ready/semantic_manifest.json')=='f630af767fbf5f26085cfe52f926339384cf1d61f926f76070effe30802fe19e'
 assert all(digest(s['path'])==s['sha256'] for s in old['sounds'])
 assert digest('/data/datasets/VCTK-Corpus-0.92/speaker-info.txt')=='ee6447edab776828e0c2e726c04a4965c4602c32047b14fac2f5294d18728eee'
 sound_issues=[]
 for s in manifest['sounds']:
  if digest(s['path'])!=s['sha256']:sound_issues.append({'id':s['sound_asset_id'],'reason':'audio_hash_changed'})
  if s['automatic_content_readback'].get('audio_sha256')!=s['sha256']:sound_issues.append({'id':s['sound_asset_id'],'reason':'blind_ASR_not_bound_to_audio_SHA'})
  if s['automatic_content_readback'].get('reference_text_or_answer_prompted_into_ASR') is not False:sound_issues.append({'id':s['sound_asset_id'],'reason':'ASR_blind_flag_not_false'})
  if s['language_identification'].get('free_language_forced') is not False:sound_issues.append({'id':s['sound_asset_id'],'reason':'free_language_forced'})
 assert not sound_issues,sound_issues
 records=[read_json(p) for p in (STAGE/'task_records').glob('*.json')]
 assert all(len(r['attempts'])<=3 and len({a['seed'] for a in r['attempts']})==len(r['attempts']) for r in records)
 # Allow-list only: all recorded PIDs plus descendants in this task's guardian receipts.
 jobs=read_json(ROOT/'JOBS.json');allowed={}
 for row in jobs['jobs']:
  if 'pid' in row:allowed[(row['pid'],str(row.get('process_start_ticks','')))]=row['name']
 def gather(value,source):
  if isinstance(value,dict):
   if isinstance(value.get('pid'),int) and ('start' in value or 'process_start_ticks' in value):
    allowed[(value['pid'],str(value.get('start') or value.get('process_start_ticks')))]=source
   for child in value.values():gather(child,source)
  elif isinstance(value,list):
   for child in value:gather(child,source)
 for p in (STAGE/'evidence').glob('*thread_stop*.json'):gather(read_json(p),str(p))
 process_audit=[]
 for (pid,start),source in sorted(allowed.items()):
  p=Path('/proc')/str(pid);active=False;state='absent'
  if p.exists():
   stat=(p/'stat').read_text().rsplit(')',1)[1].split();active=stat[19]==start;state='same_owned_process_alive' if active else 'pid_reused_unowned'
  process_audit.append({'pid':pid,'expected_start_ticks':start,'source':source,'state':state,'owned_process_alive':active})
 assert not any(r['owned_process_alive'] for r in process_audit),'recorded own process still alive; do not claim delivery closed'
 save(STAGE/'evidence/final_process_audit.json',{'timestamp':stamp(),'scope':'recorded task PIDs and recorded guardian descendants only; no broad process scan or signal','owned_processes_alive':0,'processes':process_audit})
 # Preserve inconsistent historical bookkeeping, then annotate instead of inventing exit times.
 annotations=[]
 for row in jobs['jobs']:
  if row.get('finished_utc') and row.get('started_utc') and row['finished_utc']<row['started_utc']:
   annotations.append({'name':row['name'],'old_finished_utc':row['finished_utc'],'reason':'stale field inherited when prior job record was reused; actual exact exit time unverified'})
   row['stale_finished_utc']=row.pop('finished_utc')
  if row.get('status')=='running':
   row.update(status='interrupted',exit_verified_utc=stamp(),reason='PID/start identity is no longer alive; no normal finish receipt was recorded')
 save(STAGE/'evidence/job_history_annotations.json',annotations);save(ROOT/'JOBS.json',jobs)
 names=['baseline_time.txt','calibration_time.txt','references_time_interrupted.txt','references_time.txt','plan_time.txt','pilot_time_startup_initial.txt','pilot_time_startup_rayon.txt','pilot_time.txt','batch_time_pre_encoder.txt','encoder_equivalence_time.txt','batch_time_pre_single_thread.txt','single_thread_pilot_time.txt','batch_time.txt','final_time.txt','related_tests_time.txt','related_final_time.txt','all_grouped_test_phases_pre_affinity_time.txt','all_grouped_test_phases_time.txt']
 meters=[timer(STAGE/'evidence'/name) for name in names if (STAGE/'evidence'/name).exists()]
 for name in ['batch_pipe_cpu_time.json','interim_report_cpu.json','final_report_cpu.json']:
  p=STAGE/'evidence'/name
  if p.exists():
   data=read_json(p);meters.append({'source':str(p),'cpu_user_seconds':data['cpu_user_seconds'],'cpu_system_seconds':data['cpu_system_seconds']})
 cpu_seconds=sum(m['cpu_user_seconds']+m['cpu_system_seconds'] for m in meters)
 physical=subprocess.check_output(['du','-s','-B1',str(ROOT),str(wt)],text=True).splitlines()
 apparent=subprocess.check_output(['du','-sb',str(ROOT),str(wt)],text=True).splitlines()
 physical_bytes=sum(int(row.split()[0]) for row in physical);apparent_bytes=sum(int(row.split()[0]) for row in apparent);free=shutil.disk_usage('/data').free
 code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=wt,text=True).strip();code_status=subprocess.check_output(['git','status','--porcelain'],cwd=wt,text=True).strip()
 files=subprocess.check_output(['git','diff','--name-only','9e4e63de893692883ba735190449edc70a4a5184','HEAD'],cwd=wt,text=True).splitlines()
 assert not code_status
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.links=[];self.external=[]
  def handle_starttag(self,tag,attrs):
   values=dict(attrs)
   if tag=='a' and values.get('href'):self.links.append(values['href'])
   if tag in {'script','img','link'} and (values.get('src') or values.get('href')):self.external.append(values.get('src') or values.get('href'))
 page=Links();page.feed((ROOT/'report.html').read_text());assert not page.external
 missing=[href for href in page.links if not (ROOT/href).exists() and href!='delivery_receipt.json'];assert not missing,missing
 started=datetime.datetime.fromisoformat(conf['started_utc']).timestamp();elapsed=time.time()-started
 resource_events=[]
 for name in ['reference_budget_stop.json','pilot_thread_stop_startup_initial.json','pilot_thread_stop_startup_rayon.json','final_thread_violation.json','baseline_monitor_issue.json','final_groups_thread_stop_ffmpeg.json']:
  p=STAGE/'evidence'/name
  if p.exists():resource_events.append({'source':str(p),'details':read_json(p) if 'startup' not in name else {k:v for k,v in read_json(p).items() if k in {'threads','timestamp'}}})
 receipt={'machine':'48g','connection':'ssh 48g-jump','status':summary['status'],'started_utc':conf['started_utc'],'finished_utc':stamp(),'wall_seconds_this_round':elapsed,'deadline_utc':conf['deadline_utc'],'interruption_authorized_extension_seconds':conf.get('interruption_authorized_extension_seconds',0),'within_authorized_wall_budget':elapsed<=18000+conf.get('interruption_authorized_extension_seconds',0),'wall_seconds_after_authorized_interruption_allowance':elapsed-conf.get('interruption_authorized_extension_seconds',0),'code_worktree':str(wt),'branch':'codex/v7-dialogue-voices-20261012','base_commit':'9e4e63de893692883ba735190449edc70a4a5184','code_commit':code_commit,'code_status_clean':not code_status,'pushed':False,'changed_code_files':files,'summary':summary,'cpu_metered_seconds':cpu_seconds,'cpu_metered_core_hours':cpu_seconds/3600,'CPU_accounting_method':'sum major /usr/bin/time-v phase receipts plus pipe-controller getrusage SELF+reaped CHILDREN; duplicate preserved timers and nested grouped-subtest timers excluded','CPU_accounting_limits':'small setup/read-only validation/Git/SSH commands not individually metered; startup orphan compiler children may not be fully included','CPU_sources':meters,'prior_stage_CPU_receipt':str(STAGE/'evidence/stage1_delivery_receipt.json'),'disk_physical_bytes_ROOT_plus_worktree':physical_bytes,'disk_apparent_bytes_ROOT_plus_worktree':apparent_bytes,'disk_command_sources':{'physical':['du','-s','-B1',str(ROOT),str(wt)],'apparent':['du','-sb',str(ROOT),str(wt)]},'data_free_bytes':free,'disk_limits_satisfied':physical_bytes<=20000000000 and apparent_bytes<=20000000000 and free>=150000000000,'all_530_original_audio_SHA_verified_unchanged':True,'old_plan_and_manifest_SHA_verified_unchanged':True,'accepted_audio_ASR_SHA_binding_issues':sound_issues,'maximum_attempts_new_or_repair':max(len(r['attempts']) for r in records),'background_owned_processes_remaining':0,'process_audit':str(STAGE/'evidence/final_process_audit.json'),'resource_guard_events':resource_events,'batch_resource_receipt':str(STAGE/'evidence/batch_pipe_resources.json'),'license_receipt':str(ROOT/'license_receipt.json'),'semantic_manifest':str(ROOT/'semantic_manifest.json'),'html':str(ROOT/'report.html'),'HTML_single_file_without_external_assets':True,'HTML_relative_evidence_links_exist':True,'not_done':['no production sampler integration','no training','no push'],'unprocessed_new_variants':len(summary['unprocessed_variant_ids'])}
 save(ROOT/'delivery_receipt.json',receipt)
 progress('交付收据完成：提交'+code_commit+'；CPU主要作业累计'+str(round(cpu_seconds/3600,3))+'核时；物理磁盘'+str(round(physical_bytes/1e9,3))+'GB；常驻进程0；验收'+summary['status']+'。')
 print('DELIVERY',json.dumps({k:receipt[k] for k in ['status','code_commit','cpu_metered_core_hours','wall_seconds_this_round','within_authorized_wall_budget','disk_physical_bytes_ROOT_plus_worktree','disk_apparent_bytes_ROOT_plus_worktree','data_free_bytes','background_owned_processes_remaining']}),flush=True)
if __name__=='__main__':main()
