"""Independently calibrate the English gate against v7 positives and Chinese negatives."""
from _dialogue_voice_runtime import *
import multiprocessing,sys
from concurrent.futures import ProcessPoolExecutor

def worker(index,workers,rows):
 logfile=STAGE/'logs'/f'calibration_worker_{index}.log';log=logfile.open('a',buffering=1);sys.stdout=log;sys.stderr=log;name=f'stage2_calibration_worker_{index}';tick=time.monotonic();register(name,logfile,threads=2);judge=WhisperJudge(2);counts={}
 for x in rows[index::workers]:
  guard();target=STAGE/'calibration_records'/(x['id']+'.json')
  if target.exists():
   r=json.loads(target.read_text());assert r['audio_sha256']==digest(x['path'])
  else:
   r=judge.evaluate(x['path'],x.get('reference'),x.get('slots'));r.update(id=x['id'],set=x['set'],source=x,terminal=True);save(target,r)
  counts[x['set']]=counts.get(x['set'],0)+1;print('RESULT',x['id'],r['language_gate']['status'],r['language_identification']['top_language'],r['ASR'].get('wer'),flush=True)
  if sum(counts.values())%25==0:progress(f'新语种校准 worker {index} 完成 {sum(counts.values())} 条。')
 finish(name,tick,checked=sum(counts.values()));return counts

def main():
 tick=time.monotonic();register('stage2_calibration_parent',STAGE/'logs/calibration.log',workers=4,threads_per_worker=2)
 old=json.loads((R/'natural_batch_ready/semantic_manifest.json').read_text())['sounds'];neg=json.loads((STAGE/'negative_selection.json').read_text())
 rows=[{'id':s['sound_asset_id'],'set':'v7_positive','path':s['path'],'reference':s['transcript'],'slots':s['semantic_answer']['critical_semantic_slots'],'expected_sha256':s['sha256']} for s in old]+[dict(x,id='negative_'+str(i).zfill(3),set='Chinese_negative') for i,x in enumerate(neg)]
 with ProcessPoolExecutor(4,mp_context=multiprocessing.get_context('spawn')) as pool:
  for c in pool.map(worker,range(4),[4]*4,[rows]*4):print('WORKER_DONE',c,flush=True)
 records=[json.loads((STAGE/'calibration_records'/(x['id']+'.json')).read_text()) for x in rows];p=[x for x in records if x['set']=='v7_positive'];n=[x for x in records if x['set']=='Chinese_negative'];failed=[x['id'] for x in p if x['language_gate']['status']!='pass']
 expected_bad='daily_reminder_home_living_room__role_2';pospass=len(p)==530 and failed==[expected_bad] and all(x['audio_sha256']==x['source']['expected_sha256'] for x in p);negpass=len(n)>=100 and all(x['language_gate']['status']=='blocked' for x in n)
 result={'status':'pass' if pospass and negpass else 'blocked','rules':json.loads((STAGE/'config.json').read_text())['language_rule'],'English_probability_threshold':None,'fresh_forced_English_ASR':True,'fresh_free_language_ASR':True,'fresh_language_identification':True,'positive_expected_count':530,'positive_checked':len(p),'positive_passed':len(p)-len(failed),'positive_rejected_ids':failed,'positive_calibration_pass':pospass,'negative_selected_by_owner_scan_content':True,'negative_checked':len(n),'negative_rejected':sum(x['language_gate']['status']=='blocked' for x in n),'negative_calibration_pass':negpass,'negative_top_zh':sum(x['language_identification']['top_language']=='zh' for x in n),'negative_free_has_Han':sum(has_han(x['language_identification']['transcript_free_language']) for x in n),'records_directory':str(STAGE/'calibration_records'),'source_manifest':str(R/'natural_batch_ready/semantic_manifest.json'),'source_manifest_sha256':digest(R/'natural_batch_ready/semantic_manifest.json'),'negative_selection':str(STAGE/'negative_selection.json'),'finished_utc':stamp(),'wall_seconds':time.monotonic()-tick,'unpassed_positives':[x for x in p if x['language_gate']['status']!='pass']}
 save(ROOT/'language_gate_calibration.json',result);progress(f'新语种校准收尾：正例 {result["positive_passed"]}/530，反例拒绝 {result["negative_rejected"]}/{len(n)}；校准 {result["status"]}。');finish('stage2_calibration_parent',tick,status='finished',calibration_status=result['status']);print('CALIBRATION',json.dumps({k:v for k,v in result.items() if k!='unpassed_positives'}),flush=True)
if __name__=='__main__':main()
