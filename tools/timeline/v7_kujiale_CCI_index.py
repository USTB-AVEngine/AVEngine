"""Index SHA-verified own native deliveries as an additive training release."""
from pathlib import Path
from collections import Counter
import json,csv,os
ROOT=Path('/mnt/afs/xinyuanqian/jzy/hardbank/generation/v7_s_kujiale_20261005')
def atomic(p,v):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def original_budget_audit(releases):
 # Evaluate the unmodified frozen 33b5b12 function in a private namespace;
 # avoid imports/pyc writes into anyone else's checkout or environment.
 path=Path('/mnt/afs/xinyuanqian/jzy/hardbank/generation/v7_s_20261005_batch1/code/33b5b12df1a45ca265bf0c4fbe9b26b664577a7c/src/avengine/timeline/v7_production_contract.py')
 import hashlib
 source=path.read_bytes()
 if hashlib.sha256(source).hexdigest()!='89e281eceeda38720e871bc17b5aafabf2eea57318c60d1fc0c82058b488312b':raise ValueError('frozen 33b5b12 contract differs')
 namespace={'__name__':'private_frozen_v7_contract'};exec(compile(source,str(path),'exec'),namespace)
 windows=[];binding=set();speakers=set();classes=set();parent_ids=set()
 for p in releases:
  for parent in (p/'parents').iterdir():
   group=json.loads((parent/'private/counterfactual_audit.json').read_text());binding.update(group.get('verified_binding_logical_question_ids',[]))
   delivery=json.loads((parent/'delivery_manifest.json').read_text())
   for win in delivery['windows']:
    wid=win['window_id'];q=json.loads((parent/'windows'/wid/'questions.json').read_text());g=json.loads((parent/'private'/(wid+'_answers.json')).read_text());byid={r['question_id']:r for r in g}
    windows.append((q,g))
    for question in q:
     value=byid[question['question_id']]['canonical_answer']
     if question['qa_id']=='STARSS-SPEAKERS':speakers.add(int(value))
     if question['qa_id']=='V7-CLASS34':classes.add(value)
   parent_ids.add(parent.name)
 result=namespace['audit_batch_budget'](windows,{'STARSS-SPEAKERS':len(windows),'V7-CLASS34':len(windows)},verified_binding_ids=binding)
 if speakers!=set(range(9)):result['failure_reasons'].append('additional_train_incomplete_0_to_8_coverage')
 if len(classes)!=34:result['failure_reasons'].append('additional_train_incomplete_34_class_coverage')
 if len(parent_ids)!=132:result['failure_reasons'].append('additional_train_not_132_parents')
 result['status']='fail' if result['failure_reasons'] else 'pass'
 result['frozen_contract_sha256']='89e281eceeda38720e871bc17b5aafabf2eea57318c60d1fc0c82058b488312b'
 return result

def main(queue,receipts,complete=False):
 allrows=[];summaries=[];parents=[];windows=[];verified=[]
 withdrawn=set(queue.get('withdrawn_batches',[]))
 active=[item for item in queue['batches'] if item['batch_id'] not in withdrawn]
 for item in active:
  proof=receipts.get(item['batch_id'])
  if not proof:continue
  p=Path(proof['release_directory']);assert p.is_relative_to(ROOT) and proof['status']=='verified'
  summary=json.loads((p/'summary.json').read_text());summaries.append(summary);prefix=p.relative_to(ROOT)
  for line in (p/'train.jsonl').read_text().splitlines():
   row=json.loads(line)
   for k in ('video','audio_foa','audio_mono','private_evidence'):
    target=p/row[k];assert target.is_file() and target.resolve().is_relative_to(ROOT.resolve());row[k]=str(prefix/row[k])
   row['delivery_batch_id']=item['batch_id'];allrows.append(row)
  for file,target in [('parents.csv',parents),('windows.csv',windows)]:
   with (p/file).open() as f:target.extend(csv.DictReader(f))
  verified.append({**proof,'batch_id':item['batch_id']})
 qids=[r['question_id'] for r in allrows];lids=[r['logical_question_id'] for r in allrows]
 if len(qids)!=len(set(qids)) or len(lids)!=len(set(lids)):raise ValueError('duplicate additive identities')
 parent_ids={r['independent_parent_scene_id'] for r in allrows};wins={r['window_id'] for r in allrows};families=Counter(r['qa_id'] for r in allrows)
 if complete and (len(verified)!=len(active) or len(parent_ids)!=132):raise ValueError('not the complete 132-parent additional release')
 tmp=ROOT/'train.partial.jsonl';tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in allrows))
 speakers=Counter()
 for s in summaries:speakers.update(s['actual_speakers_per_window'])
 result={'schema':'kujiale-v7-S-additional-training-v1','status':'sealed' if complete else 'partial_not_training_ready','generator_commit':'33b5b12df1a45ca265bf0c4fbe9b26b664577a7c','spec_commit':'8910727','render_path':'ue_spear','independent_parent_scenes':len(parent_ids),'formal_windows':len(wins),'formal_logical_questions':len(lids),'family_counts':dict(families),'actual_speakers_per_window':dict(speakers),'modality_counts':dict(Counter(r['modality'] for r in allrows)),'train_parent_scenes':len(parent_ids),'dev_parent_scenes':0,'test_house_access':0,'additional_training_batch':True,'expected_additional_parents':132,'primary_500_parent_release':False,'CCI_SHA_verified_batches':verified,'withdrawn_archival_batches':sorted(withdrawn),'consumer_boundary':'same formal row/media ABI; current family25 waiting script only accepts 400 train +100 dev, so additive batch needs explicit downstream intake; do not fake development houses'}
 if complete or queue.get('dispatch_finished'):
  budget=original_budget_audit([Path(v['release_directory']) for v in verified]);result['original_batch_budget_audit']=budget
  atomic(ROOT/'BATCH_BUDGET_AUDIT.json',budget)
  if budget['status']!='pass' or not complete:
   result['status']='blocked_not_training_ready';atomic(ROOT/'ADDITIONAL_DELIVERY_PROGRESS.json',result);return result
  if (ROOT/'train.jsonl').exists():raise ValueError('completed aggregate immutable')
  tmp.replace(ROOT/'train.jsonl')
  for name,rows in [('parents.csv',parents),('windows.csv',windows)]:
   with (ROOT/name).open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  atomic(ROOT/'summary.json',result);atomic(ROOT/'DELIVERY_MANIFEST.json',result)
 else:atomic(ROOT/'ADDITIONAL_DELIVERY_PROGRESS.json',result)
 return result
if __name__=='__main__':
 import sys
 queue=json.loads(Path(sys.argv[1]).read_text());receipts=json.loads(Path(sys.argv[2]).read_text());print(json.dumps(main(queue,receipts,complete=bool(queue.get('complete'))),ensure_ascii=False))
