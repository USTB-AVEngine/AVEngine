"""Index SHA-verified own native deliveries as an additive training release."""
from pathlib import Path
from collections import Counter
import json,csv,os
ROOT=Path('/mnt/afs/xinyuanqian/jzy/hardbank/generation/v7_s_kujiale_20261005')
def atomic(p,v):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def main(queue,receipts,complete=False):
 allrows=[];summaries=[];parents=[];windows=[];verified=[]
 for item in queue['batches']:
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
 if complete and (len(verified)!=len(queue['batches']) or len(parent_ids)!=132):raise ValueError('not the complete 132-parent additional release')
 tmp=ROOT/'train.partial.jsonl';tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in allrows))
 speakers=Counter()
 for s in summaries:speakers.update(s['actual_speakers_per_window'])
 result={'schema':'kujiale-v7-S-additional-training-v1','status':'sealed' if complete else 'partial_not_training_ready','generator_commit':'33b5b12df1a45ca265bf0c4fbe9b26b664577a7c','spec_commit':'8910727','render_path':'ue_spear','independent_parent_scenes':len(parent_ids),'formal_windows':len(wins),'formal_logical_questions':len(lids),'family_counts':dict(families),'actual_speakers_per_window':dict(speakers),'modality_counts':dict(Counter(r['modality'] for r in allrows)),'train_parent_scenes':len(parent_ids),'dev_parent_scenes':0,'test_house_access':0,'additional_training_batch':True,'expected_additional_parents':132,'primary_500_parent_release':False,'CCI_SHA_verified_batches':verified,'consumer_boundary':'same formal row/media ABI; current family25 waiting script only accepts 400 train +100 dev, so additive batch needs explicit downstream intake; do not fake development houses'}
 if complete:
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
