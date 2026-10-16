"""Collect only declared, successful owned geometry receipts into the current new-asset manifest."""
import argparse,json
from pathlib import Path
from v7_object_prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--worktree',type=Path,required=True);a=p.parse_args();T=a.task_root;requests=[]
 for filename in ['v7_object_variants_plan_20261012.json','v7_object_supplemental_plan_20261012.json','v7_object_topup_plan_20261012.json']:
  q=a.worktree/'examples/assets'/filename
  if q.exists():requests+=json.loads(q.read_text())['requests']
 assert len({r['asset_id'] for r in requests})==len(requests);prepared=[];pixal=[];proofs=[]
 for request in requests:
  aid=request['asset_id'];D=T/'new'/aid;meta=T/'assets'/aid/'asset.json';ok=False
  for stage in ['pixal','trellis']:
   q=D/(stage+'.json')
   if q.exists():
    receipt=json.loads(q.read_text());proofs.append({'stage':stage,'asset_id':aid,'status':receipt.get('status'),'path':str(q)})
    if stage=='pixal':pixal.append(receipt)
    ok|=receipt.get('status') in ['pass','pass_preflight']
  if ok and meta.exists():
   m=json.loads(meta.read_text())
   if not m.get('preflight_issues') and Path(m.get('derived_path','__missing__')).is_file():prepared.append(m)
 dump(T/'prepared_new.json',prepared);dump(T/'pixal_results.json',pixal);dump(T/'geometry_collection.json',{'declared':len(requests),'prepared':len(prepared),'proofs':proofs});print('COLLECT',len(prepared),'prepared;',len(pixal),'Pixal receipts;',len(requests),'declared')
if __name__=='__main__':main()
