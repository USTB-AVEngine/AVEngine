"""Merge owned native batches, retaining only evidence for each current derived GLB."""
import argparse,json,hashlib
from pathlib import Path
from v7_object_prepare import dump
from v7_object_color_recheck import recheck_record

def main():
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);a=p.parse_args();T=a.task_root;metas=json.loads((T/'prepared_new.json').read_text());by={m['asset_id']:m for m in metas};merged={};inputs=[];ignored=[];hashes={}
 cal_path=T/'color_calibration_refined.json';cal=json.loads(cal_path.read_text()) if cal_path.exists() else None
 if cal is not None:assert cal['status']=='pass'
 paths=sorted(list(T.glob('native_new_batch*.json'))+list(T.glob('native_new_repair*.json')),key=lambda q:(q.stat().st_mtime,q.name))
 for q in paths:
  try:rows=json.loads(q.read_text())
  except json.JSONDecodeError:continue
  if not isinstance(rows,list) or not all(isinstance(r,dict) and 'asset_id' in r and 'house' in r for r in rows):continue
  inputs.append({'path':str(q),'records':len(rows)})
  for r in rows:
   aid=r['asset_id'];meta=by.get(aid)
   if r.get('review_method_version')!='object-attachment-body-v5':continue
   if meta is None:continue
   path=Path(meta['derived_path'])
   if r.get('derived_glb') is not None or r.get('status')=='pass':
    if r.get('derived_glb')!=str(path):ignored.append({'asset_id':aid,'house':r['house'],'reason':'GLB path differs from current derivation','batch':str(q)});continue
    if str(path) not in hashes:hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    if r.get('derived_sha256')!=hashes[str(path)]:ignored.append({'asset_id':aid,'house':r['house'],'reason':'GLB SHA256 differs from current derivation','batch':str(q)});continue
   if cal is not None:r=recheck_record(r,meta,T,cal,cal_path)
   merged[(aid,r['house'])]=r
 rows=[merged[k] for k in sorted(merged)];dump(T/'native_new.json',rows);dump(T/'native_collection.json',{'inputs':inputs,'records':len(rows),'ignored_stale_evidence':ignored});print('NATIVE_COLLECT',len(rows),'records;',len(ignored),'stale proofs ignored')
if __name__=='__main__':main()
