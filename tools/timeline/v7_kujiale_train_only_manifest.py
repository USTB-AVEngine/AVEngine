"""Snapshot SHA-verified Kujiale rows for an explicit train-only v7-S intake."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
import csv,json,os

ROOT=Path('/mnt/afs/xinyuanqian/jzy/hardbank/generation/v7_s_kujiale_20261005')

def digest(path):
    h=sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def atomic_text(path,value):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('x') as stream:
        stream.write(value)
        stream.flush();os.fsync(stream.fileno())
    tmp.replace(path)

def main():
    progress=json.loads((ROOT/'ADDITIONAL_DELIVERY_PROGRESS.json').read_text())
    if progress['status']!='partial_not_training_ready' or progress['test_house_access']!=0 or progress['dev_parent_scenes']!=0:
        raise ValueError('only the verified partial train-only progress may be snapshotted')
    source=ROOT/'train.partial.jsonl'
    data=source.read_bytes();rows=[json.loads(line) for line in data.splitlines() if line.strip()]
    parents=defaultdict(lambda:{'house':None,'windows':set(),'questions':0,'batch':None})
    seenq=set();seenl=set();seenw=set();media=set()
    for row in rows:
        if row['split']!='train' or not row['house'].startswith('kujiale_'):
            raise ValueError('non-train or non-Kujiale row in own aggregate')
        q=row['question_id'];l=row['logical_question_id'];wid=row['window_id'];pid=row['independent_parent_scene_id']
        if q in seenq or l in seenl:raise ValueError('duplicate question identity')
        seenq.add(q);seenl.add(l);seenw.add(wid)
        parent=parents[pid]
        if parent['house'] not in (None,row['house']):raise ValueError('one parent crosses houses')
        parent['house']=row['house'];parent['windows'].add(wid);parent['questions']+=1
        if parent['batch'] not in (None,row['delivery_batch_id']):raise ValueError('one parent crosses batches')
        parent['batch']=row['delivery_batch_id']
        for key in ('video','audio_foa','audio_mono','private_evidence'):
            p=ROOT/row[key]
            if not p.resolve().is_relative_to(ROOT.resolve()) or not p.is_file() or p.is_symlink():
                raise ValueError('missing, symlinked, or escaped delivered media')
            media.add(p)
    if len(parents)!=progress['independent_parent_scenes'] or len(seenw)!=progress['formal_windows'] or len(seenl)!=progress['formal_logical_questions']:
        raise ValueError('aggregate counts differ from SHA-verified progress')
    batch_ids={v['batch_id'] for v in progress['CCI_SHA_verified_batches']}
    if {v['batch'] for v in parents.values()}!=batch_ids:
        raise ValueError('train rows and verified batch receipts differ')
    snapshot=ROOT/f'train_only_append_round18_{len(parents)}parents.jsonl'
    if snapshot.exists():
        if digest(snapshot)!=sha256(data).hexdigest():raise ValueError('existing immutable snapshot differs')
    else:atomic_text(snapshot,data.decode())
    parent_path=ROOT/f'TRAIN_ONLY_ADDITIONAL_PARENTS_{len(parents)}.csv'
    import io
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=['independent_parent_scene_id','house','split','window_count','logical_question_count','delivery_batch_id'])
    writer.writeheader()
    for pid,value in sorted(parents.items()):writer.writerow({'independent_parent_scene_id':pid,'house':value['house'],'split':'train','window_count':len(value['windows']),'logical_question_count':value['questions'],'delivery_batch_id':value['batch']})
    if parent_path.exists():
        if parent_path.read_text()!=stream.getvalue():raise ValueError('existing immutable parent list differs')
    else:atomic_text(parent_path,stream.getvalue())
    manifest={'schema':'v7_s_kujiale_train_only_additional_v1','status':'verified_partial_not_training_ready',
              'created_utc':datetime.now(timezone.utc).isoformat(),'render_path':'ue_spear','split':'train',
              'test_house_access':0,'dev_rows':0,'additional_training_batch':True,
              'training_rows_path':str(snapshot),'training_rows_sha256':digest(snapshot),
              'media_root':str(ROOT),'parent_list_path':str(parent_path),'parent_list_sha256':digest(parent_path),
              'independent_parent_scenes':len(parents),'formal_windows':len(seenw),'formal_logical_questions':len(seenl),
              'distinct_media_and_private_files':len(media),'verified_batches':progress['CCI_SHA_verified_batches'],
              'request_total':132,'remaining_generation_or_retries_pending':True,
              'consumer_contract':'Rows match v7 train.jsonl ABI. Trainer must load as an explicit additional train source with media_root; do not append to dev/test or pass as the sealed 500-parent main release.',
              'sha_pin_required_at_train_intake':True,'primary_release_overlap_audit':'pending primary second-batch release pointer'}
    marker=ROOT/f'TRAIN_ONLY_ADDITIONAL_MANIFEST_{len(parents)}.json'
    if marker.exists():
        prior=json.loads(marker.read_text());prior.pop('created_utc',None);compare=dict(manifest);compare.pop('created_utc',None)
        if prior!=compare:raise ValueError('existing immutable train-only manifest differs')
    else:atomic_text(marker,json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':manifest['status'],'parents':len(parents),'windows':len(seenw),'questions':len(seenl),'training_rows_sha256':manifest['training_rows_sha256'],'manifest':str(marker),'parent_csv':str(parent_path)},ensure_ascii=False))

if __name__=='__main__':main()
