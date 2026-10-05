"""Verify an own AFS shard before returning permission to delete its 48g copy.

This is readback, not generation. Media paths in train.jsonl stay relative;
capture_by_path identities and every original answer/receipt stay byte exact.
"""
import argparse,hashlib,json,os,socket,tarfile,time,uuid
from pathlib import Path
AFS_ROOT=Path('/mnt/afs/xinyuanqian/jzy/hardbank/generation/v7_full_preproduction1500_20261005')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    with temp.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def verify_header(ready,archive):
    index=ready['shard_id']
    if type(index)is not int or not 0<=index<30:raise ValueError('invalid own shard ID')
    expected=AFS_ROOT/'shards'/f'shard_{index:03d}'
    if expected.is_symlink() or not expected.resolve().is_relative_to(AFS_ROOT.resolve()):raise ValueError('AFS destination symlink escaped scope')
    if ready['AFS_destination']!=str(expected):raise ValueError('AFS destination escaped owner scope')
    if not archive.resolve().is_relative_to(AFS_ROOT/'incoming') or archive.is_symlink():raise ValueError('archive outside own incoming root')
    if archive.stat().st_size!=ready['archive_bytes'] or sha(archive)!=ready['archive_SHA256']:raise ValueError('archive SHA/length differs; no local deletion')
    if any(c not in '0123456789abcdef'for c in ready['source_commit']) or len(ready['source_commit'])!=40:raise ValueError('source commit absent')
    if ready.get('test_house_access')!=0 or ready.get('GPU_started')is not False:raise ValueError('scope receipt differs')
    return expected

def extract_verified(archive,destination):
    destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as tf:
        members=tf.getmembers();names=set()
        for member in members:
            parts=Path(member.name).parts
            if not parts or Path(member.name).is_absolute() or '..' in parts or member.name in names:raise ValueError('unsafe or duplicate tar entry')
            if not (member.isfile() or member.isdir()):raise ValueError('links/special files are not allowed in immutable shard tar')
            names.add(member.name)
            target=destination/member.name
            if not target.resolve().is_relative_to(destination.resolve()) or target.is_symlink():raise ValueError('unsafe extract path')
            if member.isdir():target.mkdir(parents=True,exist_ok=True);continue
            stream=tf.extractfile(member)
            if stream is None:raise ValueError('missing tar stream')
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                # An AFS restart can leave complete files. Verify, don't overwrite.
                digest=hashlib.sha256();length=0
                for data in iter(lambda:stream.read(4*1024*1024),b''):digest.update(data);length+=len(data)
                if length!=target.stat().st_size or digest.hexdigest()!=sha(target):raise ValueError('retained extract file differs; select a new audited attempt')
            else:
                temp=target.with_name('.'+target.name+'.partial_'+uuid.uuid4().hex)
                with temp.open('xb') as f:
                    for data in iter(lambda:stream.read(4*1024*1024),b''):f.write(data)
                    f.flush();os.fsync(f.fileno())
                os.replace(temp,target)
    inventory=json.loads((destination/'FILE_SHA256.json').read_text())
    files={str(f.relative_to(destination)) for f in destination.rglob('*') if f.is_file() and '.partial_' not in f.name}
    if files!=set(inventory)|{'FILE_SHA256.json'}:raise ValueError('file inventory does not cover every payload file')
    for name,digest in inventory.items():
        file=destination/name
        if not file.resolve().is_relative_to(destination.resolve()) or file.is_symlink() or sha(file)!=digest:raise ValueError('payload file SHA differs: '+name)
    return inventory

def main():
    global AFS_ROOT
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ready',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--smoke',action='store_true',help='isolated transfer smoke; never produces a full-shard deletion receipt');a=p.parse_args()
    if a.smoke:AFS_ROOT=AFS_ROOT/'transfer_smoke'
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('explicit CPU-only environment required')
    ready=json.loads(a.ready.read_text());dest=verify_header(ready,a.archive)
    if not a.receipt.resolve().is_relative_to(AFS_ROOT/'receipts'):raise ValueError('receipt escaped own AFS root')
    start=time.monotonic();stage=AFS_ROOT/'extracting'/f"shard_{ready['shard_id']:03d}_{ready['archive_SHA256'][:16]}"
    if dest.exists():
        # Existing verified bytes can be read again after a disconnected return.
        inventory=json.loads((dest/'FILE_SHA256.json').read_text())
        files={str(f.relative_to(dest)) for f in dest.rglob('*') if f.is_file()}
        if files!=set(inventory)|{'FILE_SHA256.json'}:raise ValueError('retained payload has unlisted files')
        for name,digest in inventory.items():
            if (dest/name).is_symlink() or not (dest/name).resolve().is_relative_to(dest.resolve()):raise ValueError('retained inventory escaped scope')
            if sha(dest/name)!=digest:raise ValueError('previous destination was mutated')
        active=dest
    else:inventory=extract_verified(a.archive,stage);active=stage
    provenance=json.loads((active/'SOURCE_PROVENANCE.json').read_text())
    if provenance['source_commit']!=ready['source_commit'] or provenance['test_house_access']!=0:raise ValueError('source provenance differs')
    release=active/'release'
    if ready['qualified_parents']:
        from avengine.timeline.v7_s_batch_checks import check_critical_delivery,open_delivered_media
        coverage=check_critical_delivery(release,require_full=False);media=open_delivered_media(release)
        manifest=json.loads((release/'DELIVERY_MANIFEST.json').read_text())
        if coverage['status']!='pass' or media['status']!='pass':raise ValueError('unchanged media/question readback failed')
        if (manifest['independent_parent_scenes'],manifest['formal_windows'],manifest['formal_logical_questions'])!=(ready['qualified_parents'],ready['windows'],ready['questions']):raise ValueError('formal counts differ from sealed transfer')
    else:
        if ready['windows'] or ready['questions'] or release.exists():raise ValueError('empty shard claims nonempty formal data')
        coverage={'status':'empty_no_qualified_parent'};media={'status':'pass','actual_decoded_windows':0,'GPU_started':False,'empty_failed_shard_only':True}
    if active!=dest:dest.parent.mkdir(parents=True,exist_ok=True);os.rename(active,dest)
    receipt={'status':'verified','shard_id':ready['shard_id'],'archive_SHA256':ready['archive_SHA256'],'archive_bytes':ready['archive_bytes'],'source_commit':ready['source_commit'],'AFS_destination':str(dest),'every_file_SHA256_verified':True,'verified_file_count':len(inventory),'actual_media_readback':media,'coverage':coverage,'requested_parents':ready['requested_parents'],'qualified_parents':ready['qualified_parents'],'windows':ready['windows'],'questions':ready['questions'],'wall_seconds':time.monotonic()-start,'hostname':socket.gethostname(),'checked_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'48g_copy_deletion_allowed_only_for_matching_own_shard':True,'capture_id_or_answer_rewrite':False}
    atomic(a.receipt,receipt);print(json.dumps(receipt,ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
