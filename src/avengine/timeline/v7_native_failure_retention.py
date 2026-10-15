"""Retire only newly owned, failed, unpublished native intermediate media.

An explicit task policy and an empty-at-creation marker are both required.
Old attempts, completed native evidence, logs and JSON receipts are retained.
"""
from pathlib import Path
import datetime,hashlib,json,os,time,traceback
from avengine.timeline.v7_storage_budget import ENV_KEY


def mark_new_attempt(attempt):
    raw=os.environ.get(ENV_KEY)
    if not raw:return None
    root=Path(json.loads(raw)['root']).resolve();attempt=Path(attempt).resolve();scope=root/'inputs/task_scope.json'
    if not attempt.is_relative_to(root)or attempt==root or not scope.is_file():return None
    policy=json.loads(scope.read_text()).get('new_failed_attempt_retention')
    if not policy or not policy.get('enabled'):return None
    boundary=datetime.datetime.fromisoformat(policy['created_after_utc']).timestamp()
    if time.time()<boundary or any(attempt.iterdir()):return None
    marker={'schema':'new-owned-native-attempt-v1','root':str(root),'attempt':str(attempt),'created_unix':time.time(),'authority':policy['authority'],'policy':policy}
    path=attempt/'new_owned_attempt.json';path.write_text(json.dumps(marker,indent=2)+'\n');return path


def retire_failed_media(attempt,marker,error):
    if marker is None:return {'status':'retained_no_new_owned_marker'}
    attempt=Path(attempt).resolve();marker=Path(marker)
    if not marker.exists():return {'status':'retained_marker_not_present'}
    declaration=json.loads(marker.read_text());root=Path(declaration['root']).resolve()
    if declaration['attempt']!=str(attempt)or not attempt.is_relative_to(root):raise ValueError('failed native retention escaped own marked attempt')
    policy=declaration['policy'];files=[]
    for path in attempt.rglob('*'):
        if not path.is_file()or path.is_symlink()or path.suffix not in {'.mp4','.npy','.npz','.exr'}or path.stat().st_size<int(policy.get('minimum_file_bytes',8000000)):continue
        digest=hashlib.sha256()
        with path.open('rb')as stream:
            for data in iter(lambda:stream.read(1024*1024),b''):digest.update(data)
        stat=path.stat();files.append({'path':str(path),'sha256':digest.hexdigest(),'bytes':stat.st_size,'inode':stat.st_ino})
    receipt={'status':'failed_unpublished_native_attempt','error':repr(error),'traceback':traceback.format_exc(),'created_unix':declaration['created_unix'],'authority':declaration['authority'],'intermediate_media':files,'native_JSON_logs_and_receipts_retained':True,'old_files_removed':False}
    target=attempt/'failed_intermediate_retention_receipt.json'
    with target.open('w')as stream:json.dump(receipt,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    # Native encoder/UE must already be closed by the caller. Check the exact
    # recorded inode/size so this never removes another writer's replacement.
    for item in files:
        path=Path(item['path']);stat=path.stat()
        if stat.st_ino!=item['inode']or stat.st_size!=item['bytes']:raise ValueError('failed media changed after receipt; retained')
        path.unlink()
    return receipt
