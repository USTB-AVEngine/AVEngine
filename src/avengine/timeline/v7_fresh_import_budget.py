"""Reserve task-owned space before copying a read-only helper delivery.

Only admission is delayed. Existing producers, source media and native seals
are never changed or deleted by a space decision.
"""
from contextlib import contextmanager
from pathlib import Path
import fcntl,json,os,shutil
from avengine.timeline.v7_storage_budget import allocated_bytes,launch_minimum_free_bytes


class FreshImportDeferred(RuntimeError):
    pass


def import_space_decision(manifest,payload_bytes,*,allocated,free,active_parents=0):
    margin=max(64_000_000,int(payload_bytes)//10)
    reserved=int(payload_bytes)+margin
    growth=int(manifest.get('per_parent_disk_reserve_bytes',0))*int(active_parents)
    reason=''
    if allocated+reserved+growth>int(manifest.get('maximum_task_bytes',60_000_000_000)):
        reason='task_output_reservation_no_new_import'
    elif free-reserved-growth<launch_minimum_free_bytes(manifest.get('launch_disk_policy')):
        reason='filesystem_headroom_reservation_no_new_import'
    return {'status':'paused_before_import'if reason else'admitted','reason':reason,
        'payload_bytes':int(payload_bytes),'reserved_bytes':reserved,'active_parent_growth_bytes':growth,
        'allocated_before_bytes':allocated,'filesystem_free_before_bytes':free,'active_processes_signalled':False}


def active_import_reservation(root):
    path=Path(root)/'evidence/import_space_reservation.json'
    if not path.is_file():return 0
    value=json.loads(path.read_text())
    if value.get('state')!='reserved':return 0
    try:
        stat=(Path('/proc')/str(int(value['pid']))/'stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z'or stat[19]!=value['start_ticks']:return 0
    except (OSError,ValueError,KeyError,IndexError):return 0
    return int(value['reserved_bytes'])


def measured_import_payload_bytes(native):
    supplied=native.get('export',{}).get('payload_logical_bytes')
    if supplied is not None:return int(supplied)
    pointer_path=Path(native['export']['source_seal']);pointer=json.loads(pointer_path.read_text())
    seal=pointer_path.parent/pointer['attempt'];files={pointer_path.resolve()}
    for name in pointer['required_files']:
        if Path(name).is_absolute()or '..'in Path(name).parts:raise ValueError('invalid seal payload path')
        files.add((seal/name).resolve())
    for win in pointer['result']['windows']:
        wid=win['window_id'];member=win.get('member')or wid.rsplit('__',2)[-2]
        marker=json.loads((pointer_path.parent/f'selected_windows_{member}_completed.json').read_text())
        files.add((Path(marker['recipe']['pool_directory'])/wid/'candidate_audit.json').resolve())
    return sum(p.stat().st_size for p in files)


@contextmanager
def admit_fresh_import(manifest,native):
    root=Path(manifest['output_root']).resolve();path=root/'evidence/import_space_reservation.json'
    def save(value):
        temp=path.with_name(path.name+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
    with(root/'inputs/import_budget.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        payload=measured_import_payload_bytes(native)
        active=sum(json.loads(p.read_text()).get('status')=='running'for p in(root/'evidence/jobs').glob('*.json'))
        decision=import_space_decision(manifest,int(payload),allocated=allocated_bytes(root),free=shutil.disk_usage(root).free,active_parents=active)
        state={**decision,'pid':os.getpid(),'start_ticks':(Path('/proc')/str(os.getpid())/'stat').read_text().rsplit(')',1)[1].split()[19]}
        if decision['status']!='admitted':
            save({**state,'state':'paused'})
            raise FreshImportDeferred(json.dumps(decision,sort_keys=True))
        save({**state,'state':'reserved'})
        try:yield decision
        finally:save({**state,'state':'released'})
