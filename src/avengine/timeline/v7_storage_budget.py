"""CPU production disk headroom; never treat shared filesystem free as quota.

Local review writers inherit an explicit own-output budget through their
spawned processes. The 48g /data policy is mandatory: 100 decimal GB free,
10 decimal GB for this task, at most eight parent scenes. Allocated size
counts hardlinks once and is conservative for XFS reflinks.
"""
from __future__ import annotations
import json,os,shutil,time,datetime
from pathlib import Path

def launch_minimum_free_bytes(policy=None, now=None):
    """Read a time-limited task launch override; running writes are separate."""
    policy=policy or {}
    current=now or datetime.datetime.now(datetime.timezone.utc)
    expiry=policy.get('expires_utc')
    if expiry and current>=datetime.datetime.fromisoformat(expiry.replace('Z','+00:00')):
        return int(policy.get('after_expiry_minimum_free_bytes',100_000_000_000))
    return int(policy.get('minimum_free_bytes',100_000_000_000))


ENV_KEY="AVENGINE_V7_STORAGE_BUDGET"
def resolve_launch_disk_policy(policy=None,*,root=None):
    """Read only an explicit caller policy or an optional task configuration.

    No output-directory name has special meaning. An absent configuration is
    the ordinary launch policy, including the normal 100GB filesystem floor.
    """
    if policy is not None:return dict(policy)
    if root is None:return {}
    path=Path(root)/'inputs/task_scope.json'
    if not path.is_file():return {}
    configured=json.loads(path.read_text()).get('temporary_disk_policy')
    if not configured:return {}
    return {'minimum_free_bytes':int(configured['minimum_free_bytes']),
        'expires_utc':configured['expires_utc'],
        'after_expiry_minimum_free_bytes':int(configured['normal_minimum_free_bytes'])}


def acceptance_minimum_free_bytes(now=None,policy=None):
    """Compatibility name for a supplied policy; default has no task exception."""
    return launch_minimum_free_bytes(policy,now)


def policy_minimum_free_bytes(policy,now=None):
    if policy['minimum_free_bytes']==0:return 0
    configured=policy.get('launch_disk_policy')
    return launch_minimum_free_bytes(configured,now)if configured else policy['minimum_free_bytes']

class StorageBudgetExceeded(RuntimeError):
    pass


def task_disk_policy(root,declared_floor):
    """Read an explicit task-local temporary floor, including its expiry."""
    import datetime
    path=Path(root)/'inputs/task_scope.json'
    scope=json.loads(path.read_text())if path.is_file()else{}
    temporary=scope.get('temporary_disk_policy')
    floor=int(declared_floor)
    if temporary:
        expiry=datetime.datetime.fromisoformat(temporary['expires_utc'])
        now=datetime.datetime.now(datetime.timezone.utc)
        floor=(int(temporary['minimum_free_bytes'])if now<expiry else
               max(floor,int(temporary['normal_minimum_free_bytes'])))
    return {'minimum_free_bytes':floor,'pause_new_jobs_only':scope.get('disk_guard_mode')=='pause_new_jobs_only'}


def allocated_bytes(root):
    """Count allocated blocks once per inode without following symlinks.

    DirEntry caches metadata, avoiding the repeated path construction and
    lstat calls of an os.walk/Path scan. Every check still scans the whole
    task root: no cached budget estimate can hide concurrent writes.
    """
    total=0;seen=set()
    if not Path(root).exists():return 0
    pending=[os.fspath(root)]
    while pending:
        folder=pending.pop()
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():continue
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(entry.path);continue
                        s=entry.stat(follow_symlinks=False)
                    except FileNotFoundError:continue
                    key=(s.st_dev,s.st_ino)
                    if key in seen:continue
                    seen.add(key)
                    total+=getattr(s,"st_blocks",(s.st_size+511)//512)*512
        except FileNotFoundError:continue
    return total


def configure_local_review(output,budget_root,*,parents,workers):
    output=Path(output).resolve()
    if not output.is_relative_to(Path("/data")):
        return {"policy":"AFS owner quota must be checked by its scheduler", "local_48g":False}
    if parents>8:raise StorageBudgetExceeded("48g review is limited to eight parents; production belongs on CCI AFS")
    if budget_root is None:raise StorageBudgetExceeded("48g requires the explicit task-owned --storage-budget-root")
    root=Path(budget_root).resolve()
    if not output.is_relative_to(root) or root==Path("/data"):
        raise StorageBudgetExceeded("output must be inside the declared task-owned storage budget")
    policy={"root":str(root),"filesystem":"/data","minimum_free_bytes":100_000_000_000,
            "maximum_task_bytes":10_000_000_000,"concurrent_write_margin_bytes":max(64_000_000,workers*64_000_000),
            "policy":"48g explicit owner disk discipline; decimal bytes", "local_48g":True}
    os.environ[ENV_KEY]=json.dumps(policy,sort_keys=True)
    check_write(output,expected_bytes=0)
    return policy



def effective_minimum_free_bytes(policy):
    if time.time()<float(policy.get('temporary_minimum_free_until_unix',0)):
        return int(policy.get('temporary_minimum_free_bytes',policy['minimum_free_bytes']))
    return int(policy['minimum_free_bytes'])

def check_write(path,*,expected_bytes=0):
    raw=os.environ.get(ENV_KEY)
    if not raw:return None
    policy=json.loads(raw);root=Path(policy["root"]).resolve();path=Path(path).resolve()
    if not path.is_relative_to(root):
        raise StorageBudgetExceeded("write escaped the declared own task output root")
    # The acceptance scheduler admits new jobs against the shared resource
    # limits. Once admitted, those limits must not abort an active producer.
    # Keep the own-output boundary enforced for every write.
    if policy.get("resource_admission_managed") is True:
        return {"resource_admission_managed":True,"expected_write_bytes":max(0,int(expected_bytes)),
                "checked_unix":time.time(),"policy":policy}
    free=shutil.disk_usage(policy["filesystem"]).free
    requested=max(0,int(expected_bytes));margin=policy.get("concurrent_write_margin_bytes",0)
    if policy.get("admission_only"):
        return {"free_bytes":free,"own_allocated_bytes":allocated_bytes(root),
                "expected_write_bytes":requested,"checked_unix":time.time(),
                "policy":policy,"resource_limits_enforced_at":"parent_admission"}
    disk_policy=task_disk_policy(root,min(policy_minimum_free_bytes(policy),effective_minimum_free_bytes(policy)))
    minimum=disk_policy["minimum_free_bytes"]
    below_floor=free-requested-margin<minimum
    if below_floor and not disk_policy["pause_new_jobs_only"]:
        raise StorageBudgetExceeded(f"disk headroom stop: free={free}, expected={requested}, required={minimum}")
    used=allocated_bytes(root)
    if used+requested+margin>policy["maximum_task_bytes"]:
        raise StorageBudgetExceeded(f"own output budget stop: allocated={used}, expected={requested}, limit={policy['maximum_task_bytes']}")
    return {"free_bytes":free,"own_allocated_bytes":used,"expected_write_bytes":requested,
            "checked_unix":time.time(),"policy":policy,"effective_disk_policy":disk_policy,"pause_new_jobs":below_floor}


def configure_integration_pilot(output,budget_root,*,parents,workers,minimum_free_bytes=150_000_000_000,launch_disk_policy=None):
    """Explicit owner-authorized small pilot; the older review policy stays intact."""
    output=Path(output).resolve()
    if budget_root is None:raise StorageBudgetExceeded('integration pilot requires its own budget root')
    root=Path(budget_root).resolve()
    if not root.is_relative_to(Path('/data')) or root==Path('/data') or not output.is_relative_to(root):
        raise StorageBudgetExceeded('integration pilot output escaped its task root')
    if not 1<=parents<=30 or not 1<=workers<=16:raise StorageBudgetExceeded('integration pilot is limited to 30 parents and 16 workers')
    launch_disk_policy=resolve_launch_disk_policy(launch_disk_policy,root=root)
    if minimum_free_bytes<launch_minimum_free_bytes(launch_disk_policy):raise StorageBudgetExceeded('explicit task headroom is below configured launch floor')
    policy={'root':str(root),'filesystem':'/data','minimum_free_bytes':minimum_free_bytes,
        'maximum_task_bytes':99_000_000_000,'concurrent_write_margin_bytes':workers*64_000_000,
        'policy':'explicit task-local launch and storage configuration','local_48g':True,'launch_disk_policy':launch_disk_policy}
    os.environ[ENV_KEY]=json.dumps(policy,sort_keys=True);check_write(output,expected_bytes=0)
    return policy


def configure_kujiale_production(output,budget_root,*,workers):
    """Owner-authorized Round14 native UE production, with bounded offload."""
    output=Path(output).resolve()
    if budget_root is None:raise StorageBudgetExceeded("native production needs explicit own output root")
    root=Path(budget_root).resolve()
    if root.name not in {'ue_round14','ue_round21_pilot'} or not str(root).startswith('/data/jzy/tmp/claude_hardbank_20260923/v7_kujiale_20261003/') or not output.is_relative_to(root):
        raise StorageBudgetExceeded("Kujiale native production output escaped authorized task")
    if not 1<=workers<=4:raise StorageBudgetExceeded("Kujiale resource profile supports one to four native workers")
    policy={"root":str(root),"filesystem":"/data","minimum_free_bytes":100_000_000_000,
            "maximum_task_bytes":24_000_000_000,"concurrent_write_margin_bytes":workers*128_000_000,
            "policy":"Round14 native production; separate full task growth monitor <=30GB including UE Content/DDC/worktree", "local_48g":True}
    if root.name=='ue_round21_pilot':
        policy.update(minimum_free_bytes=150_000_000_000,maximum_task_bytes=60_000_000_000,policy='Round21 train-only native pilot; no training export or transfer')
        scope_path=root/'inputs/task_scope.json'
        if scope_path.is_file():
            scope=json.loads(scope_path.read_text())
            policy['maximum_task_bytes']=min(policy['maximum_task_bytes'],int(scope['maximum_task_bytes']))
            policy['minimum_free_bytes']=max(policy['minimum_free_bytes'],int(scope['minimum_disk_free_bytes']))
    os.environ[ENV_KEY]=json.dumps(policy,sort_keys=True)
    check_write(output,expected_bytes=0)
    return policy
