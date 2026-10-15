"""CPU production disk headroom; never treat shared filesystem free as quota.

Local review writers inherit an explicit own-output budget through their
spawned processes. The 48g /data policy is mandatory: 100 decimal GB free,
10 decimal GB for this task, at most eight parent scenes. Allocated size
counts hardlinks once and is conservative for XFS reflinks.
"""
from __future__ import annotations
import json,os,shutil,time
from pathlib import Path

ENV_KEY="AVENGINE_V7_STORAGE_BUDGET"
class StorageBudgetExceeded(RuntimeError):
    pass


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


def check_write(path,*,expected_bytes=0):
    raw=os.environ.get(ENV_KEY)
    if not raw:return None
    policy=json.loads(raw);root=Path(policy["root"]).resolve();path=Path(path).resolve()
    if not path.is_relative_to(root):
        raise StorageBudgetExceeded("write escaped the declared own task output root")
    free=shutil.disk_usage(policy["filesystem"]).free
    requested=max(0,int(expected_bytes));margin=policy.get("concurrent_write_margin_bytes",0)
    # Some owner-authorized batches check headroom before launching a parent.
    # Once admitted, later checks retain path containment and measurements;
    # another user's disk growth must not abort that already running parent.
    if policy.get("admission_only"):
        return {"free_bytes":free,"own_allocated_bytes":allocated_bytes(root),
                "expected_write_bytes":requested,"checked_unix":time.time(),
                "policy":policy,"resource_limits_enforced_at":"parent_admission"}
    if free-requested-margin<policy["minimum_free_bytes"]:
        raise StorageBudgetExceeded(f"disk headroom stop: free={free}, expected={requested}, required={policy['minimum_free_bytes']}")
    used=allocated_bytes(root)
    if used+requested+margin>policy["maximum_task_bytes"]:
        raise StorageBudgetExceeded(f"own output budget stop: allocated={used}, expected={requested}, limit={policy['maximum_task_bytes']}")
    return {"free_bytes":free,"own_allocated_bytes":used,"expected_write_bytes":requested,
            "checked_unix":time.time(),"policy":policy}


def configure_integration_pilot(output,budget_root,*,parents,workers,minimum_free_bytes=150_000_000_000):
    """Explicit owner-authorized small pilot; the older review policy stays intact."""
    output=Path(output).resolve()
    if budget_root is None:raise StorageBudgetExceeded('integration pilot requires its own budget root')
    root=Path(budget_root).resolve()
    if not root.is_relative_to(Path('/data')) or root==Path('/data') or not output.is_relative_to(root):
        raise StorageBudgetExceeded('integration pilot output escaped its task root')
    if not 1<=parents<=30 or not 1<=workers<=16:raise StorageBudgetExceeded('integration pilot is limited to 30 parents and 16 workers')
    if minimum_free_bytes<100_000_000_000:raise StorageBudgetExceeded('explicit task headroom cannot be below owner 100GB floor')
    policy={'root':str(root),'filesystem':'/data','minimum_free_bytes':minimum_free_bytes,
        'maximum_task_bytes':99_000_000_000,'concurrent_write_margin_bytes':workers*64_000_000,
        'policy':'owner stage-one integration pilot: <=100GB including worktree, >=150GB shared free','local_48g':True}
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
