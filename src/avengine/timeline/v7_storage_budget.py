"""CPU production disk headroom; never treat shared filesystem free as quota.

Local review writers inherit an explicit own-output budget through their
spawned processes. The 48g /data policy is mandatory: 50 decimal GB free,
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
    total=0;seen=set()
    if not Path(root).exists():return 0
    for folder,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if not (Path(folder)/d).is_symlink()]
        for name in files:
            p=Path(folder)/name
            try:s=p.lstat()
            except FileNotFoundError:continue
            if p.is_symlink():continue
            key=(s.st_dev,s.st_ino)
            if key in seen:continue
            seen.add(key);total+=getattr(s,"st_blocks",(s.st_size+511)//512)*512
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
    policy={"root":str(root),"filesystem":"/data","minimum_free_bytes":50_000_000_000,
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
    if free-requested-margin<policy["minimum_free_bytes"]:
        raise StorageBudgetExceeded(f"disk headroom stop: free={free}, expected={requested}, required={policy['minimum_free_bytes']}")
    used=allocated_bytes(root)
    if used+requested+margin>policy["maximum_task_bytes"]:
        raise StorageBudgetExceeded(f"own output budget stop: allocated={used}, expected={requested}, limit={policy['maximum_task_bytes']}")
    return {"free_bytes":free,"own_allocated_bytes":used,"expected_write_bytes":requested,
            "checked_unix":time.time(),"policy":policy}
