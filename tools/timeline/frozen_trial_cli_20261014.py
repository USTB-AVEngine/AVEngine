"""Frozen CLI/modules with only the task's temporary owner resource policy.

The launch logic must run only in the original process.  multiprocessing spawn
loads this file as __mp_main__ before unpickling a worker from the frozen module.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,sys


def main():
 entry=Path(sys.argv[1]).resolve();T=Path('/data/jzy/tmp/claude_hardbank_20260923/v7_production_ready_20261014');root=T/'code'
 if not entry.is_relative_to(root) or entry.name!='generate_v7_phase2_pilot.py':raise ValueError('only this task-owned frozen pilot CLI is accepted')
 expires=datetime(2026,10,14,17,30,tzinfo=timezone.utc)
 if datetime.now(timezone.utc)>=expires:raise ValueError('owner temporary acceptance resources expired at01:30SGT')
 raw=entry.read_bytes();text=raw.decode();changes=[]
 for old in ['0<allowance<=35_000_000_000','0<allowance<=40_000_000_000','0<allowance<=60_000_000_000']:
  if old in text:text=text.replace(old,'0<allowance<=80_000_000_000');changes.append(old+' -> 0<allowance<=80_000_000_000')
 if '0<allowance<=80_000_000_000' not in text:raise ValueError('unknown CLI resource check; frozen code not altered')
 from avengine.timeline import v7_storage_budget as storage
 original_check=storage.check_write
 def own_temporary_check(path,*,expected_bytes=0):
  raw_policy=os.environ.get(storage.ENV_KEY)
  if raw_policy:
   policy=json.loads(raw_policy)
   if Path(policy['root']).resolve()==T:
    policy['minimum_free_bytes']=100_000_000_000
    policy['maximum_task_bytes']=min(int(policy['maximum_task_bytes']),80_000_000_000)
    policy['temporary_owner_resource_expiry_utc']=expires.isoformat()
    os.environ[storage.ENV_KEY]=json.dumps(policy,sort_keys=True)
  return original_check(path,expected_bytes=expected_bytes)
 storage.check_write=own_temporary_check
 print(json.dumps({'task_owned_resource_shim':__file__,'original_frozen_entry':str(entry),'original_entry_sha256':hashlib.sha256(raw).hexdigest(),'resource_only_changes':changes,'scope':str(T),'owner_additional_limit_bytes':80_000_000_000,'minimum_data_free_bytes':100_000_000_000,'temporary_expiry_utc':expires.isoformat(),'native_and_question_modules_from_frozen_batch':True,'spawn_safe_entry':True}),flush=True)
 sys.argv=[str(entry)]+sys.argv[2:]
 exec(compile(text,str(entry),'exec'),{'__name__':'__main__','__file__':str(entry),'__package__':None})


if __name__ == '__main__':
 main()
