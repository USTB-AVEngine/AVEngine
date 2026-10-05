"""Read the restartable personal AFS quota cache; wait on unknown measurements."""
from pathlib import Path
import argparse,json,time
from avengine.timeline.v7_owner_quota import CachedOwnerQuota,atomic


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--owner-root',type=Path,required=True);p.add_argument('--task-root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--expected-bytes',type=int,default=0);p.add_argument('--quota-bytes',type=int,default=2_000_000_000_000);p.add_argument('--reserve-bytes',type=int,default=80_000_000_000);a=p.parse_args()
    guard=CachedOwnerQuota(a.owner_root,a.task_root,a.cache,a.quota_bytes,a.reserve_bytes)
    try:
        while True:
            start=time.perf_counter();state=guard.poll(a.expected_bytes);state['fast_check_seconds']=time.perf_counter()-start;atomic(a.receipt,state)
            if state['new_dispatch_allowed']:print(json.dumps(state));return 0
            if state['confirmed_full']:print(json.dumps(state));return 2
            time.sleep(5)
    finally:guard.close()


if __name__=='__main__':raise SystemExit(main())
