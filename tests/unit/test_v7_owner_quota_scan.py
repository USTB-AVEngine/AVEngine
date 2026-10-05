import threading,time
from pathlib import Path
import pytest
from avengine.timeline.v7_owner_quota import CachedOwnerQuota


def settle(q):
    end=time.monotonic()+2
    while time.monotonic()<end:
        s=q.poll()
        if s['status'] in ('pass','confirmed_full'):return s
        time.sleep(.005)
    return q.poll()


def setup(tmp,reader,clock=time.time):
    owner=tmp/'owner';task=owner/'task';task.mkdir(parents=True,exist_ok=True)
    return CachedOwnerQuota(owner,task,task/'cache.json',100,10,reader=reader,clock=clock)


def test_initial_unknown_waits_without_dispatch_or_failure(tmp_path):
    gate=threading.Event()
    def read(root):gate.wait(2);return {'bytes':10 if root.name=='task'else 50,'wall_seconds':.001}
    q=setup(tmp_path,read)
    try:
        s=q.poll();assert not s['new_dispatch_allowed'] and not s['confirmed_full']
        gate.set();assert settle(q)['new_dispatch_allowed']
    finally:gate.set();q.close()


def test_fresh_measured_reserve_exhaustion_is_confirmed(tmp_path):
    q=setup(tmp_path,lambda root:{'bytes':10 if root.name=='task'else 90,'wall_seconds':.001})
    try:assert settle(q)['confirmed_full']
    finally:q.close()


def test_transient_du_timeout_is_retry_without_termination(tmp_path):
    clock=[0];calls=[]
    def read(root):
        if root.name!='task':
            calls.append(root)
            if len(calls)==1:raise TimeoutError('slow AFS')
        return {'bytes':10 if root.name=='task'else 50,'wall_seconds':.001}
    q=setup(tmp_path,read,lambda:clock[0])
    try:
        time.sleep(.03);s=q.poll();assert not s['confirmed_full'] and not s['new_dispatch_allowed']
        assert 'global'in s['transient_errors'];assert s['retry_not_before_unix']['global']==30
        clock[0]=31;q.poll();assert settle(q)['new_dispatch_allowed'];assert len(calls)==2
    finally:q.close()


def test_own_growth_added_without_own_deletion_credit(tmp_path):
    clock=[0];size=[10]
    q=setup(tmp_path,lambda root:{'bytes':size[0]if root.name=='task'else 50,'wall_seconds':.001},lambda:clock[0])
    try:
        assert settle(q)['estimated_owner_allocated_bytes']==50
        size[0]=25;clock[0]=31;q.poll();time.sleep(.03);s=q.poll();assert s['own_growth_bytes']==15 and s['estimated_owner_allocated_bytes']==65
        size[0]=3;clock[0]=62;q.poll();time.sleep(.03);assert q.poll()['estimated_owner_allocated_bytes']==50
    finally:q.close()


def test_stale_cache_pauses_and_survives_restart(tmp_path):
    clock=[0];gate=threading.Event()
    def read(root):
        if clock[0]>0 and root.name!='task':gate.wait(2)
        return {'bytes':10 if root.name=='task'else 50,'wall_seconds':.001}
    q=setup(tmp_path,read,lambda:clock[0])
    try:
        assert settle(q)['new_dispatch_allowed'];clock[0]=901;s=q.poll();assert not s['new_dispatch_allowed'] and not s['confirmed_full']
    finally:gate.set();q.close()
    clock[0]=902;q2=setup(tmp_path,read,lambda:clock[0])
    try:assert settle(q2)['new_dispatch_allowed']
    finally:q2.close()


def test_growth_upper_bound_requests_refresh_not_false_confirmation(tmp_path):
    q=setup(tmp_path,lambda root:{'bytes':10 if root.name=='task'else 50,'wall_seconds':.001})
    try:
        assert settle(q)['new_dispatch_allowed'];s=q.poll(expected_bytes=45)
        assert s['status']=='awaiting_headroom_refresh' and not s['confirmed_full']
    finally:q.close()


@pytest.mark.parametrize('bad',[-1,True,None])
def test_invalid_reading_never_grants_dispatch(tmp_path,bad):
    q=setup(tmp_path,lambda root:{'bytes':bad,'wall_seconds':0})
    try:
        time.sleep(.03);s=q.poll();assert not s['new_dispatch_allowed'] and not s['confirmed_full']
    finally:q.close()


def test_cache_scope_mismatch_rejected(tmp_path):
    q=setup(tmp_path,lambda root:{'bytes':10 if root.name=='task'else 50,'wall_seconds':0})
    try:assert settle(q)['new_dispatch_allowed']
    finally:q.close()
    import json
    f=tmp_path/'owner/task/cache.json';d=json.loads(f.read_text());d['owner_root']='/wrong';f.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='identity mismatch'):setup(tmp_path,lambda root:{'bytes':0,'wall_seconds':0})



def test_cached_full_is_refreshed_before_termination(tmp_path):
    q=setup(tmp_path,lambda root:{'bytes':10 if root.name=='task'else 90,'wall_seconds':0})
    try:assert settle(q)['confirmed_full']
    finally:q.close()
    gate=threading.Event()
    def read(root):
        if root.name!='task':gate.wait(2)
        return {'bytes':10 if root.name=='task'else 50,'wall_seconds':0}
    q=setup(tmp_path,read)
    try:
        s=q.poll();assert not s['confirmed_full'] and not s['new_dispatch_allowed']
        gate.set();assert settle(q)['new_dispatch_allowed']
    finally:gate.set();q.close()
