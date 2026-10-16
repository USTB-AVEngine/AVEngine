import json
from types import SimpleNamespace
import pytest
from avengine.timeline import v7_native_resource_slots as resource
from avengine.timeline.v7_storage_budget import ENV_KEY
pytestmark=pytest.mark.fast_unit

def configure(tmp_path,monkeypatch,section):
    (tmp_path/'inputs').mkdir()
    (tmp_path/'inputs/native_resource_policy.json').write_text(json.dumps({'rir':section}))
    monkeypatch.setenv(ENV_KEY,json.dumps({'root':str(tmp_path)}))
    import psutil
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=80_000_000_000))

def test_default_does_not_add_a_limit(monkeypatch):
    monkeypatch.delenv(ENV_KEY,raising=False)
    assert resource.acquire_native_resource('rir') is None

def test_locks_limit_concurrent_native_operations_and_release(tmp_path,monkeypatch):
    configure(tmp_path,monkeypatch,{'slots':2})
    a=resource.acquire_native_resource('rir');b=resource.acquire_native_resource('rir')
    assert {a.receipt['slot'],b.receipt['slot']}=={0,1}
    a.close();c=resource.acquire_native_resource('rir')
    assert c.receipt['slot']==0
    b.close();c.close()

def test_memory_headroom_waits_then_allows_new_operation(tmp_path,monkeypatch):
    configure(tmp_path,monkeypatch,{'slots':1,'minimum_available_bytes':30_000_000_000})
    import psutil
    available=iter([20_000_000_000,80_000_000_000,80_000_000_000]);slept=[]
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=next(available)))
    monkeypatch.setattr(resource.time,'sleep',lambda seconds:slept.append(seconds))
    a=resource.acquire_native_resource('rir');assert slept==[.5];a.close()

def test_invalid_slot_count_is_rejected_before_native_work(tmp_path,monkeypatch):
    configure(tmp_path,monkeypatch,{'slots':0})
    with pytest.raises(ValueError,match='resource slots'):resource.acquire_native_resource('rir')


def test_finished_operation_does_not_keep_reserving_memory(tmp_path,monkeypatch):
    configure(tmp_path,monkeypatch,{'slots':2,'minimum_available_bytes':30_000_000_000})
    first=resource.acquire_native_resource('rir',expected_incremental_bytes=10_000_000_000)
    first.close()
    import psutil
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=45_000_000_000))
    monkeypatch.setattr(resource.time,'sleep',lambda seconds:(_ for _ in ()).throw(AssertionError('stale reservation blocked legal work')))
    second=resource.acquire_native_resource('rir',expected_incremental_bytes=10_000_000_000)
    second.close()


def test_explicit_shared_budget_can_admit_more_than_sixteen_workers(tmp_path,monkeypatch):
    configure(tmp_path,monkeypatch,{'slots':21})
    leases=[]
    try:
        leases=[resource.acquire_native_resource('rir')for _ in range(21)]
        assert {lease.receipt['slot']for lease in leases}==set(range(21))
    finally:
        for lease in leases:lease.close()
