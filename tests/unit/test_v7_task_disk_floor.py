import json
from types import SimpleNamespace
import pytest
from avengine.timeline.v7_storage_budget import task_disk_policy,check_write,ENV_KEY,StorageBudgetExceeded
pytestmark=pytest.mark.fast_unit

def test_temporary_floor_expires_and_existing_writer_only_requests_pause(tmp_path,monkeypatch):
    (tmp_path/'inputs').mkdir();scope=tmp_path/'inputs/task_scope.json'
    data={'disk_guard_mode':'pause_new_jobs_only','temporary_disk_policy':{'minimum_free_bytes':60_000_000_000,
        'normal_minimum_free_bytes':100_000_000_000,'expires_utc':'2100-01-01T00:00:00+00:00'}}
    scope.write_text(json.dumps(data));assert task_disk_policy(tmp_path,100_000_000_000)['minimum_free_bytes']==60_000_000_000
    monkeypatch.setenv(ENV_KEY,json.dumps({'root':str(tmp_path),'filesystem':'/data','minimum_free_bytes':100_000_000_000,'maximum_task_bytes':1_000_000,'concurrent_write_margin_bytes':0}))
    monkeypatch.setattr('avengine.timeline.v7_storage_budget.shutil.disk_usage',lambda p:SimpleNamespace(free=70_000_000_000))
    assert check_write(tmp_path)['pause_new_jobs'] is False
    monkeypatch.setattr('avengine.timeline.v7_storage_budget.shutil.disk_usage',lambda p:SimpleNamespace(free=50_000_000_000))
    assert check_write(tmp_path)['pause_new_jobs'] is True
    data['temporary_disk_policy']['expires_utc']='2000-01-01T00:00:00+00:00';scope.write_text(json.dumps(data))
    assert task_disk_policy(tmp_path,60_000_000_000)['minimum_free_bytes']==100_000_000_000
    scope.write_text('{}')
    with pytest.raises(StorageBudgetExceeded,match='disk headroom'):check_write(tmp_path)
