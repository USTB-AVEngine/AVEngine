import datetime
import json
from pathlib import Path
import pytest
from tools.timeline.monitor_v7_acceptance import launch_pause_reason
from avengine.timeline.v7_storage_budget import ENV_KEY, StorageBudgetExceeded, check_write

pytestmark = pytest.mark.fast_unit

@pytest.mark.parametrize("override,reason", [
    ({"pss": 141_000_000_000}, "PSS_limit_no_new_jobs"),
    ({"free": 99_000_000_000}, "disk_headroom_no_new_jobs"),
    ({"available_memory": 29_000_000_000}, "memory_headroom_no_new_jobs"),
    ({"allocated_size": 61_000_000_000}, "task_disk_limit_no_new_jobs"),
    ({"now": datetime.datetime(2026,10,15,4,tzinfo=datetime.timezone.utc)}, "delivery_deadline_no_new_jobs"),
])
def test_resource_pressure_only_changes_new_job_admission(override,reason):
    fields=dict(now=datetime.datetime(2026,10,15,0,tzinfo=datetime.timezone.utc),
                deadline=datetime.datetime(2026,10,15,4,tzinfo=datetime.timezone.utc),
                pss=1, limit=140_000_000_000, free=120_000_000_000,
                allocated_size=1, available_memory=50_000_000_000)
    fields.update(override)
    assert launch_pause_reason(**fields)==reason


def test_active_job_resource_policy_keeps_own_output_boundary(tmp_path,monkeypatch):
    root=tmp_path/'own';root.mkdir()
    monkeypatch.setenv(ENV_KEY,json.dumps(dict(root=str(root),resource_admission_managed=True)))
    # No resource read or allocation can interrupt the admitted producer.
    assert check_write(root/'media',expected_bytes=10**15)['resource_admission_managed']
    with pytest.raises(StorageBudgetExceeded,match='escaped'):
        check_write(tmp_path/'other')
