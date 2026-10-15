import pytest
from avengine.timeline import v7_kujiale_scheduler as scheduler
from avengine.timeline import v7_kujiale_host_resources as host
pytestmark=pytest.mark.fast_unit


def snapshot():return {'MemAvailable_bytes':50_000_000_000,'PSS_bytes':5_000_000_000,'sampler_pid':-1}


def test_new_host_coordinator_collects_missing_sample_without_manual_daemon(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(host,'read_host',lambda root:None)
    monkeypatch.setattr(host,'sample_host',lambda root:calls.append(root)or snapshot())
    monkeypatch.setattr(scheduler,'can_collect_host_sample',lambda:True)
    result=scheduler.resources(tmp_path)
    assert calls==[tmp_path] and result['PSS_bytes']==5_000_000_000 and result['MemAvailable_bytes']==50_000_000_000


def test_fresh_external_host_sample_is_preserved(tmp_path,monkeypatch):
    monkeypatch.setattr(host,'read_host',lambda root:snapshot())
    monkeypatch.setattr(host,'sample_host',lambda root:pytest.fail('duplicate host collection'))
    assert scheduler.resources(tmp_path)['PSS_bytes']==5_000_000_000


def test_isolated_worker_cannot_replace_missing_host_pss_with_undercounted_namespace(tmp_path,monkeypatch):
    monkeypatch.setattr(host,'read_host',lambda root:None)
    monkeypatch.setattr(host,'sample_host',lambda root:pytest.fail('isolated PSS collection'))
    monkeypatch.setattr(scheduler,'can_collect_host_sample',lambda:False)
    result=scheduler.resources(tmp_path)
    assert not scheduler.can_start(result,active_parents=0)
    assert result['PSS_bytes']>40_000_000_000


@pytest.mark.parametrize('mapping,expected',[('0 0 4294967295\n',True),('0 1002 1\n',False),('',False)])
def test_host_uid_map_is_readable_without_privileged_proc1_namespace_access(monkeypatch,mapping,expected):
    from pathlib import Path
    monkeypatch.setattr(Path,'read_text',lambda self:mapping)
    assert scheduler.can_collect_host_sample() is expected


def test_actual_host_collector_creates_its_own_new_output_directory(tmp_path,monkeypatch):
    monkeypatch.setattr(host,'owned_processes',lambda root:[])
    root=tmp_path/'new_campaign'
    assert not root.exists()
    value=host.sample_host(root)
    assert (root/'evidence/host_resource_sample.json').is_file()
    assert host.read_host(root)['sample_unix']==value['sample_unix']
