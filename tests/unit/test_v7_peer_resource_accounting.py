import pytest
from tools.timeline import monitor_v7_acceptance as monitor
pytestmark=pytest.mark.fast_unit


def test_readonly_baseline_parent_root_does_not_double_count_owned_work(monkeypatch):
    manifest={'output_root':'/baseline/phase2','peer_output_roots':['/baseline','/other']}
    own=[{'pid':1,'pss_bytes':10},{'pid':2,'pss_bytes':20}];all_records=[*own,{'pid':3,'pss_bytes':30}]
    monkeypatch.setattr(monitor,'output_processes',lambda roots:own if roots==[manifest['output_root']]else all_records)
    peers=monitor.peer_processes(manifest)
    assert [p['pid']for p in peers]==[3]
    assert sum(r['pss_bytes']for r in own+peers)==60
    assert monitor.peer_processes(manifest,own_pids={1,2})==peers


def test_no_peer_roots_needs_no_process_tree_scan(monkeypatch):
    monkeypatch.setattr(monitor,'output_processes',lambda roots:pytest.fail('unnecessary process scan'))
    assert monitor.peer_processes({'output_root':'/own'})==[]
