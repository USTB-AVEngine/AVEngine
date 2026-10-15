import time
import pytest
from avengine.timeline.v7_kujiale_host_resources import read_host,subtree_pss
from avengine.timeline.v7_kujiale_manifest import atomic_write
pytestmark=pytest.mark.fast_unit

def test_stale_or_missing_host_pss_cannot_authorize_a_new_isolated_job(tmp_path):
    assert read_host(tmp_path)is None
    (tmp_path/'evidence').mkdir();path=tmp_path/'evidence/host_resource_sample.json';atomic_write(path,{'sample_unix':time.time()-60,'PSS_bytes':1})
    assert read_host(tmp_path)is None
    atomic_write(path,{'sample_unix':time.time(),'PSS_bytes':35_000_000_000});assert read_host(tmp_path)['PSS_bytes']==35_000_000_000

def test_native_memory_reservation_counts_renderer_descendants_once():
    sample={'processes':[{'pid':1,'ppid':0,'pss_bytes':1},{'pid':2,'ppid':1,'pss_bytes':20},{'pid':3,'ppid':2,'pss_bytes':300},{'pid':4,'ppid':0,'pss_bytes':4000}]}
    assert subtree_pss(sample,1)==321 and subtree_pss(sample,2)==320


def test_host_bridge_pauses_only_free_leases_and_keeps_running_lease(monkeypatch,tmp_path):
    import fcntl,json,os
    from avengine.timeline import v7_kujiale_cpu_phases
    from avengine.timeline.v7_kujiale_host_resources import LegacyAdmissionBridge
    monkeypatch.setattr(v7_kujiale_cpu_phases,'legacy_workers',lambda *a:[])
    monkeypatch.setattr(os,'kill',lambda *a:pytest.fail('resource pressure must never stop a process'))
    slots=tmp_path/'evidence/cpu_slots';slots.mkdir(parents=True);active=(slots/'slot_0.json').open('a+')
    fcntl.flock(active,fcntl.LOCK_EX);active.write(json.dumps({'pid':123,'stage':'native_v0','estimate_bytes':6000000000}));active.flush();original=(slots/'slot_0.json').read_bytes()
    bridge=LegacyAdmissionBridge(tmp_path)
    try:
        bridge.update({'time_utc':'high','PSS_bytes':41000000000,'processes':[]})
        assert len(bridge.held)==15 and (slots/'slot_0.json').read_bytes()==original
        bridge.update({'time_utc':'recovered','PSS_bytes':25000000000,'processes':[]})
        assert len(bridge.held)==14 and (slots/'slot_0.json').read_bytes()==original
        receipt=json.loads((tmp_path/'evidence/legacy_admission_bridge.json').read_text())
        assert receipt['new_phase_allowance']==1 and receipt['running_workers_stopped']is False
    finally:
        for stream in bridge.held.values():stream.close()
        active.close()


def test_new_category_reserves_room_geometry_and_renderer_before_admission():
    from avengine.timeline.v7_kujiale_cpu_phases import phase_memory_estimate,phase_fits
    estimate=phase_memory_estimate('new_native_category',2500000000)
    assert estimate==12000000000
    assert not phase_fits(30000000000,0,estimate,1,0,100000000000)
    assert phase_fits(20000000000,0,estimate,1,0,100000000000)
    assert phase_memory_estimate('native_v1',4000000000)==6000000000
