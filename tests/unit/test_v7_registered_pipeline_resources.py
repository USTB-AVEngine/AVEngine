import pytest
from avengine.timeline.v7_kujiale_cpu_phases import phase_fits
from avengine.timeline.v7_kujiale_scheduler import can_start
pytestmark=pytest.mark.fast_unit

def test_native_and_qa_share_sixteen_cpu_slots_and_reserve_pending_memory():
    assert phase_fits(20e9,4e9,2e9,6,8,50e9)
    assert not phase_fits(20e9,4e9,2e9,8,8,50e9)
    assert not phase_fits(35e9,4e9,2e9,1,0,50e9)
    assert not phase_fits(20e9,0,2e9,1,0,29e9)

def test_new_parent_reserves_its_geometry_working_set_without_stopping_old_jobs():
    s={'MemAvailable_bytes':50e9,'PSS_bytes':37e9,'data_free_bytes':80e9,'minimum_disk_free_bytes':60e9,'new_parent_memory_reserve_bytes':4e9}
    assert not can_start(s,active_parents=3,max_parents=16)
    s['PSS_bytes']=32e9;assert can_start(s,active_parents=12,max_parents=16)
