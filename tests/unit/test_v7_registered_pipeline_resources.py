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

def test_waiting_native_worker_carries_paths_instead_of_large_admission_payload():
    import pickle
    from avengine.timeline.v7_registered_native_workers import native_job
    job=native_job('/own/parent/input_context.json','/own','v0')
    assert len(pickle.dumps(job))<512 and set(job)=={'context_path','task_root','visual'}

def test_cpu_only_phases_fill_cores_released_by_native_renderers():
    from avengine.timeline.v7_kujiale_cpu_balance import assign_cpu_phases
    allocation=assign_cpu_phases(range(100,108),{0,1,2,4,5,8,9})
    assert len(set(allocation.values()))==8 and not(set(allocation.values())&{0,1,2,4,5,8,9})
    assert len(set(assign_cpu_phases(range(16),set()).values()))==16
