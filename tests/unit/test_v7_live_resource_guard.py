import pytest
from avengine.timeline.v7_live_resource_guard import decide
pytestmark=pytest.mark.fast_unit


def sample():
    return {'MemAvailable_bytes':80_000_000_000,'own_pss_bytes':20_000_000_000,'combined_pss_bytes':50_000_000_000,
        'processes':10,'data_free_bytes':120_000_000_000,'minimum_disk_free_bytes':100_000_000_000,'ranked_parent_trees':[]}


def test_live_overgrown_tree_blocks_new_operations_with_ranked_scene_identity():
    s=sample();s['ranked_parent_trees']=[{'owned':True,'PSS_bytes':31_000_000_000,'limit_bytes':30_000_000_000,'scene_id':'growing'}]
    r=decide(s,{})
    assert r['pause_new_native_operations']and r['blocked_owned_scene_ids']==['growing']
    assert r['active_processes_signalled']is False


def test_low_memory_is_detected_even_when_total_workload_is_below_budget():
    s=sample();s['MemAvailable_bytes']=19_000_000_000;r=decide(s,{})
    assert r['critical_below_20GB']and r['maximum_native_processes']==0
    assert r['pause_new_parents']and not r['active_processes_signalled']


def test_peer_PSS_counts_toward_combined_ceiling_without_taking_process_ownership():
    s=sample();s['combined_pss_bytes']=189_000_000_000;r=decide(s,{})
    assert 'combined_PSS_reservation'in r['reasons']and r['blocked_owned_scene_ids']==[]


def test_disk_policy_only_pauses_new_work_and_retains_all_running_processes():
    s=sample();s['data_free_bytes']=80_000_000_000;r=decide(s,{})
    assert r['pause_new_parents']and 'disk_floor_for_new_parents'in r['reasons']
    assert not r['active_processes_signalled']


def test_explicit_aggregate_RIR_budget_does_not_become_one_parents_worker_count():
    r=decide(sample(),{'maximum_native_processes':21,'fresh_rir_workers':7})
    assert r['maximum_native_processes']==21 and not r['pause_new_native_operations']
    low=sample();low['MemAvailable_bytes']=19_000_000_000
    assert decide(low,{'maximum_native_processes':21,'fresh_rir_workers':7})['maximum_native_processes']==0
