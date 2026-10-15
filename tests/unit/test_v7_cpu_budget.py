import pytest
from avengine.timeline.v7_cpu_budget import compute_process_count
pytestmark=pytest.mark.fast_unit

def test_idle_metadata_shells_do_not_block_an_independent_native_operation():
    rows=[{'cmd':'python /tools/monitor.py','state':'S'}for _ in range(100)]
    rows += [{'cmd':'bash -s','state':'S'},{'cmd':'python -m avengine.timeline.v7_phase2_fresh_rir --request /own/request.json','state':'R'}]
    assert compute_process_count(rows)==1

def test_all_admitted_compute_workers_remain_reserved_even_while_waiting():
    rows=[{'cmd':f'python /frozen/tools/timeline/run_v7_acceptance.py --manifest /own/plan --worker {i}','state':'S'}for i in range(80)]
    assert compute_process_count(rows)==80
    rows.append({'cmd':'ffmpeg -threads 1 -i input','state':'S'})
    assert compute_process_count(rows)==81
    rows.append({'cmd':'python /other/actual_cpu_work.py','state':'R'})
    assert compute_process_count(rows)==82
