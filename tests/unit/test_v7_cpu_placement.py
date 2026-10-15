import pytest
from avengine.timeline.v7_cpu_placement import choose_cpu
pytestmark=pytest.mark.fast_unit

def test_no_change_without_shared_contention_or_without_an_idle_allowed_core():
    assert choose_cpu(2,{2:1.,4:0.},set(),{2,4})is None
    assert choose_cpu(2,{2:1.,4:.8},{2},{2,4})is None
    assert choose_cpu(2,{2:1.,4:0.},{2},{2,4},{4})is None

def test_rebalance_preserves_the_allowed_set_and_avoids_foreign_or_reserved_cores():
    usage={1:0.,2:1.,3:0.,4:.1,5:0.}
    assert choose_cpu(2,usage,{2,3},{2,3,4},{5})==4
