import pytest
from avengine.timeline.v7_acceptance_tasks import task_by_slot
pytestmark=pytest.mark.fast_unit


def test_reordering_and_sparse_slots_cannot_dispatch_a_different_scene():
    original={'slot':7,'row':{'scene_id':'original'}};priority={'slot':101,'row':{'scene_id':'new_gap'}}
    m={'parents':[priority,original]}
    assert task_by_slot(m,7)is original and task_by_slot(m,101)is priority
    m['parents'].reverse()
    assert task_by_slot(m,7)is original
    with pytest.raises(ValueError,match='exactly one task'):task_by_slot(m,0)
    m['parents'].append(dict(original))
    with pytest.raises(ValueError,match='exactly one task'):task_by_slot(m,7)
