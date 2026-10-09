import pytest
from shapely.geometry import box
from tools.rooms.room_split_auto.rectangle_audit import scope_metrics,distribution
pytestmark=pytest.mark.fast_unit
def test_other_floor_in_l_shaped_bbox_is_measured_without_own_overlap():
    own=box(0,0,4,2).union(box(0,2,2,4))
    other=box(2,2,4,4).union(box(0,0,1,1))
    m=scope_metrics(own,box(0,0,4,4),box(0,0,4,4),other)
    assert m["floor_over_bbox_ratio"]==pytest.approx(.75)
    assert m["walkable_on_other_ground_m2"]==pytest.approx(4)
    assert m["on_other_fraction_of_bbox_walkable"]==pytest.approx(.25)
    assert m["ground_label_overlap_m2"]==pytest.approx(1)
def test_empty_navmesh_is_unknown_fraction_not_fabricated_zero():
    m=scope_metrics(box(0,0,3,3),box(0,0,3,3),box(5,5,6,6),box(5,5,6,6))
    assert m["bbox_walkable_area_m2"]==0
    assert m["on_other_fraction_of_bbox_walkable"] is None
def test_distribution_retains_zero_and_missing_counts():
    d=distribution([0,.25,.75,1,None])
    assert d["n"]==4 and d["zero_count"]==1 and d["median"]==.5
