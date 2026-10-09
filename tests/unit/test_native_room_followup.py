"""Meaningful native cleanup regressions; originals gain no disk/visibility gate."""
import pytest
import shapely
from shapely.geometry import box
from tools.rooms.room_split_auto.seam_connectivity import Connectivity
from tools.rooms.room_split_auto.native_followup import width_scope,witness_in,old_minimum_recovery
pytestmark=pytest.mark.fast_unit

def test_large_courtyard_remains_void_for_width_diagnostic():
    main=box(0,0,8,8).difference(box(2.5,2.5,5.5,5.5))
    g=main.difference(box(.5,.5,.6,.6))
    scope,holes=width_scope(g,Connectivity(main))
    assert len(holes)==1 and holes[0].area==pytest.approx(9)
    assert scope.area==pytest.approx(55)
    assert scope.intersection(box(2.5,2.5,5.5,5.5)).area<1e-8
    assert g.area==pytest.approx(54.99)

def test_reused_placement_must_be_inside_actual_floor_not_bbox():
    g=box(0,0,8,8).difference(box(2,2,6,6))
    witness=dict(found=True,camera_m=[1,1.5,1],source_1_m=[7,1.2,1],source_2_m=[4,1.2,4])
    assert not witness_in(g,witness)
    witness["source_2_m"]=[7,1.2,7]
    assert witness_in(g,witness)


def test_recovery_counts_both_small_seam_parts_including_largest():
    from shapely.geometry import mapping
    g=box(0,0,1.49,4).union(box(1.51,0,3,4))
    retained=[dict(decision="retain",floor_polygon_xz_m=mapping(g))]
    small,short=old_minimum_recovery(g,retained)
    assert small==pytest.approx(11.92) and short==0

def test_recovery_does_not_count_a_true_fragment_still_discarded():
    from shapely.geometry import mapping
    main=box(0,0,4,4);fragment=box(6,0,7,4)
    g=main.union(fragment)
    retained=[dict(decision="retain",floor_polygon_xz_m=mapping(main)),
              dict(decision="discard",floor_polygon_xz_m=mapping(fragment))]
    assert old_minimum_recovery(g,retained)==(0,0)

def test_recovery_separates_old_short_side_failure_from_old_small_area():
    from shapely.geometry import mapping
    g=box(0,0,1.6,4).union(box(1.79,0,5.79,4))
    small,short=old_minimum_recovery(g,[dict(decision="retain",floor_polygon_xz_m=mapping(g))])
    assert small==0 and short==pytest.approx(6.4)
