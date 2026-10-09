"""Shape and provenance regressions for owner round-five cap35."""
import math
import pytest
import shapely
from shapely.geometry import box,Polygon,LineString
from shapely.affinity import rotate
from tools.rooms.room_split_auto import capped_split as c
from tools.rooms.room_split_auto.connected_split import FurnitureList

pytestmark=pytest.mark.fast_unit

def test_holes_only_fill_for_shape_and_original_area_is_preserved():
    raw=box(0,0,5,4).difference(box(1,1,1.1,1.1))
    assert c.connection_count(raw)==1
    assert c.room_circle(raw)["fits"]
    assert raw.area==pytest.approx(19.99)
    assert c.outline(raw).area==20

def test_actual_detached_island_never_attaches_to_main_room():
    raw=shapely.union_all([box(0,0,6,4),box(8,0,9,1)])
    parts,cuts=c.connected_parts(raw,FurnitureList([]),0)
    assert sorted(p.area for p in parts)==[1.,24.]
    assert all(c.connection_count(p)==1 for p in parts)
    assert not cuts
    assert shapely.union_all(parts).symmetric_difference(raw).area==0

def test_real_half_metre_neck_splits_after_hole_filling():
    raw=shapely.union_all([box(0,0,4,4),box(7,0,11,4),box(4,1.75,7,2.25)])
    parts,cuts=c.connected_parts(raw,FurnitureList([]),0)
    assert len(parts)>=2
    assert all(c.connection_count(p)<=1 for p in parts)
    assert all(max(c.angle_errors(line,0),default=0)<1e-8 for line,_,_,_,_ in cuts)
    assert shapely.union_all(parts).symmetric_difference(raw).area<1e-7

def test_short_side_alone_does_not_admit_a_triangular_wedge():
    wedge=Polygon([(0,0),(3,0),(0,6)])
    assert not c.room_circle(wedge)["fits"]

def test_rotated_main_wall_axes_and_room_disk_candidates():
    raw=rotate(box(0,0,10,4),17,origin=(0,0))
    axis=c.main_axis(raw)
    assert axis==pytest.approx(17)
    choices=c.candidates(raw,FurnitureList([]),35,axis)
    assert choices
    for _,line,children,_ in choices:
        assert len(line.coords)-1<=3
        assert max(c.angle_errors(line,axis))<1e-7
        assert all(c.room_circle(p)["fits"] and p.area>=6 for p in children)
        assert shapely.union_all(children).symmetric_difference(raw).area<1e-7

def test_sofa_dining_gap_beats_a_furniture_crossing_cut():
    raw=box(0,0,10,4)
    furniture=FurnitureList([
        dict(instance_id=1,category="sofa",geometry=box(.5,.5,4.5,3.5)),
        dict(instance_id=2,category="dining table",geometry=box(5.5,.5,9.5,3.5)),
    ])
    choices=c.candidates(raw,furniture,35,0)
    assert choices
    assert choices[0][3]["furniture_intersection_length_m"]==0
    assert all(6<=p.area<=35 for p in choices[0][2])

def test_corridor_broad_lobe_is_safeguarded_from_whole_block_discard():
    raw=shapely.union_all([box(0,0,4,4),box(4,1.5,14,2.5)])
    assert c.broad_components(raw)
    parts,cuts=c.split_corridor(raw,FurnitureList([]),0)
    assert cuts
    assert any(c.room_circle(p)["fits"] for p,e in parts)
    assert shapely.union_all([p for p,e in parts]).symmetric_difference(raw).area<1e-7
    assert all(not c.broad_components(p) for p,e in parts if not c.room_circle(p)["fits"])
