import numpy as np
import pytest
from shapely.geometry import Polygon,box,LineString
from tools.rooms.room_split_auto.connected_split import connection_count,connected_parts,line_candidates,cut_measure,FurnitureList
pytestmark=pytest.mark.fast_unit
def test_two_rooms_with_sub_0_6_channel_are_split_without_losing_ground():
    g=box(0,0,4,4).union(box(6,0,10,4)).union(box(4,1.75,6,2.25))
    assert connection_count(g)>1
    pieces,cuts=connected_parts(g)
    assert len(pieces)>=2
    assert sum(x.area for x in pieces)==pytest.approx(g.area)
    assert all(connection_count(x)==1 for x in pieces if x.area>=6)
    assert all(len(line.coords)-1<=3 for line,_,_ in cuts)
def test_wide_channel_remains_one_room():
    g=box(0,0,4,4).union(box(6,0,10,4)).union(box(4,1.6,6,2.4))
    assert connection_count(g)==1
    assert len(connected_parts(g)[0])==1
def test_small_raw_island_cannot_hide_behind_large_eroded_component():
    g=box(0,0,4,4).union(box(8,0,8.4,.4))
    assert connection_count(g)==2
    parts,_=connected_parts(g)
    assert sorted(x.area for x in parts)==pytest.approx([.16,16.])
def test_cut_avoids_furniture_and_area_cap_is_a_parameter():
    g=box(0,0,10,8)
    furniture=FurnitureList([dict(instance_id=3,category="kitchen island",geometry=box(4,3,6,5))])
    choices,_=line_candidates(g,furniture,50)
    assert choices and choices[0][3]["furniture_intersection_length_m"]==0
    assert all(len(c[1].coords)-1<=3 for c in choices)
    assert sum(np.ceil(x.area/50) for x in choices[0][2] if x.area>=6)==2
def test_cut_overlap_area_is_a_real_nonzero_strip_measure():
    g=box(0,0,10,8);f=FurnitureList([dict(instance_id=1,category="sofa",geometry=box(4,3,6,5))])
    m=cut_measure(LineString([(5,-2),(5,10)]),g,f,.25)
    assert m["furniture_intersection_length_m"]==pytest.approx(2)
    assert m["furniture_intersection_area_m2"]==pytest.approx(.5)

def test_scan_holes_do_not_create_false_necks_or_change_area():
    holes=[[(x,y),(x+.15,y),(x+.15,y+.15),(x,y+.15)] for x in np.arange(.2,3.8,.3) for y in np.arange(.2,3.8,.3)]
    g=Polygon(box(0,0,4,4).exterior.coords,holes)
    assert connection_count(g)==1
    pieces,cuts=connected_parts(g)
    assert len(pieces)==1 and not cuts
    assert pieces[0].area==pytest.approx(g.area)
    assert len(pieces[0].interiors)==len(holes)

def test_visibility_failure_does_not_throw_away_valid_sibling(monkeypatch):
    from tools.rooms.room_split_auto import connected_split as cs
    from tools.rooms.room_split_auto.connected_refine import solve_partial,PARTIAL
    g=box(0,0,12,5);line=LineString([(6,-2),(6,7)]);children=cs.split_geometry(g,line)
    measure=cs.cut_measure(line,g,())
    def choices(part,*args,**kwargs):
        return ([((),line,children,measure)],1) if part.area>50 else ([],0)
    monkeypatch.setattr(cs,"line_candidates",choices)
    class Cutter:
        cap=50;attempts=0;furniture=();audit=[]
        def visibility(self,part):return {"meets_visibility":part.bounds[2]<=6.00001}
    leaves,cuts,error=solve_partial(Cutter(),g)
    assert error==PARTIAL and len(cuts)==1
    assert sum(x.area for x,v in leaves)==pytest.approx(g.area)
    assert sum(x.area for x,v in leaves if v and v["meets_visibility"])==pytest.approx(30)
    assert sum(x.area for x,v in leaves if v is None)==pytest.approx(30)

def test_cut_guidance_includes_foreign_region_furniture():
    from types import SimpleNamespace
    from tools.rooms.room_split_auto.connected_split import furniture_for
    tri=np.array([[[0,.7,0],[1,.7,0],[1,.7,1]],[[0,.7,0],[1,.7,1],[0,.7,1]]])
    scene=SimpleNamespace(instances=[dict(instance_id=99,region_id=8,category="table",role="blocker",triangles=tri)])
    f=furniture_for(scene,1,0,{},box(-1,-1,2,2))
    assert len(f)==1 and f[0]["instance_id"]==99 and f[0]["geometry"].area==pytest.approx(1)
def test_single_step_alias_excludes_only_step_not_adjacent_flat_ground():
    from tools.rooms.room_split_auto.connected_split import partition_stair_aliases
    tri=np.array([[[1,.1,1],[2,.1,1],[2,.1,2]],[[1,.1,1],[2,.1,2],[1,.1,2]]])
    marker=dict(instance_id=42,category="step",triangles=tri,semantic_source="readonly.semantic.glb")
    normal,stairs,audit=partition_stair_aliases(box(0,0,5,5),0,[marker],.3)
    assert normal.area==pytest.approx(24)
    assert sum(g.area for g in stairs)==pytest.approx(1)
    assert normal.area+sum(g.area for g in stairs)==pytest.approx(25)
    assert connection_count(normal)==1
