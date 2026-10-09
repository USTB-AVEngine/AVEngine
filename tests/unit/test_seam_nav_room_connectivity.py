"""Regressions for the owner-authorized seam/native-navmesh connectivity rule."""
import pytest
import shapely
from shapely.geometry import box,LineString,GeometryCollection
from tools.rooms.room_split_auto.seam_connectivity import Connectivity,exterior
from tools.rooms.room_split_auto import walkable_split as w,capped_split as c,connected_split as cs
pytestmark=pytest.mark.fast_unit

@pytest.fixture
def install(monkeypatch):
    monkeypatch.setattr(c.Cutter,"solve",c.Cutter.solve)
    for module,keys in [(c,["outline","room_circle","connection_count","candidates","split_at_chord","connected_parts","split_corridor"]),
                        (cs,["filled_footprint","split_geometry"])]:
        for key in keys:monkeypatch.setattr(module,key,getattr(module,key))
    yield w.install
    c.room_circle.cache_clear();c.opened.cache_clear();c.broad_components.cache_clear()

def test_floor_seam_retains_semantic_area_and_fits_full_room_disk(install):
    g=shapely.union_all([box(0,0,1.49,4),box(1.51,0,3,4)])
    ctx=Connectivity(GeometryCollection());install(ctx)
    assert len(ctx.groups(g))==1 and ctx.count(g)==1
    assert c.room_circle(g)["fits"]
    assert g.area==pytest.approx(11.92)
    assert ctx.certificate(g)["links"][0]["kind"]=="seam_distance_le_0_05"

def test_point_touching_floor_parts_are_seams():
    g=shapely.union_all([box(0,0,4,4),box(4,4,8,8)])
    ctx=Connectivity(GeometryCollection())
    assert len(ctx.groups(g))==1
    assert ctx.count(g)==1

def test_door_threshold_nav_support_connects_and_wall_blocks():
    a=box(0,0,4,4);b=box(4.19,0,8.19,4);g=a.union(b)
    open_ctx=Connectivity(box(0,0,8.19,4))
    wall_ctx=Connectivity(box(0,0,3.8,4).union(box(4.39,0,8.19,4)))
    assert open_ctx.count(g)==1
    assert wall_ctx.count(g)==2
    assert open_ctx.certificate(g)["links"][0]["kind"]=="native_navmesh_local_direct_passage"
    assert g.area==pytest.approx(32)

def test_distant_island_does_not_use_whole_house_detour():
    a=box(0,0,4,4);b=box(5.64,0,9.64,4);g=a.union(b)
    ctx=Connectivity(box(0,0,20,20))
    assert ctx.count(g)==2 and len(ctx.groups(g))==2

def test_scan_patch_inside_filled_hole_is_not_a_detached_room():
    main=box(0,0,5,4).difference(box(1,1,2,2));patch=box(1.2,1.2,1.5,1.5)
    g=main.union(patch);ctx=Connectivity(GeometryCollection())
    assert ctx.count(g)==1
    assert len(ctx.groups(g))==1
    assert g.area==pytest.approx(19.09)

def test_design_cut_separates_seam_without_inventing_ground():
    g=box(0,0,4,4).union(box(4.19,0,8.19,4));ctx=Connectivity(box(0,0,8.19,4))
    children=ctx.split(g,LineString([(4.1,-2),(4.1,6)]))
    assert len(children)==2 and all(ctx.count(p)==1 for p in children)
    assert sum(p.area for p in children)==pytest.approx(g.area)
    assert shapely.union_all(children).symmetric_difference(g).area<1e-8

def test_invalid_cut_does_not_change_connectivity_mask():
    g=box(0,0,4,4);ctx=Connectivity(g)
    assert not ctx.split(g,LineString([(10,-1),(10,6)]))
    assert g.wkb not in ctx.masks
    assert ctx.envelope(g).equals(g)

def test_two_leg_room_cut_has_only_wall_axes_and_disks(install):
    g=box(0,0,5,8);ctx=Connectivity(g);install(ctx)
    choices=w.two_leg_candidates(g,cs.FurnitureList([]),35,0)
    assert choices
    for _,line,children,_ in choices:
        assert len(line.coords)-1==2
        assert max(c.angle_errors(line,0))<1e-8
        assert all(c.room_circle(p)["fits"] for p in children if p.area>=6)
        assert shapely.union_all(children).symmetric_difference(g).area<1e-7


def test_local_corridor_chord_preserves_wide_remote_lobe(install):
    g=shapely.union_all([box(0,0,4,4),box(4,1.5,12,2.5)])
    ctx=Connectivity(g);install(ctx)
    choices=w.finite_corridor_candidates(g,cs.FurnitureList([]),0)
    assert choices
    _,line,children,measure=choices[0]
    assert len(line.coords)-1==1 and max(c.angle_errors(line,0))<1e-8
    assert any(q.area>=6 and c.room_circle(q)["fits"] for q in children)
    assert all(not c.broad_components(q) for q in children if not c.room_circle(q)["fits"])
    assert shapely.union_all(children).symmetric_difference(g).area<1e-7


def test_proxy_precision_retry_does_not_change_ground_area(monkeypatch):
    from tools.rooms.room_split_auto import seam_connectivity as sc
    original=shapely.intersection;called=[]
    def flaky(a,b,*args,**kwargs):
        called.append(kwargs.get("grid_size"))
        if len(called)==1:raise shapely.GEOSException("non-noded proxy operation")
        return original(a,b,*args,**kwargs)
    monkeypatch.setattr(shapely,"intersection",flaky)
    a=box(0,0,3,4);b=box(1,1,5,6);raw=a.union(b);area=raw.area
    joined=sc.proxy_intersection(a,b)
    assert joined.area==pytest.approx(6)
    assert called==[None,1e-8]
    assert raw.area==area


def test_cap_search_budget_still_salvages_broad_room_from_corridor(install,monkeypatch):
    import numpy as np
    g=shapely.union_all([box(0,0,4,4),box(4,1.7,39,2.3)])
    ctx=Connectivity(g)
    monkeypatch.setattr(w,"CORRIDOR_JUNCTION_SEARCH_M",2.4)
    monkeypatch.setattr(w,"CORRIDOR_ELBOW_VERTEX_REFINEMENT",True)
    install(ctx)
    cutter=c.Cutter(None,np.empty((0,3)),g,{},dict(coverage=.8,max_distance=6),35,cs.FurnitureList([]),0)
    cutter.attempts=36
    monkeypatch.setattr(c.Cutter,"visibility",lambda self,q:dict(coverage_fraction=0.,meets_visibility=False))
    leaves,cuts=cutter.solve(g)
    assert cuts
    assert sum(q.area for q,vis,error in leaves)==pytest.approx(g.area)
    broad=[q for q,vis,error in leaves if q.area>=6 and c.room_circle(q)["fits"]]
    assert broad and all(q.area<=35+1e-8 for q in broad)
    assert all(error is None for q,vis,error in leaves)
    assert all(len(line.coords)-1<=2 for line,*_ in cuts)


@pytest.mark.parametrize("width,expected",[(2.4,True),(2.3999999,True),(2.3999997,False),(2.3,False),(3.0,True)])
def test_efficient_circle_has_same_inclusive_frozen_diameter(width,expected,install):
    g=box(0,0,width,4);ctx=Connectivity(g);install(ctx)
    assert c.room_circle(g)["fits"] is expected
    assert w.BASE_ROOM_CIRCLE(g)["fits"] is expected
