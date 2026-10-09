import math
import pytest
import shapely
from shapely.geometry import box,LineString,shape
from tools.rooms.room_split_auto.shape_quality_geometry import own_outline,defects,disk,PartConnectivity,Repair,wrap_count,raw_cells
from tools.rooms.room_split_auto.connected_split import FurnitureList

pytestmark=pytest.mark.fast_unit

def furniture(g):
    return FurnitureList([dict(instance_id=1,category="sofa",region_id=0,geometry=g)])

def test_navmesh_does_not_widen_same_part_neck():
    g=shapely.union_all([box(0,0,5,4),box(5,1.7,6,2.1),box(6,0,9,4)])
    ctx=PartConnectivity(box(-1,-1,10,5))
    assert len(ctx.groups(g))==1
    assert ctx.envelope(g).equals(own_outline(g))
    assert defects(g)["neck_count"]==1

def test_navmesh_bridge_only_between_parts_and_excludes_neighbour():
    a,b=box(0,0,4,4),box(4.2,0,8.2,4)
    g=shapely.union_all([a,b]);nav=box(0,0,8.2,4)
    ctx=PartConnectivity(nav)
    assert len(ctx.groups(g))==1
    for link in ctx.certificate(g)["links"]:
        if link["kind"]=="distinct_parts_pair_overlap_native_nav":
            bridge=shape(link["bridge_geometry_xz_m"])
            assert bridge.difference(a.buffer(.3,join_style=2).intersection(b.buffer(.3,join_style=2)).intersection(nav)).area<1e-8
    blocked=PartConnectivity(nav,box(4,0,4.2,4))
    assert len(blocked.groups(g))==2
    assert ctx.envelope(g).equals(own_outline(g))

@pytest.mark.parametrize("gap,expected",[(0.,1),(.04,1),(.2,2),(1.6,2)])
def test_no_nav_far_part_is_not_a_walkable_bridge(gap,expected):
    g=shapely.union_all([box(0,0,4,4),box(4+gap,0,8+gap,4)])
    assert len(PartConnectivity(box(0,0,4,4)).groups(g))==expected

def test_short_alcove_is_not_corridor():
    g=box(0,0,6,4).union(box(6,1,8,2))
    assert defects(g)["corridor_count"]==0

def test_long_corridor_is_cut_away_without_losing_wide_room():
    g=box(0,0,6,4).union(box(6,1,14,2))
    assert defects(g)["corridor_count"]==1
    rr,cc=Repair(FurnitureList([]),0).solve(g)
    assert all(x["error"] is None for x in rr)
    kept=[x["g"] for x in rr if not x["forced"]]
    gone=[x["g"] for x in rr if x["forced"]=="CORRIDOR"]
    assert kept and gone
    assert shapely.union_all(kept).intersection(box(6,1,14,2)).area<1e-6
    assert all(not defects(x)["corridor_count"] and disk(x)["fits"] for x in kept)
    assert box(.1,.1,5.9,3.9).difference(shapely.union_all(kept)).area<1e-8
    assert abs(sum(x["g"].area for x in rr)-g.area)<1e-6
    assert all(len(c[0].coords)-1<=3 for c in cc)

def test_neck_small_lobe_has_explicit_neck_fragment_reason():
    g=box(0,0,6,4).union(box(6,1.8,7,2.2)).union(box(7,1,9,3))
    rr,cc=Repair(FurnitureList([]),0).solve(g)
    assert any(x["forced"]=="NECK_FRAGMENT" for x in rr)
    assert all(not defects(x["g"])["neck_count"] for x in rr if not x["forced"] and not x["error"])
    assert abs(sum(x["g"].area for x in rr)-g.area)<1e-6

def test_room_cut_avoids_furniture_and_does_not_create_undersize_wedge():
    g=box(0,0,10,4)
    rr,cc=Repair(furniture(box(4.5,-1,5.5,5)),0).solve(g)
    assert len(rr)==2 and all(not x["forced"] and not x["error"] for x in rr)
    assert all(6<=x["g"].area<=35 and disk(x["g"])["fits"] for x in rr)
    assert all(c[2]["furniture_intersection_length_m"]<=.5 for c in cc)
    assert wrap_count([x["g"] for x in rr])==0

def test_half_plane_raw_area_preserves_holes():
    g=box(0,0,10,4).difference(box(1,1,2,2))
    cells=raw_cells(g,LineString([(5,-5),(5,8)]))
    assert len(cells)==2 and abs(sum(x.area for x in cells)-g.area)<1e-8
    assert sum(len(p.interiors) for c in cells for p in ([c] if c.geom_type=="Polygon" else c.geoms))==1

def test_two_point_four_m_circle_is_shape_only_and_no_navigation_needed():
    assert disk(box(0,0,2.4,7))["fits"]
    assert not disk(box(0,0,2.3,7))["fits"]

def test_island_inside_a_filled_hole_is_not_a_zero_gap_seam():
    outer=box(0,0,10,10).difference(box(2,2,8,8))
    island=box(4,4,5,5)
    g=shapely.union_all([outer,island])
    ctx=PartConnectivity(outer)
    assert outer.distance(island)>1
    assert len(ctx.groups(g))==2
    assert not ctx.certificate(g)["links"]

def test_removed_cut_crossing_a_new_boundary_is_not_still_active():
    from tools.rooms.room_split_auto.shape_quality_geometry import surviving_design
    line=LineString([(0,-5),(0,5)])
    crossing=LineString([(-2,0),(2,0)]).intersection(line.buffer(1e-6))
    assert surviving_design(line,crossing).is_empty

def test_surviving_cut_span_counts_furniture_across_scan_holes():
    from tools.rooms.room_split_auto.shape_quality_geometry import surviving_design
    from tools.rooms.room_split_auto.connected_split import cut_measure
    line=LineString([(0,0),(0,10)])
    interfaces=shapely.union_all([LineString([(0,1),(0,4)]),LineString([(0,6),(0,8)])])
    live=surviving_design(line,interfaces)
    assert live.length==pytest.approx(7)
    m=cut_measure(live,box(-2,0,2,10),furniture(box(-.2,4.5,.2,5.5)))
    assert m["furniture_intersection_length_m"]==pytest.approx(1)

def test_bent_wall_axis_partition_preserves_floor_holes():
    line=LineString([(5,-10),(5,2),(20,2)])
    g=box(0,0,10,6).difference(box(1,1,2,2)).difference(box(7,4,8,5))
    cells=raw_cells(g,line)
    assert len(cells)==2
    assert shapely.union_all(cells).symmetric_difference(g).area<1e-8
    assert abs(sum(x.area for x in cells)-g.area)<1e-8

def test_metadata_only_change_cannot_replace_frozen_region():
    import copy
    from shapely.geometry import mapping
    from tools.rooms.room_split_auto.shape_quality_delivery import same_layout
    old=dict(blocks=[dict(decision="retain",floor_id="F0",floor_polygon_xz_m=mapping(box(0,0,4,4)))],cut_lines=[])
    new=copy.deepcopy(old)
    new["cut_lines"]=[dict(active_in_final_partition=False,line_geometry_xz_m=mapping(LineString([(2,-1),(2,5)])),floor_id="F0")]
    assert same_layout(old,new)
    new["blocks"][0]["floor_polygon_xz_m"]=mapping(box(0,0,3.9,4))
    assert not same_layout(old,new)

def test_joint_replan_replaces_narrow_l_cell_with_two_wide_cells():
    from shapely.geometry import mapping
    from tools.rooms.room_split_auto.shape_quality_repair import repair_floor
    a=box(0,0,2,7).union(box(2,5,6,7));b=box(2,0,6,5)
    assert not disk(a)["fits"]
    total=a.union(b)
    old=dict(wall_axes={"F0":{"primary_deg":0}},blocks=[
        dict(id="a",floor_id="F0",decision="retain",floor_polygon_xz_m=mapping(a)),
        dict(id="b",floor_id="F0",decision="retain",floor_polygon_xz_m=mapping(b))],cut_lines=[])
    floor=dict(floor_id="F0",floor_polygon=mapping(total))
    stable,leaves,cuts,axis,audit=repair_floor(old,floor,total,FurnitureList([]))
    assert any(x["status"] in ("joint_floor_replan","local_circle_boundary_repositioned") for x in audit)
    kept=[x["g"] for x in leaves if not x["forced"] and not x["error"]]
    assert len(kept)==2 and all(disk(x)["fits"] and 6<=x.area<=35 for x in kept)
    assert wrap_count(kept)==0
    assert abs(sum(x["g"].area for x in leaves)-total.area)<1e-8

def test_new_cut_cannot_measure_an_unrelated_collinear_boundary():
    from shapely.geometry import mapping
    from tools.rooms.room_split_auto.shape_quality_repair import evaluate_interfaces
    geoms=[box(0,0,1.5,4),box(1.5,0,3,4),box(0,5,1.5,9),box(1.5,5,3,9)]
    blocks=[dict(id=str(i),floor_id="F0",floor_polygon_xz_m=mapping(g),decision="retain") for i,g in enumerate(geoms)]
    line=LineString([(1.5,-2),(1.5,12)])
    cut=dict(id="Q0",floor_id="F0",type="visibility",line_xz_m=list(line.coords),line_geometry_xz_m=mapping(line),applied_parent_geometry_xz_m=mapping(box(0,0,3,4)),furniture_intersection_length_m=0.)
    result=dict(blocks=blocks,cut_lines=[cut])
    evaluate_interfaces(result,{"F0":furniture(box(1,6,2,8))})
    assert cut["active_in_final_partition"]
    assert cut["furniture_intersection_length_m"]==0.
    assert shape(cut["current_design_span_geometry_xz_m"]).length==pytest.approx(4)

def test_degenerate_preview_candidate_does_not_abort_all_candidates(monkeypatch):
    import tools.rooms.room_split_auto.shape_quality_geometry as geom
    original=geom.defects
    g=box(0,0,10,4)
    def checked(poly):
        if abs(poly.area-20)<1e-8:
            raise shapely.GEOSException("transient candidate-only non-noded intersection")
        return original(poly)
    monkeypatch.setattr(geom,"defects",checked)
    choices=Repair(FurnitureList([]),0).choices(g,fallback=False)
    assert choices
    assert all(abs(raw_cells(g,line)[0].area-20)>1e-8 for _,line,_ in choices)

def test_delivery_and_review_use_the_measurement_cut_span_without_mutating_attempt():
    from shapely.geometry import mapping
    from tools.rooms.room_split_auto.shape_quality_delivery import canonical_interfaces
    long=mapping(LineString([(0,0),(0,10)]));current=mapping(LineString([(0,2),(0,4)]))
    attempt=dict(cut_lines=[dict(final_interface_geometry_xz_m=long,current_design_span_geometry_xz_m=current)])
    final=canonical_interfaces(attempt)
    assert shape(final["cut_lines"][0]["final_interface_geometry_xz_m"]).length==2
    assert shape(attempt["cut_lines"][0]["final_interface_geometry_xz_m"]).length==10

def test_circle_boundary_repair_keeps_both_large_parts():
    from shapely.geometry import mapping
    from tools.rooms.room_split_auto.shape_quality_repair import local_circle_repair
    a,b=box(0,0,2.35,4),box(2.35,0,6,4)
    prior=[dict(id=str(i),floor_polygon_xz_m=mapping(g)) for i,g in enumerate([a,b])]
    leaves,cuts,audit,ok=local_circle_repair(prior,FurnitureList([]),0)
    assert ok and len(leaves)==2
    assert all(not x["forced"] and not x["error"] and disk(x["g"])["fits"] for x in leaves)
    assert abs(sum(x["g"].area for x in leaves)-a.union(b).area)<1e-8
    assert all(len(cut[0].coords)-1<=3 for cut in cuts)
