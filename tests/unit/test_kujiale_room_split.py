"""Contract tests for recorded transforms, exact shapes and real CPU review."""
from types import SimpleNamespace
import numpy as np
import pytest
import shapely
from shapely.geometry import box, Polygon, Point
from tools.rooms.room_split_kujiale.adapter import transform_points, cad_polygon, wall_axis
from tools.rooms.room_split_kujiale.pipeline import raw_disk, witness_check
from tools.rooms.room_split_kujiale.raster import raster_all
from tools.rooms.room_split_kujiale.usd_mesh import mesh_arrays

pytestmark=pytest.mark.fast_unit
M=np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],float)


def test_recorded_matrix_preserves_y_back_sign_and_floor():
    actual=transform_points([[2,7,.03],[-1,-4,.03]],M)
    np.testing.assert_allclose(actual,[[2,.03,7],[-1,.03,-4]])
    g,e=cad_polygon({'polygon':[[1,2],[5,2],[5,6],[1,6]]},M,.03)
    assert g.bounds==(1.,2.,5.,6.) and g.area==16
    assert e['transform_trials']==1 and e['source_floor_z_m']==.03


def test_recorded_affine_transform_translations_are_not_discarded():
    m=M.copy();m[:3,3]=[3,4,5]
    g,e=cad_polygon({'polygon':[[0,0],[3,0],[3,3],[0,3]]},m,4.2)
    assert g.bounds==(3.,5.,6.,8.)
    assert e['source_floor_z_m']==pytest.approx(.2)


def test_nonhorizontal_cad_transform_stops_instead_of_guessing():
    m=M.copy();m[1,0]=.1
    with pytest.raises(ValueError,match='horizontal'):cad_polygon({'polygon':[[0,0],[3,0],[3,3]]},m,0)


def test_invalid_cad_keeps_all_real_components():
    g,e=cad_polygon({'polygon':[[0,0],[4,4],[0,4],[4,0]]},M,0)
    assert not e['original_valid'] and g.area==8 and g.geom_type=='MultiPolygon'


def test_wall_axes_follow_rotated_cad_walls():
    from shapely.affinity import rotate
    r=wall_axis(rotate(box(0,0,9,4),17,origin=(0,0)))
    assert r['primary_deg']==pytest.approx(17)
    assert r['directional_concentration']==pytest.approx(1)


def test_circle_certificate_cannot_fill_real_floor_void():
    g=box(0,0,6,6).difference(box(1,1,5,5))
    assert not raw_disk(g)['fits']
    assert raw_disk(box(0,0,2.4,4))['fits']


def test_original_usd_composed_world_matrix_and_holes():
    from pxr import Usd,UsdGeom,Gf
    stage=Usd.Stage.CreateInMemory();root=UsdGeom.Xform.Define(stage,'/Root')
    root.AddTranslateOp().Set(Gf.Vec3d(2,3,4))
    m=UsdGeom.Mesh.Define(stage,'/Root/Mesh');m.GetPointsAttr().Set([(0,0,0),(1,0,0),(0,1,0),(1,1,0)])
    m.GetFaceVertexCountsAttr().Set([3,3]);m.GetFaceVertexIndicesAttr().Set([0,1,2,1,3,2]);m.GetHoleIndicesAttr().Set([1])
    vertices,faces,ids=mesh_arrays(m.GetPrim(),UsdGeom.XformCache())
    np.testing.assert_allclose(vertices[0],[2,3,4]);assert faces.tolist()==[[0,1,2]] and ids.tolist()==[0]


def test_cpu_depth_buffer_keeps_upper_real_triangle_colour():
    v=np.array([[-1,0,-1],[1,0,-1],[0,0,1],[-1,.7,-1],[1,.7,-1],[0,.7,1]],np.float32)
    f=np.array([[0,1,2],[3,4,5]],int);col=np.array([[1,0,0],[0,1,0]],np.float32)
    im,d,rendered,hidden=raster_all(v,f,col,np.zeros(2,bool),np.zeros(2,bool),0,0,0,4,40)
    assert d[20,20]==pytest.approx(.7)
    assert im[20,20,1]>200 and im[20,20,0]==0 and rendered==2 and hidden==0


def test_ceiling_hiding_does_not_recolour_floor():
    v=np.array([[-1,0,-1],[1,0,-1],[0,0,1],[-1,3,-1],[1,3,-1],[0,3,1]],np.float32)
    f=np.array([[0,1,2],[3,4,5]],int);col=np.array([[1,0,0],[0,0,1]],np.float32)
    im,d,n,h=raster_all(v,f,col,np.zeros(2,bool),np.array([False,True]),0,0,0,4,40)
    assert d[20,20]==0 and im[20,20,0]>200 and im[20,20,2]==0 and h==1


def test_witness_outside_actual_polygon_is_rejected():
    class PF:
        def snap_point(self,p):return np.array(p)
    class Rays:
        def intersects_location(self,o,d,multiple_hits=False):return np.empty((0,3)),np.empty(0,int),np.empty(0,int)
    w=dict(found=True,camera_m=[1,1.5,1],source_1_m=[2,1.2,2],source_2_m=[9,1.2,2])
    p=dict(camera_height_m=1.5,source_height_m=1.2,ray_endpoint_tolerance_m=.03)
    got=witness_check(w,box(0,0,5,5),SimpleNamespace(ray=Rays()),PF(),p)
    assert not got['found'] and got['validation']['status']=='fail'
    assert not got['validation']['support_checks'][2]['inside_real_polygon']


def test_embedded_review_has_no_eager_image_src(tmp_path):
    import json,re
    from PIL import Image
    from tools.rooms.room_split_kujiale.review import build
    root=tmp_path
    def put(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))
    put(root/'kujiale_room_list_draft_v1/summary.json',dict(total_rooms=1,total_area_m2=12,pending=0))
    put(root/'comparison_old_33_rooms_v1.json',dict(old_admitted=[]))
    image=root/'actual.jpg';Image.new('RGB',(30,30),(100,150,200)).save(image)
    put(root/'owner_overlays_all_sources_v1/index.json',[dict(region='kujiale_test/R0',images=[str(image)])])
    block=dict(id='kujiale_test__R0__F0__K000',decision='retain',floor_area_m2=12,short_side_m=3,placement_witness=dict(found=True),discard_reasons=[],unresolved_reasons=[])
    reg=dict(house='kujiale_test',source_region='R0',requires_split=True,source_floor_area_m2=36,source_geometry=dict(source_room_type='living room'),blocks=[block])
    put(root/'kujiale_delivery_v1/final_v1/regions/kujiale_test__R0.json',reg)
    build(root);page=(root/'KUJIALE_REVIEW_v1.html').read_text()
    tags=re.findall(r'<img\b[^>]*>',page)
    assert len(tags)==2 and all(not re.search(r'\bsrc\s*=',tag) for tag in tags)
    assert '<script id="embedded-jpegs" type="application/json">' in page
    assert 'data:image/jpeg;base64,' in page and "rootMargin:'0px'" in page
    assert page.index('1. 被切的客厅')<page.index('2. 与旧')<page.index('3. 其余保留')


@pytest.mark.parametrize('borrow_other_room',[False,True])
def test_direct_nav_path_cannot_borrow_another_cad_room(borrow_other_room):
    from shapely.geometry import mapping, GeometryCollection
    from tools.rooms.room_split_kujiale.pipeline import connectivity_check
    g=shapely.union_all([box(0,0,3,3),box(3.2,0,6.2,3)])
    bridge=box(3,0,3.2,3)
    class Context:
        def record(self,g):return dict(links=[dict(bridge_geometry_xz_m=mapping(bridge))])
        def certificate(self,g):return dict(test_pair_bridge=True)
    class Pathfinder:
        def find_path(self,p):
            p.points=[p.requested_start,p.requested_end]
            p.geodesic_distance=float(np.linalg.norm(p.requested_end-p.requested_start))
            return True
    points=np.array([[1.,0.,1.5],[4.2,0.,1.5]])
    result=connectivity_check(g,Pathfinder(),SimpleNamespace(ShortestPath=SimpleNamespace),points,Context(),bridge if borrow_other_room else GeometryCollection())
    assert result['pass_'] is (not borrow_other_room)
    assert result['far_paths'][0]['route_in_other_room_floor_m']==pytest.approx(.2 if borrow_other_room else 0.)


def test_committed_v6_cut_keeps_real_circle_and_avoids_furniture():
    pytest.importorskip('tools.rooms.room_split_auto.shape_quality_geometry',reason='committed HM3D v6 must be merged before final split')
    from tools.rooms.room_split_kujiale.pipeline import make_repair
    from tools.rooms.room_split_auto.connected_split import FurnitureList,cut_measure
    from tools.rooms.room_selection.geometry import short_side
    g=box(0,0,8,6);furniture=FurnitureList([dict(instance_id='central_cabinet',category='cabinet',geometry=box(3.9,0,4.1,6))])
    repair=make_repair(furniture,0);leaves,cuts=repair.solve(g)
    assert len(leaves)==2 and all(q['forced'] is None and q['error'] is None for q in leaves)
    assert all(6<=q['g'].area<=35 and short_side(q['g'])>=2.4 and raw_disk(q['g'])['fits'] for q in leaves)
    assert sum(q['g'].area for q in leaves)==pytest.approx(g.area)
    assert cuts and all(cut_measure(line,parent,furniture)['furniture_intersection_length_m']<=.5 for line,kind,m,parent in cuts)
    assert all(len(line.coords)-1==1 for line,kind,m,parent in cuts)


@pytest.mark.parametrize('supported_bridge',[False,True])
def test_postcut_sliver_is_detached_only_without_pair_nav_support(supported_bridge):
    from tools.rooms.room_split_auto.shape_quality_geometry import PartConnectivity
    from tools.rooms.room_split_kujiale.pipeline import post_cut_components
    main=box(0,0,4.5,4);sliver=box(4.7,1,4.7015,2.27)
    g=shapely.union_all([main,sliver])
    nav=box(0,0,4.8,4) if supported_bridge else main
    items,audit=post_cut_components([dict(g=g,forced=None,error=None)],PartConnectivity(nav))
    assert sum(x['g'].area for x in items)==pytest.approx(g.area)
    if supported_bridge:
        assert len(items)==1 and items[0]['forced'] is None and not audit
        assert items[0]['g'].equals(g)
    else:
        assert len(items)==2 and len(audit)==1
        assert items[0]['g'].equals(main) and items[0]['forced'] is None
        assert items[1]['g'].equals(sliver) and items[1]['forced']=='DETACHED_FRAGMENT'
