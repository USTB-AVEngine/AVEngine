"""Structural cuts, true mesh occlusion, and area-based manual matching."""
import numpy as np
import pytest
import shapely
from shapely.geometry import box,LineString
import trimesh
from tools.rooms.room_split_auto.contours import chord,split_at_chord,structural_partition
from tools.rooms.room_split_auto.visibility import grid_atoms,visibility_matrix
from tools.rooms.room_split_auto.cpu_rays import CPUScanRayIntersector
pytestmark=pytest.mark.fast_unit


def joined_rooms():return shapely.union_all([box(0,0,4,4),box(4,1.6,5,2.4),box(5,0,9,4)])

def test_narrow_cut_preserves_unrounded_area_and_wall_endpoints():
    g=joined_rooms();c=chord(g,[4.5,2]);assert c['width_m']==pytest.approx(0.8)
    parts=split_at_chord(g,c['line'],6)
    assert len(parts)==2
    assert sum(x.area for x in parts)==pytest.approx(g.area)
    assert parts[0].intersection(parts[1]).area==0
    assert shapely.union_all(parts).symmetric_difference(g).area<1e-8
    assert all(g.boundary.distance(shapely.Point(p))<1e-7 for p in c['line'].coords)

def test_narrow_side_under_six_is_not_a_room_cut():
    g=shapely.union_all([box(0,0,2,2),box(2,.7,3,1.3),box(3,0,7,4)])
    c=chord(g,[2.5,1]);assert split_at_chord(g,c['line'],6) is None

def test_finite_cut_does_not_slice_a_remote_component():
    g=joined_rooms().union(box(4,10,5,14));c=chord(joined_rooms(),[4.5,2]);parts=split_at_chord(g,c['line'],6)
    assert any(p.intersection(box(4,10,5,14)).area==pytest.approx(4) for p in parts)
    assert shapely.union_all(parts).symmetric_difference(g).area<1e-8

def test_door_provenance_and_unreliable_fallback():
    d={'instance_id':7,'category':'door frame','semantic_source':'fixture.semantic.txt','height_range_m':[0,2],'centre_xz_m':[4.5,2],'axis_angle_deg':90,'extent_xz_m':[.1,.8]}
    parts,cuts,audit=structural_partition(joined_rooms(),[d],0,.6,1.6)
    assert len(parts)==2 and cuts[0]['type']=='door' and cuts[0]['semantic_instance_id']==7
    d['centre_xz_m']=[100,100];parts,cuts,audit=structural_partition(joined_rooms(),[d],0,.6,1.6)
    assert audit[0]['status'].startswith('unreliable')
    assert any(c['type']=='narrow' for c in cuts)

def test_grid_atoms_keep_holes_and_do_not_subtract_furniture():
    g=box(0,0,5,4).difference(box(2,1,2.5,2));points=np.array([[1,0,1],[4,0,3]],float)
    atoms,fw,nw=grid_atoms(g,g,points,.25)
    assert sum(fw)==pytest.approx(g.area) and sum(nw)==pytest.approx(g.area)
    assert shapely.union_all(atoms).intersection(box(2,1,2.5,2)).area==0

def test_raw_mesh_occlusion_changes_visibility():
    wall=trimesh.creation.box(extents=[.1,3,10]);wall.apply_translation([3,1.5,2]);wall.ray=CPUScanRayIntersector(wall)
    points=np.array([[1,0,2],[2,0,2],[5,0,2]])
    p={'camera_height_m':1.5,'source_height_m':1.2,'ray_endpoint_tolerance_m':.03}
    visible,dist=visibility_matrix(wall,points,[0],p,6)
    assert visible[0,1] and not visible[0,2]
    assert wall.ray.receipt['omitted_faces']==0

def test_zero_area_raw_faces_are_preserved_but_do_not_block_rays():
    mesh=trimesh.Trimesh(vertices=[[2,0,0],[2,0,0],[2,0,0]],faces=[[0,1,2]],process=False)
    ray=CPUScanRayIntersector(mesh)
    loc,ids,_=ray.intersects_location(np.array([[0,0,0.]]),np.array([[1,0,0.]]))
    assert len(ids)==0 and ray.receipt['input_faces']==1 and ray.receipt['omitted_faces']==0


def test_stair_footprint_is_separate_without_losing_ground():
    from tools.rooms.room_split_auto.pipeline import stair_partition
    scope=box(0,0,8,5)
    marker={'instance_id':12,'category':'stairs','semantic_source':'fixture.semantic.txt','triangles':np.array([[[0,0,0],[2,0,0],[2,0,2]],[[0,0,0],[2,0,2],[0,0,2]]],float)}
    rooms,stairs,audit=stair_partition(scope,0,[marker],.3)
    assert len(stairs)==1 and stairs[0].area==pytest.approx(4)
    assert rooms.area==pytest.approx(36) and rooms.intersection(stairs[0]).area==0
    assert rooms.union(stairs[0]).symmetric_difference(scope).area<1e-8
    assert audit[0]['semantic_instance_id']==12


def test_partial_atlas_pixels_cannot_certify_scan_black():
    from tools.rooms.room_split_auto.pipeline import measured_black
    from PIL import Image
    entry={'floor_y_m':0,'view':np.eye(4).tolist(),'coverage_geometry_xz_m':shapely.geometry.mapping(box(-2,-2,0,2))}
    im={'projection':[[.5,0,0,0],[0,0,.5,0],[0,1,0,0],[0,0,0,1]],'span_m':4}
    result=measured_black(box(-1,-1,1,1),0,(entry,im,Image.new('RGB',(32,32),'white'),{}),{'floor_height_separation_m':.3,'black_pixel_channel_lt':8})
    assert result['black_fraction'] is None and result['world_floor_in_frame_fraction']==pytest.approx(.5)


def test_manual_iou_uses_floor_holes_and_one_to_one_matching(tmp_path):
    import json
    from tools.rooms.room_split_auto.evaluation import score_reference_pair
    floor=box(0,0,4,4).difference(box(1,1,3,3))
    refs=[]
    for i in range(2):
        path=tmp_path/f'{i}.json';path.write_text(json.dumps({'id':str(i),'bbox_xz_m':[[0,0],[4,4]],'floor_y_m':0}))
        refs.append({'path':str(path)})
    region={'house':'hm3d_fixture','source_region':'R1','status':'unchanged','source_geometry':{'floors':[{'floor_polygon':shapely.geometry.mapping(floor),'floor_y_m':0}]},'blocks':[{'id':'one','decision':'unchanged','floor_y_m':0,'floor_polygon_xz_m':shapely.geometry.mapping(floor)}]}
    result=score_reference_pair(region,refs)
    assert result['source_mean_iou']==pytest.approx(.5)
    assert sorted(e['matched_iou'] for e in result['entries'])==[0,1]
    assert all(e['manual_floor_area_m2']==pytest.approx(12) and e['manual_bbox_area_m2']==16 for e in result['entries'])


def test_medial_body_width_distinguishes_corridor_from_room():
    from tools.rooms.room_split_auto.contours import body_width
    assert body_width(box(0,0,10,1.2))<1.5
    assert body_width(box(0,0,10,3))>2.4


def test_merged_internal_cut_is_not_a_final_interface():
    from tools.rooms.room_split_auto.pipeline import finalize_interfaces
    blocks=[{'floor_id':'F0','floor_polygon_xz_m':shapely.geometry.mapping(g)} for g in (box(0,0,4,4),box(4,0,8,4))]
    cuts=[{'id':name,'floor_id':'F0','type':'narrow','line_xz_m':[[x,0],[x,4]]} for name,x in (('actual',4),('merged',2))]
    finalize_interfaces(blocks,cuts)
    assert cuts[0]['active_in_final_partition'] and not cuts[1]['active_in_final_partition']
    assert all(b['cut_ids']==['actual'] for b in blocks)
