"""Geometric counterexamples for conservative acoustic-only mesh repairs."""
import numpy as np
import pytest
from avengine.acoustics.hole_patch import (loop_candidates,nav_prism_intersections,
    check_direct_paths,exit_clusters,export_derived_glb)
from avengine.acoustics.hole_patch_checks import rays_through_boxes

pytestmark=pytest.mark.fast_unit


def horizontal_ring():
    v=np.array([[-2,2,-2],[2,2,-2],[2,2,2],[-2,2,2],[-.5,2,-.5],[.5,2,-.5],[.5,2,.5],[-.5,2,.5]],float)
    t=np.array([[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    return v,t


def leakage():
    return [{'room_label':'R0','role':'camera_m','origin_m':[0,1,0],
             'escaped_rays':[{'direction_index':0,'direction':[0,1,0],'aabb_exit_m':[0,2,0],'end_m':[0,31,0]}]}]


def test_confirmed_ceiling_hole_is_outside_and_outer_perimeter_is_rejected():
    v,t=horizontal_ring();c,r,clusters,stats=loop_candidates(v,t,['ceiling']*len(t),leakage())
    accepted=[x for x in c if x['status']=='candidate']
    assert len(accepted)==1
    assert np.allclose(np.array(accepted[0]['quad_m'])[:,1],2.03)
    assert accepted[0]['ray_ids']==[0]
    assert stats['excluded_loop_counts']['surface_inside_loop']==1


def test_door_label_cannot_be_relabelled_as_a_scan_hole():
    v,t=horizontal_ring();c,*_=loop_candidates(v,t,['door']*len(t),leakage())
    assert c and all(x['status']=='review' for x in c)
    assert 'UNCERTAIN_SURFACE_SEMANTICS' in c[0]['reasons']


def test_unknown_label_stays_unpatched():
    v,t=horizontal_ring();c,*_=loop_candidates(v,t,['unknown']*len(t),leakage())
    assert c and not any(x['status']=='candidate' for x in c)


def test_direct_line_intersection_detects_quad_interior_and_preserves_parallel_line():
    quad=[[-1,1,-1],[1,1,-1],[1,1,1],[-1,1,1]]
    pairs=[{'listener_m':[0,0,0],'source_m':[0,2,0]}, {'listener_m':[0,0,0],'source_m':[1,0,0]}]
    assert check_direct_paths(quad,pairs)==pairs[:1]


def test_narrow_navigable_prism_is_detected_without_sampling():
    nav=np.array([[0,0,0],[.02,0,0],[0,0,.02]])
    quad=[[-.01,.5,-.01],[.01,.5,-.01],[.01,.5,.01],[-.01,.5,.01]]
    assert nav_prism_intersections(quad,nav,[[0,1,2]])==[0]


def test_floor_patch_below_supported_floor_does_not_enter_walkable_volume():
    nav=np.array([[-1,0,-1],[1,0,-1],[-1,0,1]])
    quad=[[-.1,-.03,-.1],[.1,-.03,-.1],[.1,-.03,.1],[-.1,-.03,.1]]
    assert nav_prism_intersections(quad,nav,[[0,1,2]])==[]


def test_aabb_exit_clusters_never_merge_different_faces():
    rays=[{'aabb_exit_m':[1,.1,0]},{'aabb_exit_m':[1,.2,0]},{'aabb_exit_m':[.9,1,0]}]
    groups=exit_clusters(rays,[[-1,-1,-1],[1,1,1]],radius=2)
    assert sorted(len(g['ray_ids']) for g in groups)==[1,2]


def test_opening_box_parallel_outside_ray_is_not_visible_opening():
    boxes=[{'bounds_m':[[1,0,0],[2,1,1]]}]
    o=np.array([[0,.5,.5],[1.5,.5,-1]])
    d=np.array([[0,0,1],[0,0,1]])
    assert rays_through_boxes(o,d,np.array([5.,5.]),boxes).tolist()==[False,True]


def test_continuous_opening_cone_finds_a_thin_aperture_between_grid_samples():
    from avengine.acoustics.hole_patch_checks import opening_cone_overlaps
    q=[[3,0,0],[3,1,0],[3,1,1],[3,0,1]]
    cams=[{'position_m':[0,.333,.333]}]
    boxes=[{'bounds_m':[[1,.3331,.3331],[1.01,.334,.334]]}]
    assert opening_cone_overlaps(q,cams,boxes)


def test_opening_behind_camera_does_not_prohibit_a_forward_patch():
    from avengine.acoustics.hole_patch_checks import opening_cone_overlaps
    q=[[3,0,0],[3,1,0],[3,1,1],[3,0,1]]
    cams=[{'position_m':[0,.333,.333]}]
    boxes=[{'bounds_m':[[-2,.3,.3],[-1,.4,.4]]}]
    assert opening_cone_overlaps(q,cams,boxes)==[]


def test_export_keeps_base_coordinates_faces_and_material_and_only_appends_quad(tmp_path):
    import json
    from avengine.acoustics.gltf import extract_triangle_scene
    v,t=horizontal_ring();v=v.astype('<f4');t=t.astype('<u4')
    np.save(tmp_path/'vertices.npy',v);np.save(tmp_path/'triangles.npy',t)
    matrix=np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]],float)
    m={'arrays':{'vertices':{'path':'vertices.npy'},'triangles':{'path':'triangles.npy'}},
       'geometry':{'source_to_canonical':{'matrix_row_major':matrix.ravel().tolist()}},
       'objects':[{'object_id':'original','source_material_name':'ceiling','vertex_offset':0,'vertex_count':len(v),'triangle_offset':0,'triangle_count':len(t)}]}
    p=tmp_path/'manifest.json';p.write_text(json.dumps(m))
    patch={'candidate_id':'patch','semantic_label':'ceiling','quad_m':[[-.5,2.03,-.5],[.5,2.03,-.5],[.5,2.03,.5],[-.5,2.03,.5]]}
    out=tmp_path/'derived.glb';export_derived_glb(p,[patch],out)
    scene=extract_triangle_scene(out)
    canonical=(scene.vertices@matrix[:3,:3].T).astype('<f4')
    assert np.array_equal(canonical[:len(v)],v)
    assert np.array_equal(scene.triangles[:len(t)],t)
    assert len(scene.vertices)==len(v)+4 and len(scene.triangles)==len(t)+2
    assert {o['source_material_name'] for o in scene.objects}=={'ceiling'}
