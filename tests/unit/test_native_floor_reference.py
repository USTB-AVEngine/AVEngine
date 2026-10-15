import json
from pathlib import Path
import numpy as np
import pytest
from avengine.capture.native_floor_reference import NativeFloorReference,file_sha256

pytestmark=pytest.mark.fast_unit


def floor(tmp_path):
    vertices=np.array([[-2,0,-2],[2,0,-2],[2,0,2],[-2,0,2]],dtype='<f4')
    vp=tmp_path/'vertices.f32';(vertices[:,[0,2,1]]*100).astype('<f4').tofile(vp)
    tp=tmp_path/'triangles.u32';np.array([[0,1,2],[0,2,3]],dtype='<u4').tofile(tp)
    ap=tmp_path/'audit.json';ap.write_text(json.dumps({'native_render_geometry':{'world_vertices_ue_cm_f32':str(vp),'triangles_u32':str(tp)}}))
    declaration={'native_editor_audit':str(ap),'native_editor_audit_sha256':file_sha256(ap),
                 'native_vertices_sha256':file_sha256(vp),'native_triangles_sha256':file_sha256(tp)}
    return declaration


def probes(y=0.):
    return {'probes':[{'status':'measured','witness_index':i,'role':role,'nav_point_m':[i*.1,.04,0.],
                       'native_visual_floor_y_m':y} for i in range(3) for role in ('camera','source_1','source_2')]}


def test_spatial_floor_grounding_preserves_horizontal_nav_and_original_mesh(tmp_path):
    declaration=floor(tmp_path);reference=NativeFloorReference(declaration)
    assert reference.sample([.5,.04,-.5])==[.5,0.,-.5]
    assert reference.validate_probes(probes())['status']=='pass'
    assert file_sha256(tmp_path/'vertices.f32')==declaration['native_vertices_sha256']


def test_missing_visual_support_never_teleports_actor(tmp_path):
    with pytest.raises(ValueError,match='missing'):NativeFloorReference(floor(tmp_path)).sample([3.,0.,0.])


def test_navigation_height_outlier_keeps_the_existing_tolerance(tmp_path):
    with pytest.raises(ValueError,match='snap tolerance'):NativeFloorReference(floor(tmp_path)).sample([0.,.2,0.])


def test_no_consistent_witness_requires_fresh_native_contact_without_claiming_agreement(tmp_path):
    reference=NativeFloorReference(floor(tmp_path));result=reference.validate_probes(probes(.03))
    assert result['status']=='fresh_native_construction_required'
    assert result['checks']==[] and len(result['excluded_probes'])==9
    assert reference.room_floor_y_m is None and reference.proposal_floor_y_m==0.
    assert result['proposal_is_contact_admission'] is False
    assert reference.sample([.5,.04,-.5])==[.5,0.,-.5]


def test_one_bad_depth_probe_does_not_discard_eight_consistent_native_points(tmp_path):
    reference=NativeFloorReference(floor(tmp_path));witnesses=probes();witnesses['probes'][0]['native_visual_floor_y_m']=.2
    result=reference.validate_probes(witnesses)
    assert result['status']=='pass' and result['retained_probe_count']==8
    assert len(result['excluded_probes'])==1 and reference.room_floor_y_m==0.
    assert result['native_depth_tolerance_m']==.02


def test_unmeasured_or_nonfloor_witness_is_excluded_but_bad_support_stays_rejected(tmp_path):
    reference=NativeFloorReference(floor(tmp_path));witnesses=probes();witnesses['probes'][0]['status']='unmeasured'
    witnesses['probes'][1]['nav_point_m']=[3.,.04,0.]
    result=reference.validate_probes(witnesses)
    assert result['status']=='pass' and result['retained_probe_count']==7
    assert len(result['excluded_probes'])==2
    with pytest.raises(ValueError,match='missing'):reference.sample([3.,.04,0.])


def test_geometry_mutation_is_rejected_before_actor_spawn(tmp_path):
    declaration=floor(tmp_path);(tmp_path/'vertices.f32').write_bytes(b'changed')
    with pytest.raises(ValueError,match='geometry identity'):NativeFloorReference(declaration)


def test_a_bed_surface_cannot_pass_as_floor_even_with_perfect_foot_contact(tmp_path):
    declaration=floor(tmp_path)
    vp=tmp_path/'vertices.f32';tp=tmp_path/'triangles.u32';ap=tmp_path/'audit.json'
    raw=np.fromfile(vp,dtype='<f4').reshape(-1,3)
    bed=raw.copy();bed[:,2]=53.6
    np.concatenate([raw,bed]).astype('<f4').tofile(vp)
    np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]],dtype='<u4').tofile(tp)
    audit=json.loads(ap.read_text());audit['native_render_geometry']['sections']=[
        {'triangle_count':2,'material_path':'/Game/Imported/Root/Meshes/floor/floor_0/Looks/MI'},
        {'triangle_count':2,'material_path':'/Game/Imported/Root/Meshes/bedroom/bed_0/Looks/MI'}]
    ap.write_text(json.dumps(audit))
    declaration.update(native_editor_audit_sha256=file_sha256(ap),native_vertices_sha256=file_sha256(vp),native_triangles_sha256=file_sha256(tp))
    reference=NativeFloorReference(declaration)
    assert reference.sample([0,.01,0])==[0.,0.,0.]
    with pytest.raises(ValueError,match='not room floor'):reference.sample([0,.54,0])


def test_room_probe_baseline_rejects_an_elevated_unclassified_support(tmp_path):
    declaration=floor(tmp_path);reference=NativeFloorReference(declaration)
    reference.validate_probes(probes())
    # A new support cannot inherit room-floor identity from local contact alone.
    reference.mesh.vertices[:,1]=.1
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    reference.mesh.ray=RayMeshIntersector(reference.mesh)
    with pytest.raises(ValueError,match='measured room floor baseline'):reference.sample([0,.1,0])


def lineage_floor(tmp_path, *, with_bed=False):
    declaration=floor(tmp_path)
    ap=tmp_path/'audit.json';vp=tmp_path/'vertices.f32';tp=tmp_path/'triangles.u32'
    audit=json.loads(ap.read_text())
    sections=[{'triangle_count':2,'material_path':'/Game/Imported/Root/Meshes/kitchen/cabinet/Looks/Wood'}]
    if with_bed:
        raw=np.fromfile(vp,dtype='<f4').reshape(-1,3)
        bed=raw.copy();bed[:,2]=53.6
        np.concatenate([raw,bed]).astype('<f4').tofile(vp)
        np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]],dtype='<u4').tofile(tp)
        sections.append({'triangle_count':2,'material_path':'/Game/Imported/Root/Meshes/bedroom/bed/Looks/Wood'})
    audit['native_render_geometry']['sections']=sections
    ap.write_text(json.dumps(audit))
    declaration.update(native_editor_audit_sha256=file_sha256(ap),native_vertices_sha256=file_sha256(vp),native_triangles_sha256=file_sha256(tp))
    source=tmp_path/'source_floor.usd';source.write_text('frozen source floor geometry fixture')
    record={'path':str(source),'sha256':file_sha256(source)}
    receipt={'schema':'kujiale_native_source_floor_lineage_v1','source_original':record,'source_expanded':record,'source_prim_layer_files':[record],
             'native_audit':{'path':str(ap),'sha256':file_sha256(ap)},
             'native_vertices':{'path':str(vp),'sha256':file_sha256(vp)},
             'native_triangles':{'path':str(tp),'sha256':file_sha256(tp)},'match_tolerance_m':1e-5,
             'floor_mesh_lineage':[{'source_prim':'/Root/Meshes/floor/floor_0/Mesh/SM_Floor_0',
               'source_points_canonical_m':[[-2,0,-2],[2,0,-2],[2,0,2],[-2,0,2]],
               'source_triangles':[[0,1,2],[0,2,3]],
               'native_face_to_source_triangle':[{'native_triangle_index':0,'source_floor_triangle_index':0},{'native_triangle_index':1,'source_floor_triangle_index':1}]}]}
    path=tmp_path/'lineage.json';path.write_text(json.dumps(receipt))
    declaration.update(source_floor_face_lineage_path=str(path),source_floor_face_lineage_sha256=file_sha256(path))
    return declaration,receipt


def replace_lineage_receipt(declaration,receipt):
    path=Path(declaration['source_floor_face_lineage_path']);path.write_text(json.dumps(receipt))
    declaration['source_floor_face_lineage_sha256']=file_sha256(path)


def test_exact_source_floor_survives_a_deduplicated_cabinet_material_donor(tmp_path):
    declaration,_=lineage_floor(tmp_path)
    reference=NativeFloorReference(declaration)
    assert reference.validate_probes(probes())['status']=='pass'
    assert reference.sample([.5,.04,-.5])==[.5,0.,-.5]
    assert reference.last_support_readback['surface_identity']=='verified source-floor triangle lineage'


def test_source_floor_provenance_never_accepts_a_bed_with_the_same_material(tmp_path):
    declaration,_=lineage_floor(tmp_path,with_bed=True)
    reference=NativeFloorReference(declaration)
    reference.validate_probes(probes())
    assert reference.sample([0,.01,0])==[0.,0.,0.]
    with pytest.raises(ValueError,match='not room floor'):reference.sample([0,.54,0])


def test_changed_frozen_floor_receipt_cannot_grant_floor_identity(tmp_path):
    declaration,receipt=lineage_floor(tmp_path)
    receipt['floor_mesh_lineage'][0]['source_prim']='/Root/Meshes/floor/SM_Floor_changed'
    Path(declaration['source_floor_face_lineage_path']).write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='receipt identity'):NativeFloorReference(declaration)


def test_changed_original_floor_source_rejects_a_valid_looking_receipt(tmp_path):
    declaration,_=lineage_floor(tmp_path)
    (tmp_path/'source_floor.usd').write_text('mutated')
    with pytest.raises(ValueError,match='source geometry identity'):NativeFloorReference(declaration)


def test_floor_receipt_for_another_native_audit_is_rejected(tmp_path):
    declaration,receipt=lineage_floor(tmp_path)
    receipt['native_audit']['sha256']='0'*64;replace_lineage_receipt(declaration,receipt)
    with pytest.raises(ValueError,match='different native geometry'):NativeFloorReference(declaration)


def test_floor_source_name_does_not_rescue_different_triangle_geometry(tmp_path):
    declaration,receipt=lineage_floor(tmp_path)
    receipt['floor_mesh_lineage'][0]['source_points_canonical_m'][0][1]=.001
    replace_lineage_receipt(declaration,receipt)
    with pytest.raises(ValueError,match='triangle geometry differs'):NativeFloorReference(declaration)


def test_coincident_cabinet_source_faces_do_not_gain_floor_identity(tmp_path):
    declaration,receipt=lineage_floor(tmp_path)
    receipt['floor_mesh_lineage'][0]['source_prim']='/Root/Meshes/kitchen/cabinet/SM_Cabinet'
    replace_lineage_receipt(declaration,receipt)
    with pytest.raises(ValueError,match='not a source floor mesh'):NativeFloorReference(declaration)


def test_duplicate_lineage_face_mapping_is_rejected(tmp_path):
    declaration,receipt=lineage_floor(tmp_path)
    receipt['floor_mesh_lineage'][0]['native_face_to_source_triangle'][1]['native_triangle_index']=0
    replace_lineage_receipt(declaration,receipt)
    with pytest.raises(ValueError,match='duplicates a face'):NativeFloorReference(declaration)
