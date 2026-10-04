import json
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


def test_floor_cannot_claim_native_agreement_without_nine_measured_probes(tmp_path):
    reference=NativeFloorReference(floor(tmp_path))
    with pytest.raises(ValueError,match='nine'):reference.validate_probes(probes(.03))


def test_geometry_mutation_is_rejected_before_actor_spawn(tmp_path):
    declaration=floor(tmp_path);(tmp_path/'vertices.f32').write_bytes(b'changed')
    with pytest.raises(ValueError,match='geometry identity'):NativeFloorReference(declaration)
