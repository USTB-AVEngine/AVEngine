import importlib.util
from pathlib import Path
import numpy as np
import pytest
import json,struct
pytestmark=pytest.mark.fast_unit
REPO=Path(__file__).resolve().parents[1]
def module(name):
    spec=importlib.util.spec_from_file_location(name,REPO/'tools/acoustics'/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_zero_surface_policy_retains_real_small_triangles():
    m=module('derive_zero_area_glb')
    vertices=np.array([[0,0,0],[1,0,0],[0,1e-12,0],[2,0,0]],dtype=np.float32)
    mask,receipt=m.zero_surface_mask(vertices,np.array([[0,1,2],[0,1,3],[0,0,2]]))
    assert mask.tolist()==[False,True,True]
    assert receipt['exact_zero']==2 and receipt['maximum_omitted_area_m2']==0

def test_native_rounding_residue_requires_explicit_area_bound():
    m=module('derive_zero_area_glb')
    vertices=np.array([[0,0,0],[1,1,0],[2,2,0],[0,0,1]],dtype=np.float32)
    mask,receipt=m.zero_surface_mask(vertices,np.array([[0,1,2],[0,1,3]]))
    assert mask.tolist()==[True,False]
    assert receipt['total_omitted_area_m2']==0

def test_no_waveform_native_failure_is_not_reclassified_as_pass():
    m=module('train_house_followup')
    r={'status':'review','reasons':['NATIVE_LOAD_FAILED'],'rirs':[],'input':{'house':'fixture'}}
    corrected=m.corrected_result(r,{})
    assert corrected['status']=='review' and corrected['reasons']==r['reasons']

@pytest.mark.parametrize('separate_zero_mesh',[False,True])
def test_glb_derivation_preserves_original_binary_and_positive_face_indices(tmp_path,separate_zero_mesh):
    m=module('derive_zero_area_glb')
    vertices=np.array([[0,0,0],[1,0,0],[0,1e-12,0],[2,0,0]],dtype='<f4')
    indices=np.array([0,1,2,0,1,3],dtype='<u2')
    binary=vertices.tobytes()+indices.tobytes()+b'RGBA'
    doc={'asset':{'version':'2.0'},'buffers':[{'byteLength':len(binary)}],
         'bufferViews':[{'buffer':0,'byteOffset':0,'byteLength':vertices.nbytes},{'buffer':0,'byteOffset':vertices.nbytes,'byteLength':indices.nbytes},{'buffer':0,'byteOffset':vertices.nbytes+indices.nbytes,'byteLength':4}],
         'accessors':[{'bufferView':0,'componentType':5126,'count':4,'type':'VEC3'},{'bufferView':1,'componentType':5123,'count':6,'type':'SCALAR'}],
         'meshes':[{'primitives':[{'attributes':{'POSITION':0},'indices':1,'material':0}]}],
         'materials':[{'name':'unchanged_material'}],'nodes':[{'mesh':0}],'scenes':[{'nodes':[0]}],'scene':0}
    if separate_zero_mesh:
        doc['accessors'] += [dict(doc['accessors'][1],count=3,byteOffset=6),dict(doc['accessors'][1],count=3)]
        doc['meshes']=[{'name':'empty','primitives':[{'attributes':{'POSITION':0},'indices':2,'material':0}]},{'name':'positive','primitives':[{'attributes':{'POSITION':0},'indices':3,'material':0}]}]
        doc['nodes']=[{'name':'empty_node','mesh':0},{'name':'positive_node','mesh':1}];doc['scenes']=[{'nodes':[0,1]}]
    encoded=json.dumps(doc).encode();encoded+=b' '*((-len(encoded))%4)
    source=tmp_path/'raw.glb';source.write_bytes(struct.pack('<4sII',b'glTF',2,28+len(encoded)+len(binary))+struct.pack('<II',len(encoded),0x4E4F534A)+encoded+struct.pack('<II',len(binary),0x004E4942)+binary)
    original=source.read_bytes();derived=tmp_path/'derived.glb';receipt=m.derive(source,derived)
    after=m.load_glb_bytes(derived.read_bytes())
    assert source.read_bytes()==original and after.binary[:len(binary)]==binary
    assert after.document['materials']==doc['materials']
    if separate_zero_mesh:
        assert after.document['nodes']==[{'name':'empty_node'},{'name':'positive_node','mesh':0}]
        assert len(after.document['meshes'])==1 and receipt['empty_primitives_removed']==1
    else:assert after.document['nodes']==doc['nodes']
    new_indices=m._decode_accessor(after,after.document['meshes'][0]['primitives'][0]['indices'],expected_type='SCALAR',allowed_component_types={5123})
    assert new_indices.tolist()==[0,1,2] and receipt['removed_faces']==1

@pytest.mark.parametrize('verdict',['use','skip','unsure',None])
def test_human_labels_and_training_exposure_do_not_filter_the_pool(verdict,tmp_path,monkeypatch):
    m=module('report_train_house_followup')
    monkeypatch.setattr(m,'exposure',lambda output,h:dict(so_seen=True,m1_seen=True,v6_seen=False))
    room={'room_label':'R1','human':{'verdict':verdict},'stage1':{'status':'pass'},'metrics':{'floor_area_m2':5,'short_side_m':2.4,'nav_main_area_m2':3,'black_fraction':.1}}
    h={'house':'fixture','family':'hm3d','house_gate':{'status':'pass'},'candidate_room_count':1,'registry_source':'fixture','acoustic_rooms':[{'room_label':'R1'}]}
    r={'status':'pass','reasons':[],'rirs':[{'status':'valid','room_label':'R1'}]*6,'result_directory':str(tmp_path),'visual':{'status':'pass'}}
    out=m.annotate(r,h,tmp_path,[room],set())
    assert out['training_eligible'] and out['known_exposure'] and not out['human_labels_filter']
    assert out['smy_verdict_by_room']['R1']==verdict
    held=m.annotate(r,h,tmp_path,[room],{'fixture'})
    assert not held['training_eligible'] and held['batch_exclusion']=='CURRENT_HOLDOUT_RESERVED'

def test_cpu_reference_rays_keep_original_faces_when_native_raw_upload_fails(tmp_path,monkeypatch):
    import trimesh
    m=module('verify_train_house_derivation')
    mesh=trimesh.Trimesh(vertices=[[-3,0,-4],[3,0,-4],[3,4,-4],[-3,4,-4]],faces=[[0,1,2],[0,2,3],[0,0,1]],process=False)
    monkeypatch.setattr(m,'load_raw_scan',lambda p:(mesh,np.full((3,3),175,dtype=np.uint8),['fixture']))
    def reject(*args):raise ValueError("object 'raw_visual_scan_ray_diagnostic' contains a zero-area triangle")
    monkeypatch.setattr(m,'NativeRawRayIntersector',reject)
    d={'probe':'fixture','listener_m':[0,1.5,0],'source_m':[0,1.2,-2],'camera_forward':[0,0,-1],'check':{'angle_deg':0,'foa_world_direction':[0,0,-1]}}
    r=m.visual({'foa_checks':[dict(d,probe=str(i)) for i in range(3)]},{},tmp_path/'fixture.glb',tmp_path/'views')
    assert len(mesh.faces)==3 and r['triangle_count']==3
    assert r['ray_engine']=='Numba CPU scan BVH reference' and r['native_raw_upload_error']
    assert r['ray_bvh_setup']['omitted_faces']==0
    assert r['status']=='pass' and r['native_invalid_faces_remaining']==1

def test_cpu_reference_does_not_hide_other_native_raw_upload_errors(tmp_path,monkeypatch):
    import trimesh
    m=module('verify_train_house_derivation')
    mesh=trimesh.creation.box()
    monkeypatch.setattr(m,'load_raw_scan',lambda p:(mesh,np.full((len(mesh.faces),3),175,dtype=np.uint8),['fixture']))
    def reject(*args):raise ValueError('Native runtime configuration missing')
    monkeypatch.setattr(m,'NativeRawRayIntersector',reject)
    with pytest.raises(ValueError,match='Native runtime configuration missing'):
        m.visual({'foa_checks':[]},{},tmp_path/'fixture.glb',tmp_path/'views')
