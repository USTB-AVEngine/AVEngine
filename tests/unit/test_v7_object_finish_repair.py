"""A material-only export must preserve actual geometry and front controls."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/assets'))
from v7_object_finish_repair import repair_mesh,canonical_triangles,walnut_albedo
from measure_static_upright_correction import load_glb
from avengine.assets.source_materials import read_glb
from v7_object_color import classify

def test_casing_finish_preserves_transformed_triangle_geometry_and_functional_material(tmp_path):
    mesh=trimesh.creation.box(extents=[1.,1.,1.])
    mesh.visual=TextureVisuals(material=PBRMaterial(name='fixture_controls',baseColorFactor=[.01,.02,.03,1.]))
    scene=trimesh.Scene();matrix=np.eye(4);matrix[1,3]=.5
    scene.add_geometry(mesh,transform=matrix)
    source=tmp_path/'before.glb';source.write_bytes(scene.export(file_type='glb'));original=source.read_bytes()
    v,f=load_glb(source);meta={'object_type':'tablet','geometry':{'bbox_m':[v.min(axis=0).tolist(),v.max(axis=0).tolist()]}}
    target=tmp_path/'after.glb';receipt=repair_mesh(source,target,meta,'red')
    v2,f2=load_glb(target)
    assert receipt['preserved_functional_faces']==2 and receipt['casing_faces']==10
    assert np.allclose(canonical_triangles(v,f),canonical_triangles(v2,f2),atol=1e-6)
    assert source.read_bytes()==original
    d0,_=read_glb(source);d1,_=read_glb(target)
    old=d0['materials'][0]['pbrMetallicRoughness']['baseColorFactor']
    preserved=[m['pbrMetallicRoughness']['baseColorFactor'] for m in d1['materials'] if m.get('name')=='fixture_controls']
    assert len(preserved)==1 and np.allclose(preserved[0],old)

def test_walnut_albedo_retains_luminance_variation_and_measures_as_brown_family():
    gray=np.array([[[15,15,15],[80,80,80]],[[150,150,150],[220,220,220]]],dtype=np.uint8)
    result=np.asarray(walnut_albedo(Image.fromarray(gray))).reshape((-1,4))
    assert len(np.unique(result[:,:3],axis=0))==4
    thresholds={'neutral_chroma_max':10,'black_L_max':28,'white_L_min':52,'main_share_min':.55}
    assert set(classify(result[:,:3],thresholds))=={'wood_brown'}

def test_duplicate_wrapper_names_do_not_discard_actual_glb_transforms(tmp_path):
    from v7_object_prepare import wrapped
    from v7_object_finish_repair import geometry_match
    mesh=trimesh.creation.box(extents=[.9,.2,.6])
    scene=trimesh.Scene(mesh);base=tmp_path/'raw.glb';base.write_bytes(scene.export(file_type='glb'))
    one=tmp_path/'one.glb';two=tmp_path/'two.glb'
    first=np.diag([.7,.9,1.1,1.]);wrapped(base,one,first)
    second=np.eye(4);second[1,3]=.4;wrapped(one,two,second)
    v,f=load_glb(two);meta={'object_type':'tablet','geometry':{'bbox_m':[v.min(axis=0).tolist(),v.max(axis=0).tolist()]}}
    target=tmp_path/'finished.glb';repair_mesh(two,target,meta,'gray_silver')
    after,faces=load_glb(target)
    assert geometry_match(v,f,after,faces)['status']=='pass'

def test_triangle_match_detects_a_real_vertex_displacement():
    from v7_object_finish_repair import geometry_match
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    faces=np.array([[0,1,2]])
    near=v.copy();near[1,0]+=1e-8
    assert geometry_match(v,faces,near,faces)['status']=='pass'
    moved=v.copy();moved[1,0]+=.01
    assert geometry_match(v,faces,moved,faces)['status']=='fail'

def test_inactive_display_is_explicitly_dark_without_emission_and_keeps_geometry(tmp_path):
    from v7_object_finish_repair import geometry_match
    mesh=trimesh.creation.box(extents=[.2,.3,.25]);mesh.apply_translation([0,.15,0])
    source=tmp_path/'original.glb';source.write_bytes(trimesh.Scene(mesh).export(file_type='glb'))
    v,f=load_glb(source);meta={'object_type':'tablet','geometry':{'bbox_m':[v.min(0).tolist(),v.max(0).tolist()]}}
    target=tmp_path/'inactive.glb';r=repair_mesh(source,target,meta,'gray_silver',dark_functional=True)
    after,faces=load_glb(target)
    assert geometry_match(v,f,after,faces)['status']=='pass'
    doc,_=read_glb(target);material=next(m for m in doc['materials'] if m.get('name')=='v7_inactive_dark_functional')
    assert max(material['pbrMetallicRoughness']['baseColorFactor'][:3])<.02
    assert material['emissiveFactor']==[0.,0.,0.]
    assert r['functional_state']=='inactive_dark_screen'
