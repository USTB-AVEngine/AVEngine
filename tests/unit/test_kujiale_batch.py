"""Regression coverage for design geometry adaptation and portable references."""
from pathlib import Path
import pytest
from tools.rooms.kujiale_batch import type_of
pytestmark=pytest.mark.fast_unit

def test_design_semantics_keep_outdoor_and_transit_exclusions():
    assert type_of('Bedroom')=='bedroom'
    assert type_of('study room')=='office'
    assert type_of('balcony')=='outdoor'
    assert type_of('storage')=='storage'
    assert type_of('UNREGISTERED_TYPE')=='unknown'

def test_whole_house_reference_preserves_cross_room_material_binding(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd,UsdGeom,UsdShade,Sdf
    from tools.rooms.prepare_interioragent_kujiale_adapter import _create_reference_stage
    source=tmp_path/'source.usda'
    stage=Usd.Stage.CreateNew(str(source))
    root=UsdGeom.Xform.Define(stage,'/Root');stage.SetDefaultPrim(root.GetPrim())
    for scope in ['bedroom_1','livingroom_2','wall','ceiling','floor']:
        UsdGeom.Scope.Define(stage,'/Root/Meshes/'+scope)
    material=UsdShade.Material.Define(stage,'/Root/Meshes/bedroom_1/Material')
    shader=UsdShade.Shader.Define(stage,material.GetPath().AppendChild('Shader'))
    shader.CreateIdAttr('UsdPreviewSurface');shader.CreateOutput('surface',Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    mesh=UsdGeom.Mesh.Define(stage,'/Root/Meshes/livingroom_2/Object')
    mesh.CreatePointsAttr([(0,0,0),(1,0,0),(0,1,0)])
    mesh.CreateFaceVertexCountsAttr([3]);mesh.CreateFaceVertexIndicesAttr([0,1,2])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    plan={'selected_scopes':['bedroom_1','livingroom_2','wall','ceiling','floor'],
          'include_rendering_scope':True,'backend_role':'production_visual'}
    prepared=_create_reference_stage(source=source,output=tmp_path/'adapter.usda',plan=plan,whole_stage=True)
    bound,_=UsdShade.MaterialBindingAPI(prepared.GetPrimAtPath(mesh.GetPath())).ComputeBoundMaterial()
    assert bound and bound.GetPath()==material.GetPath()
    with pytest.raises(ValueError,match='requires all scopes'):
        _create_reference_stage(source=source,output=tmp_path/'subset.usda',plan={**plan,'selected_scopes':['floor']},whole_stage=True)

def test_portable_stage_keeps_empty_mesh_children_and_world_transform(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd,UsdGeom,Gf
    from tools.rooms.kujiale_batch import portable_preview
    source=tmp_path/'source.usda';stage=Usd.Stage.CreateNew(str(source))
    root=UsdGeom.Xform.Define(stage,'/Root');stage.SetDefaultPrim(root.GetPrim())
    parent=UsdGeom.Mesh.Define(stage,'/Root/Meshes/floor/Container')
    parent.CreatePointsAttr([]);parent.CreateFaceVertexCountsAttr([]);parent.CreateFaceVertexIndicesAttr([])
    UsdGeom.Xformable(parent).AddTranslateOp().Set(Gf.Vec3d(2,3,4))
    child=UsdGeom.Mesh.Define(stage,'/Root/Meshes/floor/Container/Triangle')
    child.CreatePointsAttr([(0,0,0),(1,0,0),(0,1,0)])
    child.CreateFaceVertexCountsAttr([3]);child.CreateFaceVertexIndicesAttr([0,1,2])
    before=UsdGeom.XformCache().GetLocalToWorldTransform(child.GetPrim())
    stage.GetRootLayer().Save()
    receipt=portable_preview(source,tmp_path/'portable')
    output=Usd.Stage.Open(receipt['derived_stage'])
    assert output.GetPrimAtPath(parent.GetPath()).GetTypeName()=='Xform'
    assert output.GetPrimAtPath(child.GetPath()).GetTypeName()=='Mesh'
    assert UsdGeom.XformCache().GetLocalToWorldTransform(output.GetPrimAtPath(child.GetPath()))==before
    assert receipt['empty_mesh_containers_preserved_as_xform']==[str(parent.GetPath())]
    assert receipt['unresolved_dependencies']==[]

def test_extended_v2_uses_source_constants_and_texture_mode():
    from tools.rooms.kujiale_material_translation import numeric_plan
    pxr=pytest.importorskip('pxr.Gf')
    plan=numeric_plan({'ColorColor':pxr.Vec4f(.2,.3,.4,0),'ColorIsTexType':0.,
                       'MetallicFloat':.7,'RoughnessFloat':.11})
    assert plan['family']=='base_v2'
    assert plan['constants']['diffuseColor']==pytest.approx((.2,.3,.4))
    assert plan['constants']['metallic']==pytest.approx(.7)
    assert plan['constants']['roughness']==pytest.approx(.11)
    plan=numeric_plan({'ColorIsTexType':1.,'ColorUVW':(2.,3.,0.,1.),'ColorOffset':(.1,.2,0.,1.),'ColorTex':'sentinel'})
    assert 'diffuseColor' not in plan['constants']
    texture=next(x for x in plan['textures'] if x['target']=='diffuseColor')
    assert texture['key']=='ColorTex'
    assert texture['color']==(1.,1.,1.)
    assert texture['uva']==pytest.approx((2.,3.,.1,-2.2))


def test_extended_v2_does_not_certify_falloff_or_rotation():
    from tools.rooms.kujiale_material_translation import numeric_plan
    falloff=numeric_plan({'ColorIsTexType':2.,'ColorColor':(.1,.2,.3,1.)})
    assert 'diffuseColor' not in falloff['constants']
    assert 'Color:VIEW_DEPENDENT_FALLOFF_NOT_TRANSLATED' in falloff['review']
    rotated=numeric_plan({'ColorIsTexType':1.,'ColorUVW':(1.,1.,.25,1.)})
    assert 'Color:ROTATED_UV_NOT_TRANSLATED' in rotated['review']


def test_portable_native_preview_graph_retains_authored_missing_texture_fallback(tmp_path):
    Usd=pytest.importorskip('pxr.Usd');UsdGeom=pytest.importorskip('pxr.UsdGeom')
    UsdShade=pytest.importorskip('pxr.UsdShade');Sdf=pytest.importorskip('pxr.Sdf');Gf=pytest.importorskip('pxr.Gf')
    from tools.rooms.kujiale_batch import portable_preview
    source=tmp_path/'original.usda';stage=Usd.Stage.CreateNew(str(source))
    root=UsdGeom.Xform.Define(stage,'/Root');stage.SetDefaultPrim(root.GetPrim())
    material=UsdShade.Material.Define(stage,'/Root/Material')
    shader=UsdShade.Shader.Define(stage,'/Root/Material/OriginalPreview');shader.CreateIdAttr('UsdPreviewSurface')
    texture=UsdShade.Shader.Define(stage,'/Root/Material/OriginalTexture');texture.CreateIdAttr('UsdUVTexture')
    texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('missing.png'))
    texture.CreateInput('fallback',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(.2,.3,.4,1.))
    texture.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(),'rgb')
    shader.CreateOutput('surface',Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    stage.GetRootLayer().Save()
    receipt=portable_preview(source,tmp_path/'engine',preserve_preview_graphs=True)
    assert len(receipt['missing_texture_fallbacks'])==1
    assert receipt['unresolved_dependencies']==[]
    opened=Usd.Stage.Open(receipt['derived_stage'])
    retained=UsdShade.Shader(opened.GetPrimAtPath('/Root/Material/OriginalTexture'))
    assert retained.GetInput('fallback').Get()==Gf.Vec4f(.2,.3,.4,1.)
    assert retained.GetInput('file').Get().path==''
    assert '/Root/Material/OriginalPreview' in receipt['original_preview_shader_paths_retained']

def test_extended_v1_reads_native_gf_vectors_without_sequence_default():
    from tools.rooms.kujiale_material_translation import numeric_plan
    Gf=pytest.importorskip('pxr.Gf')
    plan=numeric_plan({'BaseColor_Color':Gf.Vec4f(.2,.3,.4,1.),'Gloss_Color':Gf.Vec4f(.2,.2,.2,1.),
                       'Metallic_Color':Gf.Vec4f(.6,.6,.6,1.),'IsBaseColorTex':1.,'BaseColor_Tex':'sentinel'})
    assert plan['family']=='base_v1'
    assert plan['constants']['diffuseColor']==pytest.approx((.2,.3,.4))
    assert plan['constants']['roughness']==pytest.approx(.8)
    assert plan['constants']['metallic']==pytest.approx(.6)
    texture=next(x for x in plan['textures'] if x['target']=='diffuseColor')
    assert texture['color']==pytest.approx((.2,.3,.4))
