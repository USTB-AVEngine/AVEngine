"""Physical world area and original material provenance regressions."""
import hashlib
import math
import runpy
from pathlib import Path
import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
from avengine.capture.native_ue_qa import source_materials

pytestmark = pytest.mark.fast_unit


def adapter():
    return runpy.run_path(str(Path(__file__).resolve().parents[2]/'tools/ue/build_kujiale_authored_light_sidecar.py'))


def test_centimeter_fixture_flux_and_mirror_keep_original_emitting_normal():
    stage=Usd.Stage.CreateInMemory()
    lamp=UsdLux.SphereLight.Define(stage,'/Lamp')
    lamp.GetRadiusAttr().Set(15);lamp.GetIntensityAttr().Set(5000);lamp.GetExposureAttr().Set(1)
    transform=Gf.Matrix4d().SetScale(Gf.Vec3d(.01));transform.SetTranslateOnly(Gf.Vec3d(1,2,3))
    bake=adapter()['world_light_dimensions']
    matrix,values,record=bake(lamp.GetPrim(),transform,1.)
    assert values['inputs:radius']==pytest.approx(.15)
    # 10,000 nits on a 15cm spherical emitter, not a 15m emitter.
    assert record['expected_native_intensity_lumens']==pytest.approx(35530.5758439,rel=1e-9)
    assert np.allclose(matrix.ExtractTranslation(),[1,2,3])
    panel=UsdLux.RectLight.Define(stage,'/Panel');panel.GetWidthAttr().Set(5);panel.GetHeightAttr().Set(20)
    mirror=Gf.Matrix4d().SetScale(Gf.Vec3d(-.01,.01,.01));mirror.SetTranslateOnly(Gf.Vec3d(1,2,3))
    rigid,values,record=bake(panel.GetPrim(),mirror,1.)
    assert values['inputs:width']==pytest.approx(.05) and values['inputs:height']==pytest.approx(.2)
    assert np.linalg.det(np.asarray(rigid)[:3,:3])==pytest.approx(1.)
    assert record['emitting_normal_world']==pytest.approx([0,0,-1])


def test_qa_material_reference_retains_original_ue_namespace_and_source(tmp_path):
    original=tmp_path/'source_world_baked.usda'
    stage=Usd.Stage.CreateNew(str(original));UsdGeom.Scope.Define(stage,'/Root')
    fixture=UsdGeom.Cube.Define(stage,'/Root/Meshes/Fixture')
    material=UsdShade.Material.Define(stage,'/Root/Meshes/Fixture/Looks/MI_Body')
    shader=UsdShade.Shader.Define(stage,str(material.GetPath())+'/PreviewSurface')
    shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(.12)
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.2,.3,.4))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    stage.GetRootLayer().Save();original_sha=hashlib.sha256(original.read_bytes()).hexdigest()
    output_dir=tmp_path/'qa_only';output_dir.mkdir();output=output_dir/original.name
    adapter()['build_material_reference'](stage,stage.Flatten(),output)
    native='/Game/Own/source_world_baked/Root/Meshes/Fixture/Looks/MI_Body.MI_Body'
    audit={'source_usd':str(original),'materials':[{'path':native,'default_material':False,'blend_mode':'BLEND_OPAQUE','textures':[]}]}
    before=source_materials(audit);after=source_materials({**audit,'source_usd':str(output)})
    assert before==after
    assert after[native]['source_status']=='preview_surface'
    assert after[native]['expected_opacity']==pytest.approx(.12)
    reference_stage=Usd.Stage.Open(str(output))
    assert not any(prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube) for prim in reference_stage.TraverseAll())
    assert hashlib.sha256(original.read_bytes()).hexdigest()==original_sha
