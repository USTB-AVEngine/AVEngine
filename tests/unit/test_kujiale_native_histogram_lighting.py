import hashlib
import json
import runpy
import sys
from pathlib import Path
import numpy as np
import pytest
from avengine.capture.kujiale_native_exposure import (
    PARAMETERS, PROFILE, apply_histogram_exposure, bind_histogram_profile,
    exposure_console_commands, require_authored_light_audit, is_histogram,
)
from avengine.capture.kujiale_native_lighting import assert_authored_light_positions, reflect_native_light_pose
pytestmark = pytest.mark.fast_unit


class NativeCapture:
    def __init__(self, bad=None):
        self.values = {'CaptureSource':'SCS_FinalColorLDR','ShowFlagSettings':[{'ShowFlagName':'Bloom','Enabled':True},{'ShowFlagName':'EyeAdaptation','Enabled':False}]}; self.bad = bad
    def SetShowFlagSettings(self, *, InShowFlagSettings):
        self.values['ShowFlagSettings'] = InShowFlagSettings
    def set_property_value(self, *, property_name, property_value):
        self.values[property_name] = property_value
    def get_property_value(self, *, property_name):
        if property_name == self.bad:
            return False if isinstance(self.values[property_name], bool) else -99
        return self.values[property_name]


def test_readbacks_reject_disabled_adaptation_and_wrong_ev():
    for bad in ['bAlwaysPersistRenderingState', 'PostProcessSettings.AutoExposureMaxBrightness',
                'PostProcessSettings.bOverride_AutoExposureMethod']:
        with pytest.raises(ValueError, match='differs'):
            apply_histogram_exposure(NativeCapture(bad))
    camera = NativeCapture()
    got = apply_histogram_exposure(camera)
    assert {'ShowFlagName':'Bloom','Enabled':True} in camera.values['ShowFlagSettings']
    bad = NativeCapture(); bad.values['CaptureSource'] = 'SCS_SceneColorHDR'
    with pytest.raises(ValueError, match='final-color'):apply_histogram_exposure(bad)
    assert got['parameters']['min_ev100'] < got['parameters']['max_ev100']
    assert got['parameters']['low_percent'] < got['parameters']['high_percent']
    assert got['brightness_match_verified'] is False


def test_no_per_room_overrides_and_legacy_map_does_not_enter_fresh_generation(tmp_path):
    original = {'visual_profile': 'legacy', 'exposure_bias_ev': 30, 'white_balance_temp_k': 9000,
                'output_root': str(tmp_path)}
    current = bind_histogram_profile(original)
    assert original['exposure_bias_ev'] == 30
    assert current['exposure_bias_ev'] == 0 and current['white_balance_temp_k'] == 4000
    assert 'r.EyeAdaptationQuality 2' in exposure_console_commands(current)
    assert 'r.DefaultFeature.AutoExposure.ExtendDefaultLuminanceRange 1' in exposure_console_commands(current)
    with pytest.raises(ValueError, match='requires measured'):
        require_authored_light_audit(current, '/Game/OwnNewMap')
    assert is_histogram({'visual_profile':'kujiale-native-authored-lights-histogram-v1'})
    audit_path = tmp_path/'audit.json'
    audit = {'status': 'pass', 'map_path': '/Game/OwnNewMap', 'expected_light_count': 1,
             'light_components': [{}], 'light_transform_assertions_passed': True, 'ad_hoc_fill_count': 0}
    audit_path.write_text(json.dumps(audit)); current['native_lighting_audit'] = str(audit_path)
    with pytest.raises(ValueError, match='world-size photometry'):
        require_authored_light_audit(current, '/Game/OwnNewMap')
    audit['world_photometry_assertions_passed']=True; audit_path.write_text(json.dumps(audit))
    assert require_authored_light_audit(current, '/Game/OwnNewMap') == audit
    with pytest.raises(ValueError, match='another map'):
        require_authored_light_audit(current, '/Game/LegacyMap')
    audit['ad_hoc_fill_count'] = 1; audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='ad hoc fill'):
        require_authored_light_audit(current, '/Game/OwnNewMap')


def test_light_pose_reflection_is_world_y_only_and_reversible():
    position, rotation = reflect_native_light_pose([100, -200, 300], [15, 30, -45])
    assert position == [100, 200, 300] and rotation == [15, -30, 45]
    assert reflect_native_light_pose(position, rotation) == ([100, -200, 300], [15, 30, -45])


def test_lamps_at_origin_or_zero_intensity_do_not_pass():
    expected = [{'type': 'SphereLight', 'prim_path': '/Room/TableLamp',
                 'expected_native_ue_cm': [100, 200, 250], 'effective_usd_intensity': 1000}]
    actual = [{'component_class': 'PointLightComponent', 'component': 'OwnLamp',
               'position_ue_cm': [100, 200, 250], 'intensity': 125}]
    assert assert_authored_light_positions(expected, actual)[0]['native_position_error_m'] == 0
    actual[0]['position_ue_cm'] = [0, 0, 0]
    with pytest.raises(ValueError, match='moved'):
        assert_authored_light_positions(expected, actual)
    actual[0]['position_ue_cm'] = [100, 200, 250]; actual[0]['intensity'] = 0
    with pytest.raises(ValueError, match='became zero'):
        assert_authored_light_positions(expected, actual)


@pytest.mark.parametrize('nested_light', [False, True])
def test_native_bake_keeps_geometry_and_authored_light_world_transform(tmp_path, monkeypatch, nested_light):
    from pxr import Gf, Usd, UsdGeom, UsdLux
    source = tmp_path/'original.usda'; output = tmp_path/'new_baked.usdc'; receipt = tmp_path/'receipt.json'
    stage = Usd.Stage.CreateNew(str(source)); UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    parent = UsdGeom.Xform.Define(stage, '/Root'); parent.AddTranslateOp().Set(Gf.Vec3d(2, 3, 4))
    parent.AddRotateZOp().Set(35)
    mesh = UsdGeom.Mesh.Define(stage, '/Root/Mesh')
    mesh.GetPointsAttr().Set([Gf.Vec3f(0,0,0),Gf.Vec3f(1,0,0),Gf.Vec3f(0,1,0)])
    mesh.GetFaceVertexCountsAttr().Set([3]); mesh.GetFaceVertexIndicesAttr().Set([0,1,2])
    light = UsdLux.RectLight.Define(stage, '/Root/Lamp'); UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(1,2,3))
    light.CreateIntensityAttr(1000); light.CreateExposureAttr(1); light.CreateWidthAttr(.7)
    if nested_light:
        child = UsdLux.SphereLight.Define(stage, '/Root/Lamp/Nested')
        UsdGeom.Xformable(child).AddTranslateOp().Set(Gf.Vec3d(2,1,0)); child.CreateIntensityAttr(500)
    stage.GetRootLayer().Save()
    original_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    before = {str(p.GetPath()): np.asarray(cache.GetLocalToWorldTransform(p)) for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)}
    points = [cache.GetLocalToWorldTransform(mesh.GetPrim()).Transform(Gf.Vec3d(*p)) for p in mesh.GetPointsAttr().Get()]
    script = Path(__file__).resolve().parents[2]/'tools/ue/bake_kujiale_native_usd.py'
    monkeypatch.setattr(sys, 'argv', [str(script), '--source', str(source), '--output', str(output), '--receipt', str(receipt)])
    runpy.run_path(str(script), run_name='__main__')
    baked = Usd.Stage.Open(str(output)); fresh = UsdGeom.XformCache(Usd.TimeCode.Default())
    for path, matrix in before.items():
        assert np.allclose(np.asarray(fresh.GetLocalToWorldTransform(baked.GetPrimAtPath(path))), matrix, atol=1e-9)
    assert np.allclose(np.asarray(UsdGeom.Mesh(baked.GetPrimAtPath('/Root/Mesh')).GetPointsAttr().Get()), np.asarray(points), atol=1e-6)
    assert UsdLux.RectLight(baked.GetPrimAtPath('/Root/Lamp')).GetIntensityAttr().Get() == 1000
    assert UsdLux.RectLight(baked.GetPrimAtPath('/Root/Lamp')).GetWidthAttr().Get() == pytest.approx(.7)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_sha
    assert json.loads(receipt.read_text())['authored_light_count'] == len(before)
