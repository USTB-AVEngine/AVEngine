import pytest
from avengine.capture.kujiale_native_exposure import design_white_balance, sample_scene_exposure, apply_histogram_exposure, LEGACY_PROFILE
from test_kujiale_native_histogram_lighting import NativeCapture
pytestmark = pytest.mark.fast_unit

def light(kind,intensity,temperature=5500,enabled=True,units='LUMENS'):
    return dict(component_class=kind,intensity=intensity,temperature=temperature,use_temperature=enabled,intensity_units=units)

def test_environment_units_are_not_mixed_with_local_flux_or_sky_multiplier():
    got=design_white_balance([light('DirectionalLightComponent',2000,enabled=False),light('SkyLightComponent',1e9),light('PointLightComponent',1e9,4000)])
    assert got['temperature_k']==6500 and got['weight_units']=='lux'
    assert len(got['selected_lights'])==1

def test_mixed_local_white_points_and_disabled_emitters():
    got=design_white_balance([light('PointLightComponent',100,4500),light('RectLightComponent',100,6500)])
    assert got['temperature_k']==pytest.approx(2/(1/4500+1/6500))
    assert design_white_balance([])['temperature_k']==6500

def test_domain_randomization_is_reproducible_scene_scoped_and_bounded():
    values=[sample_scene_exposure('parent-scene-'+str(i)) for i in range(300)]
    assert all(-1<=v<=1 for v in values) and min(values)<-.9 and max(values)>.9
    assert sample_scene_exposure('parent-scene-1')==sample_scene_exposure('parent-scene-1')
    assert len(set(values))==len(values)
    with pytest.raises(ValueError):sample_scene_exposure(None)

def test_new_white_balance_changes_no_other_show_flags_and_legacy_is_replayable():
    capture=NativeCapture();before=list(capture.values['ShowFlagSettings'])
    got=apply_histogram_exposure(capture)
    assert got['native_readbacks']['WhiteTemp']==6500 and got['native_readbacks']['WhiteTint']==0
    assert capture.values['ShowFlagSettings'][0]==before[0]
    assert apply_histogram_exposure(NativeCapture(),{'visual_profile':LEGACY_PROFILE})['native_readbacks']['WhiteTemp']==4000
