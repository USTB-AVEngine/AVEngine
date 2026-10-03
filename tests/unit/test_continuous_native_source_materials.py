"""v7 streaming RGB uses the same source appearance gate as multi-actor RGB."""
import numpy as np
import pytest
from types import SimpleNamespace
from avengine.capture.continuous_native_video import NativeContinuousVideo

pytestmark=pytest.mark.fast_unit


def renderer():
    video=NativeContinuousVideo.__new__(NativeContinuousVideo)
    video.material_checks=[]
    video.actors=[{'semantic_id':1001,'endpoint_id':'voice1','declaration':{'asset_id':'colored_cloth'},
                   'source_material':{'source_palette_srgb':[[150/255,40/255,20/255]],
                                      'profiles':[{'max_highlight_ratio':.04}]}}]
    return video


def test_v7_streaming_material_failure_aborts_with_the_source_and_frame():
    video=renderer();rgb=np.full((40,40,3),100,dtype='uint8');semantic=np.full((40,40),1001)
    with pytest.raises(ValueError,match='source_texture_color_mismatch'):
        video._check_source_frame(rgb,semantic,75)
    assert video.material_checks[-1]['status']=='fail'
    assert video.material_checks[-1]['frame_index']==75
    assert video.material_checks[-1]['asset_id']=='colored_cloth'


def test_v7_streaming_material_records_valid_texture_and_offscreen_separately():
    video=renderer();rgb=np.full((40,40,3),[150,40,20],dtype='uint8');semantic=np.full((40,40),1001)
    video._check_source_frame(rgb,semantic,0)
    video._check_source_frame(rgb,np.zeros_like(semantic),149)
    assert [r['status'] for r in video.material_checks]==['pass','not_run']


def test_source_configuration_uses_the_existing_simulator_mediator():
    from avengine.assets.source_materials import configure_source_pbr
    attrs=SimpleNamespace(direct_light_intensity=3.14)
    manager=SimpleNamespace(create_template=lambda *_a,**_k:attrs,register_template=lambda *_a:1)
    mediator=SimpleNamespace(pbr_shader_template_manager=manager,
                             set_curr_default_pbr_attributes_handle=lambda _handle:True)
    config=SimpleNamespace(sim_cfg=object())
    result=configure_source_pbr(config,object(),metadata_mediator=mediator)
    assert config.metadata_mediator is mediator and result['enable_ibl'] is False
    assert attrs.map_mat_txtr_to_linear and attrs.map_output_to_srgb
