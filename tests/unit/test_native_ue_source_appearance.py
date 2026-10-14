import json
from types import SimpleNamespace
import numpy as np
import pytest
from avengine.capture.native_ue_source_appearance import measure_ue_source_highlights,verify_original_ue_materials
from avengine.assets.source_materials import measure_source_appearance

pytestmark=pytest.mark.fast_unit


def test_ue_colored_shadow_does_not_use_habitat_color_gate():
    reference=np.full((20,20,3),[180,80,40],dtype=np.uint8)
    rendered=np.full_like(reference,[35,25,15]);mask=np.ones((20,20),bool)
    assert measure_source_appearance(rendered,mask,palette=np.array([[.7,.3,.2]]),max_highlight_ratio=.02,reference_rgb=reference,strict=False)['status']=='fail'
    metric=measure_ue_source_highlights(rendered,mask,max_highlight_ratio=.02,reference_rgb=reference)
    assert metric['status']=='pass' and metric['highlight_ratio']==0 and metric['color_checks_used'] is False


def test_highlight_ratio_and_gate_equal_frozen_metric():
    reference=np.full((20,20,3),[40,40,40],dtype=np.uint8)
    rendered=reference.copy();rendered[:2]=255;mask=np.ones((20,20),bool)
    old=measure_source_appearance(rendered,mask,palette=np.array([[.2,.2,.2]]),max_highlight_ratio=.04,reference_rgb=reference,strict=False)
    new=measure_ue_source_highlights(rendered,mask,max_highlight_ratio=.04,reference_rgb=reference)
    assert old['highlight_ratio']==new['highlight_ratio']==.1
    assert new['status']=='fail' and new['failures']==['excess_source_highlights']


def test_missing_native_body_pixels_cannot_pass():
    image=np.zeros((20,20,3),np.uint8)
    assert measure_ue_source_highlights(image,np.zeros((20,20),bool),max_highlight_ratio=.02,reference_rgb=image)['status']=='not_run'


def test_original_material_readback_never_writes_component(tmp_path):
    path=tmp_path/'bindings.json';path.write_text(json.dumps({'status':'pass','asset_bindings':{'a':['/Game/Original.M']},'restore_receipt':'proof.json'}))
    component=SimpleNamespace(GetNumMaterials=lambda:1,GetMaterial=lambda **kw:123)
    service=SimpleNamespace(load_object=lambda **kw:123)
    receipt=verify_original_ue_materials(SimpleNamespace(unreal_service=service),{'component':component},'a',path)
    assert receipt['status']=='pass' and receipt['material_overrides_applied'] is False


def test_wrong_native_material_is_rejected(tmp_path):
    path=tmp_path/'bindings.json';path.write_text(json.dumps({'status':'pass','asset_bindings':{'a':['/Game/Original.M']},'restore_receipt':'proof.json'}))
    component=SimpleNamespace(GetNumMaterials=lambda:1,GetMaterial=lambda **kw:456)
    service=SimpleNamespace(load_object=lambda **kw:123)
    with pytest.raises(ValueError):verify_original_ue_materials(SimpleNamespace(unreal_service=service),{'component':component},'a',path)
