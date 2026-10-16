"""Independent CIE reference values and foreground/mixed-colour admission cases."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

FILE=Path(__file__).resolve().parents[2]/'tools/assets/v7_object_color.py'
spec=importlib.util.spec_from_file_location('v7_object_color_test_subject',FILE)
subject=importlib.util.module_from_spec(spec);spec.loader.exec_module(subject)
THRESHOLDS={'neutral_chroma_max':15,'black_L_max':32,'white_L_min':60,'main_share_min':.55}

@pytest.mark.parametrize('rgb,expected',[
    ([255,255,255],[100,0,0]),([0,0,0],[0,0,0]),
    ([255,0,0],[53.2408,80.0925,67.2032]),
    ([0,255,0],[87.7347,-86.1827,83.1793]),
    ([0,0,255],[32.2970,79.1875,-107.8602]),
])
def test_cielab_d65_reference(rgb,expected):
    assert np.allclose(subject.lab(np.array(rgb)),expected,atol=.002)

def test_white_background_does_not_change_black_foreground_label():
    rgb=np.full((64,64,3),255,dtype=np.uint8);mask=np.zeros((64,64),bool);mask[20:40,20:40]=True;rgb[mask]=[18,18,18]
    result=subject.measure(rgb,mask,THRESHOLDS)
    assert result['status']=='pass' and result['label']=='black' and result['main_share']==1

def test_equal_black_red_regions_have_no_dominant_label():
    rgb=np.zeros((20,20,3),dtype=np.uint8);rgb[10:]=[170,20,20]
    assert subject.measure(rgb,np.ones((20,20),bool),THRESHOLDS)['status']=='fail'

def test_insufficient_body_pixels_fail_closed():
    assert subject.measure(np.full((4,4,3),255,dtype=np.uint8),np.ones((4,4),bool),THRESHOLDS)['status']=='fail'

def test_one_large_view_cannot_override_required_independent_agreement():
    colors=[{'status':'pass','label':'red','main_share':1,'body_pixels':20000,'family_shares':{'red':1}}]
    colors += [{'status':'fail','label':'gray_silver','main_share':.5,'body_pixels':100,'family_shares':{'gray_silver':.5,'black':.5}}]*3
    assert subject.summarize_views(colors,THRESHOLDS)['status']=='fail'

def test_two_agreeing_views_still_require_pooled_majority():
    colors=[{'status':'pass','label':'red','main_share':.6,'body_pixels':100,'family_shares':{'red':.6,'gray_silver':.4}}]*2
    colors += [{'status':'pass','label':'gray_silver','main_share':1,'body_pixels':1000,'family_shares':{'gray_silver':1}}]
    result=subject.summarize_views(colors,THRESHOLDS)
    assert result['status']=='fail' and result['label'] is None

def test_pooled_body_family_accepts_two_views_with_sufficient_share():
    colors=[{'status':'pass','label':'red','main_share':.8,'body_pixels':100,'family_shares':{'red':.8,'gray_silver':.2}}]*2
    colors += [{'status':'fail','label':'red','main_share':.5,'body_pixels':100,'family_shares':{'red':.5,'gray_silver':.5}}]
    result=subject.summarize_views(colors,THRESHOLDS)
    assert result['status']=='pass' and result['label']=='red' and result['main_share']==pytest.approx(.7)
