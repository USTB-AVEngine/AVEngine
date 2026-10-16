"""Attachment probes must follow the physical foot/back plate, including television overhang."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/assets'))
import v7_object_native as native


def test_television_panel_overhang_is_not_a_table_contact(monkeypatch):
    foot=np.array([[-.10,0,-.15],[-.10,0,.15],[.10,0,-.15],[.10,0,.15],[0,0,0]])
    panel=np.array([[-.03,.20,-.55],[-.03,.20,.55],[.03,.80,-.55],[.03,.80,.55]])
    vertices=np.concatenate([foot,panel]);monkeypatch.setattr(native,'load_glb',lambda path:(vertices,np.empty((0,3),int)))
    anchor,points,info=native.attachment_template({'derived_path':'fixture.glb','attachment_surface':'table'})
    assert np.allclose(anchor,[0,0,0])
    assert np.allclose(np.ptp(np.asarray(points),axis=0),[.20,0,.30])
    assert info['distinct_probe_vertices']==5
    assert all(any(np.array_equal(point,v) for v in foot) for point in points)


@pytest.mark.parametrize('surface,axis,extreme',[('wall',0,-.02),('ceiling',1,.06)])
def test_mounting_probes_use_the_actual_outer_plate(monkeypatch,surface,axis,extreme):
    plate=np.array([[-.02,.06,-.06],[-.02,.06,.06],[-.02,.06,0],[-.02,.06,-.03],[-.02,.06,.03]])
    # Spread points within the mounting plane; non-contact front/body lies away from it.
    if surface=='wall':plate[:,1]=np.linspace(0,.12,5)
    else:plate[:,0]=np.linspace(-.02,.04,5)
    body=np.array([[.04,0,-.06],[.04,.04,.06]])
    vertices=np.concatenate([plate,body]);monkeypatch.setattr(native,'load_glb',lambda path:(vertices,np.empty((0,3),int)))
    anchor,points,info=native.attachment_template({'derived_path':'fixture.glb','attachment_surface':surface})
    assert anchor[axis]==pytest.approx(extreme)
    assert np.allclose(np.asarray(points)[:,axis],extreme)
    assert info['outer_band_m']<=.002
