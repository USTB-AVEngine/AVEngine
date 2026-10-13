"""Cached skin must match the existing pose implementation on all influences."""
import numpy as np
import pytest
from test_roster_motion_multiple_skin_weights import fixture_document
from avengine.assets.roster_motion.pose import pose,accessor
from avengine.assets.animal_motion.skin_cache import CachedSkin

@pytest.mark.parametrize('extra,integer,repeated',[(False,False,False),(True,False,False),(True,True,False),(True,False,True)])
def test_cached_skin_matches_reference_with_bind_transforms(extra,integer,repeated):
    d,b,g,_,attrs=fixture_document(extra=extra,integer=integer)
    data=bytearray(b)
    if repeated:
        ac=d['accessors'][attrs['JOINTS_1']];view=d['bufferViews'][ac['bufferView']]
        values=np.tile([0,1,2,3],(3,1)).astype(np.uint16)
        data[view['byteOffset']:view['byteOffset']+view['byteLength']]=values.tobytes()
    ibm=np.tile(np.eye(4),(8,1,1))
    for j in range(8):
        angle=.12*(j+1);c,s=np.cos(angle),np.sin(angle)
        ibm[j,:3,3]=[.03*j,-.04*j,.02*j]
        g[j][:3,:3]=(1+.1*j)*np.array([[c,-s,0],[s,c,0],[0,0,1]])
    ai=d['skins'][0]['inverseBindMatrices'];view=d['bufferViews'][d['accessors'][ai]['bufferView']]
    data[view['byteOffset']:view['byteOffset']+view['byteLength']]=ibm.transpose(0,2,1).astype(np.float32).tobytes()
    b=bytes(data);expected=pose(d,b,globals_override=g,canonical_flip=False,ground=False)[0]
    cache=CachedSkin(d,b);actual=cache(g)
    np.testing.assert_allclose(actual[0]['positions'],expected[0]['positions'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(actual[0]['normals'],expected[0]['normals'],rtol=0,atol=1e-12)
    g[7][:3,3]+=[.25,-.12,.2]
    expected=pose(d,b,globals_override=g,canonical_flip=False,ground=False)[0]
    np.testing.assert_allclose(cache(g)[0]['positions'],expected[0]['positions'],rtol=0,atol=1e-12)

def test_unskinned_parts_and_missing_normals_preserve_reference_behavior():
    d,b,g,_,_=fixture_document()
    d['nodes'].append({'mesh':0});g[9]=np.eye(4);g[9][:3,3]=[.2,.3,.4]
    d['meshes'][0]['primitives'][0]['attributes'].pop('NORMAL')
    expected=pose(d,b,globals_override=g,canonical_flip=False,ground=False)[0]
    actual=CachedSkin(d,b)(g)
    assert len(actual)==2
    for a,e in zip(actual,expected):
        np.testing.assert_allclose(a['positions'],e['positions'],rtol=0,atol=1e-12)
        assert a['normals'] is e['normals'] is None

pytestmark=pytest.mark.fast_unit
