"""Extra glTF skin influences must affect real posed geometry."""
import numpy as np
import pytest
from avengine.assets.roster_motion.pose import pose,skin_influences


def fixture_document(extra=True,integer=False):
    binary=bytearray();views=[];accessors=[]
    def add(values,kind,component=5126,normalized=False):
        dtype={5126:np.float32,5123:np.uint16,5121:np.uint8}[component]
        x=np.asarray(values,dtype);binary.extend(bytes((-len(binary))%4))
        vi=len(views);views.append({'buffer':0,'byteOffset':len(binary),'byteLength':x.nbytes});binary.extend(x.tobytes())
        ac={'bufferView':vi,'componentType':component,'count':len(x),'type':kind}
        if normalized:ac['normalized']=True
        ai=len(accessors);accessors.append(ac);return ai
    positions=np.array([[0.,0.,0.],[1.,0.,0.],[0.,0.,1.]])
    weights0=np.tile([32,32,32,32] if integer else [.125]*4,(3,1))
    weights1=np.tile([32,32,32,31] if integer else [.125]*4,(3,1))
    component=5121 if integer else 5126
    attributes={'POSITION':add(positions,'VEC3'),
                'NORMAL':add([[0,1,0]]*3,'VEC3'),
                'JOINTS_0':add([[0,1,2,3]]*3,'VEC4',5123),
                'WEIGHTS_0':add(weights0,'VEC4',component,integer)}
    if extra:
        attributes['JOINTS_1']=add([[4,5,6,7]]*3,'VEC4',5123)
        attributes['WEIGHTS_1']=add(weights1,'VEC4',component,integer)
    ibm=add(np.tile(np.eye(4).reshape(1,16),(8,1)),'MAT4')
    d={'nodes':[{'name':f'joint_{i}'} for i in range(8)]+[{'mesh':0,'skin':0}],
       'meshes':[{'primitives':[{'attributes':attributes}]}],
       'skins':[{'joints':list(range(8)),'inverseBindMatrices':ibm}],
       'accessors':accessors,'bufferViews':views}
    globals_override={i:np.eye(4) for i in range(9)}
    for i in range(8):globals_override[i][0,3]=i+1
    return d,bytes(binary),globals_override,positions,attributes


@pytest.mark.parametrize('integer',[False,True])
def test_second_set_moves_geometry_with_all_eight_bones(integer):
    d,b,g,rest,attributes=fixture_document(integer=integer)
    parts,_=pose(d,b,globals_override=g,ground=False,canonical_flip=False)
    expected=4.5 if not integer else sum((i+1)*w for i,w in enumerate([32]*7+[31]))/255
    np.testing.assert_allclose(parts[0]['positions'],rest+[expected,0,0],atol=1e-7)
    np.testing.assert_allclose(parts[0]['normals'],[[0,1,0]]*3,atol=1e-7)


def test_existing_four_influence_geometry_is_preserved():
    d,b,g,rest,_=fixture_document(extra=False)
    parts,_=pose(d,b,globals_override=g,ground=False,canonical_flip=False)
    np.testing.assert_allclose(parts[0]['positions'],rest+[2.5,0,0],atol=1e-7)


def test_zero_total_weights_cannot_create_false_support_geometry():
    d,b,g,rest,attributes=fixture_document(extra=False)
    a=d['accessors'][attributes['WEIGHTS_0']];v=d['bufferViews'][a['bufferView']];data=bytearray(b);data[v['byteOffset']:v['byteOffset']+v['byteLength']]=bytes(v['byteLength'])
    with pytest.raises(ValueError,match='skin_weights_must_be_finite'):
        skin_influences(d,bytes(data),attributes)

pytestmark = pytest.mark.fast_unit
