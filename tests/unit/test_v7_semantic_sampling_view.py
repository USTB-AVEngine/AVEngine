import json,struct
import pytest
from avengine.capture.v7_semantic_sampling_view import prepare_semantic_sampling_view
pytestmark=pytest.mark.fast_unit


def glb(document,binary):
    header=json.dumps(document).encode();header+=b' '*((-len(header))%4)
    return struct.pack('<III',0x46546c67,2,28+len(header)+len(binary))+struct.pack('<II',len(header),0x4e4f534a)+header+struct.pack('<II',len(binary),0x004e4942)+binary


def test_unused_leaf_view_keeps_source_and_all_geometry_bytes(tmp_path):
    document={'asset':{'version':'2.0'},'nodes':[{'name':'Camera','translation':[4.,5.,6.]},{'mesh':0,'name':'actual_room'}],'meshes':[{'primitives':[{'attributes':{'POSITION':0,'COLOR_0':1},'indices':2}]}],'accessors':[{'count':3,'type':'VEC3'}],'buffers':[{'byteLength':16}],'scenes':[{'nodes':[0,1]}],'scene':0}
    binary=bytes(range(16));before=glb(document,binary);source=tmp_path/'original.glb';source.write_bytes(before);view=tmp_path/'private.glb'
    receipt=prepare_semantic_sampling_view(source,view)
    assert source.read_bytes()==before
    with view.open('rb')as f:
        f.read(12);size,kind=struct.unpack('<II',f.read(8));adapted=json.loads(f.read(size));size,kind=struct.unpack('<II',f.read(8));assert f.read(size)==binary
    assert adapted['nodes']==[{'name':'Camera'},document['nodes'][1]]
    assert {k:v for k,v in adapted.items()if k!='nodes'}=={k:v for k,v in document.items()if k!='nodes'}
    assert receipt['binary_bytes_exactly_equal'] and receipt['source_mutated'] is False


def test_real_geometry_transform_cannot_be_removed_as_empty_metadata(tmp_path):
    document={'nodes':[{'mesh':0,'translation':[1.,0.,0.]}],'meshes':[],'buffers':[{'byteLength':4}]}
    source=tmp_path/'original.glb';source.write_bytes(glb(document,b'1234'));view=tmp_path/'private.glb'
    with pytest.raises(ValueError,match='real geometry'):prepare_semantic_sampling_view(source,view)
    assert not view.exists()
