"""Private input representation for unused, mesh-free semantic scan nodes.

Original mesh JSON, accessors, indices, colours and binary bytes stay exact.
No native attachment, collision, visibility or material predicate is changed.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,json,struct


def prepare_semantic_sampling_view(source,destination):
    source=Path(source);destination=Path(destination)
    with source.open('rb')as f:
        magic,version,total=struct.unpack('<III',f.read(12))
        if magic!=0x46546c67 or version!=2:raise ValueError('expected original GLB v2')
        length,kind=struct.unpack('<II',f.read(8))
        if kind!=0x4e4f534a:raise ValueError('missing original GLB JSON')
        original=json.loads(f.read(length));length,kind=struct.unpack('<II',f.read(8))
        if kind!=0x004e4942:raise ValueError('missing original GLB binary')
        binary=f.read(length)
        if f.read(1):raise ValueError('unexpected additional original GLB chunk')
    if original.get('animations')or original.get('skins'):raise ValueError('only original static unskinned scan inputs supported')
    d=deepcopy(original);changes=[]
    for i,node in enumerate(d['nodes']):
        extra=set(node)-{'name','mesh','children'}
        if not extra:continue
        if 'mesh'in node or node.get('children')or extra-{'translation','rotation','scale','matrix'}:raise ValueError('real geometry or nontransform node feature requires separate review')
        # These leaf placeholders own no drawable, child, camera, light or
        # extension. Their transforms cannot affect any original triangle.
        removed={key:node.pop(key)for key in sorted(extra)}
        changes.append({'original_node_id':i,'name':node.get('name'),'removed_unused_leaf_metadata':removed,'mesh':None,'children':[]})
    if not changes:raise ValueError('no rejected unused node metadata to adapt')
    if any(original[k]!=d[k]for k in original if k!='nodes'):raise ValueError('input adaptation changed geometry metadata')
    encoded=json.dumps(d,separators=(',',':'),ensure_ascii=False).encode();encoded+=b' '*((-len(encoded))%4)
    payload=struct.pack('<III',magic,version,12+8+len(encoded)+8+len(binary))+struct.pack('<II',len(encoded),0x4e4f534a)+encoded+struct.pack('<II',len(binary),0x004e4942)+binary
    with destination.open('xb')as f:f.write(payload)
    with destination.open('rb')as f:
        f.read(12);n,kind=struct.unpack('<II',f.read(8));f.read(n);n,kind=struct.unpack('<II',f.read(8))
        if f.read(n)!=binary:raise ValueError('original geometry and label binary bytes changed')
    return {'source':str(source),'private_sampling_view':str(destination),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'view_sha256':hashlib.sha256(payload).hexdigest(),'original_binary_sha256':hashlib.sha256(binary).hexdigest(),'binary_bytes_exactly_equal':True,'mesh_json_accessors_indices_colours_exactly_equal':True,'source_mutated':False,'unused_leaf_node_metadata_changes':changes,'native_original_gate_status':'not_run until fresh native census/parent proof'}
