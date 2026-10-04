"""Auditable GLB index-only derivation for zero-surface native triangles."""
from __future__ import annotations
import copy,json,struct
from pathlib import Path
import numpy as np
from avengine.acoustics.gltf import load_glb_bytes,_decode_accessor

def zero_surface_mask(vertices,faces,*,strict=False):
    """No generic small-area cutoff: zero in float64 or zero in native float32.

    A float32-zero face with a double residue is accepted only within the
    explicit cross-product rounding bound 8*eps32*|edge1|*|edge2|. The per-face
    area tolerance is half that bound, recorded separately from exact zeros.
    """
    points=np.asarray(vertices,dtype=np.float64)[faces]
    e1=points[:,1]-points[:,0];e2=points[:,2]-points[:,0]
    c64=np.cross(e1,e2);area=.5*np.linalg.norm(c64,axis=1)
    p32=np.asarray(vertices,dtype=np.float32)[faces]
    c32=np.cross(p32[:,1]-p32[:,0],p32[:,2]-p32[:,0])
    exact=np.all(c64==0,axis=1);native_zero=np.all(c32==0,axis=1)
    tolerance=4*np.finfo(np.float32).eps*np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1)
    rounding=native_zero & ~exact & (area<=tolerance)
    if strict:rounding[:]=False
    return exact|rounding,dict(exact_zero=int(exact.sum()),native_float32_zero_with_double_residue=int(rounding.sum()),maximum_omitted_area_m2=float(area[exact|rounding].max(initial=0)),total_omitted_area_m2=float(area[exact|rounding].sum()),maximum_used_roundoff_area_tolerance_m2=float(tolerance[rounding].max(initial=0)),roundoff_rule=('disabled: strict double-coordinate cross is exactly zero; tolerance 0 m2' if strict else 'area <= 4*eps(float32)*norm(edge1)*norm(edge2), AND native float32 cross product exactly zero; no positive-area-only cutoff'))

def derive(source,destination,*,strict=False):
    source=Path(source);destination=Path(destination)
    if source.resolve()==destination.resolve():raise ValueError('Derived destination must differ from source')
    original=load_glb_bytes(source.read_bytes(),source_path=source);doc=copy.deepcopy(original.document)
    # These static scans use identity mesh nodes. Reject unsupported transforms
    # rather than silently changing the precision in which area is evaluated.
    if any(any(k in node for k in ('matrix','translation','rotation','scale','skin')) for node in doc.get('nodes',[])):
        raise ValueError('Nonidentity GLB node transform requires separate verification')
    binary=bytearray(original.binary);records=[];removed={};total=0;empty_primitives=[]
    for mi,mesh in enumerate(doc['meshes']):
        for pi,primitive in enumerate(mesh['primitives']):
            if primitive.get('mode',4)!=4 or primitive.get('extensions') or primitive.get('targets'):
                raise ValueError('Only uncompressed static indexed triangle primitives supported')
            positions=_decode_accessor(original,primitive['attributes']['POSITION'],expected_type='VEC3',allowed_component_types={5126})
            if 'indices' not in primitive:raise ValueError('Nonindexed primitive needs separate verification')
            old_accessor=primitive['indices']
            indices=_decode_accessor(original,old_accessor,expected_type='SCALAR',allowed_component_types={5121,5123,5125})
            if len(indices)%3:raise ValueError('Triangle index count not divisible by three')
            faces=indices.reshape(-1,3);mask,metrics=zero_surface_mask(positions,faces,strict=strict)
            if not mask.any():continue
            kept=indices.reshape(-1,3)[~mask].ravel()
            key=f'mesh_{mi}_primitive_{pi}';removed[key]=np.flatnonzero(mask)
            record=dict(mesh_index=mi,primitive_index=pi,original_faces=len(faces),removed_faces=int(mask.sum()),**metrics)
            records.append(record);total+=record['removed_faces']
            if not len(kept):
                record['all_zero_primitive_removed']=True;empty_primitives.append((mi,pi));continue
            while len(binary)%4:binary.append(0)
            start=len(binary);raw=kept.astype(indices.dtype).tobytes();binary.extend(raw)
            view=len(doc['bufferViews']);doc['bufferViews'].append(dict(buffer=0,byteOffset=start,byteLength=len(raw),target=34963))
            accessor=copy.deepcopy(doc['accessors'][old_accessor]);accessor.update(bufferView=view,byteOffset=0,count=len(kept),min=[int(kept.min())],max=[int(kept.max())])
            primitive['indices']=len(doc['accessors']);doc['accessors'].append(accessor)
            assert np.array_equal(kept.reshape(-1,3),faces[~mask])
    for mi,mesh in enumerate(doc['meshes']):
        mesh['primitives']=[p for pi,p in enumerate(mesh['primitives']) if (mi,pi) not in empty_primitives]
    # A glTF mesh must contain a primitive. Empty nodes remain as containers;
    # only their mesh binding disappears. Nonempty mesh references are reindexed.
    mesh_map={};nonempty=[];empty_meshes=[]
    for mi,mesh in enumerate(doc['meshes']):
        if mesh['primitives']:mesh_map[mi]=len(nonempty);nonempty.append(mesh)
        else:empty_meshes.append(dict(original_mesh_index=mi,name=mesh.get('name')))
    doc['meshes']=nonempty
    for node in doc.get('nodes',[]):
        if 'mesh' in node:
            if node['mesh'] in mesh_map:node['mesh']=mesh_map[node['mesh']]
            else:del node['mesh']
    doc['buffers'][0]['byteLength']=len(binary)
    while len(binary)%4:binary.append(0)
    encoded=json.dumps(doc,separators=(',',':'),ensure_ascii=False).encode()
    encoded+=b' '*((-len(encoded))%4)
    payload=struct.pack('<4sII',b'glTF',2,12+8+len(encoded)+8+len(binary))+struct.pack('<II',len(encoded),0x4E4F534A)+encoded+struct.pack('<II',len(binary),0x004E4942)+binary
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as f:f.write(payload)
    after=load_glb_bytes(destination.read_bytes(),source_path=destination)
    assert after.binary[:len(original.binary)]==original.binary
    assert original.sha256==load_glb_bytes(source.read_bytes(),source_path=source).sha256
    index_path=destination.with_suffix('.removed_faces.npz');np.savez_compressed(index_path,**removed)
    return dict(strict_zero_only=strict,tolerance_m2=0.0 if strict else None,source=str(source),derived=str(destination),source_sha256=original.sha256,derived_sha256=after.sha256,operation='派生几何：清退化三角形',removed_faces=total,by_primitive=records,omitted_indices=str(index_path),vertices_normals_uv_materials_textures_unchanged=True,node_attributes_except_mesh_bindings_unchanged=True,empty_primitives_removed=len(empty_primitives),empty_meshes=empty_meshes,mesh_reference_map=mesh_map,original_binary_prefix_unchanged=True,original_source_unchanged=True,maximum_omitted_area_m2=max((r['maximum_omitted_area_m2'] for r in records),default=0),total_omitted_area_m2=sum(r['total_omitted_area_m2'] for r in records),exact_zero_faces=sum(r['exact_zero'] for r in records),float32_roundoff_zero_faces=sum(r['native_float32_zero_with_double_residue'] for r in records))
