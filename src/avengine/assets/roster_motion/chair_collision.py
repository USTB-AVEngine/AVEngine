"""Retain original chair geometry while exposing its parts to native collision.

Bullet rigid objects convexify each collision primitive. A primitive exposing
all chair vertices fills the empty space between its seat, back and legs even
when its indices select only one part. These collision-only GLBs partition the
original connected components and compact each attribute to its referenced
vertices. Render GLBs, positions, normals, UVs, materials and triangle winding
are retained; no bounding-box or replacement-chair geometry is introduced.
"""
from pathlib import Path
from copy import deepcopy
import numpy as np
from avengine.assets.roster_motion.io import read_glb,write_glb

def prepare_chair_collision(source,target):
 source=Path(source);target=Path(target);doc,blob=read_glb(source);result=deepcopy(doc);proof=[]
 def array(index):
  a=doc['accessors'][index];assert not a.get('sparse')and not a.get('normalized')
  view=doc['bufferViews'][a['bufferView']];assert view.get('buffer',0)==0
  dtype=np.dtype({5121:'u1',5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']]);width={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']]
  offset=view.get('byteOffset',0)+a.get('byteOffset',0);stride=view.get('byteStride',width*dtype.itemsize)
  return np.ndarray((a['count'],width),dtype=dtype,buffer=blob,offset=offset,strides=(stride,dtype.itemsize)).copy()
 for mi,mesh in enumerate(doc['meshes']):
  primitives=[]
  for pi,primitive in enumerate(mesh['primitives']):
   assert primitive.get('mode',4)==4
   positions=array(primitive['attributes']['POSITION']);indices=array(primitive['indices']).astype(np.uint32).ravel().reshape(-1,3)
   _,inverse=np.unique(positions,axis=0,return_inverse=True);parents=list(range(int(inverse.max())+1))
   def root(value):
    value=int(value)
    while parents[value]!=value:
     parents[value]=parents[parents[value]];value=parents[value]
    return value
   for face in inverse[indices]:
    first=root(face[0])
    for value in face[1:]:parents[root(value)]=first
   components={}
   for i,face in enumerate(indices):components.setdefault(root(inverse[face[0]]),[]).append(i)
   restored=[]
   for rows in components.values():
    original_triangles=indices[rows];restored.extend(rows)
    used,inverse_used=np.unique(original_triangles,return_inverse=True)
    triangles=inverse_used.astype('<u4').reshape(-1,3)
    split=deepcopy(primitive)
    # Bullet convex hulls consume the full POSITION array. Each separate
    # primitive must expose only its own original component's vertices.
    for attribute,index in primitive['attributes'].items():
     values=array(index);subset=values[used]
     assert np.array_equal(subset[triangles],values[original_triangles])
     attr_payload=subset.tobytes();blob+=b'\0'*(-len(blob)%4)
     attr_view=len(result['bufferViews']);result['bufferViews'].append({'buffer':0,'byteOffset':len(blob),'byteLength':len(attr_payload),'target':34962});blob+=attr_payload
     attr_accessor=deepcopy(doc['accessors'][index]);attr_accessor['bufferView']=attr_view;attr_accessor.pop('byteOffset',None);attr_accessor['count']=len(subset)
     if 'min'in attr_accessor:attr_accessor['min']=subset.min(0).tolist()
     if 'max'in attr_accessor:attr_accessor['max']=subset.max(0).tolist()
     new_index=len(result['accessors']);result['accessors'].append(attr_accessor);split['attributes'][attribute]=new_index
    payload=triangles.tobytes();blob+=b'\0'*(-len(blob)%4)
    view_index=len(result['bufferViews']);result['bufferViews'].append({'buffer':0,'byteOffset':len(blob),'byteLength':len(payload),'target':34963});blob+=payload
    accessor_index=len(result['accessors']);result['accessors'].append({'bufferView':view_index,'componentType':5125,'count':int(triangles.size),'type':'SCALAR','min':[int(triangles.min())],'max':[int(triangles.max())]})
    split['indices']=accessor_index;primitives.append(split)
   assert sorted(restored)==list(range(len(indices)))
   proof.append({'mesh':mi,'primitive':pi,'original_triangles':len(indices),'components':len(components),'all_original_triangle_indices_retained_once':True,'original_positions_normals_uvs_materials_retained':True})
  result['meshes'][mi]['primitives']=primitives
 result['buffers'][0]['byteLength']=len(blob)
 assert result['nodes']==doc['nodes']and result.get('materials')==doc.get('materials')
 target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():
  oldd,oldb=read_glb(target);assert oldd==result and oldb.rstrip(b'\0')==blob.rstrip(b'\0')
 else:write_glb(target,result,blob)
 return {'source':str(source),'collision':str(target),'components':proof,'representation':'native compound convex hulls of original connected components; exact full-triangle checks remain separate','join_collision_meshes':False}
