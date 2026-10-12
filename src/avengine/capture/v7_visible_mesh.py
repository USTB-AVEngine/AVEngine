"""Original static GLB geometry resolved by node index, including all transforms.

A registered generated GLB can contain a node named ``world`` inside physical
correction parents. Name-keyed trimesh scene loading collides with its synthetic
world node and drops those parents. Index traversal retains the authored graph;
no source geometry or runtime scale is changed.
"""
from pathlib import Path
import numpy as np
from avengine.assets.glb import load_glb


def indexed_static_mesh(path):
    import trimesh
    doc=load_glb(Path(path));gltf=doc.json;nodes=gltf['nodes'];vertices=[];faces=[];total=0
    def accessor(index):
        a=gltf['accessors'][index]
        if a.get('sparse') or a.get('normalized'):raise ValueError('unsupported static mesh accessor storage')
        view=gltf['bufferViews'][a['bufferView']]
        if view.get('buffer',0)!=0:raise ValueError('static mesh needs the original embedded buffer')
        dtype=np.dtype({5121:'u1',5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']]);width={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']]
        stride=int(view.get('byteStride',dtype.itemsize*width));offset=int(view.get('byteOffset',0))+int(a.get('byteOffset',0));count=int(a['count'])
        if count<=0 or stride<width*dtype.itemsize or int(a.get('byteOffset',0))+(count-1)*stride+width*dtype.itemsize>view['byteLength']:raise ValueError('static accessor leaves the original buffer view')
        values=np.ndarray((count,width),dtype=dtype,buffer=doc.binary,offset=offset,strides=(stride,dtype.itemsize)).copy()
        if not np.isfinite(values).all():raise ValueError('non-finite static mesh accessor')
        return values
    def local_matrix(node):
        if 'matrix' in node:
            if any(k in node for k in ('translation','rotation','scale')):raise ValueError('ambiguous original node matrix/TRS')
            matrix=np.asarray(node['matrix'],float).reshape(4,4).T
        else:
            x,y,z,w=np.asarray(node.get('rotation',[0,0,0,1]),float)
            if abs(x*x+y*y+z*z+w*w-1)>1e-4:raise ValueError('unnormalized original node quaternion')
            rot=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
            matrix=np.eye(4);matrix[:3,:3]=rot@np.diag(node.get('scale',[1,1,1]));matrix[:3,3]=node.get('translation',[0,0,0])
        if not np.isfinite(matrix).all() or not np.allclose(matrix[3],[0,0,0,1]):raise ValueError('invalid original node transform')
        return matrix
    def walk(index,parent,ancestry):
        nonlocal total
        if index in ancestry:raise ValueError('cycle in original GLB node indices')
        if not 0<=index<len(nodes):raise ValueError('original GLB node index is out of range')
        node=nodes[index];world=parent@local_matrix(node)
        if 'mesh' in node:
            for primitive in gltf['meshes'][node['mesh']]['primitives']:
                if primitive.get('mode',4)!=4:raise ValueError('original static mesh is not triangles')
                local=accessor(primitive['attributes']['POSITION']).astype(float)
                if local.shape[1]!=3:raise ValueError('original static POSITION is not VEC3')
                inds=accessor(primitive['indices']).astype(np.int64).ravel() if 'indices' in primitive else np.arange(len(local))
                if len(inds)%3 or not len(inds) or np.min(inds)<0 or np.max(inds)>=len(local):raise ValueError('invalid original triangle indices')
                vertices.append(local@world[:3,:3].T+world[:3,3]);faces.append(inds.reshape(-1,3)+total);total+=len(local)
        for child in node.get('children',[]):walk(int(child),world,ancestry|{index})
    for root in gltf['scenes'][gltf.get('scene',0)]['nodes']:walk(int(root),np.eye(4),set())
    if not vertices:raise ValueError('original registered mesh is empty')
    return trimesh.Trimesh(np.concatenate(vertices),np.concatenate(faces),process=False)
