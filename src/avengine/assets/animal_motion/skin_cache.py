"""Precompute linear skin contributions without dropping any influences."""
import numpy as np
from avengine.assets.roster_motion.pose import accessor,skin_influences
class CachedSkin:
    def __init__(self,document,binary):
        self.parts=[]
        for ni,node in enumerate(document['nodes']):
            if 'mesh' not in node:continue
            for pi,primitive in enumerate(document['meshes'][node['mesh']]['primitives']):
                attrs=primitive['attributes']
                pos=accessor(document,binary,attrs['POSITION']).astype(float)
                normal=accessor(document,binary,attrs['NORMAL']).astype(float) if 'NORMAL' in attrs else None
                item={'node':ni,'primitive':pi,'source_positions':pos,'source_normals':normal}
                if 'skin' in node:
                    skin=document['skins'][node['skin']]
                    joints=skin['joints'];count=len(joints)
                    ibm=accessor(document,binary,skin['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1)
                    js,ws=skin_influences(document,binary,attrs)
                    weights=np.zeros((len(pos),count))
                    rows=np.arange(len(pos))
                    for k in range(js.shape[1]):np.add.at(weights,(rows,js[:,k]),ws[:,k])
                    bound=np.empty((len(pos),count,4))
                    normals=np.empty((len(pos),count,3)) if normal is not None else None
                    homo=np.c_[pos,np.ones(len(pos))]
                    for j in range(count):
                        bound[:,j]=(homo@ibm[j].T)*weights[:,j,None]
                        if normals is not None:normals[:,j]=(normal@ibm[j,:3,:3].T)*weights[:,j,None]
                    item.update(joints=joints,bound=bound.reshape(len(pos),count*4),
                                bound_normals=normals.reshape(len(pos),count*3) if normals is not None else None)
                self.parts.append(item)
    def __call__(self,globals):
        result=[]
        for item in self.parts:
            if 'joints' in item:
                matrices=np.stack([globals[j] for j in item['joints']])
                pos=item['bound']@matrices[:,:3,:].transpose(0,2,1).reshape(-1,3)
                norm=item['bound_normals']@matrices[:,:3,:3].transpose(0,2,1).reshape(-1,3) if item['bound_normals'] is not None else None
            else:
                m=globals[item['node']]
                pos=item['source_positions']@m[:3,:3].T+m[:3,3]
                norm=item['source_normals']@np.linalg.inv(m[:3,:3]) if item['source_normals'] is not None else None
            if norm is not None:norm/=np.maximum(np.linalg.norm(norm,axis=1,keepdims=True),1e-9)
            result.append({'node':item['node'],'primitive':item['primitive'],'positions':pos,'normals':norm})
        return result
