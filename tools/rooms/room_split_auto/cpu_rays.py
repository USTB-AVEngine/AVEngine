"""CPU BVH reference rays on intact raw scan faces, without native upload."""
from __future__ import annotations
import numpy as np
from numba import njit

@njit(cache=False)
def trace(tris,order,low,high,left,right,start,count,origins,directions):
    distances=np.full(len(origins),np.inf);faces=np.full(len(origins),-1,np.int64)
    for ri in range(len(origins)):
        o=origins[ri];d=directions[ri];stack=np.empty(128,np.int64);top=1;stack[0]=0
        while top:
            top-=1;node=stack[top];near=0.;far=distances[ri];hit=True
            for axis in range(3):
                if abs(d[axis])<1e-15:
                    if o[axis]<low[node,axis] or o[axis]>high[node,axis]:hit=False;break
                else:
                    a=(low[node,axis]-o[axis])/d[axis];b=(high[node,axis]-o[axis])/d[axis]
                    near=max(near,min(a,b));far=min(far,max(a,b))
                    if far<near:hit=False;break
            if not hit:continue
            if count[node]==0:
                stack[top]=left[node];stack[top+1]=right[node];top+=2;continue
            for j in range(start[node],start[node]+count[node]):
                fi=order[j];a=tris[fi,0];e1=tris[fi,1]-a;e2=tris[fi,2]-a
                p=np.array([d[1]*e2[2]-d[2]*e2[1],d[2]*e2[0]-d[0]*e2[2],d[0]*e2[1]-d[1]*e2[0]])
                det=e1[0]*p[0]+e1[1]*p[1]+e1[2]*p[2]
                if det==0:continue
                inv=1/det;s=o-a;u=(s[0]*p[0]+s[1]*p[1]+s[2]*p[2])*inv
                if u< -1e-10 or u>1+1e-10:continue
                q=np.array([s[1]*e1[2]-s[2]*e1[1],s[2]*e1[0]-s[0]*e1[2],s[0]*e1[1]-s[1]*e1[0]])
                v=(d[0]*q[0]+d[1]*q[1]+d[2]*q[2])*inv
                if v< -1e-10 or u+v>1+1e-10:continue
                t=(e2[0]*q[0]+e2[1]*q[1]+e2[2]*q[2])*inv
                if t>.005 and t<distances[ri]:distances[ri]=t;faces[ri]=fi
    return distances,faces

class CPUScanRayIntersector:
    """Median triangle BVH and double-precision Moller-Trumbore queries.

    All faces, including mathematical zeros, stay in the input arrays. A zero
    determinant cannot contribute a ray hit. No face-area threshold is used.
    """
    def __init__(self,mesh,leaf_size=32):
        self.tris=np.ascontiguousarray(mesh.triangles,dtype=np.float64)
        self.order=np.arange(len(self.tris),dtype=np.int64)
        lo=self.tris.min(axis=1);hi=self.tris.max(axis=1);centers=(lo+hi)*.5
        low=[];high=[];left=[];right=[];start=[];count=[]
        def build(a,b):
            node=len(low);ids=self.order[a:b]
            low.append(lo[ids].min(axis=0));high.append(hi[ids].max(axis=0));left.append(-1);right.append(-1);start.append(a);count.append(b-a)
            if b-a>leaf_size:
                axis=int(np.argmax(np.ptp(centers[ids],axis=0)));k=(b-a)//2
                partition=np.argpartition(centers[ids,axis],k);self.order[a:b]=ids[partition]
                mid=a+k;left[node]=build(a,mid);right[node]=build(mid,b);count[node]=0
            return node
        if not len(self.tris):raise ValueError('Raw scan contains no faces')
        build(0,len(self.tris))
        self.nodes=tuple(np.ascontiguousarray(v,dtype=np.float64 if i<2 else np.int64) for i,v in enumerate([low,high,left,right,start,count]))
        self.receipt=dict(backend='Numba CPU scan BVH reference',input_faces=len(self.tris),omitted_faces=0,leaf_size=leaf_size,nodes=len(low),minimum_ray_distance_m=.005,determinant_policy='exact zero only',barycentric_roundoff_tolerance=1e-10,compute_device='CPU',gpu_renderer_created=False)
    def intersects_location(self,origins,directions,multiple_hits=False):
        if multiple_hits:raise ValueError('Only first-hit scan diagnosis supported')
        origins=np.ascontiguousarray(origins,dtype=np.float64);directions=np.ascontiguousarray(directions,dtype=np.float64)
        distances,faces=trace(self.tris,self.order,*self.nodes,origins,directions)
        ids=np.flatnonzero(faces>=0)
        return origins[ids]+distances[ids,None]*directions[ids],ids,faces[ids]
